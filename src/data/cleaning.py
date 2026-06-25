# =============================================================================
# src/data/cleaning.py
# California Housing Project — Data Cleaning (EDA-driven, Phase 1)
#
# Decisions source: notebooks/03_eda.ipynb → configs/data_config.yaml (eda_derived)
#
# WHAT THIS MODULE DOES (in order):
#   1. Global median imputation on total_bedrooms  (MCAR confirmed, p=0.3095)
#   2. Add is_capped flag on target                (>= cap_threshold from config)
#   3. log1p transform on count-based columns      (from config: log1p_columns)
#   4. median_income: NO transform                 (RobustScaler handled in preprocessing)
#   5. LOF outlier flag                            (contamination from config)
#
# CONFIG-DRIVEN DESIGN:
#   All EDA-derived values (log1p columns, cap threshold, LOF contamination/
#   features) are read from configs/data_config.yaml -> eda_derived section.
#   This is the SAME file produced by notebooks/03_eda.ipynb's extraction cell.
#   Re-running EDA and updating the YAML automatically updates this module's
#   behaviour - no hardcoded constants to keep in sync manually.
#
#   Module-level constants below are FALLBACK DEFAULTS ONLY, used if the
#   config file or eda_derived section is missing. A warning is logged
#   whenever a fallback is used.
#
# WHAT THIS MODULE DOES NOT DO:
#   - Feature engineering  -> feature_engineering.py
#   - Scaling/encoding     -> preprocessing.py
#   - Dropping raw columns -> feature_engineering.py (after ratio creation)
#
# FIT/TRANSFORM RULE (prevents data leakage):
#   All statistics (median, LOF) are fit on TRAIN only.
#   Val and test are transformed using train-derived values.
#
# INPUT  : data/interim/train.csv | val.csv | test.csv
# OUTPUT : data/processed/train_clean.csv | val_clean.csv | test_clean.csv
#          + artifacts/cleaning_artifacts.json  (train-fit statistics)
#
# DVC: output directory is tracked as data/processed/
# =============================================================================

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import LocalOutlierFactor

from src.utils.logger import get_logger
from src.utils.paths import (
    PROJECT_DIR,
    DVC_REMOTE_NAME,
    is_dvc_initialized,
    ensure_path,
    configure_git_identity,
)

logger = get_logger(__name__)

# -- Fallback defaults - used ONLY if data_config.yaml is missing eda_derived --
# These mirror the last known EDA run, but config.yaml is the source of truth.
_FALLBACK_LOG1P_COLS: list[str] = [
    "total_rooms", "total_bedrooms", "population", "households",
]
_FALLBACK_CAP_THRESHOLD: float = 500_001.0
_FALLBACK_LOF_CONTAMINATION: float = 0.02
_FALLBACK_LOF_N_NEIGHBORS: int = 20
_FALLBACK_LOF_FEATURES: list[str] = [
    "median_income", "total_rooms", "population",
    "households", "longitude", "latitude",
]
_TARGET: str = "median_house_value"


# =============================================================================
# CONFIG LOADER - reads eda_derived section from data_config.yaml
# =============================================================================

@dataclass
class EdaConfig:
    """
    Typed container for EDA-derived cleaning parameters.

    Populated from configs/data_config.yaml -> eda_derived section.
    Falls back to module-level defaults (with a warning) if any key is missing.
    """
    log1p_columns: list[str]
    cap_threshold: float
    lof_contamination: float
    lof_n_neighbors: int
    lof_features: list[str]
    imputation_strategy: str = "global_median"
    source: str = "config"   # "config" or "fallback" - for traceability


def load_eda_config(config_path: str | Path = "configs/data_config.yaml") -> EdaConfig:
    """
    Load EDA-derived cleaning parameters from data_config.yaml.

    Reads the `eda_derived` section produced by notebooks/03_eda.ipynb's
    extraction cell. If the file or section is missing, falls back to
    last-known-good defaults and logs a warning - the pipeline still runs,
    but you should re-run EDA and update the config.

    Returns
    -------
    EdaConfig with all parameters needed by the cleaning steps below.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(
            f"Config not found: {config_path} - using fallback EDA defaults. "
            "Run notebooks/03_eda.ipynb and save data_config.yaml for accurate values."
        )
        return EdaConfig(
            log1p_columns=_FALLBACK_LOG1P_COLS,
            cap_threshold=_FALLBACK_CAP_THRESHOLD,
            lof_contamination=_FALLBACK_LOF_CONTAMINATION,
            lof_n_neighbors=_FALLBACK_LOF_N_NEIGHBORS,
            lof_features=_FALLBACK_LOF_FEATURES,
            source="fallback",
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    eda = cfg.get("eda_derived")
    if not eda:
        logger.warning(
            f"'{config_path}' has no 'eda_derived' section - using fallback defaults. "
            "Run notebooks/03_eda.ipynb's extraction cell and update the YAML."
        )
        return EdaConfig(
            log1p_columns=_FALLBACK_LOG1P_COLS,
            cap_threshold=_FALLBACK_CAP_THRESHOLD,
            lof_contamination=_FALLBACK_LOF_CONTAMINATION,
            lof_n_neighbors=_FALLBACK_LOF_N_NEIGHBORS,
            lof_features=_FALLBACK_LOF_FEATURES,
            source="fallback",
        )

    log1p_cols   = eda.get("log1p_columns", _FALLBACK_LOG1P_COLS)
    cap_thresh   = eda.get("target_summary", {}).get("cap_threshold", _FALLBACK_CAP_THRESHOLD)
    lof_cfg      = eda.get("lof", {})
    contamination = lof_cfg.get("contamination", _FALLBACK_LOF_CONTAMINATION)
    n_neighbors    = lof_cfg.get("n_neighbors", _FALLBACK_LOF_N_NEIGHBORS)
    lof_features   = lof_cfg.get("features", _FALLBACK_LOF_FEATURES)
    imputation_strategy = (
        eda.get("missingness", {})
           .get("total_bedrooms", {})
           .get("imputation_strategy", "global_median")
    )

    logger.info(
        f"EDA config loaded: log1p_cols={log1p_cols}, "
        f"cap_threshold={cap_thresh}, lof_contamination={contamination}"
    )

    return EdaConfig(
        log1p_columns=log1p_cols,
        cap_threshold=float(cap_thresh),
        lof_contamination=float(contamination),
        lof_n_neighbors=int(n_neighbors),
        lof_features=lof_features,
        imputation_strategy=imputation_strategy,
        source="config",
    )


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class CleaningResult:
    """Holds cleaned DataFrames, artifacts, and a run summary."""

    train: pd.DataFrame
    val:   pd.DataFrame
    test:  pd.DataFrame

    train_path: Optional[Path] = None
    val_path:   Optional[Path] = None
    test_path:  Optional[Path] = None

    # Train-fit statistics (serialised to artifacts/cleaning_artifacts.json)
    imputation_median: Optional[float] = None
    lof_n_outliers_train: int = 0
    eda_config_source: str = "config"   # "config" or "fallback" - traceability

    timestamp:   datetime = field(default_factory=datetime.now)
    dvc_tracked: bool = False
    warnings:    list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  CLEANING RESULT SUMMARY",
            "=" * 60,
            f"  Train : {len(self.train):,} rows x {self.train.shape[1]} cols",
            f"  Val   : {len(self.val):,} rows x {self.val.shape[1]} cols",
            f"  Test  : {len(self.test):,} rows x {self.test.shape[1]} cols",
            "",
            f"  EDA config source             : {self.eda_config_source}",
            f"  total_bedrooms median (train) : {self.imputation_median}",
            f"  LOF outliers flagged (train)  : {self.lof_n_outliers_train}",
            f"  DVC tracked                   : {'yes' if self.dvc_tracked else 'no'}",
        ]
        if self.eda_config_source == "fallback":
            lines.append(
                "  WARNING: using fallback EDA defaults, not data_config.yaml!"
            )
        if self.train_path:
            lines += [
                "",
                "  Saved to:",
                f"    train -> {self.train_path}",
                f"    val   -> {self.val_path}",
                f"    test  -> {self.test_path}",
            ]
        if self.warnings:
            lines.append("\n  Warnings:")
            for w in self.warnings:
                lines.append(f"    !  {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# STEP 1 - IMPUTATION
# =============================================================================

def fit_imputer(train: pd.DataFrame) -> dict[str, float]:
    """
    Compute train-only imputation statistics.

    Returns a dict mapping column -> fill value.
    Only columns with actual nulls are included.

    Strategy is global median/mode regardless of MCAR/MAR - grouped
    imputation (e.g. by ocean_proximity) is not yet implemented even if
    eda_config.imputation_strategy says otherwise. See module docstring.
    """
    stats: dict[str, float] = {}
    null_cols = [c for c in train.columns if train[c].isnull().any()]

    for col in null_cols:
        if pd.api.types.is_numeric_dtype(train[col]):
            median_val = train[col].median()
            stats[col] = float(median_val)
            logger.info(f"Imputer fit: '{col}' -> median = {median_val:.4f}")
        else:
            mode_val = train[col].mode()[0]
            stats[col] = mode_val
            logger.info(f"Imputer fit: '{col}' -> mode = {mode_val}")

    return stats


def apply_imputer(df: pd.DataFrame, imputer_stats: dict[str, float]) -> pd.DataFrame:
    """Apply train-fit imputation to any split (train / val / test)."""
    df = df.copy()
    for col, fill_val in imputer_stats.items():
        if col not in df.columns:
            logger.warning(f"Imputer: column '{col}' not found in DataFrame - skipping.")
            continue
        n_filled = df[col].isnull().sum()
        if n_filled > 0:
            df[col] = df[col].fillna(fill_val)
            logger.info(f"Imputed '{col}': filled {n_filled} nulls with {fill_val}")
    return df


# =============================================================================
# STEP 2 - TARGET FLAG
# =============================================================================

def add_is_capped_flag(
    df: pd.DataFrame,
    cap_threshold: float = _FALLBACK_CAP_THRESHOLD,
    target: str = _TARGET,
) -> pd.DataFrame:
    """
    Add binary `is_capped` column (1 if target >= cap_threshold).

    cap_threshold should come from EdaConfig.cap_threshold (data_config.yaml
    -> eda_derived.target_summary.cap_threshold), not the fallback default.

    Reason: capped rows are NOT errors - they are real districts whose
    value exceeds the survey ceiling. Flagging lets models treat them
    differently without removing them.
    """
    df = df.copy()
    if target not in df.columns:
        logger.warning(f"Target column '{target}' not found - is_capped flag skipped.")
        return df
    df["is_capped"] = (df[target] >= cap_threshold).astype(int)
    n_capped = df["is_capped"].sum()
    logger.info(
        f"is_capped flag added (threshold={cap_threshold}): "
        f"{n_capped:,} rows ({n_capped/len(df):.2%})"
    )
    return df


# =============================================================================
# STEP 3 - LOG1P TRANSFORM
# =============================================================================

def apply_log1p(
    df: pd.DataFrame,
    cols: list[str] = _FALLBACK_LOG1P_COLS,
) -> pd.DataFrame:
    """
    Apply log1p to count-based columns.

    `cols` should come from EdaConfig.log1p_columns (data_config.yaml ->
    eda_derived.log1p_columns), not the fallback default.

    NOT transformed: median_income - not a count variable. RobustScaler
    handles this in preprocessing.py for linear models; tree models need
    no transformation at all.

    log1p is used instead of log to safely handle any zero values.
    """
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            logger.warning(f"log1p: column '{col}' not found - skipping.")
            continue
        if (df[col] < 0).any():
            logger.warning(
                f"log1p: '{col}' has negative values - skipping to avoid NaN."
            )
            continue
        before_skew = df[col].skew()
        df[col] = np.log1p(df[col])
        after_skew = df[col].skew()
        logger.info(
            f"log1p '{col}': skew {before_skew:.3f} -> {after_skew:.3f}"
        )
    return df


# =============================================================================
# STEP 4 - LOF OUTLIER FLAG
# =============================================================================

def fit_lof(
    train: pd.DataFrame,
    features: list[str] = _FALLBACK_LOF_FEATURES,
    contamination: float = _FALLBACK_LOF_CONTAMINATION,
    n_neighbors: int = _FALLBACK_LOF_N_NEIGHBORS,
) -> tuple[Optional[LocalOutlierFactor], list[str]]:
    """
    Fit LOF on train set and return (fitted_lof, used_features).

    `features`, `contamination`, and `n_neighbors` should come from
    EdaConfig (data_config.yaml -> eda_derived.lof), not the fallback
    defaults - those are last-resort values only.

    LOF is fit-only on train; for val/test we use .predict() (novelty=True).

    Returns (None, available) if there aren't enough rows to fit
    (len(lof_data) <= n_neighbors) - caller must handle None.
    """
    available = [c for c in features if c in train.columns]
    missing   = set(features) - set(available)
    if missing:
        logger.warning(f"LOF: features not found and skipped: {missing}")

    lof_data = train[available].dropna()

    # Guard: LOF requires at least n_neighbors + 1 rows to fit
    if len(lof_data) <= n_neighbors:
        logger.warning(
            f"LOF skipped: only {len(lof_data)} rows available after dropna, "
            f"need > n_neighbors={n_neighbors}. "
            "lof_outlier column will be set to -99 (unknown) for all rows."
        )
        return None, available   # caller checks for None

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,          # novelty=True allows predict on new data
    )
    lof.fit(lof_data)
    logger.info(
        f"LOF fit on {len(lof_data):,} train rows | "
        f"features={available} | contamination={contamination} | "
        f"n_neighbors={n_neighbors}"
    )
    return lof, available


def apply_lof_flag(
    df: pd.DataFrame,
    lof: Optional[LocalOutlierFactor],
    features: list[str],
) -> pd.DataFrame:
    """
    Add `lof_outlier` flag (1 = outlier) using a train-fit LOF model.

    Rows with nulls in LOF features are flagged as -99 (unknown).
    If lof is None (skipped due to insufficient rows), all rows get -99.
    """
    df = df.copy()

    if lof is None:
        df["lof_outlier"] = -99
        logger.warning("LOF model is None - all rows flagged as -99 (unknown).")
        return df

    lof_data = df[features].dropna()

    predictions = lof.predict(lof_data)          # -1 = outlier, +1 = inlier
    flag_series = pd.Series(
        (predictions == -1).astype(int),
        index=lof_data.index,
    )

    df["lof_outlier"] = -99                       # default: unknown (null rows)
    df.loc[flag_series.index, "lof_outlier"] = flag_series

    n_outliers = (df["lof_outlier"] == 1).sum()
    n_unknown  = (df["lof_outlier"] == -99).sum()
    logger.info(
        f"LOF flag applied: {n_outliers} outliers, {n_unknown} unknown (nulls in features)"
    )
    return df


# =============================================================================
# ARTIFACTS - save/load train-fit statistics
# =============================================================================

def save_artifacts(
    imputer_stats: dict,
    lof: Optional[LocalOutlierFactor],
    lof_features: list[str],
    eda_config: EdaConfig,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Persist train-fit statistics to artifacts/cleaning_artifacts.json.

    These are needed to apply the same transformations to val/test and
    to new production data without re-fitting. Includes the EdaConfig
    values actually used for this run (for full reproducibility).
    """
    import pickle

    artifacts_dir = output_dir or (PROJECT_DIR / "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Save LOF model as pickle (may be None if LOF was skipped)
    lof_path = artifacts_dir / "lof_model.pkl"
    with open(lof_path, "wb") as f:
        pickle.dump(lof, f)

    # Save JSON-serialisable stats - uses the EdaConfig actually used this run
    meta = {
        "imputer_stats": imputer_stats,
        "lof_features":  lof_features,
        "lof_contamination": eda_config.lof_contamination,
        "lof_n_neighbors": eda_config.lof_n_neighbors,
        "log1p_cols":    eda_config.log1p_columns,
        "cap_threshold": eda_config.cap_threshold,
        "eda_config_source": eda_config.source,
        "timestamp":     datetime.now().isoformat(),
    }
    meta_path = artifacts_dir / "cleaning_artifacts.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Artifacts saved -> {artifacts_dir}")
    return meta_path


def load_artifacts(artifacts_dir: Optional[Path] = None) -> tuple[dict, LocalOutlierFactor, list[str]]:
    """
    Load train-fit statistics from artifacts/.

    Returns (imputer_stats, lof_model, lof_features).
    """
    import pickle

    artifacts_dir = artifacts_dir or (PROJECT_DIR / "artifacts")
    meta_path = artifacts_dir / "cleaning_artifacts.json"
    lof_path  = artifacts_dir / "lof_model.pkl"

    if not meta_path.exists():
        raise FileNotFoundError(f"Cleaning artifacts not found: {meta_path}")
    if not lof_path.exists():
        raise FileNotFoundError(f"LOF model artifact not found: {lof_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    with open(lof_path, "rb") as f:
        lof = pickle.load(f)

    logger.info(f"Artifacts loaded from {artifacts_dir}")
    return meta["imputer_stats"], lof, meta["lof_features"]


# =============================================================================
# SAVE TO DISK (data/processed/)
# =============================================================================

def save_cleaned_splits(
    result: CleaningResult,
    output_dir: Optional[Path] = None,
) -> CleaningResult:
    """Save cleaned CSVs to data/processed/."""
    out = output_dir or ensure_path("processed")
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out / "train_clean.csv",
        "val":   out / "val_clean.csv",
        "test":  out / "test_clean.csv",
    }

    result.train.to_csv(paths["train"], index=False)
    result.val.to_csv(paths["val"],   index=False)
    result.test.to_csv(paths["test"], index=False)

    result.train_path = paths["train"]
    result.val_path   = paths["val"]
    result.test_path  = paths["test"]

    for name, path in paths.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {name}_clean.csv -> {path} ({size_mb:.2f} MB)")

    return result


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================

class CleaningError(Exception):
    """Raised on unrecoverable cleaning failures (e.g. critical DVC steps)."""
    pass


# =============================================================================
# DVC TRACKING
# =============================================================================

def _run_cmd(
    cmd: list[str],
    cwd: Path,
    timeout: int = 120,
    raise_on_failure: bool = False,
) -> bool:
    """
    Run a subprocess command.

    Parameters
    ----------
    raise_on_failure : If True, raises CleaningError on non-zero return code.
                       Use for critical steps (dvc add) where silent failure
                       would leave the pipeline in an inconsistent state.
                       False (default) for best-effort steps (dvc push, git push).
    """
    try:
        r = subprocess.run(
            cmd, cwd=cwd, capture_output=True,
            text=True, errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        msg = f"Command timed out after {timeout}s: {' '.join(cmd)}"
        logger.error(msg)
        if raise_on_failure:
            raise CleaningError(msg)
        return False
    except Exception as e:
        msg = f"Command raised unexpected error: {' '.join(cmd)} - {e}"
        logger.error(msg)
        if raise_on_failure:
            raise CleaningError(msg)
        return False

    if r.returncode != 0:
        msg = f"Command failed: {' '.join(cmd)}\n  stderr: {r.stderr.strip()}"
        if raise_on_failure:
            logger.error(msg)
            raise CleaningError(msg)
        logger.warning(msg)
        return False

    return True


def track_processed_with_dvc(
    result: CleaningResult,
    remote_name: str = DVC_REMOTE_NAME,
    timeout: int = 300,
) -> bool:
    """Track data/processed/ with DVC and push to remote."""
    if not is_dvc_initialized():
        logger.warning("DVC not initialized - skipping tracking.")
        return False

    if not result.train_path:
        logger.warning("No paths in CleaningResult - call save_cleaned_splits() first.")
        return False

    processed_dir = result.train_path.parent
    try:
        rel = processed_dir.relative_to(PROJECT_DIR)
    except ValueError:
        logger.error(f"{processed_dir} is outside PROJECT_DIR.")
        return False

    # dvc add is CRITICAL - if it fails the pointer file won't exist
    logger.info(f"DVC add: {rel}")
    _run_cmd(
        ["dvc", "add", str(rel)],
        cwd=PROJECT_DIR,
        timeout=timeout,
        raise_on_failure=True,   # raises CleaningError on failure
    )

    # dvc push is best-effort - data is saved locally even if remote push fails
    logger.info(f"DVC push -> {remote_name}")
    push_ok = _run_cmd(
        ["dvc", "push", f"--remote={remote_name}", str(rel)],
        cwd=PROJECT_DIR,
        timeout=timeout,
        raise_on_failure=False,
    )
    if not push_ok:
        logger.warning(
            "DVC push failed - data saved locally but NOT on remote. "
            "Run `dvc push` manually when the remote is available."
        )
        result.warnings.append("DVC push failed - run `dvc push` manually.")

    configure_git_identity(PROJECT_DIR)
    gitignore = rel.parent / ".gitignore"
    _run_cmd(
        ["git", "add", f"{rel}.dvc", str(gitignore)],
        cwd=PROJECT_DIR,
        timeout=30,
        raise_on_failure=False,
    )

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_DIR, capture_output=True,
        text=True, errors="replace", timeout=15,
    )
    if not status.stdout.strip():
        logger.info("No new git changes - processed already tracked.")
        result.dvc_tracked = True
        return True

    _run_cmd(
        ["git", "commit", "-m", "data: track processed cleaned splits with DVC"],
        cwd=PROJECT_DIR,
        timeout=60,
        raise_on_failure=False,
    )
    ok = _run_cmd(
        ["git", "push"],
        cwd=PROJECT_DIR,
        timeout=120,
        raise_on_failure=False,
    )
    if ok:
        logger.info("Processed splits tracked and pushed to DVC remote.")
        result.dvc_tracked = True
    else:
        logger.error(
            "git push failed - .dvc pointer committed locally but not on GitHub. "
            "Run `git push` manually."
        )
        result.warnings.append("git push failed - run `git push` manually.")
    return ok


# =============================================================================
# MAIN PIPELINE ENTRY POINT
# =============================================================================

def run_cleaning(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    config_path: str | Path = "configs/data_config.yaml",
    output_dir: Optional[Path] = None,
    auto_track_dvc: bool = True,
    save_artifacts_flag: bool = True,
) -> CleaningResult:
    """
    Full cleaning pipeline: impute -> flag -> transform -> LOF -> save -> DVC.

    All EDA-derived parameters (log1p columns, cap threshold, LOF settings)
    are loaded from `config_path` -> eda_derived section. If that section is
    missing, fallback defaults are used and CleaningResult.eda_config_source
    will read "fallback" - check this before trusting the output.

    FIT/TRANSFORM RULE:
        All statistics are computed on `train` only.
        The same fitted objects are applied to `val` and `test`.

    Parameters
    ----------
    train / val / test  : Split DataFrames from splitting.py
    config_path         : Path to data_config.yaml (must contain eda_derived)
    output_dir          : Override for data/processed/
    auto_track_dvc      : Track output with DVC after saving
    save_artifacts_flag : Persist train-fit stats to artifacts/

    Returns
    -------
    CleaningResult with all three cleaned DataFrames and file paths.

    Usage
    -----
        from src.data.data_loader import DataLoader
        from src.data.cleaning import run_cleaning

        loader = DataLoader()
        train  = loader.load_interim("train.csv")
        val    = loader.load_interim("val.csv")
        test   = loader.load_interim("test.csv")

        result = run_cleaning(train, val, test)
        train_clean = result.train
    """
    logger.info("=" * 60)
    logger.info("  Data cleaning started")
    logger.info("=" * 60)

    # -- Input guards ----------------------------------------------------------
    for name, df in [("train", train), ("val", val), ("test", test)]:
        if df.empty:
            raise CleaningError(f"Input '{name}' DataFrame is empty - cannot clean.")
    if _TARGET not in train.columns:
        raise CleaningError(
            f"Target column '{_TARGET}' not found in train. "
            "Ensure splitting.py ran successfully before cleaning."
        )

    # -- Load EDA-derived config (single source of truth) --------------------
    eda_config = load_eda_config(config_path)
    if eda_config.source == "fallback":
        logger.warning(
            "Cleaning is running with FALLBACK EDA defaults, not data_config.yaml. "
            "Results may not reflect the latest EDA findings."
        )

    # -- Step 1: Imputation ------------------------------------------------------
    logger.info("Step 1/4 - Imputation")
    imputer_stats = fit_imputer(train)
    train = apply_imputer(train, imputer_stats)
    val   = apply_imputer(val,   imputer_stats)
    test  = apply_imputer(test,  imputer_stats)

    # -- Step 2: is_capped flag --------------------------------------------------
    logger.info("Step 2/4 - is_capped flag")
    train = add_is_capped_flag(train, cap_threshold=eda_config.cap_threshold)
    val   = add_is_capped_flag(val,   cap_threshold=eda_config.cap_threshold)
    test  = add_is_capped_flag(test,  cap_threshold=eda_config.cap_threshold)

    # -- Step 3: log1p transform ---------------------------------------------------
    logger.info("Step 3/4 - log1p transform (count columns only)")
    train = apply_log1p(train, cols=eda_config.log1p_columns)
    val   = apply_log1p(val,   cols=eda_config.log1p_columns)
    test  = apply_log1p(test,  cols=eda_config.log1p_columns)

    # -- Step 4: LOF outlier flag ----------------------------------------------------
    logger.info("Step 4/4 - LOF outlier flag (fit on train)")
    lof_model, lof_features_used = fit_lof(
        train,
        features=eda_config.lof_features,
        contamination=eda_config.lof_contamination,
        n_neighbors=eda_config.lof_n_neighbors,
    )
    train = apply_lof_flag(train, lof_model, lof_features_used)
    val   = apply_lof_flag(val,   lof_model, lof_features_used)
    test  = apply_lof_flag(test,  lof_model, lof_features_used)

    n_lof_train = int((train["lof_outlier"] == 1).sum())

    # -- Assemble result ---------------------------------------------------------
    result = CleaningResult(
        train=train,
        val=val,
        test=test,
        imputation_median=imputer_stats.get("total_bedrooms"),
        lof_n_outliers_train=n_lof_train,
        eda_config_source=eda_config.source,
    )

    # -- Save artifacts ------------------------------------------------------------
    if save_artifacts_flag:
        save_artifacts(imputer_stats, lof_model, lof_features_used, eda_config)

    # -- Save CSVs ----------------------------------------------------------------
    result = save_cleaned_splits(result, output_dir=output_dir)

    # -- DVC tracking --------------------------------------------------------------
    if auto_track_dvc:
        track_processed_with_dvc(result)
    else:
        logger.info("auto_track_dvc=False - skipping DVC tracking.")

    logger.info("\n" + result.summary())
    return result


# =============================================================================
# CLI  ->  python -m src.data.cleaning
# =============================================================================

if __name__ == "__main__":
    import sys
    from src.data.data_loader import DataLoader
    from src.data.validation import validate_dataframe, ValidationError

    config = "configs/data_config.yaml"
    loader = DataLoader()

    logger.info("Loading interim splits...")
    train = loader.load_interim("train.csv")
    val   = loader.load_interim("val.csv")
    test  = loader.load_interim("test.csv")

    try:
        validate_dataframe(train, config_path=config, raise_on_failure=True)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

    result = run_cleaning(train, val, test, config_path=config)
    print(result.summary())
    sys.exit(0)


__all__ = [
    "CleaningError",
    "CleaningResult",
    "EdaConfig",
    "load_eda_config",
    "fit_imputer",
    "apply_imputer",
    "add_is_capped_flag",
    "apply_log1p",
    "fit_lof",
    "apply_lof_flag",
    "save_artifacts",
    "load_artifacts",
    "run_cleaning",
]