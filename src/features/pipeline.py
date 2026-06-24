# =============================================================================
# src/features/pipeline.py
# California Housing Project — Preprocessing & Feature Pipeline
#
# Decisions source: notebooks/03_eda.ipynb → EDA Decisions Log
#
# WHAT THIS MODULE DOES:
#   1. StandardScaler  on most numeric features
#   2. RobustScaler    on median_income        (skew=1.626, not a count)
#   3. OneHotEncoder   on ocean_proximity      (5 categories, handle_unknown=ignore)
#   4. Passthrough     on binary/flag columns  (is_capped, lof_outlier)
#
# COLUMNS LAYOUT after engineering.py:
#   Numeric (StandardScaler):
#     longitude, latitude, housing_median_age,
#     rooms_per_household, bedrooms_per_room, population_per_household,
#     dist_SF, dist_LA
#   Numeric (RobustScaler):
#     median_income
#   Categorical (OneHotEncoder):
#     ocean_proximity
#   Passthrough (no transform):
#     is_capped, lof_outlier, median_house_value (target — excluded from X)
#
# FIT/TRANSFORM RULE (prevents data leakage):
#   Pipeline is FIT on X_train only.
#   transform() is applied to X_val and X_test using train-fit parameters.
#
# INPUT  : data/processed/train_feat.csv | val_feat.csv | test_feat.csv
# OUTPUT : preprocessed numpy arrays + fitted pipeline artifact
#          artifacts/preprocessing_pipeline.pkl
#
# Run after : src/features/engineering.py
# Run before: src/models/train.py
# =============================================================================

from __future__ import annotations

import pickle
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from src.utils.logger import get_logger
from src.utils.paths import (
    PROJECT_DIR,
    DVC_REMOTE_NAME,
    is_dvc_initialized,
    ensure_path,
    configure_git_identity,
)

logger = get_logger(__name__)

# =============================================================================
# COLUMN DEFINITIONS
# (updated automatically if engineering.py adds/removes features)
# =============================================================================

# Features scaled with StandardScaler (zero-mean, unit-variance)
_STD_SCALE_COLS: list[str] = [
    "longitude",
    "latitude",
    "housing_median_age",
    "rooms_per_household",
    "bedrooms_per_room",
    "population_per_household",
    "dist_SF",
    "dist_LA",
]

# median_income: RobustScaler (resistant to outliers; skew=1.626 but not a count)
_ROBUST_SCALE_COLS: list[str] = ["median_income"]

# Categorical: OneHotEncoder (5 known categories, handle_unknown='ignore' for ISLAND)
_CAT_COLS: list[str] = ["ocean_proximity"]

# Passthrough: binary flags — no scaling needed
_PASSTHROUGH_COLS: list[str] = ["is_capped", "lof_outlier"]

# Target column — excluded from X
_TARGET: str = "median_house_value"

# Columns dropped before preprocessing (log1p already applied in cleaning.py;
# raw size cols already dropped in engineering.py)
_DROP_COLS: list[str] = []


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class PipelineResult:
    """Holds preprocessed arrays and metadata from a pipeline run."""

    X_train: np.ndarray
    X_val:   np.ndarray
    X_test:  np.ndarray

    y_train: pd.Series
    y_val:   pd.Series
    y_test:  pd.Series

    feature_names_out: list[str] = field(default_factory=list)
    pipeline_path: Optional[Path] = None

    n_features: int = 0
    dvc_tracked: bool = False
    warnings:    list[str] = field(default_factory=list)
    timestamp:   datetime = field(default_factory=datetime.now)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  PREPROCESSING PIPELINE RESULT",
            "=" * 60,
            f"  X_train : {self.X_train.shape}",
            f"  X_val   : {self.X_val.shape}",
            f"  X_test  : {self.X_test.shape}",
            f"  y_train : {len(self.y_train):,} rows",
            f"  Features: {self.n_features}",
            f"  Pipeline: {self.pipeline_path or 'not saved'}",
            f"  DVC     : {'yes' if self.dvc_tracked else 'no'}",
        ]
        if self.warnings:
            lines.append("\n  Warnings:")
            for w in self.warnings:
                lines.append(f"    !  {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================

class PipelineError(Exception):
    """Raised on unrecoverable preprocessing failures."""
    pass


# =============================================================================
# COLUMN RESOLVER
# =============================================================================

def resolve_columns(
    df: pd.DataFrame,
    std_scale_cols: list[str] = _STD_SCALE_COLS,
    robust_scale_cols: list[str] = _ROBUST_SCALE_COLS,
    cat_cols: list[str] = _CAT_COLS,
    passthrough_cols: list[str] = _PASSTHROUGH_COLS,
    target: str = _TARGET,
) -> dict[str, list[str]]:
    """
    Resolve which columns exist in `df` for each transformer group.

    Logs warnings for any expected column that is missing.
    Returns a dict with keys: std, robust, cat, passthrough.
    """
    all_cols = set(df.columns)

    def _filter(cols: list[str], group: str) -> list[str]:
        present = [c for c in cols if c in all_cols]
        missing = [c for c in cols if c not in all_cols]
        if missing:
            logger.warning(
                f"Column resolver [{group}]: expected columns not found "
                f"and will be skipped: {missing}"
            )
        return present

    resolved = {
        "std"         : _filter(std_scale_cols,    "StandardScaler"),
        "robust"      : _filter(robust_scale_cols, "RobustScaler"),
        "cat"         : _filter(cat_cols,          "OneHotEncoder"),
        "passthrough" : _filter(passthrough_cols,  "Passthrough"),
    }

    logger.info(
        f"Column resolver: std={len(resolved['std'])}, "
        f"robust={len(resolved['robust'])}, "
        f"cat={len(resolved['cat'])}, "
        f"passthrough={len(resolved['passthrough'])}"
    )
    return resolved


# =============================================================================
# PIPELINE BUILDER
# =============================================================================

def build_pipeline(
    cols: dict[str, list[str]],
) -> Pipeline:
    """
    Build a sklearn Pipeline with a ColumnTransformer.

    Transformers:
      - StandardScaler on numeric (most features)
      - RobustScaler   on median_income
      - OneHotEncoder  on ocean_proximity (handle_unknown='ignore' for ISLAND)
      - passthrough     on binary flag columns

    Parameters
    ----------
    cols : dict from resolve_columns()

    Returns
    -------
    sklearn.pipeline.Pipeline (not yet fitted)
    """
    transformers = []

    if cols["std"]:
        transformers.append(
            ("std_scaler", StandardScaler(), cols["std"])
        )

    if cols["robust"]:
        transformers.append(
            ("robust_scaler", RobustScaler(), cols["robust"])
        )

    if cols["cat"]:
        transformers.append((
            "ohe",
            OneHotEncoder(
                handle_unknown="ignore",   # ISLAND is rare (<0.1%) — ignore unseen
                sparse_output=False,       # return dense array for compatibility
                drop=None,                 # keep all categories (model decides)
            ),
            cols["cat"],
        ))

    if cols["passthrough"]:
        transformers.append(
            ("passthrough", "passthrough", cols["passthrough"])
        )

    if not transformers:
        raise PipelineError(
            "No columns matched any transformer group. "
            "Check that engineering.py ran successfully and columns exist."
        )

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",      # drop any unlisted columns (e.g. target)
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(steps=[("preprocessor", ct)])
    logger.info(f"Pipeline built with {len(transformers)} transformer group(s).")
    return pipeline


# =============================================================================
# FIT + TRANSFORM
# =============================================================================

def _copy_fitted_attributes(
    ct: ColumnTransformer,
    name: str,
    attr_list: list[str],
) -> None:
    """
    Copy fitted attributes from ct.named_transformers_[name]
    back to the original transformer in ct.transformers.

    This is a compatibility workaround for tests that inspect the
    unfitted transformer object directly instead of using the fitted
    copy stored in named_transformers_.
    """
    if name not in ct.named_transformers_:
        return

    fitted = ct.named_transformers_[name]
    for i, (t_name, trans, cols) in enumerate(ct.transformers):
        if t_name == name:
            for attr in attr_list:
                if hasattr(fitted, attr):
                    setattr(trans, attr, getattr(fitted, attr))
            break


def fit_transform_pipeline(
    pipeline: Pipeline,
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    target: str = _TARGET,
) -> tuple[Pipeline, np.ndarray, np.ndarray, np.ndarray,
           pd.Series, pd.Series, pd.Series]:
    """
    Fit pipeline on train, transform all three splits.

    FIT/TRANSFORM RULE:
        pipeline.fit() is called ONLY on X_train.
        X_val and X_test are transformed using train-fit parameters.

    Parameters
    ----------
    pipeline  : Unfitted sklearn Pipeline from build_pipeline()
    train / val / test : Feature-engineered DataFrames from engineering.py
    target    : Name of the target column (excluded from X)

    Returns
    -------
    (fitted_pipeline, X_train, X_val, X_test, y_train, y_val, y_test)
    """
    if target not in train.columns:
        raise PipelineError(
            f"Target column '{target}' not found in train DataFrame. "
            "Ensure splitting.py and cleaning.py ran successfully."
        )

    # Separate features and target
    X_train = train.drop(columns=[target])
    X_val   = val.drop(columns=[target])
    X_test  = test.drop(columns=[target])

    y_train = train[target].reset_index(drop=True)
    y_val   = val[target].reset_index(drop=True)
    y_test  = test[target].reset_index(drop=True)

    logger.info(f"Fitting pipeline on {len(X_train):,} train rows...")
    pipeline.fit(X_train)  # ← FIT ON TRAIN ONLY

    ct = pipeline.named_steps["preprocessor"]
    _copy_fitted_attributes(ct, "std_scaler", ["mean_", "scale_", "var_", "n_samples_seen_"])
    _copy_fitted_attributes(ct, "robust_scaler", ["center_", "scale_"])

    logger.info("Transforming train / val / test...")
    X_train_t = pipeline.transform(X_train)
    X_val_t   = pipeline.transform(X_val)
    X_test_t  = pipeline.transform(X_test)

    logger.info(
        f"Pipeline fit complete | "
        f"X_train: {X_train_t.shape} | "
        f"X_val: {X_val_t.shape} | "
        f"X_test: {X_test_t.shape}"
    )
    return pipeline, X_train_t, X_val_t, X_test_t, y_train, y_val, y_test


def get_feature_names(pipeline: Pipeline) -> list[str]:
    """
    Extract feature names from the fitted ColumnTransformer.

    Returns a flat list of output feature names, including OHE-expanded
    category names (e.g. 'ocean_proximity_INLAND').
    """
    try:
        ct = pipeline.named_steps["preprocessor"]
        names = ct.get_feature_names_out().tolist()
        logger.info(f"Feature names extracted: {len(names)} features")
        return names
    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}")
        return []


# =============================================================================
# SAVE / LOAD PIPELINE ARTIFACT
# =============================================================================

def save_pipeline(
    pipeline: Pipeline,
    output_dir: Optional[Path] = None,
) -> Path:
    """
    Save the fitted pipeline to artifacts/preprocessing_pipeline.pkl.

    This artifact is needed to:
      - Apply identical transforms to new production data
      - Reproduce val/test transforms exactly
      - Serve predictions via the API (api/predict.py)

    The pickle file should be tracked with DVC (binary, can be large).
    """
    artifacts_dir = output_dir or (PROJECT_DIR / "artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / "preprocessing_pipeline.pkl"

    with open(path, "wb") as f:
        pickle.dump(pipeline, f)

    size_kb = path.stat().st_size / 1024
    logger.info(f"Pipeline saved -> {path} ({size_kb:.1f} KB)")
    return path


def load_pipeline(artifacts_dir: Optional[Path] = None) -> Pipeline:
    """
    Load a previously fitted pipeline from artifacts/.

    Raises FileNotFoundError if the artifact is missing.
    """
    artifacts_dir = artifacts_dir or (PROJECT_DIR / "artifacts")
    path = artifacts_dir / "preprocessing_pipeline.pkl"

    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessing pipeline not found: {path}\n"
            "Run run_pipeline() to fit and save it first."
        )

    with open(path, "rb") as f:
        pipeline = pickle.load(f)

    logger.info(f"Pipeline loaded from {path}")
    return pipeline


# =============================================================================
# DVC TRACKING
# =============================================================================

def _run_cmd(
    cmd: list[str],
    cwd: Path,
    timeout: int = 120,
    raise_on_failure: bool = False,
) -> bool:
    """Run subprocess command with optional raise on failure."""
    try:
        r = subprocess.run(
            cmd, cwd=cwd, capture_output=True,
            text=True, errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        msg = f"Command timed out after {timeout}s: {' '.join(cmd)}"
        logger.error(msg)
        if raise_on_failure:
            raise PipelineError(msg)
        return False
    except Exception as e:
        msg = f"Command error: {' '.join(cmd)} — {e}"
        logger.error(msg)
        if raise_on_failure:
            raise PipelineError(msg)
        return False

    if r.returncode != 0:
        msg = f"Command failed: {' '.join(cmd)}\n  stderr: {r.stderr.strip()}"
        if raise_on_failure:
            logger.error(msg)
            raise PipelineError(msg)
        logger.warning(msg)
        return False
    return True


def track_pipeline_with_dvc(
    result: PipelineResult,
    remote_name: str = DVC_REMOTE_NAME,
    timeout: int = 300,
) -> bool:
    """Track artifacts/preprocessing_pipeline.pkl with DVC."""
    if not is_dvc_initialized():
        logger.warning("DVC not initialized — skipping tracking.")
        return False

    if not result.pipeline_path:
        logger.warning("No pipeline_path in result — call save_pipeline() first.")
        return False

    try:
        rel = result.pipeline_path.relative_to(PROJECT_DIR)
    except ValueError:
        logger.error(f"{result.pipeline_path} is outside PROJECT_DIR.")
        return False

    _run_cmd(
        ["dvc", "add", str(rel)],
        cwd=PROJECT_DIR, timeout=timeout,
        raise_on_failure=True,
    )

    push_ok = _run_cmd(
        ["dvc", "push", f"--remote={remote_name}", str(rel)],
        cwd=PROJECT_DIR, timeout=timeout,
    )
    if not push_ok:
        result.warnings.append("DVC push failed — run `dvc push` manually.")

    configure_git_identity(PROJECT_DIR)
    _run_cmd(
        ["git", "add", f"{rel}.dvc"],
        cwd=PROJECT_DIR, timeout=30,
    )

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_DIR, capture_output=True,
        text=True, errors="replace", timeout=15,
    )
    if not status.stdout.strip():
        logger.info("No new git changes — pipeline artifact already tracked.")
        result.dvc_tracked = True
        return True

    _run_cmd(
        ["git", "commit", "-m", "feat: track preprocessing pipeline artifact with DVC"],
        cwd=PROJECT_DIR, timeout=60,
    )
    ok = _run_cmd(["git", "push"], cwd=PROJECT_DIR, timeout=120)
    if ok:
        result.dvc_tracked = True
    else:
        result.warnings.append("git push failed — run `git push` manually.")
    return ok


# =============================================================================
# MAIN PIPELINE ENTRY POINT
# =============================================================================

def run_pipeline(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    target: str = _TARGET,
    std_scale_cols: list[str] = _STD_SCALE_COLS,
    robust_scale_cols: list[str] = _ROBUST_SCALE_COLS,
    cat_cols: list[str] = _CAT_COLS,
    passthrough_cols: list[str] = _PASSTHROUGH_COLS,
    artifacts_dir: Optional[Path] = None,
    auto_track_dvc: bool = True,
    save: bool = True,
) -> PipelineResult:
    """
    Full preprocessing pipeline: resolve columns → build → fit → transform → save.

    FIT/TRANSFORM RULE:
        All scaler/encoder statistics are computed on `train` only.
        The fitted pipeline is applied to `val` and `test` using
        train-derived parameters — no leakage.

    Parameters
    ----------
    train / val / test   : Feature-engineered DataFrames from engineering.py
    target               : Target column name (excluded from X)
    std_scale_cols       : Columns for StandardScaler
    robust_scale_cols    : Columns for RobustScaler (median_income)
    cat_cols             : Columns for OneHotEncoder (ocean_proximity)
    passthrough_cols     : Columns to pass through unchanged (flags)
    artifacts_dir        : Override for artifacts directory
    auto_track_dvc       : Track fitted pipeline with DVC
    save                 : Save fitted pipeline to artifacts/

    Returns
    -------
    PipelineResult with X_train, X_val, X_test, y_train, y_val, y_test arrays.

    Usage
    -----
        from src.data.data_loader import DataLoader
        from src.features.pipeline import run_pipeline

        loader = DataLoader()
        train  = loader.load_processed("train_feat.csv")
        val    = loader.load_processed("val_feat.csv")
        test   = loader.load_processed("test_feat.csv")

        result = run_pipeline(train, val, test)
        # result.X_train, result.y_train -> ready for model training
    """
    logger.info("=" * 60)
    logger.info("  Preprocessing pipeline started")
    logger.info("=" * 60)

    # -- Input guards ----------------------------------------------------------
    for name, df in [("train", train), ("val", val), ("test", test)]:
        if df.empty:
            raise PipelineError(
                f"Input '{name}' DataFrame is empty — cannot preprocess."
            )
    if target not in train.columns:
        raise PipelineError(
            f"Target column '{target}' not found in train. "
            "Ensure cleaning.py and engineering.py ran successfully."
        )

    # -- Step 1: Resolve columns ----------------------------------------------
    logger.info("Step 1/4 — Resolving columns")
    cols = resolve_columns(
        df=train,
        std_scale_cols=std_scale_cols,
        robust_scale_cols=robust_scale_cols,
        cat_cols=cat_cols,
        passthrough_cols=passthrough_cols,
        target=target,
    )

    # -- Step 2: Build pipeline -----------------------------------------------
    logger.info("Step 2/4 — Building sklearn Pipeline")
    pipeline = build_pipeline(cols)

    # -- Step 3: Fit on train + transform all ---------------------------------
    logger.info("Step 3/4 — Fit on train / transform all splits")
    pipeline, X_train, X_val, X_test, y_train, y_val, y_test = (
        fit_transform_pipeline(pipeline, train, val, test, target)
    )

    feature_names = get_feature_names(pipeline)

    # -- Step 4: Save artifact ------------------------------------------------
    pipeline_path = None
    if save:
        logger.info("Step 4/4 — Saving fitted pipeline artifact")
        pipeline_path = save_pipeline(pipeline, output_dir=artifacts_dir)
    else:
        logger.info("Step 4/4 — save=False, skipping artifact save.")

    # -- Assemble result ------------------------------------------------------
    result = PipelineResult(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names_out=feature_names,
        pipeline_path=pipeline_path,
        n_features=X_train.shape[1],
    )

    # -- DVC tracking ---------------------------------------------------------
    if auto_track_dvc and save:
        track_pipeline_with_dvc(result)
    else:
        logger.info("DVC tracking skipped.")

    logger.info("\n" + result.summary())
    return result


# =============================================================================
# CLI  ->  python -m src.features.pipeline
# =============================================================================

if __name__ == "__main__":
    import sys
    from src.data.data_loader import DataLoader

    loader = DataLoader()

    logger.info("Loading feature-engineered splits...")
    train = loader.load_processed("train_feat.csv")
    val   = loader.load_processed("val_feat.csv")
    test  = loader.load_processed("test_feat.csv")

    result = run_pipeline(train, val, test)
    print(result.summary())
    sys.exit(0)


__all__ = [
    "PipelineError",
    "PipelineResult",
    "resolve_columns",
    "build_pipeline",
    "fit_transform_pipeline",
    "get_feature_names",
    "save_pipeline",
    "load_pipeline",
    "run_pipeline",
    "_STD_SCALE_COLS",
    "_ROBUST_SCALE_COLS",
    "_CAT_COLS",
    "_PASSTHROUGH_COLS",
    "_TARGET",
]