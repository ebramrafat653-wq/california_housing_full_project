# =============================================================================
# src/features/engineering.py
# California Housing Project — Feature Engineering
#
# Decisions source: configs/data_config.yaml -> eda_derived.engineered_features
#
# WHAT THIS MODULE DOES:
#   1. Create ratio features      (from EDA multicollinearity analysis)
#   2. Create distance features   (from EDA geographic analysis)
#   3. Drop raw size columns      (replaced by ratios — reduces multicollinearity)
#
# FEATURES CREATED (from EDA VIF analysis — all VIF < 10):
#   ratios:
#     rooms_per_household      = total_rooms / households        (VIF=5.459)
#     bedrooms_per_room        = total_bedrooms / total_rooms    (VIF=3.476)
#     population_per_household = population / households         (VIF=1.074)
#   distances:
#     dist_SF  = Euclidean distance to San Francisco (37.77, -122.42)
#     dist_LA  = Euclidean distance to Los Angeles   (34.05, -118.24)
#
# FEATURES DROPPED (after ratio creation):
#   total_rooms, total_bedrooms, population, households
#   These raw counts were highly multicollinear (inter-corr > 0.9)
#   and their signal is preserved in the ratio features.
#
# CONFIG-DRIVEN DESIGN:
#   All feature definitions are read from data_config.yaml -> eda_derived.
#   Re-running EDA and updating the YAML automatically updates this module.
#   Module-level constants are FALLBACK DEFAULTS ONLY.
#
# NO FIT NEEDED:
#   All transforms here are pure mathematical operations (division, sqrt).
#   No statistics are fit on train — this module is safe to apply to any split
#   without risk of data leakage.
#
# INPUT  : data/processed/train_clean.csv | val_clean.csv | test_clean.csv
# OUTPUT : data/processed/train_feat.csv  | val_feat.csv  | test_feat.csv
#          (same directory, DVC-tracked)
#
# Run after : src/data/cleaning.py
# Run before: src/features/pipeline.py  (scaling + encoding)
# =============================================================================

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml

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
# FALLBACK DEFAULTS
# (used only if data_config.yaml is missing eda_derived.engineered_features)
# =============================================================================

_FALLBACK_RATIOS: dict[str, str] = {
    "rooms_per_household"      : "total_rooms / households",
    "bedrooms_per_room"        : "total_bedrooms / total_rooms",
    "population_per_household" : "population / households",
}

_FALLBACK_DISTANCES: dict[str, dict[str, float]] = {
    "dist_SF": {"lat": 37.77, "lon": -122.42},
    "dist_LA": {"lat": 34.05, "lon": -118.24},
}

_FALLBACK_DROP_COLS: list[str] = [
    "total_rooms", "total_bedrooms", "population", "households",
]


# =============================================================================
# CONFIG LOADER
# =============================================================================

@dataclass
class FeatureConfig:
    """
    Typed container for feature engineering parameters.
    Populated from configs/data_config.yaml -> eda_derived.engineered_features.
    Falls back to module-level defaults if the section is missing.
    """
    ratios:    dict[str, str]              # name -> "col_a / col_b"
    distances: dict[str, dict[str, float]] # name -> {lat, lon}
    drop_cols: list[str]                   # raw cols to drop after ratio creation
    source:    str = "config"              # "config" or "fallback"


def load_feature_config(
    config_path: str | Path = "configs/data_config.yaml",
) -> FeatureConfig:
    """
    Load feature engineering parameters from data_config.yaml.

    Reads eda_derived.engineered_features section. Falls back to
    module-level defaults if the file or section is missing.

    Returns
    -------
    FeatureConfig with all parameters needed by the engineering steps.
    """
    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(
            f"Config not found: {config_path} — using fallback feature defaults."
        )
        return FeatureConfig(
            ratios=_FALLBACK_RATIOS,
            distances=_FALLBACK_DISTANCES,
            drop_cols=_FALLBACK_DROP_COLS,
            source="fallback",
        )

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    eng = cfg.get("eda_derived", {}).get("engineered_features")
    if not eng:
        logger.warning(
            f"No 'eda_derived.engineered_features' in '{config_path}' — "
            "using fallback feature defaults."
        )
        return FeatureConfig(
            ratios=_FALLBACK_RATIOS,
            distances=_FALLBACK_DISTANCES,
            drop_cols=_FALLBACK_DROP_COLS,
            source="fallback",
        )

    ratios    = eng.get("ratios",    _FALLBACK_RATIOS)
    distances = eng.get("distances", _FALLBACK_DISTANCES)
    drop_cols = eng.get("drop_after_engineering", _FALLBACK_DROP_COLS)

    logger.info(
        f"Feature config loaded: {len(ratios)} ratios, "
        f"{len(distances)} distances, {len(drop_cols)} cols to drop"
    )
    return FeatureConfig(
        ratios=ratios,
        distances=distances,
        drop_cols=drop_cols,
        source="config",
    )


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class EngineeringResult:
    """Holds feature-engineered DataFrames and a run summary."""

    train: pd.DataFrame
    val:   pd.DataFrame
    test:  pd.DataFrame

    train_path: Optional[Path] = None
    val_path:   Optional[Path] = None
    test_path:  Optional[Path] = None

    features_added:  list[str] = field(default_factory=list)
    features_dropped: list[str] = field(default_factory=list)
    feature_config_source: str = "config"

    timestamp:   datetime = field(default_factory=datetime.now)
    dvc_tracked: bool = False
    warnings:    list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "  FEATURE ENGINEERING RESULT SUMMARY",
            "=" * 60,
            f"  Train  : {len(self.train):,} rows x {self.train.shape[1]} cols",
            f"  Val    : {len(self.val):,} rows x {self.val.shape[1]} cols",
            f"  Test   : {len(self.test):,} rows x {self.test.shape[1]} cols",
            "",
            f"  Config source    : {self.feature_config_source}",
            f"  Features added   : {self.features_added}",
            f"  Features dropped : {self.features_dropped}",
            f"  DVC tracked      : {'yes' if self.dvc_tracked else 'no'}",
        ]
        if self.feature_config_source == "fallback":
            lines.append(
                "  WARNING: using fallback feature defaults, not data_config.yaml!"
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
# STEP 1 — RATIO FEATURES
# =============================================================================

def add_ratio_features(
    df: pd.DataFrame,
    ratios: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create ratio features from a config dict.

    Each entry is: feature_name -> "numerator_col / denominator_col"

    Guards:
    - Skips if numerator or denominator column is missing.
    - Skips if denominator has any zero values (would produce inf).
    - Clips inf values to NaN as a last resort.

    Returns
    -------
    (df_with_ratios, list_of_added_feature_names)
    """
    df = df.copy()
    added: list[str] = []

    for feat_name, formula in ratios.items():
        try:
            parts = [p.strip() for p in formula.split("/")]
            if len(parts) != 2:
                logger.warning(
                    f"Ratio formula '{formula}' must be 'col_a / col_b' — skipping."
                )
                continue

            num_col, den_col = parts

            if num_col not in df.columns:
                logger.warning(
                    f"Ratio '{feat_name}': numerator '{num_col}' not found — skipping."
                )
                continue
            if den_col not in df.columns:
                logger.warning(
                    f"Ratio '{feat_name}': denominator '{den_col}' not found — skipping."
                )
                continue

            # Check for zero denominator — happens in log1p-transformed data
            # log1p(0) = 0, so households or total_rooms could legitimately be 0
            n_zero_denom = (df[den_col] == 0).sum()
            if n_zero_denom > 0:
                logger.warning(
                    f"Ratio '{feat_name}': {n_zero_denom} zero values in '{den_col}'. "
                    "These rows will produce inf -> clipped to NaN."
                )

            df[feat_name] = df[num_col] / df[den_col]

            # Clip any inf produced by zero-division
            n_inf = np.isinf(df[feat_name]).sum()
            if n_inf > 0:
                df[feat_name] = df[feat_name].replace([np.inf, -np.inf], np.nan)
                logger.warning(
                    f"Ratio '{feat_name}': {n_inf} inf values clipped to NaN."
                )

            added.append(feat_name)
            logger.info(
                f"Created ratio '{feat_name}' = {num_col} / {den_col} | "
                f"mean={df[feat_name].mean():.3f} | nulls={df[feat_name].isna().sum()}"
            )

        except Exception as e:
            logger.warning(f"Ratio '{feat_name}' failed: {e} — skipping.")

    return df, added


# =============================================================================
# STEP 2 — DISTANCE FEATURES
# =============================================================================

def add_distance_features(
    df: pd.DataFrame,
    distances: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create Euclidean distance features from lat/lon to fixed hub coordinates.

    Distance is computed in degrees (not km) — sufficient for tree models
    as a relative spatial signal. Preprocessing.py will scale for linear models.

    Each entry in `distances` is: feature_name -> {lat: float, lon: float}

    Guards:
    - Skips if 'latitude' or 'longitude' columns are missing.
    - Skips if hub dict is missing 'lat' or 'lon' keys.

    Returns
    -------
    (df_with_distances, list_of_added_feature_names)
    """
    df = df.copy()
    added: list[str] = []

    if "latitude" not in df.columns or "longitude" not in df.columns:
        logger.warning(
            "Distance features skipped: 'latitude' or 'longitude' not found."
        )
        return df, added

    for feat_name, hub in distances.items():
        try:
            hub_lat = hub.get("lat")
            hub_lon = hub.get("lon")

            if hub_lat is None or hub_lon is None:
                logger.warning(
                    f"Distance '{feat_name}': hub dict missing 'lat' or 'lon' — skipping."
                )
                continue

            df[feat_name] = np.sqrt(
                (df["latitude"]  - hub_lat) ** 2 +
                (df["longitude"] - hub_lon) ** 2
            )
            added.append(feat_name)
            logger.info(
                f"Created distance '{feat_name}' to ({hub_lat}, {hub_lon}) | "
                f"mean={df[feat_name].mean():.3f}"
            )

        except Exception as e:
            logger.warning(f"Distance '{feat_name}' failed: {e} — skipping.")

    return df, added


# =============================================================================
# STEP 3 — DROP RAW SIZE COLUMNS
# =============================================================================

def drop_raw_columns(
    df: pd.DataFrame,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop raw size columns that have been replaced by ratio features.

    Only drops columns that actually exist — skips missing ones with a warning.

    Returns
    -------
    (df_without_raw_cols, list_of_actually_dropped_cols)
    """
    df = df.copy()
    actually_dropped: list[str] = []

    for col in drop_cols:
        if col not in df.columns:
            logger.warning(
                f"Drop: column '{col}' not found — already dropped or renamed?"
            )
            continue
        df = df.drop(columns=[col])
        actually_dropped.append(col)
        logger.info(f"Dropped raw column: '{col}'")

    return df, actually_dropped


# =============================================================================
# SAVE TO DISK
# =============================================================================

def save_featured_splits(
    result: EngineeringResult,
    output_dir: Optional[Path] = None,
) -> EngineeringResult:
    """
    Save feature-engineered CSVs to data/processed/.

    Files are saved alongside the cleaned files (same directory):
      train_feat.csv | val_feat.csv | test_feat.csv
    """
    out = output_dir or ensure_path("processed")
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out / "train_feat.csv",
        "val":   out / "val_feat.csv",
        "test":  out / "test_feat.csv",
    }

    result.train.to_csv(paths["train"], index=False)
    result.val.to_csv(paths["val"],   index=False)
    result.test.to_csv(paths["test"], index=False)

    result.train_path = paths["train"]
    result.val_path   = paths["val"]
    result.test_path  = paths["test"]

    for name, path in paths.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {name}_feat.csv -> {path} ({size_mb:.2f} MB)")

    return result


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
            raise FeatureEngineeringError(msg)
        return False
    except Exception as e:
        msg = f"Command error: {' '.join(cmd)} — {e}"
        logger.error(msg)
        if raise_on_failure:
            raise FeatureEngineeringError(msg)
        return False

    if r.returncode != 0:
        msg = f"Command failed: {' '.join(cmd)}\n  stderr: {r.stderr.strip()}"
        if raise_on_failure:
            logger.error(msg)
            raise FeatureEngineeringError(msg)
        logger.warning(msg)
        return False
    return True


def track_featured_with_dvc(
    result: EngineeringResult,
    remote_name: str = DVC_REMOTE_NAME,
    timeout: int = 300,
) -> bool:
    """Track data/processed/ with DVC after adding featured files."""
    if not is_dvc_initialized():
        logger.warning("DVC not initialized — skipping tracking.")
        return False

    if not result.train_path:
        logger.warning("No paths in EngineeringResult — call save_featured_splits() first.")
        return False

    processed_dir = result.train_path.parent
    try:
        rel = processed_dir.relative_to(PROJECT_DIR)
    except ValueError:
        logger.error(f"{processed_dir} is outside PROJECT_DIR.")
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
    gitignore = rel.parent / ".gitignore"
    _run_cmd(
        ["git", "add", f"{rel}.dvc", str(gitignore)],
        cwd=PROJECT_DIR, timeout=30,
    )

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_DIR, capture_output=True,
        text=True, errors="replace", timeout=15,
    )
    if not status.stdout.strip():
        logger.info("No new git changes — already tracked.")
        result.dvc_tracked = True
        return True

    _run_cmd(
        ["git", "commit", "-m", "data: track feature-engineered splits with DVC"],
        cwd=PROJECT_DIR, timeout=60,
    )
    ok = _run_cmd(["git", "push"], cwd=PROJECT_DIR, timeout=120)
    if ok:
        result.dvc_tracked = True
    else:
        result.warnings.append("git push failed — run `git push` manually.")
    return ok


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================

class FeatureEngineeringError(Exception):
    """Raised on unrecoverable feature engineering failures."""
    pass


# =============================================================================
# MAIN PIPELINE ENTRY POINT
# =============================================================================

def run_feature_engineering(
    train: pd.DataFrame,
    val:   pd.DataFrame,
    test:  pd.DataFrame,
    config_path: str | Path = "configs/data_config.yaml",
    output_dir: Optional[Path] = None,
    auto_track_dvc: bool = True,
) -> EngineeringResult:
    """
    Full feature engineering pipeline: ratios -> distances -> drop -> save -> DVC.

    NO DATA LEAKAGE RISK:
        All transforms are pure math (division, sqrt) — no statistics
        are fit on train. The same operations are safely applied to all splits.

    Parameters
    ----------
    train / val / test : Cleaned DataFrames from cleaning.py
    config_path        : Path to data_config.yaml (must contain eda_derived)
    output_dir         : Override for data/processed/
    auto_track_dvc     : Track output with DVC after saving

    Returns
    -------
    EngineeringResult with all three DataFrames and file paths.

    Usage
    -----
        from src.data.data_loader import DataLoader
        from src.features.engineering import run_feature_engineering

        loader = DataLoader()
        train  = loader.load_processed("train_clean.csv")
        val    = loader.load_processed("val_clean.csv")
        test   = loader.load_processed("test_clean.csv")

        result = run_feature_engineering(train, val, test)
        train_feat = result.train
    """
    logger.info("=" * 60)
    logger.info("  Feature engineering started")
    logger.info("=" * 60)

    # -- Input guards ----------------------------------------------------------
    for name, df in [("train", train), ("val", val), ("test", test)]:
        if df.empty:
            raise FeatureEngineeringError(
                f"Input '{name}' DataFrame is empty — cannot engineer features."
            )

    # -- Load config -----------------------------------------------------------
    feat_cfg = load_feature_config(config_path)
    if feat_cfg.source == "fallback":
        logger.warning(
            "Feature engineering running with FALLBACK defaults, not data_config.yaml."
        )

    all_added:   list[str] = []
    all_dropped: list[str] = []

    # -- Step 1: Ratio features -----------------------------------------------
    logger.info("Step 1/3 — Ratio features")
    train, added = add_ratio_features(train, feat_cfg.ratios)
    val,   _     = add_ratio_features(val,   feat_cfg.ratios)
    test,  _     = add_ratio_features(test,  feat_cfg.ratios)
    all_added.extend(added)

    # -- Step 2: Distance features --------------------------------------------
    logger.info("Step 2/3 — Distance features")
    train, added = add_distance_features(train, feat_cfg.distances)
    val,   _     = add_distance_features(val,   feat_cfg.distances)
    test,  _     = add_distance_features(test,  feat_cfg.distances)
    all_added.extend(added)

    # -- Step 3: Drop raw columns ---------------------------------------------
    logger.info("Step 3/3 — Drop raw size columns")
    train, dropped = drop_raw_columns(train, feat_cfg.drop_cols)
    val,   _       = drop_raw_columns(val,   feat_cfg.drop_cols)
    test,  _       = drop_raw_columns(test,  feat_cfg.drop_cols)
    all_dropped.extend(dropped)

    # -- Assemble result ------------------------------------------------------
    result = EngineeringResult(
        train=train, val=val, test=test,
        features_added=all_added,
        features_dropped=all_dropped,
        feature_config_source=feat_cfg.source,
    )

    # -- Save -----------------------------------------------------------------
    result = save_featured_splits(result, output_dir=output_dir)

    # -- DVC ------------------------------------------------------------------
    if auto_track_dvc:
        track_featured_with_dvc(result)
    else:
        logger.info("auto_track_dvc=False — skipping DVC tracking.")

    logger.info("\n" + result.summary())
    return result


# =============================================================================
# CLI  ->  python -m src.features.engineering
# =============================================================================

if __name__ == "__main__":
    import sys
    from src.data.data_loader import DataLoader

    config = "configs/data_config.yaml"
    loader = DataLoader()

    logger.info("Loading cleaned splits...")
    train = loader.load_processed("train_clean.csv")
    val   = loader.load_processed("val_clean.csv")
    test  = loader.load_processed("test_clean.csv")

    result = run_feature_engineering(train, val, test, config_path=config)
    print(result.summary())
    sys.exit(0)


__all__ = [
    "FeatureEngineeringError",
    "FeatureConfig",
    "EngineeringResult",
    "load_feature_config",
    "add_ratio_features",
    "add_distance_features",
    "drop_raw_columns",
    "save_featured_splits",
    "run_feature_engineering",
]