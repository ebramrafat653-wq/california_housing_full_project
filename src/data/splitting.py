# =============================================================================
# src/data/splitting.py
# California Housing Project — Stratified Train / Val / Test Split
#
# WHY STRATIFIED?
#   median_house_value is skewed — a random split risks putting most
#   high-value or low-value districts in one set only.
#   Solution: bin the target into quantile-based strata, stratify on those
#   bins, then drop the helper column. This guarantees each split sees the
#   full price range in the correct proportion.
#
# SPLIT RATIO: 70% train / 15% val / 15% test
#
# OUTPUT FILES (saved to data/interim/ — DVC-tracked):
#   train.csv  |  val.csv  |  test.csv
#
# DVC INTEGRATION:
#   After saving, files are tracked with `dvc add` and pushed to remote.
#   The .dvc pointer files are committed to git automatically.
#
# Run after : src/data/validation.py
# Run before: EDA (on train.csv only) → cleaning.py
# =============================================================================

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import yaml
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.utils.paths import (
    PATHS,
    PROJECT_DIR,
    DVC_REMOTE_NAME,
    is_dvc_initialized,
    ensure_path,
    configure_git_identity,
)

logger = get_logger(__name__)

# Number of quantile-based bins for stratification on the target
_N_BINS: int = 5
_STRATA_COL: str = "_price_stratum"


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class SplitResult:
    """Holds the three DataFrames and metadata from a completed split."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    train_path: Optional[Path] = None
    val_path:   Optional[Path] = None
    test_path:  Optional[Path] = None

    train_ratio: float = 0.0
    val_ratio:   float = 0.0
    test_ratio:  float = 0.0

    n_bins:      int = _N_BINS
    random_state: int = 42
    timestamp:   datetime = field(default_factory=datetime.now)
    dvc_tracked: bool = False
    warnings:    list[str] = field(default_factory=list)

    def summary(self) -> str:
        total = len(self.train) + len(self.val) + len(self.test)
        lines = [
            "=" * 60,
            "  SPLIT RESULT SUMMARY",
            "=" * 60,
            f"  Total rows : {total:,}",
            f"  Train      : {len(self.train):,}  ({len(self.train)/total:.1%})",
            f"  Val        : {len(self.val):,}  ({len(self.val)/total:.1%})",
            f"  Test       : {len(self.test):,}  ({len(self.test)/total:.1%})",
            f"  DVC tracked: {'✅ yes' if self.dvc_tracked else '⚪ no'}",
        ]
        if self.train_path:
            lines += [
                "",
                "  Saved to:",
                f"    train → {self.train_path}",
                f"    val   → {self.val_path}",
                f"    test  → {self.test_path}",
            ]
        if self.warnings:
            lines.append("\n  Warnings:")
            for w in self.warnings:
                lines.append(f"    ⚠ {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# STRATIFICATION HELPER
# =============================================================================

def _add_price_strata(df: pd.DataFrame, target: str, n_bins: int) -> pd.DataFrame:
    """
    Add a temporary quantile-based stratum column to `df`.

    Uses pd.qcut (equal-frequency bins) so each bin has roughly the same
    number of rows — guaranteeing proportional representation across splits.

    Args:
        df      : Input DataFrame (must contain `target` column).
        target  : Name of the regression target column.
        n_bins  : Number of quantile bins (default 5 → quintiles).

    Returns:
        Copy of df with an extra integer column `_price_stratum`.
    """
    df = df.copy()
    try:
        df[_STRATA_COL] = pd.qcut(
            df[target],
            q=n_bins,
            labels=False,
            duplicates="drop",   # handles tied edges gracefully
        )
    except ValueError as e:
        logger.warning(f"qcut failed ({e}) — falling back to pd.cut (equal-width bins)")
        df[_STRATA_COL] = pd.cut(
            df[target],
            bins=n_bins,
            labels=False,
        )

    null_strata = df[_STRATA_COL].isna().sum()
    if null_strata > 0:
        logger.warning(
            f"{null_strata} rows could not be assigned a stratum "
            "and will be stratified randomly."
        )
        # Fill with the most frequent stratum so they still participate
        df[_STRATA_COL].fillna(df[_STRATA_COL].mode()[0], inplace=True)

    df[_STRATA_COL] = df[_STRATA_COL].astype(int)
    return df


def _verify_stratification(
    full: pd.DataFrame,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    result: SplitResult,
) -> None:
    """
    Log per-stratum distribution across splits.
    Warns if any stratum deviates more than ±3 pp from the expected ratio.
    """
    expected_val  = result.val_ratio
    expected_test = result.test_ratio

    full_dist  = full[_STRATA_COL].value_counts(normalize=True).sort_index()
    train_dist = train[_STRATA_COL].value_counts(normalize=True).sort_index()
    val_dist   = val[_STRATA_COL].value_counts(normalize=True).sort_index()
    test_dist  = test[_STRATA_COL].value_counts(normalize=True).sort_index()

    logger.info("Stratum distribution (proportion of each split):")
    logger.info(f"{'Bin':<6} {'Full':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
    for bin_id in full_dist.index:
        logger.info(
            f"{bin_id:<6} "
            f"{full_dist.get(bin_id, 0):>8.1%} "
            f"{train_dist.get(bin_id, 0):>8.1%} "
            f"{val_dist.get(bin_id, 0):>8.1%} "
            f"{test_dist.get(bin_id, 0):>8.1%}"
        )

    # Warn if val or test deviate more than 3 pp from expected
    threshold = 0.03
    for split_name, dist, expected in [
        ("val",  val_dist,  expected_val),
        ("test", test_dist, expected_test),
    ]:
        for bin_id in full_dist.index:
            actual = dist.get(bin_id, 0.0)
            full_p = full_dist.get(bin_id, 0.0)
            if abs(actual - full_p) > threshold:
                result.warnings.append(
                    f"Stratum {bin_id} in {split_name} set deviates "
                    f"{abs(actual - full_p):.1%} from full distribution ({full_p:.1%}). "
                    "Consider increasing n_bins or checking for ties."
                )


# =============================================================================
# CORE SPLIT LOGIC
# =============================================================================

def stratified_split(
    df: pd.DataFrame,
    target: str,
    val_size: float = 0.15,
    test_size: float = 0.15,
    n_bins: int = _N_BINS,
    random_state: int = 42,
) -> SplitResult:
    """
    Split `df` into train / val / test using stratified sampling on `target`.

    Strategy
    --------
    1. Bin `target` into `n_bins` quantile-based strata.
    2. Split off `test_size` as test — stratified on bins.
    3. Split the remainder into train / val — stratified on bins.
    4. Verify and log per-stratum distributions.
    5. Drop the helper strata column from all three sets.

    Args:
        df           : Full cleaned DataFrame (post-validation, pre-EDA).
        target       : Regression target column name.
        val_size     : Fraction of total data for validation set.
        test_size    : Fraction of total data for test set.
        n_bins       : Number of quantile bins for stratification.
        random_state : Reproducibility seed.

    Returns:
        SplitResult with .train / .val / .test DataFrames.

    Raises:
        ValueError: If target is missing, sizes are invalid, or df is empty.
    """
    # ── Guards ────────────────────────────────────────────────────────────────
    if df.empty:
        raise ValueError("Input DataFrame is empty — cannot split.")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    if not (0 < val_size < 1) or not (0 < test_size < 1):
        raise ValueError("val_size and test_size must be between 0 and 1.")
    if val_size + test_size >= 1.0:
        raise ValueError(
            f"val_size ({val_size}) + test_size ({test_size}) must be < 1.0"
        )

    train_size = round(1.0 - val_size - test_size, 6)
    logger.info(
        f"Splitting {len(df):,} rows → "
        f"train {train_size:.0%} / val {val_size:.0%} / test {test_size:.0%}"
    )

    # ── Step 1: Add strata ────────────────────────────────────────────────────
    df_strat = _add_price_strata(df, target, n_bins)
    logger.info(f"Strata value counts:\n{df_strat[_STRATA_COL].value_counts().sort_index()}")

    # ── Step 2: Split off test ────────────────────────────────────────────────
    temp_df, test_df = train_test_split(
        df_strat,
        test_size=test_size,
        random_state=random_state,
        stratify=df_strat[_STRATA_COL],
    )

    # ── Step 3: Split remainder into train / val ──────────────────────────────
    # val fraction relative to the remaining temp_df
    relative_val_size = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        temp_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=temp_df[_STRATA_COL],
    )

    # ── Step 4: Initialize Result & Verify (BEFORE dropping column) ───────────
    total = len(train_df) + len(val_df) + len(test_df)
    result = SplitResult(
        train=train_df,
        val=val_df,
        test=test_df,
        train_ratio=len(train_df) / total,
        val_ratio=len(val_df) / total,
        test_ratio=len(test_df) / total,
        n_bins=n_bins,
        random_state=random_state,
    )
    
    # Verify distributions using the dataframes that still have _STRATA_COL
    _verify_stratification(df_strat, train_df, val_df, test_df, result)

    # ── Step 5: Drop helper column ────────────────────────────────────────────
    for split in (train_df, val_df, test_df):
        split.drop(columns=[_STRATA_COL], inplace=True)

    # ── Step 6: Reset indices & Update Result ─────────────────────────────────
    result.train = train_df.reset_index(drop=True)
    result.val   = val_df.reset_index(drop=True)
    result.test  = test_df.reset_index(drop=True)

    logger.info(
        f"Split complete — "
        f"train: {len(result.train):,} | val: {len(result.val):,} | test: {len(result.test):,}"
    )
    return result

# =============================================================================
# SAVE TO DISK (data/interim/)
# =============================================================================

def save_splits(
    result: SplitResult,
    output_dir: Optional[Path] = None,
) -> SplitResult:
    """
    Save train / val / test CSVs to data/interim/.

    Files are saved to `data/interim/` because they are derived from raw data
    but not yet fully processed — correct DVC stage for interim artifacts.

    Args:
        result     : SplitResult from stratified_split().
        output_dir : Override output directory (default: PATHS["interim"]).

    Returns:
        Same SplitResult with .train_path / .val_path / .test_path populated.
    """
    out = output_dir or ensure_path("interim")
    out.mkdir(parents=True, exist_ok=True)

    paths = {
        "train": out / "train.csv",
        "val":   out / "val.csv",
        "test":  out / "test.csv",
    }

    result.train.to_csv(paths["train"], index=False)
    result.val.to_csv(paths["val"],   index=False)
    result.test.to_csv(paths["test"], index=False)

    result.train_path = paths["train"]
    result.val_path   = paths["val"]
    result.test_path  = paths["test"]

    for name, path in paths.items():
        size_mb = path.stat().st_size / (1024 * 1024)
        logger.info(f"Saved {name}.csv → {path} ({size_mb:.2f} MB)")

    return result


# =============================================================================
# DVC TRACKING
# =============================================================================

def _run_cmd(cmd: list[str], cwd: Path, timeout: int = 120) -> bool:
    """Run a subprocess command; return True on success."""
    r = subprocess.run(
        cmd, cwd=cwd,
        capture_output=True, text=True,
        errors="replace", timeout=timeout,
    )
    if r.returncode != 0:
        logger.warning(f"Command failed {cmd}: {r.stderr.strip()}")
        return False
    return True


def track_splits_with_dvc(
    result: SplitResult,
    remote_name: str = DVC_REMOTE_NAME,
    timeout: int = 300,
) -> bool:
    """
    Track the interim split files with DVC and push to remote.

    Tracks data/interim/ as a whole (not file-by-file) to keep the
    .dvc pointer files clean and avoid redundant git commits.

    Args:
        result      : SplitResult with populated paths.
        remote_name : DVC remote name from paths.py.
        timeout     : Subprocess timeout in seconds.

    Returns:
        True if tracking and push succeeded, False otherwise.
    """
    if not is_dvc_initialized():
        logger.warning("DVC not initialized — skipping tracking.")
        return False

    if not result.train_path:
        logger.warning("No paths in SplitResult — call save_splits() first.")
        return False

    interim_dir = result.train_path.parent
    try:
        rel = interim_dir.relative_to(PROJECT_DIR)
    except ValueError:
        logger.error(f"{interim_dir} is outside PROJECT_DIR — cannot track with DVC.")
        return False

    logger.info(f"DVC add: {rel}")
    if not _run_cmd(["dvc", "add", str(rel)], cwd=PROJECT_DIR, timeout=timeout):
        return False

    logger.info(f"DVC push → {remote_name}")
    _run_cmd(
        ["dvc", "push", f"--remote={remote_name}", str(rel)],
        cwd=PROJECT_DIR, timeout=timeout,
    )

    # Git: commit the .dvc pointer
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
        logger.info("No new git changes — interim already tracked.")
        result.dvc_tracked = True
        return True

    _run_cmd(
        ["git", "commit", "-m", "data: track interim splits with DVC"],
        cwd=PROJECT_DIR, timeout=60,
    )
    ok = _run_cmd(["git", "push"], cwd=PROJECT_DIR, timeout=120)
    if ok:
        logger.info("✅ Interim splits tracked and pushed to DVC remote.")
        result.dvc_tracked = True
    else:
        logger.error("git push failed — splits saved locally but not versioned.")

    return ok


# =============================================================================
# CONFIG LOADER
# =============================================================================

def _load_split_config(config_path: Path) -> dict:
    """Extract the `split` section from data_config.yaml."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


# =============================================================================
# MAIN PIPELINE ENTRY POINT
# =============================================================================

def run_splitting(
    df: pd.DataFrame,
    config_path: str | Path = "configs/data_config.yaml",
    output_dir: Optional[Path] = None,
    auto_track_dvc: bool = True,
) -> SplitResult:
    """
    Full split pipeline: stratify → save → DVC track.

    This is the function called by notebooks and downstream pipeline steps.

    Args:
        df             : Validated DataFrame from validation.py.
        config_path    : Path to data_config.yaml.
        output_dir     : Override for output directory (default: data/interim/).
        auto_track_dvc : Track splits with DVC after saving.

    Returns:
        SplitResult with all three DataFrames and file paths.

    Usage
    -----
        from src.data.data_loader import DataLoader
        from src.data.validation import validate_dataframe
        from src.data.splitting import run_splitting

        df     = DataLoader().load_raw("housing.csv")
        report = validate_dataframe(df)
        result = run_splitting(df)

        # EDA goes here — on result.train only
        train_df = result.train
    """
    logger.info("=" * 60)
    logger.info("  Data splitting started")
    logger.info("=" * 60)

    cfg = _load_split_config(Path(config_path))

    target       = cfg["project"]["target"]
    split_cfg    = cfg.get("split", {})
    val_size     = split_cfg.get("val_size",     0.15)
    test_size    = split_cfg.get("test_size",    0.15)
    random_state = split_cfg.get("random_state", 42)

    # ── Split ────────────────────────────────────────────────────────────────
    result = stratified_split(
        df=df,
        target=target,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    # ── Save ─────────────────────────────────────────────────────────────────
    result = save_splits(result, output_dir=output_dir)

    # ── DVC ──────────────────────────────────────────────────────────────────
    if auto_track_dvc:
        track_splits_with_dvc(result)
    else:
        logger.info("auto_track_dvc=False — skipping DVC tracking.")

    logger.info("\n" + result.summary())
    return result


# =============================================================================
# CLI  →  python -m src.data.splitting
# =============================================================================

if __name__ == "__main__":
    import sys
    from src.data.data_loader import DataLoader
    from src.data.validation import validate_dataframe, ValidationError

    config = "configs/data_config.yaml"

    df = DataLoader().load_raw("housing.csv")
    logger.info(f"Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    try:
        validate_dataframe(df, config_path=config, raise_on_failure=True)
    except ValidationError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

    result = run_splitting(df, config_path=config)
    print(result.summary())
    sys.exit(0)


__all__ = [
    "SplitResult",
    "stratified_split",
    "save_splits",
    "track_splits_with_dvc",
    "run_splitting",
]