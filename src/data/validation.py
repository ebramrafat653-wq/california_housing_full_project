# =============================================================================
# src/data/validation.py
# California Housing Project — Data Validation (Preliminary)
# Run after: ingestion.py  |  Run before: EDA / preprocessing
#
# Two-tier validation strategy:
#   - CRITICAL checks → raise ValidationError  (pipeline stops)
#   - WARNING checks  → logger.warning          (pipeline continues)
#
# Checks order (priority):
#   1. Not empty            [CRITICAL]
#   2. Required columns     [CRITICAL]
#   3. No fully-null cols   [CRITICAL]
#   4. Target column        [CRITICAL]
#   5. Missing values       [CRITICAL if unexpected col / WARNING if allowed col]
#   6. Duplicates           [WARNING]
#   7. Categorical values   [WARNING]
#   8. Numeric boundaries   [WARNING]
#
# After EDA, a second pass will add:
#   - Statistical / distributional checks
#   - Outlier thresholds derived from real distributions
#   - Business rules discovered during exploration
# =============================================================================

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml

from src.data.data_loader import DataLoader
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Custom Exception
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised on critical data quality failures that must stop the pipeline."""
    pass


# ---------------------------------------------------------------------------
# Validation Report
# ---------------------------------------------------------------------------

@dataclass
class ValidationReport:
    passed: bool = True
    critical_errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_critical(self, msg: str) -> None:
        self.critical_errors.append(msg)
        self.passed = False
        logger.error(f"[CRITICAL] {msg}")

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)
        logger.warning(f"[WARNING]  {msg}")

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "VALIDATION REPORT",
            "=" * 60,
            f"Status          : {'PASSED ✓' if self.passed else 'FAILED ✗'}",
            f"Critical errors : {len(self.critical_errors)}",
            f"Warnings        : {len(self.warnings)}",
        ]
        if self.critical_errors:
            lines.append("\nCritical Errors:")
            for e in self.critical_errors:
                lines.append(f"  ✗ {e}")
        if self.warnings:
            lines.append("\nWarnings:")
            for w in self.warnings:
                lines.append(f"  ⚠ {w}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Config Loader
# ---------------------------------------------------------------------------

def _load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Check Functions
# ---------------------------------------------------------------------------

def _check_not_empty(df: pd.DataFrame, cfg: dict, report: ValidationReport) -> None:
    """CRITICAL: DataFrame must meet minimum row and column thresholds."""
    min_rows = cfg["validation"]["min_rows"]
    min_cols = cfg["validation"]["min_columns"]

    if df.shape[0] < min_rows:
        report.add_critical(
            f"Dataset has {df.shape[0]} rows — minimum required is {min_rows}."
        )
    else:
        logger.info(f"Row count OK: {df.shape[0]:,} rows")

    if df.shape[1] < min_cols:
        report.add_critical(
            f"Dataset has {df.shape[1]} columns — minimum required is {min_cols}."
        )
    else:
        logger.info(f"Column count OK: {df.shape[1]} columns")


def _check_required_columns(df: pd.DataFrame, cfg: dict, report: ValidationReport) -> None:
    """CRITICAL: All required columns must be present."""
    required = set(cfg["validation"]["required_columns"])
    missing = required - set(df.columns)
    if missing:
        report.add_critical(f"Missing required columns: {sorted(missing)}")
    else:
        logger.info("All required columns present ✓")


def _check_fully_null_columns(df: pd.DataFrame, cfg: dict, report: ValidationReport) -> None:
    """CRITICAL: No column should be 100% null (unless explicitly allowed)."""
    if cfg["validation"].get("allow_fully_null_columns", False):
        return  # user opted out of this check

    fully_null = [col for col in df.columns if df[col].isna().all()]
    if fully_null:
        report.add_critical(f"Fully-null columns detected: {fully_null}")
    else:
        logger.info("No fully-null columns ✓")


def _check_target_column(df: pd.DataFrame, cfg: dict, report: ValidationReport) -> None:
    """CRITICAL: Target must exist, be numeric, and have zero nulls."""
    target = cfg["project"]["target"]

    if target not in df.columns:
        report.add_critical(f"Target column '{target}' not found in dataset.")
        return  # no point checking further

    if not pd.api.types.is_numeric_dtype(df[target]):
        report.add_critical(f"Target column '{target}' must be numeric.")

    null_count = int(df[target].isna().sum())
    if null_count > 0:
        report.add_critical(
            f"Target column '{target}' has {null_count} null values — "
            "cannot train without a complete target."
        )
    else:
        logger.info(f"Target column '{target}' OK ✓")


def _check_missing_values(df: pd.DataFrame, cfg: dict, report: ValidationReport) -> None:
    """
    Per-column missing value check.

    Rules:
      - Columns in allowed_null_columns:
          * ratio > max_missing_ratio → WARNING (still usable but risky)
          * ratio <= max_missing_ratio → INFO (expected, within limit)
      - All other columns:
          * any null → CRITICAL (schema violation — should never happen)

    Note: fully-null columns are already caught by _check_fully_null_columns,
    so here we only see partial nulls.
    """
    allowed_null_cols = set(cfg["validation"].get("allowed_null_columns", []))
    max_ratio = cfg["validation"]["max_missing_ratio"]
    n = len(df)

    for col in df.columns:
        null_count = int(df[col].isna().sum())
        if null_count == 0:
            continue

        ratio = null_count / n

        if col in allowed_null_cols:
            if ratio > max_ratio:
                report.add_warning(
                    f"'{col}': {null_count} nulls ({ratio:.1%}) exceeds "
                    f"max_missing_ratio ({max_ratio:.0%}). Review before modelling."
                )
            else:
                logger.info(
                    f"'{col}' nulls: {null_count} ({ratio:.1%}) — within allowed limit"
                )
        else:
            report.add_critical(
                f"Unexpected nulls in '{col}': {null_count} ({ratio:.1%}). "
                "Column is not listed in allowed_null_columns."
            )


def _check_duplicates(df: pd.DataFrame, report: ValidationReport) -> None:
    """WARNING: Duplicate rows are suspicious (profiling baseline = 0)."""
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        report.add_warning(
            f"{n_dupes} duplicate rows detected. "
            "Consider deduplication before train/test split."
        )
    else:
        logger.info("No duplicate rows ✓")


def _check_categorical_values(df: pd.DataFrame, cfg: dict, report: ValidationReport) -> None:
    """WARNING: Unseen categories signal data drift or upstream schema change."""
    cat_rules: dict = cfg["validation"].get("categorical_features", {})

    for col, allowed_vals in cat_rules.items():
        if col not in df.columns:
            continue  # already caught by _check_required_columns

        allowed_set = set(allowed_vals)
        actual_set = set(df[col].dropna().unique())
        unexpected = actual_set - allowed_set

        if unexpected:
            report.add_warning(
                f"'{col}' has unexpected categories: {sorted(unexpected)}. "
                f"Expected: {sorted(allowed_set)}"
            )
        else:
            logger.info(f"'{col}' categories OK ✓")


def _check_numeric_boundaries(df: pd.DataFrame, cfg: dict, report: ValidationReport) -> None:
    """
    WARNING: Values outside profiling-derived min/max bounds.
    These are hard limits from profiling (absolute min/max), NOT outlier thresholds.
    Outlier thresholds will be added post-EDA using IQR / z-score analysis.
    """
    boundaries: dict = cfg["validation"].get("numeric_boundaries", {})

    for col, bounds in boundaries.items():
        if col not in df.columns:
            continue

        series = df[col].dropna()
        lo = bounds.get("min")
        hi = bounds.get("max")
        col_ok = True

        if lo is not None:
            n_below = int((series < lo).sum())
            if n_below > 0:
                col_ok = False
                report.add_warning(
                    f"'{col}': {n_below} value(s) below expected min ({lo}). "
                    "Possible data drift or corruption."
                )

        if hi is not None:
            n_above = int((series > hi).sum())
            if n_above > 0:
                col_ok = False
                report.add_warning(
                    f"'{col}': {n_above} value(s) above expected max ({hi}). "
                    "Possible data drift or corruption."
                )

        if col_ok:
            logger.info(f"'{col}' numeric boundaries OK ✓")


# ---------------------------------------------------------------------------
# Main Public Function
# ---------------------------------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame,
    config_path: str | Path = "configs/data_config.yaml",
    raise_on_failure: bool = True,
) -> ValidationReport:
    """
    Run all preliminary validation checks against `df`.

    Parameters
    ----------
    df               : DataFrame returned by ingestion.py → run_ingestion()
    config_path      : Path to data_config.yaml
    raise_on_failure : If True (default), raises ValidationError on any critical failure.
                       Set False only in notebooks for exploratory inspection.

    Returns
    -------
    ValidationReport with full results (always returned, even on failure).

    Raises
    ------
    ValidationError   if any CRITICAL check fails and raise_on_failure=True.
    FileNotFoundError if config_path does not exist.

    Usage
    -----
        from src.data.ingestion import run_ingestion
        from src.data.validation import validate_dataframe

        report = run_ingestion()          # returns DownloadReport
        df     = pd.read_csv(...)         # or however you load after ingestion
        report = validate_dataframe(df)   # raises on critical failure
    """
    logger.info("=" * 60)
    logger.info("  Starting data validation (preliminary)")
    logger.info("=" * 60)

    cfg = _load_config(config_path)
    report = ValidationReport()

    # ── CRITICAL checks — stop adding column-level checks if schema is broken ──
    _check_not_empty(df, cfg, report)

    if not report.critical_errors:
        # Only meaningful if we have rows and columns
        _check_required_columns(df, cfg, report)
        _check_fully_null_columns(df, cfg, report)

    if not report.critical_errors:
        # Only meaningful if all required columns are present
        _check_target_column(df, cfg, report)
        _check_missing_values(df, cfg, report)

    # ── WARNING checks — always run (independent of schema) ──────────────────
    _check_duplicates(df, report)
    _check_categorical_values(df, cfg, report)
    _check_numeric_boundaries(df, cfg, report)

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("\n" + report.summary())

    if not report.passed and raise_on_failure:
        raise ValidationError(
            f"Validation failed with {len(report.critical_errors)} critical error(s). "
            "See report above for details."
        )

    return report


# ---------------------------------------------------------------------------
# CLI  →  python -m src.data.validation
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    config = "configs/data_config.yaml"

    loader = DataLoader()
    df = loader.load_raw("housing.csv")
    logger.info(f"Loaded DataFrame: {df.shape[0]:,} rows × {df.shape[1]} columns")

    try:
        validate_dataframe(df, config_path=config, raise_on_failure=True)
        sys.exit(0)
    except ValidationError as e:
        logger.error(str(e))
        sys.exit(1)


__all__ = [
    "ValidationError",
    "ValidationReport",
    "validate_dataframe",
]
