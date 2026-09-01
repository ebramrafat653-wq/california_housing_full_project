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
# OUTPUT:
#   - reports/validation/validation_report_{timestamp}.json  (machine-readable)
#   - reports/validation/validation_report_{timestamp}.md    (human-readable)
# =============================================================================

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

from src.data.data_loader import DataLoader
from src.utils.logger import get_logger
from src.utils.paths import PROJECT_DIR

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
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    dataset_shape: tuple[int, int] = (0, 0)

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
            f"Dataset shape   : {self.dataset_shape[0]:,} rows × {self.dataset_shape[1]} cols",
            f"Critical errors : {len(self.critical_errors)}",
            f"Warnings        : {len(self.warnings)}",
            f"Timestamp       : {self.timestamp}",
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
        return

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
        return

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
          * ratio > max_missing_ratio → WARNING
          * ratio <= max_missing_ratio → INFO
      - All other columns:
          * any null → CRITICAL
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
    """WARNING: Duplicate rows are suspicious."""
    n_dupes = int(df.duplicated().sum())
    if n_dupes > 0:
        report.add_warning(
            f"{n_dupes} duplicate rows detected. "
            "Consider deduplication before train/test split."
        )
    else:
        logger.info("No duplicate rows ✓")


def _check_categorical_values(df: pd.DataFrame, cfg: dict, report: ValidationReport) -> None:
    """WARNING: Unseen categories signal data drift."""
    cat_rules: dict = cfg["validation"].get("categorical_features", {})

    for col, allowed_vals in cat_rules.items():
        if col not in df.columns:
            continue

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
    """WARNING: Values outside profiling-derived min/max bounds."""
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
                    f"'{col}': {n_below} value(s) below expected min ({lo})."
                )

        if hi is not None:
            n_above = int((series > hi).sum())
            if n_above > 0:
                col_ok = False
                report.add_warning(
                    f"'{col}': {n_above} value(s) above expected max ({hi})."
                )

        if col_ok:
            logger.info(f"'{col}' numeric boundaries OK ✓")


# ---------------------------------------------------------------------------
# Export Validation Report (JSON + Markdown)
# ---------------------------------------------------------------------------

def export_validation_report(
    report: ValidationReport,
    output_dir: str | Path | None = None,
) -> tuple[Path, Path]:
    """
    Export validation report to both JSON (machine-readable) and Markdown (human-readable).

    Parameters
    ----------
    report     : ValidationReport instance
    output_dir : Directory to save reports (default: reports/validation/)

    Returns
    -------
    tuple[Path, Path] : (json_path, markdown_path)
    """
    output_dir = Path(output_dir) if output_dir else PROJECT_DIR / "reports" / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamp-based filename
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"validation_report_{timestamp_str}.json"
    md_path = output_dir / f"validation_report_{timestamp_str}.md"

    # ── Export JSON ─────────────────────────────────────────────────────────
    report_dict = {
        "passed": report.passed,
        "timestamp": report.timestamp,
        "dataset_shape": {
            "rows": report.dataset_shape[0],
            "columns": report.dataset_shape[1],
        },
        "critical_errors": report.critical_errors,
        "warnings": report.warnings,
        "summary": {
            "n_critical": len(report.critical_errors),
            "n_warnings": len(report.warnings),
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)

    logger.info(f"Validation report (JSON) saved → {json_path}")

    # ── Export Markdown ─────────────────────────────────────────────────────
    md_lines = [
        "# Data Validation Report",
        "",
        f"**Timestamp**: {report.timestamp}",
        f"**Status**: {'✅ PASSED' if report.passed else '❌ FAILED'}",
        "",
        "## Dataset Summary",
        "",
        f"- **Rows**: {report.dataset_shape[0]:,}",
        f"- **Columns**: {report.dataset_shape[1]}",
        "",
        "## Summary Statistics",
        "",
        f"- **Critical Errors**: {len(report.critical_errors)}",
        f"- **Warnings**: {len(report.warnings)}",
        "",
    ]

    if report.critical_errors:
        md_lines.extend([
            "## ❌ Critical Errors",
            "",
        ])
        for i, err in enumerate(report.critical_errors, 1):
            md_lines.append(f"{i}. {err}")
        md_lines.append("")

    if report.warnings:
        md_lines.extend([
            "## ⚠️ Warnings",
            "",
        ])
        for i, warn in enumerate(report.warnings, 1):
            md_lines.append(f"{i}. {warn}")
        md_lines.append("")

    if report.passed and not report.critical_errors and not report.warnings:
        md_lines.extend([
            "## ✅ All Checks Passed",
            "",
            "No critical errors or warnings detected.",
            "",
        ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    logger.info(f"Validation report (Markdown) saved → {md_path}")

    return json_path, md_path


# ---------------------------------------------------------------------------
# Main Public Function
# ---------------------------------------------------------------------------

def validate_dataframe(
    df: pd.DataFrame,
    config_path: str | Path = "configs/data_config.yaml",
    raise_on_failure: bool = True,
    export_report: bool = True,
    output_dir: str | Path | None = None,
) -> ValidationReport:
    """
    Run all preliminary validation checks against `df`.

    Parameters
    ----------
    df               : DataFrame returned by ingestion.py → run_ingestion()
    config_path      : Path to data_config.yaml
    raise_on_failure : If True (default), raises ValidationError on any critical failure.
    export_report    : If True (default), exports report to JSON + Markdown.
    output_dir       : Directory to save reports (default: reports/validation/)

    Returns
    -------
    ValidationReport with full results (always returned, even on failure).

    Raises
    ------
    ValidationError   if any CRITICAL check fails and raise_on_failure=True.
    FileNotFoundError if config_path does not exist.
    """
    logger.info("=" * 60)
    logger.info("  Starting data validation (preliminary)")
    logger.info("=" * 60)

    cfg = _load_config(config_path)
    report = ValidationReport()
    report.dataset_shape = df.shape

    # ── CRITICAL checks ─────────────────────────────────────────────────────
    _check_not_empty(df, cfg, report)

    if not report.critical_errors:
        _check_required_columns(df, cfg, report)
        _check_fully_null_columns(df, cfg, report)

    if not report.critical_errors:
        _check_target_column(df, cfg, report)
        _check_missing_values(df, cfg, report)

    # ── WARNING checks ──────────────────────────────────────────────────────
    _check_duplicates(df, report)
    _check_categorical_values(df, cfg, report)
    _check_numeric_boundaries(df, cfg, report)

    # ── Summary ─────────────────────────────────────────────────────────────
    logger.info("\n" + report.summary())

    # ── Export report ───────────────────────────────────────────────────────
    if export_report:
        try:
            json_path, md_path = export_validation_report(report, output_dir)
        except Exception as e:
            logger.error(f"Failed to export validation report: {e}")

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
    "export_validation_report",
]