# src/data/cleaning.py
# California Housing Project — Data Cleaning
#
# RESPONSIBILITY
# --------------
# This module is responsible ONLY for:
#
#   1. Configuration loading
#   2. Input/data-contract validation
#   3. Strict missing-value validation
#   4. Saving train/validation/test datasets
#   5. Saving cleaning metadata
#
# THIS MODULE DOES NOT
# --------------------
#   - Perform feature engineering
#   - Create ratios
#   - Apply log1p transformations
#   - Create target-derived features
#   - Fit imputers
#   - Fit scalers
#   - Fit LOF
#   - Fit encoders
#   - Train ML models
#   - Perform model preprocessing
#   - Execute DVC commands
#   - Execute Git commands
#
# IMPORTANT CV POLICY
# -------------------
# Any operation that LEARNS statistics/parameters from data must happen
# inside the Cross-Validation / Training Pipeline.
#
# Therefore:
#
#   Imputer       -> training.py / sklearn Pipeline
#   Scaler        -> training.py / sklearn Pipeline
#   OneHotEncoder -> training.py / sklearn Pipeline
#   LOF           -> training.py / sklearn Pipeline
#   Model         -> training.py
#
# PIPELINE POSITION
# -----------------
#
#   data/interim/
#       train.csv
#       val.csv
#       test.csv
#             |
#             v
#        cleaning.py
#             |
#             +--> train_clean.csv
#             +--> val_clean.csv
#             +--> test_clean.csv
#             |
#             +--> artifacts/
#                   cleaning_metadata.json
#
# DVC
# ---
# DVC is intentionally NOT used inside this module.
# DVC orchestration belongs to dvc.yaml / pipeline orchestration.
#
# =============================================================================

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.utils.logger import get_logger
from src.utils.paths import PROJECT_DIR, ensure_path


logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

_TARGET = "median_house_value"

_DEFAULT_CONFIG_PATH = "configs/data_config.yaml"

_CLEANED_TRAIN_FILENAME = "train_clean.csv"
_CLEANED_VAL_FILENAME = "val_clean.csv"
_CLEANED_TEST_FILENAME = "test_clean.csv"

_ARTIFACTS_DIRNAME = "artifacts"
_CLEANING_METADATA_FILENAME = "cleaning_metadata.json"


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================

class CleaningError(Exception):
    """Raised when the cleaning pipeline cannot safely continue."""

    pass


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class EdaConfig:
    """
    Typed representation of the EDA-derived configuration required by
    the cleaning stage.

    Expected source:

        configs/data_config.yaml

    Relevant section:

        eda_derived:
            missingness:
                ...
            target_summary:
                ...

    Notes
    -----
    LOF configuration is intentionally NOT loaded here.

    LOF is a learned ML transformation and belongs inside the training /
    cross-validation pipeline.

    Imputation configuration is also intentionally NOT fitted here.

    The actual imputation must happen inside the sklearn Pipeline during
    Cross-Validation so that each fold learns its own statistics.
    """

    impute_columns: list[str]
    imputation_strategy: dict[str, str]

    cap_threshold: float


def load_eda_config(
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
) -> EdaConfig:
    """
    Load and validate EDA-derived configuration.

    This function ONLY reads configuration.
[9/3/2026 4:42 AM] Ebram Rafat: It does not fit an imputer, scaler, LOF, encoder, or model.
    """

    config_path = Path(config_path)

    # -------------------------------------------------------------------------
    # Config file must exist
    # -------------------------------------------------------------------------

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}. "
            "Cleaning cannot continue without a valid data configuration."
        )

    # -------------------------------------------------------------------------
    # Load YAML
    # -------------------------------------------------------------------------

    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

    except yaml.YAMLError as exc:
        raise CleaningError(
            f"Failed to parse YAML config: {config_path}"
        ) from exc

    if not isinstance(cfg, dict):
        raise CleaningError(
            f"Invalid configuration format in {config_path}. "
            "Expected a YAML mapping."
        )

    # -------------------------------------------------------------------------
    # eda_derived
    # -------------------------------------------------------------------------

    eda = cfg.get("eda_derived")

    if not isinstance(eda, dict):
        raise CleaningError(
            f"'eda_derived' section is missing or invalid in {config_path}."
        )

    # -------------------------------------------------------------------------
    # Missingness configuration
    #
    # IMPORTANT:
    # We read the configuration here, but we DO NOT fit the imputer.
    #
    # Actual imputation belongs inside the training/CV Pipeline.
    # -------------------------------------------------------------------------

    missingness = eda.get("missingness")

    if not isinstance(missingness, dict):
        raise CleaningError(
            f"'eda_derived.missingness' is missing or invalid "
            f"in {config_path}."
        )

    impute_columns: list[str] = []
    imputation_strategy: dict[str, str] = {}

    for column, column_config in missingness.items():

        if not isinstance(column_config, dict):
            raise CleaningError(
                f"Invalid missingness configuration for column '{column}'."
            )

        should_impute = column_config.get("impute", False)

        if should_impute:

            impute_columns.append(column)

            strategy = column_config.get(
                "imputation_strategy",
                "median",
            )

            if not isinstance(strategy, str):
                raise CleaningError(
                    f"Invalid imputation strategy for '{column}'."
                )

            supported_strategies = {
                "median",
                "mean",
                "mode",
            }

            if strategy not in supported_strategies:
                raise CleaningError(
                    f"Unsupported imputation strategy '{strategy}' "
                    f"for '{column}'. "
                    f"Supported strategies: "
                    f"{sorted(supported_strategies)}."
                )

            imputation_strategy[column] = strategy

    # -------------------------------------------------------------------------
    # Target summary / cap threshold
    #
    # IMPORTANT:
    # cap_threshold is metadata only.
    #
    # We DO NOT create:
    #
    #     is_capped = median_house_value >= cap_threshold
    #
    # because that would create a target-derived feature.
    # -------------------------------------------------------------------------

    target_summary = eda.get("target_summary")

    if not isinstance(target_summary, dict):
        raise CleaningError(
            "'eda_derived.target_summary' is missing or invalid."
        )

    if "cap_threshold" not in target_summary:
        raise CleaningError(
            "'eda_derived.target_summary.cap_threshold' is missing."
        )
    try:
        cap_threshold = float(
            target_summary["cap_threshold"]
        )

    except (TypeError, ValueError) as exc:
        raise CleaningError(
            "cap_threshold must be numeric."
        ) from exc

    logger.info(
        "EDA configuration loaded successfully."
    )

    logger.info(
        f"Configured imputation columns: {impute_columns}"
    )

    logger.info(
        f"Configured imputation strategies: "
        f"{imputation_strategy}"
    )

    logger.info(
        f"Cap threshold metadata: {cap_threshold}"
    )

    return EdaConfig(
        impute_columns=impute_columns,
        imputation_strategy=imputation_strategy,
        cap_threshold=cap_threshold,
    )


# =============================================================================
# RESULT OBJECT
# =============================================================================

@dataclass
class CleaningResult:
    """Result returned by the cleaning stage."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    train_path: Path | None = None
    val_path: Path | None = None
    test_path: Path | None = None

    timestamp: datetime = field(
        default_factory=datetime.now
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def summary(self) -> str:
        """Return a human-readable cleaning summary."""

        lines = [
            "=" * 70,
            "  DATA CLEANING RESULT",
            "=" * 70,
            "",
            (
                f"  Train : "
                f"{len(self.train):,} rows x "
                f"{self.train.shape[1]} cols"
            ),
            (
                f"  Val   : "
                f"{len(self.val):,} rows x "
                f"{self.val.shape[1]} cols"
            ),
            (
                f"  Test  : "
                f"{len(self.test):,} rows x "
                f"{self.test.shape[1]} cols"
            ),
            "",
            "  Learned preprocessing:",
            "    Imputer : DEFERRED TO CV/TRAINING PIPELINE",
            "    Scaler  : DEFERRED TO CV/TRAINING PIPELINE",
            "    LOF     : DEFERRED TO CV/TRAINING PIPELINE",
            "    Encoder : DEFERRED TO CV/TRAINING PIPELINE",
            "",
            "  Feature engineering: DEFERRED",
            "  DVC tracking: HANDLED OUTSIDE CLEANING",
            "  Git operations: HANDLED OUTSIDE CLEANING",
        ]

        if self.train_path:

            lines.extend(
                [
                    "",
                    "  Saved outputs:",
                    f"    train -> {self.train_path}",
                    f"    val   -> {self.val_path}",
                    f"    test  -> {self.test_path}",
                ]
            )

        if self.warnings:

            lines.append("")
            lines.append("  Warnings:")

            for warning in self.warnings:

                lines.append(
                    f"    ! {warning}"
                )

        lines.append("=" * 70)

        return "\n".join(lines)


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_cleaning_inputs(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Validate the basic structure of train/val/test inputs.

    This function does not modify the data.
    """

    splits = {
        "train": train,
        "val": val,
        "test": test,
    }

    # -------------------------------------------------------------------------
    # Basic DataFrame validation
    # -------------------------------------------------------------------------

    for split_name, df in splits.items():

        if not isinstance(df, pd.DataFrame):
            raise CleaningError(
                f"{split_name} must be a pandas DataFrame."
            )

        if df.empty:
            raise CleaningError(
                f"{split_name} DataFrame is empty."
            )
    # Target existence
    # -------------------------------------------------------------------------

    for split_name, df in splits.items():

        if _TARGET not in df.columns:
            raise CleaningError(
                f"Target column '{_TARGET}' is missing "
                f"from {split_name}."
            )

    # -------------------------------------------------------------------------
    # Target must be numeric
    # -------------------------------------------------------------------------

    for split_name, df in splits.items():

        if not pd.api.types.is_numeric_dtype(
            df[_TARGET]
        ):
            raise CleaningError(
                f"[{split_name}] Target column '{_TARGET}' "
                "must be numeric."
            )

    # -------------------------------------------------------------------------
    # Target missing values are never acceptable
    # -------------------------------------------------------------------------

    for split_name, df in splits.items():

        if df[_TARGET].isna().any():

            missing_count = int(
                df[_TARGET].isna().sum()
            )

            raise CleaningError(
                f"[{split_name}] Target column '{_TARGET}' "
                f"contains {missing_count:,} missing values. "
                "Target values must be present before training."
            )


# =============================================================================
# STRICT MISSING-VALUE VALIDATION
# =============================================================================

def check_missing_values(
    df: pd.DataFrame,
    allowed_cols: list[str],
    split_name: str,
) -> None:
    """
    Validate missing values without modifying the dataframe.

    Missing values are allowed ONLY in explicitly configured columns.

    Important:
    This function does not perform imputation.

    Imputation is intentionally deferred to the CV-safe training pipeline.
    """

    nan_columns = df.columns[
        df.isna().any()
    ].tolist()

    # -------------------------------------------------------------------------
    # Target missingness is never acceptable
    # -------------------------------------------------------------------------

    if _TARGET in nan_columns:

        raise CleaningError(
            f"[{split_name}] Target column '{_TARGET}' "
            "contains missing values. "
            "Target values must be present before training."
        )

    # -------------------------------------------------------------------------
    # Unexpected missingness
    # -------------------------------------------------------------------------

    unexpected_nans = [
        column
        for column in nan_columns
        if column not in allowed_cols
    ]

    if unexpected_nans:

        raise CleaningError(
            f"[{split_name}] Unexpected missing values found "
            f"in columns: {sorted(unexpected_nans)}. "
            f"Only explicitly configured columns may be "
            f"handled by the training pipeline: "
            f"{sorted(allowed_cols)}."
        )

    # -------------------------------------------------------------------------
    # Logging
    # -------------------------------------------------------------------------

    if nan_columns:

        for column in nan_columns:

            count = int(
                df[column].isna().sum()
            )

            logger.info(
                f"[{split_name}] Missing values detected: "
                f"{column} = {count:,}"
            )


# =============================================================================
# COLUMN VALIDATION
# =============================================================================

def validate_configured_columns(
    train: pd.DataFrame,
    config: EdaConfig,
) -> None:
    """
    Validate that configured imputation columns exist.

    No fitting or transformation occurs here.
    """
    missing_columns = [
        column
        for column in config.impute_columns
        if column not in train.columns
    ]

    if missing_columns:

        raise CleaningError(
            "Configured imputation columns do not exist "
            f"in the dataset: {sorted(missing_columns)}"
        )


# =============================================================================
# CLEANING METADATA
# =============================================================================

def save_cleaning_metadata(
    config: EdaConfig,
    output_dir: Path | None = None,
) -> Path:
    """
    Save cleaning-stage metadata.

    This file contains configuration metadata only.

    It does NOT contain fitted:
        - imputer
        - scaler
        - LOF
        - encoder
        - model
    """

    artifacts_dir = (
        output_dir
        if output_dir is not None
        else PROJECT_DIR / _ARTIFACTS_DIRNAME
    )

    artifacts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    metadata = {
        "target": _TARGET,
        "impute_columns": config.impute_columns,
        "imputation_strategy": config.imputation_strategy,
        "cap_threshold": config.cap_threshold,
        "learned_preprocessing": {
            "imputer": "training_pipeline",
            "scaler": "training_pipeline",
            "lof": "training_pipeline",
            "encoder": "training_pipeline",
        },
        "target_derived_features": {
            "is_capped": False,
        },
        "timestamp": datetime.now().isoformat(),
    }

    metadata_path = (
        artifacts_dir /
        _CLEANING_METADATA_FILENAME
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2,
        )

    logger.info(
        f"Cleaning metadata saved to: {metadata_path}"
    )

    return metadata_path


# =============================================================================
# SAVE CLEANED DATA
# =============================================================================

def save_cleaned_splits(
    result: CleaningResult,
    output_dir: Path | None = None,
) -> CleaningResult:
    """
    Save train/validation/test datasets.

    No learned transformation is applied here.
    """

    output_dir = (
        output_dir
        if output_dir is not None
        else ensure_path("processed")
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "train": output_dir / _CLEANED_TRAIN_FILENAME,
        "val": output_dir / _CLEANED_VAL_FILENAME,
        "test": output_dir / _CLEANED_TEST_FILENAME,
    }

    result.train.to_csv(
        paths["train"],
        index=False,
    )

    result.val.to_csv(
        paths["val"],
        index=False,
    )

    result.test.to_csv(
        paths["test"],
        index=False,
    )

    result.train_path = paths["train"]
    result.val_path = paths["val"]
    result.test_path = paths["test"]

    for name, path in paths.items():

        size_mb = (
            path.stat().st_size /
            (1024 * 1024)
        )

        logger.info(
            f"Saved {name}: "
            f"{path} "
            f"({size_mb:.2f} MB)"
        )

    return result


# =============================================================================
# MAIN CLEANING PIPELINE
# =============================================================================

def run_cleaning(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    output_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    save_artifacts_flag: bool = True,
) -> CleaningResult:
    """
    Execute the data-cleaning stage.

    Pipeline
    --------
        1. Validate inputs
        2. Load EDA configuration
        3. Validate configured columns
        4. Validate missing-value contract
        5. Save datasets
        6. Save cleaning metadata
[9/3/2026 4:42 AM] Ebram Rafat: IMPORTANT
    ---------
    No learned preprocessing occurs here.

    The following operations are intentionally deferred to the
    CV-safe training pipeline:

        - Imputation
        - Scaling
        - LOF
        - Encoding
        - Model fitting

    This prevents preprocessing leakage during Cross-Validation.

    No DVC/Git operations are performed here.
    """

    logger.info("=" * 70)
    logger.info("DATA CLEANING STARTED")
    logger.info("=" * 70)

    # =========================================================================
    # STEP 1 — INPUT VALIDATION
    # =========================================================================

    logger.info(
        "Step 1/4 - Validating cleaning inputs..."
    )

    validate_cleaning_inputs(
        train,
        val,
        test,
    )

    # =========================================================================
    # STEP 2 — LOAD CONFIG
    # =========================================================================

    logger.info(
        "Step 2/4 - Loading EDA-derived configuration..."
    )

    eda_config = load_eda_config(
        config_path
    )

    # =========================================================================
    # STEP 3 — DATA QUALITY VALIDATION
    # =========================================================================

    logger.info(
        "Step 3/4 - Validating missing-value contract..."
    )

    validate_configured_columns(
        train=train,
        config=eda_config,
    )

    check_missing_values(
        df=train,
        allowed_cols=eda_config.impute_columns,
        split_name="train",
    )

    check_missing_values(
        df=val,
        allowed_cols=eda_config.impute_columns,
        split_name="val",
    )

    check_missing_values(
        df=test,
        allowed_cols=eda_config.impute_columns,
        split_name="test",
    )

    # =========================================================================
    # STEP 4 — SAVE
    # =========================================================================

    logger.info(
        "Step 4/4 - Saving cleaned datasets and metadata..."
    )

    result = CleaningResult(
        train=train.copy(),
        val=val.copy(),
        test=test.copy(),
    )

    # -------------------------------------------------------------------------
    # Save metadata
    # -------------------------------------------------------------------------

    if save_artifacts_flag:

        save_cleaning_metadata(
            config=eda_config,
            output_dir=artifacts_dir,
        )

    # -------------------------------------------------------------------------
    # Save datasets
    # -------------------------------------------------------------------------

    result = save_cleaned_splits(
        result,
        output_dir=output_dir,
    )

    logger.info("")
    logger.info(
        result.summary()
    )

    logger.info(
        "DATA CLEANING COMPLETED SUCCESSFULLY"
    )

    return result


# =============================================================================
# CLI
# =============================================================================
#
# Run:
#
#     python -m src.data.cleaning
#
# DVC should call the module/function from the pipeline layer.
#
# =============================================================================

if __name__ == "__main__":

    import sys

    from src.data.data_loader import DataLoader
    from src.data.validation import (
        ValidationError,
        validate_dataframe,
    )

    config = _DEFAULT_CONFIG_PATH

    loader = DataLoader()

    logger.info(
        "Loading interim datasets..."
    )

    try:

        train = loader.load_interim(
            "train.csv"
        )

        val = loader.load_interim(
            "val.csv"
        )

        test = loader.load_interim(
            "test.csv"
        )

    except Exception as exc:

        logger.error(
            f"Failed to load interim datasets: {exc}"
        )

        sys.exit(1)
    # Validate train against project data contract.
    #
    # Validation does NOT modify the dataframe.
    # -------------------------------------------------------------------------

    try:

        validate_dataframe(
            train,
            config_path=config,
            raise_on_failure=True,
        )

    except ValidationError as exc:

        logger.error(
            f"Validation failed: {exc}"
        )

        sys.exit(1)

    # -------------------------------------------------------------------------
    # Run cleaning.
    # -------------------------------------------------------------------------

    try:

        result = run_cleaning(
            train=train,
            val=val,
            test=test,
            config_path=config,
        )

    except (
        CleaningError,
        FileNotFoundError,
        KeyError,
    ) as exc:

        logger.error(
            f"Cleaning failed: {exc}"
        )

        sys.exit(1)

    print(
        result.summary()
    )

    sys.exit(0)


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "CleaningError",
    "CleaningResult",
    "EdaConfig",
    "load_eda_config",
    "validate_cleaning_inputs",
    "check_missing_values",
    "validate_configured_columns",
    "save_cleaning_metadata",
    "save_cleaned_splits",
    "run_cleaning",
]