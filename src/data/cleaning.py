# =============================================================================
# src/data/cleaning.py
# California Housing Project — Data Cleaning
#
# RESPONSIBILITY
# --------------
# This module is responsible ONLY for:
#   1. Configuration loading
#   2. Strict missing-value validation
#   3. Train-only imputation
#   4. Train-fitted LOF outlier detection
#   5. Saving cleaned train/validation/test datasets
#   6. Saving train-fitted cleaning artifacts
#
# THIS MODULE DOES NOT
# --------------------
#   - Perform feature engineering
#   - Create ratios
#   - Apply log1p transformations
#   - Create target-derived features
#   - Train ML models
#   - Perform model preprocessing
#   - Execute DVC commands
#   - Execute Git commands
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
#                   cleaning_artifacts.json
#                   lof_model.pkl
#                   lof_scaler.pkl
#
# LEAKAGE POLICY
# --------------
# All learned statistics/models are fitted on TRAIN ONLY.
#
#   TRAIN
#      |
#      +--> fit imputer
#      +--> fit scaler
#      +--> fit LOF
#      |
#      +--> transform TRAIN
#      +--> transform VALIDATION
#      +--> transform TEST
#
# DVC
# ---
# DVC is intentionally NOT used inside this module.
# DVC orchestration belongs to dvc.yaml / pipeline orchestration.
#
# =============================================================================

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

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

_IMPUTER_ARTIFACT_FILENAME = "cleaning_artifacts.json"
_LOF_MODEL_FILENAME = "lof_model.pkl"
_LOF_SCALER_FILENAME = "lof_scaler.pkl"


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
    Typed representation of the EDA-derived configuration required by cleaning.

    Expected source:

        configs/data_config.yaml

    Relevant section:

        eda_derived:
            missingness:
                ...
            lof:
                ...
            target_summary:
                ...

    Notes
    -----
    log1p configuration is intentionally NOT loaded here.

    log1p belongs to feature engineering and therefore must not be handled
    by cleaning.py.
    """

    impute_columns: list[str]

    imputation_strategy: dict[str, str]

    cap_threshold: float

    lof_contamination: float

    lof_n_neighbors: int

    lof_features: list[str]


def load_eda_config(
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
) -> EdaConfig:
    """
    Load and validate EDA-derived configuration.

    The function intentionally fails fast if the required configuration
    is missing or malformed.

    Parameters
    ----------
    config_path:
        Path to configs/data_config.yaml.

    Returns
    -------
    EdaConfig

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.

    CleaningError
        If the configuration is malformed or contains invalid values.
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
    # eda_derived is mandatory
    # -------------------------------------------------------------------------

    eda = cfg.get("eda_derived")

    if not isinstance(eda, dict):
        raise CleaningError(
            f"'eda_derived' section is missing or invalid in {config_path}."
        )

    # -------------------------------------------------------------------------
    # Missingness configuration
    # -------------------------------------------------------------------------

    missingness = eda.get("missingness")

    if not isinstance(missingness, dict):
        raise CleaningError(
            f"'eda_derived.missingness' is missing or invalid in {config_path}."
        )

    impute_columns: list[str] = []
    imputation_strategy: dict[str, str] = {}

    for column, column_config in missingness.items():

        if not isinstance(column_config, dict):
            raise CleaningError(
                f"Invalid missingness configuration for column '{column}'."
            )

        # Explicit opt-in.
        #
        # Example:
        #
        # total_bedrooms:
        #     impute: true
        #     imputation_strategy: median
        #
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

            imputation_strategy[column] = strategy

    # -------------------------------------------------------------------------
    # Target summary / cap threshold
    #
    # NOTE:
    # cap_threshold is retained as configuration metadata if it is needed
    # elsewhere in the project.
    #
    # It is NOT used to create is_capped here.
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
        cap_threshold = float(target_summary["cap_threshold"])
    except (TypeError, ValueError) as exc:
        raise CleaningError(
            "cap_threshold must be numeric."
        ) from exc

    # -------------------------------------------------------------------------
    # LOF configuration
    # -------------------------------------------------------------------------

    lof_cfg = eda.get("lof")

    if not isinstance(lof_cfg, dict):
        raise CleaningError(
            "'eda_derived.lof' is missing or invalid."
        )

    required_lof_keys = {
        "contamination",
        "n_neighbors",
        "features",
    }

    missing_lof_keys = required_lof_keys - set(lof_cfg.keys())

    if missing_lof_keys:
        raise CleaningError(
            "Missing required LOF configuration keys: "
            f"{sorted(missing_lof_keys)}"
        )

    # contamination

    try:
        lof_contamination = float(lof_cfg["contamination"])
    except (TypeError, ValueError) as exc:
        raise CleaningError(
            "LOF contamination must be numeric."
        ) from exc

    if not 0 < lof_contamination < 0.5:
        raise CleaningError(
            "LOF contamination must be between 0 and 0.5."
        )

    # n_neighbors

    try:
        lof_n_neighbors = int(lof_cfg["n_neighbors"])
    except (TypeError, ValueError) as exc:
        raise CleaningError(
            "LOF n_neighbors must be an integer."
        ) from exc

    if lof_n_neighbors < 2:
        raise CleaningError(
            "LOF n_neighbors must be >= 2."
        )

    # features

    lof_features = lof_cfg["features"]

    if not isinstance(lof_features, list) or not lof_features:
        raise CleaningError(
            "LOF features must be a non-empty list."
        )

    if not all(isinstance(col, str) for col in lof_features):
        raise CleaningError(
            "All LOF feature names must be strings."
        )

    logger.info(
        "EDA configuration loaded successfully."
    )

    logger.info(
        f"Imputation columns: {impute_columns}"
    )

    logger.info(
        f"LOF features: {lof_features}"
    )

    logger.info(
        f"LOF contamination: {lof_contamination}"
    )

    logger.info(
        f"LOF n_neighbors: {lof_n_neighbors}"
    )

    return EdaConfig(
        impute_columns=impute_columns,
        imputation_strategy=imputation_strategy,
        cap_threshold=cap_threshold,
        lof_contamination=lof_contamination,
        lof_n_neighbors=lof_n_neighbors,
        lof_features=lof_features,
    )


# =============================================================================
# RESULT OBJECT
# =============================================================================

@dataclass
class CleaningResult:
    """
    Result returned by the cleaning pipeline.
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    train_path: Path | None = None
    val_path: Path | None = None
    test_path: Path | None = None

    imputation_statistics: dict[str, Any] = field(
        default_factory=dict
    )

    lof_n_outliers_train: int = 0

    timestamp: datetime = field(
        default_factory=datetime.now
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def summary(self) -> str:
        """
        Return a human-readable cleaning summary.
        """

        lines = [
            "=" * 70,
            "  DATA CLEANING RESULT",
            "=" * 70,
            "",
            f"  Train : {len(self.train):,} rows x {self.train.shape[1]} cols",
            f"  Val   : {len(self.val):,} rows x {self.val.shape[1]} cols",
            f"  Test  : {len(self.test):,} rows x {self.test.shape[1]} cols",
            "",
            "  Imputation statistics:",
        ]

        if self.imputation_statistics:
            for column, value in self.imputation_statistics.items():
                lines.append(
                    f"    {column}: {value}"
                )
        else:
            lines.append(
                "    None"
            )

        lines.extend(
            [
                "",
                f"  LOF outliers flagged (train): "
                f"{self.lof_n_outliers_train:,}",
                "",
                "  Feature engineering: DEFERRED",
                "  DVC tracking: HANDLED OUTSIDE CLEANING",
                "  Git operations: HANDLED OUTSIDE CLEANING",
            ]
        )

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
    """

    splits = {
        "train": train,
        "val": val,
        "test": test,
    }

    for split_name, df in splits.items():

        if not isinstance(df, pd.DataFrame):
            raise CleaningError(
                f"{split_name} must be a pandas DataFrame."
            )

        if df.empty:
            raise CleaningError(
                f"{split_name} DataFrame is empty."
            )

    # Target must exist in all splits because the project is supervised.

    for split_name, df in splits.items():

        if _TARGET not in df.columns:
            raise CleaningError(
                f"Target column '{_TARGET}' is missing from {split_name}."
            )


# =============================================================================
# STRICT MISSING-VALUE VALIDATION
# =============================================================================

def _check_unexpected_nans(
    df: pd.DataFrame,
    allowed_cols: list[str],
    split_name: str,
) -> None:
    """
    Ensure that missing values exist only in explicitly declared
    imputation columns.

    Target is intentionally excluded from imputation and therefore
    missing target values are considered an error.
    """

    nan_columns = df.columns[
        df.isna().any()
    ].tolist()

    # Target missingness is never acceptable.

    if _TARGET in nan_columns:

        raise CleaningError(
            f"[{split_name}] Target column '{_TARGET}' "
            "contains missing values. "
            "Target values must be present before cleaning."
        )

    unexpected_nans = [
        column
        for column in nan_columns
        if column not in allowed_cols
    ]

    if unexpected_nans:

        raise CleaningError(
            f"[{split_name}] Unexpected missing values found in "
            f"columns: {sorted(unexpected_nans)}. "
            f"Only explicitly configured columns may be imputed: "
            f"{sorted(allowed_cols)}."
        )


# =============================================================================
# IMPUTATION
# =============================================================================

def fit_imputer(
    train: pd.DataFrame,
    allowed_cols: list[str],
    strategies: dict[str, str],
) -> dict[str, Any]:
    """
    Fit imputation statistics using TRAIN ONLY.

    No validation or test data is used to calculate statistics.

    Returns
    -------
    dict
        Mapping:

            column -> train-derived fill value
    """

    _check_unexpected_nans(
        train,
        allowed_cols,
        "train",
    )

    stats: dict[str, Any] = {}

    for column in allowed_cols:

        if column not in train.columns:

            raise CleaningError(
                f"Configured imputation column '{column}' "
                "does not exist in train dataset."
            )

        strategy = strategies.get(
            column,
            "median",
        )

        # ---------------------------------------------------------------------
        # Numeric median
        # ---------------------------------------------------------------------

        if strategy == "median":

            if not pd.api.types.is_numeric_dtype(
                train[column]
            ):
                raise CleaningError(
                    f"Column '{column}' is configured for median "
                    "imputation but is not numeric."
                )

            median_value = train[column].median()

            if pd.isna(median_value):

                raise CleaningError(
                    f"Cannot calculate median for '{column}'. "
                    "The column contains no valid values."
                )

            stats[column] = float(
                median_value
            )

            logger.info(
                f"Imputer fitted on train: "
                f"{column} -> median={median_value}"
            )

        # ---------------------------------------------------------------------
        # Numeric mean
        # ---------------------------------------------------------------------

        elif strategy == "mean":

            if not pd.api.types.is_numeric_dtype(
                train[column]
            ):
                raise CleaningError(
                    f"Column '{column}' is configured for mean "
                    "imputation but is not numeric."
                )

            mean_value = train[column].mean()

            if pd.isna(mean_value):

                raise CleaningError(
                    f"Cannot calculate mean for '{column}'."
                )

            stats[column] = float(
                mean_value
            )

            logger.info(
                f"Imputer fitted on train: "
                f"{column} -> mean={mean_value}"
            )

        # ---------------------------------------------------------------------
        # Mode
        # ---------------------------------------------------------------------

        elif strategy == "mode":

            mode_values = train[column].mode()

            if mode_values.empty:

                raise CleaningError(
                    f"Cannot calculate mode for '{column}'."
                )

            stats[column] = mode_values.iloc[0]

            logger.info(
                f"Imputer fitted on train: "
                f"{column} -> mode={stats[column]}"
            )

        else:

            raise CleaningError(
                f"Unsupported imputation strategy '{strategy}' "
                f"for column '{column}'. "
                "Supported strategies: median, mean, mode."
            )

    return stats


def apply_imputer(
    df: pd.DataFrame,
    imputer_stats: dict[str, Any],
    allowed_cols: list[str],
    split_name: str,
) -> pd.DataFrame:
    """
    Apply train-fitted imputation statistics to a dataset.

    IMPORTANT:
    This function NEVER calculates new statistics.
    """

    _check_unexpected_nans(
        df,
        allowed_cols,
        split_name,
    )

    result = df.copy()

    for column, fill_value in imputer_stats.items():

        if column not in result.columns:

            raise CleaningError(
                f"[{split_name}] Imputation column '{column}' "
                "is missing from the dataset."
            )

        missing_count = int(
            result[column].isna().sum()
        )

        if missing_count > 0:

            result[column] = result[column].fillna(
                fill_value
            )

            logger.info(
                f"[{split_name}] Imputed '{column}': "
                f"{missing_count:,} values."
            )

    # -------------------------------------------------------------------------
    # Final safety check
    # -------------------------------------------------------------------------

    remaining_nans = result.columns[
        result.isna().any()
    ].tolist()

    if remaining_nans:

        raise CleaningError(
            f"[{split_name}] Missing values remain after "
            f"imputation: {remaining_nans}"
        )

    return result


# =============================================================================
# LOF
# =============================================================================

def fit_lof(
    train: pd.DataFrame,
    features: list[str],
    contamination: float,
    n_neighbors: int,
) -> tuple[
    LocalOutlierFactor,
    StandardScaler,
    list[str],
]:
    """
    Fit StandardScaler and LOF using TRAIN ONLY.

    Returns
    -------
    lof
        Train-fitted LocalOutlierFactor.

    scaler
        Train-fitted StandardScaler.

    features
        Exact feature list used by LOF.
    """

    # -------------------------------------------------------------------------
    # Feature existence is a hard requirement.
    # -------------------------------------------------------------------------

    missing_features = [
        column
        for column in features
        if column not in train.columns
    ]

    if missing_features:

        raise CleaningError(
            "LOF configuration references missing features: "
            f"{sorted(missing_features)}"
        )

    # -------------------------------------------------------------------------
    # LOF features must not contain NaN.
    # -------------------------------------------------------------------------

    lof_data = train[features]

    if lof_data.isna().any().any():

        nan_features = lof_data.columns[
            lof_data.isna().any()
        ].tolist()

        raise CleaningError(
            "LOF cannot be fitted because the following features "
            f"contain missing values: {nan_features}"
        )

    # -------------------------------------------------------------------------
    # Validate sample size.
    # -------------------------------------------------------------------------

    if len(lof_data) <= n_neighbors:

        raise CleaningError(
            f"LOF requires more rows than n_neighbors. "
            f"Rows={len(lof_data)}, "
            f"n_neighbors={n_neighbors}."
        )

    # -------------------------------------------------------------------------
    # Validate numeric data.
    # -------------------------------------------------------------------------

    non_numeric = [
        column
        for column in features
        if not pd.api.types.is_numeric_dtype(
            train[column]
        )
    ]

    if non_numeric:

        raise CleaningError(
            "LOF features must be numeric. "
            f"Non-numeric features: {non_numeric}"
        )

    # -------------------------------------------------------------------------
    # TRAIN-ONLY scaler
    # -------------------------------------------------------------------------

    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        lof_data
    )

    # -------------------------------------------------------------------------
    # TRAIN-ONLY LOF
    # -------------------------------------------------------------------------

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
    )

    lof.fit(train_scaled)

    logger.info(
        "LOF fitted successfully."
    )

    logger.info(
        f"Rows: {len(lof_data):,}"
    )

    logger.info(
        f"Features: {features}"
    )

    logger.info(
        f"Contamination: {contamination}"
    )

    logger.info(
        f"n_neighbors: {n_neighbors}"
    )

    return (
        lof,
        scaler,
        features,
    )


def apply_lof_flag(
    df: pd.DataFrame,
    lof: LocalOutlierFactor,
    scaler: StandardScaler,
    features: list[str],
    split_name: str,
) -> pd.DataFrame:
    """
    Apply a TRAIN-FITTED LOF model to a dataset.

    No fitting occurs here.
    """

    result = df.copy()

    # -------------------------------------------------------------------------
    # Validate required features.
    # -------------------------------------------------------------------------

    missing_features = [
        column
        for column in features
        if column not in result.columns
    ]

    if missing_features:

        raise CleaningError(
            f"[{split_name}] LOF features missing: "
            f"{sorted(missing_features)}"
        )

    lof_data = result[features]

    # -------------------------------------------------------------------------
    # No missing values allowed.
    # -------------------------------------------------------------------------

    if lof_data.isna().any().any():

        nan_features = lof_data.columns[
            lof_data.isna().any()
        ].tolist()

        raise CleaningError(
            f"[{split_name}] LOF features contain NaN: "
            f"{nan_features}"
        )

    # -------------------------------------------------------------------------
    # Transform using TRAIN-FITTED scaler.
    # -------------------------------------------------------------------------

    scaled_data = scaler.transform(
        lof_data
    )

    # -------------------------------------------------------------------------
    # Predict using TRAIN-FITTED LOF.
    # -------------------------------------------------------------------------

    predictions = lof.predict(
        scaled_data
    )

    # sklearn:
    #
    #   +1 = inlier
    #   -1 = outlier
    #
    result["lof_outlier"] = (
        predictions == -1
    ).astype("int8")

    n_outliers = int(
        result["lof_outlier"].sum()
    )

    logger.info(
        f"[{split_name}] LOF outliers: "
        f"{n_outliers:,} / {len(result):,}"
    )

    return result


# =============================================================================
# ARTIFACTS
# =============================================================================

def save_artifacts(
    imputer_stats: dict[str, Any],
    lof: LocalOutlierFactor,
    lof_scaler: StandardScaler,
    lof_features: list[str],
    eda_config: EdaConfig,
    output_dir: Path | None = None,
) -> Path:
    """
    Save train-fitted cleaning artifacts.

    Artifacts contain everything required to reproduce the cleaning
    transformation on future data.
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

    # -------------------------------------------------------------------------
    # LOF model
    # -------------------------------------------------------------------------

    lof_path = (
        artifacts_dir /
        _LOF_MODEL_FILENAME
    )

    with open(
        lof_path,
        "wb",
    ) as f:

        pickle.dump(
            lof,
            f,
        )

    # -------------------------------------------------------------------------
    # LOF scaler
    # -------------------------------------------------------------------------

    scaler_path = (
        artifacts_dir /
        _LOF_SCALER_FILENAME
    )

    with open(
        scaler_path,
        "wb",
    ) as f:

        pickle.dump(
            lof_scaler,
            f,
        )

    # -------------------------------------------------------------------------
    # JSON metadata
    # -------------------------------------------------------------------------

    metadata = {
        "target": _TARGET,

        "imputer_stats": imputer_stats,

        "impute_columns": eda_config.impute_columns,

        "imputation_strategy": eda_config.imputation_strategy,

        "lof_features": lof_features,

        "lof_contamination": eda_config.lof_contamination,

        "lof_n_neighbors": eda_config.lof_n_neighbors,

        "timestamp": datetime.now().isoformat(),
    }

    metadata_path = (
        artifacts_dir /
        _IMPUTER_ARTIFACT_FILENAME
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
        f"Cleaning artifacts saved to: {artifacts_dir}"
    )

    return metadata_path


def load_artifacts(
    artifacts_dir: Path | None = None,
) -> tuple[
    dict[str, Any],
    LocalOutlierFactor,
    StandardScaler,
    list[str],
]:
    """
    Load train-fitted cleaning artifacts.
    """

    artifacts_dir = (
        artifacts_dir
        if artifacts_dir is not None
        else PROJECT_DIR / _ARTIFACTS_DIRNAME
    )

    metadata_path = (
        artifacts_dir /
        _IMPUTER_ARTIFACT_FILENAME
    )

    lof_path = (
        artifacts_dir /
        _LOF_MODEL_FILENAME
    )

    scaler_path = (
        artifacts_dir /
        _LOF_SCALER_FILENAME
    )

    # -------------------------------------------------------------------------
    # Validate artifact existence.
    # -------------------------------------------------------------------------

    for path in (
        metadata_path,
        lof_path,
        scaler_path,
    ):

        if not path.exists():

            raise FileNotFoundError(
                f"Required cleaning artifact not found: {path}"
            )

    # -------------------------------------------------------------------------
    # Load metadata
    # -------------------------------------------------------------------------

    with open(
        metadata_path,
        encoding="utf-8",
    ) as f:

        metadata = json.load(f)

    # -------------------------------------------------------------------------
    # Load LOF
    # -------------------------------------------------------------------------

    with open(
        lof_path,
        "rb",
    ) as f:

        lof = pickle.load(f)

    # -------------------------------------------------------------------------
    # Load scaler
    # -------------------------------------------------------------------------

    with open(
        scaler_path,
        "rb",
    ) as f:

        scaler = pickle.load(f)

    logger.info(
        f"Cleaning artifacts loaded from: {artifacts_dir}"
    )

    return (
        metadata["imputer_stats"],
        lof,
        scaler,
        metadata["lof_features"],
    )


# =============================================================================
# SAVE CLEANED DATA
# =============================================================================

def save_cleaned_splits(
    result: CleaningResult,
    output_dir: Path | None = None,
) -> CleaningResult:
    """
    Save cleaned train/validation/test datasets.

    No target-derived temporary metadata exists in this version,
    so there is nothing to purge before saving.
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
    Execute the complete data-cleaning pipeline.

    Pipeline
    --------
        1. Validate inputs
        2. Load EDA config
        3. Fit imputer on train
        4. Apply imputer to train/val/test
        5. Fit StandardScaler on train for LOF
        6. Fit LOF on train
        7. Predict LOF on train/val/test
        8. Save train-fitted artifacts
        9. Save cleaned datasets

    IMPORTANT
    ---------
    No DVC/Git operations are performed here.
    """

    logger.info("=" * 70)
    logger.info("DATA CLEANING STARTED")
    logger.info("=" * 70)

    # =========================================================================
    # STEP 1 — INPUT VALIDATION
    # =========================================================================

    logger.info(
        "Step 1/6 - Validating cleaning inputs..."
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
        "Step 2/6 - Loading EDA-derived configuration..."
    )

    eda_config = load_eda_config(
        config_path
    )

    # =========================================================================
    # STEP 3 — IMPUTATION
    # =========================================================================

    logger.info(
        "Step 3/6 - Train-only imputation..."
    )

    imputer_stats = fit_imputer(
        train=train,
        allowed_cols=eda_config.impute_columns,
        strategies=eda_config.imputation_strategy,
    )

    train = apply_imputer(
        df=train,
        imputer_stats=imputer_stats,
        allowed_cols=eda_config.impute_columns,
        split_name="train",
    )

    val = apply_imputer(
        df=val,
        imputer_stats=imputer_stats,
        allowed_cols=eda_config.impute_columns,
        split_name="val",
    )

    test = apply_imputer(
        df=test,
        imputer_stats=imputer_stats,
        allowed_cols=eda_config.impute_columns,
        split_name="test",
    )

    # =========================================================================
    # STEP 4 — LOF FIT
    # =========================================================================

    logger.info(
        "Step 4/6 - Fitting LOF on train only..."
    )

    lof_model, lof_scaler, lof_features = fit_lof(
        train=train,
        features=eda_config.lof_features,
        contamination=eda_config.lof_contamination,
        n_neighbors=eda_config.lof_n_neighbors,
    )

    # =========================================================================
    # STEP 5 — LOF TRANSFORMATION
    # =========================================================================

    logger.info(
        "Step 5/6 - Applying train-fitted LOF..."
    )

    train = apply_lof_flag(
        df=train,
        lof=lof_model,
        scaler=lof_scaler,
        features=lof_features,
        split_name="train",
    )

    val = apply_lof_flag(
        df=val,
        lof=lof_model,
        scaler=lof_scaler,
        features=lof_features,
        split_name="val",
    )

    test = apply_lof_flag(
        df=test,
        lof=lof_model,
        scaler=lof_scaler,
        features=lof_features,
        split_name="test",
    )

    # =========================================================================
    # STEP 6 — SAVE
    # =========================================================================

    logger.info(
        "Step 6/6 - Saving outputs and artifacts..."
    )

    result = CleaningResult(
        train=train,
        val=val,
        test=test,
        imputation_statistics=imputer_stats,
        lof_n_outliers_train=int(
            train["lof_outlier"].sum()
        ),
    )

    # -------------------------------------------------------------------------
    # Save train-fitted artifacts
    # -------------------------------------------------------------------------

    if save_artifacts_flag:

        save_artifacts(
            imputer_stats=imputer_stats,
            lof=lof_model,
            lof_scaler=lof_scaler,
            lof_features=lof_features,
            eda_config=eda_config,
            output_dir=artifacts_dir,
        )

    # -------------------------------------------------------------------------
    # Save cleaned datasets
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
# The CLI is intentionally simple.
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

    # -------------------------------------------------------------------------
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

    except (CleaningError, FileNotFoundError, KeyError) as exc:

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
    "fit_imputer",
    "apply_imputer",
    "fit_lof",
    "apply_lof_flag",
    "save_artifacts",
    "load_artifacts",
    "save_cleaned_splits",
    "run_cleaning",
]
