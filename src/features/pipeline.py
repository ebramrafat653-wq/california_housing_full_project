# =============================================================================
# src/features/pipeline.py
# California Housing Project — CV-Safe ML Pipeline
#
# RESPONSIBILITY
# --------------
# Build leakage-safe sklearn pipelines for:
#   - Cross-Validation
#   - Hyperparameter tuning
#   - Final training
#   - Inference
#
# ARCHITECTURE
# ------------
#
#   train_clean.csv
#          |
#          v
#   DataFrameImputer
#   (configured numeric columns only)
#          |
#          v
#      LOFTransformer
#   (fit inside each CV fold)
#          |
#          v
#      FeatureEngineer
#   (single source of truth from engineering.py)
#          |
#          v
#     ColumnTransformer
#       |      |      |       |
#       v      v      v       v
#     Std    Robust   OHE   Passthrough
#          |
#          v
#         Model
#
# IMPORTANT CV POLICY
# -------------------
# Every learned transformation is inside the sklearn Pipeline:
#   - Imputation
#   - LOF
#   - Scaling
#   - Encoding
#   - Model
#
# Therefore sklearn clones/fits the complete pipeline independently
# inside every CV fold.
#
# NO preprocessing is fitted outside the sklearn pipeline.
#
# FEATURE ENGINEERING
# -------------------
# FeatureEngineer is imported directly from:
#
#     src.features.engineering
#
# This guarantees a single source of truth for deterministic feature
# engineering logic.
#
# =============================================================================

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

# -----------------------------------------------------------------------------
# SINGLE SOURCE OF TRUTH FOR FEATURE ENGINEERING
# -----------------------------------------------------------------------------

from src.features.engineering import FeatureEngineer
from src.utils.logger import get_logger
from src.utils.paths import PROJECT_DIR


logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

_TARGET: str = "median_house_value"
_FORBIDDEN_FEATURES: set[str] = {"is_capped"}

_DEFAULT_CONFIG_PATH: Path = (
    PROJECT_DIR / "configs" / "data_config.yaml"
)

_DEFAULT_ARTIFACTS_DIR: Path = PROJECT_DIR / "artifacts"

_FINAL_PIPELINE_NAME: str = "final_model_pipeline.pkl"

_DEFAULT_LOF_OUTPUT_COLUMN: str = "lof_outlier"


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================


class PipelineError(Exception):
    """Raised when the ML pipeline cannot be constructed or used safely."""


# =============================================================================
# DATAFRAME-PRESERVING NUMERIC IMPUTER
# =============================================================================


class DataFrameImputer(
    BaseEstimator,
    TransformerMixin,
):
    """
    sklearn-compatible numeric imputer that preserves DataFrame structure.

    Why this transformer exists
    ---------------------------
    Standard SimpleImputer returns a NumPy array.

    The next pipeline stages need DataFrame column names because:
        - LOF selects columns by name.
        - FeatureEngineer selects columns by name.

    Therefore we preserve the DataFrame interface.

    Parameters
    ----------
    strategy:
        Imputation strategy used for configured columns.

    columns:
        Explicit list of numeric columns to impute.

        If None, all numeric columns are selected. This fallback exists
        for development compatibility, but production usage in this project
        should pass explicit configured columns.
    """

    def __init__(
        self,
        strategy: str = "median",
        columns: list[str] | None = None,
    ) -> None:
        self.strategy = strategy
        self.columns = columns

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "DataFrameImputer":
        """
        Fit the imputer using only the current CV training fold.
        """

        if not isinstance(X, pd.DataFrame):
            raise PipelineError(
                "DataFrameImputer expects a pandas DataFrame."
            )

        if X.empty:
            raise PipelineError(
                "DataFrameImputer received an empty DataFrame."
            )

        # Preserve original schema.
        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )

        # ---------------------------------------------------------------------
        # Resolve explicit columns.
        # ---------------------------------------------------------------------

        if self.columns is not None:
            self.impute_cols_ = list(self.columns)

            missing = [
                column
                for column in self.impute_cols_
                if column not in X.columns
            ]

            if missing:
                raise PipelineError(
                    "Configured imputation columns are missing from input: "
                    f"{sorted(missing)}"
                )

        else:
            # Development fallback only.
            self.impute_cols_ = (
                X.select_dtypes(
                    include=[np.number]
                )
                .columns
                .tolist()
            )

        if not self.impute_cols_:
            raise PipelineError(
                "DataFrameImputer has no columns to impute."
            )

        # ---------------------------------------------------------------------
        # Validate numeric dtype.
        # ---------------------------------------------------------------------

        non_numeric = [
            column
            for column in self.impute_cols_
            if not pd.api.types.is_numeric_dtype(
                X[column]
            )
        ]

        if non_numeric:
            raise PipelineError(
                "DataFrameImputer received non-numeric columns: "
                f"{sorted(non_numeric)}"
            )

        # ---------------------------------------------------------------------
        # Fit sklearn imputer.
        # ---------------------------------------------------------------------

        self.imputer_ = SimpleImputer(
            strategy=self.strategy,
        )

        self.imputer_.fit(
            X[self.impute_cols_]
        )

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply previously-fitted imputation while preserving DataFrame schema.
        """

        if not hasattr(self, "imputer_"):
            raise PipelineError(
                "DataFrameImputer must be fitted before transform."
            )

        if not isinstance(X, pd.DataFrame):
            raise PipelineError(
                "DataFrameImputer expects a pandas DataFrame."
            )

        missing = [
            column
            for column in self.impute_cols_
            if column not in X.columns
        ]

        if missing:
            raise PipelineError(
                "DataFrameImputer cannot transform input because required "
                f"columns are missing: {sorted(missing)}"
            )

        X_out = X.copy()

        transformed = self.imputer_.transform(
            X[self.impute_cols_]
        )

        X_out.loc[:, self.impute_cols_] = transformed

        return X_out

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        """
        Preserve the DataFrame column schema for sklearn tooling.
        """

        if not hasattr(self, "feature_names_in_"):
            raise PipelineError(
                "DataFrameImputer must be fitted before "
                "get_feature_names_out()."
            )

        if input_features is None:
            features = self.feature_names_in_
        else:
            features = np.asarray(
                input_features,
                dtype=object,
            )

        return np.asarray(
            features,
            dtype=object,
        )


# =============================================================================
# CONFIGURATION
# =============================================================================


def load_pipeline_config(
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    """
    Load all pipeline-related configuration from data_config.yaml.

    Returns
    -------
    dict
        {
            "std": [...],
            "robust": [...],
            "cat": [...],
            "passthrough": [...],
            "lof": {
                "features": [...],
                "n_neighbors": int,
                "contamination": float,
            },
            "imputation": {
                "columns": [...],
                "strategies": {
                    "column_name": "median",
                    ...
                }
            }
        }
    """

    config_path = Path(config_path)

    if not config_path.exists():
        raise PipelineError(
            f"Config file not found: {config_path}"
        )

    try:
        with open(
            config_path,
            encoding="utf-8",
        ) as file:
            cfg = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise PipelineError(
            f"Failed to parse YAML config: {config_path}"
        ) from exc

    if not isinstance(cfg, dict):
        raise PipelineError(
            "data_config.yaml must contain a mapping."
        )

    # -------------------------------------------------------------------------
    # FEATURES CONFIG
    # -------------------------------------------------------------------------

    features_cfg = cfg.get(
        "features_config",
        {},
    )

    if not isinstance(features_cfg, dict):
        raise PipelineError(
            "'features_config' section is invalid."
        )

    std_columns = list(
        features_cfg.get(
            "numerical_standard_scaler",
            [],
        )
    )

    robust_columns = list(
        features_cfg.get(
            "numerical_robust_scaler",
            [],
        )
    )

    categorical_columns = list(
        features_cfg.get(
            "categorical_one_hot",
            [],
        )
    )

    passthrough_columns = list(
        features_cfg.get(
            "passthrough",
            [],
        )
    )

    # -------------------------------------------------------------------------
    # EDA CONFIG
    # -------------------------------------------------------------------------

    eda_cfg = cfg.get(
        "eda_derived",
        {},
    )

    if not isinstance(eda_cfg, dict):
        raise PipelineError(
            "'eda_derived' section is invalid."
        )

    # -------------------------------------------------------------------------
    # LOF CONFIG
    # -------------------------------------------------------------------------

    lof_cfg = eda_cfg.get(
        "lof",
        {},
    )

    if not isinstance(lof_cfg, dict):
        raise PipelineError(
            "'eda_derived.lof' section is invalid."
        )

    lof_features = list(
        lof_cfg.get(
            "features",
            [
                "median_income",
                "total_rooms",
                "population",
                "households",
                "longitude",
                "latitude",
            ],
        )
    )

    try:
        lof_n_neighbors = int(
            lof_cfg.get(
                "n_neighbors",
                20,
            )
        )

        lof_contamination = float(
            lof_cfg.get(
                "contamination",
                0.02,
            )
        )

    except (TypeError, ValueError) as exc:
        raise PipelineError(
            "LOF n_neighbors and contamination must be numeric."
        ) from exc

    # -------------------------------------------------------------------------
    # IMPUTATION CONFIG
    # -------------------------------------------------------------------------

    missingness_cfg = eda_cfg.get(
        "missingness",
        {},
    )

    if not isinstance(missingness_cfg, dict):
        raise PipelineError(
            "'eda_derived.missingness' section is invalid."
        )

    impute_columns: list[str] = []
    imputation_strategies: dict[str, str] = {}

    supported_strategies = {
        "mean",
        "median",
        "most_frequent",
        "constant",
    }

    for column, column_cfg in missingness_cfg.items():

        if not isinstance(column_cfg, dict):
            raise PipelineError(
                f"Invalid missingness configuration for '{column}'."
            )

        should_impute = bool(
            column_cfg.get(
                "impute",
                False,
            )
        )

        if not should_impute:
            continue

        strategy = str(
            column_cfg.get(
                "imputation_strategy",
                "median",
            )
        )

        if strategy not in supported_strategies:
            raise PipelineError(
                f"Unsupported imputation strategy '{strategy}' "
                f"for column '{column}'. "
                f"Supported values: {sorted(supported_strategies)}"
            )

        impute_columns.append(column)
        imputation_strategies[column] = strategy

    # -------------------------------------------------------------------------
    # CONFIGURATION VALIDATION
    # -------------------------------------------------------------------------

    all_preprocessor_columns = (
        std_columns
        + robust_columns
        + categorical_columns
        + passthrough_columns
    )

    duplicate_preprocessor_columns = {
        column
        for column in all_preprocessor_columns
        if all_preprocessor_columns.count(column) > 1
    }

    if duplicate_preprocessor_columns:
        raise PipelineError(
            "Features are assigned to multiple preprocessing groups: "
            f"{sorted(duplicate_preprocessor_columns)}"
        )

    if _TARGET in (
        set(all_preprocessor_columns)
        | set(lof_features)
        | set(impute_columns)
    ):
        raise PipelineError(
            f"Target '{_TARGET}' cannot be used as an input feature."
        )

    forbidden_in_config = (
        _FORBIDDEN_FEATURES
        & (
            set(all_preprocessor_columns)
            | set(lof_features)
            | set(impute_columns)
        )
    )

    if forbidden_in_config:
        raise PipelineError(
            "Target-derived features cannot enter the ML pipeline: "
            f"{sorted(forbidden_in_config)}"
        )

    if lof_n_neighbors < 1:
        raise PipelineError(
            "LOF n_neighbors must be >= 1."
        )

    if not 0 < lof_contamination <= 0.5:
        raise PipelineError(
            "LOF contamination must be in (0, 0.5]."
        )

    logger.info(
        "Pipeline configuration loaded successfully."
    )

    return {
        "std": std_columns,
        "robust": robust_columns,
        "cat": categorical_columns,
        "passthrough": passthrough_columns,
        "lof": {
            "features": lof_features,
            "n_neighbors": lof_n_neighbors,
            "contamination": lof_contamination,
        },
        "imputation": {
            "columns": impute_columns,
            "strategies": imputation_strategies,
        },
    }


# =============================================================================
# CONFIGURATION RESOLUTION
# =============================================================================


def resolve_columns(
    config: dict[str, Any],
) -> dict[str, list[str]]:
    """
    Resolve feature groups directly from configuration.

    IMPORTANT
    ---------
    This function intentionally does NOT check whether engineered columns
    exist in the input DataFrame.

    Example:
        rooms_per_household
        bedrooms_per_room
        dist_SF
        dist_LA

    are created later by FeatureEngineer.

    Raw input schema validation belongs to the training/data layer.
    """

    if not isinstance(config, dict):
        raise PipelineError(
            "Pipeline configuration must be a dictionary."
        )

    lof_cfg = config.get(
        "lof",
        {},
    )

    if not isinstance(lof_cfg, dict):
        raise PipelineError(
            "'lof' configuration must be a dictionary."
        )

    return {
        "std": list(
            config.get(
                "std",
                [],
            )
        ),
        "robust": list(
            config.get(
                "robust",
                [],
            )
        ),
        "cat": list(
            config.get(
                "cat",
                [],
            )
        ),
        "passthrough": list(
            config.get(
                "passthrough",
                [],
            )
        ),
        "lof": list(
            lof_cfg.get(
                "features",
                [],
            )
        ),
    }


# =============================================================================
# RAW INPUT VALIDATION
# =============================================================================


def validate_raw_input_schema(
    X: pd.DataFrame,
    cols: dict[str, list[str]],
    config: dict[str, Any],
) -> None:
    """
    Validate that all columns required BEFORE FeatureEngineer exist.

    This function validates ONLY raw/input requirements.

    It does NOT require engineered columns such as:
        rooms_per_household
        bedrooms_per_room
        population_per_household
        dist_SF
        dist_LA
    """

    if not isinstance(X, pd.DataFrame):
        raise PipelineError(
            "Pipeline input must be a pandas DataFrame."
        )

    if X.empty:
        raise PipelineError(
            "Pipeline input DataFrame is empty."
        )

    if _TARGET in X.columns:
        raise PipelineError(
            f"Target '{_TARGET}' must not be present inside X."
        )

    forbidden_present = (
        _FORBIDDEN_FEATURES
        & set(X.columns)
    )

    if forbidden_present:
        raise PipelineError(
            "Forbidden target-derived columns detected in X: "
            f"{sorted(forbidden_present)}"
        )

    # -------------------------------------------------------------------------
    # Required raw columns for LOF.
    # -------------------------------------------------------------------------

    missing_lof = [
        column
        for column in cols["lof"]
        if column not in X.columns
    ]

    if missing_lof:
        raise PipelineError(
            "LOF requires missing raw columns: "
            f"{sorted(missing_lof)}"
        )

    # -------------------------------------------------------------------------
    # Required raw columns for deterministic FeatureEngineer.
    #
    # We derive the required columns from config rather than hardcoding
    # engineering formulas here.
    # -------------------------------------------------------------------------

    feature_cfg = config.get(
        "eda_derived",
        {},
    )

    if not isinstance(feature_cfg, dict):
        raise PipelineError(
            "'eda_derived' configuration is invalid."
        )

    engineered_cfg = feature_cfg.get(
        "engineered_features",
        {},
    )

    if not isinstance(engineered_cfg, dict):
        raise PipelineError(
            "'eda_derived.engineered_features' configuration is invalid."
        )

    required_engineering_columns: set[str] = set()

    ratios = engineered_cfg.get(
        "ratios",
        {},
    )

    if not isinstance(ratios, dict):
        raise PipelineError(
            "'engineered_features.ratios' must be a dictionary."
        )

    for formula in ratios.values():

        if not isinstance(formula, str):
            raise PipelineError(
                "Every ratio formula must be a string."
            )

        parts = [
            part.strip()
            for part in formula.split("/")
        ]

        if len(parts) != 2:
            raise PipelineError(
                f"Invalid ratio formula: '{formula}'"
            )

        required_engineering_columns.update(
            parts
        )

    distances = engineered_cfg.get(
        "distances",
        {},
    )

    if not isinstance(distances, dict):
        raise PipelineError(
            "'engineered_features.distances' must be a dictionary."
        )

    if distances:
        required_engineering_columns.update(
            {
                "latitude",
                "longitude",
            }
        )

    drop_columns = engineered_cfg.get(
        "drop_after_engineering",
        [],
    )

    if not isinstance(drop_columns, list):
        raise PipelineError(
            "'drop_after_engineering' must be a list."
        )

    required_engineering_columns.update(
        str(column)
        for column in drop_columns
    )

    missing_engineering_inputs = [
        column
        for column in required_engineering_columns
        if column not in X.columns
    ]

    if missing_engineering_inputs:
        raise PipelineError(
            "FeatureEngineer requires missing raw columns: "
            f"{sorted(missing_engineering_inputs)}"
        )


# =============================================================================
# LOF TRANSFORMER
# =============================================================================


class LOFTransformer(
    BaseEstimator,
    TransformerMixin,
):
    """
    sklearn-compatible Local Outlier Factor transformer.

    Important
    ---------
    This transformer intentionally DOES NOT contain its own imputer.

    Missing-value handling belongs to the preceding DataFrameImputer so
    imputation is performed exactly once and remains inside the CV pipeline.

    LOF is fitted only on the current training fold.

    novelty=True is used so that:
        fit()  -> training fold
        transform() -> validation/test/new observations
    """

    def __init__(
        self,
        features: list[str],
        n_neighbors: int = 20,
        contamination: float = 0.02,
        output_column: str = _DEFAULT_LOF_OUTPUT_COLUMN,
    ) -> None:
        self.features = features
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.output_column = output_column

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "LOFTransformer":
        """
        Fit LOF only on the current training fold.
        """

        if not isinstance(X, pd.DataFrame):
            raise PipelineError(
                "LOFTransformer expects a pandas DataFrame."
            )

        if X.empty:
            raise PipelineError(
                "LOFTransformer received an empty DataFrame."
            )

        missing = [
            column
            for column in self.features
            if column not in X.columns
        ]

        if missing:
            raise PipelineError(
                "LOF features are missing from input: "
                f"{sorted(missing)}"
            )

        if self.n_neighbors < 1:
            raise PipelineError(
                "LOF n_neighbors must be >= 1."
            )

        if not 0 < float(self.contamination) <= 0.5:
            raise PipelineError(
                "LOF contamination must be in (0, 0.5]."
            )

        X_lof = X[self.features].copy()

        # ---------------------------------------------------------------------
        # LOF expects numeric finite input.
        # ---------------------------------------------------------------------

        non_numeric = [
            column
            for column in self.features
            if not pd.api.types.is_numeric_dtype(
                X_lof[column]
            )
        ]

        if non_numeric:
            raise PipelineError(
                "LOF features must be numeric. "
                f"Non-numeric columns: {sorted(non_numeric)}"
            )

        if X_lof.isna().any().any():
            missing_values = (
                X_lof.columns[
                    X_lof.isna().any()
                ]
                .tolist()
            )

            raise PipelineError(
                "LOF received missing values after the upstream "
                "imputation stage. This indicates a pipeline/configuration "
                f"error. Columns: {sorted(missing_values)}"
            )

        values = X_lof.to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise PipelineError(
                "LOF received NaN or infinite values."
            )

        # ---------------------------------------------------------------------
        # Protect against too many neighbors for small folds.
        # ---------------------------------------------------------------------

        effective_neighbors = min(
            int(self.n_neighbors),
            max(
                1,
                len(X_lof) - 1,
            ),
        )

        self.effective_n_neighbors_ = effective_neighbors

        self.lof_ = LocalOutlierFactor(
            n_neighbors=effective_neighbors,
            contamination=float(self.contamination),
            novelty=True,
        )

        self.lof_.fit(values)

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add the LOF outlier flag to the input DataFrame.
        """

        if not hasattr(self, "lof_"):
            raise PipelineError(
                "LOFTransformer must be fitted before transform."
            )

        if not isinstance(X, pd.DataFrame):
            raise PipelineError(
                "LOFTransformer expects a pandas DataFrame."
            )

        missing = [
            column
            for column in self.features
            if column not in X.columns
        ]

        if missing:
            raise PipelineError(
                "LOF features are missing from transform input: "
                f"{sorted(missing)}"
            )

        result = X.copy()

        X_lof = result[
            self.features
        ].copy()

        if X_lof.isna().any().any():
            missing_values = (
                X_lof.columns[
                    X_lof.isna().any()
                ]
                .tolist()
            )

            raise PipelineError(
                "LOF received missing values during transform. "
                "The upstream imputer was not applied correctly. "
                f"Columns: {sorted(missing_values)}"
            )

        values = X_lof.to_numpy(
            dtype=float
        )

        if not np.isfinite(values).all():
            raise PipelineError(
                "LOF received NaN or infinite values during transform."
            )

        predictions = self.lof_.predict(
            values
        )

        result[
            self.output_column
        ] = (
            predictions == -1
        ).astype(
            np.int8
        )

        return result

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        """
        Return input features plus the LOF output column.
        """

        if not hasattr(
            self,
            "feature_names_in_",
        ):
            raise PipelineError(
                "LOFTransformer must be fitted before "
                "get_feature_names_out()."
            )

        features = (
            self.feature_names_in_
            if input_features is None
            else np.asarray(
                input_features,
                dtype=object,
            )
        )

        output = [
            str(feature)
            for feature in features
        ]

        if self.output_column not in output:
            output.append(
                self.output_column
            )

        return np.asarray(
            output,
            dtype=object,
        )


# =============================================================================
# COLUMN TRANSFORMER
# =============================================================================


def build_column_transformer(
    cols: dict[str, list[str]],
) -> ColumnTransformer:
    """
    Build the final preprocessing ColumnTransformer.

    IMPORTANT
    ---------
    The configured columns represent the schema AFTER:
        LOF
        FeatureEngineer

    Therefore engineered columns are valid here even though they do not
    exist in the original train_clean.csv.
    """

    transformers: list[tuple[str, Any, list[str]]] = []

    # -------------------------------------------------------------------------
    # Standard-scaled numeric features
    # -------------------------------------------------------------------------

    if cols["std"]:
        transformers.append(
            (
                "standard_scaler",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            StandardScaler(),
                        ),
                    ]
                ),
                cols["std"],
            )
        )

    # -------------------------------------------------------------------------
    # Robust-scaled numeric features
    # -------------------------------------------------------------------------

    if cols["robust"]:
        transformers.append(
            (
                "robust_scaler",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        ),
                        (
                            "scaler",
                            RobustScaler(),
                        ),
                    ]
                ),
                cols["robust"],
            )
        )

    # -------------------------------------------------------------------------
    # Categorical features
    # -------------------------------------------------------------------------

    if cols["cat"]:
        transformers.append(
            (
                "categorical_encoder",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "one_hot_encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                sparse_output=False,
                                drop=None,
                            ),
                        ),
                    ]
                ),
                cols["cat"],
            )
        )

    # -------------------------------------------------------------------------
    # Passthrough features
    # -------------------------------------------------------------------------

    if cols["passthrough"]:
        transformers.append(
            (
                "passthrough",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median"
                            ),
                        )
                    ]
                ),
                cols["passthrough"],
            )
        )

    if not transformers:
        raise PipelineError(
            "No preprocessing transformers were created."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )


# =============================================================================
# CV-SAFE PREPROCESSING PIPELINE
# =============================================================================


def build_preprocessing_pipeline(
    cols: dict[str, list[str]],
    lof_n_neighbors: int | None = None,
    lof_contamination: float | None = None,
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> Pipeline:
    """
    Build the CV-safe preprocessing pipeline WITHOUT a model.

    Order
    -----
        DataFrameImputer
            ↓
        LOF
            ↓
        FeatureEngineer
            ↓
        ColumnTransformer

    IMPORTANT
    ---------
    When LOF parameters are not explicitly overridden, they are loaded from
    data_config.yaml.

    This avoids hidden hardcoded production parameters.
    """

    config_path = Path(config_path)

    full_cfg = load_pipeline_config(
        config_path
    )

    # -------------------------------------------------------------------------
    # Resolve LOF configuration.
    # -------------------------------------------------------------------------

    configured_lof = full_cfg["lof"]

    resolved_n_neighbors = (
        configured_lof["n_neighbors"]
        if lof_n_neighbors is None
        else int(lof_n_neighbors)
    )

    resolved_contamination = (
        configured_lof["contamination"]
        if lof_contamination is None
        else float(lof_contamination)
    )

    # -------------------------------------------------------------------------
    # Resolve imputation configuration.
    # -------------------------------------------------------------------------

    imputation_cfg = full_cfg.get(
        "imputation",
        {},
    )

    impute_columns = list(
        imputation_cfg.get(
            "columns",
            [],
        )
    )

    strategies = dict(
        imputation_cfg.get(
            "strategies",
            {},
        )
    )

    # -------------------------------------------------------------------------
    # Current project uses median imputation for total_bedrooms.
    #
    # For a more complex future config with multiple strategies, fail
    # explicitly rather than silently applying the wrong strategy.
    # -------------------------------------------------------------------------

    unique_strategies = {
        str(strategy)
        for strategy in strategies.values()
    }

    if len(unique_strategies) > 1:
        raise PipelineError(
            "Multiple initial imputation strategies are configured. "
            "The current DataFrameImputer stage expects one common numeric "
            "strategy. Split this into multiple explicit stages before "
            "introducing mixed strategies."
        )

    initial_imputation_strategy = (
        next(
            iter(unique_strategies)
        )
        if unique_strategies
        else "median"
    )

    # -------------------------------------------------------------------------
    # LOF transformer
    # -------------------------------------------------------------------------

    lof = LOFTransformer(
        features=list(
            cols["lof"]
        ),
        n_neighbors=resolved_n_neighbors,
        contamination=resolved_contamination,
        output_column=_DEFAULT_LOF_OUTPUT_COLUMN,
    )

    # -------------------------------------------------------------------------
    # Final ColumnTransformer
    # -------------------------------------------------------------------------

    column_transformer = build_column_transformer(
        cols
    )

    # -------------------------------------------------------------------------
    # Complete preprocessing chain.
    # -------------------------------------------------------------------------

    preprocessing = Pipeline(
        steps=[
            (
                "initial_imputer",
                DataFrameImputer(
                    strategy=initial_imputation_strategy,
                    columns=impute_columns,
                ),
            ),
            (
                "lof",
                lof,
            ),
            (
                "feature_engineering",
                FeatureEngineer(
                    config_path=config_path,
                    allow_fallback=False,
                ),
            ),
            (
                "preprocessor",
                column_transformer,
            ),
        ]
    )

    logger.info(
        "CV-safe preprocessing pipeline created | "
        f"imputation_columns={impute_columns} | "
        f"imputation_strategy={initial_imputation_strategy} | "
        f"lof_neighbors={resolved_n_neighbors} | "
        f"lof_contamination={resolved_contamination}"
    )

    return preprocessing


# =============================================================================
# COMPLETE MODEL PIPELINE
# =============================================================================


def build_model_pipeline(
    cols: dict[str, list[str]],
    model: Any,
    lof_n_neighbors: int | None = None,
    lof_contamination: float | None = None,
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> Pipeline:
    """
    Build the COMPLETE model pipeline.

    Structure
    ---------
        preprocessing
            ├── initial_imputer
            ├── lof
            ├── feature_engineering
            └── preprocessor

        model
    """

    if model is None:
        raise PipelineError(
            "A valid model estimator is required."
        )

    preprocessing = build_preprocessing_pipeline(
        cols=cols,
        lof_n_neighbors=lof_n_neighbors,
        lof_contamination=lof_contamination,
        config_path=config_path,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessing",
                preprocessing,
            ),
            (
                "model",
                model,
            ),
        ]
    )

    # -------------------------------------------------------------------------
    # Structural safety check.
    # -------------------------------------------------------------------------

    assert_pipeline_is_cv_safe(
        pipeline
    )

    logger.info(
        "Complete model pipeline created | "
        f"model={model.__class__.__name__}"
    )

    return pipeline


# =============================================================================
# DATA SPLIT VALIDATION
# =============================================================================


def validate_split_columns(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target: str = _TARGET,
) -> None:
    """
    Validate basic train/validation/test schema consistency.

    This function does NOT fit or learn anything.
    """

    datasets = {
        "train": train,
        "val": val,
        "test": test,
    }

    for name, df in datasets.items():

        if not isinstance(
            df,
            pd.DataFrame,
        ):
            raise PipelineError(
                f"{name} must be a pandas DataFrame."
            )

        if df.empty:
            raise PipelineError(
                f"{name} DataFrame is empty."
            )

        if target not in df.columns:
            raise PipelineError(
                f"Target '{target}' is missing from {name}."
            )

        if "is_capped" in df.columns:
            raise PipelineError(
                f"Forbidden target-derived column 'is_capped' "
                f"found in {name}."
            )

    train_features = [
        column
        for column in train.columns
        if column != target
    ]

    val_features = [
        column
        for column in val.columns
        if column != target
    ]

    test_features = [
        column
        for column in test.columns
        if column != target
    ]

    if train_features != val_features:
        raise PipelineError(
            "Train and validation feature schemas do not match."
        )

    if train_features != test_features:
        raise PipelineError(
            "Train and test feature schemas do not match."
        )


# =============================================================================
# SPLIT TARGET
# =============================================================================


def split_features_target(
    df: pd.DataFrame,
    target: str = _TARGET,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate X and y while blocking target-derived leakage.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise PipelineError(
            "Input must be a pandas DataFrame."
        )

    if df.empty:
        raise PipelineError(
            "Input DataFrame is empty."
        )

    if target not in df.columns:
        raise PipelineError(
            f"Target '{target}' not found."
        )

    X = df.drop(
        columns=[target]
    ).copy()

    y = df[target].copy()

    if target in X.columns:
        raise PipelineError(
            f"Target '{target}' leaked into X."
        )

    leaked = (
        _FORBIDDEN_FEATURES
        & set(X.columns)
    )

    if leaked:
        raise PipelineError(
            "Target-derived columns detected in X: "
            f"{sorted(leaked)}"
        )

    return X, y


# =============================================================================
# FEATURE NAMES
# =============================================================================


def get_feature_names(
    pipeline: Pipeline,
) -> list[str]:
    """
    Return final feature names produced by ColumnTransformer.
    """

    if not isinstance(
        pipeline,
        Pipeline,
    ):
        raise PipelineError(
            "Expected an sklearn Pipeline."
        )

    try:
        preprocessing = pipeline.named_steps[
            "preprocessing"
        ]

        preprocessor = preprocessing.named_steps[
            "preprocessor"
        ]

        names = (
            preprocessor
            .get_feature_names_out()
            .tolist()
        )

        logger.info(
            f"Extracted {len(names)} final feature names."
        )

        return names

    except Exception as exc:
        logger.error(
            f"Could not extract feature names: {exc}"
        )

        raise PipelineError(
            "Failed to extract feature names."
        ) from exc


# =============================================================================
# FIT FINAL PIPELINE
# =============================================================================


def fit_final_pipeline(
    pipeline: Pipeline,
    train: pd.DataFrame,
    target: str = _TARGET,
) -> Pipeline:
    """
    Fit a complete pipeline on the supplied training dataset.

    The caller is responsible for supplying the correct final training
    dataset, e.g. Train + Validation.
    """

    if not isinstance(
        pipeline,
        Pipeline,
    ):
        raise PipelineError(
            "Expected an sklearn Pipeline."
        )

    X_train, y_train = split_features_target(
        train,
        target=target,
    )

    logger.info(
        "Fitting final model pipeline | "
        f"rows={len(X_train):,}"
    )

    pipeline.fit(
        X_train,
        y_train,
    )

    logger.info(
        "Final model pipeline fitted successfully."
    )

    return pipeline


# =============================================================================
# SAVE FINAL PIPELINE
# =============================================================================


def save_pipeline(
    pipeline: Pipeline,
    output_dir: Path | None = None,
    filename: str = _FINAL_PIPELINE_NAME,
) -> Path:
    """
    Serialize the complete fitted sklearn pipeline.
    """

    if not isinstance(
        pipeline,
        Pipeline,
    ):
        raise PipelineError(
            "Only sklearn Pipeline objects can be saved."
        )

    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else _DEFAULT_ARTIFACTS_DIR
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = output_dir / filename

    with open(
        path,
        "wb",
    ) as file:
        pickle.dump(
            pipeline,
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    size_kb = (
        path.stat().st_size
        / 1024
    )

    logger.info(
        f"Final pipeline saved -> {path} "
        f"({size_kb:.1f} KB)"
    )

    return path


# =============================================================================
# LOAD FINAL PIPELINE
# =============================================================================


def load_pipeline(
    artifacts_dir: Path | None = None,
    filename: str = _FINAL_PIPELINE_NAME,
) -> Pipeline:
    """
    Load a previously serialized complete sklearn pipeline.
    """

    artifacts_dir = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else _DEFAULT_ARTIFACTS_DIR
    )

    path = (
        artifacts_dir
        / filename
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Final model pipeline not found: {path}"
        )

    with open(
        path,
        "rb",
    ) as file:
        pipeline = pickle.load(
            file
        )

    if not isinstance(
        pipeline,
        Pipeline,
    ):
        raise PipelineError(
            "Loaded artifact is not an sklearn Pipeline."
        )

    # Structural verification after deserialization.
    assert_pipeline_is_cv_safe(
        pipeline
    )

    logger.info(
        f"Final pipeline loaded from {path}"
    )

    return pipeline


# =============================================================================
# PREDICTION HELPER
# =============================================================================


def predict(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Generate predictions using the complete fitted pipeline.

    Input must contain the same RAW feature schema expected during training.
    """

    if not isinstance(
        pipeline,
        Pipeline,
    ):
        raise PipelineError(
            "Expected an sklearn Pipeline."
        )

    if not isinstance(
        X,
        pd.DataFrame,
    ):
        raise PipelineError(
            "Prediction input must be a pandas DataFrame."
        )

    if X.empty:
        raise PipelineError(
            "Prediction input is empty."
        )

    if _TARGET in X.columns:
        raise PipelineError(
            f"Target '{_TARGET}' must not be included in prediction input."
        )

    forbidden = (
        _FORBIDDEN_FEATURES
        & set(X.columns)
    )

    if forbidden:
        raise PipelineError(
            "Forbidden target-derived columns found in prediction input: "
            f"{sorted(forbidden)}"
        )

    predictions = pipeline.predict(
        X
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    if not np.isfinite(
        predictions
    ).all():
        raise PipelineError(
            "Model generated NaN or infinite predictions."
        )

    return predictions


# =============================================================================
# CV SAFETY STRUCTURAL CHECK
# =============================================================================


def assert_pipeline_is_cv_safe(
    pipeline: Pipeline,
) -> None:
    """
    Verify the expected leakage-safe pipeline structure.

    This is a structural guard.
    It does not mathematically prove that a custom transformer is leakage-free,
    but it catches accidental architectural changes.
    """

    if not isinstance(
        pipeline,
        Pipeline,
    ):
        raise PipelineError(
            "CV safety check expects an sklearn Pipeline."
        )

    # -------------------------------------------------------------------------
    # Top-level structure
    # -------------------------------------------------------------------------

    required_top_level = {
        "preprocessing",
        "model",
    }

    actual_top_level = set(
        pipeline.named_steps.keys()
    )

    missing_top_level = (
        required_top_level
        - actual_top_level
    )

    if missing_top_level:
        raise PipelineError(
            "Pipeline is missing required top-level steps: "
            f"{sorted(missing_top_level)}"
        )

    expected_top_level_order = [
        "preprocessing",
        "model",
    ]

    actual_top_level_order = list(
        pipeline.named_steps.keys()
    )

    if actual_top_level_order != expected_top_level_order:
        raise PipelineError(
            "Unexpected top-level pipeline order. "
            f"Expected={expected_top_level_order}, "
            f"Actual={actual_top_level_order}"
        )

    # -------------------------------------------------------------------------
    # Preprocessing structure
    # -------------------------------------------------------------------------

    preprocessing = pipeline.named_steps[
        "preprocessing"
    ]

    if not isinstance(
        preprocessing,
        Pipeline,
    ):
        raise PipelineError(
            "'preprocessing' must itself be an sklearn Pipeline."
        )

    expected_preprocessing_order = [
        "initial_imputer",
        "lof",
        "feature_engineering",
        "preprocessor",
    ]

    actual_preprocessing_order = list(
        preprocessing.named_steps.keys()
    )

    if (
        actual_preprocessing_order
        != expected_preprocessing_order
    ):
        raise PipelineError(
            "Unexpected preprocessing step order. "
            f"Expected={expected_preprocessing_order}, "
            f"Actual={actual_preprocessing_order}"
        )

    # -------------------------------------------------------------------------
    # Verify transformer classes.
    # -------------------------------------------------------------------------

    initial_imputer = (
        preprocessing.named_steps[
            "initial_imputer"
        ]
    )

    if not isinstance(
        initial_imputer,
        DataFrameImputer,
    ):
        raise PipelineError(
            "'initial_imputer' must be a DataFrameImputer."
        )

    lof = (
        preprocessing.named_steps[
            "lof"
        ]
    )

    if not isinstance(
        lof,
        LOFTransformer,
    ):
        raise PipelineError(
            "'lof' must be a LOFTransformer."
        )

    feature_engineering = (
        preprocessing.named_steps[
            "feature_engineering"
        ]
    )

    if not isinstance(
        feature_engineering,
        FeatureEngineer,
    ):
        raise PipelineError(
            "'feature_engineering' must use the canonical "
            "FeatureEngineer from src.features.engineering."
        )

    preprocessor = (
        preprocessing.named_steps[
            "preprocessor"
        ]
    )

    if not isinstance(
        preprocessor,
        ColumnTransformer,
    ):
        raise PipelineError(
            "'preprocessor' must be a ColumnTransformer."
        )

    # -------------------------------------------------------------------------
    # Explicitly block the old duplicated transformer name/design.
    # -------------------------------------------------------------------------

    if feature_engineering.__class__.__name__ == (
        "FeatureEngineerTransformer"
    ):
        raise PipelineError(
            "Duplicated FeatureEngineerTransformer detected. "
            "Use src.features.engineering.FeatureEngineer only."
        )

    logger.info(
        "CV-safety structural check passed."
    )


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "PipelineError",
    "DataFrameImputer",
    "LOFTransformer",
    "FeatureEngineer",
    "load_pipeline_config",
    "resolve_columns",
    "validate_raw_input_schema",
    "build_column_transformer",
    "build_preprocessing_pipeline",
    "build_model_pipeline",
    "validate_split_columns",
    "split_features_target",
    "fit_final_pipeline",
    "get_feature_names",
    "save_pipeline",
    "load_pipeline",
    "predict",
    "assert_pipeline_is_cv_safe",
]