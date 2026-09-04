# =============================================================================
# src/features/engineering.py
# California Housing Project — Feature Engineering
#
# RESPONSIBILITY
# --------------
# This module is responsible ONLY for deterministic feature engineering.
#
# It provides two interfaces:
#
# 1. Standalone DVC/data-lineage interface:
#       run_feature_engineering(...)
#
# 2. sklearn-compatible transformer:
#       FeatureEngineer(...)
#
# IMPORTANT CV POLICY
# -------------------
# This transformer does NOT learn statistics from data.
#
# Therefore:
#   - fit() does NOT calculate data statistics.
#   - transform() only performs deterministic transformations.
#
# Learned preprocessing MUST NOT live here:
#   - Imputer       -> CV Pipeline
#   - Scaler        -> CV Pipeline
#   - Encoder       -> CV Pipeline
#   - LOF           -> CV Pipeline
#   - Model         -> Training Pipeline
#
# IMPORTANT ORDER
# ---------------
# For the final training pipeline, FeatureEngineer is intended to run AFTER
# imputation because:
#
#     total_bedrooms
#            ↓
#        Imputer
#            ↓
#     FeatureEngineer
#            ↓
#     bedrooms_per_room
#
# This prevents total_bedrooms missing values from being converted into
# missing engineered features before the CV-safe imputer has run.
#
# DVC POSITION
# ------------
# cleaning.py
#      ↓
# train_clean.csv / val_clean.csv / test_clean.csv
#      ↓
# engineering.py
#      ↓
# train_feat.csv / val_feat.csv / test_feat.csv
#
# NOTE
# ----
# The standalone DVC engineering output is deterministic and is mainly useful
# for data lineage, inspection, and reproducibility.
#
# The SAME FeatureEngineer implementation is also used inside the training
# sklearn Pipeline to guarantee consistency between DVC outputs and training.
#
# This module does NOT:
#   - fit imputers
#   - fit scalers
#   - fit LOF
#   - fit encoders
#   - train models
#   - execute DVC commands
#   - execute Git commands
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin

from src.utils.logger import get_logger
from src.utils.paths import ensure_path


logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

_TARGET = "median_house_value"

_DEFAULT_CONFIG_PATH = "configs/data_config.yaml"

_PROCESSED_DIR = "processed"

_TRAIN_FILENAME = "train_feat.csv"
_VAL_FILENAME = "val_feat.csv"
_TEST_FILENAME = "test_feat.csv"


# =============================================================================
# FALLBACK DEFAULTS
# =============================================================================
#
# These are retained only as explicit development fallbacks.
#
# IMPORTANT:
# The production/DVC path uses strict configuration loading by default.
# =============================================================================

_FALLBACK_RATIOS: dict[str, str] = {
    "rooms_per_household": "total_rooms / households",
    "bedrooms_per_room": "total_bedrooms / total_rooms",
    "population_per_household": "population / households",
}


_FALLBACK_DISTANCES: dict[str, dict[str, float]] = {
    "dist_SF": {
        "lat": 37.77,
        "lon": -122.42,
    },
    "dist_LA": {
        "lat": 34.05,
        "lon": -118.24,
    },
}


_FALLBACK_DROP_COLS: list[str] = [
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
]


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================

class FeatureEngineeringError(Exception):
    """Raised when feature engineering cannot continue safely."""


# =============================================================================
# CONFIG
# =============================================================================

@dataclass(frozen=True)
class FeatureConfig:
    """
    Configuration for deterministic feature engineering.

    No fitted statistics are stored here.
    """

    ratios: dict[str, str]
    distances: dict[str, dict[str, float]]
    drop_cols: list[str]

    source: str = "config"


def _build_fallback_config() -> FeatureConfig:
    """Return development-only fallback configuration."""

    return FeatureConfig(
        ratios=_FALLBACK_RATIOS.copy(),
        distances={
            name: hub.copy()
            for name, hub in _FALLBACK_DISTANCES.items()
        },
        drop_cols=_FALLBACK_DROP_COLS.copy(),
        source="fallback",
    )


def load_feature_config(
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    *,
    allow_fallback: bool = False,
) -> FeatureConfig:
    """
    Load feature engineering configuration.

    Expected YAML structure:

        eda_derived:
          engineered_features:
            ratios:
              rooms_per_household: "total_rooms / households"
              bedrooms_per_room: "total_bedrooms / total_rooms"
              population_per_household: "population / households"

            distances:
              dist_SF:
                lat: 37.77
                lon: -122.42

            drop_after_engineering:
              - total_rooms
              - total_bedrooms
              - population
              - households

    Parameters
    ----------
    config_path:
        Path to project configuration.

    allow_fallback:
        If True, missing/incomplete configuration uses development defaults.

        For DVC/training usage this should remain False.

    Returns
    -------
    FeatureConfig
        Validated deterministic feature configuration.
    """

    config_path = Path(config_path)

    # -------------------------------------------------------------------------
    # Config existence
    # -------------------------------------------------------------------------

    if not config_path.exists():

        if allow_fallback:
            logger.warning(
                f"Config not found: {config_path}. "
                "Using development fallback feature configuration."
            )

            return _build_fallback_config()

        raise FeatureEngineeringError(
            f"Feature engineering config not found: {config_path}. "
            "Refusing to continue without an explicit configuration."
        )

    # -------------------------------------------------------------------------
    # YAML loading
    # -------------------------------------------------------------------------

    try:

        with open(
            config_path,
            encoding="utf-8",
        ) as file:

            cfg = yaml.safe_load(file)

    except yaml.YAMLError as exc:

        raise FeatureEngineeringError(
            f"Malformed YAML configuration: {config_path}"
        ) from exc

    # -------------------------------------------------------------------------
    # Root validation
    # -------------------------------------------------------------------------

    if not isinstance(cfg, dict):

        raise FeatureEngineeringError(
            f"Invalid configuration format: {config_path}. "
            "Expected a YAML mapping."
        )

    # -------------------------------------------------------------------------
    # engineered_features section
    # -------------------------------------------------------------------------

    eda = cfg.get("eda_derived")

    if not isinstance(eda, dict):

        raise FeatureEngineeringError(
            "'eda_derived' section is missing or invalid."
        )

    eng = eda.get("engineered_features")

    if not isinstance(eng, dict):

        if allow_fallback:

            logger.warning(
                "Missing 'eda_derived.engineered_features'. "
                "Using development fallback configuration."
            )

            return _build_fallback_config()

        raise FeatureEngineeringError(
            "'eda_derived.engineered_features' is missing or invalid."
        )

    # -------------------------------------------------------------------------
    # Ratios
    # -------------------------------------------------------------------------

    ratios = eng.get("ratios")

    if ratios is None:

        if allow_fallback:
            ratios = _FALLBACK_RATIOS.copy()
        else:
            raise FeatureEngineeringError(
                "'eda_derived.engineered_features.ratios' "
                "is missing."
            )

    if not isinstance(ratios, dict):

        raise FeatureEngineeringError(
            "'ratios' must be a dictionary."
        )

    validated_ratios: dict[str, str] = {}

    for feature_name, formula in ratios.items():

        if not isinstance(feature_name, str) or not feature_name.strip():

            raise FeatureEngineeringError(
                "Ratio feature names must be non-empty strings."
            )

        if not isinstance(formula, str) or not formula.strip():

            raise FeatureEngineeringError(
                f"Ratio '{feature_name}' must have a valid formula string."
            )

        parts = [
            part.strip()
            for part in formula.split("/")
        ]

        if len(parts) != 2:

            raise FeatureEngineeringError(
                f"Invalid ratio formula for '{feature_name}': "
                f"'{formula}'. Expected 'column_a / column_b'."
            )

        numerator, denominator = parts

        if not numerator or not denominator:

            raise FeatureEngineeringError(
                f"Invalid ratio formula for '{feature_name}': "
                f"'{formula}'."
            )

        validated_ratios[feature_name.strip()] = (
            f"{numerator} / {denominator}"
        )

    # -------------------------------------------------------------------------
    # Distances
    # -------------------------------------------------------------------------

    distances = eng.get("distances")

    if distances is None:

        if allow_fallback:
            distances = _FALLBACK_DISTANCES.copy()
        else:
            raise FeatureEngineeringError(
                "'eda_derived.engineered_features.distances' "
                "is missing."
            )

    if not isinstance(distances, dict):

        raise FeatureEngineeringError(
            "'distances' must be a dictionary."
        )

    validated_distances: dict[str, dict[str, float]] = {}

    for feature_name, hub in distances.items():

        if not isinstance(feature_name, str) or not feature_name.strip():

            raise FeatureEngineeringError(
                "Distance feature names must be non-empty strings."
            )

        if not isinstance(hub, dict):

            raise FeatureEngineeringError(
                f"Distance '{feature_name}' must be a dictionary."
            )

        if "lat" not in hub or "lon" not in hub:

            raise FeatureEngineeringError(
                f"Distance '{feature_name}' must contain "
                "'lat' and 'lon'."
            )

        try:

            lat = float(hub["lat"])
            lon = float(hub["lon"])

        except (TypeError, ValueError) as exc:

            raise FeatureEngineeringError(
                f"Distance '{feature_name}' has invalid "
                "'lat'/'lon' values."
            ) from exc

        if not np.isfinite(lat) or not np.isfinite(lon):

            raise FeatureEngineeringError(
                f"Distance '{feature_name}' has non-finite "
                "'lat'/'lon' values."
            )

        validated_distances[feature_name.strip()] = {
            "lat": lat,
            "lon": lon,
        }

    # -------------------------------------------------------------------------
    # Drop columns
    # -------------------------------------------------------------------------

    drop_cols = eng.get("drop_after_engineering")

    if drop_cols is None:

        if allow_fallback:
            drop_cols = _FALLBACK_DROP_COLS.copy()
        else:
            raise FeatureEngineeringError(
                "'eda_derived.engineered_features."
                "drop_after_engineering' is missing."
            )

    if not isinstance(drop_cols, list):

        raise FeatureEngineeringError(
            "'drop_after_engineering' must be a list."
        )

    validated_drop_cols: list[str] = []

    for column in drop_cols:

        if not isinstance(column, str) or not column.strip():

            raise FeatureEngineeringError(
                "All drop columns must be non-empty strings."
            )

        validated_drop_cols.append(column.strip())

    # -------------------------------------------------------------------------
    # Protect target
    # -------------------------------------------------------------------------

    if _TARGET in validated_drop_cols:

        raise FeatureEngineeringError(
            f"Target column '{_TARGET}' cannot be dropped "
            "during feature engineering."
        )

    # -------------------------------------------------------------------------
    # Protect target-derived feature
    # -------------------------------------------------------------------------

    if "is_capped" in validated_ratios:
        raise FeatureEngineeringError(
            "'is_capped' is target-derived and must never be "
            "created by feature engineering."
        )

    # -------------------------------------------------------------------------
    # Return
    # -------------------------------------------------------------------------

    logger.info(
        "Feature configuration loaded successfully | "
        f"ratios={len(validated_ratios)} | "
        f"distances={len(validated_distances)} | "
        f"drop_cols={len(validated_drop_cols)}"
    )

    return FeatureConfig(
        ratios=validated_ratios,
        distances=validated_distances,
        drop_cols=validated_drop_cols,
        source="config",
    )


# =============================================================================
# INPUT VALIDATION HELPERS
# =============================================================================

def _validate_dataframe(
    X: pd.DataFrame,
    *,
    context: str,
) -> None:
    """Validate that the input is a non-empty DataFrame."""

    if not isinstance(X, pd.DataFrame):

        raise FeatureEngineeringError(
            f"{context}: expected pandas DataFrame, "
            f"got {type(X).__name__}."
        )

    if X.empty:

        raise FeatureEngineeringError(
            f"{context}: input DataFrame is empty."
        )


def _validate_required_columns(
    df: pd.DataFrame,
    columns: set[str],
    *,
    context: str,
) -> None:
    """Validate required input columns."""

    missing = sorted(
        column
        for column in columns
        if column not in df.columns
    )

    if missing:

        raise FeatureEngineeringError(
            f"{context}: missing required columns: {missing}."
        )


def _validate_finite_features(
    df: pd.DataFrame,
    *,
    context: str,
) -> None:
    """
    Ensure engineered numeric features do NOT contain infinity.

    NOTE:
        NaN is intentionally allowed here because downstream CV-safe
        preprocessing (SimpleImputer) will handle it.

        This function ONLY checks for +inf and -inf, not for NaN.
    """

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    if numeric_df.empty:
        return

    # ✅ Check for infinity only, not NaN
    has_inf = np.isinf(numeric_df.to_numpy()).any()

    if has_inf:

        # Find columns with infinite values
        inf_columns = []
        for col in numeric_df.columns:
            if np.isinf(numeric_df[col]).any():
                inf_columns.append(col)

        # Find first few row indices with infinity
        examples = []
        for col in inf_columns[:3]:
            inf_indices = np.where(np.isinf(numeric_df[col]))[0][:3]
            for idx in inf_indices:
                examples.append((idx, col))

        raise FeatureEngineeringError(
            f"{context}: infinite numeric values detected. "
            f"Columns with infinity: {inf_columns}. "
            f"Examples: {examples[:10]}"
        )


# =============================================================================
# RATIO FEATURES
# =============================================================================

def add_ratio_features(
    df: pd.DataFrame,
    ratios: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create deterministic ratio features safely.

    Rules
    -----
    - Zero denominator -> NaN
    - Missing numerator/denominator -> NaN
    - +inf / -inf -> NaN
    - No imputation is performed here
    - No statistics are learned here

    Imputation remains the responsibility of the CV-safe sklearn pipeline.
    """

    _validate_dataframe(
        df,
        context="Ratio feature engineering",
    )

    result = df.copy()
    added: list[str] = []

    for feature_name, formula in ratios.items():

        # ---------------------------------------------------------------------
        # Validate formula
        # ---------------------------------------------------------------------

        if not isinstance(formula, str):
            raise FeatureEngineeringError(
                f"Ratio '{feature_name}' formula must be a string."
            )

        parts = [
            part.strip()
            for part in formula.split("/")
        ]

        if len(parts) != 2:
            raise FeatureEngineeringError(
                f"Invalid ratio formula for '{feature_name}': "
                f"'{formula}'."
            )

        numerator, denominator = parts

        _validate_required_columns(
            result,
            {
                numerator,
                denominator,
            },
            context=f"Ratio '{feature_name}'",
        )

        # ---------------------------------------------------------------------
        # Convert explicitly to numeric arrays.
        #
        # This avoids pandas dtype surprises and gives us deterministic
        # numerical behaviour.
        # ---------------------------------------------------------------------

        numerator_values = pd.to_numeric(
            result[numerator],
            errors="coerce",
        ).to_numpy(
            dtype=float,
        )

        denominator_values = pd.to_numeric(
            result[denominator],
            errors="coerce",
        ).to_numpy(
            dtype=float,
        )

        # ---------------------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------------------

        zero_denominator_count = int(
            np.count_nonzero(
                denominator_values == 0
            )
        )

        if zero_denominator_count > 0:
            logger.warning(
                f"Ratio '{feature_name}': "
                f"{zero_denominator_count:,} zero denominator values. "
                "They will become NaN."
            )

        # ---------------------------------------------------------------------
        # Safe division
        #
        # np.divide with where= prevents division by zero from generating
        # infinity in the first place.
        # ---------------------------------------------------------------------

        ratio_values = np.full(
            shape=numerator_values.shape,
            fill_value=np.nan,
            dtype=float,
        )

        valid_denominator = (
            denominator_values != 0
        )

        np.divide(
            numerator_values,
            denominator_values,
            out=ratio_values,
            where=valid_denominator,
        )

        # ---------------------------------------------------------------------
        # Any non-finite result becomes NaN.
        #
        # This catches:
        #   +inf
        #   -inf
        #   unexpected numerical overflow
        # ---------------------------------------------------------------------

        non_finite_mask = ~np.isfinite(
            ratio_values
        )

        non_finite_count = int(
            np.count_nonzero(
                non_finite_mask
            )
        )

        if non_finite_count > 0:
            logger.warning(
                f"Ratio '{feature_name}': "
                f"{non_finite_count:,} non-finite values converted to NaN."
            )

            ratio_values[
                non_finite_mask
            ] = np.nan

        # ---------------------------------------------------------------------
        # Store as pandas Series while preserving original index.
        # ---------------------------------------------------------------------

        result[feature_name] = pd.Series(
            ratio_values,
            index=result.index,
            dtype="float64",
        )

        added.append(
            feature_name
        )

        logger.info(
            f"Created ratio '{feature_name}' | "
            f"nulls="
            f"{int(result[feature_name].isna().sum()):,} | "
            f"finite="
            f"{int(np.isfinite(ratio_values).sum()):,}"
        )

    return result, added


# =============================================================================
# DISTANCE FEATURES
# =============================================================================

def add_distance_features(
    df: pd.DataFrame,
    distances: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create deterministic Euclidean distance features.

    Distance is measured in latitude/longitude degree space.

    Example
    -------
    dist_SF =
        sqrt(
            (latitude - SF_lat)^2 +
            (longitude - SF_lon)^2
        )

    NOTE
    ----
    This is NOT a kilometer/mile geographic distance.
    """

    _validate_dataframe(
        df,
        context="Distance feature engineering",
    )

    result = df.copy()

    required = {
        "latitude",
        "longitude",
    }

    _validate_required_columns(
        result,
        required,
        context="Distance feature engineering",
    )

    added: list[str] = []

    for feature_name, hub in distances.items():

        if not isinstance(hub, dict):

            raise FeatureEngineeringError(
                f"Distance '{feature_name}' configuration "
                "must be a dictionary."
            )

        if "lat" not in hub or "lon" not in hub:

            raise FeatureEngineeringError(
                f"Distance '{feature_name}' must define "
                "'lat' and 'lon'."
            )

        try:

            hub_lat = float(hub["lat"])
            hub_lon = float(hub["lon"])

        except (TypeError, ValueError) as exc:

            raise FeatureEngineeringError(
                f"Distance '{feature_name}' contains "
                "invalid coordinates."
            ) from exc

        result[feature_name] = np.sqrt(
            (
                result["latitude"] - hub_lat
            ) ** 2
            +
            (
                result["longitude"] - hub_lon
            ) ** 2
        )

        # Explicitly protect against unexpected numerical overflow.
        result[feature_name] = (
            result[feature_name]
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
        )

        added.append(feature_name)

        logger.info(
            f"Created distance '{feature_name}' "
            f"to ({hub_lat}, {hub_lon})"
        )

    return result, added


# =============================================================================
# DROP RAW COLUMNS
# =============================================================================

def drop_raw_columns(
    df: pd.DataFrame,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop raw columns after deterministic feature creation.

    Missing requested columns are treated as an error rather than silently
    ignored. This prevents a configuration/data mismatch from producing
    a silently different feature schema.
    """

    _validate_dataframe(
        df,
        context="Raw-column removal",
    )

    if _TARGET in drop_cols:

        raise FeatureEngineeringError(
            f"Target '{_TARGET}' cannot be dropped."
        )

    missing = [
        column
        for column in drop_cols
        if column not in df.columns
    ]

    if missing:

        raise FeatureEngineeringError(
            "Configured columns to drop are missing from "
            f"the input DataFrame: {missing}."
        )

    result = df.drop(
        columns=drop_cols
    )

    for column in drop_cols:

        logger.info(
            f"Dropped raw column: '{column}'"
        )

    return result, drop_cols.copy()


# =============================================================================
# SKLEARN TRANSFORMER
# =============================================================================

class FeatureEngineer(
    BaseEstimator,
    TransformerMixin,
):
    """
    sklearn-compatible deterministic feature engineering transformer.

    This class is specifically designed to be placed inside the CV-safe
    training Pipeline.

    Example
    -------
    Pipeline([
        ("imputer", ...),
        ("feature_engineering", FeatureEngineer(...)),
        ("lof", ...),
        ("preprocessor", ...),
        ("model", ...),
    ])

    Leakage policy
    --------------
    fit()
        Does not learn statistics from X.

    transform()
        Applies deterministic feature formulas only.

    No target values are used.
    """

    def __init__(
        self,
        config_path: str | Path = _DEFAULT_CONFIG_PATH,
        allow_fallback: bool = False,
    ) -> None:

        self.config_path = str(config_path)
        self.allow_fallback = allow_fallback

    def fit(
        self,
        X: pd.DataFrame,
        y: Any = None,
    ) -> "FeatureEngineer":
        """
        Validate input and load deterministic configuration.

        No statistics are learned from X.
        """

        _validate_dataframe(
            X,
            context="FeatureEngineer.fit",
        )

        # ---------------------------------------------------------------------
        # Target must never be part of the feature matrix.
        #
        # We do not automatically drop it because silently dropping target
        # could hide an upstream training-pipeline bug.
        # ---------------------------------------------------------------------

        if _TARGET in X.columns:

            raise FeatureEngineeringError(
                f"FeatureEngineer.fit received target column "
                f"'{_TARGET}' inside X. "
                "Separate X and y before building the training pipeline."
            )

        # ---------------------------------------------------------------------
        # Target-derived feature must never enter X.
        # ---------------------------------------------------------------------

        if "is_capped" in X.columns:

            raise FeatureEngineeringError(
                "Target-derived feature 'is_capped' detected in X. "
                "It must never be used as a model feature."
            )

        # ---------------------------------------------------------------------
        # Load configuration.
        #
        # This is configuration loading, not statistical learning.
        # ---------------------------------------------------------------------

        self.feature_config_ = load_feature_config(
            self.config_path,
            allow_fallback=self.allow_fallback,
        )

        # ---------------------------------------------------------------------
        # Determine input columns for fitted transformer validation.
        # ---------------------------------------------------------------------

        self.feature_names_in_ = np.asarray(
            X.columns,
            dtype=object,
        )

        self.n_features_in_ = X.shape[1]

        logger.debug(
            "FeatureEngineer fitted. "
            "No data statistics were learned."
        )

        return self

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Apply deterministic feature engineering.
        """

        if not hasattr(
            self,
            "feature_config_",
        ):

            raise FeatureEngineeringError(
                "FeatureEngineer has not been fitted. "
                "Call fit() before transform()."
            )

        _validate_dataframe(
            X,
            context="FeatureEngineer.transform",
        )

        # ---------------------------------------------------------------------
        # Target protection.
        # ---------------------------------------------------------------------

        if _TARGET in X.columns:

            raise FeatureEngineeringError(
                f"FeatureEngineer.transform received target column "
                f"'{_TARGET}' inside X."
            )

        if "is_capped" in X.columns:

            raise FeatureEngineeringError(
                "Target-derived feature 'is_capped' detected in X."
            )

        # ---------------------------------------------------------------------
        # Make a copy.
        # ---------------------------------------------------------------------

        result = X.copy()

        # ---------------------------------------------------------------------
        # Validate expected original columns.
        #
        # Note:
        # FeatureEngineer is intentionally strict.
        # The upstream Imputer must preserve the required source columns.
        # ---------------------------------------------------------------------

        expected_source_columns: set[str] = set()

        for formula in self.feature_config_.ratios.values():

            numerator, denominator = [
                part.strip()
                for part in formula.split("/")
            ]

            expected_source_columns.add(
                numerator
            )

            expected_source_columns.add(
                denominator
            )

        if self.feature_config_.distances:

            expected_source_columns.update(
                {
                    "latitude",
                    "longitude",
                }
            )

        # Columns scheduled to be dropped also need to exist before dropping.
        expected_source_columns.update(
            self.feature_config_.drop_cols
        )

        _validate_required_columns(
            result,
            expected_source_columns,
            context="FeatureEngineer.transform",
        )

        # ---------------------------------------------------------------------
        # Step 1 — Ratios
        # ---------------------------------------------------------------------

        result, _ = add_ratio_features(
            result,
            self.feature_config_.ratios,
        )

        # ---------------------------------------------------------------------
        # Step 2 — Distances
        # ---------------------------------------------------------------------

        result, _ = add_distance_features(
            result,
            self.feature_config_.distances,
        )

        # ---------------------------------------------------------------------
        # Step 3 — Drop raw columns
        # ---------------------------------------------------------------------

        result, _ = drop_raw_columns(
            result,
            self.feature_config_.drop_cols,
        )

        # ---------------------------------------------------------------------
        # Final numeric safety check.
        # ---------------------------------------------------------------------

        _validate_finite_features(
            result,
            context="FeatureEngineer.transform",
        )

        # ---------------------------------------------------------------------
        # Ensure deterministic column order.
        #
        # DataFrame column order after transformations is preserved naturally,
        # but this explicit return keeps the implementation predictable.
        # ---------------------------------------------------------------------

        return result

    def get_feature_names_out(
        self,
        input_features: Any = None,
    ) -> np.ndarray:
        """
        Return output feature names.

        This method allows compatibility with sklearn tooling.
        """

        if not hasattr(
            self,
            "feature_names_in_",
        ):

            raise FeatureEngineeringError(
                "FeatureEngineer has not been fitted."
            )

        input_features_array = np.asarray(
            self.feature_names_in_
            if input_features is None
            else input_features,
            dtype=object,
        )

        if _TARGET in input_features_array:

            raise FeatureEngineeringError(
                f"Target '{_TARGET}' cannot be part of "
                "FeatureEngineer input features."
            )

        if "is_capped" in input_features_array:

            raise FeatureEngineeringError(
                "Target-derived feature 'is_capped' cannot "
                "be part of FeatureEngineer input features."
            )

        output_columns = [
            str(column)
            for column in input_features_array
        ]

        # ---------------------------------------------------------------------
        # Add ratio features.
        # ---------------------------------------------------------------------

        for feature_name in self.feature_config_.ratios:

            if feature_name not in output_columns:

                output_columns.append(
                    feature_name
                )

        # ---------------------------------------------------------------------
        # Add distance features.
        # ---------------------------------------------------------------------

        for feature_name in self.feature_config_.distances:

            if feature_name not in output_columns:

                output_columns.append(
                    feature_name
                )

        # ---------------------------------------------------------------------
        # Remove configured raw columns.
        # ---------------------------------------------------------------------

        drop_set = set(
            self.feature_config_.drop_cols
        )

        output_columns = [
            column
            for column in output_columns
            if column not in drop_set
        ]

        return np.asarray(
            output_columns,
            dtype=object,
        )


# =============================================================================
# RESULT OBJECT
# =============================================================================

@dataclass
class EngineeringResult:
    """Result of the standalone feature engineering stage."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    train_path: Path | None = None
    val_path: Path | None = None
    test_path: Path | None = None

    features_added: list[str] = field(
        default_factory=list
    )

    features_dropped: list[str] = field(
        default_factory=list
    )

    feature_config_source: str = "config"

    timestamp: datetime = field(
        default_factory=datetime.now
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def summary(self) -> str:
        """Return a readable engineering summary."""

        lines = [
            "=" * 70,
            "FEATURE ENGINEERING RESULT",
            "=" * 70,
            "",
            (
                f"Train : "
                f"{len(self.train):,} rows x "
                f"{self.train.shape[1]} cols"
            ),
            (
                f"Val   : "
                f"{len(self.val):,} rows x "
                f"{self.val.shape[1]} cols"
            ),
            (
                f"Test  : "
                f"{len(self.test):,} rows x "
                f"{self.test.shape[1]} cols"
            ),
            "",
            f"Config source    : {self.feature_config_source}",
            f"Features added   : {self.features_added}",
            f"Features dropped : {self.features_dropped}",
            "",
            "Learned preprocessing:",
            "  Imputer : DEFERRED TO CV/TRAINING PIPELINE",
            "  Scaler  : DEFERRED TO CV/TRAINING PIPELINE",
            "  LOF     : DEFERRED TO CV/TRAINING PIPELINE",
            "  Encoder : DEFERRED TO CV/TRAINING PIPELINE",
        ]

        if self.train_path:

            lines.extend(
                [
                    "",
                    "Saved outputs:",
                    f"  train -> {self.train_path}",
                    f"  val   -> {self.val_path}",
                    f"  test  -> {self.test_path}",
                ]
            )

        if self.warnings:

            lines.append("")
            lines.append("Warnings:")

            for warning in self.warnings:

                lines.append(
                    f"  - {warning}"
                )

        lines.append(
            "=" * 70
        )

        return "\n".join(lines)


# =============================================================================
# SPLIT VALIDATION
# =============================================================================

def validate_engineered_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Validate consistency across train/validation/test.

    This function does not fit or learn anything.
    """

    splits = {
        "train": train,
        "val": val,
        "test": test,
    }

    # -------------------------------------------------------------------------
    # Basic DataFrame validation
    # -------------------------------------------------------------------------

    for name, df in splits.items():

        _validate_dataframe(
            df,
            context=f"Engineered {name}",
        )

    # -------------------------------------------------------------------------
    # Target must remain available in standalone engineered datasets.
    # -------------------------------------------------------------------------

    for name, df in splits.items():

        if _TARGET not in df.columns:

            raise FeatureEngineeringError(
                f"Target '{_TARGET}' missing from {name}."
            )

    # -------------------------------------------------------------------------
    # Target-derived feature must not exist.
    # -------------------------------------------------------------------------

    for name, df in splits.items():

        if "is_capped" in df.columns:

            raise FeatureEngineeringError(
                f"Target-derived feature 'is_capped' "
                f"found in engineered {name} dataset."
            )

    # -------------------------------------------------------------------------
    # Same schema.
    # -------------------------------------------------------------------------

    train_columns = list(
        train.columns
    )

    val_columns = list(
        val.columns
    )

    test_columns = list(
        test.columns
    )

    if train_columns != val_columns:

        raise FeatureEngineeringError(
            "Train and validation feature schemas do not match."
        )

    if train_columns != test_columns:

        raise FeatureEngineeringError(
            "Train and test feature schemas do not match."
        )

    # -------------------------------------------------------------------------
    # No infinity.
    # -------------------------------------------------------------------------

    for name, df in splits.items():

        _validate_finite_features(
            df,
            context=f"Engineered {name}",
        )


# =============================================================================
# SAVE
# =============================================================================

def save_featured_splits(
    result: EngineeringResult,
    output_dir: Path | None = None,
) -> EngineeringResult:
    """
    Save feature-engineered train/validation/test datasets.
    """

    output_dir = (
        output_dir
        if output_dir is not None
        else ensure_path(_PROCESSED_DIR)
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "train": output_dir / _TRAIN_FILENAME,
        "val": output_dir / _VAL_FILENAME,
        "test": output_dir / _TEST_FILENAME,
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
            path.stat().st_size
            / (1024 * 1024)
        )

        logger.info(
            f"Saved {name} feature dataset -> "
            f"{path} ({size_mb:.2f} MB)"
        )

    return result


# =============================================================================
# STANDALONE DVC/DATA-LINEAGE ENTRY POINT
# =============================================================================

def run_feature_engineering(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    output_dir: Path | None = None,
    *,
    allow_fallback: bool = False,
) -> EngineeringResult:
    """
    Run deterministic feature engineering on train/validation/test.

    IMPORTANT
    ---------
    This standalone function performs NO statistical fitting.

    The transformations are deterministic.

    For the final model-training workflow, the same FeatureEngineer class
    should be placed inside the sklearn CV Pipeline after the CV-safe imputer.
    """

    logger.info("=" * 70)
    logger.info("FEATURE ENGINEERING STARTED")
    logger.info("=" * 70)

    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------

    for name, df in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:

        _validate_dataframe(
            df,
            context=f"Input {name}",
        )

        if _TARGET not in df.columns:

            raise FeatureEngineeringError(
                f"Target '{_TARGET}' missing from input {name}."
            )

        if "is_capped" in df.columns:

            raise FeatureEngineeringError(
                f"Target-derived feature 'is_capped' "
                f"found in input {name}."
            )

    # -------------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------------

    feature_config = load_feature_config(
        config_path,
        allow_fallback=allow_fallback,
    )

    # -------------------------------------------------------------------------
    # The standalone stage keeps target untouched.
    # Separate target from X so FeatureEngineer can enforce the same contract
    # that the training pipeline will use.
    # -------------------------------------------------------------------------

    train_y = train[_TARGET].copy()
    val_y = val[_TARGET].copy()
    test_y = test[_TARGET].copy()

    train_X = train.drop(
        columns=[_TARGET]
    )

    val_X = val.drop(
        columns=[_TARGET]
    )

    test_X = test.drop(
        columns=[_TARGET]
    )

    # -------------------------------------------------------------------------
    # Use the SAME sklearn transformer implementation that training will use.
    # -------------------------------------------------------------------------

    transformer = FeatureEngineer(
        config_path=config_path,
        allow_fallback=allow_fallback,
    )

    # Fit only loads/validates deterministic configuration.
    # It does NOT learn statistics.
    transformer.fit(train_X)

    train_feat_X = transformer.transform(
        train_X
    )

    val_feat_X = transformer.transform(
        val_X
    )

    test_feat_X = transformer.transform(
        test_X
    )

    # -------------------------------------------------------------------------
    # Reattach target for DVC lineage outputs.
    # -------------------------------------------------------------------------

    train_feat = train_feat_X.copy()
    train_feat[_TARGET] = train_y.to_numpy()

    val_feat = val_feat_X.copy()
    val_feat[_TARGET] = val_y.to_numpy()

    test_feat = test_feat_X.copy()
    test_feat[_TARGET] = test_y.to_numpy()

    # -------------------------------------------------------------------------
    # Validate final datasets.
    # -------------------------------------------------------------------------

    validate_engineered_splits(
        train_feat,
        val_feat,
        test_feat,
    )

    # -------------------------------------------------------------------------
    # Determine added/dropped features for reporting.
    # -------------------------------------------------------------------------

    features_added = [
        *feature_config.ratios.keys(),
        *feature_config.distances.keys(),
    ]

    features_dropped = [
        column
        for column in feature_config.drop_cols
        if column in train.columns
    ]

    # -------------------------------------------------------------------------
    # Build result.
    # -------------------------------------------------------------------------

    result = EngineeringResult(
        train=train_feat,
        val=val_feat,
        test=test_feat,
        features_added=features_added,
        features_dropped=features_dropped,
        feature_config_source=feature_config.source,
    )

    # -------------------------------------------------------------------------
    # Save.
    # -------------------------------------------------------------------------

    result = save_featured_splits(
        result,
        output_dir=output_dir,
    )

    logger.info("")
    logger.info(
        result.summary()
    )

    logger.info(
        "FEATURE ENGINEERING COMPLETED SUCCESSFULLY"
    )

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":

    from src.data.data_loader import DataLoader

    logger.info(
        "Loading cleaned splits..."
    )

    loader = DataLoader()

    try:

        train = loader.load_processed(
            "train_clean.csv"
        )

        val = loader.load_processed(
            "val_clean.csv"
        )

        test = loader.load_processed(
            "test_clean.csv"
        )

    except Exception as exc:

        logger.error(
            f"Failed to load cleaned datasets: {exc}"
        )

        raise SystemExit(1) from exc

    try:

        result = run_feature_engineering(
            train=train,
            val=val,
            test=test,
            config_path=_DEFAULT_CONFIG_PATH,
            allow_fallback=False,
        )

    except FeatureEngineeringError as exc:

        logger.error(
            f"Feature engineering failed: {exc}"
        )

        raise SystemExit(1) from exc

    print(
        result.summary()
    )


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "FeatureEngineeringError",
    "FeatureConfig",
    "FeatureEngineer",
    "EngineeringResult",
    "load_feature_config",
    "add_ratio_features",
    "add_distance_features",
    "drop_raw_columns",
    "validate_engineered_splits",
    "save_featured_splits",
    "run_feature_engineering",
]