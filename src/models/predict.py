# =============================================================================
# src/models/predict.py
# California Housing Project — Production Inference Engine
#
# RESPONSIBILITY
# --------------
# Load the fitted production pipeline and perform predictions on NEW / UNSEEN
# data.
#
# CONTRACT
# --------
# - No fitting happens here.
# - No preprocessing is reimplemented here.
# - Feature engineering, imputation, LOF, scaling, encoding, and the model
#   are already stored inside the fitted production pipeline.
# - Input must contain ONLY the 9 raw features expected by the pipeline.
# - Target and target-derived columns are forbidden.
#
# PERFORMANCE
# -----------
# - The fitted production pipeline is cached in memory after the first load.
# - Subsequent inference requests reuse the same loaded pipeline.
# - No disk I/O / deserialization occurs for every prediction request.
# =============================================================================

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.pipeline import load_pipeline
from src.utils.paths import PROJECT_DIR


# =============================================================================
# CONSTANTS & CONTRACTS
# =============================================================================

DEFAULT_MODEL_PATH = (
    PROJECT_DIR
    / "artifacts"
    / "final_model_pipeline.pkl"
)

REQUIRED_FEATURES: tuple[str, ...] = (
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
    "ocean_proximity",
)

TARGET_COLUMN = "median_house_value"

FORBIDDEN_COLUMNS: tuple[str, ...] = (
    TARGET_COLUMN,
    "is_capped",
)

NUMERIC_FEATURES: tuple[str, ...] = (
    "longitude",
    "latitude",
    "housing_median_age",
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
    "median_income",
)

ALLOWED_MISSING_FEATURES: frozenset[str] = frozenset(
    {"total_bedrooms"}
)


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================


class PredictionError(Exception):
    """Raised when inference input or model loading is invalid."""


# =============================================================================
# MODEL LOADING
# =============================================================================


@lru_cache(maxsize=1)
def load_model(
    model_path: str = str(DEFAULT_MODEL_PATH),
):
    """
    Load and cache the fitted production sklearn pipeline.

    The first call loads the pipeline from disk and stores it in memory.
    Subsequent calls with the same model path return the cached pipeline
    without re-reading or deserializing the artifact.

    Parameters
    ----------
    model_path:
        Absolute path or project-relative path to the fitted production
        pipeline.

    Returns
    -------
    object
        Fitted sklearn-compatible pipeline exposing predict().

    Raises
    ------
    PredictionError
        If the artifact does not exist, cannot be loaded, or does not expose
        a predict() method.

    Notes
    -----
    The cache is process-local. If the API runs with multiple worker
    processes, each worker maintains its own in-memory model instance.
    """

    path = Path(model_path)

    # Relative paths are interpreted relative to the project root.
    if not path.is_absolute():
        path = PROJECT_DIR / path

    if not path.exists():
        raise PredictionError(
            f"Final model pipeline not found: {path}. "
            "Run the training pipeline first to create "
            "artifacts/final_model_pipeline.pkl."
        )

    if not path.is_file():
        raise PredictionError(
            f"Model path is not a file: {path}"
        )

    try:
        pipeline = load_pipeline(
            artifacts_dir=path.parent,
            filename=path.name,
        )
    except Exception as exc:
        raise PredictionError(
            f"Failed to load final model pipeline from '{path}': {exc}"
        ) from exc

    if pipeline is None:
        raise PredictionError(
            f"Loaded model pipeline is None: {path}"
        )

    if not callable(getattr(pipeline, "predict", None)):
        raise PredictionError(
            "Loaded artifact does not expose a callable predict() method."
        )

    return pipeline


# =============================================================================
# INPUT NORMALIZATION
# =============================================================================


def _to_dataframe(
    data: (
        pd.DataFrame
        | Mapping[str, Any]
        | Sequence[Mapping[str, Any]]
    ),
) -> pd.DataFrame:
    """
    Convert supported inference inputs to a DataFrame.

    Supported inputs
    ----------------
    - pandas.DataFrame
    - single mapping/dictionary
    - sequence of mappings/dictionaries

    Raises
    ------
    PredictionError
        If the input type is not supported.
    """

    if isinstance(data, pd.DataFrame):
        return data.copy()

    if isinstance(data, Mapping):
        return pd.DataFrame([dict(data)])

    if isinstance(data, Sequence) and not isinstance(
        data,
        (str, bytes, bytearray),
    ):
        try:
            return pd.DataFrame(list(data))
        except Exception as exc:
            raise PredictionError(
                f"Could not convert input sequence to DataFrame: {exc}"
            ) from exc

    raise PredictionError(
        "Input must be one of: pandas.DataFrame, mapping/dict, "
        "or a sequence of dictionaries."
    )


# =============================================================================
# INPUT VALIDATION
# =============================================================================


def validate_input(data: pd.DataFrame) -> None:
    """
    Validate inference input without modifying it.

    Important
    ---------
    Validation happens here, while all actual ML transformations remain
    inside the fitted production pipeline.

    The validation rules are derived from data_config.yaml:
        - 9 required raw features
        - total_bedrooms is the ONLY allowed missing column
        - target and target-derived columns are forbidden
        - numeric columns must be finite
        - ocean_proximity must be a valid string
    """

    if not isinstance(data, pd.DataFrame):
        raise PredictionError(
            "Inference data must be a pandas DataFrame."
        )

    if data.empty:
        raise PredictionError(
            "Inference input is empty."
        )

    # -------------------------------------------------------------------------
    # 1. Forbidden columns
    # -------------------------------------------------------------------------

    forbidden_present = [
        column
        for column in FORBIDDEN_COLUMNS
        if column in data.columns
    ]

    if forbidden_present:
        raise PredictionError(
            "Inference input contains forbidden columns: "
            f"{sorted(forbidden_present)}. "
            "Do not send the target or target-derived columns."
        )

    # -------------------------------------------------------------------------
    # 2. Required columns
    # -------------------------------------------------------------------------

    missing = [
        column
        for column in REQUIRED_FEATURES
        if column not in data.columns
    ]

    if missing:
        raise PredictionError(
            "Inference input is missing required columns: "
            f"{sorted(missing)}"
        )

    # -------------------------------------------------------------------------
    # 3. Unexpected columns
    # -------------------------------------------------------------------------

    unexpected = [
        column
        for column in data.columns
        if column not in REQUIRED_FEATURES
    ]

    if unexpected:
        raise PredictionError(
            "Inference input contains unexpected columns: "
            f"{sorted(unexpected)}. "
            f"Expected exactly: {list(REQUIRED_FEATURES)}"
        )

    # -------------------------------------------------------------------------
    # 4. Numeric feature types
    # -------------------------------------------------------------------------

    non_numeric = [
        column
        for column in NUMERIC_FEATURES
        if not pd.api.types.is_numeric_dtype(data[column])
    ]

    if non_numeric:
        raise PredictionError(
            "The following columns must be numeric: "
            f"{sorted(non_numeric)}"
        )

    # -------------------------------------------------------------------------
    # 5. Missing values
    #
    # total_bedrooms is intentionally allowed to be missing because the
    # production pipeline is responsible for its imputation.
    # -------------------------------------------------------------------------

    invalid_missing = [
        column
        for column in NUMERIC_FEATURES
        if column not in ALLOWED_MISSING_FEATURES
        and data[column].isna().any()
    ]

    if invalid_missing:
        raise PredictionError(
            "Missing values are not allowed in: "
            f"{sorted(invalid_missing)}"
        )

    # -------------------------------------------------------------------------
    # 6. Infinite / invalid numeric values
    # -------------------------------------------------------------------------

    for column in NUMERIC_FEATURES:
        values = pd.to_numeric(
            data[column],
            errors="coerce",
        )

        # NaN is allowed ONLY for total_bedrooms.
        if column in ALLOWED_MISSING_FEATURES:
            values_to_check = values.dropna()
        else:
            values_to_check = values

        if not np.isfinite(
            values_to_check.to_numpy(dtype=float)
        ).all():
            raise PredictionError(
                f"Column '{column}' contains invalid or infinite "
                "numeric values."
            )

    # -------------------------------------------------------------------------
    # 7. Categorical feature validation
    # -------------------------------------------------------------------------

    ocean_proximity = data["ocean_proximity"]

    if ocean_proximity.isna().any():
        raise PredictionError(
            "Column 'ocean_proximity' must not contain missing values."
        )

    if ocean_proximity.astype(str).str.strip().eq("").any():
        raise PredictionError(
            "Column 'ocean_proximity' contains empty values."
        )


# =============================================================================
# PREDICTION
# =============================================================================


def predict(
    data: (
        pd.DataFrame
        | Mapping[str, Any]
        | Sequence[Mapping[str, Any]]
    ),
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> pd.DataFrame:
    """
    Generate predictions for one or many housing records.

    Parameters
    ----------
    data:
        pandas.DataFrame, one dictionary, or a sequence of dictionaries.

    model_path:
        Path to the fitted production pipeline.

    Returns
    -------
    pandas.DataFrame
        Copy of the validated input data with an additional prediction
        column named 'median_house_value_prediction'.

    Raises
    ------
    PredictionError
        If prediction fails or output size does not match input size.

    Notes
    -----
    This function performs inference only.

    It does NOT:
        - fit anything
        - impute anything manually
        - scale anything manually
        - perform feature engineering manually
        - run LOF manually
        - encode categories manually

    The production pipeline is loaded once and cached in memory.
    """

    input_df = _to_dataframe(data)

    validate_input(input_df)

    # Convert Path to string so the lru_cache receives a stable,
    # consistently hashable cache key.
    model = load_model(str(model_path))

    try:
        predictions = model.predict(input_df)
    except Exception as exc:
        raise PredictionError(
            f"Prediction failed: {exc}"
        ) from exc

    # -------------------------------------------------------------------------
    # Ensure predictions are a 1D array and match input size
    # -------------------------------------------------------------------------

    predictions = np.asarray(predictions)

    if predictions.ndim != 1:
        predictions = predictions.reshape(-1)

    if len(predictions) != len(input_df):
        raise PredictionError(
            "Prediction output size does not match input size: "
            f"{len(predictions)} predictions for "
            f"{len(input_df)} records."
        )

    # -------------------------------------------------------------------------
    # Build result DataFrame
    # -------------------------------------------------------------------------

    prediction_column = f"{TARGET_COLUMN}_prediction"

    result = input_df.copy()

    result[prediction_column] = pd.Series(
        predictions,
        index=result.index,
        dtype="float64",
    )

    return result


# =============================================================================
# SINGLE-RECORD HELPER
# =============================================================================


def predict_single(
    data: Mapping[str, Any],
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> float:
    """
    Predict the median house value for exactly one record.

    Parameters
    ----------
    data:
        Single dictionary containing the 9 raw features.

    model_path:
        Path to the fitted production pipeline.

    Returns
    -------
    float
        Predicted median house value.

    Raises
    ------
    PredictionError
        If more than one record is provided.
    """

    result = predict(
        data=data,
        model_path=model_path,
    )

    if len(result) != 1:
        raise PredictionError(
            "predict_single() expected exactly one record, "
            f"received {len(result)}."
        )

    prediction_column = f"{TARGET_COLUMN}_prediction"

    return float(
        result.iloc[0][prediction_column]
    )


# =============================================================================
# BATCH HELPER
# =============================================================================


def predict_batch(
    data: (
        pd.DataFrame
        | Sequence[Mapping[str, Any]]
    ),
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> list[float]:
    """
    Predict multiple housing records.

    Parameters
    ----------
    data:
        pandas.DataFrame or sequence of dictionaries.

    model_path:
        Path to the fitted production pipeline.

    Returns
    -------
    list[float]
        Predictions in the same row order as the input.
    """

    result = predict(
        data=data,
        model_path=model_path,
    )

    prediction_column = f"{TARGET_COLUMN}_prediction"

    return result[prediction_column].astype(float).tolist()


# =============================================================================
# CLI SMOKE TEST
# =============================================================================


def _demo() -> None:
    """
    Minimal local inference smoke test.

    Run from the project root:

        python -m src.models.predict
    """

    sample = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "ocean_proximity": "NEAR BAY",
    }

    try:
        prediction = predict_single(sample)

        print("=" * 70)
        print("INFERENCE TEST RESULT")
        print("=" * 70)
        print(f"Predicted median house value: ${prediction:,.2f}")
        print("=" * 70)

    except Exception as err:
        print(f"Inference Test Failed: {err}")


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "PredictionError",
    "REQUIRED_FEATURES",
    "TARGET_COLUMN",
    "load_model",
    "validate_input",
    "predict",
    "predict_single",
    "predict_batch",
]


if __name__ == "__main__":
    _demo()

