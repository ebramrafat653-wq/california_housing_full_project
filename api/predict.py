# =============================================================================
# api/predict.py
# California Housing Project — API Prediction Adapter
#
# RESPONSIBILITY
# --------------
# Act as the ADAPTER layer between the HTTP/Pydantic layer and the core ML
# inference engine.
#
# This file translates:
#
#     HTTP Request
#         ↓
#     Pydantic Request Schema   (api/schemas.py)
#         ↓
#     Native Python structures  (dict / list[dict])   ← this file
#         ↓
#     Core Inference Engine     (src/models/predict.py)
#         ↓
#     Pydantic Response Schema  (api/schemas.py)
#
# ARCHITECTURE
# ------------
# Client
#   ↓
# FastAPI (api/main.py)
#   ↓
# Pydantic Request Validation (api/schemas.py)
#   ↓
# API Prediction Adapter (api/predict.py)          ← THIS FILE
#   ↓
# Core Inference Engine (src/models/predict.py)
#   ↓
# Fitted Production Pipeline
#   ↓
# Pydantic Response Schema (api/schemas.py)
#
# IMPORTANT
# ---------
# This file is a pure orchestration layer.
#
# It does NOT perform:
#   - Data cleaning
#   - Imputation
#   - Feature engineering
#   - Scaling
#   - Encoding
#   - LOF fitting
#   - Model loading
#   - Model inference
#   - Pydantic field validation
#   - HTTP error handling
#
# Those responsibilities belong to:
#   - api/schemas.py           → field validation
#   - src/models/predict.py    → ML inference
#   - api/main.py              → HTTP routing + error handling
#
# The only responsibilities of this file are:
#   1. Convert Pydantic request models into native Python structures.
#   2. Call the corresponding function in the core inference engine.
#   3. Wrap the raw result in the correct Pydantic response schema.
# =============================================================================


from __future__ import annotations

from api.schemas import (
    BatchHouseInputSchema,
    BatchPredictionResponseSchema,
    HouseInputSchema,
    PredictionResponseSchema,
)
from src.models.predict import (
    predict_batch,
    predict_single,
)


# =============================================================================
# CONSTANTS
# =============================================================================


# Identifier of the production model pipeline, exposed to API clients.
#
# NOTE:
#   This value is intentionally centralized here so that the adapter has a
#   single source of truth when building PredictionResponseSchema.
#
#   It must stay in sync with the default value declared in
#   api/schemas.py::PredictionResponseSchema.model_version.
MODEL_VERSION: str = "final_production_pipeline"


# =============================================================================
# SINGLE PREDICTION ADAPTER
# =============================================================================


def get_single_prediction(
    house: HouseInputSchema,
) -> PredictionResponseSchema:
    """
    Adapt a single-house request into a single-house response.

    This function bridges the API contract (Pydantic) and the ML inference
    engine (pure Python).

    Steps
    -----
    1. Convert the validated HouseInputSchema into a native dictionary.
    2. Delegate inference to src.models.predict.predict_single().
    3. Wrap the numeric result in PredictionResponseSchema.

    Parameters
    ----------
    house:
        A fully validated HouseInputSchema instance. All field-level
        validation (types, ranges, enums, forbidden extras) has already
        been performed by Pydantic before this function is called.

    Returns
    -------
    PredictionResponseSchema
        The API response containing the predicted median house value and
        the production model version identifier.

    Raises
    ------
    PredictionError
        Propagated as-is from the core inference engine.
        The caller (api/main.py) is responsible for translating it into
        an HTTP 400 response through the registered exception handler.

    Notes
    -----
    This function must not:
        - perform any additional validation of the input
        - modify, impute, or transform any feature
        - access the model artifact directly
        - touch HTTP concerns (status codes, JSONResponse, Request)
    """

    # -------------------------------------------------------------------------
    # 1. Pydantic → native dict
    #
    # model_dump() produces a plain dictionary preserving Python-native values
    # (floats stay floats, enums are already converted to their string values
    # thanks to HouseInputSchema.model_config.use_enum_values=True).
    # -------------------------------------------------------------------------

    input_data: dict = house.model_dump()

    # -------------------------------------------------------------------------
    # 2. Delegate to the core inference engine
    #
    # Inversion of Control: this adapter does not know how the prediction is
    # produced. It only knows which function to call and what it returns.
    # -------------------------------------------------------------------------

    prediction_value: float = predict_single(input_data)

    # -------------------------------------------------------------------------
    # 3. Wrap the result in the response schema
    # -------------------------------------------------------------------------

    return PredictionResponseSchema(
        predicted_median_house_value=prediction_value,
        model_version=MODEL_VERSION,
    )


# =============================================================================
# BATCH PREDICTION ADAPTER
# =============================================================================


def get_batch_prediction(
    batch: BatchHouseInputSchema,
) -> BatchPredictionResponseSchema:
    """
    Adapt a batch-house request into a batch-house response.

    This function mirrors get_single_prediction(), but operates on a list
    of houses instead of a single house.

    Steps
    -----
    1. Convert each HouseInputSchema inside the batch into a native dict.
    2. Delegate inference to src.models.predict.predict_batch().
    3. Wrap the list of numeric results in BatchPredictionResponseSchema.

    Parameters
    ----------
    batch:
        A fully validated BatchHouseInputSchema instance. The list has
        already been guaranteed to contain between 1 and 1000 houses by
        Pydantic's Field(min_length=1, max_length=1000) constraint.

    Returns
    -------
    BatchPredictionResponseSchema
        The API response containing the list of predictions and the total
        number of records that were processed.

    Raises
    ------
    PredictionError
        Propagated as-is from the core inference engine.
        The caller (api/main.py) is responsible for translating it into
        an HTTP 400 response through the registered exception handler.

    Notes
    -----
    - The order of the returned predictions MUST match the order of the
      input houses. This invariant is guaranteed by the core engine and
      is part of the public API contract.
    - This function must not perform any additional validation, feature
      engineering, or HTTP-level handling.
    """

    # -------------------------------------------------------------------------
    # 1. Pydantic batch → native list[dict]
    #
    # A list comprehension is used so that the conversion stays explicit and
    # preserves the exact order of the incoming houses.
    # -------------------------------------------------------------------------

    input_data: list[dict] = [
        house.model_dump()
        for house in batch.houses
    ]

    # -------------------------------------------------------------------------
    # 2. Delegate to the core batch inference engine
    # -------------------------------------------------------------------------

    predictions: list[float] = predict_batch(input_data)

    # -------------------------------------------------------------------------
    # 3. Wrap the results in the batch response schema
    #
    # total_records is derived from the prediction list itself rather than
    # from the input list, so that the reported count always reflects the
    # actual number of successful predictions.
    # -------------------------------------------------------------------------

    return BatchPredictionResponseSchema(
        predictions=predictions,
        total_records=len(predictions),
    )


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "MODEL_VERSION",
    "get_single_prediction",
    "get_batch_prediction",
]