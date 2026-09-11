# =============================================================================
# api/main.py
# California Housing Project — FastAPI Application
#
# RESPONSIBILITY
# --------------
# Expose the production ML inference engine through a REST API.
#
# ARCHITECTURE
# ------------
# Client
#   ↓
# FastAPI (this file)
#   ↓
# Pydantic Request Validation (api/schemas.py)
#   ↓
# API Prediction Adapter (api/predict.py)
#   ↓
# Core Inference Engine (src/models/predict.py)
#   ↓
# Fitted Production Pipeline
#   ↓
# Prediction Response
#
# IMPORTANT
# ---------
# This file contains API orchestration only.
#
# It does NOT perform:
#   - Data cleaning
#   - Imputation
#   - Feature engineering
#   - Scaling
#   - Encoding
#   - LOF fitting
#   - Model fitting
#   - Manual ML transformations
#   - Schema ↔ inference conversion (belongs to api/predict.py)
#
# All ML inference logic belongs to:
#     src/models/predict.py
#
# All schema-to-inference adaptation belongs to:
#     api/predict.py
# =============================================================================


from __future__ import annotations

from typing import Any

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from api.predict import (
    get_batch_prediction,
    get_single_prediction,
)
from api.schemas import (
    BatchHouseInputSchema,
    BatchPredictionResponseSchema,
    ErrorResponseSchema,
    HealthCheckResponseSchema,
    HouseInputSchema,
    PredictionResponseSchema,
)
from src.models.predict import (
    DEFAULT_MODEL_PATH,
    PredictionError,
    load_model,
)


# =============================================================================
# APPLICATION CONFIGURATION
# =============================================================================


APP_TITLE = "California Housing Prediction API"
APP_DESCRIPTION = """
Production inference API for the California Housing regression model.

The API accepts the nine raw features required by the production pipeline.
All preprocessing and machine-learning transformations are handled by the
fitted production pipeline.
"""

APP_VERSION = "1.0.0"


# =============================================================================
# FASTAPI APPLICATION
# =============================================================================


app = FastAPI(
    title=APP_TITLE,
    description=APP_DESCRIPTION,
    version=APP_VERSION,
)


# =============================================================================
# ROOT ENDPOINT
# =============================================================================


@app.get(
    "/",
    summary="API information",
)
def root() -> dict[str, Any]:
    """
    Return basic API information.

    This endpoint is intentionally lightweight and does not load the model.
    """

    return {
        "service": "California Housing Prediction API",
        "version": APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


# =============================================================================
# HEALTH CHECK
# =============================================================================


@app.get(
    "/health",
    response_model=HealthCheckResponseSchema,
    summary="Health check",
)
def health_check() -> HealthCheckResponseSchema:
    """
    Check whether the production model can be loaded.

    The endpoint verifies the actual production artifact rather than merely
    checking whether the FastAPI application is running.
    """

    try:
        model = load_model()
        model_loaded = model is not None

        return HealthCheckResponseSchema(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            model_path=str(DEFAULT_MODEL_PATH),
        )

    except PredictionError:
        return HealthCheckResponseSchema(
            status="unhealthy",
            model_loaded=False,
            model_path=str(DEFAULT_MODEL_PATH),
        )

    except Exception:
        return HealthCheckResponseSchema(
            status="unhealthy",
            model_loaded=False,
            model_path=str(DEFAULT_MODEL_PATH),
        )


# =============================================================================
# SINGLE PREDICTION
# =============================================================================


@app.post(
    "/predict",
    response_model=PredictionResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Predict house value",
)
def predict_house(
    house: HouseInputSchema,
) -> PredictionResponseSchema:
    """
    Generate a prediction for a single house.

    Request validation is performed by Pydantic (HouseInputSchema).

    The validated input is delegated to the API prediction adapter
    (api/predict.py), which bridges the Pydantic schema and the core
    inference engine.

    Errors raised by the adapter (PredictionError) are automatically
    converted into HTTP 400 responses by the registered exception handler.
    """

    return get_single_prediction(house)


# =============================================================================
# BATCH PREDICTION
# =============================================================================


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponseSchema,
    status_code=status.HTTP_200_OK,
    summary="Predict multiple house values",
)
def predict_houses_batch(
    batch: BatchHouseInputSchema,
) -> BatchPredictionResponseSchema:
    """
    Generate predictions for multiple houses.

    The request must contain between 1 and 1000 houses (enforced by
    BatchHouseInputSchema). Every house is independently validated by
    HouseInputSchema before reaching the inference engine.

    The validated batch is delegated to the API prediction adapter
    (api/predict.py), which bridges the Pydantic schema and the core
    batch inference engine.

    Errors raised by the adapter (PredictionError) are automatically
    converted into HTTP 400 responses by the registered exception handler.
    """

    return get_batch_prediction(batch)


# =============================================================================
# PREDICTION ERROR HANDLER
# =============================================================================


@app.exception_handler(PredictionError)
async def prediction_error_handler(
    request: Request,
    exc: PredictionError,
) -> JSONResponse:
    """
    Convert internal PredictionError exceptions into a standardized
    HTTP 400 response.
    """

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponseSchema(
            error="Prediction error",
            details=str(exc),
        ).model_dump(),
    )


# =============================================================================
# REQUEST VALIDATION ERROR HANDLER
# =============================================================================


@app.exception_handler(RequestValidationError)
async def validation_error_handler(
    request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    """
    Convert FastAPI / Pydantic validation errors into the project's
    standardized ErrorResponseSchema.

    This handles errors such as:
        - Missing required fields
        - Invalid numeric values
        - Invalid enum values
        - Extra forbidden fields
        - Invalid batch structure
    """

    errors = exc.errors()

    formatted_errors: list[str] = []

    for error in errors:
        location = " -> ".join(
            str(part)
            for part in error.get("loc", [])
        )

        message = error.get(
            "msg",
            "Invalid value",
        )

        formatted_errors.append(
            f"{location}: {message}"
        )

    details = "; ".join(formatted_errors)

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponseSchema(
            error="Request validation error",
            details=details,
        ).model_dump(),
    )


# =============================================================================
# GENERIC ERROR HANDLER
# =============================================================================


@app.exception_handler(Exception)
async def generic_error_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Final safety net for unexpected application errors.

    Internal exception details are intentionally not exposed to the client.
    """

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponseSchema(
            error="Internal server error",
            details="An unexpected error occurred while processing the request.",
        ).model_dump(),
    )


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "app",
]


# =============================================================================
# LOCAL DEVELOPMENT ENTRY POINT
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )