# =============================================================================
# api/schemas.py
# California Housing Project — Pydantic API Contracts
#
# RESPONSIBILITY
# --------------
# Define strict request/response contracts for the FastAPI layer.
#
# CONTRACT SOURCE
# ---------------
# Validation rules are derived from configs/data_config.yaml.
#
# IMPORTANT
# ---------
# The API accepts ONLY the 9 raw input features required by the production
# inference pipeline.
#
# The API does NOT perform:
#   - Imputation
#   - Feature engineering
#   - Scaling
#   - Encoding
#   - LOF fitting
#   - Model inference
#
# Those responsibilities remain inside the ML inference pipeline.
# =============================================================================

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# 1. CATEGORICAL ENUM
# =============================================================================


class OceanProximityEnum(str, Enum):
    """
    Valid ocean proximity categories defined by data_config.yaml.
    """

    NEAR_BAY = "NEAR BAY"
    LESS_THAN_1H_OCEAN = "<1H OCEAN"
    INLAND = "INLAND"
    NEAR_OCEAN = "NEAR OCEAN"
    ISLAND = "ISLAND"


# =============================================================================
# 2. SINGLE HOUSE INPUT
# =============================================================================


class HouseInputSchema(BaseModel):
    """
    Strict request schema for a single house prediction.

    The schema contains exactly the 9 raw features expected by the production
    inference pipeline.

    total_bedrooms is allowed to be null because the data contract allows
    missing values for this feature. Imputation is handled downstream by the
    trained ML pipeline.
    """

    # -------------------------------------------------------------------------
    # Pydantic configuration
    # -------------------------------------------------------------------------

    model_config = ConfigDict(
        extra="forbid",
        use_enum_values=True,
    )

    # -------------------------------------------------------------------------
    # Geographic features
    # -------------------------------------------------------------------------

    longitude: float = Field(
        ...,
        ge=-124.35,
        le=-114.0,
        description="District longitude coordinate.",
        examples=[-122.23],
    )

    latitude: float = Field(
        ...,
        ge=32.0,
        le=42.0,
        description="District latitude coordinate.",
        examples=[37.88],
    )

    # -------------------------------------------------------------------------
    # Housing features
    # -------------------------------------------------------------------------

    housing_median_age: float = Field(
        ...,
        ge=1.0,
        le=52.0,
        description="Median age of houses in the district.",
        examples=[41.0],
    )

    total_rooms: float = Field(
        ...,
        gt=0.0,
        description="Total number of rooms in the district.",
        examples=[880.0],
    )

    total_bedrooms: Optional[float] = Field(
        default=None,
        gt=0.0,
        description=(
            "Total number of bedrooms in the district. "
            "May be null; missing values are handled by the ML pipeline."
        ),
        examples=[129.0],
    )

    population: float = Field(
        ...,
        gt=0.0,
        description="Total population residing in the district.",
        examples=[322.0],
    )

    households: float = Field(
        ...,
        gt=0.0,
        description="Total number of households in the district.",
        examples=[126.0],
    )

    # -------------------------------------------------------------------------
    # Economic feature
    # -------------------------------------------------------------------------

    median_income: float = Field(
        ...,
        ge=0.0,
        le=16.0,
        description=(
            "Median income of residents, represented in tens of thousands "
            "of US dollars."
        ),
        examples=[8.3252],
    )

    # -------------------------------------------------------------------------
    # Categorical feature
    # -------------------------------------------------------------------------

    ocean_proximity: OceanProximityEnum = Field(
        ...,
        description="Location category relative to the ocean.",
        examples=["NEAR BAY"],
    )


# =============================================================================
# 3. SINGLE PREDICTION RESPONSE
# =============================================================================


class PredictionResponseSchema(BaseModel):
    """
    Response schema for a single house prediction.
    """

    model_config = ConfigDict(extra="forbid")

    predicted_median_house_value: float = Field(
        ...,
        description="Predicted median house value in US dollars.",
        examples=[452600.0],
    )

    model_version: str = Field(
        default="final_production_pipeline",
        description="Identifier of the production model pipeline used.",
        examples=["final_production_pipeline"],
    )


# =============================================================================
# 4. BATCH INPUT
# =============================================================================


class BatchHouseInputSchema(BaseModel):
    """
    Strict request schema for batch predictions.

    At least one house must be supplied. Every house must independently
    satisfy HouseInputSchema.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    houses: List[HouseInputSchema] = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Non-empty list of houses to predict.",
        examples=[
            [
                {
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
            ]
        ],
    )


# =============================================================================
# 5. BATCH PREDICTION RESPONSE
# =============================================================================


class BatchPredictionResponseSchema(BaseModel):
    """
    Response schema for batch predictions.
    """

    model_config = ConfigDict(extra="forbid")

    predictions: List[float] = Field(
        ...,
        description="Predicted median house values in US dollars.",
        examples=[[452600.0, 315000.0]],
    )

    total_records: int = Field(
        ...,
        ge=0,
        description="Total number of records successfully processed.",
        examples=[2],
    )


# =============================================================================
# 6. HEALTH CHECK RESPONSE
# =============================================================================


class HealthCheckResponseSchema(BaseModel):
    """
    Response schema for the API health check endpoint.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    status: str = Field(
        default="healthy",
        description="Current API service health status.",
        examples=["healthy"],
    )

    model_loaded: bool = Field(
        ...,
        description="Whether the production model pipeline is loaded.",
        examples=[True],
    )

    model_path: str = Field(
        ...,
        description="Path of the loaded production model artifact.",
        examples=["artifacts/final_model_pipeline.pkl"],
    )


# =============================================================================
# 7. ERROR RESPONSE
# =============================================================================


class ErrorResponseSchema(BaseModel):
    """
    Standardized application error response.
    """

    model_config = ConfigDict(
        extra="forbid",
    )

    error: str = Field(
        ...,
        description="Human-readable error message.",
        examples=["Validation error: longitude must be >= -124.35"],
    )

    details: Optional[str] = Field(
        default=None,
        description="Optional additional error details.",
        examples=["longitude must be greater than or equal to -124.35"],
    )


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "OceanProximityEnum",
    "HouseInputSchema",
    "PredictionResponseSchema",
    "BatchHouseInputSchema",
    "BatchPredictionResponseSchema",
    "HealthCheckResponseSchema",
    "ErrorResponseSchema",
]