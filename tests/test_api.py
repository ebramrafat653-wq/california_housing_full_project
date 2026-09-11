# =============================================================================
# tests/test_api.py
# California Housing Project — FastAPI Tests
#
# RESPONSIBILITY
# --------------
# Verify the public API contract without depending on the real production
# model artifact.
#
# TESTING STRATEGY
# ----------------
# - API endpoints are tested through FastAPI TestClient.
# - Inference functions are mocked where appropriate.
# - Pydantic validation is tested through real HTTP requests.
# - The production model is NOT required for unit-level API tests.
# - The "Six Principles" of ML APIs are verified:
#     1. Input Validation
#     2. Health Endpoint
#     3. Model Loaded Once (cached)
#     4. Sync Handlers (CPU-bound → threadpool)
#     5. Response Schema
#     6. Auto-generated Docs (Swagger + ReDoc + OpenAPI)
#
# ARCHITECTURE (post-adapter)
# ---------------------------
#     HTTP Request
#         ↓
#     api/main.py            (routing only)
#         ↓
#     api/predict.py         (adapter: Pydantic ↔ inference)
#         ↓
#     src/models/predict.py  (core inference engine)
#
# MOCKING STRATEGY
# ----------------
# Inference functions are patched where they are CONSUMED, not where they
# are DEFINED. This is required because `from X import Y` creates a local
# reference at import time. Patching the source module would NOT affect the
# already-bound reference inside api.predict.
#
# Correct patch targets:
#   - api.predict.predict_single   ← consumed inside get_single_prediction()
#   - api.predict.predict_batch    ← consumed inside get_batch_prediction()
#   - api.main.load_model          ← consumed inside health_check()
# =============================================================================

from __future__ import annotations

import copy
import inspect
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.main import app, predict_house, predict_houses_batch
from src.models.predict import PredictionError, load_model


# =============================================================================
# TEST CLIENT
# =============================================================================


@pytest.fixture(scope="module")
def client() -> TestClient:
    """Provide a reusable FastAPI test client."""
    return TestClient(app)


# =============================================================================
# VALID PAYLOADS
# =============================================================================


@pytest.fixture
def valid_house_payload() -> dict[str, Any]:
    """
    Provide a valid single-house prediction payload.

    The payload follows HouseInputSchema exactly.
    """

    return {
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


@pytest.fixture
def valid_batch_payload(
    valid_house_payload: dict[str, Any],
) -> dict[str, Any]:
    """
    Provide a valid batch prediction payload.

    Uses deepcopy to guarantee that mutating the second house never
    affects the first (avoids shared-reference surprises).
    """

    second_house = {
        **copy.deepcopy(valid_house_payload),
        "longitude": -122.22,
        "latitude": 37.86,
        "median_income": 8.3014,
    }

    return {
        "houses": [
            copy.deepcopy(valid_house_payload),
            second_house,
        ]
    }


# =============================================================================
# ROOT ENDPOINT
# =============================================================================


def test_root_endpoint(client: TestClient) -> None:
    """Root endpoint should return HTTP 200 with service metadata."""

    response = client.get("/")

    assert response.status_code == 200

    body = response.json()

    assert body["service"] == "California Housing Prediction API"
    assert body["version"] == "1.0.0"
    assert body["status"] == "running"
    assert body["docs"] == "/docs"
    assert body["health"] == "/health"


# =============================================================================
# HEALTH ENDPOINT
# =============================================================================


def test_health_endpoint_when_model_is_available(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Health endpoint should report healthy when the production model
    loads successfully.
    """

    class DummyModel:
        """Minimal fake model for health testing."""

    # load_model is imported directly into api.main, so patching at
    # api.main.load_model is the correct target here.
    monkeypatch.setattr(
        "api.main.load_model",
        lambda: DummyModel(),
    )

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert "model_path" in body


def test_health_endpoint_when_model_fails_to_load(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Health endpoint should return HTTP 200 with an unhealthy status
    when the current API implementation catches model-loading errors.
    """

    def failing_load_model() -> None:
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr(
        "api.main.load_model",
        failing_load_model,
    )

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "unhealthy"
    assert body["model_loaded"] is False


def test_health_endpoint_handles_prediction_error(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    PredictionError raised by load_model() should also be converted into
    an unhealthy health response.
    """

    def failing_load_model() -> None:
        raise PredictionError("Model artifact unavailable")

    monkeypatch.setattr(
        "api.main.load_model",
        failing_load_model,
    )

    response = client.get("/health")

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "unhealthy"
    assert body["model_loaded"] is False


# =============================================================================
# SINGLE PREDICTION — HAPPY PATH
# =============================================================================


def test_predict_single_success(
    client: TestClient,
    valid_house_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid single prediction request should return a prediction."""

    expected_prediction = 452600.0

    # Patch at the CONSUMPTION site inside the adapter.
    # api.predict imports predict_single via `from src.models.predict import ...`
    # so the local reference must be replaced here.
    monkeypatch.setattr(
        "api.predict.predict_single",
        lambda payload: expected_prediction,
    )

    response = client.post(
        "/predict",
        json=valid_house_payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["predicted_median_house_value"] == expected_prediction
    assert body["model_version"] == "final_production_pipeline"


def test_predict_single_accepts_missing_total_bedrooms(
    client: TestClient,
    valid_house_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    total_bedrooms may be None because the production pipeline is
    responsible for downstream imputation.
    """

    payload = {
        **valid_house_payload,
        "total_bedrooms": None,
    }

    monkeypatch.setattr(
        "api.predict.predict_single",
        lambda payload: 300000.0,
    )

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    body = response.json()

    assert body["predicted_median_house_value"] == 300000.0


# =============================================================================
# SINGLE PREDICTION — INPUT VALIDATION
# =============================================================================


def test_predict_rejects_extra_fields(
    client: TestClient,
    valid_house_payload: dict[str, Any],
) -> None:
    """
    API must reject target / forbidden / unknown fields because
    HouseInputSchema uses extra='forbid'.
    """

    payload = {
        **valid_house_payload,
        "median_house_value": 500001.0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

    body = response.json()
    assert body["error"] == "Request validation error"
    # The offending field should be mentioned somewhere in the details.
    assert "median_house_value" in response.text.lower()


def test_predict_rejects_is_capped_field(
    client: TestClient,
    valid_house_payload: dict[str, Any],
) -> None:
    """
    Target-derived field must be rejected at inference.

    Data leakage guard: is_capped is derived from the target and must
    never be accepted from clients.
    """

    payload = {
        **valid_house_payload,
        "is_capped": True,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    # Strengthened: the response must specifically mention the offending field.
    assert "is_capped" in response.text.lower()


def test_predict_rejects_missing_required_field(
    client: TestClient,
    valid_house_payload: dict[str, Any],
) -> None:
    """Missing required raw feature must result in HTTP 422."""

    payload = valid_house_payload.copy()
    del payload["median_income"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_rejects_invalid_ocean_proximity(
    client: TestClient,
    valid_house_payload: dict[str, Any],
) -> None:
    """Unknown ocean proximity category must be rejected."""

    payload = {
        **valid_house_payload,
        "ocean_proximity": "UNKNOWN_LOCATION",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("longitude", -200.0),
        ("longitude", -100.0),
        ("latitude", 20.0),
        ("latitude", 50.0),
        ("housing_median_age", 0.0),
        ("housing_median_age", 60.0),
        ("total_rooms", 0.0),
        ("population", 0.0),
        ("households", 0.0),
        ("median_income", -1.0),
        ("median_income", 20.0),
    ],
)
def test_predict_rejects_out_of_range_values(
    client: TestClient,
    valid_house_payload: dict[str, Any],
    field: str,
    value: float,
) -> None:
    """API must enforce the configured numeric input ranges."""

    payload = {
        **valid_house_payload,
        field: value,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


# =============================================================================
# PREDICTION ERROR HANDLING
# =============================================================================


def test_predict_handles_prediction_error(
    client: TestClient,
    valid_house_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """PredictionError should be converted into HTTP 400."""

    def failing_prediction(payload: dict[str, Any]) -> float:
        raise PredictionError("Prediction failed")

    monkeypatch.setattr(
        "api.predict.predict_single",
        failing_prediction,
    )

    response = client.post(
        "/predict",
        json=valid_house_payload,
    )

    assert response.status_code == 400

    body = response.json()
    assert body["error"] == "Prediction error"


# =============================================================================
# BATCH PREDICTION
# =============================================================================


def test_predict_batch_success(
    client: TestClient,
    valid_batch_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Valid batch request should return one prediction per input row."""

    expected_predictions = [452600.0, 358500.0]

    monkeypatch.setattr(
        "api.predict.predict_batch",
        lambda payload: expected_predictions,
    )

    response = client.post(
        "/predict/batch",
        json=valid_batch_payload,
    )

    assert response.status_code == 200

    body = response.json()
    assert body["predictions"] == expected_predictions
    assert body["total_records"] == 2


def test_predict_batch_preserves_input_order(
    client: TestClient,
    valid_batch_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Batch predictions must preserve the exact order of the input rows.

    This protects against regressions where parallel processing could
    silently reorder the output.
    """

    monkeypatch.setattr(
        "api.predict.predict_batch",
        lambda payload: [111.0, 222.0],
    )

    response = client.post("/predict/batch", json=valid_batch_payload)

    assert response.status_code == 200

    body = response.json()
    assert body["predictions"][0] == 111.0
    assert body["predictions"][1] == 222.0


def test_predict_batch_rejects_empty_batch(
    client: TestClient,
) -> None:
    """Batch endpoint must reject an empty houses list."""

    payload = {"houses": []}

    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 422


def test_predict_batch_rejects_oversized_batch(
    client: TestClient,
    valid_house_payload: dict[str, Any],
) -> None:
    """
    Batch endpoint should refuse unreasonably large payloads to protect
    against accidental DoS. The exact limit is defined by the API.
    """

    payload = {"houses": [valid_house_payload] * 10_000}

    response = client.post("/predict/batch", json=payload)

    # Accept any 4xx: the API may signal this as 400, 413, or 422.
    assert 400 <= response.status_code < 500


# =============================================================================
# API CONTRACT CONSISTENCY
# =============================================================================


def test_prediction_response_contains_only_expected_fields(
    client: TestClient,
    valid_house_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prediction response must expose no internal implementation details."""

    monkeypatch.setattr(
        "api.predict.predict_single",
        lambda payload: 250000.0,
    )

    response = client.post("/predict", json=valid_house_payload)

    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {
        "predicted_median_house_value",
        "model_version",
    }


def test_batch_response_contains_only_expected_fields(
    client: TestClient,
    valid_batch_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Batch response must follow the strict output schema."""

    monkeypatch.setattr(
        "api.predict.predict_batch",
        lambda payload: [250000.0, 300000.0],
    )

    response = client.post("/predict/batch", json=valid_batch_payload)

    assert response.status_code == 200

    body = response.json()
    assert set(body.keys()) == {"predictions", "total_records"}


def test_prediction_response_types_are_numeric(
    client: TestClient,
    valid_house_payload: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The prediction value must be numeric, not a string or None."""

    monkeypatch.setattr(
        "api.predict.predict_single",
        lambda payload: 250000.0,
    )

    response = client.post("/predict", json=valid_house_payload)

    assert response.status_code == 200

    body = response.json()
    assert isinstance(body["predicted_median_house_value"], (int, float))
    assert isinstance(body["model_version"], str)


# =============================================================================
# INFRASTRUCTURE & ARCHITECTURE (The Six Principles)
# =============================================================================


def test_model_loaded_once_via_lru_cache() -> None:
    """
    Verify Principle 3: Model is cached in memory.

    Two independent checks:
      1. Identity — repeated calls return the exact same object.
      2. Mechanism — load_model exposes lru_cache bookkeeping.
    """

    # Mechanism check: load_model must be wrapped in functools.lru_cache.
    assert hasattr(load_model, "cache_info"), (
        "load_model must be wrapped in functools.lru_cache"
    )
    assert hasattr(load_model, "cache_clear")

    try:
        model_first = load_model()
        model_second = load_model()
    except PredictionError:
        pytest.skip("Model artifact not found; skipping cache identity test.")

    # Identity check: same instance, not merely equal.
    assert model_first is model_second

    # Cache-hit bookkeeping: the second call must be a hit.
    info = load_model.cache_info()
    assert info.hits >= 1


def test_api_handlers_are_synchronous() -> None:
    """
    Verify Principle 4: Handlers are 'def', not 'async def'.

    ML inference is CPU-bound; running in a thread pool (FastAPI's default
    behavior for sync def) prevents blocking the event loop.
    """

    assert not inspect.iscoroutinefunction(predict_house)
    assert not inspect.iscoroutinefunction(predict_houses_batch)


def test_auto_generated_docs_available(
    client: TestClient,
) -> None:
    """Verify Principle 6: Swagger UI and OpenAPI spec are active."""

    # 1. Swagger UI
    docs_response = client.get("/docs")
    assert docs_response.status_code == 200
    assert "swagger-ui" in docs_response.text.lower()

    # 2. OpenAPI Schema
    openapi_response = client.get("/openapi.json")
    assert openapi_response.status_code == 200

    schema = openapi_response.json()
    assert "paths" in schema
    assert "/predict" in schema["paths"]
    assert schema["info"]["title"] == "California Housing Prediction API"


def test_auto_generated_redoc_available(
    client: TestClient,
) -> None:
    """Verify Principle 6 (continued): ReDoc UI is also active."""

    redoc_response = client.get("/redoc")

    assert redoc_response.status_code == 200
    assert "redoc" in redoc_response.text.lower()


def test_openapi_schema_declares_core_endpoints(
    client: TestClient,
) -> None:
    """
    The generated OpenAPI schema must declare every public endpoint so
    that client SDK generators work correctly.
    """

    schema = client.get("/openapi.json").json()
    paths = schema["paths"]

    for endpoint in ("/", "/health", "/predict", "/predict/batch"):
        assert endpoint in paths, f"Missing endpoint in OpenAPI schema: {endpoint}"