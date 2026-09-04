# =============================================================================
# tests/test_pipeline.py
# California Housing Project — Unit Tests for src/pipeline.py
#
# ARCHITECTURAL ALIGNMENT:
# This test suite matches the NEW, streamlined pipeline.py architecture.
# It tests the pipeline using RAW input data (like train_clean.csv),
# verifying that the internal transformers (Imputer, LOF, FeatureEngineer)
# work correctly and safely within the CV structure.
# =============================================================================

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from src.features.engineering import FeatureEngineer, FeatureEngineeringError
from src.features.pipeline import (
    _TARGET,
    DataFrameImputer,
    LOFTransformer,
    PipelineError,
    assert_pipeline_is_cv_safe,
    build_model_pipeline,
    fit_final_pipeline,
    get_feature_names,
    load_pipeline,
    load_pipeline_config,
    predict,
    resolve_columns,
    save_pipeline,
    validate_raw_input_schema,
)

# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture
def raw_df() -> pd.DataFrame:
    """
    Minimal RAW California Housing DataFrame (like train_clean.csv).
    Contains raw columns needed for FeatureEngineer and LOF.
    Includes intentional NaNs in 'total_bedrooms' to test the imputer.
    """
    np.random.seed(42)
    n = 40
    df = pd.DataFrame({
        "longitude": np.random.uniform(-124.0, -114.5, n),
        "latitude": np.random.uniform(32.5, 42.0, n),
        "housing_median_age": np.random.uniform(1, 52, n),
        "total_rooms": np.random.randint(500, 8000, n).astype(float),
        "total_bedrooms": np.random.randint(100, 1500, n).astype(float),
        "population": np.random.randint(200, 3000, n).astype(float),
        "households": np.random.randint(80, 1200, n).astype(float),
        "median_income": np.random.uniform(0.5, 15.0, n),
        "median_house_value": np.random.uniform(15000, 490000, n),
        "ocean_proximity": np.random.choice(["NEAR BAY", "INLAND", "<1H OCEAN"], n),
    })
    # Inject NaN to test DataFrameImputer
    df.loc[0, "total_bedrooms"] = np.nan
    return df


@pytest.fixture
def mock_config_path(tmp_path) -> Path:
    """Create a temporary valid data_config.yaml for testing."""
    cfg = {
        "features_config": {
            "numerical_standard_scaler": ["longitude", "latitude", "rooms_per_household"],
            "numerical_robust_scaler": ["median_income"],
            "categorical_one_hot": ["ocean_proximity"],
            "passthrough": ["lof_outlier"],
        },
        "eda_derived": {
            "missingness": {
                "total_bedrooms": {"impute": True, "imputation_strategy": "median"}
            },
            "lof": {
                "features": ["median_income", "total_rooms", "population", "households", "longitude", "latitude"],
                "n_neighbors": 5,
                "contamination": 0.1,
            },
            "engineered_features": {
                "ratios": {"rooms_per_household": "total_rooms / households"},
                "distances": {},
                "drop_after_engineering": ["total_rooms", "total_bedrooms", "population", "households"],
            }
        }
    }
    path = tmp_path / "data_config.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return path


@pytest.fixture
def mock_config_dict(mock_config_path) -> dict:
    """Return the parsed (processed) config dictionary."""
    return load_pipeline_config(mock_config_path)


@pytest.fixture
def raw_mock_config(mock_config_path) -> dict:
    """
    Return the raw, unprocessed config dictionary (as loaded directly from YAML).
    This is needed for validate_raw_input_schema because it expects the full
    original structure containing 'eda_derived'.
    """
    import yaml
    with open(mock_config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def resolved_cols(mock_config_dict) -> dict:
    """Return resolved columns from the mock config."""
    return resolve_columns(mock_config_dict)


# =============================================================================
# 1. Configuration & Resolution
# =============================================================================

class TestConfiguration:
    def test_load_pipeline_config_returns_valid_dict(self, mock_config_path):
        cfg = load_pipeline_config(mock_config_path)
        assert "std" in cfg
        assert "lof" in cfg
        assert "imputation" in cfg

    def test_resolve_columns_extracts_lists(self, mock_config_dict):
        cols = resolve_columns(mock_config_dict)
        assert isinstance(cols["std"], list)
        assert isinstance(cols["lof"], list)
        assert "rooms_per_household" in cols["std"]


# =============================================================================
# 2. Raw Input Validation
# =============================================================================

class TestValidateRawInputSchema:
    def test_passes_on_valid_raw_data(self, raw_df, resolved_cols, mock_config_dict):
        validate_raw_input_schema(
            raw_df.drop(columns=[_TARGET]),
            resolved_cols,
            mock_config_dict
        )

    def test_raises_if_target_is_present(self, raw_df, resolved_cols, mock_config_dict):
        with pytest.raises(PipelineError, match="must not be present inside X"):
            validate_raw_input_schema(raw_df, resolved_cols, mock_config_dict)

    def test_raises_if_forbidden_feature_present(self, raw_df, resolved_cols, mock_config_dict):
        df_leaked = raw_df.drop(columns=[_TARGET]).copy()
        df_leaked["is_capped"] = 0
        with pytest.raises(PipelineError, match="Forbidden target-derived columns"):
            validate_raw_input_schema(df_leaked, resolved_cols, mock_config_dict)

    def test_raises_if_raw_column_missing(self, raw_df, resolved_cols, raw_mock_config):
        """
        Use raw_mock_config (unprocessed) because validate_raw_input_schema
        expects the full config including 'eda_derived'.
        """
        df_missing = raw_df.drop(columns=[_TARGET, "total_bedrooms"]).copy()
        with pytest.raises(PipelineError, match="FeatureEngineer requires missing raw columns"):
            validate_raw_input_schema(df_missing, resolved_cols, raw_mock_config)


# =============================================================================
# 3. Pipeline Construction & Safety
# =============================================================================

class TestBuildModelPipeline:
    def test_builds_valid_sklearn_pipeline(self, resolved_cols, mock_config_path):
        model = LinearRegression()
        pipeline = build_model_pipeline(
            cols=resolved_cols,
            model=model,
            config_path=mock_config_path
        )
        assert isinstance(pipeline, Pipeline)
        assert "preprocessing" in pipeline.named_steps
        assert "model" in pipeline.named_steps

    def test_passes_structural_safety_check(self, resolved_cols, mock_config_path):
        model = LinearRegression()
        pipeline = build_model_pipeline(
            cols=resolved_cols,
            model=model,
            config_path=mock_config_path
        )
        assert_pipeline_is_cv_safe(pipeline)

    def test_uses_canonical_feature_engineer(self, resolved_cols, mock_config_path):
        model = LinearRegression()
        pipeline = build_model_pipeline(
            cols=resolved_cols,
            model=model,
            config_path=mock_config_path
        )
        preprocessing = pipeline.named_steps["preprocessing"]
        fe_step = preprocessing.named_steps["feature_engineering"]
        assert isinstance(fe_step, FeatureEngineer)


# =============================================================================
# 4. End-to-End Fit & Predict
# =============================================================================

class TestFitAndPredict:
    def test_fit_final_pipeline_succeeds(self, raw_df, resolved_cols, mock_config_path):
        model = LinearRegression()
        pipeline = build_model_pipeline(
            cols=resolved_cols,
            model=model,
            config_path=mock_config_path
        )
        fitted_pipeline = fit_final_pipeline(pipeline, raw_df)
        assert fitted_pipeline is not None

    def test_predict_generates_valid_output(self, raw_df, resolved_cols, mock_config_path):
        model = LinearRegression()
        pipeline = build_model_pipeline(
            cols=resolved_cols,
            model=model,
            config_path=mock_config_path
        )
        fitted_pipeline = fit_final_pipeline(pipeline, raw_df)

        X_new = raw_df.drop(columns=[_TARGET]).head(5)
        predictions = predict(fitted_pipeline, X_new)

        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == 5
        assert np.isfinite(predictions).all()

    def test_predict_rejects_target_in_input(self, raw_df, resolved_cols, mock_config_path):
        model = LinearRegression()
        pipeline = build_model_pipeline(
            cols=resolved_cols,
            model=model,
            config_path=mock_config_path
        )
        fitted_pipeline = fit_final_pipeline(pipeline, raw_df)

        with pytest.raises(PipelineError, match="must not be included in prediction input"):
            predict(fitted_pipeline, raw_df)  # raw_df still has target


# =============================================================================
# 5. Serialization
# =============================================================================

class TestSaveLoadPipeline:
    def test_save_creates_pkl_file(self, raw_df, resolved_cols, mock_config_path, tmp_path):
        model = LinearRegression()
        pipeline = build_model_pipeline(
            cols=resolved_cols,
            model=model,
            config_path=mock_config_path
        )
        fitted_pipeline = fit_final_pipeline(pipeline, raw_df)

        path = save_pipeline(fitted_pipeline, output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".pkl"

    def test_round_trip_produces_identical_predictions(self, raw_df, resolved_cols, mock_config_path, tmp_path):
        model = LinearRegression()
        pipeline = build_model_pipeline(
            cols=resolved_cols,
            model=model,
            config_path=mock_config_path
        )
        fitted_pipeline = fit_final_pipeline(pipeline, raw_df)

        save_pipeline(fitted_pipeline, output_dir=tmp_path)
        loaded_pipeline = load_pipeline(artifacts_dir=tmp_path)

        X_new = raw_df.drop(columns=[_TARGET]).head(5)
        pred_original = predict(fitted_pipeline, X_new)
        pred_loaded = predict(loaded_pipeline, X_new)

        np.testing.assert_array_almost_equal(pred_original, pred_loaded)


# =============================================================================
# 6. No Data Leakage Guards
# =============================================================================

class TestNoDataLeakage:
    def test_imputer_uses_train_stats_only(self, raw_df, resolved_cols, mock_config_path):
        train = raw_df.iloc[:30].copy()
        train.loc[0, "total_bedrooms"] = 100.0
        train_median = train["total_bedrooms"].median()

        model = LinearRegression()
        pipeline = build_model_pipeline(
            cols=resolved_cols,
            model=model,
            config_path=mock_config_path
        )
        pipeline.fit(train.drop(columns=[_TARGET]), train[_TARGET])

        preprocessing = pipeline.named_steps["preprocessing"]
        imputer = preprocessing.named_steps["initial_imputer"]
        impute_col_idx = imputer.impute_cols_.index("total_bedrooms")
        learned_stat = imputer.imputer_.statistics_[impute_col_idx]

        assert learned_stat == pytest.approx(train_median, rel=1e-5)

    def test_pipeline_fails_if_is_capped_leaks_into_fit(self, raw_df, resolved_cols, mock_config_path):
        train_leaked = raw_df.copy()
        train_leaked["is_capped"] = 0

        model = LinearRegression()
        pipeline = build_model_pipeline(
            cols=resolved_cols,
            model=model,
            config_path=mock_config_path
        )

        with pytest.raises(FeatureEngineeringError, match="Target-derived feature 'is_capped' detected in X"):
            pipeline.fit(train_leaked.drop(columns=[_TARGET]), train_leaked[_TARGET])