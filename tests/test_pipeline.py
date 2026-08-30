# =============================================================================
# tests/test_pipeline.py
# California Housing Project — Unit Tests for src/features/pipeline.py
# =============================================================================

import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.features.pipeline import (
    _TARGET,
    PipelineError,
    PipelineResult,
    build_pipeline,
    fit_transform_pipeline,
    get_feature_names,
    load_pipeline,
    resolve_columns,
    run_pipeline,
    save_pipeline,
)

# =============================================================================
# MOCK CONFIGURATION
# (Replaces the removed hardcoded constants from pipeline.py)
# =============================================================================

MOCK_FEATURES_CONFIG = {
    "std": [
        "longitude", "latitude", "housing_median_age",
        "rooms_per_household", "bedrooms_per_room",
        "population_per_household", "dist_SF", "dist_LA",
    ],
    "robust": ["median_income"],
    "cat": ["ocean_proximity"],
    "passthrough": ["lof_outlier"],
}

# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture
def mock_config():
    return MOCK_FEATURES_CONFIG

@pytest.fixture
def feat_df() -> pd.DataFrame:
    """
    Minimal post-engineering DataFrame (40 rows).
    NOTE: `is_capped` is intentionally EXCLUDED here. The new pipeline.py 
    enforces a strict target-leakage guard that raises PipelineError if 
    `is_capped` is present in X_train.
    """
    np.random.seed(42)
    n = 40
    return pd.DataFrame({
        # Standard-scaled
        "longitude"              : np.random.uniform(-124.0, -114.5, n),
        "latitude"               : np.random.uniform(32.5, 42.0, n),
        "housing_median_age"     : np.random.uniform(1, 52, n),
        "rooms_per_household"    : np.random.uniform(3, 10, n),
        "bedrooms_per_room"      : np.random.uniform(0.1, 0.5, n),
        "population_per_household": np.random.uniform(1, 5, n),
        "dist_SF"                : np.random.uniform(0, 10, n),
        "dist_LA"                : np.random.uniform(0, 10, n),
        # Robust-scaled
        "median_income"          : np.random.uniform(0.5, 15.0, n),
        # Categorical
        "ocean_proximity"        : np.random.choice(
            ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN"], n
        ),
        # Passthrough flags (is_capped removed to satisfy leakage guard)
        "lof_outlier"            : np.zeros(n, dtype=int),
        # Target
        "median_house_value"     : np.random.uniform(15000, 490000, n),
    })

@pytest.fixture
def three_feat_splits(feat_df):
    """Split feat_df into train(24)/val(8)/test(8)."""
    train = feat_df.iloc[:24].reset_index(drop=True)
    val   = feat_df.iloc[24:32].reset_index(drop=True)
    test  = feat_df.iloc[32:].reset_index(drop=True)
    return train, val, test

@pytest.fixture
def resolved_cols(feat_df, mock_config):
    """Pre-resolved column dict for the standard feat_df layout."""
    return resolve_columns(feat_df, config=mock_config)

@pytest.fixture
def fitted_pipeline_and_data(three_feat_splits, mock_config):
    """Return a fitted pipeline + transformed arrays for reuse."""
    train, val, test = three_feat_splits
    cols = resolve_columns(train, config=mock_config)
    pipeline = build_pipeline(cols)
    pipeline, X_tr, X_v, X_te, y_tr, y_v, y_te = fit_transform_pipeline(
        pipeline, train, val, test
    )
    return pipeline, X_tr, X_v, X_te, y_tr, y_v, y_te


# =============================================================================
# 1. resolve_columns
# =============================================================================

class TestResolveColumns:

    def test_std_scale_cols_present(self, feat_df, mock_config):
        cols = resolve_columns(feat_df, config=mock_config)
        for col in mock_config["std"]:
            assert col in cols["std"]

    def test_robust_scale_cols_present(self, feat_df, mock_config):
        cols = resolve_columns(feat_df, config=mock_config)
        assert "median_income" in cols["robust"]

    def test_cat_cols_present(self, feat_df, mock_config):
        cols = resolve_columns(feat_df, config=mock_config)
        assert "ocean_proximity" in cols["cat"]

    def test_passthrough_cols_present(self, feat_df, mock_config):
        cols = resolve_columns(feat_df, config=mock_config)
        assert "lof_outlier" in cols["passthrough"]

    def test_missing_col_raises_error(self, feat_df, mock_config):
        """resolve_columns must raise PipelineError when columns are missing."""
        df = feat_df.drop(columns=["median_income"])
        with pytest.raises(PipelineError, match="Missing columns for 'robust'"):
            resolve_columns(df, config=mock_config)

    def test_target_not_in_any_group(self, feat_df, mock_config):
        cols = resolve_columns(feat_df, config=mock_config)
        all_assigned = (
            cols["std"] + cols["robust"] + cols["cat"] + cols["passthrough"]
        )
        assert _TARGET not in all_assigned

    def test_returns_dict_with_four_keys(self, feat_df, mock_config):
        cols = resolve_columns(feat_df, config=mock_config)
        assert set(cols.keys()) == {"std", "robust", "cat", "passthrough"}

    def test_no_overlap_between_groups(self, feat_df, mock_config):
        cols = resolve_columns(feat_df, config=mock_config)
        all_cols = (
            cols["std"] + cols["robust"] + cols["cat"] + cols["passthrough"]
        )
        assert len(all_cols) == len(set(all_cols)), "Duplicate columns across groups"


# =============================================================================
# 2. build_pipeline
# =============================================================================

class TestBuildPipeline:

    def test_returns_sklearn_pipeline(self, resolved_cols):
        pipeline = build_pipeline(resolved_cols)
        assert isinstance(pipeline, Pipeline)

    def test_pipeline_has_preprocessor_step(self, resolved_cols):
        pipeline = build_pipeline(resolved_cols)
        assert "preprocessor" in pipeline.named_steps

    def test_standard_scaler_in_transformers(self, resolved_cols):
        pipeline = build_pipeline(resolved_cols)
        ct = pipeline.named_steps["preprocessor"]
        transformer_names = [name for name, _, _ in ct.transformers]
        assert "standard_scaler" in transformer_names

    def test_robust_scaler_in_transformers(self, resolved_cols):
        pipeline = build_pipeline(resolved_cols)
        ct = pipeline.named_steps["preprocessor"]
        transformer_names = [name for name, _, _ in ct.transformers]
        assert "robust_scaler" in transformer_names

    def test_one_hot_encoder_in_transformers(self, resolved_cols):
        pipeline = build_pipeline(resolved_cols)
        ct = pipeline.named_steps["preprocessor"]
        transformer_names = [name for name, _, _ in ct.transformers]
        assert "one_hot_encoder" in transformer_names

    def test_ohe_handle_unknown_is_ignore(self, resolved_cols):
        pipeline = build_pipeline(resolved_cols)
        ct = pipeline.named_steps["preprocessor"]
        ohe = next(t for name, t, _ in ct.transformers if name == "one_hot_encoder")
        assert ohe.handle_unknown == "ignore"

    def test_raises_when_no_columns_match(self):
        empty_cols = {"std": [], "robust": [], "cat": [], "passthrough": []}
        with pytest.raises(PipelineError, match="No preprocessing transformers were created"):
            build_pipeline(empty_cols)

    def test_pipeline_not_yet_fitted(self, resolved_cols):
        pipeline = build_pipeline(resolved_cols)
        with pytest.raises(Exception):   
            pipeline.transform(pd.DataFrame({"dummy": [1]}))


# =============================================================================
# 3. fit_transform_pipeline
# =============================================================================

class TestFitTransformPipeline:

    def test_returns_seven_items(self, three_feat_splits, mock_config):
        train, val, test = three_feat_splits
        cols = resolve_columns(train, config=mock_config)
        pipeline = build_pipeline(cols)
        result = fit_transform_pipeline(pipeline, train, val, test)
        assert len(result) == 7

    def test_output_shapes_consistent(self, three_feat_splits, mock_config):
        train, val, test = three_feat_splits
        cols = resolve_columns(train, config=mock_config)
        pipeline = build_pipeline(cols)
        _, X_tr, X_v, X_te, y_tr, y_v, y_te = fit_transform_pipeline(
            pipeline, train, val, test
        )
        assert X_tr.shape[0] == len(train)
        assert X_v.shape[0]  == len(val)
        assert X_te.shape[0] == len(test)

    def test_same_number_of_features_across_splits(self, three_feat_splits, mock_config):
        train, val, test = three_feat_splits
        cols = resolve_columns(train, config=mock_config)
        pipeline = build_pipeline(cols)
        _, X_tr, X_v, X_te, *_ = fit_transform_pipeline(
            pipeline, train, val, test
        )
        assert X_tr.shape[1] == X_v.shape[1] == X_te.shape[1]

    def test_output_is_numpy_array(self, three_feat_splits, mock_config):
        train, val, test = three_feat_splits
        cols = resolve_columns(train, config=mock_config)
        pipeline = build_pipeline(cols)
        _, X_tr, X_v, X_te, *_ = fit_transform_pipeline(
            pipeline, train, val, test
        )
        for arr in [X_tr, X_v, X_te]:
            assert isinstance(arr, np.ndarray)

    def test_target_not_in_X(self, three_feat_splits, mock_config):
        train, val, test = three_feat_splits
        cols = resolve_columns(train, config=mock_config)
        pipeline = build_pipeline(cols)
        _, X_tr, *_ = fit_transform_pipeline(pipeline, train, val, test)
        assert X_tr.shape[1] >= len(mock_config["std"]) + len(mock_config["robust"]) + len(mock_config["passthrough"])

    def test_y_series_length_matches_split(self, three_feat_splits, mock_config):
        train, val, test = three_feat_splits
        cols = resolve_columns(train, config=mock_config)
        pipeline = build_pipeline(cols)
        _, _, _, _, y_tr, y_v, y_te = fit_transform_pipeline(
            pipeline, train, val, test
        )
        assert len(y_tr) == len(train)
        assert len(y_v)  == len(val)
        assert len(y_te) == len(test)

    def test_raises_when_target_missing(self, three_feat_splits, mock_config):
        train, val, test = three_feat_splits
        train_no_target = train.drop(columns=[_TARGET])
        cols = resolve_columns(train_no_target, config=mock_config)
        pipeline = build_pipeline(cols)
        with pytest.raises(PipelineError, match="Target 'median_house_value' is missing from train"):
            fit_transform_pipeline(pipeline, train_no_target, val, test)

    def test_no_nan_in_output_arrays(self, three_feat_splits, mock_config):
        train, val, test = three_feat_splits
        cols = resolve_columns(train, config=mock_config)
        pipeline = build_pipeline(cols)
        _, X_tr, X_v, X_te, *_ = fit_transform_pipeline(
            pipeline, train, val, test
        )
        for name, arr in [("train", X_tr), ("val", X_v), ("test", X_te)]:
            assert not np.isnan(arr).any(), f"NaN in {name} output array"
            
    def test_raises_on_target_leakage(self, three_feat_splits, mock_config):
        """Ensure pipeline fails fast if target-derived metadata leaks into X."""
        train, val, test = three_feat_splits
        train_leaked = train.copy()
        train_leaked["is_capped"] = 0  # Inject forbidden column
        
        cols = resolve_columns(train_leaked.drop(columns=["is_capped"]), config=mock_config)
        pipeline = build_pipeline(cols)
        
        with pytest.raises(PipelineError, match="Target leakage detected"):
            fit_transform_pipeline(pipeline, train_leaked, val, test)


# =============================================================================
# 4. get_feature_names
# =============================================================================

class TestGetFeatureNames:

    def test_returns_list_of_strings(self, fitted_pipeline_and_data):
        pipeline, *_ = fitted_pipeline_and_data
        names = get_feature_names(pipeline)
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)

    def test_length_matches_output_columns(self, fitted_pipeline_and_data):
        pipeline, X_tr, *_ = fitted_pipeline_and_data
        names = get_feature_names(pipeline)
        assert len(names) == X_tr.shape[1]

    def test_ohe_categories_in_names(self, fitted_pipeline_and_data):
        pipeline, *_ = fitted_pipeline_and_data
        names = get_feature_names(pipeline)
        ohe_names = [n for n in names if "ocean_proximity" in n]
        assert len(ohe_names) >= 2, "Expected at least 2 OHE categories in feature names"

    def test_standard_scaled_cols_in_names(self, fitted_pipeline_and_data):
        pipeline, *_ = fitted_pipeline_and_data
        names = get_feature_names(pipeline)
        assert "longitude" in names
        assert "latitude"  in names

    def test_robust_scaled_col_in_names(self, fitted_pipeline_and_data):
        pipeline, *_ = fitted_pipeline_and_data
        names = get_feature_names(pipeline)
        assert "median_income" in names

    def test_passthrough_cols_in_names(self, fitted_pipeline_and_data):
        pipeline, *_ = fitted_pipeline_and_data
        names = get_feature_names(pipeline)
        assert "lof_outlier"  in names

    def test_target_not_in_names(self, fitted_pipeline_and_data):
        pipeline, *_ = fitted_pipeline_and_data
        names = get_feature_names(pipeline)
        assert _TARGET not in names


# =============================================================================
# 5. save_pipeline / load_pipeline
# =============================================================================

class TestSaveLoadPipeline:

    def test_save_creates_pkl_file(self, fitted_pipeline_and_data, tmp_path):
        pipeline, *_ = fitted_pipeline_and_data
        path = save_pipeline(pipeline, output_dir=tmp_path)
        assert path.exists()
        assert path.suffix == ".pkl"

    def test_round_trip_produces_identical_predictions(
        self, fitted_pipeline_and_data, three_feat_splits, tmp_path
    ):
        pipeline, X_tr, *_ = fitted_pipeline_and_data
        train, val, test = three_feat_splits

        save_pipeline(pipeline, output_dir=tmp_path)
        loaded = load_pipeline(artifacts_dir=tmp_path)

        X_loaded = loaded.transform(train.drop(columns=[_TARGET]))
        np.testing.assert_array_almost_equal(X_tr, X_loaded)

    def test_load_raises_when_file_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_pipeline(artifacts_dir=tmp_path)

    def test_loaded_pipeline_is_sklearn_pipeline(
        self, fitted_pipeline_and_data, tmp_path
    ):
        pipeline, *_ = fitted_pipeline_and_data
        save_pipeline(pipeline, output_dir=tmp_path)
        loaded = load_pipeline(artifacts_dir=tmp_path)
        assert isinstance(loaded, Pipeline)

    def test_pkl_is_valid_pickle(self, fitted_pipeline_and_data, tmp_path):
        pipeline, *_ = fitted_pipeline_and_data
        path = save_pipeline(pipeline, output_dir=tmp_path)
        with open(path, "rb") as f:
            obj = pickle.load(f)
        assert isinstance(obj, Pipeline)


# =============================================================================
# 6. PipelineResult
# =============================================================================

class TestPipelineResult:

    def test_summary_contains_shapes(self, fitted_pipeline_and_data):
        pipeline, X_tr, X_v, X_te, y_tr, y_v, y_te = fitted_pipeline_and_data
        result = PipelineResult(
            X_train=X_tr, X_val=X_v, X_test=X_te,
            y_train=y_tr, y_val=y_v, y_test=y_te,
            n_features=X_tr.shape[1],
        )
        summary = result.summary()
        assert str(X_tr.shape) in summary

    def test_summary_shows_warnings(self, fitted_pipeline_and_data):
        pipeline, X_tr, X_v, X_te, y_tr, y_v, y_te = fitted_pipeline_and_data
        result = PipelineResult(
            X_train=X_tr, X_val=X_v, X_test=X_te,
            y_train=y_tr, y_val=y_v, y_test=y_te,
            warnings=["DVC push failed"],
        )
        assert "DVC push failed" in result.summary()

    def test_summary_shows_n_features(self, fitted_pipeline_and_data):
        pipeline, X_tr, X_v, X_te, y_tr, y_v, y_te = fitted_pipeline_and_data
        result = PipelineResult(
            X_train=X_tr, X_val=X_v, X_test=X_te,
            y_train=y_tr, y_val=y_v, y_test=y_te,
            n_features=X_tr.shape[1],
        )
        assert str(X_tr.shape[1]) in result.summary()


# =============================================================================
# 7. run_pipeline (integration)
# =============================================================================

class TestRunPipeline:

    @pytest.mark.integration
    def test_returns_pipeline_result(self, three_feat_splits, tmp_path):
        train, val, test = three_feat_splits
        result = run_pipeline(
            train, val, test,
            artifacts_dir=tmp_path,
        )
        assert isinstance(result, PipelineResult)

    @pytest.mark.integration
    def test_output_arrays_are_numpy(self, three_feat_splits, tmp_path):
        train, val, test = three_feat_splits
        result = run_pipeline(
            train, val, test,
            artifacts_dir=tmp_path,
        )
        for arr in [result.X_train, result.X_val, result.X_test]:
            assert isinstance(arr, np.ndarray)

    @pytest.mark.integration
    def test_pipeline_artifact_saved(self, three_feat_splits, tmp_path):
        train, val, test = three_feat_splits
        result = run_pipeline(
            train, val, test,
            artifacts_dir=tmp_path,
            save=True,
        )
        assert result.pipeline_path is not None
        assert result.pipeline_path.exists()

    @pytest.mark.integration
    def test_no_save_when_save_false(self, three_feat_splits, tmp_path):
        train, val, test = three_feat_splits
        result = run_pipeline(
            train, val, test,
            artifacts_dir=tmp_path,
            save=False,
        )
        assert result.pipeline_path is None

    @pytest.mark.integration
    def test_feature_names_populated(self, three_feat_splits, tmp_path):
        train, val, test = three_feat_splits
        result = run_pipeline(
            train, val, test,
            artifacts_dir=tmp_path,
        )
        assert len(result.feature_names_out) > 0
        assert len(result.feature_names_out) == result.n_features

    @pytest.mark.integration
    def test_n_features_matches_array_shape(self, three_feat_splits, tmp_path):
        train, val, test = three_feat_splits
        result = run_pipeline(
            train, val, test,
            artifacts_dir=tmp_path,
        )
        assert result.n_features == result.X_train.shape[1]

    @pytest.mark.integration
    def test_raises_on_empty_train(self, empty_df, feat_df, tmp_path):
        with pytest.raises(PipelineError, match="empty"):
            run_pipeline(
                empty_df, feat_df.head(5), feat_df.head(5),
                artifacts_dir=tmp_path,
            )

    @pytest.mark.integration
    def test_raises_when_target_missing(self, three_feat_splits, tmp_path):
        train, val, test = three_feat_splits
        train_no_target = train.drop(columns=[_TARGET])
        with pytest.raises(PipelineError, match="Target 'median_house_value' is missing from train"):
            run_pipeline(
                train_no_target, val, test,
                artifacts_dir=tmp_path,
            )

    @pytest.mark.integration
    def test_no_nan_in_result_arrays(self, three_feat_splits, tmp_path):
        train, val, test = three_feat_splits
        result = run_pipeline(
            train, val, test,
            artifacts_dir=tmp_path,
        )
        for name, arr in [
            ("X_train", result.X_train),
            ("X_val",   result.X_val),
            ("X_test",  result.X_test),
        ]:
            assert not np.isnan(arr).any(), f"NaN in {name}"

    @pytest.mark.integration
    def test_result_summary_is_string(self, three_feat_splits, tmp_path):
        train, val, test = three_feat_splits
        result = run_pipeline(
            train, val, test,
            artifacts_dir=tmp_path,
        )
        assert isinstance(result.summary(), str)
        assert len(result.summary()) > 0

    @pytest.mark.integration
    def test_unknown_ohe_category_handled(self, three_feat_splits, tmp_path):
        train, val, test = three_feat_splits
        test = test.copy()
        test.loc[0, "ocean_proximity"] = "ISLAND"  # unseen in train

        result = run_pipeline(
            train, val, test,
            artifacts_dir=tmp_path,
        )
        assert not np.isnan(result.X_test).any()


# =============================================================================
# 8. No Data Leakage
# =============================================================================

class TestNoDataLeakage:

    def test_scaler_mean_comes_from_train_only(self, three_feat_splits, mock_config):
        train, val, test = three_feat_splits
        cols = resolve_columns(train, config=mock_config)
        pipeline = build_pipeline(cols)
        fit_transform_pipeline(pipeline, train, val, test)

        ct = pipeline.named_steps["preprocessor"]
        scaler = ct.named_transformers_["standard_scaler"]

        std_cols = cols["std"]
        lon_idx = std_cols.index("longitude")

        train_mean = train["longitude"].mean()
        assert scaler.mean_[lon_idx] == pytest.approx(train_mean, rel=1e-5)

    def test_val_transform_uses_train_statistics(self, three_feat_splits, mock_config):
        train, val, test = three_feat_splits

        cols = resolve_columns(train, config=mock_config)
        pipeline_train = build_pipeline(cols)
        _, _, X_val_from_train, *_ = fit_transform_pipeline(
            pipeline_train, train, val, test
        )

        cols_val = resolve_columns(val, config=mock_config)
        pipeline_val = build_pipeline(cols_val)
        pipeline_val.fit(val.drop(columns=[_TARGET]))
        pipeline_val.transform(val.drop(columns=[_TARGET]))

        ct_train = pipeline_train.named_steps["preprocessor"]
        ct_val   = pipeline_val.named_steps["preprocessor"]
        scaler_train = ct_train.named_transformers_["standard_scaler"]
        scaler_val   = ct_val.named_transformers_["standard_scaler"]

        assert not np.allclose(scaler_train.mean_, scaler_val.mean_), (
            "Train and val scalers have identical means — "
            "train and val may be identical, or leakage occurred"
        )

    @pytest.mark.integration
    def test_shuffle_train_does_not_change_val_output(self, three_feat_splits, tmp_path):
        train, val, test = three_feat_splits

        result1 = run_pipeline(
            train, val, test,
            artifacts_dir=tmp_path / "run1",
        )

        train_shuffled = train.sample(frac=1, random_state=99).reset_index(drop=True)
        result2 = run_pipeline(
            train_shuffled, val, test,
            artifacts_dir=tmp_path / "run2",
        )

        np.testing.assert_array_almost_equal(result1.X_val, result2.X_val)
        np.testing.assert_array_almost_equal(result1.X_test, result2.X_test)

    def test_pipeline_not_refitted_on_val(self, three_feat_splits, mock_config):
        train, val, test = three_feat_splits
        cols = resolve_columns(train, config=mock_config)
        pipeline = build_pipeline(cols)

        pipeline, *_ = fit_transform_pipeline(pipeline, train, val, test)

        X_val_1 = pipeline.transform(val.drop(columns=[_TARGET]))
        X_val_2 = pipeline.transform(val.drop(columns=[_TARGET]))
        np.testing.assert_array_equal(X_val_1, X_val_2)

    @pytest.mark.integration
    def test_same_seed_same_result(self, feat_df, tmp_path):
        train = feat_df.iloc[:24].reset_index(drop=True)
        val   = feat_df.iloc[24:32].reset_index(drop=True)
        test  = feat_df.iloc[32:].reset_index(drop=True)

        result1 = run_pipeline(
            train, val, test,
            artifacts_dir=tmp_path / "r1",
        )
        result2 = run_pipeline(
            train, val, test,
            artifacts_dir=tmp_path / "r2",
        )

        np.testing.assert_array_almost_equal(result1.X_train, result2.X_train)
        np.testing.assert_array_almost_equal(result1.X_val,   result2.X_val)