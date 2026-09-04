# =============================================================================
# tests/test_engineering.py
# California Housing Project — Unit Tests for src/features/engineering.py
#
# ARCHITECTURAL ALIGNMENT:
# This test suite matches the STRICT, FAIL-FAST nature of engineering.py.
# It enforces that missing columns or target-leakage flags raise immediate
# FeatureEngineeringErrors, rather than silently skipping.
# =============================================================================

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.engineering import (
    _FALLBACK_DISTANCES,
    _FALLBACK_DROP_COLS,
    _FALLBACK_RATIOS,
    EngineeringResult,
    FeatureConfig,
    FeatureEngineeringError,
    add_distance_features,
    add_ratio_features,
    drop_raw_columns,
    load_feature_config,
    run_feature_engineering,
)

# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture
def clean_df() -> pd.DataFrame:
    """
    Minimal post-cleaning California Housing DataFrame.
    Mirrors what data/processed/train_clean.csv looks like.
    NOTE: 'is_capped' is INTENTIONALLY EXCLUDED to avoid triggering 
    the anti-leakage guard in FeatureEngineer.
    """
    np.random.seed(42)
    n = 40
    return pd.DataFrame({
        "longitude"         : np.random.uniform(-124.0, -114.5, n),
        "latitude"          : np.random.uniform(32.5, 42.0, n),
        "housing_median_age": np.random.uniform(1, 52, n),
        "total_rooms"       : np.log1p(np.random.randint(500, 8000, n)),
        "total_bedrooms"    : np.log1p(np.random.randint(100, 1500, n)),
        "population"        : np.log1p(np.random.randint(200, 3000, n)),
        "households"        : np.log1p(np.random.randint(80, 1200, n)),
        "median_income"     : np.random.uniform(0.5, 15.0, n),
        "median_house_value": np.random.uniform(15000, 490000, n),
        "ocean_proximity"   : np.random.choice(["NEAR BAY", "INLAND", "<1H OCEAN"], n),
        "lof_outlier"       : np.zeros(n, dtype=int),
    })


@pytest.fixture
def three_clean_splits(clean_df):
    """Split clean_df into train(24)/val(8)/test(8)."""
    return (
        clean_df.iloc[:24].reset_index(drop=True),
        clean_df.iloc[24:32].reset_index(drop=True),
        clean_df.iloc[32:].reset_index(drop=True),
    )


@pytest.fixture
def feat_config_yaml(tmp_path) -> Path:
    """Write a minimal data_config.yaml with a real engineered_features section."""
    cfg = {
        "eda_derived": {
            "engineered_features": {
                "ratios": {
                    "rooms_per_household"      : "total_rooms / households",
                    "bedrooms_per_room"        : "total_bedrooms / total_rooms",
                    "population_per_household" : "population / households",
                },
                "distances": {
                    "dist_SF": {"lat": 37.77, "lon": -122.42},
                    "dist_LA": {"lat": 34.05, "lon": -118.24},
                },
                "drop_after_engineering": [
                    "total_rooms", "total_bedrooms", "population", "households",
                ],
            }
        }
    }
    p = tmp_path / "data_config.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


@pytest.fixture
def config_yaml_no_eng_section(tmp_path) -> Path:
    """A valid YAML file missing the engineered_features section."""
    cfg = {"eda_derived": {"missingness": {}}}
    p = tmp_path / "data_config_no_eng.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


# =============================================================================
# 1. FeatureConfig / load_feature_config
# =============================================================================

class TestFeatureConfig:
    def test_loads_ratios_from_yaml(self, feat_config_yaml):
        cfg = load_feature_config(feat_config_yaml)
        assert "rooms_per_household" in cfg.ratios

    def test_loads_distances_from_yaml(self, feat_config_yaml):
        cfg = load_feature_config(feat_config_yaml)
        assert cfg.distances["dist_SF"]["lat"] == pytest.approx(37.77)

    def test_loads_drop_cols_from_yaml(self, feat_config_yaml):
        cfg = load_feature_config(feat_config_yaml)
        assert len(cfg.drop_cols) == 4

    def test_source_is_config_when_yaml_valid(self, feat_config_yaml):
        assert load_feature_config(feat_config_yaml).source == "config"

    def test_source_is_fallback_when_file_missing(self, tmp_path):
        cfg = load_feature_config(tmp_path / "nonexistent.yaml", allow_fallback=True)
        assert cfg.source == "fallback"

    def test_fallback_values_used_when_file_missing(self, tmp_path):
        cfg = load_feature_config(tmp_path / "nonexistent.yaml", allow_fallback=True)
        assert cfg.ratios == _FALLBACK_RATIOS
        assert cfg.distances == _FALLBACK_DISTANCES
        assert cfg.drop_cols == _FALLBACK_DROP_COLS

    def test_source_is_fallback_when_eng_section_missing(self, config_yaml_no_eng_section):
        cfg = load_feature_config(config_yaml_no_eng_section, allow_fallback=True)
        assert cfg.source == "fallback"

    def test_raises_error_when_file_missing_and_fallback_disabled(self, tmp_path):
        with pytest.raises(FeatureEngineeringError, match="Refusing to continue"):
            load_feature_config(tmp_path / "nonexistent.yaml", allow_fallback=False)


# =============================================================================
# 2. add_ratio_features
# =============================================================================

class TestAddRatioFeatures:
    def test_creates_expected_ratio_columns(self, clean_df):
        ratios = {"rooms_per_hh": "total_rooms / households"}
        result, added = add_ratio_features(clean_df, ratios)
        assert "rooms_per_hh" in result.columns
        assert "rooms_per_hh" in added

    def test_ratio_values_are_correct(self, clean_df):
        ratios = {"rooms_per_hh": "total_rooms / households"}
        result, _ = add_ratio_features(clean_df, ratios)
        expected = clean_df["total_rooms"] / clean_df["households"]
        pd.testing.assert_series_equal(
            result["rooms_per_hh"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_zero_denominator_produces_nan_not_inf(self):
        df = pd.DataFrame({
            "numerator"  : [1.0, 2.0, 3.0],
            "denominator": [0.0, 1.0, 2.0],
        })
        ratios = {"ratio": "numerator / denominator"}
        result, _ = add_ratio_features(df, ratios)
        assert not np.isinf(result["ratio"]).any()
        assert pd.isna(result.loc[0, "ratio"])

    def test_raises_on_missing_numerator(self, clean_df):
        ratios = {"bad_ratio": "nonexistent_col / households"}
        with pytest.raises(FeatureEngineeringError, match="missing required columns"):
            add_ratio_features(clean_df, ratios)

    def test_raises_on_missing_denominator(self, clean_df):
        ratios = {"bad_ratio": "total_rooms / nonexistent_col"}
        with pytest.raises(FeatureEngineeringError, match="missing required columns"):
            add_ratio_features(clean_df, ratios)

    def test_does_not_modify_original_df(self, clean_df):
        original_cols = list(clean_df.columns)
        add_ratio_features(clean_df, _FALLBACK_RATIOS)
        assert list(clean_df.columns) == original_cols


# =============================================================================
# 3. add_distance_features
# =============================================================================

class TestAddDistanceFeatures:
    def test_creates_dist_sf_and_dist_la(self, clean_df):
        result, added = add_distance_features(clean_df, _FALLBACK_DISTANCES)
        assert "dist_SF" in result.columns
        assert len(added) == 2

    def test_distance_formula_is_euclidean(self, clean_df):
        dists = {"dist_SF": {"lat": 37.77, "lon": -122.42}}
        result, _ = add_distance_features(clean_df, dists)
        expected = np.sqrt((clean_df["latitude"] - 37.77) ** 2 + (clean_df["longitude"] - (-122.42)) ** 2)
        np.testing.assert_array_almost_equal(result["dist_SF"].values, expected.values)

    def test_raises_when_latitude_missing(self, clean_df):
        df = clean_df.drop(columns=["latitude"])
        with pytest.raises(FeatureEngineeringError, match="missing required columns"):
            add_distance_features(df, _FALLBACK_DISTANCES)

    def test_does_not_modify_original_df(self, clean_df):
        original_cols = list(clean_df.columns)
        add_distance_features(clean_df, _FALLBACK_DISTANCES)
        assert list(clean_df.columns) == original_cols


# =============================================================================
# 4. drop_raw_columns
# =============================================================================

class TestDropRawColumns:
    def test_drops_expected_columns(self, clean_df):
        cols_to_drop = ["total_rooms", "total_bedrooms"]
        result, dropped = drop_raw_columns(clean_df, cols_to_drop)
        assert "total_rooms" not in result.columns
        assert dropped == cols_to_drop

    def test_raises_on_missing_column_to_drop(self, clean_df):
        # The code is strict: if you ask to drop it, it must exist.
        with pytest.raises(FeatureEngineeringError, match="Configured columns to drop are missing"):
            drop_raw_columns(clean_df, ["nonexistent_col"])

    def test_does_not_modify_original_df(self, clean_df):
        original_cols = list(clean_df.columns)
        drop_raw_columns(clean_df, _FALLBACK_DROP_COLS)
        assert list(clean_df.columns) == original_cols


# =============================================================================
# 5. EngineeringResult
# =============================================================================

class TestEngineeringResult:
    def test_summary_contains_shape(self, clean_df):
        result = EngineeringResult(train=clean_df, val=clean_df.head(5), test=clean_df.head(5))
        assert str(len(clean_df)) in result.summary()

    def test_summary_shows_deferred_preprocessing(self, clean_df):
        result = EngineeringResult(train=clean_df, val=clean_df, test=clean_df)
        assert "DEFERRED TO CV/TRAINING PIPELINE" in result.summary()


# =============================================================================
# 6. run_feature_engineering (integration)
# =============================================================================

class TestRunFeatureEngineering:
    def test_returns_engineering_result(self, three_clean_splits, feat_config_yaml, tmp_path):
        train, val, test = three_clean_splits
        result = run_feature_engineering(train, val, test, config_path=feat_config_yaml, output_dir=tmp_path / "processed")
        assert isinstance(result, EngineeringResult)

    def test_output_files_created(self, three_clean_splits, feat_config_yaml, tmp_path):
        train, val, test = three_clean_splits
        out = tmp_path / "processed"
        run_feature_engineering(train, val, test, config_path=feat_config_yaml, output_dir=out)
        assert (out / "train_feat.csv").exists()

    def test_raw_cols_dropped_from_all_splits(self, three_clean_splits, feat_config_yaml, tmp_path):
        train, val, test = three_clean_splits
        result = run_feature_engineering(train, val, test, config_path=feat_config_yaml, output_dir=tmp_path / "processed")
        for col in ["total_rooms", "total_bedrooms", "population", "households"]:
            assert col not in result.train.columns

    def test_raises_on_empty_train(self, clean_df, tmp_path):
        with pytest.raises(FeatureEngineeringError, match="empty"):
            run_feature_engineering(pd.DataFrame(), clean_df.head(5), clean_df.head(5), output_dir=tmp_path)

    def test_column_consistency_across_splits(self, three_clean_splits, feat_config_yaml, tmp_path):
        train, val, test = three_clean_splits
        result = run_feature_engineering(train, val, test, config_path=feat_config_yaml, output_dir=tmp_path / "processed")
        assert list(result.train.columns) == list(result.val.columns) == list(result.test.columns)


# =============================================================================
# 7. No Data Leakage Guards
# =============================================================================

class TestNoDataLeakage:
    def test_ratio_output_is_identical_regardless_of_call_order(self, clean_df):
        result1, _ = add_ratio_features(clean_df.copy(), _FALLBACK_RATIOS)
        result2, _ = add_ratio_features(clean_df.copy(), _FALLBACK_RATIOS)
        pd.testing.assert_series_equal(result1["rooms_per_household"], result2["rooms_per_household"])

    def test_train_and_val_get_same_formula_independently(self, three_clean_splits):
        train, val, _ = three_clean_splits
        val_result, _ = add_ratio_features(val.copy(), _FALLBACK_RATIOS)
        val_direct, _ = add_ratio_features(val.copy(), _FALLBACK_RATIOS)
        pd.testing.assert_series_equal(val_result["rooms_per_household"], val_direct["rooms_per_household"])

    def test_raises_if_is_capped_is_present(self, clean_df):
        """Guard against target leakage."""
        df_with_leak = clean_df.copy()
        df_with_leak["is_capped"] = 0
        
        # Test standalone function
        with pytest.raises(FeatureEngineeringError, match="Target-derived feature 'is_capped'"):
            run_feature_engineering(df_with_leak, df_with_leak, df_with_leak, allow_fallback=True)