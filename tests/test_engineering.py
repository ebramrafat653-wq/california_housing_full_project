# =============================================================================
# tests/test_engineering.py
# California Housing Project — Unit Tests for src/features/engineering.py
#
# Fixtures from conftest.py used:
#   - empty_df : completely empty DataFrame
#
# All other fixtures defined inline for precise control.
#
# Test classes:
#   1.  TestFeatureConfig          — FeatureConfig dataclass + load_feature_config()
#   2.  TestAddRatioFeatures       — ratio creation, zero-denom guard, inf clip
#   3.  TestAddDistanceFeatures    — distance creation, missing col guards
#   4.  TestDropRawColumns         — drop logic, missing col handling
#   5.  TestEngineeringResult      — dataclass + summary()
#   6.  TestRunFeatureEngineering  — end-to-end integration
#   7.  TestNoDataLeakage          — pure math, no fit on train
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
    Mirrors what data/processed/train_clean.csv looks like:
    - log1p applied to count cols (values are small, < 10)
    - is_capped and lof_outlier flags present
    - total_bedrooms has no nulls (imputed)
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
        "is_capped"         : np.zeros(n, dtype=int),
        "lof_outlier"       : np.zeros(n, dtype=int),
    })


@pytest.fixture
def three_clean_splits(clean_df):
    """Split clean_df into train(24)/val(8)/test(8)."""
    train = clean_df.iloc[:24].reset_index(drop=True)
    val   = clean_df.iloc[24:32].reset_index(drop=True)
    test  = clean_df.iloc[32:].reset_index(drop=True)
    return train, val, test


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
        assert "bedrooms_per_room" in cfg.ratios
        assert "population_per_household" in cfg.ratios

    def test_loads_distances_from_yaml(self, feat_config_yaml):
        cfg = load_feature_config(feat_config_yaml)
        assert "dist_SF" in cfg.distances
        assert "dist_LA" in cfg.distances
        assert cfg.distances["dist_SF"]["lat"] == pytest.approx(37.77)
        assert cfg.distances["dist_LA"]["lon"] == pytest.approx(-118.24)

    def test_loads_drop_cols_from_yaml(self, feat_config_yaml):
        cfg = load_feature_config(feat_config_yaml)
        assert "total_rooms" in cfg.drop_cols
        assert "households" in cfg.drop_cols
        assert len(cfg.drop_cols) == 4

    def test_source_is_config_when_yaml_valid(self, feat_config_yaml):
        cfg = load_feature_config(feat_config_yaml)
        assert cfg.source == "config"

    def test_source_is_fallback_when_file_missing(self, tmp_path):
        cfg = load_feature_config(tmp_path / "nonexistent.yaml")
        assert cfg.source == "fallback"

    def test_fallback_values_used_when_file_missing(self, tmp_path):
        cfg = load_feature_config(tmp_path / "nonexistent.yaml")
        assert cfg.ratios == _FALLBACK_RATIOS
        assert cfg.distances == _FALLBACK_DISTANCES
        assert cfg.drop_cols == _FALLBACK_DROP_COLS

    def test_source_is_fallback_when_eng_section_missing(self, config_yaml_no_eng_section):
        cfg = load_feature_config(config_yaml_no_eng_section)
        assert cfg.source == "fallback"

    def test_string_path_accepted(self, feat_config_yaml):
        cfg = load_feature_config(str(feat_config_yaml))
        assert cfg.source == "config"

    def test_returns_feature_config_instance(self, feat_config_yaml):
        cfg = load_feature_config(feat_config_yaml)
        assert isinstance(cfg, FeatureConfig)


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

    def test_all_three_ratios_created(self, clean_df):
        result, added = add_ratio_features(clean_df, _FALLBACK_RATIOS)
        for name in _FALLBACK_RATIOS:
            assert name in result.columns
            assert name in added

    def test_skips_missing_numerator_with_warning(self, clean_df):
        ratios = {"bad_ratio": "nonexistent_col / households"}
        result, added = add_ratio_features(clean_df, ratios)
        assert "bad_ratio" not in result.columns
        assert "bad_ratio" not in added

    def test_skips_missing_denominator_with_warning(self, clean_df):
        ratios = {"bad_ratio": "total_rooms / nonexistent_col"}
        result, added = add_ratio_features(clean_df, ratios)
        assert "bad_ratio" not in result.columns
        assert "bad_ratio" not in added

    def test_skips_malformed_formula(self, clean_df):
        ratios = {"bad_ratio": "total_rooms * households"}   # not "a / b"
        result, added = add_ratio_features(clean_df, ratios)
        assert "bad_ratio" not in result.columns

    def test_zero_denominator_produces_nan_not_inf(self):
        df = pd.DataFrame({
            "numerator"  : [1.0, 2.0, 3.0],
            "denominator": [0.0, 1.0, 2.0],   # first row is zero
        })
        ratios = {"ratio": "numerator / denominator"}
        result, _ = add_ratio_features(df, ratios)
        assert not np.isinf(result["ratio"]).any(), "inf values should be clipped to NaN"
        assert result.loc[0, "ratio"] != result.loc[0, "ratio"]  # NaN check (NaN != NaN)

    def test_does_not_modify_original_df(self, clean_df):
        original_cols = list(clean_df.columns)
        add_ratio_features(clean_df, _FALLBACK_RATIOS)
        assert list(clean_df.columns) == original_cols

    def test_no_nulls_introduced_on_clean_data(self, clean_df):
        result, _ = add_ratio_features(clean_df, _FALLBACK_RATIOS)
        for name in _FALLBACK_RATIOS:
            assert result[name].isnull().sum() == 0, \
                f"Unexpected nulls in '{name}'"

    def test_empty_ratios_dict_returns_df_unchanged(self, clean_df):
        result, added = add_ratio_features(clean_df, {})
        assert list(result.columns) == list(clean_df.columns)
        assert added == []


# =============================================================================
# 3. add_distance_features
# =============================================================================

class TestAddDistanceFeatures:

    def test_creates_dist_sf_and_dist_la(self, clean_df):
        result, added = add_distance_features(clean_df, _FALLBACK_DISTANCES)
        assert "dist_SF" in result.columns
        assert "dist_LA" in result.columns
        assert len(added) == 2

    def test_distance_values_are_non_negative(self, clean_df):
        result, _ = add_distance_features(clean_df, _FALLBACK_DISTANCES)
        assert (result["dist_SF"] >= 0).all()
        assert (result["dist_LA"] >= 0).all()

    def test_distance_formula_is_euclidean(self, clean_df):
        dists = {"dist_SF": {"lat": 37.77, "lon": -122.42}}
        result, _ = add_distance_features(clean_df, dists)
        expected = np.sqrt(
            (clean_df["latitude"]  - 37.77) ** 2 +
            (clean_df["longitude"] - (-122.42)) ** 2
        )
        np.testing.assert_array_almost_equal(
            result["dist_SF"].values, expected.values
        )

    def test_skips_when_latitude_missing(self, clean_df):
        df = clean_df.drop(columns=["latitude"])
        result, added = add_distance_features(df, _FALLBACK_DISTANCES)
        assert "dist_SF" not in result.columns
        assert added == []

    def test_skips_when_longitude_missing(self, clean_df):
        df = clean_df.drop(columns=["longitude"])
        result, added = add_distance_features(df, _FALLBACK_DISTANCES)
        assert "dist_SF" not in result.columns
        assert added == []

    def test_skips_hub_with_missing_lat_key(self, clean_df):
        dists = {"bad_hub": {"lon": -122.42}}   # no "lat"
        result, added = add_distance_features(clean_df, dists)
        assert "bad_hub" not in result.columns
        assert added == []

    def test_skips_hub_with_missing_lon_key(self, clean_df):
        dists = {"bad_hub": {"lat": 37.77}}   # no "lon"
        result, added = add_distance_features(clean_df, dists)
        assert "bad_hub" not in result.columns
        assert added == []

    def test_does_not_modify_original_df(self, clean_df):
        original_cols = list(clean_df.columns)
        add_distance_features(clean_df, _FALLBACK_DISTANCES)
        assert list(clean_df.columns) == original_cols

    def test_empty_distances_dict_returns_df_unchanged(self, clean_df):
        result, added = add_distance_features(clean_df, {})
        assert list(result.columns) == list(clean_df.columns)
        assert added == []

    def test_no_nulls_in_distance_features(self, clean_df):
        result, added = add_distance_features(clean_df, _FALLBACK_DISTANCES)
        for name in added:
            assert result[name].isnull().sum() == 0


# =============================================================================
# 4. drop_raw_columns
# =============================================================================

class TestDropRawColumns:

    def test_drops_expected_columns(self, clean_df):
        cols_to_drop = ["total_rooms", "total_bedrooms"]
        result, dropped = drop_raw_columns(clean_df, cols_to_drop)
        assert "total_rooms" not in result.columns
        assert "total_bedrooms" not in result.columns
        assert dropped == cols_to_drop

    def test_all_four_raw_cols_dropped(self, clean_df):
        result, dropped = drop_raw_columns(clean_df, _FALLBACK_DROP_COLS)
        for col in _FALLBACK_DROP_COLS:
            assert col not in result.columns
        assert set(dropped) == set(_FALLBACK_DROP_COLS)

    def test_skips_missing_column_with_warning(self, clean_df):
        result, dropped = drop_raw_columns(clean_df, ["nonexistent_col"])
        assert "nonexistent_col" not in dropped
        # Other columns unchanged
        assert len(result.columns) == len(clean_df.columns)

    def test_partial_drop_when_some_cols_missing(self, clean_df):
        result, dropped = drop_raw_columns(
            clean_df, ["total_rooms", "nonexistent_col"]
        )
        assert "total_rooms" not in result.columns
        assert dropped == ["total_rooms"]   # only the existing one

    def test_does_not_modify_original_df(self, clean_df):
        original_cols = list(clean_df.columns)
        drop_raw_columns(clean_df, _FALLBACK_DROP_COLS)
        assert list(clean_df.columns) == original_cols

    def test_empty_drop_list_returns_df_unchanged(self, clean_df):
        result, dropped = drop_raw_columns(clean_df, [])
        assert list(result.columns) == list(clean_df.columns)
        assert dropped == []


# =============================================================================
# 5. EngineeringResult
# =============================================================================

class TestEngineeringResult:

    def test_summary_contains_shape(self, clean_df):
        result = EngineeringResult(
            train=clean_df,
            val=clean_df.head(5),
            test=clean_df.head(5),
        )
        summary = result.summary()
        assert str(len(clean_df)) in summary

    def test_summary_shows_features_added(self, clean_df):
        result = EngineeringResult(
            train=clean_df, val=clean_df, test=clean_df,
            features_added=["rooms_per_household", "dist_SF"],
        )
        assert "rooms_per_household" in result.summary()
        assert "dist_SF" in result.summary()

    def test_summary_shows_features_dropped(self, clean_df):
        result = EngineeringResult(
            train=clean_df, val=clean_df, test=clean_df,
            features_dropped=["total_rooms", "population"],
        )
        assert "total_rooms" in result.summary()

    def test_summary_shows_fallback_warning(self, clean_df):
        result = EngineeringResult(
            train=clean_df, val=clean_df, test=clean_df,
            feature_config_source="fallback",
        )
        assert "WARNING" in result.summary() or "fallback" in result.summary()

    def test_summary_shows_warnings(self, clean_df):
        result = EngineeringResult(
            train=clean_df, val=clean_df, test=clean_df,
            warnings=["DVC push failed"],
        )
        assert "DVC push failed" in result.summary()


# =============================================================================
# 6. run_feature_engineering (integration)
# =============================================================================

class TestRunFeatureEngineering:

    def test_returns_engineering_result(self, three_clean_splits, feat_config_yaml, tmp_path):
        train, val, test = three_clean_splits
        result = run_feature_engineering(
            train, val, test,
            config_path=feat_config_yaml,
            output_dir=tmp_path / "processed",
        )
        assert isinstance(result, EngineeringResult)

    def test_output_files_created(self, three_clean_splits, feat_config_yaml, tmp_path):
        train, val, test = three_clean_splits
        out = tmp_path / "processed"
        run_feature_engineering(
            train, val, test,
            config_path=feat_config_yaml,
            output_dir=out,
        )
        assert (out / "train_feat.csv").exists()
        assert (out / "val_feat.csv").exists()
        assert (out / "test_feat.csv").exists()

    def test_ratio_features_present_in_all_splits(self, three_clean_splits, feat_config_yaml, tmp_path):
        train, val, test = three_clean_splits
        result = run_feature_engineering(
            train, val, test,
            config_path=feat_config_yaml,
            output_dir=tmp_path / "processed",
        )
        for ratio_name in ["rooms_per_household", "bedrooms_per_room", "population_per_household"]:
            for name, df in [("train", result.train), ("val", result.val), ("test", result.test)]:
                assert ratio_name in df.columns, f"'{ratio_name}' missing from {name}"

    def test_distance_features_present_in_all_splits(self, three_clean_splits, feat_config_yaml, tmp_path):
        train, val, test = three_clean_splits
        result = run_feature_engineering(
            train, val, test,
            config_path=feat_config_yaml,
            output_dir=tmp_path / "processed",
        )
        for dist_name in ["dist_SF", "dist_LA"]:
            for name, df in [("train", result.train), ("val", result.val), ("test", result.test)]:
                assert dist_name in df.columns, f"'{dist_name}' missing from {name}"

    def test_raw_cols_dropped_from_all_splits(self, three_clean_splits, feat_config_yaml, tmp_path):
        train, val, test = three_clean_splits
        result = run_feature_engineering(
            train, val, test,
            config_path=feat_config_yaml,
            output_dir=tmp_path / "processed",
        )
        for col in ["total_rooms", "total_bedrooms", "population", "households"]:
            for name, df in [("train", result.train), ("val", result.val), ("test", result.test)]:
                assert col not in df.columns, f"'{col}' should be dropped from {name}"

    def test_raises_on_empty_train(self, empty_df, clean_df, tmp_path):
        with pytest.raises(FeatureEngineeringError, match="empty"):
            run_feature_engineering(
                empty_df, clean_df.head(5), clean_df.head(5),
                output_dir=tmp_path,
            )

    def test_result_summary_is_string(self, three_clean_splits, feat_config_yaml, tmp_path):
        train, val, test = three_clean_splits
        result = run_feature_engineering(
            train, val, test,
            config_path=feat_config_yaml,
            output_dir=tmp_path / "processed",
        )
        assert isinstance(result.summary(), str)
        assert len(result.summary()) > 0

    def test_config_source_is_config_when_yaml_valid(self, three_clean_splits, feat_config_yaml, tmp_path):
        train, val, test = three_clean_splits
        result = run_feature_engineering(
            train, val, test,
            config_path=feat_config_yaml,
            output_dir=tmp_path / "processed",
        )
        assert result.feature_config_source == "config"

    def test_config_source_is_fallback_when_yaml_missing(self, three_clean_splits, tmp_path):
        train, val, test = three_clean_splits
        result = run_feature_engineering(
            train, val, test,
            config_path=tmp_path / "nonexistent.yaml",
            output_dir=tmp_path / "processed",
        )
        assert result.feature_config_source == "fallback"

    def test_column_consistency_across_splits(self, three_clean_splits, feat_config_yaml, tmp_path):
        train, val, test = three_clean_splits
        result = run_feature_engineering(
            train, val, test,
            config_path=feat_config_yaml,
            output_dir=tmp_path / "processed",
        )
        train_cols = list(result.train.columns)
        assert list(result.val.columns)  == train_cols
        assert list(result.test.columns) == train_cols

    def test_no_nulls_in_new_features(self, three_clean_splits, feat_config_yaml, tmp_path):
        train, val, test = three_clean_splits
        result = run_feature_engineering(
            train, val, test,
            config_path=feat_config_yaml,
            output_dir=tmp_path / "processed",
        )
        new_cols = ["rooms_per_household", "bedrooms_per_room",
                    "population_per_household", "dist_SF", "dist_LA"]
        for col in new_cols:
            nulls = result.train[col].isnull().sum()
            assert nulls == 0, f"'{col}' has {nulls} nulls in train"


# =============================================================================
# 7. No Data Leakage
# =============================================================================

class TestNoDataLeakage:

    def test_ratio_output_is_identical_regardless_of_call_order(self, clean_df):
        """
        Ratio features are pure math — same input always produces same output.
        No state is stored between calls.
        """
        result1, _ = add_ratio_features(clean_df.copy(), _FALLBACK_RATIOS)
        result2, _ = add_ratio_features(clean_df.copy(), _FALLBACK_RATIOS)
        for name in _FALLBACK_RATIOS:
            pd.testing.assert_series_equal(result1[name], result2[name])

    def test_distance_output_is_identical_regardless_of_call_order(self, clean_df):
        result1, _ = add_distance_features(clean_df.copy(), _FALLBACK_DISTANCES)
        result2, _ = add_distance_features(clean_df.copy(), _FALLBACK_DISTANCES)
        for name in _FALLBACK_DISTANCES:
            pd.testing.assert_series_equal(result1[name], result2[name])

    def test_train_and_val_get_same_formula_independently(self, three_clean_splits):
        """
        Val features must be computed from val data only — not influenced by train.
        """
        train, val, _ = three_clean_splits

        # Apply to train and val separately
        train_result, _ = add_ratio_features(train, _FALLBACK_RATIOS)
        val_result, _   = add_ratio_features(val,   _FALLBACK_RATIOS)

        # Apply to val directly
        val_direct, _   = add_ratio_features(val.copy(), _FALLBACK_RATIOS)

        # val result should be identical whether or not train was processed first
        for name in _FALLBACK_RATIOS:
            pd.testing.assert_series_equal(
                val_result[name].reset_index(drop=True),
                val_direct[name].reset_index(drop=True),
            )

    def test_no_statistics_computed_from_train(self, three_clean_splits, feat_config_yaml, tmp_path):
        """
        run_feature_engineering must not compute any global statistics.
        The engineering result should be the same if we shuffle train rows.
        """
        train, val, test = three_clean_splits

        result1 = run_feature_engineering(
            train, val, test,
            config_path=feat_config_yaml,
            output_dir=tmp_path / "run1",
        )

        # Shuffle train rows — should produce identical val and test output
        train_shuffled = train.sample(frac=1, random_state=99).reset_index(drop=True)
        result2 = run_feature_engineering(
            train_shuffled, val, test,
            config_path=feat_config_yaml,
            output_dir=tmp_path / "run2",
        )

        # Val and test must be identical regardless of train row order
        for col in ["rooms_per_household", "dist_SF"]:
            pd.testing.assert_series_equal(
                result1.val[col].reset_index(drop=True),
                result2.val[col].reset_index(drop=True),
                check_names=False,
            )
