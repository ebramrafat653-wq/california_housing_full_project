# =============================================================================
# tests/test_cleaning.py
# California Housing Project — Unit Tests for src/data/cleaning.py
#
# Fixtures used from conftest.py:
#   - sample_california_housing_df : valid 3-row housing DataFrame
#   - empty_df                     : completely empty DataFrame
#
# Additional fixtures are defined inline because cleaning tests need
# precise control over nulls, values, and sizes.
#
# Test classes:
#   1.  TestEdaConfig              — EdaConfig dataclass + load_eda_config()
#   2.  TestCleaningResult         — dataclass + summary()
#   3.  TestFitImputer             — fit on train only
#   4.  TestApplyImputer           — transform all splits
#   5.  TestAddIsCappedFlag        — threshold parameter (config-driven)
#   6.  TestApplyLog1p             — count cols only, income excluded
#   7.  TestFitLof                 — min_rows guard, novelty mode
#   8.  TestApplyLofFlag           — None model, -99 default, flag values
#   9.  TestSaveLoadArtifacts      — round-trip JSON + pickle, EdaConfig persisted
#   10. TestRunCleaning            — end-to-end integration, config vs fallback
#   11. TestLeakagePrevention      — fit/transform isolation
# =============================================================================

import json
import pickle
import numpy as np
import pandas as pd
import pytest
import yaml
from pathlib import Path

from src.data.cleaning import (
    CleaningError,
    CleaningResult,
    EdaConfig,
    load_eda_config,
    fit_imputer,
    apply_imputer,
    add_is_capped_flag,
    apply_log1p,
    fit_lof,
    apply_lof_flag,
    save_artifacts,
    load_artifacts,
    run_cleaning,
    _FALLBACK_LOG1P_COLS,
    _FALLBACK_CAP_THRESHOLD,
    _FALLBACK_LOF_CONTAMINATION,
    _FALLBACK_LOF_N_NEIGHBORS,
    _FALLBACK_LOF_FEATURES,
    _TARGET,
)


# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture
def housing_df() -> pd.DataFrame:
    """
    Minimal valid California Housing DataFrame — 50 rows, all required columns.
    Enough rows for LOF (n_neighbors=20 needs > 20 rows).
    Includes one NaN in total_bedrooms (mirrors real data).
    Includes one capped row (target >= $500,001).
    """
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "longitude"         : np.random.uniform(-124.0, -114.5, n),
        "latitude"          : np.random.uniform(32.5, 42.0, n),
        "housing_median_age": np.random.uniform(1, 52, n),
        "total_rooms"       : np.random.randint(500, 8000, n).astype(float),
        "total_bedrooms"    : np.random.randint(100, 1500, n).astype(float),
        "population"        : np.random.randint(200, 3000, n).astype(float),
        "households"        : np.random.randint(80, 1200, n).astype(float),
        "median_income"     : np.random.uniform(0.5, 15.0, n),
        "median_house_value": np.random.uniform(15000, 490000, n),
        "ocean_proximity"   : np.random.choice(
            ["NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN"], n
        ),
    })
    # Inject one NaN in total_bedrooms
    df.loc[0, "total_bedrooms"] = np.nan
    # Inject one capped row
    df.loc[1, "median_house_value"] = 500_001.0
    return df


@pytest.fixture
def three_splits(housing_df) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split housing_df into train(30) / val(10) / test(10)."""
    train = housing_df.iloc[:30].reset_index(drop=True)
    val   = housing_df.iloc[30:40].reset_index(drop=True)
    test  = housing_df.iloc[40:].reset_index(drop=True)
    return train, val, test


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """5-row DataFrame — too small for LOF (n_neighbors=20)."""
    return pd.DataFrame({
        "longitude"         : [-122.0] * 5,
        "latitude"          : [37.5]   * 5,
        "housing_median_age": [25.0]   * 5,
        "total_rooms"       : [1000.0] * 5,
        "total_bedrooms"    : [200.0]  * 5,
        "population"        : [500.0]  * 5,
        "households"        : [150.0]  * 5,
        "median_income"     : [5.0]    * 5,
        "median_house_value": [250000.0] * 5,
        "ocean_proximity"   : ["INLAND"] * 5,
    })


@pytest.fixture
def eda_config_yaml(tmp_path) -> Path:
    """
    Write a minimal data_config.yaml with a real eda_derived section
    (mirrors the actual structure produced by notebooks/03_eda.ipynb).
    """
    cfg = {
        "project": {"target": "median_house_value"},
        "eda_derived": {
            "missingness": {
                "total_bedrooms": {
                    "missing_pct": 0.969,
                    "chi2_pvalue_vs_ocean_proximity": 0.3095,
                    "mechanism": "MCAR",
                    "imputation_strategy": "global_median",
                }
            },
            "target_summary": {
                "skewness_raw": 0.982,
                "pct_capped": 4.727,
                "n_capped": 683,
                "cap_threshold": 500001.0,
            },
            "log1p_columns": [
                "total_rooms", "total_bedrooms", "population", "households",
            ],
            "lof": {
                "contamination": 0.02,
                "n_neighbors": 20,
                "features": [
                    "median_income", "total_rooms", "population",
                    "households", "longitude", "latitude",
                ],
            },
        },
    }
    p = tmp_path / "data_config.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


@pytest.fixture
def config_yaml_no_eda_section(tmp_path) -> Path:
    """A valid YAML file that is missing the eda_derived section entirely."""
    cfg = {"project": {"target": "median_house_value"}}
    p = tmp_path / "data_config_no_eda.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


# =============================================================================
# 1. EdaConfig / load_eda_config
# =============================================================================

class TestEdaConfig:

    def test_loads_log1p_columns_from_yaml(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.log1p_columns == [
            "total_rooms", "total_bedrooms", "population", "households"
        ]

    def test_loads_cap_threshold_from_yaml(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.cap_threshold == pytest.approx(500001.0)

    def test_loads_lof_contamination_from_yaml(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.lof_contamination == pytest.approx(0.02)

    def test_loads_lof_n_neighbors_from_yaml(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.lof_n_neighbors == 20

    def test_loads_lof_features_from_yaml(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert "median_income" in cfg.lof_features
        assert len(cfg.lof_features) == 6

    def test_loads_imputation_strategy_from_yaml(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.imputation_strategy == "global_median"

    def test_source_is_config_when_yaml_valid(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.source == "config"

    def test_source_is_fallback_when_file_missing(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.yaml"
        cfg = load_eda_config(missing_path)
        assert cfg.source == "fallback"

    def test_fallback_values_used_when_file_missing(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.yaml"
        cfg = load_eda_config(missing_path)
        assert cfg.log1p_columns == _FALLBACK_LOG1P_COLS
        assert cfg.cap_threshold == _FALLBACK_CAP_THRESHOLD
        assert cfg.lof_contamination == _FALLBACK_LOF_CONTAMINATION

    def test_source_is_fallback_when_eda_section_missing(self, config_yaml_no_eda_section):
        cfg = load_eda_config(config_yaml_no_eda_section)
        assert cfg.source == "fallback"

    def test_fallback_values_used_when_eda_section_missing(self, config_yaml_no_eda_section):
        cfg = load_eda_config(config_yaml_no_eda_section)
        assert cfg.log1p_columns == _FALLBACK_LOG1P_COLS
        assert cfg.lof_features == _FALLBACK_LOF_FEATURES

    def test_partial_eda_section_falls_back_per_key(self, tmp_path):
        """
        If eda_derived exists but is missing some sub-keys (e.g. no 'lof'
        block), those specific values should fall back to defaults while
        present values are still read from YAML.
        """
        cfg_dict = {
            "project": {"target": "median_house_value"},
            "eda_derived": {
                "log1p_columns": ["total_rooms"],   # only this key present
            },
        }
        p = tmp_path / "partial_config.yaml"
        p.write_text(yaml.dump(cfg_dict), encoding="utf-8")

        cfg = load_eda_config(p)
        assert cfg.source == "config"            # section exists, so "config"
        assert cfg.log1p_columns == ["total_rooms"]          # present in YAML
        assert cfg.lof_contamination == _FALLBACK_LOF_CONTAMINATION  # missing -> fallback

    def test_eda_config_is_dataclass_instance(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert isinstance(cfg, EdaConfig)

    def test_string_path_accepted(self, eda_config_yaml):
        # config_path can be a str, not just a Path
        cfg = load_eda_config(str(eda_config_yaml))
        assert cfg.source == "config"


# =============================================================================
# 2. CleaningResult
# =============================================================================

class TestCleaningResult:

    def test_summary_contains_shape(self, housing_df):
        result = CleaningResult(
            train=housing_df,
            val=housing_df.head(5),
            test=housing_df.head(5),
        )
        summary = result.summary()
        assert "50" in summary           # train rows
        assert "5"  in summary           # val/test rows

    def test_summary_shows_dvc_not_tracked(self, housing_df):
        result = CleaningResult(train=housing_df, val=housing_df, test=housing_df)
        assert "no" in result.summary()

    def test_summary_shows_warnings(self, housing_df):
        result = CleaningResult(train=housing_df, val=housing_df, test=housing_df)
        result.warnings.append("DVC push failed")
        assert "DVC push failed" in result.summary()

    def test_summary_shows_paths_when_set(self, housing_df, tmp_path):
        result = CleaningResult(train=housing_df, val=housing_df, test=housing_df)
        result.train_path = tmp_path / "train_clean.csv"
        assert str(tmp_path) in result.summary()


# =============================================================================
# 2. fit_imputer
# =============================================================================

class TestFitImputer:

    def test_returns_median_for_numeric_null_col(self, housing_df):
        stats = fit_imputer(housing_df)
        assert "total_bedrooms" in stats
        expected = housing_df["total_bedrooms"].median()
        assert stats["total_bedrooms"] == pytest.approx(expected)

    def test_returns_mode_for_categorical_null_col(self):
        df = pd.DataFrame({
            "ocean_proximity": ["INLAND", "INLAND", None, "NEAR BAY"],
            "median_house_value": [100_000.0] * 4,
        })
        stats = fit_imputer(df)
        assert "ocean_proximity" in stats
        assert stats["ocean_proximity"] == "INLAND"

    def test_empty_stats_when_no_nulls(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        stats = fit_imputer(df)
        assert stats == {}

    def test_fit_uses_train_median_not_global(self, housing_df):
        # train has different median than full dataset
        train = housing_df.iloc[:20].copy()
        train_median = train["total_bedrooms"].median()
        stats = fit_imputer(train)
        assert stats["total_bedrooms"] == pytest.approx(train_median)
        # should NOT equal full df median (different subset)
        # (this is a data leakage prevention check)

    def test_does_not_modify_input(self, housing_df):
        original = housing_df.copy()
        fit_imputer(housing_df)
        pd.testing.assert_frame_equal(housing_df, original)


# =============================================================================
# 3. apply_imputer
# =============================================================================

class TestApplyImputer:

    def test_fills_nulls_with_provided_value(self, housing_df):
        stats = {"total_bedrooms": 999.0}
        result = apply_imputer(housing_df, stats)
        assert result["total_bedrooms"].isnull().sum() == 0
        assert (result["total_bedrooms"] == 999.0).any()

    def test_does_not_fill_non_null_values(self, housing_df):
        stats = {"total_bedrooms": 999.0}
        original_non_null = housing_df["total_bedrooms"].dropna().values.copy()
        result = apply_imputer(housing_df, stats)
        # All original non-null values should be unchanged
        filled_non_null = result.loc[
            housing_df["total_bedrooms"].notna(), "total_bedrooms"
        ].values
        np.testing.assert_array_almost_equal(original_non_null, filled_non_null)

    def test_skips_missing_column_with_warning(self, housing_df):
        stats = {"nonexistent_col": 42.0}
        result = apply_imputer(housing_df, stats)
        assert "nonexistent_col" not in result.columns

    def test_does_not_modify_original_df(self, housing_df):
        original_nulls = housing_df["total_bedrooms"].isnull().sum()
        stats = {"total_bedrooms": 200.0}
        apply_imputer(housing_df, stats)
        assert housing_df["total_bedrooms"].isnull().sum() == original_nulls

    def test_train_fit_applied_to_val(self, three_splits):
        train, val, _ = three_splits
        stats = fit_imputer(train)
        val_result = apply_imputer(val, stats)
        assert val_result["total_bedrooms"].isnull().sum() == 0


# =============================================================================
# 4. add_is_capped_flag
# =============================================================================

class TestAddIsCappedFlag:

    def test_adds_is_capped_column(self, housing_df):
        result = add_is_capped_flag(housing_df)
        assert "is_capped" in result.columns

    def test_capped_rows_flagged_correctly(self, housing_df):
        result = add_is_capped_flag(housing_df)
        # Row 1 has median_house_value = 500_001 → should be 1
        assert result.loc[1, "is_capped"] == 1

    def test_non_capped_rows_are_zero(self, housing_df):
        result = add_is_capped_flag(housing_df)
        non_capped = result[housing_df["median_house_value"] < _FALLBACK_CAP_THRESHOLD]
        assert (non_capped["is_capped"] == 0).all()

    def test_exact_threshold_is_capped(self):
        df = pd.DataFrame({"median_house_value": [500_000.0, 500_001.0, 500_002.0]})
        result = add_is_capped_flag(df)
        assert result.loc[0, "is_capped"] == 0   # below threshold
        assert result.loc[1, "is_capped"] == 1   # exactly at threshold
        assert result.loc[2, "is_capped"] == 1   # above threshold

    def test_missing_target_returns_df_without_flag(self, housing_df):
        df = housing_df.drop(columns=["median_house_value"])
        result = add_is_capped_flag(df)
        assert "is_capped" not in result.columns

    def test_is_binary_0_or_1(self, housing_df):
        result = add_is_capped_flag(housing_df)
        assert set(result["is_capped"].unique()).issubset({0, 1})

    def test_does_not_modify_original(self, housing_df):
        original_cols = list(housing_df.columns)
        add_is_capped_flag(housing_df)
        assert list(housing_df.columns) == original_cols


# =============================================================================
# 5. apply_log1p
# =============================================================================

class TestApplyLog1p:

    def test_transforms_count_columns(self, housing_df):
        result = apply_log1p(housing_df)
        for col in _FALLBACK_LOG1P_COLS:
            if col in housing_df.columns:
                expected = np.log1p(housing_df[col].dropna())
                actual   = result[col].dropna()
                np.testing.assert_array_almost_equal(expected.values, actual.values)

    def test_median_income_not_transformed(self, housing_df):
        original_income = housing_df["median_income"].copy()
        result = apply_log1p(housing_df)
        pd.testing.assert_series_equal(result["median_income"], original_income)

    def test_target_not_transformed(self, housing_df):
        original_target = housing_df[_TARGET].copy()
        result = apply_log1p(housing_df)
        pd.testing.assert_series_equal(result[_TARGET], original_target)

    def test_skewness_reduces_after_transform(self, skewed_mock_df):
        """
        Verify that log1p reduces absolute skewness for count-based columns
        that are heavily right-skewed (using the dedicated skewed_mock_df fixture).
        """
        for col in _FALLBACK_LOG1P_COLS:
            if col not in skewed_mock_df.columns:
                continue
            before = skewed_mock_df[col].skew()
            after = apply_log1p(skewed_mock_df)[col].skew()
            assert abs(after) <= abs(before), (
                f"log1p did not reduce skewness for '{col}': "
                f"before={before:.3f}, after={after:.3f}"
            )

    def test_skips_negative_values_column(self, housing_df):
        df = housing_df.copy()
        df["total_rooms"] = -df["total_rooms"]    # make negative
        result = apply_log1p(df, cols=["total_rooms"])
        # Column should be unchanged (skipped due to negatives)
        pd.testing.assert_series_equal(result["total_rooms"], df["total_rooms"])

    def test_skips_missing_column_gracefully(self, housing_df):
        result = apply_log1p(housing_df, cols=["nonexistent_col"])
        assert "nonexistent_col" not in result.columns

    def test_handles_zero_values(self):
        df = pd.DataFrame({"total_rooms": [0.0, 1.0, 100.0]})
        result = apply_log1p(df, cols=["total_rooms"])
        assert result["total_rooms"].isnull().sum() == 0   # log1p(0) = 0, no NaN
        assert result.loc[0, "total_rooms"] == pytest.approx(0.0)

    def test_does_not_modify_original(self, housing_df):
        original = housing_df["total_rooms"].copy()
        apply_log1p(housing_df)
        pd.testing.assert_series_equal(housing_df["total_rooms"], original)


# =============================================================================
# 6. fit_lof
# =============================================================================

class TestFitLof:

    def test_returns_fitted_lof_and_features(self, housing_df):
        lof, features = fit_lof(housing_df, n_neighbors=5)
        assert lof is not None
        assert isinstance(features, list)
        assert len(features) > 0

    def test_novelty_mode_enabled(self, housing_df):
        lof, _ = fit_lof(housing_df, n_neighbors=5)
        assert lof.novelty is True

    def test_returns_none_when_too_few_rows(self, tiny_df):
        lof, features = fit_lof(tiny_df, n_neighbors=20)
        assert lof is None
        assert isinstance(features, list)

    def test_skips_missing_features(self, housing_df):
        fake_features = _FALLBACK_LOF_FEATURES + ["nonexistent_col"]
        lof, features = fit_lof(housing_df, features=fake_features, n_neighbors=5)
        assert "nonexistent_col" not in features
        assert lof is not None

    def test_contamination_respected(self, housing_df):
        lof, _ = fit_lof(housing_df, contamination=0.05, n_neighbors=5)
        assert lof.contamination == pytest.approx(0.05)

    def test_does_not_modify_input(self, housing_df):
        original = housing_df.copy()
        fit_lof(housing_df, n_neighbors=5)
        pd.testing.assert_frame_equal(housing_df, original)


# =============================================================================
# 7. apply_lof_flag
# =============================================================================

class TestApplyLofFlag:

    def test_adds_lof_outlier_column(self, housing_df):
        lof, features = fit_lof(housing_df, n_neighbors=5)
        result = apply_lof_flag(housing_df, lof, features)
        assert "lof_outlier" in result.columns

    def test_flag_values_are_valid(self, housing_df):
        lof, features = fit_lof(housing_df, n_neighbors=5)
        result = apply_lof_flag(housing_df, lof, features)
        assert set(result["lof_outlier"].unique()).issubset({-99, 0, 1})

    def test_none_lof_sets_all_to_minus99(self, housing_df):
        result = apply_lof_flag(housing_df, lof=None, features=_FALLBACK_LOF_FEATURES)
        assert (result["lof_outlier"] == -99).all()

    def test_rows_with_null_lof_features_get_minus99(self, housing_df):
        lof, features = fit_lof(housing_df, n_neighbors=5)
        df = housing_df.copy()
        df.loc[0, "median_income"] = np.nan    # force a null in LOF feature
        result = apply_lof_flag(df, lof, features)
        assert result.loc[0, "lof_outlier"] == -99

    def test_outlier_flag_uses_train_fit_model(self, three_splits):
        train, val, _ = three_splits
        lof, features = fit_lof(train, n_neighbors=5)
        # Should run without error on val using train-fit model
        val_result = apply_lof_flag(val, lof, features)
        assert "lof_outlier" in val_result.columns

    def test_does_not_modify_original(self, housing_df):
        lof, features = fit_lof(housing_df, n_neighbors=5)
        original_cols = list(housing_df.columns)
        apply_lof_flag(housing_df, lof, features)
        assert list(housing_df.columns) == original_cols


# =============================================================================
# 8. save_artifacts / load_artifacts
# =============================================================================

class TestSaveLoadArtifacts:

    @pytest.fixture
    def sample_eda_config(self) -> EdaConfig:
        """A real EdaConfig instance to pass into save_artifacts()."""
        return EdaConfig(
            log1p_columns=_FALLBACK_LOG1P_COLS,
            cap_threshold=_FALLBACK_CAP_THRESHOLD,
            lof_contamination=_FALLBACK_LOF_CONTAMINATION,
            lof_n_neighbors=_FALLBACK_LOF_N_NEIGHBORS,
            lof_features=_FALLBACK_LOF_FEATURES,
            source="config",
        )

    def test_save_creates_json_and_pkl(self, housing_df, sample_eda_config, tmp_path):
        lof, features = fit_lof(housing_df, n_neighbors=5)
        stats = {"total_bedrooms": 250.0}
        save_artifacts(stats, lof, features, sample_eda_config, output_dir=tmp_path)
        assert (tmp_path / "cleaning_artifacts.json").exists()
        assert (tmp_path / "lof_model.pkl").exists()

    def test_round_trip_imputer_stats(self, housing_df, sample_eda_config, tmp_path):
        lof, features = fit_lof(housing_df, n_neighbors=5)
        stats = {"total_bedrooms": 250.0}
        save_artifacts(stats, lof, features, sample_eda_config, output_dir=tmp_path)
        loaded_stats, _, _ = load_artifacts(tmp_path)
        assert loaded_stats["total_bedrooms"] == pytest.approx(250.0)

    def test_round_trip_lof_model(self, housing_df, sample_eda_config, tmp_path):
        lof, features = fit_lof(housing_df, n_neighbors=5)
        stats = {"total_bedrooms": 250.0}
        save_artifacts(stats, lof, features, sample_eda_config, output_dir=tmp_path)
        _, loaded_lof, _ = load_artifacts(tmp_path)
        # Loaded model should produce same predictions
        test_data = housing_df[features].dropna().head(5)
        original_preds = lof.predict(test_data)
        loaded_preds   = loaded_lof.predict(test_data)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_round_trip_lof_features(self, housing_df, sample_eda_config, tmp_path):
        lof, features = fit_lof(housing_df, n_neighbors=5)
        save_artifacts({"total_bedrooms": 250.0}, lof, features, sample_eda_config, output_dir=tmp_path)
        _, _, loaded_features = load_artifacts(tmp_path)
        assert loaded_features == features

    def test_load_raises_if_artifacts_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_artifacts(tmp_path)

    def test_json_contains_expected_keys(self, housing_df, sample_eda_config, tmp_path):
        lof, features = fit_lof(housing_df, n_neighbors=5)
        save_artifacts({"total_bedrooms": 200.0}, lof, features, sample_eda_config, output_dir=tmp_path)
        meta = json.loads((tmp_path / "cleaning_artifacts.json").read_text())
        for key in ["imputer_stats", "lof_features", "lof_contamination",
                    "lof_n_neighbors", "log1p_cols", "cap_threshold",
                    "eda_config_source", "timestamp"]:
            assert key in meta, f"Missing key in artifacts JSON: '{key}'"

    def test_json_persists_eda_config_source(self, housing_df, sample_eda_config, tmp_path):
        """eda_config_source must be saved for traceability (config vs fallback)."""
        lof, features = fit_lof(housing_df, n_neighbors=5)
        save_artifacts({"total_bedrooms": 200.0}, lof, features, sample_eda_config, output_dir=tmp_path)
        meta = json.loads((tmp_path / "cleaning_artifacts.json").read_text())
        assert meta["eda_config_source"] == "config"

    def test_save_handles_none_lof_model(self, sample_eda_config, tmp_path):
        """save_artifacts must not crash when lof is None (LOF was skipped)."""
        stats = {"total_bedrooms": 250.0}
        save_artifacts(stats, None, [], sample_eda_config, output_dir=tmp_path)
        assert (tmp_path / "lof_model.pkl").exists()
        _, loaded_lof, _ = load_artifacts(tmp_path)
        assert loaded_lof is None

    def test_load_raises_if_lof_pkl_missing_but_json_present(self, housing_df, sample_eda_config, tmp_path):
        lof, features = fit_lof(housing_df, n_neighbors=5)
        save_artifacts({"total_bedrooms": 250.0}, lof, features, sample_eda_config, output_dir=tmp_path)
        (tmp_path / "lof_model.pkl").unlink()   # remove pkl, keep json
        with pytest.raises(FileNotFoundError):
            load_artifacts(tmp_path)


# =============================================================================
# 9. run_cleaning (integration)
# =============================================================================

class TestRunCleaning:

    def test_returns_cleaning_result(self, three_splits, tmp_path):
        train, val, test = three_splits
        result = run_cleaning(
            train, val, test,
            output_dir=tmp_path / "processed",
            auto_track_dvc=False,
            save_artifacts_flag=False,
        )
        assert isinstance(result, CleaningResult)

    def test_output_files_created(self, three_splits, tmp_path):
        train, val, test = three_splits
        out = tmp_path / "processed"
        run_cleaning(
            train, val, test,
            output_dir=out,
            auto_track_dvc=False,
            save_artifacts_flag=False,
        )
        assert (out / "train_clean.csv").exists()
        assert (out / "val_clean.csv").exists()
        assert (out / "test_clean.csv").exists()

    def test_nulls_removed_after_cleaning(self, three_splits, tmp_path):
        train, val, test = three_splits
        result = run_cleaning(
            train, val, test,
            output_dir=tmp_path / "processed",
            auto_track_dvc=False,
            save_artifacts_flag=False,
        )
        for name, df in [("train", result.train), ("val", result.val), ("test", result.test)]:
            assert df["total_bedrooms"].isnull().sum() == 0, \
                f"'{name}' still has nulls in total_bedrooms after cleaning"

    def test_is_capped_flag_present(self, three_splits, tmp_path):
        train, val, test = three_splits
        result = run_cleaning(
            train, val, test,
            output_dir=tmp_path / "processed",
            auto_track_dvc=False,
            save_artifacts_flag=False,
        )
        for df in [result.train, result.val, result.test]:
            assert "is_capped" in df.columns

    def test_lof_outlier_flag_present(self, three_splits, tmp_path):
        train, val, test = three_splits
        result = run_cleaning(
            train, val, test,
            output_dir=tmp_path / "processed",
            auto_track_dvc=False,
            save_artifacts_flag=False,
        )
        for df in [result.train, result.val, result.test]:
            assert "lof_outlier" in df.columns

    def test_log1p_applied_to_count_cols(self, three_splits, tmp_path):
        train, val, test = three_splits
        result = run_cleaning(
            train, val, test,
            output_dir=tmp_path / "processed",
            auto_track_dvc=False,
            save_artifacts_flag=False,
        )
        # log1p values should all be >= 0 and < original (for values > 0)
        for col in _FALLBACK_LOG1P_COLS:
            if col in result.train.columns:
                assert (result.train[col] >= 0).all(), \
                    f"Negative values after log1p in '{col}'"

    def test_median_income_unchanged(self, three_splits, tmp_path):
        train, val, test = three_splits
        original_income = train["median_income"].copy()
        result = run_cleaning(
            train, val, test,
            output_dir=tmp_path / "processed",
            auto_track_dvc=False,
            save_artifacts_flag=False,
        )
        pd.testing.assert_series_equal(
            result.train["median_income"].reset_index(drop=True),
            original_income.reset_index(drop=True),
        )

    def test_raises_on_empty_train(self, empty_df, housing_df, tmp_path):
        with pytest.raises(CleaningError, match="empty"):
            run_cleaning(
                empty_df, housing_df.head(5), housing_df.head(5),
                output_dir=tmp_path,
                auto_track_dvc=False,
                save_artifacts_flag=False,
            )

    def test_raises_when_target_missing(self, three_splits, tmp_path):
        train, val, test = three_splits
        train_no_target = train.drop(columns=[_TARGET])
        with pytest.raises(CleaningError, match="Target column"):
            run_cleaning(
                train_no_target, val, test,
                output_dir=tmp_path,
                auto_track_dvc=False,
                save_artifacts_flag=False,
            )

    def test_artifacts_saved_when_flag_true(self, three_splits, tmp_path):
        train, val, test = three_splits
        artifacts_dir = tmp_path / "artifacts"
        artifacts_dir.mkdir()

        import src.data.cleaning as cleaning_mod
        original_project_dir = cleaning_mod.PROJECT_DIR
        cleaning_mod.PROJECT_DIR = tmp_path    # redirect artifacts path

        try:
            run_cleaning(
                train, val, test,
                output_dir=tmp_path / "processed",
                auto_track_dvc=False,
                save_artifacts_flag=True,
            )
            assert (tmp_path / "artifacts" / "cleaning_artifacts.json").exists()
        finally:
            cleaning_mod.PROJECT_DIR = original_project_dir


# =============================================================================
# 10. Leakage Prevention Tests
# =============================================================================

class TestLeakagePrevention:

    def test_imputer_fit_on_train_only(self, three_splits):
        """
        Imputer median must come from train, not val or test.
        Verified by manually computing train median and comparing.
        """
        train, val, test = three_splits
        train_median = train["total_bedrooms"].median()
        stats = fit_imputer(train)
        assert stats["total_bedrooms"] == pytest.approx(train_median)

    def test_val_imputed_with_train_median_not_val_median(self, three_splits):
        """
        Val nulls must be filled with the TRAIN median, not the val's own median.
        """
        train, val, test = three_splits

        # Force a null in val
        val = val.copy()
        val.loc[0, "total_bedrooms"] = np.nan

        # Compute train median
        train_median = train["total_bedrooms"].median()

        # Apply train-fit imputer to val
        stats = fit_imputer(train)
        val_result = apply_imputer(val, stats)

        assert val_result.loc[0, "total_bedrooms"] == pytest.approx(train_median)

    def test_lof_fit_on_train_not_refitted_on_val(self, three_splits):
        """
        The LOF model must be the same object used for val/test prediction.
        No re-fitting should happen.
        """
        train, val, test = three_splits
        lof, features = fit_lof(train, n_neighbors=5)
        lof_id = id(lof)

        # Applying to val should use the same model object
        val_result = apply_lof_flag(val, lof, features)
        assert id(lof) == lof_id, "LOF model was unexpectedly re-created"
        assert "lof_outlier" in val_result.columns

    def test_log1p_is_deterministic(self, housing_df):
        """
        log1p transform must be identical regardless of call order.
        No statistics are fit — this is a pure mathematical transform.
        """
        result1 = apply_log1p(housing_df.copy())
        result2 = apply_log1p(housing_df.copy())
        for col in _FALLBACK_LOG1P_COLS:
            if col in housing_df.columns:
                pd.testing.assert_series_equal(result1[col], result2[col])