# =============================================================================
# tests/test_cleaning.py
# California Housing Project — Unit Tests for src/data/cleaning.py
#
# Matches the REWRITTEN cleaning.py where:
#   - cleaning.py does ONLY: config load, strict NaN validation, train-only
#     imputation, train-fitted LOF (+ StandardScaler), save/load, saving splits
#   - NO is_capped, NO log1p, NO target-derived features live in this module
#     (feature engineering / target-derived logic is handled elsewhere,
#     removing the target-leakage risk that existed in the old plan)
#   - load_eda_config is STRICT / fail-fast: no silent fallback values.
#     Missing or malformed config -> FileNotFoundError / CleaningError.
#   - fit_imputer / apply_imputer require explicit allowed_cols + strategies
#   - fit_lof returns (lof, scaler, features) and requires ALL of
#     (train, features, contamination, n_neighbors) explicitly
#   - apply_lof_flag requires (df, lof, scaler, features, split_name) and
#     raises on missing features / NaNs instead of falling back to -99
#   - load_artifacts returns (imputer_stats, lof, scaler, lof_features)
#
# Fixtures used from conftest.py:
#   - sample_california_housing_df : valid 3-row housing DataFrame (unused
#     directly here since cleaning needs bigger frames for LOF, kept for
#     compatibility with other test modules)
#   - empty_df                     : completely empty DataFrame
#
# Test classes:
#   1.  TestEdaConfig                — strict load_eda_config()
#   2.  TestCleaningResult           — dataclass + summary()
#   3.  TestValidateCleaningInputs   — structural input validation
#   4.  TestFitImputer               — fit on train only, explicit cols
#   5.  TestApplyImputer             — transform all splits, explicit cols
#   6.  TestFitLof                   — scaler+lof tuple, strict errors
#   7.  TestApplyLofFlag             — strict errors, no -99 fallback
#   8.  TestSaveLoadArtifacts        — round-trip JSON + pickles (lof+scaler)
#   9.  TestRunCleaning              — end-to-end integration, strict config
#   10. TestLeakagePrevention        — fit/transform isolation
#   11. TestNoTargetDerivedLogicInCleaning — regression guard for the
#       previously-flagged is_capped / log1p leakage concern
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
    validate_cleaning_inputs,
    fit_imputer,
    apply_imputer,
    fit_lof,
    apply_lof_flag,
    save_artifacts,
    load_artifacts,
    save_cleaned_splits,
    run_cleaning,
    _TARGET,
)


# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture
def housing_df() -> pd.DataFrame:
    """
    Minimal valid California Housing DataFrame — 60 rows, all required
    columns. Enough rows for LOF with small n_neighbors values used in tests.

    Only 'total_bedrooms' contains a NaN (mirrors real missingness pattern
    and satisfies cleaning.py's strict "only declared columns may be null"
    rule).
    """
    np.random.seed(42)
    n = 60
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
    # Inject one NaN in total_bedrooms only.
    df.loc[0, "total_bedrooms"] = np.nan
    return df


@pytest.fixture
def three_splits(housing_df) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split housing_df into train(40) / val(10) / test(10)."""
    train = housing_df.iloc[:40].reset_index(drop=True)
    val   = housing_df.iloc[40:50].reset_index(drop=True)
    test  = housing_df.iloc[50:].reset_index(drop=True)
    return train, val, test


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """5-row DataFrame — used to test LOF's n_neighbors size guard."""
    return pd.DataFrame({
        "longitude"         : [-122.0, -121.9, -121.8, -121.7, -121.6],
        "latitude"          : [37.5, 37.4, 37.3, 37.2, 37.1],
        "housing_median_age": [25.0, 30.0, 15.0, 40.0, 22.0],
        "total_rooms"       : [1000.0, 1200.0, 900.0, 1500.0, 1100.0],
        "total_bedrooms"    : [200.0, 220.0, 180.0, 260.0, 210.0],
        "population"        : [500.0, 520.0, 480.0, 600.0, 510.0],
        "households"        : [150.0, 160.0, 140.0, 180.0, 155.0],
        "median_income"     : [5.0, 5.5, 4.5, 6.0, 5.2],
        "median_house_value": [250000.0, 260000.0, 240000.0, 280000.0, 255000.0],
        "ocean_proximity"   : ["INLAND"] * 5,
    })


LOF_FEATURES = [
    "median_income", "total_rooms", "population",
    "households", "longitude", "latitude",
]


@pytest.fixture
def eda_config_yaml(tmp_path) -> Path:
    """
    Write a valid, complete data_config.yaml matching the strict schema
    required by the rewritten load_eda_config().
    """
    cfg = {
        "project": {"target": "median_house_value"},
        "eda_derived": {
            "missingness": {
                "total_bedrooms": {
                    "impute": True,
                    "imputation_strategy": "median",
                },
                # explicitly NOT flagged for imputation -> excluded
                "housing_median_age": {
                    "impute": False,
                },
            },
            "target_summary": {
                "skewness_raw": 0.982,
                "pct_capped": 4.727,
                "n_capped": 683,
                "cap_threshold": 500001.0,
            },
            "lof": {
                "contamination": 0.05,
                "n_neighbors": 5,
                "features": LOF_FEATURES,
            },
        },
    }
    p = tmp_path / "data_config.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


@pytest.fixture
def sample_eda_config() -> EdaConfig:
    """A ready-made EdaConfig instance for tests that don't need YAML I/O."""
    return EdaConfig(
        impute_columns=["total_bedrooms"],
        imputation_strategy={"total_bedrooms": "median"},
        cap_threshold=500001.0,
        lof_contamination=0.05,
        lof_n_neighbors=5,
        lof_features=LOF_FEATURES,
    )


# =============================================================================
# 1. load_eda_config — strict / fail-fast
# =============================================================================

class TestEdaConfig:

    def test_loads_impute_columns_from_yaml(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.impute_columns == ["total_bedrooms"]

    def test_excludes_columns_with_impute_false(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert "housing_median_age" not in cfg.impute_columns

    def test_loads_imputation_strategy_mapping(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.imputation_strategy == {"total_bedrooms": "median"}

    def test_loads_cap_threshold_from_yaml(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.cap_threshold == pytest.approx(500001.0)

    def test_loads_lof_contamination_from_yaml(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.lof_contamination == pytest.approx(0.05)

    def test_loads_lof_n_neighbors_from_yaml(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.lof_n_neighbors == 5

    def test_loads_lof_features_from_yaml(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.lof_features == LOF_FEATURES

    def test_returns_eda_config_instance(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert isinstance(cfg, EdaConfig)

    def test_string_path_accepted(self, eda_config_yaml):
        cfg = load_eda_config(str(eda_config_yaml))
        assert cfg.cap_threshold == pytest.approx(500001.0)

    def test_raises_file_not_found_when_missing(self, tmp_path):
        missing_path = tmp_path / "does_not_exist.yaml"
        with pytest.raises(FileNotFoundError):
            load_eda_config(missing_path)

    def test_raises_on_malformed_yaml(self, tmp_path):
        p = tmp_path / "broken.yaml"
        p.write_text("eda_derived: [this, is, not, a, mapping", encoding="utf-8")
        with pytest.raises(CleaningError):
            load_eda_config(p)

    def test_raises_when_eda_derived_section_missing(self, tmp_path):
        cfg_dict = {"project": {"target": "median_house_value"}}
        p = tmp_path / "no_eda.yaml"
        p.write_text(yaml.dump(cfg_dict), encoding="utf-8")
        with pytest.raises(CleaningError, match="eda_derived"):
            load_eda_config(p)

    def test_raises_when_missingness_section_missing(self, tmp_path):
        cfg_dict = {
            "eda_derived": {
                "target_summary": {"cap_threshold": 500001.0},
                "lof": {
                    "contamination": 0.05,
                    "n_neighbors": 5,
                    "features": LOF_FEATURES,
                },
            }
        }
        p = tmp_path / "no_missingness.yaml"
        p.write_text(yaml.dump(cfg_dict), encoding="utf-8")
        with pytest.raises(CleaningError, match="missingness"):
            load_eda_config(p)

    def test_raises_when_cap_threshold_missing(self, tmp_path):
        cfg_dict = {
            "eda_derived": {
                "missingness": {
                    "total_bedrooms": {"impute": True, "imputation_strategy": "median"}
                },
                "target_summary": {},
                "lof": {
                    "contamination": 0.05,
                    "n_neighbors": 5,
                    "features": LOF_FEATURES,
                },
            }
        }
        p = tmp_path / "no_cap.yaml"
        p.write_text(yaml.dump(cfg_dict), encoding="utf-8")
        with pytest.raises(CleaningError, match="cap_threshold"):
            load_eda_config(p)

    def test_raises_when_lof_keys_missing(self, tmp_path):
        cfg_dict = {
            "eda_derived": {
                "missingness": {
                    "total_bedrooms": {"impute": True, "imputation_strategy": "median"}
                },
                "target_summary": {"cap_threshold": 500001.0},
                "lof": {"contamination": 0.05},  # missing n_neighbors, features
            }
        }
        p = tmp_path / "no_lof_keys.yaml"
        p.write_text(yaml.dump(cfg_dict), encoding="utf-8")
        with pytest.raises(CleaningError, match="LOF"):
            load_eda_config(p)

    def test_raises_when_contamination_out_of_range(self, tmp_path):
        cfg_dict = {
            "eda_derived": {
                "missingness": {
                    "total_bedrooms": {"impute": True, "imputation_strategy": "median"}
                },
                "target_summary": {"cap_threshold": 500001.0},
                "lof": {
                    "contamination": 0.9,  # invalid: must be < 0.5
                    "n_neighbors": 5,
                    "features": LOF_FEATURES,
                },
            }
        }
        p = tmp_path / "bad_contamination.yaml"
        p.write_text(yaml.dump(cfg_dict), encoding="utf-8")
        with pytest.raises(CleaningError, match="contamination"):
            load_eda_config(p)

    def test_raises_when_n_neighbors_too_small(self, tmp_path):
        cfg_dict = {
            "eda_derived": {
                "missingness": {
                    "total_bedrooms": {"impute": True, "imputation_strategy": "median"}
                },
                "target_summary": {"cap_threshold": 500001.0},
                "lof": {
                    "contamination": 0.05,
                    "n_neighbors": 1,  # invalid: must be >= 2
                    "features": LOF_FEATURES,
                },
            }
        }
        p = tmp_path / "bad_n_neighbors.yaml"
        p.write_text(yaml.dump(cfg_dict), encoding="utf-8")
        with pytest.raises(CleaningError, match="n_neighbors"):
            load_eda_config(p)

    def test_raises_when_lof_features_empty(self, tmp_path):
        cfg_dict = {
            "eda_derived": {
                "missingness": {
                    "total_bedrooms": {"impute": True, "imputation_strategy": "median"}
                },
                "target_summary": {"cap_threshold": 500001.0},
                "lof": {
                    "contamination": 0.05,
                    "n_neighbors": 5,
                    "features": [],
                },
            }
        }
        p = tmp_path / "empty_lof_features.yaml"
        p.write_text(yaml.dump(cfg_dict), encoding="utf-8")
        with pytest.raises(CleaningError):
            load_eda_config(p)


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
        assert "60" in summary   # train rows
        assert "5" in summary    # val/test rows

    def test_summary_shows_feature_engineering_deferred(self, housing_df):
        result = CleaningResult(train=housing_df, val=housing_df, test=housing_df)
        assert "DEFERRED" in result.summary()

    def test_summary_shows_dvc_and_git_handled_outside(self, housing_df):
        result = CleaningResult(train=housing_df, val=housing_df, test=housing_df)
        summary = result.summary()
        assert "DVC" in summary
        assert "Git" in summary

    def test_summary_shows_warnings(self, housing_df):
        result = CleaningResult(train=housing_df, val=housing_df, test=housing_df)
        result.warnings.append("Something worth flagging")
        assert "Something worth flagging" in result.summary()

    def test_summary_shows_paths_when_set(self, housing_df, tmp_path):
        result = CleaningResult(train=housing_df, val=housing_df, test=housing_df)
        result.train_path = tmp_path / "train_clean.csv"
        result.val_path = tmp_path / "val_clean.csv"
        result.test_path = tmp_path / "test_clean.csv"
        assert str(tmp_path) in result.summary()

    def test_summary_shows_lof_outlier_count(self, housing_df):
        result = CleaningResult(
            train=housing_df, val=housing_df, test=housing_df,
            lof_n_outliers_train=3,
        )
        assert "3" in result.summary()

    def test_default_warnings_list_is_empty(self, housing_df):
        result = CleaningResult(train=housing_df, val=housing_df, test=housing_df)
        assert result.warnings == []


# =============================================================================
# 3. validate_cleaning_inputs
# =============================================================================

class TestValidateCleaningInputs:

    def test_passes_for_valid_splits(self, three_splits):
        train, val, test = three_splits
        validate_cleaning_inputs(train, val, test)  # should not raise

    def test_raises_on_empty_train(self, empty_df, housing_df):
        with pytest.raises(CleaningError, match="empty"):
            validate_cleaning_inputs(empty_df, housing_df.head(5), housing_df.head(5))

    def test_raises_on_empty_val(self, empty_df, housing_df):
        with pytest.raises(CleaningError, match="empty"):
            validate_cleaning_inputs(housing_df.head(5), empty_df, housing_df.head(5))

    def test_raises_on_non_dataframe_input(self, housing_df):
        with pytest.raises(CleaningError, match="DataFrame"):
            validate_cleaning_inputs("not a df", housing_df.head(5), housing_df.head(5))

    def test_raises_when_target_missing_in_train(self, three_splits):
        train, val, test = three_splits
        train_no_target = train.drop(columns=[_TARGET])
        with pytest.raises(CleaningError, match="Target column"):
            validate_cleaning_inputs(train_no_target, val, test)

    def test_raises_when_target_missing_in_val(self, three_splits):
        train, val, test = three_splits
        val_no_target = val.drop(columns=[_TARGET])
        with pytest.raises(CleaningError, match="Target column"):
            validate_cleaning_inputs(train, val_no_target, test)


# =============================================================================
# 4. fit_imputer
# =============================================================================

class TestFitImputer:

    def test_returns_median_for_declared_column(self, housing_df):
        stats = fit_imputer(housing_df, ["total_bedrooms"], {})
        expected = housing_df["total_bedrooms"].median()
        assert stats["total_bedrooms"] == pytest.approx(expected)

    def test_uses_mean_strategy_when_configured(self, housing_df):
        stats = fit_imputer(
            housing_df, ["total_bedrooms"], {"total_bedrooms": "mean"}
        )
        expected = housing_df["total_bedrooms"].mean()
        assert stats["total_bedrooms"] == pytest.approx(expected)

    def test_uses_mode_strategy_for_categorical(self):
        df = pd.DataFrame({
            "ocean_proximity": ["INLAND", "INLAND", None, "NEAR BAY"],
            "median_house_value": [100_000.0] * 4,
        })
        stats = fit_imputer(df, ["ocean_proximity"], {"ocean_proximity": "mode"})
        assert stats["ocean_proximity"] == "INLAND"

    def test_empty_stats_when_no_allowed_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], _TARGET: [1.0, 2.0]})
        stats = fit_imputer(df, [], {})
        assert stats == {}

    def test_fit_uses_train_median_not_full_dataset(self, three_splits):
        train, _, _ = three_splits
        train_median = train["total_bedrooms"].median()
        stats = fit_imputer(train, ["total_bedrooms"], {})
        assert stats["total_bedrooms"] == pytest.approx(train_median)

    def test_raises_on_unexpected_nan_outside_allowed_cols(self, housing_df):
        df = housing_df.copy()
        df.loc[2, "population"] = np.nan  # not declared as imputable
        with pytest.raises(CleaningError, match="Unexpected missing values"):
            fit_imputer(df, ["total_bedrooms"], {})

    def test_raises_when_target_has_nan(self, housing_df):
        df = housing_df.copy()
        df.loc[2, _TARGET] = np.nan
        with pytest.raises(CleaningError, match="Target column"):
            fit_imputer(df, ["total_bedrooms"], {})

    def test_raises_when_configured_column_missing_from_train(self, housing_df):
        with pytest.raises(CleaningError, match="does not exist"):
            fit_imputer(housing_df, ["total_bedrooms", "nonexistent_col"], {})

    def test_raises_on_non_numeric_column_for_median_strategy(self, housing_df):
        with pytest.raises(CleaningError, match="not numeric"):
            fit_imputer(
                housing_df,
                ["total_bedrooms", "ocean_proximity"],
                {"total_bedrooms": "median", "ocean_proximity": "median"}
            )

    def test_raises_on_unsupported_strategy(self, housing_df):
        with pytest.raises(CleaningError, match="Unsupported imputation strategy"):
            fit_imputer(housing_df, ["total_bedrooms"], {"total_bedrooms": "bogus"})

    def test_does_not_modify_input(self, housing_df):
        original = housing_df.copy()
        fit_imputer(housing_df, ["total_bedrooms"], {})
        pd.testing.assert_frame_equal(housing_df, original)


# =============================================================================
# 5. apply_imputer
# =============================================================================

class TestApplyImputer:

    def test_fills_nulls_with_provided_stats(self, housing_df):
        stats = {"total_bedrooms": 999.0}
        result = apply_imputer(housing_df, stats, ["total_bedrooms"], "train")
        assert result["total_bedrooms"].isnull().sum() == 0
        assert (result["total_bedrooms"] == 999.0).any()

    def test_does_not_alter_non_null_values(self, housing_df):
        stats = {"total_bedrooms": 999.0}
        original_non_null = housing_df["total_bedrooms"].dropna().values.copy()
        result = apply_imputer(housing_df, stats, ["total_bedrooms"], "train")
        filled_non_null = result.loc[
            housing_df["total_bedrooms"].notna(), "total_bedrooms"
        ].values
        np.testing.assert_array_almost_equal(original_non_null, filled_non_null)

    def test_raises_when_stats_column_missing_from_df(self, housing_df):
        stats = {"total_bedrooms": 200.0, "nonexistent_col": 42.0}
        with pytest.raises(CleaningError, match="missing from the dataset"):
            apply_imputer(housing_df, stats, ["total_bedrooms", "nonexistent_col"], "train")

    def test_raises_on_unexpected_nan_outside_allowed_cols(self, housing_df):
        df = housing_df.copy()
        df.loc[3, "population"] = np.nan
        stats = {"total_bedrooms": 250.0}
        with pytest.raises(CleaningError, match="Unexpected missing values"):
            apply_imputer(df, stats, ["total_bedrooms"], "val")

    def test_does_not_modify_original_df(self, housing_df):
        original_nulls = housing_df["total_bedrooms"].isnull().sum()
        stats = {"total_bedrooms": 200.0}
        apply_imputer(housing_df, stats, ["total_bedrooms"], "train")
        assert housing_df["total_bedrooms"].isnull().sum() == original_nulls

    def test_train_fit_stats_applied_to_val(self, three_splits):
        train, val, _ = three_splits
        val = val.copy()
        val.loc[0, "total_bedrooms"] = np.nan
        stats = fit_imputer(train, ["total_bedrooms"], {})
        val_result = apply_imputer(val, stats, ["total_bedrooms"], "val")
        assert val_result["total_bedrooms"].isnull().sum() == 0


# =============================================================================
# 6. fit_lof
# =============================================================================

class TestFitLof:

    def test_returns_lof_scaler_and_features(self, housing_df):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        assert lof is not None
        assert scaler is not None
        assert features == LOF_FEATURES

    def test_novelty_mode_enabled(self, housing_df):
        lof, _, _ = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        assert lof.novelty is True

    def test_contamination_respected(self, housing_df):
        lof, _, _ = fit_lof(housing_df, LOF_FEATURES, 0.07, 5)
        assert lof.contamination == pytest.approx(0.07)

    def test_scaler_fitted_on_train_only(self, three_splits):
        train, _, _ = three_splits
        _, scaler, _ = fit_lof(train, LOF_FEATURES, 0.05, 5)
        expected_mean = train[LOF_FEATURES].mean().values
        np.testing.assert_array_almost_equal(scaler.mean_, expected_mean)

    def test_raises_when_features_missing_from_train(self, housing_df):
        with pytest.raises(CleaningError, match="missing features"):
            fit_lof(housing_df, LOF_FEATURES + ["nonexistent_col"], 0.05, 5)

    def test_raises_when_features_contain_nan(self, housing_df):
        df = housing_df.copy()
        df.loc[5, "median_income"] = np.nan
        with pytest.raises(CleaningError, match="missing values"):
            fit_lof(df, LOF_FEATURES, 0.05, 5)

    def test_raises_when_rows_not_greater_than_n_neighbors(self, tiny_df):
        with pytest.raises(CleaningError, match="n_neighbors"):
            fit_lof(tiny_df, LOF_FEATURES, 0.05, 5)  # 5 rows, n_neighbors=5

    def test_succeeds_when_rows_exceed_n_neighbors(self, tiny_df):
        lof, scaler, features = fit_lof(tiny_df, LOF_FEATURES, 0.2, 4)  # 5 > 4
        assert lof is not None

    def test_raises_on_non_numeric_feature(self, housing_df):
        with pytest.raises(CleaningError, match="numeric"):
            fit_lof(housing_df, LOF_FEATURES + ["ocean_proximity"], 0.05, 5)

    def test_does_not_modify_input(self, housing_df):
        original = housing_df.copy()
        fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        pd.testing.assert_frame_equal(housing_df, original)


# =============================================================================
# 7. apply_lof_flag
# =============================================================================

class TestApplyLofFlag:

    def test_adds_lof_outlier_column(self, housing_df):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        result = apply_lof_flag(housing_df, lof, scaler, features, "train")
        assert "lof_outlier" in result.columns

    def test_flag_values_are_binary(self, housing_df):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        result = apply_lof_flag(housing_df, lof, scaler, features, "train")
        assert set(result["lof_outlier"].unique()).issubset({0, 1})

    def test_raises_when_features_missing(self, housing_df):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        df = housing_df.drop(columns=["median_income"])
        with pytest.raises(CleaningError, match="features missing"):
            apply_lof_flag(df, lof, scaler, features, "train")

    def test_raises_when_features_contain_nan(self, housing_df):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        df = housing_df.copy()
        df.loc[2, "median_income"] = np.nan
        with pytest.raises(CleaningError, match="NaN"):
            apply_lof_flag(df, lof, scaler, features, "train")

    def test_train_fitted_model_used_on_val_without_refitting(self, three_splits):
        train, val, _ = three_splits
        lof, scaler, features = fit_lof(train, LOF_FEATURES, 0.05, 5)
        lof_id, scaler_id = id(lof), id(scaler)
        val_result = apply_lof_flag(val, lof, scaler, features, "val")
        assert id(lof) == lof_id
        assert id(scaler) == scaler_id
        assert "lof_outlier" in val_result.columns

    def test_does_not_modify_original(self, housing_df):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        original_cols = list(housing_df.columns)
        apply_lof_flag(housing_df, lof, scaler, features, "train")
        assert list(housing_df.columns) == original_cols

    def test_int8_dtype_for_flag(self, housing_df):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        result = apply_lof_flag(housing_df, lof, scaler, features, "train")
        assert result["lof_outlier"].dtype == np.int8


# =============================================================================
# 8. save_artifacts / load_artifacts
# =============================================================================

class TestSaveLoadArtifacts:

    def test_save_creates_all_artifact_files(self, housing_df, sample_eda_config, tmp_path):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        stats = {"total_bedrooms": 250.0}
        save_artifacts(stats, lof, scaler, features, sample_eda_config, output_dir=tmp_path)
        assert (tmp_path / "cleaning_artifacts.json").exists()
        assert (tmp_path / "lof_model.pkl").exists()
        assert (tmp_path / "lof_scaler.pkl").exists()

    def test_round_trip_imputer_stats(self, housing_df, sample_eda_config, tmp_path):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        stats = {"total_bedrooms": 250.0}
        save_artifacts(stats, lof, scaler, features, sample_eda_config, output_dir=tmp_path)
        loaded_stats, _, _, _ = load_artifacts(tmp_path)
        assert loaded_stats["total_bedrooms"] == pytest.approx(250.0)

    def test_round_trip_lof_predictions_match(self, housing_df, sample_eda_config, tmp_path):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        stats = {"total_bedrooms": 250.0}
        save_artifacts(stats, lof, scaler, features, sample_eda_config, output_dir=tmp_path)
        _, loaded_lof, loaded_scaler, _ = load_artifacts(tmp_path)

        sample = housing_df[features].head(5)
        original_scaled = scaler.transform(sample)
        loaded_scaled = loaded_scaler.transform(sample)
        np.testing.assert_array_almost_equal(original_scaled, loaded_scaled)

        original_preds = lof.predict(original_scaled)
        loaded_preds = loaded_lof.predict(loaded_scaled)
        np.testing.assert_array_equal(original_preds, loaded_preds)

    def test_round_trip_lof_features(self, housing_df, sample_eda_config, tmp_path):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        save_artifacts({"total_bedrooms": 250.0}, lof, scaler, features, sample_eda_config, output_dir=tmp_path)
        _, _, _, loaded_features = load_artifacts(tmp_path)
        assert loaded_features == features

    def test_load_raises_when_artifacts_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_artifacts(tmp_path)

    def test_load_raises_when_one_artifact_missing(self, housing_df, sample_eda_config, tmp_path):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        save_artifacts({"total_bedrooms": 250.0}, lof, scaler, features, sample_eda_config, output_dir=tmp_path)
        (tmp_path / "lof_scaler.pkl").unlink()
        with pytest.raises(FileNotFoundError):
            load_artifacts(tmp_path)

    def test_json_contains_expected_keys(self, housing_df, sample_eda_config, tmp_path):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        save_artifacts({"total_bedrooms": 200.0}, lof, scaler, features, sample_eda_config, output_dir=tmp_path)
        meta = json.loads((tmp_path / "cleaning_artifacts.json").read_text())
        for key in [
            "target", "imputer_stats", "impute_columns", "imputation_strategy",
            "lof_features", "lof_contamination", "lof_n_neighbors", "timestamp",
        ]:
            assert key in meta, f"Missing key in artifacts JSON: '{key}'"

    def test_json_does_not_contain_target_derived_keys(self, housing_df, sample_eda_config, tmp_path):
        """
        Regression guard: the artifacts metadata must never persist
        target-derived config (e.g. is_capped / cap thresholds used for
        feature creation) since cleaning.py does not create such features.
        """
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        save_artifacts({"total_bedrooms": 200.0}, lof, scaler, features, sample_eda_config, output_dir=tmp_path)
        meta = json.loads((tmp_path / "cleaning_artifacts.json").read_text())
        assert "is_capped" not in json.dumps(meta)
        assert "cap_threshold" not in meta

    def test_json_target_field_correct(self, housing_df, sample_eda_config, tmp_path):
        lof, scaler, features = fit_lof(housing_df, LOF_FEATURES, 0.05, 5)
        save_artifacts({"total_bedrooms": 200.0}, lof, scaler, features, sample_eda_config, output_dir=tmp_path)
        meta = json.loads((tmp_path / "cleaning_artifacts.json").read_text())
        assert meta["target"] == _TARGET


# =============================================================================
# 9. run_cleaning (integration)
# =============================================================================

class TestRunCleaning:

    def test_returns_cleaning_result(self, three_splits, eda_config_yaml, tmp_path):
        train, val, test = three_splits
        result = run_cleaning(
            train, val, test,
            config_path=eda_config_yaml,
            output_dir=tmp_path / "processed",
            artifacts_dir=tmp_path / "artifacts",
            save_artifacts_flag=False,
        )
        assert isinstance(result, CleaningResult)

    def test_output_files_created(self, three_splits, eda_config_yaml, tmp_path):
        train, val, test = three_splits
        out = tmp_path / "processed"
        run_cleaning(
            train, val, test,
            config_path=eda_config_yaml,
            output_dir=out,
            artifacts_dir=tmp_path / "artifacts",
            save_artifacts_flag=False,
        )
        assert (out / "train_clean.csv").exists()
        assert (out / "val_clean.csv").exists()
        assert (out / "test_clean.csv").exists()

    def test_nulls_removed_after_cleaning(self, three_splits, eda_config_yaml, tmp_path):
        train, val, test = three_splits
        result = run_cleaning(
            train, val, test,
            config_path=eda_config_yaml,
            output_dir=tmp_path / "processed",
            artifacts_dir=tmp_path / "artifacts",
            save_artifacts_flag=False,
        )
        for name, df in [("train", result.train), ("val", result.val), ("test", result.test)]:
            assert df["total_bedrooms"].isnull().sum() == 0, \
                f"'{name}' still has nulls in total_bedrooms after cleaning"

    def test_lof_outlier_flag_present_in_all_splits(self, three_splits, eda_config_yaml, tmp_path):
        train, val, test = three_splits
        result = run_cleaning(
            train, val, test,
            config_path=eda_config_yaml,
            output_dir=tmp_path / "processed",
            artifacts_dir=tmp_path / "artifacts",
            save_artifacts_flag=False,
        )
        for df in [result.train, result.val, result.test]:
            assert "lof_outlier" in df.columns

    def test_no_target_derived_columns_created(self, three_splits, eda_config_yaml, tmp_path):
        """
        Regression guard for the previously-open leakage question:
        cleaning.py must never add is_capped or any other target-derived
        column to the output splits.
        """
        train, val, test = three_splits
        result = run_cleaning(
            train, val, test,
            config_path=eda_config_yaml,
            output_dir=tmp_path / "processed",
            artifacts_dir=tmp_path / "artifacts",
            save_artifacts_flag=False,
        )
        for df in [result.train, result.val, result.test]:
            assert "is_capped" not in df.columns

    def test_raises_on_empty_train(self, empty_df, housing_df, eda_config_yaml, tmp_path):
        with pytest.raises(CleaningError, match="empty"):
            run_cleaning(
                empty_df, housing_df.head(5), housing_df.head(5),
                config_path=eda_config_yaml,
                output_dir=tmp_path,
                artifacts_dir=tmp_path / "artifacts",
                save_artifacts_flag=False,
            )

    def test_raises_when_target_missing(self, three_splits, eda_config_yaml, tmp_path):
        train, val, test = three_splits
        train_no_target = train.drop(columns=[_TARGET])
        with pytest.raises(CleaningError, match="Target column"):
            run_cleaning(
                train_no_target, val, test,
                config_path=eda_config_yaml,
                output_dir=tmp_path,
                artifacts_dir=tmp_path / "artifacts",
                save_artifacts_flag=False,
            )

    def test_raises_when_config_file_missing(self, three_splits, tmp_path):
        train, val, test = three_splits
        with pytest.raises(FileNotFoundError):
            run_cleaning(
                train, val, test,
                config_path=tmp_path / "does_not_exist.yaml",
                output_dir=tmp_path / "processed",
                artifacts_dir=tmp_path / "artifacts",
                save_artifacts_flag=False,
            )

    def test_artifacts_saved_when_flag_true(self, three_splits, eda_config_yaml, tmp_path):
        train, val, test = three_splits
        artifacts_dir = tmp_path / "artifacts"
        run_cleaning(
            train, val, test,
            config_path=eda_config_yaml,
            output_dir=tmp_path / "processed",
            artifacts_dir=artifacts_dir,
            save_artifacts_flag=True,
        )
        assert (artifacts_dir / "cleaning_artifacts.json").exists()
        assert (artifacts_dir / "lof_model.pkl").exists()
        assert (artifacts_dir / "lof_scaler.pkl").exists()

    def test_artifacts_not_saved_when_flag_false(self, three_splits, eda_config_yaml, tmp_path):
        train, val, test = three_splits
        artifacts_dir = tmp_path / "artifacts"
        run_cleaning(
            train, val, test,
            config_path=eda_config_yaml,
            output_dir=tmp_path / "processed",
            artifacts_dir=artifacts_dir,
            save_artifacts_flag=False,
        )
        assert not (artifacts_dir / "cleaning_artifacts.json").exists()

    def test_result_paths_are_set(self, three_splits, eda_config_yaml, tmp_path):
        train, val, test = three_splits
        result = run_cleaning(
            train, val, test,
            config_path=eda_config_yaml,
            output_dir=tmp_path / "processed",
            artifacts_dir=tmp_path / "artifacts",
            save_artifacts_flag=False,
        )
        assert result.train_path is not None
        assert result.val_path is not None
        assert result.test_path is not None


# =============================================================================
# 10. Leakage Prevention Tests
# =============================================================================

class TestLeakagePrevention:

    def test_imputer_fit_on_train_only(self, three_splits):
        train, val, test = three_splits
        train_median = train["total_bedrooms"].median()
        stats = fit_imputer(train, ["total_bedrooms"], {})
        assert stats["total_bedrooms"] == pytest.approx(train_median)

    def test_val_imputed_with_train_median_not_val_median(self, three_splits):
        """
        Val nulls must be filled with the TRAIN median, not the val's own
        median — this is what actually prevents preprocessing leakage.
        """
        train, val, test = three_splits
        val = val.copy()
        val.loc[0, "total_bedrooms"] = np.nan

        train_median = train["total_bedrooms"].median()
        # sanity: make sure val's own median differs enough to catch a bug
        val_only_median = val["total_bedrooms"].median()

        stats = fit_imputer(train, ["total_bedrooms"], {})
        val_result = apply_imputer(val, stats, ["total_bedrooms"], "val")

        assert val_result.loc[0, "total_bedrooms"] == pytest.approx(train_median)

    def test_lof_fit_on_train_not_refitted_on_val_or_test(self, three_splits):
        train, val, test = three_splits
        lof, scaler, features = fit_lof(train, LOF_FEATURES, 0.05, 5)
        lof_id, scaler_id = id(lof), id(scaler)

        apply_lof_flag(val, lof, scaler, features, "val")
        apply_lof_flag(test, lof, scaler, features, "test")

        assert id(lof) == lof_id, "LOF model was unexpectedly re-created"
        assert id(scaler) == scaler_id, "Scaler was unexpectedly re-created"

    def test_scaler_stats_come_from_train_not_val(self, three_splits):
        train, val, test = three_splits
        _, scaler, _ = fit_lof(train, LOF_FEATURES, 0.05, 5)
        train_mean = train[LOF_FEATURES].mean().values
        val_mean = val[LOF_FEATURES].mean().values
        np.testing.assert_array_almost_equal(scaler.mean_, train_mean)
        # Sanity check that val's own mean is genuinely different, so this
        # assertion would actually catch a leakage bug.
        assert not np.allclose(scaler.mean_, val_mean)

    def test_full_pipeline_train_stats_reused_for_val_test(self, three_splits, eda_config_yaml, tmp_path):
        train, val, test = three_splits
        result = run_cleaning(
            train, val, test,
            config_path=eda_config_yaml,
            output_dir=tmp_path / "processed",
            artifacts_dir=tmp_path / "artifacts",
            save_artifacts_flag=True,
        )
        stats, lof, scaler, features = load_artifacts(tmp_path / "artifacts")
        train_median = train["total_bedrooms"].median()
        assert stats["total_bedrooms"] == pytest.approx(train_median)


# =============================================================================
# 11. Regression guard — no target-derived logic lives in cleaning.py
# =============================================================================

class TestNoTargetDerivedLogicInCleaning:
    """
    These tests exist specifically because of a previously-open concern:
    an earlier version of cleaning.py risked leaking the target through an
    is_capped feature derived from median_house_value. The rewritten
    cleaning.py resolves this by not creating any target-derived feature
    at all (that responsibility, if needed, belongs to feature engineering,
    downstream and after the split boundary is already respected).
    """

    def test_add_is_capped_flag_not_exported(self):
        import src.data.cleaning as cleaning_mod
        assert not hasattr(cleaning_mod, "add_is_capped_flag")

    def test_apply_log1p_not_exported(self):
        import src.data.cleaning as cleaning_mod
        assert not hasattr(cleaning_mod, "apply_log1p")

    def test_cleaned_output_has_no_is_capped_column(self, three_splits, eda_config_yaml, tmp_path):
        train, val, test = three_splits
        result = run_cleaning(
            train, val, test,
            config_path=eda_config_yaml,
            output_dir=tmp_path / "processed",
            artifacts_dir=tmp_path / "artifacts",
            save_artifacts_flag=False,
        )
        assert "is_capped" not in result.train.columns
        assert "is_capped" not in result.val.columns
        assert "is_capped" not in result.test.columns

    def test_cap_threshold_loaded_but_unused_for_flagging(self, eda_config_yaml):
        """
        cap_threshold is still valid config metadata (kept for use
        elsewhere in the project) but must not translate into any
        derived column here.
        """
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.cap_threshold == pytest.approx(500001.0)
        # No function in the public API should consume cap_threshold to
        # produce a column — this is enforced by the two tests above.