# =============================================================================
# tests/test_cleaning.py
# California Housing Project — Unit Tests for src/data/cleaning.py
#
# ARCHITECTURAL ALIGNMENT:
# This test suite matches the PURE VALIDATION version of cleaning.py.
# 
# cleaning.py NOW ONLY DOES:
#   1. Config loading (strict, fail-fast).
#   2. Schema and structural validation.
#   3. Strict missing-value contract enforcement (no unexpected NaNs).
#   4. Saving clean splits and metadata.
#
# cleaning.py DOES NOT DO (Anymore):
#   - Fit imputers, scalers, LOF, or encoders.
#   - Feature engineering or target-derived logic.
#   (All fitting is deferred to the CV-safe pipeline in pipeline.py)
# =============================================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.data.cleaning import (
    _TARGET,
    CleaningError,
    CleaningResult,
    EdaConfig,
    check_missing_values,
    load_eda_config,
    run_cleaning,
    save_cleaned_splits,
    save_cleaning_metadata,
    validate_cleaning_inputs,
    validate_configured_columns,
)

# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture
def housing_df() -> pd.DataFrame:
    """Minimal valid California Housing DataFrame — 60 rows."""
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
    # Inject one NaN in total_bedrooms only (allowed by contract).
    df.loc[0, "total_bedrooms"] = np.nan
    return df


@pytest.fixture
def three_splits(housing_df) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split housing_df into train(40) / val(10) / test(10)."""
    return (
        housing_df.iloc[:40].reset_index(drop=True),
        housing_df.iloc[40:50].reset_index(drop=True),
        housing_df.iloc[50:].reset_index(drop=True),
    )


@pytest.fixture
def eda_config_yaml(tmp_path) -> Path:
    """Write a valid data_config.yaml matching the strict schema."""
    cfg = {
        "project": {"target": "median_house_value"},
        "eda_derived": {
            "missingness": {
                "total_bedrooms": {
                    "impute": True,
                    "imputation_strategy": "median",
                },
                "housing_median_age": {"impute": False},
            },
            "target_summary": {
                "skewness_raw": 0.982,
                "pct_capped": 4.727,
                "n_capped": 683,
                "cap_threshold": 500001.0,
            },
        },
    }
    p = tmp_path / "data_config.yaml"
    p.write_text(yaml.dump(cfg), encoding="utf-8")
    return p


# =============================================================================
# 1. load_eda_config — Strict / Fail-Fast
# =============================================================================

class TestEdaConfig:
    def test_loads_impute_columns_from_yaml(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.impute_columns == ["total_bedrooms"]

    def test_excludes_columns_with_impute_false(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert "housing_median_age" not in cfg.impute_columns

    def test_loads_cap_threshold_from_yaml(self, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        assert cfg.cap_threshold == pytest.approx(500001.0)

    def test_raises_file_not_found_when_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_eda_config(tmp_path / "does_not_exist.yaml")

    def test_raises_on_malformed_yaml(self, tmp_path):
        p = tmp_path / "broken.yaml"
        p.write_text("eda_derived: [this, is, not, a, mapping", encoding="utf-8")
        with pytest.raises(CleaningError):
            load_eda_config(p)

    def test_raises_when_eda_derived_section_missing(self, tmp_path):
        p = tmp_path / "no_eda.yaml"
        p.write_text(yaml.dump({"project": {"target": "median_house_value"}}), encoding="utf-8")
        with pytest.raises(CleaningError, match="eda_derived"):
            load_eda_config(p)


# =============================================================================
# 2. Input & Missing Value Validation (The Core Leakage Guards)
# =============================================================================

class TestValidateCleaningInputs:
    def test_passes_for_valid_splits(self, three_splits):
        validate_cleaning_inputs(*three_splits)  # Should not raise

    def test_raises_on_empty_train(self, three_splits):
        train, val, test = three_splits
        with pytest.raises(CleaningError, match="empty"):
            validate_cleaning_inputs(pd.DataFrame(), val, test)

    def test_raises_when_target_missing(self, three_splits):
        train, val, test = three_splits
        with pytest.raises(CleaningError, match="Target column"):
            validate_cleaning_inputs(train.drop(columns=[_TARGET]), val, test)


class TestCheckMissingValues:
    def test_passes_when_only_allowed_cols_have_nans(self, housing_df):
        # housing_df only has NaN in 'total_bedrooms', which is allowed.
        check_missing_values(housing_df, allowed_cols=["total_bedrooms"], split_name="train")

    def test_raises_on_unexpected_nan(self, housing_df):
        df = housing_df.copy()
        df.loc[2, "population"] = np.nan  # Not in allowed_cols
        with pytest.raises(CleaningError, match="Unexpected missing values"):
            check_missing_values(df, allowed_cols=["total_bedrooms"], split_name="train")

    def test_raises_when_target_has_nan(self, housing_df):
        df = housing_df.copy()
        df.loc[2, _TARGET] = np.nan
        with pytest.raises(CleaningError, match="Target column.*missing values"):
            check_missing_values(df, allowed_cols=["total_bedrooms"], split_name="train")


class TestValidateConfiguredColumns:
    def test_passes_when_columns_exist(self, housing_df, eda_config_yaml):
        cfg = load_eda_config(eda_config_yaml)
        validate_configured_columns(housing_df, cfg)  # Should not raise

    def test_raises_when_configured_column_missing(self, housing_df):
        # Create a fake config asking for a missing column
        cfg = EdaConfig(impute_columns=["nonexistent_column"], imputation_strategy={}, cap_threshold=500000.0)
        with pytest.raises(CleaningError, match="do not exist"):
            validate_configured_columns(housing_df, cfg)


# =============================================================================
# 3. CleaningResult & Metadata Saving
# =============================================================================

class TestCleaningResult:
    def test_summary_contains_shape_and_deferred_notice(self, housing_df):
        result = CleaningResult(train=housing_df, val=housing_df, test=housing_df)
        summary = result.summary()
        assert "60" in summary
        assert "DEFERRED TO CV/TRAINING PIPELINE" in summary


class TestSaveCleaningMetadata:
    def test_saves_correct_metadata_structure(self, eda_config_yaml, tmp_path):
        cfg = load_eda_config(eda_config_yaml)
        path = save_cleaning_metadata(cfg, output_dir=tmp_path)
        
        assert path.exists()
        meta = json.loads(path.read_text())
        
        assert meta["target"] == _TARGET
        assert meta["impute_columns"] == ["total_bedrooms"]
        assert meta["learned_preprocessing"]["imputer"] == "training_pipeline"
        assert "is_capped" in meta["target_derived_features"]
        assert meta["target_derived_features"]["is_capped"] is False


# =============================================================================
# 4. run_cleaning (End-to-End Integration)
# =============================================================================

class TestRunCleaning:
    def test_returns_cleaning_result_and_creates_files(self, three_splits, eda_config_yaml, tmp_path):
        train, val, test = three_splits
        out_dir = tmp_path / "processed"
        art_dir = tmp_path / "artifacts"
        
        result = run_cleaning(
            train, val, test,
            config_path=eda_config_yaml,
            output_dir=out_dir,
            artifacts_dir=art_dir,
            save_artifacts_flag=True,
        )
        
        assert isinstance(result, CleaningResult)
        assert (out_dir / "train_clean.csv").exists()
        assert (out_dir / "val_clean.csv").exists()
        assert (out_dir / "test_clean.csv").exists()
        assert (art_dir / "cleaning_metadata.json").exists()

    def test_data_is_unmodified_pass_through(self, three_splits, eda_config_yaml, tmp_path):
        """Cleaning should NOT impute or transform, just validate and save."""
        train, val, test = three_splits
        original_train_nans = train["total_bedrooms"].isna().sum()
        
        result = run_cleaning(
            train, val, test,
            config_path=eda_config_yaml,
            output_dir=tmp_path / "processed",
            save_artifacts_flag=False,
        )
        
        # The NaN should STILL be there, because imputation is deferred to the Pipeline!
        assert result.train["total_bedrooms"].isna().sum() == original_train_nans


# =============================================================================
# 5. Regression Guards (Leakage Prevention)
# =============================================================================

class TestNoTargetDerivedLogicInCleaning:
    """
    Guard against the historical risk of creating 'is_capped' or 
    applying log1p in the cleaning stage.
    """
    def test_cleaning_module_does_not_export_target_derived_funcs(self):
        import src.data.cleaning as cleaning_mod
        assert not hasattr(cleaning_mod, "add_is_capped_flag")
        assert not hasattr(cleaning_mod, "apply_log1p")

    def test_cleaned_output_has_no_is_capped_column(self, three_splits, eda_config_yaml, tmp_path):
        train, val, test = three_splits
        result = run_cleaning(
            train, val, test,
            config_path=eda_config_yaml,
            output_dir=tmp_path / "processed",
            save_artifacts_flag=False,
        )
        assert "is_capped" not in result.train.columns
        assert "is_capped" not in result.val.columns
        assert "is_capped" not in result.test.columns