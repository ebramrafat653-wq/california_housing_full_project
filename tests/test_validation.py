# =============================================================================
# tests/test_validation.py
# California Housing Project — Unit Tests for src/data/validation.py
#
# Fixtures used from conftest.py:
#   - sample_california_housing_df : valid 3-row housing DataFrame
#   - empty_df                     : completely empty DataFrame
#   - df_with_null_columns         : DataFrame with a fully-null column
#   - mock_config_yaml             : minimal YAML (not full schema)
#
# Full-schema config is built inline via `full_config_yaml` fixture
# because mock_config_yaml in conftest is intentionally minimal.
# =============================================================================

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data.validation import (
    ValidationError,
    ValidationReport,
    validate_dataframe,
    _check_not_empty,
    _check_required_columns,
    _check_fully_null_columns,
    _check_target_column,
    _check_missing_values,
    _check_duplicates,
    _check_categorical_values,
    _check_numeric_boundaries,
)


# =============================================================================
# HELPERS & FIXTURES
# =============================================================================

@pytest.fixture(scope="module")
def full_config(tmp_path_factory) -> dict:
    """
    Full data_config.yaml structure as a dict — mirrors configs/data_config.yaml.
    Used directly by check functions (no file I/O needed for unit tests).
    """
    return {
        "project": {
            "target": "median_house_value",
        },
        "validation": {
            "max_missing_ratio": 0.3,
            "min_rows": 2,
            "min_columns": 5,
            "allow_fully_null_columns": False,
            "required_columns": [
                "longitude", "latitude", "housing_median_age",
                "total_rooms", "total_bedrooms", "population",
                "households", "median_income",
                "median_house_value", "ocean_proximity",
            ],
            "allowed_null_columns": ["total_bedrooms"],
            "categorical_features": {
                "ocean_proximity": [
                    "NEAR BAY", "<1H OCEAN", "INLAND", "NEAR OCEAN", "ISLAND"
                ]
            },
            "numeric_boundaries": {
                "longitude":          {"min": -124.35, "max": -114.0},
                "latitude":           {"min": 32.0,    "max": 42.0},
                "housing_median_age": {"min": 1.0,     "max": 52.0},
                "median_house_value": {"min": 14999.0, "max": 500001.0},
                "median_income":      {"min": 0.0,     "max": 16.0},
            },
        },
    }


@pytest.fixture(scope="module")
def full_config_yaml(tmp_path_factory, full_config) -> Path:
    """Write full_config to a temp YAML file for validate_dataframe() tests."""
    import yaml
    p = tmp_path_factory.mktemp("cfg") / "data_config.yaml"
    p.write_text(yaml.dump(full_config), encoding="utf-8")
    return p


def fresh_report() -> ValidationReport:
    """Return a clean ValidationReport for each isolated check test."""
    return ValidationReport()


# =============================================================================
# 1. ValidationReport
# =============================================================================

class TestValidationReport:

    def test_initial_state_is_passing(self):
        report = fresh_report()
        assert report.passed is True
        assert report.critical_errors == []
        assert report.warnings == []

    def test_add_critical_sets_passed_false(self):
        report = fresh_report()
        report.add_critical("Something is broken")
        assert report.passed is False
        assert len(report.critical_errors) == 1

    def test_add_warning_does_not_affect_passed(self):
        report = fresh_report()
        report.add_warning("Something looks odd")
        assert report.passed is True
        assert len(report.warnings) == 1

    def test_summary_contains_status_passed(self):
        report = fresh_report()
        assert "PASSED" in report.summary()

    def test_summary_contains_status_failed(self):
        report = fresh_report()
        report.add_critical("boom")
        assert "FAILED" in report.summary()

    def test_summary_lists_errors_and_warnings(self):
        report = fresh_report()
        report.add_critical("critical msg")
        report.add_warning("warning msg")
        summary = report.summary()
        assert "critical msg" in summary
        assert "warning msg" in summary


# =============================================================================
# 2. _check_not_empty
# =============================================================================

class TestCheckNotEmpty:

    def test_valid_df_passes(self, sample_california_housing_df, full_config):
        report = fresh_report()
        _check_not_empty(sample_california_housing_df, full_config, report)
        assert report.passed is True
        assert report.critical_errors == []

    def test_empty_df_adds_critical(self, empty_df, full_config):
        report = fresh_report()
        _check_not_empty(empty_df, full_config, report)
        assert report.passed is False
        assert any("rows" in e for e in report.critical_errors)

    def test_too_few_columns_adds_critical(self, full_config):
        df = pd.DataFrame({"a": [1, 2, 3]})
        report = fresh_report()
        _check_not_empty(df, full_config, report)
        assert report.passed is False
        assert any("columns" in e for e in report.critical_errors)

    def test_exactly_min_rows_passes(self, full_config):
        min_rows = full_config["validation"]["min_rows"]
        df = pd.DataFrame({f"col_{i}": range(min_rows) for i in range(10)})
        report = fresh_report()
        _check_not_empty(df, full_config, report)
        assert report.passed is True


# =============================================================================
# 3. _check_required_columns
# =============================================================================

class TestCheckRequiredColumns:

    def test_all_columns_present_passes(self, sample_california_housing_df, full_config):
        report = fresh_report()
        _check_required_columns(sample_california_housing_df, full_config, report)
        assert report.passed is True

    def test_missing_one_column_adds_critical(self, sample_california_housing_df, full_config):
        df = sample_california_housing_df.drop(columns=["median_house_value"])
        report = fresh_report()
        _check_required_columns(df, full_config, report)
        assert report.passed is False
        assert any("median_house_value" in e for e in report.critical_errors)

    def test_missing_multiple_columns_adds_one_critical(self, sample_california_housing_df, full_config):
        df = sample_california_housing_df.drop(columns=["longitude", "latitude"])
        report = fresh_report()
        _check_required_columns(df, full_config, report)
        # One critical error listing all missing columns
        assert len(report.critical_errors) == 1
        assert "longitude" in report.critical_errors[0]
        assert "latitude" in report.critical_errors[0]

    def test_extra_columns_still_passes(self, sample_california_housing_df, full_config):
        df = sample_california_housing_df.copy()
        df["extra_col"] = 999
        report = fresh_report()
        _check_required_columns(df, full_config, report)
        assert report.passed is True


# =============================================================================
# 4. _check_fully_null_columns
# =============================================================================

class TestCheckFullyNullColumns:

    def test_no_null_columns_passes(self, sample_california_housing_df, full_config):
        report = fresh_report()
        _check_fully_null_columns(sample_california_housing_df, full_config, report)
        assert report.passed is True

    def test_fully_null_column_adds_critical(self, df_with_null_columns, full_config):
        report = fresh_report()
        _check_fully_null_columns(df_with_null_columns, full_config, report)
        assert report.passed is False
        assert any("null_col" in e for e in report.critical_errors)

    def test_allow_fully_null_skips_check(self, df_with_null_columns, full_config):
        cfg = {**full_config}
        cfg["validation"] = {**full_config["validation"], "allow_fully_null_columns": True}
        report = fresh_report()
        _check_fully_null_columns(df_with_null_columns, cfg, report)
        assert report.passed is True

    def test_partial_null_column_is_not_flagged(self, sample_california_housing_df, full_config):
        # total_bedrooms has 1 NaN — should NOT be flagged here
        report = fresh_report()
        _check_fully_null_columns(sample_california_housing_df, full_config, report)
        assert report.passed is True


# =============================================================================
# 5. _check_target_column
# =============================================================================

class TestCheckTargetColumn:

    def test_valid_target_passes(self, sample_california_housing_df, full_config):
        report = fresh_report()
        _check_target_column(sample_california_housing_df, full_config, report)
        assert report.passed is True

    def test_missing_target_adds_critical(self, sample_california_housing_df, full_config):
        df = sample_california_housing_df.drop(columns=["median_house_value"])
        report = fresh_report()
        _check_target_column(df, full_config, report)
        assert report.passed is False
        assert any("not found" in e for e in report.critical_errors)

    def test_non_numeric_target_adds_critical(self, sample_california_housing_df, full_config):
        df = sample_california_housing_df.copy()
        df["median_house_value"] = df["median_house_value"].astype(str)
        report = fresh_report()
        _check_target_column(df, full_config, report)
        assert report.passed is False
        assert any("numeric" in e for e in report.critical_errors)

    def test_target_with_nulls_adds_critical(self, sample_california_housing_df, full_config):
        df = sample_california_housing_df.copy()
        df.loc[0, "median_house_value"] = np.nan
        report = fresh_report()
        _check_target_column(df, full_config, report)
        assert report.passed is False
        assert any("null" in e for e in report.critical_errors)


# =============================================================================
# 6. _check_missing_values
# =============================================================================

class TestCheckMissingValues:

    def test_allowed_null_within_ratio_is_info_only(self, sample_california_housing_df, full_config):
        # total_bedrooms has 1/3 nulls = 33% → exceeds 30% → WARNING
        # Let's use a df where it's under the limit (1 null out of 10 rows)
        df = sample_california_housing_df.copy()
        for _ in range(7):
            df = pd.concat([df, sample_california_housing_df.iloc[[0]]], ignore_index=True)
        # 10 rows, 1 null in total_bedrooms = 10% → within 30% limit
        df.loc[0, "total_bedrooms"] = np.nan
        df.loc[1:, "total_bedrooms"] = 150.0
        report = fresh_report()
        _check_missing_values(df, full_config, report)
        assert report.passed is True
        assert report.warnings == []

    def test_allowed_null_above_ratio_adds_warning(self, full_config):
        # Create df where total_bedrooms has > 30% nulls
        n = 10
        df = pd.DataFrame({
            col: [1.0] * n
            for col in full_config["validation"]["required_columns"]
            if col != "ocean_proximity"
        })
        df["ocean_proximity"] = "NEAR BAY"
        df.loc[:4, "total_bedrooms"] = np.nan   # 50% nulls
        report = fresh_report()
        _check_missing_values(df, full_config, report)
        assert report.passed is True          # warning, not critical
        assert len(report.warnings) == 1
        assert "total_bedrooms" in report.warnings[0]

    def test_unexpected_null_in_non_allowed_column_adds_critical(
        self, sample_california_housing_df, full_config
    ):
        df = sample_california_housing_df.copy()
        df.loc[0, "longitude"] = np.nan      # longitude not in allowed_null_columns
        report = fresh_report()
        _check_missing_values(df, full_config, report)
        assert report.passed is False
        assert any("longitude" in e for e in report.critical_errors)

    def test_no_nulls_produces_no_issues(self, full_config):
        df = pd.DataFrame({
            col: [1.0] * 5
            for col in full_config["validation"]["required_columns"]
            if col != "ocean_proximity"
        })
        df["ocean_proximity"] = "NEAR BAY"
        report = fresh_report()
        _check_missing_values(df, full_config, report)
        assert report.passed is True
        assert report.warnings == []


# =============================================================================
# 7. _check_duplicates
# =============================================================================

class TestCheckDuplicates:

    def test_no_duplicates_passes(self, sample_california_housing_df):
        report = fresh_report()
        _check_duplicates(sample_california_housing_df, report)
        assert report.passed is True
        assert report.warnings == []

    def test_duplicate_rows_add_warning(self, sample_california_housing_df):
        df = pd.concat(
            [sample_california_housing_df, sample_california_housing_df.iloc[[0]]],
            ignore_index=True,
        )
        report = fresh_report()
        _check_duplicates(df, report)
        assert report.passed is True          # warning, not critical
        assert len(report.warnings) == 1
        assert "duplicate" in report.warnings[0].lower()

    def test_warning_count_reflects_actual_dupes(self, sample_california_housing_df):
            df = pd.concat([sample_california_housing_df] * 2, ignore_index=True)  
            report = fresh_report()
            _check_duplicates(df, report)
            # 6 rows total, 3 originals → 3 duplicates
            assert "3" in report.warnings[0]


# =============================================================================
# 8. _check_categorical_values
# =============================================================================

class TestCheckCategoricalValues:

    def test_known_categories_pass(self, sample_california_housing_df, full_config):
        report = fresh_report()
        _check_categorical_values(sample_california_housing_df, full_config, report)
        assert report.passed is True
        assert report.warnings == []

    def test_unknown_category_adds_warning(self, sample_california_housing_df, full_config):
        df = sample_california_housing_df.copy()
        df.loc[0, "ocean_proximity"] = "LAKEFRONT"   # not in allowed set
        report = fresh_report()
        _check_categorical_values(df, full_config, report)
        assert report.passed is True          # warning, not critical
        assert any("LAKEFRONT" in w for w in report.warnings)

    def test_missing_categorical_column_is_skipped(self, full_config):
        # ocean_proximity missing → already caught by _check_required_columns
        # this check should silently skip it
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        report = fresh_report()
        _check_categorical_values(df, full_config, report)
        assert report.passed is True
        assert report.warnings == []

    def test_all_valid_categories_accepted(self, full_config):
        allowed = full_config["validation"]["categorical_features"]["ocean_proximity"]
        rows = [{"ocean_proximity": cat} for cat in allowed]
        df = pd.DataFrame(rows)
        report = fresh_report()
        _check_categorical_values(df, full_config, report)
        assert report.warnings == []


# =============================================================================
# 9. _check_numeric_boundaries
# =============================================================================

class TestCheckNumericBoundaries:

    def test_in_bounds_values_pass(self, sample_california_housing_df, full_config):
        report = fresh_report()
        _check_numeric_boundaries(sample_california_housing_df, full_config, report)
        assert report.passed is True
        assert report.warnings == []

    def test_value_below_min_adds_warning(self, sample_california_housing_df, full_config):
        df = sample_california_housing_df.copy()
        df.loc[0, "longitude"] = -200.0      # below min -124.35
        report = fresh_report()
        _check_numeric_boundaries(df, full_config, report)
        assert report.passed is True          # warning, not critical
        assert any("longitude" in w and "below" in w for w in report.warnings)

    def test_value_above_max_adds_warning(self, sample_california_housing_df, full_config):
        df = sample_california_housing_df.copy()
        df.loc[0, "median_income"] = 999.0   # above max 16.0
        report = fresh_report()
        _check_numeric_boundaries(df, full_config, report)
        assert any("median_income" in w and "above" in w for w in report.warnings)

    def test_missing_column_is_skipped(self, full_config):
        df = pd.DataFrame({"other_col": [1, 2, 3]})
        report = fresh_report()
        _check_numeric_boundaries(df, full_config, report)
        assert report.warnings == []

    def test_nulls_are_ignored_in_boundary_check(self, full_config):
        # NaN in a bounded column should not trigger a warning
        df = pd.DataFrame({"longitude": [np.nan, -120.0, -118.0]})
        report = fresh_report()
        _check_numeric_boundaries(df, full_config, report)
        assert report.warnings == []


# =============================================================================
# 10. validate_dataframe (integration-level unit tests)
# =============================================================================

class TestValidateDataframe:

    def test_valid_df_returns_passing_report(
        self, sample_california_housing_df, full_config_yaml
    ):
        report = validate_dataframe(
            sample_california_housing_df,
            config_path=full_config_yaml,
            raise_on_failure=False,
        )
        assert report.passed is True
        assert report.critical_errors == []

    def test_valid_df_does_not_raise(
        self, sample_california_housing_df, full_config_yaml
    ):
        # Should complete without raising
        validate_dataframe(
            sample_california_housing_df,
            config_path=full_config_yaml,
            raise_on_failure=True,
        )

    def test_missing_column_raises_validation_error(
        self, sample_california_housing_df, full_config_yaml
    ):
        df = sample_california_housing_df.drop(columns=["median_house_value"])
        with pytest.raises(ValidationError):
            validate_dataframe(df, config_path=full_config_yaml, raise_on_failure=True)

    def test_raise_on_failure_false_returns_report_not_raise(
        self, sample_california_housing_df, full_config_yaml
    ):
        df = sample_california_housing_df.drop(columns=["median_house_value"])
        report = validate_dataframe(
            df, config_path=full_config_yaml, raise_on_failure=False
        )
        assert report.passed is False
        assert report.critical_errors != []

    def test_missing_config_raises_file_not_found(
        self, sample_california_housing_df
    ):
        with pytest.raises(FileNotFoundError):
            validate_dataframe(
                sample_california_housing_df,
                config_path="non_existent_config.yaml",
            )

    def test_warnings_do_not_cause_raise(
        self, sample_california_housing_df, full_config_yaml
    ):
        # Duplicate rows → warning only → should NOT raise
        df = pd.concat(
            [sample_california_housing_df, sample_california_housing_df.iloc[[0]]],
            ignore_index=True,
        )
        report = validate_dataframe(
            df, config_path=full_config_yaml, raise_on_failure=True
        )
        assert report.passed is True
        assert len(report.warnings) >= 1

    def test_empty_df_raises_validation_error(self, empty_df, full_config_yaml):
        with pytest.raises(ValidationError):
            validate_dataframe(empty_df, config_path=full_config_yaml)

    def test_report_summary_is_string(
        self, sample_california_housing_df, full_config_yaml
    ):
        report = validate_dataframe(
            sample_california_housing_df,
            config_path=full_config_yaml,
            raise_on_failure=False,
        )
        assert isinstance(report.summary(), str)
        assert len(report.summary()) > 0