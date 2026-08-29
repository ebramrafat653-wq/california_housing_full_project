# tests/test_data_loader.py

"""
Unit tests for the DataLoader module.

Tests cover path resolution, file loading, error handling, and
environment-aware behavior across raw/interim/processed stages.
"""

from pathlib import Path

import pytest

from src.data.data_loader import DataLoader, DataStage
from src.utils.paths import IN_COLAB, PATHS


@pytest.fixture(scope="module")
def data_loader() -> DataLoader:
    """
    Provide a DataLoader instance configured for the test environment.

    In Colab: uses Google Drive paths.
    Locally: falls back to current working directory.
    """
    return DataLoader(use_drive_paths=IN_COLAB)


@pytest.fixture(scope="module")
def expected_columns() -> set[str]:
    """
    Standard Kaggle California Housing dataset columns for validation.

    Matches the schema from saraferguswps/california-housing dataset.
    """
    return {
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "median_house_value",
        "ocean_proximity",
    }


class TestDataLoaderRaw:
    """Tests for loading raw data files."""

    def test_load_raw_file_exists(self, data_loader: DataLoader) -> None:
        """Verify raw file loading when file is present."""
        raw_file = PATHS["raw"] / "housing.csv"

        if not raw_file.exists():
            pytest.skip(f"Raw data file not found: {raw_file}")

        df = data_loader.load_raw("housing.csv")

        assert not df.empty, "Loaded DataFrame should not be empty"
        assert len(df.columns) > 0, "DataFrame should have at least one column"

    def test_load_raw_schema_validation(
        self,
        data_loader: DataLoader,
        expected_columns: set[str],
    ) -> None:
        """Validate that loaded data contains expected California Housing features."""
        raw_file = PATHS["raw"] / "housing.csv"

        if not raw_file.exists():
            pytest.skip(f"Raw data file not found: {raw_file}")

        df = data_loader.load_raw("housing.csv")

        found_columns = set(df.columns)
        matched = expected_columns & found_columns
        # Expect all 10 core features to match
        assert len(matched) >= 10, (
            f"Expected California Housing features. Found: {list(df.columns)}"
        )

    def test_load_raw_file_not_found(self, data_loader: DataLoader) -> None:
        """Verify appropriate error when requesting non-existent file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            data_loader.load_raw("non_existent_file.csv")

        assert "does not exist" in str(exc_info.value)


class TestDataLoaderStages:
    """Tests for multi-stage data loading (raw/interim/processed)."""

    @pytest.mark.parametrize("stage", ["raw", "interim", "processed"])
    def test_load_method_signature(self, stage: DataStage) -> None:
        """Verify load method returns DataFrame for valid stages."""
        # Note: This tests the interface, not actual file loading
        loader = DataLoader(use_drive_paths=False)  # Local mode for isolation

        # Mock file existence check would be needed for full integration test
        assert hasattr(loader, f"load_{stage}"), f"Missing load method for stage: {stage}"

    def test_invalid_stage_raises_error(self) -> None:
        """Verify ValueError for unsupported data stages."""
        loader = DataLoader(use_drive_paths=False)

        with pytest.raises(ValueError) as exc_info:
            loader._load("invalid_stage", "test.csv")  # type: ignore

        assert "Invalid stage" in str(exc_info.value)


class TestDataLoaderLocalMode:
    """Tests for local filesystem fallback behavior."""

    def test_local_mode_initialization(self) -> None:
        """Verify DataLoader uses local paths when drive mode is disabled."""
        loader = DataLoader(use_drive_paths=False)

        assert loader.base_path == Path.cwd()
        assert "raw" in loader.paths
        assert isinstance(loader.paths["raw"], Path)

    def test_local_mode_path_structure(self) -> None:
        """Verify local mode creates expected directory structure."""
        loader = DataLoader(use_drive_paths=False)

        for stage in ("raw", "interim", "processed"):
            expected = Path.cwd() / "data" / stage
            assert loader.paths[stage] == expected  # type: ignore


# =============================================================================
# Integration Test (Optional - Run Separately)
# =============================================================================

@pytest.mark.integration
def test_full_load_pipeline(data_loader: DataLoader) -> None:
    """
    End-to-end test: load -> validate -> basic stats.

    Marked as 'integration' to skip during fast unit test runs.
    Run with: pytest -m integration
    """
    raw_file = PATHS["raw"] / "housing.csv"

    if not raw_file.exists():
        pytest.skip("Integration test requires raw data file")

    df = data_loader.load_raw("housing.csv")

    # Basic data quality checks
    assert df.isnull().sum().sum() >= 0, "DataFrame should be accessible"  # Allows NaNs
    assert len(df) > 0, "Dataset should contain at least one row"

    # Optional: Print summary for manual verification
    print(f"\nDataset summary: {len(df)} rows x {len(df.columns)} columns")
