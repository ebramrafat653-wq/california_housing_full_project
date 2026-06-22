# tests/conftest.py

"""
Pytest configuration and shared fixtures for the California Housing project.

This module:
- Resolves project root across environments (Colab, local, CI)
- Configures Python path for imports
- Provides reusable fixtures for testing data modules
"""

import logging
import sys
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import pytest

from src.utils.logger import get_logger, setup_logging

setup_logging(level=logging.INFO)

logger = get_logger(__name__)

def resolve_project_root(
    candidates: Optional[list[Path]] = None,
    markers: tuple[str, ...] = ("pyproject.toml", ".git", "src"),
) -> Path:
    """
    Resolve the project root directory using heuristic markers.

    Searches candidate paths for directories containing standard
    project markers (pyproject.toml, .git, src/). Falls back to
    environment-specific defaults if needed.

    Args:
        candidates: Optional list of Path objects to search.
                   Defaults to [Colab path, current working directory].
        markers: Tuple of filenames/dirnames that indicate project root.

    Returns:
        Path object pointing to the resolved project root.

    Raises:
        RuntimeError: If no valid project root can be determined.
    """
    search_paths = candidates or [
        Path("/content/california_housing_full_project"),  
        Path.cwd().resolve(),                              
    ]

    for candidate in search_paths:
        if candidate.exists() and any(
            (candidate / marker).exists() for marker in markers
        ):
            logger.debug(f"Project root resolved: {candidate}")
            return candidate

    # Fallback: try to find root by walking up from __file__
    try:
        root = Path(__file__).resolve().parent.parent
        if any((root / marker).exists() for marker in markers):
            logger.debug(f"Project root resolved via fallback: {root}")
            return root
    except NameError:
        pass

    raise RuntimeError(
        f"Project root not found. Searched: {search_paths}. "
        f"Expected markers: {markers}"
    )


# Resolve and configure project root at module load time
_PROJECT_ROOT: Path = resolve_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
    logger.debug(f"Added to sys.path: {_PROJECT_ROOT}")


def pytest_configure(config: pytest.Config) -> None:
    """
    Pytest hook: Configure test environment before collection.

    Initializes logging and adds custom markers for test categorization.
    """
    # Initialize logging for test output
    setup_logging(level=logging.INFO)

    # Register custom markers to avoid pytest warnings
    config.addinivalue_line(
        "markers", "integration: mark test as end-to-end integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as requiring significant execution time"
    )
    config.addinivalue_line(
        "markers", "requires_kaggle: mark test as requiring Kaggle API credentials"
    )
    config.addinivalue_line(
        "markers", "requires_dvc: mark test as requiring DVC initialization"
    )


@pytest.fixture(scope="session")
def project_root() -> Path:
    """
    Provide the resolved project root path to tests.

    Yields:
        Path: Project root directory.
    """
    return _PROJECT_ROOT


@pytest.fixture(scope="session")
def test_data_dir(project_root: Path) -> Path:
    """
    Provide path to the test data directory.

    Args:
        project_root: Injected project root path.

    Yields:
        Path: Directory containing test fixtures/data.
    """
    test_dir = project_root / "tests" / "data"
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


@pytest.fixture(scope="function")
def sample_california_housing_df() -> pd.DataFrame:
    """
    Provide a minimal valid California Housing DataFrame for testing.
    Matches the Kaggle schema and includes intentional flaws (NaNs, Outliers) 
    to test the DataCleaner.
    """
    data = {
        "longitude": [-122.23, -122.22, -122.24],
        "latitude": [37.88, 37.86, 37.85],
        "housing_median_age": [41.0, 21.0, 52.0],
        "total_rooms": [880.0, 7099.0, 1467.0],
        "total_bedrooms": [129.0, np.nan, 190.0],  
        "population": [322.0, 2401.0, 496.0],
        "households": [126.0, 1138.0, 177.0],
        "median_income": [8.3252, 8.3014, 7.2574],
        "median_house_value": [452600.0, 358500.0, 500001.0], 
        "ocean_proximity": ["NEAR BAY", "NEAR BAY", "NEAR BAY"]
    }
    return pd.DataFrame(data)


@pytest.fixture(scope="function")
def empty_df() -> pd.DataFrame:
    """Provide an empty DataFrame for edge-case testing."""
    return pd.DataFrame()


@pytest.fixture(scope="function")
def df_with_null_columns() -> pd.DataFrame:
    """Provide a DataFrame containing fully null columns for validation testing."""
    return pd.DataFrame({
        "valid_col": [1, 2, 3],
        "null_col": [None, None, None],
        "mixed_col": [1, None, 3],
    })


@pytest.fixture(scope="function")
def mock_kaggle_credentials(tmp_path: Path) -> Iterator[Path]:
    """
    Provide a temporary mock kaggle.json for testing credential setup.

    Args:
        tmp_path: Pytest built-in temporary directory fixture.

    Yields:
        Path: Path to the mock credentials file.
    """
    creds_file = tmp_path / "kaggle.json"
    creds_file.write_text(
        '{"username":"test_user","key":"test_api_key"}',
        encoding="utf-8",
    )
    yield creds_file
    # Cleanup handled automatically by tmp_path


@pytest.fixture(scope="function")
def mock_config_yaml(tmp_path: Path) -> Path:
    """
    Provide a temporary mock configuration file for testing.

    Args:
        tmp_path: Pytest built-in temporary directory fixture.

    Returns:
        Path: Path to the mock YAML config file.
    """
    config_content = """
project:
  name: "CaliforniaHousing"
  target: "median_house_value"

dataset:
  kaggle_id: "test/test-dataset"
  expected_files:
    - "test.csv"
  raw_path: "data/raw"
  interim_path: "data/interim"

validation:
  min_rows: 10
  min_cols: 2
  required_columns:
    - "longitude"
    - "latitude"
    - "median_house_value"
"""
    config_file = tmp_path / "test_config.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    return config_file


@pytest.fixture(scope="function")
def skewed_mock_df() -> pd.DataFrame:
    """
    Provide a DataFrame with heavily right-skewed numerical features 
    specifically for testing transformations like log1p.
    """
    np.random.seed(42)

    return pd.DataFrame({
        "total_rooms": np.random.lognormal(mean=4.0, sigma=1.0, size=1000),
        "total_bedrooms": np.random.lognormal(mean=3.0, sigma=0.8, size=1000),
        "population": np.random.lognormal(mean=4.5, sigma=1.2, size=1000),
        "households": np.random.lognormal(mean=3.5, sigma=0.9, size=1000),
    })


# =============================================================================
# DVC TEST FIXTURES (Optional)
# =============================================================================

@pytest.fixture(scope="session")
def dvc_initialized(project_root: Path) -> bool:
    """Check if DVC is initialized; skip tests if not."""
    from src.utils.paths import is_dvc_initialized
    if not is_dvc_initialized():
        pytest.skip("DVC not initialized; run initialize_environment() first")
    return True


@pytest.fixture(scope="function")
def mock_dvc_remote(tmp_path: Path) -> Path:
    """Create a temporary local DVC remote for testing."""
    remote_dir = tmp_path / "dvc_remote"
    remote_dir.mkdir()
    return remote_dir


@pytest.fixture(scope="function")
def dvc_config_yaml(tmp_path: Path) -> Path:
    """
    Provide a temporary mock DVC configuration file for testing.

    Returns:
        Path: Path to the mock DVC config YAML.
    """
    config_content = """
dataset:
  kaggle_id: "test/test-dataset"

dvc:
  enabled: true
  remote:
    gdrive_id: "test_folder_id"
    name: "test_remote"
    default: true
  tracked_paths:
    - "data/raw"
    - "data/processed"
  auto_pull:
    enabled: true
    force: false
    targets:
      - "data/raw"
"""
    config_file = tmp_path / "dvc_config.yaml"
    config_file.write_text(config_content, encoding="utf-8")
    return config_file


# Optional: Skip decorator helper for integration tests
def requires_integration(func):
    """Decorator to skip test unless --integration flag is passed."""
    return pytest.mark.integration(func)


# =============================================================================
# PUBLIC API (for explicit imports if needed)
# =============================================================================

__all__ = [
    "resolve_project_root",
    "project_root",
    "test_data_dir",
    "sample_california_housing_df",
    "empty_df",
    "df_with_null_columns",
    "mock_kaggle_credentials",
    "mock_config_yaml",
    "dvc_initialized",
    "mock_dvc_remote",
    "dvc_config_yaml",
    "requires_integration",
]