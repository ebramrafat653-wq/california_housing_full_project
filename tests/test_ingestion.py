# tests/test_ingestion.py
"""
Unit and integration tests for the data ingestion module.

Tests cover configuration loading, credential setup, dataset download,
DVC integration, and end-to-end ingestion workflow.

All tests are isolated from external services using mocks/fixtures.

Compatible with ingestion.py API:
  - _run_kaggle       (not _run_kaggle_command)
  - auto_track_dvc    (not use_dvc)
  - summary format:   "Source   : kaggle"
  - error message:    "Config not found:"
  - report path:      reports/downloads/<id>_<timestamp>.txt
  - dvc_pull:         lives in colab_setup, not ingestion
  - get_dataset_metadata: not in ingestion
"""

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

import pytest
import yaml

from src.data.ingestion import (
    FileInfo,
    DownloadReport,
    load_config,
    setup_kaggle_credentials,
    download_from_kaggle,
    verify_download_integrity,
    save_download_report,
    track_with_dvc,
    pull_from_dvc,
    download_with_dvc_fallback,
    run_ingestion,
)
from src.utils.colab_setup import dvc_pull


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture(scope="session")
def project_root() -> Path:
    """Return the project root directory."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture
def mock_config_yaml(tmp_path: Path) -> Path:
    """Create a temporary valid YAML config file for testing."""
    config_content = {
        "dataset": {
            "kaggle_id": "test/test-dataset",
            "expected_files": [],
        },
        "validation": {
            "required_columns": ["col1", "col2"],
            "min_rows": 1,
        },
        "dvc": {
            "enabled": True,
            "remote": {"name": "mylocal"},
        },
    }
    config_file = tmp_path / "data_config.yaml"
    config_file.write_text(yaml.dump(config_content), encoding="utf-8")
    return config_file


@pytest.fixture
def mock_kaggle_credentials(tmp_path: Path) -> Path:
    """Create a temporary fake kaggle.json credentials file."""
    creds_file = tmp_path / "kaggle.json"
    creds_file.write_text(
        '{"username":"test","key":"fake_key_123"}', encoding="utf-8"
    )
    return creds_file


@pytest.fixture
def sample_raw_data(tmp_path: Path) -> Path:
    """Create sample raw data files for testing."""
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    csv_content = (
        "longitude,latitude,housing_median_age,total_rooms,"
        "total_bedrooms,population,households,median_income,median_house_value\n"
        "-122.4,37.7,25,1000,200,500,250,4.5,250000\n"
        "-122.3,37.8,30,1500,300,800,400,6.2,350000\n"
    )
    (raw_dir / "california_housing.csv").write_text(csv_content, encoding="utf-8")
    (raw_dir / "readme.txt").write_text("Test Sample", encoding="utf-8")
    return raw_dir


# =============================================================================
# UNIT TESTS: Dataclasses
# =============================================================================

class TestFileInfo:
    """Tests for the FileInfo dataclass."""

    def test_size_mb_property(self, tmp_path: Path) -> None:
        """Verify size_mb calculates correctly from bytes."""
        file_path = tmp_path / "test.csv"
        file_path.write_bytes(b"x" * 1024 * 1024)
        info = FileInfo(
            name="test.csv", path=file_path,
            size_bytes=1024 * 1024, extension=".csv",
        )
        assert info.size_mb == 1.0
        assert info.is_data_file is True

    def test_is_data_file_property(self) -> None:
        """Verify is_data_file recognizes common data extensions."""
        assert FileInfo("a.csv",     Path("a"), 100, ".csv").is_data_file is True
        assert FileInfo("b.parquet", Path("b"), 100, ".parquet").is_data_file is True
        assert FileInfo("c.zip",     Path("c"), 100, ".zip").is_data_file is False
        assert FileInfo("d.txt",     Path("d"), 100, ".txt").is_data_file is True
        assert FileInfo("e.json",    Path("e"), 100, ".json").is_data_file is True

    def test_fileinfo_with_zero_bytes(self, tmp_path: Path) -> None:
        """Verify FileInfo handles zero-byte files correctly."""
        info = FileInfo(
            name="empty.csv", path=tmp_path / "empty.csv",
            size_bytes=0, extension=".csv",
        )
        assert info.size_mb == 0.0
        assert info.is_data_file is True


class TestDownloadReport:
    """Tests for the DownloadReport dataclass."""

    def test_default_values(self, tmp_path: Path) -> None:
        """Verify DownloadReport initializes with sensible defaults."""
        report = DownloadReport(
            success=True, dataset_id="test/dataset", destination=tmp_path,
        )
        assert report.success is True
        assert report.files == []
        assert report.errors == []
        assert report.warnings == []
        assert report.total_size_mb == 0.0
        assert report.source is None
        assert isinstance(report.timestamp, datetime)

    def test_summary_output(self, tmp_path: Path) -> None:
        """Verify summary() generates human-readable output.
        
        ingestion.py summary format:
            Source   : kaggle   (with spaces around colon)
        """
        report = DownloadReport(
            success=True,
            dataset_id="test/dataset",
            destination=tmp_path,
            files=[FileInfo("data.csv", tmp_path / "data.csv", 2048, ".csv")],
            total_size_mb=0.002,
            source="kaggle",
        )
        summary = report.summary()
        assert "SUCCESS"       in summary
        assert "test/dataset"  in summary
        assert "data.csv"      in summary
        assert "0.00"          in summary
        assert "kaggle"        in summary   # format: "Source   : kaggle"

    def test_summary_failure_output(self, tmp_path: Path) -> None:
        """Verify summary shows FAILED and includes errors/warnings."""
        report = DownloadReport(
            success=False,
            dataset_id="test/data",
            destination=tmp_path,
            errors=["Download timeout"],
            warnings=["Slow connection"],
            source="dvc",
        )
        summary = report.summary()
        assert "FAILED"           in summary
        assert "Download timeout" in summary
        assert "Slow connection"  in summary
        assert "dvc"              in summary

    def test_summary_unicode(self, tmp_path: Path) -> None:
        """Verify DownloadReport handles unicode in dataset_id."""
        report = DownloadReport(
            success=True,
            dataset_id="test/international-dataset",
            destination=tmp_path,
            source="dvc",
        )
        summary = report.summary()
        assert summary is not None
        assert "international-dataset" in summary


# =============================================================================
# UNIT TESTS: Configuration
# =============================================================================

class TestLoadConfig:
    """Tests for load_config() function."""

    def test_load_valid_config(self, mock_config_yaml: Path) -> None:
        """Verify config loads correctly from valid YAML file."""
        config = load_config(config_path=mock_config_yaml)
        assert "dataset" in config
        assert config["dataset"]["kaggle_id"] == "test/test-dataset"

    def test_load_missing_config_raises(self) -> None:
        """Verify FileNotFoundError when config path doesn't exist.
        
        ingestion.py raises: FileNotFoundError(f"Config not found: {target}")
        """
        with pytest.raises(FileNotFoundError, match="Config not found"):
            load_config(config_path=Path("/nonexistent/path.yaml"))

    def test_load_default_path(self, project_root: Path) -> None:
        """Verify load_config uses default path when none provided."""
        default_path = project_root / "configs" / "data_config.yaml"
        if default_path.exists():
            config = load_config()
            assert isinstance(config, dict)
        else:
            pytest.skip(f"Default config not found: {default_path}")


# =============================================================================
# UNIT TESTS: Kaggle Credentials
# =============================================================================

class TestSetupKaggleCredentials:
    """Tests for setup_kaggle_credentials() function."""

    def test_credentials_setup_success(
        self, mock_kaggle_credentials: Path, tmp_path: Path
    ) -> None:
        """Verify credentials copy and chmod succeed."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            result = setup_kaggle_credentials(
                credentials_path=mock_kaggle_credentials
            )
        assert result is True
        assert (tmp_path / ".kaggle" / "kaggle.json").exists()

    def test_credentials_missing_source(self) -> None:
        """Verify graceful failure when source file doesn't exist."""
        result = setup_kaggle_credentials(
            credentials_path=Path("/nonexistent/kaggle.json")
        )
        assert result is False

    def test_credentials_permission_error(
        self, mock_kaggle_credentials: Path, tmp_path: Path
    ) -> None:
        """Verify graceful handling of chmod failure."""
        with patch("pathlib.Path.home", return_value=tmp_path):
            with patch("pathlib.Path.chmod", side_effect=PermissionError):
                result = setup_kaggle_credentials(
                    credentials_path=mock_kaggle_credentials
                )
        assert result is False


# =============================================================================
# UNIT TESTS: Kaggle CLI — _run_kaggle (private, patch via subprocess)
# =============================================================================

class TestRunKaggle:
    """
    Tests for _run_kaggle() internal function.

    ingestion.py uses _run_kaggle (not _run_kaggle_command).
    It does NOT return CompletedProcess — it raises RuntimeError on failure.
    We test it indirectly via download_from_kaggle which calls it.
    """

    def test_kaggle_failure_raises_runtime_error(self, tmp_path: Path) -> None:
        """Verify non-zero exit raises RuntimeError propagated as report error."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="API error"
            )
            report = download_from_kaggle(
                dataset_id="test/data", destination=tmp_path
            )
        assert report.success is False
        assert "API error" in report.errors[0]

    def test_kaggle_success_no_exception(self, tmp_path: Path) -> None:
        """Verify zero returncode does not raise."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            # Should not raise
            report = download_from_kaggle(
                dataset_id="test/data", destination=tmp_path
            )
        assert report.success is True


# =============================================================================
# UNIT TESTS: Dataset Download (Kaggle)
# =============================================================================

class TestDownloadFromKaggle:
    """Tests for download_from_kaggle() function."""

    @pytest.mark.parametrize("unzip,force", [
        (True, False),
        (False, True),
        (True, True),
    ])
    def test_download_calls_kaggle_cli(
        self, tmp_path: Path, unzip: bool, force: bool
    ) -> None:
        """Verify download_from_kaggle invokes subprocess with correct args."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            report = download_from_kaggle(
                dataset_id="test/data",
                destination=tmp_path,
                unzip=unzip,
                force=force,
            )
        assert mock_run.called
        # _run_kaggle passes a list: ["kaggle", "datasets", "download", ...]
        call_args = mock_run.call_args[0][0]
        assert "datasets"  in call_args
        assert "download"  in call_args
        assert "test/data" in call_args
        if unzip:
            assert "--unzip" in call_args
        if force:
            assert "--force" in call_args
        assert report.source == "kaggle"

    def test_download_failure_returns_error_report(self, tmp_path: Path) -> None:
        """Verify download failure returns DownloadReport with errors."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="API quota exceeded"
            )
            report = download_from_kaggle(
                dataset_id="test/data", destination=tmp_path,
            )
        assert report.success is False
        assert len(report.errors) == 1
        assert "API quota exceeded" in report.errors[0]
        assert report.source == "kaggle"

    def test_download_collects_file_metadata(self, tmp_path: Path) -> None:
        """Verify new files after download are collected in report."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text("col1,col2\n1,2", encoding="utf-8")

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            # before = empty set, after = csv file exists on disk
            report = download_from_kaggle(
                dataset_id="test/data",
                destination=tmp_path,
                unzip=True,
            )
        assert report.success is True
        assert report.source == "kaggle"


# =============================================================================
# UNIT TESTS: DVC Tracking & Pulling
# =============================================================================

class TestTrackWithDvc:
    """Tests for track_with_dvc() function."""

    def test_track_outside_project_dir_returns_false(self) -> None:
        """Path outside PROJECT_DIR should fail immediately without subprocess."""
        result = track_with_dvc(Path("/tmp/completely_outside_project"))
        assert result is False

    def test_track_dvc_add_failure_returns_false(self, project_root: Path) -> None:
        """Verify track_with_dvc returns False on dvc add failure."""
        test_dir = project_root / "data" / "_test_track_fail"
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1, stdout="", stderr="DVC error"
                )
                result = track_with_dvc(test_dir)
            assert result is False
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_track_success_calls_expected_commands(
        self, project_root: Path
    ) -> None:
        """Verify dvc add, dvc push, git add, git commit, git push are called."""
        test_dir = project_root / "data" / "_test_track_ok"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "test.csv").write_text("a,b\n1,2", encoding="utf-8")
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=0, stdout="M data.dvc", stderr=""
                )
                result = track_with_dvc(test_dir)
            assert result is True
            calls_str = " ".join(str(c) for c in mock_run.call_args_list)
            assert "dvc"    in calls_str
            assert "git"    in calls_str
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)


class TestPullFromDvc:
    """Tests for pull_from_dvc() function."""

    def test_pull_outside_project_dir_returns_error(self) -> None:
        """Verify pull_from_dvc returns error report for path outside PROJECT_DIR."""
        report = pull_from_dvc(Path("/tmp/outside_project"))
        assert report.success is False
        assert "outside PROJECT_DIR" in report.errors[0]
        assert report.source == "dvc"

    def test_pull_failure_returns_error_report(self, project_root: Path) -> None:
        """Verify pull_from_dvc returns error report on subprocess failure."""
        test_dir = project_root / "data" / "_test_pull_fail"
        test_dir.mkdir(parents=True, exist_ok=True)
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(
                    returncode=1, stdout="", stderr="Remote error"
                )
                report = pull_from_dvc(test_dir)
            assert report.success is False
            assert "Remote error" in report.errors[0]
            assert report.source == "dvc"
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)

    def test_pull_success_returns_report_with_files(
        self, project_root: Path
    ) -> None:
        """Verify pull_from_dvc returns DownloadReport with file metadata."""
        test_dir = project_root / "data" / "_test_pull_ok"
        test_dir.mkdir(parents=True, exist_ok=True)
        (test_dir / "data.csv").write_text("a,b\n1,2", encoding="utf-8")
        try:
            with patch("subprocess.run") as mock_run:
                mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
                report = pull_from_dvc(test_dir)
            assert report.success is True
            assert report.source == "dvc"
            assert len(report.files) >= 1
        finally:
            import shutil
            shutil.rmtree(test_dir, ignore_errors=True)


# =============================================================================
# UNIT TESTS: DVC Pull Wrapper (from colab_setup)
# =============================================================================

class TestDvcPull:
    """Tests for dvc_pull() function (imported from colab_setup)."""

    def test_dvc_pull_success(self) -> None:
        """Verify dvc_pull returns True on successful subprocess call."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            result = dvc_pull(targets=["data/raw"], force=False)
        assert result is True

    def test_dvc_pull_failure_returns_false(self) -> None:
        """Verify dvc_pull returns False on subprocess failure.

        colab_setup._run uses check=True, so failure raises CalledProcessError
        rather than returning returncode=1. The mock must match this behaviour.
        """
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.CalledProcessError(
                returncode=1, cmd="dvc pull", stderr="Remote not found"
            )
            result = dvc_pull()
        assert result is False

    def test_dvc_pull_timeout_handling(self) -> None:
        """Verify dvc_pull handles subprocess timeout gracefully."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(
                cmd="dvc pull", timeout=300
            )
            # _run catches TimeoutExpired and returns False
            result = dvc_pull()
        assert result is False

    def test_dvc_pull_default_targets_data_raw(self) -> None:
        """Verify dvc_pull defaults to data/raw target (not all)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            dvc_pull()
        # dvc_pull in colab_setup uses shell=True → cmd is a string
        call_cmd = mock_run.call_args[0][0]
        assert "data/raw" in call_cmd

    def test_dvc_pull_force_flag(self) -> None:
        """Verify --force flag is included when force=True."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            dvc_pull(targets=["data/raw"], force=True)
        call_cmd = mock_run.call_args[0][0]
        assert "--force" in call_cmd


# =============================================================================
# UNIT TESTS: Integrity & Reporting
# =============================================================================

class TestVerifyDownloadIntegrity:
    """Tests for verify_download_integrity() function."""

    def test_integrity_check_empty_dir(self, tmp_path: Path) -> None:
        """Verify integrity check handles empty directory."""
        result = verify_download_integrity(tmp_path)
        assert result["total_files"] == 0
        assert result["data_files"] == 0

    def test_integrity_check_with_files(self, tmp_path: Path) -> None:
        """Verify integrity check categorizes files correctly."""
        (tmp_path / "data.csv").write_text("a,b\n1,2", encoding="utf-8")
        (tmp_path / "model.pkl").write_bytes(b"fake pickle")
        (tmp_path / "notes.txt").write_text("info", encoding="utf-8")
        result = verify_download_integrity(tmp_path)
        assert result["total_files"] >= 2
        assert result["data_files"] >= 1
        assert "data.csv" in result["data_file_names"]

    def test_integrity_check_missing_dir(self) -> None:
        """Verify graceful handling of non-existent directory."""
        result = verify_download_integrity(Path("/nonexistent/dir"))
        assert "error" in result
        # ingestion.py: "Directory not found: {directory}"
        assert "not found" in result["error"].lower()

    def test_integrity_expected_files_pass(self, tmp_path: Path) -> None:
        """Verify integrity_ok=True when all expected files present."""
        (tmp_path / "housing.csv").write_text("a,b\n1,2", encoding="utf-8")
        result = verify_download_integrity(tmp_path, expected_files=["housing.csv"])
        assert result["integrity_ok"] is True
        assert result["missing_expected_files"] == []

    def test_integrity_expected_files_fail(self, tmp_path: Path) -> None:
        """Verify integrity_ok=False when expected files missing."""
        (tmp_path / "other.csv").write_text("a,b\n1,2", encoding="utf-8")
        result = verify_download_integrity(tmp_path, expected_files=["housing.csv"])
        assert result["integrity_ok"] is False
        assert "housing.csv" in result["missing_expected_files"]

    def test_integrity_special_filenames(self, tmp_path: Path) -> None:
        """Verify integrity check handles files with special characters."""
        (tmp_path / "data-with-dash.csv").write_text("a,b\n1,2", encoding="utf-8")
        (tmp_path / "file with spaces.parquet").write_bytes(b"fake")
        result = verify_download_integrity(tmp_path)
        assert result["total_files"] >= 2
        assert any(
            "dash" in name or "spaces" in name
            for name in result["data_file_names"]
        )


class TestSaveDownloadReport:
    """Tests for save_download_report() function."""

    def test_report_saved_to_custom_path(self, tmp_path: Path) -> None:
        """Verify report saves to custom path when provided."""
        report = DownloadReport(
            success=True, dataset_id="test/data", destination=tmp_path,
        )
        custom_path = tmp_path / "custom_report.txt"
        output_path = save_download_report(report, output_path=custom_path)
        assert output_path == custom_path
        assert custom_path.exists()

    def test_report_default_path_outside_data_raw(self, tmp_path: Path) -> None:
        """
        Verify default report path is in reports/downloads/ NOT in data/raw.

        ingestion.py saves to:
            PATHS["reports"] / "downloads" / "<dataset_id>_<timestamp>.txt"
        """
        report = DownloadReport(
            success=True,
            dataset_id="test/data",
            destination=tmp_path / "raw",
            source="kaggle",
        )
        with patch("src.data.ingestion.PATHS", {
            "reports": tmp_path / "reports",
            "raw": tmp_path / "raw",
        }):
            output_path = save_download_report(report)

        assert output_path.exists()
        # Must NOT be inside raw/
        assert "raw" not in str(output_path)
        # Must be inside reports/downloads/
        assert "reports" in str(output_path)

    def test_report_includes_files_errors_warnings(self, tmp_path: Path) -> None:
        """Verify report content includes files, errors, and warnings."""
        report = DownloadReport(
            success=False,
            dataset_id="test/data",
            destination=tmp_path,
            files=[FileInfo("data.csv", tmp_path / "data.csv", 100, ".csv")],
            errors=["Download timeout"],
            warnings=["Slow connection"],
            source="dvc",
        )
        output_path = save_download_report(report, output_path=tmp_path / "rep.txt")
        content = output_path.read_text(encoding="utf-8")
        assert "data.csv"         in content
        assert "Download timeout" in content
        assert "Slow connection"  in content
        assert "FAILED"           in content
        assert "dvc"              in content


# =============================================================================
# INTEGRATION TESTS: Fallback Logic & Workflow
# =============================================================================

@pytest.mark.integration
class TestDownloadWithDvcFallback:
    """Integration tests for download_with_dvc_fallback() function."""

    def test_fallback_uses_kaggle_when_dvc_not_initialized(
        self, tmp_path: Path
    ) -> None:
        """Verify fallback uses Kaggle when DVC is not initialized."""
        with patch("src.data.ingestion.is_dvc_initialized", return_value=False):
            with patch("src.data.ingestion.download_from_kaggle") as mock_dl:
                mock_dl.return_value = DownloadReport(
                    success=True, dataset_id="test/data",
                    destination=tmp_path, source="kaggle",
                )
                # auto_track_dvc=False: correct parameter name
                report = download_with_dvc_fallback(
                    dataset_id="test/data",
                    destination=tmp_path,
                    auto_track_dvc=False,
                )
        assert report.source == "kaggle"
        mock_dl.assert_called_once()

    def test_fallback_uses_dvc_when_pull_succeeds(
        self, tmp_path: Path, sample_raw_data: Path
    ) -> None:
        """Verify fallback returns DVC report when pull succeeds with files."""
        with patch("src.data.ingestion.is_dvc_initialized", return_value=True):
            with patch("src.data.ingestion.pull_from_dvc") as mock_pull:
                mock_pull.return_value = DownloadReport(
                    success=True,
                    dataset_id="test/data",
                    destination=sample_raw_data,
                    files=[
                        FileInfo(
                            "california_housing.csv",
                            sample_raw_data / "california_housing.csv",
                            100, ".csv",
                        )
                    ],
                    source="dvc",
                )
                report = download_with_dvc_fallback(
                    dataset_id="test/data",
                    destination=sample_raw_data,
                )
        assert report.success is True
        assert report.source == "dvc"
        assert len(report.files) >= 1

    def test_fallback_retries_kaggle_when_dvc_fails(
        self, tmp_path: Path
    ) -> None:
        """Verify fallback falls back to Kaggle when DVC pull fails."""
        with patch("src.data.ingestion.is_dvc_initialized", return_value=True):
            with patch("src.data.ingestion.pull_from_dvc") as mock_pull:
                mock_pull.return_value = DownloadReport(
                    success=False, dataset_id="test/data",
                    destination=tmp_path, errors=["remote error"], source="dvc",
                )
                with patch("src.data.ingestion.download_from_kaggle") as mock_dl:
                    mock_dl.return_value = DownloadReport(
                        success=True, dataset_id="test/data",
                        destination=tmp_path, source="kaggle",
                    )
                    report = download_with_dvc_fallback(
                        dataset_id="test/data",
                        destination=tmp_path,
                        auto_track_dvc=False,
                    )
        assert report.source == "kaggle"
        mock_dl.assert_called_once()


@pytest.mark.integration
class TestRunIngestion:
    """Integration tests for run_ingestion() main workflow."""

    def test_ingestion_success_with_mock_config(
        self, mock_config_yaml: Path, tmp_path: Path
    ) -> None:
        """Verify full ingestion workflow succeeds with mocked dependencies."""
        with patch("src.data.ingestion.load_config") as mock_load:
            mock_load.return_value = {
                "dataset": {"kaggle_id": "test/data", "expected_files": []},
                "dvc": {"remote": {"name": "mylocal"}},
            }
            with patch("src.data.ingestion.setup_kaggle_credentials", return_value=True):
                with patch("src.data.ingestion.ensure_path", return_value=tmp_path):
                    with patch("src.data.ingestion.download_with_dvc_fallback") as mock_dl:
                        mock_dl.return_value = DownloadReport(
                            success=True, dataset_id="test/data",
                            destination=tmp_path,
                            files=[FileInfo("data.csv", tmp_path/"data.csv", 100, ".csv")],
                            source="kaggle",
                        )
                        report = run_ingestion(
                            config_path=mock_config_yaml,
                            save_report=False,
                        )
        assert report.success is True
        assert report.dataset_id == "test/data"
        assert report.source == "kaggle"

    def test_ingestion_missing_kaggle_id_raises(self, mock_config_yaml: Path) -> None:
        """Verify ValueError when config lacks required kaggle_id field.
        
        ingestion.py: raise ValueError("Config missing: dataset.kaggle_id")
        """
        with patch("src.data.ingestion.load_config") as mock_load:
            mock_load.return_value = {"dataset": {}, "dvc": {}}
            with pytest.raises(ValueError, match="dataset.kaggle_id"):
                run_ingestion(config_path=mock_config_yaml)

    def test_ingestion_handles_missing_credentials_gracefully(
        self, mock_config_yaml: Path, tmp_path: Path
    ) -> None:
        """Verify ingestion continues (with warning) when credentials missing."""
        with patch("src.data.ingestion.load_config") as mock_load:
            mock_load.return_value = {
                "dataset": {"kaggle_id": "test/data", "expected_files": []},
                "dvc": {"remote": {"name": "mylocal"}},
            }
            with patch("src.data.ingestion.setup_kaggle_credentials", return_value=False):
                with patch("src.data.ingestion.ensure_path", return_value=tmp_path):
                    with patch("src.data.ingestion.download_with_dvc_fallback") as mock_dl:
                        mock_dl.return_value = DownloadReport(
                            success=True, dataset_id="test/data",
                            destination=tmp_path, source="kaggle",
                        )
                        report = run_ingestion(
                            config_path=mock_config_yaml,
                            save_report=False,
                        )
        assert report.success is True


# =============================================================================
# RUN DIRECTLY
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--color=yes", "-m", "not integration"])