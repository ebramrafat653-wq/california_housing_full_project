# src/data/ingestion.py

"""
Data ingestion module — California Housing project.

Workflow:
  1. Load YAML config
  2. Setup Kaggle credentials from Drive
  3. DVC pull (fast path if data already tracked)
  4. Kaggle download fallback (first time or forced refresh)
  5. Auto-track new downloads with DVC (add → push → git commit)
  6. Integrity check against expected_files
  7. Save download report outside data/raw (avoids DVC noise)
"""

import logging
import shutil
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import yaml

from src.utils.logger import get_logger
from src.utils.paths import (
    PATHS, PROJECT_DIR, ensure_path, is_dvc_initialized,
    DVC_REMOTE_NAME, configure_git_identity,
)

logger = get_logger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FileInfo:
    """Metadata for a single file."""

    name: str
    path: Path
    size_bytes: int
    extension: str

    _DATA_EXTENSIONS = {
        ".csv", ".json", ".parquet",
        ".xlsx", ".xls", ".txt", ".tsv", ".pkl", ".feather",
    }

    @property
    def size_mb(self) -> float:
        return round(self.size_bytes / (1024 * 1024), 2)

    @property
    def is_data_file(self) -> bool:
        return self.extension.lower() in self._DATA_EXTENSIONS


@dataclass
class DownloadReport:
    """Full result of a download or DVC-pull operation."""

    success: bool
    dataset_id: str
    destination: Path
    files: List[FileInfo] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    total_size_mb: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None  # "kaggle" | "dvc"

    def summary(self) -> str:
        status = "SUCCESS" if self.success else "FAILED"
        lines = [
            f"\n{'=' * 60}",
            f"  Download Report : {status}",
            f"{'=' * 60}",
            f"  Dataset  : {self.dataset_id}",
            f"  Time     : {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"  Location : {self.destination}",
            f"  Source   : {self.source or 'unknown'}",
            f"  Files    : {len(self.files)}",
            f"  Size     : {self.total_size_mb:.2f} MB",
        ]
        if self.files:
            lines.append("\n  Files:")
            for f in self.files:
                tag = "+" if f.is_data_file else "-"
                lines.append(f"    {tag} {f.name} ({f.size_mb:.2f} MB)")
        if self.errors:
            lines.append("\n  Errors:")
            lines.extend(f"    ! {e}" for e in self.errors)
        if self.warnings:
            lines.append("\n  Warnings:")
            lines.extend(f"    * {w}" for w in self.warnings)
        lines.append(f"{'=' * 60}\n")
        return "\n".join(lines)


# =============================================================================
# CONFIGURATION
# =============================================================================

def load_config(config_path: Optional[Path] = None) -> dict:
    """Load and parse the project YAML config file."""
    target = config_path or (PATHS["configs"] / "data_config.yaml")
    if not target.exists():
        raise FileNotFoundError(f"Config not found: {target}")
    with open(target, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    logger.info(f"Config loaded: {target}")
    return config


# =============================================================================
# KAGGLE CREDENTIALS
# =============================================================================

def setup_kaggle_credentials(credentials_path: Optional[Path] = None) -> bool:
    """
    Copy kaggle.json from Drive to ~/.kaggle/ with correct permissions.
    Expected: MyDrive/kaggle.json
    """
    source = credentials_path or PATHS["kaggle_json"]
    if not source.exists():
        logger.error(f"kaggle.json not found: {source}")
        logger.info("Get it from: https://www.kaggle.com/settings → API")
        return False

    dest_dir = Path.home() / ".kaggle"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "kaggle.json"

    try:
        shutil.copy2(source, dest)
        dest.chmod(0o600)
        logger.info(f"Kaggle credentials installed: {dest}")
        return True
    except Exception as e:
        logger.error(f"Failed to install Kaggle credentials: {e}")
        return False


# =============================================================================
# KAGGLE DOWNLOAD
# =============================================================================

def _run_kaggle(args: List[str], timeout: int = 300) -> None:
    """Run kaggle CLI command; raise RuntimeError on failure."""
    cmd = ["kaggle"] + args
    logger.debug(f"Running: {' '.join(cmd)}")
    r = subprocess.run(
        cmd, capture_output=True, text=True,
        errors="replace", timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"kaggle CLI error: {r.stderr.strip()}")


def _scan_directory(directory: Path, skip_zip: bool = True) -> tuple[List[FileInfo], float]:
    """Scan directory; return (FileInfo list, total_mb)."""
    infos: List[FileInfo] = []
    total = 0.0
    for fp in directory.iterdir():
        if not fp.is_file():
            continue
        if skip_zip and fp.suffix.lower() == ".zip":
            continue
        try:
            info = FileInfo(
                name=fp.name, path=fp,
                size_bytes=fp.stat().st_size,
                extension=fp.suffix,
            )
            infos.append(info)
            total += info.size_mb
        except OSError:
            continue
    return infos, total


def download_from_kaggle(
    dataset_id: str,
    destination: Path,
    unzip: bool = True,
    force: bool = False,
    timeout: int = 300,
) -> DownloadReport:
    """Download a Kaggle dataset to `destination`."""
    destination.mkdir(parents=True, exist_ok=True)
    before = {f.name for f in destination.iterdir() if f.is_file()}

    logger.info(f"Downloading from Kaggle: {dataset_id} → {destination}")

    args = ["datasets", "download", "-d", dataset_id, "-p", str(destination)]
    if unzip:
        args.append("--unzip")
    if force:
        args.append("--force")

    try:
        _run_kaggle(args, timeout=timeout)
    except RuntimeError as e:
        return DownloadReport(
            success=False, dataset_id=dataset_id,
            destination=destination, errors=[str(e)], source="kaggle",
        )

    after    = {f.name for f in destination.iterdir() if f.is_file()}
    new_names = after - before
    infos: List[FileInfo] = []
    total = 0.0

    for name in new_names:
        if name.endswith(".zip") and unzip:
            continue
        fp = destination / name
        try:
            info = FileInfo(
                name=name, path=fp,
                size_bytes=fp.stat().st_size,
                extension=fp.suffix,
            )
            infos.append(info)
            total += info.size_mb
        except OSError as e:
            logger.warning(f"Cannot stat {name}: {e}")

    logger.info(f"Kaggle download: {len(infos)} file(s), {total:.2f} MB")
    return DownloadReport(
        success=True, dataset_id=dataset_id,
        destination=destination, files=infos,
        total_size_mb=total, source="kaggle",
    )


# =============================================================================
# DVC — GIT IDENTITY (shared helper)
# =============================================================================

# configure_git_identity imported from paths.py (single source of truth)


# =============================================================================
# DVC TRACKING
# =============================================================================

def track_with_dvc(
    file_path: Path,
    remote_name: str = DVC_REMOTE_NAME,
    timeout: int = 600,
) -> bool:
    """
    Track file_path with DVC, push content to Drive remote,
    commit .dvc pointer file to GitHub.
    file_path must be inside PROJECT_DIR.
    """
    try:
        rel = file_path.relative_to(PROJECT_DIR)
    except ValueError:
        logger.error(
            f"{file_path} is outside PROJECT_DIR ({PROJECT_DIR}).\n"
            "Ensure setup_drive_symlinks() ran before ingestion."
        )
        return False

    # dvc add
    logger.info(f"DVC add: {rel}")
    add = subprocess.run(
        ["dvc", "add", str(rel)],
        cwd=PROJECT_DIR, capture_output=True,
        text=True, errors="replace", timeout=timeout,
    )
    if add.returncode != 0:
        logger.error(f"dvc add failed:\n{add.stderr}")
        return False

    # dvc push
    logger.info(f"DVC push → {remote_name}")
    push = subprocess.run(
        ["dvc", "push", f"--remote={remote_name}", str(rel)],
        cwd=PROJECT_DIR, capture_output=True,
        text=True, errors="replace", timeout=timeout,
    )
    if push.returncode != 0:
        logger.warning(f"dvc push warnings:\n{push.stderr.strip()}")

    # git: add .dvc pointer + .gitignore → commit → push
    _ensure_git_identity(PROJECT_DIR)
    subprocess.run(
        ["git", "add", f"{rel}.dvc", str(rel.parent / ".gitignore")],
        cwd=PROJECT_DIR, capture_output=True,
        errors="replace", timeout=30,
    )

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_DIR, capture_output=True,
        text=True, errors="replace", timeout=15,
    )
    if not status.stdout.strip():
        logger.info("Data already tracked (no new git changes)")
        return True

    subprocess.run(
        ["git", "commit", "-m", f"data: track {rel} with DVC"],
        cwd=PROJECT_DIR, capture_output=True,
        errors="replace", timeout=60,
    )
    git_push = subprocess.run(
        ["git", "push"],
        cwd=PROJECT_DIR, capture_output=True,
        text=True, errors="replace", timeout=120,
    )
    if git_push.returncode != 0:
        logger.error(f"git push failed: {git_push.stderr.strip()}")
        return False

    logger.info(f"✅ DVC tracked + .dvc pointer saved to GitHub: {rel}")
    return True


# =============================================================================
# DVC PULL
# =============================================================================

def pull_from_dvc(
    target_path: Path,
    remote_name: str = DVC_REMOTE_NAME,
    force: bool = False,
    dataset_id: Optional[str] = None,
    timeout: int = 300,
) -> DownloadReport:
    """Pull a specific path from the DVC remote."""
    try:
        rel = target_path.relative_to(PROJECT_DIR)
    except ValueError:
        return DownloadReport(
            success=False,
            dataset_id=dataset_id or "unknown",
            destination=target_path,
            errors=[f"{target_path} is outside PROJECT_DIR"],
            source="dvc",
        )

    cmd = ["dvc", "pull", f"--remote={remote_name}", str(rel)]
    if force:
        cmd.append("--force")

    logger.info(f"DVC pull: {rel}")
    r = subprocess.run(
        cmd, cwd=PROJECT_DIR, capture_output=True,
        text=True, errors="replace", timeout=timeout,
    )

    if r.returncode != 0:
        return DownloadReport(
            success=False,
            dataset_id=dataset_id or "unknown",
            destination=target_path,
            errors=[r.stderr.strip()],
            source="dvc",
        )

    infos, total = (
        _scan_directory(target_path)
        if target_path.exists() else ([], 0.0)
    )
    return DownloadReport(
        success=True,
        dataset_id=dataset_id or "unknown",
        destination=target_path,
        files=infos, total_size_mb=total,
        source="dvc",
    )


# =============================================================================
# ORCHESTRATOR: DVC-first with Kaggle fallback
# =============================================================================

def download_with_dvc_fallback(
    dataset_id: str,
    destination: Path,
    remote_name: str = DVC_REMOTE_NAME,
    unzip: bool = True,
    force: bool = False,
    timeout: int = 300,
    auto_track_dvc: bool = True,
) -> DownloadReport:
    """
    1. DVC pull (fast — no internet if data is cached on Drive)
    2. Kaggle download (fallback — first time or forced)
    3. DVC track + push new downloads (if auto_track_dvc=True)
    """
    if is_dvc_initialized():
        logger.info("DVC initialized — trying pull first…")
        report = pull_from_dvc(
            destination,
            remote_name=remote_name,
            force=force,
            dataset_id=dataset_id,
            timeout=timeout,
        )
        if report.success and report.files:
            logger.info(f"✅ Data ready via DVC ({len(report.files)} file(s))")
            return report
        if report.success:
            logger.warning("DVC pull succeeded but no files found — falling back to Kaggle")
        else:
            logger.warning(f"DVC pull failed ({report.errors}) — falling back to Kaggle")

    report = download_from_kaggle(
        dataset_id=dataset_id,
        destination=destination,
        unzip=unzip, force=force, timeout=timeout,
    )

    if report.success and is_dvc_initialized() and auto_track_dvc:
        logger.info("Tracking new data with DVC…")
        ok = track_with_dvc(destination, remote_name=remote_name, timeout=timeout)
        if ok:
            logger.info("✅ Future sessions will use DVC pull (no Kaggle needed)")
        else:
            report.warnings.append(
                "DVC tracking failed — data downloaded but not versioned yet"
            )
    elif report.success and is_dvc_initialized() and not auto_track_dvc:
        logger.info("auto_track_dvc=False — skipping DVC tracking")

    return report


# =============================================================================
# INTEGRITY CHECK
# =============================================================================

def verify_download_integrity(
    directory: Path,
    expected_files: Optional[List[str]] = None,
) -> dict:
    """
    Verify files exist and expected list is satisfied.
    Returns summary with integrity_ok flag.
    """
    if not directory.exists():
        return {"error": f"Directory not found: {directory}", "integrity_ok": False}

    infos, total_mb = _scan_directory(directory, skip_zip=False)
    data_files = [f for f in infos if f.is_data_file]
    all_names  = [f.name for f in infos]

    summary: dict = {
        "total_files":     len(infos),
        "data_files":      len(data_files),
        "total_size_mb":   round(total_mb, 2),
        "data_file_names": [f.name for f in data_files],
        "integrity_ok":    True,
    }

    if expected_files:
        missing = [f for f in expected_files if f not in all_names]
        summary["missing_expected_files"] = missing
        if missing:
            summary["integrity_ok"] = False
            logger.error(f"Integrity FAILED — missing: {missing}")
        else:
            logger.info("✅ Integrity check passed")

    logger.info(
        f"Integrity: {summary['total_files']} file(s), "
        f"{summary['total_size_mb']:.2f} MB, ok={summary['integrity_ok']}"
    )
    return summary


# =============================================================================
# REPORT — saved outside data/raw to avoid DVC tracking noise
# =============================================================================

def save_download_report(
    report: DownloadReport,
    output_path: Optional[Path] = None,
) -> Path:
    """
    Save download report to reports/downloads/ (NOT inside data/raw).
    Keeping reports outside DVC-tracked directories prevents pointer-file noise.
    """
    target = output_path or (
        PATHS["reports"] / "downloads"
        / f"{report.dataset_id.replace('/', '_')}_{report.timestamp.strftime('%Y%m%d_%H%M%S')}.txt"
    )
    target.parent.mkdir(parents=True, exist_ok=True)

    with open(target, "w", encoding="utf-8") as f:
        f.write(report.summary())

    logger.info(f"Report saved: {target}")
    return target


# =============================================================================
# MAIN PIPELINE ENTRY POINT
# =============================================================================

def run_ingestion(
    config_path: Optional[Path] = None,
    credentials_path: Optional[Path] = None,
    unzip: bool = True,
    force: bool = False,
    save_report: bool = True,
    auto_track_dvc: bool = True,
    timeout: int = 300,
) -> DownloadReport:
    """
    Full ingestion pipeline: config → credentials → download → verify → report.

    Usage:
        from src.data.ingestion import run_ingestion
        report = run_ingestion()
        print(report.summary())
    """
    logger.info("─" * 60)
    logger.info("  Data ingestion started")
    logger.info("─" * 60)

    config      = load_config(config_path)
    dataset_cfg = config.get("dataset", {})
    dataset_id  = dataset_cfg.get("kaggle_id")
    expected    = dataset_cfg.get("expected_files", [])
    remote_name = DVC_REMOTE_NAME  # Single source of truth from paths.py

    if not dataset_id:
        raise ValueError("Config missing: dataset.kaggle_id")

    setup_kaggle_credentials(credentials_path)

    destination = ensure_path("raw")
    report = download_with_dvc_fallback(
        dataset_id=dataset_id,
        destination=destination,
        remote_name=remote_name,
        unzip=unzip, force=force,
        timeout=timeout,
        auto_track_dvc=auto_track_dvc,
    )

    if report.success:
        integrity = verify_download_integrity(destination, expected)
        if not integrity.get("integrity_ok", True):
            report.warnings.append(
                f"Missing expected files: {integrity.get('missing_expected_files')}"
            )
        if save_report:
            save_download_report(report)
        logger.info(f"✅ Ingestion complete — source: {report.source}")
    else:
        logger.error(f"❌ Ingestion failed: {report.errors}")

    return report


# =============================================================================
# CLI
# =============================================================================

def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Ingest California Housing dataset (Kaggle → DVC → Drive)"
    )
    parser.add_argument("--config",      default=None)
    parser.add_argument("--credentials", default=None)
    parser.add_argument("--no-unzip",    action="store_true")
    parser.add_argument("--force",       action="store_true")
    parser.add_argument("--no-report",   action="store_true")
    parser.add_argument("--no-dvc",      action="store_true")
    parser.add_argument("--timeout",     type=int, default=300)
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    from src.utils.logger import setup_logging
    setup_logging(level=getattr(logging, args.log_level.upper(), logging.INFO))

    try:
        report = run_ingestion(
            config_path=Path(args.config) if args.config else None,
            credentials_path=Path(args.credentials) if args.credentials else None,
            unzip=not args.no_unzip,
            force=args.force,
            save_report=not args.no_report,
            auto_track_dvc=not args.no_dvc,
            timeout=args.timeout,
        )
        print(report.summary())
        return 0 if report.success else 1
    except Exception as e:
        logger.error(f"Fatal: {type(e).__name__}: {e}")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())


__all__ = [
    "FileInfo", "DownloadReport",
    "load_config", "setup_kaggle_credentials",
    "download_from_kaggle", "track_with_dvc", "pull_from_dvc",
    "download_with_dvc_fallback", "verify_download_integrity",
    "save_download_report", "run_ingestion", "main",
]