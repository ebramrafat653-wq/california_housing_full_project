# src/utils/paths.py

"""
Project path management module.

ARCHITECTURE:
  - DATA_ROOT / MODELS_ROOT always reside INSIDE PROJECT_DIR → DVC tracks them normally.
  - In Colab: symlinks transparently redirect those paths to Google Drive for persistence.
  - Drive must be mounted BEFORE this module is imported in Colab.
    Call setup_drive_symlinks() explicitly after mount_drive().
"""

import os
import platform
import shutil
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    yaml = None

logger = get_logger(__name__)

# =============================================================================
# CONSTANTS
# =============================================================================

PROJECT_NAME: str = "california_housing_full_project"

# Reliable Colab detection (env var set by Colab runtime)
IN_COLAB: bool = "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content/drive")

# Name used for DVC remote — single source of truth across all modules
DVC_REMOTE_NAME: str = "mylocal"

# =============================================================================
# PROJECT ROOT
# =============================================================================

def get_project_root() -> Path:
    """Resolve project root from __file__, Colab paths, or cwd — in that order."""
    try:
        candidate = Path(__file__).resolve().parent.parent.parent
        if (candidate / ".git").exists() or (candidate / "pyproject.toml").exists():
            return candidate
    except NameError:
        pass

    if IN_COLAB:
        for p in [Path(f"/content/{PROJECT_NAME}"), Path("/content")]:
            if p.exists() and (p / "src").exists():
                return p

    return Path.cwd().resolve()


PROJECT_DIR: Path = get_project_root()

# =============================================================================
# GOOGLE DRIVE BASE PATH
# =============================================================================

def _resolve_drive_base() -> Path:
    if IN_COLAB:
        return Path("/content/drive/MyDrive")

    env = os.environ.get("GOOGLE_DRIVE_PATH")
    if env:
        return Path(env)

    system = platform.system()
    home = Path.home()

    if system == "Windows":
        for p in ["G:/My Drive", "F:/My Drive", "E:/My Drive", "D:/My Drive"]:
            if Path(p).exists():
                logger.debug(f"Windows Drive resolved: {p}")
                return Path(p)
        return home / "GoogleDrive"

    if system == "Darwin":
        return home / "Library/CloudStorage/GoogleDrive-My Drive"

    return home / "GoogleDrive"


DRIVE_BASE: Path = _resolve_drive_base()

# =============================================================================
# DATA & MODEL ROOTS — always inside PROJECT_DIR (DVC-safe)
# =============================================================================

DATA_ROOT:   Path = PROJECT_DIR / "data"
MODELS_ROOT: Path = PROJECT_DIR / "models"

# =============================================================================
# DRIVE SYMLINKS (Colab persistence)
# =============================================================================

def _make_symlink(local: Path, target: Path) -> None:
    """
    Create symlink local → target, removing empty git placeholders first.
    No-op if symlink already points to the correct target.
    """
    if not DRIVE_BASE.exists():
        logger.error(
            "Google Drive not mounted — cannot create symlink. "
            "Call mount_drive() before setup_drive_symlinks()."
        )
        return

    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Cannot create Drive directory {target}: {e}")
        return

    if local.is_symlink():
        if local.resolve() == target.resolve():
            logger.debug(f"Symlink already correct: {local}")
            return
        local.unlink()
        logger.debug(f"Removed stale symlink: {local}")

    elif local.exists():
        try:
            shutil.rmtree(local)
            logger.debug(f"Removed empty git placeholder: {local}")
        except OSError as e:
            logger.error(f"Cannot remove {local} to create symlink: {e}")
            return

    try:
        local.symlink_to(target, target_is_directory=True)
        logger.info(f"Symlink created: {local} → {target}")
    except OSError as e:
        logger.error(f"Cannot create symlink {local} → {target}: {e}")


def setup_drive_symlinks() -> None:
    """
    Redirect DATA_ROOT and MODELS_ROOT to Google Drive via symlinks.

    Must be called AFTER mount_drive() in Colab.
    Safe to call multiple times (idempotent).
    """
    if not IN_COLAB:
        logger.debug("Not in Colab — skipping Drive symlinks")
        return

    _make_symlink(DATA_ROOT,   DRIVE_BASE / PROJECT_NAME / "data")
    _make_symlink(MODELS_ROOT, DRIVE_BASE / PROJECT_NAME / "models")

# =============================================================================
# CENTRALIZED PATH REGISTRY
# =============================================================================

PATHS: dict[str, Path] = {
    "raw":        DATA_ROOT  / "raw",
    "interim":    DATA_ROOT  / "interim",
    "processed":  DATA_ROOT  / "processed",
    "models":     MODELS_ROOT,
    "kaggle_json": DRIVE_BASE / "kaggle.json",
    "configs":    PROJECT_DIR / "configs",
    "notebooks":  PROJECT_DIR / "notebooks",
    "reports":    PROJECT_DIR / "reports",
    "src":        PROJECT_DIR / "src",
}

# DVC storage on Drive — used as the local DVC remote
DVC_STORAGE: Path = DRIVE_BASE / "dvc_storage"

# =============================================================================
# DVC HELPERS
# =============================================================================

DVC_CONFIG: Path = PROJECT_DIR / ".dvc" / "config"
DVC_CACHE:  Path = PROJECT_DIR / ".dvc" / "cache"

_DEFAULT_DVC_PATHS: list[Path] = [
    PATHS["raw"], PATHS["interim"], PATHS["processed"], PATHS["models"],
]


def is_dvc_initialized() -> bool:
    """Return True if .dvc/ exists and has a config file."""
    return (PROJECT_DIR / ".dvc").exists() and DVC_CONFIG.exists()


def get_dvc_tracked_paths() -> list[Path]:
    """Return DVC-tracked paths from config YAML, falling back to defaults."""
    if not _YAML_AVAILABLE:
        return _DEFAULT_DVC_PATHS
    try:
        cfg = PATHS["configs"] / "data_config.yaml"
        if not cfg.exists():
            return _DEFAULT_DVC_PATHS
        with open(cfg, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        raw = data.get("dvc", {}).get("tracked_paths", [])
        return [PROJECT_DIR / Path(p) for p in raw] if raw else _DEFAULT_DVC_PATHS
    except Exception as e:
        logger.warning(f"get_dvc_tracked_paths failed: {e}")
        return _DEFAULT_DVC_PATHS

# =============================================================================
# GIT CONFIGURATION
# =============================================================================

def configure_git_identity(cwd: Optional[Path] = None) -> None:
    """
    Configure git user identity (prevents commit errors in fresh Colab sessions).
    Single source of truth — imported and used across colab_setup.py and ingestion.py.
    """
    import subprocess
    cwd = cwd or PROJECT_DIR
    for key, val in [("user.name", "Colab Bot"), ("user.email", "colab@bot.local")]:
        subprocess.run(
            ["git", "config", key, val],
            cwd=str(cwd), capture_output=True, text=True, errors="replace", timeout=15,
        )

# =============================================================================
# PATH UTILITIES
# =============================================================================

def get_path(stage: str, filename: Optional[str] = None) -> Path:
    """Return registered path without creating directories."""
    base = PATHS.get(stage, DATA_ROOT / stage)
    return base / filename if filename else base


def ensure_path(stage: str, filename: Optional[str] = None) -> Path:
    """Return registered path AND create directories if missing."""
    base = PATHS.get(stage, DATA_ROOT / stage)
    base.mkdir(parents=True, exist_ok=True)
    return base / filename if filename else base


def verify_paths() -> None:
    """Print a diagnostic table of all registered paths."""
    sep = "=" * 60
    print(sep)
    print(f"  Environment  : {'Google Colab' if IN_COLAB else 'Local'}")
    print(f"  Project root : {PROJECT_DIR}")
    print(f"  Drive base   : {DRIVE_BASE}")
    print(f"  Data root    : {DATA_ROOT}"
          + (f"  →  {DATA_ROOT.resolve()}" if DATA_ROOT.is_symlink() else ""))
    print(f"  DVC remote   : {DVC_REMOTE_NAME}  →  {DVC_STORAGE}")
    print(f"  DVC ready    : {is_dvc_initialized()}")
    print(sep)
    for name, path in PATHS.items():
        ok  = "✓" if path.exists() else "✗"
        sym = "  [→ Drive]"  if path.is_symlink() else ""
        print(f"  {ok}  {name:<14} {path}{sym}")
    print(sep)


def ensure_drive_mounted() -> bool:
    """Verify Drive is accessible (Colab only)."""
    if IN_COLAB and not DRIVE_BASE.exists():
        logger.warning("Drive not mounted — run: drive.mount('/content/drive')")
        return False
    return True


__all__ = [
    # Constants
    "PROJECT_NAME", "IN_COLAB", "DVC_REMOTE_NAME",
    # Paths
    "PROJECT_DIR", "DRIVE_BASE", "DATA_ROOT", "MODELS_ROOT",
    "PATHS", "DVC_STORAGE", "DVC_CONFIG", "DVC_CACHE",
    # Functions
    "setup_drive_symlinks", "is_dvc_initialized", "get_dvc_tracked_paths",
    "get_path", "ensure_path", "verify_paths", "ensure_drive_mounted",
    "configure_git_identity",
    "logger",
]