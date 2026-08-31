# src/utils/paths.py

"""
Project path management module.

ARCHITECTURE:
  - All paths (data, models, artifacts) reside INSIDE PROJECT_DIR.
  - DVC handles pulling data from the remote (Google Drive) to these local paths.
  - No symlinks are used. DVC cache is local (.dvc/cache).
  - DVC remote is configured locally in .dvc/config.local (ignored by Git).
"""

import os
import platform
import subprocess
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
# DATA & MODEL ROOTS
# =============================================================================

DATA_ROOT:   Path = PROJECT_DIR / "data"
MODELS_ROOT: Path = PROJECT_DIR / "models"
ARTIFACTS_ROOT: Path = PROJECT_DIR / "artifacts"

# =============================================================================
# CENTRALIZED PATH REGISTRY
# =============================================================================

PATHS: dict[str, Path] = {
    "raw":        DATA_ROOT  / "raw",
    "interim":    DATA_ROOT  / "interim",
    "processed":  DATA_ROOT  / "processed",
    "models":     MODELS_ROOT,
    "artifacts":  ARTIFACTS_ROOT,
    "kaggle_json": DRIVE_BASE / "kaggle.json",
    "configs":    PROJECT_DIR / "configs",
    "notebooks":  PROJECT_DIR / "notebooks",
    "reports":    PROJECT_DIR / "reports",
    "src":        PROJECT_DIR / "src",
}

# DVC storage on Drive — used as the local DVC remote (configured in .dvc/config.local)
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
    """
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
    print(f"  Data root    : {DATA_ROOT}")
    print(f"  DVC remote   : {DVC_REMOTE_NAME}  →  {DVC_STORAGE}")
    print(f"  DVC ready    : {is_dvc_initialized()}")
    print(sep)
    for name, path in PATHS.items():
        ok  = "✓" if path.exists() else "✗"
        print(f"  {ok}  {name:<14} {path}")
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
    "PROJECT_DIR", "DRIVE_BASE", "DATA_ROOT", "MODELS_ROOT", "ARTIFACTS_ROOT",
    "PATHS", "DVC_STORAGE", "DVC_CONFIG", "DVC_CACHE",
    # Functions
    "is_dvc_initialized", "get_dvc_tracked_paths",
    "get_path", "ensure_path", "verify_paths", "ensure_drive_mounted",
    "configure_git_identity",
    "logger",
]