# src/utils/paths.py

"""
Project path management module.

Handles environment detection, project root resolution, and centralized
path configuration for local, Colab, and Google Drive environments.
Integrates with DVC for data versioning support.
"""

import os
import sys
from pathlib import Path
from typing import Optional

from src.utils.logger import get_logger

# Optional YAML support for DVC config parsing
try:
    import yaml

    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    yaml = None  # type: ignore

logger = get_logger(__name__)

# =============================================================================
# ENVIRONMENT & ROOT DETECTION
# =============================================================================

IN_COLAB: bool = os.path.exists("/content")


def get_project_root() -> Path:
    """
    Resolve project root from any execution context.

    Searches upward from this file, with fallbacks for Colab and
    environments without pyproject.toml or .git markers.

    Returns:
        Path object pointing to the project root directory.
    """
    try:
        root = Path(__file__).resolve().parent.parent.parent
    except NameError:
        root = Path.cwd().resolve()

    if not (root / "pyproject.toml").exists() and not (root / ".git").exists():
        if IN_COLAB:
            root = Path("/content/california_housing_full_project")

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    return root


PROJECT_DIR: Path = get_project_root()

# =============================================================================
# PERSISTENT STORAGE CONFIGURATION
# =============================================================================

if IN_COLAB:
    DRIVE_BASE = Path("/content/drive/MyDrive")
else:
    DRIVE_BASE = (
        Path("H:/My Drive") if os.name == "nt" else Path.home() / "GoogleDrive"
    )

DRIVE_PROJECT: Path = DRIVE_BASE / "MLprojects" / "california_housing_full_project"

# =============================================================================
# CENTRALIZED PATH REGISTRY
# =============================================================================

PATHS: dict[str, Path] = {
    # Persistent data storage (Google Drive)
    "raw": DRIVE_PROJECT / "data" / "raw",
    "interim": DRIVE_PROJECT / "data" / "interim",
    "processed": DRIVE_PROJECT / "data" / "processed",
    "models": DRIVE_PROJECT / "models",
    "kaggle_json": DRIVE_BASE / "kaggle.json",
    # Local repository paths (ephemeral in Colab)
    "configs": PROJECT_DIR / "configs",
    "notebooks": PROJECT_DIR / "notebooks",
    "reports": PROJECT_DIR / "reports",
    "src": PROJECT_DIR / "src",
}

# =============================================================================
# DVC INTEGRATION PATHS & HELPERS
# =============================================================================

DVC_CONFIG: Path = PROJECT_DIR / ".dvc" / "config"
DVC_CACHE: Path = PROJECT_DIR / ".dvc" / "cache"
DVC_LOCK: Path = PROJECT_DIR / ".dvc" / "lock"

_DEFAULT_DVC_PATHS: list[Path] = [
    PROJECT_DIR / "data" / "raw",
    PROJECT_DIR / "data" / "interim",
    PROJECT_DIR / "data" / "processed",
    PROJECT_DIR / "models",
]


def is_dvc_initialized() -> bool:
    """
    Check if DVC is properly initialized in this project.

    Returns:
        True if .dvc directory and config file exist.
    """
    return (PROJECT_DIR / ".dvc").exists() and DVC_CONFIG.exists()


def get_dvc_tracked_paths() -> list[Path]:
    """
    Return list of paths configured for DVC tracking.

    Reads from configs/data_config.yaml if available and PyYAML is installed;
    otherwise returns sensible defaults.

    Returns:
        List of Path objects configured for DVC tracking.
    """
    if not _YAML_AVAILABLE:
        logger.debug("PyYAML not available; using default DVC paths")
        return _DEFAULT_DVC_PATHS

    try:
        config_path = PROJECT_DIR / "configs" / "data_config.yaml"
        if not config_path.exists():
            logger.debug(f"DVC config not found: {config_path}; using defaults")
            return _DEFAULT_DVC_PATHS

        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        dvc_paths = config.get("dvc", {}).get("tracked_paths", [])
        if dvc_paths:
            return [PROJECT_DIR / p for p in dvc_paths]

        logger.debug("No 'dvc.tracked_paths' in config; using defaults")
        return _DEFAULT_DVC_PATHS

    except (yaml.YAMLError, KeyError, OSError) as e:
        logger.debug(f"Could not load DVC tracked paths: {type(e).__name__}: {e}")
        return _DEFAULT_DVC_PATHS


# =============================================================================
# PATH UTILITIES
# =============================================================================


def get_path(stage: str, filename: Optional[str] = None) -> Path:
    """
    Retrieve or construct a path for a given data stage.

    Creates parent directories automatically if they do not exist.

    Args:
        stage: Path category key (e.g., 'raw', 'processed').
        filename: Optional filename to append to the directory path.

    Returns:
        Resolved Path object for the target file or directory.
    """
    target_dir = PATHS.get(stage, PROJECT_DIR / "data" / stage)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename if filename else target_dir


def verify_paths() -> None:
    """Print diagnostic summary of all configured paths."""
    separator = "=" * 50
    print(separator)
    print(f"Environment : {'Colab' if IN_COLAB else 'Local'}")
    print(f"Project Root: {PROJECT_DIR}")
    print(f"Drive Base  : {DRIVE_BASE}")
    print(separator)
    for name, path in PATHS.items():
        status = "✓" if path.exists() else "✗"
        print(f"  {status} {name:12} → {path}")
    print(separator)


def ensure_drive_mounted() -> bool:
    """
    Verify Google Drive is accessible in Colab environment.

    Returns:
        True if Drive is mounted or not in Colab; False otherwise.
    """
    if IN_COLAB and not DRIVE_BASE.exists():
        logger.warning("Google Drive not mounted. Execute: drive.mount('/content/drive')")
        return False
    return True


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "PROJECT_DIR",
    "DRIVE_BASE",
    "DRIVE_PROJECT",
    "PATHS",
    "logger",
    "get_path",
    "verify_paths",
    "ensure_drive_mounted",
    "IN_COLAB",
    "DVC_CONFIG",
    "DVC_CACHE",
    "DVC_LOCK",
    "is_dvc_initialized",
    "get_dvc_tracked_paths",
]