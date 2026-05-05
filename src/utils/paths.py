# src/utils/paths.py
import os
import sys
import logging
from pathlib import Path

# ============================================================
# 1. ENVIRONMENT DETECTION
# ============================================================
IN_COLAB = os.path.exists('/content')

# ============================================================
# 2. PROJECT ROOT (finds itself from anywhere)
# ============================================================
def get_project_root() -> Path:
    try:
        root = Path(__file__).resolve().parent.parent.parent
    except NameError:
        root = Path(os.getcwd()).resolve()
    
    # Confirm this is actually the root
    if not (root / 'pyproject.toml').exists() and not (root / '.git').exists():
        if IN_COLAB:
            root = Path('/content/california_housing_full_project')
    
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    
    return root

PROJECT_DIR = get_project_root()

# ============================================================
# 3. DRIVE PATHS (persistent data)
# ============================================================
if IN_COLAB:
    DRIVE_BASE = Path('/content/drive/MyDrive')
else:
    DRIVE_BASE = Path('H:/My Drive') if os.name == 'nt' else Path.home() / 'GoogleDrive'

# All data on Drive
DRIVE_PROJECT = DRIVE_BASE / 'MLprojects' / 'california_housing'

# ============================================================
# 4. ALL PATHS IN ONE PLACE
# ============================================================
PATHS = {
    # ---- Data on Drive (persistent) ----
    "raw":           DRIVE_PROJECT / "data" / "raw",
    "interim":       DRIVE_PROJECT / "data" / "interim",
    "processed":     DRIVE_PROJECT / "data" / "processed",
    "models":        DRIVE_PROJECT / "models",
    "kaggle_json":   DRIVE_BASE / "kaggle.json",

    # ---- Files inside repo (on Colab) ----
    "configs":       PROJECT_DIR / "configs",
    "notebooks":     PROJECT_DIR / "notebooks",
    "reports":       PROJECT_DIR / "reports",
    "src":           PROJECT_DIR / "src",
}

# ============================================================
# 5. LOGGING
# ============================================================
def setup_logger(name=__name__):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name)

logger = setup_logger()

# ============================================================
# 6. SMART PATH GETTER
# ============================================================
def get_path(stage: str, filename: str = None) -> Path:
    """
    Returns path and creates directory automatically.
    Usage: get_path("raw") or get_path("raw", "housing.csv")
    """
    target_dir = PATHS.get(stage, PROJECT_DIR / 'data' / stage)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename if filename else target_dir

# ============================================================
# 7. QUICK VERIFICATION
# ============================================================
def verify_paths():
    """Print all paths for confirmation"""
    print(f"{'='*50}")
    print(f"Environment : {'Colab' if IN_COLAB else 'Local'}")
    print(f"Project Root: {PROJECT_DIR}")
    print(f"Drive Base  : {DRIVE_BASE}")
    print(f"{'='*50}")
    for name, path in PATHS.items():
        exists = "✅" if path.exists() else "⚠️ "
        print(f"  {exists} {name:12} → {path}")
    print(f"{'='*50}")

__all__ = ['PROJECT_DIR', 'DRIVE_BASE', 'PATHS', 'logger', 'get_path', 'IN_COLAB', 'verify_paths']