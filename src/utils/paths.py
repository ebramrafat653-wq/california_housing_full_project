import os
import sys
import logging
from pathlib import Path

# --- 1. SMART ENVIRONMENT DETECTION ---
# Detect environment more accurately
IN_COLAB = 'google.colab' in sys.modules

# --- 2. DYNAMIC PROJECT ROOT DETECTION ---
def get_project_root() -> Path:
    """
    Dynamically determines the project root.
    If in Colab: uses the current working directory.
    If running as a script: uses the script's file path.
    """
    try:
        # If running as a .py script
        return Path(__file__).resolve().parent.parent.parent
    except NameError:
        # If running in a Notebook (Colab/Jupyter)
        return Path(os.getcwd()).resolve()

PROJECT_DIR = get_project_root()

# --- 3. AUTO-PATH INJECTION (The Secret Sauce) ---
# Automatically add project root and src folder to sys.path 
# so imports work from anywhere
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

# --- 4. BASE DRIVE PATH LOGIC ---
if IN_COLAB:
    BASE_DRIVE_PATH = Path('/content/drive/MyDrive')
else:
    # Automatically detects Windows and macOS
    BASE_DRIVE_PATH = Path('H:/My Drive') if os.name == 'nt' else Path.home() / 'GoogleDrive'

# --- 5. STANDARDIZED DATA STRUCTURE ---
# Link data to drive to ensure persistence
DATA_BASE = BASE_DRIVE_PATH / 'MLprojects' / 'california_housing'
PATHS = {
    "raw": DATA_BASE / "data/raw",
    "interim": DATA_BASE / "data/interim",
    "processed": DATA_BASE / "data/processed",
    "models": DATA_BASE / "models",
    "configs": PROJECT_DIR / "configs"  # Configs usually stay inside the repo itself
}

# --- 6. LOGGING CONFIGURATION ---
def setup_logger(name=__name__):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(name)

logger = setup_logger()

# --- 7. SMART DIRECTORY CREATOR ---
def get_path(stage: str, filename: str = None) -> Path:
    """
    Returns the path and automatically creates the directory if it doesn't exist.
    """
    target_dir = PATHS.get(stage, PROJECT_DIR / 'data' / stage)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir / filename if filename else target_dir

__all__ = ['PROJECT_DIR', 'logger', 'get_path', 'IN_COLAB', 'PATHS']