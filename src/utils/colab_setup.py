# src/utils/colab_setup.py
"""
Colab environment initialization module.

Provides a single entry point for setting up Google Colab runtime:
- Mount Google Drive for persistent storage
- Configure SSH keys for private repository access
- Clone or update the project repository
- Initialize DVC with Google Drive remote (optional)
- Configure Python path and working directory

This module is designed to be idempotent and safe to re-run.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================================
# UTILITIES
# ============================================================================

def _is_colab() -> bool:
    """Check if running in Google Colab environment."""
    return os.path.exists("/content")


def _run_command(cmd: str, cwd: Optional[Path] = None, silent: bool = False) -> bool:
    """
    Execute a shell command with error handling.

    Args:
        cmd: Shell command to execute.
        cwd: Optional working directory for command execution.
        silent: If True, suppress command output in logs.

    Returns:
        True if command succeeded, False otherwise.
    """
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            check=True,
        )
        if not silent:
            logger.debug(f"Command succeeded: {cmd}")
        return True
    except subprocess.CalledProcessError as e:
        if not silent:
            logger.error(f"Command failed: {cmd}\nstderr: {e.stderr}")
        return False


# ============================================================================
# CORE SETUP FUNCTIONS
# ============================================================================

def mount_drive() -> Path:
    """
    Mount Google Drive if not already mounted.

    Returns:
        Path to the mounted Drive base directory.

    Raises:
        RuntimeError: If running outside Colab and Drive cannot be mounted.
    """
    drive_base = Path("/content/drive/MyDrive")

    if not _is_colab():
        logger.warning("Not running in Colab; skipping Drive mount")
        return drive_base

    if drive_base.exists():
        logger.info("Google Drive already mounted")
        return drive_base

    try:
        from google.colab import drive  # type: ignore

        drive.mount("/content/drive")
        logger.info("Google Drive mounted successfully")
        return drive_base
    except ImportError:
        logger.error("google.colab not available; cannot mount Drive")
        raise RuntimeError("Mount Drive requires Google Colab environment")
    except Exception as e:
        logger.error(f"Failed to mount Drive: {e}")
        raise


def configure_ssh(ssh_key_path: Optional[str] = None) -> bool:
    """
    Configure SSH keys for GitHub authentication.

    Args:
        ssh_key_path: Optional custom path to private key on Drive.
                    Defaults to /content/drive/MyDrive/ssh_config/housing_key

    Returns:
        True if SSH configured successfully, False otherwise.
    """
    if not _is_colab():
        logger.debug("Skipping SSH config outside Colab")
        return True

    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)

    key_source = Path(ssh_key_path) if ssh_key_path else (
        Path("/content/drive/MyDrive/ssh_config/housing_key")
    )
    key_dest = ssh_dir / "id_rsa"

    if not key_source.exists():
        logger.warning(f"SSH key not found: {key_source}")
        logger.info("Falling back to HTTPS for git operations")
        return False

    try:
        shutil.copy2(key_source, key_dest)
        key_dest.chmod(0o600)

        # Add GitHub to known_hosts to prevent interactive prompt
        _run_command("ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null", silent=True)

        logger.info("SSH keys configured successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to configure SSH: {e}")
        return False


def clone_or_update_repo(
    repo_url: str,
    target_path: Path,
    use_ssh: bool = True,
) -> bool:
    """
    Clone repository if absent, or pull latest changes if present.

    Args:
        repo_url: HTTPS or SSH URL of the repository.
        target_path: Local path where repository should reside.
        use_ssh: If True, convert HTTPS URL to SSH format for auth.

    Returns:
        True if operation succeeded, False otherwise.
    """
    if target_path.exists():
        logger.info(f"Repository exists at {target_path}; pulling updates...")
        success = _run_command("git pull --rebase", cwd=target_path)
        if success:
            logger.info("Repository updated successfully")
        return success

    logger.info(f"Cloning repository to {target_path}...")
    
    # Convert HTTPS to SSH if needed for authentication
    git_url = repo_url
    if use_ssh and "github.com" in repo_url and "git@github.com" not in repo_url:
        parts = repo_url.rstrip(".git").replace("https://github.com/", "")
        git_url = f"git@github.com:{parts}.git"
        logger.debug(f"Using SSH URL: {git_url}")

    success = _run_command(f"git clone {git_url} {target_path}")
    if success:
        logger.info(f"Repository cloned successfully to {target_path}")
    return success


def configure_python_path(repo_path: Path) -> None:
    """
    Add repository to sys.path for module imports.

    Args:
        repo_path: Path to the project root directory.
    """
    repo_str = str(repo_path.resolve())
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
        logger.debug(f"Added to sys.path: {repo_str}")


def set_working_directory(repo_path: Path) -> None:
    """
    Change current working directory to project root.

    Args:
        repo_path: Path to the project root directory.
    """
    os.chdir(repo_path)
    logger.debug(f"Working directory set to: {repo_path}")


def install_dependencies(requirements_file: Optional[str] = None) -> bool:
    """
    Install Python dependencies from requirements file.

    Args:
        requirements_file: Path to requirements.txt relative to repo root.
                        Defaults to 'requirements.txt' in repo root.

    Returns:
        True if installation succeeded or was skipped, False on error.
    """
    if not _is_colab():
        logger.debug("Skipping dependency installation outside Colab")
        return True

    req_path = Path(requirements_file) if requirements_file else Path("requirements.txt")
    
    if not req_path.exists():
        logger.warning(f"Requirements file not found: {req_path}; skipping installation")
        return True

    logger.info(f"Installing dependencies from {req_path}...")
    success = _run_command(f"pip install -q -r {req_path}")
    
    if success:
        logger.info("Dependencies installed successfully")
    return success


# ============================================================================
# DVC INTEGRATION (Google Drive Remote)
# ============================================================================

def _ensure_dvc_installed(repo_path: Path) -> bool:
    """Install DVC with Google Drive support if not present."""
    try:
        # Check if dvc command is available
        result = subprocess.run(
            "which dvc >/dev/null 2>&1 || pip install -q 'dvc[gdrive]'",
            shell=True, cwd=str(repo_path), capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to install DVC: {e}")
        return False


def initialize_dvc(repo_path: Path, remote_id: Optional[str] = None) -> bool:
    """
    Initialize DVC repository and configure Google Drive remote.

    Args:
        repo_path: Path to the project root.
        remote_id: Google Drive folder ID for DVC remote storage.

    Returns:
        True if DVC configured successfully, False otherwise.
    """
    try:
        # Ensure DVC is installed
        if not _ensure_dvc_installed(repo_path):
            logger.error("Failed to install DVC")
            return False

        # Initialize DVC if .dvc folder doesn't exist
        dvc_dir = repo_path / ".dvc"
        if not dvc_dir.exists():
            logger.info("Initializing DVC repository...")
            if not _run_command("dvc init", cwd=repo_path):
                logger.error("Failed to run 'dvc init'")
                return False
            # Auto-add DVC config to git
            _run_command("git add .dvc .dvcignore", cwd=repo_path, silent=True)
            logger.debug("DVC config files added to git staging")

        # Configure Google Drive remote if ID provided
        if remote_id:
            logger.info(f"Configuring DVC remote to Google Drive: {remote_id}")
            
            # Add remote (use -d to set as default)
            if not _run_command(f"dvc remote add -d mydrive gdrive://{remote_id}", cwd=repo_path):
                logger.error("Failed to add DVC remote")
                return False
            
            # Configure authentication for Colab interactive mode
            _run_command(
                "dvc remote modify mydrive gdrive_use_service_account false",
                cwd=repo_path, silent=True
            )
            logger.info("✅ DVC remote configured successfully")
        
        return True
    except Exception as e:
        logger.error(f"Failed to initialize DVC: {e}")
        return False


def dvc_pull(
    targets: Optional[List[str]] = None,
    repo_path: Optional[Path] = None,
    force: bool = False
) -> bool:
    """
    Pull data from DVC remote storage.

    Args:
        targets: List of DVC-tracked paths to pull (e.g., ['data/raw']).
        repo_path: Project root path (defaults to current dir).
        force: If True, force re-download even if local cache exists.

    Returns:
        True if pull succeeded, False otherwise.
    """
    try:
        cwd = repo_path or Path.cwd()
        
        # Build command
        cmd_parts = ["dvc", "pull"]
        if force:
            cmd_parts.append("--force")
        if targets:
            cmd_parts.extend(targets)
        
        cmd = " ".join(cmd_parts)
        logger.info(f"Pulling data from DVC remote: {' '.join(targets or ['all'])}")
        
        return _run_command(cmd, cwd=cwd)
    except Exception as e:
        logger.error(f"DVC pull failed: {e}")
        return False


def dvc_push(
    targets: Optional[List[str]] = None,
    repo_path: Optional[Path] = None,
    run_cache: bool = False
) -> bool:
    """
    Push data to DVC remote storage.

    Args:
        targets: List of DVC-tracked paths to push.
        repo_path: Project root path.
        run_cache: If True, also push run cache for pipeline reproducibility.

    Returns:
        True if push succeeded, False otherwise.
    """
    try:
        cwd = repo_path or Path.cwd()
        
        # Build command
        cmd_parts = ["dvc", "push"]
        if run_cache:
            cmd_parts.append("--run-cache")
        if targets:
            cmd_parts.extend(targets)
        
        cmd = " ".join(cmd_parts)
        logger.info(f"Pushing data to DVC remote: {' '.join(targets or ['all'])}")
        
        return _run_command(cmd, cwd=cwd)
    except Exception as e:
        logger.error(f"DVC push failed: {e}")
        return False


def dvc_status(repo_path: Optional[Path] = None) -> Optional[dict]:
    """
    Get DVC status for tracked data.

    Args:
        repo_path: Project root path.

    Returns:
        Dict with status info, or None if command failed.
    """
    try:
        cwd = repo_path or Path.cwd()
        result = subprocess.run(
            "dvc status --json",
            shell=True, cwd=str(cwd),
            capture_output=True, text=True, check=True
        )
        import json
        return json.loads(result.stdout)
    except Exception as e:
        logger.warning(f"Could not get DVC status: {e}")
        return None


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def initialize_environment(
    repo_name: str = "california_housing_full_project",
    repo_owner: str = "ebramrafat653-wq",
    use_ssh: bool = True,
    install_deps: bool = True,
    # DVC Configuration
    dvc_remote_id: Optional[str] = None,
    dvc_auto_pull: bool = True,
    dvc_pull_targets: Optional[List[str]] = None,
    dvc_force_pull: bool = False,
) -> Path:
    """
    Main entry point: Initialize complete Colab environment.

    This function orchestrates the full setup workflow:
    1. Mount Google Drive for persistent storage
    2. Configure SSH keys for private repo access (optional)
    3. Clone or update the project repository
    4. Install Python dependencies (optional)
    5. Initialize DVC with Google Drive remote (optional)
    6. Auto-pull tracked data from DVC (optional)
    7. Configure Python path and working directory

    Args:
        repo_name: Name of the GitHub repository.
        repo_owner: GitHub username/organization owning the repo.
        use_ssh: If True, use SSH for git operations; else HTTPS.
        install_deps: If True, install requirements.txt dependencies.
        dvc_remote_id: Google Drive folder ID for DVC remote storage.
        dvc_auto_pull: If True, automatically pull DVC-tracked data after setup.
        dvc_pull_targets: Specific paths to pull from DVC (pulls all if None).
        dvc_force_pull: If True, force re-download even if local cache exists.

    Returns:
        Path object pointing to the project root directory.

    Example:
        >>> from src.utils.colab_setup import initialize_environment
        >>> project_root = initialize_environment(
        ...     dvc_remote_id="1A2B3C4D5E6F7G8H9I0J",
        ...     dvc_pull_targets=["data/raw"]
        ... )
        >>> from src.data.data_loader import DataLoader  # Now importable
    """
    logger.info(f"Initializing Colab environment for '{repo_name}'...")

    # 1. Mount Drive
    drive_base = mount_drive()

    # 2. Configure SSH (optional)
    ssh_configured = configure_ssh() if use_ssh else False

    # 3. Build repository URL
    https_url = f"https://github.com/{repo_owner}/{repo_name}.git"
    repo_url = (
        f"git@github.com:{repo_owner}/{repo_name}.git"
        if use_ssh and ssh_configured else https_url
    )

    # 4. Clone or update repository
    repo_path = Path(f"/content/{repo_name}")
    if not clone_or_update_repo(repo_url, repo_path, use_ssh=use_ssh and ssh_configured):
        logger.error("Failed to clone/update repository")
        raise RuntimeError("Repository setup failed")

    # 5. Install dependencies (optional)
    if install_deps:
        install_dependencies(repo_path / "requirements.txt")

    # 6. Initialize DVC (optional but recommended)
    if dvc_remote_id:
        logger.info("Setting up DVC integration...")
        if initialize_dvc(repo_path, remote_id=dvc_remote_id):
            if dvc_auto_pull:
                logger.info("Auto-pulling DVC-tracked data...")
                success = dvc_pull(
                    targets=dvc_pull_targets,
                    repo_path=repo_path,
                    force=dvc_force_pull
                )
                if success:
                    logger.info("✅ DVC data pulled successfully")
                else:
                    logger.warning("⚠️ DVC pull completed with warnings")
        else:
            logger.warning("⚠️ DVC initialization failed; continuing without DVC")

    # 7. Configure Python environment
    configure_python_path(repo_path)
    set_working_directory(repo_path)

    # 8. Final verification & summary
    logger.info("=" * 60)
    logger.info("✅ ENVIRONMENT READY")
    logger.info(f"📍 Working directory: {os.getcwd()}")
    logger.info(f"📚 Project path: {sys.path[0]}")
    logger.info(f"🗂️ DVC: {'✅ Configured' if (repo_path / '.dvc').exists() else '⚪ Not set'}")
    logger.info(f"🔐 Kaggle: {'✅ Ready' if (Path.home() / '.kaggle' / 'kaggle.json').exists() else '⚪ Not configured'}")
    logger.info("=" * 60)

    return repo_path


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    # Core setup
    "initialize_environment",
    "mount_drive",
    "configure_ssh",
    "clone_or_update_repo",
    "install_dependencies",
    "configure_python_path",
    "set_working_directory",
    # DVC operations
    "initialize_dvc",
    "dvc_pull",
    "dvc_push",
    "dvc_status",
]