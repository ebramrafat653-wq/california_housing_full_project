# =============================================================================
# src/utils/colab_setup.py
# California Housing Project — Colab Environment Initialization
#
# RESPONSIBILITY:
#   Bootstrap a fully functional development environment inside Google Colab.
#
# WORKFLOW:
#   1. mount_drive()
#   2. configure_ssh()
#   3. clone_or_update_repo()
#   4. install_dependencies()   ← Uses pyproject.toml ONLY
#   5. configure_dvc_local()    ← Configure local DVC remote (config.local)
#   6. sys.path + os.chdir()
#
# NOTE: DVC pull and pipeline execution are done manually in notebook cells.
# =============================================================================

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

from src.utils.logger import get_logger
from src.utils.paths import (
    DRIVE_BASE,
    DVC_REMOTE_NAME,
    DVC_STORAGE,
    IN_COLAB,
    PROJECT_NAME,
)

try:
    import yaml
    _YAML_AVAILABLE = True
except ImportError:
    _YAML_AVAILABLE = False
    yaml = None  # type: ignore

logger = get_logger(__name__)


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _run(
    cmd: str,
    cwd: Optional[Path] = None,
    silent: bool = False,
    timeout: Optional[int] = None,
) -> bool:
    """Run a shell command; return True on success."""
    try:
        subprocess.run(
            cmd, shell=True,
            cwd=str(cwd) if cwd else None,
            capture_output=True, text=True,
            errors="replace", check=True,
            timeout=timeout,
        )
        if not silent:
            logger.debug(f"OK: {cmd}")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"Timeout ({timeout}s): {cmd}")
        return False
    except subprocess.CalledProcessError as e:
        if not silent:
            logger.error(f"FAILED: {cmd}\n  {e.stderr.strip()}")
        return False
    except Exception as e:
        logger.error(f"ERROR: {cmd} → {e}")
        return False


def _run_out(
    cmd: str,
    cwd: Optional[Path] = None,
    timeout: int = 30,
) -> Optional[str]:
    """Run a shell command; return stdout string or None on failure."""
    try:
        r = subprocess.run(
            cmd, shell=True,
            cwd=str(cwd) if cwd else None,
            capture_output=True, text=True,
            errors="replace", timeout=timeout,
        )
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


# =============================================================================
# STEP 1 — MOUNT DRIVE
# =============================================================================

def mount_drive() -> Path:
    """Mount Google Drive and return MyDrive path."""
    drive_base = Path("/content/drive/MyDrive")

    if not IN_COLAB:
        logger.warning("Not in Colab — skipping Drive mount")
        return drive_base

    if drive_base.exists():
        logger.info("Drive already mounted")
        return drive_base

    try:
        from google.colab import drive  # type: ignore
        drive.mount("/content/drive")
        logger.info("Drive mounted successfully")
        return drive_base
    except ImportError:
        raise RuntimeError("google.colab not available") from None
    except Exception as e:
        raise RuntimeError(f"Drive mount failed: {e}") from e


# =============================================================================
# STEP 2 — SSH CONFIGURATION
# =============================================================================

def configure_ssh(ssh_key_path: Optional[Union[str, Path]] = None) -> bool:
    """
    Copy SSH private key from Drive to ~/.ssh/id_rsa.
    Default key location: MyDrive/ssh_config/housing_key
    """
    if not IN_COLAB:
        logger.debug("Not in Colab — skipping SSH config")
        return True

    source = (
        Path(ssh_key_path) if ssh_key_path
        else DRIVE_BASE / "ssh_config" / "housing_key"
    )

    if not source.exists():
        logger.error(f"SSH key not found: {source}")
        logger.info("Fix: place your private key at MyDrive/ssh_config/housing_key")
        return False

    ssh_dir = Path.home() / ".ssh"
    ssh_dir.mkdir(parents=True, exist_ok=True)
    dest = ssh_dir / "id_rsa"

    try:
        shutil.copy2(source, dest)
        dest.chmod(0o600)
        _run("ssh-keyscan -H github.com >> ~/.ssh/known_hosts 2>/dev/null",
             silent=True, timeout=15)
        logger.info("SSH configured successfully")
        return True
    except Exception as e:
        logger.error(f"SSH config failed: {e}")
        return False


# =============================================================================
# STEP 3 — CLONE / UPDATE REPO
# =============================================================================

def clone_or_update_repo(
    repo_owner: str,
    repo_name: str,
    repo_path: Path,
    ssh_configured: bool = True,
) -> bool:
    """Clone via SSH (or HTTPS fallback); pull if repo already exists."""
    url = (
        f"git@github.com:{repo_owner}/{repo_name}.git"
        if ssh_configured
        else f"https://github.com/{repo_owner}/{repo_name}.git"
    )

    if (repo_path / ".git").exists():
        logger.info(f"Repo exists — updating via {url}")
        ok = _run("git pull --rebase", cwd=repo_path, timeout=120)
        if ok:
            logger.info("Repo updated")
        return ok

    logger.info(f"Cloning {url} → {repo_path}")
    ok = _run(f"git clone {url} {repo_path}", timeout=180)
    if ok:
        logger.info("Repo cloned successfully")
    return ok


# =============================================================================
# STEP 4 — DEPENDENCIES (pyproject.toml ONLY)
# =============================================================================

def install_dependencies(install_dev: bool = True) -> bool:
    """
    Install project dependencies from pyproject.toml (Colab only).

    Uses `pip install -q -e .` (or `.[dev]` if install_dev is True).
    The editable install (-e) ensures any code edits made directly in
    Colab are immediately reflected without needing to reinstall.
    """
    if not IN_COLAB:
        logger.debug("Skipping dependency install outside Colab")
        return True

    pyproject = Path("pyproject.toml")
    if not pyproject.exists():
        logger.error(
            f"pyproject.toml not found at {pyproject.absolute()}. "
            "Cannot install dependencies. Ensure you are in the repo root."
        )
        return False

    logger.info("Installing dependencies from pyproject.toml…")

    extra = ".[dev]" if install_dev else "."
    cmd = f"pip install -q -e {extra}"

    # Increased timeout to 600s because dev deps (like mypy, ipykernel)
    # and ML libs can take a while to build/install in Colab.
    ok = _run(cmd, timeout=600)

    if ok:
        logger.info("✅ Dependencies installed successfully")
    else:
        logger.error("❌ Dependency installation failed")
    return ok


# =============================================================================
# STEP 5 — CONFIGURE DVC LOCAL REMOTE (config.local ONLY)
# =============================================================================

def configure_dvc_local(repo_path: Path) -> bool:
    """
    Configure DVC local remote settings in .dvc/config.local.

    This function:
    - Checks that .dvc/ directory exists (DVC must be initialized in repo)
    - Sets local remote name to 'mylocal'
    - Sets local remote URL to Google Drive path
    - Does NOT modify .dvc/config (tracked by Git)
    - Does NOT perform git add/commit/push

    The settings are written to .dvc/config.local which is ignored by Git.
    """
    if not IN_COLAB:
        logger.debug("Not in Colab — skipping DVC local config")
        return True

    dvc_dir = repo_path / ".dvc"
    if not dvc_dir.exists():
        logger.error(
            f"DVC not initialized in {repo_path}. "
            "Please run 'dvc init' in your repository first."
        )
        return False

    # Ensure DVC storage directory exists on Drive
    DVC_STORAGE.mkdir(parents=True, exist_ok=True)

    # Configure local remote (writes to .dvc/config.local)
    logger.info(f"Configuring DVC local remote: {DVC_REMOTE_NAME} → {DVC_STORAGE}")

    # Set default remote to 'mylocal'
    if not _run(
        f"dvc config --local core.remote {DVC_REMOTE_NAME}",
        cwd=repo_path,
        silent=True,
    ):
        logger.error("Failed to set core.remote")
        return False

    # Set remote URL to Google Drive path
    if not _run(
        f"dvc config --local remote.{DVC_REMOTE_NAME}.url '{DVC_STORAGE}'",
        cwd=repo_path,
        silent=True,
    ):
        logger.error("Failed to set remote URL")
        return False

    logger.info("✅ DVC local remote configured in .dvc/config.local")
    logger.info("   (This file is ignored by Git)")

    return True


# =============================================================================
# STEP 6 — DVC PULL / STATUS (Manual helpers)
# =============================================================================

def dvc_pull(
    targets: Optional[list[str]] = None,
    repo_path: Optional[Path] = None,
    force: bool = False,
    timeout: int = 300,
) -> bool:
    """
    Pull DVC-tracked files from configured remote.
    Default target: ["data/raw/housing.csv"] — matches dvc.yaml ingestion output.

    NOTE: This is a helper function. In notebooks, you can also run:
        !dvc pull
    """
    cwd = repo_path or Path.cwd()

    # Default to pulling raw data only
    effective_targets = targets if targets is not None else ["data/raw/housing.csv"]

    parts = ["dvc", "pull"]
    if force:
        parts.append("--force")
    parts.extend(effective_targets)

    cmd = " ".join(parts)
    logger.info(f"DVC pull: {effective_targets}")
    ok = _run(cmd, cwd=cwd, timeout=timeout)

    if ok:
        logger.info("✅ DVC pull complete")
    else:
        logger.warning(
            "⚠️  DVC pull had warnings — this is normal on the very first run "
            "(no data pushed yet). Run ingestion to download and push data."
        )
    return ok


def dvc_status(repo_path: Optional[Path] = None) -> Optional[dict]:
    """Return DVC status as dict, or None on failure."""
    out = _run_out("dvc status --json", cwd=repo_path or Path.cwd(), timeout=60)
    try:
        return json.loads(out) if out else None
    except json.JSONDecodeError:
        return None


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def initialize_environment(
    repo_name: str = PROJECT_NAME,
    repo_owner: Optional[str] = None,
    ssh_key_path: Optional[Union[str, Path]] = None,
    install_deps: bool = True,
    git_user_name: str = "Colab Bot",
    git_user_email: str = "colab@bot.local",
) -> Path:
    """
    Full Colab session bootstrap.

    Usage (every session, top of your notebook):
        from src.utils.colab_setup import initialize_environment
        repo_path = initialize_environment(repo_owner="your_github_username")

    Returns:
        Path to /content/<repo_name>

    After this function completes, you should manually run:
        !dvc pull
        !dvc status
        !dvc repro
    """
    if repo_owner is None:
        repo_owner = os.environ.get("GITHUB_REPO_OWNER")
        if not repo_owner:
            raise ValueError(
                "repo_owner is required. "
                "Pass it explicitly or set the GITHUB_REPO_OWNER env var."
            )

    logger.info("=" * 60)
    logger.info(f"  Colab Setup: {repo_owner}/{repo_name}")
    logger.info("=" * 60)

    # 1. Mount Drive
    mount_drive()

    # 2. SSH
    ssh_ok = configure_ssh(ssh_key_path)
    if not ssh_ok:
        logger.warning("SSH unavailable — falling back to HTTPS (git push will fail)")

    # 3. Clone / pull repo
    repo_path = Path(f"/content/{repo_name}")
    if not clone_or_update_repo(repo_owner, repo_name, repo_path, ssh_ok):
        raise RuntimeError("Repository clone/update failed")

    # 4. Dependencies (pyproject.toml only)
    if install_deps:
        # Ensure we are in the repo root before installing so pip finds pyproject.toml
        os.chdir(repo_path)
        install_dependencies(install_dev=True)

    # 5. Configure DVC local remote (writes to .dvc/config.local)
    dvc_ok = configure_dvc_local(repo_path)

    # 6. Python path + working directory
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    os.chdir(repo_path)

    # ── Status summary ────────────────────────────────────────────
    dvc_ready = (repo_path / ".dvc").exists()
    kaggle_ready = (Path.home() / ".kaggle" / "kaggle.json").exists()

    logger.info("=" * 60)
    logger.info("  ✅ ENVIRONMENT READY")
    logger.info(f"  📍 Working dir : {os.getcwd()}")
    logger.info(f"  🗂️  DVC         : {'✅ configured' if dvc_ready else '⚪ not set'}")
    logger.info(f"  🔐 Kaggle      : {'✅ ready' if kaggle_ready else '⚪ not configured'}")
    logger.info(f"  💾 DVC remote  : {DVC_REMOTE_NAME} → {DVC_STORAGE}")
    logger.info("")
    logger.info("  Next steps:")
    logger.info("    !dvc pull")
    logger.info("    !dvc status")
    logger.info("    !dvc repro")
    logger.info("=" * 60)

    return repo_path


__all__ = [
    "initialize_environment",
    "mount_drive",
    "configure_ssh",
    "clone_or_update_repo",
    "install_dependencies",
    "configure_dvc_local",
    "dvc_pull",
    "dvc_status",
]