# src/utils/colab_setup.py

"""
Colab environment initialization module.

Authentication  : SSH key stored on Google Drive (MyDrive/ssh_config/housing_key).
DVC remote type : local path on Drive  (/content/drive/MyDrive/dvc_storage).
                  No OAuth, no gdrive:// — Drive is already mounted as a filesystem.

Session workflow:
  1. mount_drive()
  2. configure_ssh()
  3. clone_or_update_repo()
  4. setup_drive_symlinks()   ← AFTER clone (repo must exist)
  5. install_dependencies()   ← Uses pyproject.toml ONLY
  6. initialize_dvc()         ← always ensures remote + cache are set
  7. dvc_pull()               ← pull data/raw from Drive remote
  8. sys.path + os.chdir()
"""

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
    configure_git_identity,
    setup_drive_symlinks,
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


def _configure_git_identity(repo_path: Path, name: str, email: str) -> None:
    """Set git user.name and user.email (prevents commit errors in Colab)."""
    _run(f"git config user.name  '{name}'",  cwd=repo_path, silent=True)
    _run(f"git config user.email '{email}'", cwd=repo_path, silent=True)


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
# STEP 5 — DVC INITIALIZATION (ALWAYS CONFIGURE REMOTE + CACHE)
# =============================================================================

def _ensure_dvc_installed(repo_path: Path) -> bool:
    """Install DVC (plain, no gdrive extra — remote is local path on Drive)."""
    try:
        r = subprocess.run(
            "which dvc >/dev/null 2>&1 || pip install -q dvc",
            shell=True, cwd=str(repo_path),
            capture_output=True, text=True,
            errors="replace", timeout=180,
        )
        return r.returncode == 0
    except Exception as e:
        logger.error(f"DVC install failed: {e}")
        return False


def initialize_dvc(
    repo_path: Path,
    git_user_name:  str = "Colab Bot",
    git_user_email: str = "colab@bot.local",
) -> bool:
    """
    Ensure DVC is initialized and the Drive remote + cache are configured.

    Unlike the previous version, this function ALWAYS checks/reconfigures
    the remote and cache in every session, because the Drive remote may
    have been lost or changed, and we want DVC to use the Drive cache
    (not the local .dvc/cache) to avoid re-downloading large files.
    """
    if not IN_COLAB:
        logger.debug("Not in Colab — skipping DVC init")
        return True

    if not _ensure_dvc_installed(repo_path):
        logger.error("DVC installation failed")
        return False

    # -------------------------------------------------------------------------
    # 1. Initialize DVC if not already done
    # -------------------------------------------------------------------------
    if not (repo_path / ".dvc").exists():
        logger.info("First-time DVC initialization…")
        if not _run("dvc init", cwd=repo_path, timeout=60):
            logger.error("dvc init failed")
            return False
        logger.info("DVC initialized locally")
    else:
        logger.info("DVC already initialized (using existing .dvc/)")

    # -------------------------------------------------------------------------
    # 2. Always configure DVC cache on Drive (survives session restarts)
    # -------------------------------------------------------------------------
    drive_cache = DRIVE_BASE / f"{PROJECT_NAME}_dvc_cache"
    drive_cache.mkdir(parents=True, exist_ok=True)

    if not _run(f"dvc cache dir '{drive_cache}'", cwd=repo_path):
        logger.warning("Failed to set DVC cache directory — using default")

    _run("dvc config cache.type symlink", cwd=repo_path, silent=True)
    _run("dvc config cache.protected true", cwd=repo_path, silent=True)
    logger.info(f"DVC cache → {drive_cache}")

    # -------------------------------------------------------------------------
    # 3. Always configure DVC remote (local path on Drive)
    # -------------------------------------------------------------------------
    DVC_STORAGE.mkdir(parents=True, exist_ok=True)

    # Remove existing remote if it exists, then add fresh
    _run(f"dvc remote remove {DVC_REMOTE_NAME}", cwd=repo_path, silent=True, timeout=10)

    if not _run(
        f"dvc remote add -d -f {DVC_REMOTE_NAME} '{DVC_STORAGE}'",
        cwd=repo_path, timeout=30,
    ):
        logger.error("Failed to add DVC remote")
        return False
    logger.info(f"DVC remote '{DVC_REMOTE_NAME}' → {DVC_STORAGE}")

    # -------------------------------------------------------------------------
    # 4. Commit .dvc/config changes if any (so GitHub tracks the remote)
    # -------------------------------------------------------------------------
    configure_git_identity(repo_path)

    # Stage .dvc/config if it changed
    _run("git add .dvc/config .dvcignore", cwd=repo_path, silent=True)

    has_changes = bool(_run_out("git status --porcelain", cwd=repo_path))
    if has_changes:
        _run(
            "git commit -m 'chore: update DVC remote/cache configuration'",
            cwd=repo_path, timeout=60,
        )
        ok = _run("git push", cwd=repo_path, timeout=120)
        if not ok:
            ok = _run("git push -u origin HEAD", cwd=repo_path, timeout=120)
        if ok:
            logger.info("✅ .dvc/config changes pushed to GitHub")
        else:
            logger.error("git push failed — remote config only local")
    else:
        logger.info("No changes to .dvc/config")

    logger.info("✅ DVC ready (remote + cache verified)")
    return True


# =============================================================================
# STEP 6 — DVC PULL / STATUS
# =============================================================================

def dvc_pull(
    targets: Optional[list[str]] = None,
    repo_path: Optional[Path] = None,
    force: bool = False,
    timeout: int = 300,
) -> bool:
    """
    Pull DVC-tracked files from Drive local remote.
    Default targets: ["data/raw"]  — avoids pulling unfinished interim/processed.
    """
    cwd = repo_path or Path.cwd()

    # Safe default: pull only raw data, not everything
    effective_targets = targets if targets is not None else ["data/raw"]

    parts = ["dvc", "pull", f"--remote={DVC_REMOTE_NAME}"]
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
    repo_name:        str = PROJECT_NAME,
    repo_owner:       Optional[str] = None,
    ssh_key_path:     Optional[Union[str, Path]] = None,
    install_deps:     bool = True,
    dvc_auto_pull:    bool = True,
    dvc_pull_targets: Optional[list[str]] = None,   # None → ["data/raw"]
    dvc_force_pull:   bool = False,
    git_user_name:    str = "Colab Bot",
    git_user_email:   str = "colab@bot.local",
) -> Path:
    """
    Full Colab session bootstrap.

    Usage (every session, top of your notebook):
        from src.utils.colab_setup import initialize_environment
        repo_path = initialize_environment(repo_owner="ebramrafat653-wq")

    Returns:
        Path to /content/<repo_name>
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

    # 4. NOW create Drive symlinks (after repo exists)
    setup_drive_symlinks()
    logger.info("Drive symlinks created (data/ and models/ → Drive)")

    # 5. Dependencies (pyproject.toml only)
    if install_deps:
        # Ensure we are in the repo root before installing so pip finds pyproject.toml
        os.chdir(repo_path)
        install_dependencies(install_dev=True)

    # 6. DVC init (always verifies remote + cache)
    dvc_ok = initialize_dvc(
        repo_path,
        git_user_name=git_user_name,
        git_user_email=git_user_email,
    )

    # 7. DVC pull
    if dvc_ok and dvc_auto_pull:
        dvc_pull(
            targets=dvc_pull_targets,   # defaults to ["data/raw"] inside dvc_pull
            repo_path=repo_path,
            force=dvc_force_pull,
        )

    # 8. Python path + working directory
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))
    os.chdir(repo_path)

    # ── Status summary ────────────────────────────────────────────
    dvc_ready    = (repo_path / ".dvc").exists()
    kaggle_ready = (Path.home() / ".kaggle" / "kaggle.json").exists()

    logger.info("=" * 60)
    logger.info("  ✅ ENVIRONMENT READY")
    logger.info(f"  📍 Working dir : {os.getcwd()}")
    logger.info(f"  🗂️  DVC         : {'✅ configured' if dvc_ready    else '⚪ not set'}")
    logger.info(f"  🔐 Kaggle      : {'✅ ready'      if kaggle_ready  else '⚪ not configured'}")
    logger.info(f"  💾 DVC remote  : {DVC_REMOTE_NAME} → {DVC_STORAGE}")
    logger.info("=" * 60)

    return repo_path


__all__ = [
    "initialize_environment",
    "mount_drive",
    "configure_ssh",
    "clone_or_update_repo",
    "install_dependencies",
    "initialize_dvc",
    "dvc_pull",
    "dvc_status",
]
