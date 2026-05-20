# src/utils/colab_setup.py

"""
Colab environment initialization module.

Authentication  : SSH key stored on Google Drive (MyDrive/ssh_config/housing_key).
DVC remote type : local path on Drive  (/content/drive/MyDrive/dvc_storage).
                  No OAuth, no gdrive:// — Drive is already mounted as a filesystem.

Session workflow:
  1. mount_drive()
  2. setup_drive_symlinks()   ← data/ and models/ → Drive
  3. configure_ssh()
  4. clone_or_update_repo()
  5. install_dependencies()
  6. initialize_dvc()         ← first time: init + configure; later: no-op
  7. dvc_pull()               ← pull data/raw from Drive remote
  8. sys.path + os.chdir()
"""

import json
import os
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Optional, List, Union

from src.utils.logger import get_logger
from src.utils.paths import (
    IN_COLAB, PROJECT_NAME, DRIVE_BASE,
    DVC_REMOTE_NAME, DVC_STORAGE,
    setup_drive_symlinks, configure_git_identity,
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
        raise RuntimeError("google.colab not available")
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
# STEP 4 — DEPENDENCIES
# =============================================================================

def install_dependencies(req_file: Optional[Union[str, Path]] = None) -> bool:
    """Install pip requirements (Colab only)."""
    if not IN_COLAB:
        logger.debug("Skipping dependency install outside Colab")
        return True

    req = Path(req_file) if req_file else Path("requirements.txt")
    if not req.exists():
        logger.warning(f"requirements.txt not found: {req} — skipping")
        return True

    logger.info("Installing dependencies…")
    ok = _run(f"pip install -q -r {req}", timeout=300)
    if ok:
        logger.info("Dependencies installed")
    return ok


# =============================================================================
# STEP 5 — DVC INITIALIZATION
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
    First time → dvc init + cache on Drive + local remote on Drive + git push
    Subsequent → .dvc/ already cloned from GitHub → just verify DVC installed
    """
    if not IN_COLAB:
        logger.debug("Not in Colab — skipping DVC init")
        return True

    if not _ensure_dvc_installed(repo_path):
        logger.error("DVC installation failed")
        return False

    if (repo_path / ".dvc").exists():
        logger.info("✅ DVC already initialized (config from GitHub)")
        return True

    logger.info("First-time DVC initialization…")

    # 1. dvc init
    if not _run("dvc init", cwd=repo_path, timeout=60):
        logger.error("dvc init failed")
        return False

    # 2. DVC cache on Drive (survives session restarts — no re-download needed)
    drive_cache = DRIVE_BASE / f"{PROJECT_NAME}_dvc_cache"
    drive_cache.mkdir(parents=True, exist_ok=True)
    _run(f"dvc cache dir '{drive_cache}'",  cwd=repo_path)
    _run("dvc config cache.type symlink",   cwd=repo_path)
    _run("dvc config cache.protected true", cwd=repo_path)
    logger.info(f"DVC cache → {drive_cache}")

    # 3. DVC remote = local path on Drive (no OAuth, no internet for push/pull)
    DVC_STORAGE.mkdir(parents=True, exist_ok=True)
    if not _run(
        f"dvc remote add -d -f {DVC_REMOTE_NAME} '{DVC_STORAGE}'",
        cwd=repo_path, timeout=30,
    ):
        logger.error("Failed to add DVC remote")
        return False
    logger.info(f"DVC remote '{DVC_REMOTE_NAME}' → {DVC_STORAGE}")

    # 4. Commit .dvc/config to GitHub (so next session clones it automatically)
    configure_git_identity(repo_path)
    _run("git add .dvc/config .dvcignore", cwd=repo_path, silent=True)

    has_changes = bool(_run_out("git status --porcelain", cwd=repo_path))
    if has_changes:
        _run(
            "git commit -m 'chore: initialize DVC with Drive local remote'",
            cwd=repo_path, timeout=60,
        )
        ok = _run("git push", cwd=repo_path, timeout=120)
        if not ok:
            ok = _run("git push -u origin HEAD", cwd=repo_path, timeout=120)
        if ok:
            logger.info("✅ .dvc/config pushed to GitHub")
        else:
            logger.error(
                "git push FAILED.\n"
                "Check: GitHub → Settings → SSH keys has Read/Write scope."
            )
            return False
    else:
        logger.info("No git changes to commit")

    logger.info("✅ DVC initialized successfully")
    return True


# =============================================================================
# STEP 6 — DVC PULL / STATUS
# =============================================================================

def dvc_pull(
    targets: Optional[List[str]] = None,
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
    dvc_pull_targets: Optional[List[str]] = None,   # None → ["data/raw"]
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

    # 2. Symlinks (data/ and models/ → Drive) — BEFORE clone so paths exist
    setup_drive_symlinks()

    # 3. SSH
    ssh_ok = configure_ssh(ssh_key_path)
    if not ssh_ok:
        logger.warning("SSH unavailable — falling back to HTTPS (git push will fail)")

    # 4. Clone / pull repo
    repo_path = Path(f"/content/{repo_name}")
    if not clone_or_update_repo(repo_owner, repo_name, repo_path, ssh_ok):
        raise RuntimeError("Repository clone/update failed")

    # 5. Dependencies
    if install_deps:
        install_dependencies(repo_path / "requirements.txt")

    # 6. DVC init
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