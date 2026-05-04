# src/utils/colab_setup.py

import os
import sys
import shutil
from pathlib import Path

def initialize_environment(repo_name: str = "california_housing_full_project"):
    """
    Sets up the Colab environment: Mounts Drive, configures SSH, 
    clones/updates the repo, and configures Python paths.
    """

    # 1. Mount Google Drive
    drive_path = Path('/content/drive')
    if not (drive_path / 'MyDrive').exists():
        from google.colab import drive
        drive.mount('/content/drive')
        print("✅ Drive mounted")
    else:
        print("✅ Drive already mounted")

    # 2. SSH Configuration
    ssh_dir = Path.home() / '.ssh'
    ssh_dir.mkdir(parents=True, exist_ok=True) # Ensure .ssh exists
    
    ssh_key_source = Path('/content/drive/MyDrive/ssh_config/housing_key')
    ssh_key_dest = ssh_dir / 'id_rsa'

    if ssh_key_source.exists():
        shutil.copy(ssh_key_source, ssh_key_dest)
        ssh_key_dest.chmod(0o600) # Equivalent to chmod 600
        os.system('ssh-keyscan github.com >> ~/.ssh/known_hosts 2>/dev/null')
        print("✅ SSH keys configured")
    else:
        print("⚠️ Warning: SSH key not found on Drive. Private repo access might fail.")

    # 3. Clone or Update Repository
    repo_path = Path(f'/content/{repo_name}')
    if not repo_path.exists():
        print(f"🚀 Cloning {repo_name}...")
        os.system(f'git clone git@github.com:ebramrafat653-wq/{repo_name}.git {repo_path}')
    else:
        print(f"🔄 Repository exists, pulling latest changes...")
        os.system(f'git -C {repo_path} pull')

    # 4. Path Configuration
    repo_str = str(repo_path)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    # 5. Set Working Directory
    os.chdir(repo_path)

    # 6. Verification
    print("-" * 30)
    print(f"📍 Working Dir: {os.getcwd()}")
    print(f"📚 Project Path added to sys.path")
    print("-" * 30)

    return repo_path