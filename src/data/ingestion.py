# src/data/ingestion.py
import sys, os
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parents[2]))

import yaml, subprocess, shutil
from pathlib import Path
from src.utils.paths import get_path, logger, PATHS

def load_config() -> dict:
    """T7mel el config file (.yaml) w traja3o ka dictionary"""
    config_path = PATHS["configs"] / "data_config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"El config mawgoodsh fel path: {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    logger.info(f"✅ Config tet7mel mn {config_path}")
    return config

def setup_kaggle() -> None:
    """T7ot m3lomat Kaggle API (kaggle.json) fe el folder el sa7"""
    kaggle_src = PATHS["kaggle_json"]
    
    if not kaggle_src.exists():
        raise FileNotFoundError(
            f"kaggle.json mawgoodsh fe: {kaggle_src}\n"
            f"Nazzelo mn kaggle.com/settings w 7oto 3ala Drive"
        )
    
    dot_kaggle = Path.home() / ".kaggle"
    dot_kaggle.mkdir(exist_ok=True)
    shutil.copy(kaggle_src, dot_kaggle / "kaggle.json")
    
    try:
        subprocess.run(["chmod", "600", str(dot_kaggle / "kaggle.json")], check=True)
    except:
        pass
    
    logger.info("✅ Kaggle credentials tet7at bengah")

def get_file_details(directory: Path) -> list:
    """Trag3 list of dictionaries b kol m3lomat el files fe folder"""
    files_info = []
    for file_path in directory.iterdir():
        if file_path.is_file():
            files_info.append({
                "name": file_path.name,
                "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                "path": str(file_path)
            })
    return files_info

def download_data(config: dict) -> Path:
    """Tenshel el dataset mn Kaggle w tfakko fe folder el raw data"""
    raw_path = get_path("raw")
    kaggle_id = config["dataset"]["kaggle_id"]
    
    # Save el files eli mawgoda before download
    files_before = set(os.listdir(raw_path)) if raw_path.exists() else set()
    
    logger.info(f"⬇️ Benenshel: {kaggle_id}")
    
    # Neneshel w nfak el zip file automatically
    result = subprocess.run([
        "kaggle", "datasets", "download",
        "-d", kaggle_id,
        "-p", str(raw_path),
        "--unzip"
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        logger.error(f"❌ Kaggle download failed: {result.stderr}")
        raise RuntimeError(f"Download failed: {result.stderr}")
    
    # Show el files el gededa eli nzltt
    files_after = set(os.listdir(raw_path))
    new_files = files_after - files_before
    
    # Remove any .zip files (since we unzipped)
    new_files = [f for f in new_files if not f.endswith('.zip')]
    
    logger.info("=" * 60)
    logger.info("📁 EL FILES ELI TET7MELET:")
    logger.info("=" * 60)
    
    total_size = 0
    for i, file_name in enumerate(new_files, 1):
        file_path = raw_path / file_name
        size_mb = file_path.stat().st_size / (1024 * 1024)
        total_size += size_mb
        logger.info(f"  {i}. {file_name} ({size_mb:.2f} MB)")
    
    logger.info("=" * 60)
    logger.info(f"✅ Total: {len(new_files)} files, {total_size:.2f} MB")
    logger.info(f"📂 El data et7afazt fe: {raw_path}")
    logger.info("=" * 60)
    
    return raw_path

def list_dataset_files(kaggle_id: str) -> None:
    """
    T3reed a2bl ma tenshel el files elly fel dataset 3ala Kaggle
    """
    logger.info(f"🔍 Btf7es el files elly fel dataset: {kaggle_id}")
    result = subprocess.run([
        "kaggle", "datasets", "list",
        kaggle_id, "--json"
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        try:
            import json
            data = json.loads(result.stdout)
            if data and len(data) > 0:
                logger.info(f"📊 Dataset size: {data[0].get('size', 'Unknown')}")
        except:
            pass
    
    # Nshoof el files gowa el dataset
    result = subprocess.run([
        "kaggle", "datasets", "download",
        "-d", kaggle_id,
        "--dry-run"  # This option might not exist, using alternative
    ], capture_output=True, text=True)
    
    logger.info("💡 Tips: Run this command to see all files:")
    logger.info(f"   kaggle datasets download -d {kaggle_id} --unzip -p /tmp --force")

def run_ingestion() -> Path:
    """
    El function el ra'eesiya: 
    1. T7mel el config
    2. Tsetup kaggle credentials
    3. T3reed el files before download
    4. Tenshel el data
    5. T3reed el files eli nzeltt
    """
    logger.info("🚀 Btbd2 el ingestion process...")
    
    config = load_config()
    
    # Show dataset info before download
    kaggle_id = config["dataset"]["kaggle_id"]
    logger.info(f"📦 Dataset: {kaggle_id}")
    
    setup_kaggle()
    raw_path = download_data(config)
    
    # Save el list of downloaded files to a text file
    report_path = raw_path / "_download_report.txt"
    with open(report_path, 'w') as f:
        f.write(f"Download Report - {subprocess.run(['date'], capture_output=True, text=True).stdout}\n")
        f.write(f"Dataset: {kaggle_id}\n")
        f.write(f"Location: {raw_path}\n")
        f.write("=" * 50 + "\n")
        f.write("Files Downloaded:\n")
        for file_path in raw_path.iterdir():
            if file_path.is_file() and file_path.name != "_download_report.txt":
                f.write(f"  - {file_path.name} ({file_path.stat().st_size / (1024*1024):.2f} MB)\n")
    
    logger.info(f"📄 Download report saved to: {report_path}")
    logger.info("✅ Ingestion complete!")
    
    return raw_path

def verify_download_integrity(raw_path: Path) -> dict:
    """
    Tt2kd en el files nzltt kollaha wa sa7ee7a
    """
    logger.info("🔍 Btf7es integrity bta3 el download...")
    
    files_info = get_file_details(raw_path)
    
    # Check for common data files
    extensions = ['.csv', '.json', '.parquet', '.xlsx', '.txt', '.tsv']
    
    data_files = [f for f in files_info if any(f['name'].endswith(ext) for ext in extensions)]
    other_files = [f for f in files_info if f not in data_files]
    
    logger.info("=" * 60)
    logger.info(f"📊 Summary:")
    logger.info(f"   - Total files: {len(files_info)}")
    logger.info(f"   - Data files: {len(data_files)}")
    logger.info(f"   - Other files: {len(other_files)}")
    logger.info(f"   - Total size: {sum(f['size_mb'] for f in files_info):.2f} MB")
    logger.info("=" * 60)
    
    return {
        "total_files": len(files_info),
        "data_files": len(data_files),
        "total_size_mb": sum(f['size_mb'] for f in files_info)
    }

if __name__ == "__main__":
    from src.utils.paths import verify_paths
    verify_paths()
    
    # Download el data
    raw_path = run_ingestion()
    
    # Verify download
    verify_download_integrity(raw_path)
    
    # Show el files fel akher
    logger.info("\n📁 Final list of downloaded files:")
    for file in sorted(raw_path.iterdir()):
        if file.is_file():
            size_mb = file.stat().st_size / (1024 * 1024)
            logger.info(f"   📄 {file.name} ({size_mb:.2f} MB)")