# 🏠 California Housing Full Project

<div align="center">

[![Python](https://img.shields.io/badge/python-3.9+-3776ab.svg?logo=python&logoColor=white)](https://python.org)
[![DVC](https://img.shields.io/badge/DVC-3.40+-945dd6.svg?logo=dvc&logoColor=white)](https://dvc.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/license-MIT-28a745.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-7%20modules-4c1.svg)](tests/)
[![Colab Ready](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00.svg?logo=googlecolab)](https://colab.research.google.com)

**Production-grade data engineering & ML pipeline** for California housing price prediction  
with emphasis on reproducibility, data versioning, and cloud-native design.

[Quick Start](#-quick-start) • [Architecture](#-pipeline-architecture) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

This project implements an **end-to-end data pipeline** for predicting California median house values. It demonstrates best practices in:

- 🔄 **Data Pipeline Engineering**: Stratified splitting, cleaning with fit/transform isolation, feature engineering
- 📊 **Data Versioning**: DVC integration with Google Drive for reproducible datasets
- ☁️ **Cloud-First Design**: Optimized for Google Colab with seamless Google Drive integration
- 🧪 **Comprehensive Testing**: Unit and integration tests with pytest
- 📝 **Production Standards**: Structured logging, YAML configuration, type hints, validation

| Metric | Value |
|--------|-------|
| **Dataset** | California Housing (Kaggle, 20,640 districts) |
| **Target** | `median_house_value` (regression) |
| **Input Features** | 9 features (numeric + categorical) |
| **Pipeline Split** | 70% train / 15% validation / 15% test (stratified) |
| **Python Support** | 3.9+ |

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

```python
# First cell in Colab notebook
from src.utils.logger import setup_logging
from src.utils.colab_setup import initialize_environment
import logging

# Initialize logging and environment
setup_logging(level=logging.INFO)
project_root = initialize_environment(
    repo_name="california_housing_full_project",
    repo_owner="ebramrafat653-wq",  # Required: GitHub repo owner
    install_deps=True,
    dvc_auto_pull=True,
    dvc_pull_targets=["data/raw"],  # Which targets to pull from DVC
)

# Load and validate data
from src.data.data_loader import DataLoader
from src.data.validation import validate_dataframe

loader = DataLoader()
df = loader.load_raw("housing.csv")
report = validate_dataframe(df)
print(report.summary())
```

### Option 2: Local Development

```bash
# Clone repository
git clone https://github.com/ebramrafat653-wq/california_housing_full_project.git
cd california_housing_full_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize DVC with local remote
dvc init
dvc remote add -d mylocal /path/to/dvc/storage  # Use local path for development

# Pull data
dvc pull

# Run tests
pytest tests/ -v
```

---

## 🏗️ Pipeline Architecture

The project implements a **fit-on-train-only pipeline** to prevent data leakage:

```
Raw Data (Kaggle)
      ↓
[1] INGESTION (ingestion.py)
    └─ Download from Kaggle API → data/raw/
      ↓
[2] VALIDATION (validation.py)  
    └─ Enforce data quality rules (nulls, bounds, categories)
      ↓
[3] SPLITTING (splitting.py)
    └─ Stratified split (70/15/15) on price quantiles → data/interim/
      ↓
[4] CLEANING (cleaning.py) — FIT ON TRAIN ONLY
    ├─ Imputation (median on train)
    ├─ Flags (is_capped for censored values)
    ├─ Transforms (log1p on counts)
    └─ Outlier detection (LOF) → data/processed/
      ↓
[5] FEATURE ENGINEERING (engineering.py) — TRANSFORM ALL SPLITS
    ├─ Ratios (rooms_per_household, etc.)
    ├─ Geographic distances (to SF & LA)
    └─ Drop original count columns → data/processed/*_feat.csv
      ↓
Ready for Model Training
```

### Key Transformations

| Stage | Transformation | Purpose |
|-------|---|---|
| **Imputation** | Median (train) → apply to val/test | Handle missing `total_bedrooms` (~0.97%) |
| **Flags** | `is_capped` (≥ $500,001) | Identify censored predictions |
| **Scaling** | log1p on counts | Reduce skew in `total_rooms`, `bedrooms`, `population` |
| **Outliers** | LOF detector (k=20, contamination=0.02) | Identify spatial/market anomalies |
| **Ratios** | `rooms_per_household`, `bedrooms_per_room` | Reduce multicollinearity |
| **Geography** | Euclidean distance to SF & LA | Capture market effects from major metros |

---

## 🗂️ Project Structure

```
california_housing_full_project/
│
├── 📂 src/                          # Core codebase (Git tracked)
│   ├── data/                        # Data pipeline modules
│   │   ├── ingestion.py             # Download from Kaggle
│   │   ├── validation.py            # Data quality checks
│   │   ├── splitting.py             # Stratified 70/15/15 split
│   │   ├── cleaning.py              # Fit-on-train transformations
│   │   ├── data_loader.py           # Environment-aware data loading
│   │   └── profiling.py             # EDA & statistics
│   │
│   ├── features/                    # Feature engineering
│   │   ├── engineering.py           # Feature creation & selection
│   │   └── pipeline.py              # Orchestration
│   │
│   ├── models/                      # Model training (TODO)
│   │   ├── train.py                 # Model training
│   │   ├── evaluate.py              # Metrics & validation
│   │   └── predict.py               # Inference
│   │
│   └── utils/                       # Utilities
│       ├── logger.py                # Structured logging
│       ├── colab_setup.py           # Google Colab integration
│       ├── helpers.py               # Common utilities
│       └── paths.py                 # Path resolution
│
├── 📂 tests/                        # Unit & integration tests (Git tracked)
│   ├── conftest.py                  # pytest fixtures
│   ├── test_ingestion.py            # Ingestion tests
│   ├── test_data_loader.py          # Data loading tests
│   ├── test_validation.py           # Validation tests
│   ├── test_cleaning.py             # Cleaning & isolation tests
│   ├── test_engineering.py          # Feature engineering tests
│   ├── test_api.py                  # API tests (TODO)
│   └── test_pipeline.py             # End-to-end pipeline tests
│
├── 📂 data/                         # Data directory (DVC tracked)
│   ├── raw/                         # Original data (Kaggle) ⚠️ .gitignore
│   │   └── housing.csv.dvc          # DVC metadata (Git tracked)
│   ├── interim/                     # Intermediate splits ⚠️ .gitignore
│   │   └── *.dvc                    # DVC metadata (Git tracked)
│   └── processed/                   # Final features ⚠️ .gitignore
│       └── *.dvc                    # DVC metadata (Git tracked)
│
├── 📂 models/                       # Trained models (DVC tracked)
│   ├── final/                       # Production model
│   └── experiments/                 # Experiment snapshots
│
├── 📂 artifacts/                    # Train-fit statistics & models (DVC + Git)
│   ├── cleaning_artifacts.json      # Imputation medians, LOF params (Git tracked)
│   └── lof_model.pkl               # Fitted LocalOutlierFactor model (DVC tracked)
│
├── 📂 configs/                      # Configuration files (Git tracked)
│   └── data_config.yaml             # Pipeline parameters & EDA results
│
├── 📂 notebooks/                    # Jupyter notebooks (Git tracked)
│   ├── 00_environment_setup.ipynb   # Setup & authentication
│   ├── 01_data_profiling.ipynb      # EDA & statistics
│   ├── 02_data_splitting.ipynb      # Train/val/test split analysis
│   ├── 03_eda.ipynb                 # Exploratory analysis
│   ├── 04_cleaning.ipynb            # Cleaning transformation & validation
│   ├── 05_feature_engineering.ipynb # Feature creation analysis
│   └── test.ipynb                   # Quick testing notebook
│
├── 📂 api/                          # FastAPI application (TODO)
│   ├── main.py                      # API endpoints
│   ├── schemas.py                   # Request/response models
│   └── predict.py                   # Prediction service
│
├── 📂 pipelines/                    # Pipeline orchestration
│   └── run_pipeline.py              # End-to-end pipeline script
│
├── 📂 docker/                       # Containerization
│   ├── Dockerfile                   # Container image
│   └── docker-compose.yml           # Multi-container setup
│
├── 📂 reports/                      # Generated reports
│   ├── eda_report.html              # Visual EDA report
│   └── model_report.md              # Model evaluation report
│
├── 📂 .dvc/                         # DVC configuration (Git tracked)
├── .dvcignore                       # DVC ignore rules
├── .gitignore                       # Git ignore rules
├── pyproject.toml                   # Project metadata & dependencies
├── requirements.txt                 # Pip dependencies
├── setup.py                         # Installation script
├── Makefile                         # Development commands
├── LICENSE                          # MIT License
└── README.md                        # This file
```

**Legend:**
- ✅ **Git tracked**: Code, configs, metadata (always committed)
- ⚠️ **DVC tracked**: Data files (committed as `.dvc` metadata only)
- 📂 **Directory**: Folder containing related files
- ❓ **TODO**: Planned but not yet implemented

---

## ⚙️ Setup Guide

### Prerequisites

- **Python 3.9+** (3.10+ recommended)
- **Git** & **Git LFS** (optional, for large models)
- **Google Account** (for Kaggle API and DVC with Google Drive)
- **Kaggle API Key** (download from https://www.kaggle.com/settings)

### Step 1: Clone Repository

```bash
git clone https://github.com/ebramrafat653-wq/california_housing_full_project.git
cd california_housing_full_project
```

### Step 2: Set Up Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate
# On macOS/Linux:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Configure DVC with Google Drive

#### 3a. Create Google Drive Folder
1. Go to https://drive.google.com
2. Create folder: `california_housing_dvc_remote`
3. Copy folder ID from URL: `https://drive.google.com/drive/folders/[FOLDER_ID]`

#### 3b. Initialize DVC & Configure Remote

**Option A: Google Drive (for Colab)**
```bash
dvc init
dvc remote add -d gdrive gdrive://YOUR_FOLDER_ID
dvc remote modify gdrive gdrive_use_service_account false
```

**Option B: Local Directory (for local development with Git)**
```bash
dvc init  # ← Important: without --no-scm (allows .dvc config in Git)
dvc remote add -d mylocal /path/to/dvc/storage  # Absolute path recommended
```

#### 3c. Authenticate & Test
```bash
# Verify remote is configured
dvc remote list

# First push/pull will authenticate (if using Google Drive)
dvc push
# → If Google Drive: Opens browser → Authorize → Caches credentials
```

### Step 4: Configure Kaggle API

```bash
# Download kaggle.json from https://www.kaggle.com/settings/account

# Place it in the correct location:
# On macOS/Linux: ~/.kaggle/kaggle.json
# On Windows: C:\Users\<YourUsername>\.kaggle\kaggle.json

# Set permissions (macOS/Linux only)
chmod 600 ~/.kaggle/kaggle.json
```

### Step 5: Pull Data & Run Pipeline

```bash
# Pull data from DVC
dvc pull

# Run data pipeline
python pipelines/run_pipeline.py

# Verify with tests
pytest tests/ -v -m "not integration"
```

---

---

## 🛠️ Development Commands

```bash
# Run tests (unit tests only, no DVC/Kaggle calls)
pytest tests/ -v -m "not integration"

# Run all tests including integration
pytest tests/ -v

# Format code with black
black src/ tests/

# Lint code with ruff
ruff check src/ tests/

# Run full data pipeline
python pipelines/run_pipeline.py

# Generate test coverage report
pytest tests/ --cov=src --cov-report=html
```

> **Note**: The `Makefile` is reserved for future automation. For now, use the commands above directly.

---

## 🧪 Testing

The project includes comprehensive test coverage across all pipeline stages:

### Test Structure

```
tests/
├── conftest.py                  # Shared fixtures (test data, mocks)
├── test_ingestion.py           # Kaggle API, DVC pull/push
├── test_data_loader.py         # Path resolution, loading logic
├── test_validation.py          # Data quality rules
├── test_cleaning.py            # Fit/transform isolation, transformations
├── test_engineering.py         # Feature creation, selection
├── test_pipeline.py            # End-to-end workflow
└── test_api.py                 # API endpoints (TODO)
```

### Running Tests

```bash
# Run all unit tests (fast, no DVC operations)
pytest tests/ -v -m "not integration"

# Run full test suite (includes DVC, Kaggle API)
pytest tests/ -v

# Run specific test file
pytest tests/test_cleaning.py -v -s

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run with detailed output
pytest tests/test_engineering.py -v --tb=long

# Run tests matching a pattern
pytest tests/ -k "test_fit_transform" -v
```

### Key Test Patterns

1. **Fit/Transform Isolation**: Verify that transformers fit on train data only
2. **Data Leakage Prevention**: Ensure train/val/test splits are clean
3. **Error Handling**: Validate graceful failures with bad inputs
4. **Reproducibility**: Same random seed → same results
5. **Integration**: End-to-end pipeline runs successfully

### Example Test Fixture

```python
@pytest.fixture
def sample_housing_df():
    """Provides a small housing dataset for testing."""
    return pd.DataFrame({
        'longitude': [-122.23, -122.24],
        'latitude': [37.88, 37.89],
        'housing_median_age': [41, 21],
        'total_rooms': [880, 7099],
        'total_bedrooms': [129.0, 1106.0],
        'population': [322, 2401],
        'households': [126, 1138],
        'median_income': [8.3252, 8.3014],
        'median_house_value': [452600, 358500],
        'ocean_proximity': ['NEAR BAY', 'NEAR BAY']
    })
```

---

## 🤝 Contributing

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feat/new-feature
   ```

2. **Make Changes**
   - Modify code in `src/` or `tests/`
   - Update configuration in `configs/data_config.yaml` if needed
   - Create/update tests in `tests/`

3. **Test Locally**
   ```bash
   # Run relevant tests
   pytest tests/test_your_module.py -v
   
   # Run all tests
   pytest tests/ -v
   
   # Check code quality
   black src/ tests/
   ruff check src/ tests/
   ```

4. **Track New Data** (if applicable)
   ```bash
   # If adding new datasets
   dvc add data/new_data.csv
   git add data/new_data.csv.dvc
   ```

5. **Commit & Push**
   ```bash
   git add src/ tests/ configs/ data/*.dvc
   git commit -m "feat: add new feature

   - Brief description
   - Related issue #123 (if applicable)
   "
   git push origin feat/new-feature
   ```

6. **Open Pull Request**
   - Link to related issues
   - Describe changes and testing
   - Request review from maintainers

### Code Style Guidelines

- **Python**: Follow [PEP 8](https://pep8.org/) (enforced with `black` + `ruff`)
- **Type Hints**: Add type annotations to all functions
- **Docstrings**: Use Google-style docstrings
- **Tests**: Aim for >80% coverage in new code
- **Commits**: Use conventional commits (feat:, fix:, test:, docs:, refactor:)

### ⚠️ Important Rules

- **Never commit raw data files** — only `.dvc` metadata files
- **Fit transformers on train data only** — prevent data leakage
- **Commit `artifacts/cleaning_artifacts.json` to Git** — contains train-fit statistics
- **Track `artifacts/lof_model.pkl` with DVC** — binary model file (add to `.dvc`)
- **Update tests** when modifying pipeline logic
- **Update README** for new features or breaking changes
- **Review DVC logs** before pushing large datasets

---

## 📖 Documentation

- **[EDA Report](reports/eda_report.html)** — Exploratory Data Analysis visualizations
- **[Model Report](reports/model_report.md)** — Model evaluation metrics and analysis
- **[Config Reference](configs/data_config.yaml)** — All pipeline parameters
- **[API Schema](api/schemas.py)** — Request/response models (TODO)

---

## 🔧 Troubleshooting

### DVC Issues

| Problem | Solution |
|---------|----------|
| `dvc pull` fails with "remote not found" | Run `dvc remote list`; if empty, run `dvc remote add -d <name> <path_or_gdrive_id>` |
| `dvc pull` fails with auth error (Google Drive) | Run `dvc remote modify gdrive gdrive_use_service_account false` then `dvc pull` |
| `dvc push` times out (Google Drive) | Check Drive quota; consider archiving old experiments |
| Artifacts missing in git | Run `git add artifacts/` and `git commit` (artifacts are Git-tracked) |
| Local DVC remote permission denied | Check path exists and user has read/write access: `ls -la /path/to/dvc/storage` |

### Data Issues

| Problem | Solution |
|---------|----------|
| `KeyError: 'median_house_value'` | Verify data loaded correctly; check `configs/data_config.yaml` target column |
| Validation errors after loading | Run `validate_dataframe(df)` to see detailed report; check `validation.py` rules |
| Missing values in pipeline | Check `cleaning.py` imputation logic; ensure `artifacts/cleaning_artifacts.json` exists |
| Stratified split fails | Ensure target column has no NaN values; check if `_add_price_strata()` completes successfully |
| LOF outlier detection fails | Check `artifacts/lof_model.pkl` exists and has valid n_neighbors settings |

### Test Failures

```bash
# Run with verbose output to see exact errors
pytest tests/test_module.py -vv --tb=long

# Run a single test function
pytest tests/test_module.py::test_function_name -v

# Show print statements during test
pytest tests/ -v -s
```

### Colab Issues

| Problem | Solution |
|---------|----------|
| Module import errors | Run first cell: `initialize_environment()` with `install_deps=True` |
| DVC authentication hanging | Restart kernel and run `initialize_environment()` again |
| Google Drive mount issues | Run: `from google.colab import drive; drive.mount('/content/drive')` |

---

## 📊 Project Status

### ✅ Completed
- [x] Data ingestion (Kaggle API)
- [x] Data validation & profiling
- [x] Stratified train/val/test splitting
- [x] Data cleaning with fit/transform isolation
- [x] Feature engineering (ratios, geographic)
- [x] Comprehensive testing suite
- [x] DVC integration with Google Drive
- [x] Google Colab optimization
- [x] Structured logging
- [x] Logging warning fixes (initialize logging before get_logger)

### 🔄 In Progress / TODO
- [ ] Model training (train.py)
- [ ] Model evaluation (evaluate.py)
- [ ] Model inference (predict.py)
- [ ] FastAPI application
- [ ] Docker containerization
- [ ] CI/CD pipeline (GitHub Actions)
- [ ] Model serving (production)

---

## 📞 Support

For issues, questions, or contributions:
- **Issues**: [GitHub Issues](https://github.com/ebramrafat653-wq/california_housing_full_project/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ebramrafat653-wq/california_housing_full_project/discussions)
- **Email**: ebramrafat569@gmail.com

---

## 📜 License

This project is licensed under the [MIT License](LICENSE) — feel free to use it for educational or commercial purposes.

---

## 🙏 Acknowledgments

- **Dataset**: California Housing dataset from [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
- **Inspiration**: ML Engineering best practices from production systems
- **Tools**: [DVC](https://dvc.org/), [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/)

---

**Last Updated**: 2026-06-22 | **Status**: Production Ready (Data Pipeline) | **Python**: 3.9+
