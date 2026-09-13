# 🏠 California Housing Full Project

<div align="center">

[![Python](https://img.shields.io/badge/python-3.9+-3776ab.svg?logo=python&logoColor=white)](https://python.org)
[![DVC](https://img.shields.io/badge/DVC-3.40+-945dd6.svg?logo=dvc&logoColor=white)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.8+-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/license-MIT-28a745.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-189%20functions-4c1.svg)](tests/)
[![Colab Ready](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00.svg?logo=googlecolab)](https://colab.research.google.com)

**Production-grade data engineering and ML pipeline** for California housing price prediction  
with reproducibility, data versioning, leakage-safe training, experiment tracking, and REST inference.

[Quick Start](#-quick-start) • [Architecture](#-pipeline-architecture) • [Model Comparison](#-model-comparison) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

This project implements an end-to-end California housing regression workflow. It combines DVC-managed data preparation, CV-safe scikit-learn preprocessing, model benchmarking and tuning, MLflow tracking, final test evaluation, and a FastAPI inference service.

- 🔄 **Data Pipeline Engineering**: Kaggle ingestion, validation, stratified splitting, cleaning, and deterministic feature engineering
- 🧠 **ML Pipeline**: Four configured regressors, 5-fold benchmarking, RandomizedSearchCV, final fitting, and evaluation
- 📊 **Reproducibility**: DVC stages and file-backed MLflow tracking
- 🔒 **Leakage Prevention**: Learned preprocessing remains inside the sklearn Pipeline
- 🚀 **Inference**: FastAPI single and batch prediction endpoints
- 🧪 **Testing**: 189 pytest test functions across data, ML, and API behavior

| Metric | Value |
|--------|-------|
| **Dataset** | California Housing (Kaggle, 20,640 districts) |
| **Target** | `median_house_value` (regression) |
| **Input Features** | 9 raw features (8 numeric + `ocean_proximity`) |
| **Pipeline Split** | 70% train / 15% validation / 15% test (stratified) |
| **Models Benchmarked** | 4 |
| **Best Model** | `random_forest` |
| **Test RMSE** | 47,408.2041 |
| **Test R²** | 0.8257 |
| **Tracking** | MLflow (`california_housing_regression`) |
| **Recorded MLflow Runs** | 17 (5 stages + per-model benchmarks) |
| **Python Support** | 3.9+ |

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

```python
from src.utils.logger import setup_logging
from src.utils.colab_setup import initialize_environment
import logging

setup_logging(level=logging.INFO)
project_root = initialize_environment(
    repo_name="california_housing_full_project",
    repo_owner="ebramrafat653-wq",
    install_deps=True,
)
```

The repository includes setup notebooks under `notebooks/` for environment initialization, profiling, splitting, EDA, cleaning, feature engineering, Power BI export, and preprocessing.

### Option 2: Local Development

```bash
git clone https://github.com/ebramrafat653-wq/california_housing_full_project.git
cd california_housing_full_project

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -e ".[all]"
```

If data is stored in the configured DVC remote:

```bash
dvc pull
```

Run the test suite:

```bash
pytest tests/ -v
```

---

## 🏗️ Pipeline Architecture

### Data Pipeline

```text
Raw Data (Kaggle)
      ↓
[1] INGESTION
      ↓
[2] VALIDATION
      ↓
[3] STRATIFIED SPLITTING (70/15/15)
      ↓
[4] CLEANING
      ↓
[5] DETERMINISTIC FEATURE ENGINEERING
      ↓
Clean train / validation / test datasets
```

### Key Data Transformations

| Stage | Transformation | Purpose |
|-------|---|---|
| **Validation** | Required columns, missing-value contract, categories, numeric bounds | Stop invalid data early |
| **Splitting** | Five target quantile strata | Preserve the target distribution across splits |
| **Cleaning** | Contract validation and split outputs | Keep learned preprocessing out of the data layer |
| **Feature Engineering** | Ratios and distances to San Francisco and Los Angeles | Add deterministic geographic and household features |
| **ML Pipeline** | Imputation, LOF, feature engineering, scaling, encoding, model | Fit every learned operation within each CV fold |

---

## 📊 Model Comparison

### Key Results

Random Forest was the strongest model on both cross-validation (CV RMSE 48,423.25, CV R² 0.8244) and the untouched test set (Test RMSE 47,408.20, Test R² 0.8257). Gradient Boosting came second with slightly worse accuracy but ≈3.6× faster CV fitting time (17.24s vs 61.74s), making it a strong speed-accuracy trade-off. Linear Regression and Ridge both plateaued around R² ≈ 0.62, confirming the housing-target relationship is fundamentally non-linear.

### Validation Benchmark (5-Fold CV on Train)

| Rank | Model | CV RMSE (mean) | CV RMSE (std) | CV MAE (mean) | CV R² (mean) | Fit Time (s) |
|:----:|:---|---:|---:|---:|---:|---:|
| 1 | **random_forest** | 48,423.2517 | 830.2330 | 31,278.9071 | 0.8244 | 61.74 |
| 2 | gradient_boosting | 50,118.7228 | 273.3605 | 34,261.4323 | 0.8118 | 17.24 |
| 3 | linear_regression | 71,175.7024 | 1,152.2162 | 51,717.5684 | 0.6203 | 0.41 |
| 4 | ridge | 71,179.0483 | 1,147.0636 | 51,719.1352 | 0.6202 | 0.60 |

_Lower CV RMSE is better. Values are from the successful MLflow benchmark runs._

### Final Test Evaluation

| Final Model | Test RMSE | Test MAE | Test R² |
|:---|---:|---:|---:|
| `random_forest` | 47,408.2041 | 29,466.7301 | **0.8257** |

_The test set was used once, only for final evaluation, on 3,096 samples._

---

## 🧠 ML Training Pipeline

The ML lifecycle is orchestrated by `pipelines/run_pipeline.py`:

1. **Stage 1 — Benchmarking:** Run 5-fold CV on Train for all enabled models and rank them by `val_rmse_mean`.
2. **Stage 2 — Tuning:** Run `RandomizedSearchCV` on Train only. The configured search uses 30 iterations and 5-fold CV.
3. **Stage 3 — Final Fit:** Fit the selected model on Train + Validation with the tuned parameters.
4. **Stage 4 — Test Evaluation:** Evaluate the saved final pipeline once on the untouched Test set.

The complete preprocessing chain is inside the sklearn Pipeline: imputation, LOF, deterministic feature engineering, scaling, encoding, and model fitting. No preprocessing is fitted outside that pipeline, and Test data is excluded from benchmarking, tuning, and final fitting.

---

## 🔬 MLflow Integration

MLflow is enabled in `configs/model_config.yaml`:

- **Experiment:** `california_housing_regression`
- **Tracking URI:** `file:./mlruns`
- **Parent run:** `california_housing_pipeline`
- **Nested stages:** `stage_1_benchmarking`, `stage_2_tuning`, `stage_3_final_fit`, and `stage_4_test_evaluation`
- **Recorded runs:** 17 in the local MLflow store, including successful and failed pipeline attempts

The benchmarking stage also creates nested per-model runs such as `benchmark_random_forest`.

Launch the local MLflow UI from the repository root:

```bash
mlflow ui --backend-store-uri ./mlruns
```

To disable tracking, set `mlflow.enabled` to `false` in `configs/model_config.yaml`. Comparison tables are generated from the tracked MLflow data and the artifacts under `reports/`.

### Configuration Highlights

| Setting | Value |
|---|---|
| Enabled models | `linear_regression`, `ridge`, `random_forest`, `gradient_boosting` |
| CV folds | 5 (shuffle=True, random_state=42) |
| Primary metric | `neg_root_mean_squared_error` (minimize) |
| Tuning | RandomizedSearchCV, 30 iterations, 5-fold CV |
| Final training strategy | Train + Validation |
| Test used for training | No |
| MLflow enabled | Yes |
| Tracking URI | `file:./mlruns` |
| Experiment name | `california_housing_regression` |
| Final artifact | `artifacts/final_model_pipeline.pkl` |

---

## 🔁 Reproducing Results

> All commands below must be run from the repository root.

### Full data + ML pipeline

```bash
dvc repro
```

This runs the DVC stages for ingestion, validation, splitting, cleaning, feature engineering, and `ml_pipeline`.

### ML pipeline only

```bash
python -m pipelines.run_pipeline
```

### Benchmarking only

```bash
python -m src.models.training
```

### Tune a specific model

```bash
python -m src.models.tuning random_forest
```

Configured model names are `linear_regression`, `ridge`, `random_forest`, and `gradient_boosting`; search spaces are defined for Ridge, Random Forest, and Gradient Boosting.

### Evaluate a saved model

```bash
python -m src.models.evaluation
```

The evaluation command expects the configured final pipeline artifact at `artifacts/final_model_pipeline.pkl`.

---

## 🗂️ Project Structure

```text
california_housing_full_project/
│
├── 📂 api/
│   ├── main.py                      # FastAPI application
│   ├── predict.py                   # API-to-inference adapter
│   └── schemas.py                   # Pydantic request/response contracts
│
├── 📂 src/
│   ├── data/
│   │   ├── ingestion.py             # Kaggle/DVC data ingestion
│   │   ├── validation.py             # Data quality validation
│   │   ├── splitting.py             # Stratified train/val/test split
│   │   ├── cleaning.py              # Clean split validation and outputs
│   │   ├── data_loader.py           # Environment-aware CSV loading
│   │   └── profiling.py             # Present, currently empty
│   ├── features/
│   │   ├── engineering.py           # Deterministic feature engineering
│   │   └── pipeline.py              # CV-safe sklearn pipeline construction
│   ├── models/
│   │   ├── model_factory.py         # Config-driven estimator creation
│   │   ├── training.py              # CV benchmarking and final fitting
│   │   ├── tuning.py                # RandomizedSearchCV tuning
│   │   ├── evaluation.py            # Final test evaluation and plots
│   │   └── predict.py               # Production inference engine
│   └── utils/
│       ├── logger.py                # Centralized logging
│       ├── colab_setup.py           # Colab environment bootstrap
│       ├── helpers.py               # Present, currently empty
│       └── paths.py                 # Project and DVC path management
│
├── 📂 configs/
│   ├── data_config.yaml             # Dataset and data-pipeline configuration
│   └── model_config.yaml            # Models, CV, tuning, MLflow, artifacts
│
├── 📂 pipelines/
│   └── run_pipeline.py              # ML lifecycle orchestrator
│
├── 📂 tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_cleaning.py
│   ├── test_data_loader.py
│   ├── test_engineering.py
│   ├── test_ingestion.py
│   ├── test_pipeline.py
│   └── test_validation.py
│
├── 📂 artifacts/
│   ├── cleaning/                      # Cleaning metadata artifacts
│   ├── preprocessing_pipeline.pkl     # Present: preprocessing pipeline
│   └── final_model_pipeline.pkl       # Expected: not yet committed
│
├── 📂 reports/
│   ├── cv_results.csv
│   ├── eda_report.html
│   ├── feature_importance.png
│   ├── metrics.json
│   ├── model_report.md
│   ├── pred_vs_actual.png
│   ├── residuals.png
│   ├── test_predictions.csv
│   ├── test_results.json
│   ├── training_report.json
│   └── tuning_report_random_forest.json
│
├── 📂 data/                         # DVC-managed data directories
├── 📂 notebooks/                    # Environment, EDA, preprocessing, and export notebooks
├── 📂 mlruns/                       # Local MLflow tracking store
├── 📂 docker/                       # Dockerfile and docker-compose.yml
├── dvc.yaml                         # DVC file with six declared stages
├── dvc.lock                         # DVC lockfile
├── Makefile                         # Install, test, lint, DVC, and cleanup commands
├── pyproject.toml                   # Packaging and dependency specification
├── LICENSE
└── README.md
```

> The DVC file contains six declared stages: ingestion, validation, splitting, cleaning, engineering, and `ml_pipeline`.

---

## ⚙️ Setup Guide

### Prerequisites

- **Python 3.9+**
- **Git**
- **DVC 3.40+** when reproducing the data pipeline
- **Kaggle API credentials** for fresh dataset ingestion
- **Google Account** when using Google Drive DVC storage or Colab

### Step 1: Clone Repository

```bash
git clone https://github.com/ebramrafat653-wq/california_housing_full_project.git
cd california_housing_full_project
```

### Step 2: Set Up Python Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install --upgrade pip setuptools wheel
pip install -e ".[all]"
```

### Step 3: Configure DVC

Configure the DVC remote appropriate for your environment, then pull tracked data:

```bash
dvc remote list
dvc pull
```

### Step 4: Configure Kaggle API

Download `kaggle.json` from https://www.kaggle.com/settings/account and place it in the location expected by your environment. In Colab, `src.utils.colab_setup` can configure credentials and the local DVC remote.

### Step 5: Run Data and ML Pipelines

```bash
# Data pipeline only
dvc repro

# ML lifecycle only, after clean data exists
python -m pipelines.run_pipeline

# Tests
pytest tests/ -v
```

---

## 🛠️ Development Commands

| Command | Action |
|---|---|
| `make help` | Print available targets |
| `make install` | Upgrade pip and install package (production deps) |
| `make install-dev` | Install package with dev + test dependencies |
| `make test` | Run full pytest suite |
| `make test-fast` | Run pytest excluding `integration` marker |
| `make lint` | Run `ruff check src/ tests/` |
| `make format` | Run `ruff format` then `ruff check --fix` |
| `make pipeline` | Run `dvc repro` (full data + ML lifecycle) |
| `make pull` / `make push` | Sync DVC remote |
| `make status` | Show DVC status |
| `make clean` | Remove caches, build artifacts, coverage files |

The Makefile maps `make pipeline` to `dvc repro`; use `python -m pipelines.run_pipeline` when only the ML lifecycle should run.

---

## 🚀 API Usage

```bash
# Start the API
uvicorn api.main:app --reload

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "longitude": -122.23,
    "latitude": 37.88,
    "housing_median_age": 41,
    "total_rooms": 880,
    "total_bedrooms": 129,
    "population": 322,
    "households": 126,
    "median_income": 8.3252,
    "ocean_proximity": "NEAR BAY"
  }'
```

---

## 🧪 Testing

The test suite contains 189 test functions:

| File | Coverage | Tests |
|---|---|---:|
| `tests/test_api.py` | FastAPI endpoints, validation, error handling, OpenAPI | 24 |
| `tests/test_cleaning.py` | Cleaning contracts, metadata, deferred preprocessing | 20 |
| `tests/test_data_loader.py` | Data stages, paths, loading behavior | 8 |
| `tests/test_engineering.py` | Ratios, distances, dropped columns, leakage safety | 31 |
| `tests/test_ingestion.py` | Kaggle/DVC helpers, reports, credentials, downloads | 44 |
| `tests/test_pipeline.py` | sklearn pipeline construction and behavior | 16 |
| `tests/test_validation.py` | Data quality rules and validation reports | 46 |

> `tests/test_data.py` exists as a placeholder but currently contains no test functions. It is excluded from the count above.

```bash
pytest tests/ -v
pytest tests/ -v -m "not integration"
pytest tests/test_api.py -v
pytest tests/ --cov=src --cov-report=html
```

---

## 🤝 Contributing

### Development Workflow

1. **Create Feature Branch**
   ```bash
   git checkout -b feat/new-feature
   ```
2. **Make Changes**
   - Modify code in `src/`, `api/`, or `tests/`
   - Update YAML configuration when needed
   - Add or update focused tests
3. **Test Locally**
   ```bash
   pytest tests/ -v
   ruff check src/ api/ tests/
   ruff format src/ api/ tests/
   ```
4. **Track New Data**
   ```bash
   dvc add data/new_data.csv
   git add data/new_data.csv.dvc
   ```
5. **Commit & Push**
   ```bash
   git add src/ api/ tests/ configs/ data/*.dvc
   git commit -m "feat: describe the change"
   git push origin feat/new-feature
   ```
6. **Open a Pull Request**

### Code Style Guidelines

- **Linting & Formatting**: Ruff (`ruff check` + `ruff format`)
- **Type checking**: mypy is configured but not enforced in CI
- **Testing**: pytest with coverage (`pytest --cov=src`)
- **Docstrings**: Document public behavior
- **Commits**: Use conventional commits (`feat:`, `fix:`, `test:`, `docs:`)

### ⚠️ Important Rules

- **Never commit raw data files**; commit DVC metadata instead.
- **Keep learned preprocessing inside the sklearn Pipeline**.
- **Never use Test data for model selection or fitting**.
- **Update tests and reports when changing pipeline behavior**.

---

## 📖 Documentation

- **[EDA Report](reports/eda_report.html)** — Exploratory Data Analysis visualizations
- **[Model Report](reports/model_report.md)** — Model report placeholder currently present in the repository
- **[Config Reference](configs/data_config.yaml)** — Data-pipeline parameters
- **[Model Config](configs/model_config.yaml)** — Models, CV, tuning, MLflow, and artifacts
- **[API Schema](api/schemas.py)** — Request and response contracts

---

## 🔧 Troubleshooting

### DVC Issues

| Problem | Solution |
|---------|----------|
| `dvc pull` fails with "remote not found" | Run `dvc remote list` and configure a remote for the current environment. |
| `dvc pull` fails with an authentication error | Check the configured remote credentials and Google Drive access. |
| Data stages are out of date | Run `dvc status`, then `dvc repro`. |

### Data Issues

| Problem | Solution |
|---------|----------|
| `KeyError: 'median_house_value'` | Verify the input schema and `configs/data_config.yaml`. |
| Validation errors after loading | Run the validation stage and inspect `reports/validation/` when generated. |
| Missing values in the pipeline | `total_bedrooms` is the only allowed missing feature; imputation occurs inside the ML pipeline. |
| Stratified split fails | Check that the target exists, is numeric, and contains no nulls. |

### MLflow Issues

| Problem | Solution |
|---------|----------|
| MLflow is not recording runs | Confirm `mlflow.enabled: true` and install the `pipeline` extra. |
| UI shows no experiments | Launch `mlflow ui --backend-store-uri ./mlruns` from the repository root. |
| Tracking is not wanted | Set `mlflow.enabled: false` in `configs/model_config.yaml`. |

### API Issues

| Problem | Solution |
|---------|----------|
| API returns 503 or fails to load model | Confirm `artifacts/final_model_pipeline.pkl` exists; run `python -m pipelines.run_pipeline` to generate it. |

```bash
uvicorn api.main:app --reload
```

The API exposes `/`, `/health`, `/predict`, `/predict/batch`, `/docs`, and `/redoc`. Prediction requires the configured final artifact `artifacts/final_model_pipeline.pkl`.

---

## 📊 Project Status

### ✅ Completed

- [x] Data ingestion and validation
- [x] Stratified train/validation/test splitting
- [x] Data cleaning and deterministic feature engineering
- [x] Leakage-safe sklearn preprocessing pipeline
- [x] Model benchmarking across four regressors
- [x] Hyperparameter tuning
- [x] Final model training
- [x] Final test evaluation and reports
- [x] MLflow experiment tracking
- [x] FastAPI prediction API
- [x] Docker files
- [x] CI workflow configuration
- [x] Google Colab setup

### 🔄 In Progress / TODO

- [ ] Populate `reports/model_report.md` (currently empty)
- [ ] Generate and commit `artifacts/final_model_pipeline.pkl` (required by the API and evaluation stage)

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
- **Inspiration**: ML engineering best practices from production systems
- **Tools**: [DVC](https://dvc.org/), [MLflow](https://mlflow.org/), [scikit-learn](https://scikit-learn.org/), [pandas](https://pandas.pydata.org/)

---

**Last Updated**: 2026-09-13 | **Status**: Data + ML Pipeline Complete | **Python**: 3.9+
