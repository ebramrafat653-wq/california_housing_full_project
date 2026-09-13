# 🏠 California Housing Full Project

<div align="center">

[![Python](https://img.shields.io/badge/python-3.9+-3776ab.svg?logo=python&logoColor=white)](https://python.org)
[![DVC](https://img.shields.io/badge/DVC-3.40+-945dd6.svg?logo=dvc&logoColor=white)](https://dvc.org)
[![MLflow](https://img.shields.io/badge/MLflow-2.8+-0194E2.svg?logo=mlflow&logoColor=white)](https://mlflow.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-f7931e.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License](https://img.shields.io/badge/license-MIT-28a745.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/test%20functions-189-4c1.svg)](tests/)
[![Colab Ready](https://img.shields.io/badge/Google%20Colab-Ready-F9AB00.svg?logo=googlecolab)](https://colab.research.google.com)

**Production-oriented end-to-end data and machine learning pipeline**  
for California housing price prediction, with reproducible data processing, leakage-safe model training, experiment tracking, automated testing, and REST inference.

</div>

---

## 🏆 Final Test Performance

| Metric | Result |
|---|---:|
| **Model** | Random Forest Regressor |
| **Test RMSE** | 47,408.2041 |
| **Test MAE** | 29,466.7301 |
| **Test R²** | **0.8257** |
| **Test samples** | 3,096 |

> The test set was used **once**, only for final evaluation, on a model selected entirely from cross-validation on the training data.

### Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` · `MLflow` · `DVC` · `FastAPI` · `Pydantic` · `pytest` · `Ruff` · `Docker` · `Google Colab`

---

## 📋 Overview

This project implements an end-to-end machine learning workflow for predicting California district housing values.

The project is designed around **reproducibility**, **leakage prevention**, **modularity**, and **production-oriented ML engineering practices**, rather than a notebook-only modeling workflow.

The complete workflow covers:

- Data ingestion and validation
- Stratified train/validation/test splitting
- Data cleaning and contract validation
- Deterministic feature engineering
- Leakage-safe scikit-learn preprocessing
- Model benchmarking and cross-validation
- Hyperparameter tuning
- Final model training
- Final evaluation on an untouched test set
- MLflow experiment tracking
- DVC pipeline orchestration and data versioning
- Automated testing with pytest
- FastAPI inference
- Docker configuration
- CI workflow configuration
- Google Colab environment setup

### Project Highlights

- 🔄 **Reproducible Data Pipeline** — DVC-managed multi-stage workflow
- 🛡️ **Leakage Prevention** — learned preprocessing is fitted inside the scikit-learn pipeline
- 🧪 **Model Benchmarking** — four regression models compared using 5-fold cross-validation
- 🎯 **Hyperparameter Tuning** — RandomizedSearchCV with a dedicated tuning stage
- 📊 **Final Evaluation** — test data is held out until final evaluation
- 🔬 **Experiment Tracking** — MLflow tracks pipeline stages, models, parameters, and metrics
- 🚀 **Inference API** — FastAPI exposes single and batch prediction endpoints
- 🧪 **Automated Testing** — 189 pytest test functions covering data, ML, and API behavior
- 🐳 **Containerization Ready** — Docker configuration included
- ☁️ **Colab Ready** — environment bootstrap utilities for reproducible development

---

## 🎯 Project Objective

The objective is to build a reliable regression system that predicts `median_house_value` from nine raw housing features.

Rather than focusing only on model accuracy, the project focuses on building a complete ML lifecycle:

```text
Raw Data
   ↓
Ingestion
   ↓
Validation
   ↓
Stratified Splitting
   ↓
Cleaning
   ↓
Feature Engineering
   ↓
Leakage-Safe ML Pipeline
   ↓
Benchmarking
   ↓
Hyperparameter Tuning
   ↓
Final Training
   ↓
Test Evaluation
   ↓
Serialized Model
   ↓
FastAPI Inference
```

---

## 📊 Dataset

The project uses the California Housing Prices dataset containing information about California districts.

| Property | Value |
|---|---|
| Dataset | California Housing Prices |
| Source | Kaggle |
| Observations | 20,640 |
| Target | `median_house_value` |
| Raw input features | 9 |
| Task | Regression |
| Train split | 70% |
| Validation split | 15% |
| Test split | 15% |
| Random state | 42 |

### Input Features

The model receives the following nine raw features:

- `longitude`
- `latitude`
- `housing_median_age`
- `total_rooms`
- `total_bedrooms`
- `population`
- `households`
- `median_income`
- `ocean_proximity`

The target variable is `median_house_value`.

---

## 🏗️ Architecture

### High-Level Architecture

```text
                    ┌──────────────────────┐
                    │  California Housing  │
                    │       Dataset        │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Ingestion       │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │      Validation      │
                    └──────────┬───────────┘
                               │
                               ▼
                    ┌──────────────────────┐
                    │ Stratified Splitting │
                    │     70 / 15 / 15     │
                    └──────────┬───────────┘
                               │
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
            Train            Valid            Test
              │                │                │
              └────────┬───────┘                │
                       ▼                        │
             ┌──────────────────────┐           │
             │ Feature Engineering  │           │
             └──────────┬───────────┘           │
                        ▼                       │
             ┌──────────────────────┐           │
             │ sklearn ML Pipeline  │           │
             │                      │           │
             │ • Imputation         │           │
             │ • LOF                │           │
             │ • Features           │           │
             │ • Scaling            │           │
             │ • Encoding           │           │
             │ • Estimator          │           │
             └──────────┬───────────┘           │
                        ▼                       │
             ┌──────────────────────┐           │
             │ Benchmarking / Tuning│           │
             └──────────┬───────────┘           │
                        ▼                       │
             ┌──────────────────────┐           │
             │ Final Model Training │           │
             │ Train + Validation   │           │
             └──────────┬───────────┘           │
                        ▼                       │
             ┌──────────────────────┐           │
             │ Final Model Artifact │           │
             └──────────┬───────────┘           │
                        │                       │
                        └───────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │   Final Test Set     │
                         │   One-Time Eval      │
                         └──────────────────────┘
```

---

## 🔄 Data Pipeline

The data workflow is orchestrated through DVC.

```text
Raw Data
   ↓
[1] Ingestion
   ↓
[2] Validation
   ↓
[3] Stratified Splitting
   ↓
[4] Cleaning
   ↓
[5] Feature Engineering
   ↓
Clean Train / Validation / Test
   ↓
ML Pipeline
```

The DVC pipeline contains six declared stages:

1. `ingestion`
2. `validation`
3. `splitting`
4. `cleaning`
5. `engineering`
6. `ml_pipeline`

Run the complete pipeline with:

```bash
dvc repro
```

---

## 🛡️ Data Validation

The validation layer checks the dataset before downstream processing.

Validation includes:

- Required columns
- Target availability
- Missing-value rules
- Numeric constraints
- Categorical values
- Data types
- Value ranges
- Schema consistency

Only `total_bedrooms` is allowed to contain missing values among the model's raw input features.

This prevents invalid or unexpected data from silently entering the ML pipeline.

---

## ✂️ Data Splitting

The dataset is divided into:

- 70% Train
- 15% Validation
- 15% Test

The split uses target-based stratification to preserve the distribution of housing prices across the three datasets.

Configuration:

```yaml
random_state: 42
```

The test set remains isolated throughout:

- model benchmarking
- hyperparameter tuning
- final model fitting

The test set is used only once for final evaluation.

---

## 🧹 Data Cleaning & Feature Engineering

Feature engineering includes deterministic transformations designed to provide additional information to the regression models.

### Engineered Features

Examples include:

- rooms per household
- bedrooms per room
- population per household
- distance to San Francisco
- distance to Los Angeles

The project separates **deterministic feature generation** from **learned preprocessing**.

This distinction is important because transformations that learn parameters from data must not be fitted using validation or test information.

---

## 🔒 Leakage-Safe Machine Learning Pipeline

A major design goal of this project is preventing preprocessing leakage.

The learned preprocessing operations are contained inside the scikit-learn pipeline.

Conceptually:

```text
Raw Features
     ↓
Imputation
     ↓
LOF Transformation
     ↓
Feature Engineering
     ↓
Scaling
     ↓
One-Hot Encoding
     ↓
Model
```

The pipeline is fitted independently inside each cross-validation fold.

This ensures that validation folds do not influence preprocessing parameters learned from the training fold.

### Why this matters

Incorrect preprocessing can create information leakage and produce unrealistically optimistic validation results.

The project therefore avoids fitting learned preprocessing outside the cross-validation pipeline.

---

## 🤖 Model Benchmarking

Four regression models are benchmarked:

1. Linear Regression
2. Ridge Regression
3. Random Forest Regressor
4. Gradient Boosting Regressor

Benchmarking uses:

- 5-Fold Cross-Validation

Primary metric:

- RMSE

Additional metrics:

- MAE
- R²

---

## 📈 Model Comparison

### Cross-Validation Results

| Rank | Model | CV RMSE Mean | CV RMSE Std | CV MAE Mean | CV R² Mean | Fit Time |
|:---:|:---|---:|---:|---:|---:|---:|
| 1 | **Random Forest** | 48,423.25 | 830.23 | 31,278.91 | 0.8244 | 61.74s |
| 2 | Gradient Boosting | 50,118.72 | 273.36 | 34,261.43 | 0.8118 | 17.24s |
| 3 | Linear Regression | 71,175.70 | 1,152.22 | 51,717.57 | 0.6203 | 0.41s |
| 4 | Ridge | 71,179.05 | 1,147.06 | 51,719.14 | 0.6202 | 0.60s |

Lower RMSE is better. Higher R² is better.

Random Forest achieved the best overall predictive performance during cross-validation.

Gradient Boosting provided a competitive accuracy/speed trade-off, with substantially lower fitting time.

---

## 🎯 Hyperparameter Tuning

After model benchmarking, the selected model is tuned using:

- RandomizedSearchCV

Configuration:

- Search iterations: 30
- Cross-validation folds: 5
- Primary metric: Negative RMSE

Tuning is performed using **training data only**.

The test set is not used for hyperparameter selection.

Search spaces are configured for:

- Ridge
- Random Forest
- Gradient Boosting

The final selected model is then retrained using the combined **Train + Validation** dataset.

---

## 🏆 Final Model

The final selected model is:

- **Random Forest Regressor**

The final pipeline is serialized as:

```text
artifacts/final_model_pipeline.pkl
```

The serialized artifact contains the **complete fitted inference pipeline** rather than only the estimator.

This allows inference to start from the same raw input schema used during training.

---

## 🧪 Final Test Evaluation

The final model is evaluated on the untouched test set.

### Final Results

| Metric | Result |
|---|---:|
| Test samples | 3,096 |
| RMSE | 47,408.2041 |
| MAE | 29,466.7301 |
| R² | 0.8257 |

### Interpretation

The final model explains approximately **82.57%** of the variance in the test target according to R².

The test set is intentionally excluded from:

- benchmarking
- hyperparameter tuning
- final fitting

This keeps the reported test performance as an **independent final evaluation**.

---

## 📊 Evaluation Reports

The evaluation stage generates artifacts such as:

```text
reports/
├── cv_results.csv
├── test_predictions.csv
├── test_results.json
├── training_report.json
├── tuning_report_random_forest.json
├── metrics.json
├── feature_importance.png
├── pred_vs_actual.png
└── residuals.png
```

These artifacts provide both numerical and visual evaluation of the trained model.

---

## 🔬 MLflow Experiment Tracking

MLflow is integrated into the ML lifecycle.

**Experiment:**

```text
california_housing_regression
```

**Tracking URI:**

```text
file:./mlruns
```

**Pipeline stages:**

```text
california_housing_pipeline
│
├── stage_1_benchmarking
│   ├── benchmark_linear_regression
│   ├── benchmark_ridge
│   ├── benchmark_random_forest
│   └── benchmark_gradient_boosting
│
├── stage_2_tuning
│
├── stage_3_final_fit
│
└── stage_4_test_evaluation
```

The repository currently contains **17 recorded MLflow runs**, including benchmark and pipeline runs.

MLflow tracks information such as:

- model parameters
- metrics
- run status
- pipeline stages
- experiment metadata
- model-related artifacts

Launch the MLflow UI:

```bash
mlflow ui --backend-store-uri ./mlruns
```

Then open the local MLflow interface provided by the command.

---

## 📦 DVC & Reproducibility

DVC is used to manage the data and pipeline lifecycle.

The project tracks the relationship between:

```text
Data
 ↓
Processing
 ↓
Features
 ↓
Training
 ↓
Model
 ↓
Reports
```

The DVC pipeline allows the workflow to be reproduced using:

```bash
dvc repro
```

Check pipeline status:

```bash
dvc status
```

Pull tracked data:

```bash
dvc pull
```

Push updated tracked data:

```bash
dvc push
```

---

## 🚀 FastAPI Inference

The trained model is exposed through a FastAPI service.

Architecture:

```text
Client
  ↓
FastAPI
  ↓
Pydantic Validation
  ↓
Inference Adapter
  ↓
Production Inference Engine
  ↓
Fitted sklearn Pipeline
  ↓
Prediction
```

The API supports:

| Method | Endpoint |
|---|---|
| GET | `/` |
| GET | `/health` |
| POST | `/predict` |
| POST | `/predict/batch` |
| GET | `/docs` |
| GET | `/redoc` |

Start the API:

```bash
uvicorn api.main:app --reload
```

---

## 📥 Prediction Example

### Request

```json
{
  "longitude": -122.23,
  "latitude": 37.88,
  "housing_median_age": 41,
  "total_rooms": 880,
  "total_bedrooms": 129,
  "population": 322,
  "households": 126,
  "median_income": 8.3252,
  "ocean_proximity": "NEAR BAY"
}
```

### Endpoint

```text
POST /predict
```

The API validates the input before passing the raw features to the fitted production pipeline.

The API does **not** manually reproduce training transformations.

---

## 🧠 Inference Design

The inference engine is implemented separately from the API layer.

Responsibilities include:

- Loading the fitted model pipeline
- Validating input structure
- Rejecting unexpected fields
- Preventing target leakage
- Supporting single predictions
- Supporting batch predictions
- Reusing the loaded model within the process

The model is loaded from:

```text
artifacts/final_model_pipeline.pkl
```

The inference layer does **not**:

- fit models
- retrain preprocessing
- perform manual feature engineering
- fit LOF
- modify the production pipeline

This keeps the API layer thin and the inference behavior consistent with training.

---

## 🧪 Testing

The project contains **189 pytest test functions**.

| Test Module | Main Coverage | Test Functions |
|---|---|---:|
| `test_api.py` | API endpoints, validation, errors, OpenAPI | 24 |
| `test_cleaning.py` | Cleaning contracts and metadata | 20 |
| `test_data_loader.py` | Data loading and paths | 8 |
| `test_engineering.py` | Features, ratios, distances, leakage safety | 31 |
| `test_ingestion.py` | Ingestion, DVC/Kaggle helpers | 44 |
| `test_pipeline.py` | sklearn pipeline behavior | 16 |
| `test_validation.py` | Data quality and validation rules | 46 |

Run the full suite:

```bash
pytest tests/ -v
```

Run tests excluding integration tests:

```bash
pytest tests/ -v -m "not integration"
```

Run API tests:

```bash
pytest tests/test_api.py -v
```

Run with coverage:

```bash
pytest tests/ --cov=src --cov-report=html
```

---

## 🐳 Docker

Docker configuration is included under:

```text
docker/
```

The Docker setup is intended to package the inference service and its runtime dependencies consistently across environments.

The project is therefore structured so that the trained model can be moved from development into a containerized inference environment.

> Production cloud deployment is not currently claimed as part of this repository.

---

## ⚙️ CI Workflow

A CI workflow configuration is included under:

```text
.github/workflows/
```

The repository also provides development commands through the Makefile.

Typical development checks include:

```bash
make test
make lint
make format
```

The project uses:

- pytest for testing
- Ruff for linting and formatting
- mypy configuration for type checking

Type checking is configured but is not currently enforced as a blocking CI requirement.

---

## 📁 Project Structure

```text
california_housing_full_project/
│
├── api/
│   ├── main.py
│   ├── predict.py
│   └── schemas.py
│
├── src/
│   ├── data/
│   │   ├── ingestion.py
│   │   ├── validation.py
│   │   ├── splitting.py
│   │   ├── cleaning.py
│   │   ├── data_loader.py
│   │   └── profiling.py
│   │
│   ├── features/
│   │   ├── engineering.py
│   │   └── pipeline.py
│   │
│   ├── models/
│   │   ├── model_factory.py
│   │   ├── training.py
│   │   ├── tuning.py
│   │   ├── evaluation.py
│   │   └── predict.py
│   │
│   └── utils/
│       ├── logger.py
│       ├── colab_setup.py
│       ├── helpers.py
│       └── paths.py
│
├── configs/
│   ├── data_config.yaml
│   └── model_config.yaml
│
├── pipelines/
│   └── run_pipeline.py
│
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_cleaning.py
│   ├── test_data_loader.py
│   ├── test_engineering.py
│   ├── test_ingestion.py
│   ├── test_pipeline.py
│   └── test_validation.py
│
├── artifacts/
│   ├── cleaning/
│   └── final_model_pipeline.pkl
│
├── reports/
│   ├── cv_results.csv
│   ├── eda_report.html
│   ├── feature_importance.png
│   ├── metrics.json
│   ├── pred_vs_actual.png
│   ├── residuals.png
│   ├── test_predictions.csv
│   ├── test_results.json
│   ├── training_report.json
│   └── tuning_report_random_forest.json
│
├── data/
├── notebooks/
├── docker/
├── mlruns/
│
├── .github/
│   └── workflows/
│
├── dvc.yaml
├── dvc.lock
├── Makefile
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## ⚡ Quick Start

### Option 1 — Google Colab

The repository includes environment setup utilities for Google Colab.

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

### Option 2 — Local Development

Clone the repository:

```bash
git clone https://github.com/ebramrafat653-wq/california_housing_full_project.git
cd california_housing_full_project
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

Linux / macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install --upgrade pip setuptools wheel
pip install -e ".[all]"
```

If the required DVC remote is configured:

```bash
dvc pull
```

Run tests:

```bash
pytest tests/ -v
```

---

## 🔁 Reproducing the Project

### Complete Pipeline

Run the complete DVC workflow:

```bash
dvc repro
```

This executes:

```text
Ingestion
→ Validation
→ Splitting
→ Cleaning
→ Feature Engineering
→ ML Pipeline
```

### ML Lifecycle Only

After the processed datasets are available:

```bash
python -m pipelines.run_pipeline
```

### Benchmarking

```bash
python -m src.models.training
```

### Tune a Model

```bash
python -m src.models.tuning random_forest
```

Available configured models:

- `linear_regression`
- `ridge`
- `random_forest`
- `gradient_boosting`

### Evaluation

```bash
python -m src.models.evaluation
```

---

## 🛠️ Development Commands

| Command | Purpose |
|---|---|
| `make help` | Show available commands |
| `make install` | Install production dependencies |
| `make install-dev` | Install development and test dependencies |
| `make test` | Run the complete test suite |
| `make test-fast` | Run non-integration tests |
| `make lint` | Run Ruff checks |
| `make format` | Format and fix code with Ruff |
| `make pipeline` | Run `dvc repro` |
| `make pull` | Pull DVC-tracked data |
| `make push` | Push DVC-tracked data |
| `make status` | Show DVC status |
| `make clean` | Remove local caches and generated build files |

---

## ⚙️ Configuration

Model and training behavior is controlled through:

```text
configs/model_config.yaml
```

The configuration defines:

- target variable
- train/validation/test paths
- random state
- enabled models
- cross-validation settings
- evaluation metrics
- hyperparameter search
- MLflow settings
- final model artifact paths

Example:

```yaml
task: regression
target: median_house_value
random_state: 42
```

The data workflow is configured through:

```text
configs/data_config.yaml
```

This keeps project behavior configurable without hard-coding operational parameters throughout the codebase.

---

## 📚 Documentation & Reports

The repository contains generated analysis and model artifacts under:

```text
reports/
```

Important outputs include:

- EDA report
- cross-validation results
- training report
- tuning report
- final test metrics
- test predictions
- residual analysis
- predicted-vs-actual visualization
- feature importance

The configuration files also serve as references for:

```text
configs/data_config.yaml
configs/model_config.yaml
```

---

## 🔐 Important ML Engineering Rules

This project follows several explicit rules.

### 1. Never use test data for model selection

The test set is reserved for final evaluation.

### 2. Keep learned preprocessing inside the ML pipeline

Learned transformations must be fitted only on the relevant training data.

### 3. Prevent target leakage

Target-derived information must not be used as an input feature during model training.

### 4. Keep inference consistent with training

The API accepts raw features and relies on the serialized fitted pipeline for preprocessing and prediction.

### 5. Version data and pipeline state

DVC is used to maintain reproducibility of data-dependent workflows.

---

## 📌 Project Status

### ✅ Completed

- [x] Data ingestion
- [x] Data validation
- [x] Stratified train/validation/test splitting
- [x] Data cleaning
- [x] Deterministic feature engineering
- [x] Leakage-safe scikit-learn preprocessing
- [x] Four-model benchmarking
- [x] 5-fold cross-validation
- [x] Hyperparameter tuning
- [x] Final model training
- [x] Final test evaluation
- [x] Evaluation reports
- [x] MLflow experiment tracking
- [x] DVC pipeline
- [x] FastAPI inference service
- [x] Single prediction endpoint
- [x] Batch prediction endpoint
- [x] API validation and contracts
- [x] Automated tests
- [x] Docker configuration
- [x] CI workflow configuration
- [x] Google Colab setup

### 🔄 Remaining Production Extensions

The core ML project and inference API are complete.

The following are considered extensions beyond the current core project:

- [ ] Deploy the API to a cloud platform
- [ ] Add production monitoring
- [ ] Add data/model drift detection
- [ ] Add automated retraining triggers
- [ ] Add a production model registry workflow
- [ ] Add end-to-end deployed API monitoring

These are intentionally treated as production extensions, not prerequisites for the completed ML pipeline.

---

## 🚧 Production Scope

The current repository should be viewed as a:

> **Production-oriented ML project with a container-ready REST inference service**

rather than claiming to be a fully deployed production system.

The project demonstrates the engineering practices required to move from experimentation toward production:

```text
Data
 ↓
Reproducible Pipeline
 ↓
Validated Features
 ↓
Leakage-Safe Training
 ↓
Experiment Tracking
 ↓
Model Evaluation
 ↓
Serialized Model
 ↓
REST API
 ↓
Containerization
```

Cloud deployment, monitoring, and automated retraining are intentionally left as future production extensions.

---

## 🧭 Future Improvements

Potential future improvements include:

### Deployment

Deploy the FastAPI service to a cloud environment.

### Monitoring

Add:

- latency monitoring
- request/error monitoring
- prediction distribution monitoring
- data drift detection
- model performance monitoring

### Model Lifecycle

Add:

- model registry
- model version promotion
- automated retraining
- scheduled evaluation
- rollback strategy

### Infrastructure

Potential future additions:

- CI/CD deployment pipeline
- cloud object storage
- managed MLflow
- container registry
- orchestration

These improvements are outside the current core ML implementation.

---

## 🧩 Repository Design Principles

The project follows these principles:

### Modularity

Each stage has a focused responsibility.

### Reproducibility

Data and pipeline stages are tracked through DVC.

### Configuration-driven behavior

Models and pipeline behavior are controlled through YAML configuration.

### Leakage prevention

Learned preprocessing is kept inside the training pipeline.

### Testability

Core data, ML, and API behaviors are covered by automated tests.

### Separation of concerns

Data processing, model training, evaluation, inference, and API serving are separated into dedicated modules.

### Production-oriented inference

The API consumes raw features and delegates transformation and prediction to the fitted production pipeline.

---

## 🤝 Contributing

Contributions should follow the existing project structure.

### 1. Create a feature branch

```bash
git checkout -b feat/new-feature
```

### 2. Implement the change

Modify the relevant modules under:

```text
src/
api/
configs/
tests/
```

### 3. Run tests

```bash
pytest tests/ -v
```

### 4. Run linting

```bash
ruff check src/ api/ tests/
```

### 5. Format code

```bash
ruff format src/ api/ tests/
```

### 6. Update DVC when data changes

```bash
dvc add data/new_data.csv
```

### 7. Commit changes

```bash
git add .
git commit -m "feat: describe the change"
```

### 8. Push the branch

```bash
git push origin feat/new-feature
```

---

## 🐛 Troubleshooting

### DVC remote not configured

Check configured remotes:

```bash
dvc remote list
```

Then configure the appropriate remote for the environment.

### DVC pull fails

Try:

```bash
dvc status
dvc pull
```

Check authentication and remote configuration if required.

### API cannot load the model

Make sure the final model artifact exists:

```text
artifacts/final_model_pipeline.pkl
```

If it does not exist locally, run the ML pipeline:

```bash
python -m pipelines.run_pipeline
```

### MLflow shows no experiments

Verify:

```yaml
mlflow:
  enabled: true
```

Then launch:

```bash
mlflow ui --backend-store-uri ./mlruns
```

---

## 📖 References

### Dataset

California Housing Prices dataset from Kaggle:

https://www.kaggle.com/datasets/camnugent/california-housing-prices

### Main Technologies

- Python
- pandas
- NumPy
- scikit-learn
- DVC
- MLflow
- FastAPI
- Pydantic
- pytest
- Ruff
- Docker
- Google Colab

---

## 📄 License

This project is licensed under the MIT License.

See [LICENSE](LICENSE) for details.

---

## 👤 Author

**Ebram Rafat**

Computer Science Graduate | Machine Learning / Data Science

GitHub: https://github.com/ebramrafat653-wq

---

<div align="center">

**California Housing Full Project**

End-to-end data and machine learning pipeline focused on reproducibility, leakage prevention, model evaluation, and production-oriented inference.

*Last Updated: September 13, 2026*

</div>