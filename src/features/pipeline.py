# =============================================================================
# src/features/pipeline.py
# California Housing Project — Preprocessing Pipeline
#
# RESPONSIBILITY:
#   Build, fit, transform, and save the preprocessing pipeline.
#
# PIPELINE POSITION:
#
#   cleaning.py
#       ↓
#   train_clean.csv / val_clean.csv / test_clean.csv
#       ↓
#   engineering.py
#       ↓
#   train_feat.csv / val_feat.csv / test_feat.csv
#       ↓
#   pipeline.py
#       ↓
#   X_train / X_val / X_test + fitted pipeline
#       ↓
#   training.py
#
# IMPORTANT:
#   - FIT is performed on X_train ONLY.
#   - Validation and test are transformed using train-fitted parameters.
#   - Target-derived metadata (is_capped) is explicitly excluded.
#   - This module does NOT manage DVC.
#   - This module does NOT manage Git.
#   - DVC orchestration will be handled later by dvc.yaml.
# =============================================================================

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    RobustScaler,
    StandardScaler,
)

from src.utils.logger import get_logger
from src.utils.paths import PROJECT_DIR

logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

_TARGET: str = "median_house_value"
_DEFAULT_CONFIG_PATH: Path = PROJECT_DIR / "configs" / "data_config.yaml"


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================

class PipelineError(Exception):
    """Raised when preprocessing cannot continue safely."""


# =============================================================================
# CONFIGURATION LOADER (FAIL-FAST)
# =============================================================================

def load_features_config(
    config_path: Path = _DEFAULT_CONFIG_PATH,
) -> dict[str, list[str]]:
    """
    Load feature lists from data_config.yaml.
    Fails fast if the config is missing or malformed.
    """
    if not config_path.exists():
        raise PipelineError(f"Config file not found: {config_path}")

    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        raise PipelineError(f"Failed to parse YAML config: {config_path}") from exc

    features_cfg = cfg.get("features_config")
    if not isinstance(features_cfg, dict):
        raise PipelineError(
            "'features_config' section is missing or invalid in data_config.yaml."
        )

    return {
        "std": features_cfg.get("numerical_standard_scaler", []),
        "robust": features_cfg.get("numerical_robust_scaler", []),
        "cat": features_cfg.get("categorical_one_hot", []),
        "passthrough": features_cfg.get("passthrough", []),
    }


# =============================================================================
# RESULT
# =============================================================================

@dataclass
class PipelineResult:
    """
    Container for the result of the preprocessing pipeline.
    """

    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray

    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    feature_names_out: list[str] = field(
        default_factory=list
    )

    pipeline_path: Path | None = None

    n_features: int = 0

    timestamp: datetime = field(
        default_factory=datetime.now
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def summary(self) -> str:
        """Return a readable summary of preprocessing results."""

        lines = [
            "=" * 60,
            "PREPROCESSING PIPELINE RESULT",
            "=" * 60,
            f"X_train : {self.X_train.shape}",
            f"X_val   : {self.X_val.shape}",
            f"X_test  : {self.X_test.shape}",
            f"y_train : {len(self.y_train):,}",
            f"y_val   : {len(self.y_val):,}",
            f"y_test  : {len(self.y_test):,}",
            f"Features: {self.n_features}",
            (
                "Pipeline: "
                f"{self.pipeline_path or 'not saved'}"
            ),
        ]

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")

            for warning in self.warnings:
                lines.append(f"  - {warning}")

        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# COLUMN RESOLUTION
# =============================================================================

def resolve_columns(
    df: pd.DataFrame,
    config: dict[str, list[str]],
) -> dict[str, list[str]]:
    """
    Resolve expected columns against the actual DataFrame.

    Returns:
        {
            "std": [...],
            "robust": [...],
            "cat": [...],
            "passthrough": [...]
        }

    Missing expected columns are reported as errors rather than silently
    ignored, because silently dropping features can hide pipeline bugs.
    """

    all_cols = set(df.columns)
    resolved: dict[str, list[str]] = {}

    for group_name, columns in config.items():

        missing = [
            column
            for column in columns
            if column not in all_cols
        ]

        if missing:
            raise PipelineError(
                f"Missing columns for '{group_name}': "
                f"{missing}"
            )

        resolved[group_name] = list(columns)

    logger.info(
        "Columns resolved successfully | "
        f"std={len(resolved['std'])}, "
        f"robust={len(resolved['robust'])}, "
        f"cat={len(resolved['cat'])}, "
        f"passthrough={len(resolved['passthrough'])}"
    )

    return resolved


# =============================================================================
# PIPELINE BUILDER
# =============================================================================

def build_pipeline(
    cols: dict[str, list[str]],
) -> Pipeline:
    """
    Build an unfitted sklearn preprocessing pipeline.

    Transformations:

        StandardScaler
            ↓
        Standard numeric features

        RobustScaler
            ↓
        median_income

        OneHotEncoder
            ↓
        ocean_proximity

        passthrough
            ↓
        lof_outlier (is_capped is strictly excluded)

    Any unlisted column is dropped.
    This prevents the target from entering X.
    """

    transformers = []

    # -------------------------------------------------------------------------
    # StandardScaler
    # -------------------------------------------------------------------------

    if cols["std"]:
        transformers.append(
            (
                "standard_scaler",
                StandardScaler(),
                cols["std"],
            )
        )

    # -------------------------------------------------------------------------
    # RobustScaler
    # -------------------------------------------------------------------------

    if cols["robust"]:
        transformers.append(
            (
                "robust_scaler",
                RobustScaler(),
                cols["robust"],
            )
        )

    # -------------------------------------------------------------------------
    # OneHotEncoder
    # -------------------------------------------------------------------------

    if cols["cat"]:
        transformers.append(
            (
                "one_hot_encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                    drop=None,
                ),
                cols["cat"],
            )
        )

    # -------------------------------------------------------------------------
    # Passthrough
    # -------------------------------------------------------------------------

    if cols["passthrough"]:
        transformers.append(
            (
                "passthrough",
                "passthrough",
                cols["passthrough"],
            )
        )

    if not transformers:
        raise PipelineError(
            "No preprocessing transformers were created."
        )

    column_transformer = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                column_transformer,
            )
        ]
    )

    logger.info(
        f"Preprocessing pipeline built with "
        f"{len(transformers)} transformer groups."
    )

    return pipeline


# =============================================================================
# INPUT VALIDATION
# =============================================================================

def validate_split_columns(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target: str = _TARGET,
) -> None:
    """
    Validate that train/validation/test have compatible schemas.
    """

    for name, df in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:

        if not isinstance(df, pd.DataFrame):
            raise PipelineError(
                f"{name} must be a pandas DataFrame."
            )

        if df.empty:
            raise PipelineError(
                f"{name} DataFrame is empty."
            )

        if target not in df.columns:
            raise PipelineError(
                f"Target '{target}' is missing from {name}."
            )

    train_features = set(train.columns) - {target}
    val_features = set(val.columns) - {target}
    test_features = set(test.columns) - {target}

    if train_features != val_features:
        raise PipelineError(
            "Train and validation feature columns do not match."
        )

    if train_features != test_features:
        raise PipelineError(
            "Train and test feature columns do not match."
        )


# =============================================================================
# FIT + TRANSFORM
# =============================================================================

def fit_transform_pipeline(
    pipeline: Pipeline,
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target: str = _TARGET,
) -> tuple[
    Pipeline,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """
    Fit preprocessing on X_train only and transform all splits.

    IMPORTANT:

        pipeline.fit(X_train)

    is called exactly once.

    Validation and test are NEVER used to calculate:
        - means
        - standard deviations
        - medians
        - category mappings
        - any other fitted preprocessing parameters
    """

    validate_split_columns(
        train,
        val,
        test,
        target=target,
    )

    # -------------------------------------------------------------------------
    # Separate X and y
    # -------------------------------------------------------------------------

    X_train = train.drop(
        columns=[target]
    )

    X_val = val.drop(
        columns=[target]
    )

    X_test = test.drop(
        columns=[target]
    )

    y_train = train[target].reset_index(
        drop=True
    )

    y_val = val[target].reset_index(
        drop=True
    )

    y_test = test[target].reset_index(
        drop=True
    )

    # -------------------------------------------------------------------------
    # TARGET LEAKAGE GUARD
    # -------------------------------------------------------------------------

    forbidden = {"is_capped"}
    leaked = forbidden.intersection(X_train.columns)
    if leaked:
        raise PipelineError(
            f"Target leakage detected in X! Forbidden columns found: {leaked}. "
            "These must be excluded before preprocessing."
        )

    # -------------------------------------------------------------------------
    # FIT — TRAIN ONLY
    # -------------------------------------------------------------------------

    logger.info(
        f"Fitting preprocessing pipeline "
        f"on {len(X_train):,} training rows..."
    )

    pipeline.fit(X_train)

    logger.info(
        "Pipeline fitted successfully on TRAIN only."
    )

    # -------------------------------------------------------------------------
    # TRANSFORM
    # -------------------------------------------------------------------------

    logger.info(
        "Transforming train / validation / test..."
    )

    X_train_transformed = pipeline.transform(
        X_train
    )

    X_val_transformed = pipeline.transform(
        X_val
    )

    X_test_transformed = pipeline.transform(
        X_test
    )

    # -------------------------------------------------------------------------
    # Numerical safety checks
    # -------------------------------------------------------------------------

    arrays = {
        "X_train": X_train_transformed,
        "X_val": X_val_transformed,
        "X_test": X_test_transformed,
    }

    for name, array in arrays.items():

        if not np.isfinite(array).all():
            raise PipelineError(
                f"{name} contains NaN or infinite values "
                "after preprocessing."
            )

    logger.info(
        "Transformation complete | "
        f"X_train={X_train_transformed.shape} | "
        f"X_val={X_val_transformed.shape} | "
        f"X_test={X_test_transformed.shape}"
    )

    return (
        pipeline,
        X_train_transformed,
        X_val_transformed,
        X_test_transformed,
        y_train,
        y_val,
        y_test,
    )


# =============================================================================
# FEATURE NAMES
# =============================================================================

def get_feature_names(
    pipeline: Pipeline,
) -> list[str]:
    """
    Return feature names after preprocessing.

    OHE categories are expanded automatically.

    Example:
        ocean_proximity_INLAND
        ocean_proximity_NEAR BAY
    """

    try:

        preprocessor = pipeline.named_steps[
            "preprocessor"
        ]

        feature_names = (
            preprocessor
            .get_feature_names_out()
            .tolist()
        )

        logger.info(
            f"Extracted {len(feature_names)} output features."
        )

        return feature_names

    except Exception as exc:

        logger.error(
            f"Could not extract feature names: {exc}"
        )

        raise PipelineError(
            "Failed to extract preprocessing feature names."
        ) from exc


# =============================================================================
# SAVE PIPELINE
# =============================================================================

def save_pipeline(
    pipeline: Pipeline,
    output_dir: Path | None = None,
) -> Path:
    """
    Save the fitted preprocessing pipeline.

    Output:
        artifacts/preprocessing_pipeline.pkl

    DVC tracking is intentionally NOT performed here.
    """

    output_dir = (
        output_dir
        if output_dir is not None
        else PROJECT_DIR / "artifacts"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        output_dir
        / "preprocessing_pipeline.pkl"
    )

    with open(path, "wb") as file:
        pickle.dump(
            pipeline,
            file,
        )

    size_kb = (
        path.stat().st_size / 1024
    )

    logger.info(
        f"Pipeline saved -> "
        f"{path} ({size_kb:.1f} KB)"
    )

    return path


# =============================================================================
# LOAD PIPELINE
# =============================================================================

def load_pipeline(
    artifacts_dir: Path | None = None,
) -> Pipeline:
    """
    Load a previously fitted preprocessing pipeline.
    """

    artifacts_dir = (
        artifacts_dir
        if artifacts_dir is not None
        else PROJECT_DIR / "artifacts"
    )

    path = (
        artifacts_dir
        / "preprocessing_pipeline.pkl"
    )

    if not path.exists():
        raise FileNotFoundError(
            f"Preprocessing pipeline not found: {path}"
        )

    with open(path, "rb") as file:
        pipeline = pickle.load(file)

    logger.info(
        f"Pipeline loaded from {path}"
    )

    return pipeline


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    target: str = _TARGET,
    config_path: str | Path = _DEFAULT_CONFIG_PATH,
    artifacts_dir: Path | None = None,
    save: bool = True,
) -> PipelineResult:
    """
    Execute the complete preprocessing stage.

    Steps:

        1. Load Config (Fail-Fast)
        2. Validate input
        3. Resolve feature columns
        4. Build sklearn pipeline
        5. Fit on train only
        6. Transform train/validation/test
        7. Extract feature names
        8. Save fitted pipeline artifact
    """

    logger.info("=" * 60)
    logger.info("PREPROCESSING PIPELINE STARTED")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Step 1 — Load Config
    # -------------------------------------------------------------------------

    logger.info(
        "Step 1/6 — Loading features configuration..."
    )

    config = load_features_config(Path(config_path))

    # -------------------------------------------------------------------------
    # Step 2 — Validate
    # -------------------------------------------------------------------------

    logger.info(
        "Step 2/6 — Validating input splits"
    )

    validate_split_columns(
        train,
        val,
        test,
        target=target,
    )

    # -------------------------------------------------------------------------
    # Step 3 — Resolve columns
    # -------------------------------------------------------------------------

    logger.info(
        "Step 3/6 — Resolving feature columns"
    )

    columns = resolve_columns(
        df=train.drop(
            columns=[target]
        ),
        config=config,
    )

    # -------------------------------------------------------------------------
    # Step 4 — Build pipeline
    # -------------------------------------------------------------------------

    logger.info(
        "Step 4/6 — Building sklearn pipeline"
    )

    pipeline = build_pipeline(
        columns
    )

    # -------------------------------------------------------------------------
    # Step 5 — Fit + transform
    # -------------------------------------------------------------------------

    logger.info(
        "Step 5/6 — Fit on train / transform all splits"
    )

    (
        fitted_pipeline,
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
    ) = fit_transform_pipeline(
        pipeline,
        train,
        val,
        test,
        target=target,
    )

    # -------------------------------------------------------------------------
    # Step 6 — Feature names
    # -------------------------------------------------------------------------

    logger.info(
        "Step 6/6 — Extracting feature names"
    )

    feature_names = get_feature_names(
        fitted_pipeline
    )

    if len(feature_names) != X_train.shape[1]:
        raise PipelineError(
            "Number of feature names does not match "
            "number of transformed features."
        )

    # -------------------------------------------------------------------------
    # Step 7 — Save
    # -------------------------------------------------------------------------

    pipeline_path = None

    if save:

        logger.info(
            "Step 7/7 — Saving fitted pipeline"
        )

        pipeline_path = save_pipeline(
            fitted_pipeline,
            output_dir=artifacts_dir,
        )

    else:

        logger.info(
            "Step 7/7 — save=False, "
            "skipping artifact save."
        )

    # -------------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------------

    result = PipelineResult(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        feature_names_out=feature_names,
        pipeline_path=pipeline_path,
        n_features=X_train.shape[1],
    )

    logger.info(
        "\n" + result.summary()
    )

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":

    from src.data.data_loader import DataLoader

    loader = DataLoader()

    logger.info(
        "Loading feature-engineered splits..."
    )

    train = loader.load_processed(
        "train_feat.csv"
    )

    val = loader.load_processed(
        "val_feat.csv"
    )

    test = loader.load_processed(
        "test_feat.csv"
    )

    result = run_pipeline(
        train,
        val,
        test,
    )

    print(
        result.summary()
    )


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "PipelineError",
    "PipelineResult",
    "load_features_config",
    "resolve_columns",
    "build_pipeline",
    "validate_split_columns",
    "fit_transform_pipeline",
    "get_feature_names",
    "save_pipeline",
    "load_pipeline",
    "run_pipeline",
]
