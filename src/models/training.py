# =============================================================================
# src/models/training.py
# California Housing Project — Pure ML Training & Benchmarking Engine
#
# RESPONSIBILITY
# --------------
# 1. Validate training-data schema.
# 2. Run CV-safe benchmarking across enabled candidate models.
# 3. Select the best candidate using model_config.yaml.
# 4. Fit the final selected pipeline on Train + Validation.
# 5. Serialize the complete fitted sklearn pipeline.
#
# DESIGN
# ------
# - ZERO MLflow dependency.
# - ZERO DVC dependency.
# - ZERO Git dependency.
# - All learned preprocessing stays inside sklearn Pipeline.
# - Candidate benchmarking uses Cross-Validation.
# - Final fitting never uses Test.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features.pipeline import (
    build_model_pipeline,
    load_pipeline_config,
    resolve_columns,
    save_pipeline,
    split_features_target,
)
from src.models.model_factory import (
    create_model,
    get_enabled_models,
    load_model_config,
)
from src.utils.logger import get_logger
from src.utils.paths import PROJECT_DIR


logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

_DEFAULT_DATA_CONFIG_PATH = (
    PROJECT_DIR / "configs" / "data_config.yaml"
)

_DEFAULT_MODEL_CONFIG_PATH = (
    PROJECT_DIR / "configs" / "model_config.yaml"
)

_DEFAULT_TARGET = "median_house_value"

_DEFAULT_REPORTS_DIR = PROJECT_DIR / "reports"

_DEFAULT_FINAL_MODEL_FILENAME = (
    "final_model_pipeline.pkl"
)


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================


class TrainingEngineError(Exception):
    """Raised when training or model selection fails safely."""


# =============================================================================
# RAW TRAINING SCHEMA
# =============================================================================


def validate_training_schema(
    df: pd.DataFrame,
    target_col: str = _DEFAULT_TARGET,
) -> None:
    """
    Validate that the supplied dataset is a RAW/CLEAN training dataset.

    The training pipeline expects the original columns because
    feature engineering happens internally inside the sklearn Pipeline.

    Therefore train_feat.csv must not be supplied here.
    """

    if not isinstance(df, pd.DataFrame):
        raise TrainingEngineError(
            "Training data must be a pandas DataFrame."
        )

    if df.empty:
        raise TrainingEngineError(
            "Training DataFrame is empty."
        )

    required_raw_columns = {
        "longitude",
        "latitude",
        "housing_median_age",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
        "median_income",
        "ocean_proximity",
        target_col,
    }

    missing = sorted(
        required_raw_columns - set(df.columns)
    )

    if missing:
        raise TrainingEngineError(
            "Training data schema validation failed.\n"
            f"Missing required raw columns: {missing}\n"
            "The training engine expects train_clean.csv / "
            "val_clean.csv, not train_feat.csv."
        )

    forbidden = {"is_capped"}

    forbidden_present = (
        forbidden & set(df.columns)
    )

    if forbidden_present:
        raise TrainingEngineError(
            "Target-derived columns detected in training data: "
            f"{sorted(forbidden_present)}"
        )

    if target_col not in df.columns:
        raise TrainingEngineError(
            f"Target column '{target_col}' is missing."
        )

    if not pd.api.types.is_numeric_dtype(
        df[target_col]
    ):
        raise TrainingEngineError(
            f"Target column '{target_col}' must be numeric."
        )

    if df[target_col].isna().any():
        raise TrainingEngineError(
            f"Target column '{target_col}' contains missing values."
        )

    logger.info(
        "Training input schema validation passed."
    )


# =============================================================================
# METRIC HELPERS
# =============================================================================


def _format_scores(
    raw_scores: np.ndarray,
    *,
    is_negative_metric: bool,
) -> dict[str, Any]:
    """
    Convert sklearn CV scores into human-readable values.

    sklearn represents loss metrics such as RMSE and MAE as negative
    values because its scoring API maximizes scores.
    """

    scores = (
        -raw_scores
        if is_negative_metric
        else raw_scores
    )

    scores = np.asarray(
        scores,
        dtype=float,
    )

    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "values": [
            float(value)
            for value in scores
        ],
    }


def _get_primary_metric_config(
    model_config: dict[str, Any],
) -> tuple[str, str, bool]:
    """
    Resolve:
        - configured primary metric
        - result-column name
        - optimization direction

    Returns
    -------
    tuple
        (metric_name, result_column, direction)
    """

    selection_cfg = model_config.get(
        "model_selection",
        {},
    )

    primary_metric = selection_cfg.get(
        "primary_metric",
        "neg_root_mean_squared_error",
    )

    optimization_direction = selection_cfg.get(
        "optimization_direction",
        "minimize",
    )

    metric_to_result_column = {
        "neg_root_mean_squared_error": "val_rmse_mean",
        "neg_mean_absolute_error": "val_mae_mean",
        "r2": "val_r2_mean",
    }

    if primary_metric not in metric_to_result_column:
        raise TrainingEngineError(
            "Unsupported primary metric: "
            f"'{primary_metric}'. "
            f"Supported values: "
            f"{sorted(metric_to_result_column)}"
        )

    if optimization_direction not in {
        "minimize",
        "maximize",
    }:
        raise TrainingEngineError(
            "optimization_direction must be either "
            "'minimize' or 'maximize'."
        )

    return (
        primary_metric,
        metric_to_result_column[primary_metric],
        optimization_direction,
    )


# =============================================================================
# SINGLE MODEL CV EVALUATION
# =============================================================================


def evaluate_single_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    pipeline_cols: dict[str, list[str]],
    pipeline_config: dict[str, Any],
    model_config: dict[str, Any],
    cv_splitter: KFold,
    params_override: dict[str, Any] | None = None,
    cv_n_jobs: int = 1,
) -> dict[str, Any]:
    """
    Run leakage-safe CV for one candidate model.
    """

    logger.info(
        f"--- Starting Cross-Validation for "
        f"[{model_name}] ---"
    )

    start_time = time.time()

    # -------------------------------------------------------------------------
    # Create raw estimator
    # -------------------------------------------------------------------------

    raw_model = create_model(
        model_name=model_name,
        params_override=params_override,
        config=model_config,
    )

    # -------------------------------------------------------------------------
    # Build COMPLETE sklearn pipeline
    # -------------------------------------------------------------------------

    pipeline = build_model_pipeline(
        cols=pipeline_cols,
        model=raw_model,
        config_path=(
            PROJECT_DIR
            / "configs"
            / "data_config.yaml"
        ),
        lof_n_neighbors=(
            pipeline_config["lof"]["n_neighbors"]
        ),
        lof_contamination=(
            pipeline_config["lof"]["contamination"]
        ),
    )

    # -------------------------------------------------------------------------
    # Scoring
    # -------------------------------------------------------------------------

    scoring = {
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }

    # -------------------------------------------------------------------------
    # Cross-validation
    # -------------------------------------------------------------------------

    try:
        cv_results = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=cv_splitter,
            scoring=scoring,
            return_train_score=True,
            n_jobs=cv_n_jobs,
        )

    except Exception as exc:
        raise TrainingEngineError(
            f"Cross-validation failed for "
            f"model '{model_name}': {exc}"
        ) from exc

    elapsed = time.time() - start_time

    # -------------------------------------------------------------------------
    # Format metrics
    # -------------------------------------------------------------------------

    val_rmse = _format_scores(
        cv_results["test_rmse"],
        is_negative_metric=True,
    )

    train_rmse = _format_scores(
        cv_results["train_rmse"],
        is_negative_metric=True,
    )

    val_mae = _format_scores(
        cv_results["test_mae"],
        is_negative_metric=True,
    )

    val_r2 = _format_scores(
        cv_results["test_r2"],
        is_negative_metric=False,
    )

    fit_time_mean = float(
        np.mean(
            cv_results["fit_time"]
        )
    )

    score_time_mean = float(
        np.mean(
            cv_results["score_time"]
        )
    )

    logger.info(
        f"[{model_name}] CV completed in "
        f"{elapsed:.2f}s | "
        f"RMSE={val_rmse['mean']:.2f} ± "
        f"{val_rmse['std']:.2f} | "
        f"MAE={val_mae['mean']:.2f} | "
        f"R²={val_r2['mean']:.4f}"
    )

    return {
        "model_name": model_name,

        "val_rmse_mean": val_rmse["mean"],
        "val_rmse_std": val_rmse["std"],
        "fold_val_rmse": val_rmse["values"],

        "val_mae_mean": val_mae["mean"],
        "val_mae_std": val_mae["std"],

        "val_r2_mean": val_r2["mean"],
        "val_r2_std": val_r2["std"],

        "train_rmse_mean": train_rmse["mean"],

        "fit_time_mean": fit_time_mean,
        "score_time_mean": score_time_mean,

        "elapsed_seconds": elapsed,
    }


# =============================================================================
# BENCHMARKING
# =============================================================================


def run_training_benchmarks(
    model_config_path: str | Path = (
        _DEFAULT_MODEL_CONFIG_PATH
    ),
    data_config_path: str | Path = (
        _DEFAULT_DATA_CONFIG_PATH
    ),
    output_dir: str | Path | None = None,
) -> pd.DataFrame:
    """
    Benchmark all enabled candidate models using 5-fold CV.

    Returns
    -------
    pandas.DataFrame
        Model comparison sorted according to YAML selection configuration.
    """

    logger.info("=" * 80)
    logger.info(
        "MODEL TRAINING & BENCHMARKING STARTED"
    )
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Load config
    # -------------------------------------------------------------------------

    model_cfg = load_model_config(
        model_config_path
    )

    pipeline_cfg = load_pipeline_config(
        Path(data_config_path)
    )

    target_col = (
        model_cfg
        .get("project", {})
        .get(
            "target",
            _DEFAULT_TARGET,
        )
    )

    # -------------------------------------------------------------------------
    # Load train data
    # -------------------------------------------------------------------------

    train_path = (
        PROJECT_DIR
        / model_cfg["data"]["train_path"]
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: "
            f"{train_path}"
        )

    train_df = pd.read_csv(
        train_path
    )

    validate_training_schema(
        train_df,
        target_col=target_col,
    )

    # -------------------------------------------------------------------------
    # X / y
    # -------------------------------------------------------------------------

    cols = resolve_columns(
        pipeline_cfg
    )

    X_train, y_train = (
        split_features_target(
            train_df,
            target=target_col,
        )
    )

    # -------------------------------------------------------------------------
    # CV configuration
    # -------------------------------------------------------------------------

    cv_cfg = model_cfg.get(
        "cross_validation",
        {},
    )

    n_splits = int(
        cv_cfg.get(
            "n_splits",
            5,
        )
    )

    shuffle = bool(
        cv_cfg.get(
            "shuffle",
            True,
        )
    )

    random_state = int(
        cv_cfg.get(
            "random_state",
            42,
        )
    )

    cv_n_jobs = int(
        cv_cfg.get(
            "n_jobs",
            1,
        )
    )

    if n_splits < 2:
        raise TrainingEngineError(
            "cross_validation.n_splits must be >= 2."
        )

    cv_splitter = KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    # -------------------------------------------------------------------------
    # Models
    # -------------------------------------------------------------------------

    enabled_models = get_enabled_models(
        model_cfg
    )

    if not enabled_models:
        raise TrainingEngineError(
            "No enabled models found in model_config.yaml."
        )

    # -------------------------------------------------------------------------
    # Benchmark
    # -------------------------------------------------------------------------

    results_list: list[dict[str, Any]] = []

    for model_name in enabled_models:

        result = evaluate_single_model(
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            pipeline_cols=cols,
            pipeline_config=pipeline_cfg,
            model_config=model_cfg,
            cv_splitter=cv_splitter,
            cv_n_jobs=cv_n_jobs,
        )

        results_list.append(
            result
        )

    # -------------------------------------------------------------------------
    # Ranking
    # -------------------------------------------------------------------------

    df_results = pd.DataFrame(
        results_list
    )

    (
        primary_metric,
        primary_column,
        optimization_direction,
    ) = _get_primary_metric_config(
        model_cfg
    )

    ascending = (
        optimization_direction
        == "minimize"
    )

    df_results = (
        df_results
        .sort_values(
            by=primary_column,
            ascending=ascending,
        )
        .reset_index(drop=True)
    )

    best_model = str(
        df_results.iloc[0][
            "model_name"
        ]
    )

    best_score = float(
        df_results.iloc[0][
            primary_column
        ]
    )

    logger.info("=" * 80)
    logger.info(
        f"BEST MODEL: [{best_model}] | "
        f"{primary_metric} -> {best_score:.6f}"
    )
    logger.info("=" * 80)

    # -------------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------------

    reports_dir = (
        Path(output_dir)
        if output_dir is not None
        else _DEFAULT_REPORTS_DIR
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cv_results_path = (
        reports_dir
        / "cv_results.csv"
    )

    df_results.to_csv(
        cv_results_path,
        index=False,
    )

    training_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "n_folds": n_splits,
        "primary_metric": primary_metric,
        "primary_result_column": primary_column,
        "optimization_direction": (
            optimization_direction
        ),
        "best_model": best_model,
        "best_score": best_score,
        "benchmarks": results_list,
    }

    training_report_path = (
        reports_dir
        / "training_report.json"
    )

    with open(
        training_report_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            training_report,
            file,
            indent=2,
        )

    logger.info(
        f"CV results saved -> "
        f"{cv_results_path}"
    )

    logger.info(
        f"Training report saved -> "
        f"{training_report_path}"
    )

    return df_results


# =============================================================================
# FINAL FIT
# =============================================================================


def train_and_save_final_model(
    model_name: str,
    best_params: dict[str, Any],
    model_config_path: str | Path = (
        _DEFAULT_MODEL_CONFIG_PATH
    ),
    data_config_path: str | Path = (
        _DEFAULT_DATA_CONFIG_PATH
    ),
    output_dir: str | Path | None = None,
) -> tuple[Path, Pipeline]:
    """
    Fit final selected pipeline on Train + Validation.

    Test is NEVER loaded here.

    Parameters
    ----------
    model_name:
        Winning model selected during benchmarking.

    best_params:
        Best hyperparameters returned by tuning.py.

    Returns
    -------
    tuple
        (saved_artifact_path, fitted_pipeline)
    """

    logger.info("=" * 80)
    logger.info(
        f"FINAL FIT STARTED | MODEL={model_name}"
    )
    logger.info(
        "Training on TRAIN + VALIDATION only."
    )
    logger.info("=" * 80)

    model_cfg = load_model_config(
        model_config_path
    )

    pipeline_cfg = load_pipeline_config(
        Path(data_config_path)
    )

    target_col = (
        model_cfg
        .get("project", {})
        .get(
            "target",
            _DEFAULT_TARGET,
        )
    )

    # -------------------------------------------------------------------------
    # Load Train + Validation
    # -------------------------------------------------------------------------

    train_path = (
        PROJECT_DIR
        / model_cfg["data"]["train_path"]
    )

    val_path = (
        PROJECT_DIR
        / model_cfg["data"]["validation_path"]
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train dataset not found: {train_path}"
        )

    if not val_path.exists():
        raise FileNotFoundError(
            f"Validation dataset not found: {val_path}"
        )

    train_df = pd.read_csv(
        train_path
    )

    val_df = pd.read_csv(
        val_path
    )

    validate_training_schema(
        train_df,
        target_col=target_col,
    )

    validate_training_schema(
        val_df,
        target_col=target_col,
    )

    # -------------------------------------------------------------------------
    # Final training strategy
    # -------------------------------------------------------------------------

    final_cfg = model_cfg.get(
        "final_training",
        {},
    )

    strategy = final_cfg.get(
        "strategy",
        "train_plus_validation",
    )

    if strategy == "train_plus_validation":

        combined_df = pd.concat(
            [
                train_df,
                val_df,
            ],
            ignore_index=True,
        )

        logger.info(
            f"Final dataset = Train "
            f"({len(train_df):,}) + "
            f"Validation "
            f"({len(val_df):,}) = "
            f"{len(combined_df):,} rows."
        )

    elif strategy == "train_only":

        combined_df = train_df.copy()

        logger.info(
            f"Final dataset = Train only "
            f"({len(combined_df):,} rows)."
        )

    else:
        raise TrainingEngineError(
            "Unsupported final_training.strategy: "
            f"'{strategy}'"
        )

    use_test_for_training = bool(
        final_cfg.get(
            "use_test_for_training",
            False,
        )
    )

    if use_test_for_training:
        raise TrainingEngineError(
            "Test data must NEVER be used for final training."
        )

    # -------------------------------------------------------------------------
    # Hyperparameter cleanup
    # -------------------------------------------------------------------------

    cleaned_params = {}

    for key, value in best_params.items():

        if key.startswith(
            "model__"
        ):
            cleaned_key = key[
                len("model__") :
            ]
        else:
            cleaned_key = key

        cleaned_params[
            cleaned_key
        ] = value

    logger.info(
        f"Final model parameters: "
        f"{cleaned_params}"
    )

    # -------------------------------------------------------------------------
    # Create raw model
    # -------------------------------------------------------------------------

    raw_model = create_model(
        model_name=model_name,
        params_override=cleaned_params,
        config=model_cfg,
    )

    # -------------------------------------------------------------------------
    # Build complete pipeline
    # -------------------------------------------------------------------------

    cols = resolve_columns(
        pipeline_cfg
    )

    pipeline = build_model_pipeline(
        cols=cols,
        model=raw_model,
        config_path=Path(
            data_config_path
        ),
        lof_n_neighbors=(
            pipeline_cfg["lof"]["n_neighbors"]
        ),
        lof_contamination=(
            pipeline_cfg["lof"]["contamination"]
        ),
    )

    # -------------------------------------------------------------------------
    # Split X / y
    # -------------------------------------------------------------------------

    X_full, y_full = (
        split_features_target(
            combined_df,
            target=target_col,
        )
    )

    # -------------------------------------------------------------------------
    # Fit
    # -------------------------------------------------------------------------

    start_fit = time.time()

    pipeline.fit(
        X_full,
        y_full,
    )

    fit_time = (
        time.time()
        - start_fit
    )

    logger.info(
        f"Final pipeline fitted in "
        f"{fit_time:.2f}s."
    )

    # -------------------------------------------------------------------------
    # Save artifact
    # -------------------------------------------------------------------------

    artifacts_dir = (
        Path(output_dir)
        if output_dir is not None
        else (
            PROJECT_DIR
            / model_cfg
            .get(
                "artifacts",
                {}
            )
            .get(
                "output_dir",
                "artifacts",
            )
        )
    )

    saved_path = save_pipeline(
        pipeline=pipeline,
        output_dir=artifacts_dir,
        filename=_DEFAULT_FINAL_MODEL_FILENAME,
    )

    logger.info(
        f"Final production pipeline saved -> "
        f"{saved_path}"
    )

    return saved_path, pipeline


# =============================================================================
# CLI
# =============================================================================


if __name__ == "__main__":

    import sys

    try:

        df_summary = (
            run_training_benchmarks()
        )

        print(
            "\n"
            + "=" * 90
        )

        print(
            "CV BENCHMARK SUMMARY"
        )

        print(
            "=" * 90
        )

        display_columns = [
            "model_name",
            "val_rmse_mean",
            "val_rmse_std",
            "val_mae_mean",
            "val_r2_mean",
            "fit_time_mean",
        ]

        available_columns = [
            column
            for column in display_columns
            if column in df_summary.columns
        ]

        print(
            df_summary[
                available_columns
            ].to_string(
                index=False
            )
        )

        print(
            "=" * 90
            + "\n"
        )

        sys.exit(0)

    except Exception as exc:

        logger.exception(
            f"Training failed: {exc}"
        )

        sys.exit(1)


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "TrainingEngineError",
    "validate_training_schema",
    "evaluate_single_model",
    "run_training_benchmarks",
    "train_and_save_final_model",
]