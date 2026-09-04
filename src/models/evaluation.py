# =============================================================================
# src/models/evaluation.py
# California Housing Project — Final Model Evaluation Engine
#
# RESPONSIBILITY
# --------------
# 1. Load the untouched Test dataset.
# 2. Validate the Test schema.
# 3. Evaluate an already-fitted final sklearn Pipeline.
# 4. Calculate RMSE, MAE, and R².
# 5. Save predictions and evaluation reports.
# 6. Generate evaluation plots.
#
# DESIGN
# ------
# - ZERO MLflow dependency.
# - ZERO DVC dependency.
# - ZERO Git dependency.
# - NEVER fits the model.
# - NEVER fits preprocessing.
# - NEVER modifies the final model.
# - Test data is used ONLY for the final evaluation.
#
# EXPECTED FLOW
# -------------
#
#   training.py
#        ↓
#   Train + Validation
#        ↓
#   final_model_pipeline.pkl
#        ↓
#   evaluation.py
#        ↓
#   untouched test_clean.csv
#        ↓
#   predictions
#        ↓
#   RMSE / MAE / R²
#        ↓
#   reports/
#
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline

from src.features.pipeline import (
    predict,
    split_features_target,
)
from src.models.model_factory import load_model_config
from src.utils.logger import get_logger
from src.utils.paths import PROJECT_DIR


logger = get_logger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

_DEFAULT_MODEL_CONFIG_PATH = (
    PROJECT_DIR / "configs" / "model_config.yaml"
)

_DEFAULT_DATA_CONFIG_PATH = (
    PROJECT_DIR / "configs" / "data_config.yaml"
)

_DEFAULT_TARGET = "median_house_value"

_DEFAULT_ARTIFACTS_DIR = (
    PROJECT_DIR / "artifacts"
)

_DEFAULT_REPORTS_DIR = (
    PROJECT_DIR / "reports"
)

_DEFAULT_FINAL_MODEL_FILENAME = (
    "final_model_pipeline.pkl"
)

_DEFAULT_METRICS_FILENAME = (
    "metrics.json"
)

_DEFAULT_TEST_RESULTS_FILENAME = (
    "test_results.json"
)

_DEFAULT_PREDICTIONS_FILENAME = (
    "test_predictions.csv"
)

_DEFAULT_RESIDUALS_FILENAME = (
    "residuals.png"
)

_DEFAULT_FEATURE_IMPORTANCE_FILENAME = (
    "feature_importance.png"
)

_DEFAULT_PRED_VS_ACTUAL_FILENAME = (
    "pred_vs_actual.png"
)


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================


class EvaluationEngineError(Exception):
    """Raised when final model evaluation fails safely."""


# =============================================================================
# TEST SCHEMA VALIDATION
# =============================================================================


def validate_test_schema(
    df: pd.DataFrame,
    target_col: str = _DEFAULT_TARGET,
) -> None:
    """
    Validate that the supplied dataset is the untouched clean Test schema.

    The final pipeline performs all model-side transformations internally,
    therefore evaluation expects test_clean.csv rather than test_feat.csv.
    """

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise EvaluationEngineError(
            "Test data must be a pandas DataFrame."
        )

    if df.empty:
        raise EvaluationEngineError(
            "Test DataFrame is empty."
        )

    required_columns = {
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
        required_columns - set(df.columns)
    )

    if missing:
        raise EvaluationEngineError(
            "Test schema validation failed.\n"
            f"Missing required columns: {missing}\n"
            "Evaluation expects test_clean.csv, not test_feat.csv."
        )

    if "is_capped" in df.columns:
        raise EvaluationEngineError(
            "Target-derived feature 'is_capped' must not be present "
            "in the Test dataset."
        )

    if not pd.api.types.is_numeric_dtype(
        df[target_col]
    ):
        raise EvaluationEngineError(
            f"Target column '{target_col}' must be numeric."
        )

    if df[target_col].isna().any():
        raise EvaluationEngineError(
            f"Target column '{target_col}' contains missing values."
        )

    logger.info(
        "Test input schema validation passed."
    )


# =============================================================================
# FINAL PIPELINE VALIDATION
# =============================================================================


def validate_fitted_pipeline(
    pipeline: Pipeline,
) -> None:
    """
    Validate that the supplied object is a fitted sklearn Pipeline.

    This function intentionally does NOT fit anything.
    """

    if not isinstance(
        pipeline,
        Pipeline,
    ):
        raise EvaluationEngineError(
            "Evaluation requires an sklearn Pipeline."
        )

    required_steps = {
        "preprocessing",
        "model",
    }

    actual_steps = set(
        pipeline.named_steps.keys()
    )

    missing_steps = (
        required_steps - actual_steps
    )

    if missing_steps:
        raise EvaluationEngineError(
            "Final pipeline is missing required steps: "
            f"{sorted(missing_steps)}"
        )

    # -------------------------------------------------------------------------
    # Verify that final model appears fitted.
    #
    # We do not rely on a single universal sklearn fitted-attribute because
    # different estimators expose different learned attributes.
    #
    # Instead we perform a prediction smoke test later on the actual Test data.
    # -------------------------------------------------------------------------

    logger.info(
        "Final pipeline structure validation passed."
    )


# =============================================================================
# METRIC CALCULATION
# =============================================================================


def calculate_regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> dict[str, float]:
    """
    Calculate final regression metrics.

    Metrics
    -------
    RMSE
    MAE
    R²
    """

    y_true_array = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred_array = np.asarray(
        y_pred,
        dtype=float,
    )

    if y_true_array.shape != y_pred_array.shape:
        raise EvaluationEngineError(
            "y_true and y_pred must have identical shapes."
        )

    if y_true_array.size == 0:
        raise EvaluationEngineError(
            "Cannot calculate metrics on empty arrays."
        )

    if not np.isfinite(
        y_true_array
    ).all():
        raise EvaluationEngineError(
            "y_true contains NaN or infinite values."
        )

    if not np.isfinite(
        y_pred_array
    ).all():
        raise EvaluationEngineError(
            "y_pred contains NaN or infinite values."
        )

    rmse = float(
        np.sqrt(
            mean_squared_error(
                y_true_array,
                y_pred_array,
            )
        )
    )

    mae = float(
        mean_absolute_error(
            y_true_array,
            y_pred_array,
        )
    )

    r2 = float(
        r2_score(
            y_true_array,
            y_pred_array,
        )
    )

    return {
        "test_rmse": rmse,
        "test_mae": mae,
        "test_r2": r2,
    }


# =============================================================================
# RESIDUAL CALCULATION
# =============================================================================


def calculate_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> np.ndarray:
    """
    Calculate residuals:

        residual = actual - predicted
    """

    residuals = (
        np.asarray(
            y_true,
            dtype=float,
        )
        - np.asarray(
            y_pred,
            dtype=float,
        )
    )

    if not np.isfinite(
        residuals
    ).all():
        raise EvaluationEngineError(
            "Residuals contain NaN or infinite values."
        )

    return residuals


# =============================================================================
# PREDICTION REPORT
# =============================================================================


def build_prediction_dataframe(
    test_df: pd.DataFrame,
    target_col: str,
    predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Build a compact prediction report.

    Contains:
        actual
        prediction
        residual
        absolute_error
    """

    actual = test_df[
        target_col
    ].to_numpy(
        dtype=float
    )

    predictions = np.asarray(
        predictions,
        dtype=float,
    )

    residuals = (
        actual - predictions
    )

    absolute_errors = np.abs(
        residuals
    )

    result = pd.DataFrame(
        {
            "actual": actual,
            "prediction": predictions,
            "residual": residuals,
            "absolute_error": absolute_errors,
        }
    )

    return result


# =============================================================================
# PLOT 1 — PREDICTED VS ACTUAL
# =============================================================================


def save_predicted_vs_actual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> Path:
    """
    Save Predicted vs Actual scatter plot.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.scatter(
        y_true,
        y_pred,
        alpha=0.35,
        s=20,
    )

    minimum = float(
        min(
            np.min(y_true),
            np.min(y_pred),
        )
    )

    maximum = float(
        max(
            np.max(y_true),
            np.max(y_pred),
        )
    )

    axis.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        linewidth=2,
    )

    axis.set_title(
        "Predicted vs Actual"
    )

    axis.set_xlabel(
        "Actual Median House Value"
    )

    axis.set_ylabel(
        "Predicted Median House Value"
    )

    axis.grid(
        alpha=0.2
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    logger.info(
        f"Predicted-vs-Actual plot saved -> "
        f"{output_path}"
    )

    return output_path


# =============================================================================
# PLOT 2 — RESIDUALS
# =============================================================================


def save_residuals_plot(
    y_pred: np.ndarray,
    residuals: np.ndarray,
    output_path: Path,
) -> Path:
    """
    Save residuals vs predicted values plot.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(8, 6)
    )

    axis.scatter(
        y_pred,
        residuals,
        alpha=0.35,
        s=20,
    )

    axis.axhline(
        y=0,
        linestyle="--",
        linewidth=2,
    )

    axis.set_title(
        "Residuals vs Predicted"
    )

    axis.set_xlabel(
        "Predicted Median House Value"
    )

    axis.set_ylabel(
        "Residual (Actual - Predicted)"
    )

    axis.grid(
        alpha=0.2
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    logger.info(
        f"Residuals plot saved -> "
        f"{output_path}"
    )

    return output_path


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================


def _extract_model_feature_importance(
    pipeline: Pipeline,
) -> tuple[list[str], np.ndarray] | None:
    """
    Extract feature importance from the fitted final pipeline.

    Supported model interfaces:
        - feature_importances_
        - coef_

    Returns None for unsupported models such as a generic estimator without
    either attribute.
    """

    try:

        preprocessing = pipeline.named_steps[
            "preprocessing"
        ]

        preprocessor = preprocessing.named_steps[
            "preprocessor"
        ]

        feature_names = (
            preprocessor
            .get_feature_names_out()
            .tolist()
        )

        model = pipeline.named_steps[
            "model"
        ]

        # ---------------------------------------------------------------------
        # Tree-based models
        # ---------------------------------------------------------------------

        if hasattr(
            model,
            "feature_importances_",
        ):

            importance = np.asarray(
                model.feature_importances_,
                dtype=float,
            )

        # ---------------------------------------------------------------------
        # Linear models
        # ---------------------------------------------------------------------

        elif hasattr(
            model,
            "coef_",
        ):

            coefficients = np.asarray(
                model.coef_,
                dtype=float,
            )

            # Regression normally has 1D coefficients.
            if coefficients.ndim == 1:

                importance = np.abs(
                    coefficients
                )

            elif coefficients.ndim == 2:

                importance = np.mean(
                    np.abs(coefficients),
                    axis=0,
                )

            else:

                return None

        else:

            return None

        if len(feature_names) != len(importance):
            logger.warning(
                "Feature-name count does not match model importance count."
            )
            return None

        if not np.isfinite(
            importance
        ).all():
            return None

        return (
            feature_names,
            importance,
        )

    except Exception as exc:

        logger.warning(
            f"Feature importance extraction unavailable: {exc}"
        )

        return None


def save_feature_importance_plot(
    pipeline: Pipeline,
    output_path: Path,
    top_n: int = 20,
) -> Path | None:
    """
    Save a top-N feature importance plot when the final estimator supports it.

    If the estimator does not expose feature importance or coefficients,
    no plot is generated.
    """

    extracted = (
        _extract_model_feature_importance(
            pipeline
        )
    )

    if extracted is None:

        logger.info(
            "Feature importance is not available "
            "for the final model. Skipping plot."
        )

        return None

    feature_names, importance = extracted

    order = np.argsort(
        importance
    )[::-1]

    order = order[
        : min(
            top_n,
            len(order),
        )
    ]

    selected_names = [
        feature_names[index]
        for index in reversed(order)
    ]

    selected_importance = [
        importance[index]
        for index in reversed(order)
    ]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure, axis = plt.subplots(
        figsize=(10, 8)
    )

    axis.barh(
        selected_names,
        selected_importance,
    )

    axis.set_title(
        f"Top {len(selected_names)} Feature Importance"
    )

    axis.set_xlabel(
        "Importance"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.grid(
        axis="x",
        alpha=0.2,
    )

    figure.tight_layout()

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(
        figure
    )

    logger.info(
        f"Feature importance plot saved -> "
        f"{output_path}"
    )

    return output_path


# =============================================================================
# REPORT SAVING
# =============================================================================


def save_evaluation_reports(
    metrics: dict[str, float],
    prediction_df: pd.DataFrame,
    output_dir: Path,
    model_name: str | None = None,
) -> tuple[Path, Path]:
    """
    Save metrics JSON files and test predictions CSV.
    """

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------------------------------
    # metrics.json
    # -------------------------------------------------------------------------

    metrics_path = (
        output_dir
        / _DEFAULT_METRICS_FILENAME
    )

    with open(
        metrics_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metrics,
            file,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # test_results.json
    # -------------------------------------------------------------------------

    test_results = {
        "model_name": model_name,
        "test_metrics": metrics,
        "n_test_samples": int(
            len(prediction_df)
        ),
        "evaluation_stage": (
            "final_test_evaluation"
        ),
        "test_used_for_training": False,
    }

    test_results_path = (
        output_dir
        / _DEFAULT_TEST_RESULTS_FILENAME
    )

    with open(
        test_results_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            test_results,
            file,
            indent=2,
        )

    # -------------------------------------------------------------------------
    # Predictions
    # -------------------------------------------------------------------------

    predictions_path = (
        output_dir
        / _DEFAULT_PREDICTIONS_FILENAME
    )

    prediction_df.to_csv(
        predictions_path,
        index=False,
    )

    logger.info(
        f"Metrics saved -> {metrics_path}"
    )

    logger.info(
        f"Test results saved -> {test_results_path}"
    )

    logger.info(
        f"Predictions saved -> {predictions_path}"
    )

    return (
        metrics_path,
        test_results_path,
    )


# =============================================================================
# MAIN EVALUATION FUNCTION
# =============================================================================


def run_evaluation_on_test(
    pipeline: Pipeline,
    model_config_path: str | Path = (
        _DEFAULT_MODEL_CONFIG_PATH
    ),
    output_dir: str | Path | None = None,
) -> tuple[dict[str, float], list[Path]]:
    """
    Evaluate the already-fitted final pipeline on the untouched Test dataset.

    IMPORTANT
    ---------
    This function does NOT call fit() anywhere.

    Returns
    -------
    tuple
        (
            test_metrics,
            plot_paths
        )
    """

    logger.info("=" * 80)
    logger.info(
        "FINAL TEST EVALUATION STARTED"
    )
    logger.info("=" * 80)

    start_time = time.time()

    # =========================================================================
    # 1. VALIDATE PIPELINE
    # =========================================================================

    validate_fitted_pipeline(
        pipeline
    )

    # =========================================================================
    # 2. LOAD MODEL CONFIG
    # =========================================================================

    model_cfg = load_model_config(
        model_config_path
    )

    target_col = (
        model_cfg
        .get("project", {})
        .get(
            "target",
            _DEFAULT_TARGET,
        )
    )

    test_path = (
        PROJECT_DIR
        / model_cfg["data"]["test_path"]
    )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test dataset not found: {test_path}"
        )

    # =========================================================================
    # 3. LOAD TEST DATA ONLY
    # =========================================================================

    test_df = pd.read_csv(
        test_path
    )

    validate_test_schema(
        test_df,
        target_col=target_col,
    )

    logger.info(
        f"Test dataset loaded: "
        f"{test_path} | "
        f"rows={len(test_df):,}"
    )

    # =========================================================================
    # 4. SPLIT X / y
    # =========================================================================

    X_test, y_test = (
        split_features_target(
            test_df,
            target=target_col,
        )
    )

    # =========================================================================
    # 5. PREDICTION ONLY
    # =========================================================================
    #
    # IMPORTANT:
    # pipeline.predict() performs transforms using already-fitted objects.
    #
    # NO FIT occurs here.
    #
    # =========================================================================

    logger.info(
        "Generating predictions on untouched Test data..."
    )

    try:

        predictions = predict(
            pipeline,
            X_test,
        )

    except Exception as exc:

        raise EvaluationEngineError(
            f"Final Test prediction failed: {exc}"
        ) from exc

    # =========================================================================
    # 6. METRICS
    # =========================================================================

    metrics = calculate_regression_metrics(
        y_true=y_test,
        y_pred=predictions,
    )

    residuals = calculate_residuals(
        y_true=y_test,
        y_pred=predictions,
    )

    logger.info(
        f"Test RMSE: {metrics['test_rmse']:,.4f}"
    )

    logger.info(
        f"Test MAE:  {metrics['test_mae']:,.4f}"
    )

    logger.info(
        f"Test R²:   {metrics['test_r2']:.6f}"
    )

    # =========================================================================
    # 7. MODEL NAME
    # =========================================================================

    model_name: str | None = None

    if (
        "model"
        in pipeline.named_steps
    ):

        model_name = (
            pipeline
            .named_steps["model"]
            .__class__
            .__name__
        )

    # =========================================================================
    # 8. BUILD PREDICTION REPORT
    # =========================================================================

    prediction_df = (
        build_prediction_dataframe(
            test_df=test_df,
            target_col=target_col,
            predictions=predictions,
        )
    )

    # =========================================================================
    # 9. OUTPUT DIRECTORY
    # =========================================================================

    reports_dir = (
        Path(output_dir)
        if output_dir is not None
        else _DEFAULT_REPORTS_DIR
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # 10. SAVE REPORTS
    # =========================================================================

    save_evaluation_reports(
        metrics=metrics,
        prediction_df=prediction_df,
        output_dir=reports_dir,
        model_name=model_name,
    )

    # =========================================================================
    # 11. SAVE PLOTS
    # =========================================================================

    plot_paths: list[Path] = []

    # -------------------------------------------------------------------------
    # Predicted vs Actual
    # -------------------------------------------------------------------------

    pred_vs_actual_path = (
        reports_dir
        / _DEFAULT_PRED_VS_ACTUAL_FILENAME
    )

    save_predicted_vs_actual_plot(
        y_true=np.asarray(
            y_test,
            dtype=float,
        ),
        y_pred=predictions,
        output_path=pred_vs_actual_path,
    )

    plot_paths.append(
        pred_vs_actual_path
    )

    # -------------------------------------------------------------------------
    # Residuals
    # -------------------------------------------------------------------------

    residuals_path = (
        reports_dir
        / _DEFAULT_RESIDUALS_FILENAME
    )

    save_residuals_plot(
        y_pred=predictions,
        residuals=residuals,
        output_path=residuals_path,
    )

    plot_paths.append(
        residuals_path
    )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    feature_importance_path = (
        reports_dir
        / _DEFAULT_FEATURE_IMPORTANCE_FILENAME
    )

    generated_feature_importance = (
        save_feature_importance_plot(
            pipeline=pipeline,
            output_path=feature_importance_path,
        )
    )

    if generated_feature_importance is not None:
        plot_paths.append(
            generated_feature_importance
        )

    # =========================================================================
    # 12. EVALUATION SUMMARY
    # =========================================================================

    elapsed = (
        time.time()
        - start_time
    )

    logger.info("=" * 80)
    logger.info(
        "FINAL TEST EVALUATION COMPLETED"
    )
    logger.info(
        f"Model: {model_name}"
    )
    logger.info(
        f"Test rows: {len(test_df):,}"
    )
    logger.info(
        f"RMSE: {metrics['test_rmse']:,.4f}"
    )
    logger.info(
        f"MAE:  {metrics['test_mae']:,.4f}"
    )
    logger.info(
        f"R²:   {metrics['test_r2']:.6f}"
    )
    logger.info(
        f"Duration: {elapsed:.2f}s"
    )
    logger.info("=" * 80)

    return (
        metrics,
        plot_paths,
    )


# =============================================================================
# OPTIONAL CONVENIENCE WRAPPER
# =============================================================================


def evaluate_saved_final_model(
    model_config_path: str | Path = (
        _DEFAULT_MODEL_CONFIG_PATH
    ),
    artifacts_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, float], list[Path]]:
    """
    Load the serialized final pipeline and evaluate it on Test.

    This is useful for:
        python -m src.models.evaluation

    It still performs NO fitting.
    """

    from src.features.pipeline import load_pipeline

    resolved_artifacts_dir = (
        Path(artifacts_dir)
        if artifacts_dir is not None
        else _DEFAULT_ARTIFACTS_DIR
    )

    pipeline = load_pipeline(
        artifacts_dir=resolved_artifacts_dir,
        filename=_DEFAULT_FINAL_MODEL_FILENAME,
    )

    return run_evaluation_on_test(
        pipeline=pipeline,
        model_config_path=model_config_path,
        output_dir=output_dir,
    )


# =============================================================================
# CLI
# =============================================================================


if __name__ == "__main__":

    import sys

    try:

        metrics, plots = (
            evaluate_saved_final_model()
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            "FINAL TEST EVALUATION SUMMARY"
        )

        print(
            "=" * 80
        )

        print(
            f"Test RMSE : "
            f"{metrics['test_rmse']:,.4f}"
        )

        print(
            f"Test MAE  : "
            f"{metrics['test_mae']:,.4f}"
        )

        print(
            f"Test R²   : "
            f"{metrics['test_r2']:.6f}"
        )

        print(
            "\nGenerated plots:"
        )

        for plot in plots:
            print(
                f"  - {plot}"
            )

        print(
            "=" * 80
            + "\n"
        )

        sys.exit(0)

    except Exception as exc:

        logger.exception(
            f"Evaluation failed: {exc}"
        )

        sys.exit(1)


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "EvaluationEngineError",
    "validate_test_schema",
    "validate_fitted_pipeline",
    "calculate_regression_metrics",
    "calculate_residuals",
    "build_prediction_dataframe",
    "save_predicted_vs_actual_plot",
    "save_residuals_plot",
    "save_feature_importance_plot",
    "save_evaluation_reports",
    "run_evaluation_on_test",
    "evaluate_saved_final_model",
]