# =============================================================================
# pipelines/run_pipeline.py
# California Housing Project — Master ML Pipeline Orchestrator
#
# RESPONSIBILITY
# --------------
# This module coordinates the ML lifecycle and integrates MLflow tracking.
#
# It intentionally does NOT contain ML algorithms.
#
# Execution flow:
#
#   1. Benchmarking
#          ↓
#   2. Hyperparameter Tuning
#          ↓
#   3. Final Fit on Train + Validation
#          ↓
#   4. Test Evaluation
#
# MLflow hierarchy:
#
#   Parent Run
#       ├── stage_1_benchmarking
#       ├── stage_2_tuning
#       ├── stage_3_final_fit
#       └── stage_4_test_evaluation
#
# IMPORTANT
# ---------
# MLflow is an external tracking layer.
# The src/models package remains MLflow-independent.
# =============================================================================

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np  # Used for numeric type detection in MLflow logging


# =============================================================================
# PROJECT ROOT
# =============================================================================

PROJECT_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_DIR),
    )


# =============================================================================
# PROJECT IMPORTS
# =============================================================================

from src.models.evaluation import (
    run_evaluation_on_test,
)

from src.models.model_factory import (
    load_model_config,
)

from src.models.training import (
    run_training_benchmarks,
    train_and_save_final_model,
)

from src.models.tuning import (
    run_tuning_for_model,
)

from src.utils.logger import get_logger


logger = get_logger(
    "pipeline_orchestrator"
)


# =============================================================================
# MLFLOW OPTIONAL IMPORT
# =============================================================================


def _setup_mlflow(
    model_cfg: dict[str, Any],
):
    """
    Configure MLflow if enabled.

    Returns
    -------
    tuple
        (use_mlflow, mlflow_module)
    """

    mlflow_cfg = model_cfg.get(
        "mlflow",
        {},
    )

    enabled = bool(
        mlflow_cfg.get(
            "enabled",
            True,
        )
    )

    if not enabled:
        logger.info(
            "MLflow disabled by configuration."
        )

        return (
            False,
            None,
        )

    try:

        import mlflow
        import mlflow.sklearn

    except ImportError:

        logger.warning(
            "MLflow is enabled in YAML but "
            "not installed. Continuing without "
            "MLflow tracking."
        )

        return (
            False,
            None,
        )

    tracking_uri = mlflow_cfg.get(
        "tracking_uri",
        "file:./mlruns",
    )

    experiment_name = mlflow_cfg.get(
        "experiment_name",
        "california_housing_regression",
    )

    mlflow.set_tracking_uri(
        tracking_uri
    )

    mlflow.set_experiment(
        experiment_name
    )

    logger.info(
        "MLflow active | "
        f"experiment='{experiment_name}' | "
        f"tracking_uri='{tracking_uri}'"
    )

    return (
        True,
        mlflow,
    )


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================


def execute_full_pipeline() -> None:
    """
    Execute the complete ML lifecycle.

    Order:
        Benchmarking
        → Tuning
        → Final Fit
        → Test Evaluation
    """

    logger.info("=" * 90)
    logger.info(
        "MASTER ML PIPELINE STARTED"
    )
    logger.info("=" * 90)

    pipeline_start_time = time.time()

    # -------------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------------

    model_cfg_path = (
        PROJECT_DIR
        / "configs"
        / "model_config.yaml"
    )

    model_cfg = load_model_config(
        model_cfg_path
    )

    # -------------------------------------------------------------------------
    # Setup MLflow
    # -------------------------------------------------------------------------

    use_mlflow, mlflow = (
        _setup_mlflow(
            model_cfg
        )
    )

    # -------------------------------------------------------------------------
    # Parent MLflow run
    # -------------------------------------------------------------------------

    if use_mlflow:

        parent_context = (
            mlflow.start_run(
                run_name=(
                    "california_housing_pipeline"
                )
            )
        )

    else:

        parent_context = (
            nullcontext()
        )

    with parent_context:

        # =====================================================================
        # STAGE 1 — MODEL BENCHMARKING
        # =====================================================================

        logger.info(
            "\n>>> "
            "[STAGE 1/4] "
            "MODEL BENCHMARKING"
        )

        df_results = (
            run_training_benchmarks()
        )

        if df_results.empty:
            raise RuntimeError(
                "Benchmarking returned no model results."
            )

        best_candidate_name = str(
            df_results.iloc[0][
                "model_name"
            ]
        )

        best_candidate_rmse = float(
            df_results.iloc[0][
                "val_rmse_mean"
            ]
        )

        if use_mlflow:

            with mlflow.start_run(
                run_name="stage_1_benchmarking",
                nested=True,
            ):

                mlflow.set_tag(
                    "stage",
                    "benchmarking",
                )

                mlflow.set_tag(
                    "selected_candidate",
                    best_candidate_name,
                )

                mlflow.log_metric(
                    "best_cv_rmse",
                    best_candidate_rmse,
                )

                # -------------------------------------------------------------
                # Individual model runs
                # -------------------------------------------------------------

                for _, row in (
                    df_results.iterrows()
                ):

                    model_name = str(
                        row["model_name"]
                    )

                    with mlflow.start_run(
                        run_name=(
                            f"benchmark_{model_name}"
                        ),
                        nested=True,
                    ):

                        mlflow.set_tag(
                            "stage",
                            "benchmarking_model",
                        )

                        mlflow.set_tag(
                            "model_name",
                            model_name,
                        )

                        mlflow.log_metric(
                            "cv_rmse_mean",
                            float(
                                row[
                                    "val_rmse_mean"
                                ]
                            ),
                        )

                        mlflow.log_metric(
                            "cv_rmse_std",
                            float(
                                row[
                                    "val_rmse_std"
                                ]
                            ),
                        )

                        mlflow.log_metric(
                            "cv_mae_mean",
                            float(
                                row[
                                    "val_mae_mean"
                                ]
                            ),
                        )

                        mlflow.log_metric(
                            "cv_r2_mean",
                            float(
                                row[
                                    "val_r2_mean"
                                ]
                            ),
                        )

                        mlflow.log_metric(
                            "fit_time_mean",
                            float(
                                row[
                                    "fit_time_mean"
                                ]
                            ),
                        )

        logger.info(
            f"Best benchmark candidate: "
            f"{best_candidate_name}"
        )

        # =====================================================================
        # STAGE 2 — HYPERPARAMETER TUNING
        # =====================================================================

        logger.info(
            "\n>>> "
            "[STAGE 2/4] "
            f"TUNING [{best_candidate_name}]"
        )

        best_params, tuned_cv_rmse = (
            run_tuning_for_model(
                model_name=best_candidate_name
            )
        )

        if use_mlflow:

            with mlflow.start_run(
                run_name="stage_2_tuning",
                nested=True,
            ):

                mlflow.set_tag(
                    "stage",
                    "tuning",
                )

                mlflow.set_tag(
                    "model_name",
                    best_candidate_name,
                )

                mlflow.log_metric(
                    "tuned_cv_rmse",
                    float(
                        tuned_cv_rmse
                    ),
                )

                for (
                    param_name,
                    param_value,
                ) in best_params.items():

                    # MLflow params should be serializable.
                    mlflow.log_param(
                        param_name,
                        str(param_value),
                    )

        # =====================================================================
        # STAGE 3 — FINAL FIT
        # =====================================================================

        logger.info(
            "\n>>> "
            "[STAGE 3/4] "
            "FINAL FIT ON TRAIN + VALIDATION"
        )

        final_artifact_path, fitted_pipeline = (
            train_and_save_final_model(
                model_name=best_candidate_name,
                best_params=best_params,
            )
        )

        if use_mlflow:

            with mlflow.start_run(
                run_name="stage_3_final_fit",
                nested=True,
            ):

                mlflow.set_tag(
                    "stage",
                    "final_fit",
                )

                mlflow.set_tag(
                    "model_name",
                    best_candidate_name,
                )

                # -------------------------------------------------------------
                # Local artifact
                # -------------------------------------------------------------

                if Path(
                    final_artifact_path
                ).exists():

                    mlflow.log_artifact(
                        str(
                            final_artifact_path
                        ),
                        artifact_path=(
                            "local_artifact"
                        ),
                    )

                # -------------------------------------------------------------
                # Native sklearn MLflow model (with skops serialization)
                # -------------------------------------------------------------

                # ✅ FIX: Use skops with trusted types to avoid MlflowException
                mlflow.sklearn.log_model(
                    sk_model=fitted_pipeline,
                    artifact_path="final_pipeline_model",
                    serialization_format="skops",
                    skops_trusted_types=[
                        "numpy.dtype",
                        "sklearn.compose._column_transformer._RemainderColsList",
                        "sklearn.metrics._dist_metrics.EuclideanDistance64",
                        "sklearn.neighbors._kd_tree.KDTree",
                        "src.features.engineering.FeatureConfig",
                        "src.features.engineering.FeatureEngineer",
                        "src.features.pipeline.DataFrameImputer",
                        "src.features.pipeline.LOFTransformer",
                    ],
                )

                logger.info(
                    "Final sklearn pipeline "
                    "logged to MLflow (with skops)."
                )

        # =====================================================================
        # STAGE 4 — TEST EVALUATION
        # =====================================================================

        logger.info(
            "\n>>> "
            "[STAGE 4/4] "
            "FINAL TEST EVALUATION"
        )

        test_metrics, plot_paths = (
            run_evaluation_on_test(
                pipeline=fitted_pipeline,
            )
        )

        if use_mlflow:

            with mlflow.start_run(
                run_name=(
                    "stage_4_test_evaluation"
                ),
                nested=True,
            ):

                mlflow.set_tag(
                    "stage",
                    "test_evaluation",
                )

                mlflow.set_tag(
                    "model_name",
                    best_candidate_name,
                )

                # -------------------------------------------------------------
                # Test metrics
                # -------------------------------------------------------------

                numeric_test_metrics = {}

                for key, value in (
                    test_metrics.items()
                ):

                    if isinstance(
                        value,
                        (
                            int,
                            float,
                            np.integer,
                            np.floating,
                        ),
                    ):

                        numeric_test_metrics[
                            key
                        ] = float(value)

                if numeric_test_metrics:

                    mlflow.log_metrics(
                        numeric_test_metrics
                    )

                # -------------------------------------------------------------
                # Evaluation plots
                # -------------------------------------------------------------

                for plot_path in plot_paths:

                    plot_file = Path(
                        plot_path
                    )

                    if plot_file.exists():

                        mlflow.log_artifact(
                            str(
                                plot_file
                            ),
                            artifact_path=(
                                "evaluation_plots"
                            ),
                        )

            # -----------------------------------------------------------------
            # Parent summary
            # -----------------------------------------------------------------

            mlflow.log_metrics(
                numeric_test_metrics
            )

            mlflow.set_tag(
                "final_model",
                best_candidate_name,
            )

            mlflow.set_tag(
                "pipeline_status",
                "completed",
            )

        # =====================================================================
        # SUMMARY
        # =====================================================================

        total_duration = (
            time.time()
            - pipeline_start_time
        )

        logger.info("=" * 90)

        logger.info(
            "MASTER ML PIPELINE COMPLETED"
        )

        logger.info(
            f"Duration: "
            f"{total_duration:.1f}s"
        )

        logger.info(
            f"Best model: "
            f"{best_candidate_name}"
        )

        logger.info(
            f"Final artifact: "
            f"{final_artifact_path}"
        )

        if "test_rmse" in test_metrics:

            logger.info(
                f"Test RMSE: "
                f"${float(test_metrics['test_rmse']):,.2f}"
            )

        if "test_mae" in test_metrics:

            logger.info(
                f"Test MAE: "
                f"${float(test_metrics['test_mae']):,.2f}"
            )

        if "test_r2" in test_metrics:

            logger.info(
                f"Test R²: "
                f"{float(test_metrics['test_r2']):.4f}"
            )

        logger.info("=" * 90)


# =============================================================================
# CLI ENTRY POINT
# =============================================================================


if __name__ == "__main__":

    try:

        execute_full_pipeline()

        sys.exit(0)

    except Exception as exc:

        logger.exception(
            f"Master pipeline execution failed: {exc}"
        )

        sys.exit(1)