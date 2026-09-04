# =============================================================================
# src/models/tuning.py
# California Housing Project — Pure Hyperparameter Tuning Engine
#
# RESPONSIBILITY
# --------------
# 1. Load the already-selected candidate model.
# 2. Load its hyperparameter search space from model_config.yaml.
# 3. Build the COMPLETE CV-safe sklearn pipeline.
# 4. Execute RandomizedSearchCV using TRAINING DATA ONLY.
# 5. Return the best hyperparameters and best CV score.
# 6. Save tuning results/reports.
#
# DESIGN
# ------
# - ZERO MLflow dependency.
# - ZERO DVC dependency.
# - ZERO Git dependency.
# - Validation/Test data are NEVER loaded.
# - All learned preprocessing remains inside the sklearn Pipeline.
# - Hyperparameter tuning operates on the COMPLETE pipeline.
# - Final Train + Validation fitting is intentionally NOT performed here.
#
# IMPORTANT
# ---------
# The returned best_params use sklearn pipeline parameter names such as:
#
#     model__alpha
#     model__n_estimators
#
# training.py is responsible for removing the "model__" prefix when creating
# the final raw estimator.
#
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
)

from src.features.pipeline import (
    build_model_pipeline,
    load_pipeline_config,
    resolve_columns,
    split_features_target,
)
from src.models.model_factory import (
    create_model,
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

_DEFAULT_REPORTS_DIR = (
    PROJECT_DIR / "reports"
)

_DEFAULT_TRAIN_TARGET = "median_house_value"


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================


class TuningEngineError(Exception):
    """Raised when hyperparameter tuning fails safely."""


# =============================================================================
# TRAINING DATA SCHEMA VALIDATION
# =============================================================================


def validate_tuning_schema(
    df: pd.DataFrame,
    target_col: str = _DEFAULT_TRAIN_TARGET,
) -> None:
    """
    Validate that tuning receives the original clean training schema.

    Hyperparameter tuning must receive train_clean.csv rather than train_feat.csv
    because the complete sklearn pipeline performs feature engineering internally.
    """

    if not isinstance(df, pd.DataFrame):
        raise TuningEngineError(
            "Tuning data must be a pandas DataFrame."
        )

    if df.empty:
        raise TuningEngineError(
            "Tuning DataFrame is empty."
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
        raise TuningEngineError(
            "Tuning input schema validation failed.\n"
            f"Missing required columns: {missing}\n"
            "RandomizedSearchCV must receive train_clean.csv "
            "with the original raw feature columns."
        )

    if "is_capped" in df.columns:
        raise TuningEngineError(
            "Target-derived feature 'is_capped' must never enter tuning."
        )

    if df[target_col].isna().any():
        raise TuningEngineError(
            f"Target column '{target_col}' contains missing values."
        )

    if not pd.api.types.is_numeric_dtype(
        df[target_col]
    ):
        raise TuningEngineError(
            f"Target column '{target_col}' must be numeric."
        )

    logger.info(
        "Tuning input schema validation passed."
    )


# =============================================================================
# SEARCH-SPACE VALIDATION
# =============================================================================


def validate_search_space(
    model_name: str,
    search_space: dict[str, Any],
) -> None:
    """
    Validate the configured RandomizedSearchCV parameter space.
    """

    if not isinstance(
        search_space,
        dict,
    ):
        raise TuningEngineError(
            f"Search space for '{model_name}' must be a dictionary."
        )

    if not search_space:
        raise TuningEngineError(
            f"No hyperparameter search space configured for "
            f"model '{model_name}'."
        )

    for parameter_name, values in search_space.items():

        if not isinstance(
            parameter_name,
            str,
        ):
            raise TuningEngineError(
                "Hyperparameter names must be strings."
            )

        if not parameter_name.startswith(
            "model__"
        ):
            raise TuningEngineError(
                f"Invalid tuning parameter '{parameter_name}' "
                f"for model '{model_name}'. "
                "Pipeline model parameters must start with "
                "'model__'."
            )

        if values is None:
            raise TuningEngineError(
                f"Search-space values for '{parameter_name}' "
                "cannot be None."
            )

        if isinstance(
            values,
            (str, bytes),
        ):
            raise TuningEngineError(
                f"Search-space value for '{parameter_name}' "
                "must be a list/distribution, not a string."
            )

    logger.info(
        f"Search space validated for [{model_name}] | "
        f"parameters={len(search_space)}"
    )


# =============================================================================
# CV CONFIGURATION
# =============================================================================


def build_tuning_cv(
    tuning_config: dict[str, Any],
) -> KFold:
    """
    Build the Cross-Validation splitter used by RandomizedSearchCV.
    """

    cv_config = tuning_config.get(
        "cv",
        {},
    )

    if not isinstance(
        cv_config,
        dict,
    ):
        raise TuningEngineError(
            "'tuning.cv' must be a dictionary."
        )

    strategy = cv_config.get(
        "strategy",
        "kfold",
    )

    if strategy != "kfold":
        raise TuningEngineError(
            f"Unsupported tuning CV strategy: '{strategy}'. "
            "This regression project expects 'kfold'."
        )

    n_splits = int(
        cv_config.get(
            "n_splits",
            5,
        )
    )

    shuffle = bool(
        cv_config.get(
            "shuffle",
            True,
        )
    )

    random_state = int(
        cv_config.get(
            "random_state",
            42,
        )
    )

    if n_splits < 2:
        raise TuningEngineError(
            "tuning.cv.n_splits must be >= 2."
        )

    return KFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )


# =============================================================================
# TUNING RESULT EXTRACTION
# =============================================================================


def extract_best_tuning_result(
    search: RandomizedSearchCV,
) -> tuple[dict[str, Any], float]:
    """
    Extract the best parameters and best CV score from RandomizedSearchCV.
    """

    if not hasattr(
        search,
        "best_params_",
    ):
        raise TuningEngineError(
            "RandomizedSearchCV did not produce best_params_."
        )

    if not hasattr(
        search,
        "best_score_",
    ):
        raise TuningEngineError(
            "RandomizedSearchCV did not produce best_score_."
        )

    best_params = dict(
        search.best_params_
    )

    # sklearn returns the configured negative scoring value.
    # For neg_root_mean_squared_error, invert it back to RMSE.
    raw_best_score = float(
        search.best_score_
    )

    scoring = str(
        search.scoring
    )

    if scoring == "neg_root_mean_squared_error":
        best_score = -raw_best_score
    elif scoring == "neg_mean_absolute_error":
        best_score = -raw_best_score
    else:
        best_score = raw_best_score

    return (
        best_params,
        float(best_score),
    )


# =============================================================================
# MAIN TUNING FUNCTION
# =============================================================================


def run_tuning_for_model(
    model_name: str,
    model_config_path: str | Path = (
        _DEFAULT_MODEL_CONFIG_PATH
    ),
    data_config_path: str | Path = (
        _DEFAULT_DATA_CONFIG_PATH
    ),
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], float]:
    """
    Run RandomizedSearchCV for the selected candidate model.

    IMPORTANT
    ---------
    Only train_clean.csv is loaded.

    Validation and Test are deliberately not loaded because:

        Validation/Test
            ↓
        must remain untouched

    RandomizedSearchCV performs its own CV exclusively on the training data.

    Returns
    -------
    tuple
        (
            best_params,
            best_cv_rmse
        )
    """

    logger.info("=" * 80)
    logger.info(
        f"HYPERPARAMETER TUNING STARTED | MODEL=[{model_name}]"
    )
    logger.info("=" * 80)

    start_time = time.time()

    # =========================================================================
    # 1. LOAD CONFIGURATION
    # =========================================================================

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
            _DEFAULT_TRAIN_TARGET,
        )
    )

    tuning_cfg = model_cfg.get(
        "tuning",
        {},
    )

    if not isinstance(
        tuning_cfg,
        dict,
    ):
        raise TuningEngineError(
            "'tuning' section must be a dictionary."
        )

    tuning_enabled = bool(
        tuning_cfg.get(
            "enabled",
            True,
        )
    )

    if not tuning_enabled:
        raise TuningEngineError(
            "Hyperparameter tuning is disabled in model_config.yaml."
        )

    # =========================================================================
    # 2. LOAD TRAINING DATA ONLY
    # =========================================================================

    train_path = (
        PROJECT_DIR
        / model_cfg["data"]["train_path"]
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {train_path}"
        )

    train_df = pd.read_csv(
        train_path
    )

    validate_tuning_schema(
        train_df,
        target_col=target_col,
    )

    logger.info(
        f"Tuning dataset loaded: "
        f"{train_path} | rows={len(train_df):,}"
    )

    # =========================================================================
    # 3. SPLIT X / y
    # =========================================================================

    cols = resolve_columns(
        pipeline_cfg
    )

    X_train, y_train = (
        split_features_target(
            train_df,
            target=target_col,
        )
    )

    # =========================================================================
    # 4. SEARCH SPACE
    # =========================================================================

    all_search_spaces = model_cfg.get(
        "search_spaces",
        {},
    )

    if not isinstance(
        all_search_spaces,
        dict,
    ):
        raise TuningEngineError(
            "'search_spaces' must be a dictionary."
        )

    if model_name not in all_search_spaces:
        raise TuningEngineError(
            f"No search space found for model '{model_name}'."
        )

    search_space = (
        all_search_spaces[model_name]
    )

    validate_search_space(
        model_name=model_name,
        search_space=search_space,
    )

    # =========================================================================
    # 5. CREATE BASE MODEL
    # =========================================================================

    raw_model = create_model(
        model_name=model_name,
        params_override=None,
        config=model_cfg,
    )

    # =========================================================================
    # 6. BUILD COMPLETE CV-SAFE PIPELINE
    # =========================================================================

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

    logger.info(
        "Complete CV-safe model pipeline built "
        "for RandomizedSearchCV."
    )

    # =========================================================================
    # 7. BUILD CV SPLITTER
    # =========================================================================

    cv_splitter = build_tuning_cv(
        tuning_cfg
    )

    # =========================================================================
    # 8. TUNING CONFIG
    # =========================================================================

    scoring = tuning_cfg.get(
        "scoring",
        "neg_root_mean_squared_error",
    )

    n_iter = int(
        tuning_cfg.get(
            "n_iter",
            30,
        )
    )

    n_jobs = int(
        tuning_cfg.get(
            "n_jobs",
            -1,
        )
    )

    random_state = int(
        tuning_cfg.get(
            "random_state",
            42,
        )
    )

    if n_iter < 1:
        raise TuningEngineError(
            "tuning.n_iter must be >= 1."
        )

    # =========================================================================
    # 9. IMPORTANT: NO REFIT DURING TUNING
    # =========================================================================
    #
    # The final production fit is deliberately handled by:
    #
    #     training.py
    #
    # on:
    #
    #     Train + Validation
    #
    # Therefore RandomizedSearchCV does not refit the best estimator here.
    #
    # This prevents an unnecessary duplicate fit on Train only.
    #
    # =========================================================================

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=search_space,
        n_iter=n_iter,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv_splitter,
        refit=False,
        random_state=random_state,
        return_train_score=True,
        error_score="raise",
    )

    # =========================================================================
    # 10. EXECUTE RANDOMIZED SEARCH
    # =========================================================================

    logger.info(
        "Starting RandomizedSearchCV | "
        f"n_iter={n_iter} | "
        f"cv={cv_splitter.n_splits} | "
        f"n_jobs={n_jobs} | "
        f"scoring={scoring}"
    )

    try:

        search.fit(
            X_train,
            y_train,
        )

    except Exception as exc:

        raise TuningEngineError(
            f"RandomizedSearchCV failed for "
            f"model '{model_name}': {exc}"
        ) from exc

    # =========================================================================
    # 11. EXTRACT BEST RESULT
    # =========================================================================

    best_params, best_cv_score = (
        extract_best_tuning_result(
            search
        )
    )

    elapsed = (
        time.time()
        - start_time
    )

    logger.info("=" * 80)
    logger.info(
        f"BEST PARAMETERS [{model_name}]"
    )
    logger.info(
        f"{best_params}"
    )
    logger.info(
        f"Best CV Score: {best_cv_score:.6f}"
    )
    logger.info(
        f"Tuning Duration: {elapsed:.2f}s"
    )
    logger.info("=" * 80)

    # =========================================================================
    # 12. BUILD TUNING RESULTS DATAFRAME
    # =========================================================================

    cv_results = pd.DataFrame(
        search.cv_results_
    )

    # -------------------------------------------------------------------------
    # Convert sklearn negative RMSE into positive RMSE for reporting.
    # -------------------------------------------------------------------------

    if (
        "mean_test_score"
        in cv_results.columns
    ):

        if scoring == (
            "neg_root_mean_squared_error"
        ):

            cv_results[
                "mean_test_rmse"
            ] = -cv_results[
                "mean_test_score"
            ]

            cv_results[
                "std_test_rmse"
            ] = cv_results[
                "std_test_score"
            ]

    # -------------------------------------------------------------------------
    # Sort results according to the scoring function.
    # -------------------------------------------------------------------------

    if (
        "rank_test_score"
        in cv_results.columns
    ):

        cv_results = (
            cv_results
            .sort_values(
                by="rank_test_score"
            )
            .reset_index(
                drop=True
            )
        )

    # =========================================================================
    # 13. SAVE REPORTS
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

    tuning_results_path = (
        reports_dir
        / f"tuning_results_{model_name}.csv"
    )

    cv_results.to_csv(
        tuning_results_path,
        index=False,
    )

    # -------------------------------------------------------------------------
    # JSON-safe best parameters
    # -------------------------------------------------------------------------

    serializable_params = {}

    for key, value in best_params.items():

        if isinstance(
            value,
            np.generic,
        ):
            value = value.item()

        serializable_params[
            key
        ] = value

    tuning_report = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "model_name": model_name,
        "scoring": scoring,
        "n_iter": n_iter,
        "cv": {
            "strategy": "kfold",
            "n_splits": cv_splitter.n_splits,
            "shuffle": cv_splitter.shuffle,
            "random_state": cv_splitter.random_state,
        },
        "n_jobs": n_jobs,
        "random_state": random_state,
        "best_cv_score": best_cv_score,
        "best_params": serializable_params,
        "duration_seconds": elapsed,
        "train_rows": len(train_df),
        "validation_used": False,
        "test_used": False,
        "refit_during_tuning": False,
    }

    tuning_report_path = (
        reports_dir
        / f"tuning_report_{model_name}.json"
    )

    with open(
        tuning_report_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            tuning_report,
            file,
            indent=2,
        )

    logger.info(
        f"Tuning results saved -> "
        f"{tuning_results_path}"
    )

    logger.info(
        f"Tuning report saved -> "
        f"{tuning_report_path}"
    )

    logger.info(
        "HYPERPARAMETER TUNING COMPLETED SUCCESSFULLY."
    )

    return (
        best_params,
        best_cv_score,
    )


# =============================================================================
# CLI
# =============================================================================


if __name__ == "__main__":

    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Run RandomizedSearchCV for a selected "
            "California Housing model."
        )
    )

    parser.add_argument(
        "model_name",
        type=str,
        help=(
            "Model name exactly as configured in "
            "model_config.yaml."
        ),
    )

    args = parser.parse_args()

    try:

        best_params, best_cv_score = (
            run_tuning_for_model(
                model_name=args.model_name
            )
        )

        print(
            "\n"
            + "=" * 80
        )

        print(
            f"TUNING SUMMARY — {args.model_name}"
        )

        print(
            "=" * 80
        )

        print(
            f"Best CV RMSE: {best_cv_score:,.4f}"
        )

        print(
            "\nBest Parameters:"
        )

        for key, value in best_params.items():

            print(
                f"  {key}: {value}"
            )

        print(
            "=" * 80
            + "\n"
        )

        sys.exit(0)

    except Exception as exc:

        logger.exception(
            f"Tuning failed: {exc}"
        )

        sys.exit(1)


# =============================================================================
# PUBLIC API
# =============================================================================


__all__ = [
    "TuningEngineError",
    "validate_tuning_schema",
    "validate_search_space",
    "build_tuning_cv",
    "extract_best_tuning_result",
    "run_tuning_for_model",
]