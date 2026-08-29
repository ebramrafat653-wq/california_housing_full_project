# =============================================================================
# src/features/engineering.py
# California Housing Project — Feature Engineering
#
# RESPONSIBILITY:
#   Feature engineering only.
#
# Pipeline position:
#   cleaning.py
#       ↓
#   train_clean.csv / val_clean.csv / test_clean.csv
#       ↓
#   engineering.py
#       ↓
#   train_feat.csv / val_feat.csv / test_feat.csv
#       ↓
#   pipeline.py
#
# IMPORTANT:
#   This module does NOT manage DVC or Git.
#   DVC orchestration will be handled by dvc.yaml.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.utils.logger import get_logger
from src.utils.paths import ensure_path

logger = get_logger(__name__)


# =============================================================================
# FALLBACK DEFAULTS
# =============================================================================
# These are only development fallbacks.
# In the final DVC pipeline, missing configuration should ideally fail fast.
# =============================================================================

_FALLBACK_RATIOS: dict[str, str] = {
    "rooms_per_household": "total_rooms / households",
    "bedrooms_per_room": "total_bedrooms / total_rooms",
    "population_per_household": "population / households",
}

_FALLBACK_DISTANCES: dict[str, dict[str, float]] = {
    "dist_SF": {
        "lat": 37.77,
        "lon": -122.42,
    },
    "dist_LA": {
        "lat": 34.05,
        "lon": -118.24,
    },
}

_FALLBACK_DROP_COLS: list[str] = [
    "total_rooms",
    "total_bedrooms",
    "population",
    "households",
]


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""

    ratios: dict[str, str]
    distances: dict[str, dict[str, float]]
    drop_cols: list[str]

    source: str = "config"


def load_feature_config(
    config_path: str | Path = "configs/data_config.yaml",
) -> FeatureConfig:
    """
    Load feature engineering configuration.

    Expected YAML structure:

    eda_derived:
      engineered_features:
        ratios:
          rooms_per_household: "total_rooms / households"

        distances:
          dist_SF:
            lat: 37.77
            lon: -122.42

        drop_after_engineering:
          - total_rooms
          - total_bedrooms
          - population
          - households
    """

    config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(
            f"Config not found: {config_path}. "
            "Using fallback feature configuration."
        )

        return FeatureConfig(
            ratios=_FALLBACK_RATIOS.copy(),
            distances=_FALLBACK_DISTANCES.copy(),
            drop_cols=_FALLBACK_DROP_COLS.copy(),
            source="fallback",
        )

    try:
        with open(config_path, encoding="utf-8") as file:
            cfg = yaml.safe_load(file)

    except yaml.YAMLError as exc:
        raise FeatureEngineeringError(
            f"Malformed YAML configuration: {config_path}"
        ) from exc

    if not isinstance(cfg, dict):
        raise FeatureEngineeringError(
            f"Invalid configuration format: {config_path}"
        )

    eng = (
        cfg
        .get("eda_derived", {})
        .get("engineered_features")
    )

    if not eng:
        logger.warning(
            "Missing 'eda_derived.engineered_features'. "
            "Using fallback feature configuration."
        )

        return FeatureConfig(
            ratios=_FALLBACK_RATIOS.copy(),
            distances=_FALLBACK_DISTANCES.copy(),
            drop_cols=_FALLBACK_DROP_COLS.copy(),
            source="fallback",
        )

    ratios = eng.get(
        "ratios",
        _FALLBACK_RATIOS,
    )

    distances = eng.get(
        "distances",
        _FALLBACK_DISTANCES,
    )

    drop_cols = eng.get(
        "drop_after_engineering",
        _FALLBACK_DROP_COLS,
    )

    if not isinstance(ratios, dict):
        raise FeatureEngineeringError(
            "'ratios' must be a dictionary."
        )

    if not isinstance(distances, dict):
        raise FeatureEngineeringError(
            "'distances' must be a dictionary."
        )

    if not isinstance(drop_cols, list):
        raise FeatureEngineeringError(
            "'drop_after_engineering' must be a list."
        )

    logger.info(
        f"Feature config loaded: "
        f"{len(ratios)} ratios, "
        f"{len(distances)} distances, "
        f"{len(drop_cols)} columns to drop."
    )

    return FeatureConfig(
        ratios=ratios,
        distances=distances,
        drop_cols=drop_cols,
        source="config",
    )


# =============================================================================
# RESULT
# =============================================================================

@dataclass
class EngineeringResult:
    """Result of the feature engineering stage."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    train_path: Path | None = None
    val_path: Path | None = None
    test_path: Path | None = None

    features_added: list[str] = field(default_factory=list)
    features_dropped: list[str] = field(default_factory=list)

    feature_config_source: str = "config"

    timestamp: datetime = field(
        default_factory=datetime.now
    )

    warnings: list[str] = field(
        default_factory=list
    )

    def summary(self) -> str:
        """Return a readable summary of the engineering result."""

        lines = [
            "=" * 60,
            "FEATURE ENGINEERING RESULT",
            "=" * 60,
            (
                f"Train : "
                f"{len(self.train):,} rows x "
                f"{self.train.shape[1]} cols"
            ),
            (
                f"Val   : "
                f"{len(self.val):,} rows x "
                f"{self.val.shape[1]} cols"
            ),
            (
                f"Test  : "
                f"{len(self.test):,} rows x "
                f"{self.test.shape[1]} cols"
            ),
            "",
            f"Config source    : {self.feature_config_source}",
            f"Features added   : {self.features_added}",
            f"Features dropped : {self.features_dropped}",
        ]

        if self.train_path:
            lines.extend(
                [
                    "",
                    "Saved outputs:",
                    f"  train -> {self.train_path}",
                    f"  val   -> {self.val_path}",
                    f"  test  -> {self.test_path}",
                ]
            )

        if self.warnings:
            lines.append("")
            lines.append("Warnings:")

            for warning in self.warnings:
                lines.append(f"  - {warning}")

        lines.append("=" * 60)

        return "\n".join(lines)


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================

class FeatureEngineeringError(Exception):
    """Raised when feature engineering cannot continue safely."""


# =============================================================================
# STEP 1 — RATIO FEATURES
# =============================================================================

def add_ratio_features(
    df: pd.DataFrame,
    ratios: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create ratio features.

    Example:
        rooms_per_household =
            total_rooms / households

    Zero denominators produce NaN instead of infinity.
    """

    df = df.copy()
    added: list[str] = []

    for feature_name, formula in ratios.items():

        try:
            parts = [
                part.strip()
                for part in formula.split("/")
            ]

            if len(parts) != 2:
                logger.warning(
                    f"Invalid ratio formula "
                    f"'{formula}'. Expected 'col_a / col_b'."
                )
                continue

            numerator, denominator = parts

            if numerator not in df.columns:
                logger.warning(
                    f"Ratio '{feature_name}': "
                    f"missing numerator '{numerator}'."
                )
                continue

            if denominator not in df.columns:
                logger.warning(
                    f"Ratio '{feature_name}': "
                    f"missing denominator '{denominator}'."
                )
                continue

            denominator_values = df[denominator]

            zero_count = (
                denominator_values == 0
            ).sum()

            if zero_count > 0:
                logger.warning(
                    f"Ratio '{feature_name}': "
                    f"{zero_count:,} zero denominator values."
                )

            # Avoid inf explicitly.
            safe_denominator = denominator_values.replace(
                0,
                np.nan,
            )

            df[feature_name] = (
                df[numerator] /
                safe_denominator
            )

            inf_count = np.isinf(
                df[feature_name]
            ).sum()

            if inf_count > 0:
                df[feature_name] = (
                    df[feature_name]
                    .replace(
                        [np.inf, -np.inf],
                        np.nan,
                    )
                )

            added.append(feature_name)

            logger.info(
                f"Created ratio '{feature_name}' | "
                f"nulls={df[feature_name].isna().sum():,}"
            )

        except Exception as exc:
            logger.warning(
                f"Ratio '{feature_name}' failed: {exc}"
            )

    return df, added


# =============================================================================
# STEP 2 — DISTANCE FEATURES
# =============================================================================

def add_distance_features(
    df: pd.DataFrame,
    distances: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Create Euclidean geographic distance features.

    Distance is measured in latitude/longitude degree space.

    Example:
        dist_SF =
            sqrt(
                (latitude - SF_lat)^2 +
                (longitude - SF_lon)^2
            )
    """

    df = df.copy()
    added: list[str] = []

    required = {
        "latitude",
        "longitude",
    }

    missing = required - set(df.columns)

    if missing:
        logger.warning(
            "Distance features skipped. "
            f"Missing columns: {sorted(missing)}"
        )
        return df, added

    for feature_name, hub in distances.items():

        try:
            if not isinstance(hub, dict):
                logger.warning(
                    f"Distance '{feature_name}': "
                    "hub configuration must be a dictionary."
                )
                continue

            hub_lat = hub.get("lat")
            hub_lon = hub.get("lon")

            if hub_lat is None or hub_lon is None:
                logger.warning(
                    f"Distance '{feature_name}': "
                    "missing 'lat' or 'lon'."
                )
                continue

            df[feature_name] = np.sqrt(
                (
                    df["latitude"] - hub_lat
                ) ** 2
                +
                (
                    df["longitude"] - hub_lon
                ) ** 2
            )

            added.append(feature_name)

            logger.info(
                f"Created distance '{feature_name}' "
                f"to ({hub_lat}, {hub_lon})"
            )

        except Exception as exc:
            logger.warning(
                f"Distance '{feature_name}' failed: {exc}"
            )

    return df, added


# =============================================================================
# STEP 3 — DROP RAW COLUMNS
# =============================================================================

def drop_raw_columns(
    df: pd.DataFrame,
    drop_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """
    Drop raw count columns after creating ratio features.
    """

    df = df.copy()

    existing = [
        col
        for col in drop_cols
        if col in df.columns
    ]

    missing = [
        col
        for col in drop_cols
        if col not in df.columns
    ]

    if missing:
        logger.warning(
            f"Columns requested for drop but not found: "
            f"{missing}"
        )

    if existing:
        df = df.drop(
            columns=existing
        )

    for col in existing:
        logger.info(
            f"Dropped raw column: '{col}'"
        )

    return df, existing


# =============================================================================
# VALIDATION
# =============================================================================

def validate_engineered_splits(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    """
    Validate basic consistency across train/val/test.
    """

    if train.empty:
        raise FeatureEngineeringError(
            "Engineered train DataFrame is empty."
        )

    if val.empty:
        raise FeatureEngineeringError(
            "Engineered validation DataFrame is empty."
        )

    if test.empty:
        raise FeatureEngineeringError(
            "Engineered test DataFrame is empty."
        )

    train_columns = set(train.columns)
    val_columns = set(val.columns)
    test_columns = set(test.columns)

    if train_columns != val_columns:
        raise FeatureEngineeringError(
            "Train and validation columns do not match."
        )

    if train_columns != test_columns:
        raise FeatureEngineeringError(
            "Train and test columns do not match."
        )

    # Target must remain available for downstream preprocessing/training.
    target = "median_house_value"

    for name, df in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:
        if target not in df.columns:
            raise FeatureEngineeringError(
                f"Target '{target}' missing from {name}."
            )


# =============================================================================
# SAVE
# =============================================================================

def save_featured_splits(
    result: EngineeringResult,
    output_dir: Path | None = None,
) -> EngineeringResult:
    """
    Save feature-engineered splits.

    Outputs:
        train_feat.csv
        val_feat.csv
        test_feat.csv
    """

    output_dir = (
        output_dir
        if output_dir is not None
        else ensure_path("processed")
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = {
        "train": output_dir / "train_feat.csv",
        "val": output_dir / "val_feat.csv",
        "test": output_dir / "test_feat.csv",
    }

    result.train.to_csv(
        paths["train"],
        index=False,
    )

    result.val.to_csv(
        paths["val"],
        index=False,
    )

    result.test.to_csv(
        paths["test"],
        index=False,
    )

    result.train_path = paths["train"]
    result.val_path = paths["val"]
    result.test_path = paths["test"]

    for name, path in paths.items():
        size_mb = (
            path.stat().st_size
            / (1024 * 1024)
        )

        logger.info(
            f"Saved {name}_feat.csv -> "
            f"{path} ({size_mb:.2f} MB)"
        )

    return result


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def run_feature_engineering(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
    config_path: str | Path = "configs/data_config.yaml",
    output_dir: Path | None = None,
) -> EngineeringResult:
    """
    Run the complete feature engineering stage.

    Order:
        1. Load configuration
        2. Create ratio features
        3. Create distance features
        4. Drop raw count columns
        5. Validate splits
        6. Save outputs

    No fitting/statistical learning happens here.
    Therefore, this stage does not introduce train/validation/test
    leakage.
    """

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING STARTED")
    logger.info("=" * 60)

    # -------------------------------------------------------------------------
    # Input validation
    # -------------------------------------------------------------------------

    for name, df in [
        ("train", train),
        ("val", val),
        ("test", test),
    ]:

        if not isinstance(df, pd.DataFrame):
            raise FeatureEngineeringError(
                f"'{name}' must be a pandas DataFrame."
            )

        if df.empty:
            raise FeatureEngineeringError(
                f"Input '{name}' DataFrame is empty."
            )

    # -------------------------------------------------------------------------
    # Load configuration
    # -------------------------------------------------------------------------

    feature_config = load_feature_config(
        config_path
    )

    if feature_config.source == "fallback":
        logger.warning(
            "Using fallback feature configuration."
        )

    # -------------------------------------------------------------------------
    # Step 1 — Ratios
    # -------------------------------------------------------------------------

    logger.info(
        "Step 1/3 — Creating ratio features"
    )

    train, train_added = add_ratio_features(
        train,
        feature_config.ratios,
    )

    val, _ = add_ratio_features(
        val,
        feature_config.ratios,
    )

    test, _ = add_ratio_features(
        test,
        feature_config.ratios,
    )

    # -------------------------------------------------------------------------
    # Step 2 — Distances
    # -------------------------------------------------------------------------

    logger.info(
        "Step 2/3 — Creating distance features"
    )

    train, distance_added = add_distance_features(
        train,
        feature_config.distances,
    )

    val, _ = add_distance_features(
        val,
        feature_config.distances,
    )

    test, _ = add_distance_features(
        test,
        feature_config.distances,
    )

    # -------------------------------------------------------------------------
    # Step 3 — Drop raw columns
    # -------------------------------------------------------------------------

    logger.info(
        "Step 3/3 — Dropping raw count columns"
    )

    train, train_dropped = drop_raw_columns(
        train,
        feature_config.drop_cols,
    )

    val, _ = drop_raw_columns(
        val,
        feature_config.drop_cols,
    )

    test, _ = drop_raw_columns(
        test,
        feature_config.drop_cols,
    )

    # -------------------------------------------------------------------------
    # Result
    # -------------------------------------------------------------------------

    result = EngineeringResult(
        train=train,
        val=val,
        test=test,
        features_added=(
            train_added +
            distance_added
        ),
        features_dropped=train_dropped,
        feature_config_source=(
            feature_config.source
        ),
    )

    # -------------------------------------------------------------------------
    # Validate
    # -------------------------------------------------------------------------

    validate_engineered_splits(
        result.train,
        result.val,
        result.test,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    result = save_featured_splits(
        result,
        output_dir=output_dir,
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
        "Loading cleaned splits..."
    )

    train = loader.load_processed(
        "train_clean.csv"
    )

    val = loader.load_processed(
        "val_clean.csv"
    )

    test = loader.load_processed(
        "test_clean.csv"
    )

    result = run_feature_engineering(
        train,
        val,
        test,
    )

    print(result.summary())


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "FeatureEngineeringError",
    "FeatureConfig",
    "EngineeringResult",
    "load_feature_config",
    "add_ratio_features",
    "add_distance_features",
    "drop_raw_columns",
    "validate_engineered_splits",
    "save_featured_splits",
    "run_feature_engineering",
]
