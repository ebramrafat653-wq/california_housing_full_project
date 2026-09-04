# =============================================================================
# src/models/model_factory.py
# California Housing Project — Dynamic Model Factory
#
# RESPONSIBILITY:
#   Instantiate and configure ML model estimators dynamically based on
#   configs/model_config.yaml using dynamic imports and signature inspection.
#
# DESIGN:
#   - Open/Closed Principle: Add new models via YAML without changing Python code.
#   - Validates that imported classes inherit from sklearn.base.BaseEstimator.
#   - Uses inspect.signature to safely detect and inject random_state.
#   - Supports runtime parameter overrides for Hyperparameter Tuning (HPO).
# =============================================================================

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Any

from sklearn.base import BaseEstimator
import yaml

from src.utils.logger import get_logger
from src.utils.paths import PROJECT_DIR

logger = get_logger(__name__)

_DEFAULT_MODEL_CONFIG_PATH = PROJECT_DIR / "configs" / "model_config.yaml"


# =============================================================================
# CUSTOM EXCEPTION
# =============================================================================


class ModelFactoryError(Exception):
  """Raised when an error occurs during model creation."""


# =============================================================================
# CONFIG LOADER
# =============================================================================


def load_model_config(
    config_path: str | Path = _DEFAULT_MODEL_CONFIG_PATH,
) -> dict[str, Any]:
  """Load and validate model_config.yaml."""
  config_path = Path(config_path)

  if not config_path.exists():
    raise ModelFactoryError(f"Model config file not found: {config_path}")

  try:
    with open(config_path, encoding="utf-8") as f:
      cfg = yaml.safe_load(f)
  except yaml.YAMLError as exc:
    raise ModelFactoryError(
        f"Failed to parse model configuration: {config_path}"
    ) from exc

  if not isinstance(cfg, dict):
    raise ModelFactoryError("model_config.yaml must contain a valid mapping.")

  return cfg


# =============================================================================
# DYNAMIC CLASS IMPORTER & INTROSPECTION
# =============================================================================


def _import_estimator_class(class_path: str) -> type[BaseEstimator]:
  """Dynamically import an estimator class from a full module path string.

  Example
  -------
  'sklearn.ensemble.RandomForestRegressor' -> RandomForestRegressor class
  """
  if not isinstance(class_path, str) or "." not in class_path:
    raise ModelFactoryError(
        f"Invalid class path '{class_path}'. Expected format:"
        " 'module.submodule.ClassName'"
    )

  module_path, class_name = class_path.rsplit(".", 1)

  try:
    module = importlib.import_module(module_path)
  except ImportError as exc:
    raise ModelFactoryError(
        f"Could not import module '{module_path}': {exc}"
    ) from exc

  try:
    estimator_cls = getattr(module, class_name)
  except AttributeError as exc:
    raise ModelFactoryError(
        f"Module '{module_path}' has no class named '{class_name}': {exc}"
    ) from exc

  # التأكد من توافق الكلاس مع واجهة Scikit-Learn
  if not (
      isinstance(estimator_cls, type) and issubclass(estimator_cls, BaseEstimator)
  ):
    raise ModelFactoryError(
        f"Class '{class_path}' must be a subclass of sklearn.base.BaseEstimator."
    )

  return estimator_cls


def _supports_random_state(estimator_cls: type) -> bool:
  """Inspect whether the estimator's __init__ accepts 'random_state'."""
  try:
    sig = inspect.signature(estimator_cls.__init__)
    return "random_state" in sig.parameters
  except (ValueError, TypeError):
    return False


# =============================================================================
# MODEL FACTORY FUNCTION
# =============================================================================


def create_model(
    model_name: str,
    params_override: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    config_path: str | Path = _DEFAULT_MODEL_CONFIG_PATH,
) -> BaseEstimator:
  """Instantiate a model estimator dynamically based on model_config.yaml.

  Parameters
  ----------
  model_name : str
      Key identifying the model in config (e.g. 'ridge', 'random_forest',
      'gradient_boosting').
  params_override : dict[str, Any] | None
      Parameters to override defaults (used during HPO / tuning).
  config : dict[str, Any] | None
      Optional pre-loaded configuration dictionary.
  config_path : str | Path
      Path to configuration file if config is not provided.

  Returns
  -------
  BaseEstimator
      Unfitted, configured scikit-learn estimator instance.
  """
  model_name = model_name.lower().strip()

  # 1. Load configuration
  if config is None:
    config = load_model_config(config_path)

  models_section = config.get("models", {})
  if model_name not in models_section:
    raise ModelFactoryError(
        f"Model '{model_name}' not defined in models section of"
        f" {config_path}. Available models: {list(models_section.keys())}"
    )

  model_entry = models_section[model_name]
  if not isinstance(model_entry, dict):
    raise ModelFactoryError(
        f"Configuration entry for model '{model_name}' must be a dictionary."
    )

  # 2. Dynamic class loading from YAML
  class_path = model_entry.get("class")
  if not class_path:
    raise ModelFactoryError(
        f"Model '{model_name}' is missing the 'class' field in configuration."
    )

  estimator_cls = _import_estimator_class(class_path)

  # 3. Extract parameters and global seed
  params = model_entry.get("params", {}).copy()
  project_seed = config.get("project", {}).get("random_state", 42)

  # 4. Safely propagate random_state using introspection
  if _supports_random_state(estimator_cls):
    params.setdefault("random_state", project_seed)

  # 5. Apply runtime parameter overrides
  if params_override:
    params.update(params_override)

  # 6. Instantiate estimator
  try:
    estimator = estimator_cls(**params)
    logger.info(
        f"Successfully created '{model_name}' ({class_path}) with params:"
        f" {params}"
    )
    return estimator
  except Exception as exc:
    raise ModelFactoryError(
        f"Failed to instantiate '{model_name}' ({class_path}) with params"
        f" {params}: {exc}"
    ) from exc


def get_enabled_models(
    config: dict[str, Any] | None = None,
    config_path: str | Path = _DEFAULT_MODEL_CONFIG_PATH,
) -> list[str]:
  """Return a list of all model names enabled in model_config.yaml."""
  if config is None:
    config = load_model_config(config_path)

  models_cfg = config.get("models", {})
  enabled_models = [
      name
      for name, settings in models_cfg.items()
      if isinstance(settings, dict) and settings.get("enabled", True)
  ]
  return enabled_models


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    "ModelFactoryError",
    "load_model_config",
    "create_model",
    "get_enabled_models",
]