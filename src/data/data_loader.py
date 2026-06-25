# src/data/data_loader.py

"""
Data loading module for California Housing project.

Provides a unified interface for loading CSV datasets from configured
storage paths (raw, interim, processed) with environment-aware resolution.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd

from src.utils.logger import get_logger
from src.utils.paths import PATHS, IN_COLAB

DataStage = Literal["raw", "interim", "processed"]

logger = get_logger(__name__)


class DataLoader:
    """
    Centralized data loader with environment-aware path resolution.

    Loads CSV files from configured data stages using the project's
    centralized PATHS registry. Supports both Google Drive (Colab)
    and local filesystem paths.

    Attributes:
        paths: Dictionary mapping data stages to resolved Path objects.
        base_path: Reference base path for fallback/local mode.
    """

    def __init__(self, use_drive_paths: bool | None = None) -> None:
        """
        Initialize DataLoader with appropriate path configuration.

        Args:
            use_drive_paths: Force Drive paths. Defaults to IN_COLAB detection.
        """
        self._use_drive = (
            use_drive_paths if use_drive_paths is not None else IN_COLAB
        )
        self.paths = self._resolve_paths()
        self.base_path = self._resolve_base_path()
        logger.info(f"DataLoader initialized | Drive mode: {self._use_drive}")

    def _resolve_paths(self) -> dict[DataStage, Path]:
        """Resolve path dictionary based on environment configuration."""
        if self._use_drive:
            return {
                "raw": PATHS["raw"],
                "interim": PATHS["interim"],
                "processed": PATHS["processed"],
            }
        fallback = Path.cwd()
        return {
            stage: fallback / "data" / stage
            for stage in ("raw", "interim", "processed")
        }

    def _resolve_base_path(self) -> Path:
        """Resolve base path reference for the current mode."""
        if self._use_drive:
            return PATHS["raw"].parent.parent
        return Path.cwd()

    def load_raw(self, filename: str) -> pd.DataFrame:
        """Load a CSV file from the raw data stage."""
        return self._load("raw", filename)

    def load_interim(self, filename: str) -> pd.DataFrame:
        """Load a CSV file from the interim data stage."""
        return self._load("interim", filename)

    def load_processed(self, filename: str) -> pd.DataFrame:
        """Load a CSV file from the processed data stage."""
        return self._load("processed", filename)

    def _load(self, stage: DataStage, filename: str) -> pd.DataFrame:
        """
        Internal loader with validation and error handling.

        Args:
            stage: Data lifecycle stage ('raw', 'interim', 'processed').
            filename: Target CSV filename.

        Returns:
            Loaded pandas DataFrame.

        Raises:
            ValueError: If stage is invalid.
            FileNotFoundError: If target file does not exist.
        """
        if stage not in self.paths:
            raise ValueError(f"Invalid stage: {stage}. Available: {list(self.paths)}")

        file_path = self.paths[stage] / filename
        logger.info(f"Loading: {file_path}")

        if not file_path.exists():
            available = (
                [f.name for f in self.paths[stage].glob("*.csv")]
                if self.paths[stage].exists()
                else []
            )
            logger.error(f"File not found: {file_path}")
            if available:
                logger.error(f"Available files in '{stage}': {available}")
            raise FileNotFoundError(f"{file_path} does not exist")

        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded '{filename}' | shape={df.shape} | stage={stage}")
            return df
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            raise


__all__ = ["DataLoader", "DataStage"]