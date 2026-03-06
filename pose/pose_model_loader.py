from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional

from config import PathsConfig
from utils.file_utils import load_joblib_or_none


@dataclass
class PoseModelBundle:
    model: Optional[Any]
    scaler: Optional[Any]
    top_features: Optional[List[str]]


class PoseModelLoader:
    def __init__(self, paths: PathsConfig, logger: logging.Logger) -> None:
        self.paths = paths
        self.logger = logger

    def load(self) -> PoseModelBundle:
        model = load_joblib_or_none(Path(self.paths.pose_rf_model_path))
        scaler = load_joblib_or_none(Path(self.paths.pose_scaler_path))

        top_features = None
        if Path(self.paths.pose_top_features_path).exists():
            with open(self.paths.pose_top_features_path, "r", encoding="utf-8") as f:
                top_features = json.load(f)

        if model is None:
            self.logger.warning("Pose RF model not found at %s. Using fallback inference.", self.paths.pose_rf_model_path)

        return PoseModelBundle(model=model, scaler=scaler, top_features=top_features)
