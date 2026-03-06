from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from config import PathsConfig
from utils.file_utils import load_joblib_or_none


@dataclass
class SoundModelBundle:
    rf_model: Optional[Any]
    xgb_model: Optional[Any]
    cnn_lstm_model: Optional[Any]
    scaler: Optional[Any]
    hybrid_scaler: Optional[Any]
    label_encoder: Optional[Any]


class SoundModelLoader:
    def __init__(self, paths: PathsConfig, logger: logging.Logger) -> None:
        self.paths = paths
        self.logger = logger

    def _load_keras_model(self, path: Path) -> Optional[Any]:
        if not path.exists():
            return None
        try:
            from tensorflow.keras.models import load_model  # type: ignore

            return load_model(path)
        except Exception as exc:
            self.logger.warning("Could not load CNN-LSTM model (%s): %s", path, exc)
            return None

    def load(self) -> SoundModelBundle:
        rf_model = load_joblib_or_none(Path(self.paths.sound_rf_model_path))
        xgb_model = load_joblib_or_none(Path(self.paths.sound_xgb_model_path))
        scaler = load_joblib_or_none(Path(self.paths.sound_scaler_path))
        hybrid_scaler = load_joblib_or_none(Path(self.paths.sound_hybrid_scaler_path))
        label_encoder = load_joblib_or_none(Path(self.paths.label_encoder_path))
        cnn_lstm = self._load_keras_model(Path(self.paths.sound_cnn_lstm_model_path))

        if rf_model is None and xgb_model is None and cnn_lstm is None:
            self.logger.warning("No sound models found. Runtime will use heuristic fallback.")

        return SoundModelBundle(
            rf_model=rf_model,
            xgb_model=xgb_model,
            cnn_lstm_model=cnn_lstm,
            scaler=scaler,
            hybrid_scaler=hybrid_scaler,
            label_encoder=label_encoder,
        )
