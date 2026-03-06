from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from sound.sound_feature_extractor import SoundFeatureExtractor
from sound.sound_model_loader import SoundModelBundle


@dataclass
class SoundPrediction:
    label: str
    score_aggressive: float
    model_votes: Dict[str, float]
    features: Dict[str, float]
    success: bool


class SoundRuntimePredictor:
    """Hybrid sound predictor combining RF, XGB, CNN-LSTM and rule logic."""

    def __init__(self, model_bundle: SoundModelBundle, logger: logging.Logger) -> None:
        self.model_bundle = model_bundle
        self.logger = logger
        self.extractor = SoundFeatureExtractor()
        self.model_weights = {
            "rf": 0.3,
            "xgb": 0.3,
            "cnn_lstm": 0.3,
            "rule": 0.1,
        }

    def predict(self, audio: np.ndarray, sample_rate: int) -> SoundPrediction:
        try:
            features = self.extractor.extract(audio, sample_rate).to_dict()
            x = np.array([list(features.values())], dtype=float)

            if self.model_bundle.scaler is not None:
                x_scaled = self.model_bundle.scaler.transform(x)
            else:
                x_scaled = x

            votes = {
                "rf": self._predict_sklearn(self.model_bundle.rf_model, x_scaled),
                "xgb": self._predict_sklearn(self.model_bundle.xgb_model, x_scaled),
                "cnn_lstm": self._predict_cnn(audio),
                "rule": self._rule_score(features),
            }

            final_score = self._weighted_score(votes)
            label = "Aggressive" if final_score >= 0.5 else "Normal"
            self.logger.info("Sound prediction: label=%s score_aggressive=%.3f votes=%s", label, final_score, votes)
            return SoundPrediction(
                label=label,
                score_aggressive=final_score,
                model_votes=votes,
                features=features,
                success=True,
            )
        except Exception as exc:
            self.logger.exception("Sound prediction failed: %s", exc)
            return SoundPrediction(label="Unknown", score_aggressive=0.0, model_votes={}, features={}, success=False)

    def _predict_sklearn(self, model: object, x: np.ndarray) -> float:
        if model is None:
            return 0.0
        if hasattr(model, "predict_proba"):
            return float(model.predict_proba(x)[0][1])
        pred = float(model.predict(x)[0])
        return float(np.clip(pred, 0.0, 1.0))

    def _predict_cnn(self, audio: np.ndarray) -> float:
        model = self.model_bundle.cnn_lstm_model
        if model is None:
            return 0.0
        # Simple reshape for placeholder sequence interface.
        x = np.expand_dims(np.expand_dims(audio[:512], axis=0), axis=-1)
        pred = float(np.squeeze(model.predict(x, verbose=0)))
        return float(np.clip(pred, 0.0, 1.0))

    @staticmethod
    def _rule_score(features: Dict[str, float]) -> float:
        rms = features.get("rms", 0.0)
        centroid = features.get("spectral_centroid", 0.0)
        zcr = features.get("zcr", 0.0)
        score = (rms * 1.2) + (zcr * 0.8) + min(1.0, centroid / 3000.0) * 0.5
        return float(np.clip(score / 2.0, 0.0, 1.0))

    def _weighted_score(self, votes: Dict[str, float]) -> float:
        available_models: List[str] = [k for k, v in votes.items() if v > 0.0 or k == "rule"]
        if not available_models:
            return 0.0

        weights = {k: self.model_weights[k] for k in available_models}
        total = sum(weights.values()) + 1e-6
        return float(sum(votes[k] * weights[k] for k in available_models) / total)
