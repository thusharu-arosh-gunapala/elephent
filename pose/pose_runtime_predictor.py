from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from pose.pose_feature_extractor import PoseFeatureExtractor
from pose.pose_model_loader import PoseModelBundle


@dataclass
class PosePrediction:
    label: str
    prob_aggressive: float
    features: Dict[str, float]
    success: bool


class PoseRuntimePredictor:
    def __init__(self, model_bundle: PoseModelBundle, logger: logging.Logger) -> None:
        self.model_bundle = model_bundle
        self.logger = logger
        self.extractor = PoseFeatureExtractor()

    def predict_from_frame(self, frame: np.ndarray) -> PosePrediction:
        try:
            feat_obj = self.extractor.extract_from_frame(frame)
            features = feat_obj.to_dict()

            selected_keys = self.model_bundle.top_features or list(features.keys())
            x = np.array([[features.get(k, 0.0) for k in selected_keys]], dtype=float)

            if self.model_bundle.scaler is not None:
                x = self.model_bundle.scaler.transform(x)

            if self.model_bundle.model is not None:
                prob_aggressive = float(self.model_bundle.model.predict_proba(x)[0][1])
            else:
                # Heuristic fallback score.
                prob_aggressive = float(np.clip((features["trunk_length"] + features["ear_spread"]) / 2.0, 0.0, 1.0))

            label = "Aggressive" if prob_aggressive >= 0.5 else "Normal"
            self.logger.info("Pose prediction: label=%s prob_aggressive=%.3f", label, prob_aggressive)
            return PosePrediction(label=label, prob_aggressive=prob_aggressive, features=features, success=True)
        except Exception as exc:
            self.logger.exception("Pose prediction failed: %s", exc)
            return PosePrediction(label="Unknown", prob_aggressive=0.0, features={}, success=False)
