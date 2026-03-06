from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np


@dataclass
class PoseFeatures:
    ear_spread: float
    trunk_length: float
    tail_length: float
    front_leg_stance: float
    back_leg_stance: float
    body_ratio: float
    shoulder_angle: float
    hip_angle: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "ear_spread": self.ear_spread,
            "trunk_length": self.trunk_length,
            "tail_length": self.tail_length,
            "front_leg_stance": self.front_leg_stance,
            "back_leg_stance": self.back_leg_stance,
            "body_ratio": self.body_ratio,
            "shoulder_angle": self.shoulder_angle,
            "hip_angle": self.hip_angle,
        }


class PoseFeatureExtractor:
    """Extract engineered geometry features from pose keypoints.

    In prototype mode, if no keypoints are available, frame-level heuristics can be used.
    """

    def extract_from_keypoints(self, keypoints: Dict[str, Tuple[float, float]]) -> PoseFeatures:
        def dist(a: str, b: str) -> float:
            pa, pb = np.array(keypoints[a]), np.array(keypoints[b])
            return float(np.linalg.norm(pa - pb))

        ear_spread = dist("left_ear", "right_ear")
        trunk_length = dist("trunk_base", "trunk_tip")
        tail_length = dist("tail_base", "tail_tip")
        front_leg_stance = (dist("left_front_leg", "right_front_leg") + 1e-6) / (
            dist("left_shoulder", "right_shoulder") + 1e-6
        )
        back_leg_stance = (dist("left_back_leg", "right_back_leg") + 1e-6) / (
            dist("left_hip", "right_hip") + 1e-6
        )
        body_ratio = (dist("left_shoulder", "right_shoulder") + 1e-6) / (
            dist("left_hip", "right_hip") + 1e-6
        )

        shoulder_angle = self._angle(
            keypoints["left_shoulder"], keypoints["neck"], keypoints["right_shoulder"]
        )
        hip_angle = self._angle(keypoints["left_hip"], keypoints["spine"], keypoints["right_hip"])

        return PoseFeatures(
            ear_spread=ear_spread,
            trunk_length=trunk_length,
            tail_length=tail_length,
            front_leg_stance=front_leg_stance,
            back_leg_stance=back_leg_stance,
            body_ratio=body_ratio,
            shoulder_angle=shoulder_angle,
            hip_angle=hip_angle,
        )

    def extract_from_frame(self, frame: np.ndarray) -> PoseFeatures:
        """Fallback extractor based on frame statistics when keypoints are unavailable."""
        gray = frame.mean(axis=2) if frame.ndim == 3 else frame
        h, w = gray.shape[:2]
        col_var = float(np.var(gray.mean(axis=0)))
        row_var = float(np.var(gray.mean(axis=1)))
        energy = float(np.mean(np.abs(np.diff(gray, axis=0))))

        return PoseFeatures(
            ear_spread=min(1.0, col_var / 1000.0),
            trunk_length=min(1.0, energy / 20.0),
            tail_length=min(1.0, row_var / 1000.0),
            front_leg_stance=min(1.0, (h / (w + 1e-6))),
            back_leg_stance=min(1.0, (w / (h + 1e-6))),
            body_ratio=(w + 1e-6) / (h + 1e-6),
            shoulder_angle=min(180.0, col_var / 10.0),
            hip_angle=min(180.0, row_var / 10.0),
        )

    @staticmethod
    def _angle(a: Iterable[float], b: Iterable[float], c: Iterable[float]) -> float:
        pa, pb, pc = np.array(a), np.array(b), np.array(c)
        ba, bc = pa - pb, pc - pb
        denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-6
        cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
        return float(np.degrees(np.arccos(cosine)))
