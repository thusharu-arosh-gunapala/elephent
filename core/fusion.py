from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FusionResult:
    final_label: str
    final_score: float


class DecisionFusion:
    def __init__(self, pose_weight: float, sound_weight: float, threshold: float) -> None:
        total = pose_weight + sound_weight
        self.pose_weight = pose_weight / total if total > 0 else 0.5
        self.sound_weight = sound_weight / total if total > 0 else 0.5
        self.threshold = threshold

    def fuse(self, pose_prob_aggressive: float, sound_score_aggressive: float) -> FusionResult:
        final_score = (self.pose_weight * pose_prob_aggressive) + (
            self.sound_weight * sound_score_aggressive
        )
        final_label = "Aggressive" if final_score >= self.threshold else "Normal"
        return FusionResult(final_label=final_label, final_score=float(final_score))
