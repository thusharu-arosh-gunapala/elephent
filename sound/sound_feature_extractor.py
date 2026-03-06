from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SoundFeatures:
    rms: float
    zcr: float
    spectral_centroid: float
    spectral_bandwidth: float
    mfcc_1: float
    mfcc_2: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "rms": self.rms,
            "zcr": self.zcr,
            "spectral_centroid": self.spectral_centroid,
            "spectral_bandwidth": self.spectral_bandwidth,
            "mfcc_1": self.mfcc_1,
            "mfcc_2": self.mfcc_2,
        }


class SoundFeatureExtractor:
    def extract(self, audio: np.ndarray, sample_rate: int) -> SoundFeatures:
        audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        if len(audio) == 0:
            raise ValueError("Empty audio signal")

        rms = float(np.sqrt(np.mean(np.square(audio))))
        zcr = float(np.mean(np.abs(np.diff(np.signbit(audio)).astype(np.float32))))

        magnitude = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
        mag_sum = np.sum(magnitude) + 1e-6

        spectral_centroid = float(np.sum(freqs * magnitude) / mag_sum)
        spectral_bandwidth = float(np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / mag_sum))

        # Lightweight MFCC-like placeholders from log mel bins.
        mel_bins = np.log1p(magnitude[: min(64, len(magnitude))])
        mfcc_1 = float(np.mean(mel_bins))
        mfcc_2 = float(np.std(mel_bins))

        return SoundFeatures(
            rms=rms,
            zcr=zcr,
            spectral_centroid=spectral_centroid,
            spectral_bandwidth=spectral_bandwidth,
            mfcc_1=mfcc_1,
            mfcc_2=mfcc_2,
        )
