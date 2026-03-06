from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.io import wavfile


@dataclass
class AudioChunk:
    audio: Optional[np.ndarray]
    sample_rate: int
    success: bool


class AudioInput:
    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def from_wav(self, path: Path) -> AudioChunk:
        try:
            if not path.exists():
                self.logger.warning("Audio file not found: %s", path)
                return AudioChunk(audio=None, sample_rate=0, success=False)
            sample_rate, audio = wavfile.read(path)
            audio = self._normalize(audio)
            self.logger.info("Loaded WAV audio file: %s", path)
            return AudioChunk(audio=audio, sample_rate=sample_rate, success=True)
        except Exception as exc:
            self.logger.exception("Failed to read WAV file: %s", exc)
            return AudioChunk(audio=None, sample_rate=0, success=False)

    def record(self, duration_s: float, sample_rate: int, channels: int = 1) -> AudioChunk:
        try:
            import sounddevice as sd  # type: ignore

            self.logger.info("Recording audio: duration=%ss sample_rate=%s", duration_s, sample_rate)
            rec = sd.rec(
                int(duration_s * sample_rate),
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
            )
            sd.wait()
            audio = np.squeeze(rec)
            return AudioChunk(audio=audio, sample_rate=sample_rate, success=True)
        except Exception as exc:
            self.logger.warning("Audio recording unavailable. Falling back: %s", exc)
            return AudioChunk(audio=None, sample_rate=0, success=False)

    @staticmethod
    def _normalize(audio: np.ndarray) -> np.ndarray:
        if np.issubdtype(audio.dtype, np.integer):
            maxv = np.iinfo(audio.dtype).max
            return audio.astype(np.float32) / max(maxv, 1)
        return audio.astype(np.float32)
