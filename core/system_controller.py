from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

from config import AppConfig
from core.fusion import DecisionFusion
from io_modules.audio_input import AudioInput
from io_modules.camera_input import CameraInput
from io_modules.gps_provider import GPSProvider
from io_modules.lora_sender import LoRaSender
from pose.pose_model_loader import PoseModelLoader
from pose.pose_runtime_predictor import PoseRuntimePredictor
from sound.sound_model_loader import SoundModelLoader
from sound.sound_runtime_predictor import SoundRuntimePredictor
from utils.env_detect import EnvironmentInfo


@dataclass
class SystemOutput:
    pose_label: str
    pose_prob_aggressive: float
    sound_label: str
    sound_score_aggressive: float
    final_label: str
    final_score: float
    elephant_id: int
    latitude: float
    longitude: float
    lora_payload: Dict[str, object]


class SystemController:
    def __init__(self, config: AppConfig, env_info: EnvironmentInfo, logger: logging.Logger) -> None:
        self.config = config
        self.env_info = env_info
        self.logger = logger

        pose_bundle = PoseModelLoader(config.paths, logger).load()
        sound_bundle = SoundModelLoader(config.paths, logger).load()

        self.pose_predictor = PoseRuntimePredictor(pose_bundle, logger)
        self.sound_predictor = SoundRuntimePredictor(sound_bundle, logger)
        self.audio_input = AudioInput(logger)

        source = self._resolve_camera_source(config.runtime.camera_source)
        self.camera_input = CameraInput(
            source=source,
            width=config.runtime.camera_width,
            height=config.runtime.camera_height,
            logger=logger,
        )

        self.gps_provider = GPSProvider(config.runtime.mock_latitude, config.runtime.mock_longitude, logger)
        self.lora_sender = LoRaSender(config.runtime.serial_port, config.runtime.serial_baudrate, logger)
        self.fusion = DecisionFusion(
            pose_weight=config.runtime.pose_weight,
            sound_weight=config.runtime.sound_weight,
            threshold=config.runtime.aggressive_threshold,
        )

    def _resolve_camera_source(self, source: str):
        return int(source) if source.isdigit() else source

    def run_once(self) -> SystemOutput:
        pose_label, pose_prob = "Unknown", 0.0
        sound_label, sound_score = "Unknown", 0.0

        if self.camera_input.cap is None:
            self.camera_input.open()

        frame_data = self.camera_input.read()
        if frame_data.success and frame_data.frame is not None:
            pose_pred = self.pose_predictor.predict_from_frame(frame_data.frame)
            if pose_pred.success:
                pose_label, pose_prob = pose_pred.label, pose_pred.prob_aggressive

        audio_chunk = self._get_audio_chunk()
        if audio_chunk is not None:
            sound_pred = self.sound_predictor.predict(audio_chunk[0], audio_chunk[1])
            if sound_pred.success:
                sound_label, sound_score = sound_pred.label, sound_pred.score_aggressive

        if pose_label == "Unknown" and sound_label == "Unknown":
            final_label, final_score = "Unknown", 0.0
        else:
            fusion_result = self.fusion.fuse(pose_prob, sound_score)
            final_label, final_score = fusion_result.final_label, fusion_result.final_score

        gps = self.gps_provider.get_location()
        payload = {
            "elephant_id": self.config.runtime.elephant_id,
            "latitude": gps.latitude,
            "longitude": gps.longitude,
            "behavior": final_label,
            "confidence": round(final_score, 3),
        }

        if final_label == "Aggressive":
            self.lora_sender.send_payload(payload)
        else:
            self.logger.info("No LoRa alert sent. Final behavior is %s", final_label)

        output = SystemOutput(
            pose_label=pose_label,
            pose_prob_aggressive=round(pose_prob, 3),
            sound_label=sound_label,
            sound_score_aggressive=round(sound_score, 3),
            final_label=final_label,
            final_score=round(final_score, 3),
            elephant_id=self.config.runtime.elephant_id,
            latitude=gps.latitude,
            longitude=gps.longitude,
            lora_payload=payload,
        )

        self.logger.info("System output: %s", output)
        return output

    def _get_audio_chunk(self) -> Optional[tuple]:
        sample_audio = self.config.paths.sample_audio_path
        if sample_audio is not None:
            audio_data = self.audio_input.from_wav(sample_audio)
            if audio_data.success and audio_data.audio is not None:
                return audio_data.audio, audio_data.sample_rate

        rec = self.audio_input.record(
            duration_s=self.config.runtime.audio_duration_seconds,
            sample_rate=self.config.runtime.audio_sample_rate,
            channels=self.config.runtime.audio_channels,
        )
        if rec.success and rec.audio is not None:
            return rec.audio, rec.sample_rate

        self.logger.warning("Audio modality unavailable for this cycle.")
        return None

    def monitor(self) -> None:
        try:
            while True:
                self.run_once()
                time.sleep(self.config.runtime.monitoring_interval_seconds)
        finally:
            self.camera_input.close()
