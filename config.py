from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PathsConfig:
    """Paths for model and sample assets."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent)
    model_dir: Path = field(init=False)
    sample_video_path: Optional[Path] = None
    sample_audio_path: Optional[Path] = None

    pose_rf_model_path: Path = field(init=False)
    pose_scaler_path: Path = field(init=False)
    pose_top_features_path: Path = field(init=False)

    sound_rf_model_path: Path = field(init=False)
    sound_xgb_model_path: Path = field(init=False)
    sound_cnn_lstm_model_path: Path = field(init=False)
    sound_scaler_path: Path = field(init=False)
    sound_hybrid_scaler_path: Path = field(init=False)
    label_encoder_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.model_dir = self.project_root / "models"

        self.pose_rf_model_path = self.model_dir / "pose_rf.pkl"
        self.pose_scaler_path = self.model_dir / "pose_scaler.pkl"
        self.pose_top_features_path = self.model_dir / "pose_top_features.json"

        self.sound_rf_model_path = self.model_dir / "sound_rf.pkl"
        self.sound_xgb_model_path = self.model_dir / "sound_xgb.pkl"
        self.sound_cnn_lstm_model_path = self.model_dir / "sound_cnn_lstm.h5"
        self.sound_scaler_path = self.model_dir / "sound_scaler.pkl"
        self.sound_hybrid_scaler_path = self.model_dir / "sound_hybrid_scaler.pkl"
        self.label_encoder_path = self.model_dir / "label_encoder.pkl"


@dataclass
class RuntimeConfig:
    test_mode: bool = True
    monitoring_interval_seconds: float = 2.0

    # Camera input: int for webcam index or file path.
    camera_source: str = "0"
    camera_width: int = 640
    camera_height: int = 480

    # Audio input options.
    audio_duration_seconds: float = 4.0
    audio_sample_rate: int = 16000
    audio_channels: int = 1

    # Hardware / communication.
    serial_port: str = "/dev/ttyUSB0"
    serial_baudrate: int = 9600

    # Fusion weights and threshold.
    pose_weight: float = 0.5
    sound_weight: float = 0.5
    aggressive_threshold: float = 0.6

    # Mock metadata.
    elephant_id: int = 1
    mock_latitude: float = 7.8731
    mock_longitude: float = 80.7718


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_to_file: bool = False
    log_file_path: Path = field(default_factory=lambda: Path("logs/runtime.log"))


@dataclass
class AppConfig:
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)


def get_default_config() -> AppConfig:
    """Build default configuration for both PC and Raspberry Pi usage."""
    return AppConfig()
