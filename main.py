"""
Elephant Behavior Classification Prototype (PC + Raspberry Pi)

Quick README:
1) PC test mode (default):
   - `python main.py --once` runs one full cycle.
   - Uses webcam index 0 by default (or set --camera-source to a video path).
   - Uses --sample-audio if provided, else tries microphone recording.
   - GPS defaults to mock coordinates from config.
   - LoRa payload is printed if serial hardware is unavailable.

2) Raspberry Pi deployment mode:
   - Keep the SAME `main.py` and project structure.
   - Change runtime config/CLI values only (camera source, serial port, mode).
   - Real hardware is auto-used when libraries/devices are available.
   - If any hardware fails, system gracefully falls back without crashing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import get_default_config
from core.logger import setup_logger
from core.system_controller import SystemController
from utils.env_detect import detect_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Elephant behavior classification runtime")
    parser.add_argument("--once", action="store_true", help="Run one monitoring cycle only")
    parser.add_argument("--test-mode", action="store_true", default=True, help="Enable test mode")
    parser.add_argument("--camera-source", type=str, default="0", help="Camera index or video file path")
    parser.add_argument("--sample-audio", type=str, default=None, help="Path to .wav sample for sound prediction")
    parser.add_argument("--serial-port", type=str, default=None, help="LoRa serial port override")
    parser.add_argument("--pose-weight", type=float, default=None)
    parser.add_argument("--sound-weight", type=float, default=None)
    parser.add_argument("--threshold", type=float, default=None)
    return parser.parse_args()


def apply_overrides(args: argparse.Namespace):
    config = get_default_config()
    config.runtime.test_mode = args.test_mode
    config.runtime.camera_source = args.camera_source

    if args.sample_audio:
        config.paths.sample_audio_path = Path(args.sample_audio)

    if args.serial_port:
        config.runtime.serial_port = args.serial_port

    if args.pose_weight is not None:
        config.runtime.pose_weight = args.pose_weight
    if args.sound_weight is not None:
        config.runtime.sound_weight = args.sound_weight
    if args.threshold is not None:
        config.runtime.aggressive_threshold = args.threshold

    return config


def main() -> None:
    args = parse_args()
    config = apply_overrides(args)
    logger = setup_logger(config.logging)
    env = detect_environment()

    logger.info("Environment detected: is_raspberry_pi=%s machine=%s system=%s", env.is_raspberry_pi, env.machine, env.system)

    controller = SystemController(config=config, env_info=env, logger=logger)

    if args.once:
        output = controller.run_once()
        print(json.dumps(output.__dict__, indent=2))
        controller.camera_input.close()
    else:
        controller.monitor()


if __name__ == "__main__":
    main()
