from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Dict


@dataclass
class LoRaSendResult:
    sent: bool
    mode: str
    message: str


class LoRaSender:
    def __init__(self, serial_port: str, baudrate: int, logger: logging.Logger) -> None:
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.logger = logger

    def send_payload(self, payload: Dict[str, object]) -> LoRaSendResult:
        encoded = json.dumps(payload, separators=(",", ":"))
        try:
            import serial  # type: ignore

            with serial.Serial(self.serial_port, self.baudrate, timeout=2) as ser:
                ser.write(encoded.encode("utf-8") + b"\n")
            self.logger.info("LoRa payload sent over serial: %s", encoded)
            return LoRaSendResult(sent=True, mode="serial", message=encoded)
        except Exception as exc:
            self.logger.info("LoRa serial unavailable, printing payload only: %s", encoded)
            return LoRaSendResult(sent=False, mode="print", message=f"{encoded} | reason={exc}")
