from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Union

import cv2
import numpy as np


@dataclass
class CameraFrame:
    frame: Optional[np.ndarray]
    success: bool


class CameraInput:
    def __init__(self, source: Union[int, str], width: int, height: int, logger: logging.Logger) -> None:
        self.source = source
        self.width = width
        self.height = height
        self.logger = logger
        self.cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self.cap = cv2.VideoCapture(self.source)
        if self.cap is None or not self.cap.isOpened():
            self.logger.warning("Could not open camera/video source: %s", self.source)
            self.cap = None
            return False

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.logger.info("Camera source opened: %s", self.source)
        return True

    def read(self) -> CameraFrame:
        if self.cap is None:
            return CameraFrame(frame=None, success=False)
        ok, frame = self.cap.read()
        if not ok:
            self.logger.warning("Failed to read frame from source: %s", self.source)
            return CameraFrame(frame=None, success=False)
        return CameraFrame(frame=frame, success=True)

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
