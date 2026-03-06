from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GPSData:
    latitude: float
    longitude: float
    is_mock: bool


class GPSProvider:
    def __init__(self, mock_lat: float, mock_lon: float, logger: logging.Logger) -> None:
        self.mock_lat = mock_lat
        self.mock_lon = mock_lon
        self.logger = logger

    def get_location(self) -> GPSData:
        gps = self._try_real_gps()
        if gps is not None:
            return gps
        self.logger.info("Using mock GPS coordinates.")
        return GPSData(latitude=self.mock_lat, longitude=self.mock_lon, is_mock=True)

    def _try_real_gps(self) -> Optional[GPSData]:
        try:
            import gpsd  # type: ignore

            gpsd.connect()
            pkt = gpsd.get_current()
            if pkt.mode >= 2:
                self.logger.info("Using real GPS data.")
                return GPSData(latitude=float(pkt.lat), longitude=float(pkt.lon), is_mock=False)
        except Exception:
            return None
        return None
