"""
Motion artifact detector using MPU6050 accelerometer data.

The MPU6050 sends 3-axis accelerometer readings alongside ECG data
when wired to the ESP8266. Motion is detected by computing the standard
deviation of the total acceleration magnitude over a short window.

High STD of |a| → patient is moving → ECG data is likely contaminated.
"""

import collections
import numpy as np
import logging

log = logging.getLogger(__name__)


class MotionArtifactDetector:
    """
    Detects patient motion using MPU6050 accelerometer magnitude.

    Uses a rolling window STD threshold:
      std(|a_total|) > threshold → motion artifact detected

    At rest: |a| ≈ 1.0 g (gravity), std ≈ 0 (no motion)
    During motion: |a| fluctuates, std rises above threshold
    """

    def __init__(self, app_config: dict):
        self.threshold = float(app_config.get("ACCEL_STD_THRESHOLD", 0.15))
        self.window = int(app_config.get("MOTION_WINDOW", 25))
        self._buffer: collections.deque = collections.deque(maxlen=self.window)
        self._enabled = False   # True once at least one accelerometer sample received

    def add_sample(self, ax: float, ay: float, az: float) -> None:
        """
        Add one MPU6050 accelerometer reading (in g units).

        Args:
            ax, ay, az: Acceleration in each axis (g). From ESP8266 protocol:
                        values are sent as int×100, so Python divides by 100.0
        """
        magnitude = float(np.sqrt(ax ** 2 + ay ** 2 + az ** 2))
        self._buffer.append(magnitude)
        self._enabled = True

    def is_motion_artifact(self) -> bool:
        """
        Returns True if current accelerometer window shows motion.
        Returns False if MPU6050 is not connected (no data received).
        """
        if not self._enabled or len(self._buffer) < 3:
            return False
        return float(np.std(list(self._buffer))) > self.threshold

    def motion_level(self) -> float:
        """
        Normalized motion intensity [0.0, 1.0] for UI display.
        0.0 = completely still, 1.0 = severe motion.
        """
        if not self._enabled or len(self._buffer) < 3:
            return 0.0
        std = float(np.std(list(self._buffer)))
        # Saturate at 3× threshold → 1.0
        return min(1.0, std / (self.threshold * 3.0))

    def has_sensor(self) -> bool:
        """True if at least one accelerometer reading has been received."""
        return self._enabled

    def reset(self) -> None:
        """Clear buffer — call at session start."""
        self._buffer.clear()
        self._enabled = False
