"""EuRoC IMU data reader.

Loads IMU measurements (gyroscope and accelerometer) from EuRoC dataset format.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class IMUMeasurement:
    """Single IMU measurement at a given timestamp.

    Attributes:
        timestamp_ns: Measurement timestamp in nanoseconds
        gyroscope: Angular velocity (wx, wy, wz) in rad/s
        accelerometer: Linear acceleration (ax, ay, az) in m/s²
    """

    timestamp_ns: int
    gyroscope: np.ndarray  # (3,) rad/s
    accelerometer: np.ndarray  # (3,) m/s²

    def __post_init__(self) -> None:
        """Ensure arrays have correct shape."""
        self.gyroscope = np.asarray(self.gyroscope, dtype=np.float64).flatten()
        self.accelerometer = np.asarray(self.accelerometer, dtype=np.float64).flatten()


@dataclass
class IMUCalibration:
    """IMU noise parameters from sensor calibration.

    Attributes:
        gyro_noise_density: Gyroscope white noise (rad/s/√Hz)
        gyro_random_walk: Gyroscope bias random walk (rad/s²/√Hz)
        accel_noise_density: Accelerometer white noise (m/s²/√Hz)
        accel_random_walk: Accelerometer bias random walk (m/s³/√Hz)
        T_body_imu: Transform from body frame to IMU frame (usually identity)
        rate_hz: IMU sampling rate in Hz
    """

    gyro_noise_density: float
    gyro_random_walk: float
    accel_noise_density: float
    accel_random_walk: float
    T_body_imu: np.ndarray  # 4x4 transform
    rate_hz: float = 200.0


class IMUReader:
    """Reader for EuRoC IMU data.

    Loads IMU measurements from imu0/data.csv and calibration from imu0/sensor.yaml.

    Example usage:
        reader = IMUReader("data/euroc/MH_01_easy/mav0")
        measurements = reader.get_measurements_between(t_start, t_end)
        for m in measurements:
            print(f"t={m.timestamp_ns}, gyro={m.gyroscope}, accel={m.accelerometer}")
    """

    def __init__(self, dataset_path: str | Path) -> None:
        """Initialize IMU reader.

        Args:
            dataset_path: Path to EuRoC mav0 directory
        """
        self._dataset_path = Path(dataset_path)
        self._imu_data_path = self._dataset_path / "imu0" / "data.csv"
        self._imu_sensor_path = self._dataset_path / "imu0" / "sensor.yaml"

        if not self._imu_data_path.exists():
            raise FileNotFoundError(
                f"IMU data not found: {self._imu_data_path}\n"
                f"Expected EuRoC format with imu0/data.csv"
            )

        # Load calibration
        self._calibration = self._load_calibration()

        # Load all measurements into memory (EuRoC has ~37k samples, ~3MB)
        self._measurements: list[IMUMeasurement] = []
        self._timestamps: list[int] = []  # For fast binary search
        self._load_measurements()

    def _load_calibration(self) -> IMUCalibration:
        """Load IMU calibration from sensor.yaml.

        Returns:
            IMUCalibration with noise parameters
        """
        if not self._imu_sensor_path.exists():
            # Return default values if file not found
            return IMUCalibration(
                gyro_noise_density=1.6968e-04,
                gyro_random_walk=1.9393e-05,
                accel_noise_density=2.0000e-3,
                accel_random_walk=3.0000e-3,
                T_body_imu=np.eye(4),
                rate_hz=200.0,
            )

        with open(self._imu_sensor_path, "r") as f:
            content = f.read()

        # Parse noise parameters
        def parse_float(pattern: str, default: float) -> float:
            match = re.search(pattern, content)
            return float(match.group(1)) if match else default

        gyro_noise = parse_float(r"gyroscope_noise_density:\s*([\d.e+-]+)", 1.6968e-04)
        gyro_walk = parse_float(r"gyroscope_random_walk:\s*([\d.e+-]+)", 1.9393e-05)
        accel_noise = parse_float(r"accelerometer_noise_density:\s*([\d.e+-]+)", 2.0000e-3)
        accel_walk = parse_float(r"accelerometer_random_walk:\s*([\d.e+-]+)", 3.0000e-3)
        rate_hz = parse_float(r"rate_hz:\s*([\d.]+)", 200.0)

        # Parse T_BS transform
        T_body_imu = np.eye(4)
        match = re.search(
            r"T_BS:\s*\n\s*cols:\s*4\s*\n\s*rows:\s*4\s*\n\s*data:\s*\[([^\]]+)\]",
            content,
        )
        if match:
            values = [float(x.strip()) for x in match.group(1).split(",")]
            if len(values) == 16:
                T_body_imu = np.array(values).reshape(4, 4)

        return IMUCalibration(
            gyro_noise_density=gyro_noise,
            gyro_random_walk=gyro_walk,
            accel_noise_density=accel_noise,
            accel_random_walk=accel_walk,
            T_body_imu=T_body_imu,
            rate_hz=rate_hz,
        )

    def _load_measurements(self) -> None:
        """Load all IMU measurements from CSV file."""
        with open(self._imu_data_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                if len(parts) < 7:
                    continue

                try:
                    timestamp_ns = int(parts[0])
                    wx, wy, wz = float(parts[1]), float(parts[2]), float(parts[3])
                    ax, ay, az = float(parts[4]), float(parts[5]), float(parts[6])

                    measurement = IMUMeasurement(
                        timestamp_ns=timestamp_ns,
                        gyroscope=np.array([wx, wy, wz]),
                        accelerometer=np.array([ax, ay, az]),
                    )
                    self._measurements.append(measurement)
                    self._timestamps.append(timestamp_ns)

                except (ValueError, IndexError):
                    continue

    def get_measurements_between(
        self, start_ns: int, end_ns: int
    ) -> list[IMUMeasurement]:
        """Get all IMU measurements between two timestamps (inclusive on start, exclusive on end).

        Args:
            start_ns: Start timestamp in nanoseconds (inclusive)
            end_ns: End timestamp in nanoseconds (exclusive)

        Returns:
            List of IMUMeasurement objects in the time range
        """
        if not self._timestamps:
            return []

        # Binary search for start and end indices
        start_idx = bisect.bisect_left(self._timestamps, start_ns)
        end_idx = bisect.bisect_left(self._timestamps, end_ns)

        return self._measurements[start_idx:end_idx]

    def get_measurement_at(self, timestamp_ns: int) -> IMUMeasurement | None:
        """Get the IMU measurement closest to the given timestamp.

        Args:
            timestamp_ns: Query timestamp in nanoseconds

        Returns:
            Closest IMUMeasurement, or None if no measurements exist
        """
        if not self._timestamps:
            return None

        idx = bisect.bisect_left(self._timestamps, timestamp_ns)

        # Check which neighbor is closer
        if idx == 0:
            return self._measurements[0]
        if idx >= len(self._timestamps):
            return self._measurements[-1]

        # Compare distances to neighbors
        if timestamp_ns - self._timestamps[idx - 1] < self._timestamps[idx] - timestamp_ns:
            return self._measurements[idx - 1]
        return self._measurements[idx]

    @property
    def calibration(self) -> IMUCalibration:
        """Return IMU calibration parameters."""
        return self._calibration

    @property
    def start_timestamp(self) -> int | None:
        """First IMU timestamp in nanoseconds."""
        return self._timestamps[0] if self._timestamps else None

    @property
    def end_timestamp(self) -> int | None:
        """Last IMU timestamp in nanoseconds."""
        return self._timestamps[-1] if self._timestamps else None

    @property
    def num_measurements(self) -> int:
        """Total number of IMU measurements."""
        return len(self._measurements)

    def __len__(self) -> int:
        """Number of IMU measurements."""
        return len(self._measurements)
