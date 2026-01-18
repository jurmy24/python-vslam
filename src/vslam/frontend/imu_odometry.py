"""IMU-based dead-reckoning odometry.

Replaces visual odometry with IMU integration for pose estimation.
Outputs poses at requested timestamps by integrating all IMU measurements.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import numpy as np

from .pose import SE3
from .imu_integrator import IMUIntegrator, IMUState
from ..io.imu_reader import IMUReader


class IMUTrackingStatus(Enum):
    """Status of IMU odometry."""

    OK = "OK"
    INITIALIZING = "INITIALIZING"
    DIVERGED = "DIVERGED"  # Pose values unreasonable (e.g., NaN, very large)


@dataclass
class IMUOdometryTiming:
    """Timing breakdown for IMU processing."""

    integration_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class IMUFrame:
    """Output of IMU odometry for a given timestamp.

    Provides a similar interface to VOFrame for compatibility.

    Attributes:
        frame_id: Sequential frame identifier
        timestamp_ns: Timestamp in nanoseconds
        pose: Estimated pose T_world_body (SE3)
        velocity: Estimated velocity in world frame
        tracking_status: Current tracking status
        num_integrated_measurements: Number of IMU samples integrated
        timing: Processing time breakdown
    """

    frame_id: int
    timestamp_ns: int
    pose: SE3
    velocity: np.ndarray
    tracking_status: IMUTrackingStatus
    num_integrated_measurements: int = 0
    timing: IMUOdometryTiming = field(default_factory=IMUOdometryTiming)

    @property
    def position(self) -> np.ndarray:
        """Return body position in world frame."""
        return self.pose.translation

    @property
    def is_tracking_ok(self) -> bool:
        """Return True if tracking is successful."""
        return self.tracking_status == IMUTrackingStatus.OK


class IMUOdometry:
    """Dead-reckoning odometry using IMU integration.

    Replaces VisualOdometry for IMU-only SLAM.
    Outputs poses at requested timestamps by integrating
    all IMU measurements up to that point.

    Note: IMU-only odometry will drift unboundedly over time.
    For long trajectories, expect position errors of 1-10% of distance traveled.
    """

    def __init__(
        self,
        imu_reader: IMUReader,
        initial_pose: SE3 | None = None,
        initial_velocity: np.ndarray | None = None,
        initial_bias_gyro: np.ndarray | None = None,
        initial_bias_accel: np.ndarray | None = None,
        gravity: np.ndarray | None = None,
    ) -> None:
        """Initialize IMU odometry.

        Args:
            imu_reader: Reader for IMU measurements
            initial_pose: Starting pose T_world_body (default: identity)
            initial_velocity: Starting velocity in world frame (default: zeros)
            initial_bias_gyro: Initial gyroscope bias (default: zeros)
            initial_bias_accel: Initial accelerometer bias (default: zeros)
            gravity: Gravity vector in world frame (default: [0,0,-9.81])
        """
        self._imu_reader = imu_reader
        self._integrator = IMUIntegrator(
            gravity=gravity,
            calibration=imu_reader.calibration,
        )

        # Initial state (from ground truth or defaults)
        self._initial_pose = initial_pose if initial_pose is not None else SE3.identity()
        self._initial_velocity = (
            np.zeros(3) if initial_velocity is None else np.asarray(initial_velocity)
        )
        self._initial_bias_gyro = (
            np.zeros(3) if initial_bias_gyro is None else np.asarray(initial_bias_gyro)
        )
        self._initial_bias_accel = (
            np.zeros(3) if initial_bias_accel is None else np.asarray(initial_bias_accel)
        )

        # State
        self._current_state: IMUState | None = None
        self._trajectory: list[SE3] = []
        self._frame_id: int = 0
        self._is_initialized: bool = False
        self._last_timestamp_ns: int | None = None

    @classmethod
    def from_dataset_path(
        cls,
        dataset_path: str | Path,
        use_gt_initial_state: bool = True,
    ) -> IMUOdometry:
        """Create IMUOdometry from EuRoC dataset path.

        Args:
            dataset_path: Path to mav0 directory
            use_gt_initial_state: If True, load initial pose, velocity, and biases from ground truth

        Returns:
            Initialized IMUOdometry
        """
        dataset_path = Path(dataset_path)
        imu_reader = IMUReader(dataset_path)

        initial_pose = None
        initial_velocity = None
        initial_bias_gyro = None
        initial_bias_accel = None

        if use_gt_initial_state:
            # Try to load initial state from ground truth
            gt_path = dataset_path / "state_groundtruth_estimate0" / "data.csv"
            if gt_path.exists():
                with open(gt_path, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        parts = line.split(",")
                        if len(parts) >= 17:
                            try:
                                # columns 1-3: position
                                position = np.array([
                                    float(parts[1]),
                                    float(parts[2]),
                                    float(parts[3]),
                                ])
                                # columns 4-7: quaternion (w, x, y, z)
                                qw = float(parts[4])
                                qx = float(parts[5])
                                qy = float(parts[6])
                                qz = float(parts[7])
                                initial_pose = SE3.from_quaternion(
                                    qw=qw, qx=qx, qy=qy, qz=qz, translation=position
                                )
                                # columns 8-10: velocity (world frame)
                                initial_velocity = np.array([
                                    float(parts[8]),
                                    float(parts[9]),
                                    float(parts[10]),
                                ])
                                # columns 11-13: gyro bias
                                initial_bias_gyro = np.array([
                                    float(parts[11]),
                                    float(parts[12]),
                                    float(parts[13]),
                                ])
                                # columns 14-16: accel bias
                                initial_bias_accel = np.array([
                                    float(parts[14]),
                                    float(parts[15]),
                                    float(parts[16]),
                                ])
                                break
                            except (ValueError, IndexError):
                                pass

        return cls(
            imu_reader=imu_reader,
            initial_pose=initial_pose,
            initial_velocity=initial_velocity,
            initial_bias_gyro=initial_bias_gyro,
            initial_bias_accel=initial_bias_accel,
        )

    def process_timestamp(self, timestamp_ns: int) -> IMUFrame:
        """Process all IMU data up to given timestamp.

        Integrates all IMU measurements from the last processed
        timestamp to the given timestamp.

        Args:
            timestamp_ns: Target timestamp in nanoseconds

        Returns:
            IMUFrame with estimated pose at timestamp
        """
        start_time = time.perf_counter()
        num_integrated = 0

        # Initialize on first call
        if not self._is_initialized:
            self._initialize(timestamp_ns)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            frame = IMUFrame(
                frame_id=self._frame_id,
                timestamp_ns=timestamp_ns,
                pose=self._current_state.pose,
                velocity=self._current_state.velocity.copy(),
                tracking_status=IMUTrackingStatus.OK,
                num_integrated_measurements=0,
                timing=IMUOdometryTiming(integration_ms=elapsed_ms, total_ms=elapsed_ms),
            )
            self._trajectory.append(frame.pose)
            self._frame_id += 1
            self._last_timestamp_ns = timestamp_ns
            return frame

        # Get IMU measurements between last timestamp and now
        integration_start = time.perf_counter()
        measurements = self._imu_reader.get_measurements_between(
            self._last_timestamp_ns, timestamp_ns + 1  # +1 to include endpoint
        )

        if measurements:
            self._current_state = self._integrator.integrate(
                measurements, self._current_state
            )
            num_integrated = len(measurements)

        integration_ms = (time.perf_counter() - integration_start) * 1000

        # Check for divergence (NaN or unreasonable values)
        status = self._check_tracking_status()

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        frame = IMUFrame(
            frame_id=self._frame_id,
            timestamp_ns=timestamp_ns,
            pose=self._current_state.pose,
            velocity=self._current_state.velocity.copy(),
            tracking_status=status,
            num_integrated_measurements=num_integrated,
            timing=IMUOdometryTiming(integration_ms=integration_ms, total_ms=elapsed_ms),
        )

        self._trajectory.append(frame.pose)
        self._frame_id += 1
        self._last_timestamp_ns = timestamp_ns

        return frame

    def _initialize(self, timestamp_ns: int) -> None:
        """Initialize IMU state at the first timestamp.

        Uses the initial rotation (for gravity direction) from ground truth but
        starts at origin position [0,0,0] to match visual odometry convention.
        Velocity is also rotated to the aligned frame.

        Args:
            timestamp_ns: First timestamp
        """
        # Start at origin but with correct orientation (important for gravity!)
        # This matches visual odometry which starts at identity pose
        initial_rotation = self._initial_pose.rotation.copy()

        self._current_state = IMUState(
            timestamp_ns=timestamp_ns,
            pose=SE3(
                rotation=initial_rotation,
                translation=np.zeros(3),  # Start at origin like VO
            ),
            velocity=self._initial_velocity.copy(),
            bias_gyro=self._initial_bias_gyro.copy(),
            bias_accel=self._initial_bias_accel.copy(),
        )
        self._is_initialized = True

    def _check_tracking_status(self) -> IMUTrackingStatus:
        """Check if the current state is reasonable.

        Returns:
            Tracking status
        """
        if self._current_state is None:
            return IMUTrackingStatus.INITIALIZING

        # Check for NaN
        if (
            np.any(np.isnan(self._current_state.pose.translation))
            or np.any(np.isnan(self._current_state.velocity))
        ):
            return IMUTrackingStatus.DIVERGED

        # Check for unreasonable position (more than 10km from origin)
        if np.linalg.norm(self._current_state.pose.translation) > 10000:
            return IMUTrackingStatus.DIVERGED

        # Check for unreasonable velocity (more than 100 m/s)
        if np.linalg.norm(self._current_state.velocity) > 100:
            return IMUTrackingStatus.DIVERGED

        return IMUTrackingStatus.OK

    def get_trajectory(self) -> list[SE3]:
        """Return all estimated poses."""
        return self._trajectory.copy()

    def get_trajectory_positions(self) -> np.ndarray:
        """Return positions as Nx3 array."""
        if not self._trajectory:
            return np.zeros((0, 3))
        return np.array([pose.translation for pose in self._trajectory])

    @property
    def current_pose(self) -> SE3 | None:
        """Get current pose estimate."""
        return self._current_state.pose if self._current_state else None

    @property
    def current_velocity(self) -> np.ndarray | None:
        """Get current velocity estimate."""
        return self._current_state.velocity.copy() if self._current_state else None

    def reset(self) -> None:
        """Reset to initial state."""
        self._current_state = None
        self._trajectory = []
        self._frame_id = 0
        self._is_initialized = False
        self._last_timestamp_ns = None
