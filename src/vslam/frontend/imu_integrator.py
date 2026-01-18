"""IMU integration for dead-reckoning odometry.

Integrates gyroscope and accelerometer measurements to estimate orientation,
velocity, and position using mid-point integration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pose import SE3
from ..io.imu_reader import IMUCalibration, IMUMeasurement


@dataclass
class IMUState:
    """Full IMU state at a given time.

    Attributes:
        timestamp_ns: State timestamp in nanoseconds
        pose: Rigid body transformation T_world_body (SE3)
        velocity: Linear velocity in world frame (3,)
        bias_gyro: Gyroscope bias (3,) in rad/s
        bias_accel: Accelerometer bias (3,) in m/s²
    """

    timestamp_ns: int
    pose: SE3
    velocity: np.ndarray  # (3,) world frame
    bias_gyro: np.ndarray  # (3,)
    bias_accel: np.ndarray  # (3,)

    def __post_init__(self) -> None:
        """Ensure arrays have correct shape and type."""
        self.velocity = np.asarray(self.velocity, dtype=np.float64).flatten()
        self.bias_gyro = np.asarray(self.bias_gyro, dtype=np.float64).flatten()
        self.bias_accel = np.asarray(self.bias_accel, dtype=np.float64).flatten()

    @property
    def position(self) -> np.ndarray:
        """Return position in world frame."""
        return self.pose.translation


def exp_so3(omega: np.ndarray) -> np.ndarray:
    """Exponential map from so(3) to SO(3) (Rodrigues formula).

    Converts an angular velocity vector (omega * dt) to a rotation matrix.

    Args:
        omega: Angular velocity vector (3,) representing axis-angle rotation

    Returns:
        3x3 rotation matrix
    """
    theta = np.linalg.norm(omega)
    if theta < 1e-10:
        # First-order approximation for small angles: R ≈ I + [omega]×
        return np.eye(3) + skew(omega)

    axis = omega / theta
    K = skew(axis)

    # Rodrigues formula: R = I + sin(θ)K + (1 - cos(θ))K²
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def skew(v: np.ndarray) -> np.ndarray:
    """Create skew-symmetric matrix from vector.

    Args:
        v: 3D vector

    Returns:
        3x3 skew-symmetric matrix [v]×
    """
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


class IMUIntegrator:
    """Integrates IMU measurements to estimate motion.

    Uses mid-point integration for better accuracy:
    - Rotation: Integrate gyroscope using exponential map
    - Velocity: Integrate accelerometer in world frame (with gravity compensation)
    - Position: Integrate velocity

    The integrator maintains bias estimates (typically constant or slowly varying)
    and removes them from raw measurements before integration.
    """

    def __init__(
        self,
        gravity: np.ndarray | None = None,
        calibration: IMUCalibration | None = None,
    ) -> None:
        """Initialize IMU integrator.

        Args:
            gravity: Gravity vector in world frame (default: [0, 0, -9.81])
            calibration: IMU noise calibration (for future uncertainty propagation)
        """
        self._gravity = (
            np.array([0, 0, -9.81], dtype=np.float64)
            if gravity is None
            else np.asarray(gravity, dtype=np.float64)
        )
        self._calibration = calibration

    def integrate(
        self,
        measurements: list[IMUMeasurement],
        initial_state: IMUState,
    ) -> IMUState:
        """Integrate a sequence of IMU measurements.

        Args:
            measurements: List of IMU measurements in chronological order
            initial_state: Starting state (pose, velocity, biases)

        Returns:
            Final state after integrating all measurements
        """
        state = initial_state

        for i, measurement in enumerate(measurements):
            # Compute dt (time since previous state)
            dt = (measurement.timestamp_ns - state.timestamp_ns) * 1e-9

            # Skip invalid time steps
            if dt <= 0:
                continue
            if dt > 0.1:  # Skip if gap > 100ms (data discontinuity)
                continue

            state = self.integrate_single(state, measurement, dt)

        return state

    def integrate_single(
        self,
        prev_state: IMUState,
        measurement: IMUMeasurement,
        dt: float,
    ) -> IMUState:
        """Integrate a single IMU measurement using mid-point integration.

        The mid-point method improves accuracy by using the average orientation
        during the time step to transform acceleration to world frame.

        Args:
            prev_state: State at previous timestep
            measurement: Current IMU measurement
            dt: Time step in seconds

        Returns:
            Updated state at measurement timestamp
        """
        # Remove biases from measurements
        omega = measurement.gyroscope - prev_state.bias_gyro  # rad/s
        accel = measurement.accelerometer - prev_state.bias_accel  # m/s²

        # 1. Rotation update using exponential map
        # R_new = R_prev @ exp(omega * dt)
        delta_angle = omega * dt
        R_delta = exp_so3(delta_angle)
        R_new = prev_state.pose.rotation @ R_delta

        # 2. Use mid-point rotation to transform acceleration
        # R_mid = R_prev @ exp(omega * dt / 2)
        R_mid = prev_state.pose.rotation @ exp_so3(delta_angle / 2)
        accel_world = R_mid @ accel

        # 3. Velocity update: v_new = v_prev + (a_world + g) * dt
        v_new = prev_state.velocity + (accel_world + self._gravity) * dt

        # 4. Position update: p_new = p_prev + v_prev * dt + 0.5 * a * dt²
        p_new = (
            prev_state.pose.translation
            + prev_state.velocity * dt
            + 0.5 * (accel_world + self._gravity) * dt**2
        )

        # Create new pose
        pose_new = SE3(rotation=R_new, translation=p_new)

        # Biases remain constant (could add random walk here for more realism)
        return IMUState(
            timestamp_ns=measurement.timestamp_ns,
            pose=pose_new,
            velocity=v_new,
            bias_gyro=prev_state.bias_gyro.copy(),
            bias_accel=prev_state.bias_accel.copy(),
        )

    @property
    def gravity(self) -> np.ndarray:
        """Return gravity vector in world frame."""
        return self._gravity.copy()
