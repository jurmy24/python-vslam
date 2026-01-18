"""Frontend components for SLAM.

Core components kept in this module:
- SE3: Rigid body transformation (used by both visual and IMU odometry)
- StereoCamera: Camera calibration (useful for camera-IMU extrinsics)
- IMU odometry: Dead-reckoning using IMU integration

Visual odometry components have been moved to the `vo` submodule
and are preserved for potential future Visual-Inertial (VIO) fusion.
"""

from .pose import SE3
from .stereo_camera import CameraIntrinsics, DistortionCoeffs, StereoCamera
from .imu_integrator import IMUIntegrator, IMUState
from .imu_odometry import IMUFrame, IMUOdometry, IMUOdometryTiming, IMUTrackingStatus

__all__ = [
    # Pose
    "SE3",
    # Camera
    "StereoCamera",
    "CameraIntrinsics",
    "DistortionCoeffs",
    # IMU Odometry
    "IMUOdometry",
    "IMUFrame",
    "IMUOdometryTiming",
    "IMUTrackingStatus",
    "IMUIntegrator",
    "IMUState",
]
