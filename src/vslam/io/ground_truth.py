"""EuRoC ground truth reader with timestamp interpolation.

Handles the coordinate frame transformation between body frame (ground truth)
and camera frame (SLAM output).
"""

from __future__ import annotations

import bisect
import re
from pathlib import Path

import numpy as np

from ..frontend import SE3


class GroundTruthReader:
    """Load and interpolate EuRoC ground truth poses in camera frame.

    EuRoC ground truth is provided at ~200Hz in state_groundtruth_estimate0/data.csv.
    Camera images are at ~20Hz, so we need to interpolate.

    IMPORTANT: Ground truth provides body/IMU frame poses, but SLAM estimates
    camera frame poses. This class transforms GT to camera frame using the
    body-to-camera extrinsics from cam0/sensor.yaml.

    The poses are also aligned to start at identity (like SLAM) by transforming
    all poses relative to the first camera pose.

    CSV format:
        #timestamp, p_RS_R_x, p_RS_R_y, p_RS_R_z, q_RS_w, q_RS_x, q_RS_y, q_RS_z, ...
    """

    def __init__(self, dataset_path: str | Path) -> None:
        """Initialize ground truth reader.

        Args:
            dataset_path: Path to EuRoC mav0 directory
        """
        self._dataset_path = Path(dataset_path)
        self._gt_path = self._dataset_path / "state_groundtruth_estimate0" / "data.csv"

        if not self._gt_path.exists():
            raise FileNotFoundError(
                f"Ground truth not found: {self._gt_path}\n"
                f"Expected EuRoC format with state_groundtruth_estimate0/data.csv"
            )

        # Load body-to-camera transform from sensor.yaml
        self._T_body_camera = self._load_body_to_camera_transform()

        # Load ground truth data (will be transformed to camera frame)
        self._timestamps: list[int] = []  # nanoseconds
        self._poses: list[SE3] = []  # Camera frame poses

        self._load_ground_truth()

        # Alignment transform: T_aligned = T_first_inv @ T_original
        # Lazily initialized on first get_pose_at() call to align with
        # the first CAMERA timestamp (not the first GT timestamp)
        self._first_pose_inv: SE3 | None = None
        self._alignment_initialized: bool = False

    def _load_body_to_camera_transform(self) -> SE3:
        """Load T_BS (body to sensor/camera) from cam0/sensor.yaml.

        Returns:
            SE3 transform from body frame to camera frame
        """
        sensor_yaml_path = self._dataset_path / "cam0" / "sensor.yaml"

        if not sensor_yaml_path.exists():
            # Fall back to identity if sensor.yaml not found
            print(f"Warning: {sensor_yaml_path} not found, using identity transform")
            return SE3.identity()

        # Parse the YAML file manually (avoid adding pyyaml dependency)
        with open(sensor_yaml_path, "r") as f:
            content = f.read()

        # Find T_BS data section
        match = re.search(r"T_BS:\s*\n\s*cols:\s*4\s*\n\s*rows:\s*4\s*\n\s*data:\s*\[([^\]]+)\]", content)
        if not match:
            print("Warning: Could not parse T_BS from sensor.yaml, using identity transform")
            return SE3.identity()

        # Parse the 16 values
        data_str = match.group(1)
        values = [float(x.strip()) for x in data_str.split(",")]

        if len(values) != 16:
            print(f"Warning: Expected 16 values for T_BS, got {len(values)}, using identity")
            return SE3.identity()

        # Reshape to 4x4 matrix (row-major order)
        T_BS = np.array(values).reshape(4, 4)

        return SE3.from_matrix(T_BS)

    def _load_ground_truth(self) -> None:
        """Load ground truth poses from CSV file."""
        with open(self._gt_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split(",")
                if len(parts) < 8:
                    continue

                try:
                    timestamp_ns = int(parts[0])
                    px, py, pz = float(parts[1]), float(parts[2]), float(parts[3])
                    qw, qx, qy, qz = (
                        float(parts[4]),
                        float(parts[5]),
                        float(parts[6]),
                        float(parts[7]),
                    )

                    # Create body frame pose from ground truth
                    pose_body = SE3.from_quaternion(
                        qw=qw,
                        qx=qx,
                        qy=qy,
                        qz=qz,
                        translation=np.array([px, py, pz]),
                    )

                    # Transform from body frame to camera frame
                    # T_world_camera = T_world_body @ T_body_camera
                    pose_camera = pose_body @ self._T_body_camera

                    self._timestamps.append(timestamp_ns)
                    self._poses.append(pose_camera)

                except (ValueError, IndexError):
                    continue

    def _get_raw_pose_at(self, timestamp_ns: int) -> SE3 | None:
        """Get interpolated pose WITHOUT alignment (internal use).

        Args:
            timestamp_ns: Query timestamp in nanoseconds

        Returns:
            Raw SE3 pose in camera frame, or None if outside GT range
        """
        if not self._timestamps:
            return None

        # Check bounds
        if timestamp_ns < self._timestamps[0] or timestamp_ns > self._timestamps[-1]:
            return None

        # Binary search for surrounding timestamps
        idx = bisect.bisect_left(self._timestamps, timestamp_ns)

        # Exact match
        if idx < len(self._timestamps) and self._timestamps[idx] == timestamp_ns:
            return self._poses[idx]

        # Interpolate between idx-1 and idx
        if idx == 0:
            return self._poses[0]
        if idx >= len(self._timestamps):
            return self._poses[-1]

        # Linear interpolation factor
        t0 = self._timestamps[idx - 1]
        t1 = self._timestamps[idx]
        alpha = (timestamp_ns - t0) / (t1 - t0)

        # Interpolate position (linear)
        pos0 = self._poses[idx - 1].translation
        pos1 = self._poses[idx].translation
        pos_interp = (1 - alpha) * pos0 + alpha * pos1

        # For rotation, use the closer pose (SLERP would be more accurate but overkill)
        # Since GT is at 200Hz and camera at 20Hz, the difference is tiny
        rot = self._poses[idx - 1].rotation if alpha < 0.5 else self._poses[idx].rotation

        return SE3(rotation=rot, translation=pos_interp)

    def get_pose_at(self, timestamp_ns: int) -> SE3 | None:
        """Get interpolated ground truth pose at given timestamp.

        The returned pose is aligned to start at identity. The alignment reference
        is set lazily on the first call, so the first queried timestamp becomes
        the origin (matching SLAM's first-frame-at-identity convention).

        Args:
            timestamp_ns: Query timestamp in nanoseconds

        Returns:
            Aligned SE3 pose, or None if timestamp is outside GT range
        """
        # Get the raw (unaligned) pose
        raw_pose = self._get_raw_pose_at(timestamp_ns)
        if raw_pose is None:
            return None

        # Lazy initialization: set alignment reference on first successful query
        # This ensures we align to the first CAMERA timestamp, not the first GT timestamp
        if not self._alignment_initialized:
            self._first_pose_inv = raw_pose.inverse()
            self._alignment_initialized = True

        return self._align_pose(raw_pose)

    def _align_pose(self, pose: SE3) -> SE3:
        """Align pose to start at identity (first GT pose = origin).

        Args:
            pose: Original GT pose in world frame

        Returns:
            Aligned pose where first GT pose is identity
        """
        if self._first_pose_inv is None:
            return pose
        return self._first_pose_inv @ pose

    def get_position_at(self, timestamp_ns: int) -> np.ndarray | None:
        """Get interpolated ground truth position at given timestamp.

        Args:
            timestamp_ns: Query timestamp in nanoseconds

        Returns:
            3D position array, or None if timestamp is outside GT range
        """
        pose = self.get_pose_at(timestamp_ns)
        if pose is None:
            return None
        return pose.position

    @property
    def start_timestamp(self) -> int | None:
        """First ground truth timestamp in nanoseconds."""
        return self._timestamps[0] if self._timestamps else None

    @property
    def end_timestamp(self) -> int | None:
        """Last ground truth timestamp in nanoseconds."""
        return self._timestamps[-1] if self._timestamps else None

    def __len__(self) -> int:
        """Number of ground truth poses."""
        return len(self._poses)
