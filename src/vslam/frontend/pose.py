"""SE(3) pose representation for rigid body transformations."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class SE3:
    """Rigid body transformation (rotation + translation) in SE(3).

    Represents a camera pose T_world_camera that transforms points from
    the camera frame to the world frame:

        p_world = R @ p_camera + t

    The SE(3) Lie group has 6 degrees of freedom: 3 for rotation (SO(3))
    and 3 for translation.

    Attributes:
        rotation: 3x3 orthonormal rotation matrix (det = +1)
        translation: 3D translation vector
    """

    rotation: np.ndarray  # 3x3 rotation matrix
    translation: np.ndarray  # (3,) translation vector

    def __post_init__(self) -> None:
        """Validate and normalize inputs."""
        self.rotation = np.asarray(self.rotation, dtype=np.float64)
        self.translation = np.asarray(self.translation, dtype=np.float64).flatten()

        if self.rotation.shape != (3, 3):
            raise ValueError(f"Rotation must be 3x3, got {self.rotation.shape}")
        if self.translation.shape != (3,):
            raise ValueError(
                f"Translation must be (3,), got {self.translation.shape}"
            )

    @classmethod
    def identity(cls) -> SE3:
        """Create identity transformation (no rotation, no translation).

        Returns:
            SE3 representing the identity transform
        """
        return cls(rotation=np.eye(3), translation=np.zeros(3))

    @classmethod
    def from_Rt(cls, R: np.ndarray, t: np.ndarray) -> SE3:
        """Create SE3 from rotation matrix and translation vector.

        Args:
            R: 3x3 rotation matrix
            t: 3D translation vector (any shape that flattens to 3)

        Returns:
            SE3 transformation
        """
        return cls(rotation=R, translation=t)

    @classmethod
    def from_matrix(cls, T: np.ndarray) -> SE3:
        """Create SE3 from 4x4 homogeneous transformation matrix.

        Args:
            T: 4x4 transformation matrix of the form:
               [[R  t]
                [0  1]]

        Returns:
            SE3 transformation
        """
        T = np.asarray(T)
        if T.shape != (4, 4):
            raise ValueError(f"Transform must be 4x4, got {T.shape}")

        R = T[:3, :3]
        t = T[:3, 3]
        return cls(rotation=R, translation=t)

    @classmethod
    def from_rvec_tvec(cls, rvec: np.ndarray, tvec: np.ndarray) -> SE3:
        """Create SE3 from OpenCV Rodrigues vector and translation.

        This is useful for converting output from cv2.solvePnP.

        Note: cv2.solvePnP returns T_camera_world. To get T_world_camera
        (which is what SE3 typically represents), you need to invert:

            pose_cam_world = SE3.from_rvec_tvec(rvec, tvec)
            pose_world_cam = pose_cam_world.inverse()

        Args:
            rvec: 3D Rodrigues rotation vector (axis * angle)
            tvec: 3D translation vector

        Returns:
            SE3 transformation
        """
        R, _ = cv2.Rodrigues(np.asarray(rvec).flatten())
        t = np.asarray(tvec).flatten()
        return cls(rotation=R, translation=t)

    @classmethod
    def from_quaternion(
        cls,
        qw: float,
        qx: float,
        qy: float,
        qz: float,
        translation: np.ndarray,
    ) -> SE3:
        """Create SE3 from quaternion and translation.

        Uses the Hamilton convention where quaternion is (w, x, y, z).
        This matches EuRoC ground truth format.

        Args:
            qw: Quaternion scalar (w) component
            qx: Quaternion x component
            qy: Quaternion y component
            qz: Quaternion z component
            translation: 3D translation vector

        Returns:
            SE3 transformation
        """
        # Normalize quaternion
        norm = np.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
        qw, qx, qy, qz = qw / norm, qx / norm, qy / norm, qz / norm

        # Quaternion to rotation matrix
        # https://www.euclideanspace.com/maths/geometry/rotations/conversions/quaternionToMatrix/
        R = np.array(
            [
                [
                    1 - 2 * qy * qy - 2 * qz * qz,
                    2 * qx * qy - 2 * qz * qw,
                    2 * qx * qz + 2 * qy * qw,
                ],
                [
                    2 * qx * qy + 2 * qz * qw,
                    1 - 2 * qx * qx - 2 * qz * qz,
                    2 * qy * qz - 2 * qx * qw,
                ],
                [
                    2 * qx * qz - 2 * qy * qw,
                    2 * qy * qz + 2 * qx * qw,
                    1 - 2 * qx * qx - 2 * qy * qy,
                ],
            ],
            dtype=np.float64,
        )

        return cls(rotation=R, translation=np.asarray(translation).flatten())

    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 homogeneous transformation matrix.

        Returns:
            4x4 transformation matrix [[R, t], [0, 1]]
        """
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.rotation
        T[:3, 3] = self.translation
        return T

    def to_rvec_tvec(self) -> tuple[np.ndarray, np.ndarray]:
        """Convert to OpenCV Rodrigues vector and translation.

        Returns:
            Tuple of (rvec, tvec) where rvec is 3D Rodrigues vector
        """
        rvec, _ = cv2.Rodrigues(self.rotation)
        return rvec.flatten(), self.translation.copy()

    def inverse(self) -> SE3:
        """Compute the inverse transformation T^{-1}.

        If T transforms from A to B, T.inverse() transforms from B to A.

        For T = [R, t], the inverse is [R^T, -R^T @ t].

        Returns:
            Inverse SE3 transformation
        """
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.translation
        return SE3(rotation=R_inv, translation=t_inv)

    def compose(self, other: SE3) -> SE3:
        """Compose with another transformation: self @ other.

        If self transforms A->B and other transforms B->C,
        then self.compose(other) transforms A->C... wait, that's wrong.

        Actually: If self = T_B_A (transforms points from A to B)
        and other = T_C_B (transforms points from B to C),
        then the result T_C_A = T_C_B @ T_B_A = other @ self.

        For "compose" we follow the convention self @ other:
        T_result = self.compose(other) means T_result = T_self @ T_other

        Example:
            T_world_prev.compose(T_prev_curr) gives T_world_curr

        Args:
            other: SE3 transformation to compose with

        Returns:
            Composed SE3 transformation (self @ other)
        """
        R = self.rotation @ other.rotation
        t = self.rotation @ other.translation + self.translation
        return SE3(rotation=R, translation=t)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """Transform points from local frame to world frame.

        Applies the transformation: p_world = R @ p_local + t

        Args:
            points: Nx3 array of 3D points in local (camera) frame

        Returns:
            Nx3 array of 3D points in world frame
        """
        points = np.asarray(points)
        if points.ndim == 1:
            points = points.reshape(1, 3)

        if points.shape[1] != 3:
            raise ValueError(f"Points must be Nx3, got {points.shape}")

        # p_world = R @ p_local + t
        return (self.rotation @ points.T).T + self.translation

    def transform_point(self, point: np.ndarray) -> np.ndarray:
        """Transform a single point from local frame to world frame.

        Args:
            point: 3D point in local (camera) frame

        Returns:
            3D point in world frame
        """
        point = np.asarray(point).flatten()
        return self.rotation @ point + self.translation

    @property
    def position(self) -> np.ndarray:
        """Return camera/body position in world frame.

        For T_world_camera, this is simply the translation component,
        which represents where the camera origin is in world coordinates.

        Returns:
            3D position vector in world frame
        """
        return self.translation.copy()

    @property
    def forward(self) -> np.ndarray:
        """Return forward direction (Z-axis) in world frame.

        In camera convention (X-right, Y-down, Z-forward), this returns
        where the camera is looking.

        Returns:
            Unit vector pointing in camera's forward direction
        """
        return self.rotation[:, 2].copy()

    @property
    def right(self) -> np.ndarray:
        """Return right direction (X-axis) in world frame.

        Returns:
            Unit vector pointing to camera's right
        """
        return self.rotation[:, 0].copy()

    @property
    def up(self) -> np.ndarray:
        """Return up direction (-Y-axis) in world frame.

        Note: In camera convention Y points down, so "up" is -Y.

        Returns:
            Unit vector pointing upward from camera's perspective
        """
        return -self.rotation[:, 1].copy()

    def __repr__(self) -> str:
        """Return string representation."""
        pos = self.position
        return f"SE3(position=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}])"

    def __matmul__(self, other: SE3) -> SE3:
        """Matrix multiplication operator for composition.

        Allows: T_result = T1 @ T2

        Args:
            other: SE3 transformation to compose with

        Returns:
            Composed SE3 transformation
        """
        return self.compose(other)
