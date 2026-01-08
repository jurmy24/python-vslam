"""Stereo camera calibration, rectification, and triangulation."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters (pinhole model)."""

    fx: float  # Focal length x (pixels)
    fy: float  # Focal length y (pixels)
    cx: float  # Principal point x (pixels)
    cy: float  # Principal point y (pixels)

    def to_matrix(self) -> np.ndarray:
        """Return 3x3 camera intrinsic matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class DistortionCoeffs:
    """Radial-tangential distortion coefficients."""

    k1: float  # Radial distortion coefficient 1
    k2: float  # Radial distortion coefficient 2
    p1: float  # Tangential distortion coefficient 1
    p2: float  # Tangential distortion coefficient 2

    def to_array(self) -> np.ndarray:
        """Return distortion coefficients as (4,) array for OpenCV."""
        return np.array([self.k1, self.k2, self.p1, self.p2], dtype=np.float64)


class StereoCamera:
    """Stereo camera handling calibration, rectification, and triangulation.

    This class loads calibration data from EuRoC-format YAML files,
    computes rectification maps, and provides methods for image
    rectification and 3D triangulation.
    """

    def __init__(self, cam0_yaml_path: str, cam1_yaml_path: str) -> None:
        """Initialize stereo camera from EuRoC calibration files.

        Args:
            cam0_yaml_path: Path to left camera sensor.yaml
            cam1_yaml_path: Path to right camera sensor.yaml

        Raises:
            FileNotFoundError: If calibration files don't exist
            ValueError: If calibration data is invalid
        """
        # Load calibration for both cameras
        self._intrinsics_left, self._distortion_left, T_BS_left = (
            self._load_calibration(cam0_yaml_path)
        )
        self._intrinsics_right, self._distortion_right, T_BS_right = (
            self._load_calibration(cam1_yaml_path)
        )

        # Compute relative pose between cameras
        self._R, self._T = self._compute_relative_pose(T_BS_left, T_BS_right)
        self._baseline = float(np.linalg.norm(self._T))

        # Load image size from calibration
        with open(cam0_yaml_path, "r") as f:
            cam0_data = yaml.safe_load(f)
        self._image_size = tuple(cam0_data["resolution"])  # (width, height)

        # Compute rectification maps
        self._compute_rectification_maps()

    @staticmethod
    def _load_calibration(
        yaml_path: str,
    ) -> tuple[CameraIntrinsics, DistortionCoeffs, np.ndarray]:
        """Parse EuRoC sensor.yaml calibration file.

        Args:
            yaml_path: Path to sensor.yaml file

        Returns:
            Tuple of (intrinsics, distortion, T_BS) where T_BS is 4x4 transform

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file format is invalid
        """
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {yaml_path}")

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Parse intrinsics [fu, fv, cu, cv]
        intrinsics_list = data.get("intrinsics")
        if intrinsics_list is None or len(intrinsics_list) != 4:
            raise ValueError(f"Invalid intrinsics in {yaml_path}")

        intrinsics = CameraIntrinsics(
            fx=intrinsics_list[0],
            fy=intrinsics_list[1],
            cx=intrinsics_list[2],
            cy=intrinsics_list[3],
        )

        # Parse distortion coefficients [k1, k2, p1, p2]
        distortion_list = data.get("distortion_coefficients")
        if distortion_list is None or len(distortion_list) != 4:
            raise ValueError(f"Invalid distortion coefficients in {yaml_path}")

        distortion = DistortionCoeffs(
            k1=distortion_list[0],
            k2=distortion_list[1],
            p1=distortion_list[2],
            p2=distortion_list[3],
        )

        # Parse T_BS (camera-to-body transform)
        T_BS_data = data.get("T_BS", {}).get("data")
        if T_BS_data is None or len(T_BS_data) != 16:
            raise ValueError(f"Invalid T_BS transform in {yaml_path}")

        T_BS = np.array(T_BS_data, dtype=np.float64).reshape(4, 4)

        return intrinsics, distortion, T_BS

    @staticmethod
    def _compute_relative_pose(
        T_BS_cam0: np.ndarray, T_BS_cam1: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute rotation and translation between cameras.

        The relative pose T_cam1_cam0 transforms points from cam0 frame to cam1 frame:
            T_cam1_cam0 = inv(T_BS_cam1) @ T_BS_cam0

        Args:
            T_BS_cam0: 4x4 transform from cam0 to body frame
            T_BS_cam1: 4x4 transform from cam1 to body frame

        Returns:
            Tuple of (R, T) where R is 3x3 rotation, T is 3x1 translation
        """
        T_cam1_cam0 = np.linalg.inv(T_BS_cam1) @ T_BS_cam0

        R = T_cam1_cam0[:3, :3]
        T = T_cam1_cam0[:3, 3:4]  # Keep as column vector

        return R, T

    def _compute_rectification_maps(self) -> None:
        """Compute undistortion and rectification maps.

        Uses cv2.stereoRectify to compute rectification transforms that make
        epipolar lines horizontal, then cv2.initUndistortRectifyMap to create
        lookup tables for fast image warping.
        """
        K_left = self._intrinsics_left.to_matrix()
        K_right = self._intrinsics_right.to_matrix()
        D_left = self._distortion_left.to_array()
        D_right = self._distortion_right.to_array()

        # Compute rectification transforms
        # R1, R2: 3x3 rotation matrices for each camera
        # P1, P2: 3x4 projection matrices in rectified coords
        # Q: 4x4 disparity-to-depth mapping matrix
        (
            self._R1,
            self._R2,
            self._P1,
            self._P2,
            self._Q,
            _roi1,
            _roi2,
        ) = cv2.stereoRectify(
            cameraMatrix1=K_left,
            distCoeffs1=D_left,
            cameraMatrix2=K_right,
            distCoeffs2=D_right,
            imageSize=self._image_size,
            R=self._R,
            T=self._T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,  # Crop to valid pixels only
        )

        # Pre-compute undistortion + rectification maps for fast remapping
        self._map1_x, self._map1_y = cv2.initUndistortRectifyMap(
            cameraMatrix=K_left,
            distCoeffs=D_left,
            R=self._R1,
            newCameraMatrix=self._P1,
            size=self._image_size,
            m1type=cv2.CV_32FC1,
        )

        self._map2_x, self._map2_y = cv2.initUndistortRectifyMap(
            cameraMatrix=K_right,
            distCoeffs=D_right,
            R=self._R2,
            newCameraMatrix=self._P2,
            size=self._image_size,
            m1type=cv2.CV_32FC1,
        )

    def rectify_images(
        self, left: np.ndarray, right: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply undistortion and rectification to stereo image pair.

        After rectification, corresponding points lie on the same horizontal
        scanline (epipolar constraint), making stereo matching a 1D search.

        Args:
            left: Left camera image (grayscale or color)
            right: Right camera image (grayscale or color)

        Returns:
            Tuple of (rectified_left, rectified_right)
        """
        left_rectified = cv2.remap(
            left,
            self._map1_x,
            self._map1_y,
            interpolation=cv2.INTER_LINEAR,
        )
        right_rectified = cv2.remap(
            right,
            self._map2_x,
            self._map2_y,
            interpolation=cv2.INTER_LINEAR,
        )

        return left_rectified, right_rectified

    def triangulate_points(
        self, pts_left: np.ndarray, pts_right: np.ndarray
    ) -> np.ndarray:
        """Triangulate 3D points from matched 2D correspondences.

        Uses the projection matrices from stereo rectification to compute
        3D coordinates via linear triangulation.

        Args:
            pts_left: Nx2 array of 2D points in rectified left image
            pts_right: Nx2 array of 2D points in rectified right image

        Returns:
            Nx3 array of 3D points in left camera frame
        """
        if len(pts_left) == 0:
            return np.empty((0, 3), dtype=np.float64)

        # OpenCV expects 2xN arrays
        pts_left_T = pts_left.T.astype(np.float64)
        pts_right_T = pts_right.T.astype(np.float64)

        # Triangulate to homogeneous coordinates (4xN)
        points_4d = cv2.triangulatePoints(
            projMatr1=self._P1,
            projMatr2=self._P2,
            projPoints1=pts_left_T,
            projPoints2=pts_right_T,
        )

        # Convert from homogeneous to 3D (divide by w)
        points_3d = points_4d[:3, :] / points_4d[3:4, :]

        return points_3d.T  # Return as Nx3

    @property
    def baseline_meters(self) -> float:
        """Return baseline distance between cameras in meters."""
        return self._baseline

    @property
    def image_size(self) -> tuple[int, int]:
        """Return image size as (width, height)."""
        return self._image_size

    @property
    def focal_length(self) -> float:
        """Return focal length (fx) of rectified left camera."""
        # P1[0,0] is fx in the rectified projection matrix
        return float(self._P1[0, 0])

    @property
    def principal_point(self) -> tuple[float, float]:
        """Return principal point (cx, cy) of rectified left camera."""
        return (float(self._P1[0, 2]), float(self._P1[1, 2]))
