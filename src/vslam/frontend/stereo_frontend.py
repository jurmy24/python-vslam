"""Stereo visual frontend for feature extraction and 3D reconstruction."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .feature_detector import FeatureDetector, Features
from .stereo_camera import StereoCamera
from .stereo_matcher import StereoMatcher, StereoMatches


@dataclass
class StereoFrame:
    """Output of stereo frontend processing for a single frame.

    Contains all intermediate and final results from the stereo processing
    pipeline, enabling downstream tracking and visualization.

    Attributes:
        timestamp_ns: Frame timestamp in nanoseconds
        left_rectified: Rectified left camera image
        right_rectified: Rectified right camera image
        features_left: Detected features in left image
        features_right: Detected features in right image
        matches: Stereo matches between left and right images
        points_3d: Triangulated 3D points in left camera frame
    """

    timestamp_ns: int
    left_rectified: np.ndarray
    right_rectified: np.ndarray
    features_left: Features
    features_right: Features
    matches: StereoMatches
    points_3d: np.ndarray

    @property
    def num_features_left(self) -> int:
        """Return number of features detected in left image."""
        return len(self.features_left)

    @property
    def num_features_right(self) -> int:
        """Return number of features detected in right image."""
        return len(self.features_right)

    @property
    def num_matches(self) -> int:
        """Return number of stereo matches."""
        return len(self.matches)

    @property
    def num_3d_points(self) -> int:
        """Return number of triangulated 3D points."""
        return len(self.points_3d)


class StereoFrontend:
    """Stereo visual frontend for feature extraction and 3D reconstruction.

    Orchestrates the stereo processing pipeline:
    1. Rectify stereo images (remove distortion, align epipolar lines)
    2. Detect ORB features in both images
    3. Match features between images
    4. Triangulate matched points to 3D

    Example:
        >>> frontend = StereoFrontend.from_dataset_path("data/euroc/MH_01_easy/mav0")
        >>> frame = frontend.process_frame(left_img, right_img, timestamp_ns)
        >>> print(f"Reconstructed {frame.num_3d_points} 3D points")
    """

    def __init__(
        self,
        stereo_camera: StereoCamera,
        feature_detector: FeatureDetector | None = None,
        stereo_matcher: StereoMatcher | None = None,
    ) -> None:
        """Initialize stereo frontend with components.

        Args:
            stereo_camera: Calibrated stereo camera for rectification/triangulation
            feature_detector: ORB feature detector. Uses defaults if None.
            stereo_matcher: Stereo matcher. Uses defaults if None.
        """
        self._camera = stereo_camera
        self._detector = feature_detector or FeatureDetector()
        self._matcher = stereo_matcher or StereoMatcher()

    @classmethod
    def from_dataset_path(
        cls,
        dataset_path: str,
        n_features: int = 1000,
    ) -> "StereoFrontend":
        """Create StereoFrontend from EuRoC dataset path.

        Convenience factory that loads calibration from sensor.yaml files
        and configures the frontend with sensible defaults.

        Args:
            dataset_path: Path to mav0 directory containing cam0/ and cam1/
            n_features: Maximum number of ORB features to detect

        Returns:
            Configured StereoFrontend instance

        Raises:
            FileNotFoundError: If calibration files don't exist
        """
        path = Path(dataset_path)
        cam0_yaml = str(path / "cam0" / "sensor.yaml")
        cam1_yaml = str(path / "cam1" / "sensor.yaml")

        camera = StereoCamera(cam0_yaml, cam1_yaml)
        detector = FeatureDetector(n_features=n_features)
        matcher = StereoMatcher()

        return cls(
            stereo_camera=camera,
            feature_detector=detector,
            stereo_matcher=matcher,
        )

    def process_frame(
        self,
        left: np.ndarray,
        right: np.ndarray,
        timestamp_ns: int,
    ) -> StereoFrame:
        """Process a stereo image pair through the full pipeline.

        Pipeline stages:
        1. Rectify images (undistort + align epipolar lines)
        2. Detect ORB features in both rectified images
        3. Match features between left and right images
        4. Triangulate matched points to 3D coordinates

        Args:
            left: Left camera image (grayscale uint8)
            right: Right camera image (grayscale uint8)
            timestamp_ns: Frame timestamp in nanoseconds

        Returns:
            StereoFrame containing all processing results
        """
        # Stage 1: Rectify images
        left_rect, right_rect = self._camera.rectify_images(left, right)

        # Stage 2: Detect features in both images
        features_left = self._detector.detect(left_rect)
        features_right = self._detector.detect(right_rect)

        # Stage 3: Match features between images
        matches = self._matcher.match(features_left, features_right)

        # Stage 4: Triangulate 3D points
        points_3d = np.empty((0, 3), dtype=np.float64)
        if len(matches) > 0:
            points_3d = self._camera.triangulate_points(
                matches.pts_left, matches.pts_right
            )

        return StereoFrame(
            timestamp_ns=timestamp_ns,
            left_rectified=left_rect,
            right_rectified=right_rect,
            features_left=features_left,
            features_right=features_right,
            matches=matches,
            points_3d=points_3d,
        )

    @property
    def stereo_camera(self) -> StereoCamera:
        """Return the stereo camera used by this frontend."""
        return self._camera

    @property
    def baseline_meters(self) -> float:
        """Return stereo baseline in meters."""
        return self._camera.baseline_meters
