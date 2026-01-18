"""Sparse stereo matching using ORB descriptors."""

from dataclasses import dataclass

import cv2
import numpy as np

from .feature_detector import Features


@dataclass
class StereoMatches:
    """Container for stereo feature matches.

    Attributes:
        pts_left: Nx2 array of matched points in left image
        pts_right: Nx2 array of matched points in right image
        disparities: N array of disparity values (u_left - u_right)
        match_distances: N array of descriptor distances (Hamming)
    """

    pts_left: np.ndarray
    pts_right: np.ndarray
    disparities: np.ndarray
    match_distances: np.ndarray

    def __len__(self) -> int:
        """Return number of matches."""
        return len(self.pts_left)

    def filter_by_disparity(
        self, min_disp: float = 1.0, max_disp: float = 200.0
    ) -> "StereoMatches":
        """Return new StereoMatches with disparity filtering applied.

        Args:
            min_disp: Minimum valid disparity (filters infinite depth)
            max_disp: Maximum valid disparity (filters too-close objects)

        Returns:
            New StereoMatches with only points in valid disparity range
        """
        mask = (self.disparities >= min_disp) & (self.disparities <= max_disp)
        return StereoMatches(
            pts_left=self.pts_left[mask],
            pts_right=self.pts_right[mask],
            disparities=self.disparities[mask],
            match_distances=self.match_distances[mask],
        )


class StereoMatcher:
    """Sparse stereo matcher using ORB binary descriptors.

    Matches features between left and right rectified images using
    brute-force Hamming distance matching with multiple filtering criteria.
    """

    def __init__(
        self,
        cross_check: bool = True,
        max_hamming_distance: int = 50,
        epipolar_threshold: float = 2.0,
        min_disparity: float = 1.0,
        max_disparity: float = 200.0,
    ) -> None:
        """Initialize stereo matcher.

        Args:
            cross_check: Enable cross-check for more robust matches.
                A match (i, j) is only accepted if feature j's best match is i.
            max_hamming_distance: Maximum Hamming distance for valid match.
                ORB descriptors are 256 bits, so max distance is 256.
                Lower values = stricter matching.
            epipolar_threshold: Maximum y-coordinate difference (pixels) for
                rectified images. After rectification, matches should be on
                the same horizontal line.
            min_disparity: Minimum valid disparity in pixels.
                Filters out matches at infinity (where u_left ≈ u_right).
            max_disparity: Maximum valid disparity in pixels.
                Filters out matches too close to the camera.
        """
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
        self._max_distance = max_hamming_distance
        self._epipolar_threshold = epipolar_threshold
        self._min_disparity = min_disparity
        self._max_disparity = max_disparity

    def match(
        self, features_left: Features, features_right: Features
    ) -> StereoMatches:
        """Match features between rectified stereo images.

        Applies three filtering stages:
        1. Hamming distance threshold (descriptor similarity)
        2. Epipolar constraint (y-coordinates must match in rectified images)
        3. Disparity range (depth must be reasonable)

        Args:
            features_left: Features from rectified left image
            features_right: Features from rectified right image

        Returns:
            StereoMatches containing filtered correspondences
        """
        # Handle empty feature sets
        if (
            len(features_left) == 0
            or len(features_right) == 0
            or features_left.descriptors is None
            or features_right.descriptors is None
        ):
            return StereoMatches(
                pts_left=np.empty((0, 2), dtype=np.float32),
                pts_right=np.empty((0, 2), dtype=np.float32),
                disparities=np.empty(0, dtype=np.float32),
                match_distances=np.empty(0, dtype=np.float32),
            )

        # Perform brute-force matching
        matches = self._bf_matcher.match(
            features_left.descriptors, features_right.descriptors
        )

        if len(matches) == 0:
            return StereoMatches(
                pts_left=np.empty((0, 2), dtype=np.float32),
                pts_right=np.empty((0, 2), dtype=np.float32),
                disparities=np.empty(0, dtype=np.float32),
                match_distances=np.empty(0, dtype=np.float32),
            )

        # Extract matched point coordinates
        pts_left = features_left.points
        pts_right = features_right.points

        # Apply filtering
        return self._filter_matches(matches, pts_left, pts_right)

    def _filter_matches(
        self,
        matches: list[cv2.DMatch],
        pts_left: np.ndarray,
        pts_right: np.ndarray,
    ) -> StereoMatches:
        """Apply filtering criteria to raw matches.

        Filters:
        1. Hamming distance ≤ max_hamming_distance
        2. |y_left - y_right| ≤ epipolar_threshold (epipolar constraint)
        3. min_disparity ≤ (x_left - x_right) ≤ max_disparity

        Args:
            matches: Raw matches from BFMatcher
            pts_left: All keypoint coordinates in left image
            pts_right: All keypoint coordinates in right image

        Returns:
            Filtered StereoMatches
        """
        filtered_left = []
        filtered_right = []
        filtered_disparities = []
        filtered_distances = []

        for match in matches:
            # Get matched point coordinates
            pt_left = pts_left[match.queryIdx]
            pt_right = pts_right[match.trainIdx]

            # Filter 1: Descriptor distance
            if match.distance > self._max_distance:
                continue

            # Filter 2: Epipolar constraint (y-coordinates should match)
            y_diff = abs(pt_left[1] - pt_right[1])
            if y_diff > self._epipolar_threshold:
                continue

            # Filter 3: Disparity range
            # In rectified images, disparity = x_left - x_right
            # Positive disparity means the point is in front of the camera
            disparity = pt_left[0] - pt_right[0]
            if disparity < self._min_disparity or disparity > self._max_disparity:
                continue

            # Match passed all filters
            filtered_left.append(pt_left)
            filtered_right.append(pt_right)
            filtered_disparities.append(disparity)
            filtered_distances.append(match.distance)

        # Convert to arrays
        if len(filtered_left) == 0:
            return StereoMatches(
                pts_left=np.empty((0, 2), dtype=np.float32),
                pts_right=np.empty((0, 2), dtype=np.float32),
                disparities=np.empty(0, dtype=np.float32),
                match_distances=np.empty(0, dtype=np.float32),
            )

        return StereoMatches(
            pts_left=np.array(filtered_left, dtype=np.float32),
            pts_right=np.array(filtered_right, dtype=np.float32),
            disparities=np.array(filtered_disparities, dtype=np.float32),
            match_distances=np.array(filtered_distances, dtype=np.float32),
        )
