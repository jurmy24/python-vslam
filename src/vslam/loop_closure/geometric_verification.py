"""Geometric verification for loop closure candidates.

After place recognition finds visually similar keyframes, geometric
verification confirms the match by computing a relative pose using
feature matching and PnP + RANSAC.

This eliminates false positives from perceptual aliasing (different
places that look similar).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..frontend import SE3


@dataclass
class VerificationResult:
    """Result of geometric verification.

    Attributes:
        is_valid: Whether verification succeeded
        relative_pose: Relative transform from query to match keyframe
        num_inliers: Number of PnP inliers
        num_matches: Total number of feature matches
        inlier_ratio: Fraction of matches that are inliers
    """

    is_valid: bool
    relative_pose: SE3 | None = None
    num_inliers: int = 0
    num_matches: int = 0
    inlier_ratio: float = 0.0


class GeometricVerifier:
    """Verifies loop closure candidates using geometric constraints.

    Uses feature matching between keyframes followed by PnP + RANSAC
    to compute the relative pose and filter false positives.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        min_inliers: int = 50,
        min_inlier_ratio: float = 0.4,
        ransac_threshold: float = 2.0,
        max_reproj_error: float = 1.5,
    ) -> None:
        """Initialize geometric verifier.

        Args:
            camera_matrix: 3x3 camera intrinsics matrix
            min_inliers: Minimum inliers for valid loop (increased for robustness)
            min_inlier_ratio: Minimum ratio of inliers to matches
            ransac_threshold: RANSAC reprojection threshold in pixels
            max_reproj_error: Maximum mean reprojection error for valid loop
        """
        self._camera_matrix = camera_matrix.astype(np.float64)
        self._min_inliers = min_inliers
        self._min_inlier_ratio = min_inlier_ratio
        self._ransac_threshold = ransac_threshold
        self._max_reproj_error = max_reproj_error

        # Feature matcher for ORB (binary descriptors)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def verify(
        self,
        query_descriptors: np.ndarray,
        query_keypoints: np.ndarray,
        match_descriptors: np.ndarray,
        match_keypoints: np.ndarray,
        match_points_3d: np.ndarray,
        match_keypoint_to_3d: dict[int, int],
    ) -> VerificationResult:
        """Verify a loop closure candidate geometrically.

        Args:
            query_descriptors: Query keyframe descriptors (N, 32)
            query_keypoints: Query keyframe 2D points (N, 2)
            match_descriptors: Match keyframe descriptors (M, 32)
            match_keypoints: Match keyframe 2D points (M, 2)
            match_points_3d: 3D map points visible in match keyframe (P, 3)
            match_keypoint_to_3d: Mapping from keypoint index to 3D point index

        Returns:
            VerificationResult with validity and relative pose
        """
        # Step 1: Match features between keyframes
        matches = self._match_features(query_descriptors, match_descriptors)

        if len(matches) < self._min_inliers:
            return VerificationResult(is_valid=False, num_matches=len(matches))

        # Step 2: Build 3D-2D correspondences
        # For each match: if the matched keypoint in match_kf has a 3D point,
        # use that 3D point with the query keypoint 2D location
        points_3d = []
        points_2d = []
        valid_matches = []

        for m in matches:
            query_idx = m.queryIdx
            match_idx = m.trainIdx

            # Check if match keypoint has associated 3D point
            if match_idx in match_keypoint_to_3d:
                point_3d_idx = match_keypoint_to_3d[match_idx]
                points_3d.append(match_points_3d[point_3d_idx])
                points_2d.append(query_keypoints[query_idx])
                valid_matches.append(m)

        if len(points_3d) < self._min_inliers:
            return VerificationResult(
                is_valid=False,
                num_matches=len(matches),
                num_inliers=0,
            )

        points_3d = np.array(points_3d, dtype=np.float64)
        points_2d = np.array(points_2d, dtype=np.float64)

        # Step 3: Run PnP + RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints=points_3d,
            imagePoints=points_2d,
            cameraMatrix=self._camera_matrix,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=self._ransac_threshold,
            iterationsCount=100,
            confidence=0.99,
        )

        if not success or inliers is None:
            return VerificationResult(
                is_valid=False,
                num_matches=len(matches),
                num_inliers=0,
            )

        num_inliers = len(inliers)
        inlier_ratio = num_inliers / len(points_3d)

        # Step 4: Check inlier thresholds
        if num_inliers < self._min_inliers or inlier_ratio < self._min_inlier_ratio:
            return VerificationResult(
                is_valid=False,
                num_matches=len(matches),
                num_inliers=num_inliers,
                inlier_ratio=inlier_ratio,
            )

        # Step 5: Verify reprojection error on inliers
        inlier_points_3d = points_3d[inliers.flatten()]
        inlier_points_2d = points_2d[inliers.flatten()]

        projected, _ = cv2.projectPoints(
            inlier_points_3d,
            rvec,
            tvec,
            self._camera_matrix,
            None,
        )
        projected = projected.reshape(-1, 2)
        reproj_errors = np.linalg.norm(projected - inlier_points_2d, axis=1)
        mean_reproj_error = float(np.mean(reproj_errors))

        if mean_reproj_error > self._max_reproj_error:
            return VerificationResult(
                is_valid=False,
                num_matches=len(matches),
                num_inliers=num_inliers,
                inlier_ratio=inlier_ratio,
            )

        # Step 6: Convert to SE3
        # PnP returns T_camera_world (transforms world points to camera frame)
        # We want T_query_match (transform from match to query)
        R_camera_world, _ = cv2.Rodrigues(rvec)
        relative_pose = SE3(
            rotation=R_camera_world,
            translation=tvec.flatten(),
        )

        return VerificationResult(
            is_valid=True,
            relative_pose=relative_pose,
            num_inliers=num_inliers,
            num_matches=len(matches),
            inlier_ratio=inlier_ratio,
        )

    def _match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_thresh: float = 0.7,
    ) -> list[cv2.DMatch]:
        """Match features using ratio test.

        Args:
            desc1: First set of descriptors
            desc2: Second set of descriptors
            ratio_thresh: Lowe's ratio test threshold

        Returns:
            List of good matches
        """
        if len(desc1) < 2 or len(desc2) < 2:
            return []

        # K-NN matching with k=2 for ratio test
        matches = self._matcher.knnMatch(desc1, desc2, k=2)

        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_thresh * n.distance:
                    good_matches.append(m)

        return good_matches
