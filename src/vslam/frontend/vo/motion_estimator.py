"""Motion estimation using PnP with RANSAC."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..pose import SE3


@dataclass
class PnPResult:
    """Result of PnP pose estimation.

    Attributes:
        success: True if pose estimation succeeded
        pose: Estimated camera pose T_world_camera (transforms points
            from camera frame to world frame). None if failed.
        inliers: Boolean mask indicating which correspondences are inliers
        num_inliers: Number of inlier correspondences
        reprojection_error: Mean reprojection error of inliers (pixels)
    """

    success: bool
    pose: SE3 | None
    inliers: np.ndarray  # (N,) bool
    num_inliers: int
    reprojection_error: float


class MotionEstimator:
    """Estimates camera motion using PnP with RANSAC.

    Given 3D-2D correspondences (known 3D map points and their observed
    2D positions in the current frame), estimates the camera pose that
    minimizes reprojection error.

    The Perspective-n-Point (PnP) problem solves for the camera pose
    given at least 4 3D-2D correspondences. RANSAC provides robustness
    to outlier correspondences (incorrect matches).
    """

    def __init__(
        self,
        reprojection_threshold: float = 2.0,
        ransac_confidence: float = 0.99,
        max_iterations: int = 1000,
        min_inliers: int = 10,
        refine_with_all_inliers: bool = True,
    ) -> None:
        """Initialize motion estimator.

        Args:
            reprojection_threshold: RANSAC inlier threshold in pixels.
                Points with reprojection error below this are inliers.
            ransac_confidence: Desired probability of finding a good model.
                Higher values = more iterations. Range: 0-1.
            max_iterations: Maximum RANSAC iterations.
            min_inliers: Minimum number of inliers for a valid pose.
            refine_with_all_inliers: If True, refine pose using all inliers
                after RANSAC. Usually improves accuracy.
        """
        self._reprojection_threshold = reprojection_threshold
        self._ransac_confidence = ransac_confidence
        self._max_iterations = max_iterations
        self._min_inliers = min_inliers
        self._refine = refine_with_all_inliers

    def estimate_pose(
        self,
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        camera_matrix: np.ndarray,
        initial_pose: SE3 | None = None,
    ) -> PnPResult:
        """Estimate camera pose from 3D-2D correspondences.

        Uses cv2.solvePnPRansac for robust estimation, then optionally
        refines using all inliers with iterative PnP.

        Important: cv2.solvePnP returns T_camera_world (transforms world
        points to camera frame). We invert this to get T_world_camera
        (the camera pose in world coordinates).

        Args:
            points_3d: Nx3 array of 3D points in world frame
            points_2d: Nx2 array of corresponding 2D pixel coordinates
            camera_matrix: 3x3 camera intrinsic matrix K
            initial_pose: Optional initial pose estimate for refinement.
                If provided, uses iterative PnP starting from this pose.

        Returns:
            PnPResult with estimated pose and inlier information
        """
        n_points = len(points_3d)

        # Need at least 4 points for PnP
        if n_points < 4:
            return PnPResult(
                success=False,
                pose=None,
                inliers=np.zeros(n_points, dtype=bool),
                num_inliers=0,
                reprojection_error=float("inf"),
            )

        # Prepare arrays for OpenCV (needs float64 for best results)
        points_3d = np.asarray(points_3d, dtype=np.float64).reshape(-1, 1, 3)
        points_2d = np.asarray(points_2d, dtype=np.float64).reshape(-1, 1, 2)
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64)

        # No distortion (we work with rectified images)
        dist_coeffs = None

        # Initial guess from previous pose if available
        use_extrinsic_guess = False
        rvec_init = None
        tvec_init = None

        if initial_pose is not None:
            # Convert T_world_camera to T_camera_world for OpenCV
            pose_cam_world = initial_pose.inverse()
            rvec_init, tvec_init = pose_cam_world.to_rvec_tvec()
            rvec_init = rvec_init.reshape(3, 1)
            tvec_init = tvec_init.reshape(3, 1)
            use_extrinsic_guess = True

        # Run PnP RANSAC
        try:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                objectPoints=points_3d,
                imagePoints=points_2d,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                rvec=rvec_init,
                tvec=tvec_init,
                useExtrinsicGuess=use_extrinsic_guess,
                iterationsCount=self._max_iterations,
                reprojectionError=self._reprojection_threshold,
                confidence=self._ransac_confidence,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            return PnPResult(
                success=False,
                pose=None,
                inliers=np.zeros(n_points, dtype=bool),
                num_inliers=0,
                reprojection_error=float("inf"),
            )

        # Handle failure cases
        if not success or inliers is None or len(inliers) < self._min_inliers:
            return PnPResult(
                success=False,
                pose=None,
                inliers=np.zeros(n_points, dtype=bool),
                num_inliers=0 if inliers is None else len(inliers),
                reprojection_error=float("inf"),
            )

        # Convert inliers to boolean mask
        inlier_mask = np.zeros(n_points, dtype=bool)
        inlier_mask[inliers.flatten()] = True
        num_inliers = int(np.sum(inlier_mask))

        # Optionally refine with all inliers
        if self._refine and num_inliers >= 4:
            points_3d_inliers = points_3d[inlier_mask]
            points_2d_inliers = points_2d[inlier_mask]

            try:
                success_refine, rvec_refined, tvec_refined = cv2.solvePnP(
                    objectPoints=points_3d_inliers,
                    imagePoints=points_2d_inliers,
                    cameraMatrix=camera_matrix,
                    distCoeffs=dist_coeffs,
                    rvec=rvec,
                    tvec=tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                if success_refine:
                    rvec, tvec = rvec_refined, tvec_refined
            except cv2.error:
                pass  # Keep RANSAC result if refinement fails

        # Check for numerical issues
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            return PnPResult(
                success=False,
                pose=None,
                inliers=np.zeros(n_points, dtype=bool),
                num_inliers=0,
                reprojection_error=float("inf"),
            )

        # Convert T_camera_world to T_world_camera
        # solvePnP returns pose that transforms world points to camera frame
        # We want pose that represents camera position in world frame
        pose_cam_world = SE3.from_rvec_tvec(rvec, tvec)
        pose_world_cam = pose_cam_world.inverse()

        # Compute mean reprojection error
        reproj_error = self._compute_reprojection_error(
            points_3d[inlier_mask].reshape(-1, 3),
            points_2d[inlier_mask].reshape(-1, 2),
            rvec,
            tvec,
            camera_matrix,
        )

        return PnPResult(
            success=True,
            pose=pose_world_cam,
            inliers=inlier_mask,
            num_inliers=num_inliers,
            reprojection_error=reproj_error,
        )

    @staticmethod
    def _compute_reprojection_error(
        points_3d: np.ndarray,
        points_2d: np.ndarray,
        rvec: np.ndarray,
        tvec: np.ndarray,
        camera_matrix: np.ndarray,
    ) -> float:
        """Compute mean reprojection error.

        Args:
            points_3d: Nx3 3D points in world frame
            points_2d: Nx2 observed 2D points
            rvec: Rotation vector (T_camera_world)
            tvec: Translation vector (T_camera_world)
            camera_matrix: 3x3 intrinsic matrix

        Returns:
            Mean reprojection error in pixels
        """
        if len(points_3d) == 0:
            return 0.0

        # Project 3D points to 2D
        projected, _ = cv2.projectPoints(
            points_3d.reshape(-1, 1, 3),
            rvec,
            tvec,
            camera_matrix,
            None,
        )
        projected = projected.reshape(-1, 2)

        # Compute L2 distance
        errors = np.linalg.norm(projected - points_2d, axis=1)
        return float(np.mean(errors))

    @property
    def reprojection_threshold(self) -> float:
        """Return RANSAC inlier threshold."""
        return self._reprojection_threshold

    @property
    def min_inliers(self) -> int:
        """Return minimum required inliers."""
        return self._min_inliers
