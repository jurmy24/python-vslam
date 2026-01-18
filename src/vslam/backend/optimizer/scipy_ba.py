"""Bundle adjustment using scipy.optimize.least_squares.

Bundle adjustment jointly optimizes camera poses and 3D point positions
by minimizing the sum of squared reprojection errors.

The optimization problem:
    minimize sum_i ||observed_i - project(pose_j, point_k)||^2

Where:
- observed_i is a 2D pixel observation
- pose_j is the camera pose for the keyframe
- point_k is the 3D position of the map point
- project() projects the 3D point to 2D using the camera intrinsics
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

if TYPE_CHECKING:
    from ..keyframe import Keyframe
    from ...frontend.vo.map_point import MapPoint


@dataclass
class BAObservation:
    """A single 2D observation for bundle adjustment."""

    keyframe_idx: int  # Index in the keyframes list (not keyframe.id)
    point_idx: int  # Index in the points list (not mappoint.id)
    pixel: np.ndarray  # (2,) observed pixel coordinates


@dataclass
class BAResult:
    """Result of bundle adjustment optimization."""

    success: bool
    # Optimized poses: keyframe.id -> SE3
    optimized_poses: dict[int, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    # Optimized points: mappoint.id -> position
    optimized_points: dict[int, np.ndarray] = field(default_factory=dict)
    initial_cost: float = 0.0
    final_cost: float = 0.0
    iterations: int = 0
    message: str = ""


def _project_point(
    point_world: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    camera_matrix: np.ndarray,
) -> np.ndarray:
    """Project a 3D world point to 2D pixel coordinates.

    Args:
        point_world: 3D point in world frame
        rotation: 3x3 rotation matrix (world to camera)
        translation: 3D translation vector (world to camera)
        camera_matrix: 3x3 camera intrinsics matrix

    Returns:
        2D pixel coordinates
    """
    # Transform to camera frame: p_cam = R @ (p_world - t_world_cam)
    # But we store T_world_camera, so: p_cam = R.T @ (p_world - translation)
    # Actually for BA we parameterize as T_camera_world for easier optimization
    # So: p_cam = R @ p_world + t
    p_cam = rotation @ point_world + translation

    if p_cam[2] <= 1e-6:
        # Point behind camera - return large coordinates
        return np.array([1e6, 1e6])

    # Project using camera matrix
    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    u = fx * (p_cam[0] / p_cam[2]) + cx
    v = fy * (p_cam[1] / p_cam[2]) + cy

    return np.array([u, v])


def _rvec_to_rotation(rvec: np.ndarray) -> np.ndarray:
    """Convert Rodrigues vector to rotation matrix."""
    R, _ = cv2.Rodrigues(rvec)
    return R


def _rotation_to_rvec(R: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to Rodrigues vector."""
    rvec, _ = cv2.Rodrigues(R)
    return rvec.flatten()


class ScipyBundleAdjustment:
    """Bundle adjustment using scipy's Levenberg-Marquardt optimizer.

    Optimizes camera poses and 3D point positions to minimize
    reprojection error. Uses sparse Jacobian structure for efficiency.
    """

    def __init__(
        self,
        max_iterations: int = 50,
        ftol: float = 1e-6,
        xtol: float = 1e-6,
        loss: str = "linear",  # or "huber", "soft_l1" for robustness
    ) -> None:
        """Initialize bundle adjustment optimizer.

        Args:
            max_iterations: Maximum LM iterations
            ftol: Function tolerance for convergence
            xtol: Parameter tolerance for convergence
            loss: Loss function ("linear", "huber", "soft_l1", "cauchy")
        """
        self._max_iterations = max_iterations
        self._ftol = ftol
        self._xtol = xtol
        self._loss = loss

    def optimize(
        self,
        keyframes: list[Keyframe],
        map_points: list[MapPoint],
        camera_matrix: np.ndarray,
        fix_first_pose: bool = True,
    ) -> BAResult:
        """Run bundle adjustment optimization.

        Args:
            keyframes: List of keyframes to optimize
            map_points: List of map points to optimize
            camera_matrix: 3x3 camera intrinsics matrix
            fix_first_pose: If True, fix the first keyframe pose (gauge freedom)

        Returns:
            BAResult with optimized poses and points
        """
        if len(keyframes) == 0 or len(map_points) == 0:
            return BAResult(success=False, message="No keyframes or map points")

        # Build index mappings
        kf_id_to_idx = {kf.id: idx for idx, kf in enumerate(keyframes)}
        mp_id_to_idx = {mp.id: idx for idx, mp in enumerate(map_points)}

        # Collect observations
        observations = self._collect_observations(
            keyframes, map_points, kf_id_to_idx, mp_id_to_idx
        )

        if len(observations) < 10:
            return BAResult(
                success=False, message=f"Too few observations: {len(observations)}"
            )

        # Pack initial parameters
        # Format: [rvec_0, tvec_0, rvec_1, tvec_1, ..., point_0, point_1, ...]
        n_poses = len(keyframes)
        n_points = len(map_points)
        n_pose_params = 0 if fix_first_pose else 6
        n_pose_params += 6 * (n_poses - 1) if fix_first_pose else 6 * n_poses

        # Count actual pose parameters
        pose_params_count = 6 * (n_poses - 1) if fix_first_pose else 6 * n_poses

        x0 = self._pack_parameters(keyframes, map_points, fix_first_pose)

        # Store fixed first pose if needed
        fixed_pose_data = None
        if fix_first_pose and len(keyframes) > 0:
            kf0 = keyframes[0]
            # Convert T_world_camera to T_camera_world for optimization
            R_world_cam = kf0.pose.rotation
            t_world_cam = kf0.pose.translation
            R_cam_world = R_world_cam.T
            t_cam_world = -R_cam_world @ t_world_cam
            fixed_pose_data = (R_cam_world, t_cam_world)

        # Compute initial cost
        initial_residuals = self._compute_residuals(
            x0,
            observations,
            camera_matrix,
            n_poses,
            n_points,
            fix_first_pose,
            fixed_pose_data,
        )
        initial_cost = 0.5 * np.sum(initial_residuals**2)

        # Build sparse Jacobian sparsity pattern
        sparsity = self._build_sparsity_matrix(
            observations, n_poses, n_points, fix_first_pose
        )

        # Run optimization
        try:
            result = least_squares(
                fun=self._compute_residuals,
                x0=x0,
                jac_sparsity=sparsity,
                args=(
                    observations,
                    camera_matrix,
                    n_poses,
                    n_points,
                    fix_first_pose,
                    fixed_pose_data,
                ),
                method="trf",  # Trust Region Reflective
                loss=self._loss,
                ftol=self._ftol,
                xtol=self._xtol,
                max_nfev=self._max_iterations * len(x0),
                verbose=0,
            )
        except Exception as e:
            return BAResult(success=False, message=f"Optimization failed: {e}")

        # Unpack results
        optimized_poses, optimized_points = self._unpack_parameters(
            result.x,
            keyframes,
            map_points,
            fix_first_pose,
            fixed_pose_data,
        )

        final_cost = 0.5 * np.sum(result.fun**2)

        # Check for divergence
        if final_cost > initial_cost * 10:
            return BAResult(
                success=False,
                message="Optimization diverged",
                initial_cost=initial_cost,
                final_cost=final_cost,
            )

        return BAResult(
            success=result.success or final_cost < initial_cost,
            optimized_poses=optimized_poses,
            optimized_points=optimized_points,
            initial_cost=initial_cost,
            final_cost=final_cost,
            iterations=result.nfev,
            message=result.message,
        )

    def _collect_observations(
        self,
        keyframes: list[Keyframe],
        map_points: list[MapPoint],
        kf_id_to_idx: dict[int, int],
        mp_id_to_idx: dict[int, int],
    ) -> list[BAObservation]:
        """Collect all 2D observations for optimization."""
        observations = []
        mp_id_set = set(mp_id_to_idx.keys())

        for kf_idx, kf in enumerate(keyframes):
            for kp_idx, mp_id in kf.keypoint_to_mappoint.items():
                if mp_id not in mp_id_set:
                    continue

                mp_idx = mp_id_to_idx[mp_id]
                pixel = kf.keypoints[kp_idx]

                observations.append(
                    BAObservation(
                        keyframe_idx=kf_idx,
                        point_idx=mp_idx,
                        pixel=pixel,
                    )
                )

        return observations

    def _pack_parameters(
        self,
        keyframes: list[Keyframe],
        map_points: list[MapPoint],
        fix_first_pose: bool,
    ) -> np.ndarray:
        """Pack poses and points into a flat parameter vector."""
        params = []

        # Pack poses (as T_camera_world for optimization)
        for idx, kf in enumerate(keyframes):
            if fix_first_pose and idx == 0:
                continue

            # Convert T_world_camera to T_camera_world
            R_world_cam = kf.pose.rotation
            t_world_cam = kf.pose.translation
            R_cam_world = R_world_cam.T
            t_cam_world = -R_cam_world @ t_world_cam

            rvec = _rotation_to_rvec(R_cam_world)
            params.extend(rvec)
            params.extend(t_cam_world)

        # Pack 3D points
        for mp in map_points:
            params.extend(mp.position_world)

        return np.array(params, dtype=np.float64)

    def _unpack_parameters(
        self,
        params: np.ndarray,
        keyframes: list[Keyframe],
        map_points: list[MapPoint],
        fix_first_pose: bool,
        fixed_pose_data: tuple[np.ndarray, np.ndarray] | None,
    ) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, np.ndarray]]:
        """Unpack parameter vector to poses and points."""
        n_poses = len(keyframes)
        n_points = len(map_points)

        # Calculate offsets
        pose_params = 6 * (n_poses - 1) if fix_first_pose else 6 * n_poses
        points_start = pose_params

        # Unpack poses
        optimized_poses: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        offset = 0

        for idx, kf in enumerate(keyframes):
            if fix_first_pose and idx == 0:
                # Use fixed pose (convert back to T_world_camera)
                R_cam_world, t_cam_world = fixed_pose_data
                R_world_cam = R_cam_world.T
                t_world_cam = -R_world_cam @ t_cam_world
                optimized_poses[kf.id] = (R_world_cam, t_world_cam)
            else:
                rvec = params[offset : offset + 3]
                tvec = params[offset + 3 : offset + 6]
                offset += 6

                # Convert T_camera_world back to T_world_camera
                R_cam_world = _rvec_to_rotation(rvec)
                t_cam_world = tvec
                R_world_cam = R_cam_world.T
                t_world_cam = -R_world_cam @ t_cam_world

                optimized_poses[kf.id] = (R_world_cam, t_world_cam)

        # Unpack points
        optimized_points: dict[int, np.ndarray] = {}
        points_flat = params[points_start:]

        for idx, mp in enumerate(map_points):
            point = points_flat[idx * 3 : idx * 3 + 3]
            optimized_points[mp.id] = point.copy()

        return optimized_poses, optimized_points

    def _compute_residuals(
        self,
        params: np.ndarray,
        observations: list[BAObservation],
        camera_matrix: np.ndarray,
        n_poses: int,
        n_points: int,
        fix_first_pose: bool,
        fixed_pose_data: tuple[np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        """Compute reprojection residuals for all observations."""
        # Calculate offsets
        pose_params = 6 * (n_poses - 1) if fix_first_pose else 6 * n_poses
        points_start = pose_params

        # Extract poses (as T_camera_world)
        poses: list[tuple[np.ndarray, np.ndarray]] = []
        offset = 0

        for idx in range(n_poses):
            if fix_first_pose and idx == 0:
                poses.append(fixed_pose_data)
            else:
                rvec = params[offset : offset + 3]
                tvec = params[offset + 3 : offset + 6]
                offset += 6
                R = _rvec_to_rotation(rvec)
                poses.append((R, tvec))

        # Extract points
        points_flat = params[points_start:]
        points_3d = points_flat.reshape(-1, 3)

        # Compute residuals
        residuals = np.zeros(len(observations) * 2)

        for i, obs in enumerate(observations):
            R, t = poses[obs.keyframe_idx]
            point = points_3d[obs.point_idx]

            projected = _project_point(point, R, t, camera_matrix)
            error = obs.pixel - projected

            residuals[i * 2] = error[0]
            residuals[i * 2 + 1] = error[1]

        return residuals

    def _build_sparsity_matrix(
        self,
        observations: list[BAObservation],
        n_poses: int,
        n_points: int,
        fix_first_pose: bool,
    ) -> lil_matrix:
        """Build sparse Jacobian structure for efficient optimization.

        The Jacobian has structure where each observation only affects:
        - 6 pose parameters (for its keyframe)
        - 3 point parameters (for its map point)
        """
        pose_params = 6 * (n_poses - 1) if fix_first_pose else 6 * n_poses
        point_params = 3 * n_points
        n_params = pose_params + point_params
        n_residuals = len(observations) * 2

        # Use lil_matrix for efficient construction
        sparsity = lil_matrix((n_residuals, n_params), dtype=int)

        for i, obs in enumerate(observations):
            row_start = i * 2

            # Pose Jacobian entries
            if fix_first_pose:
                if obs.keyframe_idx > 0:
                    pose_col_start = (obs.keyframe_idx - 1) * 6
                    for j in range(6):
                        sparsity[row_start, pose_col_start + j] = 1
                        sparsity[row_start + 1, pose_col_start + j] = 1
            else:
                pose_col_start = obs.keyframe_idx * 6
                for j in range(6):
                    sparsity[row_start, pose_col_start + j] = 1
                    sparsity[row_start + 1, pose_col_start + j] = 1

            # Point Jacobian entries
            point_col_start = pose_params + obs.point_idx * 3
            for j in range(3):
                sparsity[row_start, point_col_start + j] = 1
                sparsity[row_start + 1, point_col_start + j] = 1

        return sparsity
