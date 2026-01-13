"""Pose graph optimization for global drift correction.

When a loop closure is detected, we have a constraint between two
non-adjacent keyframes. Pose graph optimization distributes the
accumulated drift across the entire trajectory by minimizing the
error in all relative pose constraints.

Unlike bundle adjustment which optimizes poses AND 3D points,
pose graph optimization only optimizes poses (faster, global).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from ..frontend import SE3


@dataclass
class PoseEdge:
    """An edge in the pose graph.

    Attributes:
        from_id: Source keyframe ID
        to_id: Target keyframe ID
        measurement: Measured relative transform T_from_to
        information: 6x6 information matrix (inverse covariance)
        is_loop: Whether this is a loop closure edge
    """

    from_id: int
    to_id: int
    measurement: SE3
    information: np.ndarray = field(
        default_factory=lambda: np.eye(6, dtype=np.float64)
    )
    is_loop: bool = False


class PoseGraph:
    """Pose graph for global trajectory optimization.

    Stores keyframe poses and relative constraints (odometry and loop edges).
    Optimizes all poses jointly when a loop closure is added.
    """

    def __init__(self) -> None:
        """Initialize empty pose graph."""
        self._poses: dict[int, SE3] = {}
        self._odometry_edges: list[PoseEdge] = []
        self._loop_edges: list[PoseEdge] = []

    def add_keyframe(self, keyframe_id: int, pose: SE3) -> None:
        """Add a keyframe to the pose graph.

        Args:
            keyframe_id: Unique keyframe ID
            pose: Camera pose in world frame
        """
        self._poses[keyframe_id] = SE3(
            rotation=pose.rotation.copy(),
            translation=pose.translation.copy(),
        )

    def add_odometry_edge(
        self,
        from_id: int,
        to_id: int,
        relative_pose: SE3,
        information: np.ndarray | None = None,
    ) -> None:
        """Add an odometry edge between consecutive keyframes.

        Args:
            from_id: Source keyframe ID
            to_id: Target keyframe ID
            relative_pose: Measured relative transform
            information: 6x6 information matrix (default: identity)
        """
        if information is None:
            information = np.eye(6, dtype=np.float64)

        edge = PoseEdge(
            from_id=from_id,
            to_id=to_id,
            measurement=relative_pose,
            information=information.copy(),
            is_loop=False,
        )
        self._odometry_edges.append(edge)

    def add_loop_edge(
        self,
        from_id: int,
        to_id: int,
        relative_pose: SE3,
        information: np.ndarray | None = None,
    ) -> None:
        """Add a loop closure edge.

        Args:
            from_id: Query keyframe ID
            to_id: Match keyframe ID
            relative_pose: Measured relative transform from geometric verification
            information: 6x6 information matrix (default: 10x identity for higher weight)
        """
        if information is None:
            # Loop edges have higher weight (more confident)
            information = 10.0 * np.eye(6, dtype=np.float64)

        edge = PoseEdge(
            from_id=from_id,
            to_id=to_id,
            measurement=relative_pose,
            information=information.copy(),
            is_loop=True,
        )
        self._loop_edges.append(edge)

    def optimize(
        self,
        fix_first: bool = True,
        max_iterations: int = 50,
    ) -> dict[int, SE3]:
        """Optimize all poses in the graph.

        Args:
            fix_first: If True, fix the first pose (gauge freedom)
            max_iterations: Maximum optimization iterations

        Returns:
            Dictionary of optimized poses (keyframe_id -> SE3)
        """
        if len(self._poses) < 2:
            return {k: SE3(rotation=v.rotation.copy(), translation=v.translation.copy())
                    for k, v in self._poses.items()}

        # Get sorted pose IDs for consistent ordering
        pose_ids = sorted(self._poses.keys())
        n_poses = len(pose_ids)
        id_to_idx = {pid: i for i, pid in enumerate(pose_ids)}

        # Build initial parameter vector
        # Each pose: 6 params (rvec: 3, tvec: 3)
        params = []
        for pid in pose_ids:
            pose = self._poses[pid]
            rvec, _ = cv2.Rodrigues(pose.rotation)
            params.extend(rvec.flatten())
            params.extend(pose.translation.flatten())
        params = np.array(params, dtype=np.float64)

        # All edges
        all_edges = self._odometry_edges + self._loop_edges

        if len(all_edges) == 0:
            return {k: SE3(rotation=v.rotation.copy(), translation=v.translation.copy())
                    for k, v in self._poses.items()}

        # Build sparse Jacobian structure
        n_residuals = len(all_edges) * 6
        n_params = n_poses * 6 if not fix_first else (n_poses - 1) * 6

        def build_jacobian_sparsity():
            jac = lil_matrix((n_residuals, n_params), dtype=np.float64)

            for e_idx, edge in enumerate(all_edges):
                from_idx = id_to_idx[edge.from_id]
                to_idx = id_to_idx[edge.to_id]

                for r in range(6):  # 6 residuals per edge
                    res_idx = e_idx * 6 + r

                    # From pose affects residual (if not fixed)
                    if not (fix_first and from_idx == 0):
                        param_start = (from_idx - (1 if fix_first else 0)) * 6
                        if param_start >= 0:
                            for p in range(6):
                                jac[res_idx, param_start + p] = 1

                    # To pose affects residual (if not fixed)
                    if not (fix_first and to_idx == 0):
                        param_start = (to_idx - (1 if fix_first else 0)) * 6
                        if param_start >= 0:
                            for p in range(6):
                                jac[res_idx, param_start + p] = 1

            return jac.tocsr()

        jac_sparsity = build_jacobian_sparsity()

        # Residual function
        def residuals(params_opt):
            # Unpack poses
            poses = {}
            if fix_first:
                # First pose is fixed
                poses[pose_ids[0]] = self._poses[pose_ids[0]]
                for i, pid in enumerate(pose_ids[1:]):
                    offset = i * 6
                    rvec = params_opt[offset : offset + 3]
                    tvec = params_opt[offset + 3 : offset + 6]
                    R, _ = cv2.Rodrigues(rvec)
                    poses[pid] = SE3(rotation=R, translation=tvec)
            else:
                for i, pid in enumerate(pose_ids):
                    offset = i * 6
                    rvec = params_opt[offset : offset + 3]
                    tvec = params_opt[offset + 3 : offset + 6]
                    R, _ = cv2.Rodrigues(rvec)
                    poses[pid] = SE3(rotation=R, translation=tvec)

            # Compute residuals for all edges
            all_residuals = []

            for edge in all_edges:
                T_i = poses[edge.from_id]
                T_j = poses[edge.to_id]

                # Predicted relative transform: T_i^-1 @ T_j
                T_ij_pred = T_i.inverse().compose(T_j)

                # Error: log(T_ij_pred^-1 @ T_ij_meas)
                T_err = T_ij_pred.inverse().compose(edge.measurement)
                error = self._se3_log(T_err)

                # Weight by sqrt of information
                sqrt_info = np.sqrt(np.diag(edge.information))
                weighted_error = error * sqrt_info

                all_residuals.extend(weighted_error)

            return np.array(all_residuals)

        # Initial params (skip first pose if fixed)
        if fix_first:
            params_opt = params[6:]  # Skip first pose
        else:
            params_opt = params

        # Optimize using Trust Region Reflective (supports sparse Jacobians)
        result = least_squares(
            residuals,
            params_opt,
            method="trf",  # 'lm' doesn't support jac_sparsity
            jac_sparsity=jac_sparsity,
            ftol=1e-6,
            max_nfev=max_iterations * len(params_opt),
        )

        # Unpack optimized poses
        optimized_poses = {}

        if fix_first:
            optimized_poses[pose_ids[0]] = SE3(
                rotation=self._poses[pose_ids[0]].rotation.copy(),
                translation=self._poses[pose_ids[0]].translation.copy(),
            )
            for i, pid in enumerate(pose_ids[1:]):
                offset = i * 6
                rvec = result.x[offset : offset + 3]
                tvec = result.x[offset + 3 : offset + 6]
                R, _ = cv2.Rodrigues(rvec)
                optimized_poses[pid] = SE3(rotation=R, translation=tvec)
        else:
            for i, pid in enumerate(pose_ids):
                offset = i * 6
                rvec = result.x[offset : offset + 3]
                tvec = result.x[offset + 3 : offset + 6]
                R, _ = cv2.Rodrigues(rvec)
                optimized_poses[pid] = SE3(rotation=R, translation=tvec)

        # Update internal poses
        for pid, pose in optimized_poses.items():
            self._poses[pid] = pose

        return optimized_poses

    def _se3_log(self, pose: SE3) -> np.ndarray:
        """Compute SE(3) logarithm (6D error vector).

        Args:
            pose: SE3 pose

        Returns:
            6D error vector [rotation_error, translation_error]
        """
        # Rotation error (axis-angle)
        rvec, _ = cv2.Rodrigues(pose.rotation)
        rotation_error = rvec.flatten()

        # Translation error
        translation_error = pose.translation.flatten()

        return np.concatenate([rotation_error, translation_error])

    def get_pose(self, keyframe_id: int) -> SE3 | None:
        """Get pose of a keyframe.

        Args:
            keyframe_id: Keyframe ID

        Returns:
            SE3 pose if found, None otherwise
        """
        return self._poses.get(keyframe_id)

    @property
    def num_poses(self) -> int:
        """Number of poses in graph."""
        return len(self._poses)

    @property
    def num_edges(self) -> int:
        """Total number of edges in graph."""
        return len(self._odometry_edges) + len(self._loop_edges)

    @property
    def num_loop_edges(self) -> int:
        """Number of loop closure edges."""
        return len(self._loop_edges)

    @property
    def poses(self) -> dict[int, SE3]:
        """Get all poses (read-only copy)."""
        return {k: SE3(rotation=v.rotation.copy(), translation=v.translation.copy())
                for k, v in self._poses.items()}
