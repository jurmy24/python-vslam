"""Frame-to-frame stereo visual odometry pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .map_point import Map, Observation
from .motion_estimator import MotionEstimator, PnPResult
from .pose import SE3
from .stereo_frontend import StereoFrame, StereoFrontend
from .temporal_matcher import TemporalMatcher


class TrackingStatus(Enum):
    """Status of visual odometry tracking."""

    OK = "OK"
    INITIALIZING = "INITIALIZING"
    LOST = "LOST"


@dataclass
class VOTiming:
    """Timing breakdown for a single frame."""

    stereo_ms: float = 0.0
    matching_ms: float = 0.0
    pnp_ms: float = 0.0
    map_update_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class VOFrame:
    """Output of visual odometry for a single frame."""

    frame_id: int
    timestamp_ns: int
    stereo_frame: StereoFrame
    pose: SE3
    tracked_points: int
    new_points: int
    tracking_status: TrackingStatus
    inlier_ratio: float = 0.0
    num_temporal_matches: int = 0
    num_correspondences: int = 0
    timing: VOTiming = field(default_factory=VOTiming)

    @property
    def position(self) -> np.ndarray:
        """Return camera position in world frame."""
        return self.pose.position

    @property
    def is_tracking_ok(self) -> bool:
        """Return True if tracking is successful."""
        return self.tracking_status == TrackingStatus.OK


def _build_coord_to_index_map(points: np.ndarray) -> dict[tuple[int, int], int]:
    """Build a hash map from rounded coordinates to keypoint index.

    This enables O(1) lookup of keypoint indices from coordinates.

    Args:
        points: Nx2 array of keypoint coordinates

    Returns:
        Dictionary mapping (rounded_x, rounded_y) to index
    """
    coord_map: dict[tuple[int, int], int] = {}
    for i, pt in enumerate(points):
        # Round to nearest pixel for hash key
        key = (int(round(pt[0])), int(round(pt[1])))
        coord_map[key] = i
    return coord_map


def _is_valid_pose(pose: SE3) -> bool:
    """Check if a pose has valid (finite) values."""
    return (
        np.isfinite(pose.rotation).all()
        and np.isfinite(pose.translation).all()
    )


class VisualOdometry:
    """Frame-to-frame stereo visual odometry.

    Orchestrates the full VO pipeline:
    1. Stereo processing (rectify, detect features, stereo match, triangulate)
    2. Temporal feature matching (track features from previous frame)
    3. Motion estimation (PnP + RANSAC from 3D-2D correspondences)
    4. Map management (add new 3D points, update tracks)
    """

    def __init__(
        self,
        stereo_frontend: StereoFrontend,
        temporal_matcher: TemporalMatcher | None = None,
        motion_estimator: MotionEstimator | None = None,
        min_tracked_points: int = 50,
        max_depth: float = 40.0,
        min_depth: float = 0.1,
        enable_timing: bool = True,
    ) -> None:
        """Initialize visual odometry pipeline."""
        self._stereo_frontend = stereo_frontend
        self._temporal_matcher = temporal_matcher or TemporalMatcher()
        self._motion_estimator = motion_estimator or MotionEstimator()

        self._min_tracked_points = min_tracked_points
        self._max_depth = max_depth
        self._min_depth = min_depth
        self._enable_timing = enable_timing

        # State
        self._map = Map()
        self._trajectory: list[SE3] = []
        self._frame_id: int = 0
        self._is_initialized: bool = False
        self._consecutive_lost: int = 0

        # Previous frame data
        self._prev_frame: StereoFrame | None = None
        self._prev_pose: SE3 | None = None
        self._prev_kp_to_map_point: dict[int, int] = {}
        self._prev_coord_map: dict[tuple[int, int], int] = {}

        # Camera intrinsics
        K = self._stereo_frontend.stereo_camera
        self._camera_matrix = np.array(
            [
                [K.focal_length, 0, K.principal_point[0]],
                [0, K.focal_length, K.principal_point[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_dataset_path(
        cls,
        dataset_path: str,
        n_features: int = 1000,
    ) -> VisualOdometry:
        """Create VisualOdometry from EuRoC dataset path."""
        stereo_frontend = StereoFrontend.from_dataset_path(
            dataset_path, n_features=n_features
        )
        return cls(
            stereo_frontend=stereo_frontend,
            temporal_matcher=TemporalMatcher(),
            motion_estimator=MotionEstimator(),
        )

    def process_frame(
        self,
        left: np.ndarray,
        right: np.ndarray,
        timestamp_ns: int,
    ) -> VOFrame:
        """Process a stereo frame through the full VO pipeline."""
        timing = VOTiming()
        t_start = time.perf_counter()

        frame_id = self._frame_id
        self._frame_id += 1

        # Stage 1: Stereo processing
        t0 = time.perf_counter()
        stereo_frame = self._stereo_frontend.process_frame(left, right, timestamp_ns)
        timing.stereo_ms = (time.perf_counter() - t0) * 1000

        # Build coordinate map for current frame (O(1) lookups later)
        curr_coord_map = _build_coord_to_index_map(stereo_frame.features_left.points)

        # Stage 2: Initialization (first frame)
        if not self._is_initialized:
            return self._initialize(frame_id, stereo_frame, curr_coord_map, timing)

        # Stage 3: Temporal matching
        t0 = time.perf_counter()
        temporal_matches = self._temporal_matcher.match(
            self._prev_frame.features_left,
            stereo_frame.features_left,
        )
        timing.matching_ms = (time.perf_counter() - t0) * 1000

        # Stage 4: Find 3D-2D correspondences
        points_3d, points_2d, curr_kp_indices = self._find_correspondences(
            temporal_matches.prev_indices,
            temporal_matches.curr_indices,
            stereo_frame.features_left.points,
        )

        # Stage 5: Motion estimation
        t0 = time.perf_counter()
        pnp_result = self._estimate_motion(points_3d, points_2d)
        timing.pnp_ms = (time.perf_counter() - t0) * 1000

        # Determine tracking status and pose
        if pnp_result.success and _is_valid_pose(pnp_result.pose):
            current_pose = pnp_result.pose
            tracking_status = TrackingStatus.OK
            self._consecutive_lost = 0
        else:
            # Tracking failed
            self._consecutive_lost += 1
            tracking_status = TrackingStatus.LOST

            # Only use prediction if we haven't been lost too long
            if self._consecutive_lost <= 3 and self._prev_pose is not None:
                predicted = self._predict_pose()
                if _is_valid_pose(predicted):
                    current_pose = predicted
                else:
                    current_pose = self._prev_pose
            else:
                # Too many consecutive failures - use last known good pose
                current_pose = self._prev_pose if self._prev_pose else SE3.identity()

        # Stage 6: Update map
        t0 = time.perf_counter()
        new_points, curr_kp_to_map_point = self._update_map(
            frame_id,
            stereo_frame,
            temporal_matches,
            pnp_result,
            current_pose,
            curr_coord_map,
        )
        timing.map_update_ms = (time.perf_counter() - t0) * 1000

        # Stage 7: Update state
        tracked_points = pnp_result.num_inliers if pnp_result.success else 0
        self._prev_frame = stereo_frame
        self._prev_pose = current_pose
        self._prev_kp_to_map_point = curr_kp_to_map_point
        self._prev_coord_map = curr_coord_map
        self._trajectory.append(current_pose)

        timing.total_ms = (time.perf_counter() - t_start) * 1000

        inlier_ratio = 0.0
        if len(points_3d) > 0:
            inlier_ratio = pnp_result.num_inliers / len(points_3d)

        return VOFrame(
            frame_id=frame_id,
            timestamp_ns=timestamp_ns,
            stereo_frame=stereo_frame,
            pose=current_pose,
            tracked_points=tracked_points,
            new_points=new_points,
            tracking_status=tracking_status,
            inlier_ratio=inlier_ratio,
            num_temporal_matches=len(temporal_matches),
            num_correspondences=len(points_3d),
            timing=timing,
        )

    def _initialize(
        self,
        frame_id: int,
        stereo_frame: StereoFrame,
        coord_map: dict[tuple[int, int], int],
        timing: VOTiming,
    ) -> VOFrame:
        """Initialize VO with first frame."""
        initial_pose = SE3.identity()

        t0 = time.perf_counter()
        new_points, kp_to_map_point = self._add_map_points_from_stereo(
            frame_id, stereo_frame, initial_pose, coord_map, set()
        )
        timing.map_update_ms = (time.perf_counter() - t0) * 1000

        self._prev_frame = stereo_frame
        self._prev_pose = initial_pose
        self._prev_kp_to_map_point = kp_to_map_point
        self._prev_coord_map = coord_map
        self._trajectory.append(initial_pose)
        self._is_initialized = True

        return VOFrame(
            frame_id=frame_id,
            timestamp_ns=stereo_frame.timestamp_ns,
            stereo_frame=stereo_frame,
            pose=initial_pose,
            tracked_points=0,
            new_points=new_points,
            tracking_status=TrackingStatus.INITIALIZING,
            timing=timing,
        )

    def _find_correspondences(
        self,
        prev_indices: np.ndarray,
        curr_indices: np.ndarray,
        curr_keypoints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, list[int]]:
        """Find 3D-2D correspondences from temporal matches."""
        points_3d = []
        points_2d = []
        curr_kp_indices = []

        for prev_idx, curr_idx in zip(prev_indices, curr_indices):
            prev_idx = int(prev_idx)
            curr_idx = int(curr_idx)

            map_point_id = self._prev_kp_to_map_point.get(prev_idx)
            if map_point_id is None:
                continue

            map_point = self._map.get_point(map_point_id)
            if map_point is None:
                continue

            # Validate map point position
            if not np.isfinite(map_point.position_world).all():
                continue

            points_3d.append(map_point.position_world)
            points_2d.append(curr_keypoints[curr_idx])
            curr_kp_indices.append(curr_idx)

        if len(points_3d) == 0:
            return (
                np.empty((0, 3), dtype=np.float64),
                np.empty((0, 2), dtype=np.float32),
                [],
            )

        return (
            np.array(points_3d, dtype=np.float64),
            np.array(points_2d, dtype=np.float32),
            curr_kp_indices,
        )

    def _estimate_motion(
        self, points_3d: np.ndarray, points_2d: np.ndarray
    ) -> PnPResult:
        """Estimate camera motion using PnP."""
        return self._motion_estimator.estimate_pose(
            points_3d=points_3d,
            points_2d=points_2d,
            camera_matrix=self._camera_matrix,
            initial_pose=self._prev_pose,
        )

    def _predict_pose(self) -> SE3:
        """Predict current pose using constant velocity model."""
        if len(self._trajectory) < 2:
            return self._prev_pose if self._prev_pose else SE3.identity()

        T_prev = self._trajectory[-1]
        T_prev_prev = self._trajectory[-2]

        # Validate previous poses
        if not _is_valid_pose(T_prev) or not _is_valid_pose(T_prev_prev):
            return self._prev_pose if self._prev_pose else SE3.identity()

        # delta = T_prev * T_prev_prev^-1
        T_prev_prev_inv = T_prev_prev.inverse()
        delta = T_prev @ T_prev_prev_inv

        # predicted = delta * T_prev
        return delta @ T_prev

    def _update_map(
        self,
        frame_id: int,
        stereo_frame: StereoFrame,
        temporal_matches,
        pnp_result: PnPResult,
        current_pose: SE3,
        coord_map: dict[tuple[int, int], int],
    ) -> tuple[int, dict[int, int]]:
        """Update map with new points and observations.

        Returns:
            Tuple of (num_new_points, curr_kp_to_map_point)
        """
        curr_kp_to_map_point: dict[int, int] = {}

        # Transfer map point associations from tracked matches
        for prev_idx, curr_idx in zip(
            temporal_matches.prev_indices, temporal_matches.curr_indices
        ):
            prev_idx = int(prev_idx)
            curr_idx = int(curr_idx)

            map_point_id = self._prev_kp_to_map_point.get(prev_idx)
            if map_point_id is not None:
                map_point = self._map.get_point(map_point_id)
                if map_point is not None:
                    # Add observation
                    pixel_coords = stereo_frame.features_left.points[curr_idx]
                    obs = Observation(
                        frame_id=frame_id,
                        keypoint_idx=curr_idx,
                        pixel_coords=pixel_coords,
                    )
                    self._map.add_observation(map_point_id, obs)
                    curr_kp_to_map_point[curr_idx] = map_point_id

        # Add new map points if needed
        new_points = 0
        tracked_count = len(curr_kp_to_map_point)

        # Only add new points if:
        # 1. We need more tracked points
        # 2. The current pose is valid
        if tracked_count < self._min_tracked_points and _is_valid_pose(current_pose):
            new_points, new_kp_map = self._add_map_points_from_stereo(
                frame_id,
                stereo_frame,
                current_pose,
                coord_map,
                set(curr_kp_to_map_point.keys()),
            )
            # Merge new associations
            curr_kp_to_map_point.update(new_kp_map)

        return new_points, curr_kp_to_map_point

    def _add_map_points_from_stereo(
        self,
        frame_id: int,
        stereo_frame: StereoFrame,
        pose: SE3,
        coord_map: dict[tuple[int, int], int],
        exclude_keypoints: set[int],
    ) -> tuple[int, dict[int, int]]:
        """Add new map points from stereo triangulation.

        Returns:
            Tuple of (num_new_points, keypoint_to_map_point_dict)
        """
        kp_to_map_point: dict[int, int] = {}

        matches = stereo_frame.matches
        points_3d_camera = stereo_frame.points_3d
        features_left = stereo_frame.features_left

        if len(points_3d_camera) == 0 or features_left.descriptors is None:
            return 0, kp_to_map_point

        pts_left = matches.pts_left
        new_count = 0

        for i in range(len(points_3d_camera)):
            # O(1) lookup of keypoint index using coordinate hash
            match_pt = pts_left[i]
            key = (int(round(match_pt[0])), int(round(match_pt[1])))
            kp_idx = coord_map.get(key)

            if kp_idx is None or kp_idx in exclude_keypoints:
                continue

            # Get 3D point in camera frame
            point_camera = points_3d_camera[i]

            # Validate point
            if not np.isfinite(point_camera).all():
                continue

            depth = point_camera[2]
            if depth < self._min_depth or depth > self._max_depth:
                continue

            # Transform to world frame
            point_world = pose.transform_point(point_camera)

            # Validate transformed point
            if not np.isfinite(point_world).all():
                continue

            descriptor = features_left.descriptors[kp_idx]

            obs = Observation(
                frame_id=frame_id,
                keypoint_idx=kp_idx,
                pixel_coords=match_pt,
            )

            map_point = self._map.add_point(
                position=point_world,
                descriptor=descriptor,
                observation=obs,
            )

            kp_to_map_point[kp_idx] = map_point.id
            new_count += 1

        return new_count, kp_to_map_point

    def get_trajectory(self) -> list[SE3]:
        """Return all estimated camera poses."""
        return self._trajectory.copy()

    def get_trajectory_positions(self) -> np.ndarray:
        """Return camera positions as array."""
        if len(self._trajectory) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.array([p.position for p in self._trajectory], dtype=np.float64)

    def get_map(self) -> Map:
        """Return the sparse map."""
        return self._map

    @property
    def is_initialized(self) -> bool:
        return self._is_initialized

    @property
    def num_frames(self) -> int:
        return self._frame_id

    @property
    def num_map_points(self) -> int:
        return self._map.num_points

    @property
    def current_pose(self) -> SE3 | None:
        if len(self._trajectory) == 0:
            return None
        return self._trajectory[-1]

    def reset(self) -> None:
        """Reset VO to initial state."""
        self._map.clear()
        self._trajectory.clear()
        self._frame_id = 0
        self._is_initialized = False
        self._consecutive_lost = 0
        self._prev_frame = None
        self._prev_pose = None
        self._prev_kp_to_map_point.clear()
        self._prev_coord_map.clear()
