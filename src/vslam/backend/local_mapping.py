"""Local mapping with sliding window bundle adjustment.

LocalMapping maintains a sliding window of recent keyframes and runs
local bundle adjustment to refine poses and 3D points within that window.

The sliding window approach bounds computational complexity while still
correcting drift in the local trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .covisibility import CovisibilityGraph
from .keyframe import Keyframe, KeyframeManager
from .optimizer import BAResult, ScipyBundleAdjustment

if TYPE_CHECKING:
    from ..frontend.vo.map_point import Map, MapPoint
    from ..frontend.pose import SE3


@dataclass
class LocalBAConfig:
    """Configuration for local bundle adjustment."""

    window_size: int = 10  # Max keyframes in sliding window
    min_observations: int = 20  # Minimum observations to run BA
    max_iterations: int = 50  # Max LM iterations
    loss_function: str = "huber"  # Robust loss function


@dataclass
class LocalMappingResult:
    """Result of local mapping operation."""

    success: bool
    ba_result: BAResult | None = None
    num_keyframes_optimized: int = 0
    num_points_optimized: int = 0
    message: str = ""


class LocalMapping:
    """Manages local bundle adjustment with a sliding window.

    Maintains a window of recent keyframes and their associated map points.
    When a new keyframe is added, runs local BA to refine the estimates.
    """

    def __init__(
        self,
        camera_matrix: np.ndarray,
        config: LocalBAConfig | None = None,
    ) -> None:
        """Initialize local mapping.

        Args:
            camera_matrix: 3x3 camera intrinsics matrix
            config: Configuration for local BA
        """
        self._camera_matrix = camera_matrix
        self._config = config or LocalBAConfig()

        # Components
        self._keyframe_manager = KeyframeManager()
        self._covisibility_graph = CovisibilityGraph()
        self._optimizer = ScipyBundleAdjustment(
            max_iterations=self._config.max_iterations,
            loss=self._config.loss_function,
        )

        # Map reference (set later when connected to frontend)
        self._map: Map | None = None

        # Sliding window keyframe IDs
        self._window_ids: list[int] = []

    def set_map(self, map_ref: Map) -> None:
        """Set reference to the map.

        Args:
            map_ref: Reference to the frontend's Map
        """
        self._map = map_ref

    def add_keyframe(
        self,
        keyframe: Keyframe,
        new_map_points: list[MapPoint] | None = None,
    ) -> LocalMappingResult:
        """Add a keyframe and run local bundle adjustment.

        Args:
            keyframe: New keyframe to add
            new_map_points: New map points associated with this keyframe

        Returns:
            Result of local mapping operation
        """
        # Add to managers
        self._keyframe_manager.add_keyframe(keyframe)
        self._covisibility_graph.add_keyframe(keyframe)

        # Update sliding window
        self._window_ids.append(keyframe.id)
        if len(self._window_ids) > self._config.window_size:
            # Remove oldest keyframe from window (but keep in manager)
            self._window_ids.pop(0)

        # Get keyframes for local BA
        window_keyframes = self._get_window_keyframes()

        if len(window_keyframes) < 2:
            return LocalMappingResult(
                success=True,
                message="Not enough keyframes for BA",
                num_keyframes_optimized=0,
            )

        # Get map points observed by window keyframes
        window_map_points = self._get_window_map_points(window_keyframes)

        if len(window_map_points) < self._config.min_observations // 2:
            return LocalMappingResult(
                success=True,
                message="Not enough map points for BA",
                num_keyframes_optimized=len(window_keyframes),
                num_points_optimized=0,
            )

        # Run bundle adjustment
        ba_result = self._optimizer.optimize(
            keyframes=window_keyframes,
            map_points=window_map_points,
            camera_matrix=self._camera_matrix,
            fix_first_pose=True,  # Fix oldest keyframe in window
        )

        if ba_result.success:
            # Apply corrections
            self._apply_corrections(ba_result)

        return LocalMappingResult(
            success=ba_result.success,
            ba_result=ba_result,
            num_keyframes_optimized=len(window_keyframes),
            num_points_optimized=len(window_map_points),
            message=ba_result.message,
        )

    def _get_window_keyframes(self) -> list[Keyframe]:
        """Get keyframes currently in the sliding window."""
        keyframes = []
        for kf_id in self._window_ids:
            kf = self._keyframe_manager.get_keyframe(kf_id)
            if kf is not None:
                keyframes.append(kf)
        return keyframes

    def _get_window_map_points(
        self, window_keyframes: list[Keyframe]
    ) -> list[MapPoint]:
        """Get all map points observed by window keyframes.

        Only includes map points observed by at least 2 keyframes
        in the window (triangulated with parallax).
        """
        if self._map is None:
            return []

        # Count observations per map point
        mp_observation_count: dict[int, int] = {}
        for kf in window_keyframes:
            for mp_id in kf.keypoint_to_mappoint.values():
                mp_observation_count[mp_id] = mp_observation_count.get(mp_id, 0) + 1

        # Get map points observed by at least 2 keyframes
        map_points = []
        for mp_id, count in mp_observation_count.items():
            if count >= 2:
                mp = self._map.get_point(mp_id)
                if mp is not None:
                    map_points.append(mp)

        return map_points

    def _apply_corrections(self, ba_result: BAResult) -> None:
        """Apply BA corrections to keyframes and map points."""
        # Update keyframe poses
        for kf_id, (R, t) in ba_result.optimized_poses.items():
            from ..frontend.pose import SE3

            new_pose = SE3(rotation=R, translation=t)
            self._keyframe_manager.update_keyframe_pose(kf_id, new_pose)

        # Update map point positions
        if self._map is not None:
            for mp_id, position in ba_result.optimized_points.items():
                self._map.update_point_position(mp_id, position)

    def get_corrected_poses(self) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        """Get all corrected poses from recent BA.

        Returns:
            Dictionary of kf_id -> (rotation, translation)
        """
        poses = {}
        for kf in self._keyframe_manager.get_all_keyframes():
            poses[kf.id] = (kf.pose.rotation, kf.pose.translation)
        return poses

    def get_recent_keyframe(self) -> Keyframe | None:
        """Get the most recent keyframe."""
        return self._keyframe_manager.last_keyframe

    @property
    def num_keyframes(self) -> int:
        """Return total number of keyframes."""
        return self._keyframe_manager.num_keyframes

    @property
    def window_size(self) -> int:
        """Return current sliding window size."""
        return len(self._window_ids)

    @property
    def keyframe_manager(self) -> KeyframeManager:
        """Return the keyframe manager."""
        return self._keyframe_manager

    @property
    def covisibility_graph(self) -> CovisibilityGraph:
        """Return the covisibility graph."""
        return self._covisibility_graph
