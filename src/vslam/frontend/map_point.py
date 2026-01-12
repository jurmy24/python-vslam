"""Map point and sparse map data structures for visual odometry."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Observation:
    """A single observation of a map point in a frame.

    Records where a 3D map point was observed in a specific frame,
    linking the 2D image measurement to the 3D landmark.

    Attributes:
        frame_id: Unique identifier of the observing frame
        keypoint_idx: Index into the frame's keypoints array
        pixel_coords: 2D pixel coordinates (u, v) of the observation
    """

    frame_id: int
    keypoint_idx: int
    pixel_coords: np.ndarray  # (2,) float32

    def __post_init__(self) -> None:
        """Ensure pixel_coords is proper array."""
        self.pixel_coords = np.asarray(self.pixel_coords, dtype=np.float32).flatten()


@dataclass
class MapPoint:
    """A 3D landmark in the map with its observations.

    Map points are triangulated from stereo or multi-view observations
    and represent persistent 3D features in the environment. Each map
    point tracks all frames that have observed it, enabling bundle
    adjustment and loop closure.

    Attributes:
        id: Unique identifier for this map point
        position_world: 3D position in world frame
        descriptor: Representative ORB descriptor (32 bytes) for matching
        observations: List of observations from different frames
        is_valid: False if point has been culled or merged
    """

    id: int
    position_world: np.ndarray  # (3,) float64
    descriptor: np.ndarray  # (32,) uint8
    observations: list[Observation] = field(default_factory=list)
    is_valid: bool = True

    def __post_init__(self) -> None:
        """Ensure arrays are proper types."""
        self.position_world = np.asarray(self.position_world, dtype=np.float64).flatten()
        self.descriptor = np.asarray(self.descriptor, dtype=np.uint8).flatten()

    @property
    def num_observations(self) -> int:
        """Return number of frames observing this point."""
        return len(self.observations)

    def add_observation(self, obs: Observation) -> None:
        """Add a new observation of this point.

        Args:
            obs: Observation to add
        """
        self.observations.append(obs)

    def get_observation_in_frame(self, frame_id: int) -> Observation | None:
        """Get observation in a specific frame.

        Args:
            frame_id: Frame to search for

        Returns:
            Observation if found, None otherwise
        """
        for obs in self.observations:
            if obs.frame_id == frame_id:
                return obs
        return None

    def is_observed_in_frame(self, frame_id: int) -> bool:
        """Check if this point was observed in a frame.

        Args:
            frame_id: Frame to check

        Returns:
            True if point was observed in frame
        """
        return any(obs.frame_id == frame_id for obs in self.observations)

    def update_position(self, new_position: np.ndarray) -> None:
        """Update the 3D position of this map point.

        Used during bundle adjustment or re-triangulation.

        Args:
            new_position: New 3D position in world frame
        """
        self.position_world = np.asarray(new_position, dtype=np.float64).flatten()


class Map:
    """Sparse map of 3D landmarks and their observations.

    The map stores all triangulated 3D points and provides efficient
    lookup for:
    - Finding all map points visible in a given frame
    - Getting 3D position given a map point ID
    - Adding new points and observations

    This is the core data structure that accumulates knowledge about
    the environment as the camera moves through the scene.
    """

    def __init__(self) -> None:
        """Initialize empty map."""
        self._points: dict[int, MapPoint] = {}
        self._frame_to_points: dict[int, set[int]] = {}  # frame_id -> {point_ids}
        self._next_point_id: int = 0

    def add_point(
        self,
        position: np.ndarray,
        descriptor: np.ndarray,
        observation: Observation,
    ) -> MapPoint:
        """Create and add a new map point.

        Args:
            position: 3D position in world frame
            descriptor: ORB descriptor for matching
            observation: Initial observation of this point

        Returns:
            The newly created MapPoint
        """
        point_id = self._next_point_id
        self._next_point_id += 1

        point = MapPoint(
            id=point_id,
            position_world=position,
            descriptor=descriptor,
            observations=[observation],
        )
        self._points[point_id] = point

        # Index by frame
        frame_id = observation.frame_id
        if frame_id not in self._frame_to_points:
            self._frame_to_points[frame_id] = set()
        self._frame_to_points[frame_id].add(point_id)

        return point

    def get_point(self, point_id: int) -> MapPoint | None:
        """Get map point by ID.

        Args:
            point_id: Unique point identifier

        Returns:
            MapPoint if found and valid, None otherwise
        """
        point = self._points.get(point_id)
        if point is not None and point.is_valid:
            return point
        return None

    def get_points_in_frame(self, frame_id: int) -> list[MapPoint]:
        """Get all valid map points observed in a frame.

        Args:
            frame_id: Frame identifier

        Returns:
            List of valid MapPoints observed in the frame
        """
        point_ids = self._frame_to_points.get(frame_id, set())
        return [
            self._points[pid]
            for pid in point_ids
            if pid in self._points and self._points[pid].is_valid
        ]

    def add_observation(self, point_id: int, observation: Observation) -> bool:
        """Add an observation to an existing map point.

        Args:
            point_id: ID of the map point
            observation: New observation to add

        Returns:
            True if observation was added, False if point not found
        """
        point = self.get_point(point_id)
        if point is None:
            return False

        point.add_observation(observation)

        # Update frame index
        frame_id = observation.frame_id
        if frame_id not in self._frame_to_points:
            self._frame_to_points[frame_id] = set()
        self._frame_to_points[frame_id].add(point_id)

        return True

    def remove_point(self, point_id: int) -> bool:
        """Mark a map point as invalid (soft delete).

        The point remains in memory but is excluded from queries.
        This is safer than hard deletion during optimization.

        Args:
            point_id: ID of the point to remove

        Returns:
            True if point was found and marked invalid
        """
        point = self._points.get(point_id)
        if point is None:
            return False

        point.is_valid = False
        return True

    def get_all_points(self) -> list[MapPoint]:
        """Return all valid map points.

        Returns:
            List of all valid MapPoints in the map
        """
        return [p for p in self._points.values() if p.is_valid]

    def get_all_positions(self) -> np.ndarray:
        """Return positions of all valid map points.

        Returns:
            Nx3 array of 3D positions in world frame
        """
        points = self.get_all_points()
        if len(points) == 0:
            return np.empty((0, 3), dtype=np.float64)
        return np.array([p.position_world for p in points], dtype=np.float64)

    @property
    def num_points(self) -> int:
        """Return number of valid map points."""
        return sum(1 for p in self._points.values() if p.is_valid)

    @property
    def num_frames(self) -> int:
        """Return number of frames with observations."""
        return len(self._frame_to_points)

    def clear(self) -> None:
        """Remove all points from the map."""
        self._points.clear()
        self._frame_to_points.clear()
        self._next_point_id = 0

    def __len__(self) -> int:
        """Return number of valid map points."""
        return self.num_points
