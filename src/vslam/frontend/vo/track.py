"""Feature track management for visual odometry.

Tracks are sequences of 2D observations of the same 3D point across
multiple frames. They bridge the gap between frame-to-frame matching
and persistent map points.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TrackObservation:
    """A single observation within a track.

    Attributes:
        frame_id: Frame where this observation occurred
        keypoint_idx: Index into frame's keypoints
        pixel_coords: 2D pixel coordinates (u, v)
    """

    frame_id: int
    keypoint_idx: int
    pixel_coords: np.ndarray  # (2,) float32

    def __post_init__(self) -> None:
        """Ensure pixel_coords is proper array."""
        self.pixel_coords = np.asarray(self.pixel_coords, dtype=np.float32).flatten()


@dataclass
class Track:
    """A sequence of 2D observations of the same 3D point.

    Tracks span multiple frames and are used to:
    1. Associate observations before triangulation
    2. Link 2D features to existing map points
    3. Decide when to triangulate new map points

    Attributes:
        track_id: Unique identifier for this track
        observations: List of observations in temporal order
        map_point_id: Associated map point ID (None if not triangulated)
    """

    track_id: int
    observations: list[TrackObservation] = field(default_factory=list)
    map_point_id: int | None = None

    @property
    def length(self) -> int:
        """Return number of observations in track."""
        return len(self.observations)

    @property
    def is_triangulated(self) -> bool:
        """Return True if track has associated map point."""
        return self.map_point_id is not None

    @property
    def first_frame_id(self) -> int | None:
        """Return frame ID of first observation."""
        if len(self.observations) == 0:
            return None
        return self.observations[0].frame_id

    @property
    def last_frame_id(self) -> int | None:
        """Return frame ID of last observation."""
        if len(self.observations) == 0:
            return None
        return self.observations[-1].frame_id

    def add_observation(
        self, frame_id: int, keypoint_idx: int, pixel_coords: np.ndarray
    ) -> None:
        """Add new observation to track.

        Args:
            frame_id: Frame ID of the observation
            keypoint_idx: Index into frame's keypoints
            pixel_coords: 2D pixel coordinates
        """
        obs = TrackObservation(
            frame_id=frame_id,
            keypoint_idx=keypoint_idx,
            pixel_coords=pixel_coords,
        )
        self.observations.append(obs)

    def get_latest_observation(self) -> TrackObservation | None:
        """Return most recent observation.

        Returns:
            Latest TrackObservation or None if empty
        """
        if len(self.observations) == 0:
            return None
        return self.observations[-1]

    def get_observation_in_frame(self, frame_id: int) -> TrackObservation | None:
        """Get observation in a specific frame.

        Args:
            frame_id: Frame to search

        Returns:
            TrackObservation if found, None otherwise
        """
        for obs in self.observations:
            if obs.frame_id == frame_id:
                return obs
        return None


class TrackManager:
    """Manages feature tracks across frames.

    Responsibilities:
    - Create new tracks for unmatched features
    - Extend existing tracks with new observations
    - Track which keypoints are associated with which tracks
    - Identify tracks ready for triangulation

    The TrackManager maintains a mapping from (frame_id, keypoint_idx)
    to track_id, enabling efficient lookup during temporal matching.
    """

    def __init__(self, min_track_length: int = 2) -> None:
        """Initialize track manager.

        Args:
            min_track_length: Minimum observations before a track
                is considered ready for triangulation
        """
        self._tracks: dict[int, Track] = {}
        self._next_track_id: int = 0
        self._min_track_length = min_track_length

        # Mapping from (frame_id, keypoint_idx) to track_id
        # Only stores the most recent frame's keypoints
        self._current_kp_to_track: dict[int, int] = {}  # keypoint_idx -> track_id
        self._current_frame_id: int | None = None

    def process_matches(
        self,
        frame_id: int,
        prev_indices: np.ndarray,
        curr_indices: np.ndarray,
        curr_keypoints: np.ndarray,
        prev_kp_to_track: dict[int, int] | None = None,
    ) -> dict[int, int]:
        """Process temporal matches and update tracks.

        For each match:
        - If previous keypoint has a track, extend it with current observation
        - If not, this is a new potential track (created on next match)

        Args:
            frame_id: Current frame ID
            prev_indices: Indices of matched keypoints in previous frame
            curr_indices: Indices of matched keypoints in current frame
            curr_keypoints: Nx2 array of current frame keypoint coordinates
            prev_kp_to_track: Mapping from previous keypoint index to track ID.
                If None, uses internal state from previous process_matches call.

        Returns:
            Mapping from current keypoint index to track ID
        """
        if prev_kp_to_track is None:
            prev_kp_to_track = self._current_kp_to_track

        new_kp_to_track: dict[int, int] = {}

        for prev_idx, curr_idx in zip(prev_indices, curr_indices):
            prev_idx = int(prev_idx)
            curr_idx = int(curr_idx)
            pixel_coords = curr_keypoints[curr_idx]

            if prev_idx in prev_kp_to_track:
                # Extend existing track
                track_id = prev_kp_to_track[prev_idx]
                track = self._tracks.get(track_id)
                if track is not None:
                    track.add_observation(frame_id, curr_idx, pixel_coords)
                    new_kp_to_track[curr_idx] = track_id

        # Update internal state
        self._current_kp_to_track = new_kp_to_track
        self._current_frame_id = frame_id

        return new_kp_to_track

    def create_tracks_for_new_points(
        self,
        frame_id: int,
        keypoint_indices: np.ndarray,
        keypoints: np.ndarray,
        map_point_ids: np.ndarray | None = None,
    ) -> dict[int, int]:
        """Create new tracks for keypoints not in existing tracks.

        Called when new map points are triangulated from stereo.
        Creates a track for each new point and optionally associates
        it with a map point ID.

        Args:
            frame_id: Current frame ID
            keypoint_indices: Indices of keypoints to create tracks for
            keypoints: Nx2 array of all keypoint coordinates
            map_point_ids: Optional array of map point IDs (same length as
                keypoint_indices). If provided, tracks are pre-associated.

        Returns:
            Mapping from keypoint index to newly created track ID
        """
        new_tracks: dict[int, int] = {}

        for i, kp_idx in enumerate(keypoint_indices):
            kp_idx = int(kp_idx)

            # Skip if already tracked
            if kp_idx in self._current_kp_to_track:
                continue

            # Create new track
            track_id = self._next_track_id
            self._next_track_id += 1

            track = Track(track_id=track_id)
            track.add_observation(frame_id, kp_idx, keypoints[kp_idx])

            # Associate with map point if provided
            if map_point_ids is not None:
                track.map_point_id = int(map_point_ids[i])

            self._tracks[track_id] = track
            self._current_kp_to_track[kp_idx] = track_id
            new_tracks[kp_idx] = track_id

        return new_tracks

    def get_triangulatable_tracks(self) -> list[Track]:
        """Return tracks with enough observations for triangulation.

        A track is triangulatable if:
        - It has at least min_track_length observations
        - It doesn't already have an associated map point

        Returns:
            List of tracks ready for triangulation
        """
        return [
            track
            for track in self._tracks.values()
            if track.length >= self._min_track_length and not track.is_triangulated
        ]

    def mark_triangulated(self, track_id: int, map_point_id: int) -> bool:
        """Associate a track with a triangulated map point.

        Args:
            track_id: Track ID to update
            map_point_id: Map point ID to associate

        Returns:
            True if track was found and updated
        """
        track = self._tracks.get(track_id)
        if track is None:
            return False

        track.map_point_id = map_point_id
        return True

    def get_track(self, track_id: int) -> Track | None:
        """Get track by ID.

        Args:
            track_id: Track identifier

        Returns:
            Track if found, None otherwise
        """
        return self._tracks.get(track_id)

    def get_track_for_keypoint(
        self, frame_id: int, keypoint_idx: int
    ) -> Track | None:
        """Get track associated with a keypoint in the current frame.

        Args:
            frame_id: Frame ID (must match current frame)
            keypoint_idx: Keypoint index

        Returns:
            Track if found, None otherwise
        """
        if frame_id != self._current_frame_id:
            return None

        track_id = self._current_kp_to_track.get(keypoint_idx)
        if track_id is None:
            return None

        return self._tracks.get(track_id)

    def get_current_keypoint_to_track(self) -> dict[int, int]:
        """Return current frame's keypoint to track mapping.

        Returns:
            Dictionary mapping keypoint index to track ID
        """
        return self._current_kp_to_track.copy()

    def get_current_keypoint_to_map_point(self) -> dict[int, int]:
        """Return current frame's keypoint to map point mapping.

        Only includes keypoints with triangulated tracks.

        Returns:
            Dictionary mapping keypoint index to map point ID
        """
        result: dict[int, int] = {}
        for kp_idx, track_id in self._current_kp_to_track.items():
            track = self._tracks.get(track_id)
            if track is not None and track.map_point_id is not None:
                result[kp_idx] = track.map_point_id
        return result

    @property
    def num_tracks(self) -> int:
        """Return total number of tracks."""
        return len(self._tracks)

    @property
    def num_active_tracks(self) -> int:
        """Return number of tracks active in current frame."""
        return len(self._current_kp_to_track)

    def clear(self) -> None:
        """Remove all tracks."""
        self._tracks.clear()
        self._current_kp_to_track.clear()
        self._current_frame_id = None
        self._next_track_id = 0
