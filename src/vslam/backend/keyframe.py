"""Keyframe data structure and selection logic.

A keyframe is a frame that is selected for bundle adjustment optimization.
Not every frame becomes a keyframe - only those with sufficient parallax
(rotation or translation) from the previous keyframe are selected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from ..frontend.vo.feature_detector import Features
    from ..frontend.vo.map_point import Map
    from ..frontend.pose import SE3
    from ..frontend.vo.visual_odometry import VOFrame


def _rotation_angle_from_matrix(R: np.ndarray) -> float:
    """Extract rotation angle from a 3x3 rotation matrix.

    Uses the trace formula: trace(R) = 1 + 2*cos(theta)

    Args:
        R: 3x3 rotation matrix

    Returns:
        Rotation angle in radians [0, pi]
    """
    trace = np.trace(R)
    # Clamp for numerical stability
    cos_theta = np.clip((trace - 1) / 2, -1.0, 1.0)
    return np.arccos(cos_theta)


@dataclass
class Keyframe:
    """A keyframe selected for bundle adjustment.

    Contains all data needed for optimization:
    - Camera pose in world frame
    - Feature observations (2D keypoints)
    - Associations to 3D map points
    """

    id: int  # Same as VOFrame.frame_id
    timestamp_ns: int
    pose: SE3  # Camera pose T_world_camera
    # Feature data
    keypoints: np.ndarray  # (N, 2) keypoint pixel coordinates
    descriptors: np.ndarray  # (N, 32) ORB descriptors
    # Map point associations: keypoint_idx -> mappoint_id
    keypoint_to_mappoint: dict[int, int] = field(default_factory=dict)

    @classmethod
    def from_vo_frame(
        cls,
        vo_frame: VOFrame,
        keypoint_to_mappoint: dict[int, int],
    ) -> Keyframe:
        """Create a keyframe from a VOFrame.

        Args:
            vo_frame: Visual odometry output frame
            keypoint_to_mappoint: Mapping from keypoint index to map point ID

        Returns:
            New Keyframe instance
        """
        features = vo_frame.stereo_frame.features_left
        return cls(
            id=vo_frame.frame_id,
            timestamp_ns=vo_frame.timestamp_ns,
            pose=vo_frame.pose,
            keypoints=features.points.copy(),
            descriptors=features.descriptors.copy()
            if features.descriptors is not None
            else np.empty((0, 32), dtype=np.uint8),
            keypoint_to_mappoint=keypoint_to_mappoint.copy(),
        )

    def get_observed_mappoint_ids(self) -> list[int]:
        """Return list of map point IDs observed by this keyframe."""
        return list(self.keypoint_to_mappoint.values())

    def get_observation(self, mappoint_id: int) -> np.ndarray | None:
        """Get the 2D observation (pixel coords) of a map point.

        Args:
            mappoint_id: ID of the map point

        Returns:
            2D pixel coordinates or None if not observed
        """
        for kp_idx, mp_id in self.keypoint_to_mappoint.items():
            if mp_id == mappoint_id:
                return self.keypoints[kp_idx]
        return None

    @property
    def num_observations(self) -> int:
        """Return number of map point observations."""
        return len(self.keypoint_to_mappoint)


class KeyframeManager:
    """Manages keyframe selection and storage.

    Decides when a new keyframe should be created based on:
    1. Sufficient parallax (translation or rotation) from last keyframe
    2. Minimum frames since last keyframe
    3. Number of tracked features dropping below threshold
    """

    def __init__(
        self,
        min_translation: float = 0.1,  # meters
        min_rotation: float = 5.0,  # degrees
        min_frames_between: int = 5,
        min_tracked_ratio: float = 0.5,
    ) -> None:
        """Initialize keyframe manager.

        Args:
            min_translation: Minimum translation (m) to trigger new keyframe
            min_rotation: Minimum rotation (degrees) to trigger new keyframe
            min_frames_between: Minimum frames between keyframes
            min_tracked_ratio: If tracked points drop below this ratio of
                               last keyframe's points, create new keyframe
        """
        self._min_translation = min_translation
        self._min_rotation_rad = np.deg2rad(min_rotation)
        self._min_frames_between = min_frames_between
        self._min_tracked_ratio = min_tracked_ratio

        # State
        self._keyframes: dict[int, Keyframe] = {}  # id -> Keyframe
        self._keyframe_ids: list[int] = []  # Ordered list of keyframe IDs
        self._last_keyframe: Keyframe | None = None

    def should_create_keyframe(
        self,
        vo_frame: VOFrame,
        num_tracked_from_last_kf: int | None = None,
    ) -> bool:
        """Determine if a new keyframe should be created.

        Args:
            vo_frame: Current visual odometry frame
            num_tracked_from_last_kf: Number of features tracked from last keyframe

        Returns:
            True if a new keyframe should be created
        """
        # First frame is always a keyframe
        if self._last_keyframe is None:
            return True

        # Check minimum frame gap
        frame_gap = vo_frame.frame_id - self._last_keyframe.id
        if frame_gap < self._min_frames_between:
            return False

        # Check parallax (relative pose from last keyframe)
        delta_pose = self._last_keyframe.pose.inverse().compose(vo_frame.pose)

        # Translation check
        translation_dist = np.linalg.norm(delta_pose.translation)
        if translation_dist > self._min_translation:
            return True

        # Rotation check
        rotation_angle = _rotation_angle_from_matrix(delta_pose.rotation)
        if rotation_angle > self._min_rotation_rad:
            return True

        # Tracking quality check (if provided)
        if num_tracked_from_last_kf is not None:
            last_kf_points = self._last_keyframe.num_observations
            if last_kf_points > 0:
                tracked_ratio = num_tracked_from_last_kf / last_kf_points
                if tracked_ratio < self._min_tracked_ratio:
                    return True

        return False

    def add_keyframe(self, keyframe: Keyframe) -> None:
        """Add a keyframe to the manager.

        Args:
            keyframe: Keyframe to add
        """
        self._keyframes[keyframe.id] = keyframe
        self._keyframe_ids.append(keyframe.id)
        self._last_keyframe = keyframe

    def get_keyframe(self, kf_id: int) -> Keyframe | None:
        """Get a keyframe by ID.

        Args:
            kf_id: Keyframe ID

        Returns:
            Keyframe or None if not found
        """
        return self._keyframes.get(kf_id)

    def get_recent_keyframes(self, n: int) -> list[Keyframe]:
        """Get the N most recent keyframes.

        Args:
            n: Number of keyframes to return

        Returns:
            List of keyframes (most recent last)
        """
        recent_ids = self._keyframe_ids[-n:]
        return [self._keyframes[kf_id] for kf_id in recent_ids]

    def get_all_keyframes(self) -> list[Keyframe]:
        """Get all keyframes in order.

        Returns:
            List of all keyframes (oldest first)
        """
        return [self._keyframes[kf_id] for kf_id in self._keyframe_ids]

    @property
    def last_keyframe(self) -> Keyframe | None:
        """Return the most recent keyframe."""
        return self._last_keyframe

    @property
    def num_keyframes(self) -> int:
        """Return total number of keyframes."""
        return len(self._keyframes)

    def update_keyframe_pose(self, kf_id: int, new_pose: SE3) -> bool:
        """Update the pose of a keyframe (after bundle adjustment).

        Args:
            kf_id: Keyframe ID
            new_pose: New optimized pose

        Returns:
            True if keyframe was found and updated
        """
        kf = self._keyframes.get(kf_id)
        if kf is None:
            return False

        # Create new keyframe with updated pose (dataclass is immutable-ish)
        self._keyframes[kf_id] = Keyframe(
            id=kf.id,
            timestamp_ns=kf.timestamp_ns,
            pose=new_pose,
            keypoints=kf.keypoints,
            descriptors=kf.descriptors,
            keypoint_to_mappoint=kf.keypoint_to_mappoint,
        )

        # Update last keyframe reference if needed
        if self._last_keyframe is not None and self._last_keyframe.id == kf_id:
            self._last_keyframe = self._keyframes[kf_id]

        return True
