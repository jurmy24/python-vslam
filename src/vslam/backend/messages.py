"""Inter-process message types for frontend-backend communication.

These dataclasses define the message protocol between the frontend (VisualOdometry)
and backend (LocalMapping/BundleAdjustment) processes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..frontend.pose import SE3


@dataclass
class KeyframeData:
    """Serializable keyframe data for inter-process transfer.

    Note: We use primitive arrays instead of complex objects to ensure
    clean serialization across process boundaries.
    """

    id: int
    timestamp_ns: int
    # Pose as separate arrays (avoids serializing SE3 object)
    rotation: np.ndarray  # (3, 3) rotation matrix
    translation: np.ndarray  # (3,) translation vector
    # Feature data
    keypoints: np.ndarray  # (N, 2) keypoint coordinates
    descriptors: np.ndarray  # (N, 32) ORB descriptors
    # Map point associations: keypoint_idx -> mappoint_id
    keypoint_to_mappoint: dict[int, int] = field(default_factory=dict)


@dataclass
class MapPointData:
    """Serializable map point data for inter-process transfer."""

    id: int
    position: np.ndarray  # (3,) world position
    descriptor: np.ndarray  # (32,) ORB descriptor


@dataclass
class KeyframeMessage:
    """Frontend -> Backend: New keyframe to process.

    Sent when the frontend decides a frame should become a keyframe.
    The backend will add this to its local window and run bundle adjustment.
    """

    keyframe: KeyframeData
    new_map_points: list[MapPointData] = field(default_factory=list)


@dataclass
class CorrectionMessage:
    """Backend -> Frontend: Pose and point corrections from bundle adjustment.

    After BA optimizes poses and 3D points, corrections are sent back
    to the frontend to update its state.
    """

    # keyframe_id -> (rotation, translation)
    pose_corrections: dict[int, tuple[np.ndarray, np.ndarray]] = field(
        default_factory=dict
    )
    # mappoint_id -> new_position
    point_corrections: dict[int, np.ndarray] = field(default_factory=dict)


@dataclass
class ShutdownMessage:
    """Signal to stop the backend process gracefully."""

    pass


# Type alias for any message that can be sent between processes
BackendMessage = KeyframeMessage | ShutdownMessage
FrontendMessage = CorrectionMessage
