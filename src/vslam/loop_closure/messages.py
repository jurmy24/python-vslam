"""Message types for loop closure inter-process communication.

These messages are sent between the local BA process and the
loop closure process, and from loop closure back to the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LoopKeyframeData:
    """Keyframe data for loop closure detection.

    Serializable data that can be sent between processes.
    Uses primitive arrays instead of custom objects.

    Attributes:
        id: Keyframe ID
        pose_rotation: 3x3 rotation matrix
        pose_translation: 3x1 translation vector
        descriptors: ORB descriptors (N, 32)
        keypoints: 2D keypoint locations (N, 2)
        points_3d: Associated 3D map points (M, 3)
        keypoint_to_3d: Mapping from keypoint index to 3D point index
    """

    id: int
    pose_rotation: np.ndarray  # (3, 3)
    pose_translation: np.ndarray  # (3,)
    descriptors: np.ndarray  # (N, 32)
    keypoints: np.ndarray  # (N, 2)
    points_3d: np.ndarray  # (M, 3)
    keypoint_to_3d: dict[int, int]


@dataclass
class LoopKeyframeMessage:
    """Message containing keyframe data for loop closure.

    Sent from LocalBA process to LoopClosure process.
    """

    keyframe: LoopKeyframeData


@dataclass
class GlobalCorrectionMessage:
    """Global pose corrections from loop closure optimization.

    Sent from LoopClosure process back to frontend.
    Contains optimized poses for ALL keyframes after pose graph optimization.

    Attributes:
        pose_corrections: Mapping from keyframe ID to optimized pose (4x4 matrix)
        loop_from_id: Query keyframe ID of detected loop (for visualization)
        loop_to_id: Match keyframe ID of detected loop (for visualization)
    """

    pose_corrections: dict[int, np.ndarray]  # kf_id -> 4x4 homogeneous matrix
    loop_from_id: int = -1
    loop_to_id: int = -1


@dataclass
class LoopClosureShutdownMessage:
    """Signal to stop the loop closure process."""

    pass
