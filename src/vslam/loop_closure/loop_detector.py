"""Loop closure detector combining place recognition and geometric verification.

This module orchestrates the full loop closure pipeline:
1. Place Recognition: Find visually similar keyframes using BoW
2. Geometric Verification: Confirm candidates using PnP + RANSAC
3. Pose Graph: Add loop edges and optimize globally

The loop detector maintains its own copy of keyframe data for the
loop closure process, separate from the frontend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..frontend import SE3
from .geometric_verification import GeometricVerifier, VerificationResult
from .place_recognition import PlaceDatabase, PlaceEntry
from .pose_graph import PoseGraph
from .vocabulary import VisualVocabulary


@dataclass
class LoopClosureResult:
    """Result of loop closure detection.

    Attributes:
        detected: Whether a loop was detected
        query_keyframe_id: ID of query keyframe
        match_keyframe_id: ID of matched keyframe (if detected)
        relative_pose: Relative pose from query to match (if detected)
        similarity_score: BoW similarity score
        num_inliers: PnP inlier count
    """

    detected: bool = False
    query_keyframe_id: int = -1
    match_keyframe_id: int = -1
    relative_pose: SE3 | None = None
    similarity_score: float = 0.0
    num_inliers: int = 0


@dataclass
class KeyframeData:
    """Data needed for loop closure from a keyframe.

    Attributes:
        id: Keyframe ID
        pose: Camera pose in world frame
        descriptors: ORB descriptors (N, 32)
        keypoints: 2D keypoint locations (N, 2)
        points_3d: Associated 3D map points (M, 3)
        keypoint_to_3d: Mapping from keypoint index to 3D point index
    """

    id: int
    pose: SE3
    descriptors: np.ndarray
    keypoints: np.ndarray
    points_3d: np.ndarray
    keypoint_to_3d: dict[int, int]


class LoopDetector:
    """Detects loop closures and manages pose graph optimization.

    Combines place recognition (visual similarity) with geometric
    verification (spatial consistency) to reliably detect loops.
    """

    def __init__(
        self,
        vocabulary: VisualVocabulary,
        camera_matrix: np.ndarray,
        min_score: float = 0.5,
        min_keyframe_gap: int = 30,
        n_candidates: int = 3,
        min_inliers: int = 50,
    ) -> None:
        """Initialize loop detector.

        Args:
            vocabulary: Visual vocabulary for BoW
            camera_matrix: 3x3 camera intrinsics
            min_score: Minimum BoW similarity for candidates (increased for robustness)
            min_keyframe_gap: Minimum keyframe ID gap (skip recent frames)
            n_candidates: Number of candidates to verify per query
            min_inliers: Minimum PnP inliers for valid loop (increased for robustness)
        """
        self._vocabulary = vocabulary
        self._camera_matrix = camera_matrix
        self._min_score = min_score
        self._min_keyframe_gap = min_keyframe_gap
        self._n_candidates = n_candidates

        # Components
        self._place_db = PlaceDatabase(vocabulary)
        self._verifier = GeometricVerifier(
            camera_matrix=camera_matrix,
            min_inliers=min_inliers,
        )
        self._pose_graph = PoseGraph()

        # Keyframe storage for geometric verification
        self._keyframes: dict[int, KeyframeData] = {}

        # Previous keyframe ID for odometry edges
        self._prev_keyframe_id: int | None = None

    @classmethod
    def from_vocabulary_path(
        cls,
        vocabulary_path: str | Path,
        camera_matrix: np.ndarray,
        **kwargs,
    ) -> LoopDetector:
        """Create loop detector from vocabulary file.

        Args:
            vocabulary_path: Path to vocabulary .npz file
            camera_matrix: 3x3 camera intrinsics
            **kwargs: Additional arguments passed to __init__

        Returns:
            Configured LoopDetector
        """
        vocabulary = VisualVocabulary.load(vocabulary_path)
        return cls(vocabulary=vocabulary, camera_matrix=camera_matrix, **kwargs)

    def add_keyframe(self, keyframe: KeyframeData) -> LoopClosureResult:
        """Add a keyframe and check for loop closure.

        Args:
            keyframe: Keyframe data

        Returns:
            LoopClosureResult indicating if loop was detected
        """
        # Store keyframe
        self._keyframes[keyframe.id] = keyframe

        # Add to pose graph
        self._pose_graph.add_keyframe(keyframe.id, keyframe.pose)

        # Add odometry edge from previous keyframe
        if self._prev_keyframe_id is not None:
            prev_pose = self._keyframes[self._prev_keyframe_id].pose
            relative_pose = prev_pose.inverse().compose(keyframe.pose)
            self._pose_graph.add_odometry_edge(
                self._prev_keyframe_id,
                keyframe.id,
                relative_pose,
            )
        self._prev_keyframe_id = keyframe.id

        # Query for loop candidates BEFORE adding to database
        result = self._detect_loop(keyframe)

        # Add to place database (for future queries)
        self._place_db.add(
            keyframe_id=keyframe.id,
            descriptors=keyframe.descriptors,
            keypoints=keyframe.keypoints,
        )

        return result

    def _detect_loop(self, query_kf: KeyframeData) -> LoopClosureResult:
        """Detect loop closure for a query keyframe.

        Args:
            query_kf: Query keyframe data

        Returns:
            LoopClosureResult
        """
        # Skip if database is too small
        if self._place_db.size < self._min_keyframe_gap:
            return LoopClosureResult(
                detected=False,
                query_keyframe_id=query_kf.id,
            )

        # Query place database
        candidates = self._place_db.query(
            descriptors=query_kf.descriptors,
            n_candidates=self._n_candidates,
            min_score=self._min_score,
            exclude_recent=self._min_keyframe_gap,
            query_keyframe_id=query_kf.id,
        )

        if not candidates:
            return LoopClosureResult(
                detected=False,
                query_keyframe_id=query_kf.id,
            )

        # Try geometric verification on each candidate
        for candidate in candidates:
            match_kf = self._keyframes.get(candidate.keyframe_id)
            if match_kf is None:
                continue

            # Get match keyframe's place entry for descriptors
            match_entry = self._place_db.get_entry(candidate.keyframe_id)
            if match_entry is None:
                continue

            # Geometric verification
            verification = self._verifier.verify(
                query_descriptors=query_kf.descriptors,
                query_keypoints=query_kf.keypoints,
                match_descriptors=match_entry.descriptors,
                match_keypoints=match_entry.keypoints,
                match_points_3d=match_kf.points_3d,
                match_keypoint_to_3d=match_kf.keypoint_to_3d,
            )

            if verification.is_valid:
                # Valid loop closure found!
                # Add loop edge to pose graph
                self._pose_graph.add_loop_edge(
                    from_id=query_kf.id,
                    to_id=candidate.keyframe_id,
                    relative_pose=verification.relative_pose,
                )

                return LoopClosureResult(
                    detected=True,
                    query_keyframe_id=query_kf.id,
                    match_keyframe_id=candidate.keyframe_id,
                    relative_pose=verification.relative_pose,
                    similarity_score=candidate.similarity,
                    num_inliers=verification.num_inliers,
                )

        return LoopClosureResult(
            detected=False,
            query_keyframe_id=query_kf.id,
        )

    def optimize(self) -> dict[int, SE3]:
        """Run pose graph optimization.

        Returns:
            Dictionary of optimized poses (keyframe_id -> SE3)
        """
        return self._pose_graph.optimize()

    def get_optimized_poses(self) -> dict[int, SE3]:
        """Get current poses from pose graph.

        Returns:
            Dictionary of poses (keyframe_id -> SE3)
        """
        return self._pose_graph.poses

    @property
    def num_keyframes(self) -> int:
        """Number of keyframes in detector."""
        return len(self._keyframes)

    @property
    def num_loop_closures(self) -> int:
        """Number of detected loop closures."""
        return self._pose_graph.num_loop_edges
