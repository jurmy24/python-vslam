"""Temporal feature matching between consecutive frames."""

from dataclasses import dataclass

import cv2
import numpy as np

from .feature_detector import Features


@dataclass
class TemporalMatches:
    """Container for temporal feature matches between consecutive frames.

    Attributes:
        prev_indices: Indices into previous frame's keypoints array
        curr_indices: Indices into current frame's keypoints array
        distances: Hamming distances between matched descriptors
    """

    prev_indices: np.ndarray  # (N,) int
    curr_indices: np.ndarray  # (N,) int
    distances: np.ndarray  # (N,) float32

    def __len__(self) -> int:
        """Return number of matches."""
        return len(self.prev_indices)

    def filter_by_distance(self, max_distance: int) -> "TemporalMatches":
        """Return new TemporalMatches with distance filtering applied.

        Args:
            max_distance: Maximum Hamming distance to keep

        Returns:
            New TemporalMatches with only matches below threshold
        """
        mask = self.distances <= max_distance
        return TemporalMatches(
            prev_indices=self.prev_indices[mask],
            curr_indices=self.curr_indices[mask],
            distances=self.distances[mask],
        )


class TemporalMatcher:
    """Matches features between consecutive frames for tracking.

    Uses brute-force Hamming matching with Lowe's ratio test for robustness.
    The ratio test rejects ambiguous matches where the best match distance
    is similar to the second-best match distance.
    """

    def __init__(
        self,
        ratio_threshold: float = 0.75,
        max_hamming_distance: int = 50,
    ) -> None:
        """Initialize temporal matcher.

        Args:
            ratio_threshold: Lowe's ratio test threshold. A match is accepted
                only if best_distance < ratio * second_best_distance.
                Lower values = stricter matching. Typical range: 0.7-0.8
            max_hamming_distance: Maximum Hamming distance for valid match.
                ORB descriptors are 256 bits, so max possible is 256.
        """
        # Use knnMatch with k=2 for ratio test (can't use crossCheck with knn)
        self._bf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self._ratio_threshold = ratio_threshold
        self._max_distance = max_hamming_distance

    def match(
        self, features_prev: Features, features_curr: Features
    ) -> TemporalMatches:
        """Match features from previous frame to current frame.

        Uses Lowe's ratio test: a match is accepted if the best match
        distance is significantly better than the second best. This filters
        out ambiguous matches that could be incorrect.

        Args:
            features_prev: Features from previous frame
            features_curr: Features from current frame

        Returns:
            TemporalMatches containing filtered correspondences
        """
        # Handle empty feature sets
        if (
            len(features_prev) == 0
            or len(features_curr) == 0
            or features_prev.descriptors is None
            or features_curr.descriptors is None
        ):
            return TemporalMatches(
                prev_indices=np.empty(0, dtype=np.int32),
                curr_indices=np.empty(0, dtype=np.int32),
                distances=np.empty(0, dtype=np.float32),
            )

        # Perform k-nearest neighbor matching (k=2 for ratio test)
        knn_matches = self._bf_matcher.knnMatch(
            features_prev.descriptors,
            features_curr.descriptors,
            k=2,
        )

        # Apply ratio test and distance threshold
        prev_indices = []
        curr_indices = []
        distances = []

        for match_pair in knn_matches:
            # Need at least 2 matches for ratio test
            if len(match_pair) < 2:
                # Only one match found - accept if distance is good
                if len(match_pair) == 1 and match_pair[0].distance <= self._max_distance:
                    m = match_pair[0]
                    prev_indices.append(m.queryIdx)
                    curr_indices.append(m.trainIdx)
                    distances.append(m.distance)
                continue

            best, second_best = match_pair[0], match_pair[1]

            # Ratio test: best must be significantly better than second best
            if best.distance > self._ratio_threshold * second_best.distance:
                continue

            # Distance threshold
            if best.distance > self._max_distance:
                continue

            prev_indices.append(best.queryIdx)
            curr_indices.append(best.trainIdx)
            distances.append(best.distance)

        # Convert to arrays
        if len(prev_indices) == 0:
            return TemporalMatches(
                prev_indices=np.empty(0, dtype=np.int32),
                curr_indices=np.empty(0, dtype=np.int32),
                distances=np.empty(0, dtype=np.float32),
            )

        return TemporalMatches(
            prev_indices=np.array(prev_indices, dtype=np.int32),
            curr_indices=np.array(curr_indices, dtype=np.int32),
            distances=np.array(distances, dtype=np.float32),
        )

    @property
    def ratio_threshold(self) -> float:
        """Return the ratio test threshold."""
        return self._ratio_threshold

    @property
    def max_hamming_distance(self) -> int:
        """Return the maximum Hamming distance threshold."""
        return self._max_distance
