"""Covisibility graph for tracking shared observations between keyframes.

The covisibility graph is a weighted undirected graph where:
- Nodes are keyframes
- Edges connect keyframes that share observations of the same map points
- Edge weights represent the number of shared map points

This structure enables efficient queries like "which keyframes should be
optimized together in local bundle adjustment?"
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .keyframe import Keyframe


@dataclass
class CovisibilityEdge:
    """Edge in the covisibility graph."""

    kf1_id: int
    kf2_id: int
    weight: int  # Number of shared map points

    @property
    def other(self) -> tuple[int, int]:
        """Return both keyframe IDs."""
        return (self.kf1_id, self.kf2_id)


class CovisibilityGraph:
    """Graph tracking which keyframes share map point observations.

    Used to determine which keyframes should be jointly optimized
    in local bundle adjustment.
    """

    def __init__(self, min_shared_points: int = 15) -> None:
        """Initialize covisibility graph.

        Args:
            min_shared_points: Minimum shared points to create an edge
        """
        self._min_shared = min_shared_points

        # Adjacency list: kf_id -> {other_kf_id: weight}
        self._adjacency: dict[int, dict[int, int]] = defaultdict(dict)

        # Inverted index: mappoint_id -> set of kf_ids observing it
        self._mappoint_to_keyframes: dict[int, set[int]] = defaultdict(set)

        # Keyframe data: kf_id -> set of observed mappoint_ids
        self._keyframe_observations: dict[int, set[int]] = {}

    def add_keyframe(self, keyframe: Keyframe) -> None:
        """Add a keyframe to the covisibility graph.

        Updates edges to all existing keyframes that share map points.

        Args:
            keyframe: Keyframe to add
        """
        kf_id = keyframe.id
        observed_mappoints = set(keyframe.keypoint_to_mappoint.values())

        # Store this keyframe's observations
        self._keyframe_observations[kf_id] = observed_mappoints

        # Find all keyframes that share observations
        shared_counts: dict[int, int] = defaultdict(int)

        for mp_id in observed_mappoints:
            # Count shared points with existing keyframes
            for other_kf_id in self._mappoint_to_keyframes[mp_id]:
                if other_kf_id != kf_id:
                    shared_counts[other_kf_id] += 1

            # Add this keyframe to the inverted index
            self._mappoint_to_keyframes[mp_id].add(kf_id)

        # Create edges to keyframes with sufficient shared points
        for other_kf_id, count in shared_counts.items():
            if count >= self._min_shared:
                self._adjacency[kf_id][other_kf_id] = count
                self._adjacency[other_kf_id][kf_id] = count

    def update_keyframe(self, keyframe: Keyframe) -> None:
        """Update keyframe observations (e.g., after adding new associations).

        Args:
            keyframe: Keyframe with potentially updated observations
        """
        kf_id = keyframe.id

        # Get old and new observations
        old_observations = self._keyframe_observations.get(kf_id, set())
        new_observations = set(keyframe.keypoint_to_mappoint.values())

        # Find added map points
        added = new_observations - old_observations

        # Update inverted index for new observations
        for mp_id in added:
            self._mappoint_to_keyframes[mp_id].add(kf_id)

        # Update stored observations
        self._keyframe_observations[kf_id] = new_observations

        # Recompute edges (simple approach - recalculate all)
        self._recompute_edges_for_keyframe(kf_id)

    def _recompute_edges_for_keyframe(self, kf_id: int) -> None:
        """Recompute all edges for a specific keyframe."""
        # Remove existing edges
        if kf_id in self._adjacency:
            for other_kf_id in list(self._adjacency[kf_id].keys()):
                if other_kf_id in self._adjacency:
                    self._adjacency[other_kf_id].pop(kf_id, None)
            self._adjacency[kf_id].clear()

        # Recalculate shared counts
        observed = self._keyframe_observations.get(kf_id, set())
        shared_counts: dict[int, int] = defaultdict(int)

        for mp_id in observed:
            for other_kf_id in self._mappoint_to_keyframes[mp_id]:
                if other_kf_id != kf_id:
                    shared_counts[other_kf_id] += 1

        # Create new edges
        for other_kf_id, count in shared_counts.items():
            if count >= self._min_shared:
                self._adjacency[kf_id][other_kf_id] = count
                self._adjacency[other_kf_id][kf_id] = count

    def get_connected_keyframes(
        self,
        kf_id: int,
        min_shared: int | None = None,
    ) -> list[tuple[int, int]]:
        """Get keyframes connected to a given keyframe.

        Args:
            kf_id: Keyframe ID
            min_shared: Minimum shared points (uses default if None)

        Returns:
            List of (kf_id, weight) tuples, sorted by weight descending
        """
        min_shared = min_shared if min_shared is not None else self._min_shared

        connections = [
            (other_id, weight)
            for other_id, weight in self._adjacency.get(kf_id, {}).items()
            if weight >= min_shared
        ]

        # Sort by weight descending
        return sorted(connections, key=lambda x: x[1], reverse=True)

    def get_local_keyframes(
        self,
        kf_id: int,
        n: int = 10,
    ) -> list[int]:
        """Get the N most connected keyframes (for local BA window).

        Args:
            kf_id: Reference keyframe ID
            n: Maximum number of keyframes to return

        Returns:
            List of keyframe IDs (most connected first), including kf_id
        """
        connected = self.get_connected_keyframes(kf_id, min_shared=1)
        local_ids = [kf_id]  # Include the reference keyframe

        for other_id, _ in connected[:n - 1]:
            local_ids.append(other_id)

        return local_ids

    def get_shared_mappoints(self, kf1_id: int, kf2_id: int) -> set[int]:
        """Get map points observed by both keyframes.

        Args:
            kf1_id: First keyframe ID
            kf2_id: Second keyframe ID

        Returns:
            Set of shared map point IDs
        """
        obs1 = self._keyframe_observations.get(kf1_id, set())
        obs2 = self._keyframe_observations.get(kf2_id, set())
        return obs1 & obs2

    def get_keyframes_observing(self, mappoint_id: int) -> set[int]:
        """Get all keyframes observing a map point.

        Args:
            mappoint_id: Map point ID

        Returns:
            Set of keyframe IDs
        """
        return self._mappoint_to_keyframes.get(mappoint_id, set()).copy()

    def remove_keyframe(self, kf_id: int) -> None:
        """Remove a keyframe from the graph.

        Args:
            kf_id: Keyframe ID to remove
        """
        # Remove from adjacency
        if kf_id in self._adjacency:
            for other_kf_id in self._adjacency[kf_id]:
                if other_kf_id in self._adjacency:
                    self._adjacency[other_kf_id].pop(kf_id, None)
            del self._adjacency[kf_id]

        # Remove from inverted index
        observed = self._keyframe_observations.pop(kf_id, set())
        for mp_id in observed:
            self._mappoint_to_keyframes[mp_id].discard(kf_id)

    def get_covisibility_weight(self, kf1_id: int, kf2_id: int) -> int:
        """Get the covisibility weight between two keyframes.

        Args:
            kf1_id: First keyframe ID
            kf2_id: Second keyframe ID

        Returns:
            Number of shared map points (0 if not connected)
        """
        return self._adjacency.get(kf1_id, {}).get(kf2_id, 0)

    @property
    def num_keyframes(self) -> int:
        """Return number of keyframes in the graph."""
        return len(self._keyframe_observations)

    @property
    def num_edges(self) -> int:
        """Return number of edges in the graph."""
        # Each edge is counted twice in adjacency list
        return sum(len(adj) for adj in self._adjacency.values()) // 2
