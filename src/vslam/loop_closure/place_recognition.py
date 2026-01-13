"""Place recognition database for loop closure detection.

This module stores keyframe BoW representations and enables fast
similarity queries to find loop closure candidates.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .vocabulary import VisualVocabulary


@dataclass
class PlaceEntry:
    """An entry in the place recognition database.

    Attributes:
        keyframe_id: ID of the keyframe
        bow_vector: Bag of Words vector (normalized, TF-IDF weighted)
        descriptors: Original ORB descriptors for geometric verification
        keypoints: 2D keypoint locations, shape (N, 2)
    """

    keyframe_id: int
    bow_vector: np.ndarray  # (n_words,) float32
    descriptors: np.ndarray  # (N, 32) uint8
    keypoints: np.ndarray  # (N, 2) float32


@dataclass
class QueryResult:
    """Result from a place recognition query.

    Attributes:
        keyframe_id: ID of the matching keyframe
        similarity: Cosine similarity score [0, 1]
    """

    keyframe_id: int
    similarity: float


class PlaceDatabase:
    """Database of visited places for loop closure queries.

    Stores BoW representations of keyframes and enables fast
    similarity queries using cosine distance.
    """

    def __init__(self, vocabulary: VisualVocabulary) -> None:
        """Initialize place database.

        Args:
            vocabulary: Visual vocabulary for BoW conversion
        """
        self._vocabulary = vocabulary
        self._entries: dict[int, PlaceEntry] = {}

        # BoW matrix for fast batch queries: (n_entries, n_words)
        self._bow_matrix: np.ndarray | None = None
        self._keyframe_ids: list[int] = []

        # For IDF updates
        self._document_frequencies: np.ndarray = np.zeros(
            vocabulary.n_words, dtype=np.int32
        )

    def add(
        self,
        keyframe_id: int,
        descriptors: np.ndarray,
        keypoints: np.ndarray,
    ) -> PlaceEntry:
        """Add a keyframe to the database.

        Args:
            keyframe_id: Unique keyframe ID
            descriptors: ORB descriptors, shape (N, 32)
            keypoints: 2D keypoint locations, shape (N, 2)

        Returns:
            The created PlaceEntry
        """
        # Compute BoW vector
        bow_vector = self._vocabulary.describe(descriptors)

        # Create entry
        entry = PlaceEntry(
            keyframe_id=keyframe_id,
            bow_vector=bow_vector,
            descriptors=descriptors.copy(),
            keypoints=keypoints.copy(),
        )

        self._entries[keyframe_id] = entry
        self._keyframe_ids.append(keyframe_id)

        # Update BoW matrix for fast queries
        if self._bow_matrix is None:
            self._bow_matrix = bow_vector.reshape(1, -1)
        else:
            self._bow_matrix = np.vstack([self._bow_matrix, bow_vector])

        # Update document frequencies for IDF
        word_indices = np.where(bow_vector > 0)[0]
        self._document_frequencies[word_indices] += 1

        return entry

    def query(
        self,
        descriptors: np.ndarray,
        n_candidates: int = 5,
        min_score: float = 0.0,
        exclude_recent: int = 0,
        query_keyframe_id: int | None = None,
    ) -> list[QueryResult]:
        """Query database for similar places.

        Args:
            descriptors: Query ORB descriptors, shape (N, 32)
            n_candidates: Maximum number of candidates to return
            min_score: Minimum similarity score threshold
            exclude_recent: Exclude keyframes within this ID gap of query
            query_keyframe_id: ID of query keyframe (for exclusion)

        Returns:
            List of QueryResults sorted by similarity (highest first)
        """
        if self._bow_matrix is None or len(self._bow_matrix) == 0:
            return []

        # Compute query BoW
        query_bow = self._vocabulary.describe(descriptors)

        # Compute cosine similarities (dot product since vectors are normalized)
        similarities = self._bow_matrix @ query_bow  # (n_entries,)

        # Create mask for exclusion
        mask = np.ones(len(self._keyframe_ids), dtype=bool)

        if query_keyframe_id is not None and exclude_recent > 0:
            for i, kf_id in enumerate(self._keyframe_ids):
                if abs(kf_id - query_keyframe_id) < exclude_recent:
                    mask[i] = False

        # Apply minimum score threshold
        mask &= similarities >= min_score

        # Get valid indices and sort by similarity
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return []

        valid_similarities = similarities[valid_indices]
        sorted_order = np.argsort(valid_similarities)[::-1]

        # Return top N candidates
        results = []
        for i in sorted_order[:n_candidates]:
            idx = valid_indices[i]
            results.append(
                QueryResult(
                    keyframe_id=self._keyframe_ids[idx],
                    similarity=float(similarities[idx]),
                )
            )

        return results

    def get_entry(self, keyframe_id: int) -> PlaceEntry | None:
        """Get a place entry by keyframe ID.

        Args:
            keyframe_id: Keyframe ID to look up

        Returns:
            PlaceEntry if found, None otherwise
        """
        return self._entries.get(keyframe_id)

    def update_idf(self) -> None:
        """Update IDF weights based on current document frequencies.

        Call this periodically as the database grows for better
        discrimination between places.
        """
        n_documents = len(self._entries)
        if n_documents > 0:
            self._vocabulary.update_idf(self._document_frequencies, n_documents)

            # Recompute all BoW vectors with new IDF
            for i, kf_id in enumerate(self._keyframe_ids):
                entry = self._entries[kf_id]
                entry.bow_vector = self._vocabulary.describe(entry.descriptors)
                self._bow_matrix[i] = entry.bow_vector

    @property
    def size(self) -> int:
        """Return number of entries in database."""
        return len(self._entries)

    def __len__(self) -> int:
        return len(self._entries)
