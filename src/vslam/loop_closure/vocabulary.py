"""Visual vocabulary for Bag of Visual Words place recognition.

A visual vocabulary enables fast image similarity comparison by:
1. Clustering descriptors into "visual words" (k-means centers)
2. Representing images as histograms of visual word occurrences
3. Comparing images via histogram similarity (cosine distance)

The vocabulary is trained offline on a large dataset of images,
then used at runtime to quickly describe and compare keyframes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class VisualVocabulary:
    """Bag of Visual Words vocabulary for ORB descriptors.

    Attributes:
        words: Cluster centers (visual words), shape (n_words, 32)
        n_words: Number of visual words in vocabulary
        idf: Inverse document frequency weights, shape (n_words,)
    """

    words: np.ndarray  # (n_words, 32) float32 cluster centers
    n_words: int
    idf: np.ndarray  # (n_words,) IDF weights

    def describe(self, descriptors: np.ndarray) -> np.ndarray:
        """Convert image descriptors to Bag of Words vector.

        Args:
            descriptors: ORB descriptors, shape (N, 32) uint8

        Returns:
            BoW vector, shape (n_words,), L2 normalized with TF-IDF weighting
        """
        if descriptors is None or len(descriptors) == 0:
            return np.zeros(self.n_words, dtype=np.float32)

        # Assign each descriptor to nearest visual word
        # For ORB (binary descriptors), we use Hamming distance
        # But k-means uses Euclidean, so we treat descriptors as float vectors
        descriptors_float = descriptors.astype(np.float32)

        # Compute distances to all words: (N, n_words)
        # Using broadcasting: (N, 1, 32) - (1, n_words, 32) -> (N, n_words, 32)
        # Then sum squared differences
        diff = descriptors_float[:, np.newaxis, :] - self.words[np.newaxis, :, :]
        distances = np.sum(diff**2, axis=2)  # (N, n_words)

        # Find nearest word for each descriptor
        word_indices = np.argmin(distances, axis=1)  # (N,)

        # Build histogram (term frequency)
        histogram = np.bincount(word_indices, minlength=self.n_words).astype(np.float32)

        # Apply TF-IDF weighting
        # TF = term frequency (already have this)
        # IDF = inverse document frequency (pre-computed)
        tfidf = histogram * self.idf

        # L2 normalize for cosine similarity
        norm = np.linalg.norm(tfidf)
        if norm > 0:
            tfidf = tfidf / norm

        return tfidf

    def similarity(self, bow1: np.ndarray, bow2: np.ndarray) -> float:
        """Compute cosine similarity between two BoW vectors.

        Args:
            bow1: First BoW vector (L2 normalized)
            bow2: Second BoW vector (L2 normalized)

        Returns:
            Cosine similarity in [0, 1]
        """
        return float(np.dot(bow1, bow2))

    def save(self, path: str | Path) -> None:
        """Save vocabulary to .npz file.

        Args:
            path: Output file path
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            words=self.words,
            n_words=self.n_words,
            idf=self.idf,
        )

    @classmethod
    def load(cls, path: str | Path) -> VisualVocabulary:
        """Load vocabulary from .npz file.

        Args:
            path: Input file path

        Returns:
            Loaded vocabulary
        """
        data = np.load(path)
        return cls(
            words=data["words"],
            n_words=int(data["n_words"]),
            idf=data["idf"],
        )

    @classmethod
    def from_words(cls, words: np.ndarray) -> VisualVocabulary:
        """Create vocabulary from cluster centers with uniform IDF.

        Args:
            words: Cluster centers, shape (n_words, 32)

        Returns:
            Vocabulary with uniform IDF weights (to be updated later)
        """
        n_words = len(words)
        return cls(
            words=words.astype(np.float32),
            n_words=n_words,
            idf=np.ones(n_words, dtype=np.float32),
        )

    def update_idf(self, document_frequencies: np.ndarray, n_documents: int) -> None:
        """Update IDF weights based on document frequencies.

        IDF(word) = log(N / df(word))

        where N is total documents and df(word) is documents containing word.

        Args:
            document_frequencies: Count of documents containing each word, shape (n_words,)
            n_documents: Total number of documents
        """
        # Avoid division by zero with smoothing
        df_smoothed = np.maximum(document_frequencies, 1)
        self.idf = np.log(n_documents / df_smoothed).astype(np.float32)
