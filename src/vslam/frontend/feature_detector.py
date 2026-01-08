"""ORB feature detection for visual SLAM."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Features:
    """Container for detected image features.

    Attributes:
        keypoints: Tuple of OpenCV KeyPoint objects
        descriptors: Nx32 array of ORB binary descriptors (uint8), or None if no features
    """

    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray | None

    @property
    def points(self) -> np.ndarray:
        """Return Nx2 array of keypoint (x, y) coordinates."""
        if len(self.keypoints) == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.array([kp.pt for kp in self.keypoints], dtype=np.float32)

    def __len__(self) -> int:
        """Return number of detected features."""
        return len(self.keypoints)


class FeatureDetector:
    """ORB feature detector for sparse feature extraction.

    ORB (Oriented FAST and Rotated BRIEF) is a fast, rotation-invariant
    feature detector that produces binary descriptors suitable for
    real-time SLAM applications.
    """

    def __init__(
        self,
        n_features: int = 1000,
        scale_factor: float = 1.2,
        n_levels: int = 8,
        edge_threshold: int = 31,
        fast_threshold: int = 20,
    ) -> None:
        """Initialize ORB detector with configurable parameters.

        Args:
            n_features: Maximum number of features to retain (sorted by score)
            scale_factor: Pyramid decimation ratio (>1.0). 1.2 means each level
                is 1.2x smaller than the previous.
            n_levels: Number of pyramid levels for multi-scale detection
            edge_threshold: Border margin (pixels) where features are not detected
            fast_threshold: Threshold for FAST corner detection. Lower values
                detect more corners but may include noise.
        """
        self._orb = cv2.ORB_create(
            nfeatures=n_features,
            scaleFactor=scale_factor,
            nlevels=n_levels,
            edgeThreshold=edge_threshold,
            fastThreshold=fast_threshold,
        )
        self._n_features = n_features

    def detect(self, image: np.ndarray, mask: np.ndarray | None = None) -> Features:
        """Detect ORB features in an image.

        Args:
            image: Grayscale image (uint8). Color images will work but
                are converted internally.
            mask: Optional binary mask where 255 = detect, 0 = ignore.
                Must be same size as image.

        Returns:
            Features object containing keypoints and descriptors

        Example:
            >>> detector = FeatureDetector(n_features=500)
            >>> features = detector.detect(grayscale_image)
            >>> print(f"Detected {len(features)} features")
        """
        keypoints, descriptors = self._orb.detectAndCompute(image, mask)

        # Handle case where no features are detected
        if keypoints is None:
            keypoints = []
        if descriptors is None:
            descriptors = None

        return Features(keypoints=tuple(keypoints), descriptors=descriptors)

    @property
    def n_features(self) -> int:
        """Return maximum number of features to detect."""
        return self._n_features
