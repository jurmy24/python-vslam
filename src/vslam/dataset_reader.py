"""EuRoC MAV dataset reader for stereo camera images."""

from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class DatasetReader:
    """Reader for EuRoC MAV dataset stereo camera data."""

    def __init__(self, dataset_path: str = "data/euroc/MH_01_easy/mav0") -> None:
        """Initialize reader with path to dataset.

        Args:
            dataset_path: Path to mav0 directory

        Raises:
            FileNotFoundError: If dataset path or required directories don't exist
            ValueError: If data.csv is empty or invalid
        """
        self.dataset_path = Path(dataset_path)

        # Setup data paths
        self.cam0_path = self.dataset_path / "cam0"
        self.cam1_path = self.dataset_path / "cam1"
        self.cam0_data_path = self.cam0_path / "data"
        self.cam1_data_path = self.cam1_path / "data"
        # Extensible if we want the imu, groundtruth, and leica data

        # Validate paths exist
        self._validate_paths()

        # Load image list from CSV
        self._image_list = self._load_image_list()

        if not self._image_list:
            raise ValueError(f"No images found in {self.cam0_path / 'data.csv'}")

        # Initialize iterator state
        self._current_idx = 0

    def _validate_paths(self) -> None:
        """Validate that all required paths exist."""
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        if not self.cam0_path.exists():
            raise FileNotFoundError(
                f"cam0 directory not found: {self.cam0_path}\n"
                f"Expected structure: {self.dataset_path}/cam0/"
            )

        if not self.cam1_path.exists():
            raise FileNotFoundError(
                f"cam1 directory not found: {self.cam1_path}\n"
                f"Expected structure: {self.dataset_path}/cam1/"
            )

        if not self.cam0_data_path.exists():
            raise FileNotFoundError(
                f"cam0/data directory not found: {self.cam0_data_path}"
            )

        if not self.cam1_data_path.exists():
            raise FileNotFoundError(
                f"cam1/data directory not found: {self.cam1_data_path}"
            )

        csv_path = self.cam0_path / "data.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"cam0/data.csv not found: {csv_path}\n"
                f"This file is required to list image timestamps and filenames."
            )

    def _load_image_list(self) -> list[tuple[int, str]]:
        """Parse cam0/data.csv to get image list.

        CSV format:
            #timestamp [ns],filename
            1403636579763555584,1403636579763555584.png
            1403636579813555456,1403636579813555456.png

        Returns:
            List of (timestamp_ns, filename) tuples in chronological order
        """
        csv_path = self.cam0_path / "data.csv"
        image_list = []

        with open(csv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                try:
                    timestamp_str, filename = line.split(",")
                    timestamp_ns = int(timestamp_str.strip())
                    image_list.append((timestamp_ns, filename.strip()))
                except ValueError as e:
                    raise ValueError(
                        f"Invalid line in {csv_path}: '{line}'\n"
                        f"Expected format: timestamp,filename"
                    ) from e

        return image_list

    def _load_image_pair(self, filename: str) -> tuple[np.ndarray, np.ndarray]:
        """Load synchronized stereo pair by filename.

        Args:
            filename: Image filename (e.g., '1403636579763555584.png')

        Returns:
            Tuple of (left_image, right_image) as grayscale numpy arrays

        Raises:
            FileNotFoundError: If either image file doesn't exist
            ValueError: If image loading fails
        """
        left_path = self.cam0_data_path / filename
        right_path = self.cam1_data_path / filename

        # Check files exist
        if not left_path.exists():
            raise FileNotFoundError(f"Left camera image not found: {left_path}")

        if not right_path.exists():
            raise FileNotFoundError(f"Right camera image not found: {right_path}")

        # Load images as grayscale
        left_img = cv2.imread(str(left_path), cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(str(right_path), cv2.IMREAD_GRAYSCALE)

        # Check loading succeeded
        if left_img is None:
            raise ValueError(f"Failed to load left image: {left_path}")

        if right_img is None:
            raise ValueError(f"Failed to load right image: {right_path}")

        return left_img, right_img

    def get_next_stereo_pair(
        self,
    ) -> tuple[np.ndarray, np.ndarray, int] | None:
        """Get next synchronized stereo image pair.

        Returns:
            Tuple of (left_image, right_image, timestamp_ns) where:
                - left_image: Left camera image (grayscale numpy array)
                - right_image: Right camera image (grayscale numpy array)
                - timestamp_ns: Timestamp in nanoseconds
            Returns None if no more images available.

        Example:
            >>> reader = DatasetReader('data/euroc/MH_01_easy/mav0')
            >>> while (pair := reader.get_next_stereo_pair()) is not None:
            ...     left, right, timestamp = pair
            ...     print(f"Processing frame at {timestamp}ns")
        """
        if self._current_idx >= len(self._image_list):
            return None

        timestamp_ns, filename = self._image_list[self._current_idx]
        left, right = self._load_image_pair(filename)

        self._current_idx += 1
        return left, right, timestamp_ns

    def reset(self) -> None:
        """Reset iterator to beginning of dataset."""
        self._current_idx = 0

    def __len__(self) -> int:
        """Return total number of stereo pairs in dataset."""
        return len(self._image_list)

    def __iter__(self) -> Iterator[tuple[np.ndarray, np.ndarray, int]]:
        """Allow iteration over dataset.

        Yields:
            Tuple of (left_image, right_image, timestamp_ns)

        Example:
            >>> reader = DatasetReader('data/euroc/MH_01_easy/mav0')
            >>> for left, right, timestamp in reader:
            ...     print(f"Frame at {timestamp}ns")
        """
        self.reset()
        return self

    def __next__(self) -> tuple[np.ndarray, np.ndarray, int]:
        """Get next stereo pair for iterator protocol.

        Returns:
            Tuple of (left_image, right_image, timestamp_ns)

        Raises:
            StopIteration: When no more images available
        """
        pair = self.get_next_stereo_pair()
        if pair is None:
            raise StopIteration
        return pair
