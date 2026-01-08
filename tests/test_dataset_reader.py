"""Tests for DatasetReader class."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from vslam.dataset_reader import DatasetReader


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    """Create a mock EuRoC dataset structure for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to mock mav0 directory
    """
    # Create directory structure
    mav0 = tmp_path / "mav0"
    cam0_data = mav0 / "cam0" / "data"
    cam1_data = mav0 / "cam1" / "data"

    cam0_data.mkdir(parents=True)
    cam1_data.mkdir(parents=True)

    # Create test images (simple 100x100 grayscale images)
    timestamps = [1403636579763555584, 1403636579813555456, 1403636579863555328]

    for i, timestamp in enumerate(timestamps):
        filename = f"{timestamp}.png"

        # Create left image (with value = i * 50)
        left_img = np.full((100, 100), i * 50, dtype=np.uint8)
        cv2.imwrite(str(cam0_data / filename), left_img)

        # Create right image (with value = i * 50 + 25)
        right_img = np.full((100, 100), i * 50 + 25, dtype=np.uint8)
        cv2.imwrite(str(cam1_data / filename), right_img)

    # Create data.csv
    csv_content = "#timestamp [ns],filename\n"
    for timestamp in timestamps:
        csv_content += f"{timestamp},{timestamp}.png\n"

    (mav0 / "cam0" / "data.csv").write_text(csv_content)

    return mav0


class TestDatasetReader:
    """Test suite for DatasetReader class."""

    def test_initialization(self, mock_dataset: Path):
        """Test that DatasetReader initializes correctly."""
        reader = DatasetReader(str(mock_dataset))

        assert reader.dataset_path == mock_dataset
        assert len(reader) == 3
        assert reader._current_idx == 0

    def test_initialization_missing_dataset_path(self, tmp_path: Path):
        """Test that initialization fails with missing dataset path."""
        nonexistent = tmp_path / "nonexistent"

        with pytest.raises(FileNotFoundError, match="Dataset path does not exist"):
            DatasetReader(str(nonexistent))

    def test_initialization_missing_cam0(self, tmp_path: Path):
        """Test that initialization fails when cam0 directory is missing."""
        mav0 = tmp_path / "mav0"
        mav0.mkdir()

        with pytest.raises(FileNotFoundError, match="cam0 directory not found"):
            DatasetReader(str(mav0))

    def test_initialization_missing_cam1(self, tmp_path: Path):
        """Test that initialization fails when cam1 directory is missing."""
        mav0 = tmp_path / "mav0"
        cam0 = mav0 / "cam0"
        cam0.mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="cam1 directory not found"):
            DatasetReader(str(mav0))

    def test_initialization_missing_data_csv(self, tmp_path: Path):
        """Test that initialization fails when data.csv is missing."""
        mav0 = tmp_path / "mav0"
        (mav0 / "cam0" / "data").mkdir(parents=True)
        (mav0 / "cam1" / "data").mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="cam0/data.csv not found"):
            DatasetReader(str(mav0))

    def test_initialization_empty_csv(self, tmp_path: Path):
        """Test that initialization fails with empty data.csv."""
        mav0 = tmp_path / "mav0"
        cam0 = mav0 / "cam0"
        cam1 = mav0 / "cam1"

        (cam0 / "data").mkdir(parents=True)
        (cam1 / "data").mkdir(parents=True)

        # Create empty CSV (only header)
        (cam0 / "data.csv").write_text("#timestamp [ns],filename\n")

        with pytest.raises(ValueError, match="No images found"):
            DatasetReader(str(mav0))

    def test_get_next_stereo_pair(self, mock_dataset: Path):
        """Test getting next stereo pair."""
        reader = DatasetReader(str(mock_dataset))

        # Get first pair
        result = reader.get_next_stereo_pair()
        assert result is not None

        left, right, timestamp = result
        assert left.shape == (100, 100)
        assert right.shape == (100, 100)
        assert timestamp == 1403636579763555584
        assert np.all(left == 0)  # First image has value 0
        assert np.all(right == 25)  # First right image has value 25

    def test_get_next_stereo_pair_exhausted(self, mock_dataset: Path):
        """Test that get_next_stereo_pair returns None when exhausted."""
        reader = DatasetReader(str(mock_dataset))

        # Get all pairs
        for _ in range(3):
            result = reader.get_next_stereo_pair()
            assert result is not None

        # Should return None when exhausted
        result = reader.get_next_stereo_pair()
        assert result is None

    def test_reset(self, mock_dataset: Path):
        """Test that reset returns iterator to beginning."""
        reader = DatasetReader(str(mock_dataset))

        # Advance to second frame
        reader.get_next_stereo_pair()
        reader.get_next_stereo_pair()
        assert reader._current_idx == 2

        # Reset
        reader.reset()
        assert reader._current_idx == 0

        # Should get first frame again
        result = reader.get_next_stereo_pair()
        assert result is not None
        _, _, timestamp = result
        assert timestamp == 1403636579763555584

    def test_len(self, mock_dataset: Path):
        """Test that len returns correct number of frames."""
        reader = DatasetReader(str(mock_dataset))
        assert len(reader) == 3

    def test_iterator_protocol(self, mock_dataset: Path):
        """Test that DatasetReader works with Python iterator protocol."""
        reader = DatasetReader(str(mock_dataset))

        timestamps = []
        for left, right, timestamp in reader:
            timestamps.append(timestamp)
            assert left.shape == (100, 100)
            assert right.shape == (100, 100)

        assert len(timestamps) == 3
        assert timestamps[0] == 1403636579763555584
        assert timestamps[1] == 1403636579813555456
        assert timestamps[2] == 1403636579863555328

    def test_multiple_iterations(self, mock_dataset: Path):
        """Test that reader can be iterated multiple times."""
        reader = DatasetReader(str(mock_dataset))

        # First iteration
        count1 = sum(1 for _ in reader)
        assert count1 == 3

        # Second iteration (should work due to automatic reset)
        count2 = sum(1 for _ in reader)
        assert count2 == 3

    def test_image_loading_order(self, mock_dataset: Path):
        """Test that images are loaded in chronological order."""
        reader = DatasetReader(str(mock_dataset))

        expected_timestamps = [
            1403636579763555584,
            1403636579813555456,
            1403636579863555328,
        ]

        for expected_ts in expected_timestamps:
            result = reader.get_next_stereo_pair()
            assert result is not None
            _, _, timestamp = result
            assert timestamp == expected_ts

    def test_synchronized_images(self, mock_dataset: Path):
        """Test that left and right images are properly synchronized."""
        reader = DatasetReader(str(mock_dataset))

        for i in range(3):
            result = reader.get_next_stereo_pair()
            assert result is not None

            left, right, _ = result

            # Verify left and right have expected values (from fixture)
            expected_left_value = i * 50
            expected_right_value = i * 50 + 25

            assert np.all(left == expected_left_value)
            assert np.all(right == expected_right_value)

    def test_missing_left_image(self, mock_dataset: Path):
        """Test error when left camera image is missing."""
        # Remove one left image
        (mock_dataset / "cam0" / "data" / "1403636579763555584.png").unlink()

        reader = DatasetReader(str(mock_dataset))

        with pytest.raises(FileNotFoundError, match="Left camera image not found"):
            reader.get_next_stereo_pair()

    def test_missing_right_image(self, mock_dataset: Path):
        """Test error when right camera image is missing."""
        # Remove one right image
        (mock_dataset / "cam1" / "data" / "1403636579763555584.png").unlink()

        reader = DatasetReader(str(mock_dataset))

        with pytest.raises(FileNotFoundError, match="Right camera image not found"):
            reader.get_next_stereo_pair()

    def test_grayscale_loading(self, mock_dataset: Path):
        """Test that images are loaded as grayscale."""
        reader = DatasetReader(str(mock_dataset))

        result = reader.get_next_stereo_pair()
        assert result is not None

        left, right, _ = result

        # Grayscale images should be 2D
        assert left.ndim == 2
        assert right.ndim == 2
        assert left.dtype == np.uint8
        assert right.dtype == np.uint8

    def test_csv_parsing_with_whitespace(self, tmp_path: Path):
        """Test CSV parsing handles whitespace correctly."""
        mav0 = tmp_path / "mav0"
        cam0_data = mav0 / "cam0" / "data"
        cam1_data = mav0 / "cam1" / "data"

        cam0_data.mkdir(parents=True)
        cam1_data.mkdir(parents=True)

        timestamp = 1403636579763555584
        filename = f"{timestamp}.png"

        # Create images
        img = np.zeros((10, 10), dtype=np.uint8)
        cv2.imwrite(str(cam0_data / filename), img)
        cv2.imwrite(str(cam1_data / filename), img)

        # Create CSV with extra whitespace
        csv_content = "#timestamp [ns],filename\n"
        csv_content += f"  {timestamp}  ,  {filename}  \n"

        (mav0 / "cam0" / "data.csv").write_text(csv_content)

        reader = DatasetReader(str(mav0))
        assert len(reader) == 1

        result = reader.get_next_stereo_pair()
        assert result is not None
