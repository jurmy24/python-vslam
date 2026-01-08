# Python VSLAM

Visual SLAM (Simultaneous Localization and Mapping) implementation in Python, designed to work with the EuRoC MAV dataset.

## Project Status

This is an early-stage implementation. Currently implemented:

### ✅ DatasetReader

A robust reader for EuRoC MAV stereo camera datasets.

**What it does:**

- Parses `cam0/data.csv` to get image list with timestamps
- Loads synchronized stereo image pairs from cam0/cam1 directories
- Returns images as grayscale numpy arrays (ready for feature detection)
- Provides Python iterator interface for easy frame-by-frame processing
- Lazy loading (images loaded on-demand for memory efficiency)
- Comprehensive error handling with clear messages

**What it doesn't do (yet):**

- ❌ Image undistortion (returns raw distorted images from cameras)
- ❌ Parse `sensor.yaml` for camera intrinsics/calibration parameters
- ❌ Load ground truth trajectory data
- ❌ Rectification for stereo processing
- ❌ Any calibration or preprocessing

**Usage:**

```python
from vslam.dataset_reader import DatasetReader

# Initialize with path to mav0 directory
reader = DatasetReader('data/euroc/MH_01_easy/mav0')

print(f"Dataset contains {len(reader)} stereo pairs")

# Iterate through frames
for left_img, right_img, timestamp_ns in reader:
    # left_img, right_img: grayscale numpy arrays (H, W)
    # timestamp_ns: frame timestamp in nanoseconds

    # Your SLAM processing here...
    print(f"Processing frame at {timestamp_ns}ns")
```

**Alternative usage:**

```python
# Manual iteration with get_next_stereo_pair()
while (pair := reader.get_next_stereo_pair()) is not None:
    left, right, timestamp = pair
    # Process frame...

# Reset to beginning
reader.reset()
```

## Setup

See [claude.md](claude.md) for detailed setup instructions.

**Quick start:**

```bash
uv sync --dev              # Install dependencies
uv run python -m pytest    # Run tests
```

## Dataset

This project uses the **EuRoC MAV Dataset** for stereo visual SLAM.

**Dataset Citation:**

```bibtex
@article{Burri25012016,
  author  = {Burri, Michael and Nikolic, Janosch and Gohl, Pascal and Schneider, Thomas and Rehder, Joern and Omari, Sammy and Achtelik, Markus W and Siegwart, Roland},
  title   = {The EuRoC micro aerial vehicle datasets},
  year    = {2016},
  doi     = {10.1177/0278364915620033},
  URL     = {http://ijr.sagepub.com/content/early/2016/01/21/0278364915620033.abstract},
  eprint  = {http://ijr.sagepub.com/content/early/2016/01/21/0278364915620033.full.pdf+html},
  journal = {The International Journal of Robotics Research}
}
```

## Not Implemented

The following features are not implemented yet:

- [ ] Camera calibration/intrinsics parser
- [ ] Image undistortion and rectification
- [ ] Feature detection (ORB, SIFT, etc.)
- [ ] Stereo matching and depth estimation
- [ ] Visual odometry
- [ ] Loop closure detection
- [ ] Map building and optimization
