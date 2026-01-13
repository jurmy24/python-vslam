# Python VSLAM

Visual SLAM (Simultaneous Localization and Mapping) implementation in Python for the EuRoC MAV dataset.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            SLAM System                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────┐     ┌─────────────────────────────────────┐   │
│  │   Frontend (~20Hz)  │     │        Backend Processes            │   │
│  │                     │     │                                     │   │
│  │  Stereo Camera      │────>│  Local BA (~10Hz)                   │   │
│  │  ORB Features       │     │  - Sliding window optimization      │   │
│  │  Stereo Matching    │     │  - Scipy bundle adjustment          │   │
│  │  Frame-to-Frame     │     │  - Covisibility graph               │   │
│  │    Tracking (PnP)   │     │                                     │   │
│  │  SE3 Poses          │────>│  Loop Closure (~2-5Hz)              │   │
│  │  Sparse Map         │     │  - Bag of Visual Words              │   │
│  │                     │     │  - Geometric verification           │   │
│  └─────────────────────┘     │  - Pose graph optimization          │   │
│                              └─────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────┐     ┌─────────────────────────────────────┐   │
│  │   Visualization     │     │        Ground Truth                 │   │
│  │   Rerun 3D viewer   │     │   EuRoC GT reader + interpolation   │   │
│  └─────────────────────┘     └─────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# Install dependencies
uv sync --dev

# Download EuRoC dataset (MH_01_easy recommended)
mkdir -p data/euroc
# Download from: https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

# Run visual odometry demo
uv run python examples/vo_demo.py

# Train vocabulary for loop closure (one-time)
uv run python scripts/train_vocabulary.py

# Run full SLAM with loop closure
uv run python examples/slam_with_loop_closure.py
```

## Components

### Frontend (`src/vslam/frontend/`)

Real-time visual odometry with stereo cameras:
- **StereoCamera**: Calibration, rectification, and undistortion
- **FeatureDetector**: ORB feature extraction
- **StereoMatcher**: Left-right feature matching for depth
- **TemporalMatcher**: Frame-to-frame feature tracking
- **MotionEstimator**: PnP pose estimation with RANSAC
- **VisualOdometry**: Complete VO pipeline with sparse map

### Backend (`src/vslam/backend/`)

Asynchronous optimization running in separate process:
- **KeyframeManager**: Keyframe selection and management
- **CovisibilityGraph**: Tracks shared observations between keyframes
- **ScipyBundleAdjustment**: Sparse bundle adjustment using scipy
- **LocalMapping**: Sliding window BA for recent keyframes
- **BackendProcess**: Multi-process architecture

### Loop Closure (`src/vslam/loop_closure/`)

Global drift correction via place recognition:
- **VisualVocabulary**: Bag of Visual Words from ORB descriptors
- **PlaceDatabase**: Efficient place retrieval
- **GeometricVerifier**: PnP-based loop candidate verification
- **PoseGraph**: Global pose optimization after loop detection
- **LoopClosureProcess**: Separate process for loop detection

### I/O (`src/vslam/io/`)

- **DatasetReader**: EuRoC stereo image loader with timestamps
- **GroundTruthReader**: Ground truth poses with interpolation

### Visualization (`src/vslam/visualization/`)

- **RerunVisualizer**: Real-time 3D visualization of trajectory, map, and loop closures

## Demo Scripts

| Script | Description |
|--------|-------------|
| `examples/stereo_demo.py` | Basic stereo processing |
| `examples/vo_demo.py` | Visual odometry only |
| `examples/slam_demo.py` | SLAM with backend BA |
| `examples/slam_with_loop_closure.py` | Full pipeline with loop closure |

## Usage Example

```python
from vslam import DatasetReader, SLAMSystem, RerunVisualizer

# Initialize
reader = DatasetReader("data/euroc/MH_01_easy/mav0")
visualizer = RerunVisualizer("my-slam")

with SLAMSystem.from_dataset_path(
    "data/euroc/MH_01_easy/mav0",
    enable_backend=True,
    enable_loop_closure=True,
    vocabulary_path="data/vocabulary.npz",
) as slam:
    for left, right, timestamp_ns in reader:
        result = slam.process_frame(left, right, timestamp_ns)

        if result.is_keyframe:
            print(f"New keyframe {result.keyframe_id}")
        if result.loop_closure_detected:
            print(f"Loop: {result.loop_from_id} -> {result.loop_to_id}")
```

## Dataset

This project uses the **EuRoC MAV Dataset**. Download sequences from:
https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

Recommended: Start with `MH_01_easy` (Machine Hall, easy difficulty).

**Citation:**

```bibtex
@article{Burri25012016,
  author  = {Burri, Michael and Nikolic, Janosch and Gohl, Pascal and Schneider, Thomas and Rehder, Joern and Omari, Sammy and Achtelik, Markus W and Siegwart, Roland},
  title   = {The EuRoC micro aerial vehicle datasets},
  year    = {2016},
  doi     = {10.1177/0278364915620033},
  URL     = {http://ijr.sagepub.com/content/early/2016/01/21/0278364915620033.abstract},
  journal = {The International Journal of Robotics Research}
}
```

## Development

```bash
uv sync --dev          # Install dependencies
uv run pytest          # Run tests
uv run mypy src/       # Type checking
uv run ruff check .    # Linting
uv run ruff format .   # Formatting
```

See [CLAUDE.md](CLAUDE.md) for detailed development instructions.
