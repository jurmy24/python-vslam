"""Python VSLAM - Visual SLAM implementation in Python."""

__version__ = "0.1.0"

# Re-export main classes for convenient imports
from .dataset_reader import DatasetReader
from .frontend import (
    Map,
    SE3,
    StereoCamera,
    StereoFrame,
    StereoFrontend,
    TrackingStatus,
    VisualOdometry,
    VOFrame,
)
from .visualization import RerunVisualizer

__all__ = [
    "__version__",
    # Dataset
    "DatasetReader",
    # Visual Odometry
    "VisualOdometry",
    "VOFrame",
    "TrackingStatus",
    # Pose
    "SE3",
    # Stereo Frontend
    "StereoFrontend",
    "StereoFrame",
    "StereoCamera",
    # Map
    "Map",
    # Visualization
    "RerunVisualizer",
]
