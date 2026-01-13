"""Python VSLAM - Visual SLAM implementation in Python."""

__version__ = "0.1.0"

# Re-export main classes for convenient imports
from .dataset_reader import DatasetReader
from .io import GroundTruthReader
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
from .backend import (
    BAResult,
    Keyframe,
    KeyframeManager,
    LocalMapping,
    ScipyBundleAdjustment,
)
from .loop_closure import (
    LoopClosureProcess,
    LoopClosureResult,
    LoopDetector,
    PoseGraph,
    VisualVocabulary,
)
from .slam_system import SLAMFrame, SLAMStats, SLAMSystem
from .visualization import RerunVisualizer

__all__ = [
    "__version__",
    # Dataset / I/O
    "DatasetReader",
    "GroundTruthReader",
    # SLAM System
    "SLAMSystem",
    "SLAMFrame",
    "SLAMStats",
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
    # Backend / Bundle Adjustment
    "Keyframe",
    "KeyframeManager",
    "LocalMapping",
    "ScipyBundleAdjustment",
    "BAResult",
    # Loop Closure
    "LoopClosureProcess",
    "LoopDetector",
    "LoopClosureResult",
    "PoseGraph",
    "VisualVocabulary",
    # Visualization
    "RerunVisualizer",
]
