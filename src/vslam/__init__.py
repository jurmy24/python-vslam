"""Python VSLAM - Visual SLAM implementation in Python."""

__version__ = "0.1.0"

# Re-export main classes for convenient imports
from .dataset_reader import DatasetReader
from .io import GroundTruthReader, IMUCalibration, IMUMeasurement, IMUReader
from .frontend import (
    SE3,
    StereoCamera,
    IMUFrame,
    IMUIntegrator,
    IMUOdometry,
    IMUState,
    IMUTrackingStatus,
)
from .frontend.vo import (
    Map,
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
    "IMUReader",
    "IMUMeasurement",
    "IMUCalibration",
    # SLAM System
    "SLAMSystem",
    "SLAMFrame",
    "SLAMStats",
    # IMU Odometry
    "IMUOdometry",
    "IMUFrame",
    "IMUIntegrator",
    "IMUState",
    "IMUTrackingStatus",
    # Visual Odometry (moved to frontend.vo)
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
