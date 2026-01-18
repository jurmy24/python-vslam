"""Visual odometry components (preserved for future Visual-Inertial fusion).

This module contains the stereo visual odometry pipeline that has been
replaced by IMU-based odometry. It is kept for potential future use in
Visual-Inertial Odometry (VIO) fusion systems.

Components:
- VisualOdometry: Main VO pipeline with 7-stage processing
- StereoFrontend: Stereo feature extraction and triangulation
- TemporalMatcher: Frame-to-frame feature tracking
- MotionEstimator: PnP+RANSAC pose estimation
- FeatureDetector: ORB feature detection
- StereoMatcher: Left-right stereo matching
- Map/MapPoint: Sparse 3D landmark management
- Track/TrackManager: Feature track management
"""

from .feature_detector import FeatureDetector, Features
from .map_point import Map, MapPoint, Observation
from .motion_estimator import MotionEstimator, PnPResult
from .stereo_frontend import StereoFrame, StereoFrontend
from .stereo_matcher import StereoMatcher, StereoMatches
from .temporal_matcher import TemporalMatcher, TemporalMatches
from .track import Track, TrackManager, TrackObservation
from .visual_odometry import TrackingStatus, VisualOdometry, VOFrame, VOTiming

__all__ = [
    # Visual Odometry
    "VisualOdometry",
    "VOFrame",
    "VOTiming",
    "TrackingStatus",
    # Stereo Frontend
    "StereoFrontend",
    "StereoFrame",
    # Features
    "FeatureDetector",
    "Features",
    # Stereo Matching
    "StereoMatcher",
    "StereoMatches",
    # Temporal Matching
    "TemporalMatcher",
    "TemporalMatches",
    # Map
    "Map",
    "MapPoint",
    "Observation",
    # Tracking
    "Track",
    "TrackManager",
    "TrackObservation",
    # Motion Estimation
    "MotionEstimator",
    "PnPResult",
]
