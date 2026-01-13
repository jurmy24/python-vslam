"""Loop closure detection for Visual SLAM.

This module provides place recognition and pose graph optimization
to detect when the camera revisits a previously seen location and
correct accumulated drift.

Key components:
- VisualVocabulary: Bag of Visual Words for image similarity
- PlaceDatabase: Database of visited places
- GeometricVerifier: PnP-based verification of candidates
- PoseGraph: Global pose optimization
- LoopDetector: Main detection pipeline
- LoopClosureProcess: Separate process for loop detection
"""

from .geometric_verification import GeometricVerifier, VerificationResult
from .loop_closure_process import LoopClosureProcess
from .loop_detector import KeyframeData, LoopClosureResult, LoopDetector
from .messages import (
    GlobalCorrectionMessage,
    LoopClosureShutdownMessage,
    LoopKeyframeData,
    LoopKeyframeMessage,
)
from .place_recognition import PlaceDatabase, PlaceEntry, QueryResult
from .pose_graph import PoseEdge, PoseGraph
from .vocabulary import VisualVocabulary

__all__ = [
    # Vocabulary
    "VisualVocabulary",
    # Place Recognition
    "PlaceDatabase",
    "PlaceEntry",
    "QueryResult",
    # Geometric Verification
    "GeometricVerifier",
    "VerificationResult",
    # Pose Graph
    "PoseGraph",
    "PoseEdge",
    # Loop Detector
    "LoopDetector",
    "LoopClosureResult",
    "KeyframeData",
    # Process
    "LoopClosureProcess",
    # Messages
    "LoopKeyframeData",
    "LoopKeyframeMessage",
    "GlobalCorrectionMessage",
    "LoopClosureShutdownMessage",
]
