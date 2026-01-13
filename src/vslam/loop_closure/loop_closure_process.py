"""Loop closure process running in a separate Python process.

This process receives keyframes from the local BA process, detects
loop closures, and sends global pose corrections back to the frontend.

Running in a separate process ensures that slow loop detection
(~100-500ms) doesn't block the local BA (~50-100ms).
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Process, Queue
from pathlib import Path
from queue import Empty
from typing import TYPE_CHECKING

import numpy as np

from ..frontend import SE3
from .loop_detector import KeyframeData, LoopDetector
from .messages import (
    GlobalCorrectionMessage,
    LoopClosureShutdownMessage,
    LoopKeyframeData,
    LoopKeyframeMessage,
)
from .vocabulary import VisualVocabulary

if TYPE_CHECKING:
    from multiprocessing import Queue as QueueType


def _loop_closure_main(
    from_local_ba: "QueueType",
    to_frontend: "QueueType",
    vocabulary_path: str,
    camera_matrix: np.ndarray,
    min_score: float,
    min_keyframe_gap: int,
) -> None:
    """Main loop for the loop closure process.

    Args:
        from_local_ba: Queue to receive keyframes from local BA
        to_frontend: Queue to send corrections to frontend
        vocabulary_path: Path to vocabulary .npz file
        camera_matrix: 3x3 camera intrinsics
        min_score: Minimum BoW similarity score
        min_keyframe_gap: Minimum keyframe ID gap
    """
    # Initialize loop detector
    loop_detector = LoopDetector.from_vocabulary_path(
        vocabulary_path=vocabulary_path,
        camera_matrix=camera_matrix,
        min_score=min_score,
        min_keyframe_gap=min_keyframe_gap,
    )

    print("[LoopClosure] Process started")

    while True:
        try:
            # Wait for message from local BA
            msg = from_local_ba.get(timeout=1.0)

            if isinstance(msg, LoopClosureShutdownMessage):
                print("[LoopClosure] Shutdown received")
                break

            if isinstance(msg, LoopKeyframeMessage):
                # Convert message to KeyframeData
                kf_data = msg.keyframe
                keyframe = KeyframeData(
                    id=kf_data.id,
                    pose=SE3(
                        rotation=kf_data.pose_rotation,
                        translation=kf_data.pose_translation,
                    ),
                    descriptors=kf_data.descriptors,
                    keypoints=kf_data.keypoints,
                    points_3d=kf_data.points_3d,
                    keypoint_to_3d=kf_data.keypoint_to_3d,
                )

                # Add keyframe and detect loop
                result = loop_detector.add_keyframe(keyframe)

                # Debug: print 3D point info
                n_3d_pts = len(keyframe.points_3d) if keyframe.points_3d is not None else 0
                n_kp_3d = len(keyframe.keypoint_to_3d)
                if kf_data.id % 20 == 0:  # Print every 20 keyframes
                    print(
                        f"[LoopClosure] KF {kf_data.id}: "
                        f"{n_3d_pts} 3D pts, {n_kp_3d} kp→3d mappings, "
                        f"db_size={loop_detector.num_keyframes}"
                    )

                if result.detected:
                    print(
                        f"[LoopClosure] ✓ Loop detected: KF {result.query_keyframe_id} "
                        f"→ KF {result.match_keyframe_id} "
                        f"(score={result.similarity_score:.2f}, "
                        f"inliers={result.num_inliers})"
                    )

                    # Run pose graph optimization
                    optimized_poses = loop_detector.optimize()

                    # Convert to serializable format (4x4 matrices)
                    pose_corrections = {}
                    for kf_id, pose in optimized_poses.items():
                        pose_corrections[kf_id] = pose.to_matrix()

                    # Send corrections to frontend
                    correction = GlobalCorrectionMessage(
                        pose_corrections=pose_corrections,
                        loop_from_id=result.query_keyframe_id,
                        loop_to_id=result.match_keyframe_id,
                    )
                    to_frontend.put(correction)

        except Empty:
            # No message received, continue
            continue
        except Exception as e:
            print(f"[LoopClosure] Error: {e}")
            continue

    print("[LoopClosure] Process stopped")


class LoopClosureProcess:
    """Manages the loop closure process.

    Provides start/stop lifecycle and message passing to/from
    the loop closure subprocess.
    """

    def __init__(
        self,
        vocabulary_path: str | Path,
        camera_matrix: np.ndarray,
        min_score: float = 0.5,  # Increased for robustness in repetitive environments
        min_keyframe_gap: int = 30,  # Reduced to catch loops earlier
    ) -> None:
        """Initialize loop closure process manager.

        Args:
            vocabulary_path: Path to vocabulary .npz file
            camera_matrix: 3x3 camera intrinsics
            min_score: Minimum BoW similarity score
            min_keyframe_gap: Minimum keyframe ID gap
        """
        self._vocabulary_path = str(vocabulary_path)
        self._camera_matrix = camera_matrix.copy()
        self._min_score = min_score
        self._min_keyframe_gap = min_keyframe_gap

        self._process: Process | None = None
        self._to_loop_closure: Queue | None = None
        self._from_loop_closure: Queue | None = None

    def start(self) -> None:
        """Start the loop closure process."""
        if self._process is not None:
            return

        # Create queues
        ctx = mp.get_context("spawn")
        self._to_loop_closure = ctx.Queue(maxsize=100)
        self._from_loop_closure = ctx.Queue(maxsize=100)

        # Start process
        self._process = ctx.Process(
            target=_loop_closure_main,
            args=(
                self._to_loop_closure,
                self._from_loop_closure,
                self._vocabulary_path,
                self._camera_matrix,
                self._min_score,
                self._min_keyframe_gap,
            ),
            daemon=True,
        )
        self._process.start()

    def stop(self) -> None:
        """Stop the loop closure process."""
        if self._process is None:
            return

        # Send shutdown message
        if self._to_loop_closure is not None:
            try:
                self._to_loop_closure.put(LoopClosureShutdownMessage(), timeout=1.0)
            except:
                pass

        # Wait for process to finish
        self._process.join(timeout=5.0)

        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)

        self._process = None
        self._to_loop_closure = None
        self._from_loop_closure = None

    def send_keyframe(self, msg: LoopKeyframeMessage) -> bool:
        """Send a keyframe to the loop closure process.

        Args:
            msg: Keyframe message

        Returns:
            True if sent successfully
        """
        if self._to_loop_closure is None:
            return False

        try:
            self._to_loop_closure.put_nowait(msg)
            return True
        except:
            return False

    def get_corrections(self, timeout: float = 0.0) -> GlobalCorrectionMessage | None:
        """Get pose corrections from loop closure.

        Args:
            timeout: Timeout in seconds (0 for non-blocking)

        Returns:
            GlobalCorrectionMessage if available, None otherwise
        """
        if self._from_loop_closure is None:
            return None

        try:
            if timeout > 0:
                return self._from_loop_closure.get(timeout=timeout)
            else:
                return self._from_loop_closure.get_nowait()
        except Empty:
            return None

    @property
    def is_running(self) -> bool:
        """Check if process is running."""
        return self._process is not None and self._process.is_alive()
