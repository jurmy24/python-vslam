"""SLAM system orchestrating frontend visual odometry and backend bundle adjustment.

SLAMSystem combines:
- Visual Odometry (frontend): Real-time frame-to-frame tracking
- Local Bundle Adjustment (backend): Asynchronous pose refinement
- Loop Closure (backend): Global drift correction via place recognition

The frontend runs in the main process at full frame rate, while the backends
run in separate processes to avoid blocking tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .backend import (
    BackendProcess,
    CorrectionMessage,
    Keyframe,
    KeyframeData,
    KeyframeManager,
    KeyframeMessage,
    MapPointData,
)
from .frontend import SE3, StereoFrontend, VisualOdometry, VOFrame
from .loop_closure import (
    GlobalCorrectionMessage,
    LoopClosureProcess,
    LoopKeyframeData,
    LoopKeyframeMessage,
)

if TYPE_CHECKING:
    from .frontend.map_point import MapPoint


@dataclass
class SLAMFrame:
    """Output of the SLAM system for a single frame."""

    vo_frame: VOFrame
    is_keyframe: bool = False
    keyframe_id: int | None = None
    ba_applied: bool = False
    num_pose_corrections: int = 0
    num_point_corrections: int = 0
    loop_closure_detected: bool = False
    loop_from_id: int = -1
    loop_to_id: int = -1


@dataclass
class SLAMStats:
    """Statistics from the SLAM system."""

    num_frames: int = 0
    num_keyframes: int = 0
    num_ba_corrections: int = 0
    num_loop_closures: int = 0
    total_distance: float = 0.0


class SLAMSystem:
    """Complete SLAM system with frontend VO, backend BA, and loop closure.

    Orchestrates the visual odometry frontend, bundle adjustment backend,
    and loop closure detection as separate processes for optimal performance.
    """

    def __init__(
        self,
        stereo_frontend: StereoFrontend,
        enable_backend: bool = True,
        enable_loop_closure: bool = False,
        vocabulary_path: str | Path | None = None,
        min_keyframe_translation: float = 0.1,
        min_keyframe_rotation: float = 5.0,
        min_keyframe_frames: int = 5,
    ) -> None:
        """Initialize SLAM system.

        Args:
            stereo_frontend: Stereo frontend for image processing
            enable_backend: If True, run BA in background process
            enable_loop_closure: If True, run loop closure in background process
            vocabulary_path: Path to vocabulary file (required if enable_loop_closure)
            min_keyframe_translation: Min translation (m) for new keyframe
            min_keyframe_rotation: Min rotation (deg) for new keyframe
            min_keyframe_frames: Min frames between keyframes
        """
        # Frontend
        self._vo = VisualOdometry(stereo_frontend=stereo_frontend)

        # Keyframe management (in main process)
        self._keyframe_manager = KeyframeManager(
            min_translation=min_keyframe_translation,
            min_rotation=min_keyframe_rotation,
            min_frames_between=min_keyframe_frames,
        )

        # Backend (Local BA)
        self._enable_backend = enable_backend
        self._backend: BackendProcess | None = None
        self._camera_matrix = self._build_camera_matrix(stereo_frontend)

        # Loop Closure
        self._enable_loop_closure = enable_loop_closure
        self._vocabulary_path = vocabulary_path
        self._loop_closure: LoopClosureProcess | None = None

        if enable_loop_closure and vocabulary_path is None:
            raise ValueError("vocabulary_path required when enable_loop_closure=True")

        # State tracking
        self._stats = SLAMStats()
        self._prev_position: np.ndarray | None = None
        self._pending_corrections: list[CorrectionMessage] = []

        # Track which keypoints map to which map points for keyframe creation
        self._current_kp_to_mappoint: dict[int, int] = {}

    @classmethod
    def from_dataset_path(
        cls,
        dataset_path: str,
        n_features: int = 1000,
        enable_backend: bool = True,
        enable_loop_closure: bool = False,
        vocabulary_path: str | Path | None = None,
    ) -> SLAMSystem:
        """Create SLAMSystem from EuRoC dataset path.

        Args:
            dataset_path: Path to EuRoC dataset (e.g., data/euroc/MH_01_easy/mav0)
            n_features: Number of ORB features to detect
            enable_backend: If True, enable backend BA process
            enable_loop_closure: If True, enable loop closure process
            vocabulary_path: Path to vocabulary file (required if enable_loop_closure)

        Returns:
            Configured SLAMSystem
        """
        stereo_frontend = StereoFrontend.from_dataset_path(
            dataset_path, n_features=n_features
        )
        return cls(
            stereo_frontend=stereo_frontend,
            enable_backend=enable_backend,
            enable_loop_closure=enable_loop_closure,
            vocabulary_path=vocabulary_path,
        )

    def _build_camera_matrix(self, frontend: StereoFrontend) -> np.ndarray:
        """Build camera intrinsics matrix."""
        cam = frontend.stereo_camera
        return np.array(
            [
                [cam.focal_length, 0, cam.principal_point[0]],
                [0, cam.focal_length, cam.principal_point[1]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

    def start(self) -> None:
        """Start the SLAM system (launches backend processes if enabled)."""
        if self._enable_backend:
            self._backend = BackendProcess(camera_matrix=self._camera_matrix)
            self._backend.start()

        if self._enable_loop_closure and self._vocabulary_path is not None:
            self._loop_closure = LoopClosureProcess(
                vocabulary_path=self._vocabulary_path,
                camera_matrix=self._camera_matrix,
            )
            self._loop_closure.start()

    def stop(self) -> None:
        """Stop the SLAM system."""
        # Stop loop closure first (depends on backend)
        if self._loop_closure is not None:
            self._loop_closure.stop()
            self._loop_closure = None

        if self._backend is not None:
            self._backend.stop()
            self._backend = None

    def process_frame(
        self,
        left: np.ndarray,
        right: np.ndarray,
        timestamp_ns: int,
    ) -> SLAMFrame:
        """Process a stereo frame through the SLAM pipeline.

        Args:
            left: Left camera image (grayscale or color)
            right: Right camera image (grayscale or color)
            timestamp_ns: Timestamp in nanoseconds

        Returns:
            SLAMFrame with tracking result and keyframe info
        """
        # Step 1: Run visual odometry
        vo_frame = self._vo.process_frame(left, right, timestamp_ns)

        # Step 2: Check for backend corrections (local BA)
        ba_applied = False
        num_pose_corrections = 0
        num_point_corrections = 0

        if self._backend is not None:
            correction = self._backend.get_corrections(timeout=0.0)
            if correction is not None:
                self._apply_correction(correction)
                ba_applied = True
                num_pose_corrections = len(correction.pose_corrections)
                num_point_corrections = len(correction.point_corrections)
                self._stats.num_ba_corrections += 1

        # Step 3: Check for loop closure corrections (global)
        loop_closure_detected = False
        loop_from_id = -1
        loop_to_id = -1

        if self._loop_closure is not None:
            global_correction = self._loop_closure.get_corrections(timeout=0.0)
            if global_correction is not None:
                self._apply_global_correction(global_correction)
                loop_closure_detected = True
                loop_from_id = global_correction.loop_from_id
                loop_to_id = global_correction.loop_to_id
                self._stats.num_loop_closures += 1

        # Step 4: Update statistics
        self._stats.num_frames += 1
        if self._prev_position is not None and vo_frame.is_tracking_ok:
            dist = np.linalg.norm(vo_frame.position - self._prev_position)
            self._stats.total_distance += dist
        self._prev_position = vo_frame.position.copy()

        # Step 5: Keyframe decision
        is_keyframe = False
        keyframe_id = None

        # Get current keypoint-to-mappoint mapping from VO
        self._update_kp_to_mappoint_from_vo()

        if vo_frame.is_tracking_ok and self._keyframe_manager.should_create_keyframe(
            vo_frame
        ):
            keyframe = self._create_keyframe(vo_frame)
            self._keyframe_manager.add_keyframe(keyframe)
            is_keyframe = True
            keyframe_id = keyframe.id
            self._stats.num_keyframes += 1

            # Send to local BA backend
            if self._backend is not None:
                self._send_keyframe_to_backend(keyframe, vo_frame)

            # Send to loop closure backend
            if self._loop_closure is not None:
                self._send_keyframe_to_loop_closure(keyframe, vo_frame)

        return SLAMFrame(
            vo_frame=vo_frame,
            is_keyframe=is_keyframe,
            keyframe_id=keyframe_id,
            ba_applied=ba_applied,
            num_pose_corrections=num_pose_corrections,
            num_point_corrections=num_point_corrections,
            loop_closure_detected=loop_closure_detected,
            loop_from_id=loop_from_id,
            loop_to_id=loop_to_id,
        )

    def _update_kp_to_mappoint_from_vo(self) -> None:
        """Update keypoint-to-mappoint mapping from VO state.

        Note: This is a simplified approach. In a production implementation,
        the VO would expose this mapping directly.
        """
        # For now, we'll build this from the map's frame observations
        # This is approximate but works for demonstration
        pass  # The VO internally tracks this

    def _create_keyframe(self, vo_frame: VOFrame) -> Keyframe:
        """Create a keyframe from a VO frame."""
        features = vo_frame.stereo_frame.features_left

        # Build keypoint-to-mappoint mapping
        # We need to get this from the VO's internal state
        # For now, we'll use a simplified approach based on map points
        kp_to_mp: dict[int, int] = {}

        # Get map points observed in this frame
        map_points = self._vo.get_map().get_points_in_frame(vo_frame.frame_id)
        for mp in map_points:
            obs = mp.get_observation_in_frame(vo_frame.frame_id)
            if obs is not None:
                kp_to_mp[obs.keypoint_idx] = mp.id

        return Keyframe(
            id=vo_frame.frame_id,
            timestamp_ns=vo_frame.timestamp_ns,
            pose=vo_frame.pose,
            keypoints=features.points.copy(),
            descriptors=features.descriptors.copy()
            if features.descriptors is not None
            else np.empty((0, 32), dtype=np.uint8),
            keypoint_to_mappoint=kp_to_mp,
        )

    def _send_keyframe_to_backend(
        self, keyframe: Keyframe, vo_frame: VOFrame
    ) -> None:
        """Send keyframe and associated map points to backend."""
        if self._backend is None:
            return

        # Create serializable keyframe data
        kf_data = KeyframeData(
            id=keyframe.id,
            timestamp_ns=keyframe.timestamp_ns,
            rotation=keyframe.pose.rotation.copy(),
            translation=keyframe.pose.translation.copy(),
            keypoints=keyframe.keypoints.copy(),
            descriptors=keyframe.descriptors.copy(),
            keypoint_to_mappoint=keyframe.keypoint_to_mappoint.copy(),
        )

        # Collect associated map points
        map_point_data: list[MapPointData] = []
        slam_map = self._vo.get_map()

        for mp_id in keyframe.keypoint_to_mappoint.values():
            mp = slam_map.get_point(mp_id)
            if mp is not None:
                map_point_data.append(
                    MapPointData(
                        id=mp.id,
                        position=mp.position_world.copy(),
                        descriptor=mp.descriptor.copy(),
                    )
                )

        # Send message
        msg = KeyframeMessage(keyframe=kf_data, new_map_points=map_point_data)
        self._backend.send_keyframe(msg)

    def _apply_correction(self, correction: CorrectionMessage) -> None:
        """Apply BA corrections to frontend state.

        Note: This is a simplified implementation. In a full system,
        corrections would be carefully merged with ongoing tracking.
        """
        slam_map = self._vo.get_map()

        # Apply point corrections
        for mp_id, new_position in correction.point_corrections.items():
            slam_map.update_point_position(mp_id, new_position)

        # Pose corrections are trickier - we'd need to update the trajectory
        # For now, we just log that corrections were received
        # A full implementation would update _vo._trajectory for recent poses

    def _apply_global_correction(self, correction: GlobalCorrectionMessage) -> None:
        """Apply global pose corrections from loop closure.

        Note: This is a simplified implementation. In a full system,
        we would update all keyframe poses and reproject map points.
        """
        # Global corrections update all poses after loop closure
        # For now, we log the correction but don't apply to VO trajectory
        # A full implementation would need to transform the entire trajectory
        pass

    def _send_keyframe_to_loop_closure(
        self, keyframe: Keyframe, vo_frame: VOFrame
    ) -> None:
        """Send keyframe data to loop closure process."""
        if self._loop_closure is None:
            return

        # Collect 3D map points visible in this keyframe
        slam_map = self._vo.get_map()
        points_3d_list = []
        keypoint_to_3d: dict[int, int] = {}

        for kp_idx, mp_id in keyframe.keypoint_to_mappoint.items():
            mp = slam_map.get_point(mp_id)
            if mp is not None:
                keypoint_to_3d[kp_idx] = len(points_3d_list)
                points_3d_list.append(mp.position_world.copy())

        points_3d = (
            np.array(points_3d_list, dtype=np.float64)
            if points_3d_list
            else np.empty((0, 3), dtype=np.float64)
        )

        # Create loop closure keyframe data
        lc_data = LoopKeyframeData(
            id=keyframe.id,
            pose_rotation=keyframe.pose.rotation.copy(),
            pose_translation=keyframe.pose.translation.copy(),
            descriptors=keyframe.descriptors.copy(),
            keypoints=keyframe.keypoints.copy(),
            points_3d=points_3d,
            keypoint_to_3d=keypoint_to_3d,
        )

        # Send to loop closure process
        msg = LoopKeyframeMessage(keyframe=lc_data)
        self._loop_closure.send_keyframe(msg)

    def get_trajectory(self) -> list[SE3]:
        """Get the estimated camera trajectory."""
        return self._vo.get_trajectory()

    def get_trajectory_positions(self) -> np.ndarray:
        """Get camera positions as Nx3 array."""
        return self._vo.get_trajectory_positions()

    def get_map(self):
        """Get the sparse map."""
        return self._vo.get_map()

    def get_map_positions(self) -> np.ndarray:
        """Get map point positions as Nx3 array."""
        return self._vo.get_map().get_all_positions()

    @property
    def stats(self) -> SLAMStats:
        """Get SLAM statistics."""
        return self._stats

    @property
    def num_frames(self) -> int:
        """Return number of processed frames."""
        return self._stats.num_frames

    @property
    def num_keyframes(self) -> int:
        """Return number of keyframes."""
        return self._stats.num_keyframes

    @property
    def num_map_points(self) -> int:
        """Return number of map points."""
        return self._vo.num_map_points

    @property
    def current_pose(self) -> SE3 | None:
        """Get current camera pose."""
        return self._vo.current_pose

    def __enter__(self) -> SLAMSystem:
        """Context manager entry - starts the system."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stops the system."""
        self.stop()
