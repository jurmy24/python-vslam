"""Backend process for running bundle adjustment in a separate process.

The backend process receives keyframe messages from the frontend,
runs local bundle adjustment, and sends pose/point corrections back.

This separation allows the frontend to run at full frame rate while
BA runs asynchronously on a separate CPU core.
"""

from __future__ import annotations

import multiprocessing as mp
from queue import Empty
from typing import TYPE_CHECKING

import numpy as np

from .keyframe import Keyframe
from .local_mapping import LocalBAConfig, LocalMapping
from .messages import (
    CorrectionMessage,
    KeyframeData,
    KeyframeMessage,
    MapPointData,
    ShutdownMessage,
)

if TYPE_CHECKING:
    from ..frontend.vo.map_point import Map, MapPoint
    from ..frontend.pose import SE3


def _keyframe_data_to_keyframe(data: KeyframeData) -> Keyframe:
    """Convert KeyframeData (serializable) to Keyframe object.

    Args:
        data: Serializable keyframe data from message

    Returns:
        Keyframe object for use in backend
    """
    from ..frontend.pose import SE3

    pose = SE3(rotation=data.rotation, translation=data.translation)

    return Keyframe(
        id=data.id,
        timestamp_ns=data.timestamp_ns,
        pose=pose,
        keypoints=data.keypoints,
        descriptors=data.descriptors,
        keypoint_to_mappoint=data.keypoint_to_mappoint.copy(),
    )


def _mappoint_data_to_mappoint(data: MapPointData) -> MapPoint:
    """Convert MapPointData (serializable) to MapPoint object.

    Args:
        data: Serializable map point data from message

    Returns:
        MapPoint object
    """
    from ..frontend.vo.map_point import MapPoint

    return MapPoint(
        id=data.id,
        position_world=data.position,
        descriptor=data.descriptor,
        observations=[],  # Observations not needed for BA
    )


class BackendProcess:
    """Manages the backend bundle adjustment process.

    Handles message passing between frontend and backend processes.
    The actual BA computation runs in a separate process.
    """

    def __init__(self, camera_matrix: np.ndarray) -> None:
        """Initialize backend process manager.

        Args:
            camera_matrix: 3x3 camera intrinsics matrix
        """
        self._camera_matrix = camera_matrix
        self._process: mp.Process | None = None
        self._to_backend: mp.Queue | None = None
        self._to_frontend: mp.Queue | None = None
        self._is_running = False

    def start(self) -> None:
        """Start the backend process."""
        if self._is_running:
            return

        # Create queues for communication
        self._to_backend = mp.Queue(maxsize=10)
        self._to_frontend = mp.Queue(maxsize=10)

        # Start backend process
        self._process = mp.Process(
            target=_backend_main,
            args=(
                self._to_backend,
                self._to_frontend,
                self._camera_matrix,
            ),
            daemon=True,
        )
        self._process.start()
        self._is_running = True

    def stop(self) -> None:
        """Stop the backend process gracefully."""
        if not self._is_running:
            return

        # Send shutdown signal
        try:
            self._to_backend.put(ShutdownMessage(), timeout=1.0)
        except Exception:
            pass

        # Wait for process to finish
        if self._process is not None:
            self._process.join(timeout=2.0)
            if self._process.is_alive():
                self._process.terminate()

        self._is_running = False
        self._process = None

    def send_keyframe(self, msg: KeyframeMessage) -> bool:
        """Send a keyframe to the backend for processing.

        Args:
            msg: Keyframe message to send

        Returns:
            True if message was sent, False if queue is full
        """
        if not self._is_running or self._to_backend is None:
            return False

        try:
            self._to_backend.put_nowait(msg)
            return True
        except Exception:
            return False

    def get_corrections(self, timeout: float = 0.0) -> CorrectionMessage | None:
        """Get pose/point corrections from the backend.

        Args:
            timeout: Time to wait for corrections (0 = non-blocking)

        Returns:
            CorrectionMessage or None if no corrections available
        """
        if not self._is_running or self._to_frontend is None:
            return None

        try:
            if timeout > 0:
                return self._to_frontend.get(timeout=timeout)
            else:
                return self._to_frontend.get_nowait()
        except Empty:
            return None

    @property
    def is_running(self) -> bool:
        """Return True if backend process is running."""
        return self._is_running


def _backend_main(
    to_backend: mp.Queue,
    to_frontend: mp.Queue,
    camera_matrix: np.ndarray,
) -> None:
    """Main loop for the backend process.

    Receives keyframes, runs BA, sends corrections.

    Args:
        to_backend: Queue for receiving messages from frontend
        to_frontend: Queue for sending corrections to frontend
        camera_matrix: Camera intrinsics matrix
    """
    # Initialize local mapping
    config = LocalBAConfig(
        window_size=10,
        min_observations=20,
        max_iterations=50,
        loss_function="huber",
    )
    local_mapping = LocalMapping(camera_matrix=camera_matrix, config=config)

    # Create a simple in-memory map for the backend
    from ..frontend.vo.map_point import Map

    backend_map = Map()
    local_mapping.set_map(backend_map)

    while True:
        try:
            # Wait for message (blocking)
            msg = to_backend.get(timeout=1.0)
        except Empty:
            continue

        if isinstance(msg, ShutdownMessage):
            break

        if isinstance(msg, KeyframeMessage):
            # Convert serializable data to objects
            keyframe = _keyframe_data_to_keyframe(msg.keyframe)

            # Add new map points to backend's map
            for mp_data in msg.new_map_points:
                mp = _mappoint_data_to_mappoint(mp_data)
                # Add to backend map (simplified - just store position)
                backend_map._points[mp.id] = mp
                backend_map._next_point_id = max(
                    backend_map._next_point_id, mp.id + 1
                )

            # Run local bundle adjustment
            result = local_mapping.add_keyframe(keyframe)

            # Send corrections back to frontend
            if result.success and result.ba_result is not None:
                ba = result.ba_result
                if ba.success:
                    correction = CorrectionMessage(
                        pose_corrections=ba.optimized_poses,
                        point_corrections=ba.optimized_points,
                    )
                    try:
                        to_frontend.put_nowait(correction)
                    except Exception:
                        pass  # Queue full, skip this update
