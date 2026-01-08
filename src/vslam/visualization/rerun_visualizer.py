"""Rerun-based visualization for stereo SLAM."""

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from ..frontend.stereo_frontend import StereoFrame


class RerunVisualizer:
    """Rerun-based visualization for stereo visual SLAM.

    Provides real-time visualization of:
    - Stereo camera images (rectified)
    - Detected 2D features
    - Stereo matches
    - Triangulated 3D point cloud

    Entity hierarchy:
        camera/
            left/
                image       - Rectified left image
                features    - All detected features (green)
                matched     - Matched features (red)
            right/
                image       - Rectified right image
                features    - All detected features (green)
                matched     - Matched features (red)
        world/
            points          - 3D point cloud (colored by depth)
    """

    def __init__(self, app_name: str = "python-vslam", spawn: bool = True) -> None:
        """Initialize Rerun visualization.

        Args:
            app_name: Name for the Rerun application window
            spawn: If True, automatically spawn the Rerun viewer
        """
        rr.init(app_name, spawn=spawn)
        self._setup_coordinate_system()
        self._setup_layout()

    def _setup_coordinate_system(self) -> None:
        """Configure the 3D coordinate system.

        Uses a right-handed coordinate system with Y pointing down,
        which matches typical camera conventions (X-right, Y-down, Z-forward).
        """
        rr.log("world", rr.ViewCoordinates.RDF, static=True)

    def _setup_layout(self) -> None:
        """Configure the viewer layout"""
        blueprint = rrb.Blueprint(
            rrb.Vertical(
                contents=[
                    rrb.Horizontal(
                        contents=[
                            rrb.Spatial2DView(
                                name="Left Camera", origin="camera/left/image"
                            ),
                            rrb.Spatial2DView(
                                name="Right Camera", origin="camera/right/image"
                            ),
                        ]
                    ),
                    rrb.Spatial3DView(name="3D Points", origin="world"),
                ]
            )
        )
        rr.send_blueprint(blueprint)

    def log_frame(self, frame: StereoFrame) -> None:
        """Log a processed stereo frame to Rerun.

        Args:
            frame: StereoFrame from the stereo frontend
        """
        # Set timestamp for this frame
        timestamp_sec = frame.timestamp_ns / 1e9
        rr.set_time("timestamp", duration=timestamp_sec)

        # Log all components
        self._log_images(frame)
        self._log_features(frame)
        self._log_matches(frame)
        self._log_3d_points(frame)

    def _log_images(self, frame: StereoFrame) -> None:
        """Log rectified stereo images."""
        rr.log("camera/left/image", rr.Image(frame.left_rectified))
        rr.log("camera/right/image", rr.Image(frame.right_rectified))

    def _log_features(self, frame: StereoFrame) -> None:
        """Log detected 2D features in both images."""
        # Left image features (green)
        if len(frame.features_left) > 0:
            rr.log(
                "camera/left/features",
                rr.Points2D(
                    frame.features_left.points,
                    colors=[[0, 255, 0]],  # Green
                    radii=3.0,
                ),
            )

        # Right image features (green)
        if len(frame.features_right) > 0:
            rr.log(
                "camera/right/features",
                rr.Points2D(
                    frame.features_right.points,
                    colors=[[0, 255, 0]],  # Green
                    radii=3.0,
                ),
            )

    def _log_matches(self, frame: StereoFrame) -> None:
        """Log matched features (red) overlaid on the images."""
        if len(frame.matches) > 0:
            # Matched points in left image (red)
            rr.log(
                "camera/left/matched",
                rr.Points2D(
                    frame.matches.pts_left,
                    colors=[[255, 0, 0]],  # Red
                    radii=4.0,
                ),
            )

            # Matched points in right image (red)
            rr.log(
                "camera/right/matched",
                rr.Points2D(
                    frame.matches.pts_right,
                    colors=[[255, 0, 0]],  # Red
                    radii=4.0,
                ),
            )

    def _log_3d_points(self, frame: StereoFrame) -> None:
        """Log triangulated 3D point cloud with depth-based coloring."""
        if len(frame.points_3d) == 0:
            return

        points = frame.points_3d

        # Filter out invalid points (NaN, Inf, or extreme values)
        valid_mask = np.isfinite(points).all(axis=1) & (np.abs(points) < 100).all(
            axis=1
        )  # Filter extreme outliers
        valid_points = points[valid_mask]

        if len(valid_points) == 0:
            return

        # Color by depth (z-coordinate)
        depths = valid_points[:, 2]
        depth_min, depth_max = np.percentile(depths, [5, 95])  # Robust range
        depth_range = max(depth_max - depth_min, 0.1)  # Avoid division by zero

        # Normalize depths to [0, 1]
        normalized_depths = np.clip((depths - depth_min) / depth_range, 0, 1)

        # Create colormap: blue (close) -> green -> red (far)
        colors = np.zeros((len(valid_points), 3), dtype=np.uint8)
        colors[:, 0] = (normalized_depths * 255).astype(
            np.uint8
        )  # Red increases with depth
        colors[:, 1] = ((1 - np.abs(normalized_depths - 0.5) * 2) * 255).astype(
            np.uint8
        )  # Green peaks at mid-depth
        colors[:, 2] = ((1 - normalized_depths) * 255).astype(
            np.uint8
        )  # Blue decreases with depth

        rr.log(
            "world/points",
            rr.Points3D(valid_points, colors=colors, radii=0.02),
        )

    def log_camera_pose(
        self,
        position: np.ndarray,
        rotation: np.ndarray,
        entity_path: str = "world/camera",
    ) -> None:
        """Log camera pose as a 3D transform (for future use).

        Args:
            position: 3D position [x, y, z]
            rotation: 3x3 rotation matrix
            entity_path: Rerun entity path for the camera
        """
        rr.log(
            entity_path,
            rr.Transform3D(
                translation=position,
                mat3x3=rotation,
            ),
        )
