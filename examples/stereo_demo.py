#!/usr/bin/env python3
"""Demo script for stereo frontend visualization.

This script processes the EuRoC dataset through the stereo frontend
and visualizes the results in Rerun, including:
- Rectified stereo images
- Detected ORB features (green dots)
- Stereo matches (red dots)
- Triangulated 3D point cloud (colored by depth)

Usage:
    uv run python examples/stereo_demo.py

Requirements:
    - EuRoC dataset downloaded to data/euroc/MH_01_easy/mav0/
"""

from vslam import DatasetReader, StereoFrontend, RerunVisualizer


def main() -> None:
    """Run the stereo frontend demo."""
    # Configuration
    dataset_path = "data/euroc/MH_01_easy/mav0"
    n_features = 1000
    max_frames = None  # Set to int to limit frames, None for all

    # Initialize components
    print("Initializing stereo frontend...")
    reader = DatasetReader(
        dataset_path
    )  # Could undistort images already here instead of in the StereoFrontend
    frontend = StereoFrontend.from_dataset_path(dataset_path, n_features=n_features)
    visualizer = RerunVisualizer("python-vslam")

    print(f"Stereo baseline: {frontend.baseline_meters:.4f} m")
    print(f"Processing {len(reader)} frames...")
    print()

    # Process frames
    for i, (left, right, timestamp_ns) in enumerate(reader):
        if max_frames is not None and i >= max_frames:
            break

        # Run stereo frontend
        frame = frontend.process_frame(left, right, timestamp_ns)

        # Visualize
        visualizer.log_frame(frame)

        # Print progress every 50 frames
        if i % 50 == 0:
            print(
                f"Frame {i:4d}: "
                f"{frame.num_features_left:4d} features, "
                f"{frame.num_matches:3d} matches, "
                f"{frame.num_3d_points:3d} 3D points"
                f"Average depth: {frame.points_3d.mean():.2f} m"
            )

    print()
    print("Done! Check the Rerun viewer for visualization.")


if __name__ == "__main__":
    main()
