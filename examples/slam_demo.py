#!/usr/bin/env python3
"""Demo script for full SLAM with visual odometry and bundle adjustment.

This demo runs the complete SLAM pipeline:
- Frontend: Visual Odometry (frame-to-frame tracking)
- Backend: Local Bundle Adjustment (pose refinement)

The backend runs in a separate process for optimal performance.

Usage:
    uv run python examples/slam_demo.py
"""

import numpy as np

from vslam import DatasetReader, RerunVisualizer, SLAMSystem, TrackingStatus


def main() -> None:
    """Run the SLAM demo with backend bundle adjustment."""
    # Configuration
    dataset_path = "data/euroc/MH_01_easy/mav0"
    n_features = 1000
    max_frames = None  # Set to int to limit frames
    enable_backend = True  # Set to False to run VO-only

    # Initialize
    print("Initializing SLAM system...")
    print(f"  Backend BA: {'Enabled' if enable_backend else 'Disabled'}")
    reader = DatasetReader(dataset_path)
    visualizer = RerunVisualizer("python-vslam")

    # Use context manager to ensure clean shutdown
    with SLAMSystem.from_dataset_path(
        dataset_path,
        n_features=n_features,
        enable_backend=enable_backend,
    ) as slam:
        print(f"Processing {len(reader)} frames...")
        print()

        # Column headers
        print(
            f"{'Frame':>6} {'Status':^12} {'Track':>5} {'Inlr':>5} "
            f"{'KF':>3} {'Map':>6} {'BA':>3} | "
            f"{'Total':>7} | {'Position'}"
        )
        print("-" * 100)

        # Statistics
        lost_count = 0
        keyframe_count = 0
        ba_count = 0
        timing_total = 0.0

        # Process frames
        for i, (left, right, timestamp_ns) in enumerate(reader):
            if max_frames is not None and i >= max_frames:
                break

            result = slam.process_frame(left, right, timestamp_ns)
            vo = result.vo_frame

            # Visualize (less frequently to reduce overhead)
            if i % 5 == 0:
                visualizer.log_vo_frame(vo)
                positions = slam.get_trajectory_positions()
                visualizer.log_trajectory(positions)

                # Log map points
                map_positions = slam.get_map_positions()
                if len(map_positions) > 0:
                    visualizer.log_map_points(map_positions)

            # Track statistics
            if vo.tracking_status == TrackingStatus.LOST:
                lost_count += 1

            if result.is_keyframe:
                keyframe_count += 1

            if result.ba_applied:
                ba_count += 1

            timing_total += vo.timing.total_ms

            # Print progress every 20 frames or on special events
            should_print = (
                (i % 20 == 0)
                or result.is_keyframe
                or result.ba_applied
                or (vo.tracking_status == TrackingStatus.LOST)
            )

            if should_print:
                pos = vo.position
                status = vo.tracking_status.value
                kf_marker = "*" if result.is_keyframe else " "
                ba_marker = "+" if result.ba_applied else " "

                print(
                    f"{i:6d} {status:^12} {vo.tracked_points:5d} "
                    f"{int(vo.inlier_ratio * 100):4d}% "
                    f"{kf_marker:>3} {slam.num_map_points:6d} {ba_marker:>3} | "
                    f"{vo.timing.total_ms:6.1f}ms | "
                    f"[{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}]"
                )

        # Final statistics
        n_frames = slam.num_frames
        print()
        print("=" * 80)
        print("SLAM SUMMARY")
        print("=" * 80)
        print(f"Frames processed:   {n_frames}")
        print(f"Keyframes created:  {keyframe_count}")
        print(f"Map points:         {slam.num_map_points}")
        print(f"Lost count:         {lost_count} ({100*lost_count/max(n_frames,1):.1f}%)")
        print(f"BA corrections:     {ba_count}")
        print(f"Distance traveled:  {slam.stats.total_distance:.2f} m")
        print()
        print(f"Average frame time: {timing_total/n_frames:.1f} ms ({n_frames/(timing_total/1000):.1f} Hz)")
        print()

        if slam.current_pose is not None:
            pos = slam.current_pose.position
            print(f"Final position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

        print()
        print("Done! Check Rerun viewer.")
        print()
        print("Legend:")
        print("  KF column: '*' indicates new keyframe")
        print("  BA column: '+' indicates BA correction applied")


if __name__ == "__main__":
    main()
