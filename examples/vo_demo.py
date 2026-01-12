#!/usr/bin/env python3
"""Demo script for visual odometry with timing diagnostics.

Usage:
    uv run python examples/vo_demo.py
"""

import numpy as np

from vslam import DatasetReader, RerunVisualizer, TrackingStatus, VisualOdometry


def main() -> None:
    """Run the visual odometry demo."""
    # Configuration
    dataset_path = "data/euroc/MH_01_easy/mav0"
    n_features = 1000
    max_frames = None  # Set to int to limit frames

    # Initialize
    print("Initializing visual odometry pipeline...")
    reader = DatasetReader(dataset_path)
    vo = VisualOdometry.from_dataset_path(dataset_path, n_features=n_features)
    visualizer = RerunVisualizer("python-vslam-vo")

    print(f"Processing {len(reader)} frames...")
    print()

    # Column headers
    print(
        f"{'Frame':>6} {'Status':^12} {'Track':>5} {'Corr':>5} {'Inlr':>5} "
        f"{'New':>5} {'Map':>6} | "
        f"{'Stereo':>7} {'Match':>6} {'PnP':>6} {'Map':>6} {'Total':>7} | "
        f"{'Position'}"
    )
    print("-" * 120)

    # Statistics
    lost_count = 0
    total_distance = 0.0
    prev_position = None
    timing_totals = {"stereo": 0.0, "match": 0.0, "pnp": 0.0, "map": 0.0, "total": 0.0}

    # Process frames
    for i, (left, right, timestamp_ns) in enumerate(reader):
        if max_frames is not None and i >= max_frames:
            break

        result = vo.process_frame(left, right, timestamp_ns)

        # Visualize (less frequently to reduce overhead)
        if i % 5 == 0:
            visualizer.log_vo_frame(result)
            positions = vo.get_trajectory_positions()
            visualizer.log_trajectory(positions)

        # Track statistics
        if result.tracking_status == TrackingStatus.LOST:
            lost_count += 1

        if prev_position is not None and result.is_tracking_ok:
            total_distance += float(np.linalg.norm(result.position - prev_position))
        prev_position = result.position.copy()

        # Accumulate timing
        t = result.timing
        timing_totals["stereo"] += t.stereo_ms
        timing_totals["match"] += t.matching_ms
        timing_totals["pnp"] += t.pnp_ms
        timing_totals["map"] += t.map_update_ms
        timing_totals["total"] += t.total_ms

        # Print progress every 20 frames or on status change
        should_print = (i % 20 == 0) or (result.tracking_status == TrackingStatus.LOST)
        if should_print:
            pos = result.position
            status = result.tracking_status.value
            print(
                f"{i:6d} {status:^12} {result.tracked_points:5d} "
                f"{result.num_correspondences:5d} "
                f"{int(result.inlier_ratio * 100):4d}% "
                f"{result.new_points:5d} {vo.num_map_points:6d} | "
                f"{t.stereo_ms:6.1f}ms {t.matching_ms:5.1f}ms {t.pnp_ms:5.1f}ms "
                f"{t.map_update_ms:5.1f}ms {t.total_ms:6.1f}ms | "
                f"[{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}]"
            )

    # Final statistics
    n_frames = vo.num_frames
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Frames processed:  {n_frames}")
    print(f"Map points:        {vo.num_map_points}")
    print(f"Lost count:        {lost_count} ({100*lost_count/max(n_frames,1):.1f}%)")
    print(f"Distance traveled: {total_distance:.2f} m")
    print()
    print("Average timing per frame:")
    print(f"  Stereo:    {timing_totals['stereo']/n_frames:6.1f} ms")
    print(f"  Matching:  {timing_totals['match']/n_frames:6.1f} ms")
    print(f"  PnP:       {timing_totals['pnp']/n_frames:6.1f} ms")
    print(f"  Map:       {timing_totals['map']/n_frames:6.1f} ms")
    print(f"  Total:     {timing_totals['total']/n_frames:6.1f} ms ({n_frames/(timing_totals['total']/1000):.1f} Hz)")
    print()

    if vo.current_pose is not None:
        pos = vo.current_pose.position
        print(f"Final position: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")

    print()
    print("Done! Check Rerun viewer.")


if __name__ == "__main__":
    main()
