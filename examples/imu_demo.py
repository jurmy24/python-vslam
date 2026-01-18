#!/usr/bin/env python3
"""Demo script for IMU-based dead-reckoning odometry.

Demonstrates IMU integration and compares trajectory to ground truth
to show the drift characteristics of IMU-only odometry.

Usage:
    uv run python examples/imu_demo.py
"""

import numpy as np

from vslam import DatasetReader, GroundTruthReader, IMUOdometry, IMUTrackingStatus


def main() -> None:
    """Run the IMU odometry demo."""
    # Configuration
    dataset_path = "data/euroc/MH_01_easy/mav0"
    max_frames = None  # Set to int to limit frames

    # Initialize
    print("Initializing IMU odometry pipeline...")
    print("=" * 80)
    reader = DatasetReader(dataset_path)
    imu_odom = IMUOdometry.from_dataset_path(dataset_path, use_gt_initial_state=True)
    gt_reader = GroundTruthReader(dataset_path)

    print(f"Loaded {len(reader)} camera frames")
    print(f"Initial state from ground truth:")
    print(f"  Position:   {imu_odom._initial_pose.translation}")
    print(f"  Velocity:   {imu_odom._initial_velocity}")
    print(f"  Gyro bias:  {imu_odom._initial_bias_gyro}")
    print(f"  Accel bias: {imu_odom._initial_bias_accel}")
    print()

    # Column headers
    print(
        f"{'Frame':>6} {'Status':^12} {'IMU#':>5} | "
        f"{'Time':>7} | "
        f"{'IMU Position':^30} | "
        f"{'GT Position':^30} | "
        f"{'Error':>7}"
    )
    print("-" * 120)

    # Statistics
    total_distance_imu = 0.0
    total_distance_gt = 0.0
    prev_imu_pos = None
    prev_gt_pos = None
    errors: list[float] = []
    timing_totals = {"integration": 0.0, "total": 0.0}

    # Process frames at camera rate
    for i, (_, _, timestamp_ns) in enumerate(reader):
        if max_frames is not None and i >= max_frames:
            break

        # Process IMU up to this timestamp
        frame = imu_odom.process_timestamp(timestamp_ns)

        # Get ground truth at same timestamp
        gt_pose = gt_reader.get_pose_at(timestamp_ns)
        gt_pos = gt_pose.translation if gt_pose else None

        # Calculate error
        error = np.nan
        if gt_pos is not None:
            error = float(np.linalg.norm(frame.position - gt_pos))
            errors.append(error)

        # Track distances
        if prev_imu_pos is not None:
            total_distance_imu += float(np.linalg.norm(frame.position - prev_imu_pos))
        prev_imu_pos = frame.position.copy()

        if gt_pos is not None and prev_gt_pos is not None:
            total_distance_gt += float(np.linalg.norm(gt_pos - prev_gt_pos))
        if gt_pos is not None:
            prev_gt_pos = gt_pos.copy()

        # Accumulate timing
        timing_totals["integration"] += frame.timing.integration_ms
        timing_totals["total"] += frame.timing.total_ms

        # Print progress every 50 frames
        if i % 50 == 0:
            imu_str = f"[{frame.position[0]:8.3f}, {frame.position[1]:8.3f}, {frame.position[2]:8.3f}]"
            if gt_pos is not None:
                gt_str = f"[{gt_pos[0]:8.3f}, {gt_pos[1]:8.3f}, {gt_pos[2]:8.3f}]"
            else:
                gt_str = "[       N/A       ]"

            print(
                f"{i:6d} {frame.tracking_status.value:^12} {frame.num_integrated_measurements:5d} | "
                f"{frame.timing.total_ms:6.2f}ms | "
                f"{imu_str} | "
                f"{gt_str} | "
                f"{error:7.3f}m"
            )

    # Final statistics
    n_frames = len(imu_odom.get_trajectory())
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Frames processed:     {n_frames}")
    print(f"Distance traveled:")
    print(f"  IMU estimate:       {total_distance_imu:.2f} m")
    print(f"  Ground truth:       {total_distance_gt:.2f} m")
    print()

    if errors:
        print("Position error vs ground truth:")
        print(f"  Min:    {min(errors):7.3f} m")
        print(f"  Max:    {max(errors):7.3f} m")
        print(f"  Mean:   {np.mean(errors):7.3f} m")
        print(f"  Median: {np.median(errors):7.3f} m")
        print(f"  Final:  {errors[-1]:7.3f} m")
        print()
        print(f"  Relative error: {errors[-1]/total_distance_gt*100:.2f}% of distance traveled")

    print()
    print("Average timing per frame:")
    print(f"  Integration: {timing_totals['integration']/n_frames:6.3f} ms")
    print(f"  Total:       {timing_totals['total']/n_frames:6.3f} ms ({n_frames/(timing_totals['total']/1000):.0f} Hz theoretical)")
    print()

    if imu_odom.current_pose is not None:
        pos = imu_odom.current_pose.translation
        print(f"Final IMU position: [{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

    if gt_pos is not None:
        print(f"Final GT position:  [{gt_pos[0]:.3f}, {gt_pos[1]:.3f}, {gt_pos[2]:.3f}]")

    print()
    print("Note: IMU-only odometry drifts over time due to sensor noise and bias errors.")
    print("For accurate localization, combine with visual odometry (VIO) or GPS corrections.")


if __name__ == "__main__":
    main()
