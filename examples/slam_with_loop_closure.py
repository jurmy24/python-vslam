#!/usr/bin/env python3
"""Demo script for full SLAM with loop closure detection.

This demo runs the complete SLAM pipeline with three processes:
- Frontend: Visual Odometry (frame-to-frame tracking) ~20 Hz
- Backend: Local Bundle Adjustment (pose refinement) ~10 Hz
- Loop Closure: Place recognition + pose graph optimization ~2-5 Hz

When the camera revisits a previously seen location, loop closure
detects the match and optimizes the entire trajectory to correct
accumulated drift.

Usage:
    uv run python examples/slam_with_loop_closure.py

Requirements:
    - Trained vocabulary file at data/vocabulary.npz
    - EuRoC dataset at data/euroc/MH_01_easy/mav0

To train a vocabulary (if not already done):
    uv run python scripts/train_vocabulary.py
"""

from pathlib import Path

import numpy as np

from vslam import DatasetReader, GroundTruthReader, RerunVisualizer, SLAMSystem, TrackingStatus


def main() -> None:
    """Run the SLAM demo with loop closure detection."""
    # Configuration
    dataset_path = "data/euroc/MH_01_easy/mav0"
    vocabulary_path = Path("data/vocabulary.npz")
    n_features = 1000
    max_frames = None  # Set to int to limit frames
    enable_backend = True
    enable_loop_closure = True

    # Check vocabulary exists
    if enable_loop_closure and not vocabulary_path.exists():
        print(f"ERROR: Vocabulary not found at {vocabulary_path}")
        print("Train a vocabulary first with:")
        print("  uv run python scripts/train_vocabulary.py")
        return

    # Initialize
    print("=" * 80)
    print("SLAM WITH LOOP CLOSURE")
    print("=" * 80)
    print()
    print("Initializing SLAM system...")
    print(f"  Frontend VO:    Enabled (real-time tracking)")
    print(f"  Backend BA:     {'Enabled' if enable_backend else 'Disabled'}")
    print(f"  Loop Closure:   {'Enabled' if enable_loop_closure else 'Disabled'}")
    if enable_loop_closure:
        print(f"  Vocabulary:     {vocabulary_path}")
    print()

    reader = DatasetReader(dataset_path)
    visualizer = RerunVisualizer("python-vslam-loop-closure")

    # Load ground truth for error visualization
    try:
        gt_reader = GroundTruthReader(dataset_path)
        print(f"  Ground Truth:   Loaded ({len(gt_reader)} poses)")
    except FileNotFoundError:
        gt_reader = None
        print("  Ground Truth:   Not found (skipping GT comparison)")

    # Track loop closure positions for visualization
    loop_closure_positions: list[tuple[np.ndarray, np.ndarray]] = []

    # Track ground truth positions for visualization
    gt_positions: list[np.ndarray] = []
    gt_errors: list[float] = []
    gt_errors_xyz: list[np.ndarray] = []  # Per-axis errors (x, y, z)

    # Use context manager to ensure clean shutdown
    with SLAMSystem.from_dataset_path(
        dataset_path,
        n_features=n_features,
        enable_backend=enable_backend,
        enable_loop_closure=enable_loop_closure,
        vocabulary_path=str(vocabulary_path) if enable_loop_closure else None,
    ) as slam:
        print(f"Processing {len(reader)} frames...")
        print()

        # Column headers
        if gt_reader:
            print(
                f"{'Frame':>6} {'Status':^12} {'Track':>5} {'Inlr':>5} "
                f"{'KF':>3} {'Map':>6} {'BA':>3} {'LC':>3} | "
                f"{'Total':>7} | {'GT Err':>7} | {'dX':>6} {'dY':>6} {'dZ':>6} | {'Position'}"
            )
            print("-" * 155)
        else:
            print(
                f"{'Frame':>6} {'Status':^12} {'Track':>5} {'Inlr':>5} "
                f"{'KF':>3} {'Map':>6} {'BA':>3} {'LC':>3} | "
                f"{'Total':>7} | {'Position'}"
            )
            print("-" * 110)

        # Statistics
        lost_count = 0
        keyframe_count = 0
        ba_count = 0
        loop_closure_count = 0
        timing_total = 0.0

        # Process frames
        for i, (left, right, timestamp_ns) in enumerate(reader):
            if max_frames is not None and i >= max_frames:
                break

            result = slam.process_frame(left, right, timestamp_ns)
            vo = result.vo_frame

            # Get ground truth for this frame
            gt_error = None
            gt_error_xyz = None
            if gt_reader:
                gt_pos = gt_reader.get_position_at(timestamp_ns)
                if gt_pos is not None:
                    gt_positions.append(gt_pos.copy())
                    error_vec = vo.position - gt_pos  # Per-axis signed error
                    gt_error_xyz = error_vec
                    gt_error = float(np.linalg.norm(error_vec))
                    gt_errors.append(gt_error)
                    gt_errors_xyz.append(error_vec.copy())

            # Visualize (less frequently to reduce overhead)
            if i % 5 == 0:
                visualizer.log_vo_frame(vo)
                positions = slam.get_trajectory_positions()
                visualizer.log_trajectory(positions)

                # Log ground truth trajectory
                if len(gt_positions) > 1:
                    visualizer.log_ground_truth_trajectory(np.array(gt_positions))

                # Log map points
                map_positions = slam.get_map_positions()
                if len(map_positions) > 0:
                    visualizer.log_map_points(map_positions)

                # Re-visualize all loop closures
                for from_pos, to_pos in loop_closure_positions:
                    visualizer.log_loop_closure(from_pos, to_pos)

            # Track statistics
            if vo.tracking_status == TrackingStatus.LOST:
                lost_count += 1

            if result.is_keyframe:
                keyframe_count += 1

            if result.ba_applied:
                ba_count += 1

            # Handle loop closure detection
            if result.loop_closure_detected:
                loop_closure_count += 1
                # Store the positions for visualization
                trajectory = slam.get_trajectory_positions()
                if len(trajectory) > max(result.loop_from_id, result.loop_to_id):
                    from_pos = trajectory[result.loop_from_id]
                    to_pos = trajectory[result.loop_to_id]
                    loop_closure_positions.append((from_pos.copy(), to_pos.copy()))
                    print()
                    print(
                        f"  ╔══════════════════════════════════════════════════════════════╗"
                    )
                    print(
                        f"  ║  LOOP CLOSURE DETECTED!  KF {result.loop_from_id} → KF {result.loop_to_id:<4}               ║"
                    )
                    print(
                        f"  ╚══════════════════════════════════════════════════════════════╝"
                    )
                    print()

            timing_total += vo.timing.total_ms

            # Print progress every 20 frames or on special events
            should_print = (
                (i % 20 == 0)
                or result.is_keyframe
                or result.ba_applied
                or result.loop_closure_detected
                or (vo.tracking_status == TrackingStatus.LOST)
            )

            if should_print:
                pos = vo.position
                status = vo.tracking_status.value
                kf_marker = "*" if result.is_keyframe else " "
                ba_marker = "+" if result.ba_applied else " "
                lc_marker = "L" if result.loop_closure_detected else " "

                if gt_reader and gt_error is not None and gt_error_xyz is not None:
                    dx, dy, dz = gt_error_xyz
                    print(
                        f"{i:6d} {status:^12} {vo.tracked_points:5d} "
                        f"{int(vo.inlier_ratio * 100):4d}% "
                        f"{kf_marker:>3} {slam.num_map_points:6d} {ba_marker:>3} {lc_marker:>3} | "
                        f"{vo.timing.total_ms:6.1f}ms | "
                        f"{gt_error:6.2f}m | "
                        f"{dx:+6.2f} {dy:+6.2f} {dz:+6.2f} | "
                        f"[{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}]"
                    )
                else:
                    print(
                        f"{i:6d} {status:^12} {vo.tracked_points:5d} "
                        f"{int(vo.inlier_ratio * 100):4d}% "
                        f"{kf_marker:>3} {slam.num_map_points:6d} {ba_marker:>3} {lc_marker:>3} | "
                        f"{vo.timing.total_ms:6.1f}ms | "
                        f"[{pos[0]:7.2f}, {pos[1]:7.2f}, {pos[2]:7.2f}]"
                    )

        # Final statistics
        n_frames = slam.num_frames
        print()
        print("=" * 80)
        print("SLAM WITH LOOP CLOSURE - SUMMARY")
        print("=" * 80)
        print()
        print("Processing Statistics:")
        print(f"  Frames processed:   {n_frames}")
        print(f"  Keyframes created:  {keyframe_count}")
        print(f"  Map points:         {slam.num_map_points}")
        print(f"  Lost count:         {lost_count} ({100*lost_count/max(n_frames,1):.1f}%)")
        print()
        print("Backend Statistics:")
        print(f"  BA corrections:     {ba_count}")
        print(f"  Loop closures:      {loop_closure_count}")
        print()
        print("Trajectory Statistics:")
        print(f"  Distance traveled:  {slam.stats.total_distance:.2f} m")
        print()

        # Ground truth comparison
        if gt_reader and gt_errors and gt_errors_xyz:
            gt_errors_arr = np.array(gt_errors)
            gt_errors_xyz_arr = np.array(gt_errors_xyz)
            ate = float(np.sqrt(np.mean(gt_errors_arr**2)))  # RMSE

            # Per-axis statistics
            mean_xyz = np.mean(gt_errors_xyz_arr, axis=0)
            std_xyz = np.std(gt_errors_xyz_arr, axis=0)
            final_xyz = gt_errors_xyz_arr[-1]

            print("Ground Truth Comparison:")
            print(f"  Final position error:  {gt_errors[-1]:.3f} m")
            print(f"  ATE (RMSE):            {ate:.3f} m")
            print(f"  Max error:             {np.max(gt_errors_arr):.3f} m")
            print()
            print("  Per-axis error (X, Y, Z):")
            print(f"    Final:  [{final_xyz[0]:+.3f}, {final_xyz[1]:+.3f}, {final_xyz[2]:+.3f}] m")
            print(f"    Mean:   [{mean_xyz[0]:+.3f}, {mean_xyz[1]:+.3f}, {mean_xyz[2]:+.3f}] m")
            print(f"    Std:    [{std_xyz[0]:.3f}, {std_xyz[1]:.3f}, {std_xyz[2]:.3f}] m")
            print()

        print(
            f"Average frame time: {timing_total/n_frames:.1f} ms ({n_frames/(timing_total/1000):.1f} Hz)"
        )
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
        print("  LC column: 'L' indicates loop closure detected")
        print()
        print("In Rerun viewer:")
        print("  - Yellow line:  Estimated trajectory")
        print("  - Green line:   Ground truth trajectory")
        print("  - Red lines:    Loop closure edges")
        print("  - Magenta dots: Loop closure endpoints")


if __name__ == "__main__":
    main()
