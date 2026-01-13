#!/usr/bin/env python3
"""Train visual vocabulary on EuRoC dataset.

This script trains a Bag of Visual Words vocabulary by:
1. Extracting ORB descriptors from images across all EuRoC sequences
2. Running k-means clustering to find visual word centers
3. Saving the vocabulary for use in loop closure detection

Usage:
    uv run python scripts/train_vocabulary.py
    uv run python scripts/train_vocabulary.py --n-words 2000 --max-images 10000
    uv run python scripts/train_vocabulary.py --data-dir /path/to/euroc

The trained vocabulary is saved to data/vocabulary.npz by default.
"""

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vslam.loop_closure import VisualVocabulary


def collect_descriptors(
    data_dir: Path,
    n_features: int = 500,
    max_images: int | None = None,
    skip_every: int = 1,
) -> np.ndarray:
    """Extract ORB descriptors from all EuRoC sequences.

    Args:
        data_dir: Path to euroc data directory containing sequences
        n_features: Number of ORB features per image
        max_images: Maximum images to process (None for all)
        skip_every: Process every Nth image (for speed)

    Returns:
        Stacked descriptors array, shape (total_descriptors, 32)
    """
    orb = cv2.ORB_create(nfeatures=n_features)
    all_descriptors: list[np.ndarray] = []
    image_count = 0

    # Find all cam0 image directories
    sequence_dirs = sorted(data_dir.iterdir())
    print(f"Found {len(sequence_dirs)} potential sequences in {data_dir}")

    for sequence_dir in sequence_dirs:
        if not sequence_dir.is_dir():
            continue

        cam0_dir = sequence_dir / "mav0" / "cam0" / "data"
        if not cam0_dir.exists():
            continue

        print(f"Processing {sequence_dir.name}...", end=" ", flush=True)
        seq_count = 0

        image_paths = sorted(cam0_dir.glob("*.png"))
        for i, img_path in enumerate(image_paths):
            if max_images and image_count >= max_images:
                break

            # Skip images for speed
            if i % skip_every != 0:
                continue

            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            _, descriptors = orb.detectAndCompute(img, None)

            if descriptors is not None and len(descriptors) > 0:
                all_descriptors.append(descriptors)
                image_count += 1
                seq_count += 1

        print(f"{seq_count} images")

        if max_images and image_count >= max_images:
            print(f"Reached max_images limit ({max_images})")
            break

    print(f"\nCollected descriptors from {image_count} images")

    if not all_descriptors:
        raise ValueError(f"No descriptors found in {data_dir}")

    stacked = np.vstack(all_descriptors)
    print(f"Total descriptors: {len(stacked)}")

    return stacked


def train_vocabulary(
    descriptors: np.ndarray,
    n_words: int,
    batch_size: int = 10000,
    max_iter: int = 100,
) -> np.ndarray:
    """Train vocabulary using k-means clustering.

    Args:
        descriptors: Stacked descriptors, shape (N, 32)
        n_words: Number of visual words (clusters)
        batch_size: Mini-batch size for k-means
        max_iter: Maximum iterations

    Returns:
        Cluster centers (visual words), shape (n_words, 32)
    """
    print(f"\nTraining vocabulary with {n_words} words...")
    print(f"  Descriptors: {len(descriptors)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Max iterations: {max_iter}")

    start_time = time.time()

    # MiniBatchKMeans is much faster than regular KMeans for large datasets
    kmeans = MiniBatchKMeans(
        n_clusters=n_words,
        random_state=42,
        batch_size=batch_size,
        n_init="auto",
        max_iter=max_iter,
        verbose=1,
    )

    # Convert to float32 for k-means
    descriptors_float = descriptors.astype(np.float32)
    kmeans.fit(descriptors_float)

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Inertia: {kmeans.inertia_:.2e}")
    print(f"  Iterations: {kmeans.n_iter_}")

    return kmeans.cluster_centers_.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train visual vocabulary for loop closure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/euroc"),
        help="Path to EuRoC data directory (default: data/euroc)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/vocabulary.npz"),
        help="Output vocabulary file (default: data/vocabulary.npz)",
    )
    parser.add_argument(
        "--n-words",
        type=int,
        default=1000,
        help="Number of visual words (default: 1000)",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=500,
        help="ORB features per image (default: 500)",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Max images to process (default: all)",
    )
    parser.add_argument(
        "--skip-every",
        type=int,
        default=3,
        help="Process every Nth image (default: 3 for speed)",
    )
    args = parser.parse_args()

    # Validate data directory
    if not args.data_dir.exists():
        print(f"Error: Data directory not found: {args.data_dir}")
        print("Please download EuRoC dataset sequences to this directory.")
        sys.exit(1)

    print("=" * 60)
    print("Visual Vocabulary Training")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output file: {args.output}")
    print(f"Visual words: {args.n_words}")
    print(f"Features/image: {args.n_features}")
    print(f"Skip every: {args.skip_every}")
    if args.max_images:
        print(f"Max images: {args.max_images}")
    print()

    # Collect descriptors
    descriptors = collect_descriptors(
        args.data_dir,
        n_features=args.n_features,
        max_images=args.max_images,
        skip_every=args.skip_every,
    )

    # Train vocabulary
    words = train_vocabulary(descriptors, args.n_words)

    # Create and save vocabulary
    vocabulary = VisualVocabulary.from_words(words)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    vocabulary.save(args.output)

    print()
    print("=" * 60)
    print(f"Vocabulary saved to: {args.output}")
    print(f"  Words: {vocabulary.n_words}")
    print(f"  Word shape: {vocabulary.words.shape}")
    print("=" * 60)


if __name__ == "__main__":
    main()
