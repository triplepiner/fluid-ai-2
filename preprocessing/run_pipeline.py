#!/usr/bin/env python3
"""Run the full preprocessing pipeline in sequence.

Steps:
1. Normalize FPS
2. Sync videos
3. Stabilize
4. Normalize intrinsics (undistort)
5. COLMAP reconstruction
6. Optical flow extraction
7. Depth estimation

Usage::

    python -m preprocessing.run_pipeline --input-dir data/raw
    # or via entry point:
    fluid-preprocess --input-dir data/raw
"""

from __future__ import annotations

import argparse
import sys
import time

from preprocessing.extract_depth import main as extract_depth
from preprocessing.extract_flow import main as extract_flow
from preprocessing.normalize_fps import main as normalize_fps
from preprocessing.normalize_intrinsics import main as normalize_intrinsics
from preprocessing.run_colmap import main as run_colmap
from preprocessing.stabilize import main as stabilize
from preprocessing.sync_videos import main as sync_videos

_STEPS = [
    ("Normalize FPS", normalize_fps, ["--input-dir", "{input}", "--output-dir", "data/normalized"]),
    ("Sync Videos", sync_videos, ["--input-dir", "data/normalized", "--output-dir", "data/synced"]),
    ("Stabilize", stabilize, ["--input-dir", "data/synced", "--output-dir", "data/stabilized"]),
    (
        "Normalize Intrinsics",
        normalize_intrinsics,
        ["--input-dir", "data/stabilized", "--output-dir", "data/undistorted"],
    ),
    ("COLMAP", run_colmap, ["--input-dir", "data/undistorted", "--output-dir", "data/colmap"]),
    (
        "Optical Flow",
        extract_flow,
        ["--input-dir", "data/undistorted", "--output-dir", "data/flow"],
    ),
    (
        "Depth Estimation",
        extract_depth,
        ["--input-dir", "data/undistorted", "--output-dir", "data/depth"],
    ),
]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the full preprocessing pipeline")
    parser.add_argument(
        "--input-dir",
        type=str,
        default="data/raw",
        help="Input directory with raw multi-view video data",
    )
    parser.add_argument(
        "--start-step", type=int, default=1, help="Step to start from (1-7, default: 1)"
    )
    parser.add_argument("--end-step", type=int, default=7, help="Step to end at (1-7, default: 7)")
    args = parser.parse_args(argv)

    print("=" * 60)
    print("  Preprocessing Pipeline")
    print("=" * 60)

    total_steps = len(_STEPS)
    for i, (name, fn, arg_template) in enumerate(_STEPS, start=1):
        if i < args.start_step or i > args.end_step:
            continue

        step_args = [a.replace("{input}", args.input_dir) for a in arg_template]
        print(f"\n--- Step {i}/{total_steps}: {name} ---")
        t0 = time.time()
        try:
            fn(step_args)
        except SystemExit as e:
            if e.code and e.code != 0:
                print(f"  Step {i} ({name}) failed with exit code {e.code}")
                sys.exit(e.code)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.1f}s")

    print(f"\n{'=' * 60}")
    print("  Pipeline complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
