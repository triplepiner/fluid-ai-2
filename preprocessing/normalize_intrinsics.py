#!/usr/bin/env python3
"""Undistort frames and normalise intrinsics across all cameras.

After COLMAP, each camera may have different focal lengths and distortion
coefficients.  This script:

1. Loads ``cameras.json`` from the COLMAP step.
2. Computes a **common target intrinsic** matrix (mean focal length, centred
   principal point, zero distortion) at the maximum resolution.
3. For each camera, builds an undistortion + re-projection remap table
   (``cv2.initUndistortRectifyMap``) that simultaneously removes lens
   distortion and maps to the common focal length.
4. Applies the remap to every frame (``cv2.remap`` with Lanczos-4).
5. Writes ``cameras_normalized.json`` with shared intrinsics and unchanged
   extrinsics.
6. Saves a before/after visualisation grid.

Example
-------
::

    python -m preprocessing.normalize_intrinsics \\
        --input_dir      data/stabilized \\
        --cameras_json   data/colmap/cameras.json \\
        --output_dir     data/undistorted \\
        --output_cameras data/undistorted/cameras_normalized.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Intrinsic helpers
# ---------------------------------------------------------------------------


def camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    """Build a 3x3 camera intrinsic matrix."""
    return np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def distortion_vector(
    k1: float = 0.0,
    k2: float = 0.0,
    p1: float = 0.0,
    p2: float = 0.0,
) -> np.ndarray:
    """Build an OpenCV 4-element distortion vector [k1, k2, p1, p2]."""
    return np.array([k1, k2, p1, p2], dtype=np.float64)


def compute_common_intrinsic(
    cameras: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, int, int]:
    """Derive a common target intrinsic from per-camera intrinsics.

    Strategy:
    - Focal length: geometric mean of (fx, fy) per camera, then arithmetic
      mean across cameras.
    - Resolution: maximum width and height across cameras.
    - Principal point: centre of the target resolution.

    Returns
    -------
    K_target : np.ndarray
        3x3 target intrinsic matrix.
    target_w : int
        Target image width.
    target_h : int
        Target image height.
    """
    focal_lengths: list[float] = []
    max_w = 0
    max_h = 0

    for cam in cameras.values():
        intr = cam["intrinsic"]
        fx, fy = intr["fx"], intr["fy"]
        geo_mean = math.sqrt(fx * fy) if fx > 0 and fy > 0 else max(fx, fy)
        focal_lengths.append(geo_mean)
        max_w = max(max_w, cam["image_width"])
        max_h = max(max_h, cam["image_height"])

    mean_f = sum(focal_lengths) / len(focal_lengths) if focal_lengths else 1.0
    cx = max_w / 2.0
    cy = max_h / 2.0

    K_target = camera_matrix(mean_f, mean_f, cx, cy)
    return K_target, max_w, max_h


def build_remap_tables(
    cam_entry: dict[str, Any],
    K_target: np.ndarray,
    target_w: int,
    target_h: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Compute undistortion + re-projection remap tables for one camera.

    Returns
    -------
    map1, map2 : np.ndarray
        Remap lookup tables (float32, same shape as target image).
    max_displacement : float
        Maximum pixel displacement caused by the remap (for reporting).
    """
    intr = cam_entry["intrinsic"]
    src_w = cam_entry["image_width"]
    src_h = cam_entry["image_height"]

    K_src = camera_matrix(intr["fx"], intr["fy"], intr["cx"], intr["cy"])
    dist = distortion_vector(intr["k1"], intr["k2"], intr["p1"], intr["p2"])

    # If source resolution differs from target, scale the source intrinsic so
    # that pixel coordinates are expressed in the target frame.
    sx = target_w / src_w if src_w else 1.0
    sy = target_h / src_h if src_h else 1.0
    if not (math.isclose(sx, 1.0, abs_tol=1e-6) and math.isclose(sy, 1.0, abs_tol=1e-6)):
        K_src_scaled = K_src.copy()
        K_src_scaled[0, :] *= sx
        K_src_scaled[1, :] *= sy
    else:
        K_src_scaled = K_src

    map1, map2 = cv2.initUndistortRectifyMap(
        cameraMatrix=K_src_scaled,
        distCoeffs=dist,
        R=np.eye(3),
        newCameraMatrix=K_target,
        size=(target_w, target_h),
        m1type=cv2.CV_32FC1,
    )

    # Compute max displacement: compare remap to identity grid.
    ys, xs = np.mgrid[0:target_h, 0:target_w].astype(np.float32)
    dx = map1 - xs
    dy = map2 - ys
    max_displacement = float(np.sqrt(dx**2 + dy**2).max())

    return map1, map2, max_displacement


# ---------------------------------------------------------------------------
# Frame remapping (worker function for multiprocessing)
# ---------------------------------------------------------------------------


def _remap_frame(
    src_path: str,
    dst_path: str,
    map1_path: str,
    map2_path: str,
    src_w: int,
    src_h: int,
    target_w: int,
    target_h: int,
) -> str:
    """Remap a single frame using pre-computed tables stored on disk.

    This function is designed to be called in a worker process.  The remap
    tables are loaded from NumPy files to avoid pickling large arrays.

    Returns the destination path on success.
    """
    map1 = np.load(map1_path)
    map2 = np.load(map2_path)
    img = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"Failed to read image: {src_path}")

    # Resize if source resolution differs from target.
    h_img, w_img = img.shape[:2]
    if w_img != target_w or h_img != target_h:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

    remapped = cv2.remap(
        img,
        map1,
        map2,
        interpolation=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    cv2.imwrite(dst_path, remapped)
    return dst_path


# ---------------------------------------------------------------------------
# Per-camera pipeline
# ---------------------------------------------------------------------------


def process_camera(
    cam_name: str,
    cam_entry: dict[str, Any],
    input_dir: Path,
    output_dir: Path,
    K_target: np.ndarray,
    target_w: int,
    target_h: int,
    max_workers: int,
) -> dict[str, Any]:
    """Undistort and re-project all frames for one camera.

    Returns a report dict for this camera.
    """
    cam_input = input_dir / cam_name
    if not cam_input.is_dir():
        logger.warning("Frame directory not found for %s: %s", cam_name, cam_input)
        return {
            "camera": cam_name,
            "status": "skipped",
            "reason": "input directory not found",
        }

    frames = sorted(
        p for p in cam_input.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frames:
        logger.warning("No frames found for %s in %s", cam_name, cam_input)
        return {
            "camera": cam_name,
            "status": "skipped",
            "reason": "no frames found",
        }

    # Build remap tables.
    map1, map2, max_disp = build_remap_tables(cam_entry, K_target, target_w, target_h)

    src_w = cam_entry["image_width"]
    src_h = cam_entry["image_height"]

    # Save remap tables to temp numpy files for workers.
    cam_out = output_dir / cam_name
    cam_out.mkdir(parents=True, exist_ok=True)
    map1_path = cam_out / "_map1.npy"
    map2_path = cam_out / "_map2.npy"
    np.save(str(map1_path), map1)
    np.save(str(map2_path), map2)

    logger.info(
        "  %s: %d frames, max distortion correction = %.2f px",
        cam_name,
        len(frames),
        max_disp,
    )

    # Process frames in parallel.
    tasks = []
    for fpath in frames:
        dst_path = cam_out / fpath.name
        tasks.append(
            (
                str(fpath),
                str(dst_path),
                str(map1_path),
                str(map2_path),
                src_w,
                src_h,
                target_w,
                target_h,
            )
        )

    failed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_remap_frame, *t): t[0] for t in tasks}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception:
                failed += 1
                logger.exception("  Failed to remap %s", futures[fut])

    # Clean up temp files.
    map1_path.unlink(missing_ok=True)
    map2_path.unlink(missing_ok=True)

    intr = cam_entry["intrinsic"]
    original_f = (
        math.sqrt(intr["fx"] * intr["fy"])
        if intr["fx"] > 0 and intr["fy"] > 0
        else max(intr["fx"], intr["fy"])
    )

    return {
        "camera": cam_name,
        "status": "ok",
        "num_frames": len(frames),
        "num_failed": failed,
        "original_focal_length": round(original_f, 4),
        "target_focal_length": round(float(K_target[0, 0]), 4),
        "original_resolution": f"{src_w}x{src_h}",
        "target_resolution": f"{target_w}x{target_h}",
        "max_distortion_correction_px": round(max_disp, 4),
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_before_after(
    cameras: dict[str, dict[str, Any]],
    input_dir: Path,
    output_dir: Path,
    target_w: int,
    target_h: int,
    save_path: Path,
) -> None:
    """Save a grid of before/after frames (one per camera).

    Parameters
    ----------
    cameras : dict
        Camera entries keyed by name.
    input_dir : Path
        Directory containing original stabilized frames.
    output_dir : Path
        Directory containing undistorted frames.
    target_w, target_h : int
        Target image dimensions.
    save_path : Path
        Output PNG path.
    """
    cam_names = sorted(cameras.keys())
    n = len(cam_names)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 2, figsize=(12, 4 * n), squeeze=False)
    fig.suptitle("Intrinsic Normalisation: Before vs After", fontsize=14)

    for i, cam_name in enumerate(cam_names):
        # Find the first frame.
        src_dir = input_dir / cam_name
        dst_dir = output_dir / cam_name
        src_frames = (
            sorted(
                p for p in src_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if src_dir.is_dir()
            else []
        )
        dst_frames = (
            sorted(
                p for p in dst_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )
            if dst_dir.is_dir()
            else []
        )

        if src_frames:
            img_before = cv2.imread(str(src_frames[0]))
            if img_before is not None:
                img_before = cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB)
                axes[i, 0].imshow(img_before)
        axes[i, 0].set_title(f"{cam_name} — Original")
        axes[i, 0].axis("off")

        if dst_frames:
            img_after = cv2.imread(str(dst_frames[0]))
            if img_after is not None:
                img_after = cv2.cvtColor(img_after, cv2.COLOR_BGR2RGB)
                axes[i, 1].imshow(img_after)
        axes[i, 1].set_title(f"{cam_name} — Undistorted")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(str(save_path), dpi=150)
    plt.close(fig)
    logger.info("Before/after visualisation saved to %s", save_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def normalize_intrinsics(
    input_dir: Path,
    cameras_json: Path,
    output_dir: Path,
    output_cameras: Path,
    max_workers: int = 4,
) -> dict[str, Any]:
    """Undistort and normalise intrinsics for all cameras.

    Parameters
    ----------
    input_dir : Path
        Directory with stabilized frame sequences (``cam_XX/`` subdirs).
    cameras_json : Path
        Path to ``cameras.json`` from the COLMAP step.
    output_dir : Path
        Directory for undistorted frame sequences.
    output_cameras : Path
        Path for the updated cameras JSON with common intrinsics.
    max_workers : int
        Number of parallel workers for frame remapping.

    Returns
    -------
    dict
        Summary report with per-camera statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load COLMAP cameras.
    with open(cameras_json) as f:
        data = json.load(f)

    cameras = data.get("cameras", {})
    normalization = data.get("normalization", {})

    if not cameras:
        logger.error("No cameras found in %s", cameras_json)
        return {"status": "error", "reason": "no cameras in JSON"}

    logger.info("Loaded %d camera(s) from %s", len(cameras), cameras_json)

    # Compute common target intrinsic.
    K_target, target_w, target_h = compute_common_intrinsic(cameras)

    target_f = float(K_target[0, 0])
    logger.info(
        "Common intrinsic: f=%.2f, resolution=%dx%d, cx=%.1f, cy=%.1f",
        target_f,
        target_w,
        target_h,
        float(K_target[0, 2]),
        float(K_target[1, 2]),
    )

    # Process each camera.
    reports: list[dict[str, Any]] = []
    for cam_name in sorted(cameras):
        logger.info("Processing %s...", cam_name)
        report = process_camera(
            cam_name=cam_name,
            cam_entry=cameras[cam_name],
            input_dir=input_dir,
            output_dir=output_dir,
            K_target=K_target,
            target_w=target_w,
            target_h=target_h,
            max_workers=max_workers,
        )
        reports.append(report)

    # Build updated cameras JSON with shared intrinsics.
    common_intrinsic = {
        "fx": target_f,
        "fy": target_f,
        "cx": float(K_target[0, 2]),
        "cy": float(K_target[1, 2]),
        "k1": 0.0,
        "k2": 0.0,
        "p1": 0.0,
        "p2": 0.0,
    }

    updated_cameras: dict[str, Any] = {}
    for cam_name, cam_entry in sorted(cameras.items()):
        updated = {
            "camera_id": cam_entry["camera_id"],
            "image_width": target_w,
            "image_height": target_h,
            "intrinsic": common_intrinsic,
            "extrinsic": cam_entry["extrinsic"],
            "camera_to_world": cam_entry["camera_to_world"],
            "world_to_camera": cam_entry["world_to_camera"],
        }
        updated_cameras[cam_name] = updated

    output_json = {
        "normalization": normalization,
        "common_intrinsic": {
            "fx": target_f,
            "fy": target_f,
            "cx": float(K_target[0, 2]),
            "cy": float(K_target[1, 2]),
            "image_width": target_w,
            "image_height": target_h,
        },
        "cameras": updated_cameras,
    }

    output_cameras.parent.mkdir(parents=True, exist_ok=True)
    with open(output_cameras, "w") as f:
        json.dump(output_json, f, indent=2)
    logger.info("Normalised cameras written to %s", output_cameras)

    # Visualisation.
    plot_before_after(
        cameras,
        input_dir,
        output_dir,
        target_w,
        target_h,
        output_dir / "undistortion_comparison.png",
    )

    return {
        "num_cameras": len(cameras),
        "target_focal_length": round(target_f, 4),
        "target_resolution": f"{target_w}x{target_h}",
        "per_camera": reports,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``normalize_intrinsics``."""
    parser = argparse.ArgumentParser(
        description=("Undistort frames and normalise intrinsics across cameras."),
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory with stabilized frame sequences (cam_XX/ subdirs).",
    )
    parser.add_argument(
        "--cameras_json",
        type=Path,
        required=True,
        help="Path to cameras.json from the COLMAP step.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write undistorted frames.",
    )
    parser.add_argument(
        "--output_cameras",
        type=Path,
        required=True,
        help="Path for the updated cameras JSON.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel workers for frame processing (default: 4).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug-level logging.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Entry point for the CLI."""
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.input_dir.is_dir():
        logger.error("Input directory does not exist: %s", args.input_dir)
        sys.exit(1)

    if not args.cameras_json.is_file():
        logger.error("cameras.json not found: %s", args.cameras_json)
        sys.exit(1)

    result = normalize_intrinsics(
        input_dir=args.input_dir,
        cameras_json=args.cameras_json,
        output_dir=args.output_dir,
        output_cameras=args.output_cameras,
        max_workers=args.max_workers,
    )

    logger.info("Summary: %s", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
