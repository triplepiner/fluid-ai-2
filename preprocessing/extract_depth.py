#!/usr/bin/env python3
"""Generate pseudo-depth maps from monocular frames using Depth Anything V2.

For each camera directory (``cam_00/``, ``cam_01/``, …), this script:

1. Loads every frame.
2. Runs Depth Anything V2 (via the ``transformers`` pipeline) to produce a
   relative depth map per frame.
3. Normalises depth so that the median depth of frame 0 equals 1.0 for each
   camera, giving roughly consistent scales across cameras.
4. Saves depth as ``.npy`` (float32) and colourised visualisations (viridis).

Supports CUDA, MPS (Apple Silicon), and CPU backends.

Example
-------
::

    python -m preprocessing.extract_depth \\
        --input_dir  data/undistorted \\
        --output_dir data/depth \\
        --model_size base \\
        --device auto
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}

MODEL_NAME_MAP = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}


# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def select_device(requested: str) -> torch.device:
    """Resolve a device string to a :class:`torch.device`.

    ``"auto"`` probes for CUDA → MPS → CPU in that order.
    """
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def default_batch_size(device: torch.device) -> int:
    """Return a sensible default batch size for *device*."""
    if device.type == "cuda":
        return 8
    if device.type == "mps":
        return 4
    return 1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_pipeline(model_size: str, device: torch.device):
    """Load a Depth Anything V2 model via the transformers pipeline.

    Returns a callable ``transformers.Pipeline``.
    """
    from transformers import pipeline as hf_pipeline

    model_name = MODEL_NAME_MAP[model_size]
    logger.info("Loading %s on %s...", model_name, device)

    # The pipeline accepts device as int (cuda index) or -1 (cpu) or string.
    if device.type == "cuda":
        device_arg = device.index if device.index is not None else 0
    elif device.type == "mps":
        device_arg = "mps"
    else:
        device_arg = -1

    pipe = hf_pipeline(
        "depth-estimation",
        model=model_name,
        device=device_arg,
    )
    logger.info("Model loaded successfully.")
    return pipe


# ---------------------------------------------------------------------------
# Depth visualisation
# ---------------------------------------------------------------------------


def depth_to_colormap(depth: np.ndarray) -> np.ndarray:
    """Convert a float depth map to a viridis-coloured uint8 BGR image.

    Normalises to [0, 255] based on the depth map's own min/max.
    """
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-8:
        norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
    else:
        norm = np.zeros_like(depth, dtype=np.uint8)
    colored = cv2.applyColorMap(norm, cv2.COLORMAP_VIRIDIS)
    return colored


# ---------------------------------------------------------------------------
# Per-camera pipeline
# ---------------------------------------------------------------------------


def process_camera(
    cam_name: str,
    input_cam_dir: Path,
    output_cam_dir: Path,
    pipe,
    batch_size: int,
) -> dict[str, Any]:
    """Estimate depth for all frames of one camera.

    Returns a per-camera report dict.
    """
    frames = sorted(
        p for p in input_cam_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not frames:
        logger.warning("%s: no frames found — skipping.", cam_name)
        return {"camera": cam_name, "status": "skipped", "reason": "no frames"}

    output_cam_dir.mkdir(parents=True, exist_ok=True)

    # Load all frames as PIL images for the pipeline.
    pil_images: list[Image.Image] = []
    for fpath in frames:
        img = cv2.imread(str(fpath), cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("  Failed to read %s — skipping frame.", fpath.name)
            continue
        pil_images.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))

    if not pil_images:
        return {"camera": cam_name, "status": "skipped", "reason": "all frames unreadable"}

    # Run depth estimation in batches.
    logger.info(
        "  %s: estimating depth for %d frames (batch_size=%d)...",
        cam_name,
        len(pil_images),
        batch_size,
    )

    depth_maps: list[np.ndarray] = []
    effective_batch = batch_size

    for batch_start in tqdm(range(0, len(pil_images), effective_batch), desc=cam_name, leave=False):
        batch_end = min(batch_start + effective_batch, len(pil_images))
        batch_imgs = pil_images[batch_start:batch_end]

        try:
            results = pipe(batch_imgs, batch_size=len(batch_imgs))
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and effective_batch > 1:
                effective_batch = max(1, effective_batch // 2)
                logger.warning("  OOM — reducing batch size to %d", effective_batch)
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                # Retry with smaller batch.
                results = pipe(batch_imgs, batch_size=1)
            else:
                raise

        if not isinstance(results, list):
            results = [results]

        for r in results:
            depth = r["predicted_depth"]
            if isinstance(depth, torch.Tensor):
                depth = depth.cpu().numpy()
            depth_maps.append(depth.astype(np.float32))

    # Scale-align: normalise so median of frame 0 = 1.0.
    median_frame0 = float(np.median(depth_maps[0]))
    if median_frame0 > 1e-8:
        scale = 1.0 / median_frame0
    else:
        scale = 1.0
        logger.warning(
            "  %s: median depth of frame 0 is near zero, skipping normalisation.", cam_name
        )

    all_mins: list[float] = []
    all_maxs: list[float] = []
    all_medians: list[float] = []

    for i, depth in enumerate(depth_maps):
        depth_scaled = depth * scale

        # Save .npy.
        np.save(str(output_cam_dir / f"depth_{i:05d}.npy"), depth_scaled)

        # Save visualisation.
        vis = depth_to_colormap(depth_scaled)
        cv2.imwrite(str(output_cam_dir / f"depth_vis_{i:05d}.png"), vis)

        all_mins.append(float(depth_scaled.min()))
        all_maxs.append(float(depth_scaled.max()))
        all_medians.append(float(np.median(depth_scaled)))

    return {
        "camera": cam_name,
        "status": "ok",
        "num_frames": len(depth_maps),
        "scale_factor": round(scale, 6),
        "depth_min": round(min(all_mins), 6),
        "depth_max": round(max(all_maxs), 6),
        "depth_median": round(float(np.mean(all_medians)), 6),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def extract_depth(
    input_dir: Path,
    output_dir: Path,
    model_size: str = "base",
    device_str: str = "auto",
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run the full depth extraction pipeline.

    Parameters
    ----------
    input_dir : Path
        Directory with undistorted frame sequences (``cam_XX/`` subdirs).
    output_dir : Path
        Destination for depth outputs.
    model_size : str
        ``"small"``, ``"base"``, or ``"large"``.
    device_str : str
        ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
    batch_size : int or None
        Override batch size (default: auto from device type).

    Returns
    -------
    dict
        Summary manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    device = select_device(device_str)
    logger.info("Using device: %s", device)

    if batch_size is None:
        batch_size = default_batch_size(device)
    logger.info("Batch size: %d", batch_size)

    pipe = load_pipeline(model_size, device)

    # Discover camera directories.
    cam_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not cam_dirs:
        logger.error("No camera directories found in %s", input_dir)
        return {"status": "error", "reason": "no camera directories"}

    logger.info("Found %d camera(s): %s", len(cam_dirs), [d.name for d in cam_dirs])

    reports: list[dict[str, Any]] = []
    total_frames = 0

    for cam_dir in cam_dirs:
        logger.info("Processing %s...", cam_dir.name)
        report = process_camera(
            cam_name=cam_dir.name,
            input_cam_dir=cam_dir,
            output_cam_dir=output_dir / cam_dir.name,
            pipe=pipe,
            batch_size=batch_size,
        )
        reports.append(report)
        total_frames += report.get("num_frames", 0)

    manifest = {
        "model": MODEL_NAME_MAP[model_size],
        "model_size": model_size,
        "device": str(device),
        "batch_size": batch_size,
        "total_frames": total_frames,
        "per_camera": reports,
    }

    manifest_path = output_dir / "depth_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``extract_depth``."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate pseudo-depth maps from monocular frames using " "Depth Anything V2."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory with undistorted frame sequences (cam_XX/ subdirs).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write depth outputs.",
    )
    parser.add_argument(
        "--model_size",
        choices=["small", "base", "large"],
        default="base",
        help="Depth Anything V2 model size (default: base).",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device (default: auto).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size (default: auto based on device).",
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

    result = extract_depth(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_size=args.model_size,
        device_str=args.device,
        batch_size=args.batch_size,
    )

    logger.info("Done. Total frames: %d", result.get("total_frames", 0))


if __name__ == "__main__":
    main()
