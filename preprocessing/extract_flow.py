#!/usr/bin/env python3
"""Compute dense optical flow between consecutive frames using RAFT.

For each camera directory (``cam_00/``, ``cam_01/``, …), this script:

1. Loads consecutive frame pairs.
2. Runs RAFT (via torchvision) to produce dense 2-D flow fields.
3. Optionally computes backward flow and forward-backward consistency masks.
4. Saves flow as ``.npy`` arrays, HSV visualisations, and occlusion masks.

Supports CUDA, MPS (Apple Silicon), and CPU backends.

Example
-------
::

    python -m preprocessing.extract_flow \\
        --input_dir  data/undistorted \\
        --output_dir data/flow \\
        --device auto \\
        --backward
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
from torchvision.models.optical_flow import Raft_Large_Weights, raft_large
from tqdm import tqdm

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


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
        return 2
    return 1


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_raft(device: torch.device) -> torch.nn.Module:
    """Load a pretrained RAFT-Large model and move it to *device*."""
    weights = Raft_Large_Weights.C_T_SKHT_V2
    model = raft_large(weights=weights)
    model = model.to(device).eval()
    logger.info("RAFT-Large loaded on %s", device)
    return model


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------


def load_frame_tensor(path: Path) -> torch.Tensor:
    """Read an image and return a ``(3, H, W)`` float32 tensor in ``[-1, 1]``.

    Uses the same normalisation as ``Raft_Large_Weights.C_T_SKHT_V2.transforms()``:
    uint8 → float [0, 1] → normalise to [-1, 1].
    """
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    # Map [0, 1] → [-1, 1]
    t = t * 2.0 - 1.0
    return t


def pad_to_multiple_of_8(t: torch.Tensor) -> tuple[torch.Tensor, int, int]:
    """Pad a ``(B, C, H, W)`` tensor so H and W are multiples of 8.

    Returns the padded tensor and the original (H, W) for later cropping.
    """
    _, _, h, w = t.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        t = torch.nn.functional.pad(t, (0, pad_w, 0, pad_h), mode="replicate")
    return t, h, w


# ---------------------------------------------------------------------------
# Flow computation
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_flow_batch(
    model: torch.nn.Module,
    frames1: torch.Tensor,
    frames2: torch.Tensor,
    num_iters: int,
) -> torch.Tensor:
    """Run RAFT on a batch of frame pairs.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained RAFT model (already on device).
    frames1, frames2 : torch.Tensor
        Batches of shape ``(B, 3, H, W)`` in ``[-1, 1]``, already padded to
        multiples of 8 and on the model's device.
    num_iters : int
        Number of RAFT recurrent updates.

    Returns
    -------
    torch.Tensor
        Flow of shape ``(B, 2, H, W)`` (the final iteration's prediction).
    """
    predictions = model(frames1, frames2, num_flow_updates=num_iters)
    return predictions[-1]  # last iteration is the most refined


# ---------------------------------------------------------------------------
# Flow visualisation
# ---------------------------------------------------------------------------


def flow_to_hsv(flow: np.ndarray) -> np.ndarray:
    """Convert a ``(2, H, W)`` flow field to an HSV colour image.

    - Hue encodes direction.
    - Value encodes magnitude (normalised per-image).
    - Saturation is always 1.

    Returns a ``(H, W, 3)`` uint8 BGR image suitable for ``cv2.imwrite``.
    """
    u, v = flow[0], flow[1]
    mag = np.sqrt(u**2 + v**2)
    ang = np.arctan2(v, u)

    hsv = np.zeros((flow.shape[1], flow.shape[2], 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) / (2 * np.pi) * 179).astype(np.uint8)
    hsv[..., 1] = 255
    max_mag = mag.max()
    if max_mag > 0:
        hsv[..., 2] = np.clip(mag / max_mag * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Forward-backward consistency
# ---------------------------------------------------------------------------


def compute_fb_consistency_mask(
    fwd_flow: np.ndarray,
    bwd_flow: np.ndarray,
    threshold: float = 1.0,
) -> np.ndarray:
    """Compute a forward-backward consistency mask.

    For each pixel ``(x, y)``, we warp it by ``fwd_flow`` to get ``(x', y')``,
    then sample ``bwd_flow`` at ``(x', y')``.  If the round-trip displacement
    ``|fwd + bwd_warped|`` exceeds *threshold*, the pixel is marked as
    occluded (0); otherwise it is consistent (255).

    Parameters
    ----------
    fwd_flow : np.ndarray
        Forward flow ``(2, H, W)``.
    bwd_flow : np.ndarray
        Backward flow ``(2, H, W)``.
    threshold : float
        Maximum allowed round-trip error in pixels.

    Returns
    -------
    np.ndarray
        Binary mask ``(H, W)`` of dtype uint8 (255 = consistent, 0 = occluded).
    """
    h, w = fwd_flow.shape[1], fwd_flow.shape[2]

    # Build coordinate grid.
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))

    # Warped coordinates.
    wx = xs + fwd_flow[0]
    wy = ys + fwd_flow[1]

    # Sample backward flow at warped locations via bilinear interpolation.
    bwd_u = cv2.remap(bwd_flow[0], wx, wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    bwd_v = cv2.remap(bwd_flow[1], wx, wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    # Round-trip error.
    err_u = fwd_flow[0] + bwd_u
    err_v = fwd_flow[1] + bwd_v
    err = np.sqrt(err_u**2 + err_v**2)

    mask = np.where(err <= threshold, 255, 0).astype(np.uint8)
    return mask


# ---------------------------------------------------------------------------
# Per-camera pipeline
# ---------------------------------------------------------------------------


def process_camera(
    cam_name: str,
    input_cam_dir: Path,
    output_cam_dir: Path,
    model: torch.nn.Module,
    device: torch.device,
    num_iters: int,
    batch_size: int,
    backward: bool,
) -> dict[str, Any]:
    """Compute optical flow for all consecutive frame pairs in one camera.

    Returns a per-camera report dict.
    """
    frames = sorted(
        p for p in input_cam_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if len(frames) < 2:
        logger.warning("%s: need >= 2 frames, got %d — skipping.", cam_name, len(frames))
        return {"camera": cam_name, "status": "skipped", "reason": "< 2 frames"}

    output_cam_dir.mkdir(parents=True, exist_ok=True)
    num_pairs = len(frames) - 1

    # Collect pair indices.
    pair_indices = list(range(num_pairs))
    all_fwd_mags: list[float] = []

    desc = f"{cam_name} fwd"
    for batch_start in tqdm(range(0, num_pairs, batch_size), desc=desc, leave=False):
        batch_end = min(batch_start + batch_size, num_pairs)
        batch_idx = pair_indices[batch_start:batch_end]

        # Load frame pairs.
        tensors1 = []
        tensors2 = []
        for idx in batch_idx:
            tensors1.append(load_frame_tensor(frames[idx]))
            tensors2.append(load_frame_tensor(frames[idx + 1]))

        img1 = torch.stack(tensors1).to(device)
        img2 = torch.stack(tensors2).to(device)
        img1, orig_h, orig_w = pad_to_multiple_of_8(img1)
        img2, _, _ = pad_to_multiple_of_8(img2)

        # Forward flow.
        fwd_flow = compute_flow_batch(model, img1, img2, num_iters)
        fwd_flow = fwd_flow[:, :, :orig_h, :orig_w].cpu().numpy()

        # Backward flow (if requested).
        bwd_flow_np = None
        if backward:
            bwd_flow = compute_flow_batch(model, img2, img1, num_iters)
            bwd_flow_np = bwd_flow[:, :, :orig_h, :orig_w].cpu().numpy()

        # Save per-pair outputs.
        for i, idx in enumerate(batch_idx):
            fwd = fwd_flow[i]  # (2, H, W)
            np.save(str(output_cam_dir / f"flow_fwd_{idx:05d}.npy"), fwd)

            # Visualisation.
            vis = flow_to_hsv(fwd)
            cv2.imwrite(str(output_cam_dir / f"flow_vis_{idx:05d}.png"), vis)

            mag = np.sqrt(fwd[0] ** 2 + fwd[1] ** 2).mean()
            all_fwd_mags.append(float(mag))

            if backward and bwd_flow_np is not None:
                bwd = bwd_flow_np[i]
                np.save(str(output_cam_dir / f"flow_bwd_{idx:05d}.npy"), bwd)

                # Consistency mask.
                mask = compute_fb_consistency_mask(fwd, bwd)
                cv2.imwrite(str(output_cam_dir / f"flow_mask_{idx:05d}.png"), mask)

        # Free GPU memory.
        del img1, img2, fwd_flow
        if backward:
            del bwd_flow
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mean_mag = float(np.mean(all_fwd_mags)) if all_fwd_mags else 0.0
    max_mag = float(np.max(all_fwd_mags)) if all_fwd_mags else 0.0

    return {
        "camera": cam_name,
        "status": "ok",
        "num_pairs": num_pairs,
        "mean_flow_magnitude": round(mean_mag, 4),
        "max_flow_magnitude": round(max_mag, 4),
        "backward_computed": backward,
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def extract_flow(
    input_dir: Path,
    output_dir: Path,
    device_str: str = "auto",
    backward: bool = True,
    num_iters: int = 20,
    batch_size: int | None = None,
) -> dict[str, Any]:
    """Run the full optical-flow extraction pipeline.

    Parameters
    ----------
    input_dir : Path
        Directory with undistorted frame sequences (``cam_XX/`` subdirs).
    output_dir : Path
        Destination for flow outputs.
    device_str : str
        ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
    backward : bool
        Whether to also compute backward flow and consistency masks.
    num_iters : int
        Number of RAFT recurrent update iterations.
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

    model = load_raft(device)

    # Discover camera directories.
    cam_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir() and not d.name.startswith("."))
    if not cam_dirs:
        logger.error("No camera directories found in %s", input_dir)
        return {"status": "error", "reason": "no camera directories"}

    logger.info("Found %d camera(s): %s", len(cam_dirs), [d.name for d in cam_dirs])

    reports: list[dict[str, Any]] = []
    total_pairs = 0

    for cam_dir in cam_dirs:
        logger.info("Processing %s...", cam_dir.name)
        report = process_camera(
            cam_name=cam_dir.name,
            input_cam_dir=cam_dir,
            output_cam_dir=output_dir / cam_dir.name,
            model=model,
            device=device,
            num_iters=num_iters,
            batch_size=batch_size,
            backward=backward,
        )
        reports.append(report)
        if report.get("num_pairs"):
            total_pairs += report["num_pairs"]

    manifest = {
        "device": str(device),
        "num_iters": num_iters,
        "backward": backward,
        "batch_size": batch_size,
        "total_flow_pairs": total_pairs,
        "per_camera": reports,
    }

    manifest_path = output_dir / "flow_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``extract_flow``."""
    parser = argparse.ArgumentParser(
        description="Compute dense optical flow between consecutive frames using RAFT.",
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
        help="Directory to write flow outputs.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device (default: auto).",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        default=True,
        help="Compute backward flow and consistency masks (default: True).",
    )
    parser.add_argument(
        "--no_backward",
        action="store_true",
        help="Disable backward flow computation.",
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=20,
        help="RAFT recurrent update iterations (default: 20).",
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

    backward = args.backward and not args.no_backward

    result = extract_flow(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        device_str=args.device,
        backward=backward,
        num_iters=args.num_iters,
        batch_size=args.batch_size,
    )

    logger.info("Done. Total flow pairs: %d", result.get("total_flow_pairs", 0))


if __name__ == "__main__":
    main()
