#!/usr/bin/env python3
"""Remove camera shake from synchronised frame sequences.

For tripod-mounted cameras the intended trajectory is essentially static, so
any detected inter-frame motion is unwanted jitter.  This script:

1. Estimates per-frame similarity transforms (translation + rotation + scale)
   via sparse optical flow (Shi-Tomasi → Lucas-Kanade with forward-backward
   consistency check).
2. Builds a cumulative camera trajectory.
3. Smooths the trajectory with a Gaussian kernel to separate low-frequency
   intended motion from high-frequency jitter.
4. Warps every frame by the corrective transform.
5. Optionally crops to the maximum inscribed rectangle that avoids black
   borders across the entire sequence.

Each ``cam_XX/`` subdirectory is processed independently.

Outputs
-------
- ``cam_XX/frame_NNNNN.png`` – stabilised frame sequences.
- ``stabilization_report.json`` – per-camera jitter statistics and crop rects.
- ``trajectory_cam_XX.png`` – original vs smoothed trajectory plots.

Example
-------
::

    python -m preprocessing.stabilize \\
        --input_dir  data/synced \\
        --output_dir data/stabilized \\
        --smooth_sigma 30 \\
        --crop
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import scipy.ndimage
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class FrameTransform:
    """Inter-frame similarity-transform parameters."""

    dx: float = 0.0
    dy: float = 0.0
    da: float = 0.0  # rotation angle in radians
    valid: bool = True  # False when feature tracking failed


@dataclass
class CameraReport:
    """Stabilisation statistics for one camera."""

    camera_name: str = ""
    num_frames: int = 0
    failed_tracks: int = 0
    mean_jitter_px: float = 0.0
    max_jitter_px: float = 0.0
    mean_rotation_jitter_deg: float = 0.0
    max_rotation_jitter_deg: float = 0.0
    crop_rect: list[int] = field(default_factory=list)  # [x, y, w, h]

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dictionary."""
        return {
            "camera_name": self.camera_name,
            "num_frames": self.num_frames,
            "failed_tracks": self.failed_tracks,
            "mean_jitter_pixels": round(self.mean_jitter_px, 4),
            "max_jitter_pixels": round(self.max_jitter_px, 4),
            "mean_rotation_jitter_degrees": round(self.mean_rotation_jitter_deg, 6),
            "max_rotation_jitter_degrees": round(self.max_rotation_jitter_deg, 6),
            "crop_rect_xywh": self.crop_rect,
        }


# ---------------------------------------------------------------------------
# Feature tracking
# ---------------------------------------------------------------------------

# Shi-Tomasi corner detector parameters.
_FEATURE_PARAMS = dict(
    maxCorners=200,
    qualityLevel=0.01,
    minDistance=30,
    blockSize=3,
)

# Lucas-Kanade optical flow parameters.
_LK_PARAMS = dict(
    winSize=(15, 15),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)

_BACK_THRESH = 1.0  # px – forward-backward consistency threshold


def _estimate_interframe_transform(
    gray_prev: np.ndarray,
    gray_curr: np.ndarray,
) -> FrameTransform:
    """Estimate the similarity transform between two consecutive grey frames.

    Parameters
    ----------
    gray_prev, gray_curr : np.ndarray
        Single-channel uint8 images of the same size.

    Returns
    -------
    FrameTransform
        Translation (dx, dy) and rotation angle (da).  ``valid`` is ``False``
        when fewer than 3 inlier correspondences remain after filtering.
    """
    pts_prev = cv2.goodFeaturesToTrack(gray_prev, **_FEATURE_PARAMS)
    if pts_prev is None or len(pts_prev) < 3:
        return FrameTransform(valid=False)

    # Forward flow.
    pts_curr, st_fwd, _ = cv2.calcOpticalFlowPyrLK(
        gray_prev,
        gray_curr,
        pts_prev,
        None,
        **_LK_PARAMS,
    )
    # Backward flow for consistency check.
    pts_back, st_bwd, _ = cv2.calcOpticalFlowPyrLK(
        gray_curr,
        gray_prev,
        pts_curr,
        None,
        **_LK_PARAMS,
    )

    # Keep only points that pass both status flags and backward check.
    good = (
        (st_fwd.ravel() == 1)
        & (st_bwd.ravel() == 1)
        & (np.linalg.norm(pts_prev.reshape(-1, 2) - pts_back.reshape(-1, 2), axis=1) < _BACK_THRESH)
    )

    src = pts_prev.reshape(-1, 2)[good]
    dst = pts_curr.reshape(-1, 2)[good]

    if len(src) < 3:
        return FrameTransform(valid=False)

    # Fit similarity transform (4-DOF: tx, ty, rotation, uniform scale).
    M, inliers = cv2.estimateAffinePartial2D(
        src.reshape(-1, 1, 2),
        dst.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
    )

    if M is None:
        return FrameTransform(valid=False)

    dx = float(M[0, 2])
    dy = float(M[1, 2])
    da = float(math.atan2(M[1, 0], M[0, 0]))

    return FrameTransform(dx=dx, dy=dy, da=da, valid=True)


# ---------------------------------------------------------------------------
# Trajectory computation
# ---------------------------------------------------------------------------


def compute_transforms(
    frame_paths: list[Path],
) -> list[FrameTransform]:
    """Estimate inter-frame transforms for a sorted sequence of frame images.

    Parameters
    ----------
    frame_paths : list[Path]
        Sorted list of frame image paths.

    Returns
    -------
    list[FrameTransform]
        One entry per consecutive pair (length ``len(frame_paths) - 1``).
    """
    transforms: list[FrameTransform] = []
    prev_gray: np.ndarray | None = None

    for i, fpath in enumerate(tqdm(frame_paths, desc="  tracking", leave=False)):
        img = cv2.imread(str(fpath))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if prev_gray is not None:
            tf = _estimate_interframe_transform(prev_gray, gray)
            transforms.append(tf)

        prev_gray = gray

    return transforms


def interpolate_failed(transforms: list[FrameTransform]) -> list[FrameTransform]:
    """Replace invalid transforms by linearly interpolating from neighbours.

    Parameters
    ----------
    transforms : list[FrameTransform]
        Sequence with some entries having ``valid=False``.

    Returns
    -------
    list[FrameTransform]
        All entries will have ``valid=True`` after interpolation.
    """
    n = len(transforms)
    if n == 0:
        return transforms

    # Collect indices of valid entries.
    valid_idx = [i for i, t in enumerate(transforms) if t.valid]

    if not valid_idx:
        # Nothing to interpolate from – assume identity for all.
        return [FrameTransform(valid=True) for _ in transforms]

    result = list(transforms)

    for i in range(n):
        if result[i].valid:
            continue

        # Find nearest valid neighbours.
        prev_v = max((vi for vi in valid_idx if vi < i), default=None)
        next_v = min((vi for vi in valid_idx if vi > i), default=None)

        if prev_v is not None and next_v is not None:
            alpha = (i - prev_v) / (next_v - prev_v)
            tp, tn = result[prev_v], result[next_v]
            result[i] = FrameTransform(
                dx=tp.dx + alpha * (tn.dx - tp.dx),
                dy=tp.dy + alpha * (tn.dy - tp.dy),
                da=tp.da + alpha * (tn.da - tp.da),
                valid=True,
            )
        elif prev_v is not None:
            t = result[prev_v]
            result[i] = FrameTransform(dx=t.dx, dy=t.dy, da=t.da, valid=True)
        else:
            t = result[next_v]
            result[i] = FrameTransform(dx=t.dx, dy=t.dy, da=t.da, valid=True)

    return result


def build_trajectory(
    transforms: list[FrameTransform],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate inter-frame transforms into a cumulative trajectory.

    Parameters
    ----------
    transforms : list[FrameTransform]
        Inter-frame transforms (length N-1 for N frames).

    Returns
    -------
    cum_x, cum_y, cum_a : np.ndarray
        Cumulative translation and rotation arrays, each of length N.
        ``cum_*[0]`` is always 0.
    """
    n = len(transforms) + 1
    cum_x = np.zeros(n)
    cum_y = np.zeros(n)
    cum_a = np.zeros(n)

    for i, t in enumerate(transforms):
        cum_x[i + 1] = cum_x[i] + t.dx
        cum_y[i + 1] = cum_y[i] + t.dy
        cum_a[i + 1] = cum_a[i] + t.da

    return cum_x, cum_y, cum_a


def smooth_trajectory(
    cum_x: np.ndarray,
    cum_y: np.ndarray,
    cum_a: np.ndarray,
    sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Smooth a cumulative trajectory with a 1-D Gaussian kernel.

    Parameters
    ----------
    cum_x, cum_y, cum_a : np.ndarray
        Raw cumulative trajectory arrays (length N).
    sigma : float
        Standard deviation of the Gaussian kernel in frames.

    Returns
    -------
    smooth_x, smooth_y, smooth_a : np.ndarray
        Smoothed trajectory arrays (same length).
    """
    smooth_x = scipy.ndimage.gaussian_filter1d(cum_x, sigma=sigma)
    smooth_y = scipy.ndimage.gaussian_filter1d(cum_y, sigma=sigma)
    smooth_a = scipy.ndimage.gaussian_filter1d(cum_a, sigma=sigma)
    return smooth_x, smooth_y, smooth_a


# ---------------------------------------------------------------------------
# Frame warping
# ---------------------------------------------------------------------------


def corrective_affine(
    dx: float,
    dy: float,
    da: float,
    cx: float,
    cy: float,
) -> np.ndarray:
    """Build a 2×3 affine matrix that corrects a jitter offset.

    The correction is the *inverse* of the jitter: we translate and rotate
    the frame in the opposite direction of the residual motion.

    Parameters
    ----------
    dx, dy : float
        Translational jitter to remove (pixels).
    da : float
        Rotational jitter to remove (radians).
    cx, cy : float
        Frame centre (rotation pivot).

    Returns
    -------
    np.ndarray
        2×3 affine transformation matrix.
    """
    cos_a = math.cos(-da)
    sin_a = math.sin(-da)
    M = np.array(
        [
            [cos_a, -sin_a, -dx + cx * (1 - cos_a) + cy * sin_a],
            [sin_a, cos_a, -dy - cx * sin_a + cy * (1 - cos_a)],
        ],
        dtype=np.float64,
    )
    return M


def compute_crop_rect(
    jitter_x: np.ndarray,
    jitter_y: np.ndarray,
    jitter_a: np.ndarray,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Compute the largest axis-aligned crop that avoids all black borders.

    After stabilisation, each frame is shifted/rotated by its jitter
    correction.  The crop must be safe for the *worst-case* frame.

    Parameters
    ----------
    jitter_x, jitter_y : np.ndarray
        Per-frame translational jitter (pixels).
    jitter_a : np.ndarray
        Per-frame rotational jitter (radians).
    width, height : int
        Original frame dimensions.

    Returns
    -------
    (x, y, w, h) : tuple[int, int, int, int]
        Crop rectangle in the stabilised frame coordinate system.
    """
    # Maximum absolute translation shift.
    max_dx = float(np.max(np.abs(jitter_x)))
    max_dy = float(np.max(np.abs(jitter_y)))

    # Rotation shrinks the usable area.  For small angles the inscribed
    # axis-aligned rectangle of a rotated rectangle of size (W, H) is
    # approximately (W - H*|sin a| - ...).  We use the exact formula.
    max_abs_a = float(np.max(np.abs(jitter_a)))

    cos_a = math.cos(max_abs_a)
    sin_a = math.sin(max_abs_a)

    # Inscribed rectangle inside a rotated (width × height) rectangle.
    if max_abs_a < 1e-8:
        rot_inset_x = 0.0
        rot_inset_y = 0.0
    else:
        rot_inset_x = (height * sin_a) / (2.0)
        rot_inset_y = (width * sin_a) / (2.0)

    inset_x = math.ceil(max_dx + rot_inset_x)
    inset_y = math.ceil(max_dy + rot_inset_y)

    # Clamp so crop is at least 1×1.
    inset_x = min(inset_x, width // 2 - 1)
    inset_y = min(inset_y, height // 2 - 1)

    crop_x = inset_x
    crop_y = inset_y
    crop_w = width - 2 * inset_x
    crop_h = height - 2 * inset_y

    return crop_x, crop_y, crop_w, crop_h


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_trajectory(
    cum_x: np.ndarray,
    cum_y: np.ndarray,
    cum_a: np.ndarray,
    smooth_x: np.ndarray,
    smooth_y: np.ndarray,
    smooth_a: np.ndarray,
    output_path: Path,
    camera_name: str,
) -> None:
    """Save a 3-panel plot comparing raw and smoothed camera trajectory.

    Parameters
    ----------
    cum_x, cum_y, cum_a : np.ndarray
        Original cumulative trajectory.
    smooth_x, smooth_y, smooth_a : np.ndarray
        Smoothed trajectory.
    output_path : Path
        Destination PNG path.
    camera_name : str
        Label for the plot title.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    frames = np.arange(len(cum_x))

    for ax, raw, smo, label, unit in zip(
        axes,
        [cum_x, cum_y, cum_a],
        [smooth_x, smooth_y, smooth_a],
        ["X translation", "Y translation", "Rotation"],
        ["px", "px", "rad"],
    ):
        ax.plot(frames, raw, linewidth=0.7, alpha=0.8, label="original")
        ax.plot(frames, smo, linewidth=1.2, label="smoothed")
        ax.set_ylabel(f"{label} ({unit})")
        ax.legend(loc="upper right")

    axes[-1].set_xlabel("Frame")
    axes[0].set_title(f"Camera trajectory — {camera_name}")
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Per-camera pipeline
# ---------------------------------------------------------------------------


def stabilize_camera(
    input_cam_dir: Path,
    output_cam_dir: Path,
    smooth_sigma: float,
    do_crop: bool,
    plot_dir: Path,
) -> CameraReport:
    """Stabilise a single camera's frame sequence.

    Parameters
    ----------
    input_cam_dir : Path
        Directory containing ``frame_NNNNN.png`` files.
    output_cam_dir : Path
        Destination for stabilised frames.
    smooth_sigma : float
        Gaussian smoothing sigma in frames.
    do_crop : bool
        Whether to auto-crop black borders.
    plot_dir : Path
        Directory to write trajectory plots.

    Returns
    -------
    CameraReport
        Per-camera jitter statistics.
    """
    output_cam_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted(input_cam_dir.glob("frame_*.png"))
    n_frames = len(frame_paths)

    report = CameraReport(
        camera_name=input_cam_dir.name,
        num_frames=n_frames,
    )

    if n_frames < 2:
        logger.warning(
            "  %s: fewer than 2 frames — copying as-is.",
            input_cam_dir.name,
        )
        for fp in frame_paths:
            img = cv2.imread(str(fp))
            cv2.imwrite(str(output_cam_dir / fp.name), img)
        return report

    # --- 1. Estimate inter-frame transforms --------------------------------
    logger.info("  Estimating inter-frame transforms (%d pairs)...", n_frames - 1)
    raw_transforms = compute_transforms(frame_paths)

    failed = sum(1 for t in raw_transforms if not t.valid)
    report.failed_tracks = failed
    if failed:
        logger.warning(
            "  %d/%d frames had insufficient features — interpolating.",
            failed,
            len(raw_transforms),
        )

    transforms = interpolate_failed(raw_transforms)

    # --- 2. Build cumulative trajectory ------------------------------------
    cum_x, cum_y, cum_a = build_trajectory(transforms)

    # --- 3. Smooth trajectory ----------------------------------------------
    smooth_x, smooth_y, smooth_a = smooth_trajectory(
        cum_x,
        cum_y,
        cum_a,
        smooth_sigma,
    )

    # Jitter = original - smoothed (what we need to correct).
    jitter_x = cum_x - smooth_x
    jitter_y = cum_y - smooth_y
    jitter_a = cum_a - smooth_a

    # --- Stats -------------------------------------------------------------
    jitter_mag = np.sqrt(jitter_x**2 + jitter_y**2)
    report.mean_jitter_px = float(np.mean(jitter_mag))
    report.max_jitter_px = float(np.max(jitter_mag))
    report.mean_rotation_jitter_deg = float(np.degrees(np.mean(np.abs(jitter_a))))
    report.max_rotation_jitter_deg = float(np.degrees(np.max(np.abs(jitter_a))))

    logger.info(
        "  Jitter — mean: %.2f px, max: %.2f px, mean rot: %.4f°, max rot: %.4f°",
        report.mean_jitter_px,
        report.max_jitter_px,
        report.mean_rotation_jitter_deg,
        report.max_rotation_jitter_deg,
    )

    # --- 4. Determine crop -------------------------------------------------
    sample_img = cv2.imread(str(frame_paths[0]))
    h, w = sample_img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    if do_crop:
        crop_x, crop_y, crop_w, crop_h = compute_crop_rect(
            jitter_x,
            jitter_y,
            jitter_a,
            w,
            h,
        )
        report.crop_rect = [crop_x, crop_y, crop_w, crop_h]
        logger.info(
            "  Crop rect: x=%d y=%d w=%d h=%d (from %dx%d)",
            crop_x,
            crop_y,
            crop_w,
            crop_h,
            w,
            h,
        )
    else:
        crop_x, crop_y, crop_w, crop_h = 0, 0, w, h

    # --- 5. Warp frames ----------------------------------------------------
    logger.info("  Warping %d frames...", n_frames)
    for i, fpath in enumerate(tqdm(frame_paths, desc="  stabilizing", leave=False)):
        img = cv2.imread(str(fpath))
        M = corrective_affine(jitter_x[i], jitter_y[i], jitter_a[i], cx, cy)
        stabilized = cv2.warpAffine(
            img,
            M,
            (w, h),
            borderMode=cv2.BORDER_REFLECT,
        )
        if do_crop:
            stabilized = stabilized[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        cv2.imwrite(str(output_cam_dir / fpath.name), stabilized)

    # --- 6. Trajectory plot ------------------------------------------------
    plot_path = plot_dir / f"trajectory_{input_cam_dir.name}.png"
    plot_trajectory(
        cum_x,
        cum_y,
        cum_a,
        smooth_x,
        smooth_y,
        smooth_a,
        plot_path,
        input_cam_dir.name,
    )
    logger.info("  Trajectory plot saved to %s", plot_path)

    return report


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------


def stabilize_all(
    input_dir: Path,
    output_dir: Path,
    smooth_sigma: float,
    do_crop: bool,
) -> dict[str, Any]:
    """Stabilise all camera directories under *input_dir*.

    Parameters
    ----------
    input_dir : Path
        Parent directory containing ``cam_XX/`` subdirectories.
    output_dir : Path
        Destination parent directory.
    smooth_sigma : float
        Gaussian smoothing sigma in frames.
    do_crop : bool
        Whether to auto-crop black borders.

    Returns
    -------
    dict
        Report dictionary (also written to ``stabilization_report.json``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cam_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("cam_"))

    if not cam_dirs:
        logger.error("No cam_XX directories found in %s", input_dir)
        sys.exit(1)

    logger.info(
        "Found %d camera(s) in %s — sigma=%g, crop=%s",
        len(cam_dirs),
        input_dir,
        smooth_sigma,
        do_crop,
    )

    reports: list[dict[str, Any]] = []

    for cam_dir in cam_dirs:
        logger.info("Processing %s ...", cam_dir.name)
        report = stabilize_camera(
            input_cam_dir=cam_dir,
            output_cam_dir=output_dir / cam_dir.name,
            smooth_sigma=smooth_sigma,
            do_crop=do_crop,
            plot_dir=output_dir,
        )
        reports.append(report.to_dict())

    manifest: dict[str, Any] = {
        "smooth_sigma_frames": smooth_sigma,
        "crop_enabled": do_crop,
        "cameras": reports,
    }

    manifest_path = output_dir / "stabilization_report.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Report written to %s", manifest_path)

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``stabilize``."""
    parser = argparse.ArgumentParser(
        description=(
            "Remove camera shake from synchronised frame sequences using " "trajectory smoothing."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing cam_XX/ subdirectories with PNG frames.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write stabilised frames.",
    )
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=30.0,
        help="Gaussian smoothing sigma in frames (default: 30).",
    )
    parser.add_argument(
        "--crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Auto-crop black borders (default: enabled, use --no-crop to disable).",
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

    stabilize_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        smooth_sigma=args.smooth_sigma,
        do_crop=args.crop,
    )


if __name__ == "__main__":
    main()
