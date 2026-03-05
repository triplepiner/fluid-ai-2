#!/usr/bin/env python3
"""Resample multi-camera video files to a common target frame rate.

Given a directory of videos (potentially from different cameras at different
frame rates), this script normalises them all to a single target FPS using
FFmpeg.  Three interpolation methods are available:

- **mci**   – motion-compensated temporal interpolation (best quality,
              slowest).  Uses `minterpolate` with adaptive overlapped block
              motion compensation and bidirectional motion estimation.
- **blend** – frame blending via `minterpolate` (good quality, moderate
              speed).
- **drop**  – nearest-frame selection via the `fps` filter (lowest quality,
              fastest).  Simply drops or duplicates frames.

A JSON manifest (``fps_manifest.json``) is written to the output directory
recording per-video metadata before and after normalisation.

Example
-------
::

    python -m preprocessing.normalize_fps \\
        --input_dir  data/raw \\
        --output_dir data/normalized \\
        --target_fps 30 \\
        --method mci
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------


def probe_video(path: Path) -> dict[str, Any]:
    """Return metadata for *path* extracted via ``ffprobe``.

    Returns a dict with keys:
        fps           – average frame rate as a float
        width, height – pixel dimensions
        duration      – duration in seconds (float)
        codec         – video codec name (e.g. ``h264``)
        num_frames    – total frame count (int)
        is_vfr        – True if the container signals variable frame rate

    Raises
    ------
    RuntimeError
        If ``ffprobe`` cannot be executed or returns a non-zero exit code.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=r_frame_rate,avg_frame_rate,nb_frames,width,height," "codec_name,duration",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr.strip()}")

    data = json.loads(result.stdout)
    stream = data["streams"][0]
    fmt = data.get("format", {})

    # Parse frame rates (returned as "num/den" strings).
    avg_fps = _parse_rational(stream.get("avg_frame_rate", "0/1"))
    r_fps = _parse_rational(stream.get("r_frame_rate", "0/1"))
    is_vfr = not math.isclose(avg_fps, r_fps, rel_tol=0.02) if r_fps else False

    # Duration may live in the stream or format section.
    duration_str = stream.get("duration") or fmt.get("duration") or "0"
    duration = float(duration_str)

    # Frame count: ffprobe sometimes reports "N/A".
    nb_frames_raw = stream.get("nb_frames", "0")
    try:
        num_frames = int(nb_frames_raw)
    except (ValueError, TypeError):
        num_frames = round(duration * avg_fps) if duration and avg_fps else 0

    return {
        "fps": avg_fps,
        "width": int(stream.get("width", 0)),
        "height": int(stream.get("height", 0)),
        "duration": duration,
        "codec": stream.get("codec_name", "unknown"),
        "num_frames": num_frames,
        "is_vfr": is_vfr,
    }


def _parse_rational(s: str) -> float:
    """Convert an ``ffprobe`` rational string like ``'30000/1001'`` to float."""
    try:
        num, den = s.split("/")
        return float(num) / float(den) if float(den) != 0 else 0.0
    except (ValueError, ZeroDivisionError):
        return 0.0


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------


def resample_video(
    src: Path,
    dst: Path,
    target_fps: float,
    method: str,
) -> int:
    """Resample *src* to *target_fps* and write the result to *dst*.

    Parameters
    ----------
    src : Path
        Input video file.
    dst : Path
        Output video file.
    target_fps : float
        Desired output frame rate.
    method : str
        One of ``"mci"``, ``"blend"``, or ``"drop"``.

    Returns
    -------
    int
        Frame count of the output file.

    Raises
    ------
    RuntimeError
        If the FFmpeg process exits with a non-zero return code.
    """
    if method == "mci":
        vf = f"minterpolate=fps={target_fps}:mi_mode=mci" f":mc_mode=aobmc:me_mode=bidir:vsbmc=1"
    elif method == "blend":
        vf = f"minterpolate=fps={target_fps}:mi_mode=blend"
    elif method == "drop":
        vf = f"fps=fps={target_fps}"
    else:
        raise ValueError(f"Unknown interpolation method: {method!r}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-vf",
        vf,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "medium",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        str(dst),
    ]

    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg failed for {src}:\n{result.stderr.strip()}")

    # Probe the output to get the actual frame count.
    out_info = probe_video(dst)
    return out_info["num_frames"]


def copy_video(src: Path, dst: Path) -> None:
    """Copy *src* to *dst* without re-encoding."""
    shutil.copy2(src, dst)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def normalize_directory(
    input_dir: Path,
    output_dir: Path,
    target_fps: float = 30.0,
    method: str = "mci",
) -> list[dict[str, Any]]:
    """Normalise all videos in *input_dir* to *target_fps*.

    Parameters
    ----------
    input_dir : Path
        Directory containing raw video files.
    output_dir : Path
        Destination directory for normalised videos.
    target_fps : float
        Target frame rate.
    method : str
        Interpolation method (``"mci"``, ``"blend"``, or ``"drop"``).

    Returns
    -------
    list[dict]
        Manifest entries (one per processed video).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if not video_files:
        logger.warning("No video files found in %s", input_dir)
        return []

    logger.info(
        "Found %d video(s) in %s — target FPS: %g, method: %s",
        len(video_files),
        input_dir,
        target_fps,
        method,
    )

    manifest: list[dict[str, Any]] = []

    for idx, src in enumerate(video_files, 1):
        logger.info(
            "[%d/%d] Processing %s",
            idx,
            len(video_files),
            src.name,
        )

        # --- Probe source ---------------------------------------------------
        try:
            info = probe_video(src)
        except RuntimeError:
            logger.exception("Failed to probe %s — skipping.", src.name)
            continue

        original_fps = info["fps"]
        logger.info(
            "  %s: %dx%d, %.3f fps, %.2fs, codec=%s, frames=%d",
            src.name,
            info["width"],
            info["height"],
            original_fps,
            info["duration"],
            info["codec"],
            info["num_frames"],
        )

        if info["is_vfr"]:
            logger.warning(
                "  %s appears to have variable frame rate (avg=%.3f, "
                "r_frame_rate differs). Output timing may be approximate.",
                src.name,
                original_fps,
            )

        dst = output_dir / src.with_suffix(".mp4").name

        # --- Decide: copy or resample --------------------------------------
        fps_matches = math.isclose(original_fps, target_fps, abs_tol=0.1)

        if fps_matches:
            logger.info(
                "  FPS already matches target (%.3f ≈ %g) — copying.",
                original_fps,
                target_fps,
            )
            copy_video(src, dst)
            new_frame_count = info["num_frames"]
            used_method = "copy"
        else:
            logger.info(
                "  Resampling %.3f → %g fps (method=%s)...",
                original_fps,
                target_fps,
                method,
            )
            try:
                new_frame_count = resample_video(src, dst, target_fps, method)
            except RuntimeError:
                logger.exception(
                    "  FFmpeg failed on %s — skipping.",
                    src.name,
                )
                continue
            used_method = method

        logger.info(
            "  Done: %s → %d frames.",
            dst.name,
            new_frame_count,
        )

        manifest.append(
            {
                "original_filename": src.name,
                "output_filename": dst.name,
                "original_fps": round(original_fps, 4),
                "target_fps": target_fps,
                "method": used_method,
                "original_frame_count": info["num_frames"],
                "new_frame_count": new_frame_count,
                "duration_seconds": round(info["duration"], 4),
                "resolution": f"{info['width']}x{info['height']}",
                "is_vfr": info["is_vfr"],
            }
        )

    return manifest


def write_manifest(
    manifest: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write the manifest list to ``fps_manifest.json`` inside *output_dir*.

    Parameters
    ----------
    manifest : list[dict]
        Per-video metadata entries.
    output_dir : Path
        Directory where the JSON file is created.

    Returns
    -------
    Path
        Absolute path to the written manifest file.
    """
    path = output_dir / "fps_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s (%d entries)", path, len(manifest))
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``normalize_fps``."""
    parser = argparse.ArgumentParser(
        description=("Resample multi-camera videos to a common frame rate using FFmpeg."),
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing raw video files (mp4, mov, avi).",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write normalised videos.",
    )
    parser.add_argument(
        "--target_fps",
        type=float,
        default=30.0,
        help="Target frame rate (default: 30).",
    )
    parser.add_argument(
        "--method",
        choices=["mci", "blend", "drop"],
        default="mci",
        help=(
            "Interpolation method: "
            "mci (motion-compensated, best quality), "
            "blend (frame blending), "
            "drop (nearest frame, fastest). "
            "Default: mci."
        ),
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

    manifest = normalize_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        target_fps=args.target_fps,
        method=args.method,
    )

    if manifest:
        write_manifest(manifest, args.output_dir)
    else:
        logger.warning("No videos were processed.")


if __name__ == "__main__":
    main()
