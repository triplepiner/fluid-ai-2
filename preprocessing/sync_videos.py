#!/usr/bin/env python3
"""Temporally synchronise multi-camera video recordings.

When a fluid scene is recorded with several cameras, each camera typically
starts at a slightly different wall-clock time.  This script determines the
time offset between every camera and a chosen reference (camera 0), then
extracts the overlapping frame range so that ``frame N`` across all output
directories corresponds to the same real-world instant.

Two synchronisation strategies are available:

- **audio** (default) – extract mono 16 kHz audio tracks and cross-correlate
  them.  Best when the cameras share a common acoustic environment.
- **visual** – compute mean frame brightness per frame and cross-correlate
  the resulting 1-D signals.  Useful when audio is absent or silent.

Outputs
-------
- ``cam_XX/frame_NNNNN.png`` – synchronised PNG frame sequences.
- ``sync_manifest.json``     – per-camera offsets, confidences, overlap window.
- ``cross_correlation.png``  – visualisation of correlation curves.

Example
-------
::

    python -m preprocessing.sync_videos \\
        --input_dir  data/normalized \\
        --output_dir data/synced \\
        --fps 30 \\
        --method audio
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np
import scipy.signal
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi"}


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def extract_audio(video_path: Path, wav_path: Path, sample_rate: int = 16000) -> bool:
    """Extract a mono audio track from *video_path* as a 16-bit WAV.

    Parameters
    ----------
    video_path : Path
        Source video file.
    wav_path : Path
        Destination ``.wav`` file.
    sample_rate : int
        Output sample rate in Hz.

    Returns
    -------
    bool
        ``True`` if extraction succeeded, ``False`` if the video has no
        audio stream or FFmpeg failed.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-sample_fmt",
        "s16",
        str(wav_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(
            "Audio extraction failed for %s (video may lack an audio stream).",
            video_path.name,
        )
        return False
    if not wav_path.exists() or wav_path.stat().st_size < 100:
        logger.warning("Extracted audio file is empty for %s.", video_path.name)
        return False
    return True


def load_wav_mono(wav_path: Path) -> tuple[np.ndarray, int]:
    """Read a mono WAV file and return ``(samples, sample_rate)``.

    Parameters
    ----------
    wav_path : Path
        Path to a 16-bit mono WAV.

    Returns
    -------
    samples : np.ndarray
        1-D float64 array normalised to [-1, 1].
    sample_rate : int
        Sample rate in Hz.
    """
    with wave.open(str(wav_path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sampwidth, np.int16)
    samples = np.frombuffer(raw, dtype=dtype).astype(np.float64)

    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)

    peak = np.max(np.abs(samples))
    if peak > 0:
        samples /= peak
    return samples, sample_rate


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------


def compute_offset_audio(
    ref_audio: np.ndarray,
    tgt_audio: np.ndarray,
    sample_rate: int,
    max_offset_s: float,
) -> tuple[float, float]:
    """Find the time offset of *tgt_audio* relative to *ref_audio*.

    Parameters
    ----------
    ref_audio, tgt_audio : np.ndarray
        1-D float audio signals (same sample rate).
    sample_rate : int
        Sample rate in Hz.
    max_offset_s : float
        Maximum lag to search in seconds.

    Returns
    -------
    offset_s : float
        Offset in seconds.  Positive means *tgt* starts **after** *ref*.
    confidence : float
        Ratio of the primary peak to the strongest secondary peak (outside
        a ±0.5 s exclusion zone around the primary).  Higher is better.
    """
    max_lag = int(max_offset_s * sample_rate)

    corr = scipy.signal.correlate(ref_audio, tgt_audio, mode="full")
    lags = scipy.signal.correlation_lags(len(ref_audio), len(tgt_audio), mode="full")

    # Restrict to the allowed search window.
    mask = np.abs(lags) <= max_lag
    corr_w = corr[mask]
    lags_w = lags[mask]

    if len(corr_w) == 0:
        return 0.0, 0.0

    # Normalise for interpretability.
    norm = np.sqrt(np.sum(ref_audio**2) * np.sum(tgt_audio**2))
    if norm > 0:
        corr_w = corr_w / norm

    peak_idx = int(np.argmax(corr_w))
    peak_val = corr_w[peak_idx]
    best_lag = int(lags_w[peak_idx])
    offset_s = best_lag / sample_rate

    # Confidence: ratio to second-highest peak outside ±0.5 s exclusion.
    exclusion_samples = int(0.5 * sample_rate)
    secondary_mask = np.ones(len(corr_w), dtype=bool)
    lo = max(0, peak_idx - exclusion_samples)
    hi = min(len(corr_w), peak_idx + exclusion_samples + 1)
    secondary_mask[lo:hi] = False

    if np.any(secondary_mask):
        second_peak = float(np.max(corr_w[secondary_mask]))
        confidence = float(peak_val / second_peak) if second_peak > 0 else float("inf")
    else:
        confidence = float("inf")

    return offset_s, confidence


def compute_offset_visual(
    ref_brightness: np.ndarray,
    tgt_brightness: np.ndarray,
    fps: float,
    max_offset_s: float,
) -> tuple[float, float]:
    """Find the frame offset of *tgt* relative to *ref* using brightness signals.

    Parameters
    ----------
    ref_brightness, tgt_brightness : np.ndarray
        1-D arrays of per-frame mean brightness.
    fps : float
        Common frame rate.
    max_offset_s : float
        Maximum lag to search in seconds.

    Returns
    -------
    offset_s : float
        Offset in seconds.
    confidence : float
        Primary-to-secondary peak ratio.
    """
    max_lag = int(max_offset_s * fps)

    # Subtract mean so cross-correlation responds to *changes* in brightness.
    ref_z = ref_brightness - np.mean(ref_brightness)
    tgt_z = tgt_brightness - np.mean(tgt_brightness)

    corr = scipy.signal.correlate(ref_z, tgt_z, mode="full")
    lags = scipy.signal.correlation_lags(len(ref_z), len(tgt_z), mode="full")

    mask = np.abs(lags) <= max_lag
    corr_w = corr[mask]
    lags_w = lags[mask]

    if len(corr_w) == 0:
        return 0.0, 0.0

    norm = np.sqrt(np.sum(ref_z**2) * np.sum(tgt_z**2))
    if norm > 0:
        corr_w = corr_w / norm

    peak_idx = int(np.argmax(corr_w))
    peak_val = corr_w[peak_idx]
    best_lag = int(lags_w[peak_idx])
    offset_s = best_lag / fps

    exclusion_frames = int(0.5 * fps)
    secondary_mask = np.ones(len(corr_w), dtype=bool)
    lo = max(0, peak_idx - exclusion_frames)
    hi = min(len(corr_w), peak_idx + exclusion_frames + 1)
    secondary_mask[lo:hi] = False

    if np.any(secondary_mask):
        second_peak = float(np.max(corr_w[secondary_mask]))
        confidence = float(peak_val / second_peak) if second_peak > 0 else float("inf")
    else:
        confidence = float("inf")

    return offset_s, confidence


# ---------------------------------------------------------------------------
# Brightness extraction
# ---------------------------------------------------------------------------


def extract_brightness(video_path: Path) -> np.ndarray:
    """Return a 1-D array of per-frame mean brightness for *video_path*.

    Parameters
    ----------
    video_path : Path
        Input video file readable by OpenCV.

    Returns
    -------
    np.ndarray
        Float array of length ``num_frames``.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    brightness = []

    for _ in tqdm(range(total), desc=f"  brightness {video_path.name}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness.append(float(np.mean(gray)))

    cap.release()
    return np.array(brightness, dtype=np.float64)


# ---------------------------------------------------------------------------
# Frame extraction
# ---------------------------------------------------------------------------


def get_video_frame_count(video_path: Path) -> int:
    """Return the frame count of *video_path* via OpenCV."""
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def extract_frames(
    video_path: Path,
    output_dir: Path,
    start_frame: int,
    num_frames: int,
) -> int:
    """Write a range of frames from *video_path* as numbered PNGs.

    Parameters
    ----------
    video_path : Path
        Source video.
    output_dir : Path
        Destination directory (created if needed).
    start_frame : int
        0-based index of the first frame to extract.
    num_frames : int
        Number of frames to extract.

    Returns
    -------
    int
        Number of frames actually written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    written = 0

    for i in tqdm(range(num_frames), desc=f"  frames {video_path.name}", leave=False):
        ret, frame = cap.read()
        if not ret:
            break
        out_path = output_dir / f"frame_{i:05d}.png"
        cv2.imwrite(str(out_path), frame)
        written += 1

    cap.release()
    return written


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_correlations(
    corr_data: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save a multi-panel cross-correlation plot.

    Parameters
    ----------
    corr_data : list[dict]
        One entry per non-reference camera, each with keys:
        ``name``, ``lags``, ``corr``, ``best_lag``, ``offset_s``, ``confidence``.
    output_path : Path
        Destination PNG file.
    """
    n = len(corr_data)
    if n == 0:
        return

    fig, axes = plt.subplots(n, 1, figsize=(12, 3.5 * n), squeeze=False)

    for ax_row, entry in zip(axes, corr_data):
        ax = ax_row[0]
        lags = entry["lags"]
        corr = entry["corr"]
        best = entry["best_lag"]

        ax.plot(lags, corr, linewidth=0.6)
        ax.axvline(best, color="red", linestyle="--", linewidth=1.0, label=f"peak lag={best}")
        ax.set_title(
            f"{entry['name']}  —  offset={entry['offset_s']:.4f}s  "
            f"confidence={entry['confidence']:.2f}",
        )
        ax.set_xlabel("Lag")
        ax.set_ylabel("Normalised correlation")
        ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Cross-correlation plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def sync_videos(
    input_dir: Path,
    output_dir: Path,
    fps: float,
    method: str,
    max_offset: float,
) -> dict[str, Any]:
    """Synchronise all videos in *input_dir* and extract aligned frames.

    Parameters
    ----------
    input_dir : Path
        Directory containing fps-normalised video files.
    output_dir : Path
        Destination for synchronised PNG sequences.
    fps : float
        Common frame rate of all input videos.
    method : str
        ``"audio"`` or ``"visual"``.
    max_offset : float
        Maximum expected offset in seconds.

    Returns
    -------
    dict
        Manifest dictionary (also written to ``sync_manifest.json``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    video_files = sorted(
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
    )

    if len(video_files) < 2:
        logger.error(
            "Need at least 2 videos to synchronise, found %d in %s.",
            len(video_files),
            input_dir,
        )
        sys.exit(1)

    logger.info(
        "Found %d video(s) — method=%s, fps=%g, max_offset=%.1fs",
        len(video_files),
        method,
        fps,
        max_offset,
    )

    # ------------------------------------------------------------------
    # Step 1: compute per-camera offsets relative to camera 0 (reference)
    # ------------------------------------------------------------------
    ref_path = video_files[0]
    logger.info("Reference camera: %s", ref_path.name)

    # Pre-load reference signal.
    ref_signal: np.ndarray | None = None
    sample_rate: int = 16000
    corr_plot_data: list[dict[str, Any]] = []
    fallback_to_visual = False

    if method == "audio":
        with tempfile.TemporaryDirectory() as tmp:
            ref_wav = Path(tmp) / "ref.wav"
            if not extract_audio(ref_path, ref_wav, sample_rate):
                logger.warning(
                    "Reference video has no audio — falling back to visual method.",
                )
                fallback_to_visual = True
            else:
                ref_signal, sample_rate = load_wav_mono(ref_wav)

                # Check that audio is not silence (RMS < threshold).
                ref_rms = float(np.sqrt(np.mean(ref_signal**2)))
                if ref_rms < 1e-4:
                    logger.warning(
                        "Reference audio is near-silent (RMS=%.6f) "
                        "— falling back to visual method.",
                        ref_rms,
                    )
                    fallback_to_visual = True

            if not fallback_to_visual:
                offsets = _compute_offsets_audio(
                    ref_signal,
                    sample_rate,
                    video_files,
                    max_offset,
                    Path(tmp),
                    corr_plot_data,
                )
            else:
                method = "visual"

    if method == "visual" or fallback_to_visual:
        logger.info("Computing per-frame brightness for reference...")
        ref_brightness = extract_brightness(ref_path)
        offsets = _compute_offsets_visual(
            ref_brightness,
            fps,
            video_files,
            max_offset,
            corr_plot_data,
        )

    # ------------------------------------------------------------------
    # Step 2: determine the overlapping window
    # ------------------------------------------------------------------
    frame_counts = [get_video_frame_count(v) for v in video_files]

    # offset_frames[i] = how many frames into video i does the overlap start.
    # A positive offset means video i starts *after* the reference.
    offset_frames = [round(o * fps) for o in offsets]

    # The latest start across all cameras (in "reference time" frames).
    global_start = max(offset_frames)

    # Per-camera local start frame and available length from that point.
    local_starts: list[int] = []
    available: list[int] = []
    for i, (of, fc) in enumerate(zip(offset_frames, frame_counts)):
        local_start = global_start - of
        local_starts.append(local_start)
        available.append(fc - local_start)

    total_sync_frames = max(0, min(available))
    overlap_start_s = global_start / fps
    overlap_end_s = overlap_start_s + total_sync_frames / fps

    logger.info(
        "Overlap window: %.3f–%.3f s  (%d synchronised frames)",
        overlap_start_s,
        overlap_end_s,
        total_sync_frames,
    )

    if total_sync_frames == 0:
        logger.error("No overlapping frames — cameras may not share a common time window.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Step 3: extract synchronised frames
    # ------------------------------------------------------------------
    cam_entries: list[dict[str, Any]] = []

    for i, (vpath, ls) in enumerate(zip(video_files, local_starts)):
        cam_dir = output_dir / f"cam_{i:02d}"
        logger.info(
            "Extracting %d frames from %s (start_frame=%d) → %s",
            total_sync_frames,
            vpath.name,
            ls,
            cam_dir,
        )
        written = extract_frames(vpath, cam_dir, ls, total_sync_frames)
        logger.info("  Wrote %d frames.", written)

        cam_entries.append(
            {
                "camera_index": i,
                "video_filename": vpath.name,
                "offset_seconds": round(offsets[i], 6),
                "offset_frames": offset_frames[i],
                "confidence": round(corr_plot_data[i - 1]["confidence"], 4) if i > 0 else None,
                "local_start_frame": ls,
                "frames_written": written,
            }
        )

    # ------------------------------------------------------------------
    # Step 4: save artefacts
    # ------------------------------------------------------------------
    manifest: dict[str, Any] = {
        "method": method,
        "fps": fps,
        "max_offset_seconds": max_offset,
        "overlap_start_seconds": round(overlap_start_s, 6),
        "overlap_end_seconds": round(overlap_end_s, 6),
        "total_synchronized_frames": total_sync_frames,
        "cameras": cam_entries,
    }

    manifest_path = output_dir / "sync_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written to %s", manifest_path)

    if corr_plot_data:
        plot_correlations(corr_plot_data, output_dir / "cross_correlation.png")

    return manifest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_offsets_audio(
    ref_signal: np.ndarray,
    sample_rate: int,
    video_files: list[Path],
    max_offset: float,
    tmp_dir: Path,
    corr_plot_data: list[dict[str, Any]],
) -> list[float]:
    """Return per-camera offsets (seconds) using audio cross-correlation.

    Parameters
    ----------
    ref_signal : np.ndarray
        Reference camera audio (normalised float64).
    sample_rate : int
        Audio sample rate.
    video_files : list[Path]
        All video paths (index 0 is the reference).
    max_offset : float
        Maximum lag in seconds.
    tmp_dir : Path
        Temporary directory for intermediate WAV files.
    corr_plot_data : list[dict]
        Populated in-place with per-camera correlation curves for plotting.

    Returns
    -------
    list[float]
        Offset in seconds for each camera (index 0 is always 0.0).
    """
    offsets = [0.0]

    for i, vpath in enumerate(video_files[1:], start=1):
        logger.info("Computing audio offset for %s...", vpath.name)

        wav_path = tmp_dir / f"cam_{i}.wav"
        if not extract_audio(vpath, wav_path, sample_rate):
            logger.warning("  No audio — assuming zero offset for %s.", vpath.name)
            offsets.append(0.0)
            corr_plot_data.append(
                {
                    "name": vpath.name,
                    "lags": np.array([0]),
                    "corr": np.array([0.0]),
                    "best_lag": 0,
                    "offset_s": 0.0,
                    "confidence": 0.0,
                }
            )
            continue

        tgt_signal, _ = load_wav_mono(wav_path)
        offset_s, confidence = compute_offset_audio(
            ref_signal,
            tgt_signal,
            sample_rate,
            max_offset,
        )

        logger.info(
            "  %s: offset=%.4fs (%.1f frames @ %dHz), confidence=%.2f",
            vpath.name,
            offset_s,
            offset_s * sample_rate,
            sample_rate,
            confidence,
        )

        offsets.append(offset_s)

        # Store correlation curve for plotting.
        max_lag = int(max_offset * sample_rate)
        corr_full = scipy.signal.correlate(ref_signal, tgt_signal, mode="full")
        lags_full = scipy.signal.correlation_lags(len(ref_signal), len(tgt_signal), mode="full")
        mask = np.abs(lags_full) <= max_lag
        norm = np.sqrt(np.sum(ref_signal**2) * np.sum(tgt_signal**2))
        corr_plot_data.append(
            {
                "name": vpath.name,
                "lags": lags_full[mask],
                "corr": corr_full[mask] / norm if norm > 0 else corr_full[mask],
                "best_lag": round(offset_s * sample_rate),
                "offset_s": offset_s,
                "confidence": confidence,
            }
        )

    return offsets


def _compute_offsets_visual(
    ref_brightness: np.ndarray,
    fps: float,
    video_files: list[Path],
    max_offset: float,
    corr_plot_data: list[dict[str, Any]],
) -> list[float]:
    """Return per-camera offsets (seconds) using visual cross-correlation.

    Parameters
    ----------
    ref_brightness : np.ndarray
        Reference camera mean-brightness signal.
    fps : float
        Common frame rate.
    video_files : list[Path]
        All video paths (index 0 is the reference).
    max_offset : float
        Maximum lag in seconds.
    corr_plot_data : list[dict]
        Populated in-place with per-camera correlation curves for plotting.

    Returns
    -------
    list[float]
        Offset in seconds for each camera (index 0 is always 0.0).
    """
    offsets = [0.0]

    ref_z = ref_brightness - np.mean(ref_brightness)

    for i, vpath in enumerate(video_files[1:], start=1):
        logger.info("Computing visual offset for %s...", vpath.name)
        tgt_brightness = extract_brightness(vpath)
        offset_s, confidence = compute_offset_visual(
            ref_brightness,
            tgt_brightness,
            fps,
            max_offset,
        )
        offset_frames = round(offset_s * fps)

        logger.info(
            "  %s: offset=%.4fs (%d frames), confidence=%.2f",
            vpath.name,
            offset_s,
            offset_frames,
            confidence,
        )

        offsets.append(offset_s)

        # Store correlation curve for plotting.
        max_lag = int(max_offset * fps)
        tgt_z = tgt_brightness - np.mean(tgt_brightness)
        corr_full = scipy.signal.correlate(ref_z, tgt_z, mode="full")
        lags_full = scipy.signal.correlation_lags(len(ref_z), len(tgt_z), mode="full")
        mask = np.abs(lags_full) <= max_lag
        norm = np.sqrt(np.sum(ref_z**2) * np.sum(tgt_z**2))
        corr_plot_data.append(
            {
                "name": vpath.name,
                "lags": lags_full[mask],
                "corr": corr_full[mask] / norm if norm > 0 else corr_full[mask],
                "best_lag": offset_frames,
                "offset_s": offset_s,
                "confidence": confidence,
            }
        )

    return offsets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``sync_videos``."""
    parser = argparse.ArgumentParser(
        description=(
            "Temporally synchronise multi-camera recordings using audio or "
            "visual cross-correlation, then extract aligned frame sequences."
        ),
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        required=True,
        help="Directory containing fps-normalised video files.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Directory to write synchronised PNG sequences.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Common frame rate of the input videos (default: 30).",
    )
    parser.add_argument(
        "--method",
        choices=["audio", "visual"],
        default="audio",
        help="Synchronisation method (default: audio).",
    )
    parser.add_argument(
        "--max_offset",
        type=float,
        default=10.0,
        help="Maximum expected offset in seconds (default: 10).",
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

    sync_videos(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        fps=args.fps,
        method=args.method,
        max_offset=args.max_offset,
    )


if __name__ == "__main__":
    main()
