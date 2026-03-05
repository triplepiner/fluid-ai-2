#!/usr/bin/env python3
"""Estimate camera poses from multi-view frame sequences using COLMAP.

Given stabilised frame directories (``cam_00/``, ``cam_01/``, …) this script:

1. Selects representative frames from each camera.
2. Runs the COLMAP sparse reconstruction pipeline via subprocess
   (feature extraction → exhaustive matching → mapping → text export).
3. Parses the text output to recover intrinsics, extrinsics, and a sparse
   point cloud.
4. Averages per-frame poses from the same physical camera into a single
   robust estimate.
5. Normalises the scene so the median camera distance from the centroid
   is 1.0.
6. Exports ``cameras.json``, ``points3d.ply``, and ``camera_layout.png``.

No ``pycolmap`` dependency — only the COLMAP command-line binary is used.

Example
-------
::

    python -m preprocessing.run_colmap \\
        --input_dir  data/stabilized \\
        --output_dir data/colmap \\
        --quality medium \\
        --frames_per_cam 5
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from plyfile import PlyData, PlyElement

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

logger = logging.getLogger(__name__)

# Quality presets: (max_image_size, max_num_features)
_QUALITY = {
    "low": (800, 4096),
    "medium": (1600, 8192),
    "high": (3200, 16384),
}


# ---------------------------------------------------------------------------
# COLMAP binary helpers
# ---------------------------------------------------------------------------


def find_colmap(colmap_bin: str) -> str:
    """Resolve the COLMAP binary path and verify it is executable.

    Parameters
    ----------
    colmap_bin : str
        Name or path of the COLMAP binary.

    Returns
    -------
    str
        Absolute path to the COLMAP binary.

    Raises
    ------
    SystemExit
        If COLMAP is not found.
    """
    path = shutil.which(colmap_bin)
    if path is None:
        logger.error(
            "COLMAP binary '%s' not found on PATH.\n\n"
            "Installation instructions:\n"
            "  macOS:   brew install colmap\n"
            "  Ubuntu:  sudo apt install colmap\n"
            "  From source: https://colmap.github.io/install.html\n",
            colmap_bin,
        )
        sys.exit(1)
    return path


def colmap_version(colmap_bin: str) -> str | None:
    """Return the COLMAP version string, or ``None`` on failure.

    Parameters
    ----------
    colmap_bin : str
        Path to the COLMAP binary.

    Returns
    -------
    str or None
        Version string such as ``"3.8"`` or ``"3.13.0"``.
    """
    result = subprocess.run(
        [colmap_bin, "help"],
        capture_output=True,
        text=True,
    )
    combined = result.stdout + result.stderr
    m = re.search(r"COLMAP\s+([\d.]+)", combined)
    return m.group(1) if m else None


def _colmap_flag_style(colmap_bin: str) -> str:
    """Detect whether COLMAP uses old or new CLI flag names.

    COLMAP ≥3.10 moved ``use_gpu`` from ``SiftExtraction`` to
    ``FeatureExtraction`` and ``guided_matching`` from ``SiftMatching`` to
    ``FeatureMatching``.

    Returns ``"new"`` or ``"old"``.
    """
    result = subprocess.run(
        [colmap_bin, "feature_extractor", "--help"],
        capture_output=True,
        text=True,
    )
    text = result.stdout + result.stderr
    if "--FeatureExtraction.use_gpu" in text:
        return "new"
    return "old"


def _run_colmap(colmap_bin: str, cmd: str, args: list[str]) -> subprocess.CompletedProcess:
    """Execute a COLMAP sub-command and return the completed process.

    Parameters
    ----------
    colmap_bin : str
        Path to the COLMAP binary.
    cmd : str
        COLMAP sub-command (e.g. ``"feature_extractor"``).
    args : list[str]
        Additional CLI arguments.

    Returns
    -------
    subprocess.CompletedProcess

    Raises
    ------
    RuntimeError
        If COLMAP exits with a non-zero return code.
    """
    full_cmd = [colmap_bin, cmd] + args
    logger.debug("Running: %s", " ".join(full_cmd))
    result = subprocess.run(full_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr_tail = result.stderr.strip()[-1000:] if result.stderr else ""
        raise RuntimeError(f"COLMAP {cmd} failed (exit {result.returncode}):\n{stderr_tail}")
    return result


# ---------------------------------------------------------------------------
# Frame selection
# ---------------------------------------------------------------------------


def select_frames(
    cam_dirs: list[Path],
    frames_per_cam: int,
    dest_dir: Path,
) -> dict[str, list[str]]:
    """Copy representative frames from each camera into *dest_dir*.

    Frames are evenly spaced in time.  File names are prefixed with the
    camera name so COLMAP treats them as separate images.

    Parameters
    ----------
    cam_dirs : list[Path]
        Sorted list of ``cam_XX/`` directories.
    frames_per_cam : int
        Number of frames to select per camera.
    dest_dir : Path
        Target directory (``colmap_input/images/``).

    Returns
    -------
    dict[str, list[str]]
        Mapping from camera name to list of copied file names (basenames).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    selection: dict[str, list[str]] = {}

    for cam_dir in cam_dirs:
        all_frames = sorted(cam_dir.glob("frame_*.png"))
        if not all_frames:
            logger.warning("No frames in %s — skipping.", cam_dir.name)
            continue

        n = len(all_frames)
        if frames_per_cam >= n:
            indices = list(range(n))
        else:
            indices = [round(i * (n - 1) / (frames_per_cam - 1)) for i in range(frames_per_cam)]

        copied: list[str] = []
        for idx in indices:
            src = all_frames[idx]
            dst_name = f"{cam_dir.name}_{src.name}"
            shutil.copy2(src, dest_dir / dst_name)
            copied.append(dst_name)

        selection[cam_dir.name] = copied
        logger.info(
            "  %s: selected %d frames (%s … %s)", cam_dir.name, len(copied), copied[0], copied[-1]
        )

    return selection


# ---------------------------------------------------------------------------
# COLMAP pipeline
# ---------------------------------------------------------------------------


def run_colmap_pipeline(
    colmap_bin: str,
    image_dir: Path,
    workspace: Path,
    quality: str,
    use_gpu: bool,
) -> Path | None:
    """Run the full COLMAP sparse reconstruction pipeline.

    Parameters
    ----------
    colmap_bin : str
        Path to the COLMAP binary.
    image_dir : Path
        Directory containing the input images.
    workspace : Path
        Working directory for the database and reconstruction output.
    quality : str
        One of ``"low"``, ``"medium"``, ``"high"``.
    use_gpu : bool
        Whether to enable GPU-accelerated feature extraction/matching.

    Returns
    -------
    Path or None
        Path to the best reconstruction model directory (e.g. ``sparse/0/``),
        or ``None`` if the mapper produced no models.
    """
    max_size, max_feat = _QUALITY[quality]
    db_path = workspace / "database.db"
    sparse_dir = workspace / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Detect CLI flag style (COLMAP ≥3.10 renamed several flags).
    style = _colmap_flag_style(colmap_bin)
    gpu_val = "1" if use_gpu else "0"
    logger.debug("  COLMAP flag style: %s", style)

    # 1. Feature extraction
    logger.info("  [1/4] Feature extraction (max_size=%d, max_features=%d)...", max_size, max_feat)
    extract_args = [
        "--database_path",
        str(db_path),
        "--image_path",
        str(image_dir),
        "--ImageReader.camera_model",
        "OPENCV",
        "--ImageReader.single_camera_per_folder",
        "0",
        "--SiftExtraction.max_image_size",
        str(max_size),
        "--SiftExtraction.max_num_features",
        str(max_feat),
    ]
    if style == "new":
        extract_args += ["--FeatureExtraction.use_gpu", gpu_val]
    else:
        extract_args += ["--SiftExtraction.use_gpu", gpu_val]
    _run_colmap(colmap_bin, "feature_extractor", extract_args)

    # 2. Exhaustive matching
    logger.info("  [2/4] Exhaustive matching...")
    match_args = ["--database_path", str(db_path)]
    if style == "new":
        match_args += [
            "--FeatureMatching.guided_matching",
            "1",
            "--FeatureMatching.use_gpu",
            gpu_val,
        ]
    else:
        match_args += [
            "--SiftMatching.guided_matching",
            "1",
            "--SiftMatching.use_gpu",
            gpu_val,
        ]
    _run_colmap(colmap_bin, "exhaustive_matcher", match_args)

    # 3. Sparse reconstruction (mapper)
    logger.info("  [3/4] Sparse reconstruction (mapper)...")
    _run_colmap(
        colmap_bin,
        "mapper",
        [
            "--database_path",
            str(db_path),
            "--image_path",
            str(image_dir),
            "--output_path",
            str(sparse_dir),
            "--Mapper.ba_global_max_num_iterations",
            "50",
        ],
    )

    # Find the best model (most registered images).
    model_dirs = sorted(sparse_dir.iterdir())
    if not model_dirs:
        logger.error("COLMAP mapper produced no reconstruction models.")
        return None

    best = model_dirs[0]
    logger.info("  Using reconstruction model: %s", best.name)

    # 4. Convert to text
    txt_dir = workspace / "sparse_txt"
    txt_dir.mkdir(parents=True, exist_ok=True)
    logger.info("  [4/4] Converting model to text format...")
    _run_colmap(
        colmap_bin,
        "model_converter",
        [
            "--input_path",
            str(best),
            "--output_path",
            str(txt_dir),
            "--output_type",
            "TXT",
        ],
    )

    return txt_dir


# ---------------------------------------------------------------------------
# COLMAP text parsers
# ---------------------------------------------------------------------------


def parse_cameras_txt(path: Path) -> dict[int, dict[str, Any]]:
    """Parse a COLMAP ``cameras.txt`` file.

    Parameters
    ----------
    path : Path
        Path to ``cameras.txt``.

    Returns
    -------
    dict[int, dict]
        Mapping from camera_id to a dict with ``model``, ``width``,
        ``height``, and ``params`` (list of floats).
    """
    cameras: dict[int, dict[str, Any]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = [float(x) for x in parts[4:]]
            cameras[cam_id] = {
                "model": model,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def parse_images_txt(path: Path) -> list[dict[str, Any]]:
    """Parse a COLMAP ``images.txt`` file.

    The file alternates between image header lines (qw qx qy qz tx ty tz
    camera_id name) and 2-D point lines.  We only need the headers.

    Parameters
    ----------
    path : Path
        Path to ``images.txt``.

    Returns
    -------
    list[dict]
        One entry per registered image with keys: ``image_id``, ``qvec``
        (w,x,y,z), ``tvec`` (3-list), ``camera_id``, ``name``.
    """
    images: list[dict[str, Any]] = []
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # Lines alternate: header, points2d
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        image_id = int(parts[0])
        qvec = [float(parts[j]) for j in range(1, 5)]  # qw qx qy qz
        tvec = [float(parts[j]) for j in range(5, 8)]
        camera_id = int(parts[8])
        name = parts[9]
        images.append(
            {
                "image_id": image_id,
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
        )

    return images


def parse_points3d_txt(path: Path) -> list[dict[str, Any]]:
    """Parse a COLMAP ``points3D.txt`` file.

    Parameters
    ----------
    path : Path
        Path to ``points3D.txt``.

    Returns
    -------
    list[dict]
        One entry per 3-D point with keys ``xyz`` (3-list), ``rgb``
        (3-list of ints), ``error`` (float).
    """
    points: list[dict[str, Any]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
            rgb = [int(parts[4]), int(parts[5]), int(parts[6])]
            error = float(parts[7])
            points.append({"xyz": xyz, "rgb": rgb, "error": error})
    return points


# ---------------------------------------------------------------------------
# Quaternion / rotation utilities
# ---------------------------------------------------------------------------


def qvec_to_rotmat(qvec: list[float]) -> np.ndarray:
    """Convert a COLMAP quaternion (w, x, y, z) to a 3×3 rotation matrix.

    Parameters
    ----------
    qvec : list[float]
        Quaternion in COLMAP order ``[w, x, y, z]``.

    Returns
    -------
    np.ndarray
        3×3 rotation matrix.
    """
    w, x, y, z = qvec
    R = np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * w * x + 2 * y * z, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )
    # Re-orthogonalise via SVD.
    U, _, Vt = np.linalg.svd(R)
    return U @ Vt


def rotmat_to_qvec(R: np.ndarray) -> list[float]:
    """Convert a 3×3 rotation matrix to a quaternion (w, x, y, z).

    Parameters
    ----------
    R : np.ndarray
        3×3 rotation matrix.

    Returns
    -------
    list[float]
        Quaternion ``[w, x, y, z]``.
    """
    tr = np.trace(R)
    if tr > 0:
        s = 0.5 / math.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = [w, x, y, z]
    norm = math.sqrt(sum(v * v for v in q))
    return [v / norm for v in q]


def average_quaternions(quats: list[list[float]]) -> list[float]:
    """Average quaternions via the eigenvector method.

    Parameters
    ----------
    quats : list[list[float]]
        Each quaternion is ``[w, x, y, z]``.

    Returns
    -------
    list[float]
        Averaged quaternion ``[w, x, y, z]``, unit-normalised.
    """
    Q = np.array(quats)  # (N, 4)
    # Ensure consistent hemisphere.
    for i in range(1, len(Q)):
        if np.dot(Q[i], Q[0]) < 0:
            Q[i] *= -1
    M = Q.T @ Q
    eigvals, eigvecs = np.linalg.eigh(M)
    avg = eigvecs[:, -1]  # eigenvector with largest eigenvalue
    if avg[0] < 0:
        avg *= -1
    avg /= np.linalg.norm(avg)
    return avg.tolist()


# ---------------------------------------------------------------------------
# Pose grouping and averaging
# ---------------------------------------------------------------------------


def group_poses_by_camera(
    images: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Group registered images by their physical camera name.

    The camera name is extracted from the image file name prefix
    (e.g. ``cam_00`` from ``cam_00_frame_00010.png``).

    Parameters
    ----------
    images : list[dict]
        Parsed image entries from ``images.txt``.

    Returns
    -------
    dict[str, list[dict]]
        Mapping from camera name to its registered image entries.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for img in images:
        # Expected: cam_XX_frame_NNNNN.png
        cam_name = "_".join(img["name"].split("_")[:2])
        groups.setdefault(cam_name, []).append(img)
    return groups


def average_camera_pose(
    entries: list[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    """Average multiple COLMAP poses for the same physical camera.

    Parameters
    ----------
    entries : list[dict]
        Image entries from the same camera, each with ``qvec`` and ``tvec``.

    Returns
    -------
    avg_qvec : list[float]
        Averaged quaternion ``[w, x, y, z]``.
    avg_tvec : list[float]
        Averaged translation ``[tx, ty, tz]``.
    """
    quats = [e["qvec"] for e in entries]
    tvecs = np.array([e["tvec"] for e in entries])
    avg_q = average_quaternions(quats)
    avg_t = tvecs.mean(axis=0).tolist()
    return avg_q, avg_t


# ---------------------------------------------------------------------------
# Scene normalisation
# ---------------------------------------------------------------------------


def normalise_scene(
    cam_poses: dict[str, dict[str, Any]],
    points: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], float, list[float]]:
    """Rescale + recenter the scene so median camera distance from centroid = 1.

    Parameters
    ----------
    cam_poses : dict[str, dict]
        Per-camera data including ``camera_to_world`` (4×4 list).
    points : list[dict]
        Sparse 3-D points with ``xyz``.

    Returns
    -------
    cam_poses : dict
        Updated poses.
    points : list
        Updated point positions.
    scale : float
        The applied scale factor (original → normalised).
    translation : list[float]
        The applied translation (subtracted before scaling).
    """
    # Camera positions in world space (last column of c2w).
    positions = np.array(
        [cam_poses[c]["camera_to_world"][i][3] for c in cam_poses for i in range(3)]
    ).reshape(-1, 3)

    centroid = positions.mean(axis=0)
    dists = np.linalg.norm(positions - centroid, axis=1)
    median_dist = float(np.median(dists))
    scale = 1.0 / median_dist if median_dist > 1e-9 else 1.0

    translation = centroid.tolist()

    # Apply to cameras.
    for cam in cam_poses.values():
        c2w = np.array(cam["camera_to_world"])
        c2w[:3, 3] = (c2w[:3, 3] - centroid) * scale
        cam["camera_to_world"] = c2w.tolist()
        w2c = np.linalg.inv(c2w)
        cam["world_to_camera"] = w2c.tolist()
        cam["extrinsic"]["translation"] = c2w[:3, 3].tolist()  # updated

    # Apply to points.
    for pt in points:
        xyz = np.array(pt["xyz"])
        pt["xyz"] = ((xyz - centroid) * scale).tolist()

    return cam_poses, points, scale, translation


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


def build_camera_entry(
    cam_name: str,
    avg_qvec: list[float],
    avg_tvec: list[float],
    intrinsic: dict[str, Any],
) -> dict[str, Any]:
    """Build a single camera JSON entry.

    Parameters
    ----------
    cam_name : str
        Camera identifier (e.g. ``"cam_00"``).
    avg_qvec : list[float]
        Averaged quaternion ``[w, x, y, z]`` (world-to-camera rotation).
    avg_tvec : list[float]
        Averaged translation ``[tx, ty, tz]`` (world-to-camera).
    intrinsic : dict
        From ``parse_cameras_txt`` — has ``model``, ``width``, ``height``,
        ``params``.

    Returns
    -------
    dict
        Full camera entry ready for JSON export.
    """
    R = qvec_to_rotmat(avg_qvec)
    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = avg_tvec
    c2w = np.linalg.inv(w2c)

    params = intrinsic["params"]
    # OPENCV model: fx, fy, cx, cy, k1, k2, p1, p2
    fx = params[0] if len(params) > 0 else 0.0
    fy = params[1] if len(params) > 1 else 0.0
    cx = params[2] if len(params) > 2 else 0.0
    cy = params[3] if len(params) > 3 else 0.0
    k1 = params[4] if len(params) > 4 else 0.0
    k2 = params[5] if len(params) > 5 else 0.0
    p1 = params[6] if len(params) > 6 else 0.0
    p2 = params[7] if len(params) > 7 else 0.0

    return {
        "camera_id": cam_name,
        "image_width": intrinsic["width"],
        "image_height": intrinsic["height"],
        "intrinsic": {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "k1": k1,
            "k2": k2,
            "p1": p1,
            "p2": p2,
        },
        "extrinsic": {
            "rotation_matrix": R.tolist(),
            "quaternion_wxyz": avg_qvec,
            "translation": avg_tvec,
        },
        "camera_to_world": c2w.tolist(),
        "world_to_camera": w2c.tolist(),
    }


def export_ply(points: list[dict[str, Any]], path: Path) -> None:
    """Write sparse 3-D points to a binary PLY file.

    Parameters
    ----------
    points : list[dict]
        Each entry has ``xyz`` (3-list) and ``rgb`` (3-list of ints).
    path : Path
        Output PLY file path.
    """
    if not points:
        logger.warning("No 3-D points to export.")
        return

    dtype = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ]
    data = np.array(
        [
            (p["xyz"][0], p["xyz"][1], p["xyz"][2], p["rgb"][0], p["rgb"][1], p["rgb"][2])
            for p in points
        ],
        dtype=dtype,
    )
    el = PlyElement.describe(data, "vertex")
    PlyData([el], text=False).write(str(path))
    logger.info("Sparse point cloud written to %s (%d points)", path, len(points))


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------


def plot_camera_layout(
    cam_poses: dict[str, dict[str, Any]],
    points: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Save a 3-D scatter plot of cameras and sparse points.

    Parameters
    ----------
    cam_poses : dict[str, dict]
        Per-camera entries with ``camera_to_world``.
    points : list[dict]
        Sparse points with ``xyz`` and ``rgb``.
    output_path : Path
        Destination PNG.
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Points (subsample if many).
    if points:
        pts = np.array([p["xyz"] for p in points])
        colors = np.array([p["rgb"] for p in points]) / 255.0
        stride = max(1, len(pts) // 5000)
        ax.scatter(
            pts[::stride, 0],
            pts[::stride, 1],
            pts[::stride, 2],
            c=colors[::stride],
            s=0.5,
            alpha=0.3,
        )

    # Cameras as arrows.
    cam_colors = plt.cm.tab10(np.linspace(0, 1, max(len(cam_poses), 1)))
    for idx, (name, cam) in enumerate(sorted(cam_poses.items())):
        c2w = np.array(cam["camera_to_world"])
        pos = c2w[:3, 3]
        # Viewing direction is the negative Z axis of the camera.
        direction = -c2w[:3, 2] * 0.3
        ax.quiver(
            pos[0],
            pos[1],
            pos[2],
            direction[0],
            direction[1],
            direction[2],
            color=cam_colors[idx],
            arrow_length_ratio=0.3,
            linewidth=2,
        )
        ax.text(pos[0], pos[1], pos[2], f" {name}", fontsize=7)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Camera layout & sparse points")
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150)
    plt.close(fig)
    logger.info("Camera layout plot saved to %s", output_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    colmap_bin: str,
    quality: str,
    frames_per_cam: int,
    use_gpu: bool,
) -> dict[str, Any]:
    """Run the full COLMAP reconstruction and export pipeline.

    Parameters
    ----------
    input_dir : Path
        Directory with ``cam_XX/`` subdirectories of stabilised frames.
    output_dir : Path
        Destination for all outputs.
    colmap_bin : str
        Name or path of the COLMAP binary.
    quality : str
        ``"low"``, ``"medium"``, or ``"high"``.
    frames_per_cam : int
        Number of representative frames to select per camera.
    use_gpu : bool
        Enable GPU for SIFT extraction/matching.

    Returns
    -------
    dict
        Summary manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    colmap_path = find_colmap(colmap_bin)
    ver = colmap_version(colmap_path)
    if ver:
        logger.info("COLMAP version: %s", ver)
        major_minor = tuple(int(x) for x in ver.split(".")[:2])
        if major_minor < (3, 6):
            logger.warning("COLMAP %s is old — 3.6+ recommended.", ver)
    else:
        logger.warning("Could not determine COLMAP version.")

    # Discover cameras.
    cam_dirs = sorted(d for d in input_dir.iterdir() if d.is_dir() and d.name.startswith("cam_"))
    if not cam_dirs:
        logger.error("No cam_XX directories in %s", input_dir)
        sys.exit(1)

    logger.info("Found %d camera(s): %s", len(cam_dirs), [d.name for d in cam_dirs])

    # Work inside a temporary directory so failed runs don't leave debris.
    workspace = output_dir / "_colmap_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    image_dir = workspace / "images"

    # Step 1: select frames.
    logger.info("Selecting %d frames per camera...", frames_per_cam)
    selection = select_frames(cam_dirs, frames_per_cam, image_dir)
    total_images = sum(len(v) for v in selection.values())
    logger.info("Total images for COLMAP: %d", total_images)

    # Step 2: run COLMAP.
    txt_dir = run_colmap_pipeline(colmap_path, image_dir, workspace, quality, use_gpu)
    if txt_dir is None:
        logger.error("COLMAP reconstruction failed — no model produced.")
        sys.exit(1)

    # Step 3: parse outputs.
    cameras_raw = parse_cameras_txt(txt_dir / "cameras.txt")
    images_raw = parse_images_txt(txt_dir / "images.txt")
    points_raw = parse_points3d_txt(txt_dir / "points3D.txt")

    logger.info(
        "Parsed: %d camera models, %d registered images, %d 3-D points",
        len(cameras_raw),
        len(images_raw),
        len(points_raw),
    )

    # Report registration success.
    registered_names = {img["name"] for img in images_raw}
    all_names = {name for names in selection.values() for name in names}
    unregistered = all_names - registered_names
    if unregistered:
        logger.warning(
            "%d/%d images failed to register: %s",
            len(unregistered),
            len(all_names),
            sorted(unregistered),
        )

    # Group by physical camera and average poses.
    groups = group_poses_by_camera(images_raw)
    cam_poses: dict[str, dict[str, Any]] = {}

    for cam_name in sorted(groups):
        entries = groups[cam_name]
        avg_q, avg_t = average_camera_pose(entries)
        # Resolve intrinsic: use the camera_id from the first entry.
        cam_id = entries[0]["camera_id"]
        intrinsic = cameras_raw.get(
            cam_id,
            {
                "model": "OPENCV",
                "width": 0,
                "height": 0,
                "params": [],
            },
        )
        cam_poses[cam_name] = build_camera_entry(cam_name, avg_q, avg_t, intrinsic)
        logger.info(
            "  %s: averaged %d poses (intrinsic cam_id=%d)",
            cam_name,
            len(entries),
            cam_id,
        )

    # Check which input cameras were recovered.
    expected_cams = {d.name for d in cam_dirs}
    recovered_cams = set(cam_poses.keys())
    missing_cams = expected_cams - recovered_cams
    if missing_cams:
        logger.warning("Cameras with no recovered pose: %s", sorted(missing_cams))

    # Step 4: normalise.
    cam_poses, points_raw, scale, translation = normalise_scene(cam_poses, points_raw)
    logger.info("Scene normalised: scale=%.6f, translation=%s", scale, translation)

    # Step 5: export.
    cameras_json = {
        "normalization": {
            "scale": scale,
            "translation": translation,
        },
        "cameras": {name: pose for name, pose in sorted(cam_poses.items())},
    }
    cameras_path = output_dir / "cameras.json"
    with open(cameras_path, "w") as f:
        json.dump(cameras_json, f, indent=2)
    logger.info("Camera parameters written to %s", cameras_path)

    export_ply(points_raw, output_dir / "points3d.ply")

    plot_camera_layout(cam_poses, points_raw, output_dir / "camera_layout.png")

    # Clean up workspace.
    shutil.rmtree(workspace)
    logger.info("Cleaned up COLMAP workspace.")

    return {
        "num_cameras_input": len(cam_dirs),
        "num_cameras_recovered": len(cam_poses),
        "cameras_recovered": sorted(cam_poses.keys()),
        "cameras_missing": sorted(missing_cams),
        "num_images_submitted": total_images,
        "num_images_registered": len(images_raw),
        "num_points3d": len(points_raw),
        "normalization_scale": scale,
        "normalization_translation": translation,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for ``run_colmap``."""
    parser = argparse.ArgumentParser(
        description="Run COLMAP to estimate camera poses from multi-view frame sequences.",
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
        help="Directory to write cameras.json, points3d.ply, etc.",
    )
    parser.add_argument(
        "--colmap",
        type=str,
        default="colmap",
        help="Path to the COLMAP binary (default: 'colmap').",
    )
    parser.add_argument(
        "--quality",
        choices=["low", "medium", "high"],
        default="medium",
        help="Reconstruction quality preset (default: medium).",
    )
    parser.add_argument(
        "--frames_per_cam",
        type=int,
        default=5,
        help="Number of representative frames per camera (default: 5).",
    )
    parser.add_argument(
        "--no_gpu",
        action="store_true",
        help="Disable GPU acceleration for SIFT extraction/matching.",
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

    summary = run_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        colmap_bin=args.colmap,
        quality=args.quality,
        frames_per_cam=args.frames_per_cam,
        use_gpu=not args.no_gpu,
    )

    logger.info("Summary: %s", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
