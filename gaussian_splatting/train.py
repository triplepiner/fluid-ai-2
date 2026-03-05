#!/usr/bin/env python3
"""Training loop for dynamic 3D Gaussian splatting.

Handles the full lifecycle: point-cloud initialisation (COLMAP sparse +
depth-unprojected), per-parameter optimiser setup with learning-rate
scheduling, adaptive Gaussian densification (clone / split / prune / reset),
periodic evaluation (PSNR, SSIM), checkpointing with resume support, and
optional wandb logging.

Usage::

    python -m gaussian_splatting.train --config configs/default.yaml
    python -m gaussian_splatting.train --config configs/default.yaml --resume outputs/ckpt_10000.pt
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import sys
import time as time_mod
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from gaussian_splatting.dataset import FluidDataset, load_cameras
from gaussian_splatting.losses import LossConfig, total_loss
from gaussian_splatting.model import (
    DynamicGaussianModel,
    GaussianModel,
    inverse_sigmoid,
)
from gaussian_splatting.renderer import GaussianRenderer

# ---------------------------------------------------------------------------
# Point-cloud loading
# ---------------------------------------------------------------------------


def _load_ply_points(ply_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(N, 3)`` xyz and ``(N, 3)`` rgb from a binary-little-endian PLY.

    Falls back to ASCII parsing when the file uses ``format ascii``.

    Returns
    -------
    xyz : (N, 3) float32
    rgb : (N, 3) float32 in [0, 1]
    """
    with open(ply_path, "rb") as f:
        header = b""
        while True:
            line = f.readline()
            header += line
            if b"end_header" in line:
                break

    header_str = header.decode("ascii", errors="replace")
    lines = header_str.strip().split("\n")

    n_vertices = 0
    is_binary_le = False
    prop_names: list[str] = []
    for line in lines:
        if line.startswith("element vertex"):
            n_vertices = int(line.split()[-1])
        elif line.startswith("format binary_little_endian"):
            is_binary_le = True
        elif line.startswith("property"):
            parts = line.split()
            prop_names.append(parts[-1])

    # Determine column indices for x, y, z, red, green, blue
    x_i = prop_names.index("x") if "x" in prop_names else 0
    y_i = prop_names.index("y") if "y" in prop_names else 1
    z_i = prop_names.index("z") if "z" in prop_names else 2
    has_rgb = "red" in prop_names and "green" in prop_names and "blue" in prop_names

    if is_binary_le:
        header_len = len(header)
        with open(ply_path, "rb") as f:
            f.seek(header_len)
            # Assume all properties are float32 (COLMAP binary PLY)
            n_props = len(prop_names)
            data = np.frombuffer(f.read(n_vertices * n_props * 4), dtype=np.float32).reshape(
                n_vertices, n_props
            )
    else:
        # ASCII
        with open(ply_path, "r") as f:
            for line in f:
                if "end_header" in line:
                    break
            rows = []
            for _ in range(n_vertices):
                vals = f.readline().split()
                rows.append([float(v) for v in vals])
            data = np.array(rows, dtype=np.float32)

    xyz = data[:, [x_i, y_i, z_i]].astype(np.float32)

    if has_rgb:
        r_i = prop_names.index("red")
        g_i = prop_names.index("green")
        b_i = prop_names.index("blue")
        rgb = data[:, [r_i, g_i, b_i]].astype(np.float32)
        if rgb.max() > 1.5:
            rgb = rgb / 255.0
    else:
        rgb = np.ones_like(xyz) * 0.5

    return xyz, np.clip(rgb, 0.0, 1.0)


def _unproject_depth(
    depth: np.ndarray,
    K: np.ndarray,
    c2w: np.ndarray,
    image: np.ndarray | None = None,
    subsample: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a depth map to a coloured point cloud.

    Parameters
    ----------
    depth : (H, W) float32
    K : (3, 3)
    c2w : (4, 4)
    image : (H, W, 3) float32 in [0, 1], optional
    subsample : int
        Take every Nth pixel in each dimension.

    Returns
    -------
    xyz : (M, 3) float32  world-space points
    rgb : (M, 3) float32
    """
    H, W = depth.shape[:2]
    ys = np.arange(0, H, subsample)
    xs = np.arange(0, W, subsample)
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    yy = yy.flatten()
    xx = xx.flatten()
    zz = depth[yy, xx]

    valid = zz > 1e-6
    yy, xx, zz = yy[valid], xx[valid], zz[valid]

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    x_cam = (xx - cx) / fx * zz
    y_cam = (yy - cy) / fy * zz
    pts_cam = np.stack([x_cam, y_cam, zz, np.ones_like(zz)], axis=-1)  # (M, 4)
    pts_world = (c2w @ pts_cam.T).T[:, :3]  # (M, 3)

    if image is not None:
        rgb = image[yy, xx]
    else:
        rgb = np.ones((pts_world.shape[0], 3), dtype=np.float32) * 0.5

    return pts_world.astype(np.float32), rgb.astype(np.float32)


def initialize_point_cloud(
    cfg: DictConfig,
    data_root: Path,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build initial point cloud from COLMAP sparse points + depth unprojection.

    Returns
    -------
    xyz : (N, 3) float32 on *device*
    rgb : (N, 3) float32 on *device*
    """
    import cv2

    num_target = cfg.gaussian.num_gaussians
    all_xyz: list[np.ndarray] = []
    all_rgb: list[np.ndarray] = []

    # 1. Load sparse COLMAP points
    ply_candidates = [
        data_root / "colmap" / "sparse" / "0" / "points3D.ply",
        data_root / "colmap" / "points3d.ply",
        data_root / "colmap" / "points3D.ply",
    ]
    for ply_path in ply_candidates:
        if ply_path.exists():
            xyz_sparse, rgb_sparse = _load_ply_points(ply_path)
            all_xyz.append(xyz_sparse)
            all_rgb.append(rgb_sparse)
            print(f"  Loaded {len(xyz_sparse)} sparse points from {ply_path}")
            break
    else:
        print("  WARNING: No sparse COLMAP PLY found — using depth-only init")

    # 2. Unproject depth maps from the first frame of each camera
    cameras_json = data_root / "colmap" / "cameras_normalized.json"
    if cameras_json.exists():
        cameras = load_cameras(cameras_json)
        depth_dir = data_root / "depth"
        frames_dir = data_root / "undistorted"

        for cam in cameras:
            depth_path = depth_dir / cam["name"] / "depth_00000.npy"
            if not depth_path.exists():
                continue

            depth = np.load(str(depth_path))
            # Load corresponding image for colour
            frame_dir = frames_dir / cam["name"]
            frame_files = (
                sorted(
                    p
                    for p in frame_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                )
                if frame_dir.exists()
                else []
            )

            img = None
            if frame_files:
                img_bgr = cv2.imread(str(frame_files[0]), cv2.IMREAD_COLOR)
                if img_bgr is not None:
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                    # Resize depth/image if needed
                    if img.shape[:2] != depth.shape[:2]:
                        depth = cv2.resize(
                            depth, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
                        )

            xyz_d, rgb_d = _unproject_depth(
                depth,
                cam["K"],
                cam["c2w"],
                image=img,
                subsample=8,
            )
            all_xyz.append(xyz_d)
            all_rgb.append(rgb_d)
            print(f"  Unprojected {len(xyz_d)} points from {cam['name']} depth")

    if len(all_xyz) == 0:
        print("  WARNING: No point cloud sources found — using random init")
        xyz = torch.randn(num_target, 3, device=device) * 0.5
        rgb = torch.rand(num_target, 3, device=device)
        return xyz, rgb

    xyz = np.concatenate(all_xyz, axis=0)
    rgb = np.concatenate(all_rgb, axis=0)

    # Subsample to target count
    if len(xyz) > num_target:
        indices = np.random.choice(len(xyz), num_target, replace=False)
        xyz = xyz[indices]
        rgb = rgb[indices]
    elif len(xyz) < num_target:
        # Duplicate with small noise to reach target
        deficit = num_target - len(xyz)
        dup_idx = np.random.choice(len(xyz), deficit, replace=True)
        noise = np.random.randn(deficit, 3).astype(np.float32) * 0.001
        xyz = np.concatenate([xyz, xyz[dup_idx] + noise], axis=0)
        rgb = np.concatenate([rgb, rgb[dup_idx]], axis=0)

    print(f"  Final point cloud: {len(xyz)} points")
    return torch.from_numpy(xyz).to(device), torch.from_numpy(rgb).to(device)


# ---------------------------------------------------------------------------
# Optimizer helpers
# ---------------------------------------------------------------------------


def _build_gaussian_optimizer(
    model: DynamicGaussianModel,
    cfg: DictConfig,
) -> torch.optim.Adam:
    """Create Adam optimizer with separate per-parameter learning rates."""
    gm = model.gaussian_model
    lr = cfg.gaussian.learning_rate

    param_groups = [
        {"params": [gm._xyz], "lr": lr.position, "name": "xyz"},
        {"params": [gm._features_dc], "lr": lr.features, "name": "features_dc"},
        {"params": [gm._features_rest], "lr": lr.features / 20.0, "name": "features_rest"},
        {"params": [gm._opacity], "lr": lr.opacity, "name": "opacity"},
        {"params": [gm._scaling], "lr": lr.scaling, "name": "scaling"},
        {"params": [gm._rotation], "lr": lr.rotation, "name": "rotation"},
    ]
    return torch.optim.Adam(param_groups, lr=0.0, eps=1e-15)


def _build_deformation_optimizer(
    model: DynamicGaussianModel,
    cfg: DictConfig,
) -> torch.optim.Adam:
    """Create optimizer for the deformation network."""
    return torch.optim.Adam(
        model.deformation_network.parameters(),
        lr=cfg.gaussian.deformation.learning_rate,
        weight_decay=1e-6,
    )


def _update_position_lr(
    optimizer: torch.optim.Adam,
    iteration: int,
    max_iterations: int,
    lr_init: float,
    lr_final: float,
) -> float:
    """Exponential decay for position learning rate."""
    if max_iterations <= 1:
        return lr_init
    t = min(iteration / max_iterations, 1.0)
    # log-linear interpolation
    lr = math.exp(math.log(lr_init) * (1.0 - t) + math.log(lr_final) * t)
    for group in optimizer.param_groups:
        if group["name"] == "xyz":
            group["lr"] = lr
    return lr


# ---------------------------------------------------------------------------
# Adaptive densification
# ---------------------------------------------------------------------------


class DensificationStats:
    """Tracks per-Gaussian gradient magnitudes for densification decisions."""

    def __init__(self, num_points: int, device: torch.device):
        self.grad_accum = torch.zeros(num_points, device=device)
        self.count = torch.zeros(num_points, device=device)

    def update(self, visibility_filter: torch.Tensor, xyz_grad: torch.Tensor):
        """Accumulate gradient norms for visible Gaussians."""
        grad_norm = xyz_grad.detach().norm(dim=-1)  # (N,)
        self.grad_accum[visibility_filter] += grad_norm[visibility_filter]
        self.count[visibility_filter] += 1

    def reset(self):
        self.grad_accum.zero_()
        self.count.zero_()

    def resize(self, new_size: int, device: torch.device):
        """Expand or shrink tracking tensors."""
        old_size = len(self.grad_accum)
        if new_size == old_size:
            return
        new_grad = torch.zeros(new_size, device=device)
        new_count = torch.zeros(new_size, device=device)
        keep = min(old_size, new_size)
        new_grad[:keep] = self.grad_accum[:keep]
        new_count[:keep] = self.count[:keep]
        self.grad_accum = new_grad
        self.count = new_count


def _replace_optimizer_params(
    old_optimizer: torch.optim.Adam,
    new_params_dict: dict[str, nn.Parameter],
) -> torch.optim.Adam:
    """Create a new optimizer copying momentum state for existing entries.

    Parameters
    ----------
    old_optimizer : the current Adam optimizer
    new_params_dict : maps group "name" -> new Parameter tensor

    Returns
    -------
    A new Adam optimizer with the new parameters and carried-over state
    for the first ``old_size`` entries of each parameter.
    """
    new_groups = []
    for group in old_optimizer.param_groups:
        name = group["name"]
        new_p = new_params_dict[name]
        old_p = group["params"][0]
        new_group = {k: v for k, v in group.items() if k != "params"}
        new_group["params"] = [new_p]
        new_groups.append(new_group)

        # Copy optimizer state (exp_avg, exp_avg_sq)
        if old_p in old_optimizer.state:
            old_state = old_optimizer.state[old_p]
            new_state = {}
            for key, val in old_state.items():
                if (
                    isinstance(val, torch.Tensor)
                    and val.dim() > 0
                    and val.shape[0] == old_p.shape[0]
                ):
                    # Extend with zeros for new Gaussians
                    new_shape = list(val.shape)
                    new_shape[0] = new_p.shape[0]
                    extended = torch.zeros(new_shape, device=val.device, dtype=val.dtype)
                    extended[: old_p.shape[0]] = val
                    new_state[key] = extended
                else:
                    new_state[key] = val
            old_optimizer.state[new_p] = new_state

    new_opt = torch.optim.Adam(new_groups, lr=0.0, eps=1e-15)
    # Transfer state
    for group in new_opt.param_groups:
        p = group["params"][0]
        if p in old_optimizer.state:
            new_opt.state[p] = old_optimizer.state[p]
    return new_opt


def _densify_and_prune(
    model: DynamicGaussianModel,
    optimizer: torch.optim.Adam,
    stats: DensificationStats,
    grad_threshold: float,
    max_gaussians: int,
    iteration: int,
    device: torch.device,
) -> tuple[torch.optim.Adam, DensificationStats]:
    """Perform one round of adaptive Gaussian densification.

    1. Clone small Gaussians with high gradient.
    2. Split large Gaussians with high gradient.
    3. Prune low-opacity and oversized Gaussians.
    4. Every 3000 iterations, reset opacity.

    Returns updated optimizer and stats.
    """
    gm = model.gaussian_model
    N = gm.num_points

    avg_grad = stats.grad_accum / stats.count.clamp(min=1)
    high_grad = avg_grad > grad_threshold

    scaling = gm.get_scaling()  # (N, 3)
    max_scale = scaling.max(dim=-1).values  # (N,)
    scale_threshold = 0.01  # scene-unit threshold for small vs large

    # --- Clone: high gradient + small scale ---
    clone_mask = high_grad & (max_scale < scale_threshold)
    n_clone = clone_mask.sum().item()

    # --- Split: high gradient + large scale ---
    split_mask = high_grad & (max_scale >= scale_threshold)
    n_split = split_mask.sum().item()

    # Limit total to avoid explosion
    n_after = N + n_clone + n_split  # splits replace, but we add 2 and remove 1
    if n_after > max_gaussians:
        # Reduce clones/splits proportionally
        excess = n_after - max_gaussians
        if n_clone + n_split > 0:
            ratio = max(0, 1.0 - excess / (n_clone + n_split))
            if ratio < 1.0:
                clone_rand = torch.rand(N, device=device) < ratio
                split_rand = torch.rand(N, device=device) < ratio
                clone_mask = clone_mask & clone_rand
                split_mask = split_mask & split_rand
                n_clone = clone_mask.sum().item()
                n_split = split_mask.sum().item()

    # Build new parameters
    new_xyz_parts = [gm._xyz.data]
    new_rot_parts = [gm._rotation.data]
    new_scale_parts = [gm._scaling.data]
    new_opacity_parts = [gm._opacity.data]
    new_fdc_parts = [gm._features_dc.data]
    new_frest_parts = [gm._features_rest.data]

    # Clone: duplicate as-is
    if n_clone > 0:
        idx = clone_mask.nonzero(as_tuple=True)[0]
        new_xyz_parts.append(gm._xyz.data[idx])
        new_rot_parts.append(gm._rotation.data[idx])
        new_scale_parts.append(gm._scaling.data[idx])
        new_opacity_parts.append(gm._opacity.data[idx])
        new_fdc_parts.append(gm._features_dc.data[idx])
        new_frest_parts.append(gm._features_rest.data[idx])

    # Split: create 2 new Gaussians offset from original, with smaller scale
    if n_split > 0:
        idx = split_mask.nonzero(as_tuple=True)[0]
        n_s = idx.shape[0]

        # Two copies offset in random directions
        stds = gm.get_scaling()[idx]  # (n_s, 3)
        offset = torch.randn(n_s, 3, device=device) * stds
        new_xyz_parts.append(gm._xyz.data[idx] + offset)
        new_xyz_parts.append(gm._xyz.data[idx] - offset)

        # Reduce scale by factor 1.6
        log_scale_reduced = gm._scaling.data[idx] - math.log(1.6)
        new_scale_parts.append(log_scale_reduced)
        new_scale_parts.append(log_scale_reduced)

        for _ in range(2):
            new_rot_parts.append(gm._rotation.data[idx])
            new_opacity_parts.append(gm._opacity.data[idx])
            new_fdc_parts.append(gm._features_dc.data[idx])
            new_frest_parts.append(gm._features_rest.data[idx])

    # Concatenate
    all_xyz = torch.cat(new_xyz_parts, dim=0)
    all_rot = torch.cat(new_rot_parts, dim=0)
    all_scale = torch.cat(new_scale_parts, dim=0)
    all_opacity = torch.cat(new_opacity_parts, dim=0)
    all_fdc = torch.cat(new_fdc_parts, dim=0)
    all_frest = torch.cat(new_frest_parts, dim=0)

    # --- Prune: low opacity or oversized ---
    opacity_activated = torch.sigmoid(all_opacity).squeeze(-1)  # (N_new,)
    prune_mask = opacity_activated < 0.005
    # Also prune oversized
    all_scale_activated = torch.exp(all_scale)
    prune_mask = prune_mask | (all_scale_activated.max(dim=-1).values > 0.5)

    # For split originals: mark them for removal (they were replaced)
    if n_split > 0:
        prune_mask[:N][split_mask] = True

    keep = ~prune_mask
    all_xyz = all_xyz[keep]
    all_rot = all_rot[keep]
    all_scale = all_scale[keep]
    all_opacity = all_opacity[keep]
    all_fdc = all_fdc[keep]
    all_frest = all_frest[keep]

    # --- Opacity reset every 3000 iterations ---
    if iteration % 3000 == 0:
        all_opacity = torch.full_like(all_opacity, inverse_sigmoid(torch.tensor(0.01)).item())

    # --- Update model parameters ---
    gm._xyz = nn.Parameter(all_xyz)
    gm._rotation = nn.Parameter(all_rot)
    gm._scaling = nn.Parameter(all_scale)
    gm._opacity = nn.Parameter(all_opacity)
    gm._features_dc = nn.Parameter(all_fdc)
    gm._features_rest = nn.Parameter(all_frest)

    # Rebuild optimizer with new parameters
    new_params = {
        "xyz": gm._xyz,
        "features_dc": gm._features_dc,
        "features_rest": gm._features_rest,
        "opacity": gm._opacity,
        "scaling": gm._scaling,
        "rotation": gm._rotation,
    }
    optimizer = _replace_optimizer_params(optimizer, new_params)

    # Rebuild stats
    new_n = gm.num_points
    new_stats = DensificationStats(new_n, device)

    return optimizer, new_stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(
    model: DynamicGaussianModel,
    renderer: GaussianRenderer,
    dataset: FluidDataset,
    device: torch.device,
    max_samples: int = 50,
) -> dict[str, float]:
    """Evaluate on the validation set, computing PSNR and L1.

    Returns a dict with mean metrics.
    """
    model.eval()
    psnrs: list[float] = []
    l1s: list[float] = []

    n_eval = min(len(dataset), max_samples)
    indices = np.linspace(0, len(dataset) - 1, n_eval, dtype=int)

    for idx in indices:
        sample = dataset[int(idx)]

        gt_image = sample["image"].to(device)  # (3, H, W)
        K = sample["K"].to(device)
        w2c = sample["w2c"].to(device)
        t_norm = sample["time_normalized"]
        H, W = gt_image.shape[1], gt_image.shape[2]

        camera = {
            "K": K,
            "w2c": w2c,
            "image_height": H,
            "image_width": W,
        }

        gaussians = model.forward(time=t_norm)
        rendered = renderer.render(gaussians, camera)
        pred = rendered["render"]  # (3, H, W)

        # PSNR
        mse = ((pred - gt_image) ** 2).mean().item()
        psnr = -10.0 * math.log10(max(mse, 1e-10))
        psnrs.append(psnr)

        # L1
        l1 = (pred - gt_image).abs().mean().item()
        l1s.append(l1)

    model.train()
    return {
        "psnr": float(np.mean(psnrs)),
        "l1": float(np.mean(l1s)),
        "n_eval": n_eval,
    }


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: DynamicGaussianModel,
    gauss_opt: torch.optim.Adam,
    deform_opt: torch.optim.Adam,
    iteration: int,
    best_psnr: float,
    stats: DensificationStats,
):
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "gauss_optimizer_state_dict": gauss_opt.state_dict(),
            "deform_optimizer_state_dict": deform_opt.state_dict(),
            "best_psnr": best_psnr,
            "num_gaussians": model.gaussian_model.num_points,
            "grad_accum": stats.grad_accum,
            "grad_count": stats.count,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: DynamicGaussianModel,
    gauss_opt: torch.optim.Adam,
    deform_opt: torch.optim.Adam,
    device: torch.device,
) -> tuple[int, float, DensificationStats]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    gauss_opt.load_state_dict(ckpt["gauss_optimizer_state_dict"])
    deform_opt.load_state_dict(ckpt["deform_optimizer_state_dict"])
    iteration = ckpt["iteration"]
    best_psnr = ckpt.get("best_psnr", 0.0)
    n = model.gaussian_model.num_points
    stats = DensificationStats(n, device)
    if "grad_accum" in ckpt:
        ga = ckpt["grad_accum"]
        gc = ckpt["grad_count"]
        stats.grad_accum[: len(ga)] = ga[:n]
        stats.count[: len(gc)] = gc[:n]
    return iteration, best_psnr, stats


# ---------------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------------


def train(cfg: DictConfig, resume_path: str | None = None, device_str: str = "auto"):
    # --- Device ---
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # --- Seed ---
    seed = cfg.training.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    # --- Paths ---
    data_root = Path(cfg.data.root_dir)
    output_dir = Path(cfg.data.output_dir) / cfg.data.scene_name
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # --- Dataset ---
    res_h = cfg.data.resolution.height
    res_w = cfg.data.resolution.width
    time_range = None
    if cfg.data.get("start_frame", 0) > 0 or cfg.data.get("end_frame", -1) != -1:
        end = cfg.data.end_frame if cfg.data.end_frame != -1 else None
        if end is not None:
            time_range = (cfg.data.start_frame, end)

    train_dataset = FluidDataset(
        data_root=data_root,
        split="train",
        resolution=(res_h, res_w),
        load_flow=True,
        load_depth=True,
        time_range=time_range,
        augment=True,
    )
    val_dataset = FluidDataset(
        data_root=data_root,
        split="val",
        resolution=(res_h, res_w),
        load_flow=False,
        load_depth=True,
        time_range=time_range,
        augment=False,
    )
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # --- Model ---
    model = DynamicGaussianModel(
        sh_degree=cfg.gaussian.sh_degree,
        num_points=cfg.gaussian.num_gaussians,
    )

    # Initialize point cloud
    print("Initializing point cloud...")
    xyz, rgb = initialize_point_cloud(cfg, data_root, device)
    model.gaussian_model.initialize_from_point_cloud(xyz, rgb)
    model = model.to(device)
    model.train()

    # --- Renderer ---
    renderer = GaussianRenderer(
        image_height=res_h,
        image_width=res_w,
        background_color=(0.0, 0.0, 0.0),
        use_cuda_rasterizer="auto",
    ).to(device)

    # --- Optimizers ---
    gauss_opt = _build_gaussian_optimizer(model, cfg)
    deform_opt = _build_deformation_optimizer(model, cfg)

    # --- Training config ---
    max_iter = cfg.training.gaussian.epochs
    save_interval = cfg.training.gaussian.save_interval
    eval_interval = cfg.training.gaussian.eval_interval
    log_interval = cfg.training.gaussian.get("log_interval", 100)
    grad_clip = cfg.training.get("gradient_clip", 1.0)
    use_amp = cfg.training.get("mixed_precision", False) and device.type == "cuda"

    lr_init = cfg.gaussian.learning_rate.position
    lr_final = lr_init * cfg.gaussian.get("position_lr_decay_mult", 0.01)

    densify_cfg = cfg.gaussian.densify
    densify_start = densify_cfg.start_iter
    densify_stop = densify_cfg.stop_iter
    densify_interval = densify_cfg.interval
    densify_grad_thresh = densify_cfg.grad_threshold
    max_gaussians = densify_cfg.max_gaussians

    # Loss config
    loss_w = cfg.gaussian.get("loss_weights", {})
    loss_cfg = LossConfig(
        lambda_photo=loss_w.get("photo", 1.0),
        lambda_l1=1.0 - loss_w.get("ssim", 0.2),
        lambda_ssim=loss_w.get("ssim", 0.2),
        lambda_depth=loss_w.get("depth", 0.1),
        lambda_temporal=loss_w.get("flow", 0.05),
        lambda_smooth=loss_w.get("temporal_smooth", 0.01),
        lambda_opacity_reg=0.01,
        lambda_scale_reg=0.01,
        max_scale=0.1,
    )

    # --- Resume ---
    start_iter = 0
    best_psnr = 0.0
    stats = DensificationStats(model.gaussian_model.num_points, device)

    if resume_path is not None:
        print(f"Resuming from {resume_path}")
        start_iter, best_psnr, stats = load_checkpoint(
            Path(resume_path),
            model,
            gauss_opt,
            deform_opt,
            device,
        )
        print(f"  Resumed at iteration {start_iter}, best PSNR={best_psnr:.2f}")

    # --- Wandb ---
    use_wandb = cfg.logging.get("use_wandb", False)
    if use_wandb:
        import wandb

        wandb.init(
            project=cfg.logging.get("wandb_project", "fluid-recon"),
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow",
        )

    # --- AMP scaler ---
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    # --- Training loop ---
    n_train = len(train_dataset)
    pbar = tqdm(range(start_iter, max_iter), desc="Training", dynamic_ncols=True)

    for iteration in pbar:
        # Sample a random (camera, timestep)
        idx = np.random.randint(0, n_train)
        sample = train_dataset[idx]

        gt_image = sample["image"].to(device)  # (3, H, W)
        K = sample["K"].to(device)
        w2c = sample["w2c"].to(device)
        t_norm = sample["time_normalized"]
        H_img, W_img = gt_image.shape[1], gt_image.shape[2]

        camera = {
            "K": K,
            "w2c": w2c,
            "image_height": H_img,
            "image_width": W_img,
        }

        # ---- Forward ----
        with torch.amp.autocast("cuda", enabled=use_amp):
            gaussians = model.forward(time=t_norm)
            rendered = renderer.render(gaussians, camera)

            # Build loss kwargs
            loss_kwargs: dict[str, Any] = {
                "rendered": rendered["render"],
                "target": gt_image,
                "opacity": gaussians["opacity"],
                "scaling": gaussians["scaling"],
                "config": loss_cfg,
            }

            # Depth
            if sample["depth"] is not None:
                loss_kwargs["rendered_depth"] = rendered["depth"]
                loss_kwargs["pseudo_depth"] = sample["depth"].to(device)

            # Temporal consistency (flow warp)
            if sample["flow_fwd"] is not None:
                loss_kwargs["rendered_t"] = rendered["render"]
                loss_kwargs["flow_fwd"] = sample["flow_fwd"].to(device)
                # Load next frame's image as target_t1
                cam_idx, timestep = train_dataset._indices[idx]
                if timestep + 1 < train_dataset._num_total_frames:
                    next_sample = train_dataset._load_raw(cam_idx, timestep + 1)
                    if next_sample["image"] is not None:
                        next_img = (
                            torch.from_numpy(next_sample["image"]).permute(2, 0, 1).to(device)
                        )
                        loss_kwargs["target_t1"] = next_img

            # Temporal smoothness (need 3 consecutive timesteps)
            dt = 1.0 / max(train_dataset.total_timesteps - 1, 1)
            if t_norm - dt >= 0.0 and t_norm + dt <= 1.0:
                gaussians_prev = model.forward(time=max(0.0, t_norm - dt))
                gaussians_next = model.forward(time=min(1.0, t_norm + dt))
                loss_kwargs["xyz_t_minus_1"] = gaussians_prev["xyz"]
                loss_kwargs["xyz_t"] = gaussians["xyz"]
                loss_kwargs["xyz_t_plus_1"] = gaussians_next["xyz"]

            loss_total, loss_dict = total_loss(**loss_kwargs)

        # ---- Backward ----
        gauss_opt.zero_grad(set_to_none=True)
        deform_opt.zero_grad(set_to_none=True)

        if scaler is not None:
            scaler.scale(loss_total).backward()
            scaler.unscale_(gauss_opt)
            scaler.unscale_(deform_opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(gauss_opt)
            scaler.step(deform_opt)
            scaler.update()
        else:
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            gauss_opt.step()
            deform_opt.step()

        # ---- LR scheduling ----
        current_lr = _update_position_lr(
            gauss_opt,
            iteration,
            max_iter,
            lr_init,
            lr_final,
        )

        # ---- Densification stats ----
        if iteration >= densify_start and iteration < densify_stop:
            vis = rendered["visibility_filter"]
            if model.gaussian_model._xyz.grad is not None:
                stats.update(vis, model.gaussian_model._xyz.grad)

            # Densify every interval
            if (iteration + 1) % densify_interval == 0:
                gauss_opt, stats = _densify_and_prune(
                    model,
                    gauss_opt,
                    stats,
                    grad_threshold=densify_grad_thresh,
                    max_gaussians=max_gaussians,
                    iteration=iteration,
                    device=device,
                )

        # ---- Logging ----
        if iteration % log_interval == 0:
            n_gauss = model.gaussian_model.num_points
            loss_val = loss_total.item()
            photo_val = loss_dict.get("photometric", torch.tensor(0.0)).item()

            # Quick per-iteration PSNR from photometric MSE
            with torch.no_grad():
                mse = ((rendered["render"] - gt_image) ** 2).mean().item()
                train_psnr = -10.0 * math.log10(max(mse, 1e-10))

            pbar.set_postfix(
                loss=f"{loss_val:.4f}",
                psnr=f"{train_psnr:.1f}",
                n=f"{n_gauss // 1000}k",
                lr=f"{current_lr:.1e}",
            )

            if use_wandb:
                log_data = {
                    "train/loss": loss_val,
                    "train/psnr": train_psnr,
                    "train/num_gaussians": n_gauss,
                    "train/lr_position": current_lr,
                }
                for k, v in loss_dict.items():
                    log_data[f"train/{k}"] = v.item()
                wandb.log(log_data, step=iteration)

        # ---- Evaluation ----
        if (iteration + 1) % eval_interval == 0:
            metrics = evaluate(model, renderer, val_dataset, device)
            print(
                f"\n  [Eval @ {iteration + 1}] "
                f"PSNR={metrics['psnr']:.2f}, L1={metrics['l1']:.4f}, "
                f"N_gauss={model.gaussian_model.num_points}"
            )

            if metrics["psnr"] > best_psnr:
                best_psnr = metrics["psnr"]
                save_checkpoint(
                    ckpt_dir / "best.pt",
                    model,
                    gauss_opt,
                    deform_opt,
                    iteration + 1,
                    best_psnr,
                    stats,
                )

            if use_wandb:
                wandb.log(
                    {
                        "val/psnr": metrics["psnr"],
                        "val/l1": metrics["l1"],
                        "val/best_psnr": best_psnr,
                    },
                    step=iteration,
                )

        # ---- Checkpoint ----
        if (iteration + 1) % save_interval == 0:
            save_checkpoint(
                ckpt_dir / f"ckpt_{iteration + 1:06d}.pt",
                model,
                gauss_opt,
                deform_opt,
                iteration + 1,
                best_psnr,
                stats,
            )

    # --- Final checkpoint ---
    save_checkpoint(
        ckpt_dir / "final.pt",
        model,
        gauss_opt,
        deform_opt,
        max_iter,
        best_psnr,
        stats,
    )
    print(f"\nTraining complete. Best PSNR: {best_psnr:.2f}")
    print(f"Final Gaussian count: {model.gaussian_model.num_points}")

    if use_wandb:
        wandb.finish()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train dynamic 3D Gaussian splatting")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    train(cfg, resume_path=args.resume, device_str=args.device)


if __name__ == "__main__":
    main()
