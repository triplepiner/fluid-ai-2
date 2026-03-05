#!/usr/bin/env python3
"""Thorough integration test for gaussian_splatting/train.py.

Creates a minimal synthetic dataset, runs a short training loop, and
verifies: loss decreases, checkpointing works, resume works, evaluation
runs, and densification stats are updated.
"""

import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

checks_passed = 0
checks_total = 0


def check(cond: bool, label: str):
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if cond:
        checks_passed += 1


# =====================================================================
# Step 0: Create synthetic dataset
# =====================================================================
def create_synthetic_dataset(root: Path):
    """Create a minimal synthetic dataset with 2 cameras, 10 frames, 64x64."""
    print("\n=== Step 0: Creating synthetic dataset ===")

    H, W = 64, 64
    num_cameras = 2
    num_frames = 10
    fx, fy = 50.0, 50.0
    cx, cy = W / 2.0, H / 2.0

    # Camera poses: two cameras at z=+3 with slight angular offset
    cameras_data = {"cameras": {}}
    for cam_idx in range(num_cameras):
        cam_name = f"cam_{cam_idx:02d}"
        angle = (cam_idx - 0.5) * 0.3  # slight rotation around y

        # c2w: camera-to-world
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 0] = math.cos(angle)
        c2w[0, 2] = math.sin(angle)
        c2w[2, 0] = -math.sin(angle)
        c2w[2, 2] = math.cos(angle)
        c2w[2, 3] = 3.0  # camera at z=3

        w2c = np.linalg.inv(c2w).astype(np.float32)

        cameras_data["cameras"][cam_name] = {
            "camera_id": cam_idx,
            "image_width": W,
            "image_height": H,
            "intrinsic": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "k1": 0.0,
                "k2": 0.0,
                "p1": 0.0,
                "p2": 0.0,
            },
            "camera_to_world": c2w.tolist(),
            "world_to_camera": w2c.tolist(),
        }

        # Create frame images (simple gradient + noise varying by time)
        frames_dir = root / "undistorted" / cam_name
        frames_dir.mkdir(parents=True, exist_ok=True)

        depth_dir = root / "depth" / cam_name
        depth_dir.mkdir(parents=True, exist_ok=True)

        flow_dir = root / "flow" / cam_name
        flow_dir.mkdir(parents=True, exist_ok=True)

        for t in range(num_frames):
            # Image: gradient from red to blue with time-varying brightness
            ramp_x = np.linspace(0, 1, W, dtype=np.float32)
            ramp_y = np.linspace(0, 1, H, dtype=np.float32)
            yy, xx = np.meshgrid(ramp_y, ramp_x, indexing="ij")

            brightness = 0.3 + 0.5 * (t / max(num_frames - 1, 1))
            r = xx * brightness
            g = (1 - xx) * yy * brightness
            b = yy * brightness

            img = np.stack([r, g, b], axis=-1)
            # Add per-frame noise so temporal loss has signal
            np.random.seed(cam_idx * 1000 + t)
            img = img + np.random.randn(H, W, 3).astype(np.float32) * 0.05
            img = np.clip(img, 0, 1)

            img_bgr = (img[:, :, ::-1] * 255).astype(np.uint8)
            cv2.imwrite(str(frames_dir / f"frame_{t:05d}.png"), img_bgr)

            # Depth: synthetic depth (everything at ~3.0 ± noise)
            depth = np.full((H, W), 3.0, dtype=np.float32)
            depth += np.random.randn(H, W).astype(np.float32) * 0.1
            depth = np.clip(depth, 0.1, 10.0)
            np.save(str(depth_dir / f"depth_{t:05d}.npy"), depth)

            # Forward flow: small rightward shift (2 px/frame)
            if t < num_frames - 1:
                flow_fwd = np.zeros((2, H, W), dtype=np.float32)
                flow_fwd[0] = 2.0  # dx = 2px rightward
                np.save(str(flow_dir / f"flow_fwd_{t:05d}.npy"), flow_fwd)

                # Flow mask
                mask = np.ones((H, W), dtype=np.uint8) * 255
                cv2.imwrite(str(flow_dir / f"flow_mask_{t:05d}.png"), mask)

    # Write cameras JSON
    colmap_dir = root / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    with open(colmap_dir / "cameras_normalized.json", "w") as f:
        json.dump(cameras_data, f)

    # Write a simple ASCII PLY with 50 random points
    sparse_dir = colmap_dir / "sparse" / "0"
    sparse_dir.mkdir(parents=True, exist_ok=True)
    n_pts = 50
    np.random.seed(123)
    pts = np.random.randn(n_pts, 3).astype(np.float32) * 0.5
    colors = np.random.rand(n_pts, 3).astype(np.float32)

    with open(sparse_dir / "points3D.ply", "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {n_pts}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property float red\n")
        f.write("property float green\n")
        f.write("property float blue\n")
        f.write("end_header\n")
        for i in range(n_pts):
            f.write(
                f"{pts[i,0]} {pts[i,1]} {pts[i,2]} " f"{colors[i,0]} {colors[i,1]} {colors[i,2]}\n"
            )

    print(f"  Created synthetic dataset at {root}")
    print(f"  {num_cameras} cameras, {num_frames} frames, {H}x{W}")
    check(True, "Synthetic dataset created")
    return cameras_data


# =====================================================================
# Step 0b: Create test config
# =====================================================================
def create_test_config(data_root: Path, output_dir: Path, config_path: Path):
    print("\n=== Step 0b: Creating test config ===")

    config = {
        "data": {
            "root_dir": str(data_root),
            "output_dir": str(output_dir),
            "scene_name": "test_scene",
            "resolution": {"width": 64, "height": 64},
            "fps": 30,
            "num_cameras": 2,
            "start_frame": 0,
            "end_frame": -1,
        },
        "gaussian": {
            "num_gaussians": 200,
            "sh_degree": 2,
            "opacity_init": 0.1,
            "scale_init": 0.001,
            "learning_rate": {
                "position": 1.6e-4,
                "features": 2.5e-3,
                "opacity": 5.0e-2,
                "scaling": 5.0e-3,
                "rotation": 1.0e-3,
            },
            "position_lr_decay_mult": 0.01,
            "densify": {
                "grad_threshold": 0.0002,
                "interval": 10,
                "start_iter": 20,
                "stop_iter": 80,
                "max_gaussians": 500,
            },
            "deformation": {
                "enabled": True,
                "learning_rate": 1.0e-3,
                "mlp_width": 128,
                "mlp_depth": 6,
                "fourier_features": 64,
            },
            "loss_weights": {
                "photo": 1.0,
                "depth": 0.1,
                "flow": 0.05,
                "temporal_smooth": 0.01,
                "ssim": 0.2,
            },
        },
        "training": {
            "gaussian": {
                "epochs": 100,
                "batch_size": 1,
                "save_interval": 50,
                "eval_interval": 50,
                "log_interval": 10,
            },
            "seed": 42,
            "mixed_precision": False,
            "gradient_clip": 1.0,
            "num_workers": 0,
        },
        "logging": {
            "use_wandb": False,
            "use_tensorboard": False,
        },
    }

    from omegaconf import OmegaConf

    OmegaConf.save(config, str(config_path))
    print(f"  Config saved to {config_path}")
    check(True, "Test config created")
    return config


# =====================================================================
# Step 1: Run training for 100 iterations
# =====================================================================
def step1_run_training(config_path: Path, output_dir: Path):
    print("\n=== Step 1: Run training for 100 iterations ===")

    from omegaconf import OmegaConf

    from gaussian_splatting.train import train

    cfg = OmegaConf.load(str(config_path))

    # Capture the training output — we call train() directly
    # instead of subprocess so we can catch errors
    train(cfg, resume_path=None, device_str="cpu")

    check(True, "Training completed without crash")


# =====================================================================
# Step 2: Verify training behavior (loss decrease)
# =====================================================================
def step2_verify_loss_decrease(config_path: Path, data_root: Path, output_dir: Path):
    print("\n=== Step 2: Verify training behavior ===")

    from omegaconf import OmegaConf

    from gaussian_splatting.dataset import FluidDataset
    from gaussian_splatting.losses import LossConfig, total_loss
    from gaussian_splatting.model import DynamicGaussianModel
    from gaussian_splatting.renderer import GaussianRenderer
    from gaussian_splatting.train import (
        DensificationStats,
        _build_deformation_optimizer,
        _build_gaussian_optimizer,
        _update_position_lr,
        evaluate,
        initialize_point_cloud,
    )

    cfg = OmegaConf.load(str(config_path))
    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    # Re-create model and train a controlled 50 iterations collecting losses
    model = DynamicGaussianModel(sh_degree=2, num_points=200)
    xyz, rgb = initialize_point_cloud(cfg, data_root, device)
    model.gaussian_model.initialize_from_point_cloud(xyz, rgb)
    model = model.to(device)
    model.train()

    renderer = GaussianRenderer(64, 64, use_cuda_rasterizer="pytorch").to(device)
    gauss_opt = _build_gaussian_optimizer(model, cfg)
    deform_opt = _build_deformation_optimizer(model, cfg)

    train_dataset = FluidDataset(
        data_root=data_root,
        split="train",
        resolution=(64, 64),
        load_flow=True,
        load_depth=True,
        augment=False,
    )

    loss_cfg = LossConfig(
        lambda_photo=1.0,
        lambda_l1=0.8,
        lambda_ssim=0.2,
        lambda_depth=0.1,
        lambda_temporal=0.05,
        lambda_smooth=0.01,
        lambda_opacity_reg=0.01,
        lambda_scale_reg=0.01,
    )

    losses_log = []
    psnrs_log = []

    for it in range(50):
        idx = it % len(train_dataset)
        sample = train_dataset[idx]

        gt_image = sample["image"].to(device)
        K = sample["K"].to(device)
        w2c = sample["w2c"].to(device)
        t_norm = sample["time_normalized"]
        H, W = gt_image.shape[1], gt_image.shape[2]

        camera = {"K": K, "w2c": w2c, "image_height": H, "image_width": W}

        gaussians = model.forward(time=t_norm)
        rendered = renderer.render(gaussians, camera)

        loss_val, loss_dict = total_loss(
            rendered=rendered["render"],
            target=gt_image,
            rendered_depth=rendered["depth"],
            pseudo_depth=sample["depth"].to(device) if sample["depth"] is not None else None,
            opacity=gaussians["opacity"],
            scaling=gaussians["scaling"],
            config=loss_cfg,
        )

        gauss_opt.zero_grad(set_to_none=True)
        deform_opt.zero_grad(set_to_none=True)
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gauss_opt.step()
        deform_opt.step()

        _update_position_lr(gauss_opt, it, 50, 1.6e-4, 1.6e-6)

        lv = loss_val.item()
        with torch.no_grad():
            mse = ((rendered["render"] - gt_image) ** 2).mean().item()
            psnr = -10.0 * math.log10(max(mse, 1e-10))

        losses_log.append(lv)
        psnrs_log.append(psnr)

    loss_first = np.mean(losses_log[:5])
    loss_last = np.mean(losses_log[-5:])
    psnr_first = np.mean(psnrs_log[:5])
    psnr_last = np.mean(psnrs_log[-5:])

    print(f"  Loss first 5 avg: {loss_first:.4f}")
    print(f"  Loss last 5 avg:  {loss_last:.4f}")
    print(f"  PSNR first 5 avg: {psnr_first:.2f}")
    print(f"  PSNR last 5 avg:  {psnr_last:.2f}")

    check(loss_last < loss_first, f"Loss decreased ({loss_first:.4f} -> {loss_last:.4f})")
    check(not math.isnan(loss_last), "Loss is not NaN")
    check(not math.isinf(loss_last), "Loss is not Inf")
    check(psnr_last > psnr_first, f"PSNR increased ({psnr_first:.2f} -> {psnr_last:.2f})")
    check(psnr_last > 5.0, f"PSNR > 5.0 ({psnr_last:.2f})")


# =====================================================================
# Step 3: Verify densification
# =====================================================================
def step3_verify_densification(config_path: Path, data_root: Path):
    print("\n=== Step 3: Verify densification ===")

    from omegaconf import OmegaConf

    from gaussian_splatting.dataset import FluidDataset
    from gaussian_splatting.losses import LossConfig, total_loss
    from gaussian_splatting.model import DynamicGaussianModel
    from gaussian_splatting.renderer import GaussianRenderer
    from gaussian_splatting.train import (
        DensificationStats,
        _build_deformation_optimizer,
        _build_gaussian_optimizer,
        _densify_and_prune,
        _update_position_lr,
        initialize_point_cloud,
    )

    cfg = OmegaConf.load(str(config_path))
    device = torch.device("cpu")
    torch.manual_seed(42)
    np.random.seed(42)

    model = DynamicGaussianModel(sh_degree=2, num_points=200)
    xyz, rgb = initialize_point_cloud(cfg, data_root, device)
    model.gaussian_model.initialize_from_point_cloud(xyz, rgb)
    model = model.to(device)
    model.train()

    renderer = GaussianRenderer(64, 64, use_cuda_rasterizer="pytorch").to(device)
    gauss_opt = _build_gaussian_optimizer(model, cfg)
    deform_opt = _build_deformation_optimizer(model, cfg)

    train_dataset = FluidDataset(
        data_root=data_root,
        split="train",
        resolution=(64, 64),
        load_flow=False,
        load_depth=False,
        augment=False,
    )

    loss_cfg = LossConfig()
    stats = DensificationStats(model.gaussian_model.num_points, device)
    n_start = model.gaussian_model.num_points
    print(f"  Starting Gaussian count: {n_start}")

    # Train for densify_start + densify_interval iterations
    densify_start = 20
    densify_interval = 10
    total_iters = densify_start + densify_interval + 1

    for it in range(total_iters):
        idx = it % len(train_dataset)
        sample = train_dataset[idx]

        gt_image = sample["image"].to(device)
        K = sample["K"].to(device)
        w2c = sample["w2c"].to(device)
        t_norm = sample["time_normalized"]
        H, W = gt_image.shape[1], gt_image.shape[2]
        camera = {"K": K, "w2c": w2c, "image_height": H, "image_width": W}

        gaussians = model.forward(time=t_norm)
        rendered = renderer.render(gaussians, camera)

        loss_val, _ = total_loss(
            rendered=rendered["render"],
            target=gt_image,
            opacity=gaussians["opacity"],
            scaling=gaussians["scaling"],
            config=loss_cfg,
        )

        gauss_opt.zero_grad(set_to_none=True)
        deform_opt.zero_grad(set_to_none=True)
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gauss_opt.step()
        deform_opt.step()

        # Update densification stats when in range
        if it >= densify_start:
            vis = rendered["visibility_filter"]
            if model.gaussian_model._xyz.grad is not None:
                stats.update(vis, model.gaussian_model._xyz.grad)

            if (it + 1) % densify_interval == 0:
                n_before = model.gaussian_model.num_points
                gauss_opt, stats = _densify_and_prune(
                    model,
                    gauss_opt,
                    stats,
                    grad_threshold=0.0002,
                    max_gaussians=500,
                    iteration=it,
                    device=device,
                )
                n_after = model.gaussian_model.num_points
                print(f"  Densification at iter {it}: {n_before} -> {n_after}")

    n_end = model.gaussian_model.num_points
    print(f"  Final Gaussian count: {n_end}")

    # The count should have changed (either up from cloning or down from pruning)
    check(n_end != n_start, f"Gaussian count changed ({n_start} -> {n_end})")

    # Verify optimizer groups are aligned
    for g in gauss_opt.param_groups:
        p = g["params"][0]
        check(p.shape[0] == n_end, f"Optimizer group '{g['name']}' aligned (size={p.shape[0]})")

    # Verify model can still forward
    out = model.forward(time=0.5)
    check(out["xyz"].shape[0] == n_end, f"Model forward after densification (N={n_end})")


# =====================================================================
# Step 4: Verify checkpointing
# =====================================================================
def step4_verify_checkpointing(output_dir: Path):
    print("\n=== Step 4: Verify checkpointing ===")

    ckpt_dir = output_dir / "test_scene" / "checkpoints"

    # Check files exist
    ckpt_50 = ckpt_dir / "ckpt_000050.pt"
    ckpt_100 = ckpt_dir / "ckpt_000100.pt"
    ckpt_final = ckpt_dir / "final.pt"
    ckpt_best = ckpt_dir / "best.pt"

    check(ckpt_50.exists(), f"Checkpoint at iter 50 exists ({ckpt_50.name})")
    check(ckpt_100.exists() or ckpt_final.exists(), "Checkpoint at iter 100 or final exists")

    # Load and verify contents
    ckpt_path = ckpt_50 if ckpt_50.exists() else ckpt_final
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    check("iteration" in ckpt, "Checkpoint has 'iteration'")
    check("model_state_dict" in ckpt, "Checkpoint has 'model_state_dict'")
    check("gauss_optimizer_state_dict" in ckpt, "Checkpoint has 'gauss_optimizer_state_dict'")
    check("deform_optimizer_state_dict" in ckpt, "Checkpoint has 'deform_optimizer_state_dict'")
    check("best_psnr" in ckpt, "Checkpoint has 'best_psnr'")
    check("num_gaussians" in ckpt, "Checkpoint has 'num_gaussians'")
    check("grad_accum" in ckpt, "Checkpoint has 'grad_accum'")

    print(f"  Checkpoint iteration: {ckpt['iteration']}")
    print(f"  Checkpoint N gaussians: {ckpt['num_gaussians']}")
    print(f"  Checkpoint best PSNR: {ckpt['best_psnr']:.2f}")

    check(ckpt["iteration"] == 50, f"Iteration == 50 (got {ckpt['iteration']})")

    # Load into fresh model and verify forward pass
    from gaussian_splatting.model import DynamicGaussianModel

    n_gauss = ckpt["num_gaussians"]
    model2 = DynamicGaussianModel(sh_degree=2, num_points=n_gauss)
    # Initialize with dummy data to set correct sizes
    model2.gaussian_model.initialize_from_point_cloud(
        torch.randn(n_gauss, 3),
        torch.rand(n_gauss, 3),
    )
    model2.load_state_dict(ckpt["model_state_dict"])

    out = model2.forward(time=0.5)
    check(out["xyz"].shape == (n_gauss, 3), f"Loaded model forward: xyz shape ({n_gauss}, 3)")
    check(
        out["opacity"].shape == (n_gauss, 1), f"Loaded model forward: opacity shape ({n_gauss}, 1)"
    )
    check(
        out["covariance"].shape == (n_gauss, 3, 3),
        f"Loaded model forward: covariance shape ({n_gauss}, 3, 3)",
    )


# =====================================================================
# Step 5: Verify evaluation
# =====================================================================
def step5_verify_evaluation(config_path: Path, data_root: Path):
    print("\n=== Step 5: Verify evaluation ===")

    from omegaconf import OmegaConf

    from gaussian_splatting.dataset import FluidDataset
    from gaussian_splatting.model import DynamicGaussianModel
    from gaussian_splatting.renderer import GaussianRenderer
    from gaussian_splatting.train import evaluate, initialize_point_cloud

    cfg = OmegaConf.load(str(config_path))
    device = torch.device("cpu")
    torch.manual_seed(42)

    model = DynamicGaussianModel(sh_degree=2, num_points=200)
    xyz, rgb = initialize_point_cloud(cfg, data_root, device)
    model.gaussian_model.initialize_from_point_cloud(xyz, rgb)
    model = model.to(device)

    renderer = GaussianRenderer(64, 64, use_cuda_rasterizer="pytorch").to(device)

    val_dataset = FluidDataset(
        data_root=data_root,
        split="val",
        resolution=(64, 64),
        load_flow=False,
        load_depth=True,
        augment=False,
    )

    print(f"  Val dataset size: {len(val_dataset)}")
    check(len(val_dataset) > 0, f"Val dataset has samples ({len(val_dataset)})")

    metrics = evaluate(model, renderer, val_dataset, device, max_samples=5)

    print(f"  Eval PSNR: {metrics['psnr']:.2f}")
    print(f"  Eval L1:   {metrics['l1']:.4f}")
    print(f"  N eval:    {metrics['n_eval']}")

    check(metrics["psnr"] > 5.0, f"Eval PSNR > 5.0 ({metrics['psnr']:.2f})")
    check(0 < metrics["l1"] < 1.0, f"Eval L1 in (0, 1) ({metrics['l1']:.4f})")
    check(not math.isnan(metrics["psnr"]), "Eval PSNR is not NaN")
    check(metrics["n_eval"] > 0, f"Evaluated {metrics['n_eval']} samples")


# =====================================================================
# Step 6: Resume test
# =====================================================================
def step6_resume_test(config_path: Path, output_dir: Path):
    print("\n=== Step 6: Resume test ===")

    from omegaconf import OmegaConf

    from gaussian_splatting.dataset import FluidDataset
    from gaussian_splatting.losses import LossConfig, total_loss
    from gaussian_splatting.model import DynamicGaussianModel
    from gaussian_splatting.renderer import GaussianRenderer
    from gaussian_splatting.train import (
        DensificationStats,
        _build_deformation_optimizer,
        _build_gaussian_optimizer,
        initialize_point_cloud,
        load_checkpoint,
        save_checkpoint,
    )

    cfg = OmegaConf.load(str(config_path))
    device = torch.device("cpu")
    data_root = Path(cfg.data.root_dir)

    ckpt_dir = output_dir / "test_scene" / "checkpoints"
    ckpt_path = ckpt_dir / "ckpt_000050.pt"
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / "final.pt"

    check(ckpt_path.exists(), f"Checkpoint exists for resume ({ckpt_path.name})")

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    n_gauss = ckpt["num_gaussians"]

    model = DynamicGaussianModel(sh_degree=2, num_points=n_gauss)
    model.gaussian_model.initialize_from_point_cloud(
        torch.randn(n_gauss, 3),
        torch.rand(n_gauss, 3),
    )
    model = model.to(device)

    gauss_opt = _build_gaussian_optimizer(model, cfg)
    deform_opt = _build_deformation_optimizer(model, cfg)

    start_iter, best_psnr, stats = load_checkpoint(
        ckpt_path,
        model,
        gauss_opt,
        deform_opt,
        device,
    )

    print(f"  Resumed at iteration {start_iter}")
    check(start_iter == 50, f"Start iter == 50 (got {start_iter})")
    check(best_psnr > 0, f"Best PSNR > 0 ({best_psnr:.2f})")

    # Run 10 more iterations
    train_dataset = FluidDataset(
        data_root=data_root,
        split="train",
        resolution=(64, 64),
        load_flow=False,
        load_depth=False,
        augment=False,
    )
    renderer = GaussianRenderer(64, 64, use_cuda_rasterizer="pytorch").to(device)
    loss_cfg = LossConfig()

    model.train()
    for it in range(10):
        idx = it % len(train_dataset)
        sample = train_dataset[idx]

        gt_image = sample["image"].to(device)
        K = sample["K"].to(device)
        w2c = sample["w2c"].to(device)
        t_norm = sample["time_normalized"]
        H, W = gt_image.shape[1], gt_image.shape[2]
        camera = {"K": K, "w2c": w2c, "image_height": H, "image_width": W}

        gaussians = model.forward(time=t_norm)
        rendered = renderer.render(gaussians, camera)

        loss_val, _ = total_loss(
            rendered=rendered["render"],
            target=gt_image,
            opacity=gaussians["opacity"],
            scaling=gaussians["scaling"],
            config=loss_cfg,
        )

        gauss_opt.zero_grad(set_to_none=True)
        deform_opt.zero_grad(set_to_none=True)
        loss_val.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gauss_opt.step()
        deform_opt.step()

    check(True, "10 iterations after resume completed without crash")

    # Verify forward still works
    out = model.forward(time=0.5)
    check(out["xyz"].shape[0] == n_gauss, f"Forward works after resume training (N={n_gauss})")

    # Save the new checkpoint
    resume_ckpt_path = ckpt_dir / "resume_test.pt"
    save_checkpoint(
        resume_ckpt_path,
        model,
        gauss_opt,
        deform_opt,
        start_iter + 10,
        best_psnr,
        stats,
    )
    check(resume_ckpt_path.exists(), "Post-resume checkpoint saved")

    # Verify post-resume checkpoint loads correctly
    ckpt2 = torch.load(resume_ckpt_path, map_location="cpu", weights_only=False)
    check(
        ckpt2["iteration"] == start_iter + 10,
        f"Post-resume iteration == {start_iter + 10} (got {ckpt2['iteration']})",
    )


# =====================================================================
# Step 7: PLY loader test
# =====================================================================
def step7_ply_loader(data_root: Path):
    print("\n=== Step 7: PLY loader test ===")

    from gaussian_splatting.train import _load_ply_points

    ply_path = data_root / "colmap" / "sparse" / "0" / "points3D.ply"
    check(ply_path.exists(), f"PLY file exists")

    xyz, rgb = _load_ply_points(ply_path)
    check(xyz.shape == (50, 3), f"PLY xyz shape = (50, 3) (got {xyz.shape})")
    check(rgb.shape == (50, 3), f"PLY rgb shape = (50, 3) (got {rgb.shape})")
    check(rgb.min() >= 0.0, "PLY rgb >= 0")
    check(rgb.max() <= 1.0, "PLY rgb <= 1")
    check(xyz.dtype == np.float32, "PLY xyz dtype = float32")


# =====================================================================
# Step 8: Depth unprojection test
# =====================================================================
def step8_depth_unproject(data_root: Path):
    print("\n=== Step 8: Depth unprojection test ===")

    from gaussian_splatting.train import _unproject_depth

    K = np.array([[50, 0, 32], [0, 50, 32], [0, 0, 1]], dtype=np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[2, 3] = 3.0  # camera at z=3

    depth = np.full((64, 64), 3.0, dtype=np.float32)
    image = np.random.rand(64, 64, 3).astype(np.float32)

    xyz, rgb = _unproject_depth(depth, K, c2w, image=image, subsample=8)

    check(xyz.shape[1] == 3, f"Unprojected xyz has 3 columns")
    check(rgb.shape[1] == 3, f"Unprojected rgb has 3 columns")
    check(xyz.shape[0] == rgb.shape[0], "xyz and rgb have same count")
    check(xyz.shape[0] > 0, f"Got {xyz.shape[0]} unprojected points")

    # Center pixels at depth=3 with c2w at z=3 should be near (0,0,6)
    # (camera looks down -Z, points at z=3 in camera = z=3+3=6 in world)
    center_mask = (np.abs(xyz[:, 0]) < 1) & (np.abs(xyz[:, 1]) < 1)
    if center_mask.sum() > 0:
        mean_z = xyz[center_mask, 2].mean()
        check(abs(mean_z - 6.0) < 1.0, f"Center points z ~ 6.0 (got {mean_z:.2f})")


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("  TEST: gaussian_splatting/train.py")
    print("=" * 70)

    # Use a temp directory for all test data
    tmp_base = Path(tempfile.mkdtemp(prefix="test_train_"))
    data_root = tmp_base / "data"
    output_dir = tmp_base / "outputs"
    config_path = tmp_base / "test_train.yaml"

    try:
        create_synthetic_dataset(data_root)
        create_test_config(data_root, output_dir, config_path)

        step1_run_training(config_path, output_dir)
        step2_verify_loss_decrease(config_path, data_root, output_dir)
        step3_verify_densification(config_path, data_root)
        step4_verify_checkpointing(output_dir)
        step5_verify_evaluation(config_path, data_root)
        step6_resume_test(config_path, output_dir)
        step7_ply_loader(data_root)
        step8_depth_unproject(data_root)

    finally:
        # Step 8: Cleanup
        print("\n=== Step 9: Cleanup ===")
        shutil.rmtree(tmp_base, ignore_errors=True)
        print(f"  Cleaned up {tmp_base}")
        check(not tmp_base.exists(), "Temp directory cleaned up")

    print("\n" + "=" * 70)
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 70)

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: train.py"
            " — loss decreases, checkpointing works, resume works,"
            " evaluation runs"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
