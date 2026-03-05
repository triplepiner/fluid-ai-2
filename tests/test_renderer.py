#!/usr/bin/env python3
"""Thorough test for gaussian_splatting/renderer.py."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N = 500
H, W = 128, 128
SH_DEGREE = 0  # DC-only keeps things simple and fast

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
# Shared setup
# =====================================================================


def make_model_and_camera():
    """Create DynamicGaussianModel with 500 Gaussians + pinhole camera."""
    from gaussian_splatting.model import DynamicGaussianModel

    model = DynamicGaussianModel(sh_degree=SH_DEGREE, num_points=N)
    torch.manual_seed(42)
    xyz = torch.rand(N, 3) * 2.0 - 1.0  # uniform in [-1, 1]^3
    rgb = torch.rand(N, 3)
    model.gaussian_model.initialize_from_point_cloud(xyz, rgb)

    K = torch.tensor(
        [
            [128.0, 0.0, 64.0],
            [0.0, 128.0, 64.0],
            [0.0, 0.0, 1.0],
        ]
    )
    w2c = torch.eye(4)
    w2c[2, 3] = 3.0  # camera 3 units along +z

    camera = {"K": K, "w2c": w2c, "image_height": H, "image_width": W}
    return model, camera


# =====================================================================
# Step 1: Setup
# =====================================================================
def step1_setup():
    print("\n=== Step 1: Setup ===")
    model, camera = make_model_and_camera()

    check(model.gaussian_model.num_points == N, f"Model has {N} Gaussians")

    xyz = model.gaussian_model.get_xyz()
    check(
        xyz.min().item() >= -1.5 and xyz.max().item() <= 1.5,
        f"Gaussians in expected range (min={xyz.min():.2f}, max={xyz.max():.2f})",
    )

    K = camera["K"]
    check(K[0, 0].item() == 128.0 and K[1, 1].item() == 128.0, f"Intrinsics fx=fy=128")
    check(K[0, 2].item() == 64.0 and K[1, 2].item() == 64.0, f"Principal point at (64, 64)")
    check(camera["w2c"][2, 3].item() == 3.0, f"Camera translated 3 units along z")

    return model, camera


# =====================================================================
# Step 2: Render test
# =====================================================================
def step2_render(model, camera):
    print("\n=== Step 2: Render test ===")
    from gaussian_splatting.renderer import GaussianRenderer

    renderer = GaussianRenderer(
        image_height=H,
        image_width=W,
        background_color=(0.0, 0.0, 0.0),
        use_cuda_rasterizer="pytorch",
    )

    gaussians = model.forward(time=0.5)
    result = renderer.render(gaussians, camera)

    # --- Keys present ---
    print("  -- Output keys --")
    for key in ["render", "depth", "alpha", "visibility_filter", "radii"]:
        check(key in result, f"'{key}' in result")

    # --- Shapes ---
    print("  -- Shapes --")
    check(
        result["render"].shape == (3, H, W),
        f"render shape = (3, {H}, {W}) (got {tuple(result['render'].shape)})",
    )
    check(
        result["depth"].shape == (1, H, W),
        f"depth shape = (1, {H}, {W}) (got {tuple(result['depth'].shape)})",
    )
    check(
        result["alpha"].shape == (1, H, W),
        f"alpha shape = (1, {H}, {W}) (got {tuple(result['alpha'].shape)})",
    )
    check(result["visibility_filter"].shape == (N,), f"visibility_filter shape = ({N},)")
    check(result["radii"].shape == (N,), f"radii shape = ({N},)")

    # --- Value ranges ---
    print("  -- Value ranges --")
    check(
        result["render"].min().item() >= 0.0, f"render min >= 0 (got {result['render'].min():.6f})"
    )
    check(
        result["render"].max().item() <= 1.0, f"render max <= 1 (got {result['render'].max():.6f})"
    )
    check(result["depth"].min().item() >= 0.0, f"depth min >= 0 (got {result['depth'].min():.6f})")
    check(result["alpha"].min().item() >= 0.0, f"alpha min >= 0 (got {result['alpha'].min():.6f})")
    check(result["alpha"].max().item() <= 1.0, f"alpha max <= 1 (got {result['alpha'].max():.6f})")

    # --- Not blank ---
    print("  -- Non-trivial output --")
    render_max = result["render"].max().item()
    alpha_max = result["alpha"].max().item()
    check(render_max > 0.01, f"Image is NOT all zeros (max pixel = {render_max:.4f})")
    check(alpha_max > 0.01, f"Alpha is NOT all zeros (max alpha = {alpha_max:.4f})")

    # Spatial variation: std across spatial dims should be non-trivial
    spatial_std = result["render"].std().item()
    check(spatial_std > 0.001, f"Image has spatial variation (std = {spatial_std:.6f})")

    # Some Gaussians should be visible
    num_visible = result["visibility_filter"].sum().item()
    check(num_visible > 0, f"Some Gaussians visible ({int(num_visible)} / {N})")

    # --- Save rendered image as PNG ---
    print("  -- Save PNG --")
    out_dir = Path(__file__).resolve().parent / "_test_renderer_output"
    out_dir.mkdir(exist_ok=True)
    _save_render_png(result["render"], out_dir / "step2_render.png")
    check((out_dir / "step2_render.png").exists(), "Saved step2_render.png")

    return result, renderer


# =====================================================================
# Step 3: Gradient flow (MOST CRITICAL)
# =====================================================================
def step3_gradient_flow(model, camera):
    print("\n=== Step 3: Gradient flow test (CRITICAL) ===")
    from gaussian_splatting.renderer import GaussianRenderer

    renderer = GaussianRenderer(
        image_height=H,
        image_width=W,
        background_color=(0.0, 0.0, 0.0),
        use_cuda_rasterizer="pytorch",
    )

    model.zero_grad()
    gaussians = model.forward(time=0.5)
    result = renderer.render(gaussians, camera)

    # MSE loss against random target
    torch.manual_seed(99)
    target = torch.rand(3, H, W)
    loss = F.mse_loss(result["render"], target)
    print(f"    MSE loss = {loss.item():.6f}")
    loss.backward()

    # --- Gaussian model gradients ---
    print("  -- Gaussian model gradients --")
    xyz_grad = model.gaussian_model._xyz.grad
    check(xyz_grad is not None, "gradient on _xyz exists")
    if xyz_grad is not None:
        xyz_grad_norm = xyz_grad.norm().item()
        check(
            xyz_grad_norm > 0, f"gradient on _xyz is NOT all zeros (||grad|| = {xyz_grad_norm:.6e})"
        )
        print(f"    ||grad_xyz|| = {xyz_grad_norm:.6e}")
    else:
        check(False, "gradient on _xyz is NOT all zeros (grad is None!)")

    opacity_grad = model.gaussian_model._opacity.grad
    check(opacity_grad is not None, "gradient on _opacity exists")
    if opacity_grad is not None:
        check(
            opacity_grad.norm().item() > 0,
            f"gradient on _opacity non-zero (||grad|| = {opacity_grad.norm():.6e})",
        )

    features_grad = model.gaussian_model._features_dc.grad
    check(features_grad is not None, "gradient on _features_dc exists")
    if features_grad is not None:
        check(
            features_grad.norm().item() > 0,
            f"gradient on _features_dc non-zero (||grad|| = {features_grad.norm():.6e})",
        )

    scaling_grad = model.gaussian_model._scaling.grad
    check(scaling_grad is not None, "gradient on _scaling exists")

    rotation_grad = model.gaussian_model._rotation.grad
    check(rotation_grad is not None, "gradient on _rotation exists")

    # --- Deformation network gradients ---
    print("  -- Deformation network gradients --")
    deform_grads = [
        (name, p.grad)
        for name, p in model.deformation_network.named_parameters()
        if p.grad is not None
    ]
    check(
        len(deform_grads) > 0,
        f"DeformationNetwork has grads ({len(deform_grads)} params with grad)",
    )

    deform_grad_norm = sum(
        p.grad.norm().item()
        for _, p in model.deformation_network.named_parameters()
        if p.grad is not None
    )
    check(deform_grad_norm > 0, f"DeformationNetwork total grad norm > 0 ({deform_grad_norm:.6e})")


# =====================================================================
# Step 4: Depth rendering test
# =====================================================================
def step4_depth(model, camera):
    print("\n=== Step 4: Depth rendering test ===")
    from gaussian_splatting.renderer import GaussianRenderer

    renderer = GaussianRenderer(
        image_height=H,
        image_width=W,
        background_color=(0.0, 0.0, 0.0),
        use_cuda_rasterizer="pytorch",
    )

    with torch.no_grad():
        gaussians = model.forward(time=0.5)
        result = renderer.render(gaussians, camera)

    depth = result["depth"]  # (1, H, W)
    alpha = result["alpha"]  # (1, H, W)

    # Pixels with significant alpha
    high_alpha_mask = alpha[0] > 0.5
    num_high_alpha = high_alpha_mask.sum().item()
    print(f"    Pixels with alpha > 0.5: {int(num_high_alpha)}")

    check(num_high_alpha > 0, f"Some pixels have alpha > 0.5 ({int(num_high_alpha)} pixels)")

    if num_high_alpha > 0:
        depth_at_high_alpha = depth[0][high_alpha_mask]
        mean_depth = depth_at_high_alpha.mean().item()
        print(f"    Mean depth at high-alpha pixels: {mean_depth:.4f}")

        # Camera is at z=3, Gaussians are in [-1,1] near origin.
        # In camera space: Gaussian z_cam = world_z + 3 (since w2c is just
        # a z-translation of +3).  So z_cam ranges from 2 to 4 for
        # Gaussians in [-1,1].  The mean should be around 3.
        check(1.0 < mean_depth < 5.0, f"Mean depth in [1, 5] (got {mean_depth:.4f})")

        # Depth should be positive at visible pixels
        check(
            depth_at_high_alpha.min().item() > 0,
            f"Depth positive at visible pixels (min={depth_at_high_alpha.min():.4f})",
        )

    # Depth at zero-alpha pixels should be near zero (no contribution)
    low_alpha_mask = alpha[0] < 0.01
    if low_alpha_mask.sum().item() > 0:
        depth_at_low_alpha = depth[0][low_alpha_mask]
        mean_bg_depth = depth_at_low_alpha.mean().item()
        check(
            mean_bg_depth < 0.5, f"Depth at background pixels near zero (mean={mean_bg_depth:.6f})"
        )


# =====================================================================
# Step 5: Different viewpoints
# =====================================================================
def step5_viewpoints(model):
    print("\n=== Step 5: Different viewpoints ===")
    from gaussian_splatting.renderer import GaussianRenderer

    renderer = GaussianRenderer(
        image_height=H,
        image_width=W,
        background_color=(0.0, 0.0, 0.0),
        use_cuda_rasterizer="pytorch",
    )

    K = torch.tensor(
        [
            [128.0, 0.0, 64.0],
            [0.0, 128.0, 64.0],
            [0.0, 0.0, 1.0],
        ]
    )

    # Camera positions: front, left, right
    cameras = {}

    # Front: camera at (0, 0, -3) looking toward origin along +z
    w2c_front = torch.eye(4)
    w2c_front[2, 3] = 3.0  # t = -R @ pos = -I @ [0,0,-3] = [0,0,3]
    cameras["front"] = {"K": K, "w2c": w2c_front, "image_height": H, "image_width": W}

    # Left: camera at (-3, 0, 0) looking toward origin along +x
    # c2w columns: right=[0,0,-1], up=[0,1,0], forward=[1,0,0], pos=[-3,0,0]
    # R_c2w = [[0,0,1],[0,1,0],[-1,0,0]] -> R_w2c = R_c2w^T
    w2c_left = torch.eye(4)
    R_left = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    pos_left = torch.tensor([-3.0, 0.0, 0.0])
    w2c_left[:3, :3] = R_left
    w2c_left[:3, 3] = -(R_left @ pos_left)  # t = -R @ pos
    cameras["left"] = {"K": K, "w2c": w2c_left, "image_height": H, "image_width": W}

    # Right: camera at (3, 0, 0) looking toward origin along -x
    # c2w columns: right=[0,0,1], up=[0,1,0], forward=[-1,0,0], pos=[3,0,0]
    # R_c2w = [[0,0,-1],[0,1,0],[1,0,0]] -> R_w2c = R_c2w^T
    w2c_right = torch.eye(4)
    R_right = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ]
    )
    pos_right = torch.tensor([3.0, 0.0, 0.0])
    w2c_right[:3, :3] = R_right
    w2c_right[:3, 3] = -(R_right @ pos_right)  # t = -R @ pos
    cameras["right"] = {"K": K, "w2c": w2c_right, "image_height": H, "image_width": W}

    renders = {}
    out_dir = Path(__file__).resolve().parent / "_test_renderer_output"
    out_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        gaussians = model.forward(time=0.5)
        for name, cam in cameras.items():
            result = renderer.render(gaussians, cam)
            renders[name] = result["render"]

            png_path = out_dir / f"step5_{name}.png"
            _save_render_png(result["render"], png_path)
            check(png_path.exists(), f"Saved {png_path.name}")

            check(
                result["render"].max().item() > 0.001,
                f"'{name}' view is not blank (max={result['render'].max():.4f})",
            )

    # --- All 3 views should be different ---
    print("  -- Views are different --")
    names = list(renders.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            diff = (renders[names[i]] - renders[names[j]]).abs().mean().item()
            check(diff > 0.001, f"'{names[i]}' vs '{names[j]}' differ (mean abs diff = {diff:.6f})")


# =====================================================================
# Step 6: Empty scene test
# =====================================================================
def step6_empty_scene():
    print("\n=== Step 6: Empty scene test ===")
    from gaussian_splatting.model import DynamicGaussianModel
    from gaussian_splatting.renderer import GaussianRenderer

    renderer = GaussianRenderer(
        image_height=H,
        image_width=W,
        background_color=(0.0, 0.0, 0.0),
        use_cuda_rasterizer="pytorch",
    )

    # 10 Gaussians far behind the camera (z = -100)
    model = DynamicGaussianModel(sh_degree=SH_DEGREE, num_points=10)
    torch.manual_seed(123)
    xyz = torch.randn(10, 3) * 0.1
    xyz[:, 2] = -100.0  # far behind camera
    rgb = torch.rand(10, 3)
    model.gaussian_model.initialize_from_point_cloud(xyz, rgb)

    K = torch.tensor(
        [
            [128.0, 0.0, 64.0],
            [0.0, 128.0, 64.0],
            [0.0, 0.0, 1.0],
        ]
    )
    w2c = torch.eye(4)
    w2c[2, 3] = 3.0
    camera = {"K": K, "w2c": w2c, "image_height": H, "image_width": W}

    with torch.no_grad():
        gaussians = model.forward(time=0.0)
        result = renderer.render(gaussians, camera)

    render = result["render"]
    alpha = result["alpha"]

    print(f"    render max = {render.max().item():.6f}")
    print(f"    alpha max  = {alpha.max().item():.6f}")

    # Image should be approximately background (black)
    check(render.max().item() < 0.01, f"Render is ~background (max pixel = {render.max():.6f})")
    check(alpha.max().item() < 0.01, f"Alpha is ~zero everywhere (max = {alpha.max():.6f})")


# =====================================================================
# Helper: save render as PNG
# =====================================================================
def _save_render_png(render: torch.Tensor, path: Path):
    """Save a (3, H, W) float [0,1] tensor as PNG."""
    import numpy as np

    try:
        import cv2

        img = render.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        img_uint8 = (img * 255).astype(np.uint8)
        # OpenCV uses BGR
        cv2.imwrite(str(path), img_uint8[:, :, ::-1])
    except ImportError:
        # Fallback: write raw PPM
        img = render.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
        img_uint8 = (img * 255).astype(np.uint8)
        h, w, _ = img_uint8.shape
        ppm_path = path.with_suffix(".ppm")
        with open(ppm_path, "wb") as f:
            f.write(f"P6\n{w} {h}\n255\n".encode())
            f.write(img_uint8.tobytes())


# =====================================================================
# Cleanup
# =====================================================================
def cleanup():
    import shutil

    out_dir = Path(__file__).resolve().parent / "_test_renderer_output"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    check(not out_dir.exists(), "Test output directory cleaned up")


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("  TEST: gaussian_splatting/renderer.py")
    print("=" * 70)

    model, camera = step1_setup()
    step2_render(model, camera)
    step3_gradient_flow(model, camera)
    step4_depth(model, camera)
    step5_viewpoints(model)
    step6_empty_scene()
    cleanup()

    print("\n" + "=" * 70)
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 70)

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: renderer.py"
            " — renders valid images, gradients flow to Gaussian positions,"
            " depth is consistent, multiple viewpoints work"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
