#!/usr/bin/env python3
"""Thorough integration tests for advection, forward prediction, and novel view synthesis.

Tests:
1. advection.py — Euler, RK4, trajectory, PhysicsGuidedDeformation
2. forward_predict.py — single frame, sequence, video
3. novel_view.py — camera interpolation, orbit render, orbit video
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

import numpy as np
import torch

from gaussian_splatting.model import DynamicGaussianModel
from gaussian_splatting.renderer import GaussianRenderer
from integration.advection import (
    PhysicsGuidedDeformation,
    advect_euler,
    advect_rk4,
    advect_trajectory,
)
from integration.forward_predict import ForwardPredictor, extrapolate_density
from integration.novel_view import (
    NovelViewSynthesizer,
    _quaternion_to_rotation_matrix,
    _rotation_matrix_to_quaternion,
    interpolate_cameras,
    look_at,
    make_camera,
)
from pinn.model import FluidPINN, PINNConfig

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------

checks_passed = 0
checks_total = 0


def check(cond: bool, label: str) -> None:
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if cond:
        checks_passed += 1


# ---------------------------------------------------------------------------
# Shared setup
# ---------------------------------------------------------------------------

print("=" * 70)
print("  test_integration.py — comprehensive tests")
print("=" * 70)

torch.manual_seed(42)

# Small PINN (untrained — fine for shape/finiteness checks)
pinn = FluidPINN(PINNConfig(activation="siren", hidden_dim=64, num_layers=4))
pinn.eval()

# 200 Gaussians for forward prediction / novel view tests
N_gauss = 200
gauss_model = DynamicGaussianModel(sh_degree=0, num_points=N_gauss)
gauss_model.gaussian_model.initialize_from_point_cloud(
    torch.randn(N_gauss, 3) * 0.5,
    torch.rand(N_gauss, 3),
)
gauss_model.eval()

# Renderer at 128x128
H, W = 128, 128
renderer = GaussianRenderer(H, W, use_cuda_rasterizer="pytorch")

# Camera (same convention as renderer self-test)
K = torch.tensor(
    [
        [128.0, 0.0, W / 2.0],
        [0.0, 128.0, H / 2.0],
        [0.0, 0.0, 1.0],
    ]
)
w2c = torch.eye(4)
w2c[2, 3] = 3.0  # camera at world (0,0,-3), looking +Z
camera = {"K": K, "w2c": w2c, "image_height": H, "image_width": W}


# ===================================================================
# 1. ADVECTION TESTS
# ===================================================================
print("\n" + "=" * 70)
print("  1. ADVECTION TESTS (advection.py)")
print("=" * 70)

N = 100
xyz = torch.randn(N, 3)

# --- Euler ---
print("\n--- Euler ---")
xyz_euler = advect_euler(pinn, xyz, t=0.0, dt=0.01)
check(xyz_euler.shape == (N, 3), f"Euler shape = ({N}, 3)")
check(not torch.equal(xyz_euler, xyz), "Euler output DIFFERENT from input")
check(torch.isfinite(xyz_euler).all().item(), "Euler output: no NaN/Inf")

# --- RK4 ---
print("\n--- RK4 ---")
xyz_rk4 = advect_rk4(pinn, xyz, t=0.0, dt=0.01)
check(xyz_rk4.shape == (N, 3), f"RK4 shape = ({N}, 3)")
check(not torch.equal(xyz_rk4, xyz), "RK4 output DIFFERENT from input")
check(torch.isfinite(xyz_rk4).all().item(), "RK4 output: no NaN/Inf")

# RK4 != Euler (different integration methods)
rk4_euler_diff = (xyz_rk4 - xyz_euler).abs().max().item()
check(rk4_euler_diff > 0, f"RK4 vs Euler differ (max diff = {rk4_euler_diff:.2e})")

# --- Trajectory ---
print("\n--- Trajectory ---")
traj = advect_trajectory(pinn, xyz, t_start=0.0, t_end=0.5, num_steps=10, method="rk4")
check(traj.shape == (11, N, 3), f"Trajectory shape = (11, {N}, 3)")
check(torch.equal(traj[0], xyz), "traj[0] == xyz (start position)")
check(torch.isfinite(traj).all().item(), "Trajectory: no NaN/Inf")

# Consecutive positions should be different (particles are moving)
consecutive_diffs = []
for step in range(10):
    diff = (traj[step + 1] - traj[step]).abs().max().item()
    consecutive_diffs.append(diff)
check(
    all(d > 0 for d in consecutive_diffs),
    f"All consecutive steps differ (min diff = {min(consecutive_diffs):.2e})",
)

# --- PhysicsGuidedDeformation ---
print("\n--- PhysicsGuidedDeformation ---")
deform = PhysicsGuidedDeformation(pinn, t_canonical=0.0, num_advection_steps=5, method="rk4")

result = deform(xyz, time=0.5)
check("delta_xyz" in result, "result contains 'delta_xyz'")
check(result["delta_xyz"].shape == (N, 3), f"delta_xyz shape = ({N}, 3)")
check(
    result["delta_xyz"].abs().max().item() > 1e-6,
    f"delta_xyz is non-zero (max = {result['delta_xyz'].abs().max().item():.4e})",
)
check(torch.isfinite(result["delta_xyz"]).all().item(), "delta_xyz: no NaN/Inf")

# Also check other keys exist
check(
    "delta_rotation" in result and result["delta_rotation"].shape == (N, 4),
    "delta_rotation shape = (100, 4)",
)
check(
    "delta_scaling" in result and result["delta_scaling"].shape == (N, 3),
    "delta_scaling shape = (100, 3)",
)
check(
    "delta_opacity" in result and result["delta_opacity"].shape == (N, 1),
    "delta_opacity shape = (100, 1)",
)

# At canonical time (t=0), delta_xyz should be zero
result_t0 = deform(xyz, time=0.0)
check((result_t0["delta_xyz"].abs() < 1e-6).all().item(), "delta_xyz ~ 0 at canonical time (t=0)")


# ===================================================================
# 2. FORWARD PREDICTION TESTS
# ===================================================================
print("\n" + "=" * 70)
print("  2. FORWARD PREDICTION TESTS (forward_predict.py)")
print("=" * 70)

predictor = ForwardPredictor(
    gauss_model,
    pinn,
    renderer,
    t_canonical=0.0,
    num_advection_steps=5,
)

# --- Single frame ---
print("\n--- Single frame ---")
frame = predictor.predict_frame(camera, time=0.5)
check(frame["image"].shape == (3, H, W), f"image shape = (3, {H}, {W})")
check(frame["depth"].shape == (1, H, W), f"depth shape = (1, {H}, {W})")
check(frame["alpha"].shape == (1, H, W), f"alpha shape = (1, {H}, {W})")
check(frame["image"].min().item() >= 0.0, "image values >= 0")
check(frame["image"].max().item() <= 1.0, "image values <= 1")
check(torch.isfinite(frame["image"]).all().item(), "image: no NaN/Inf")

# Not all zero — at least some Gaussians should be visible
img_sum = frame["image"].sum().item()
check(img_sum > 0, f"image not all zero (sum = {img_sum:.2f})")

# --- Sequence ---
print("\n--- Sequence ---")
frames = predictor.predict_sequence(camera, t_start=0.0, t_end=1.0, num_frames=5)
check(len(frames) == 5, "sequence has 5 frames")

# Check keys and shapes
all_keys_ok = all("image" in f and "depth" in f and "alpha" in f and "time" in f for f in frames)
check(all_keys_ok, "all frames have image/depth/alpha/time keys")
check(
    all(f["image"].shape == (3, H, W) for f in frames), f"all frames have image shape (3, {H}, {W})"
)

# Frame 0 vs frame 4 should differ (Gaussians advected to different times)
pixel_diff = (frames[0]["image"] - frames[4]["image"]).abs().max().item()
check(pixel_diff > 0, f"frame 0 != frame 4 (max pixel diff = {pixel_diff:.4e})")

# --- Video ---
print("\n--- Video ---")
video_path = _ROOT / "outputs" / "renders" / "test_prediction.mp4"
predictor.predict_video(
    camera,
    t_start=0.0,
    t_end=1.0,
    fps=5,
    output_path=video_path,
)
# The video writer may fall back to PNG frames if no encoder is available
video_exists = video_path.exists()
frames_dir = video_path.parent / (video_path.stem + "_frames")
frames_exist = frames_dir.exists() and any(frames_dir.iterdir()) if frames_dir.exists() else False

output_created = video_exists or frames_exist
check(output_created, "video file or frame directory was created")
if video_exists:
    check(
        video_path.stat().st_size > 0,
        f"video file has non-zero size ({video_path.stat().st_size} bytes)",
    )
elif frames_exist:
    n_frames = len(list(frames_dir.glob("*.png")))
    check(n_frames > 0, f"frame directory has {n_frames} PNG files")

# Cleanup
if video_exists:
    video_path.unlink()
    print(f"    (cleaned up {video_path})")
if frames_exist:
    import shutil

    shutil.rmtree(frames_dir)
    print(f"    (cleaned up {frames_dir})")
# Remove empty parent dirs
for d in [video_path.parent, video_path.parent.parent]:
    try:
        d.rmdir()  # only removes if empty
    except OSError:
        pass

# --- extrapolate_density ---
print("\n--- extrapolate_density ---")
R = 16
vol = extrapolate_density(pinn, time=0.5, grid_resolution=R)
check(vol.shape == (R, R, R), f"density volume shape = ({R}, {R}, {R})")
check(torch.isfinite(vol).all().item(), "density volume: no NaN/Inf")
check(vol.std().item() > 1e-6, f"density has spatial variation (std = {vol.std().item():.4e})")


# ===================================================================
# 3. NOVEL VIEW SYNTHESIS TESTS
# ===================================================================
print("\n" + "=" * 70)
print("  3. NOVEL VIEW SYNTHESIS TESTS (novel_view.py)")
print("=" * 70)

intrinsics = {
    "fx": 128.0,
    "fy": 128.0,
    "cx": W / 2.0,
    "cy": H / 2.0,
    "image_height": H,
    "image_width": W,
}
synthesizer = NovelViewSynthesizer(
    gauss_model,
    pinn,
    renderer,
    camera_intrinsics=intrinsics,
    num_advection_steps=3,
)

# --- Camera interpolation ---
print("\n--- Camera interpolation ---")
cam1 = {
    "K": K.clone(),
    "w2c": torch.eye(4),
    "image_height": H,
    "image_width": W,
}
cam1["w2c"][2, 3] = 3.0  # identity rotation, translation [0,0,3]

# 90-degree rotation around y-axis
Ry90 = torch.tensor(
    [
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)
cam2 = {
    "K": K.clone(),
    "w2c": torch.eye(4),
    "image_height": H,
    "image_width": W,
}
cam2["w2c"][:3, :3] = Ry90
cam2["w2c"][:3, 3] = torch.tensor([3.0, 0.0, 0.0])

cams = interpolate_cameras(cam1, cam2, num_steps=5)
check(len(cams) == 5, "5 interpolated cameras")

# First camera ~ cam1
check(torch.allclose(cams[0]["w2c"], cam1["w2c"], atol=1e-4), "first interpolated camera ~ cam1")
# Last camera ~ cam2
check(torch.allclose(cams[-1]["w2c"], cam2["w2c"], atol=1e-4), "last interpolated camera ~ cam2")

# All intermediate cameras: valid rotation matrices (orthogonal, det=+1)
all_valid_rotations = True
for i, c in enumerate(cams):
    R_interp = c["w2c"][:3, :3]
    # Orthogonality: R @ R^T = I
    RtR = R_interp @ R_interp.T
    ortho = torch.allclose(RtR, torch.eye(3), atol=1e-4)
    # Determinant = +1 (proper rotation)
    det = torch.det(R_interp).item()
    det_ok = abs(det - 1.0) < 1e-4
    if not (ortho and det_ok):
        all_valid_rotations = False
        print(f"    cam[{i}]: ortho={ortho}, det={det:.4f}")
check(all_valid_rotations, "all interpolated cameras have valid rotation matrices (ortho, det=+1)")

# --- Orbit render ---
print("\n--- Orbit render ---")
orbit_images = synthesizer.render_orbit(
    center=(0, 0, 0),
    radius=3.0,
    time=0.5,
    num_views=4,
    elevation=30.0,
)
check(len(orbit_images) == 4, "orbit: 4 views")
check(
    all(f["image"].shape == (3, H, W) for f in orbit_images),
    f"orbit: all frames shape (3, {H}, {W})",
)
check(all(torch.isfinite(f["image"]).all() for f in orbit_images), "orbit: all frames finite")

# At least 2 views should be visually different
n_different = 0
for i in range(len(orbit_images)):
    for j in range(i + 1, len(orbit_images)):
        diff = (orbit_images[i]["image"] - orbit_images[j]["image"]).abs().max().item()
        if diff > 1e-3:
            n_different += 1
check(
    n_different >= 2,
    f"orbit: at least 2 view-pairs are visually different ({n_different} pairs differ)",
)

# --- Orbit video ---
print("\n--- Orbit video ---")
orbit_video_path = _ROOT / "outputs" / "renders" / "test_orbit.mp4"
synthesizer.render_orbit_video(
    center=(0, 0, 0),
    radius=3.0,
    time=0.5,
    num_views=8,
    elevation=30.0,
    output_path=orbit_video_path,
    fps=5,
)
orbit_video_exists = orbit_video_path.exists()
orbit_frames_dir = orbit_video_path.parent / (orbit_video_path.stem + "_frames")
orbit_frames_exist = (
    orbit_frames_dir.exists() and any(orbit_frames_dir.iterdir())
    if orbit_frames_dir.exists()
    else False
)

orbit_output_created = orbit_video_exists or orbit_frames_exist
check(orbit_output_created, "orbit video file or frame directory was created")
if orbit_video_exists:
    check(
        orbit_video_path.stat().st_size > 0,
        f"orbit video has non-zero size ({orbit_video_path.stat().st_size} bytes)",
    )
elif orbit_frames_exist:
    n_orbit_frames = len(list(orbit_frames_dir.glob("*.png")))
    check(n_orbit_frames > 0, f"orbit frame directory has {n_orbit_frames} PNG files")

# Cleanup
if orbit_video_exists:
    orbit_video_path.unlink()
    print(f"    (cleaned up {orbit_video_path})")
if orbit_frames_exist:
    import shutil

    shutil.rmtree(orbit_frames_dir)
    print(f"    (cleaned up {orbit_frames_dir})")
for d in [orbit_video_path.parent, orbit_video_path.parent.parent]:
    try:
        d.rmdir()
    except OSError:
        pass


# ===================================================================
# Summary
# ===================================================================
print(f"\n{'=' * 70}")
print(f"  Results: {checks_passed}/{checks_total} checks passed")
print(f"{'=' * 70}")

if checks_passed == checks_total:
    print(
        "\nTEST PASSED: integration/ — advection (Euler+RK4+trajectory), "
        "forward prediction (frame+sequence+video), novel view synthesis "
        "(interpolation+orbit+video) all verified"
    )
else:
    print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
    sys.exit(1)
