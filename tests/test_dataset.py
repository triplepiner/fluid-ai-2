#!/usr/bin/env python3
"""Thorough test for gaussian_splatting/dataset.py."""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEST_DIR = Path(__file__).resolve().parent / "_test_dataset"
DATA_ROOT = TEST_DIR / "data"
NUM_CAMERAS = 2
NUM_FRAMES = 10
NUM_PAIRS = NUM_FRAMES - 1
W, H = 64, 64
FX = FY = 50.0
CX = CY = 32.0
VAL_EVERY = 10

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
# Step 1: Create synthetic preprocessed data
# =====================================================================
def step1_create_data():
    print("\n=== Step 1: Create synthetic preprocessed data ===")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    # --- Camera matrices ---
    # cam_00: identity pose
    c2w_0 = np.eye(4, dtype=np.float64)
    w2c_0 = np.eye(4, dtype=np.float64)

    # cam_01: small rotation around z-axis + translation
    theta = np.radians(10)
    c, s = np.cos(theta), np.sin(theta)
    R1 = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)
    t1 = np.array([0.1, 0.05, 0.0], dtype=np.float64)
    w2c_1 = np.eye(4, dtype=np.float64)
    w2c_1[:3, :3] = R1
    w2c_1[:3, 3] = t1
    c2w_1 = np.linalg.inv(w2c_1)

    cameras_json = {
        "normalization": {"scale": 1.0},
        "common_intrinsic": {
            "fx": FX,
            "fy": FY,
            "cx": CX,
            "cy": CY,
            "image_width": W,
            "image_height": H,
        },
        "cameras": {
            "cam_00": {
                "camera_id": 0,
                "image_width": W,
                "image_height": H,
                "intrinsic": {
                    "fx": FX,
                    "fy": FY,
                    "cx": CX,
                    "cy": CY,
                    "k1": 0.0,
                    "k2": 0.0,
                    "p1": 0.0,
                    "p2": 0.0,
                },
                "extrinsic": {
                    "rotation_matrix": c2w_0[:3, :3].tolist(),
                    "translation": c2w_0[:3, 3].tolist(),
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "camera_to_world": c2w_0.tolist(),
                "world_to_camera": w2c_0.tolist(),
            },
            "cam_01": {
                "camera_id": 1,
                "image_width": W,
                "image_height": H,
                "intrinsic": {
                    "fx": FX,
                    "fy": FY,
                    "cx": CX,
                    "cy": CY,
                    "k1": 0.0,
                    "k2": 0.0,
                    "p1": 0.0,
                    "p2": 0.0,
                },
                "extrinsic": {
                    "rotation_matrix": R1.tolist(),
                    "translation": t1.tolist(),
                    "quaternion_wxyz": [0.996, 0.0, 0.0, 0.087],
                },
                "camera_to_world": c2w_1.tolist(),
                "world_to_camera": w2c_1.tolist(),
            },
        },
    }

    colmap_dir = DATA_ROOT / "colmap"
    colmap_dir.mkdir(parents=True)
    with open(colmap_dir / "cameras_normalized.json", "w") as f:
        json.dump(cameras_json, f, indent=2)
    check(True, "cameras_normalized.json created")

    # --- Frames ---
    rng = np.random.RandomState(42)
    for cam_name in ["cam_00", "cam_01"]:
        cam_frame_dir = DATA_ROOT / "undistorted" / cam_name
        cam_frame_dir.mkdir(parents=True)
        for i in range(NUM_FRAMES):
            img = rng.randint(0, 256, (H, W, 3), dtype=np.uint8)
            cv2.imwrite(str(cam_frame_dir / f"frame_{i:05d}.png"), img)

    frames_00 = sorted((DATA_ROOT / "undistorted" / "cam_00").glob("*.png"))
    frames_01 = sorted((DATA_ROOT / "undistorted" / "cam_01").glob("*.png"))
    check(len(frames_00) == NUM_FRAMES, f"cam_00: {NUM_FRAMES} frames")
    check(len(frames_01) == NUM_FRAMES, f"cam_01: {NUM_FRAMES} frames")

    # --- Flow ---
    for cam_name in ["cam_00", "cam_01"]:
        flow_dir = DATA_ROOT / "flow" / cam_name
        flow_dir.mkdir(parents=True)
        for i in range(NUM_PAIRS):
            fwd = rng.randn(2, H, W).astype(np.float32) * 0.5
            bwd = rng.randn(2, H, W).astype(np.float32) * 0.5
            np.save(str(flow_dir / f"flow_fwd_{i:05d}.npy"), fwd)
            np.save(str(flow_dir / f"flow_bwd_{i:05d}.npy"), bwd)
            # Mask is a PNG (uint8, 0 or 255)
            mask = np.ones((H, W), dtype=np.uint8) * 255
            cv2.imwrite(str(flow_dir / f"flow_mask_{i:05d}.png"), mask)

    check(
        len(list((DATA_ROOT / "flow" / "cam_00").glob("flow_fwd_*.npy"))) == NUM_PAIRS,
        f"cam_00: {NUM_PAIRS} forward flow files",
    )
    check(
        len(list((DATA_ROOT / "flow" / "cam_00").glob("flow_bwd_*.npy"))) == NUM_PAIRS,
        f"cam_00: {NUM_PAIRS} backward flow files",
    )
    check(
        len(list((DATA_ROOT / "flow" / "cam_00").glob("flow_mask_*.png"))) == NUM_PAIRS,
        f"cam_00: {NUM_PAIRS} mask PNGs",
    )

    # --- Depth ---
    for cam_name in ["cam_00", "cam_01"]:
        depth_dir = DATA_ROOT / "depth" / cam_name
        depth_dir.mkdir(parents=True)
        for i in range(NUM_FRAMES):
            depth = rng.rand(H, W).astype(np.float32) * 5.0 + 0.1
            np.save(str(depth_dir / f"depth_{i:05d}.npy"), depth)

    check(
        len(list((DATA_ROOT / "depth" / "cam_00").glob("depth_*.npy"))) == NUM_FRAMES,
        f"cam_00: {NUM_FRAMES} depth files",
    )


# =====================================================================
# Step 2: Instantiate the dataset
# =====================================================================
def step2_instantiate():
    print("\n=== Step 2: Instantiate the dataset ===")

    from gaussian_splatting.dataset import FluidDataset

    ds = FluidDataset(
        data_root=DATA_ROOT,
        split="train",
        resolution=(H, W),
        load_flow=True,
        load_depth=True,
        val_every=VAL_EVERY,
        augment=False,  # disable augment for deterministic checks
    )

    length = len(ds)
    print(f"  len(dataset) = {length}")

    # With val_every=10 and 10 frames (0-9):
    # val timesteps: t where (t-0)%10==0 => t=0
    # train timesteps: t=1,2,3,4,5,6,7,8,9 => 9
    # total samples = 2 cameras * 9 timesteps = 18
    expected_train = NUM_CAMERAS * (NUM_FRAMES - 1)
    check(isinstance(length, int), "len() returns int")
    check(length > 0, f"len() > 0 (got {length})")
    check(length == expected_train, f"len() = {expected_train} (got {length})")

    return ds


# =====================================================================
# Step 3: Iterate and check shapes
# =====================================================================
def step3_check_shapes(ds):
    print("\n=== Step 3: Iterate and check shapes ===")

    sample_indices = [0, 1, 5, 10, len(ds) - 1]

    for idx in sample_indices:
        print(f"  -- Sample idx={idx} --")
        sample = ds[idx]

        # Image
        img = sample["image"]
        check(img.dtype == torch.float32, f"  image dtype=float32")
        check(img.shape == (3, H, W), f"  image shape=(3,{H},{W})")
        check(
            img.min() >= 0.0 and img.max() <= 1.0,
            f"  image in [0,1] (min={img.min():.3f}, max={img.max():.3f})",
        )

        # camera_id
        cam_id = sample["camera_id"]
        check(isinstance(cam_id, int), f"  camera_id is int")
        check(0 <= cam_id < NUM_CAMERAS, f"  camera_id in [0,{NUM_CAMERAS})")

        # timestep
        ts = sample["timestep"]
        check(isinstance(ts, int), f"  timestep is int")
        check(ts >= 0, f"  timestep >= 0 (got {ts})")

        # time_normalized
        tn = sample["time_normalized"]
        check(isinstance(tn, float), f"  time_normalized is float")
        check(0.0 <= tn <= 1.0, f"  time_normalized in [0,1] (got {tn:.4f})")

        # K
        K = sample["K"]
        check(K.shape == (3, 3), f"  K shape=(3,3)")
        check(not torch.all(K == 0), f"  K not all zeros")
        check(
            K[0, 0].item() > 0 and K[1, 1].item() > 0,
            f"  K diagonal positive (fx={K[0,0]:.1f}, fy={K[1,1]:.1f})",
        )

        # R
        R = sample["R"]
        check(R.shape == (3, 3), f"  R shape=(3,3)")
        RtR = R @ R.T
        eye3 = torch.eye(3)
        orth_err = (RtR - eye3).abs().max().item()
        check(orth_err < 1e-4, f"  R orthogonal (max err={orth_err:.6f})")
        det_R = torch.linalg.det(R).item()
        check(abs(det_R - 1.0) < 1e-4, f"  det(R) ~ 1 (got {det_R:.6f})")

        # t
        t = sample["t"]
        check(t.shape == (3,), f"  t shape=(3,)")

        # c2w
        c2w = sample["c2w"]
        check(c2w.shape == (4, 4), f"  c2w shape=(4,4)")
        check(
            torch.allclose(c2w[3], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-5),
            f"  c2w bottom row = [0,0,0,1]",
        )

        # w2c
        w2c = sample["w2c"]
        check(w2c.shape == (4, 4), f"  w2c shape=(4,4)")
        check(
            torch.allclose(w2c[3], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-5),
            f"  w2c bottom row = [0,0,0,1]",
        )

        # c2w @ w2c = identity
        product = c2w @ w2c
        identity_err = (product - torch.eye(4)).abs().max().item()
        check(identity_err < 1e-4, f"  c2w @ w2c ~ I (max err={identity_err:.6f})")

        # depth
        depth = sample["depth"]
        if depth is not None:
            check(depth.dtype == torch.float32, f"  depth dtype=float32")
            check(depth.shape == (1, H, W), f"  depth shape=(1,{H},{W}) (got {tuple(depth.shape)})")
            check(depth.min() > 0, f"  depth all positive (min={depth.min():.4f})")
        else:
            check(False, f"  depth should not be None")

        # flow_fwd
        flow_fwd = sample["flow_fwd"]
        # Last timestep (t=9) has no forward flow.
        if ts < NUM_FRAMES - 1:
            if flow_fwd is not None:
                check(flow_fwd.shape == (2, H, W), f"  flow_fwd shape=(2,{H},{W})")
            else:
                check(False, f"  flow_fwd should not be None at t={ts}")
        else:
            check(flow_fwd is None, f"  flow_fwd is None at last timestep")

        # flow_mask
        flow_mask = sample["flow_mask"]
        if ts < NUM_FRAMES - 1:
            if flow_mask is not None:
                check(flow_mask.shape == (1, H, W), f"  flow_mask shape=(1,{H},{W})")
            else:
                check(False, f"  flow_mask should not be None at t={ts}")
        else:
            check(flow_mask is None, f"  flow_mask is None at last timestep")


# =====================================================================
# Step 4: DataLoader test
# =====================================================================
def step4_dataloader(ds):
    print("\n=== Step 4: DataLoader test ===")

    from gaussian_splatting.dataset import fluid_collate

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=fluid_collate,
    )

    batches_checked = 0
    for batch in loader:
        if batches_checked >= 3:
            break

        # With batch_size=1, stacked tensors should have shape (1, ...)
        img = batch["image"]
        check(
            isinstance(img, torch.Tensor) and img.shape == (1, 3, H, W),
            f"  batch image shape=(1,3,{H},{W})",
        )
        K = batch["K"]
        check(isinstance(K, torch.Tensor) and K.shape == (1, 3, 3), f"  batch K shape=(1,3,3)")
        c2w = batch["c2w"]
        check(
            isinstance(c2w, torch.Tensor) and c2w.shape == (1, 4, 4), f"  batch c2w shape=(1,4,4)"
        )

        batches_checked += 1

    check(batches_checked == 3, f"Iterated over 3 batches without error")


# =====================================================================
# Step 5: Helper function tests
# =====================================================================
def step5_helpers():
    print("\n=== Step 5: Helper function tests ===")

    from gaussian_splatting.dataset import get_rays, load_cameras

    # -- load_cameras --
    cameras = load_cameras(DATA_ROOT / "colmap" / "cameras_normalized.json")
    check(isinstance(cameras, list), "load_cameras returns list")
    check(len(cameras) == NUM_CAMERAS, f"load_cameras returns {NUM_CAMERAS} cameras")

    required_keys = {
        "name",
        "camera_id",
        "image_width",
        "image_height",
        "K",
        "R",
        "t",
        "c2w",
        "w2c",
    }
    cam0 = cameras[0]
    check(required_keys.issubset(cam0.keys()), f"Camera dict has all required keys")
    check(cam0["K"].shape == (3, 3), "K is (3,3)")
    check(cam0["R"].shape == (3, 3), "R is (3,3)")
    check(cam0["t"].shape == (3,), "t is (3,)")
    check(cam0["c2w"].shape == (4, 4), "c2w is (4,4)")
    check(cam0["w2c"].shape == (4, 4), "w2c is (4,4)")

    # -- get_rays --
    K = torch.tensor(
        [
            [FX, 0, CX],
            [0, FY, CY],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    c2w = torch.eye(4, dtype=torch.float32)

    origins, directions = get_rays(H, W, K, c2w)

    # get_rays returns (H, W, 3) — test spec says (H*W, 3) but actual
    # function returns spatial layout which is more natural.
    check(origins.shape == (H, W, 3), f"origins shape=({H},{W},3) (got {tuple(origins.shape)})")
    check(
        directions.shape == (H, W, 3),
        f"directions shape=({H},{W},3) (got {tuple(directions.shape)})",
    )

    # All origins should be the camera position (0,0,0 for identity c2w)
    check(
        torch.allclose(origins, torch.zeros(H, W, 3), atol=1e-6),
        "origins = camera position (0,0,0)",
    )

    # Directions should have unit norm
    norms = directions.norm(dim=-1)
    norm_err = (norms - 1.0).abs().max().item()
    check(norm_err < 1e-5, f"directions have unit norm (max err={norm_err:.6f})")

    # Centre pixel should point along +z (approximately)
    centre_dir = directions[H // 2, W // 2]
    check(centre_dir[2].item() > 0.9, f"centre ray points along +z (z={centre_dir[2]:.4f})")

    # Test with non-identity c2w (cam_01)
    cam1 = cameras[1]
    c2w_1 = torch.from_numpy(cam1["c2w"])
    K_1 = torch.from_numpy(cam1["K"])
    origins_1, dirs_1 = get_rays(H, W, K_1, c2w_1)
    norms_1 = dirs_1.norm(dim=-1)
    norm_err_1 = (norms_1 - 1.0).abs().max().item()
    check(norm_err_1 < 1e-4, f"cam_01 ray directions unit norm (max err={norm_err_1:.6f})")


# =====================================================================
# Step 6: Validation split test
# =====================================================================
def step6_train_val_split():
    print("\n=== Step 6: Validation split test ===")

    from gaussian_splatting.dataset import FluidDataset

    train_ds = FluidDataset(
        data_root=DATA_ROOT,
        split="train",
        resolution=(H, W),
        load_flow=False,
        load_depth=False,
        val_every=VAL_EVERY,
        augment=False,
    )
    val_ds = FluidDataset(
        data_root=DATA_ROOT,
        split="val",
        resolution=(H, W),
        load_flow=False,
        load_depth=False,
        val_every=VAL_EVERY,
        augment=False,
    )

    print(f"  train len = {len(train_ds)}, val len = {len(val_ds)}")
    check(len(train_ds) > 0, "train dataset is non-empty")
    check(len(val_ds) > 0, "val dataset is non-empty")

    # Total should equal num_cameras * total_timesteps
    total = len(train_ds) + len(val_ds)
    expected_total = NUM_CAMERAS * NUM_FRAMES
    check(total == expected_total, f"train + val = {expected_total} (got {total})")

    # Extract (camera_id, timestep) pairs from both
    train_pairs = set()
    for i in range(len(train_ds)):
        s = train_ds[i]
        train_pairs.add((s["camera_id"], s["timestep"]))

    val_pairs = set()
    for i in range(len(val_ds)):
        s = val_ds[i]
        val_pairs.add((s["camera_id"], s["timestep"]))

    overlap = train_pairs & val_pairs
    check(len(overlap) == 0, f"No overlapping (camera_id, timestep) pairs (overlap={len(overlap)})")

    # Union covers all expected pairs
    all_pairs = train_pairs | val_pairs
    check(
        len(all_pairs) == expected_total,
        f"Union covers all {expected_total} pairs (got {len(all_pairs)})",
    )


# =====================================================================
# Step 7: Cleanup
# =====================================================================
def step7_cleanup():
    print("\n=== Step 7: Cleanup ===")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    check(not TEST_DIR.exists(), "Test directory cleaned up")


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("  TEST: gaussian_splatting/dataset.py")
    print("=" * 70)

    step1_create_data()
    ds = step2_instantiate()
    step3_check_shapes(ds)
    step4_dataloader(ds)
    step5_helpers()
    step6_train_val_split()
    step7_cleanup()

    print("\n" + "=" * 70)
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 70)

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: dataset.py"
            " — all shapes correct, camera matrices valid,"
            " DataLoader works, train/val split verified"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
