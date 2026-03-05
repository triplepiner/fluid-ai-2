#!/usr/bin/env python3
"""Thorough test for preprocessing/normalize_intrinsics.py."""

import json
import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.normalize_intrinsics import (
    build_remap_tables,
    camera_matrix,
    compute_common_intrinsic,
    distortion_vector,
    normalize_intrinsics,
)

TEST_DIR = Path(__file__).resolve().parent / "_test_normalize_intrinsics"
checks_passed = 0
checks_total = 0


def check(condition: bool, label: str):
    global checks_passed, checks_total
    checks_total += 1
    if condition:
        checks_passed += 1
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}")


def make_grid_image(w: int, h: int, spacing: int = 20) -> np.ndarray:
    """Create a white image with a black grid pattern."""
    img = np.ones((h, w, 3), dtype=np.uint8) * 255
    for x in range(0, w, spacing):
        cv2.line(img, (x, 0), (x, h), (0, 0, 0), 1)
    for y in range(0, h, spacing):
        cv2.line(img, (0, y), (w, y), (0, 0, 0), 1)
    # Add corner markers so edge distortion is visible.
    cv2.circle(img, (30, 30), 10, (0, 0, 255), 2)
    cv2.circle(img, (w - 30, 30), 10, (0, 255, 0), 2)
    cv2.circle(img, (30, h - 30), 10, (255, 0, 0), 2)
    cv2.circle(img, (w - 30, h - 30), 10, (0, 255, 255), 2)
    return img


def build_cameras_json(cameras_dict, path, normalization=None):
    """Write a cameras.json file."""
    data = {
        "normalization": normalization or {"scale": 1.0, "translation": [0, 0, 0]},
        "cameras": cameras_dict,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def make_camera_entry(cam_id, fx, fy, cx, cy, k1, k2, p1, p2, w, h):
    """Build a camera entry matching run_colmap.py output format."""
    # Simple identity extrinsic (camera at origin looking down -Z).
    R = np.eye(3)
    t = np.array([0.0, 0.0, 0.0])

    # Add slight variation per camera so we can verify extrinsics are preserved.
    angle = float(cam_id) * 0.3
    c, s = math.cos(angle), math.sin(angle)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    t = np.array([float(cam_id) * 0.5, 0.1, 2.0])

    w2c = np.eye(4)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    c2w = np.linalg.inv(w2c)

    return {
        "camera_id": f"cam_{cam_id:02d}",
        "image_width": w,
        "image_height": h,
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
            "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],  # placeholder
            "translation": t.tolist(),
        },
        "camera_to_world": c2w.tolist(),
        "world_to_camera": w2c.tolist(),
    }


# =========================================================================
# Step 1: Create synthetic test data
# =========================================================================


def step1_create_data():
    print("\n=== Step 1: Create synthetic test data ===")

    # Clean up any previous run.
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    stab_dir = TEST_DIR / "stabilized"
    W, H = 320, 240
    NUM_FRAMES = 10

    # Camera 0: barrel distortion (k1=0.1, k2=0.01)
    cam0 = make_camera_entry(0, 500, 500, 160, 120, 0.1, 0.01, 0, 0, W, H)
    # Camera 1: pincushion distortion (k1=-0.05), different zoom (f=600)
    cam1 = make_camera_entry(1, 600, 600, 160, 120, -0.05, 0, 0, 0, W, H)

    cameras = {"cam_00": cam0, "cam_01": cam1}
    cameras_path = build_cameras_json(cameras, TEST_DIR / "cameras.json")

    # Generate grid frames.
    for cam_name in ["cam_00", "cam_01"]:
        cam_dir = stab_dir / cam_name
        cam_dir.mkdir(parents=True, exist_ok=True)
        for i in range(NUM_FRAMES):
            img = make_grid_image(W, H, spacing=20)
            # Vary brightness slightly per frame so they're not identical.
            img = np.clip(img.astype(np.int16) + i, 0, 255).astype(np.uint8)
            cv2.imwrite(str(cam_dir / f"frame_{i:05d}.png"), img)

    n0 = len(list((stab_dir / "cam_00").glob("*.png")))
    n1 = len(list((stab_dir / "cam_01").glob("*.png")))
    check(n0 == NUM_FRAMES, f"cam_00 has {NUM_FRAMES} frames (got {n0})")
    check(n1 == NUM_FRAMES, f"cam_01 has {NUM_FRAMES} frames (got {n1})")
    check(cameras_path.exists(), "cameras.json created")

    return cameras_path, stab_dir, cameras


# =========================================================================
# Step 2: Run the script
# =========================================================================


def step2_run(cameras_path, stab_dir):
    print("\n=== Step 2: Run normalize_intrinsics ===")

    output_dir = TEST_DIR / "undistorted"
    output_cameras = TEST_DIR / "cameras_normalized.json"

    result = normalize_intrinsics(
        input_dir=stab_dir,
        cameras_json=cameras_path,
        output_dir=output_dir,
        output_cameras=output_cameras,
        max_workers=2,
    )

    check(result["num_cameras"] == 2, f"Processed 2 cameras (got {result['num_cameras']})")
    check(output_cameras.exists(), "cameras_normalized.json created")
    check(output_dir.exists(), "Output directory created")

    per_cam = {r["camera"]: r for r in result["per_camera"]}
    for cam in ["cam_00", "cam_01"]:
        check(per_cam[cam]["status"] == "ok", f"{cam} processed OK")
        check(per_cam[cam]["num_failed"] == 0, f"{cam} zero failed frames")

    return output_dir, output_cameras, result


# =========================================================================
# Step 3: Verify outputs
# =========================================================================


def step3_verify(output_dir, output_cameras, original_cameras):
    print("\n=== Step 3: Verify outputs ===")

    with open(output_cameras) as f:
        norm_data = json.load(f)

    norm_cameras = norm_data["cameras"]

    # 3a: Both cameras have IDENTICAL intrinsics.
    intr0 = norm_cameras["cam_00"]["intrinsic"]
    intr1 = norm_cameras["cam_01"]["intrinsic"]

    check(intr0["fx"] == intr1["fx"], f"fx matches: {intr0['fx']} == {intr1['fx']}")
    check(intr0["fy"] == intr1["fy"], f"fy matches: {intr0['fy']} == {intr1['fy']}")
    check(intr0["cx"] == intr1["cx"], f"cx matches: {intr0['cx']} == {intr1['cx']}")
    check(intr0["cy"] == intr1["cy"], f"cy matches: {intr0['cy']} == {intr1['cy']}")

    # Verify focal length is the mean: geometric mean per cam then arithmetic mean.
    expected_f = (math.sqrt(500 * 500) + math.sqrt(600 * 600)) / 2.0  # = 550
    check(
        math.isclose(intr0["fx"], expected_f, rel_tol=1e-6),
        f"Target focal length = {expected_f} (got {intr0['fx']})",
    )

    # 3b: All distortion coefficients are zero.
    for cam_name in ["cam_00", "cam_01"]:
        intr = norm_cameras[cam_name]["intrinsic"]
        check(intr["k1"] == 0.0, f"{cam_name} k1 == 0")
        check(intr["k2"] == 0.0, f"{cam_name} k2 == 0")
        check(intr["p1"] == 0.0, f"{cam_name} p1 == 0")
        check(intr["p2"] == 0.0, f"{cam_name} p2 == 0")

    # 3c: Extrinsics UNCHANGED.
    for cam_name in ["cam_00", "cam_01"]:
        orig = original_cameras[cam_name]
        norm = norm_cameras[cam_name]

        orig_ext = orig["extrinsic"]
        norm_ext = norm["extrinsic"]
        check(
            orig_ext["rotation_matrix"] == norm_ext["rotation_matrix"],
            f"{cam_name} rotation_matrix preserved",
        )
        check(
            orig_ext["translation"] == norm_ext["translation"],
            f"{cam_name} translation preserved",
        )

        orig_c2w = np.array(orig["camera_to_world"])
        norm_c2w = np.array(norm["camera_to_world"])
        check(
            np.allclose(orig_c2w, norm_c2w, atol=1e-10),
            f"{cam_name} camera_to_world preserved",
        )

        orig_w2c = np.array(orig["world_to_camera"])
        norm_w2c = np.array(norm["world_to_camera"])
        check(
            np.allclose(orig_w2c, norm_w2c, atol=1e-10),
            f"{cam_name} world_to_camera preserved",
        )

    # 3d: Verify undistorted frames exist.
    for cam_name in ["cam_00", "cam_01"]:
        cam_out = output_dir / cam_name
        frames = sorted(cam_out.glob("*.png"))
        check(len(frames) == 10, f"{cam_name} has 10 undistorted frames (got {len(frames)})")

        # Load first frame and verify it's valid.
        img = cv2.imread(str(frames[0]))
        check(img is not None, f"{cam_name} first frame is readable")
        check(
            img.shape[0] > 0 and img.shape[1] > 0,
            f"{cam_name} first frame has valid shape {img.shape}",
        )

        # Verify it's not all black.
        mean_val = img.mean()
        check(mean_val > 10, f"{cam_name} first frame not all black (mean={mean_val:.1f})")

    # 3e: Undistorted differs from original (distortion was corrected).
    stab_dir = TEST_DIR / "stabilized"
    for cam_name in ["cam_00", "cam_01"]:
        orig_path = stab_dir / cam_name / "frame_00000.png"
        und_path = output_dir / cam_name / "frame_00000.png"
        orig_img = cv2.imread(str(orig_path))
        und_img = cv2.imread(str(und_path))

        # Resize original if shapes differ.
        if orig_img.shape != und_img.shape:
            orig_img = cv2.resize(orig_img, (und_img.shape[1], und_img.shape[0]))

        diff = np.abs(orig_img.astype(float) - und_img.astype(float))
        mean_diff = diff.mean()
        check(
            mean_diff > 1.0,
            f"{cam_name} frames differ after undistortion (mean_diff={mean_diff:.2f})",
        )

    # 3f: cam_00 (k1=0.1 barrel) should have strong edge distortion correction.
    orig_path = stab_dir / "cam_00" / "frame_00000.png"
    und_path = output_dir / "cam_00" / "frame_00000.png"
    orig_img = cv2.imread(str(orig_path)).astype(float)
    und_img = cv2.imread(str(und_path)).astype(float)
    if orig_img.shape != und_img.shape:
        orig_img = cv2.resize(orig_img, (und_img.shape[1], und_img.shape[0]))

    diff_map = np.sqrt(((orig_img - und_img) ** 2).sum(axis=2))
    # Edge region: outer 25% border.
    h, w = diff_map.shape
    border = min(h, w) // 4
    edge_mask = np.zeros_like(diff_map, dtype=bool)
    edge_mask[:border, :] = True
    edge_mask[-border:, :] = True
    edge_mask[:, :border] = True
    edge_mask[:, -border:] = True
    center_mask = ~edge_mask

    edge_diff = diff_map[edge_mask].mean()
    center_diff = diff_map[center_mask].mean()
    check(
        edge_diff > center_diff,
        f"cam_00 edge correction > center ({edge_diff:.2f} > {center_diff:.2f})",
    )

    # common_intrinsic in JSON.
    check("common_intrinsic" in norm_data, "common_intrinsic section in output JSON")
    ci = norm_data["common_intrinsic"]
    check(ci["image_width"] == 320, f"common_intrinsic width=320 (got {ci['image_width']})")
    check(ci["image_height"] == 240, f"common_intrinsic height=240 (got {ci['image_height']})")

    # Visualisation file.
    vis_path = output_dir / "undistortion_comparison.png"
    check(vis_path.exists(), "undistortion_comparison.png created")


# =========================================================================
# Step 4: Identity test
# =========================================================================


def step4_identity_test():
    print("\n=== Step 4: Identity test (zero distortion, matching focal length) ===")

    # The common intrinsic will be f=550 (from step 2).  Create a camera
    # that already has f=550, zero distortion, same resolution → output
    # should be nearly identical to input.
    identity_dir = TEST_DIR / "identity"
    stab_dir = identity_dir / "stabilized"
    W, H = 320, 240

    cam = make_camera_entry(0, 550, 550, 160, 120, 0, 0, 0, 0, W, H)
    cameras = {"cam_00": cam}
    cameras_path = build_cameras_json(cameras, identity_dir / "cameras.json")

    cam_dir = stab_dir / "cam_00"
    cam_dir.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        img = make_grid_image(W, H, spacing=20)
        cv2.imwrite(str(cam_dir / f"frame_{i:05d}.png"), img)

    output_dir = identity_dir / "undistorted"
    output_cameras = identity_dir / "cameras_normalized.json"

    result = normalize_intrinsics(
        input_dir=stab_dir,
        cameras_json=cameras_path,
        output_dir=output_dir,
        output_cameras=output_cameras,
        max_workers=2,
    )

    check(result["num_cameras"] == 1, "Identity: 1 camera processed")
    per_cam = result["per_camera"][0]
    check(per_cam["status"] == "ok", "Identity: camera processed OK")

    # Compare frames: should be nearly identical.
    diffs = []
    for i in range(5):
        orig = cv2.imread(str(stab_dir / "cam_00" / f"frame_{i:05d}.png")).astype(float)
        und = cv2.imread(str(output_dir / "cam_00" / f"frame_{i:05d}.png")).astype(float)
        if orig.shape != und.shape:
            orig = cv2.resize(orig, (und.shape[1], und.shape[0]))
        diffs.append(np.abs(orig - und).mean())

    mean_diff = sum(diffs) / len(diffs)
    check(
        mean_diff < 1.0,
        f"Identity: mean pixel diff < 1.0 (got {mean_diff:.4f})",
    )

    # Max distortion correction should be very small.
    max_disp = per_cam.get("max_distortion_correction_px", 999)
    check(
        max_disp < 1.0,
        f"Identity: max distortion correction < 1.0 px (got {max_disp:.4f})",
    )


# =========================================================================
# Step 5: Cleanup
# =========================================================================


def step5_cleanup():
    print("\n=== Step 5: Cleanup ===")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
        print(f"  Removed {TEST_DIR}")
    check(not TEST_DIR.exists(), "Test directory cleaned up")


# =========================================================================
# Main
# =========================================================================


def main():
    print("=" * 65)
    print("  TEST: preprocessing/normalize_intrinsics.py")
    print("=" * 65)

    cameras_path, stab_dir, original_cameras = step1_create_data()
    output_dir, output_cameras, result = step2_run(cameras_path, stab_dir)
    step3_verify(output_dir, output_cameras, original_cameras)
    step4_identity_test()
    step5_cleanup()

    print("\n" + "=" * 65)
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 65)

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: normalize_intrinsics.py"
            " — intrinsics unified, distortion corrected, extrinsics preserved"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
