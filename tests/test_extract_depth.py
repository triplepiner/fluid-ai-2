#!/usr/bin/env python3
"""Thorough test for preprocessing/extract_depth.py."""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEST_DIR = Path(__file__).resolve().parent / "_test_extract_depth"
INPUT_DIR = TEST_DIR / "undistorted"
OUTPUT_DIR = TEST_DIR / "depth"
CAM = "cam_00"
W, H = 320, 240
NUM_FRAMES = 5

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
# Step 1: Create synthetic test data
# =====================================================================
def step1_create_data():
    print("\n=== Step 1: Create synthetic test data ===")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)

    cam_dir = INPUT_DIR / CAM
    cam_dir.mkdir(parents=True)

    for i in range(NUM_FRAMES):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        # Left half: bright nearby object (white rectangle)
        img[:, : W // 2] = 220
        # Right half: dark far background
        img[:, W // 2 :] = 40
        # Add a slight per-frame variation so frames aren't byte-identical
        img = np.clip(img.astype(np.int16) + i, 0, 255).astype(np.uint8)
        cv2.imwrite(str(cam_dir / f"frame_{i:05d}.png"), img)

    frames = sorted(cam_dir.glob("*.png"))
    check(len(frames) == NUM_FRAMES, f"Created {NUM_FRAMES} frames (got {len(frames)})")
    print(f"  Left half: bright (220), Right half: dark (40)")


# =====================================================================
# Step 2: Run the script
# =====================================================================
def step2_run():
    print("\n=== Step 2: Run extract_depth ===")

    from preprocessing.extract_depth import extract_depth

    result = extract_depth(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        model_size="small",
        device_str="cpu",
        batch_size=1,
    )

    check(
        result.get("total_frames") == NUM_FRAMES,
        f"Total frames = {NUM_FRAMES} (got {result.get('total_frames')})",
    )
    check(
        result.get("model_size") == "small", f"Model size = small (got {result.get('model_size')})"
    )

    cam_report = result["per_camera"][0]
    check(cam_report["status"] == "ok", f"{CAM} status = ok")
    check(
        cam_report["num_frames"] == NUM_FRAMES,
        f"{CAM} num_frames = {NUM_FRAMES} (got {cam_report['num_frames']})",
    )
    check(cam_report["depth_min"] > 0, f"depth_min > 0 (got {cam_report['depth_min']})")
    check(
        cam_report["depth_max"] > cam_report["depth_min"],
        f"depth_max > depth_min ({cam_report['depth_max']} > {cam_report['depth_min']})",
    )

    return result


# =====================================================================
# Step 3: Verify outputs
# =====================================================================
def step3_verify():
    print("\n=== Step 3: Verify outputs ===")
    cam_out = OUTPUT_DIR / CAM

    # --- 3a: File existence ---
    print("  -- File existence --")
    for i in range(NUM_FRAMES):
        npy = cam_out / f"depth_{i:05d}.npy"
        vis = cam_out / f"depth_vis_{i:05d}.png"
        check(npy.exists(), f"depth_{i:05d}.npy exists")
        check(vis.exists(), f"depth_vis_{i:05d}.png exists")

    # --- 3b: Depth shape and dtype ---
    print("  -- Depth array properties --")
    depth = np.load(str(cam_out / "depth_00000.npy"))
    check(depth.ndim == 2, f"Depth is 2-D (got ndim={depth.ndim})")
    check(depth.shape == (H, W), f"Depth shape = ({H}, {W}) (got {depth.shape})")
    check(depth.dtype == np.float32, f"Depth dtype = float32 (got {depth.dtype})")

    # --- 3c: Value sanity ---
    print("  -- Value sanity --")
    check(not np.any(np.isnan(depth)), "No NaN values")
    check(not np.any(np.isinf(depth)), "No Inf values")
    check(np.all(depth > 0), f"All depth values positive (min={depth.min():.6f})")

    unique_vals = np.unique(depth)
    check(len(unique_vals) > 10, f"Depth is not all identical ({len(unique_vals)} unique values)")

    # --- 3d: Median of frame 0 should be ~1.0 (normalised) ---
    median_f0 = float(np.median(depth))
    check(abs(median_f0 - 1.0) < 0.01, f"Median of frame 0 = {median_f0:.6f} (expected ~1.0)")

    # --- 3e: Visualisation validity ---
    print("  -- Visualisation PNGs --")
    vis = cv2.imread(str(cam_out / "depth_vis_00000.png"))
    check(vis is not None, "depth_vis_00000.png is readable")
    check(vis.shape == (H, W, 3), f"Vis shape = ({H}, {W}, 3) (got {vis.shape})")
    check(vis.mean() > 5, f"Vis not all black (mean={vis.mean():.1f})")
    check(vis.mean() < 250, f"Vis not all white (mean={vis.mean():.1f})")

    # --- 3f: Manifest ---
    print("  -- Manifest --")
    manifest_path = OUTPUT_DIR / "depth_manifest.json"
    check(manifest_path.exists(), "depth_manifest.json exists")

    with open(manifest_path) as f:
        manifest = json.load(f)

    check(manifest["total_frames"] == NUM_FRAMES, f"Manifest total_frames = {NUM_FRAMES}")
    check(
        "depth-anything" in manifest["model"].lower(), f"Manifest model contains 'depth-anything'"
    )
    check("per_camera" in manifest, "Manifest has per_camera section")

    cam_entry = manifest["per_camera"][0]
    check(cam_entry["camera"] == CAM, f"Manifest camera = {CAM}")
    for key in ["depth_min", "depth_max", "depth_median", "scale_factor", "num_frames"]:
        check(key in cam_entry, f"Manifest has {key}")


# =====================================================================
# Step 4: Relative depth sanity check
# =====================================================================
def step4_relative_depth():
    print("\n=== Step 4: Relative depth sanity check ===")
    cam_out = OUTPUT_DIR / CAM

    depth = np.load(str(cam_out / "depth_00000.npy"))

    left_half = depth[:, : W // 2]
    right_half = depth[:, W // 2 :]

    mean_left = float(left_half.mean())
    mean_right = float(right_half.mean())

    print(f"    Mean depth — left (bright) half:  {mean_left:.4f}")
    print(f"    Mean depth — right (dark) half:   {mean_right:.4f}")

    diff = abs(mean_left - mean_right)
    avg = (mean_left + mean_right) / 2.0
    relative_diff = diff / avg * 100 if avg > 0 else 0

    print(f"    Absolute difference: {diff:.4f}")
    print(f"    Relative difference: {relative_diff:.1f}%")

    check(diff > 0.01, f"Left and right halves have different mean depth ({diff:.4f} > 0.01)")
    check(relative_diff > 1.0, f"Relative difference > 1% ({relative_diff:.1f}%)")

    # Additionally, check that the spatial structure is coherent:
    # Use central strips to avoid edge effects from the depth model.
    margin_y = H // 4
    margin_x = W // 8
    left_center = depth[margin_y : H - margin_y, margin_x : W // 2 - margin_x]
    right_center = depth[margin_y : H - margin_y, W // 2 + margin_x : W - margin_x]
    left_center_std = float(left_center.std())
    right_center_std = float(right_center.std())
    center_diff = abs(float(left_center.mean()) - float(right_center.mean()))
    min_center_std = min(left_center_std, right_center_std)
    print(f"    Center strip std (left):  {left_center_std:.4f}")
    print(f"    Center strip std (right): {right_center_std:.4f}")
    print(f"    Center strip cross diff:  {center_diff:.4f}")

    check(
        center_diff > min_center_std * 0.5,
        f"Cross-half center diff > 0.5 * min within-half std "
        f"({center_diff:.4f} > {min_center_std * 0.5:.4f})",
    )


# =====================================================================
# Step 5: Temporal consistency
# =====================================================================
def step5_temporal_consistency():
    print("\n=== Step 5: Temporal consistency ===")
    cam_out = OUTPUT_DIR / CAM

    d0 = np.load(str(cam_out / "depth_00000.npy"))
    d1 = np.load(str(cam_out / "depth_00001.npy"))

    mae = float(np.abs(d0 - d1).mean())
    mean_depth = float(d0.mean())
    relative_diff = mae / mean_depth * 100 if mean_depth > 0 else 0

    print(f"    Mean absolute diff (frame 0 vs 1): {mae:.6f}")
    print(f"    Mean depth of frame 0:             {mean_depth:.6f}")
    print(f"    Relative difference:               {relative_diff:.2f}%")

    check(relative_diff < 10.0, f"Temporal relative diff < 10% ({relative_diff:.2f}%)")

    # Check all consecutive pairs.
    all_rel_diffs = []
    for i in range(NUM_FRAMES - 1):
        da = np.load(str(cam_out / f"depth_{i:05d}.npy"))
        db = np.load(str(cam_out / f"depth_{i + 1:05d}.npy"))
        m = float(np.abs(da - db).mean())
        r = m / float(da.mean()) * 100 if da.mean() > 0 else 0
        all_rel_diffs.append(r)

    max_rel = max(all_rel_diffs)
    print(f"    Max relative diff across all pairs: {max_rel:.2f}%")
    check(max_rel < 10.0, f"Max temporal relative diff < 10% ({max_rel:.2f}%)")


# =====================================================================
# Step 6: Cleanup
# =====================================================================
def step6_cleanup():
    print("\n=== Step 6: Cleanup ===")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    check(not TEST_DIR.exists(), "Test directory cleaned up")


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("  TEST: preprocessing/extract_depth.py")
    print("=" * 70)

    import logging

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    step1_create_data()
    step2_run()
    step3_verify()
    step4_relative_depth()
    step5_temporal_consistency()
    step6_cleanup()

    print("\n" + "=" * 70)
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 70)

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: extract_depth.py"
            " — depth maps generated, spatial structure detected,"
            " temporal consistency verified"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
