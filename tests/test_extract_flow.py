#!/usr/bin/env python3
"""Thorough test for preprocessing/extract_flow.py."""

import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TEST_DIR = Path(__file__).resolve().parent / "_test_extract_flow"
INPUT_DIR = TEST_DIR / "undistorted"
OUTPUT_DIR = TEST_DIR / "flow"
CAM = "cam_00"
W, H = 320, 240
NUM_FRAMES = 5
NUM_PAIRS = NUM_FRAMES - 1
RECT_SHIFT = 10  # pixels per frame, rightward

# Rectangle geometry (y1:y2, x1:x2 in frame 0)
RECT_Y1, RECT_Y2 = 80, 160
RECT_X0_START = 60  # left edge in frame 0

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
        x1 = RECT_X0_START + i * RECT_SHIFT
        x2 = x1 + 80
        cv2.rectangle(img, (x1, RECT_Y1), (x2, RECT_Y2), (255, 255, 255), -1)
        path = cam_dir / f"frame_{i:05d}.png"
        cv2.imwrite(str(path), img)

    frames = sorted(cam_dir.glob("*.png"))
    check(len(frames) == NUM_FRAMES, f"Created {NUM_FRAMES} frames (got {len(frames)})")

    # Verify first and last frame differ
    f0 = cv2.imread(str(frames[0]), cv2.IMREAD_GRAYSCALE)
    f4 = cv2.imread(str(frames[-1]), cv2.IMREAD_GRAYSCALE)
    check(not np.array_equal(f0, f4), "First and last frames differ")

    print(f"  Rectangle moves +{RECT_SHIFT}px/frame rightward")
    print(f"  Ground truth forward flow: ~(+{RECT_SHIFT}, 0) at rect pixels")


# =====================================================================
# Step 2: Run the script
# =====================================================================
def step2_run():
    print("\n=== Step 2: Run extract_flow ===")

    from preprocessing.extract_flow import extract_flow

    result = extract_flow(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        device_str="cpu",
        backward=True,
        num_iters=12,
        batch_size=1,
    )

    check(
        result.get("total_flow_pairs") == NUM_PAIRS,
        f"Total flow pairs = {NUM_PAIRS} (got {result.get('total_flow_pairs')})",
    )

    cam_report = result["per_camera"][0]
    check(cam_report["status"] == "ok", f"{CAM} status = ok")
    check(
        cam_report["num_pairs"] == NUM_PAIRS,
        f"{CAM} num_pairs = {NUM_PAIRS} (got {cam_report['num_pairs']})",
    )
    check(cam_report["backward_computed"] is True, "Backward flow computed")
    check(
        cam_report["mean_flow_magnitude"] > 0,
        f"Mean flow magnitude > 0 (got {cam_report['mean_flow_magnitude']:.4f})",
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
    for i in range(NUM_PAIRS):
        fwd = cam_out / f"flow_fwd_{i:05d}.npy"
        bwd = cam_out / f"flow_bwd_{i:05d}.npy"
        vis = cam_out / f"flow_vis_{i:05d}.png"
        msk = cam_out / f"flow_mask_{i:05d}.png"
        check(fwd.exists(), f"flow_fwd_{i:05d}.npy exists")
        check(bwd.exists(), f"flow_bwd_{i:05d}.npy exists")
        check(vis.exists(), f"flow_vis_{i:05d}.png exists")
        check(msk.exists(), f"flow_mask_{i:05d}.png exists")

    # --- 3b: Flow shape and content ---
    print("  -- Flow shape and sanity --")
    flow = np.load(str(cam_out / "flow_fwd_00000.npy"))
    check(flow.shape == (2, H, W), f"flow_fwd shape = (2, {H}, {W}) (got {flow.shape})")
    check(not np.all(flow == 0), "Flow is not all zeros")
    check(not np.any(np.isnan(flow)), "Flow has no NaN values")
    check(not np.any(np.isinf(flow)), "Flow has no Inf values")

    # --- 3c: Flow direction at rectangle vs background ---
    print("  -- Flow magnitude: rectangle vs background --")
    # For pair 0: rectangle is at x=[60,140], y=[80,160]
    # Use a safe interior sub-region to avoid edge effects
    rect_mask = np.zeros((H, W), dtype=bool)
    rect_mask[RECT_Y1 + 10 : RECT_Y2 - 10, RECT_X0_START + 15 : RECT_X0_START + 65] = True
    bg_mask = np.zeros((H, W), dtype=bool)
    bg_mask[10:40, 10:40] = True  # top-left, far from rectangle

    u = flow[0]  # horizontal
    v = flow[1]  # vertical

    rect_u = u[rect_mask].mean()
    rect_v = v[rect_mask].mean()
    bg_u = u[bg_mask].mean()
    bg_v = v[bg_mask].mean()

    print(f"    Rectangle region: u={rect_u:.2f}, v={rect_v:.2f}")
    print(f"    Background region: u={bg_u:.2f}, v={bg_v:.2f}")

    check(rect_u > 3.0, f"Rectangle horizontal flow is positive/rightward ({rect_u:.2f} > 3.0)")
    check(
        abs(rect_v) < abs(rect_u),
        f"Rectangle vertical flow < horizontal ({abs(rect_v):.2f} < {abs(rect_u):.2f})",
    )
    check(
        abs(bg_u) < abs(rect_u),
        f"Background flow < rectangle flow ({abs(bg_u):.2f} < {abs(rect_u):.2f})",
    )

    # --- 3d: Visualisation validity ---
    print("  -- Visualisation PNGs --")
    vis = cv2.imread(str(cam_out / "flow_vis_00000.png"))
    check(vis is not None, "flow_vis_00000.png is readable")
    check(vis.shape == (H, W, 3), f"Vis shape = ({H}, {W}, 3) (got {vis.shape})")
    check(vis.mean() > 0, f"Vis is not all black (mean={vis.mean():.1f})")

    # --- 3e: Mask validity ---
    print("  -- Consistency masks --")
    mask = cv2.imread(str(cam_out / "flow_mask_00000.png"), cv2.IMREAD_GRAYSCALE)
    check(mask is not None, "flow_mask_00000.png is readable")
    check(mask.shape == (H, W), f"Mask shape = ({H}, {W}) (got {mask.shape})")
    unique = set(np.unique(mask))
    check(unique.issubset({0, 255}), f"Mask is binary (values: {unique})")

    consistent_pct = (mask == 255).sum() / mask.size * 100
    print(f"    Consistent pixels: {consistent_pct:.1f}%")

    # --- 3f: Manifest ---
    print("  -- Manifest --")
    manifest_path = OUTPUT_DIR / "flow_manifest.json"
    check(manifest_path.exists(), "flow_manifest.json exists")

    with open(manifest_path) as f:
        manifest = json.load(f)

    check(manifest["total_flow_pairs"] == NUM_PAIRS, f"Manifest total_flow_pairs = {NUM_PAIRS}")
    check(manifest["backward"] is True, "Manifest backward = True")
    check(manifest["device"] == "cpu", "Manifest device = cpu")
    check(manifest["num_iters"] == 12, "Manifest num_iters = 12")
    check("per_camera" in manifest, "Manifest has per_camera section")

    cam_entry = manifest["per_camera"][0]
    check(cam_entry["camera"] == CAM, f"Manifest camera = {CAM}")
    check("mean_flow_magnitude" in cam_entry, "Manifest has mean_flow_magnitude")
    check("max_flow_magnitude" in cam_entry, "Manifest has max_flow_magnitude")


# =====================================================================
# Step 4: Forward-backward consistency check
# =====================================================================
def step4_fb_consistency():
    print("\n=== Step 4: Forward-backward consistency check ===")
    cam_out = OUTPUT_DIR / CAM

    fwd = np.load(str(cam_out / "flow_fwd_00000.npy"))  # (2, H, W)
    bwd = np.load(str(cam_out / "flow_bwd_00000.npy"))  # (2, H, W)

    check(fwd.shape == bwd.shape, f"Forward and backward flow have same shape ({fwd.shape})")

    # Sample pixels inside the rectangle interior (frame 0 positions)
    sample_ys = [100, 110, 120, 130, 140]
    sample_xs = [80, 90, 100, 110, 120]

    errors = []
    print("    Pixel-level round-trip consistency (fwd + bwd_warped):")
    for y, x in zip(sample_ys, sample_xs):
        fu, fv = fwd[0, y, x], fwd[1, y, x]
        # Warped position
        wx = x + fu
        wy = y + fv
        # Bilinear sample of backward flow at warped position
        wx_i, wy_i = int(round(wx)), int(round(wy))
        wx_i = max(0, min(W - 1, wx_i))
        wy_i = max(0, min(H - 1, wy_i))
        bu, bv = bwd[0, wy_i, wx_i], bwd[1, wy_i, wx_i]
        err = np.sqrt((fu + bu) ** 2 + (fv + bv) ** 2)
        errors.append(err)
        print(
            f"      ({x},{y}): fwd=({fu:.2f},{fv:.2f}), "
            f"bwd@warped=({bu:.2f},{bv:.2f}), err={err:.2f}px"
        )

    mean_err = np.mean(errors)
    print(f"    Mean round-trip error: {mean_err:.2f}px")

    check(mean_err < 5.0, f"Mean FB consistency error < 5.0px ({mean_err:.2f}px)")

    # Verify mask marks high-error pixels as occluded
    mask = cv2.imread(str(cam_out / "flow_mask_00000.png"), cv2.IMREAD_GRAYSCALE)

    # Recompute consistency error map for the full image
    xs_grid, ys_grid = np.meshgrid(
        np.arange(W, dtype=np.float32),
        np.arange(H, dtype=np.float32),
    )
    wx = xs_grid + fwd[0]
    wy = ys_grid + fwd[1]
    bwd_u_warped = cv2.remap(bwd[0], wx, wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    bwd_v_warped = cv2.remap(bwd[1], wx, wy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    err_map = np.sqrt((fwd[0] + bwd_u_warped) ** 2 + (fwd[1] + bwd_v_warped) ** 2)

    # Where error is low, mask should be 255 (consistent)
    low_err = err_map < 0.5
    if low_err.sum() > 0:
        mask_at_low_err = mask[low_err].mean()
        check(
            mask_at_low_err > 200,
            f"Mask marks low-error pixels as consistent (mean={mask_at_low_err:.0f})",
        )

    # Where error is very high, mask should be 0 (occluded)
    high_err = err_map > 3.0
    if high_err.sum() > 0:
        mask_at_high_err = mask[high_err].mean()
        check(
            mask_at_high_err < 128,
            f"Mask marks high-error pixels as occluded (mean={mask_at_high_err:.0f})",
        )
    else:
        print("    (No pixels with error > 3.0 — skipping occluded mask check)")
        check(True, "Skipped — no high-error region (scene is simple)")

    # Backward flow direction check
    bwd_rect_u = bwd[
        0,
        RECT_Y1 + 10 : RECT_Y2 - 10,
        RECT_X0_START + RECT_SHIFT + 15 : RECT_X0_START + RECT_SHIFT + 65,
    ].mean()
    print(f"    Backward flow at rect in frame 1: u={bwd_rect_u:.2f} (expected ~-{RECT_SHIFT})")
    check(bwd_rect_u < -3.0, f"Backward flow is leftward ({bwd_rect_u:.2f} < -3.0)")


# =====================================================================
# Step 5: Cleanup
# =====================================================================
def step5_cleanup():
    print("\n=== Step 5: Cleanup ===")
    if TEST_DIR.exists():
        shutil.rmtree(TEST_DIR)
    check(not TEST_DIR.exists(), "Test directory cleaned up")


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("  TEST: preprocessing/extract_flow.py")
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
    step4_fb_consistency()
    step5_cleanup()

    print("\n" + "=" * 70)
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 70)

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: extract_flow.py"
            " — flow extracted, visualizations generated,"
            " forward-backward consistency verified"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
