#!/usr/bin/env python3
"""Thorough test for gaussian_splatting/losses.py."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

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
# Step 1: Photometric loss
# =====================================================================
def step1_photometric():
    print("\n=== Step 1: Photometric loss ===")
    from gaussian_splatting.losses import photometric_loss

    torch.manual_seed(42)
    identical = torch.rand(3, 64, 64)

    # --- Identical images: loss should be ~0 ---
    print("  -- Identical images --")
    loss_identical = photometric_loss(identical, identical)
    print(f"    loss(identical, identical) = {loss_identical.item():.8f}")
    check(
        loss_identical.item() < 1e-5,
        f"Identical images -> loss ~0 (got {loss_identical.item():.8f})",
    )

    # --- Different images: loss should be significant ---
    print("  -- Different images --")
    different = torch.rand(3, 64, 64)
    loss_different = photometric_loss(identical, different)
    print(f"    loss(identical, different) = {loss_different.item():.6f}")
    check(
        loss_different.item() > 0.01,
        f"Different images -> loss > 0.01 (got {loss_different.item():.6f})",
    )

    # --- Scalar tensor with gradient ---
    print("  -- Gradient flow --")
    rendered = torch.rand(3, 64, 64, requires_grad=True)
    target = torch.rand(3, 64, 64)
    loss = photometric_loss(rendered, target)
    check(loss.dim() == 0, f"Loss is scalar (dim={loss.dim()})")
    loss.backward()
    check(rendered.grad is not None, "Gradient exists on rendered")
    check(
        rendered.grad.norm().item() > 0, f"Gradient non-zero (||grad||={rendered.grad.norm():.6e})"
    )

    # --- Batched input ---
    print("  -- Batched input --")
    batch_r = torch.rand(2, 3, 64, 64)
    batch_t = torch.rand(2, 3, 64, 64)
    loss_batch = photometric_loss(batch_r, batch_t)
    check(loss_batch.dim() == 0, "Batched loss is scalar")
    check(loss_batch.item() > 0, "Batched loss > 0")


# =====================================================================
# Step 2: SSIM implementation check
# =====================================================================
def step2_ssim():
    print("\n=== Step 2: SSIM implementation check ===")
    from gaussian_splatting.losses import _ensure_batch, _ssim

    torch.manual_seed(42)
    img = torch.rand(1, 3, 64, 64)

    # --- SSIM with itself: should be ~1.0 ---
    ssim_self = _ssim(img, img)
    print(f"    SSIM(img, img)   = {ssim_self.item():.8f}")
    check(abs(ssim_self.item() - 1.0) < 1e-4, f"SSIM(img, img) ~1.0 (got {ssim_self.item():.8f})")

    # --- SSIM with random noise: should be near 0 ---
    noise = torch.rand(1, 3, 64, 64)
    ssim_noise = _ssim(img, noise)
    print(f"    SSIM(img, noise) = {ssim_noise.item():.6f}")
    check(ssim_noise.item() < 0.5, f"SSIM(img, noise) < 0.5 (got {ssim_noise.item():.6f})")

    # --- SSIM is symmetric ---
    ssim_ab = _ssim(img, noise)
    ssim_ba = _ssim(noise, img)
    check(
        abs(ssim_ab.item() - ssim_ba.item()) < 1e-6,
        f"SSIM is symmetric (diff={abs(ssim_ab.item() - ssim_ba.item()):.8f})",
    )


# =====================================================================
# Step 3: Depth loss (Pearson correlation)
# =====================================================================
def step3_depth():
    print("\n=== Step 3: Depth loss (Pearson correlation) ===")
    from gaussian_splatting.losses import depth_loss

    # --- Perfect linear correlation: loss should be ~0 ---
    print("  -- Perfect correlation --")
    depth_a = torch.arange(64 * 64).float().reshape(1, 64, 64)
    depth_b = depth_a * 2.5 + 10.0  # linear transform
    loss_perfect = depth_loss(depth_a, depth_b)
    print(f"    loss(a, 2.5*a+10) = {loss_perfect.item():.8f}")
    check(
        loss_perfect.item() < 1e-4,
        f"Perfect correlation -> loss ~0 (got {loss_perfect.item():.8f})",
    )

    # --- Negative correlation: loss should be ~2 ---
    print("  -- Negative correlation --")
    depth_neg = -depth_a + 100.0
    loss_neg = depth_loss(depth_a, depth_neg)
    print(f"    loss(a, -a+100) = {loss_neg.item():.6f}")
    check(
        abs(loss_neg.item() - 2.0) < 1e-4,
        f"Negative correlation -> loss ~2.0 (got {loss_neg.item():.6f})",
    )

    # --- Uncorrelated: loss should be substantial ---
    print("  -- Uncorrelated --")
    torch.manual_seed(99)
    depth_c = torch.rand(1, 64, 64)
    loss_uncorr = depth_loss(depth_a, depth_c)
    print(f"    loss(a, random) = {loss_uncorr.item():.6f}")
    check(loss_uncorr.item() > 0.3, f"Uncorrelated -> loss > 0.3 (got {loss_uncorr.item():.6f})")

    # --- Mask support ---
    print("  -- Mask support --")
    mask = torch.zeros(1, 64, 64)
    mask[:, :32, :] = 1.0  # only top half
    loss_masked = depth_loss(depth_a, depth_b, mask=mask)
    print(f"    loss with mask = {loss_masked.item():.8f}")
    check(
        loss_masked.item() < 1e-4,
        f"Masked perfect correlation -> ~0 (got {loss_masked.item():.8f})",
    )

    # --- Gradient flow ---
    print("  -- Gradient flow --")
    rd = torch.rand(1, 64, 64, requires_grad=True)
    pd = torch.rand(1, 64, 64)
    loss = depth_loss(rd, pd)
    loss.backward()
    check(rd.grad is not None, "Gradient exists on rendered_depth")
    check(rd.grad.norm().item() > 0, f"Gradient non-zero (||grad||={rd.grad.norm():.6e})")

    # --- Batched ---
    print("  -- Batched --")
    batch_rd = torch.rand(2, 1, 64, 64)
    batch_pd = torch.rand(2, 1, 64, 64)
    loss_batch = depth_loss(batch_rd, batch_pd)
    check(loss_batch.dim() == 0, "Batched depth loss is scalar")


# =====================================================================
# Step 4: Temporal consistency loss
# =====================================================================
def step4_temporal_consistency():
    print("\n=== Step 4: Temporal consistency loss ===")
    from gaussian_splatting.losses import temporal_consistency_loss

    torch.manual_seed(42)
    H, W = 64, 64

    # Create image with a horizontal gradient (predictable structure)
    rendered = torch.linspace(0, 1, W).unsqueeze(0).unsqueeze(0).expand(3, H, W).clone()

    # Target is rendered shifted 5 pixels to the right
    target = torch.zeros_like(rendered)
    target[:, :, 5:] = rendered[:, :, : W - 5]
    target[:, :, :5] = rendered[:, :, 0:1]  # border padding

    # Flow field that explains the shift: (dx=5, dy=0)
    flow_correct = torch.zeros(2, H, W)
    flow_correct[0] = 5.0  # 5 pixels in x direction

    # --- Correct flow: loss should be small ---
    print("  -- Correct flow --")
    loss_correct = temporal_consistency_loss(rendered, target, flow_correct)
    print(f"    loss with correct flow = {loss_correct.item():.6f}")

    # --- Zero flow: loss should be larger ---
    print("  -- Zero flow --")
    flow_zero = torch.zeros(2, H, W)
    loss_zero = temporal_consistency_loss(rendered, target, flow_zero)
    print(f"    loss with zero flow    = {loss_zero.item():.6f}")

    check(
        loss_correct.item() < loss_zero.item(),
        f"Correct flow < zero flow ({loss_correct.item():.6f} < {loss_zero.item():.6f})",
    )

    # --- Gradient flow ---
    print("  -- Gradient flow --")
    r = torch.rand(3, H, W, requires_grad=True)
    t = torch.rand(3, H, W)
    f_flow = torch.randn(2, H, W)
    loss = temporal_consistency_loss(r, t, f_flow)
    loss.backward()
    check(r.grad is not None, "Gradient exists on rendered_t")
    check(r.grad.norm().item() > 0, f"Gradient non-zero (||grad||={r.grad.norm():.6e})")

    # --- Batched ---
    print("  -- Batched --")
    loss_batch = temporal_consistency_loss(
        torch.rand(2, 3, H, W),
        torch.rand(2, 3, H, W),
        torch.randn(2, 2, H, W),
    )
    check(loss_batch.dim() == 0, "Batched temporal loss is scalar")


# =====================================================================
# Step 5: Temporal smoothness loss
# =====================================================================
def step5_temporal_smoothness():
    print("\n=== Step 5: Temporal smoothness loss ===")
    from gaussian_splatting.losses import temporal_smoothness_loss

    N = 100

    # --- Constant velocity (straight line): acceleration ~0 ---
    print("  -- Constant velocity --")
    t0 = torch.zeros(N, 3)
    t1 = torch.ones(N, 3)
    t2 = 2.0 * torch.ones(N, 3)

    loss_smooth = temporal_smoothness_loss(t0, t1, t2)
    # acceleration = 0 - 2*1 + 2 = 0, velocity = 1-0 = 1
    # loss = 0 + 0.1 * mean(1^2 * 3) = 0.1 * 1.0 = 0.1
    print(f"    loss (constant velocity) = {loss_smooth.item():.6f}")
    # Acceleration is zero; only velocity penalty contributes
    accel_component = 0.0
    velocity_component = 0.1 * 1.0  # 0.1 * mean(velocity^2) = 0.1
    expected = accel_component + velocity_component
    check(
        abs(loss_smooth.item() - expected) < 1e-5,
        f"Constant velocity -> loss = {expected} (got {loss_smooth.item():.6f})",
    )

    # --- Sudden acceleration ---
    print("  -- Sudden acceleration --")
    t2_accel = 5.0 * torch.ones(N, 3)
    loss_accel = temporal_smoothness_loss(t0, t1, t2_accel)
    # acceleration = 0 - 2*1 + 5 = 3, velocity = 1-0 = 1
    # accel_loss = mean(9 * 3) / 3 = 9.0, vel_penalty = 0.1 * 1.0
    print(f"    loss (sudden accel) = {loss_accel.item():.6f}")
    check(
        loss_accel.item() > loss_smooth.item(),
        f"Acceleration loss > smooth loss ({loss_accel.item():.4f} > {loss_smooth.item():.4f})",
    )
    check(
        loss_accel.item() > 1.0, f"Acceleration loss is significant (got {loss_accel.item():.4f})"
    )

    # --- Stationary: both zero ---
    print("  -- Stationary --")
    loss_still = temporal_smoothness_loss(torch.zeros(N, 3), torch.zeros(N, 3), torch.zeros(N, 3))
    check(loss_still.item() < 1e-8, f"Stationary -> loss ~0 (got {loss_still.item():.8f})")

    # --- Gradient flow ---
    print("  -- Gradient flow --")
    x0 = torch.randn(N, 3, requires_grad=True)
    x1 = torch.randn(N, 3, requires_grad=True)
    x2 = torch.randn(N, 3, requires_grad=True)
    loss = temporal_smoothness_loss(x0, x1, x2)
    loss.backward()
    check(x0.grad is not None and x0.grad.norm() > 0, "Gradient on xyz_t_minus_1")
    check(x1.grad is not None and x1.grad.norm() > 0, "Gradient on xyz_t")
    check(x2.grad is not None and x2.grad.norm() > 0, "Gradient on xyz_t_plus_1")


# =====================================================================
# Step 6: Opacity and scale regularization
# =====================================================================
def step6_regularization():
    print("\n=== Step 6: Opacity and scale regularization ===")
    from gaussian_splatting.losses import opacity_regularization, scale_regularization

    # --- Opacity at 0.5 (maximum uncertainty) ---
    print("  -- Opacity regularization --")
    opacity_mid = torch.full((100, 1), 0.5)
    loss_mid = opacity_regularization(opacity_mid)
    print(f"    opacity=0.5: loss = {loss_mid.item():.6f}")

    # --- Opacity at 0.99 (decisive) ---
    opacity_high = torch.full((100, 1), 0.99)
    loss_high = opacity_regularization(opacity_high)
    print(f"    opacity=0.99: loss = {loss_high.item():.6f}")

    # --- Opacity at 0.01 (decisive low) ---
    opacity_low = torch.full((100, 1), 0.01)
    loss_low = opacity_regularization(opacity_low)
    print(f"    opacity=0.01: loss = {loss_low.item():.6f}")

    check(
        loss_mid.item() > loss_high.item(),
        f"opacity=0.5 loss > opacity=0.99 loss ({loss_mid.item():.4f} > {loss_high.item():.4f})",
    )
    check(
        loss_mid.item() > loss_low.item(),
        f"opacity=0.5 loss > opacity=0.01 loss ({loss_mid.item():.4f} > {loss_low.item():.4f})",
    )

    # Binary entropy max is ln(2) ~ 0.693 at p=0.5
    check(
        abs(loss_mid.item() - 0.6931) < 0.01,
        f"Entropy at 0.5 ~ln(2)=0.693 (got {loss_mid.item():.4f})",
    )

    # --- Gradient flow ---
    o = torch.rand(50, 1, requires_grad=True)
    loss = opacity_regularization(o)
    loss.backward()
    check(o.grad is not None and o.grad.norm() > 0, "Gradient on opacity")

    # --- Scale regularization ---
    print("  -- Scale regularization --")

    # All below threshold: loss should be 0
    scale_small = torch.full((100, 3), 0.05)
    loss_small = scale_regularization(scale_small, max_scale=0.1)
    print(f"    scales=0.05 (below 0.1): loss = {loss_small.item():.8f}")
    check(loss_small.item() == 0.0, f"Scales below max -> loss = 0 (got {loss_small.item():.8f})")

    # All above threshold: loss should be positive
    scale_large = torch.full((100, 3), 0.3)
    loss_large = scale_regularization(scale_large, max_scale=0.1)
    expected_scale_loss = 0.3 - 0.1  # relu(0.3 - 0.1) = 0.2
    print(f"    scales=0.3 (above 0.1): loss = {loss_large.item():.6f}")
    check(
        abs(loss_large.item() - expected_scale_loss) < 1e-5,
        f"loss = relu(0.3-0.1).mean() = 0.2 (got {loss_large.item():.6f})",
    )

    # Mixed: some above, some below
    scale_mixed = torch.tensor([[0.05, 0.15, 0.05]])  # (1, 3)
    loss_mixed = scale_regularization(scale_mixed, max_scale=0.1)
    expected_mixed = (0.0 + 0.05 + 0.0) / 3.0
    print(f"    mixed scales: loss = {loss_mixed.item():.6f}")
    check(
        abs(loss_mixed.item() - expected_mixed) < 1e-5,
        f"Mixed scales -> correct mean (got {loss_mixed.item():.6f})",
    )

    # --- Gradient flow ---
    s = (torch.rand(50, 3) * 0.3).detach().requires_grad_(True)
    loss = scale_regularization(s, max_scale=0.1)
    loss.backward()
    check(s.grad is not None, "Gradient on scaling")


# =====================================================================
# Step 7: Total loss
# =====================================================================
def step7_total_loss():
    print("\n=== Step 7: Total loss ===")
    from gaussian_splatting.losses import LossConfig, total_loss

    torch.manual_seed(42)
    H, W = 64, 64

    rendered = torch.rand(3, H, W)
    target = torch.rand(3, H, W)
    rendered_depth = torch.rand(1, H, W)
    pseudo_depth = torch.rand(1, H, W)
    depth_mask = torch.ones(1, H, W)
    rendered_t = torch.rand(3, H, W)
    target_t1 = torch.rand(3, H, W)
    flow_fwd = torch.randn(2, H, W) * 2
    xyz_t0 = torch.randn(100, 3)
    xyz_t1 = xyz_t0 + torch.randn(100, 3) * 0.1
    xyz_t2 = xyz_t1 + torch.randn(100, 3) * 0.1
    opacity = torch.rand(100, 1)
    scaling = torch.rand(100, 3) * 0.2

    config = LossConfig(
        lambda_photo=1.0,
        lambda_l1=0.8,
        lambda_ssim=0.2,
        lambda_depth=0.5,
        lambda_temporal=0.1,
        lambda_smooth=0.01,
        lambda_opacity_reg=0.01,
        lambda_scale_reg=0.01,
        max_scale=0.1,
    )

    total, losses = total_loss(
        rendered=rendered,
        target=target,
        rendered_depth=rendered_depth,
        pseudo_depth=pseudo_depth,
        depth_mask=depth_mask,
        rendered_t=rendered_t,
        target_t1=target_t1,
        flow_fwd=flow_fwd,
        xyz_t_minus_1=xyz_t0,
        xyz_t=xyz_t1,
        xyz_t_plus_1=xyz_t2,
        opacity=opacity,
        scaling=scaling,
        config=config,
    )

    # --- Return types ---
    print("  -- Return types --")
    check(isinstance(total, torch.Tensor), "total is a Tensor")
    check(total.dim() == 0, f"total is scalar (dim={total.dim()})")
    check(isinstance(losses, dict), "losses is a dict")

    # --- All expected keys present ---
    print("  -- Expected keys --")
    expected_keys = {
        "photometric",
        "depth",
        "temporal_consistency",
        "temporal_smoothness",
        "opacity_reg",
        "scale_reg",
    }
    check(
        set(losses.keys()) == expected_keys,
        f"All 6 loss keys present (got {sorted(losses.keys())})",
    )

    # --- All individual losses are non-negative scalars ---
    print("  -- Individual losses --")
    for name, val in sorted(losses.items()):
        is_scalar = val.dim() == 0
        is_nonneg = val.item() >= 0
        print(f"    {name}: {val.item():.6f} (scalar={is_scalar}, >=0={is_nonneg})")
        check(is_scalar and is_nonneg, f"{name} is non-negative scalar")

    # --- Verify total == weighted sum ---
    print("  -- Weighted sum verification --")
    recomputed = (
        config.lambda_photo * losses["photometric"]
        + config.lambda_depth * losses["depth"]
        + config.lambda_temporal * losses["temporal_consistency"]
        + config.lambda_smooth * losses["temporal_smoothness"]
        + config.lambda_opacity_reg * losses["opacity_reg"]
        + config.lambda_scale_reg * losses["scale_reg"]
    )
    diff = abs(total.item() - recomputed.item())
    print(f"    total = {total.item():.8f}")
    print(f"    recomputed = {recomputed.item():.8f}")
    print(f"    diff = {diff:.2e}")
    check(diff < 1e-5, f"Total == weighted sum of components (diff={diff:.2e})")

    # --- Partial inputs (only photometric) ---
    print("  -- Partial inputs --")
    total_partial, losses_partial = total_loss(rendered, target)
    check(
        len(losses_partial) == 1 and "photometric" in losses_partial,
        f"Partial: only photometric present (got {list(losses_partial.keys())})",
    )

    # --- Gradient flow through total_loss ---
    print("  -- Gradient flow --")
    r = torch.rand(3, H, W, requires_grad=True)
    t_target = torch.rand(3, H, W)
    total_g, _ = total_loss(r, t_target)
    total_g.backward()
    check(
        r.grad is not None and r.grad.norm() > 0,
        f"Gradient flows through total_loss (||grad||={r.grad.norm():.6e})",
    )


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("  TEST: gaussian_splatting/losses.py")
    print("=" * 70)

    step1_photometric()
    step2_ssim()
    step3_depth()
    step4_temporal_consistency()
    step5_temporal_smoothness()
    step6_regularization()
    step7_total_loss()

    print("\n" + "=" * 70)
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 70)

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: losses.py"
            " — all 7 loss functions verified with expected behavior"
            " and gradient flow"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
