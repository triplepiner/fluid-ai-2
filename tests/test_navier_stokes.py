#!/usr/bin/env python3
"""Thorough test for pinn/navier_stokes.py."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinn.model import FluidDomain, FluidPINN, PINNConfig
from pinn.navier_stokes import (
    compute_derivatives,
    compute_vorticity,
    navier_stokes_residual,
    physics_loss,
)

checks_passed = 0
checks_total = 0


def check(cond: bool, label: str) -> None:
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if cond:
        checks_passed += 1


def no_nan(t: torch.Tensor) -> bool:
    return not torch.isnan(t).any().item()


def no_inf(t: torch.Tensor) -> bool:
    return not torch.isinf(t).any().item()


def not_all_zero(t: torch.Tensor, tol: float = 1e-12) -> bool:
    return t.abs().max().item() > tol


# =====================================================================
# Expected derivative keys
# =====================================================================
FIELD_KEYS = {"u", "v", "w", "p", "rho"}

FIRST_SPATIAL_KEYS = {f"d{c}_d{a}" for c in ("u", "v", "w") for a in ("x", "y", "z")}
# du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz, dw_dx, dw_dy, dw_dz

PRESSURE_GRAD_KEYS = {"dp_dx", "dp_dy", "dp_dz"}

TIME_DERIV_KEYS = {"du_dt", "dv_dt", "dw_dt"}

SECOND_SPATIAL_KEYS = {f"d2{c}_d{a}2" for c in ("u", "v", "w") for a in ("x", "y", "z")}
# d2u_dx2, d2u_dy2, d2u_dz2, d2v_dx2, ...

ALL_EXPECTED_KEYS = (
    FIELD_KEYS | FIRST_SPATIAL_KEYS | PRESSURE_GRAD_KEYS | TIME_DERIV_KEYS | SECOND_SPATIAL_KEYS
)

RESIDUAL_KEYS = {"momentum_x", "momentum_y", "momentum_z", "continuity"}


# =====================================================================
# Step 1: Derivative computation test
# =====================================================================
def step1_derivative_computation():
    print("\n=== Step 1: Derivative computation test ===")

    cfg = PINNConfig(activation="siren", hidden_dim=64, num_layers=4, omega_0=30.0)
    model = FluidPINN(cfg)

    B = 200
    x = torch.randn(B, 3, requires_grad=True)
    t = torch.randn(B, 1, requires_grad=True)
    derivatives = compute_derivatives(model, x, t)

    # -- All expected keys present --
    print("  -- Key existence --")
    actual_keys = set(derivatives.keys())
    missing = ALL_EXPECTED_KEYS - actual_keys
    check(
        len(missing) == 0,
        f"All {len(ALL_EXPECTED_KEYS)} expected keys present"
        + (f" (missing: {sorted(missing)})" if missing else ""),
    )

    # Print total key count
    print(f"    Total keys in derivatives dict: {len(actual_keys)}")
    print(f"    Expected keys: {len(ALL_EXPECTED_KEYS)}")

    # -- Shape checks --
    print("  -- Shape checks --")
    shape_ok_count = 0
    shape_fail_keys = []
    for key in sorted(ALL_EXPECTED_KEYS):
        if key in derivatives:
            val = derivatives[key]
            if val.shape == (B, 1):
                shape_ok_count += 1
            else:
                shape_fail_keys.append((key, val.shape))
    check(
        shape_ok_count == len(ALL_EXPECTED_KEYS),
        f"All {len(ALL_EXPECTED_KEYS)} values have shape ({B}, 1)",
    )
    if shape_fail_keys:
        for k, s in shape_fail_keys:
            print(f"    BAD SHAPE: {k} has {s}, expected ({B}, 1)")

    # -- NaN / Inf checks --
    print("  -- NaN and Inf checks --")
    nan_keys = [
        k for k in sorted(ALL_EXPECTED_KEYS) if k in derivatives and not no_nan(derivatives[k])
    ]
    inf_keys = [
        k for k in sorted(ALL_EXPECTED_KEYS) if k in derivatives and not no_inf(derivatives[k])
    ]
    check(
        len(nan_keys) == 0,
        f"No NaN in any derivative" + (f" (NaN in: {nan_keys})" if nan_keys else ""),
    )
    check(
        len(inf_keys) == 0,
        f"No Inf in any derivative" + (f" (Inf in: {inf_keys})" if inf_keys else ""),
    )

    # -- Non-zero checks --
    print("  -- Non-zero checks --")
    zero_keys = []
    for key in sorted(ALL_EXPECTED_KEYS):
        if key in derivatives and not not_all_zero(derivatives[key]):
            zero_keys.append(key)
    check(
        len(zero_keys) == 0,
        f"No derivative tensor is all-zero" + (f" (all-zero: {zero_keys})" if zero_keys else ""),
    )

    # Print summary statistics for a few key derivatives
    print("  -- Derivative magnitude summary --")
    for key in ("u", "du_dx", "du_dt", "d2u_dx2", "dp_dx"):
        if key in derivatives:
            val = derivatives[key]
            print(
                f"    {key}: mean={val.mean().item():.6f}, "
                f"std={val.std().item():.6f}, "
                f"max_abs={val.abs().max().item():.6f}"
            )

    return model, derivatives


# =====================================================================
# Step 2: NS residual test
# =====================================================================
def step2_ns_residual(derivatives):
    print("\n=== Step 2: NS residual test ===")

    B = 200
    residuals = navier_stokes_residual(derivatives, nu=1e-3, gravity=(0, -9.81, 0))

    # -- Key existence --
    print("  -- Key existence --")
    actual_keys = set(residuals.keys())
    check(
        RESIDUAL_KEYS.issubset(actual_keys), f"All 4 residual keys present: {sorted(RESIDUAL_KEYS)}"
    )

    # -- Shape checks --
    print("  -- Shape checks --")
    for key in sorted(RESIDUAL_KEYS):
        check(residuals[key].shape == (B, 1), f"{key} shape = ({B}, 1)")

    # -- NaN / Inf --
    print("  -- NaN and Inf --")
    for key in sorted(RESIDUAL_KEYS):
        check(no_nan(residuals[key]), f"{key} no NaN")
        check(no_inf(residuals[key]), f"{key} no Inf")

    # -- Non-zero (untrained network should not satisfy NS) --
    print("  -- Residuals are non-zero (physics violated) --")
    for key in sorted(RESIDUAL_KEYS):
        val = residuals[key]
        mean_abs = val.abs().mean().item()
        check(not_all_zero(val), f"{key} is non-zero (mean |r| = {mean_abs:.6f})")
        print(f"    {key}: mean_abs={mean_abs:.6f}, max_abs={val.abs().max().item():.6f}")

    return residuals


# =====================================================================
# Step 3: Physics loss test
# =====================================================================
def step3_physics_loss(residuals):
    print("\n=== Step 3: Physics loss test ===")

    lam_mom = 2.0
    lam_cont = 0.5
    total, details = physics_loss(residuals, lambda_momentum=lam_mom, lambda_continuity=lam_cont)

    # -- Total is positive scalar --
    check(total.dim() == 0, "total is a scalar")
    check(total.item() > 0, f"total > 0 ({total.item():.6f})")
    check(no_nan(total), "total no NaN")

    # -- Details contains per-residual MSE --
    print("  -- Details dict --")
    for key in ("momentum_x", "momentum_y", "momentum_z", "continuity"):
        check(key in details, f"'{key}' in details")
        if key in details:
            print(f"    {key} MSE = {details[key].item():.6f}")

    # -- Verify total = weighted sum --
    print("  -- Weighted sum verification --")
    expected_total = (
        lam_mom * details["momentum_x"]
        + lam_mom * details["momentum_y"]
        + lam_mom * details["momentum_z"]
        + lam_cont * details["continuity"]
    )
    rel_err = (total - expected_total).abs().item()
    check(rel_err < 1e-5, f"total matches weighted sum (err={rel_err:.2e})")

    # -- Different weights change the total --
    print("  -- Weight sensitivity --")
    total_eq, _ = physics_loss(residuals, lambda_momentum=1.0, lambda_continuity=1.0)
    total_hi_mom, _ = physics_loss(residuals, lambda_momentum=10.0, lambda_continuity=1.0)
    check(
        total_hi_mom.item() > total_eq.item(),
        f"Higher lambda_momentum -> higher loss "
        f"({total_hi_mom.item():.4f} > {total_eq.item():.4f})",
    )

    return total


# =====================================================================
# Step 4: End-to-end gradient test
# =====================================================================
def step4_gradient_test():
    print("\n=== Step 4: End-to-end gradient test ===")

    cfg = PINNConfig(activation="siren", hidden_dim=64, num_layers=4)
    model = FluidPINN(cfg)

    B = 100
    x = torch.randn(B, 3, requires_grad=True)
    t = torch.randn(B, 1, requires_grad=True)

    derivs = compute_derivatives(model, x, t)
    residuals = navier_stokes_residual(derivs, nu=1e-3, gravity=(0, -9.81, 0))
    total, _ = physics_loss(residuals)

    model.zero_grad()
    total.backward()

    # -- All model params have gradients --
    print("  -- Parameter gradients --")
    param_names = [name for name, _ in model.named_parameters()]
    n_total = len(param_names)
    grad_norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms[name] = p.grad.norm().item()
        else:
            grad_norms[name] = None

    n_with_grad = sum(1 for v in grad_norms.values() if v is not None and v > 0)
    check(
        n_with_grad == n_total,
        f"All {n_total} params have non-zero gradients ({n_with_grad}/{n_total})",
    )

    # -- Print first and last layer grad norms --
    first_name = param_names[0]
    last_name = param_names[-1]
    print(f"    First layer ({first_name}): grad_norm = {grad_norms[first_name]:.6e}")
    print(f"    Last layer ({last_name}): grad_norm = {grad_norms[last_name]:.6e}")

    # -- Verify no NaN in gradients --
    nan_grads = [
        name
        for name, p in model.named_parameters()
        if p.grad is not None and torch.isnan(p.grad).any()
    ]
    check(
        len(nan_grads) == 0,
        f"No NaN in parameter gradients" + (f" (NaN in: {nan_grads})" if nan_grads else ""),
    )

    # -- Verify x and t also got gradients (leaf tensors) --
    check(x.grad is not None, "x leaf tensor has gradient")
    check(t.grad is not None, "t leaf tensor has gradient")

    # -- Repeat for Fourier model --
    print("  -- Fourier model gradient test --")
    cfg_f = PINNConfig(activation="fourier", hidden_dim=64, num_layers=4, num_fourier_features=64)
    model_f = FluidPINN(cfg_f)

    x_f = torch.randn(B, 3, requires_grad=True)
    t_f = torch.randn(B, 1, requires_grad=True)
    derivs_f = compute_derivatives(model_f, x_f, t_f)
    res_f = navier_stokes_residual(derivs_f, nu=1e-3)
    total_f, _ = physics_loss(res_f)

    model_f.zero_grad()
    total_f.backward()

    n_grad_f = sum(
        1 for p in model_f.parameters() if p.grad is not None and p.grad.abs().max().item() > 0
    )
    n_total_f = sum(1 for _ in model_f.parameters())
    check(
        n_grad_f == n_total_f,
        f"Fourier: all {n_total_f} params have gradients ({n_grad_f}/{n_total_f})",
    )


# =====================================================================
# Step 5: Known solution test (training reduces loss)
# =====================================================================
def step5_known_solution():
    print("\n=== Step 5: Known solution / training test ===")

    # -- Part A: untrained residual is large --
    print("  -- Part A: untrained residual is large --")
    cfg = PINNConfig(activation="siren", hidden_dim=64, num_layers=4)
    model = FluidPINN(cfg)
    domain = FluidDomain()

    x, t = domain.sample_interior(500)
    derivs = compute_derivatives(model, x, t)
    residuals = navier_stokes_residual(derivs, nu=1e-3, gravity=(0, -9.81, 0))
    total_before, details_before = physics_loss(residuals)

    loss_before = total_before.item()
    print(f"    Loss before training: {loss_before:.6f}")
    check(loss_before > 0.1, f"Initial loss is substantial ({loss_before:.4f} > 0.1)")

    # Print individual residual magnitudes
    for key in ("momentum_x", "momentum_y", "momentum_z", "continuity"):
        print(f"    {key} MSE: {details_before[key].item():.6f}")

    # -- Part B: verify gravity appears in momentum_y --
    # Without gravity, momentum_y should be different
    print("  -- Part B: gravity affects momentum_y --")
    res_no_grav = navier_stokes_residual(derivs, nu=1e-3, gravity=(0, 0, 0))
    _, det_no_grav = physics_loss(res_no_grav)
    _, det_with_grav = physics_loss(residuals)

    # momentum_y should differ between gravity and no-gravity cases
    # (gravity adds a constant -9.81 to the y-momentum residual)
    diff_y = abs(det_with_grav["momentum_y"].item() - det_no_grav["momentum_y"].item())
    check(diff_y > 1.0, f"Gravity changes momentum_y MSE significantly (diff={diff_y:.4f})")

    # momentum_x should be identical (gravity only in y)
    diff_x = abs(det_with_grav["momentum_x"].item() - det_no_grav["momentum_x"].item())
    check(diff_x < 1e-5, f"Gravity does not affect momentum_x (diff={diff_x:.2e})")

    # -- Part C: training for 100 steps reduces loss --
    print("  -- Part C: training reduces physics loss --")
    model_train = FluidPINN(PINNConfig(activation="siren", hidden_dim=64, num_layers=4))
    optimizer = torch.optim.Adam(model_train.parameters(), lr=1e-4)

    # Record initial loss
    x0, t0 = domain.sample_interior(256)
    d0 = compute_derivatives(model_train, x0, t0)
    r0 = navier_stokes_residual(d0, nu=1e-3, gravity=(0, -9.81, 0))
    loss_init, _ = physics_loss(r0)
    loss_init_val = loss_init.item()

    # Reset graph
    model_train.zero_grad()
    optimizer.zero_grad()

    # Training loop
    losses = []
    for step in range(100):
        optimizer.zero_grad()
        x_s, t_s = domain.sample_interior(256)
        d_s = compute_derivatives(model_train, x_s, t_s)
        r_s = navier_stokes_residual(d_s, nu=1e-3, gravity=(0, -9.81, 0))
        loss_s, _ = physics_loss(r_s)
        loss_s.backward()
        optimizer.step()
        losses.append(loss_s.item())

    loss_final = losses[-1]
    print(f"    Loss: {loss_init_val:.4f} (init) -> {loss_final:.4f} (step 100)")

    # Use average of last 10 steps vs first 10 for robust comparison
    avg_first_10 = sum(losses[:10]) / 10
    avg_last_10 = sum(losses[-10:]) / 10
    check(
        avg_last_10 < avg_first_10,
        f"Loss decreased: first-10 avg={avg_first_10:.4f} -> last-10 avg={avg_last_10:.4f}",
    )

    # Check monotonic-ish trend (allow some noise)
    mid = len(losses) // 2
    avg_first_half = sum(losses[:mid]) / mid
    avg_second_half = sum(losses[mid:]) / (len(losses) - mid)
    check(
        avg_second_half < avg_first_half,
        f"Second half avg ({avg_second_half:.4f}) < first half avg ({avg_first_half:.4f})",
    )


# =====================================================================
# Step 6: Vorticity test
# =====================================================================
def step6_vorticity():
    print("\n=== Step 6: Vorticity test ===")

    cfg = PINNConfig(activation="siren", hidden_dim=64, num_layers=4)
    model = FluidPINN(cfg)

    B = 200
    x = torch.randn(B, 3, requires_grad=True)
    t = torch.randn(B, 1, requires_grad=True)

    derivs = compute_derivatives(model, x, t)
    vorticity = compute_vorticity(derivs)

    # -- Shape --
    check(vorticity.shape == (B, 3), f"vorticity shape = ({B}, 3)")

    # -- NaN --
    check(no_nan(vorticity), "vorticity has no NaN")
    check(no_inf(vorticity), "vorticity has no Inf")

    # -- Non-zero --
    check(not_all_zero(vorticity), "vorticity is not all zeros")

    # -- Components match definition --
    print("  -- Verify curl definition --")
    # omega_x = dw/dy - dv/dz
    omega_x_manual = derivs["dw_dy"] - derivs["dv_dz"]
    err_x = (vorticity[:, 0:1] - omega_x_manual).abs().max().item()
    check(err_x < 1e-7, f"omega_x = dw/dy - dv/dz (err={err_x:.2e})")

    # omega_y = du/dz - dw/dx
    omega_y_manual = derivs["du_dz"] - derivs["dw_dx"]
    err_y = (vorticity[:, 1:2] - omega_y_manual).abs().max().item()
    check(err_y < 1e-7, f"omega_y = du/dz - dw/dx (err={err_y:.2e})")

    # omega_z = dv/dx - du/dy
    omega_z_manual = derivs["dv_dx"] - derivs["du_dy"]
    err_z = (vorticity[:, 2:3] - omega_z_manual).abs().max().item()
    check(err_z < 1e-7, f"omega_z = dv/dx - du/dy (err={err_z:.2e})")

    # -- Magnitude summary --
    print(
        f"    vorticity magnitude: mean={vorticity.norm(dim=-1).mean().item():.6f}, "
        f"max={vorticity.norm(dim=-1).max().item():.6f}"
    )

    # -- Autograd through vorticity --
    print("  -- Autograd through vorticity --")
    loss_vort = (vorticity**2).mean()
    model.zero_grad()
    loss_vort.backward()
    n_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().max().item() > 0
    )
    n_total = sum(1 for _ in model.parameters())
    # Output bias has zero spatial derivative (d(const)/dx = 0), so vorticity
    # loss cannot reach it. All other params should receive gradients.
    check(
        n_grad >= n_total - 1,
        f"Vorticity loss reaches at least {n_total - 1}/{n_total} params ({n_grad}/{n_total})",
    )


# =====================================================================
# Step 7: Viscosity sensitivity
# =====================================================================
def step7_viscosity_sensitivity():
    print("\n=== Step 7: Viscosity sensitivity ===")

    cfg = PINNConfig(activation="siren", hidden_dim=64, num_layers=4)
    model = FluidPINN(cfg)

    B = 100
    x = torch.randn(B, 3, requires_grad=True)
    t = torch.randn(B, 1, requires_grad=True)
    derivs = compute_derivatives(model, x, t)

    # Different viscosities should yield different residuals
    res_low = navier_stokes_residual(derivs, nu=1e-6, gravity=(0, 0, 0))
    res_high = navier_stokes_residual(derivs, nu=1.0, gravity=(0, 0, 0))

    # The nu * laplacian term scales with nu, so high-nu has larger diffusion contribution
    diff_x = (res_low["momentum_x"] - res_high["momentum_x"]).abs().max().item()
    check(diff_x > 1e-6, f"Different viscosities give different momentum_x (max_diff={diff_x:.6f})")

    # Continuity should be identical (no nu dependence)
    cont_diff = (res_low["continuity"] - res_high["continuity"]).abs().max().item()
    check(cont_diff < 1e-7, f"Continuity is independent of viscosity (diff={cont_diff:.2e})")


# =====================================================================
# Step 8: Gradient checkpointing equivalence
# =====================================================================
def step8_checkpointing():
    print("\n=== Step 8: Gradient checkpointing equivalence ===")

    torch.manual_seed(42)
    cfg = PINNConfig(activation="siren", hidden_dim=64, num_layers=4)
    model = FluidPINN(cfg)

    B = 50
    x = torch.randn(B, 3, requires_grad=True)
    t = torch.randn(B, 1, requires_grad=True)

    # Without checkpointing
    derivs_normal = compute_derivatives(model, x, t, use_checkpointing=False)
    res_normal = navier_stokes_residual(derivs_normal, nu=1e-3)
    total_normal, _ = physics_loss(res_normal)

    model.zero_grad()
    total_normal.backward()
    grads_normal = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }

    # With checkpointing (need fresh x, t since graph was consumed)
    x2 = x.detach().clone().requires_grad_(True)
    t2 = t.detach().clone().requires_grad_(True)
    derivs_ckpt = compute_derivatives(model, x2, t2, use_checkpointing=True)
    res_ckpt = navier_stokes_residual(derivs_ckpt, nu=1e-3)
    total_ckpt, _ = physics_loss(res_ckpt)

    model.zero_grad()
    total_ckpt.backward()
    grads_ckpt = {
        name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None
    }

    # Loss values should match (same model, same input)
    loss_diff = abs(total_normal.item() - total_ckpt.item())
    check(loss_diff < 1e-5, f"Checkpointed loss matches normal (diff={loss_diff:.2e})")

    # Gradients should match
    max_grad_diff = 0.0
    for name in grads_normal:
        if name in grads_ckpt:
            diff = (grads_normal[name] - grads_ckpt[name]).abs().max().item()
            max_grad_diff = max(max_grad_diff, diff)
    check(
        max_grad_diff < 1e-4, f"Checkpointed gradients match normal (max_diff={max_grad_diff:.2e})"
    )


# =====================================================================
# Step 9: Domain sampling integration
# =====================================================================
def step9_domain_integration():
    print("\n=== Step 9: Domain sampling integration ===")

    cfg = PINNConfig(activation="siren", hidden_dim=64, num_layers=4)
    model = FluidPINN(cfg)
    domain = FluidDomain(x_range=(-1, 1), y_range=(-1, 1), z_range=(-1, 1), t_range=(0, 1))

    # Interior
    x_int, t_int = domain.sample_interior(300)
    d_int = compute_derivatives(model, x_int, t_int)
    r_int = navier_stokes_residual(d_int, nu=1e-3)
    total_int, _ = physics_loss(r_int)
    check(total_int.item() > 0, f"Interior loss > 0 ({total_int.item():.4f})")

    # Boundary (no-slip: velocity should be zero on walls)
    x_bnd, t_bnd = domain.sample_boundary(100, "y_min")
    d_bnd = compute_derivatives(model, x_bnd, t_bnd)
    boundary_vel = torch.cat([d_bnd["u"], d_bnd["v"], d_bnd["w"]], dim=-1)
    boundary_loss = (boundary_vel**2).mean()
    check(
        boundary_loss.item() >= 0, f"Boundary velocity loss computable ({boundary_loss.item():.6f})"
    )

    # Backward from combined loss
    model.zero_grad()
    combined = total_int + boundary_loss
    combined.backward()
    has_grad = all(p.grad is not None and p.grad.abs().max().item() > 0 for p in model.parameters())
    check(has_grad, "Combined interior + boundary loss reaches all params")

    # Initial condition
    x_init, t_init = domain.sample_initial(100)
    d_init = compute_derivatives(model, x_init, t_init)
    check(no_nan(d_init["u"]), "Initial condition: u has no NaN")
    check(d_init["u"].shape == (100, 1), "Initial condition: u shape correct")


# =====================================================================
# Step 10: Device test
# =====================================================================
def step10_device():
    print("\n=== Step 10: Device test ===")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"  Using device: {device}")

    cfg = PINNConfig(activation="siren", hidden_dim=64, num_layers=4)
    model = FluidPINN(cfg).to(device)

    B = 50
    x = torch.randn(B, 3, device=device, requires_grad=True)
    t = torch.randn(B, 1, device=device, requires_grad=True)

    derivs = compute_derivatives(model, x, t)
    check(derivs["u"].device.type == device.type, f"Derivatives on {device.type}")

    residuals = navier_stokes_residual(derivs, nu=1e-3)
    check(residuals["momentum_x"].device.type == device.type, f"Residuals on {device.type}")

    total, _ = physics_loss(residuals)
    check(total.device.type == device.type, f"Loss on {device.type}")

    vort = compute_vorticity(derivs)
    check(vort.device.type == device.type, f"Vorticity on {device.type}")

    model.zero_grad()
    total.backward()
    n_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().max().item() > 0
    )
    n_total = sum(1 for _ in model.parameters())
    check(n_grad == n_total, f"Full backward on {device.type}: {n_grad}/{n_total} params")


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("  TEST: pinn/navier_stokes.py")
    print("=" * 70)

    model, derivatives = step1_derivative_computation()
    residuals = step2_ns_residual(derivatives)
    total = step3_physics_loss(residuals)
    step4_gradient_test()
    step5_known_solution()
    step6_vorticity()
    step7_viscosity_sensitivity()
    step8_checkpointing()
    step9_domain_integration()
    step10_device()

    print(f"\n{'=' * 70}")
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 70}")

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: navier_stokes.py"
            " — all derivatives computed, residuals non-zero for untrained network,"
            " loss backpropagates to weights, vorticity computable"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
