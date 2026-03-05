#!/usr/bin/env python3
"""Thorough tests for pinn/boundary.py — boundary condition loss functions."""

import sys

import torch
from torch import Tensor

sys.path.insert(0, "/Users/makar/fluid-recon")

from pinn.boundary import (
    BoundaryConditionSet,
    BoundarySpec,
    _velocity_jacobian,
    free_surface_loss,
    inflow_loss,
    initial_condition_loss,
    no_slip_loss,
    outflow_loss,
    pressure_reference_loss,
)
from pinn.model import FluidDomain, FluidPINN, PINNConfig

checks_passed = 0
checks_total = 0


def check(cond: bool, label: str) -> None:
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if cond:
        checks_passed += 1


def finite(t: Tensor) -> bool:
    return torch.isfinite(t).all().item()


def has_all_grads(model) -> bool:
    return all(p.grad is not None and p.grad.abs().max().item() > 0 for p in model.parameters())


def count_params(model) -> int:
    return sum(1 for _ in model.parameters())


# ============================================================================
# Setup — small SIREN model (same for all tests unless otherwise stated)
# ============================================================================
cfg = PINNConfig(activation="siren", hidden_dim=64, num_layers=4)
model = FluidPINN(cfg)
n_params = count_params(model)

domain = FluidDomain(
    x_range=(-1.0, 1.0),
    y_range=(-1.0, 1.0),
    z_range=(-1.0, 1.0),
    t_range=(0.0, 1.0),
)

# ============================================================================
# Step 1 — NO-SLIP LOSS TEST
# ============================================================================
print("\n" + "=" * 60)
print("STEP 1: No-slip loss test")
print("=" * 60)

# Generate 100 boundary points on x_min face: x=-1, y and z random, t random
N1 = 100
x_wall, t_wall = domain.sample_boundary(N1, "x_min")

check(x_wall.shape == (N1, 3), f"Wall points shape = ({N1}, 3)")
check(t_wall.shape == (N1, 1), f"Wall time shape = ({N1}, 1)")
check((x_wall[:, 0] == -1.0).all().item(), "All x-coordinates == -1.0 on x_min face")

# Compute loss
l_ns = no_slip_loss(model, x_wall, t_wall)
check(l_ns.dim() == 0, "no_slip_loss returns a scalar")
check(finite(l_ns), "no_slip_loss is finite (no NaN/Inf)")
check(l_ns.item() > 0, f"no_slip_loss > 0 (untrained: {l_ns.item():.6f})")
print(f"    no_slip_loss value: {l_ns.item():.6f}")

# Backward and gradient check
model.zero_grad()
l_ns.backward()
check(
    has_all_grads(model),
    f"Gradients reach all {n_params} model params after no_slip_loss.backward()",
)

# Verify loss is MSE of velocity (sanity: recompute manually)
model.zero_grad()
x_wall2, t_wall2 = domain.sample_boundary(50, "x_min")
out_ns = model(x_wall2, t_wall2)
manual_ns = (out_ns["velocity"] ** 2).mean()
auto_ns = no_slip_loss(model, x_wall2, t_wall2)
check(
    abs(manual_ns.item() - auto_ns.item()) < 1e-6,
    f"no_slip_loss == MSE(velocity) (err={abs(manual_ns.item() - auto_ns.item()):.2e})",
)


# ============================================================================
# Step 2 — INFLOW LOSS TEST
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Inflow loss test")
print("=" * 60)

N2 = 100


def upward_jet(x: Tensor, t: Tensor) -> Tensor:
    """Simple profile: uniform upward velocity (0, 1, 0)."""
    zeros = torch.zeros_like(x[:, 0:1])
    ones = torch.ones_like(x[:, 0:1])
    return torch.cat([zeros, ones, zeros], dim=-1)


x_inflow, t_inflow = domain.sample_boundary(N2, "y_min")
check((x_inflow[:, 1] == -1.0).all().item(), "Inflow points on y_min face (y == -1)")

l_in = inflow_loss(model, x_inflow, t_inflow, upward_jet)
check(l_in.dim() == 0, "inflow_loss returns a scalar")
check(finite(l_in), "inflow_loss is finite")
check(l_in.item() > 0, f"inflow_loss > 0 (network doesn't match jet: {l_in.item():.6f})")
print(f"    inflow_loss value: {l_in.item():.6f}")

# Gradient flow
model.zero_grad()
l_in.backward()
check(has_all_grads(model), f"Gradients reach all {n_params} params after inflow_loss.backward()")

# Verify loss is MSE(velocity - target)
model.zero_grad()
x_in2, t_in2 = domain.sample_boundary(50, "y_min")
out_in = model(x_in2, t_in2)
target_vel = upward_jet(x_in2, t_in2)
manual_in = ((out_in["velocity"] - target_vel) ** 2).mean()
auto_in = inflow_loss(model, x_in2, t_in2, upward_jet)
check(
    abs(manual_in.item() - auto_in.item()) < 1e-6,
    f"inflow_loss == MSE(vel - target) (err={abs(manual_in.item() - auto_in.item()):.2e})",
)


# Different profile: radial function
def parabolic_profile(x: Tensor, t: Tensor) -> Tensor:
    r_sq = x[:, 0:1] ** 2 + x[:, 2:3] ** 2
    v_y = torch.clamp(1.0 - r_sq, min=0.0)
    zeros = torch.zeros_like(v_y)
    return torch.cat([zeros, v_y, zeros], dim=-1)


l_par = inflow_loss(model, x_inflow, t_inflow, parabolic_profile)
check(l_par.dim() == 0 and finite(l_par), "inflow_loss works with parabolic profile")


# ============================================================================
# Step 3 — PRESSURE REFERENCE TEST
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: Pressure reference loss test")
print("=" * 60)

# Single reference point
x_ref = torch.tensor([[0.0, 1.0, 0.0]])
t_ref = torch.tensor([[0.5]])

l_pr = pressure_reference_loss(model, x_ref, t_ref, p_ref=0.0)
check(l_pr.dim() == 0, "pressure_reference_loss returns a scalar")
check(finite(l_pr), "pressure_reference_loss is finite")
check(l_pr.item() >= 0, f"pressure_reference_loss >= 0 ({l_pr.item():.6f})")
print(f"    pressure_reference_loss (p_ref=0): {l_pr.item():.6f}")

# Gradient flow
model.zero_grad()
l_pr.backward()
check(
    has_all_grads(model),
    f"Gradients reach all {n_params} params after pressure_reference_loss.backward()",
)

# Non-zero p_ref
model.zero_grad()
l_pr2 = pressure_reference_loss(model, x_ref, t_ref, p_ref=5.0)
check(l_pr2.item() > 0, f"pressure_ref_loss with p_ref=5.0 > 0 ({l_pr2.item():.4f})")

# Multiple reference points
x_refs = torch.randn(10, 3)
t_refs = torch.rand(10, 1)
l_pr3 = pressure_reference_loss(model, x_refs, t_refs, p_ref=0.0)
check(l_pr3.dim() == 0 and finite(l_pr3), "Works with batch of reference points")

# Verify MSE formula
out_pr = model(x_ref, t_ref)
manual_pr = ((out_pr["pressure"] - 0.0) ** 2).mean()
check(
    abs(manual_pr.item() - l_pr.item()) < 1e-6,
    f"pressure_ref_loss == MSE(p - p_ref) (err={abs(manual_pr.item() - l_pr.item()):.2e})",
)


# ============================================================================
# Step 4 — INITIAL CONDITION TEST
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Initial condition loss test")
print("=" * 60)

N4 = 200
x_init, t_init = domain.sample_initial(N4)

check(x_init.shape == (N4, 3), f"Initial points shape = ({N4}, 3)")
check((t_init == 0.0).all().item(), "All t values == 0 at initial condition")

# Default: fluid at rest (no velocity_initial → penalize non-zero velocity)
l_ic = initial_condition_loss(model, x_init)
check(l_ic.dim() == 0, "initial_condition_loss returns a scalar")
check(finite(l_ic), "initial_condition_loss is finite")
check(
    l_ic.item() > 0,
    f"initial_condition_loss > 0 (untrained predicts non-zero vel: {l_ic.item():.6f})",
)
print(f"    initial_condition_loss (at rest): {l_ic.item():.6f}")

# Gradient flow
model.zero_grad()
l_ic.backward()
check(
    has_all_grads(model),
    f"Gradients reach all {n_params} params after initial_condition_loss.backward()",
)

# With explicit zero velocity target (should match default)
model.zero_grad()
vel_zero = torch.zeros(N4, 3)
l_ic_explicit = initial_condition_loss(model, x_init, velocity_initial=vel_zero)
# The two losses use different random seeds internally (t=0 for both), so compare
# using the same x_init
check(
    abs(l_ic_explicit.item() - l_ic.item()) < 1e-5,
    f"Explicit zero vel ≈ default at-rest (err={abs(l_ic_explicit.item() - l_ic.item()):.2e})",
)

# With density target
rho_target = torch.ones(N4, 1) * 1000.0  # target density
l_ic_rho = initial_condition_loss(model, x_init, density_initial=rho_target)
check(
    l_ic_rho.item() > l_ic.item(),
    f"Adding density target increases loss ({l_ic_rho.item():.4f} > {l_ic.item():.4f})",
)

# Verify t_initial kwarg works
t_custom = torch.full((N4, 1), 0.3)
l_ic_t03 = initial_condition_loss(model, x_init, t_initial=t_custom)
check(
    l_ic_t03.dim() == 0 and finite(l_ic_t03),
    "initial_condition_loss with custom t_initial=0.3 works",
)
# Different t should generally give different loss
# (SIREN is sensitive to input)
check(
    abs(l_ic_t03.item() - l_ic.item()) > 1e-8, "Different t_initial values produce different losses"
)


# ============================================================================
# Step 5 — OUTFLOW (NEUMANN) LOSS TEST
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: Outflow (Neumann) loss test")
print("=" * 60)

N5 = 80
x_out, t_out = domain.sample_boundary(N5, "x_max")
check(x_out.requires_grad, "Outflow x has requires_grad=True")

# With specific normal (x_max face → outward normal [1,0,0])
normal_xmax = torch.tensor([1.0, 0.0, 0.0])
l_out = outflow_loss(model, x_out, t_out, normal=normal_xmax)
check(l_out.dim() == 0, "outflow_loss (with normal) returns scalar")
check(finite(l_out), "outflow_loss (with normal) is finite")
check(l_out.item() > 0, f"outflow_loss > 0 ({l_out.item():.6f})")
print(f"    outflow_loss (normal=[1,0,0]): {l_out.item():.6f}")

model.zero_grad()
l_out.backward()
# Output bias has d(bias)/dx = 0 — spatial-derivative-only losses cannot
# reach the output layer bias.  Expect n_params - 1 at most.
n_out_grad = sum(
    1 for p in model.parameters() if p.grad is not None and p.grad.abs().max().item() > 0
)
check(
    n_out_grad >= n_params - 1,
    f"Outflow grads reach >= {n_params - 1} params ({n_out_grad}/{n_params})"
    " (output bias unreachable from spatial derivatives)",
)

# Without normal (penalizes full Jacobian)
model.zero_grad()
x_out2 = torch.randn(N5, 3, requires_grad=True)
t_out2 = torch.rand(N5, 1)
l_out_no_n = outflow_loss(model, x_out2, t_out2)
check(
    l_out_no_n.dim() == 0 and finite(l_out_no_n), "outflow_loss (no normal) returns finite scalar"
)

# Verify full-Jacobian sum-of-squares >= directional sum-of-squares
# (same points, compare total sums not means — means differ by denominator)
x_shared = torch.randn(N5, 3, requires_grad=True)
t_shared = torch.rand(N5, 1)
l_dir = outflow_loss(model, x_shared, t_shared, normal=normal_xmax)
x_shared2 = x_shared.detach().clone().requires_grad_(True)
l_full = outflow_loss(model, x_shared2, t_shared)
# l_dir = mean over (B,3) = sum/(B*3);  l_full = mean over (B,3,3) = sum/(B*9)
# So total_dir = l_dir * B*3, total_full = l_full * B*9
total_dir = l_dir.item() * N5 * 3
total_full = l_full.item() * N5 * 9
check(
    total_full >= total_dir - 1e-6,
    f"Full Jacobian sum >= directional sum ({total_full:.4f} >= {total_dir:.4f})",
)


# ============================================================================
# Step 6 — FREE SURFACE LOSS TEST
# ============================================================================
print("\n" + "=" * 60)
print("STEP 6: Free surface loss test")
print("=" * 60)

N6 = 60
x_fs = torch.randn(N6, 3, requires_grad=True)
t_fs = torch.rand(N6, 1)
n_surface = torch.tensor([0.0, 1.0, 0.0])

l_fs = free_surface_loss(model, x_fs, t_fs, n_surface)
check(l_fs.dim() == 0, "free_surface_loss returns scalar")
check(finite(l_fs), "free_surface_loss is finite")
check(l_fs.item() > 0, f"free_surface_loss > 0 ({l_fs.item():.6f})")
print(f"    free_surface_loss: {l_fs.item():.6f}")

model.zero_grad()
l_fs.backward()
check(
    has_all_grads(model),
    f"Gradients reach all {n_params} params after free_surface_loss.backward()",
)

# Verify it contains both pressure and stress components
model.zero_grad()
x_fs2 = torch.randn(N6, 3, requires_grad=True)
out_fs, jac_fs = _velocity_jacobian(model, x_fs2, t_fs)
p_only = (out_fs["pressure"] ** 2).mean()
check(p_only.item() > 0, "Pressure component of free_surface_loss > 0")

# Stress component: compute manually
n_exp = n_surface.unsqueeze(0).expand(N6, -1)
eps = 0.5 * (jac_fs + jac_fs.transpose(1, 2))
tau = torch.bmm(eps, n_exp.unsqueeze(-1)).squeeze(-1)
tau_n = (tau * n_exp).sum(dim=-1, keepdim=True)
tau_t = tau - tau_n * n_exp
stress_only = (tau_t**2).mean()
manual_total = p_only + stress_only
auto_total = free_surface_loss(model, x_fs2, t_fs, n_surface)
check(
    abs(manual_total.item() - auto_total.item()) < 1e-5,
    f"free_surface = p_loss + stress_loss (err={abs(manual_total.item() - auto_total.item()):.2e})",
)


# ============================================================================
# Step 7 — VELOCITY JACOBIAN HELPER TEST
# ============================================================================
print("\n" + "=" * 60)
print("STEP 7: _velocity_jacobian helper test")
print("=" * 60)

N7 = 40
x_jac = torch.randn(N7, 3, requires_grad=True)
t_jac = torch.rand(N7, 1)

out_jac, jac = _velocity_jacobian(model, x_jac, t_jac)
check(jac.shape == (N7, 3, 3), f"Jacobian shape = ({N7}, 3, 3)")
check(finite(jac), "Jacobian has no NaN/Inf")
check(jac.requires_grad, "Jacobian tracks gradients (create_graph=True)")

# Verify Jacobian semantics: jac[b, i, j] = du_i / dx_j
# Cross-check against autograd.grad for u component
vel = out_jac["velocity"]
du_dx_auto = torch.autograd.grad(
    vel[:, 0:1], x_jac, torch.ones(N7, 1), create_graph=True, retain_graph=True
)[
    0
]  # (N7, 3) = [du/dx, du/dy, du/dz]
check(
    torch.allclose(jac[:, 0, :], du_dx_auto, atol=1e-5),
    "jac[:, 0, :] matches autograd du/dx (row 0 of Jacobian)",
)

# Auto-enable requires_grad for x without it
x_no_grad = torch.randn(20, 3)  # no requires_grad
t_ng = torch.rand(20, 1)
out_ng, jac_ng = _velocity_jacobian(model, x_no_grad, t_ng)
check(
    jac_ng.shape == (20, 3, 3) and finite(jac_ng),
    "_velocity_jacobian handles x without requires_grad",
)


# ============================================================================
# Step 8 — BOUNDARY CONDITION SET: RISING SMOKE
# ============================================================================
print("\n" + "=" * 60)
print("STEP 8: BoundaryConditionSet — rising_smoke")
print("=" * 60)

bc_smoke = BoundaryConditionSet(domain, scenario="rising_smoke")
check(
    len(bc_smoke.specs) == 8,
    f"rising_smoke has 8 BC specs (4 no-slip + inflow + outflow + pressure_ref + initial): got {len(bc_smoke.specs)}",
)

# Verify spec types
spec_types = [s.bc_type for s in bc_smoke.specs]
check(spec_types.count("no_slip") == 4, "4 no-slip walls (x_min, x_max, z_min, z_max)")
check("inflow" in spec_types, "Has inflow BC")
check("outflow" in spec_types, "Has outflow BC")
check("pressure_ref" in spec_types, "Has pressure reference BC")
check("initial" in spec_types, "Has initial condition BC")

# Verify no-slip faces
no_slip_faces = [s.face for s in bc_smoke.specs if s.bc_type == "no_slip"]
check(
    set(no_slip_faces) == {"x_min", "x_max", "z_min", "z_max"},
    f"No-slip on side walls: {sorted(no_slip_faces)}",
)

# Verify inflow is on y_min (bottom)
inflow_spec = [s for s in bc_smoke.specs if s.bc_type == "inflow"][0]
check(inflow_spec.face == "y_min", "Inflow on y_min (bottom)")
check(inflow_spec.velocity_profile is not None, "Inflow has velocity profile")
check(inflow_spec.density_profile is not None, "Inflow has density profile")

# Verify outflow is on y_max (top)
outflow_spec = [s for s in bc_smoke.specs if s.bc_type == "outflow"][0]
check(outflow_spec.face == "y_max", "Outflow on y_max (top)")
check(outflow_spec.normal is not None, "Outflow has normal vector")

# Compute total boundary loss
model.zero_grad()
total_bc, bc_details = bc_smoke.compute_total_boundary_loss(model, n_points_per_boundary=100)

check(total_bc.dim() == 0, "total_bc is a scalar")
check(finite(total_bc), "total_bc is finite")
check(total_bc.item() > 0, f"total_bc > 0 ({total_bc.item():.4f})")

# Check details dict
expected_keys = {
    "no_slip/x_min",
    "no_slip/x_max",
    "no_slip/z_min",
    "no_slip/z_max",
    "inflow/y_min",
    "outflow/y_max",
    "pressure_ref",
    "initial",
}
check(
    set(bc_details.keys()) == expected_keys,
    f"Details has all expected keys: {sorted(bc_details.keys())}",
)

# All individual losses non-negative
all_nonneg = all(v.item() >= 0 for v in bc_details.values())
check(all_nonneg, "All individual BC losses >= 0")

# Print individual values
print("    Individual BC losses:")
for k, v in bc_details.items():
    print(f"      {k}: {v.item():.6f}")

# Backward and gradient flow
total_bc.backward()
check(has_all_grads(model), f"total_bc.backward() reaches all {n_params} params")


# ============================================================================
# Step 9 — BOUNDARY POINT SAMPLING TEST
# ============================================================================
print("\n" + "=" * 60)
print("STEP 9: Boundary point sampling test")
print("=" * 60)

N9 = 100
samples = bc_smoke.sample_boundary_points(N9)

check(isinstance(samples, list), "sample_boundary_points returns a list")
check(len(samples) == 8, f"8 sample groups for rising_smoke: got {len(samples)}")

for s in samples:
    check(isinstance(s, dict), "Each sample is a dict")
    check("x" in s and "t" in s and "spec" in s, "Dict has keys: x, t, spec")
    break  # structure check on first

# Check requires_grad on sampled points
n_has_xgrad = sum(1 for s in samples if s["x"].requires_grad)
n_has_tgrad = sum(1 for s in samples if s["t"].requires_grad or s["spec"].bc_type == "pressure_ref")
# pressure_ref samples only 1 point via _uniform which may not have requires_grad
# but boundary/initial samples should all have it
n_boundary_samples = sum(1 for s in samples if s["spec"].bc_type != "pressure_ref")
n_xgrad_boundary = sum(
    1 for s in samples if s["spec"].bc_type != "pressure_ref" and s["x"].requires_grad
)
check(
    n_xgrad_boundary == n_boundary_samples,
    f"All non-pressure-ref x tensors have requires_grad ({n_xgrad_boundary}/{n_boundary_samples})",
)

# Verify no-slip boundary points lie on domain faces
face_axis_idx = {"x_min": 0, "x_max": 0, "y_min": 1, "y_max": 1, "z_min": 2, "z_max": 2}
face_val = {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0, "z_min": -1.0, "z_max": 1.0}

for s in samples:
    spec = s["spec"]
    if spec.face is not None and spec.bc_type in ("no_slip", "inflow", "outflow"):
        x_s = s["x"]
        axis = face_axis_idx[spec.face]
        expected = face_val[spec.face]
        actual_vals = x_s[:, axis].detach()
        on_face = (actual_vals == expected).all().item()
        check(
            on_face,
            f"  {spec.bc_type}/{spec.face}: all points on face "
            f"(axis {axis} == {expected}, got range [{actual_vals.min():.4f}, {actual_vals.max():.4f}])",
        )

# Check that non-fixed coordinates are within domain
for s in samples:
    spec = s["spec"]
    if spec.face is not None and spec.bc_type == "no_slip":
        x_s = s["x"].detach()
        axis = face_axis_idx[spec.face]
        for j, (name, rng) in enumerate(
            zip(["x", "y", "z"], [domain.x_range, domain.y_range, domain.z_range])
        ):
            if j == axis:
                continue
            vals = x_s[:, j]
            in_range = (vals >= rng[0]).all().item() and (vals <= rng[1]).all().item()
            check(in_range, f"  {spec.bc_type}/{spec.face}: {name} coords in [{rng[0]}, {rng[1]}]")

# Verify initial condition points have t=0
for s in samples:
    if s["spec"].bc_type == "initial":
        t_vals = s["t"].detach()
        check((t_vals == 0.0).all().item(), "Initial condition sample: all t == 0")
        check(s["x"].shape[0] == N9, f"Initial condition sample has {N9} points")

# Verify pressure_ref is a single point at top centre
for s in samples:
    if s["spec"].bc_type == "pressure_ref":
        check(s["x"].shape[0] == 1, "pressure_ref samples 1 point")
        x_pr = s["x"].detach()
        check(abs(x_pr[0, 0].item() - 0.0) < 1e-6, "pressure_ref x == 0 (centre)")
        check(abs(x_pr[0, 1].item() - 1.0) < 1e-6, "pressure_ref y == 1 (top)")
        check(abs(x_pr[0, 2].item() - 0.0) < 1e-6, "pressure_ref z == 0 (centre)")


# ============================================================================
# Step 10 — SMOKE INFLOW PROFILE VERIFICATION
# ============================================================================
print("\n" + "=" * 60)
print("STEP 10: Smoke inflow profile shape & values")
print("=" * 60)

# Test the smoke velocity profile: should be upward (v_y > 0) near centre,
# zero far from centre
inflow_spec = [s for s in bc_smoke.specs if s.bc_type == "inflow"][0]
prof = inflow_spec.velocity_profile
dens_prof = inflow_spec.density_profile

# Point at centre of bottom face
x_centre = torch.tensor([[0.0, -1.0, 0.0]])
t_any = torch.tensor([[0.5]])
vel_centre = prof(x_centre, t_any)
check(vel_centre.shape == (1, 3), "Profile returns (1, 3)")
check(vel_centre[0, 0].item() == 0.0, "Profile: u=0 at centre")
check(vel_centre[0, 1].item() > 0.0, f"Profile: v>0 at centre (v={vel_centre[0, 1].item():.4f})")
check(vel_centre[0, 2].item() == 0.0, "Profile: w=0 at centre")

# Point far from centre (outside plume radius)
x_far = torch.tensor([[0.9, -1.0, 0.9]])
vel_far = prof(x_far, t_any)
check(vel_far[0, 1].item() == 0.0, "Profile: v=0 far from centre (outside plume)")

# Density profile
rho_centre = dens_prof(x_centre, t_any)
check(rho_centre.item() == 2.0, f"Density at centre == 2.0 (got {rho_centre.item():.2f})")
rho_far = dens_prof(x_far, t_any)
check(rho_far.item() == 1.0, f"Density far from centre == 1.0 (got {rho_far.item():.2f})")


# ============================================================================
# Step 11 — POURING WATER SCENARIO
# ============================================================================
print("\n" + "=" * 60)
print("STEP 11: BoundaryConditionSet — pouring_water")
print("=" * 60)

bc_pour = BoundaryConditionSet(domain, scenario="pouring_water")
check(len(bc_pour.specs) == 9, f"pouring_water has 9 BC specs: got {len(bc_pour.specs)}")

# Verify the mask_fn filters correctly (free_surface only outside stream)
fs_spec = [s for s in bc_pour.specs if s.bc_type == "free_surface"][0]
check(fs_spec.mask_fn is not None, "free_surface has mask_fn")

# Points inside stream should be filtered out
x_in_stream = torch.tensor([[0.0, 1.0, 0.0]])  # centre of y_max
t_any_pour = torch.tensor([[0.5]])
mask_result = fs_spec.mask_fn(x_in_stream, t_any_pour)
check(not mask_result.any().item(), "mask_fn filters out points inside stream")

# Points outside stream should pass
x_out_stream = torch.tensor([[0.8, 1.0, 0.8]])
mask_out = fs_spec.mask_fn(x_out_stream, t_any_pour)
check(mask_out.all().item(), "mask_fn keeps points outside stream")

# Full loss computation
model.zero_grad()
total_pw, details_pw = bc_pour.compute_total_boundary_loss(model, n_points_per_boundary=64)
check(
    total_pw.dim() == 0 and finite(total_pw) and total_pw.item() > 0,
    f"pouring_water total loss valid ({total_pw.item():.4f})",
)
check("free_surface/y_max" in details_pw, "Details includes free_surface/y_max")

total_pw.backward()
check(has_all_grads(model), f"pouring_water grads reach all {n_params} params")


# ============================================================================
# Step 12 — CUSTOM SCENARIO
# ============================================================================
print("\n" + "=" * 60)
print("STEP 12: Custom scenario")
print("=" * 60)

bc_custom = BoundaryConditionSet(domain, scenario="custom")
check(len(bc_custom.specs) == 0, "Custom starts empty")

# Add BCs manually
bc_custom.add_bc(BoundarySpec(bc_type="no_slip", face="y_min", weight=2.0))
bc_custom.add_bc(BoundarySpec(bc_type="no_slip", face="y_max", weight=2.0))
bc_custom.add_bc(
    BoundarySpec(
        bc_type="inflow",
        face="x_min",
        velocity_profile=lambda x, t: torch.cat(
            [torch.ones_like(x[:, 0:1]), torch.zeros_like(x[:, 0:1]), torch.zeros_like(x[:, 0:1])],
            dim=-1,
        ),
    )
)
bc_custom.add_bc(
    BoundarySpec(bc_type="outflow", face="x_max", normal=torch.tensor([1.0, 0.0, 0.0]))
)
bc_custom.add_bc(BoundarySpec(bc_type="initial"))
bc_custom.add_bc(BoundarySpec(bc_type="pressure_ref", weight=0.1, p_ref=0.0))

check(len(bc_custom.specs) == 6, "Custom has 6 BCs after add_bc calls")

model.zero_grad()
total_c, details_c = bc_custom.compute_total_boundary_loss(model, n_points_per_boundary=50)
check(total_c.dim() == 0 and finite(total_c), f"Custom total loss valid ({total_c.item():.4f})")
check(len(details_c) == 6, f"Custom details has 6 entries: got {len(details_c)}")

total_c.backward()
check(has_all_grads(model), f"Custom scenario grads reach all {n_params} params")


# ============================================================================
# Step 13 — WEIGHT SENSITIVITY TEST
# ============================================================================
print("\n" + "=" * 60)
print("STEP 13: Weight sensitivity test")
print("=" * 60)

# Compute with different weights, verify total changes
bc_w1 = BoundaryConditionSet(domain, scenario="custom")
bc_w1.add_bc(BoundarySpec(bc_type="no_slip", face="x_min", weight=1.0))
bc_w1.add_bc(BoundarySpec(bc_type="initial", weight=1.0))

bc_w10 = BoundaryConditionSet(domain, scenario="custom")
bc_w10.add_bc(BoundarySpec(bc_type="no_slip", face="x_min", weight=10.0))
bc_w10.add_bc(BoundarySpec(bc_type="initial", weight=10.0))

# Use the same random seed for fair comparison
torch.manual_seed(42)
total_w1, det_w1 = bc_w1.compute_total_boundary_loss(model, n_points_per_boundary=50)
torch.manual_seed(42)
total_w10, det_w10 = bc_w10.compute_total_boundary_loss(model, n_points_per_boundary=50)

# Individual (unweighted) losses should be the same
for key in det_w1:
    if key in det_w10:
        err = abs(det_w1[key].item() - det_w10[key].item())
        check(err < 1e-5, f"Unweighted loss '{key}' same regardless of weight (err={err:.2e})")

# Total should scale with weight
ratio = total_w10.item() / max(total_w1.item(), 1e-12)
check(abs(ratio - 10.0) < 0.5, f"Total scales ~10x with 10x weights (ratio={ratio:.2f})")


# ============================================================================
# Step 14 — FOURIER ARCHITECTURE COMPATIBILITY
# ============================================================================
print("\n" + "=" * 60)
print("STEP 14: Fourier architecture compatibility")
print("=" * 60)

cfg_f = PINNConfig(activation="fourier", hidden_dim=64, num_layers=4, num_fourier_features=64)
model_f = FluidPINN(cfg_f)
n_params_f = count_params(model_f)

# Run all BC types with Fourier model
x_f = torch.randn(50, 3, requires_grad=True)
t_f = torch.rand(50, 1)

l_ns_f = no_slip_loss(model_f, x_f, t_f)
l_in_f = inflow_loss(model_f, x_f, t_f, upward_jet)
l_out_f = outflow_loss(model_f, x_f, t_f, normal=torch.tensor([0.0, 0.0, 1.0]))
l_pr_f = pressure_reference_loss(model_f, x_f[:1], t_f[:1])
l_fs_f = free_surface_loss(model_f, x_f, t_f, torch.tensor([0.0, 1.0, 0.0]))
l_ic_f = initial_condition_loss(model_f, x_f.detach())

all_finite = all(finite(l) for l in [l_ns_f, l_in_f, l_out_f, l_pr_f, l_fs_f, l_ic_f])
check(all_finite, "All BC losses finite with Fourier model")

# Full scenario
model_f.zero_grad()
bc_f = BoundaryConditionSet(domain, scenario="rising_smoke")
total_f, _ = bc_f.compute_total_boundary_loss(model_f, n_points_per_boundary=50)
total_f.backward()

n_grad_f = sum(
    1 for p in model_f.parameters() if p.grad is not None and p.grad.abs().max().item() > 0
)
check(
    n_grad_f == n_params_f,
    f"Fourier: rising_smoke grads reach all {n_params_f} params ({n_grad_f}/{n_params_f})",
)


# ============================================================================
# Step 15 — ERROR HANDLING
# ============================================================================
print("\n" + "=" * 60)
print("STEP 15: Error handling")
print("=" * 60)

try:
    BoundaryConditionSet(domain, scenario="nonexistent")
    check(False, "Should raise ValueError for unknown scenario")
except ValueError as e:
    check("nonexistent" in str(e), f"ValueError mentions bad scenario name: {e}")

# Unknown BC type in compute
bc_bad = BoundaryConditionSet(domain, scenario="custom")
bc_bad.add_bc(BoundarySpec(bc_type="magic_wall", face="x_min"))
try:
    bc_bad.compute_total_boundary_loss(model, n_points_per_boundary=10)
    check(False, "Should raise ValueError for unknown BC type")
except ValueError as e:
    check("magic_wall" in str(e), f"ValueError mentions bad BC type: {e}")


# ============================================================================
# Summary
# ============================================================================
print(f"\n{'=' * 60}")
print(f"  Results: {checks_passed}/{checks_total} checks passed")
print(f"{'=' * 60}")

if checks_passed == checks_total:
    print(
        "\nTEST PASSED: boundary.py — all BC types produce valid losses"
        " with gradient flow, BoundaryConditionSet works for rising_smoke scenario"
    )
else:
    print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
    sys.exit(1)
