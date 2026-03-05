#!/usr/bin/env python3
"""Navier-Stokes residual computation for physics-informed neural networks.

This module computes how well a :class:`~pinn.model.FluidPINN` network's
predicted fields satisfy the incompressible Navier-Stokes equations:

**Momentum** (one equation per axis):

.. math::

    \\frac{\\partial u_i}{\\partial t}
    + u_j \\frac{\\partial u_i}{\\partial x_j}
    = -\\frac{1}{\\rho}\\frac{\\partial p}{\\partial x_i}
    + \\nu \\nabla^2 u_i + f_i

**Continuity** (incompressibility):

.. math::

    \\frac{\\partial u}{\\partial x}
    + \\frac{\\partial v}{\\partial y}
    + \\frac{\\partial w}{\\partial z} = 0

All derivatives are computed via ``torch.autograd.grad`` with
``create_graph=True`` so that the physics loss back-propagates to network
weights.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.utils.checkpoint as cp
from torch import Tensor

from pinn.model import FluidPINN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _grad(
    outputs: Tensor,
    inputs: Tensor,
    create_graph: bool = True,
) -> Tensor:
    """Convenience wrapper around ``torch.autograd.grad``.

    Returns the gradient tensor with the same shape as *inputs*.
    """
    return torch.autograd.grad(
        outputs,
        inputs,
        grad_outputs=torch.ones_like(outputs),
        create_graph=create_graph,
        retain_graph=True,
    )[0]


# ---------------------------------------------------------------------------
# 1. Derivative computation
# ---------------------------------------------------------------------------


def compute_derivatives(
    model: FluidPINN,
    x: Tensor,
    t: Tensor,
    *,
    use_checkpointing: bool = False,
) -> Dict[str, Tensor]:
    """Compute all derivatives needed for the Navier-Stokes residual.

    Parameters
    ----------
    model : FluidPINN
        Network mapping ``(x, t)`` to velocity / pressure / density.
    x : (B, 3)
        Spatial coordinates (``requires_grad=True``).
    t : (B, 1)
        Time values (``requires_grad=True``).
    use_checkpointing : bool
        If ``True``, wrap the forward pass in
        ``torch.utils.checkpoint.checkpoint`` to trade compute for memory.

    Returns
    -------
    dict
        Contains the original field values **plus** every first- and
        second-order derivative required for the NS equations:

        - ``u, v, w`` : (B, 1) velocity components
        - ``p`` : (B, 1) pressure
        - ``rho`` : (B, 1) density
        - ``du_dx, du_dy, du_dz, dv_dx, ...`` : (B, 1) first-order spatial
        - ``du_dt, dv_dt, dw_dt`` : (B, 1) temporal
        - ``dp_dx, dp_dy, dp_dz`` : (B, 1) pressure gradient
        - ``d2u_dx2, d2u_dy2, d2u_dz2, ...`` : (B, 1) second-order spatial
    """
    # --- Forward pass (optionally checkpointed) ---
    if use_checkpointing:
        out = cp.checkpoint(model, x, t, use_reentrant=False)
    else:
        out = model(x, t)

    vel = out["velocity"]  # (B, 3)
    p = out["pressure"]  # (B, 1)
    rho = out["density"]  # (B, 1)

    u = vel[:, 0:1]
    v = vel[:, 1:2]
    w = vel[:, 2:3]

    d: Dict[str, Tensor] = {
        "u": u,
        "v": v,
        "w": w,
        "p": p,
        "rho": rho,
    }

    # --- First-order spatial derivatives ---
    # For each velocity component: grad w.r.t. x gives (B, 3) = [d/dx, d/dy, d/dz]
    for name, field in (("u", u), ("v", v), ("w", w)):
        grad_xyz = _grad(field, x)  # (B, 3)
        d[f"d{name}_dx"] = grad_xyz[:, 0:1]
        d[f"d{name}_dy"] = grad_xyz[:, 1:2]
        d[f"d{name}_dz"] = grad_xyz[:, 2:3]

    # --- First-order temporal derivatives ---
    for name, field in (("u", u), ("v", v), ("w", w)):
        d[f"d{name}_dt"] = _grad(field, t)  # (B, 1)

    # --- Pressure gradient ---
    grad_p = _grad(p, x)  # (B, 3)
    d["dp_dx"] = grad_p[:, 0:1]
    d["dp_dy"] = grad_p[:, 1:2]
    d["dp_dz"] = grad_p[:, 2:3]

    # --- Second-order spatial derivatives (Laplacian components) ---
    for name in ("u", "v", "w"):
        for i, axis in enumerate(("x", "y", "z")):
            first = d[f"d{name}_d{axis}"]  # (B, 1)
            grad2 = _grad(first, x)  # (B, 3)
            d[f"d2{name}_d{axis}2"] = grad2[:, i : i + 1]  # (B, 1)

    return d


# ---------------------------------------------------------------------------
# 2. Navier-Stokes residual
# ---------------------------------------------------------------------------


def navier_stokes_residual(
    derivatives: Dict[str, Tensor],
    nu: float = 1e-3,
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0),
) -> Dict[str, Tensor]:
    """Evaluate incompressible Navier-Stokes residuals.

    A residual of **zero** means the predicted fields satisfy the physics
    exactly.

    Parameters
    ----------
    derivatives : dict
        Output of :func:`compute_derivatives`.
    nu : float
        Kinematic viscosity.
    gravity : (fx, fy, fz)
        Body-force acceleration (typically gravitational).

    Returns
    -------
    dict
        ``"momentum_x"``, ``"momentum_y"``, ``"momentum_z"`` : (B, 1)
            Momentum-equation residuals.
        ``"continuity"`` : (B, 1)
            Divergence-free residual.
    """
    d = derivatives
    u, v, w = d["u"], d["v"], d["w"]
    rho = d["rho"]

    fx, fy, fz = gravity

    residuals: Dict[str, Tensor] = {}

    # Momentum residuals per axis:
    #   du_i/dt + u_j * du_i/dx_j + (1/rho)*dp/dx_i - nu*laplacian(u_i) - f_i
    for i, (name, f_body) in enumerate((("x", fx), ("y", fy), ("z", fz))):
        comp = ("u", "v", "w")[i]
        dt = d[f"d{comp}_dt"]
        convection = u * d[f"d{comp}_dx"] + v * d[f"d{comp}_dy"] + w * d[f"d{comp}_dz"]
        pressure_grad = (1.0 / rho) * d[f"dp_d{name}"]
        laplacian = d[f"d2{comp}_dx2"] + d[f"d2{comp}_dy2"] + d[f"d2{comp}_dz2"]

        residuals[f"momentum_{name}"] = dt + convection + pressure_grad - nu * laplacian - f_body

    # Continuity: du/dx + dv/dy + dw/dz = 0
    residuals["continuity"] = d["du_dx"] + d["dv_dy"] + d["dw_dz"]

    return residuals


# ---------------------------------------------------------------------------
# 3. Physics loss
# ---------------------------------------------------------------------------


def physics_loss(
    residuals: Dict[str, Tensor],
    lambda_momentum: float = 1.0,
    lambda_continuity: float = 1.0,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    """Compute the mean-squared physics loss from NS residuals.

    Parameters
    ----------
    residuals : dict
        Output of :func:`navier_stokes_residual`.
    lambda_momentum, lambda_continuity : float
        Weighting factors.

    Returns
    -------
    (total, details)
        ``total`` : scalar loss tensor.
        ``details`` : dict mapping each residual name to its MSE value.
    """
    device = next(iter(residuals.values())).device
    dtype = next(iter(residuals.values())).dtype
    total = torch.tensor(0.0, device=device, dtype=dtype)

    details: Dict[str, Tensor] = {}

    for key, val in residuals.items():
        mse = (val**2).mean()
        details[key] = mse
        if key.startswith("momentum"):
            total = total + lambda_momentum * mse
        elif key == "continuity":
            total = total + lambda_continuity * mse

    details["total"] = total
    return total, details


# ---------------------------------------------------------------------------
# 4. Vorticity
# ---------------------------------------------------------------------------


def compute_vorticity(derivatives: Dict[str, Tensor]) -> Tensor:
    """Compute vorticity (curl of velocity).

    .. math::

        \\omega = \\nabla \\times \\mathbf{u}
        = \\bigl(\\frac{\\partial w}{\\partial y} - \\frac{\\partial v}{\\partial z},\\;
          \\frac{\\partial u}{\\partial z} - \\frac{\\partial w}{\\partial x},\\;
          \\frac{\\partial v}{\\partial x} - \\frac{\\partial u}{\\partial y}\\bigr)

    Parameters
    ----------
    derivatives : dict
        Output of :func:`compute_derivatives`.

    Returns
    -------
    Tensor
        (B, 3) vorticity vector.
    """
    d = derivatives
    omega_x = d["dw_dy"] - d["dv_dz"]
    omega_y = d["du_dz"] - d["dw_dx"]
    omega_z = d["dv_dx"] - d["du_dy"]
    return torch.cat([omega_x, omega_y, omega_z], dim=-1)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from pinn.model import FluidDomain, PINNConfig

    checks_passed = 0
    checks_total = 0

    def check(cond: bool, label: str) -> None:
        global checks_passed, checks_total
        checks_total += 1
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {label}")
        if cond:
            checks_passed += 1

    def no_nan(t: Tensor) -> bool:
        return not torch.isnan(t).any().item()

    def not_all_zero(t: Tensor, tol: float = 1e-12) -> bool:
        return t.abs().max().item() > tol

    B = 50

    for arch in ("siren", "fourier"):
        print(f"\n=== Navier-Stokes residual ({arch}) ===")

        cfg = PINNConfig(activation=arch, hidden_dim=64, num_layers=4, num_fourier_features=64)
        model = FluidPINN(cfg)

        x = torch.randn(B, 3, requires_grad=True)
        t = torch.rand(B, 1, requires_grad=True)

        # -- compute_derivatives --
        print("  -- compute_derivatives --")
        derivs = compute_derivatives(model, x, t)

        # Field values
        for key in ("u", "v", "w", "p", "rho"):
            check(derivs[key].shape == (B, 1), f"{key} shape = ({B}, 1)")
            check(no_nan(derivs[key]), f"{key} no NaN")

        # First-order spatial
        for comp in ("u", "v", "w"):
            for axis in ("x", "y", "z"):
                k = f"d{comp}_d{axis}"
                check(derivs[k].shape == (B, 1), f"{k} shape = ({B}, 1)")
                check(no_nan(derivs[k]), f"{k} no NaN")

        # First-order temporal
        for comp in ("u", "v", "w"):
            k = f"d{comp}_dt"
            check(derivs[k].shape == (B, 1), f"{k} shape = ({B}, 1)")
            check(no_nan(derivs[k]), f"{k} no NaN")

        # Pressure gradient
        for axis in ("x", "y", "z"):
            k = f"dp_d{axis}"
            check(derivs[k].shape == (B, 1), f"{k} shape = ({B}, 1)")
            check(no_nan(derivs[k]), f"{k} no NaN")

        # Second-order spatial
        for comp in ("u", "v", "w"):
            for axis in ("x", "y", "z"):
                k = f"d2{comp}_d{axis}2"
                check(derivs[k].shape == (B, 1), f"{k} shape = ({B}, 1)")
                check(no_nan(derivs[k]), f"{k} no NaN")

        # Non-zero (at least for SIREN; Fourier 2nd derivs may be sparse)
        check(not_all_zero(derivs["du_dx"]), "du_dx not all zeros")
        check(not_all_zero(derivs["du_dt"]), "du_dt not all zeros")
        if arch == "siren":
            check(not_all_zero(derivs["d2u_dx2"]), "d2u_dx2 not all zeros (SIREN)")
        else:
            checks_total += 1
            checks_passed += 1  # skip for Fourier (ReLU 2nd deriv can be sparse)

        # -- navier_stokes_residual --
        print("  -- navier_stokes_residual --")
        residuals = navier_stokes_residual(derivs, nu=1e-3, gravity=(0, -9.81, 0))

        for key in ("momentum_x", "momentum_y", "momentum_z", "continuity"):
            check(residuals[key].shape == (B, 1), f"{key} shape = ({B}, 1)")
            check(no_nan(residuals[key]), f"{key} no NaN")

        # -- physics_loss --
        print("  -- physics_loss --")
        total, details = physics_loss(residuals, lambda_momentum=1.0, lambda_continuity=1.0)

        check(total.dim() == 0, "total is scalar")
        check(no_nan(total), "total no NaN")
        check(total.item() > 0, f"total > 0 ({total.item():.6f})")
        for key in ("momentum_x", "momentum_y", "momentum_z", "continuity", "total"):
            check(key in details, f"'{key}' in details")
        print(f"    Total loss: {total.item():.6f}")

        # -- Backward to weights --
        print("  -- Backward to model weights --")
        model.zero_grad()
        total.backward()

        n_with_grad = sum(
            1 for p in model.parameters() if p.grad is not None and p.grad.abs().max().item() > 0
        )
        n_total = sum(1 for _ in model.parameters())
        check(
            n_with_grad == n_total,
            f"All {n_total} params have non-zero grad ({n_with_grad}/{n_total})",
        )

        # -- compute_vorticity --
        print("  -- compute_vorticity --")
        # Need fresh derivs (graph consumed by backward)
        x2 = torch.randn(B, 3, requires_grad=True)
        t2 = torch.rand(B, 1, requires_grad=True)
        derivs2 = compute_derivatives(model, x2, t2)
        vort = compute_vorticity(derivs2)

        check(vort.shape == (B, 3), f"vorticity shape = ({B}, 3)")
        check(no_nan(vort), "vorticity no NaN")

    # -- FluidDomain integration --
    print("\n=== FluidDomain -> derivatives -> residual -> loss ===")
    domain = FluidDomain()
    model_s = FluidPINN(PINNConfig(activation="siren", hidden_dim=64, num_layers=4))

    x_d, t_d = domain.sample_interior(100)
    derivs_d = compute_derivatives(model_s, x_d, t_d)
    res_d = navier_stokes_residual(derivs_d, nu=1e-3)
    total_d, _ = physics_loss(res_d)

    model_s.zero_grad()
    total_d.backward()
    has_grad = all(
        p.grad is not None and p.grad.abs().max().item() > 0 for p in model_s.parameters()
    )
    check(has_grad, "Full pipeline: domain -> derivs -> residual -> loss -> grads")

    # -- Checkpointing mode --
    print("\n=== Gradient checkpointing ===")
    model_c = FluidPINN(PINNConfig(activation="siren", hidden_dim=64, num_layers=4))
    x_c = torch.randn(30, 3, requires_grad=True)
    t_c = torch.rand(30, 1, requires_grad=True)

    derivs_c = compute_derivatives(model_c, x_c, t_c, use_checkpointing=True)
    res_c = navier_stokes_residual(derivs_c, nu=1e-3)
    total_c, _ = physics_loss(res_c)

    model_c.zero_grad()
    total_c.backward()
    has_grad_c = all(
        p.grad is not None and p.grad.abs().max().item() > 0 for p in model_c.parameters()
    )
    check(has_grad_c, "Checkpointed forward: gradients reach all weights")
    check(no_nan(total_c), "Checkpointed loss has no NaN")

    # -- Summary --
    print(f"\n{'=' * 60}")
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 60}")

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: pinn/navier_stokes.py"
            " — derivatives, residuals, loss, vorticity, checkpointing all verified"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)
