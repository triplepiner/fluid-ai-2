#!/usr/bin/env python3
"""Advection of Gaussian positions using the PINN velocity field.

The PINN provides a continuous velocity field u(x, t) that satisfies the
Navier-Stokes equations.  This module uses that field to transport Gaussian
positions forward (or backward) in time via numerical integration.

Key components:

- :func:`advect_euler` — single forward-Euler step.
- :func:`advect_rk4` — single fourth-order Runge-Kutta step.
- :func:`advect_trajectory` — multi-step trajectory integration.
- :class:`PhysicsGuidedDeformation` — drop-in replacement for
  :class:`DeformationNetwork` that uses PINN advection instead of a
  learned deformation MLP.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# 1. Single-step integrators
# ---------------------------------------------------------------------------


def _query_velocity(
    pinn: nn.Module,
    xyz: Tensor,
    t_val: float | Tensor,
) -> Tensor:
    """Query the PINN velocity field at given positions and time.

    Parameters
    ----------
    pinn : FluidPINN
        Trained PINN model.
    xyz : (N, 3)
        Spatial positions.
    t_val : float or scalar Tensor
        Time value (broadcast to all points).

    Returns
    -------
    (N, 3) velocity vectors.
    """
    N = xyz.shape[0]
    device = xyz.device

    if isinstance(t_val, (int, float)):
        t = torch.full((N, 1), t_val, device=device, dtype=xyz.dtype)
    else:
        t = t_val.reshape(1, 1).expand(N, 1).to(device=device, dtype=xyz.dtype)

    with torch.no_grad():
        out = pinn(xyz, t)
    return out["velocity"]  # (N, 3)


def advect_euler(
    pinn: nn.Module,
    xyz: Tensor,
    t: float,
    dt: float,
) -> Tensor:
    """Single forward-Euler advection step.

    .. math::
        x_{n+1} = x_n + u(x_n, t_n) \\cdot \\Delta t

    Parameters
    ----------
    pinn : FluidPINN
    xyz : (N, 3) current positions.
    t : float — current time.
    dt : float — time step (can be negative for backward advection).

    Returns
    -------
    (N, 3) advected positions.
    """
    vel = _query_velocity(pinn, xyz, t)
    return xyz + vel * dt


def advect_rk4(
    pinn: nn.Module,
    xyz: Tensor,
    t: float,
    dt: float,
) -> Tensor:
    """Single fourth-order Runge-Kutta advection step.

    .. math::
        k_1 &= u(x_n,\\; t_n) \\\\
        k_2 &= u(x_n + \\tfrac{dt}{2} k_1,\\; t_n + \\tfrac{dt}{2}) \\\\
        k_3 &= u(x_n + \\tfrac{dt}{2} k_2,\\; t_n + \\tfrac{dt}{2}) \\\\
        k_4 &= u(x_n + dt\\, k_3,\\; t_n + dt) \\\\
        x_{n+1} &= x_n + \\tfrac{dt}{6}(k_1 + 2k_2 + 2k_3 + k_4)

    Parameters
    ----------
    pinn : FluidPINN
    xyz : (N, 3) current positions.
    t : float — current time.
    dt : float — time step (can be negative for backward advection).

    Returns
    -------
    (N, 3) advected positions.
    """
    k1 = _query_velocity(pinn, xyz, t)
    k2 = _query_velocity(pinn, xyz + 0.5 * dt * k1, t + 0.5 * dt)
    k3 = _query_velocity(pinn, xyz + 0.5 * dt * k2, t + 0.5 * dt)
    k4 = _query_velocity(pinn, xyz + dt * k3, t + dt)
    return xyz + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


# ---------------------------------------------------------------------------
# 2. Multi-step trajectory integration
# ---------------------------------------------------------------------------


def advect_trajectory(
    pinn: nn.Module,
    xyz_start: Tensor,
    t_start: float,
    t_end: float,
    num_steps: int,
    method: str = "rk4",
) -> Tensor:
    """Integrate particle trajectories through the PINN velocity field.

    Parameters
    ----------
    pinn : FluidPINN
    xyz_start : (N, 3) starting positions.
    t_start : float — start time.
    t_end : float — end time.
    num_steps : int — number of integration steps.
    method : ``"euler"`` or ``"rk4"`` (default).

    Returns
    -------
    (num_steps + 1, N, 3) trajectory tensor.
        ``trajectory[0]`` = ``xyz_start``,
        ``trajectory[-1]`` = final advected positions.
    """
    if method == "euler":
        step_fn = advect_euler
    elif method == "rk4":
        step_fn = advect_rk4
    else:
        raise ValueError(f"Unknown method {method!r}. Use 'euler' or 'rk4'.")

    dt = (t_end - t_start) / num_steps
    trajectory = [xyz_start.clone()]
    xyz = xyz_start.clone()

    for i in range(num_steps):
        t_current = t_start + i * dt
        xyz = step_fn(pinn, xyz, t_current, dt)
        trajectory.append(xyz.clone())

    return torch.stack(trajectory, dim=0)  # (num_steps+1, N, 3)


# ---------------------------------------------------------------------------
# 3. PhysicsGuidedDeformation
# ---------------------------------------------------------------------------


class PhysicsGuidedDeformation(nn.Module):
    """Drop-in replacement for DeformationNetwork using PINN advection.

    Instead of learning deformation deltas with an MLP, this module advects
    canonical Gaussian positions from ``t_canonical`` to the query time
    using the trained PINN velocity field.

    The rotation, scaling, and opacity deltas are set to zero — the
    physics-based transport only moves the Gaussian centres.  The
    canonical rotation/scale/opacity are preserved as-is.

    Parameters
    ----------
    pinn : FluidPINN
        Trained PINN model providing the velocity field.
    t_canonical : float
        The time corresponding to the canonical Gaussian frame (usually 0.0).
    num_advection_steps : int
        Number of RK4 sub-steps for advection.
    method : str
        Integration method (``"rk4"`` or ``"euler"``).
    """

    def __init__(
        self,
        pinn: nn.Module,
        t_canonical: float = 0.0,
        num_advection_steps: int = 10,
        method: str = "rk4",
    ) -> None:
        super().__init__()
        self.pinn = pinn
        self.t_canonical = t_canonical
        self.num_advection_steps = num_advection_steps
        self.method = method

    def forward(
        self,
        canonical_xyz: Tensor,
        time: float | Tensor,
    ) -> Dict[str, Tensor]:
        """Compute deformation deltas via PINN advection.

        Matches the :class:`DeformationNetwork` interface exactly.

        Parameters
        ----------
        canonical_xyz : (N, 3)
            Canonical Gaussian positions.
        time : float or Tensor
            Target normalised time in [0, 1].

        Returns
        -------
        dict
            ``"delta_xyz"`` (N, 3), ``"delta_rotation"`` (N, 4),
            ``"delta_scaling"`` (N, 3), ``"delta_opacity"`` (N, 1).
        """
        N = canonical_xyz.shape[0]
        device = canonical_xyz.device

        if isinstance(time, Tensor):
            t_target = time.item()
        else:
            t_target = float(time)

        # Advect from canonical time to target time
        if abs(t_target - self.t_canonical) < 1e-8:
            # No advection needed at canonical time
            delta_xyz = torch.zeros(N, 3, device=device, dtype=canonical_xyz.dtype)
        else:
            # Integrate trajectory: canonical_xyz at t_canonical -> t_target
            trajectory = advect_trajectory(
                self.pinn,
                canonical_xyz.detach(),
                t_start=self.t_canonical,
                t_end=t_target,
                num_steps=self.num_advection_steps,
                method=self.method,
            )
            advected = trajectory[-1]  # (N, 3)
            delta_xyz = advected - canonical_xyz.detach()

        # Rotation, scaling, opacity: no physics-based delta
        delta_rotation = torch.zeros(N, 4, device=device, dtype=canonical_xyz.dtype)
        delta_scaling = torch.zeros(N, 3, device=device, dtype=canonical_xyz.dtype)
        delta_opacity = torch.zeros(N, 1, device=device, dtype=canonical_xyz.dtype)

        return {
            "delta_xyz": delta_xyz,
            "delta_rotation": delta_rotation,
            "delta_scaling": delta_scaling,
            "delta_opacity": delta_opacity,
        }


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

    from pinn.model import FluidPINN, PINNConfig

    checks_passed = 0
    checks_total = 0

    def check(cond: bool, label: str) -> None:
        global checks_passed, checks_total
        checks_total += 1
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {label}")
        if cond:
            checks_passed += 1

    print("=" * 60)
    print("  advection.py — self-test")
    print("=" * 60)

    # Create a tiny PINN
    pinn = FluidPINN(PINNConfig(activation="siren", hidden_dim=32, num_layers=4))
    pinn.eval()

    N = 100
    xyz = torch.randn(N, 3) * 0.5

    # -- Euler step --
    print("\n--- Euler ---")
    xyz_e = advect_euler(pinn, xyz, t=0.0, dt=0.01)
    check(xyz_e.shape == (N, 3), f"Euler shape = ({N}, 3)")
    check(torch.isfinite(xyz_e).all(), "Euler output finite")
    check(not torch.equal(xyz_e, xyz), "Euler moved points")

    # -- RK4 step --
    print("\n--- RK4 ---")
    xyz_r = advect_rk4(pinn, xyz, t=0.0, dt=0.01)
    check(xyz_r.shape == (N, 3), f"RK4 shape = ({N}, 3)")
    check(torch.isfinite(xyz_r).all(), "RK4 output finite")
    check(not torch.equal(xyz_r, xyz), "RK4 moved points")

    # RK4 and Euler should give slightly different results
    diff = (xyz_e - xyz_r).abs().max().item()
    check(diff > 0, f"Euler vs RK4 differ (max diff={diff:.2e})")

    # -- Trajectory --
    print("\n--- Trajectory ---")
    traj = advect_trajectory(pinn, xyz, t_start=0.0, t_end=1.0, num_steps=20)
    check(traj.shape == (21, N, 3), f"Trajectory shape = (21, {N}, 3)")
    check(torch.equal(traj[0], xyz), "Trajectory[0] == start")
    check(not torch.equal(traj[-1], xyz), "Trajectory[-1] != start")
    check(torch.isfinite(traj).all(), "Trajectory finite")

    # Euler trajectory
    traj_e = advect_trajectory(pinn, xyz, 0.0, 1.0, 20, method="euler")
    check(traj_e.shape == (21, N, 3), "Euler trajectory shape")

    # -- PhysicsGuidedDeformation --
    print("\n--- PhysicsGuidedDeformation ---")
    deform = PhysicsGuidedDeformation(pinn, t_canonical=0.0, num_advection_steps=5)

    # At canonical time, delta should be zero
    out_0 = deform(xyz, time=0.0)
    check(out_0["delta_xyz"].shape == (N, 3), "delta_xyz shape at t=0")
    check((out_0["delta_xyz"].abs() < 1e-6).all(), "delta_xyz ~ 0 at canonical time")
    check((out_0["delta_rotation"] == 0).all(), "delta_rotation = 0")
    check((out_0["delta_scaling"] == 0).all(), "delta_scaling = 0")
    check((out_0["delta_opacity"] == 0).all(), "delta_opacity = 0")

    # At a different time, delta should be non-zero
    out_1 = deform(xyz, time=0.5)
    check(out_1["delta_xyz"].shape == (N, 3), "delta_xyz shape at t=0.5")
    check(out_1["delta_xyz"].abs().max() > 1e-6, "delta_xyz non-zero at t=0.5")
    check(torch.isfinite(out_1["delta_xyz"]).all(), "delta_xyz finite at t=0.5")

    # Tensor time input
    out_t = deform(xyz, time=torch.tensor(0.5))
    check(torch.allclose(out_t["delta_xyz"], out_1["delta_xyz"]), "Tensor time matches float time")

    # -- Backward advection --
    print("\n--- Backward advection ---")
    traj_back = advect_trajectory(pinn, xyz, t_start=1.0, t_end=0.0, num_steps=10)
    check(traj_back.shape == (11, N, 3), "Backward trajectory shape")
    check(torch.isfinite(traj_back).all(), "Backward trajectory finite")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 60}")

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: advection.py"
            " — Euler, RK4, trajectory, PhysicsGuidedDeformation all verified"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)
