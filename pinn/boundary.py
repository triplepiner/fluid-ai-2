#!/usr/bin/env python3
"""Boundary condition loss functions for physics-informed neural networks.

Loss functions
--------------
- :func:`no_slip_loss` — velocity = 0 at solid walls.
- :func:`inflow_loss` — velocity matches a prescribed profile.
- :func:`outflow_loss` — zero-gradient (Neumann) condition: du/dn = 0.
- :func:`pressure_reference_loss` — fix pressure at a reference point.
- :func:`free_surface_loss` — p = 0 and zero tangential stress.
- :func:`initial_condition_loss` — match prescribed initial state at t = 0.

Manager class
-------------
- :class:`BoundaryConditionSet` — manages multiple BCs for predefined or
  custom physical scenarios.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor

from pinn.model import FluidDomain, FluidPINN

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _velocity_jacobian(
    model: FluidPINN,
    x: Tensor,
    t: Tensor,
) -> Tuple[Dict[str, Tensor], Tensor]:
    """Run model forward and compute velocity spatial Jacobian.

    Parameters
    ----------
    model : FluidPINN
    x : (B, 3) with ``requires_grad=True``
    t : (B, 1)

    Returns
    -------
    out : dict
        Model outputs (velocity, pressure, density).
    jac : (B, 3, 3)
        ``jac[b, i, j] = du_i / dx_j``.
    """
    if not x.requires_grad:
        x = x.detach().clone().requires_grad_(True)

    out = model(x, t)
    vel = out["velocity"]  # (B, 3)

    grads = []
    for i in range(3):
        g = torch.autograd.grad(
            vel[:, i : i + 1],
            x,
            grad_outputs=torch.ones_like(vel[:, i : i + 1]),
            create_graph=True,
            retain_graph=True,
        )[
            0
        ]  # (B, 3)
        grads.append(g)

    jac = torch.stack(grads, dim=1)  # (B, 3, 3)
    return out, jac


# ---------------------------------------------------------------------------
# 1. No-slip
# ---------------------------------------------------------------------------


def no_slip_loss(
    model: FluidPINN,
    x_boundary: Tensor,
    t_boundary: Tensor,
) -> Tensor:
    """No-slip boundary condition: velocity = 0 at solid walls.

    Parameters
    ----------
    model : FluidPINN
    x_boundary : (B, 3)
    t_boundary : (B, 1)

    Returns
    -------
    Tensor
        Scalar MSE loss.
    """
    out = model(x_boundary, t_boundary)
    return (out["velocity"] ** 2).mean()


# ---------------------------------------------------------------------------
# 2. Inflow
# ---------------------------------------------------------------------------


def inflow_loss(
    model: FluidPINN,
    x_inflow: Tensor,
    t_inflow: Tensor,
    velocity_profile: Callable[[Tensor, Tensor], Tensor],
) -> Tensor:
    """Inflow boundary condition: velocity matches a prescribed profile.

    Parameters
    ----------
    model : FluidPINN
    x_inflow : (B, 3)
    t_inflow : (B, 1)
    velocity_profile : callable ``(x, t) -> (B, 3)``
        Returns target velocity at each boundary point.

    Returns
    -------
    Tensor
        Scalar MSE loss.
    """
    out = model(x_inflow, t_inflow)
    target = velocity_profile(x_inflow, t_inflow)
    return ((out["velocity"] - target) ** 2).mean()


# ---------------------------------------------------------------------------
# 3. Outflow (Neumann)
# ---------------------------------------------------------------------------


def outflow_loss(
    model: FluidPINN,
    x_outflow: Tensor,
    t_outflow: Tensor,
    normal: Tensor | None = None,
) -> Tensor:
    """Zero-gradient (Neumann) outflow condition: du/dn = 0.

    Parameters
    ----------
    model : FluidPINN
    x_outflow : (B, 3) with ``requires_grad=True``
    t_outflow : (B, 1)
    normal : (3,) or (B, 3), optional
        Outward face normal. When provided, only the normal derivative is
        penalised. Otherwise all spatial velocity gradients are penalised.

    Returns
    -------
    Tensor
        Scalar MSE loss.
    """
    out, jac = _velocity_jacobian(model, x_outflow, t_outflow)

    if normal is not None:
        n = normal.to(device=x_outflow.device, dtype=x_outflow.dtype)
        if n.dim() == 1:
            n = n.unsqueeze(0).expand(jac.shape[0], -1)  # (B, 3)
        # du_i/dn = sum_j jac_ij * n_j  =>  jac @ n
        du_dn = torch.bmm(jac, n.unsqueeze(-1)).squeeze(-1)  # (B, 3)
        return (du_dn**2).mean()

    # Fallback: penalise all spatial velocity gradients
    return (jac**2).mean()


# ---------------------------------------------------------------------------
# 4. Pressure reference
# ---------------------------------------------------------------------------


def pressure_reference_loss(
    model: FluidPINN,
    x_ref: Tensor,
    t_ref: Tensor,
    p_ref: float = 0.0,
) -> Tensor:
    """Fix pressure at a reference point to remove gauge ambiguity.

    Parameters
    ----------
    model : FluidPINN
    x_ref : (B, 3) — typically a single point ``(1, 3)``.
    t_ref : (B, 1)
    p_ref : float
        Target pressure value.

    Returns
    -------
    Tensor
        Scalar MSE loss.
    """
    out = model(x_ref, t_ref)
    return ((out["pressure"] - p_ref) ** 2).mean()


# ---------------------------------------------------------------------------
# 5. Free surface
# ---------------------------------------------------------------------------


def free_surface_loss(
    model: FluidPINN,
    x_surface: Tensor,
    t_surface: Tensor,
    surface_normal: Tensor,
) -> Tensor:
    """Free-surface boundary condition.

    Enforces two conditions:

    1. ``p = 0`` (atmospheric pressure).
    2. Zero tangential viscous stress: the tangential component of the
       strain-rate tensor contracted with the surface normal is zero.

    Parameters
    ----------
    model : FluidPINN
    x_surface : (B, 3) with ``requires_grad=True``
    t_surface : (B, 1)
    surface_normal : (3,) or (B, 3)
        Outward-pointing unit normal at the free surface.

    Returns
    -------
    Tensor
        Scalar loss (sum of pressure and stress terms).
    """
    out, jac = _velocity_jacobian(model, x_surface, t_surface)

    # --- Pressure = 0 ---
    p_loss = (out["pressure"] ** 2).mean()

    # --- Zero tangential stress ---
    n = surface_normal.to(device=x_surface.device, dtype=x_surface.dtype)
    if n.dim() == 1:
        n = n.unsqueeze(0).expand(jac.shape[0], -1)  # (B, 3)

    # Strain-rate tensor: eps = 0.5 * (J + J^T)
    eps = 0.5 * (jac + jac.transpose(1, 2))  # (B, 3, 3)

    # Traction on surface: tau = eps @ n
    tau = torch.bmm(eps, n.unsqueeze(-1)).squeeze(-1)  # (B, 3)

    # Tangential part: tau_t = tau - (tau . n) * n
    tau_n = (tau * n).sum(dim=-1, keepdim=True)  # (B, 1)
    tau_tangential = tau - tau_n * n  # (B, 3)

    stress_loss = (tau_tangential**2).mean()

    return p_loss + stress_loss


# ---------------------------------------------------------------------------
# 6. Initial condition
# ---------------------------------------------------------------------------


def initial_condition_loss(
    model: FluidPINN,
    x_initial: Tensor,
    velocity_initial: Tensor | None = None,
    density_initial: Tensor | None = None,
    *,
    t_initial: Tensor | None = None,
) -> Tensor:
    """Initial condition at t = 0.

    Parameters
    ----------
    model : FluidPINN
    x_initial : (B, 3)
        Spatial coordinates at the initial time.
    velocity_initial : (B, 3), optional
        Target velocity. Default: fluid at rest (zero velocity).
    density_initial : (B, 1), optional
        Target density. When ``None``, density is not penalised.
    t_initial : (B, 1), optional
        Time values. Default: ``t = 0``.

    Returns
    -------
    Tensor
        Scalar MSE loss.
    """
    B = x_initial.shape[0]
    if t_initial is None:
        t_initial = torch.zeros(B, 1, device=x_initial.device, dtype=x_initial.dtype)

    out = model(x_initial, t_initial)

    loss = torch.tensor(0.0, device=x_initial.device, dtype=x_initial.dtype)

    if velocity_initial is not None:
        loss = loss + ((out["velocity"] - velocity_initial) ** 2).mean()
    else:
        # Default: fluid at rest
        loss = loss + (out["velocity"] ** 2).mean()

    if density_initial is not None:
        loss = loss + ((out["density"] - density_initial) ** 2).mean()

    return loss


# ---------------------------------------------------------------------------
# 7. BoundaryConditionSet
# ---------------------------------------------------------------------------


@dataclass
class BoundarySpec:
    """Specification for a single boundary condition."""

    bc_type: str
    face: str | None = None
    weight: float = 1.0
    velocity_profile: Callable[[Tensor, Tensor], Tensor] | None = None
    density_profile: Callable[[Tensor, Tensor], Tensor] | None = None
    normal: Tensor | None = None
    p_ref: float = 0.0
    velocity_initial: Tensor | None = None
    density_initial: Tensor | None = None
    mask_fn: Callable[[Tensor, Tensor], Tensor] | None = None


class BoundaryConditionSet:
    """Manages multiple boundary conditions for a physical scenario.

    Parameters
    ----------
    domain : FluidDomain
        Spatial-temporal domain.
    scenario : str
        ``"rising_smoke"`` | ``"pouring_water"`` | ``"custom"``.

    Predefined scenarios
    --------------------
    **rising_smoke** — Box with no-slip side walls, circular inflow at
    bottom (upward velocity + elevated density), outflow at top, fluid
    initially at rest.

    **pouring_water** — Container with no-slip on 5 walls, narrow downward
    stream at top centre, free-surface condition on the rest of the top
    face, fluid initially at rest.

    **custom** — Empty; add BCs manually via :meth:`add_bc`.
    """

    def __init__(
        self,
        domain: FluidDomain,
        scenario: str = "rising_smoke",
    ) -> None:
        self.domain = domain
        self.scenario = scenario
        self.specs: List[BoundarySpec] = []

        if scenario == "rising_smoke":
            self._setup_rising_smoke()
        elif scenario == "pouring_water":
            self._setup_pouring_water()
        elif scenario == "custom":
            pass
        else:
            raise ValueError(
                f"Unknown scenario {scenario!r}. "
                "Choose 'rising_smoke', 'pouring_water', or 'custom'."
            )

    def add_bc(self, spec: BoundarySpec) -> None:
        """Add a boundary condition specification."""
        self.specs.append(spec)

    # ---- Scenario setups ------------------------------------------------

    def _setup_rising_smoke(self) -> None:
        d = self.domain
        cx = (d.x_range[0] + d.x_range[1]) / 2
        cz = (d.z_range[0] + d.z_range[1]) / 2
        r_max = 0.2 * (d.x_range[1] - d.x_range[0])
        v_inlet = 1.0

        # Side walls: no-slip
        for face in ("x_min", "x_max", "z_min", "z_max"):
            self.specs.append(BoundarySpec(bc_type="no_slip", face=face))

        # Bottom (y_min): inflow — circular smoke plume
        def smoke_velocity(x: Tensor, t: Tensor) -> Tensor:
            r = torch.sqrt((x[:, 0:1] - cx) ** 2 + (x[:, 2:3] - cz) ** 2)
            profile = v_inlet * torch.clamp(1.0 - (r / r_max) ** 2, min=0.0)
            zeros = torch.zeros_like(profile)
            return torch.cat([zeros, profile, zeros], dim=-1)

        def smoke_density(x: Tensor, t: Tensor) -> Tensor:
            r = torch.sqrt((x[:, 0:1] - cx) ** 2 + (x[:, 2:3] - cz) ** 2)
            mask = (r < r_max).float()
            return 1.0 + mask  # rho=2 in plume, rho=1 outside

        self.specs.append(
            BoundarySpec(
                bc_type="inflow",
                face="y_min",
                velocity_profile=smoke_velocity,
                density_profile=smoke_density,
            )
        )

        # Top (y_max): outflow
        self.specs.append(
            BoundarySpec(
                bc_type="outflow",
                face="y_max",
                weight=0.5,
                normal=torch.tensor([0.0, 1.0, 0.0]),
            )
        )

        # Pressure reference: p=0 at top centre
        self.specs.append(
            BoundarySpec(
                bc_type="pressure_ref",
                weight=0.1,
                p_ref=0.0,
            )
        )

        # Initial condition: at rest
        self.specs.append(BoundarySpec(bc_type="initial"))

    def _setup_pouring_water(self) -> None:
        d = self.domain
        cx = (d.x_range[0] + d.x_range[1]) / 2
        cz = (d.z_range[0] + d.z_range[1]) / 2
        r_stream = 0.1 * (d.x_range[1] - d.x_range[0])
        v_pour = -2.0

        # Container walls (5 faces): no-slip
        for face in ("x_min", "x_max", "y_min", "z_min", "z_max"):
            self.specs.append(BoundarySpec(bc_type="no_slip", face=face))

        # Top (y_max): inflow — narrow downward stream at centre
        def pour_velocity(x: Tensor, t: Tensor) -> Tensor:
            r = torch.sqrt((x[:, 0:1] - cx) ** 2 + (x[:, 2:3] - cz) ** 2)
            in_stream = (r < r_stream).float()
            v_y = v_pour * in_stream
            zeros = torch.zeros_like(v_y)
            return torch.cat([zeros, v_y, zeros], dim=-1)

        self.specs.append(
            BoundarySpec(
                bc_type="inflow",
                face="y_max",
                velocity_profile=pour_velocity,
            )
        )

        # Free surface on top outside the stream
        _cx, _cz, _r = cx, cz, r_stream  # capture for closure

        def outside_stream(x: Tensor, t: Tensor) -> Tensor:
            r = torch.sqrt((x[:, 0:1] - _cx) ** 2 + (x[:, 2:3] - _cz) ** 2)
            return r.squeeze(-1) > _r

        self.specs.append(
            BoundarySpec(
                bc_type="free_surface",
                face="y_max",
                weight=0.3,
                normal=torch.tensor([0.0, 1.0, 0.0]),
                mask_fn=outside_stream,
            )
        )

        # Pressure reference
        self.specs.append(
            BoundarySpec(
                bc_type="pressure_ref",
                weight=0.1,
                p_ref=0.0,
            )
        )

        # Initial condition: at rest
        self.specs.append(BoundarySpec(bc_type="initial"))

    # ---- Sampling -------------------------------------------------------

    def sample_boundary_points(
        self,
        n_points_per_boundary: int,
        device: torch.device = torch.device("cpu"),
    ) -> List[Dict]:
        """Sample boundary points for each boundary condition.

        Parameters
        ----------
        n_points_per_boundary : int
            Number of points to sample per BC.
        device : torch.device

        Returns
        -------
        list of dict
            Each dict has keys ``"spec"``, ``"x"``, and ``"t"``.
        """
        results: List[Dict] = []
        d = self.domain

        for spec in self.specs:
            if spec.bc_type == "initial":
                x, t = d.sample_initial(n_points_per_boundary, device)
            elif spec.bc_type == "pressure_ref":
                # Single reference point at top centre
                x = torch.tensor(
                    [
                        [
                            (d.x_range[0] + d.x_range[1]) / 2,
                            d.y_range[1],
                            (d.z_range[0] + d.z_range[1]) / 2,
                        ]
                    ],
                    device=device,
                )
                t = d._uniform(*d.t_range, 1, device)
            elif spec.face is not None:
                x, t = d.sample_boundary(n_points_per_boundary, spec.face, device)
            else:
                continue

            # Apply point filter if present
            if spec.mask_fn is not None:
                mask = spec.mask_fn(x, t)
                if not mask.any():
                    continue
                x, t = x[mask], t[mask]

            results.append({"spec": spec, "x": x, "t": t})

        return results

    # ---- Loss computation -----------------------------------------------

    def compute_total_boundary_loss(
        self,
        model: FluidPINN,
        n_points_per_boundary: int = 128,
        device: torch.device | None = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute the weighted sum of all boundary condition losses.

        Parameters
        ----------
        model : FluidPINN
        n_points_per_boundary : int
            Points sampled per BC (stochastic mini-batch).
        device : torch.device, optional
            Defaults to the model's device.

        Returns
        -------
        (total, details)
            Scalar total loss and dict of individual (unweighted) losses.
        """
        if device is None:
            device = next(model.parameters()).device

        samples = self.sample_boundary_points(n_points_per_boundary, device)

        total = torch.tensor(0.0, device=device)
        details: Dict[str, Tensor] = {}
        name_counts: Dict[str, int] = {}

        for sample in samples:
            spec: BoundarySpec = sample["spec"]
            x, t = sample["x"], sample["t"]

            if spec.bc_type == "no_slip":
                l = no_slip_loss(model, x, t)

            elif spec.bc_type == "inflow":
                l = inflow_loss(model, x, t, spec.velocity_profile)
                # Optional density matching at inflow
                if spec.density_profile is not None:
                    out = model(x, t)
                    target_rho = spec.density_profile(x, t)
                    l = l + ((out["density"] - target_rho) ** 2).mean()

            elif spec.bc_type == "outflow":
                l = outflow_loss(model, x, t, normal=spec.normal)

            elif spec.bc_type == "pressure_ref":
                l = pressure_reference_loss(model, x, t, p_ref=spec.p_ref)

            elif spec.bc_type == "free_surface":
                l = free_surface_loss(model, x, t, spec.normal)

            elif spec.bc_type == "initial":
                l = initial_condition_loss(
                    model,
                    x,
                    velocity_initial=spec.velocity_initial,
                    density_initial=spec.density_initial,
                    t_initial=t,
                )

            else:
                raise ValueError(f"Unknown BC type: {spec.bc_type!r}")

            # Unique name for the details dict
            base = spec.bc_type
            if spec.face:
                base += f"/{spec.face}"
            count = name_counts.get(base, 0)
            name_counts[base] = count + 1
            name = base if count == 0 else f"{base}#{count + 1}"

            details[name] = l
            total = total + spec.weight * l

        return total, details


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from pinn.model import PINNConfig

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

    B = 30
    cfg = PINNConfig(activation="siren", hidden_dim=64, num_layers=4)
    model = FluidPINN(cfg)

    # ---- 1. Standalone loss functions ----
    print("\n=== Standalone loss functions ===")

    x = torch.randn(B, 3, requires_grad=True)
    t = torch.rand(B, 1, requires_grad=True)

    # no_slip_loss
    l_ns = no_slip_loss(model, x, t)
    check(l_ns.dim() == 0, "no_slip_loss is scalar")
    check(finite(l_ns), "no_slip_loss is finite")
    check(l_ns.item() > 0, f"no_slip_loss > 0 ({l_ns.item():.6f})")

    # inflow_loss
    def test_profile(x_in: Tensor, t_in: Tensor) -> Tensor:
        zeros = torch.zeros_like(x_in[:, 0:1])
        ones = torch.ones_like(x_in[:, 0:1])
        return torch.cat([zeros, ones, zeros], dim=-1)

    l_in = inflow_loss(model, x, t, test_profile)
    check(l_in.dim() == 0, "inflow_loss is scalar")
    check(finite(l_in), "inflow_loss is finite")

    # outflow_loss (with normal)
    x2 = torch.randn(B, 3, requires_grad=True)
    t2 = torch.rand(B, 1)
    normal = torch.tensor([1.0, 0.0, 0.0])
    l_out = outflow_loss(model, x2, t2, normal=normal)
    check(l_out.dim() == 0, "outflow_loss is scalar")
    check(finite(l_out), "outflow_loss is finite")

    # outflow_loss (without normal)
    x3 = torch.randn(B, 3, requires_grad=True)
    l_out2 = outflow_loss(model, x3, t2)
    check(l_out2.dim() == 0, "outflow_loss (no normal) is scalar")
    check(finite(l_out2), "outflow_loss (no normal) is finite")

    # pressure_reference_loss
    x_ref = torch.tensor([[0.0, 1.0, 0.0]])
    t_ref = torch.tensor([[0.5]])
    l_pr = pressure_reference_loss(model, x_ref, t_ref, p_ref=0.0)
    check(l_pr.dim() == 0, "pressure_reference_loss is scalar")
    check(finite(l_pr), "pressure_reference_loss is finite")

    # free_surface_loss
    x4 = torch.randn(B, 3, requires_grad=True)
    t4 = torch.rand(B, 1)
    n_surf = torch.tensor([0.0, 1.0, 0.0])
    l_fs = free_surface_loss(model, x4, t4, n_surf)
    check(l_fs.dim() == 0, "free_surface_loss is scalar")
    check(finite(l_fs), "free_surface_loss is finite")

    # initial_condition_loss (default: at rest)
    x5 = torch.randn(B, 3)
    l_ic = initial_condition_loss(model, x5)
    check(l_ic.dim() == 0, "initial_condition_loss is scalar")
    check(finite(l_ic), "initial_condition_loss is finite")

    # initial_condition_loss (with targets)
    vel_target = torch.zeros(B, 3)
    rho_target = torch.ones(B, 1)
    l_ic2 = initial_condition_loss(model, x5, vel_target, rho_target)
    check(l_ic2.dim() == 0, "initial_condition_loss (with targets) is scalar")
    check(finite(l_ic2), "initial_condition_loss (with targets) is finite")

    # ---- 2. Backward test ----
    print("\n=== Backward to model weights ===")
    model.zero_grad()
    x_bk = torch.randn(B, 3, requires_grad=True)
    t_bk = torch.rand(B, 1, requires_grad=True)
    loss_bk = no_slip_loss(model, x_bk, t_bk) + outflow_loss(model, x_bk, t_bk)
    loss_bk = loss_bk + initial_condition_loss(model, x_bk)
    loss_bk.backward()

    n_with_grad = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().max().item() > 0
    )
    n_total = sum(1 for _ in model.parameters())
    check(
        n_with_grad == n_total, f"All {n_total} params have non-zero grad ({n_with_grad}/{n_total})"
    )

    # ---- 3. BoundaryConditionSet — rising_smoke ----
    print("\n=== BoundaryConditionSet: rising_smoke ===")
    domain = FluidDomain()
    bc_set = BoundaryConditionSet(domain, scenario="rising_smoke")

    check(len(bc_set.specs) > 0, f"Has {len(bc_set.specs)} BC specs")

    samples = bc_set.sample_boundary_points(64)
    check(len(samples) > 0, f"Sampled {len(samples)} boundary groups")
    for s in samples:
        check("x" in s and "t" in s and "spec" in s, f"Sample has x, t, spec keys")
        break  # just check the first one

    model.zero_grad()
    total_rs, details_rs = bc_set.compute_total_boundary_loss(model, n_points_per_boundary=64)
    check(total_rs.dim() == 0, "rising_smoke total is scalar")
    check(finite(total_rs), "rising_smoke total is finite")
    check(total_rs.item() > 0, f"rising_smoke total > 0 ({total_rs.item():.4f})")
    check(len(details_rs) > 0, f"rising_smoke details has {len(details_rs)} entries")
    print(f"    Details: {', '.join(f'{k}={v.item():.4f}' for k, v in details_rs.items())}")

    total_rs.backward()
    n_grad_rs = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().max().item() > 0
    )
    check(
        n_grad_rs == n_total,
        f"rising_smoke grads reach all {n_total} params ({n_grad_rs}/{n_total})",
    )

    # ---- 4. BoundaryConditionSet — pouring_water ----
    print("\n=== BoundaryConditionSet: pouring_water ===")
    bc_pour = BoundaryConditionSet(domain, scenario="pouring_water")
    check(len(bc_pour.specs) > 0, f"Has {len(bc_pour.specs)} BC specs")

    model.zero_grad()
    total_pw, details_pw = bc_pour.compute_total_boundary_loss(model, n_points_per_boundary=64)
    check(total_pw.dim() == 0, "pouring_water total is scalar")
    check(finite(total_pw), "pouring_water total is finite")
    check(total_pw.item() > 0, f"pouring_water total > 0 ({total_pw.item():.4f})")
    print(f"    Details: {', '.join(f'{k}={v.item():.4f}' for k, v in details_pw.items())}")

    total_pw.backward()
    n_grad_pw = sum(
        1 for p in model.parameters() if p.grad is not None and p.grad.abs().max().item() > 0
    )
    check(
        n_grad_pw == n_total,
        f"pouring_water grads reach all {n_total} params ({n_grad_pw}/{n_total})",
    )

    # ---- 5. Custom scenario ----
    print("\n=== BoundaryConditionSet: custom ===")
    bc_custom = BoundaryConditionSet(domain, scenario="custom")
    check(len(bc_custom.specs) == 0, "custom starts empty")

    bc_custom.add_bc(BoundarySpec(bc_type="no_slip", face="x_min", weight=2.0))
    bc_custom.add_bc(BoundarySpec(bc_type="initial"))
    check(len(bc_custom.specs) == 2, "custom has 2 BCs after add_bc")

    model.zero_grad()
    total_c, details_c = bc_custom.compute_total_boundary_loss(model, n_points_per_boundary=32)
    check(total_c.dim() == 0, "custom total is scalar")
    check(finite(total_c), "custom total is finite")
    check("no_slip/x_min" in details_c, "custom details has 'no_slip/x_min'")
    check("initial" in details_c, "custom details has 'initial'")

    # ---- 6. Invalid scenario ----
    print("\n=== Error handling ===")
    try:
        BoundaryConditionSet(domain, scenario="invalid")
        check(False, "Should raise ValueError for invalid scenario")
    except ValueError:
        check(True, "Raises ValueError for invalid scenario")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 60}")

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: pinn/boundary.py"
            " — all BC losses, scenarios, and gradient flow verified"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)
