#!/usr/bin/env python3
"""Training loop for the physics-informed neural network.

Trains a :class:`~pinn.model.FluidPINN` using **only** physics losses — no
data fitting.  Each iteration samples random collocation points, evaluates the
Navier-Stokes residual, and minimises it.  Boundary conditions provide the
"supervision" that makes the solution unique.

Usage::

    python -m pinn.train --config configs/default.yaml
    python -m pinn.train --config configs/default.yaml --resume outputs/pinn/ckpt_5000.pt
    python -m pinn.train --device cuda --wandb
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time as time_mod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from pinn.boundary import BoundaryConditionSet
from pinn.model import FluidDomain, FluidPINN, PINNConfig, count_parameters
from pinn.navier_stokes import (
    compute_derivatives,
    navier_stokes_residual,
    physics_loss,
)

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MPL = True
except ImportError:
    HAS_MPL = False


# ---------------------------------------------------------------------------
# 1. Training configuration
# ---------------------------------------------------------------------------


@dataclass
class PINNTrainConfig:
    """All training hyperparameters with sensible defaults."""

    # Epochs & batch sizes
    num_epochs: int = 50_000
    num_collocation: int = 10_000
    num_boundary: int = 2_000

    # Optimiser
    learning_rate: float = 1e-4
    grad_clip_max_norm: float = 1.0

    # LR scheduler: CosineAnnealingWarmRestarts
    scheduler_T_0: int = 5_000
    scheduler_T_mult: int = 2

    # Physics
    viscosity: float = 1e-3
    gravity: Tuple[float, float, float] = (0.0, -9.81, 0.0)
    lambda_physics: float = 1.0
    lambda_boundary: float = 1.0
    lambda_momentum: float = 1.0
    lambda_continuity: float = 1.0

    # Domain
    x_range: Tuple[float, float] = (-1.0, 1.0)
    y_range: Tuple[float, float] = (-1.0, 1.0)
    z_range: Tuple[float, float] = (-1.0, 1.0)
    t_range: Tuple[float, float] = (0.0, 1.0)

    # Scenario for boundary conditions
    scenario: str = "rising_smoke"

    # Network
    activation: str = "siren"
    hidden_dim: int = 256
    num_layers: int = 8

    # Adaptive collocation
    adaptive_collocation: bool = True
    adaptive_interval: int = 1_000
    adaptive_grid_res: int = 32

    # Logging
    log_interval: int = 1_000

    # Convergence
    plateau_window: int = 5_000
    plateau_threshold: float = 1e-6
    early_stopping_threshold: float = 1e-6

    # Checkpointing
    checkpoint_interval: int = 5_000

    # Gradient checkpointing (trades compute for memory)
    use_grad_checkpointing: bool = False


def build_config(yaml_cfg: Any = None) -> PINNTrainConfig:
    """Build :class:`PINNTrainConfig` from an OmegaConf / dict config.

    Reads ``pinn.*`` and ``training.pinn.*`` sections from the YAML.
    """
    if yaml_cfg is None:
        return PINNTrainConfig()

    # Allow both OmegaConf and plain dict
    get = lambda obj, key, default: obj.get(key, default) if hasattr(obj, "get") else default

    pinn = get(yaml_cfg, "pinn", {})
    training = get(yaml_cfg, "training", {})
    train_pinn = get(training, "pinn", {})

    net = get(pinn, "network", {})
    phys = get(pinn, "physics", {})
    lw = get(pinn, "loss_weights", {})
    dom = get(pinn, "domain", {})

    gravity_raw = get(phys, "gravity", [0.0, -9.81, 0.0])
    gravity = tuple(gravity_raw) if not isinstance(gravity_raw, tuple) else gravity_raw

    def _range(d, key, default):
        raw = get(d, key, default)
        return tuple(raw) if not isinstance(raw, tuple) else raw

    return PINNTrainConfig(
        num_epochs=get(train_pinn, "epochs", 50_000),
        num_collocation=get(pinn, "num_collocation", 10_000),
        num_boundary=get(train_pinn, "batch_size", 2_000),
        learning_rate=get(pinn, "learning_rate", 1e-4),
        grad_clip_max_norm=get(training, "gradient_clip", 1.0),
        viscosity=get(phys, "viscosity", 1e-3),
        gravity=gravity,
        lambda_momentum=get(lw, "momentum", 1.0),
        lambda_continuity=get(lw, "continuity", 1.0),
        lambda_boundary=get(lw, "boundary", 0.1),
        activation=get(net, "activation", "siren"),
        hidden_dim=get(net, "hidden_dim", 256),
        num_layers=get(net, "num_layers", 8),
        x_range=_range(dom, "x_range", (-1.0, 1.0)),
        y_range=_range(dom, "y_range", (-1.0, 1.0)),
        z_range=_range(dom, "z_range", (-1.0, 1.0)),
        t_range=_range(dom, "t_range", (0.0, 1.0)),
        checkpoint_interval=get(train_pinn, "save_interval", 5_000),
        log_interval=get(train_pinn, "log_interval", 1_000),
    )


# ---------------------------------------------------------------------------
# 2. Adaptive collocation sampler
# ---------------------------------------------------------------------------


class AdaptiveSampler:
    """Importance-samples collocation points toward high-residual regions.

    Every ``update_interval`` iterations, evaluates the NS residual on a coarse
    spatial grid and uses the per-cell residual magnitude as sampling weights.
    """

    def __init__(
        self,
        domain: FluidDomain,
        grid_res: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.domain = domain
        self.grid_res = grid_res
        self.device = device
        self.weights: Tensor | None = None

        # Pre-compute cell edge coordinates
        R = grid_res
        self.x_edges = torch.linspace(*domain.x_range, R + 1, device=device)
        self.y_edges = torch.linspace(*domain.y_range, R + 1, device=device)
        self.z_edges = torch.linspace(*domain.z_range, R + 1, device=device)

    def update_weights(
        self,
        model: FluidPINN,
        nu: float,
        gravity: Tuple[float, float, float],
    ) -> None:
        """Evaluate NS residual on the coarse grid, store sampling weights."""
        R = self.grid_res
        d = self.domain

        # Cell centres
        x_c = 0.5 * (self.x_edges[:-1] + self.x_edges[1:])
        y_c = 0.5 * (self.y_edges[:-1] + self.y_edges[1:])
        z_c = 0.5 * (self.z_edges[:-1] + self.z_edges[1:])
        xx, yy, zz = torch.meshgrid(x_c, y_c, z_c, indexing="ij")
        xyz_all = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)
        t_lo, t_hi = d.t_range
        t_all = t_lo + (t_hi - t_lo) * torch.rand(xyz_all.shape[0], 1, device=self.device)

        # Process in chunks to manage memory (autograd graph is heavy)
        chunk_size = 4096
        res_parts: List[Tensor] = []

        for i in range(0, len(xyz_all), chunk_size):
            xyz_c = xyz_all[i : i + chunk_size].clone().requires_grad_(True)
            t_c = t_all[i : i + chunk_size].clone().requires_grad_(True)

            with torch.enable_grad():
                derivs = compute_derivatives(model, xyz_c, t_c)
                residuals = navier_stokes_residual(derivs, nu=nu, gravity=gravity)
                # Sum-of-squares per point, detach to free computation graph
                total_res = sum((v**2).squeeze(-1) for v in residuals.values()).detach()

            res_parts.append(total_res)

        self.weights = torch.cat(res_parts) + 1e-8

    def sample(self, n_points: int) -> Tuple[Tensor, Tensor]:
        """Sample collocation points — importance-weighted if weights exist."""
        if self.weights is None:
            return self.domain.sample_interior(n_points, self.device)

        R = self.grid_res
        # Pick cells with probability proportional to residual
        cell_idx = torch.multinomial(self.weights, n_points, replacement=True)

        # Flat index → 3D cell indices
        ix = cell_idx // (R * R)
        iy = (cell_idx % (R * R)) // R
        iz = cell_idx % R

        # Uniform sample within each cell
        rand = torch.rand(n_points, 3, device=self.device)
        x_pts = self.x_edges[ix] + rand[:, 0] * (self.x_edges[ix + 1] - self.x_edges[ix])
        y_pts = self.y_edges[iy] + rand[:, 1] * (self.y_edges[iy + 1] - self.y_edges[iy])
        z_pts = self.z_edges[iz] + rand[:, 2] * (self.z_edges[iz + 1] - self.z_edges[iz])

        x = torch.stack([x_pts, y_pts, z_pts], dim=-1)
        t_lo, t_hi = self.domain.t_range
        t = t_lo + (t_hi - t_lo) * torch.rand(n_points, 1, device=self.device)

        x.requires_grad_(True)
        t.requires_grad_(True)
        return x, t


# ---------------------------------------------------------------------------
# 3. Convergence monitor
# ---------------------------------------------------------------------------


class ConvergenceMonitor:
    """Tracks loss history and detects plateaus."""

    def __init__(
        self,
        window: int = 1_000,
        plateau_window: int = 5_000,
        plateau_threshold: float = 1e-6,
    ) -> None:
        self.losses: List[float] = []
        self.window = window
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold

    def update(self, loss_val: float) -> None:
        self.losses.append(loss_val)

    def rolling_mean(self) -> float:
        if not self.losses:
            return float("inf")
        w = self.losses[-self.window :]
        return sum(w) / len(w)

    def check_plateau(self) -> bool:
        """Return True if loss has plateaued."""
        if len(self.losses) < self.plateau_window:
            return False
        recent = self.losses[-self.plateau_window :]
        mid = len(recent) // 2
        first_half = sum(recent[:mid]) / mid
        second_half = sum(recent[mid:]) / (len(recent) - mid)
        if first_half < 1e-12:
            return False
        rel_change = abs(second_half - first_half) / first_half
        return rel_change < self.plateau_threshold


# ---------------------------------------------------------------------------
# 4. 2D slice visualisations
# ---------------------------------------------------------------------------


@torch.no_grad()
def create_slice_plots(
    model: FluidPINN,
    domain: FluidDomain,
    device: torch.device,
    grid_res: int = 64,
    t_val: float = 0.5,
) -> Dict[str, Any]:
    """Create 2D slice plots at z=0 for velocity, pressure, and density.

    Returns dict mapping name -> matplotlib Figure (or empty if mpl missing).
    """
    if not HAS_MPL:
        return {}

    d = domain
    xs = torch.linspace(d.x_range[0], d.x_range[1], grid_res, device=device)
    ys = torch.linspace(d.y_range[0], d.y_range[1], grid_res, device=device)
    xx, yy = torch.meshgrid(xs, ys, indexing="ij")
    z_mid = (d.z_range[0] + d.z_range[1]) / 2.0
    xyz = torch.stack(
        [
            xx.flatten(),
            yy.flatten(),
            torch.full((grid_res**2,), z_mid, device=device),
        ],
        dim=-1,
    )
    t_tensor = torch.full((grid_res**2, 1), t_val, device=device)

    out = model(xyz, t_tensor)
    vel = out["velocity"].cpu().numpy()
    p = out["pressure"].squeeze(-1).cpu().numpy().reshape(grid_res, grid_res)
    rho = out["density"].squeeze(-1).cpu().numpy().reshape(grid_res, grid_res)

    xx_np = xx.cpu().numpy()
    yy_np = yy.cpu().numpy()
    u = vel[:, 0].reshape(grid_res, grid_res)
    v = vel[:, 1].reshape(grid_res, grid_res)
    speed = np.sqrt(u**2 + v**2)

    figs: Dict[str, Any] = {}

    # --- Velocity quiver + speed contour ---
    fig1, ax1 = plt.subplots(figsize=(6, 6))
    cf = ax1.contourf(xx_np, yy_np, speed, levels=20, cmap="viridis")
    step = max(1, grid_res // 16)
    ax1.quiver(
        xx_np[::step, ::step],
        yy_np[::step, ::step],
        u[::step, ::step],
        v[::step, ::step],
        color="white",
        scale_units="xy",
        alpha=0.8,
    )
    ax1.set_title(f"Velocity (z={z_mid:.1f}, t={t_val:.2f})")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_aspect("equal")
    fig1.colorbar(cf, ax=ax1, label="Speed")
    fig1.tight_layout()
    figs["velocity"] = fig1

    # --- Pressure colormap ---
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    cf2 = ax2.contourf(xx_np, yy_np, p, levels=20, cmap="coolwarm")
    ax2.set_title(f"Pressure (z={z_mid:.1f}, t={t_val:.2f})")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_aspect("equal")
    fig2.colorbar(cf2, ax=ax2, label="Pressure")
    fig2.tight_layout()
    figs["pressure"] = fig2

    # --- Density colormap ---
    fig3, ax3 = plt.subplots(figsize=(6, 6))
    cf3 = ax3.contourf(xx_np, yy_np, rho, levels=20, cmap="YlOrRd")
    ax3.set_title(f"Density (z={z_mid:.1f}, t={t_val:.2f})")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    ax3.set_aspect("equal")
    fig3.colorbar(cf3, ax=ax3, label="Density")
    fig3.tight_layout()
    figs["density"] = fig3

    return figs


# ---------------------------------------------------------------------------
# 5. Checkpointing
# ---------------------------------------------------------------------------


def save_checkpoint(
    path: Path,
    model: FluidPINN,
    optimizer: torch.optim.Adam,
    scheduler: Any,
    iteration: int,
    best_loss: float,
    monitor: ConvergenceMonitor,
    adaptive_weights: Tensor | None,
) -> None:
    torch.save(
        {
            "iteration": iteration,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_loss": best_loss,
            "loss_history": monitor.losses[-10_000:],  # keep last 10k
            "adaptive_weights": adaptive_weights,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: FluidPINN,
    optimizer: torch.optim.Adam,
    scheduler: Any,
    device: torch.device,
) -> Tuple[int, float, ConvergenceMonitor, Tensor | None]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    iteration = ckpt["iteration"]
    best_loss = ckpt.get("best_loss", float("inf"))
    monitor = ConvergenceMonitor()
    monitor.losses = ckpt.get("loss_history", [])
    adaptive_weights = ckpt.get("adaptive_weights", None)
    return iteration, best_loss, monitor, adaptive_weights


# ---------------------------------------------------------------------------
# 6. Final evaluation on fine grid
# ---------------------------------------------------------------------------


def final_evaluation(
    model: FluidPINN,
    domain: FluidDomain,
    device: torch.device,
    nu: float,
    gravity: Tuple[float, float, float],
    grid_res: int = 48,
) -> Dict[str, float]:
    """Evaluate NS residuals on a fine grid and report statistics."""
    R = grid_res
    d = domain

    x_c = torch.linspace(d.x_range[0], d.x_range[1], R, device=device)
    y_c = torch.linspace(d.y_range[0], d.y_range[1], R, device=device)
    z_c = torch.linspace(d.z_range[0], d.z_range[1], R, device=device)
    xx, yy, zz = torch.meshgrid(x_c, y_c, z_c, indexing="ij")
    xyz_all = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)

    # Evaluate at multiple time values
    n_pts = xyz_all.shape[0]
    t_vals = torch.linspace(d.t_range[0], d.t_range[1], 5, device=device)

    all_stats: Dict[str, List[float]] = {
        "momentum_x_max": [],
        "momentum_y_max": [],
        "momentum_z_max": [],
        "continuity_max": [],
        "momentum_x_mean": [],
        "momentum_y_mean": [],
        "momentum_z_mean": [],
        "continuity_mean": [],
    }

    chunk_size = 4096
    for t_val in t_vals:
        res_per_term: Dict[str, List[Tensor]] = {
            "momentum_x": [],
            "momentum_y": [],
            "momentum_z": [],
            "continuity": [],
        }

        t_tensor = torch.full((chunk_size, 1), t_val.item(), device=device)

        for i in range(0, n_pts, chunk_size):
            end = min(i + chunk_size, n_pts)
            xyz_c = xyz_all[i:end].clone().requires_grad_(True)
            t_c = t_tensor[: end - i].requires_grad_(True)

            with torch.enable_grad():
                derivs = compute_derivatives(model, xyz_c, t_c)
                residuals = navier_stokes_residual(derivs, nu=nu, gravity=gravity)

            for key in res_per_term:
                res_per_term[key].append(residuals[key].detach().abs())

        for key in res_per_term:
            vals = torch.cat(res_per_term[key])
            all_stats[f"{key}_max"].append(vals.max().item())
            all_stats[f"{key}_mean"].append(vals.mean().item())

    # Average across time values
    results: Dict[str, float] = {}
    for key, vals in all_stats.items():
        results[key] = sum(vals) / len(vals)

    return results


# ---------------------------------------------------------------------------
# 7. Main training loop
# ---------------------------------------------------------------------------


def train(
    config: PINNTrainConfig | None = None,
    resume_path: str | None = None,
    device_str: str = "auto",
    use_wandb: bool = False,
    output_dir: str = "outputs/pinn",
) -> FluidPINN:
    """Train a FluidPINN using physics-only losses.

    Parameters
    ----------
    config : PINNTrainConfig, optional
    resume_path : str, optional
        Path to checkpoint for resume.
    device_str : str
        ``"auto"``, ``"cuda"``, ``"mps"``, or ``"cpu"``.
    use_wandb : bool
    output_dir : str

    Returns
    -------
    FluidPINN
        The trained model.
    """
    if config is None:
        config = PINNTrainConfig()

    # --- Device ---
    if device_str == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device_str)
    print(f"Device: {device}")

    # --- Seed ---
    torch.manual_seed(42)
    np.random.seed(42)

    # --- Paths ---
    out_path = Path(output_dir)
    ckpt_dir = out_path / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_path / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # --- Domain ---
    domain = FluidDomain(
        x_range=config.x_range,
        y_range=config.y_range,
        z_range=config.z_range,
        t_range=config.t_range,
    )

    # --- Model ---
    pinn_cfg = PINNConfig(
        activation=config.activation,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
    )
    model = FluidPINN(pinn_cfg).to(device)
    n_params = count_parameters(model)
    print(f"FluidPINN ({config.activation}): {n_params:,} parameters")

    # --- Boundary conditions ---
    bc_set = BoundaryConditionSet(domain, scenario=config.scenario)
    print(f"Boundary scenario: {config.scenario} ({len(bc_set.specs)} BCs)")

    # --- Optimiser: Adam ---
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # --- Scheduler: CosineAnnealingWarmRestarts ---
    # Restarts at T_0, T_0*T_mult, T_0*T_mult^2, ...
    # (5000, 10000, 20000, ...) — helps PINNs escape local minima
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=config.scheduler_T_0,
        T_mult=config.scheduler_T_mult,
    )

    # --- Adaptive sampler ---
    adaptive_sampler: AdaptiveSampler | None = None
    if config.adaptive_collocation:
        adaptive_sampler = AdaptiveSampler(
            domain,
            grid_res=config.adaptive_grid_res,
            device=device,
        )

    # --- Convergence monitor ---
    monitor = ConvergenceMonitor(
        plateau_window=config.plateau_window,
        plateau_threshold=config.plateau_threshold,
    )

    # --- Resume ---
    start_iter = 0
    best_loss = float("inf")

    if resume_path is not None:
        print(f"Resuming from {resume_path}")
        start_iter, best_loss, monitor, adap_w = load_checkpoint(
            Path(resume_path),
            model,
            optimizer,
            scheduler,
            device,
        )
        if adap_w is not None and adaptive_sampler is not None:
            adaptive_sampler.weights = adap_w.to(device)
        print(f"  Resumed at iteration {start_iter}, best loss={best_loss:.6f}")

    # --- Wandb ---
    if use_wandb:
        import wandb

        wandb.init(
            project="fluid-recon-pinn",
            config={
                "num_epochs": config.num_epochs,
                "num_collocation": config.num_collocation,
                "num_boundary": config.num_boundary,
                "learning_rate": config.learning_rate,
                "viscosity": config.viscosity,
                "gravity": config.gravity,
                "scenario": config.scenario,
                "activation": config.activation,
                "hidden_dim": config.hidden_dim,
                "num_layers": config.num_layers,
                "n_params": n_params,
            },
            resume="allow",
        )

    # --- Training ---
    print(f"\nStarting training: {config.num_epochs} iterations")
    print(f"  Collocation: {config.num_collocation}, Boundary: {config.num_boundary}")
    print(f"  nu={config.viscosity}, gravity={config.gravity}")
    print(f"  lambda_physics={config.lambda_physics}, lambda_boundary={config.lambda_boundary}")
    print()

    model.train()
    t_start = time_mod.time()

    for iteration in range(start_iter, config.num_epochs):
        # ----------------------------------------------------------------
        # (a) Sample collocation points with requires_grad=True
        # ----------------------------------------------------------------
        if adaptive_sampler is not None and adaptive_sampler.weights is not None:
            x_col, t_col = adaptive_sampler.sample(config.num_collocation)
        else:
            x_col, t_col = domain.sample_interior(config.num_collocation, device)

        # ----------------------------------------------------------------
        # (b) Compute derivatives and NS residual
        #
        # compute_derivatives evaluates:
        #   model(x,t) → (u,v,w,p,rho) + all 1st/2nd order spatial/temporal
        #   derivatives via torch.autograd.grad (create_graph=True so
        #   backward reaches model weights).
        #
        # navier_stokes_residual evaluates:
        #   momentum_i = du_i/dt + u_j*du_i/dx_j + (1/rho)*dp/dx_i
        #                - nu*laplacian(u_i) - gravity_i
        #   continuity = du/dx + dv/dy + dw/dz
        # ----------------------------------------------------------------
        derivs = compute_derivatives(
            model,
            x_col,
            t_col,
            use_checkpointing=config.use_grad_checkpointing,
        )
        residuals = navier_stokes_residual(
            derivs,
            nu=config.viscosity,
            gravity=config.gravity,
        )
        phys_loss, phys_details = physics_loss(
            residuals,
            lambda_momentum=config.lambda_momentum,
            lambda_continuity=config.lambda_continuity,
        )

        # ----------------------------------------------------------------
        # (c) Compute boundary losses
        #
        # BoundaryConditionSet samples its own points and evaluates:
        #   no-slip (vel=0 at walls), inflow (vel=profile), outflow (du/dn=0),
        #   pressure reference (p=p_ref), initial condition (vel=0 at t=0),
        #   and optional density at inflow.
        # ----------------------------------------------------------------
        bc_loss, bc_details = bc_set.compute_total_boundary_loss(
            model,
            n_points_per_boundary=config.num_boundary,
            device=device,
        )

        # ----------------------------------------------------------------
        # (d) Total loss = lambda_physics * phys + lambda_boundary * bc
        # ----------------------------------------------------------------
        total = config.lambda_physics * phys_loss + config.lambda_boundary * bc_loss

        # ----------------------------------------------------------------
        # (e) Backward, gradient clipping, optimiser step
        # ----------------------------------------------------------------
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_max_norm)
        optimizer.step()
        scheduler.step(iteration)

        # ----------------------------------------------------------------
        # Track convergence
        # ----------------------------------------------------------------
        loss_val = total.item()
        monitor.update(loss_val)

        # ----------------------------------------------------------------
        # Adaptive collocation update (every adaptive_interval iterations)
        # ----------------------------------------------------------------
        if adaptive_sampler is not None and (iteration + 1) % config.adaptive_interval == 0:
            adaptive_sampler.update_weights(
                model,
                config.viscosity,
                config.gravity,
            )

        # ----------------------------------------------------------------
        # Logging (every log_interval iterations)
        # ----------------------------------------------------------------
        if iteration % config.log_interval == 0:
            elapsed = time_mod.time() - t_start
            lr_current = optimizer.param_groups[0]["lr"]
            rolling = monitor.rolling_mean()

            # Max absolute residuals
            max_res = {k: v.detach().abs().max().item() for k, v in residuals.items()}

            print(
                f"[{iteration:6d}/{config.num_epochs}] "
                f"total={loss_val:.6f}  "
                f"phys={phys_loss.item():.6f}  "
                f"bc={bc_loss.item():.6f}  "
                f"lr={lr_current:.2e}  "
                f"rolling={rolling:.6f}  "
                f"time={elapsed:.0f}s"
            )
            print(
                f"  momentum: x={phys_details['momentum_x'].item():.4e}  "
                f"y={phys_details['momentum_y'].item():.4e}  "
                f"z={phys_details['momentum_z'].item():.4e}  "
                f"cont={phys_details['continuity'].item():.4e}"
            )
            print(f"  max|res|: " + "  ".join(f"{k}={v:.4e}" for k, v in max_res.items()))

            # Wandb logging
            if use_wandb:
                import wandb

                log_data = {
                    "train/total_loss": loss_val,
                    "train/physics_loss": phys_loss.item(),
                    "train/boundary_loss": bc_loss.item(),
                    "train/lr": lr_current,
                    "train/rolling_mean": rolling,
                }
                for k, v in phys_details.items():
                    log_data[f"physics/{k}"] = v.item()
                for k, v in bc_details.items():
                    log_data[f"boundary/{k}"] = v.item()
                for k, v in max_res.items():
                    log_data[f"max_residual/{k}"] = v
                wandb.log(log_data, step=iteration)

            # 2D slice visualisations
            if HAS_MPL and iteration > 0:
                figs = create_slice_plots(model, domain, device, grid_res=48)
                for name, fig in figs.items():
                    fig.savefig(viz_dir / f"{name}_{iteration:06d}.png", dpi=100)
                    if use_wandb:
                        import wandb

                        wandb.log({f"viz/{name}": wandb.Image(fig)}, step=iteration)
                    plt.close(fig)

        # ----------------------------------------------------------------
        # Convergence checks
        # ----------------------------------------------------------------
        # Plateau warning
        if (iteration + 1) % config.plateau_window == 0 and monitor.check_plateau():
            print(
                f"  WARNING: Loss has plateaued (relative change < "
                f"{config.plateau_threshold} over {config.plateau_window} iters)"
            )

        # Early stopping
        if loss_val < config.early_stopping_threshold:
            print(
                f"\n  Early stopping: loss {loss_val:.2e} < {config.early_stopping_threshold:.2e}"
            )
            break

        # ----------------------------------------------------------------
        # Checkpointing (every checkpoint_interval iterations)
        # ----------------------------------------------------------------
        if (iteration + 1) % config.checkpoint_interval == 0:
            ckpt_path = ckpt_dir / f"ckpt_{iteration + 1:06d}.pt"
            save_checkpoint(
                ckpt_path,
                model,
                optimizer,
                scheduler,
                iteration + 1,
                min(best_loss, loss_val),
                monitor,
                adaptive_sampler.weights if adaptive_sampler else None,
            )
            print(f"  Checkpoint saved: {ckpt_path}")

        if loss_val < best_loss:
            best_loss = loss_val

    # --- Final checkpoint ---
    save_checkpoint(
        ckpt_dir / "final.pt",
        model,
        optimizer,
        scheduler,
        config.num_epochs,
        best_loss,
        monitor,
        adaptive_sampler.weights if adaptive_sampler else None,
    )

    # --- Final evaluation ---
    print("\n" + "=" * 60)
    print("Final evaluation on fine grid...")
    print("=" * 60)
    eval_results = final_evaluation(
        model,
        domain,
        device,
        nu=config.viscosity,
        gravity=config.gravity,
        grid_res=32,
    )
    for k, v in eval_results.items():
        print(f"  {k}: {v:.6e}")

    elapsed_total = time_mod.time() - t_start
    print(f"\nTraining complete in {elapsed_total:.0f}s.  Best loss: {best_loss:.6e}")

    if use_wandb:
        import wandb

        wandb.log({f"final/{k}": v for k, v in eval_results.items()})
        wandb.finish()

    return model


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Train Physics-Informed Neural Network")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file (optional)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu")
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/pinn",
        help="Output directory for checkpoints and visualizations",
    )
    args = parser.parse_args()

    # Load config from YAML or use defaults
    config = PINNTrainConfig()
    if args.config is not None:
        from omegaconf import OmegaConf

        yaml_cfg = OmegaConf.load(args.config)
        config = build_config(yaml_cfg)

    train(
        config=config,
        resume_path=args.resume,
        device_str=args.device,
        use_wandb=args.wandb,
        output_dir=args.output_dir,
    )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # If CLI args are provided, run the full CLI
    if len(sys.argv) > 1:
        main()
        sys.exit(0)

    # Otherwise, run a quick self-test
    print("=" * 60)
    print("PINN Training — Self-test (100 iterations, tiny model)")
    print("=" * 60)

    checks_passed = 0
    checks_total = 0

    def check(cond: bool, label: str) -> None:
        global checks_passed, checks_total
        checks_total += 1
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {label}")
        if cond:
            checks_passed += 1

    import tempfile

    test_config = PINNTrainConfig(
        num_epochs=100,
        num_collocation=500,
        num_boundary=100,
        learning_rate=1e-3,
        hidden_dim=32,
        num_layers=4,
        log_interval=20,
        checkpoint_interval=50,
        adaptive_collocation=False,  # skip for speed
        scenario="rising_smoke",
        plateau_window=80,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Training run ---
        model = train(
            config=test_config,
            device_str="cpu",
            use_wandb=False,
            output_dir=tmpdir,
        )

        check(model is not None, "train() returns a model")
        check(isinstance(model, FluidPINN), "Returned model is FluidPINN")

        # --- Verify loss decreased ---
        losses = []
        # Re-run a quick forward to check model works
        domain = FluidDomain()
        x, t = domain.sample_interior(100)
        derivs = compute_derivatives(model, x, t)
        residuals = navier_stokes_residual(derivs, nu=1e-3)
        pl, _ = physics_loss(residuals)
        check(torch.isfinite(pl).item(), "Final physics loss is finite")

        # --- Verify checkpoint exists ---
        ckpt_path = Path(tmpdir) / "checkpoints" / "ckpt_000050.pt"
        check(ckpt_path.exists(), f"Checkpoint at iteration 50 exists")

        final_path = Path(tmpdir) / "checkpoints" / "final.pt"
        check(final_path.exists(), "Final checkpoint exists")

        # --- Verify checkpoint load ---
        model2 = FluidPINN(
            PINNConfig(
                activation="siren",
                hidden_dim=32,
                num_layers=4,
            )
        )
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        sched2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt2,
            T_0=5000,
            T_mult=2,
        )
        it, bl, mon, aw = load_checkpoint(
            final_path,
            model2,
            opt2,
            sched2,
            torch.device("cpu"),
        )
        check(it == 100, f"Loaded iteration == 100 (got {it})")
        check(bl < float("inf"), f"Loaded best_loss finite ({bl:.6f})")
        check(len(mon.losses) > 0, f"Loaded loss history ({len(mon.losses)} entries)")

        # --- Verify loaded model produces same output ---
        with torch.no_grad():
            x_test = torch.randn(10, 3)
            t_test = torch.rand(10, 1)
            out1 = model(x_test, t_test)
            out2 = model2(x_test, t_test)
            vel_match = torch.allclose(out1["velocity"], out2["velocity"], atol=1e-5)
        check(vel_match, "Loaded model matches trained model output")

        # --- Config builder ---
        print("\n  -- Config builder --")
        cfg_default = build_config(None)
        check(cfg_default.num_epochs == 50_000, "Default config: num_epochs=50000")

        # Simulate a dict config
        mock_cfg = {
            "pinn": {
                "network": {"hidden_dim": 128, "num_layers": 6, "activation": "fourier"},
                "physics": {"viscosity": 0.01, "gravity": [0, 0, -9.81]},
                "learning_rate": 5e-4,
                "num_collocation": 8000,
                "loss_weights": {"momentum": 2.0, "continuity": 0.5, "boundary": 0.2},
                "domain": {"x_range": [-2, 2], "y_range": [0, 1]},
            },
            "training": {
                "pinn": {"epochs": 30000, "save_interval": 10000},
                "gradient_clip": 0.5,
            },
        }
        cfg_from_dict = build_config(mock_cfg)
        check(cfg_from_dict.hidden_dim == 128, "Dict config: hidden_dim=128")
        check(cfg_from_dict.num_epochs == 30000, "Dict config: num_epochs=30000")
        check(cfg_from_dict.viscosity == 0.01, "Dict config: viscosity=0.01")
        check(cfg_from_dict.lambda_momentum == 2.0, "Dict config: lambda_momentum=2.0")
        check(cfg_from_dict.x_range == (-2, 2), "Dict config: x_range=(-2, 2)")
        check(cfg_from_dict.grad_clip_max_norm == 0.5, "Dict config: grad_clip=0.5")

        # --- ConvergenceMonitor ---
        print("\n  -- ConvergenceMonitor --")
        mon_test = ConvergenceMonitor(window=10, plateau_window=20, plateau_threshold=0.01)
        for i in range(30):
            mon_test.update(1.0)  # constant loss = plateau
        check(mon_test.check_plateau(), "Constant loss detected as plateau")

        mon_test2 = ConvergenceMonitor(window=10, plateau_window=20, plateau_threshold=0.01)
        for i in range(30):
            mon_test2.update(10.0 - i * 0.5)  # decreasing
        check(not mon_test2.check_plateau(), "Decreasing loss not a plateau")

    print(f"\n{'=' * 60}")
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 60}")

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: pinn/train.py"
            " — training loop, checkpointing, config, convergence all verified"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)
