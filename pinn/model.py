#!/usr/bin/env python3
"""Physics-Informed Neural Network for fluid velocity/pressure/density fields.

Architecture options
--------------------
- **SIREN** (Sinusoidal Representation Network): ``sin(omega * (Wx + b))``
  activations with specialised initialisation.
- **Fourier Feature Network**: random Fourier feature encoding followed by a
  ReLU MLP with skip connections.

Both architectures map ``(x, y, z, t) -> (u, v, w, p, rho)`` and are fully
differentiable with ``torch.autograd.grad(..., create_graph=True)`` so that
Navier-Stokes residuals can be computed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# 1. SIREN layer
# ---------------------------------------------------------------------------


class SirenLayer(nn.Module):
    """Single SIREN layer: ``sin(omega * (Wx + b))``.

    Parameters
    ----------
    in_features, out_features : int
        Linear layer dimensions.
    omega : float
        Frequency multiplier.
    is_first : bool
        If ``True``, use first-layer SIREN initialisation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega: float = 30.0,
        is_first: bool = False,
    ) -> None:
        super().__init__()
        self.omega = omega
        self.linear = nn.Linear(in_features, out_features)
        self._init_weights(is_first, in_features)

    def _init_weights(self, is_first: bool, fan_in: int) -> None:
        with torch.no_grad():
            if is_first:
                bound = 1.0 / fan_in
            else:
                bound = math.sqrt(6.0 / fan_in) / self.omega
            self.linear.weight.uniform_(-bound, bound)
            # Bias uniform in the same range (standard SIREN practice)
            self.linear.bias.uniform_(-bound, bound)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega * self.linear(x))


# ---------------------------------------------------------------------------
# 2. Fourier feature encoding
# ---------------------------------------------------------------------------


class FourierFeatureEncoding(nn.Module):
    """Random Fourier feature encoding.

    Maps an input ``v`` of dimension ``d`` to::

        [sin(2 pi B v), cos(2 pi B v)]

    where ``B`` is a fixed ``(num_features, d)`` matrix sampled from
    ``N(0, sigma^2)``.

    Parameters
    ----------
    input_dim : int
        Dimensionality of the input (typically 4 for x,y,z,t).
    num_features : int
        Number of Fourier features (output is ``2 * num_features``).
    sigma : float
        Standard deviation for the random frequency matrix.
    """

    def __init__(
        self,
        input_dim: int = 4,
        num_features: int = 256,
        sigma: float = 2.0,
    ) -> None:
        super().__init__()
        B = torch.randn(num_features, input_dim) * sigma
        self.register_buffer("B", B)  # not a learnable parameter

    @property
    def output_dim(self) -> int:
        return self.B.shape[0] * 2

    def forward(self, x: Tensor) -> Tensor:
        # x: (B_batch, input_dim)
        proj = 2.0 * math.pi * F.linear(x, self.B)  # (B_batch, num_features)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)


# ---------------------------------------------------------------------------
# 3. FluidPINN
# ---------------------------------------------------------------------------


@dataclass
class PINNConfig:
    """Configuration for :class:`FluidPINN`."""

    activation: str = "siren"  # "siren" or "fourier"
    hidden_dim: int = 256
    num_layers: int = 8
    omega_0: float = 30.0  # SIREN first-layer omega
    omega_hidden: float = 30.0  # SIREN hidden-layer omega
    num_fourier_features: int = 256
    encoding_scale: float = 2.0  # sigma for Fourier features


class FluidPINN(nn.Module):
    """Physics-informed neural network for incompressible fluid fields.

    Maps ``(x, y, z, t)`` to ``(u, v, w, p, rho)`` with full autograd
    support for computing Navier-Stokes residuals.

    Parameters
    ----------
    config : PINNConfig or dict
        Network hyperparameters.
    """

    def __init__(self, config: PINNConfig | dict | None = None) -> None:
        super().__init__()

        if config is None:
            config = PINNConfig()
        elif isinstance(config, dict):
            config = PINNConfig(
                **{k: v for k, v in config.items() if k in PINNConfig.__dataclass_fields__}
            )
        self.config = config

        input_dim = 4  # (x, y, z, t)
        H = config.hidden_dim
        L = config.num_layers
        self.skip_layer = L // 2  # skip connection injected here

        if config.activation == "siren":
            self._build_siren(input_dim, H, L)
        elif config.activation == "fourier":
            self._build_fourier(input_dim, H, L)
        else:
            raise ValueError(
                f"Unknown activation: {config.activation!r}. " "Choose 'siren' or 'fourier'."
            )

        # Output head: hidden_dim -> 5 (u, v, w, p, rho)
        self.output_linear = nn.Linear(H, 5)
        # Small-weight init so initial outputs are near zero but non-zero
        # (zero init would kill all input gradients, breaking PINN training)
        if config.activation == "siren":
            bound = math.sqrt(6.0 / H) / config.omega_hidden
            self.output_linear.weight.data.uniform_(-bound, bound)
            self.output_linear.bias.data.uniform_(-bound, bound)
        # For fourier: keep default Kaiming uniform init from nn.Linear

        # Softplus for density (ensures rho > 0)
        self.softplus = nn.Softplus(beta=1.0)

    # -- SIREN backbone --------------------------------------------------

    def _build_siren(self, input_dim: int, H: int, L: int) -> None:
        self.encoding = None

        layers: list[SirenLayer] = []
        # First layer: input_dim -> H
        layers.append(SirenLayer(input_dim, H, omega=self.config.omega_0, is_first=True))

        for i in range(1, L):
            in_dim = H * 2 if i == self.skip_layer else H
            layers.append(SirenLayer(in_dim, H, omega=self.config.omega_hidden, is_first=False))

        self.layers = nn.ModuleList(layers)

    # -- Fourier + ReLU backbone ------------------------------------------

    def _build_fourier(self, input_dim: int, H: int, L: int) -> None:
        self.encoding = FourierFeatureEncoding(
            input_dim=input_dim,
            num_features=self.config.num_fourier_features,
            sigma=self.config.encoding_scale,
        )
        enc_dim = self.encoding.output_dim  # 2 * num_fourier_features

        layers: list[nn.Module] = []
        # First layer: enc_dim -> H
        layers.append(nn.Linear(enc_dim, H))
        for i in range(1, L):
            in_dim = H * 2 if i == self.skip_layer else H
            layers.append(nn.Linear(in_dim, H))

        self.layers = nn.ModuleList(layers)

    # -- Forward -----------------------------------------------------------

    def forward(self, x: Tensor, t: Tensor) -> Dict[str, Tensor]:
        """Forward pass.

        Parameters
        ----------
        x : (B, 3)
            Spatial coordinates.
        t : (B, 1)
            Time values.

        Returns
        -------
        dict
            ``"velocity"`` (B, 3), ``"pressure"`` (B, 1), ``"density"`` (B, 1).
        """
        inp = torch.cat([x, t], dim=-1)  # (B, 4)

        if self.encoding is not None:
            inp = self.encoding(inp)  # Fourier features

        # Pass through layers with skip connection
        h = self.layers[0](inp)
        if self.config.activation == "fourier":
            h = F.relu(h)

        h_skip = h  # save for skip connection

        for i in range(1, len(self.layers)):
            if i == self.skip_layer:
                h = torch.cat([h, h_skip], dim=-1)
            h = self.layers[i](h)
            if self.config.activation == "fourier":
                h = F.relu(h)

        out = self.output_linear(h)  # (B, 5)

        velocity = out[:, :3]  # (B, 3) — u, v, w
        pressure = out[:, 3:4]  # (B, 1) — p
        density = self.softplus(out[:, 4:5])  # (B, 1) — rho > 0

        return {
            "velocity": velocity,
            "pressure": pressure,
            "density": density,
        }


# ---------------------------------------------------------------------------
# 4. FluidDomain
# ---------------------------------------------------------------------------


class FluidDomain:
    """Spatial-temporal domain for collocation point sampling.

    Parameters
    ----------
    x_range, y_range, z_range : tuple of (float, float)
        Spatial extents.
    t_range : tuple of (float, float)
        Temporal extent.
    """

    def __init__(
        self,
        x_range: Tuple[float, float] = (-1.0, 1.0),
        y_range: Tuple[float, float] = (-1.0, 1.0),
        z_range: Tuple[float, float] = (-1.0, 1.0),
        t_range: Tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self.x_range = x_range
        self.y_range = y_range
        self.z_range = z_range
        self.t_range = t_range

    def _uniform(self, lo: float, hi: float, n: int, device: torch.device) -> Tensor:
        return lo + (hi - lo) * torch.rand(n, 1, device=device)

    def sample_interior(
        self,
        n_points: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Tensor, Tensor]:
        """Uniformly sample interior collocation points.

        Returns
        -------
        x : (n_points, 3), requires_grad=True
        t : (n_points, 1), requires_grad=True
        """
        x = torch.cat(
            [
                self._uniform(*self.x_range, n_points, device),
                self._uniform(*self.y_range, n_points, device),
                self._uniform(*self.z_range, n_points, device),
            ],
            dim=-1,
        )
        t = self._uniform(*self.t_range, n_points, device)
        x.requires_grad_(True)
        t.requires_grad_(True)
        return x, t

    def sample_boundary(
        self,
        n_points: int,
        face: str,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Tensor, Tensor]:
        """Sample points on a specific domain face.

        Parameters
        ----------
        face : str
            One of ``"x_min"``, ``"x_max"``, ``"y_min"``, ``"y_max"``,
            ``"z_min"``, ``"z_max"``.

        Returns
        -------
        x : (n_points, 3), requires_grad=True
        t : (n_points, 1), requires_grad=True
        """
        coords = {
            "x": self._uniform(*self.x_range, n_points, device),
            "y": self._uniform(*self.y_range, n_points, device),
            "z": self._uniform(*self.z_range, n_points, device),
        }

        axis, side = face.rsplit("_", 1)
        if axis not in ("x", "y", "z") or side not in ("min", "max"):
            raise ValueError(
                f"Invalid face {face!r}. Expected one of: "
                "x_min, x_max, y_min, y_max, z_min, z_max"
            )
        rng = getattr(self, f"{axis}_range")
        val = rng[0] if side == "min" else rng[1]
        coords[axis] = torch.full((n_points, 1), val, device=device, dtype=torch.float32)

        x = torch.cat([coords["x"], coords["y"], coords["z"]], dim=-1)
        t = self._uniform(*self.t_range, n_points, device)
        x.requires_grad_(True)
        t.requires_grad_(True)
        return x, t

    def sample_initial(
        self,
        n_points: int,
        device: torch.device = torch.device("cpu"),
    ) -> Tuple[Tensor, Tensor]:
        """Sample points at ``t = t_range[0]`` (initial condition).

        Returns
        -------
        x : (n_points, 3), requires_grad=True
        t : (n_points, 1), requires_grad=True
        """
        x = torch.cat(
            [
                self._uniform(*self.x_range, n_points, device),
                self._uniform(*self.y_range, n_points, device),
                self._uniform(*self.z_range, n_points, device),
            ],
            dim=-1,
        )
        t = torch.full((n_points, 1), self.t_range[0], device=device, dtype=torch.float32)
        x.requires_grad_(True)
        t.requires_grad_(True)
        return x, t


# ---------------------------------------------------------------------------
# 5. Utilities
# ---------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> int:
    """Total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    checks_passed = 0
    checks_total = 0

    def check(cond: bool, label: str) -> None:
        global checks_passed, checks_total
        checks_total += 1
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {label}")
        if cond:
            checks_passed += 1

    # ---- Test both architectures ----
    for arch in ("siren", "fourier"):
        print(f"\n=== FluidPINN ({arch}) ===")

        cfg = PINNConfig(activation=arch, hidden_dim=64, num_layers=6)
        model = FluidPINN(cfg)
        n_params = count_parameters(model)
        print(f"  Parameters: {n_params:,}")
        check(n_params > 0, f"Has trainable parameters ({n_params:,})")

        B = 32
        x = torch.randn(B, 3, requires_grad=True)
        t = torch.rand(B, 1, requires_grad=True)

        out = model(x, t)

        # Shape checks
        check(out["velocity"].shape == (B, 3), f"velocity shape = ({B}, 3)")
        check(out["pressure"].shape == (B, 1), f"pressure shape = ({B}, 1)")
        check(out["density"].shape == (B, 1), f"density shape = ({B}, 1)")

        # Density positive
        check(out["density"].min().item() > 0, "density > 0")

        # --- Autograd: first derivatives ---
        print("  -- Autograd (first derivatives) --")
        u = out["velocity"][:, 0:1]  # (B, 1)

        grad_u = torch.autograd.grad(
            u,
            x,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        check(grad_u.shape == (B, 3), f"du/dx shape = ({B}, 3)")
        check(grad_u.requires_grad, "du/dx tracks gradients (create_graph)")

        grad_u_t = torch.autograd.grad(
            u,
            t,
            grad_outputs=torch.ones_like(u),
            create_graph=True,
            retain_graph=True,
        )[0]
        check(grad_u_t.shape == (B, 1), f"du/dt shape = ({B}, 1)")
        check(grad_u_t.requires_grad, "du/dt tracks gradients")

        # --- Autograd: second derivatives ---
        print("  -- Autograd (second derivatives) --")
        du_dx = grad_u[:, 0:1]  # du/dx
        grad2_u = torch.autograd.grad(
            du_dx,
            x,
            grad_outputs=torch.ones_like(du_dx),
            create_graph=True,
            retain_graph=True,
        )[0]
        check(grad2_u.shape == (B, 3), f"d2u/dx2 shape = ({B}, 3)")
        check(grad2_u.requires_grad, "d2u/dx2 tracks gradients")

        # --- Backward works ---
        print("  -- Backward --")
        loss = (u**2).mean() + (grad_u**2).mean() + (grad2_u**2).mean()
        loss.backward()
        check(x.grad is not None, "x has gradient after backward")
        check(t.grad is not None, "t has gradient after backward")

    # ---- FluidDomain ----
    print("\n=== FluidDomain ===")
    domain = FluidDomain(x_range=(-2, 2), y_range=(0, 1), z_range=(-1, 1), t_range=(0, 0.5))

    x_int, t_int = domain.sample_interior(100)
    check(x_int.shape == (100, 3), "interior x shape")
    check(t_int.shape == (100, 1), "interior t shape")
    check(x_int.requires_grad, "interior x requires_grad")
    check(t_int.requires_grad, "interior t requires_grad")
    check(x_int[:, 0].min().item() >= -2.0, "interior x in range")
    check(x_int[:, 0].max().item() <= 2.0, "interior x in range (max)")
    check(t_int.min().item() >= 0.0, "interior t >= 0")
    check(t_int.max().item() <= 0.5, "interior t <= 0.5")

    x_bnd, t_bnd = domain.sample_boundary(50, "y_min")
    check(x_bnd.shape == (50, 3), "boundary x shape")
    check((x_bnd[:, 1] == 0.0).all(), "y_min face: y == 0")
    check(x_bnd.requires_grad, "boundary x requires_grad")

    x_bnd_max, _ = domain.sample_boundary(50, "z_max")
    check((x_bnd_max[:, 2] == 1.0).all(), "z_max face: z == 1")

    x_init, t_init = domain.sample_initial(80)
    check(x_init.shape == (80, 3), "initial x shape")
    check((t_init == 0.0).all(), "initial t == 0")
    check(x_init.requires_grad, "initial x requires_grad")
    check(t_init.requires_grad, "initial t requires_grad")

    # ---- SirenLayer standalone ----
    print("\n=== SirenLayer ===")
    sl = SirenLayer(4, 64, omega=30.0, is_first=True)
    out_sl = sl(torch.randn(10, 4))
    check(out_sl.shape == (10, 64), "SirenLayer output shape")
    check(out_sl.abs().max().item() <= 1.0, "SirenLayer output in [-1, 1]")

    # ---- FourierFeatureEncoding standalone ----
    print("\n=== FourierFeatureEncoding ===")
    ffe = FourierFeatureEncoding(input_dim=4, num_features=128, sigma=2.0)
    out_ff = ffe(torch.randn(10, 4))
    check(out_ff.shape == (10, 256), "FourierFeatures output shape = (10, 256)")
    check(ffe.output_dim == 256, "output_dim property = 256")

    # ---- Dict config ----
    print("\n=== Dict config ===")
    model_d = FluidPINN({"activation": "siren", "hidden_dim": 32, "num_layers": 4})
    check(model_d.config.hidden_dim == 32, "Dict config: hidden_dim=32")
    out_d = model_d(torch.randn(5, 3, requires_grad=True), torch.rand(5, 1, requires_grad=True))
    check(out_d["velocity"].shape == (5, 3), "Dict config: forward works")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 60}")

    if checks_passed == checks_total:
        print("\nTEST PASSED: pinn/model.py — SIREN, Fourier, FluidDomain, autograd all verified")
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)
