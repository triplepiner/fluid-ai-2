#!/usr/bin/env python3
"""Dynamic 3D Gaussian splatting model for time-varying fluid scenes.

The scene is represented as a set of anisotropic 3D Gaussian primitives in a
canonical frame.  A deformation network predicts per-Gaussian offsets
conditioned on normalised time, allowing the Gaussians to move, rotate, rescale,
and change opacity over the course of a sequence.

Modules
-------
- :class:`GaussianModel` — canonical Gaussian parameters (positions, rotations,
  scales, opacities, SH colour coefficients).
- :class:`DeformationNetwork` — time-conditioned MLP that predicts per-Gaussian
  deformation deltas.
- :class:`DynamicGaussianModel` — combines the two into a single forward pass
  that returns deformed Gaussian properties at any query time.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def inverse_sigmoid(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable inverse of the sigmoid function."""
    x = x.clamp(1e-5, 1.0 - 1e-5)
    return torch.log(x / (1.0 - x))


def build_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions ``(w, x, y, z)`` to 3x3 rotation matrices.

    Parameters
    ----------
    q : torch.Tensor
        ``(..., 4)`` quaternions.

    Returns
    -------
    torch.Tensor
        ``(..., 3, 3)`` rotation matrices.
    """
    q = F.normalize(q, p=2, dim=-1)
    w, x, y, z = q.unbind(-1)

    R = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )

    return R.reshape(*q.shape[:-1], 3, 3)


def build_covariance_3d(
    scaling: torch.Tensor,
    rotation_matrix: torch.Tensor,
) -> torch.Tensor:
    """Build 3D covariance matrices from scaling and rotation.

    ``Sigma = R @ S @ S^T @ R^T`` where ``S = diag(scaling)``.

    Parameters
    ----------
    scaling : torch.Tensor
        ``(N, 3)`` positive scale values.
    rotation_matrix : torch.Tensor
        ``(N, 3, 3)`` rotation matrices.

    Returns
    -------
    torch.Tensor
        ``(N, 3, 3)`` symmetric positive semi-definite covariance matrices.
    """
    S = torch.zeros(*scaling.shape[:-1], 3, 3, device=scaling.device, dtype=scaling.dtype)
    S[..., 0, 0] = scaling[..., 0]
    S[..., 1, 1] = scaling[..., 1]
    S[..., 2, 2] = scaling[..., 2]

    # RS = R @ S
    RS = rotation_matrix @ S  # (N, 3, 3)
    return RS @ RS.transpose(-1, -2)  # (N, 3, 3)


def knn_points(xyz: torch.Tensor, k: int = 3) -> torch.Tensor:
    """Compute distances to the k nearest neighbours for each point.

    Uses brute-force pairwise distances.  Efficient enough for typical
    point-cloud sizes (< 500 k points); switch to a spatial tree for
    millions of points.

    Parameters
    ----------
    xyz : torch.Tensor
        ``(N, 3)`` point positions.
    k : int
        Number of neighbours.

    Returns
    -------
    torch.Tensor
        ``(N, k)`` squared distances to the k nearest neighbours
        (excluding self).
    """
    N = xyz.shape[0]
    k = min(k, N - 1)
    if k <= 0:
        return torch.zeros(N, 1, device=xyz.device, dtype=xyz.dtype)

    # Chunk the computation for large point clouds to avoid OOM.
    CHUNK = 8192
    all_dists: list[torch.Tensor] = []
    for i in range(0, N, CHUNK):
        chunk = xyz[i : i + CHUNK]  # (C, 3)
        dist = torch.cdist(chunk, xyz, p=2.0) ** 2  # (C, N)
        # Set self-distances to +inf so they're excluded from topk.
        dist[
            torch.arange(chunk.shape[0], device=xyz.device),
            torch.arange(i, i + chunk.shape[0], device=xyz.device),
        ] = float("inf")
        topk = dist.topk(k, dim=-1, largest=False)  # smallest k
        all_dists.append(topk.values)

    return torch.cat(all_dists, dim=0)  # (N, k)


# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------


class SinusoidalEncoding(nn.Module):
    """Sinusoidal positional encoding.

    Maps a ``D``-dimensional input to ``D + D * 2 * L`` dimensions by
    appending ``[sin(2^l pi x), cos(2^l pi x)]`` for ``l = 0 ... L-1``.
    """

    def __init__(self, input_dim: int, num_freqs: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.output_dim = input_dim + input_dim * 2 * num_freqs
        freqs = torch.tensor([2**l for l in range(num_freqs)], dtype=torch.float32) * math.pi
        self.register_buffer("freqs", freqs)  # (L,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode *x*.

        Parameters
        ----------
        x : torch.Tensor
            ``(..., D)`` input.

        Returns
        -------
        torch.Tensor
            ``(..., D + D * 2 * L)`` encoded output.
        """
        # x: (..., D), freqs: (L,)
        # outer product: (..., D, L)
        scaled = x.unsqueeze(-1) * self.freqs  # (..., D, L)
        sin_part = torch.sin(scaled)
        cos_part = torch.cos(scaled)
        # Interleave sin/cos and flatten: (..., D * 2 * L)
        encoded = torch.cat([sin_part, cos_part], dim=-1)  # (..., D, 2L)
        encoded = encoded.reshape(*x.shape[:-1], -1)  # (..., D*2*L)
        return torch.cat([x, encoded], dim=-1)  # (..., D + D*2*L)


# ---------------------------------------------------------------------------
# GaussianModel
# ---------------------------------------------------------------------------


class GaussianModel(nn.Module):
    """Canonical 3D Gaussian primitives.

    Parameters
    ----------
    sh_degree : int
        Maximum spherical-harmonics degree (0–3).  Total SH coefficients per
        colour channel: ``(sh_degree + 1)^2``.
    num_points : int
        Initial number of Gaussians (used only to pre-allocate if
        :meth:`initialize_from_point_cloud` is not called).
    """

    def __init__(self, sh_degree: int = 3, num_points: int = 100_000) -> None:
        super().__init__()
        self.sh_degree = sh_degree
        self.num_sh_coeffs = (sh_degree + 1) ** 2

        self._xyz = nn.Parameter(torch.zeros(num_points, 3))
        self._rotation = nn.Parameter(torch.zeros(num_points, 4))
        self._scaling = nn.Parameter(torch.zeros(num_points, 3))
        self._opacity = nn.Parameter(torch.zeros(num_points, 1))
        self._features_dc = nn.Parameter(torch.zeros(num_points, 1, 3))
        self._features_rest = nn.Parameter(torch.zeros(num_points, self.num_sh_coeffs - 1, 3))

        # Set identity quaternion as default.
        with torch.no_grad():
            self._rotation[:, 0] = 1.0  # (w, x, y, z) = (1, 0, 0, 0)

    @torch.no_grad()
    def initialize_from_point_cloud(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
    ) -> None:
        """Initialise from a point cloud.

        Parameters
        ----------
        xyz : torch.Tensor
            ``(N, 3)`` point positions.
        rgb : torch.Tensor
            ``(N, 3)`` point colours in ``[0, 1]``.
        """
        N = xyz.shape[0]
        device = xyz.device

        # --- Positions ---
        self._xyz = nn.Parameter(xyz.float().clone())

        # --- Scales from KNN ---
        knn_sq = knn_points(xyz.float(), k=3)  # (N, 3)
        mean_knn_dist = knn_sq.mean(dim=-1).sqrt()  # (N,)
        mean_knn_dist = mean_knn_dist.clamp(min=1e-7)
        log_scale = torch.log(mean_knn_dist).unsqueeze(-1).expand(-1, 3)
        self._scaling = nn.Parameter(log_scale.clone())

        # --- Rotations: identity quaternion ---
        rot = torch.zeros(N, 4, device=device)
        rot[:, 0] = 1.0
        self._rotation = nn.Parameter(rot)

        # --- Opacity: inverse_sigmoid(0.1) ---
        opacity_logit = inverse_sigmoid(torch.full((N, 1), 0.1, device=device))
        self._opacity = nn.Parameter(opacity_logit)

        # --- SH colour: DC term from RGB, rest zero ---
        # The DC coefficient is rgb (mapped to SH-convention range).
        features_dc = rgb.float().clamp(0, 1).unsqueeze(1)  # (N, 1, 3)
        self._features_dc = nn.Parameter(features_dc)

        features_rest = torch.zeros(N, self.num_sh_coeffs - 1, 3, device=device)
        self._features_rest = nn.Parameter(features_rest)

    # -- Property accessors ------------------------------------------------

    def get_xyz(self) -> torch.Tensor:
        """``(N, 3)`` canonical positions."""
        return self._xyz

    def get_rotation(self) -> torch.Tensor:
        """``(N, 4)`` normalised canonical quaternions."""
        return F.normalize(self._rotation, p=2, dim=-1)

    def get_scaling(self) -> torch.Tensor:
        """``(N, 3)`` positive canonical scales (exp of log-scales)."""
        return torch.exp(self._scaling)

    def get_opacity(self) -> torch.Tensor:
        """``(N, 1)`` canonical opacities in ``(0, 1)``."""
        return torch.sigmoid(self._opacity)

    def get_features(self) -> torch.Tensor:
        """``(N, (sh_degree+1)^2, 3)`` full SH coefficients."""
        return torch.cat([self._features_dc, self._features_rest], dim=1)

    def get_covariance(self, scaling_modifier: float = 1.0) -> torch.Tensor:
        """``(N, 3, 3)`` covariance matrices.

        ``Sigma = R S S^T R^T`` where ``R`` is from the quaternion and
        ``S = diag(scaling * scaling_modifier)``.
        """
        scaling = self.get_scaling() * scaling_modifier
        R = build_rotation_matrix(self.get_rotation())
        return build_covariance_3d(scaling, R)

    @property
    def num_points(self) -> int:
        return self._xyz.shape[0]


# ---------------------------------------------------------------------------
# DeformationNetwork
# ---------------------------------------------------------------------------


class DeformationNetwork(nn.Module):
    """Time-conditioned MLP that predicts per-Gaussian deformation deltas.

    Architecture
    ------------
    * Inputs: positionally-encoded canonical xyz + positionally-encoded time.
    * 4 hidden layers of width 256 with ReLU.
    * Skip connection at layer 2 (concatenate encoded input again).
    * Four small output heads (delta_xyz, delta_rotation, delta_scaling,
      delta_opacity) initialised near zero so the model starts from the
      canonical configuration.

    Parameters
    ----------
    pos_freq : int
        Number of sinusoidal frequencies for xyz encoding.
    time_freq : int
        Number of sinusoidal frequencies for time encoding.
    hidden_dim : int
        Width of the hidden layers.
    num_layers : int
        Number of hidden layers.
    skip_layer : int
        Layer index at which to inject the skip connection (0-indexed).
    """

    def __init__(
        self,
        pos_freq: int = 6,
        time_freq: int = 10,
        hidden_dim: int = 256,
        num_layers: int = 4,
        skip_layer: int = 2,
    ) -> None:
        super().__init__()
        self.skip_layer = skip_layer

        self.pos_enc = SinusoidalEncoding(3, pos_freq)
        self.time_enc = SinusoidalEncoding(1, time_freq)

        input_dim = self.pos_enc.output_dim + self.time_enc.output_dim

        # Build hidden layers.
        layers: list[nn.Module] = []
        in_dim = input_dim
        for i in range(num_layers):
            if i == skip_layer:
                in_dim += input_dim  # skip-connection concatenation
            layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.layers = nn.ModuleList(layers)

        # Output heads.
        self.head_xyz = nn.Linear(hidden_dim, 3)
        self.head_rotation = nn.Linear(hidden_dim, 4)
        self.head_scaling = nn.Linear(hidden_dim, 3)
        self.head_opacity = nn.Linear(hidden_dim, 1)

        # Initialise output heads with very small weights.
        for head in [self.head_xyz, self.head_rotation, self.head_scaling, self.head_opacity]:
            nn.init.normal_(head.weight, std=1e-4)
            nn.init.zeros_(head.bias)

    def forward(
        self,
        canonical_xyz: torch.Tensor,
        time: float | torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict deformation deltas.

        Parameters
        ----------
        canonical_xyz : torch.Tensor
            ``(N, 3)`` canonical Gaussian positions.
        time : float or torch.Tensor
            Normalised time in ``[0, 1]``.

        Returns
        -------
        dict[str, torch.Tensor]
            Keys: ``"delta_xyz"`` (N, 3), ``"delta_rotation"`` (N, 4),
            ``"delta_scaling"`` (N, 3), ``"delta_opacity"`` (N, 1).
        """
        N = canonical_xyz.shape[0]
        device = canonical_xyz.device

        # Encode time.
        if isinstance(time, (int, float)):
            time_t = torch.tensor([[time]], device=device, dtype=torch.float32)
        else:
            time_t = time.reshape(1, 1).to(device=device, dtype=torch.float32)
        time_encoded = self.time_enc(time_t).expand(N, -1)  # (N, D_time)

        # Encode position.
        pos_encoded = self.pos_enc(canonical_xyz)  # (N, D_pos)

        # Concatenate.
        x_input = torch.cat([pos_encoded, time_encoded], dim=-1)  # (N, D_in)

        h = x_input
        for i, layer in enumerate(self.layers):
            if i == self.skip_layer:
                h = torch.cat([h, x_input], dim=-1)
            h = F.relu(layer(h))

        return {
            "delta_xyz": self.head_xyz(h),
            "delta_rotation": self.head_rotation(h),
            "delta_scaling": self.head_scaling(h),
            "delta_opacity": self.head_opacity(h),
        }


# ---------------------------------------------------------------------------
# DynamicGaussianModel
# ---------------------------------------------------------------------------


class DynamicGaussianModel(nn.Module):
    """Canonical Gaussians + time-conditioned deformation network.

    Parameters
    ----------
    sh_degree : int
        Spherical-harmonics degree for the :class:`GaussianModel`.
    num_points : int
        Initial point count.
    pos_freq, time_freq : int
        Frequency counts for the :class:`DeformationNetwork` encodings.
    """

    def __init__(
        self,
        sh_degree: int = 3,
        num_points: int = 100_000,
        pos_freq: int = 6,
        time_freq: int = 10,
    ) -> None:
        super().__init__()
        self.gaussian_model = GaussianModel(
            sh_degree=sh_degree,
            num_points=num_points,
        )
        self.deformation_network = DeformationNetwork(
            pos_freq=pos_freq,
            time_freq=time_freq,
        )

    def forward(self, time: float | torch.Tensor) -> dict[str, torch.Tensor]:
        """Return deformed Gaussian properties at the given *time*.

        Returns
        -------
        dict[str, torch.Tensor]
            ``"xyz"`` (N, 3), ``"rotation"`` (N, 4), ``"scaling"`` (N, 3),
            ``"opacity"`` (N, 1), ``"features"`` (N, C, 3),
            ``"covariance"`` (N, 3, 3).
        """
        gm = self.gaussian_model

        canonical_xyz = gm.get_xyz()
        deltas = self.deformation_network(canonical_xyz, time)

        # Apply deformations.
        xyz = canonical_xyz + deltas["delta_xyz"]
        rotation = F.normalize(
            gm.get_rotation() + deltas["delta_rotation"],
            p=2,
            dim=-1,
        )
        scaling = torch.exp(gm._scaling + deltas["delta_scaling"])
        opacity = torch.sigmoid(gm._opacity + deltas["delta_opacity"])
        features = gm.get_features()

        # Covariance from deformed rotation + scaling.
        R = build_rotation_matrix(rotation)
        covariance = build_covariance_3d(scaling, R)

        return {
            "xyz": xyz,
            "rotation": rotation,
            "scaling": scaling,
            "opacity": opacity,
            "features": features,
            "covariance": covariance,
        }

    @torch.no_grad()
    def get_canonical(self) -> dict[str, torch.Tensor]:
        """Return canonical (un-deformed) Gaussian properties."""
        gm = self.gaussian_model
        scaling = gm.get_scaling()
        rotation = gm.get_rotation()
        R = build_rotation_matrix(rotation)
        return {
            "xyz": gm.get_xyz(),
            "rotation": rotation,
            "scaling": scaling,
            "opacity": gm.get_opacity(),
            "features": gm.get_features(),
            "covariance": build_covariance_3d(scaling, R),
        }
