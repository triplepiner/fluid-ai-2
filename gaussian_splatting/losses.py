#!/usr/bin/env python3
"""Loss functions for dynamic 3D Gaussian splatting training.

Loss terms
----------
- :func:`photometric_loss` — L1 + SSIM between rendered and target images.
- :func:`depth_loss` — Pearson correlation between rendered and monocular depth.
- :func:`temporal_consistency_loss` — flow-warped image L1 loss.
- :func:`temporal_smoothness_loss` — acceleration + velocity penalty on Gaussian
  trajectories.
- :func:`opacity_regularization` — binary-entropy penalty pushing opacities to
  0 or 1.
- :func:`scale_regularization` — penalises Gaussians exceeding a max scale.
- :func:`total_loss` — weighted combination of all terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_batch(x: Tensor, expected_c: int | None = None) -> Tensor:
    """Add a leading batch dimension if missing.

    Converts ``(C, H, W)`` to ``(1, C, H, W)`` and leaves
    ``(B, C, H, W)`` unchanged.
    """
    if x.dim() == 3:
        return x.unsqueeze(0)
    return x


def _gaussian_kernel_1d(
    size: int, sigma: float, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """1-D Gaussian kernel (normalised)."""
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-0.5 * (coords / sigma) ** 2)
    return g / g.sum()


def _gaussian_kernel_2d(
    size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype
) -> Tensor:
    """2-D separable Gaussian kernel shaped ``(channels, 1, size, size)``."""
    k1d = _gaussian_kernel_1d(size, sigma, device, dtype)
    k2d = k1d.unsqueeze(-1) @ k1d.unsqueeze(0)  # (size, size)
    return k2d.expand(channels, 1, -1, -1).contiguous()


# ---------------------------------------------------------------------------
# 1. Photometric loss (L1 + SSIM)
# ---------------------------------------------------------------------------


def _ssim(
    x: Tensor,
    y: Tensor,
    window_size: int = 11,
    sigma: float = 1.5,
) -> Tensor:
    """Compute mean SSIM between two image batches.

    Parameters
    ----------
    x, y : (B, C, H, W) tensors in the same value range.
    window_size : int
        Side length of the Gaussian window.
    sigma : float
        Standard deviation for the Gaussian window.

    Returns
    -------
    Tensor
        Scalar mean SSIM value in [-1, 1].
    """
    C = x.shape[1]
    kernel = _gaussian_kernel_2d(window_size, sigma, C, x.device, x.dtype)
    pad = window_size // 2

    mu_x = F.conv2d(x, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(y, kernel, padding=pad, groups=C)

    mu_x_sq = mu_x * mu_x
    mu_y_sq = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(x * x, kernel, padding=pad, groups=C) - mu_x_sq
    sigma_y_sq = F.conv2d(y * y, kernel, padding=pad, groups=C) - mu_y_sq
    sigma_xy = F.conv2d(x * y, kernel, padding=pad, groups=C) - mu_xy

    C1 = 0.01**2
    C2 = 0.03**2

    numerator = (2.0 * mu_xy + C1) * (2.0 * sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)

    ssim_map = numerator / denominator  # (B, C, H, W)
    return ssim_map.mean()


def photometric_loss(
    rendered: Tensor,
    target: Tensor,
    lambda_l1: float = 0.8,
    lambda_ssim: float = 0.2,
) -> Tensor:
    """Combined L1 + SSIM photometric loss.

    Parameters
    ----------
    rendered, target : (3, H, W) or (B, 3, H, W)
    lambda_l1 : float
        Weight for the L1 component.
    lambda_ssim : float
        Weight for the SSIM component.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    rendered = _ensure_batch(rendered)
    target = _ensure_batch(target)

    l1 = torch.abs(rendered - target).mean()
    ssim_val = _ssim(rendered, target, window_size=11, sigma=1.5)
    ssim_loss = 1.0 - ssim_val

    return lambda_l1 * l1 + lambda_ssim * ssim_loss


# ---------------------------------------------------------------------------
# 2. Depth loss (Pearson correlation)
# ---------------------------------------------------------------------------


def depth_loss(
    rendered_depth: Tensor,
    pseudo_depth: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Scale-and-shift-invariant depth loss via Pearson correlation.

    Parameters
    ----------
    rendered_depth, pseudo_depth : (1, H, W) or (B, 1, H, W)
    mask : (1, H, W) or (B, 1, H, W), optional
        Binary mask; loss is computed only where mask > 0.5.

    Returns
    -------
    Tensor
        Scalar loss in [0, 2] (0 = perfect correlation).
    """
    rendered_depth = _ensure_batch(rendered_depth)
    pseudo_depth = _ensure_batch(pseudo_depth)

    r = rendered_depth.reshape(rendered_depth.shape[0], -1)  # (B, HW)
    p = pseudo_depth.reshape(pseudo_depth.shape[0], -1)

    if mask is not None:
        mask = _ensure_batch(mask)
        m = mask.reshape(mask.shape[0], -1) > 0.5  # (B, HW) bool
    else:
        m = torch.ones_like(r, dtype=torch.bool)

    # Compute per-batch Pearson and average
    losses = []
    for i in range(r.shape[0]):
        mi = m[i]
        if mi.sum() < 2:
            continue
        ri = r[i][mi]
        pi = p[i][mi]

        ri_centered = ri - ri.mean()
        pi_centered = pi - pi.mean()

        numer = (ri_centered * pi_centered).sum()
        denom = torch.sqrt((ri_centered**2).sum() * (pi_centered**2).sum()).clamp(min=1e-8)

        corr = numer / denom
        losses.append(1.0 - corr)

    if len(losses) == 0:
        return torch.tensor(0.0, device=rendered_depth.device, dtype=rendered_depth.dtype)

    return torch.stack(losses).mean()


# ---------------------------------------------------------------------------
# 3. Temporal consistency loss (flow warp)
# ---------------------------------------------------------------------------


def _warp_with_flow(image: Tensor, flow: Tensor) -> Tensor:
    """Warp an image using optical flow via grid_sample.

    Parameters
    ----------
    image : (B, C, H, W)
    flow : (B, 2, H, W)  pixel displacements (dx, dy)

    Returns
    -------
    Tensor
        Warped image (B, C, H, W).
    """
    B, _, H, W = image.shape

    # Base grid in pixel coordinates
    yy, xx = torch.meshgrid(
        torch.arange(H, device=image.device, dtype=image.dtype),
        torch.arange(W, device=image.device, dtype=image.dtype),
        indexing="ij",
    )
    base_grid = torch.stack([xx, yy], dim=0).unsqueeze(0).expand(B, -1, -1, -1)  # (B, 2, H, W)

    # Displaced pixel coordinates (subtract forward flow for backward warp)
    displaced = base_grid - flow  # (B, 2, H, W)

    # Normalise to [-1, 1] for grid_sample
    displaced_x = 2.0 * displaced[:, 0] / (W - 1) - 1.0
    displaced_y = 2.0 * displaced[:, 1] / (H - 1) - 1.0
    grid = torch.stack([displaced_x, displaced_y], dim=-1)  # (B, H, W, 2)

    return F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)


def temporal_consistency_loss(
    rendered_t: Tensor,
    target_t1: Tensor,
    flow_fwd: Tensor,
) -> Tensor:
    """Flow-warped temporal consistency loss.

    Warps ``rendered_t`` by ``flow_fwd`` and compares to ``target_t1``.

    Parameters
    ----------
    rendered_t : (3, H, W) or (B, 3, H, W)
        Rendered image at time *t*.
    target_t1 : (3, H, W) or (B, 3, H, W)
        Target image at time *t + 1*.
    flow_fwd : (2, H, W) or (B, 2, H, W)
        Forward optical flow from *t* to *t + 1* in pixels.

    Returns
    -------
    Tensor
        Scalar L1 loss.
    """
    rendered_t = _ensure_batch(rendered_t)
    target_t1 = _ensure_batch(target_t1)
    flow_fwd = _ensure_batch(flow_fwd)

    warped = _warp_with_flow(rendered_t, flow_fwd)
    return torch.abs(warped - target_t1).mean()


# ---------------------------------------------------------------------------
# 4. Temporal smoothness loss
# ---------------------------------------------------------------------------


def temporal_smoothness_loss(
    xyz_t_minus_1: Tensor,
    xyz_t: Tensor,
    xyz_t_plus_1: Tensor,
) -> Tensor:
    """Trajectory smoothness loss (acceleration + velocity penalty).

    Parameters
    ----------
    xyz_t_minus_1, xyz_t, xyz_t_plus_1 : (N, 3) or (B, N, 3)
        Gaussian positions at three consecutive timesteps.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    acceleration = xyz_t_minus_1 - 2.0 * xyz_t + xyz_t_plus_1
    accel_loss = (acceleration**2).mean()

    velocity = xyz_t - xyz_t_minus_1
    velocity_penalty = (velocity**2).mean()

    return accel_loss + 0.1 * velocity_penalty


# ---------------------------------------------------------------------------
# 5. Opacity regularization
# ---------------------------------------------------------------------------


def opacity_regularization(opacity: Tensor) -> Tensor:
    """Binary-entropy penalty pushing opacities toward 0 or 1.

    Parameters
    ----------
    opacity : (N, 1), (N,), or (B, N, 1)
        Opacity values in (0, 1).

    Returns
    -------
    Tensor
        Scalar loss (higher for mid-range opacities).
    """
    eps = 1e-6
    o = opacity.clamp(eps, 1.0 - eps)
    entropy = -(o * torch.log(o + eps) + (1.0 - o) * torch.log(1.0 - o + eps))
    return entropy.mean()


# ---------------------------------------------------------------------------
# 6. Scale regularization
# ---------------------------------------------------------------------------


def scale_regularization(scaling: Tensor, max_scale: float = 0.1) -> Tensor:
    """Penalise Gaussians with scales exceeding *max_scale*.

    Parameters
    ----------
    scaling : (N, 3) or (B, N, 3)
        Positive scale values.
    max_scale : float
        Threshold above which the penalty activates.

    Returns
    -------
    Tensor
        Scalar loss.
    """
    return F.relu(scaling - max_scale).mean()


# ---------------------------------------------------------------------------
# 7. Combined loss
# ---------------------------------------------------------------------------


@dataclass
class LossConfig:
    """Weights for each loss term."""

    lambda_photo: float = 1.0
    lambda_l1: float = 0.8
    lambda_ssim: float = 0.2
    lambda_depth: float = 0.5
    lambda_temporal: float = 0.1
    lambda_smooth: float = 0.01
    lambda_opacity_reg: float = 0.01
    lambda_scale_reg: float = 0.01
    max_scale: float = 0.1


def total_loss(
    rendered: Tensor,
    target: Tensor,
    rendered_depth: Tensor | None = None,
    pseudo_depth: Tensor | None = None,
    depth_mask: Tensor | None = None,
    rendered_t: Tensor | None = None,
    target_t1: Tensor | None = None,
    flow_fwd: Tensor | None = None,
    xyz_t_minus_1: Tensor | None = None,
    xyz_t: Tensor | None = None,
    xyz_t_plus_1: Tensor | None = None,
    opacity: Tensor | None = None,
    scaling: Tensor | None = None,
    config: LossConfig | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute weighted combination of all loss terms.

    Only loss terms whose required inputs are not ``None`` are computed.
    Missing terms contribute 0 to the total and are omitted from the
    returned dict.

    Parameters
    ----------
    rendered, target : (3, H, W) or (B, 3, H, W)
        Rendered and ground-truth images (required).
    rendered_depth, pseudo_depth : optional depth maps.
    depth_mask : optional valid-pixel mask for depth.
    rendered_t, target_t1, flow_fwd : optional flow-warp inputs.
    xyz_t_minus_1, xyz_t, xyz_t_plus_1 : optional trajectory inputs.
    opacity : optional opacity values.
    scaling : optional scale values.
    config : :class:`LossConfig` with per-term weights.

    Returns
    -------
    (total, losses_dict)
        Scalar total loss and a dict mapping loss names to their
        (unweighted) scalar values.
    """
    if config is None:
        config = LossConfig()

    device = rendered.device
    dtype = rendered.dtype
    total = torch.tensor(0.0, device=device, dtype=dtype)
    losses: dict[str, Tensor] = {}

    # --- Photometric (always computed) ---
    l_photo = photometric_loss(
        rendered, target, lambda_l1=config.lambda_l1, lambda_ssim=config.lambda_ssim
    )
    losses["photometric"] = l_photo
    total = total + config.lambda_photo * l_photo

    # --- Depth ---
    if rendered_depth is not None and pseudo_depth is not None:
        l_depth = depth_loss(rendered_depth, pseudo_depth, mask=depth_mask)
        losses["depth"] = l_depth
        total = total + config.lambda_depth * l_depth

    # --- Temporal consistency ---
    if rendered_t is not None and target_t1 is not None and flow_fwd is not None:
        l_temporal = temporal_consistency_loss(rendered_t, target_t1, flow_fwd)
        losses["temporal_consistency"] = l_temporal
        total = total + config.lambda_temporal * l_temporal

    # --- Temporal smoothness ---
    if xyz_t_minus_1 is not None and xyz_t is not None and xyz_t_plus_1 is not None:
        l_smooth = temporal_smoothness_loss(xyz_t_minus_1, xyz_t, xyz_t_plus_1)
        losses["temporal_smoothness"] = l_smooth
        total = total + config.lambda_smooth * l_smooth

    # --- Opacity regularization ---
    if opacity is not None:
        l_opacity = opacity_regularization(opacity)
        losses["opacity_reg"] = l_opacity
        total = total + config.lambda_opacity_reg * l_opacity

    # --- Scale regularization ---
    if scaling is not None:
        l_scale = scale_regularization(scaling, max_scale=config.max_scale)
        losses["scale_reg"] = l_scale
        total = total + config.lambda_scale_reg * l_scale

    return total, losses
