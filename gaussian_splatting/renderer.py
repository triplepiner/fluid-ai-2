#!/usr/bin/env python3
"""Differentiable Gaussian splatting renderer.

Backends
--------
- **CUDA** (``gsplat``): fast tile-based rasterisation for training on GPU.
- **PyTorch** (pure): portable fallback for CPU / MPS development and
  gradient-correctness verification.  Slower, but numerically identical
  compositing logic.

The renderer takes the output of :class:`DynamicGaussianModel.forward(t)` and a
camera specification, then produces a rendered RGB image, depth map, and alpha
mask — all differentiable with respect to the Gaussian parameters.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helper: detect gsplat availability
# ---------------------------------------------------------------------------
_GSPLAT_AVAILABLE = False
try:
    import gsplat  # noqa: F401

    _GSPLAT_AVAILABLE = True
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def project_points(
    xyz: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project 3D world-space points to 2D pixel coordinates.

    Parameters
    ----------
    xyz : (N, 3)
    K   : (3, 3) intrinsic matrix
    w2c : (4, 4) world-to-camera transform

    Returns
    -------
    xy  : (N, 2) pixel coordinates (x, y)
    depth : (N,) z-depth in camera space
    """
    R = w2c[:3, :3]  # (3, 3)
    t = w2c[:3, 3]  # (3,)

    xyz_cam = R @ xyz.T + t.unsqueeze(-1)  # (3, N)

    depth = xyz_cam[2].clone()  # (N,)

    # Perspective projection
    xy_h = K @ xyz_cam  # (3, N)
    xy = xy_h[:2] / (xy_h[2:3].clamp(min=1e-8))  # (2, N)

    return xy.T, depth  # (N, 2), (N,)


def compute_2d_covariance(
    cov3d: torch.Tensor,
    K: torch.Tensor,
    w2c: torch.Tensor,
    xyz: torch.Tensor,
) -> torch.Tensor:
    """Project 3D covariance matrices to 2D screen-space.

    Uses the Jacobian of the perspective projection and the camera rotation
    to propagate the 3D covariance to a 2x2 screen-space covariance:

        Sigma_2d = (J @ W @ Sigma_3d @ W^T @ J^T)[:2, :2]

    Parameters
    ----------
    cov3d : (N, 3, 3)
    K     : (3, 3)
    w2c   : (4, 4)
    xyz   : (N, 3) world-space positions

    Returns
    -------
    cov2d : (N, 2, 2)
    """
    R = w2c[:3, :3]  # (3, 3)
    t = w2c[:3, 3]  # (3,)

    # Transform to camera space
    xyz_cam = (R @ xyz.T + t.unsqueeze(-1)).T  # (N, 3)
    tx, ty, tz = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
    tz = tz.clamp(min=1e-8)

    fx = K[0, 0]
    fy = K[1, 1]

    # Jacobian of perspective projection: d(pixel) / d(cam_xyz)
    # J = [[fx/z, 0, -fx*x/z^2],
    #       [0, fy/z, -fy*y/z^2]]
    # We use a 3x3 version (third row = [0, 0, 0]) so matrix sizes align.
    zeros = torch.zeros_like(tz)
    tz_sq = tz * tz

    J = torch.stack(
        [
            fx / tz,
            zeros,
            -fx * tx / tz_sq,
            zeros,
            fy / tz,
            -fy * ty / tz_sq,
            zeros,
            zeros,
            zeros,
        ],
        dim=-1,
    ).reshape(
        -1, 3, 3
    )  # (N, 3, 3)

    # W = camera rotation (world -> camera)
    W = R.unsqueeze(0).expand(xyz.shape[0], -1, -1)  # (N, 3, 3)

    # T = J @ W
    T = J @ W  # (N, 3, 3)

    # Sigma_cam = T @ Sigma_3d @ T^T
    cov_cam = T @ cov3d @ T.transpose(-1, -2)  # (N, 3, 3)

    # Extract top-left 2x2 block
    cov2d = cov_cam[:, :2, :2]  # (N, 2, 2)

    # Add a small diagonal for numerical stability
    cov2d = cov2d + 0.3 * torch.eye(2, device=cov2d.device, dtype=cov2d.dtype).unsqueeze(0)

    return cov2d


def sh_to_rgb(
    sh_coeffs: torch.Tensor,
    viewdir: torch.Tensor,
) -> torch.Tensor:
    """Evaluate spherical harmonics to get RGB colour.

    Parameters
    ----------
    sh_coeffs : (N, C, 3) where C = (deg+1)^2
    viewdir   : (N, 3) unit view directions (camera -> point)

    Returns
    -------
    rgb : (N, 3) in [0, 1]
    """
    C = sh_coeffs.shape[1]
    C0 = 0.28209479177387814
    result = sh_coeffs[:, 0, :] * C0  # (N, 3)

    if C > 1 and viewdir is not None:
        x = viewdir[:, 0:1]  # (N, 1)
        y = viewdir[:, 1:2]
        z = viewdir[:, 2:3]
        C1 = 0.4886025119029199

        result = result + C1 * (
            -y * sh_coeffs[:, 1, :] + z * sh_coeffs[:, 2, :] - x * sh_coeffs[:, 3, :]
        )

        if C > 4:
            xx, yy, zz = x * x, y * y, z * z
            xy, yz, xz = x * y, y * z, x * z

            result = result + (
                1.0925484305920792 * xy * sh_coeffs[:, 4, :]
                + (-1.0925484305920792) * yz * sh_coeffs[:, 5, :]
                + 0.31539156525252005 * (2.0 * zz - xx - yy) * sh_coeffs[:, 6, :]
                + (-1.0925484305920792) * xz * sh_coeffs[:, 7, :]
                + 0.5462742152960396 * (xx - yy) * sh_coeffs[:, 8, :]
            )

            if C > 9:
                result = result + (
                    (-0.5900435899266435) * y * (3.0 * xx - yy) * sh_coeffs[:, 9, :]
                    + 2.890611442640554 * xy * z * sh_coeffs[:, 10, :]
                    + (-0.4570457994644658) * y * (4.0 * zz - xx - yy) * sh_coeffs[:, 11, :]
                    + 0.3731763325901154
                    * z
                    * (2.0 * zz - 3.0 * xx - 3.0 * yy)
                    * sh_coeffs[:, 12, :]
                    + (-0.4570457994644658) * x * (4.0 * zz - xx - yy) * sh_coeffs[:, 13, :]
                    + 1.445305721320277 * z * (xx - yy) * sh_coeffs[:, 14, :]
                    + (-0.5900435899266435) * x * (xx - 3.0 * yy) * sh_coeffs[:, 15, :]
                )

    # Shift from SH space to [0, 1]
    result = result + 0.5
    return result.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# GaussianRenderer
# ---------------------------------------------------------------------------


class GaussianRenderer(nn.Module):
    """Differentiable Gaussian splatting renderer.

    Parameters
    ----------
    image_height, image_width : int
        Default output resolution (can be overridden per-call via viewpoint_camera).
    background_color : tuple[float, float, float]
        RGB background in [0, 1].
    use_cuda_rasterizer : str
        ``"auto"`` (default), ``"cuda"``, or ``"pytorch"``.
    """

    def __init__(
        self,
        image_height: int = 512,
        image_width: int = 512,
        background_color: tuple[float, float, float] = (0.0, 0.0, 0.0),
        use_cuda_rasterizer: str = "auto",
    ) -> None:
        super().__init__()
        self.image_height = image_height
        self.image_width = image_width
        self.register_buffer(
            "background",
            torch.tensor(background_color, dtype=torch.float32),
        )

        if use_cuda_rasterizer == "auto":
            self._backend = (
                "cuda" if (_GSPLAT_AVAILABLE and torch.cuda.is_available()) else "pytorch"
            )
        else:
            self._backend = use_cuda_rasterizer

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def render(
        self,
        gaussians_dict: dict[str, torch.Tensor],
        viewpoint_camera: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Render an image from the given Gaussians and camera.

        Parameters
        ----------
        gaussians_dict
            Output of ``DynamicGaussianModel.forward(t)``:
            ``"xyz"`` (N,3), ``"rotation"`` (N,4), ``"scaling"`` (N,3),
            ``"opacity"`` (N,1), ``"features"`` (N,C,3),
            ``"covariance"`` (N,3,3).
        viewpoint_camera
            ``"K"`` (3,3), ``"w2c"`` (4,4), and optionally
            ``"image_height"`` / ``"image_width"``.

        Returns
        -------
        dict
            ``"render"`` (3,H,W), ``"depth"`` (1,H,W), ``"alpha"`` (1,H,W),
            ``"visibility_filter"`` (N,), ``"radii"`` (N,).
        """
        if self._backend == "cuda":
            return self._render_cuda(gaussians_dict, viewpoint_camera)
        return self._render_pytorch(gaussians_dict, viewpoint_camera)

    # ------------------------------------------------------------------
    # Pure PyTorch fallback
    # ------------------------------------------------------------------

    def _render_pytorch(
        self,
        gaussians_dict: dict[str, torch.Tensor],
        viewpoint_camera: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Pure-PyTorch differentiable Gaussian splatting.

        Simplified for development / gradient verification:
        - Processes up to ``max_gaussians_per_tile`` nearest Gaussians per
          pixel tile.
        - Not optimised for speed; use the CUDA backend for training.
        """
        device = gaussians_dict["xyz"].device
        dtype = gaussians_dict["xyz"].dtype

        H = viewpoint_camera.get("image_height", self.image_height)
        W = viewpoint_camera.get("image_width", self.image_width)
        K = viewpoint_camera["K"].to(device=device, dtype=dtype)
        w2c = viewpoint_camera["w2c"].to(device=device, dtype=dtype)
        bg = self.background.to(device=device, dtype=dtype)

        xyz = gaussians_dict["xyz"]  # (N, 3)
        opacity = gaussians_dict["opacity"]  # (N, 1)
        features = gaussians_dict["features"]  # (N, C, 3)
        cov3d = gaussians_dict["covariance"]  # (N, 3, 3)
        N = xyz.shape[0]

        # --- Step 1: Project means to 2D ---
        xy, depths = project_points(xyz, K, w2c)  # (N, 2), (N,)

        # --- Step 2: Compute view directions for SH ---
        c2w = torch.inverse(w2c)
        cam_pos = c2w[:3, 3]  # (3,)
        viewdirs = xyz - cam_pos.unsqueeze(0)  # (N, 3)
        viewdirs = F.normalize(viewdirs, dim=-1)

        # Evaluate SH -> RGB
        rgb = sh_to_rgb(features, viewdirs)  # (N, 3)

        # --- Step 3: Compute 2D covariance ---
        cov2d = compute_2d_covariance(cov3d, K, w2c, xyz)  # (N, 2, 2)

        # --- Step 4: Filter to visible Gaussians (in front of camera, on screen) ---
        margin = 256  # pixels of margin
        visible = (
            (depths > 0.01)
            & (xy[:, 0] > -margin)
            & (xy[:, 0] < W + margin)
            & (xy[:, 1] > -margin)
            & (xy[:, 1] < H + margin)
        )
        vis_idx = visible.nonzero(as_tuple=True)[0]

        # Build full-size output tensors
        visibility_filter = visible  # (N,)

        if vis_idx.numel() == 0:
            render_img = bg.reshape(3, 1, 1).expand(3, H, W)
            depth_img = torch.zeros(1, H, W, device=device, dtype=dtype)
            alpha_img = torch.zeros(1, H, W, device=device, dtype=dtype)
            radii = torch.zeros(N, device=device, dtype=dtype)
            return {
                "render": render_img,
                "depth": depth_img,
                "alpha": alpha_img,
                "visibility_filter": visibility_filter,
                "radii": radii,
            }

        # Gather visible Gaussian properties
        v_xy = xy[vis_idx]  # (V, 2)
        v_depths = depths[vis_idx]  # (V,)
        v_opacity = opacity[vis_idx]  # (V, 1)
        v_rgb = rgb[vis_idx]  # (V, 3)
        v_cov2d = cov2d[vis_idx]  # (V, 2, 2)
        V = vis_idx.numel()

        # Approximate screen-space radii (3-sigma from larger eigenvalue)
        # eigenvalues of 2x2 symmetric matrix
        a = v_cov2d[:, 0, 0]
        b = v_cov2d[:, 0, 1]
        d = v_cov2d[:, 1, 1]
        det = a * d - b * b
        trace = a + d
        # eigenvalues via quadratic formula
        disc = (trace * trace - 4.0 * det).clamp(min=0.0)
        sqrt_disc = torch.sqrt(disc)
        lam_max = 0.5 * (trace + sqrt_disc)
        v_radii_float = 3.0 * torch.sqrt(lam_max.clamp(min=1e-8))  # (V,) in pixels

        # Build full radii
        radii = torch.zeros(N, device=device, dtype=dtype)
        radii[vis_idx] = v_radii_float

        # --- Step 5: Sort by depth (front to back) ---
        sorted_indices = torch.argsort(v_depths)
        v_xy = v_xy[sorted_indices]
        v_depths = v_depths[sorted_indices]
        v_opacity = v_opacity[sorted_indices]
        v_rgb = v_rgb[sorted_indices]
        v_cov2d = v_cov2d[sorted_indices]
        v_radii_float = v_radii_float[sorted_indices]

        # --- Step 6: Alpha compositing (out-of-place for autograd) ---
        # For the pure PyTorch renderer we process Gaussians in depth order,
        # accumulating contributions via out-of-place ops so gradients flow.

        render_img = torch.zeros(3, H, W, device=device, dtype=dtype)
        depth_img = torch.zeros(1, H, W, device=device, dtype=dtype)
        T_acc = torch.ones(1, H, W, device=device, dtype=dtype)  # transmittance

        # Precompute inverse of 2x2 covariances (out-of-place)
        cov_a = v_cov2d[:, 0, 0]
        cov_b = v_cov2d[:, 0, 1]
        cov_d = v_cov2d[:, 1, 1]
        det = (cov_a * cov_d - cov_b * cov_b).clamp(min=1e-10)
        inv_00 = cov_d / det
        inv_01 = -cov_b / det
        inv_11 = cov_a / det

        # Process Gaussians (sorted by depth, limited for tractability)
        MAX_GAUSSIANS = min(V, 8192)

        for i in range(MAX_GAUSSIANS):
            mean_x = v_xy[i, 0]
            mean_y = v_xy[i, 1]
            radius = v_radii_float[i].clamp(min=1.0)

            # Bounding box (integer pixel coordinates)
            x_min = int(max(0, (mean_x - radius).detach().item()))
            x_max = int(min(W, (mean_x + radius).detach().item() + 1))
            y_min = int(max(0, (mean_y - radius).detach().item()))
            y_max = int(min(H, (mean_y + radius).detach().item() + 1))

            if x_min >= x_max or y_min >= y_max:
                continue

            bh = y_max - y_min
            bw = x_max - x_min

            # Pixel grid for this bounding box
            py = torch.arange(y_min, y_max, device=device, dtype=dtype)
            px = torch.arange(x_min, x_max, device=device, dtype=dtype)
            grid_y, grid_x = torch.meshgrid(py, px, indexing="ij")  # (bh, bw)

            # Offset from Gaussian mean
            dx = grid_x - mean_x  # (bh, bw)
            dy = grid_y - mean_y  # (bh, bw)

            # Mahalanobis distance: d^T @ inv_cov @ d
            maha = inv_00[i] * dx * dx + 2.0 * inv_01[i] * dx * dy + inv_11[i] * dy * dy  # (bh, bw)

            # Gaussian weight
            gauss_weight = torch.exp(-0.5 * maha)  # (bh, bw)

            # Alpha for this Gaussian
            alpha_i = (v_opacity[i, 0] * gauss_weight).clamp(max=0.99)  # (bh, bw)

            # Read current transmittance (creates a view — fine for reading)
            T_patch = T_acc[0, y_min:y_max, x_min:x_max]  # (bh, bw)

            # Contribution weight
            weight = alpha_i * T_patch  # (bh, bw)

            # Accumulate colour (out-of-place via F.pad)
            colour_i = v_rgb[i]  # (3,)
            color_patch = weight.unsqueeze(0) * colour_i.reshape(3, 1, 1)  # (3, bh, bw)
            padded_color = F.pad(
                color_patch,
                (x_min, W - x_max, y_min, H - y_max),
            )
            render_img = render_img + padded_color

            # Accumulate depth (out-of-place via F.pad)
            depth_patch = (weight * v_depths[i]).unsqueeze(0)  # (1, bh, bw)
            padded_depth = F.pad(
                depth_patch,
                (x_min, W - x_max, y_min, H - y_max),
            )
            depth_img = depth_img + padded_depth

            # Update transmittance (out-of-place)
            # decay_patch = (1 - alpha_i), padded with 0, then add 1 everywhere
            # so the padded region has multiplier 1.0
            decay_offset = (-alpha_i).unsqueeze(0)  # (1, bh, bw)
            padded_decay = F.pad(
                decay_offset,
                (x_min, W - x_max, y_min, H - y_max),
            )
            T_acc = T_acc * (1.0 + padded_decay)

        # Alpha = 1 - final transmittance
        alpha_img = 1.0 - T_acc  # (1, H, W)

        # Composite background
        render_img = render_img + T_acc * bg.reshape(3, 1, 1)

        return {
            "render": render_img,
            "depth": depth_img,
            "alpha": alpha_img,
            "visibility_filter": visibility_filter,
            "radii": radii,
        }

    # ------------------------------------------------------------------
    # CUDA backend (gsplat)
    # ------------------------------------------------------------------

    def _render_cuda(
        self,
        gaussians_dict: dict[str, torch.Tensor],
        viewpoint_camera: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        """Render using the gsplat CUDA rasteriser.

        Requires ``pip install gsplat`` and a CUDA-capable GPU.
        """
        if not _GSPLAT_AVAILABLE:
            raise RuntimeError("gsplat is not installed.  Install with: pip install gsplat")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.  Use backend='pytorch' for CPU/MPS.")

        from gsplat import rasterization

        device = gaussians_dict["xyz"].device
        dtype = gaussians_dict["xyz"].dtype

        H = viewpoint_camera.get("image_height", self.image_height)
        W = viewpoint_camera.get("image_width", self.image_width)
        K = viewpoint_camera["K"].to(device=device, dtype=dtype)
        w2c = viewpoint_camera["w2c"].to(device=device, dtype=dtype)

        xyz = gaussians_dict["xyz"]  # (N, 3)
        opacity = gaussians_dict["opacity"]  # (N, 1)
        features = gaussians_dict["features"]  # (N, C, 3)
        scaling = gaussians_dict["scaling"]  # (N, 3)
        rotation = gaussians_dict["rotation"]  # (N, 4)
        N = xyz.shape[0]

        # Compute view directions for SH evaluation
        c2w = torch.inverse(w2c)
        cam_pos = c2w[:3, 3]
        viewdirs = F.normalize(xyz - cam_pos.unsqueeze(0), dim=-1)
        rgb = sh_to_rgb(features, viewdirs)  # (N, 3)

        # gsplat.rasterization expects:
        #   means: (N, 3), quats: (N, 4), scales: (N, 3),
        #   opacities: (N,), colors: (N, 3),
        #   viewmats: (C, 4, 4), Ks: (C, 3, 3),
        #   width: int, height: int
        renders, alphas, meta = rasterization(
            means=xyz,
            quats=rotation,
            scales=scaling,
            opacities=opacity.squeeze(-1),
            colors=rgb,
            viewmats=w2c.unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            render_mode="RGB+ED",
            backgrounds=self.background.to(device=device, dtype=dtype).unsqueeze(0),
        )
        # renders: (C, H, W, 4) — RGB + expected depth
        # alphas:  (C, H, W, 1)

        render_img = renders[0, :, :, :3].permute(2, 0, 1)  # (3, H, W)
        depth_img = renders[0, :, :, 3:4].permute(2, 0, 1)  # (1, H, W)
        alpha_img = alphas[0].permute(2, 0, 1)  # (1, H, W)

        # Visibility and radii from meta
        radii = meta.get("radii", torch.zeros(N, device=device, dtype=dtype))
        if isinstance(radii, torch.Tensor) and radii.dim() == 2:
            radii = radii[0]  # take first camera
        visibility_filter = radii > 0

        return {
            "render": render_img,
            "depth": depth_img,
            "alpha": alpha_img,
            "visibility_filter": visibility_filter,
            "radii": radii.float(),
        }


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    from gaussian_splatting.model import (
        DynamicGaussianModel,
        GaussianModel,
        build_rotation_matrix,
    )

    print("=" * 60)
    print("  Renderer self-test")
    print("=" * 60)

    device = torch.device("cpu")
    N = 50
    H, W = 64, 64

    # --- Create random Gaussians via the model ---
    model = DynamicGaussianModel(sh_degree=0, num_points=N)
    torch.manual_seed(42)
    xyz = torch.randn(N, 3) * 0.5
    rgb = torch.rand(N, 3)
    model.gaussian_model.initialize_from_point_cloud(xyz, rgb)
    model = model.to(device)

    gaussians = model.forward(time=0.0)

    # --- Camera looking down -Z, centered at origin ---
    K = torch.tensor(
        [
            [50.0, 0.0, W / 2.0],
            [0.0, 50.0, H / 2.0],
            [0.0, 0.0, 1.0],
        ],
        device=device,
    )

    # Camera at z=+3, looking toward origin
    w2c = torch.eye(4, device=device)
    w2c[2, 3] = 3.0  # translate camera back

    camera = {"K": K, "w2c": w2c, "image_height": H, "image_width": W}

    # --- Render ---
    renderer = GaussianRenderer(
        image_height=H,
        image_width=W,
        background_color=(0.0, 0.0, 0.0),
        use_cuda_rasterizer="pytorch",
    )
    renderer = renderer.to(device)

    out = renderer.render(gaussians, camera)

    passed = 0
    total = 0

    def check(cond, label):
        global passed, total
        total += 1
        s = "PASS" if cond else "FAIL"
        print(f"  [{s}] {label}")
        if cond:
            passed += 1

    check(out["render"].shape == (3, H, W), f"render shape = (3, {H}, {W})")
    check(out["depth"].shape == (1, H, W), f"depth shape = (1, {H}, {W})")
    check(out["alpha"].shape == (1, H, W), f"alpha shape = (1, {H}, {W})")
    check(out["visibility_filter"].shape == (N,), f"visibility_filter shape = ({N},)")
    check(out["radii"].shape == (N,), f"radii shape = ({N},)")

    check(out["render"].min() >= 0.0, "render min >= 0")
    check(out["render"].max() <= 1.0, "render max <= 1")
    check(out["alpha"].min() >= 0.0, "alpha min >= 0")
    check(out["alpha"].max() <= 1.0, "alpha max <= 1")
    check(out["depth"].min() >= 0.0, "depth min >= 0")

    # Some pixels should be non-zero (Gaussians visible)
    check(out["alpha"].max() > 0.01, "some pixels have alpha > 0")
    check(out["render"].max() > 0.01, "some pixels have colour")

    # Gradient flow
    loss = out["render"].sum() + out["depth"].sum()
    loss.backward()
    check(model.gaussian_model._xyz.grad is not None, "gradient on _xyz")
    check(model.gaussian_model._opacity.grad is not None, "gradient on _opacity")
    check(model.gaussian_model._features_dc.grad is not None, "gradient on _features_dc")

    deform_has_grad = any(p.grad is not None for p in model.deformation_network.parameters())
    check(deform_has_grad, "gradient flows to deformation network")

    print(f"\n  {passed}/{total} checks passed")
    if passed == total:
        print("  SELF-TEST PASSED")
    else:
        print("  SELF-TEST FAILED")
        sys.exit(1)
