#!/usr/bin/env python3
"""Forward prediction by coupling Gaussian splatting with PINN advection.

After training, the :class:`DynamicGaussianModel` can render observed
timesteps, and the :class:`FluidPINN` has learned a velocity field
satisfying Navier-Stokes.  This module couples them: canonical Gaussian
positions are advected through the PINN velocity field, then rendered
from arbitrary cameras — including at *extrapolated* future times that
were never observed during training.

Key components:

- :class:`ForwardPredictor` — render single frames, sequences, or MP4
  videos using physics-based Gaussian advection.
- :func:`extrapolate_density` — sample the PINN density field on a 3D
  grid for volumetric visualisation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

from integration.advection import advect_trajectory

# ---------------------------------------------------------------------------
# 1. ForwardPredictor
# ---------------------------------------------------------------------------


class ForwardPredictor:
    """Renders physics-predicted frames by advecting Gaussians with the PINN.

    Parameters
    ----------
    gaussian_model : DynamicGaussianModel
        Trained Gaussian splatting model (provides canonical Gaussians).
    pinn_model : FluidPINN
        Trained PINN providing the velocity field u(x, t).
    renderer : GaussianRenderer
        Differentiable renderer.
    t_canonical : float
        Time corresponding to the canonical Gaussian frame (default 0.0).
    num_advection_steps : int
        Number of integration sub-steps per advection (default 20).
    advection_method : str
        Integration method — ``"rk4"`` (default) or ``"euler"``.
    """

    def __init__(
        self,
        gaussian_model: nn.Module,
        pinn_model: nn.Module,
        renderer: nn.Module,
        t_canonical: float = 0.0,
        num_advection_steps: int = 20,
        advection_method: str = "rk4",
    ) -> None:
        self.gaussian_model = gaussian_model
        self.pinn = pinn_model
        self.renderer = renderer
        self.t_canonical = t_canonical
        self.num_advection_steps = num_advection_steps
        self.advection_method = advection_method

    @torch.no_grad()
    def _advect_gaussians(self, time: float) -> Dict[str, Tensor]:
        """Advect canonical Gaussians to the given time.

        Returns a gaussians_dict compatible with the renderer.
        """
        canonical = self.gaussian_model.get_canonical()
        xyz_canonical = canonical["xyz"]  # (N, 3)

        if abs(time - self.t_canonical) < 1e-8:
            return canonical

        # Advect positions from canonical time to target time
        trajectory = advect_trajectory(
            self.pinn,
            xyz_canonical,
            t_start=self.t_canonical,
            t_end=time,
            num_steps=self.num_advection_steps,
            method=self.advection_method,
        )
        advected_xyz = trajectory[-1]  # (N, 3)

        # Build a new gaussians_dict with advected positions but
        # canonical rotation, scaling, opacity, features
        return {
            "xyz": advected_xyz,
            "rotation": canonical["rotation"],
            "scaling": canonical["scaling"],
            "opacity": canonical["opacity"],
            "features": canonical["features"],
            "covariance": canonical["covariance"],
        }

    @torch.no_grad()
    def predict_frame(
        self,
        camera: Dict[str, Any],
        time: float,
    ) -> Dict[str, Tensor]:
        """Render a single predicted frame.

        Parameters
        ----------
        camera : dict
            ``"K"`` (3,3), ``"w2c"`` (4,4), and optionally
            ``"image_height"`` / ``"image_width"``.
        time : float
            Target time (can be beyond the training range for extrapolation).

        Returns
        -------
        dict
            ``"image"`` (3, H, W) in [0, 1],
            ``"depth"`` (1, H, W),
            ``"alpha"`` (1, H, W).
        """
        gaussians = self._advect_gaussians(time)
        rendered = self.renderer.render(gaussians, camera)

        return {
            "image": rendered["render"],
            "depth": rendered["depth"],
            "alpha": rendered["alpha"],
        }

    @torch.no_grad()
    def predict_sequence(
        self,
        camera: Dict[str, Any],
        t_start: float,
        t_end: float,
        num_frames: int,
    ) -> List[Dict[str, Tensor]]:
        """Render a sequence of predicted frames at evenly spaced times.

        Parameters
        ----------
        camera : dict
            Camera specification (fixed for all frames).
        t_start, t_end : float
            Time range.
        num_frames : int
            Number of frames to render.

        Returns
        -------
        list of dict
            Each dict has ``"image"``, ``"depth"``, ``"alpha"``, ``"time"``.
        """
        times = torch.linspace(t_start, t_end, num_frames).tolist()
        frames = []
        for t in times:
            frame = self.predict_frame(camera, t)
            frame["time"] = t
            frames.append(frame)
        return frames

    @torch.no_grad()
    def predict_video(
        self,
        camera: Dict[str, Any],
        t_start: float,
        t_end: float,
        fps: int,
        output_path: str | Path,
    ) -> Path:
        """Render a predicted video and save as MP4.

        Parameters
        ----------
        camera : dict
            Camera specification (fixed for all frames).
        t_start, t_end : float
            Time range.
        fps : int
            Frames per second.
        output_path : str or Path
            Output MP4 file path.

        Returns
        -------
        Path to the saved video file.
        """
        duration = abs(t_end - t_start)
        num_frames = max(2, int(duration * fps))

        frames = self.predict_sequence(camera, t_start, t_end, num_frames)

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect RGB images as numpy arrays
        images = []
        for frame in frames:
            img = frame["image"]  # (3, H, W)
            img_np = (img.permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            images.append(img_np)

        _write_mp4(images, fps, output_path)
        return output_path


# ---------------------------------------------------------------------------
# 2. Density volume extraction
# ---------------------------------------------------------------------------


@torch.no_grad()
def extrapolate_density(
    pinn: nn.Module,
    time: float,
    grid_resolution: int = 128,
    domain: Optional[Tuple[Tuple[float, float], ...]] = None,
    device: torch.device = torch.device("cpu"),
) -> Tensor:
    """Sample the PINN density field on a 3D grid.

    Useful for volumetric visualisation (e.g. with Plotly iso-surfaces or
    volume rendering).

    Parameters
    ----------
    pinn : FluidPINN
    time : float
        Time at which to sample the density.
    grid_resolution : int
        Number of samples per axis (output shape: ``(R, R, R)``).
    domain : tuple of 3 (lo, hi) pairs, optional
        Spatial domain.  Defaults to ``[(-1,1), (-1,1), (-1,1)]``.
    device : torch.device

    Returns
    -------
    (R, R, R) density volume.
    """
    if domain is None:
        domain = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))

    R = grid_resolution
    xs = torch.linspace(domain[0][0], domain[0][1], R, device=device)
    ys = torch.linspace(domain[1][0], domain[1][1], R, device=device)
    zs = torch.linspace(domain[2][0], domain[2][1], R, device=device)
    xx, yy, zz = torch.meshgrid(xs, ys, zs, indexing="ij")
    xyz = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=-1)  # (R^3, 3)
    t_tensor = torch.full((xyz.shape[0], 1), time, device=device)

    # Process in chunks to manage memory
    chunk_size = 8192
    densities = []
    for i in range(0, xyz.shape[0], chunk_size):
        out = pinn(xyz[i : i + chunk_size], t_tensor[i : i + chunk_size])
        densities.append(out["density"].squeeze(-1))

    density = torch.cat(densities)  # (R^3,)
    return density.reshape(R, R, R)


# ---------------------------------------------------------------------------
# 3. Video writing utility
# ---------------------------------------------------------------------------


def _write_mp4(images: List[np.ndarray], fps: int, path: Path) -> None:
    """Write a list of RGB uint8 images to MP4 using available backend."""
    try:
        import imageio.v3 as iio

        iio.imwrite(str(path), np.stack(images), fps=fps, codec="h264")
        return
    except (ImportError, Exception):
        pass

    try:
        import imageio

        writer = imageio.get_writer(str(path), fps=fps, codec="libx264")
        for img in images:
            writer.append_data(img)
        writer.close()
        return
    except (ImportError, Exception):
        pass

    try:
        import cv2

        H, W = images[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
        for img in images:
            writer.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        writer.release()
        return
    except (ImportError, Exception):
        pass

    # Fallback: save individual frames as PNG
    frame_dir = path.parent / (path.stem + "_frames")
    frame_dir.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image

        for i, img in enumerate(images):
            Image.fromarray(img).save(frame_dir / f"frame_{i:05d}.png")
    except ImportError:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for i, img in enumerate(images):
            plt.imsave(str(frame_dir / f"frame_{i:05d}.png"), img)
    print(f"  No video encoder available. Saved {len(images)} frames to {frame_dir}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI for forward prediction / extrapolation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Forward-predict fluid frames using trained Gaussian + PINN models",
    )
    parser.add_argument(
        "--gaussian-ckpt", type=str, required=True, help="Path to trained Gaussian model checkpoint"
    )
    parser.add_argument(
        "--pinn-ckpt", type=str, required=True, help="Path to trained PINN checkpoint"
    )
    parser.add_argument("--t-start", type=float, default=0.0, help="Start time (default: 0.0)")
    parser.add_argument("--t-end", type=float, default=1.0, help="End time (default: 1.0)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument(
        "--output", type=str, default="outputs/renders/prediction.mp4", help="Output video path"
    )
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu")
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[512, 512],
        metavar=("H", "W"),
        help="Render resolution (default: 512 512)",
    )
    args = parser.parse_args()
    print(
        f"forward_predict CLI: would render {args.t_start}->{args.t_end} "
        f"at {args.fps}fps to {args.output}"
    )
    print("  (Full implementation requires trained model checkpoints)")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # If CLI args provided, run the CLI
    if len(sys.argv) > 1:
        main()
        sys.exit(0)

    # Otherwise run self-test
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from gaussian_splatting.model import DynamicGaussianModel
    from gaussian_splatting.renderer import GaussianRenderer
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
    print("  forward_predict.py — self-test")
    print("=" * 60)

    device = torch.device("cpu")

    # -- Create mock models --
    pinn = FluidPINN(PINNConfig(activation="siren", hidden_dim=32, num_layers=4))
    pinn.eval()

    N_gauss = 50
    gauss_model = DynamicGaussianModel(sh_degree=0, num_points=N_gauss)
    torch.manual_seed(42)
    gauss_model.gaussian_model.initialize_from_point_cloud(
        torch.randn(N_gauss, 3) * 0.5,
        torch.rand(N_gauss, 3),
    )
    gauss_model.eval()

    H, W = 64, 64
    renderer = GaussianRenderer(H, W, use_cuda_rasterizer="pytorch")

    K = torch.tensor(
        [
            [50.0, 0.0, W / 2.0],
            [0.0, 50.0, H / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )
    w2c = torch.eye(4)
    w2c[2, 3] = 3.0
    camera = {"K": K, "w2c": w2c, "image_height": H, "image_width": W}

    # -- ForwardPredictor --
    print("\n--- ForwardPredictor ---")
    predictor = ForwardPredictor(
        gauss_model,
        pinn,
        renderer,
        t_canonical=0.0,
        num_advection_steps=5,
    )

    # Single frame
    frame = predictor.predict_frame(camera, time=0.0)
    check(frame["image"].shape == (3, H, W), f"Frame image shape (3, {H}, {W})")
    check(frame["depth"].shape == (1, H, W), f"Frame depth shape (1, {H}, {W})")
    check(frame["alpha"].shape == (1, H, W), f"Frame alpha shape (1, {H}, {W})")
    check(frame["image"].min() >= 0.0, "Image min >= 0")
    check(frame["image"].max() <= 1.0, "Image max <= 1")
    check(torch.isfinite(frame["image"]).all(), "Image finite")

    # Frame at different time (advected)
    frame_05 = predictor.predict_frame(camera, time=0.5)
    check(frame_05["image"].shape == (3, H, W), "Advected frame shape")
    check(torch.isfinite(frame_05["image"]).all(), "Advected frame finite")

    # Sequence
    print("\n--- Sequence ---")
    seq = predictor.predict_sequence(camera, t_start=0.0, t_end=1.0, num_frames=5)
    check(len(seq) == 5, "Sequence has 5 frames")
    check(all(f["image"].shape == (3, H, W) for f in seq), "All frames correct shape")
    check(all(torch.isfinite(f["image"]).all() for f in seq), "All frames finite")
    check(abs(seq[0]["time"] - 0.0) < 1e-6, "First frame time = 0.0")
    check(abs(seq[-1]["time"] - 1.0) < 1e-6, "Last frame time = 1.0")

    # -- extrapolate_density --
    print("\n--- extrapolate_density ---")
    R = 16
    vol = extrapolate_density(pinn, time=0.5, grid_resolution=R)
    check(vol.shape == (R, R, R), f"Density volume shape ({R}, {R}, {R})")
    check(torch.isfinite(vol).all(), "Density volume finite")
    check((vol > 0).all(), "Density > 0 everywhere (softplus)")
    check(vol.std() > 1e-6, f"Density has spatial variation (std={vol.std():.4e})")

    # Extrapolated time (beyond training range)
    vol_ext = extrapolate_density(pinn, time=2.0, grid_resolution=R)
    check(torch.isfinite(vol_ext).all(), "Extrapolated density finite")

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 60}")

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: forward_predict.py"
            " — ForwardPredictor renders frames/sequences, "
            "extrapolate_density produces valid 3D volumes"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)
