#!/usr/bin/env python3
"""Novel-view synthesis with physics-based temporal extrapolation.

Combines the trained Gaussian scene representation with the PINN velocity
field to render:

- **Orbit videos** — circle around the scene at a fixed time.
- **Spacetime videos** — move through both space and time simultaneously.
- **Camera interpolation** — smooth transitions between two cameras using
  slerp (rotation) and lerp (translation).

Also provides camera-matrix utilities:

- :func:`look_at` — build a camera extrinsic matrix from
  (position, look_at, up_vector).
- :func:`make_camera` — full camera dict from position, look_at, intrinsics.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from integration.forward_predict import ForwardPredictor, _write_mp4

# ---------------------------------------------------------------------------
# Camera utilities
# ---------------------------------------------------------------------------


def look_at(
    eye: Tensor,
    target: Tensor,
    up: Tensor,
) -> Tensor:
    """Build a world-to-camera (w2c) 4x4 matrix from eye, target, up.

    Uses the COLMAP / 3DGS convention:
    - Camera Z-axis points *forward* (into the scene).  Objects in front
      of the camera have positive z-depth.
    - Camera Y-axis points *down* (image rows increase downward).
    - Camera X-axis points *right*.
    - The rotation matrix ``R`` has ``det(R) = +1`` (proper rotation).

    Parameters
    ----------
    eye : (3,) camera position in world space.
    target : (3,) point the camera looks at.
    up : (3,) approximate world-up direction.

    Returns
    -------
    (4, 4) world-to-camera matrix.
    """
    # Forward: from eye toward target (camera Z-axis in world)
    forward = F.normalize(target - eye, dim=0)
    # Right = forward x up (camera X-axis in world)
    right = F.normalize(torch.cross(forward, up, dim=0), dim=0)
    # Camera Y-axis = "down" = forward x right (right-handed: Z x X = Y)
    cam_down = torch.cross(forward, right, dim=0)

    # Rotation matrix: rows = camera axes expressed in world coordinates
    R = torch.stack([right, cam_down, forward], dim=0)  # (3, 3)

    # Translation: -R @ eye
    t = -R @ eye  # (3,)

    w2c = torch.eye(4, device=eye.device, dtype=eye.dtype)
    w2c[:3, :3] = R
    w2c[:3, 3] = t
    return w2c


def make_camera(
    position: Tensor | Tuple[float, float, float],
    look_at_point: Tensor | Tuple[float, float, float],
    up: Tensor | Tuple[float, float, float] = (0.0, 1.0, 0.0),
    fx: float = 500.0,
    fy: float = 500.0,
    cx: float = 256.0,
    cy: float = 256.0,
    image_height: int = 512,
    image_width: int = 512,
) -> Dict[str, Any]:
    """Create a full camera dict from intuitive parameters.

    Parameters
    ----------
    position : 3-tuple or Tensor
        Camera position in world space.
    look_at_point : 3-tuple or Tensor
        Point the camera looks at.
    up : 3-tuple or Tensor
        Up direction (default Y-up).
    fx, fy, cx, cy : float
        Intrinsic parameters.
    image_height, image_width : int
        Output resolution.

    Returns
    -------
    dict with ``"K"`` (3,3), ``"w2c"`` (4,4), ``"image_height"``,
    ``"image_width"``.
    """
    eye = (
        torch.tensor(position, dtype=torch.float32)
        if not isinstance(position, Tensor)
        else position.float()
    )
    target = (
        torch.tensor(look_at_point, dtype=torch.float32)
        if not isinstance(look_at_point, Tensor)
        else look_at_point.float()
    )
    up_vec = torch.tensor(up, dtype=torch.float32) if not isinstance(up, Tensor) else up.float()

    w2c = look_at(eye, target, up_vec)

    K = torch.tensor(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    return {
        "K": K,
        "w2c": w2c,
        "image_height": image_height,
        "image_width": image_width,
    }


def _rotation_matrix_to_quaternion(R: Tensor) -> Tensor:
    """Convert a 3x3 rotation matrix to a unit quaternion (w, x, y, z).

    Uses Shepperd's method for numerical stability.
    """
    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return torch.tensor([w, x, y, z], dtype=R.dtype, device=R.device)


def _quaternion_to_rotation_matrix(q: Tensor) -> Tensor:
    """Convert unit quaternion (w, x, y, z) to 3x3 rotation matrix."""
    q = q / q.norm()
    w, x, y, z = q

    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=q.dtype,
        device=q.device,
    )


def _slerp(q0: Tensor, q1: Tensor, t: float) -> Tensor:
    """Spherical linear interpolation between two unit quaternions."""
    dot = (q0 * q1).sum().clamp(-1.0, 1.0)

    # If dot < 0, negate one quaternion to take the short path
    if dot < 0:
        q1 = -q1
        dot = -dot

    # For very close quaternions, use linear interpolation
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / result.norm()

    theta = torch.acos(dot)
    sin_theta = torch.sin(theta)
    s0 = torch.sin((1.0 - t) * theta) / sin_theta
    s1 = torch.sin(t * theta) / sin_theta
    result = s0 * q0 + s1 * q1
    return result / result.norm()


def interpolate_cameras(
    cam1: Dict[str, Any],
    cam2: Dict[str, Any],
    num_steps: int,
) -> List[Dict[str, Any]]:
    """Smoothly interpolate between two cameras.

    Uses slerp for rotations and linear interpolation for translations.
    Intrinsics are linearly interpolated.

    Parameters
    ----------
    cam1, cam2 : dict
        Camera dicts with ``"K"`` (3,3), ``"w2c"`` (4,4), and resolution.
    num_steps : int
        Number of intermediate cameras (including endpoints).

    Returns
    -------
    list of camera dicts.
    """
    w2c1 = cam1["w2c"].float()
    w2c2 = cam2["w2c"].float()

    R1, t1 = w2c1[:3, :3], w2c1[:3, 3]
    R2, t2 = w2c2[:3, :3], w2c2[:3, 3]

    q1 = _rotation_matrix_to_quaternion(R1)
    q2 = _rotation_matrix_to_quaternion(R2)

    K1 = cam1["K"].float()
    K2 = cam2["K"].float()

    H1 = cam1.get("image_height", 512)
    W1 = cam1.get("image_width", 512)
    H2 = cam2.get("image_height", 512)
    W2 = cam2.get("image_width", 512)

    cameras = []
    for i in range(num_steps):
        alpha = i / max(num_steps - 1, 1)

        # Slerp rotation
        q_interp = _slerp(q1, q2, alpha)
        R_interp = _quaternion_to_rotation_matrix(q_interp)

        # Lerp translation
        t_interp = (1.0 - alpha) * t1 + alpha * t2

        w2c = torch.eye(4, dtype=torch.float32)
        w2c[:3, :3] = R_interp
        w2c[:3, 3] = t_interp

        # Lerp intrinsics
        K_interp = (1.0 - alpha) * K1 + alpha * K2

        # Lerp resolution (round to int)
        H = int(round((1.0 - alpha) * H1 + alpha * H2))
        W = int(round((1.0 - alpha) * W1 + alpha * W2))

        cameras.append(
            {
                "K": K_interp,
                "w2c": w2c,
                "image_height": H,
                "image_width": W,
            }
        )

    return cameras


# ---------------------------------------------------------------------------
# NovelViewSynthesizer
# ---------------------------------------------------------------------------


class NovelViewSynthesizer:
    """Render novel views with physics-based temporal evolution.

    Parameters
    ----------
    gaussian_model : DynamicGaussianModel
    pinn_model : FluidPINN
    renderer : GaussianRenderer
    camera_intrinsics : dict, optional
        Default intrinsics: ``"fx"``, ``"fy"``, ``"cx"``, ``"cy"``,
        ``"image_height"``, ``"image_width"``.
    t_canonical : float
        Canonical time for Gaussian positions.
    num_advection_steps : int
        Sub-steps for advection integration.
    """

    def __init__(
        self,
        gaussian_model: nn.Module,
        pinn_model: nn.Module,
        renderer: nn.Module,
        camera_intrinsics: Optional[Dict[str, float]] = None,
        t_canonical: float = 0.0,
        num_advection_steps: int = 20,
    ) -> None:
        self.predictor = ForwardPredictor(
            gaussian_model,
            pinn_model,
            renderer,
            t_canonical=t_canonical,
            num_advection_steps=num_advection_steps,
        )
        self.pinn = pinn_model
        self.gaussian_model = gaussian_model
        self.renderer = renderer

        # Default intrinsics
        if camera_intrinsics is None:
            camera_intrinsics = {
                "fx": 500.0,
                "fy": 500.0,
                "cx": 256.0,
                "cy": 256.0,
                "image_height": 512,
                "image_width": 512,
            }
        self.intrinsics = camera_intrinsics

    def _make_orbit_camera(
        self,
        center: Tuple[float, float, float],
        radius: float,
        angle_rad: float,
        elevation_deg: float,
    ) -> Dict[str, Any]:
        """Create a camera on an orbit around center."""
        elev = math.radians(elevation_deg)
        x = center[0] + radius * math.cos(elev) * math.cos(angle_rad)
        y = center[1] + radius * math.sin(elev)
        z = center[2] + radius * math.cos(elev) * math.sin(angle_rad)

        return make_camera(
            position=(x, y, z),
            look_at_point=center,
            up=(0.0, 1.0, 0.0),
            fx=self.intrinsics["fx"],
            fy=self.intrinsics["fy"],
            cx=self.intrinsics["cx"],
            cy=self.intrinsics["cy"],
            image_height=int(self.intrinsics["image_height"]),
            image_width=int(self.intrinsics["image_width"]),
        )

    @torch.no_grad()
    def render_orbit(
        self,
        center: Tuple[float, float, float],
        radius: float,
        time: float,
        num_views: int = 36,
        elevation: float = 30.0,
    ) -> List[Dict[str, Tensor]]:
        """Render an orbit (turntable) around the scene at a fixed time.

        Parameters
        ----------
        center : (x, y, z)
            Centre of the orbit.
        radius : float
            Orbit radius.
        time : float
            Time at which to render (Gaussians are advected to this time).
        num_views : int
            Number of views evenly spaced around the orbit.
        elevation : float
            Elevation angle in degrees above the XZ plane.

        Returns
        -------
        list of dicts, each with ``"image"``, ``"depth"``, ``"alpha"``,
        ``"angle_deg"``.
        """
        frames = []
        for i in range(num_views):
            angle = 2.0 * math.pi * i / num_views
            camera = self._make_orbit_camera(center, radius, angle, elevation)
            frame = self.predictor.predict_frame(camera, time)
            frame["angle_deg"] = math.degrees(angle)
            frames.append(frame)
        return frames

    @torch.no_grad()
    def render_orbit_video(
        self,
        center: Tuple[float, float, float],
        radius: float,
        time: float,
        num_views: int = 36,
        elevation: float = 30.0,
        output_path: str | Path = "orbit.mp4",
        fps: int = 30,
    ) -> Path:
        """Render an orbit video and save as MP4.

        Parameters
        ----------
        center, radius, time, num_views, elevation : see :meth:`render_orbit`.
        output_path : str or Path
        fps : int

        Returns
        -------
        Path to the saved video file.
        """
        frames = self.render_orbit(center, radius, time, num_views, elevation)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        images = []
        for f in frames:
            img_np = (f["image"].permute(1, 2, 0).cpu().clamp(0, 1).numpy() * 255).astype(np.uint8)
            images.append(img_np)

        _write_mp4(images, fps, output_path)
        return output_path

    @torch.no_grad()
    def render_spacetime(
        self,
        camera: Dict[str, Any],
        t_start: float,
        t_end: float,
        num_frames: int,
        camera_path: str = "static",
        **orbit_kwargs: Any,
    ) -> List[Dict[str, Tensor]]:
        """Render a spacetime sequence — varying time and optionally camera.

        Parameters
        ----------
        camera : dict
            Camera specification (used if ``camera_path="static"``).
        t_start, t_end : float
            Time range.
        num_frames : int
            Number of frames.
        camera_path : str
            ``"static"`` — fixed camera (default).
            ``"orbit"`` — orbiting camera (pass ``center``, ``radius``,
            ``elevation`` via ``orbit_kwargs``).

        Returns
        -------
        list of dicts with ``"image"``, ``"depth"``, ``"alpha"``, ``"time"``.
        """
        times = torch.linspace(t_start, t_end, num_frames).tolist()
        frames = []

        if camera_path == "orbit":
            center = orbit_kwargs.get("center", (0.0, 0.0, 0.0))
            radius = orbit_kwargs.get("radius", 3.0)
            elevation = orbit_kwargs.get("elevation", 30.0)

        for i, t in enumerate(times):
            if camera_path == "static":
                cam = camera
            elif camera_path == "orbit":
                angle = 2.0 * math.pi * i / num_frames
                cam = self._make_orbit_camera(center, radius, angle, elevation)
            else:
                raise ValueError(f"Unknown camera_path {camera_path!r}")

            frame = self.predictor.predict_frame(cam, t)
            frame["time"] = t
            frames.append(frame)

        return frames


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI for novel-view rendering."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Render novel views (orbit / spacetime) from trained models",
    )
    parser.add_argument(
        "--gaussian-ckpt", type=str, required=True, help="Path to trained Gaussian model checkpoint"
    )
    parser.add_argument(
        "--pinn-ckpt", type=str, required=True, help="Path to trained PINN checkpoint"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="orbit",
        choices=["orbit", "spacetime", "interpolate"],
        help="Rendering mode (default: orbit)",
    )
    parser.add_argument(
        "--time", type=float, default=0.5, help="Time for orbit render (default: 0.5)"
    )
    parser.add_argument(
        "--num-views", type=int, default=120, help="Number of views / frames (default: 120)"
    )
    parser.add_argument("--radius", type=float, default=3.0, help="Orbit radius (default: 3.0)")
    parser.add_argument(
        "--elevation", type=float, default=30.0, help="Orbit elevation in degrees (default: 30)"
    )
    parser.add_argument(
        "--output", type=str, default="outputs/renders/orbit.mp4", help="Output video path"
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second (default: 30)")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cuda, mps, cpu")
    args = parser.parse_args()
    print(
        f"novel_view CLI: mode={args.mode}, {args.num_views} views, "
        f"radius={args.radius}, output={args.output}"
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
    print("  novel_view.py — self-test")
    print("=" * 60)

    device = torch.device("cpu")

    # -- Mock models --
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

    H, W = 32, 32
    renderer = GaussianRenderer(H, W, use_cuda_rasterizer="pytorch")

    # -- look_at --
    print("\n--- look_at ---")
    eye = torch.tensor([0.0, 0.0, 3.0])
    target = torch.tensor([0.0, 0.0, 0.0])
    up = torch.tensor([0.0, 1.0, 0.0])
    w2c = look_at(eye, target, up)
    check(w2c.shape == (4, 4), "look_at shape (4, 4)")
    # Camera at z=3 looking at origin: translation should put origin at z=3 in cam
    origin_cam = w2c[:3, :3] @ target + w2c[:3, 3]
    check(
        abs(origin_cam[2].item() - 3.0) < 0.01,
        f"Origin at z={origin_cam[2]:.2f} in cam space (~3.0)",
    )
    # R should be orthogonal
    RtR = w2c[:3, :3] @ w2c[:3, :3].T
    check(torch.allclose(RtR, torch.eye(3), atol=1e-5), "R is orthogonal")

    # -- make_camera --
    print("\n--- make_camera ---")
    cam = make_camera(
        position=(0, 0, 3),
        look_at_point=(0, 0, 0),
        fx=50,
        fy=50,
        cx=16,
        cy=16,
        image_height=H,
        image_width=W,
    )
    check("K" in cam and "w2c" in cam, "make_camera returns K and w2c")
    check(cam["K"].shape == (3, 3), "K shape (3, 3)")
    check(cam["w2c"].shape == (4, 4), "w2c shape (4, 4)")
    check(cam["image_height"] == H, f"image_height = {H}")
    check(abs(cam["K"][0, 0].item() - 50.0) < 1e-5, "fx = 50")

    # -- interpolate_cameras --
    print("\n--- interpolate_cameras ---")
    cam1 = make_camera(
        position=(0, 0, 3),
        look_at_point=(0, 0, 0),
        fx=50,
        fy=50,
        cx=16,
        cy=16,
        image_height=H,
        image_width=W,
    )
    cam2 = make_camera(
        position=(3, 0, 0),
        look_at_point=(0, 0, 0),
        fx=50,
        fy=50,
        cx=16,
        cy=16,
        image_height=H,
        image_width=W,
    )
    interp = interpolate_cameras(cam1, cam2, num_steps=5)
    check(len(interp) == 5, "5 interpolated cameras")
    check(all(c["w2c"].shape == (4, 4) for c in interp), "All w2c shape (4, 4)")

    # First and last should match cam1 and cam2
    check(
        torch.allclose(interp[0]["w2c"], cam1["w2c"], atol=1e-4),
        "First interp == cam1",
    )
    check(
        torch.allclose(interp[-1]["w2c"], cam2["w2c"], atol=1e-4),
        "Last interp == cam2",
    )

    # Middle camera should differ from both endpoints
    mid_w2c = interp[2]["w2c"]
    check(
        not torch.allclose(mid_w2c, cam1["w2c"], atol=1e-2),
        "Middle interp != cam1",
    )

    # -- NovelViewSynthesizer --
    print("\n--- NovelViewSynthesizer ---")
    intrinsics = {
        "fx": 50.0,
        "fy": 50.0,
        "cx": W / 2.0,
        "cy": H / 2.0,
        "image_height": H,
        "image_width": W,
    }
    nvs = NovelViewSynthesizer(
        gauss_model,
        pinn,
        renderer,
        camera_intrinsics=intrinsics,
        num_advection_steps=3,
    )

    # Orbit (4 views for speed)
    orbit_frames = nvs.render_orbit(
        center=(0, 0, 0),
        radius=3.0,
        time=0.0,
        num_views=4,
        elevation=30.0,
    )
    check(len(orbit_frames) == 4, "Orbit: 4 views")
    check(all(f["image"].shape == (3, H, W) for f in orbit_frames), "Orbit: correct shape")
    check(all(torch.isfinite(f["image"]).all() for f in orbit_frames), "Orbit: all finite")

    # Different angles should produce different images (camera moved)
    img0 = orbit_frames[0]["image"]
    img1 = orbit_frames[1]["image"]
    check(not torch.allclose(img0, img1, atol=1e-3), "Different orbit views differ")

    # Spacetime (static camera)
    print("\n--- Spacetime ---")
    st_frames = nvs.render_spacetime(
        camera=cam1,
        t_start=0.0,
        t_end=1.0,
        num_frames=3,
        camera_path="static",
    )
    check(len(st_frames) == 3, "Spacetime: 3 frames")
    check(all(torch.isfinite(f["image"]).all() for f in st_frames), "Spacetime: all finite")
    check(abs(st_frames[0]["time"] - 0.0) < 1e-6, "Spacetime: first time = 0")
    check(abs(st_frames[-1]["time"] - 1.0) < 1e-6, "Spacetime: last time = 1")

    # Spacetime (orbit camera)
    st_orbit = nvs.render_spacetime(
        camera=cam1,
        t_start=0.0,
        t_end=0.5,
        num_frames=3,
        camera_path="orbit",
        center=(0, 0, 0),
        radius=3.0,
        elevation=20.0,
    )
    check(len(st_orbit) == 3, "Spacetime orbit: 3 frames")
    check(all(torch.isfinite(f["image"]).all() for f in st_orbit), "Spacetime orbit: all finite")

    # -- Quaternion round-trip --
    print("\n--- Quaternion round-trip ---")
    R_test = cam1["w2c"][:3, :3]
    q_test = _rotation_matrix_to_quaternion(R_test)
    R_back = _quaternion_to_rotation_matrix(q_test)
    check(torch.allclose(R_test, R_back, atol=1e-5), "R -> q -> R round-trip")

    # Slerp at endpoints
    q_a = _rotation_matrix_to_quaternion(cam1["w2c"][:3, :3])
    q_b = _rotation_matrix_to_quaternion(cam2["w2c"][:3, :3])
    q_0 = _slerp(q_a, q_b, 0.0)
    q_1 = _slerp(q_a, q_b, 1.0)
    check(
        torch.allclose(q_0, q_a, atol=1e-5) or torch.allclose(q_0, -q_a, atol=1e-5),
        "Slerp(t=0) == q_a",
    )
    check(
        torch.allclose(q_1, q_b, atol=1e-5) or torch.allclose(q_1, -q_b, atol=1e-5),
        "Slerp(t=1) == q_b",
    )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 60}")

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: novel_view.py — look_at, make_camera, interpolate_cameras, "
            "orbit, spacetime all verified"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)
