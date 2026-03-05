#!/usr/bin/env python3
"""PyTorch dataset for multi-view dynamic fluid reconstruction.

Loads preprocessed data (undistorted frames, camera parameters, optical flow,
depth maps) and serves ``(camera, timestep)`` pairs for dynamic 3D Gaussian
splatting training.

Expected directory layout::

    data_root/
        colmap/cameras_normalized.json
        undistorted/cam_XX/frame_XXXXX.png
        flow/cam_XX/flow_fwd_XXXXX.npy
        flow/cam_XX/flow_bwd_XXXXX.npy
        flow/cam_XX/flow_mask_XXXXX.png
        depth/cam_XX/depth_XXXXX.npy
"""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


# ---------------------------------------------------------------------------
# Camera loading
# ---------------------------------------------------------------------------


def load_cameras(cameras_json_path: Path | str) -> list[dict[str, Any]]:
    """Parse ``cameras_normalized.json`` into a list of camera dicts.

    Each dict contains:

    - ``name`` (str): camera directory name, e.g. ``"cam_00"``
    - ``camera_id`` (int)
    - ``image_width``, ``image_height`` (int)
    - ``K`` (np.ndarray): ``(3, 3)`` float32 intrinsic matrix
    - ``R`` (np.ndarray): ``(3, 3)`` float32 rotation (world-to-camera)
    - ``t`` (np.ndarray): ``(3,)`` float32 translation (world-to-camera)
    - ``c2w`` (np.ndarray): ``(4, 4)`` float32 camera-to-world matrix
    - ``w2c`` (np.ndarray): ``(4, 4)`` float32 world-to-camera matrix
    """
    with open(cameras_json_path) as f:
        data = json.load(f)

    cameras_raw = data.get("cameras", {})
    cameras: list[dict[str, Any]] = []

    for cam_name in sorted(cameras_raw):
        entry = cameras_raw[cam_name]
        intr = entry["intrinsic"]

        K = np.array(
            [
                [intr["fx"], 0.0, intr["cx"]],
                [0.0, intr["fy"], intr["cy"]],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        c2w = np.array(entry["camera_to_world"], dtype=np.float32)
        w2c = np.array(entry["world_to_camera"], dtype=np.float32)

        cameras.append(
            {
                "name": cam_name,
                "camera_id": entry["camera_id"],
                "image_width": entry["image_width"],
                "image_height": entry["image_height"],
                "K": K,
                "R": w2c[:3, :3].copy(),
                "t": w2c[:3, 3].copy(),
                "c2w": c2w,
                "w2c": w2c,
            }
        )

    return cameras


# ---------------------------------------------------------------------------
# Ray generation
# ---------------------------------------------------------------------------


def get_rays(
    H: int,
    W: int,
    K: torch.Tensor,
    c2w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate ray origins and directions for every pixel.

    Parameters
    ----------
    H, W : int
        Image dimensions.
    K : torch.Tensor
        ``(3, 3)`` intrinsic matrix.
    c2w : torch.Tensor
        ``(4, 4)`` camera-to-world matrix.

    Returns
    -------
    origins : torch.Tensor
        ``(H, W, 3)`` ray origins in world coordinates.
    directions : torch.Tensor
        ``(H, W, 3)`` normalised ray directions in world coordinates.
    """
    j, i = torch.meshgrid(
        torch.arange(H, dtype=torch.float32),
        torch.arange(W, dtype=torch.float32),
        indexing="ij",
    )

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    dirs_cam = torch.stack(
        [
            (i - cx) / fx,
            (j - cy) / fy,
            torch.ones_like(i),
        ],
        dim=-1,
    )  # (H, W, 3)

    R = c2w[:3, :3]
    dirs_world = dirs_cam @ R.T  # (H, W, 3)
    dirs_world = dirs_world / dirs_world.norm(dim=-1, keepdim=True)

    origins = c2w[:3, 3].expand(H, W, -1).contiguous()

    return origins, dirs_world


# ---------------------------------------------------------------------------
# LRU cache
# ---------------------------------------------------------------------------


class _LRUCache:
    """Thread-safe LRU cache with fixed capacity."""

    def __init__(self, maxsize: int = 100):
        self._maxsize = maxsize
        self._cache: OrderedDict[Any, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Any) -> Any | None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        return None

    def put(self, key: Any, value: Any) -> None:
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            else:
                if len(self._cache) >= self._maxsize:
                    self._cache.popitem(last=False)
                self._cache[key] = value

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()


# ---------------------------------------------------------------------------
# Custom collate
# ---------------------------------------------------------------------------


def fluid_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate function that handles ``None`` values in samples.

    - Tensors are stacked along a new batch dimension.
    - Scalars (int/float) are collected into a tensor.
    - ``None`` values are kept as a list of Nones.
    - Mixed tensor/None entries are kept as a list.
    """
    result: dict[str, Any] = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        first = values[0]
        if first is None:
            result[key] = values
        elif isinstance(first, torch.Tensor):
            if any(v is None for v in values):
                result[key] = values
            else:
                result[key] = torch.stack(values)
        elif isinstance(first, (int, float)):
            result[key] = torch.tensor(values)
        else:
            result[key] = values
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FluidDataset(Dataset):
    """Multi-view video dataset for dynamic 3D Gaussian splatting.

    Each sample is a ``(camera_index, timestep)`` pair providing the image,
    camera parameters, and optionally depth, flow, and flow mask.

    Parameters
    ----------
    data_root : Path or str
        Root directory containing ``undistorted/``, ``flow/``, ``depth/``,
        and ``colmap/cameras_normalized.json``.
    split : str
        ``"train"`` or ``"val"``.  Every *val_every*-th timestep is held out
        for validation.
    resolution : tuple[int, int] or None
        Target ``(H, W)``.  Frames larger than this are resized on load.
    load_flow : bool
        Whether to load optical flow fields.
    load_depth : bool
        Whether to load depth maps.
    time_range : tuple[int, int] or None
        Restrict to timesteps in ``[start, end)``.
    val_every : int
        Hold out every Nth timestep for validation (default 10).
    cache_size : int
        LRU cache capacity for loaded samples (default 100).
    augment : bool or None
        Enable data augmentation.  Defaults to ``True`` for train split.
    """

    def __init__(
        self,
        data_root: Path | str,
        split: str = "train",
        resolution: tuple[int, int] | None = None,
        load_flow: bool = True,
        load_depth: bool = True,
        time_range: tuple[int, int] | None = None,
        val_every: int = 10,
        cache_size: int = 100,
        augment: bool | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.resolution = resolution
        self.load_flow = load_flow
        self.load_depth = load_depth
        self.val_every = val_every
        self.augment = augment if augment is not None else (split == "train")

        # Directory layout.
        self.frames_dir = self.data_root / "undistorted"
        self.flow_dir = self.data_root / "flow"
        self.depth_dir = self.data_root / "depth"
        cameras_json = self.data_root / "colmap" / "cameras_normalized.json"

        # Load cameras.
        self.cameras = load_cameras(cameras_json)
        self.num_cameras = len(self.cameras)

        # Discover timesteps from the first camera's frame directory.
        first_cam = self.cameras[0]["name"]
        frame_dir = self.frames_dir / first_cam
        frame_files = sorted(
            p for p in frame_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        self._frame_names = [f.name for f in frame_files]
        self._num_total_frames = len(self._frame_names)

        # Time range.
        if time_range is not None:
            self._t_start = max(0, time_range[0])
            self._t_end = min(self._num_total_frames, time_range[1])
        else:
            self._t_start = 0
            self._t_end = self._num_total_frames

        self._total_timesteps = self._t_end - self._t_start

        # Train/val split: hold out every val_every-th timestep.
        all_t = list(range(self._t_start, self._t_end))
        if split == "val":
            self._timesteps = [t for t in all_t if (t - self._t_start) % val_every == 0]
        else:
            self._timesteps = [t for t in all_t if (t - self._t_start) % val_every != 0]

        # Flat (camera_idx, timestep) index.
        self._indices: list[tuple[int, int]] = []
        for cam_idx in range(self.num_cameras):
            for t in self._timesteps:
                self._indices.append((cam_idx, t))

        # Read cache.
        self._cache = _LRUCache(maxsize=cache_size)

        # Pre-compute K matrices scaled to target resolution.
        self._K_scaled: list[np.ndarray] = []
        for cam in self.cameras:
            K = cam["K"].copy()
            if self.resolution is not None:
                src_h, src_w = cam["image_height"], cam["image_width"]
                tgt_h, tgt_w = self.resolution
                K[0, :] *= tgt_w / src_w
                K[1, :] *= tgt_h / src_h
            self._K_scaled.append(K)

    @property
    def total_timesteps(self) -> int:
        """Total timesteps in the time range (before train/val split)."""
        return self._total_timesteps

    def __len__(self) -> int:
        return len(self._indices)

    # -- loaders ----------------------------------------------------------

    def _load_image(self, cam_name: str, timestep: int) -> np.ndarray:
        """Load image as ``(H, W, 3)`` float32 RGB in ``[0, 1]``."""
        path = self.frames_dir / cam_name / self._frame_names[timestep]
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        if self.resolution is not None:
            tgt_h, tgt_w = self.resolution
            if img.shape[0] != tgt_h or img.shape[1] != tgt_w:
                img = cv2.resize(img, (tgt_w, tgt_h), interpolation=cv2.INTER_AREA)
        return img

    def _load_depth(self, cam_name: str, timestep: int) -> np.ndarray | None:
        """Load depth as ``(H, W)`` float32, or ``None``."""
        if not self.load_depth:
            return None
        path = self.depth_dir / cam_name / f"depth_{timestep:05d}.npy"
        if not path.exists():
            return None
        depth = np.load(str(path))
        if self.resolution is not None:
            tgt_h, tgt_w = self.resolution
            if depth.shape[0] != tgt_h or depth.shape[1] != tgt_w:
                depth = cv2.resize(depth, (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
        return depth

    def _resize_flow(self, flow: np.ndarray) -> np.ndarray:
        """Resize ``(2, H, W)`` flow and rescale displacement magnitudes."""
        if self.resolution is None:
            return flow
        tgt_h, tgt_w = self.resolution
        src_h, src_w = flow.shape[1], flow.shape[2]
        if src_h == tgt_h and src_w == tgt_w:
            return flow
        u = cv2.resize(flow[0], (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR) * (tgt_w / src_w)
        v = cv2.resize(flow[1], (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR) * (tgt_h / src_h)
        return np.stack([u, v], axis=0)

    def _load_flow_fwd(self, cam_name: str, timestep: int) -> np.ndarray | None:
        """Load forward flow ``(2, H, W)`` for frame *timestep* → *timestep+1*."""
        if not self.load_flow or timestep >= self._num_total_frames - 1:
            return None
        path = self.flow_dir / cam_name / f"flow_fwd_{timestep:05d}.npy"
        if not path.exists():
            return None
        return self._resize_flow(np.load(str(path)))

    def _load_flow_bwd(self, cam_name: str, timestep: int) -> np.ndarray | None:
        """Load backward flow ``(2, H, W)`` for frame *timestep* → *timestep-1*.

        The backward flow for pair *i* goes from frame *i+1* to frame *i*,
        so backward flow at timestep *t* is stored as ``flow_bwd_{t-1}.npy``.
        """
        if not self.load_flow or timestep <= 0:
            return None
        pair_idx = timestep - 1
        path = self.flow_dir / cam_name / f"flow_bwd_{pair_idx:05d}.npy"
        if not path.exists():
            return None
        return self._resize_flow(np.load(str(path)))

    def _load_flow_mask(self, cam_name: str, timestep: int) -> np.ndarray | None:
        """Load forward-backward consistency mask as ``(H, W)`` float32 binary."""
        if not self.load_flow or timestep >= self._num_total_frames - 1:
            return None
        path = self.flow_dir / cam_name / f"flow_mask_{timestep:05d}.png"
        if not path.exists():
            return None
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None
        mask = (mask > 127).astype(np.float32)
        if self.resolution is not None:
            tgt_h, tgt_w = self.resolution
            if mask.shape[0] != tgt_h or mask.shape[1] != tgt_w:
                mask = cv2.resize(mask, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)
        return mask

    def _load_raw(self, cam_idx: int, timestep: int) -> dict[str, Any]:
        """Load all data for a ``(camera, timestep)`` pair, with caching."""
        key = (cam_idx, timestep)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        cam_name = self.cameras[cam_idx]["name"]
        data = {
            "image": self._load_image(cam_name, timestep),
            "depth": self._load_depth(cam_name, timestep),
            "flow_fwd": self._load_flow_fwd(cam_name, timestep),
            "flow_bwd": self._load_flow_bwd(cam_name, timestep),
            "flow_mask": self._load_flow_mask(cam_name, timestep),
        }
        self._cache.put(key, data)
        return data

    # -- __getitem__ ------------------------------------------------------

    def __getitem__(self, idx: int) -> dict[str, Any]:
        cam_idx, timestep = self._indices[idx]
        cam = self.cameras[cam_idx]
        raw = self._load_raw(cam_idx, timestep)

        # Copy mutable arrays so augmentation doesn't corrupt the cache.
        image = raw["image"].copy()
        depth = raw["depth"].copy() if raw["depth"] is not None else None
        flow_fwd = raw["flow_fwd"].copy() if raw["flow_fwd"] is not None else None
        flow_bwd = raw["flow_bwd"].copy() if raw["flow_bwd"] is not None else None
        flow_mask = raw["flow_mask"].copy() if raw["flow_mask"] is not None else None

        K = self._K_scaled[cam_idx].copy()
        c2w = cam["c2w"].copy()
        w2c = cam["w2c"].copy()
        H, W = image.shape[:2]

        if self.augment:
            # --- Horizontal flip ---
            if torch.rand(1).item() < 0.5:
                image = np.ascontiguousarray(image[:, ::-1])
                if depth is not None:
                    depth = np.ascontiguousarray(depth[:, ::-1])
                if flow_fwd is not None:
                    flow_fwd = np.ascontiguousarray(flow_fwd[:, :, ::-1])
                    flow_fwd[0] *= -1  # negate horizontal displacement
                if flow_bwd is not None:
                    flow_bwd = np.ascontiguousarray(flow_bwd[:, :, ::-1])
                    flow_bwd[0] *= -1
                if flow_mask is not None:
                    flow_mask = np.ascontiguousarray(flow_mask[:, ::-1])

                # Mirror intrinsics: reflect principal point.
                K[0, 2] = W - 1.0 - K[0, 2]
                # Flip camera x-axis in both c2w and w2c.
                c2w[:3, 0] *= -1
                w2c[0, :] *= -1

            # --- Color jitter (brightness ±0.1, contrast ±0.1) ---
            brightness = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            contrast = 1.0 + (torch.rand(1).item() - 0.5) * 0.2
            mean = image.mean()
            image = (image - mean) * contrast + mean
            image = image * brightness
            np.clip(image, 0.0, 1.0, out=image)

        # --- Convert to tensors ---
        image_t = torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1)  # (3, H, W)

        time_norm = float((timestep - self._t_start) / max(self._total_timesteps - 1, 1))

        sample: dict[str, Any] = {
            "image": image_t,
            "camera_id": cam["camera_id"],
            "timestep": timestep,
            "time_normalized": time_norm,
            "K": torch.from_numpy(K),
            "R": torch.from_numpy(w2c[:3, :3].copy()),
            "t": torch.from_numpy(w2c[:3, 3].copy()),
            "c2w": torch.from_numpy(c2w),
            "w2c": torch.from_numpy(w2c),
            "depth": (torch.from_numpy(depth)[None] if depth is not None else None),
            "flow_fwd": (
                torch.from_numpy(np.ascontiguousarray(flow_fwd)) if flow_fwd is not None else None
            ),
            "flow_bwd": (
                torch.from_numpy(np.ascontiguousarray(flow_bwd)) if flow_bwd is not None else None
            ),
            "flow_mask": (torch.from_numpy(flow_mask)[None] if flow_mask is not None else None),
        }
        return sample


# ---------------------------------------------------------------------------
# DataModule
# ---------------------------------------------------------------------------


class FluidDataModule:
    """Wraps :class:`FluidDataset` with train and validation DataLoaders.

    Parameters
    ----------
    data_root : Path or str
        Passed through to :class:`FluidDataset`.
    resolution : tuple[int, int] or None
        Target ``(H, W)``.
    load_flow, load_depth : bool
        Whether to load flow / depth.
    time_range : tuple[int, int] or None
        Restrict timestep range.
    val_every : int
        Hold-out interval for validation.
    cache_size : int
        LRU cache capacity per dataset instance.
    batch_size : int
        Batch size for both loaders (default 1 for 3DGS).
    num_workers : int
        Number of DataLoader worker processes.
    """

    def __init__(
        self,
        data_root: Path | str,
        resolution: tuple[int, int] | None = None,
        load_flow: bool = True,
        load_depth: bool = True,
        time_range: tuple[int, int] | None = None,
        val_every: int = 10,
        cache_size: int = 100,
        batch_size: int = 1,
        num_workers: int = 4,
    ) -> None:
        shared = dict(
            data_root=data_root,
            resolution=resolution,
            load_flow=load_flow,
            load_depth=load_depth,
            time_range=time_range,
            val_every=val_every,
            cache_size=cache_size,
        )
        self.train_dataset = FluidDataset(split="train", augment=True, **shared)
        self.val_dataset = FluidDataset(split="val", augment=False, **shared)
        self.batch_size = batch_size
        self.num_workers = num_workers

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=fluid_collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=fluid_collate,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
