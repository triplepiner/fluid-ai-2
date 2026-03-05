#!/usr/bin/env python3
"""Verify that the fluid-recon environment is correctly configured."""

import importlib
import shutil
import subprocess
import sys


def check_python_version():
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro}")
    if v.major != 3 or v.minor < 11:
        print("  WARNING: Python 3.11+ recommended")
    else:
        print("  OK")


def check_torch():
    import torch

    print(f"PyTorch {torch.__version__}")

    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
            print(f"  CUDA device {i}: {name} ({mem:.1f} GB)")
        x = torch.randn(2, 2, device="cuda")
        print(f"  CUDA tensor test: OK (device={x.device})")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        print("  MPS: available")
        x = torch.randn(2, 2, device="mps")
        print(f"  MPS tensor test: OK (device={x.device})")
    else:
        print("  Accelerator: none (CPU only)")
        print("  WARNING: training will be extremely slow without GPU/MPS")


def check_packages():
    packages = {
        "numpy": "numpy",
        "scipy": "scipy",
        "einops": "einops",
        "tqdm": "tqdm",
        "omegaconf": "omegaconf",
        "yaml": "yaml",
        "imageio": "imageio",
        "cv2": "cv2",
        "kornia": "kornia",
        "PIL": "PIL",
        "plyfile": "plyfile",
        "open3d": "open3d",
        "wandb": "wandb",
        "tensorboard": "tensorboard",
        "timm": "timm",
    }

    missing = []
    for display_name, import_name in packages.items():
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {display_name}: {version}")
        except ImportError:
            print(f"  {display_name}: MISSING")
            missing.append(display_name)

    # gsplat is Linux/CUDA only
    try:
        importlib.import_module("gsplat")
        print("  gsplat: installed")
    except ImportError:
        import platform

        if platform.system() == "Linux":
            print("  gsplat: MISSING (required on Linux)")
            missing.append("gsplat")
        else:
            print("  gsplat: skipped (CUDA-only, not needed on macOS)")

    return missing


def check_colmap():
    path = shutil.which("colmap")
    if path:
        result = subprocess.run(
            ["colmap", "version"],
            capture_output=True,
            text=True,
        )
        # COLMAP prints version to stderr in some builds
        version_text = (result.stdout + result.stderr).strip().split("\n")[0]
        print(f"COLMAP: {version_text} ({path})")
    else:
        print("COLMAP: NOT FOUND")
        print("  Install: https://colmap.github.io/install.html")
        print("  macOS:   brew install colmap")
        print("  Ubuntu:  sudo apt install colmap")
        return False
    return True


def check_ffmpeg():
    path = shutil.which("ffmpeg")
    if path:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
        )
        version_line = result.stdout.strip().split("\n")[0]
        print(f"FFmpeg: {version_line} ({path})")
    else:
        print("FFmpeg: NOT FOUND")
        print("  Install: https://ffmpeg.org/download.html")
        print("  macOS:   brew install ffmpeg")
        print("  Ubuntu:  sudo apt install ffmpeg")
        return False
    return True


def main():
    print("=" * 60)
    print("  fluid-recon environment check")
    print("=" * 60)
    print()

    print("── Python ──────────────────────────────────────")
    check_python_version()
    print()

    print("── PyTorch & Accelerator ───────────────────────")
    try:
        check_torch()
    except ImportError:
        print("  PyTorch: NOT INSTALLED")
    print()

    print("── Python packages ────────────────────────────")
    missing = check_packages()
    print()

    print("── External tools ─────────────────────────────")
    colmap_ok = check_colmap()
    ffmpeg_ok = check_ffmpeg()
    print()

    # Final verdict
    print("=" * 60)
    issues = []
    if missing:
        issues.append(f"Missing packages: {', '.join(missing)}")
    if not colmap_ok:
        issues.append("COLMAP not found")
    if not ffmpeg_ok:
        issues.append("FFmpeg not found")

    if issues:
        print("  ISSUES FOUND:")
        for issue in issues:
            print(f"    - {issue}")
        print()
        print("  Run 'bash scripts/setup_env.sh' to install missing packages.")
        sys.exit(1)
    else:
        print("  All checks passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
