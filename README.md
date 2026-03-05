# Physics-Integrated Neural Fluid Reconstruction and Novel View Synthesis

**MBZUAI — Department of Computer Vision and Metaverse Lab**

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![PyTorch 2.2+](https://img.shields.io/badge/PyTorch-2.2%2B-ee4c2c?logo=pytorch&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)

---

## Overview

This project reconstructs dynamic fluid phenomena from multi-view video by combining **3D Gaussian Splatting** with **Physics-Informed Neural Networks (PINNs)**. A set of time-varying Gaussians captures the visual appearance across camera views, while a PINN learns a continuous velocity field that satisfies the incompressible Navier-Stokes equations using physics-only residual constraints (no data fitting to ground-truth velocities). The learned velocity field then advects Gaussians to enable **novel view synthesis** at arbitrary camera poses and time steps, as well as **forward prediction** of fluid motion beyond the observed time window.

---

## Key Features

- **Multi-view video preprocessing pipeline** — FPS normalization, temporal synchronization, camera stabilization, COLMAP pose estimation, RAFT optical flow extraction, and Depth Anything V2 monocular depth estimation
- **Dynamic 3D Gaussian Splatting** — time-varying deformation network with Fourier time encoding, adaptive densification/pruning, and multi-loss training (photometric, depth, optical flow, SSIM, temporal smoothness)
- **Physics-Informed Neural Network** — SIREN-based architecture enforcing incompressible Navier-Stokes momentum and continuity equations via automatic differentiation, with configurable boundary conditions (no-slip, inflow, outflow, pressure reference)
- **Physics-guided Gaussian advection** — RK4 numerical integration through the learned PINN velocity field transports Gaussian centres to novel time steps
- **Novel view synthesis** — render from arbitrary camera poses using orbit cameras, interpolated camera paths, or custom trajectories
- **Forward prediction** — extrapolate fluid motion and appearance beyond the training time window

---

## Architecture

```
                            ┌─────────────────────────────────────────────┐
                            │          Multi-View Video Input             │
                            └──────────────────┬──────────────────────────┘
                                               │
                          ┌────────────────────────────────────────┐
                          │    Phase 1: PREPROCESSING PIPELINE     │
                          │                                        │
                          │  Normalize FPS → Sync → Stabilize →    │
                          │  Undistort → COLMAP → Optical Flow →   │
                          │  Monocular Depth                       │
                          └────────┬──────────────────┬────────────┘
                                   │                  │
                    ┌──────────────▼──────┐   ┌──────▼──────────────────┐
                    │  Phase 2: GAUSSIAN  │   │   Phase 3: PINN         │
                    │  SPLATTING TRAINING │   │   TRAINING              │
                    │                     │   │                         │
                    │  Canonical 3DGS +   │   │  SIREN network learns   │
                    │  Deformation MLP    │   │  u(x,t), p(x,t), ρ(x,t)│
                    │  (photometric +     │   │  via Navier-Stokes      │
                    │   depth + flow      │   │  residual minimization  │
                    │   losses)           │   │                         │
                    └──────────┬──────────┘   └──────────┬──────────────┘
                               │                         │
                          ┌────▼─────────────────────────▼────┐
                          │      Phase 4: INTEGRATION         │
                          │                                   │
                          │  PINN velocity field advects      │
                          │  Gaussian positions via RK4 →     │
                          │  Render at novel views & times    │
                          └───────────────┬───────────────────┘
                                          │
                              ┌────────────▼────────────┐
                              │  Novel View Synthesis   │
                              │  Forward Prediction     │
                              │  Orbit / Path Videos    │
                              └─────────────────────────┘
```

---

## Installation

### Prerequisites

- Python 3.10+ (3.11 recommended)
- [conda](https://docs.conda.io/en/latest/miniconda.html) (Miniconda or Anaconda)
- CUDA 12.x (for GPU training on Linux) **or** macOS with Apple Silicon (for MPS-accelerated development)
- [COLMAP](https://colmap.github.io/) and [FFmpeg](https://ffmpeg.org/) (installed automatically by the setup script)

### Quick Start

```bash
git clone <repo-url> fluid-recon
cd fluid-recon
bash scripts/setup_env.sh
```

The setup script will:
1. Create a conda environment named `fluid` with Python 3.11
2. Detect your platform and install PyTorch with the appropriate backend (CUDA / MPS / CPU)
3. Install the project in editable mode with all dependencies
4. Install COLMAP and FFmpeg
5. Run environment verification

### Alternative: Manual Installation

```bash
conda create -n fluid python=3.11 -y
conda activate fluid

# PyTorch (choose one):
pip install torch torchvision torchaudio                                          # macOS (MPS)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu    # CPU only

# Project + dev tools:
pip install -e ".[dev]"

# GPU extras (Linux + CUDA only):
pip install -e ".[cuda,dev]"
```

### Verify Installation

```bash
python scripts/check_env.py
```

This checks Python version, PyTorch backend, all required packages, COLMAP, and FFmpeg.

---

## Data Preparation

### Supported Datasets

| Dataset | Description | Link |
|---------|-------------|------|
| **ScalarFlow** | Real captured scalar transport flows with calibrated multi-view setup | [TUM Download](https://ge.in.tum.de/publications/2019-scalarflow-eckert/) |
| **FluidNexus** | Synthetic and real fluid scenes with ground-truth physics | [HuggingFace](https://huggingface.co/datasets/FluidNexus/FluidNexus) |
| **Custom** | Your own multi-view video of a fluid scene | — |

### Expected Data Format

Place multi-view video files in `data/raw/`:

```
data/raw/
├── cam_00.mp4
├── cam_01.mp4
├── cam_02.mp4
└── cam_03.mp4
```

For **ScalarFlow**, camera calibration is already provided, so you can skip steps 1-4 of preprocessing and start directly from the COLMAP step with pre-calibrated intrinsics.

---

## Usage — Quick Start

The simplest end-to-end workflow:

```bash
conda activate fluid

# 1. Place multi-view videos in data/raw/

# 2. Preprocess (all 7 steps)
make preprocess DATA_DIR=data/raw

# 3. Train Gaussian Splatting
make train-gaussian CONFIG=configs/default.yaml DEVICE=cuda:0

# 4. Train PINN
make train-pinn CONFIG=configs/default.yaml DEVICE=cuda:0

# 5. Generate novel view orbit video
fluid-render \
    --gaussian-ckpt outputs/checkpoints/gaussian_best.pt \
    --pinn-ckpt outputs/checkpoints/pinn_best.pt \
    --mode orbit --num-views 120 --fps 30

# 6. Generate forward prediction video
fluid-predict \
    --gaussian-ckpt outputs/checkpoints/gaussian_best.pt \
    --pinn-ckpt outputs/checkpoints/pinn_best.pt \
    --t-start 0.0 --t-end 2.0 --fps 30
```

For rapid iteration on macOS:

```bash
make train-gaussian CONFIG=configs/mac_debug.yaml DEVICE=mps
make train-pinn CONFIG=configs/mac_debug.yaml DEVICE=mps
```

---

## Usage — Detailed

### Preprocessing

The preprocessing pipeline has 7 sequential steps. Run all at once or individually:

```bash
# All steps
make preprocess DATA_DIR=data/raw

# Or run the pipeline with step control
fluid-preprocess --input-dir data/raw --start-step 1 --end-step 7
```

| Step | Command | Description |
|------|---------|-------------|
| 1 | `python -m preprocessing.normalize_fps --input-dir data/raw --output-dir data/normalized` | Resample all videos to a consistent frame rate |
| 2 | `python -m preprocessing.sync_videos --input-dir data/normalized --output-dir data/synced` | Synchronize video start/end across cameras |
| 3 | `python -m preprocessing.stabilize --input-dir data/synced --output-dir data/stabilized` | Compensate for camera shake |
| 4 | `python -m preprocessing.normalize_intrinsics --input-dir data/stabilized --output-dir data/undistorted` | Undistort frames using calibration parameters |
| 5 | `python -m preprocessing.run_colmap --input-dir data/undistorted --output-dir data/colmap` | COLMAP structure-from-motion for camera poses |
| 6 | `python -m preprocessing.extract_flow --input-dir data/undistorted --output-dir data/flow` | Extract RAFT optical flow (forward + backward) |
| 7 | `python -m preprocessing.extract_depth --input-dir data/undistorted --output-dir data/depth` | Monocular depth via Depth Anything V2 |

### Gaussian Splatting Training

```bash
# Basic training
fluid-train-gaussian --config configs/default.yaml --device cuda:0

# Resume from checkpoint
fluid-train-gaussian --config configs/default.yaml --resume outputs/checkpoints/gaussian_iter_15000.pt

# Monitor with TensorBoard (enabled by default)
tensorboard --logdir outputs/tb_logs
```

Key training stages:
1. **Initialization** — point cloud from COLMAP sparse points + depth unprojection
2. **Optimization** — per-view photometric + depth + flow + temporal smoothness losses
3. **Densification** — adaptive clone/split of Gaussians based on position gradients (iterations 500-15,000)
4. **Deformation** — time-conditioned MLP with Fourier features learns per-Gaussian motion

### PINN Training

The PINN learns a continuous velocity field u(x,t), pressure p(x,t), and density rho(x,t) by minimizing the Navier-Stokes residual. No ground-truth velocity data is required.

```bash
# Basic training
fluid-train-pinn --config configs/default.yaml --device cuda:0

# With WandB logging
fluid-train-pinn --config configs/default.yaml --wandb

# Resume from checkpoint
fluid-train-pinn --config configs/default.yaml --resume outputs/pinn/checkpoints/pinn_iter_25000.pt
```

Training features:
- **Adaptive collocation sampling** — importance-samples points toward high-residual regions
- **Convergence monitoring** — detects loss plateaus and adjusts training
- **Periodic visualization** — 2D slice plots of velocity/pressure/density fields saved during training

### Integration and Rendering

```bash
# Orbit video (turntable around scene at fixed time)
fluid-render \
    --gaussian-ckpt outputs/checkpoints/gaussian_best.pt \
    --pinn-ckpt outputs/checkpoints/pinn_best.pt \
    --mode orbit --time 0.5 --num-views 120 --radius 3.0 --elevation 30

# Forward prediction video (fixed camera, varying time)
fluid-predict \
    --gaussian-ckpt outputs/checkpoints/gaussian_best.pt \
    --pinn-ckpt outputs/checkpoints/pinn_best.pt \
    --t-start 0.0 --t-end 2.0 --fps 30 --resolution 512 512

# Spacetime video (orbiting camera + varying time)
fluid-render \
    --gaussian-ckpt outputs/checkpoints/gaussian_best.pt \
    --pinn-ckpt outputs/checkpoints/pinn_best.pt \
    --mode spacetime --num-views 240 --fps 30
```

---

## Configuration

Configuration uses YAML files loaded via [OmegaConf](https://omegaconf.readthedocs.io/). Override configs are merged on top of `default.yaml`.

### Config Structure

```yaml
data:           # Data paths, resolution, FPS, camera count
preprocessing:  # COLMAP quality, flow model, depth model settings
gaussian:       # Gaussian count, SH degree, learning rates, densification, deformation, loss weights
pinn:           # Network architecture, physics parameters, collocation, loss weights, domain bounds
training:       # Epochs, batch size, save/eval/log intervals, seed, mixed precision
logging:        # WandB and TensorBoard settings
```

### Important Hyperparameters

| Parameter | Config Key | Default | Description |
|-----------|-----------|---------|-------------|
| Resolution | `data.resolution.width/height` | 512 | Training image resolution |
| Num Gaussians | `gaussian.num_gaussians` | 100,000 | Initial Gaussian count |
| Max Gaussians | `gaussian.densify.max_gaussians` | 500,000 | Upper limit after densification |
| SH Degree | `gaussian.sh_degree` | 3 | Spherical harmonics degree for colour |
| Deformation MLP | `gaussian.deformation.mlp_width/depth` | 128 / 6 | Deformation network size |
| PINN Hidden Dim | `pinn.network.hidden_dim` | 256 | PINN layer width |
| PINN Layers | `pinn.network.num_layers` | 8 | PINN depth |
| PINN Activation | `pinn.network.activation` | siren | siren / tanh / swish |
| Viscosity | `pinn.physics.viscosity` | 1e-3 | Kinematic viscosity |
| Collocation Points | `pinn.num_collocation` | 10,000 | Physics sampling points per batch |
| Gaussian Epochs | `training.gaussian.epochs` | 30,000 | Gaussian training iterations |
| PINN Epochs | `training.pinn.epochs` | 50,000 | PINN training iterations |

### Recommended Configurations

| Setup | Config | Notes |
|-------|--------|-------|
| Mac development | `configs/mac_debug.yaml` | 128x128, 5K Gaussians, 1K/5K epochs |
| Single RTX 4090 | `configs/default.yaml` | 512x512, 100K Gaussians, 30K/50K epochs |
| Dual RTX 4090 | `configs/default.yaml` + `make train-all` | Parallel training on cuda:0 and cuda:1 |

---

## Project Structure

```
fluid-recon/
├── configs/
│   ├── __init__.py
│   ├── default.yaml             # Full production configuration
│   ├── mac_debug.yaml           # Lightweight config for macOS development
│   └── test_pinn.yaml           # Minimal config for unit tests
│
├── preprocessing/
│   ├── __init__.py
│   ├── run_pipeline.py          # Orchestrates all 7 preprocessing steps
│   ├── normalize_fps.py         # Resample videos to consistent FPS
│   ├── sync_videos.py           # Temporal synchronization across cameras
│   ├── stabilize.py             # Camera motion stabilization
│   ├── normalize_intrinsics.py  # Lens undistortion
│   ├── run_colmap.py            # COLMAP structure-from-motion
│   ├── extract_flow.py          # RAFT optical flow extraction
│   └── extract_depth.py         # Monocular depth (Depth Anything V2)
│
├── gaussian_splatting/
│   ├── __init__.py
│   ├── model.py                 # GaussianModel, DeformationNetwork, DynamicGaussianModel
│   ├── renderer.py              # Differentiable rendering (PyTorch + gsplat CUDA backends)
│   ├── losses.py                # Photometric, depth, flow, SSIM, temporal losses
│   ├── dataset.py               # Multi-view video dataset loader
│   └── train.py                 # Gaussian splatting training loop
│
├── pinn/
│   ├── __init__.py
│   ├── model.py                 # FluidPINN (SIREN-based velocity/pressure/density network)
│   ├── navier_stokes.py         # NS residual computation via autograd
│   ├── boundary.py              # Boundary condition implementations
│   └── train.py                 # PINN training with adaptive collocation sampling
│
├── integration/
│   ├── __init__.py
│   ├── advection.py             # Euler, RK4 integrators; PhysicsGuidedDeformation
│   ├── forward_predict.py       # Future frame prediction using PINN-advected Gaussians
│   └── novel_view.py            # Camera utilities, orbit/path rendering, video output
│
├── scripts/
│   ├── __init__.py
│   ├── setup_env.sh             # Automated environment setup (conda + PyTorch + deps)
│   └── check_env.py             # Environment verification
│
├── tests/
│   ├── test_model.py            # Gaussian model unit tests
│   ├── test_renderer.py         # Differentiable renderer tests
│   ├── test_losses.py           # Loss function tests
│   ├── test_dataset.py          # Dataset loader tests
│   ├── test_train.py            # Gaussian training loop tests
│   ├── test_pinn_model.py       # PINN architecture tests
│   ├── test_navier_stokes.py    # NS residual computation tests
│   ├── test_boundary.py         # Boundary condition tests
│   ├── test_pinn_train.py       # PINN training loop tests
│   ├── test_integration.py      # Advection, forward prediction, novel view tests
│   ├── test_normalize_intrinsics.py
│   ├── test_extract_flow.py
│   └── test_extract_depth.py
│
├── data/                        # Data directories (git-ignored)
│   ├── raw/                     # Input multi-view videos
│   ├── normalized/              # FPS-normalized frames
│   ├── synced/                  # Temporally synchronized frames
│   ├── stabilized/              # Stabilized frames
│   ├── undistorted/             # Undistorted frames
│   ├── colmap/                  # COLMAP sparse reconstruction
│   ├── flow/                    # Optical flow maps
│   └── depth/                   # Monocular depth maps
│
├── outputs/                     # Training outputs (git-ignored)
│   ├── checkpoints/             # Model checkpoints (.pt)
│   ├── renders/                 # Rendered images and videos
│   └── tb_logs/                 # TensorBoard logs
│
├── pyproject.toml               # Package metadata and dependencies
├── requirements.txt             # Flat dependency list
├── environment.yml              # Conda environment specification
├── Makefile                     # Build/train/test automation
├── .flake8                      # Flake8 linting configuration
└── .gitignore                   # Git ignore rules
```

---

## Hardware Requirements

| | Minimum (Development) | Recommended (Training) |
|---|---|---|
| **GPU** | MacBook with M-series chip (MPS) | NVIDIA RTX 4090 (24 GB VRAM) |
| **RAM** | 16 GB | 32 GB+ |
| **Storage** | 20 GB (code + small dataset) | 100 GB+ (full dataset + checkpoints) |

### Estimated Training Times (RTX 4090)

| Component | Default Config | Mac Debug Config |
|-----------|---------------|-----------------|
| Gaussian Splatting (30K iter) | ~4-8 hours | ~5-15 min (MPS) |
| PINN (50K iter) | ~2-4 hours | ~5-10 min (MPS) |

### VRAM Usage

| Component | Estimated VRAM |
|-----------|---------------|
| 100K Gaussians at 512x512 | ~16 GB |
| 500K Gaussians at 512x512 | ~22 GB |
| PINN (256-dim, 8 layers) | ~4 GB |
| 5K Gaussians at 128x128 (debug) | ~1 GB |

---

## Results

> **Placeholder** — add figures, tables, and metrics here as experiments are completed.

### Novel View Synthesis Quality

| Metric | Description | Expected Range |
|--------|-------------|---------------|
| PSNR (dB) | Peak signal-to-noise ratio | 25-35 |
| SSIM | Structural similarity index | 0.85-0.95 |
| LPIPS | Learned perceptual image patch similarity (lower is better) | 0.05-0.15 |

### Physics Metrics

| Metric | Description |
|--------|-------------|
| NS Residual | Mean magnitude of Navier-Stokes momentum residual on held-out collocation points |
| Divergence Error | Mean |div(u)| measuring incompressibility violation |

<!-- Add result figures:
![Novel Views](outputs/renders/novel_views.png)
![Forward Prediction](outputs/renders/forward_prediction.png)
![Velocity Field](outputs/renders/velocity_field.png)
-->

---

## Methodology

### Dynamic 3D Gaussian Splatting

The scene is represented as a set of anisotropic 3D Gaussian primitives in a canonical frame. Each Gaussian is parameterized by position, rotation (quaternion), scale, opacity, and spherical harmonics colour coefficients. A **deformation network** — a time-conditioned MLP with Fourier feature encoding — predicts per-Gaussian offsets (position, rotation, scale, opacity deltas) given the current timestep. This allows the Gaussians to move, rotate, and change appearance over time.

Training minimizes a combination of:
- **Photometric loss** (L1 + SSIM between rendered and observed images)
- **Depth loss** (alignment with monocular depth estimates)
- **Optical flow loss** (consistency with RAFT flow predictions)
- **Temporal smoothness** (regularization on deformation magnitude)

Adaptive densification clones or splits Gaussians in under-reconstructed regions based on position gradient magnitude, running from iteration 500 to 15,000.

### Physics-Informed Neural Network

The PINN is a SIREN-based network that takes spatial coordinates (x, y, z) and time (t) as input and predicts velocity u(x,t), pressure p(x,t), and density rho(x,t). The network is trained by minimizing the residual of the incompressible Navier-Stokes equations, computed entirely via `torch.autograd.grad`:

**Momentum equation:**

$$\frac{\partial \mathbf{u}}{\partial t} + (\mathbf{u} \cdot \nabla)\mathbf{u} = -\frac{1}{\rho}\nabla p + \nu \nabla^2 \mathbf{u} + \mathbf{f}$$

**Continuity equation (incompressibility):**

$$\nabla \cdot \mathbf{u} = 0$$

Boundary conditions (no-slip walls, inflow/outflow profiles, pressure reference) are enforced as soft constraints. Collocation points are adaptively sampled toward high-residual regions during training.

### Integration: Physics-Guided Advection

The trained PINN provides a continuous, differentiable velocity field. To render at a novel time, canonical Gaussian positions are advected through this field using fourth-order Runge-Kutta (RK4) integration:

$$\mathbf{x}(t + \Delta t) = \mathbf{x}(t) + \frac{\Delta t}{6}(\mathbf{k}_1 + 2\mathbf{k}_2 + 2\mathbf{k}_3 + \mathbf{k}_4)$$

where each k_i queries the PINN velocity at intermediate positions and times. This replaces the learned deformation MLP at inference time, enabling physically consistent extrapolation to unseen timesteps.

---

## References

1. Kerbl, B., Kopanas, G., Leimkühler, T., & Drettakis, G. (2023). **3D Gaussian Splatting for Real-Time Radiance Field Rendering.** *ACM Transactions on Graphics (SIGGRAPH 2023).*
2. Mildenhall, B., Srinivasan, P. P., Tancik, M., Barron, J. T., Ramamoorthi, R., & Ng, R. (2020). **NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis.** *ECCV 2020.*
3. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). **Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations.** *Journal of Computational Physics, 378*, 686-707.
4. Zhong, Y., Deng, Y., Luo, G., Zhao, H., & Zhang, J. (2023). **FluidNexus: A Unified Neural Solver for Fluid Simulation and Rendering.** *SIGGRAPH 2023.*
5. Gao, H., et al. (2025). **FluidNexus: 3D Fluid Reconstruction and Prediction from a Single Video.** *CVPR 2025.*

---

## Team

| Role | Name | Contact |
|------|------|---------|
| Faculty Advisor | Dr. J. Alejandro Amador Herrera | jorge.herrera@mbzuai.ac.ae |
| Team Member | Ivan Kanev | |
| Team Member | Delyan Hristov | |
| Team Member | Stoyan Ganchev | |
| Team Member | Makar Ulesov | |

**Institution:** Mohamed bin Zayed University of Artificial Intelligence (MBZUAI)

---

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2025 Makar Ulesov, Ivan Kanev, Delyan Hristov, Stoyan Ganchev

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

For academic use, please cite the relevant references listed above.

---

## Acknowledgments

- **MBZUAI Computer Vision and Metaverse Lab** for compute resources and research support
- **Technical University of Munich (TUM)** for the [ScalarFlow](https://ge.in.tum.de/publications/2019-scalarflow-eckert/) dataset
- **Stanford University** for the [FluidNexus](https://huggingface.co/datasets/FluidNexus/FluidNexus) dataset
- The authors of [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/), [gsplat](https://github.com/nerfstudio-project/gsplat), and [RAFT](https://github.com/princeton-vl/RAFT) for their open-source implementations
