#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# setup_env.sh — Create and configure the "fluid" conda environment.
#
# Works on both macOS (MPS) and Linux (CUDA / CPU).
# From a fresh clone:  bash scripts/setup_env.sh
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ENV_NAME="fluid"
PYTHON_VERSION="3.11"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[ OK ]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERR ]${NC} $*"; }

# ── 1. Check conda ──────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    err "conda not found."
    echo ""
    echo "  Install Miniconda first:"
    echo "    https://docs.conda.io/en/latest/miniconda.html"
    echo ""
    echo "  Quick install (Linux):"
    echo "    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    echo "    bash Miniconda3-latest-Linux-x86_64.sh"
    echo ""
    echo "  Quick install (macOS):"
    echo "    brew install --cask miniconda"
    echo "    # or: wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh"
    echo ""
    exit 1
fi

# Allow conda activate inside scripts
eval "$(conda shell.bash hook)"

# ── 2. Detect platform ─────────────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"
info "Detected OS=$OS  ARCH=$ARCH"

HAS_CUDA=false
CUDA_TAG="cu124"  # default fallback

if [[ "$OS" == "Darwin" ]]; then
    PLATFORM="macos"
    info "Platform: macOS — will use MPS backend"
elif [[ "$OS" == "Linux" ]]; then
    PLATFORM="linux"
    if command -v nvcc &>/dev/null; then
        # Try to detect CUDA version from nvcc
        NVCC_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' || echo "")
        if [[ -n "$NVCC_VERSION" ]]; then
            HAS_CUDA=true
            # Convert e.g. "12.4" -> "cu124"
            CUDA_TAG="cu$(echo "$NVCC_VERSION" | tr -d '.')"
            info "Platform: Linux — CUDA $NVCC_VERSION detected (tag: $CUDA_TAG)"
        fi
    elif command -v nvidia-smi &>/dev/null; then
        # Fallback: detect CUDA from nvidia-smi
        DRIVER_CUDA=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo "")
        if [[ -n "$DRIVER_CUDA" ]]; then
            HAS_CUDA=true
            CUDA_TAG="cu$(echo "$DRIVER_CUDA" | tr -d '.')"
            info "Platform: Linux — NVIDIA driver supports CUDA $DRIVER_CUDA (tag: $CUDA_TAG)"
        fi
    fi
    if [[ "$HAS_CUDA" == "false" ]]; then
        warn "Platform: Linux — no CUDA detected, installing CPU-only PyTorch"
    fi
else
    err "Unsupported OS: $OS"
    exit 1
fi

# ── 3. Create / recreate conda environment ──────────────────────────────────
if conda env list | grep -qw "$ENV_NAME"; then
    warn "Environment '$ENV_NAME' already exists."
    read -rp "Recreate it from scratch? [y/N] " ans
    if [[ "$ans" =~ ^[Yy]$ ]]; then
        info "Removing existing environment..."
        conda deactivate 2>/dev/null || true
        conda env remove -n "$ENV_NAME" -y
    else
        info "Reusing existing environment."
    fi
fi

if ! conda env list | grep -qw "$ENV_NAME"; then
    info "Creating conda environment '$ENV_NAME' with Python $PYTHON_VERSION..."
    conda create -n "$ENV_NAME" python="$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"
ok "Activated environment: $ENV_NAME  (Python $(python --version 2>&1 | awk '{print $2}'))"

# ── 4. Install PyTorch ──────────────────────────────────────────────────────
info "Installing PyTorch..."
if [[ "$PLATFORM" == "macos" ]]; then
    pip install torch torchvision torchaudio
elif [[ "$HAS_CUDA" == "true" ]]; then
    pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/$CUDA_TAG"
else
    # Linux CPU-only
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
ok "PyTorch installed."

# ── 5. Install project in editable mode ─────────────────────────────────────
info "Installing fluid-recon in editable mode..."
cd "$PROJECT_DIR"
if [[ "$PLATFORM" == "linux" && "$HAS_CUDA" == "true" ]]; then
    pip install -e ".[cuda,dev]"
else
    pip install -e ".[dev]"
fi
ok "fluid-recon installed."

# ── 6. Install COLMAP ───────────────────────────────────────────────────────
if command -v colmap &>/dev/null; then
    ok "COLMAP already installed: $(which colmap)"
else
    info "Installing COLMAP..."
    if [[ "$PLATFORM" == "macos" ]]; then
        if command -v brew &>/dev/null; then
            brew install colmap || warn "COLMAP installation failed (Homebrew). Install manually if needed for preprocessing."
        else
            warn "Homebrew not found. Install COLMAP manually: https://colmap.github.io/install.html"
        fi
    elif [[ "$PLATFORM" == "linux" ]]; then
        info "(may require sudo)"
        sudo apt-get update -qq && sudo apt-get install -y colmap \
            || warn "COLMAP installation failed. Install manually if needed for preprocessing."
    fi
fi

# ── 7. Install FFmpeg ───────────────────────────────────────────────────────
if command -v ffmpeg &>/dev/null; then
    ok "FFmpeg already installed: $(which ffmpeg)"
else
    info "Installing FFmpeg..."
    if [[ "$PLATFORM" == "macos" ]]; then
        if command -v brew &>/dev/null; then
            brew install ffmpeg || warn "FFmpeg installation failed."
        else
            warn "Homebrew not found. Install FFmpeg manually: https://ffmpeg.org/download.html"
        fi
    elif [[ "$PLATFORM" == "linux" ]]; then
        sudo apt-get update -qq && sudo apt-get install -y ffmpeg \
            || warn "FFmpeg installation failed."
    fi
fi

# ── 8. Verify environment ──────────────────────────────────────────────────
echo ""
info "Running environment verification..."
python "$SCRIPT_DIR/check_env.py" || true

# ── 9. Done ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD} Setup complete!${NC}"
echo -e "${CYAN}════════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  Activate the environment:"
echo -e "    ${YELLOW}conda activate $ENV_NAME${NC}"
echo ""
echo -e "  Verify everything works:"
echo -e "    ${YELLOW}python scripts/check_env.py${NC}"
echo ""
echo -e "  Quick start:"
echo -e "    ${YELLOW}make test${NC}                    # run all tests"
echo -e "    ${YELLOW}make preprocess DATA_DIR=...${NC}  # preprocess raw data"
echo -e "    ${YELLOW}make train-gaussian${NC}           # train Gaussian model"
echo -e "    ${YELLOW}make train-pinn${NC}               # train PINN model"
echo ""
