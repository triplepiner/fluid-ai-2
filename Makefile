.PHONY: setup test preprocess train-gaussian train-pinn train-all predict render clean lint format

# Defaults (override on CLI: make train-gaussian CONFIG=path DEVICE=cuda:0)
CONFIG ?= configs/default.yaml
DEVICE ?= auto
DATA_DIR ?= data/raw

# ── Setup ────────────────────────────────────────────────────────────────────
setup:
	bash scripts/setup_env.sh

# ── Tests ────────────────────────────────────────────────────────────────────
test:
	python -m pytest tests/ -v

# ── Preprocessing ────────────────────────────────────────────────────────────
preprocess:
	@echo "=== Step 1/7: Normalize FPS ==="
	python -m preprocessing.normalize_fps --input-dir $(DATA_DIR) --output-dir data/normalized
	@echo "=== Step 2/7: Sync Videos ==="
	python -m preprocessing.sync_videos --input-dir data/normalized --output-dir data/synced
	@echo "=== Step 3/7: Stabilize ==="
	python -m preprocessing.stabilize --input-dir data/synced --output-dir data/stabilized
	@echo "=== Step 4/7: Normalize Intrinsics ==="
	python -m preprocessing.normalize_intrinsics --input-dir data/stabilized --output-dir data/undistorted
	@echo "=== Step 5/7: COLMAP ==="
	python -m preprocessing.run_colmap --input-dir data/undistorted --output-dir data/colmap
	@echo "=== Step 6/7: Extract Optical Flow ==="
	python -m preprocessing.extract_flow --input-dir data/undistorted --output-dir data/flow
	@echo "=== Step 7/7: Extract Depth ==="
	python -m preprocessing.extract_depth --input-dir data/undistorted --output-dir data/depth
	@echo "=== Preprocessing complete ==="

# ── Training ─────────────────────────────────────────────────────────────────
train-gaussian:
	python -m gaussian_splatting.train --config $(CONFIG) --device $(DEVICE)

train-pinn:
	python -m pinn.train --config $(CONFIG) --device $(DEVICE)

train-all:
	@echo "Starting Gaussian training on cuda:0 and PINN training on cuda:1 in parallel..."
	python -m gaussian_splatting.train --config $(CONFIG) --device cuda:0 &
	python -m pinn.train --config $(CONFIG) --device cuda:1 &
	wait
	@echo "Both training jobs completed."

# ── Inference ────────────────────────────────────────────────────────────────
predict:
	python -m integration.forward_predict

render:
	python -m integration.novel_view

# ── Code quality ─────────────────────────────────────────────────────────────
lint:
	black --check --line-length 100 preprocessing/ gaussian_splatting/ pinn/ integration/ tests/ scripts/
	isort --check --profile black --line-length 100 preprocessing/ gaussian_splatting/ pinn/ integration/ tests/ scripts/
	flake8 preprocessing/ gaussian_splatting/ pinn/ integration/ tests/ scripts/

format:
	black --line-length 100 preprocessing/ gaussian_splatting/ pinn/ integration/ tests/ scripts/
	isort --profile black --line-length 100 preprocessing/ gaussian_splatting/ pinn/ integration/ tests/ scripts/

# ── Cleanup ──────────────────────────────────────────────────────────────────
clean:
	rm -rf outputs/checkpoints/ outputs/renders/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned outputs and caches."
