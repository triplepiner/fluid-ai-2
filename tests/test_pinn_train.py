#!/usr/bin/env python3
"""Thorough integration test for pinn/train.py.

Tests:
  1. YAML config loading
  2. 500-iteration training run with convergence verification
  3. Logging output (visualization PNGs)
  4. Checkpointing (save, load, forward pass, residual comparison)
  5. Resume (correct iteration, no loss spike)
  6. Field evaluation (spatial structure, no NaN)
"""

from __future__ import annotations

import sys
import tempfile
import time as time_mod
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinn.boundary import BoundaryConditionSet
from pinn.model import FluidDomain, FluidPINN, PINNConfig, count_parameters
from pinn.navier_stokes import compute_derivatives, navier_stokes_residual, physics_loss
from pinn.train import (
    ConvergenceMonitor,
    PINNTrainConfig,
    build_config,
    load_checkpoint,
    train,
)

checks_passed = 0
checks_total = 0


def check(cond: bool, label: str) -> None:
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if cond:
        checks_passed += 1


def main():
    print("=" * 70)
    print("PINN Training — Integration Test (500 iterations)")
    print("=" * 70)

    # ==================================================================
    # 1. CREATE & VERIFY TEST CONFIG
    # ==================================================================
    print("\n--- Step 1: Test config ---")

    config = PINNTrainConfig(
        num_epochs=500,
        num_collocation=500,
        num_boundary=100,
        learning_rate=1e-3,
        hidden_dim=64,
        num_layers=4,
        activation="siren",
        viscosity=1e-3,
        gravity=(0.0, -9.81, 0.0),
        lambda_momentum=1.0,
        lambda_continuity=1.0,
        lambda_boundary=1.0,
        lambda_physics=1.0,
        scenario="rising_smoke",
        x_range=(-1.0, 1.0),
        y_range=(-1.0, 1.0),
        z_range=(-1.0, 1.0),
        t_range=(0.0, 1.0),
        log_interval=100,
        checkpoint_interval=250,
        adaptive_collocation=False,  # skip for speed
        plateau_window=400,
        plateau_threshold=1e-6,
        early_stopping_threshold=1e-10,  # don't trigger
    )
    check(config.num_epochs == 500, "Config: num_epochs=500")
    check(config.hidden_dim == 64, "Config: hidden_dim=64")
    check(config.scenario == "rising_smoke", "Config: scenario=rising_smoke")

    # Also verify YAML config loading works
    yaml_config_path = Path(__file__).resolve().parent.parent / "configs" / "test_pinn.yaml"
    if yaml_config_path.exists():
        try:
            from omegaconf import OmegaConf

            yaml_cfg = OmegaConf.load(str(yaml_config_path))
            cfg_from_yaml = build_config(yaml_cfg)
            check(cfg_from_yaml.num_epochs == 500, "YAML config: num_epochs=500")
            check(cfg_from_yaml.hidden_dim == 64, "YAML config: hidden_dim=64")
            check(cfg_from_yaml.num_collocation == 500, "YAML config: num_collocation=500")
            check(cfg_from_yaml.checkpoint_interval == 250, "YAML config: save_interval=250")
            check(cfg_from_yaml.log_interval == 100, "YAML config: log_interval=100")
            check(abs(cfg_from_yaml.learning_rate - 1e-3) < 1e-8, "YAML config: learning_rate=1e-3")
            check(cfg_from_yaml.lambda_boundary == 1.0, "YAML config: lambda_boundary=1.0")
        except ImportError:
            print("  [SKIP] OmegaConf not installed, skipping YAML config test")
    else:
        print(f"  [SKIP] {yaml_config_path} not found")

    # ==================================================================
    # 2. RUN TRAINING FOR 500 ITERATIONS
    # ==================================================================
    print("\n--- Step 2: Training run (500 iterations) ---")

    with tempfile.TemporaryDirectory() as tmpdir:
        t0 = time_mod.time()
        trained_model = train(
            config=config,
            device_str="cpu",
            use_wandb=False,
            output_dir=tmpdir,
        )
        elapsed = time_mod.time() - t0
        print(f"\n  Training completed in {elapsed:.1f}s")
        check(trained_model is not None, "train() returned a model")
        check(isinstance(trained_model, FluidPINN), "Returned model is FluidPINN")
        check(elapsed < 180, f"Completed under 3 minutes ({elapsed:.1f}s)")

        # ==============================================================
        # 3. VERIFY CONVERGENCE
        # ==============================================================
        print("\n--- Step 3: Verify convergence ---")

        # Load final checkpoint to get loss history
        final_ckpt = Path(tmpdir) / "checkpoints" / "final.pt"
        check(final_ckpt.exists(), "Final checkpoint exists")

        ckpt_data = torch.load(final_ckpt, map_location="cpu", weights_only=False)
        loss_history = ckpt_data.get("loss_history", [])
        check(len(loss_history) == 500, f"Loss history has 500 entries (got {len(loss_history)})")

        if len(loss_history) >= 500:
            loss_iter_0 = loss_history[0]
            loss_iter_499 = loss_history[-1]
            ratio = loss_iter_499 / loss_iter_0 if loss_iter_0 > 0 else float("inf")
            print(f"  Loss at iteration 0:   {loss_iter_0:.6f}")
            print(f"  Loss at iteration 499: {loss_iter_499:.6f}")
            print(f"  Ratio (loss_500 / loss_1): {ratio:.4f}")
            check(ratio < 0.5, f"Loss decreased by >50% (ratio={ratio:.4f} < 0.5)")

            # Check loss is monotonically trending down (rolling average)
            window = 50
            early_avg = sum(loss_history[:window]) / window
            late_avg = sum(loss_history[-window:]) / window
            check(late_avg < early_avg, f"Rolling avg decreased: {early_avg:.4f} -> {late_avg:.4f}")

        # Evaluate individual loss components at the trained model
        domain = FluidDomain(
            x_range=config.x_range,
            y_range=config.y_range,
            z_range=config.z_range,
            t_range=config.t_range,
        )
        x_eval, t_eval = domain.sample_interior(500)
        derivs = compute_derivatives(trained_model, x_eval, t_eval)
        residuals = navier_stokes_residual(derivs, nu=config.viscosity, gravity=config.gravity)
        phys_l, phys_details = physics_loss(residuals)

        momentum_loss = (
            phys_details["momentum_x"].item()
            + phys_details["momentum_y"].item()
            + phys_details["momentum_z"].item()
        )
        continuity_loss = phys_details["continuity"].item()

        bc_set = BoundaryConditionSet(domain, scenario="rising_smoke")
        bc_loss_val, bc_details = bc_set.compute_total_boundary_loss(
            trained_model,
            n_points_per_boundary=100,
        )

        print(f"  Momentum loss (sum x+y+z): {momentum_loss:.6f}")
        print(f"  Continuity loss:           {continuity_loss:.6f}")
        print(f"  Boundary loss:             {bc_loss_val.item():.6f}")
        print(f"  BC components: {', '.join(f'{k}={v.item():.4e}' for k, v in bc_details.items())}")

        check(torch.isfinite(phys_l), "Physics loss is finite")
        check(torch.isfinite(bc_loss_val), "Boundary loss is finite")
        # Continuity typically converges faster because it's a simpler constraint
        # (just divergence=0, no nonlinear convection or pressure gradient)
        check(
            continuity_loss < momentum_loss,
            f"Continuity ({continuity_loss:.4e}) < momentum ({momentum_loss:.4e}) — typical PINN behavior",
        )

        # ==============================================================
        # 4. VERIFY LOGGING
        # ==============================================================
        print("\n--- Step 4: Verify logging ---")

        viz_dir = Path(tmpdir) / "visualizations"
        if viz_dir.exists():
            pngs = list(viz_dir.glob("*.png"))
            print(f"  Found {len(pngs)} visualization PNGs")
            # log_interval=100, visualization saved at iterations 100, 200, 300, 400
            # (iteration 0 skips viz: `if iteration > 0`)
            check(len(pngs) >= 4, f"At least 4 visualization PNGs saved (got {len(pngs)})")

            # Check for velocity, pressure, density plots
            vel_pngs = [p for p in pngs if "velocity" in p.name]
            pres_pngs = [p for p in pngs if "pressure" in p.name]
            dens_pngs = [p for p in pngs if "density" in p.name]
            check(len(vel_pngs) > 0, f"Velocity PNGs exist ({len(vel_pngs)} found)")
            check(len(pres_pngs) > 0, f"Pressure PNGs exist ({len(pres_pngs)} found)")
            check(len(dens_pngs) > 0, f"Density PNGs exist ({len(dens_pngs)} found)")
        else:
            print("  [SKIP] No visualization directory (matplotlib may not be available)")

        # ==============================================================
        # 5. VERIFY CHECKPOINTING
        # ==============================================================
        print("\n--- Step 5: Verify checkpointing ---")

        ckpt_dir = Path(tmpdir) / "checkpoints"
        ckpt_250 = ckpt_dir / "ckpt_000250.pt"
        ckpt_500 = ckpt_dir / "ckpt_000500.pt"

        check(ckpt_250.exists(), "Checkpoint at iteration 250 exists")
        check(ckpt_500.exists(), "Checkpoint at iteration 500 exists")

        # Load checkpoint into a fresh model
        model_loaded = FluidPINN(
            PINNConfig(
                activation="siren",
                hidden_dim=64,
                num_layers=4,
            )
        )
        opt_loaded = torch.optim.Adam(model_loaded.parameters(), lr=1e-3)
        sched_loaded = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt_loaded,
            T_0=5000,
            T_mult=2,
        )
        loaded_iter, loaded_best, loaded_mon, loaded_aw = load_checkpoint(
            final_ckpt,
            model_loaded,
            opt_loaded,
            sched_loaded,
            torch.device("cpu"),
        )
        check(loaded_iter == 500, f"Loaded iteration == 500 (got {loaded_iter})")
        check(loaded_best < float("inf"), f"Loaded best_loss finite ({loaded_best:.6f})")

        # Forward pass with loaded model — verify shapes
        x_test = torch.randn(50, 3)
        t_test = torch.rand(50, 1)
        with torch.no_grad():
            out_loaded = model_loaded(x_test, t_test)
        check(out_loaded["velocity"].shape == (50, 3), "Loaded model velocity shape (50,3)")
        check(out_loaded["pressure"].shape == (50, 1), "Loaded model pressure shape (50,1)")
        check(out_loaded["density"].shape == (50, 1), "Loaded model density shape (50,1)")

        # Verify loaded model matches trained model
        with torch.no_grad():
            out_trained = trained_model(x_test, t_test)
            vel_match = torch.allclose(
                out_trained["velocity"],
                out_loaded["velocity"],
                atol=1e-5,
            )
        check(vel_match, "Loaded model output matches trained model")

        # Compare physics residual: trained model vs fresh model
        model_fresh = FluidPINN(
            PINNConfig(
                activation="siren",
                hidden_dim=64,
                num_layers=4,
            )
        )
        x_res, t_res = domain.sample_interior(300)

        derivs_trained = compute_derivatives(
            trained_model,
            x_res.clone().requires_grad_(True),
            t_res.clone().requires_grad_(True),
        )
        res_trained = navier_stokes_residual(derivs_trained, nu=1e-3, gravity=(0, -9.81, 0))
        pl_trained, _ = physics_loss(res_trained)

        derivs_fresh = compute_derivatives(
            model_fresh,
            x_res.clone().requires_grad_(True),
            t_res.clone().requires_grad_(True),
        )
        res_fresh = navier_stokes_residual(derivs_fresh, nu=1e-3, gravity=(0, -9.81, 0))
        pl_fresh, _ = physics_loss(res_fresh)

        print(
            f"  Physics residual — trained: {pl_trained.item():.4f}, fresh: {pl_fresh.item():.4f}"
        )
        check(
            pl_trained.item() < pl_fresh.item(),
            f"Trained residual ({pl_trained.item():.4f}) < fresh residual ({pl_fresh.item():.4f})",
        )

        # ==============================================================
        # 6. VERIFY RESUME
        # ==============================================================
        print("\n--- Step 6: Verify resume ---")

        # Get loss at iteration 249 (just before checkpoint 250) from original run
        loss_at_250 = loss_history[249] if len(loss_history) > 249 else float("inf")
        print(f"  Loss at iteration 249 (original run): {loss_at_250:.6f}")

        # Resume from checkpoint 250, train 50 more iterations (to 300)
        resume_config = PINNTrainConfig(
            num_epochs=300,  # will run iterations 250..299
            num_collocation=500,
            num_boundary=100,
            learning_rate=1e-3,
            hidden_dim=64,
            num_layers=4,
            activation="siren",
            viscosity=1e-3,
            gravity=(0.0, -9.81, 0.0),
            lambda_momentum=1.0,
            lambda_continuity=1.0,
            lambda_boundary=1.0,
            lambda_physics=1.0,
            scenario="rising_smoke",
            log_interval=25,
            checkpoint_interval=1000,  # don't checkpoint during resume
            adaptive_collocation=False,
            early_stopping_threshold=1e-10,
        )

        with tempfile.TemporaryDirectory() as resume_dir:
            resumed_model = train(
                config=resume_config,
                resume_path=str(ckpt_250),
                device_str="cpu",
                use_wandb=False,
                output_dir=resume_dir,
            )

            # Load the final checkpoint from the resumed run
            resume_final = Path(resume_dir) / "checkpoints" / "final.pt"
            resume_ckpt = torch.load(resume_final, map_location="cpu", weights_only=False)
            resume_losses = resume_ckpt.get("loss_history", [])
            resume_end_iter = resume_ckpt.get("iteration", -1)

            check(
                resume_end_iter == 300, f"Resumed run ends at iteration 300 (got {resume_end_iter})"
            )

            if len(resume_losses) > 0:
                # The resumed monitor starts with the loaded history, then appends
                # Number of losses from original checkpoint
                ckpt_250_data = torch.load(ckpt_250, map_location="cpu", weights_only=False)
                n_original = len(ckpt_250_data.get("loss_history", []))

                if len(resume_losses) > n_original:
                    first_resume_loss = resume_losses[n_original]  # first new loss
                    last_resume_loss = resume_losses[-1]

                    print(f"  First loss after resume: {first_resume_loss:.6f}")
                    print(f"  Last loss of resumed run: {last_resume_loss:.6f}")

                    # No sudden spike: first resumed loss should be within 2x of
                    # loss at checkpoint
                    spike_ratio = (
                        first_resume_loss / loss_at_250 if loss_at_250 > 0 else float("inf")
                    )
                    check(
                        spike_ratio < 2.0,
                        f"No loss spike after resume (ratio={spike_ratio:.2f}, expect <2.0)",
                    )

                    # Loss should not increase overall during resumed training
                    check(
                        last_resume_loss <= first_resume_loss * 1.5,
                        f"Loss stable/decreasing during resumed training "
                        f"({first_resume_loss:.4f} -> {last_resume_loss:.4f})",
                    )
                else:
                    print(
                        f"  [WARN] Resume history shorter than expected "
                        f"({len(resume_losses)} <= {n_original})"
                    )
                    check(False, "Resume added new loss entries")

        # ==============================================================
        # 7. FIELD EVALUATION TEST
        # ==============================================================
        print("\n--- Step 7: Field evaluation ---")

        grid_res = 20
        xs = torch.linspace(-1.0, 1.0, grid_res)
        ys = torch.linspace(-1.0, 1.0, grid_res)
        xx, yy = torch.meshgrid(xs, ys, indexing="ij")
        xyz = torch.stack(
            [
                xx.flatten(),
                yy.flatten(),
                torch.zeros(grid_res * grid_res),  # z=0
            ],
            dim=-1,
        )
        t_field = torch.full((grid_res * grid_res, 1), 0.5)  # t=0.5

        with torch.no_grad():
            out_field = trained_model(xyz, t_field)

        vel = out_field["velocity"]
        pres = out_field["pressure"]
        dens = out_field["density"]

        check(vel.shape == (400, 3), f"Velocity shape (400,3) — got {tuple(vel.shape)}")
        check(pres.shape == (400, 1), f"Pressure shape (400,1) — got {tuple(pres.shape)}")
        check(dens.shape == (400, 1), f"Density shape (400,1) — got {tuple(dens.shape)}")

        check(not torch.isnan(vel).any(), "No NaN in velocity field")
        check(not torch.isnan(pres).any(), "No NaN in pressure field")
        check(not torch.isnan(dens).any(), "No NaN in density field")

        # Spatial structure: std should be non-zero (not perfectly uniform)
        vel_std = vel.std(dim=0)  # std per component across grid
        pres_std = pres.std()
        dens_std = dens.std()

        print(
            f"  Velocity std (per component): u={vel_std[0]:.6f}, v={vel_std[1]:.6f}, w={vel_std[2]:.6f}"
        )
        print(f"  Pressure std: {pres_std:.6f}")
        print(f"  Density std:  {dens_std:.6f}")

        check(vel_std[0] > 1e-6, f"u-velocity has spatial structure (std={vel_std[0]:.2e})")
        check(vel_std[1] > 1e-6, f"v-velocity has spatial structure (std={vel_std[1]:.2e})")
        check(pres_std > 1e-6, f"Pressure has spatial structure (std={pres_std:.2e})")
        check(dens_std > 1e-6, f"Density has spatial structure (std={dens_std:.2e})")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print(f"\n{'=' * 70}")
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print(f"{'=' * 70}")

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: pinn/train.py — loss decreases by at least 50% in 500 "
            "iterations, checkpointing works, resume works, trained field shows "
            "spatial structure"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
