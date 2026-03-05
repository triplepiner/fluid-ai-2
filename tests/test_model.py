#!/usr/bin/env python3
"""Thorough test for gaussian_splatting/model.py."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N = 1000
SH_DEGREE = 3

checks_passed = 0
checks_total = 0


def check(cond: bool, label: str):
    global checks_passed, checks_total
    checks_total += 1
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if cond:
        checks_passed += 1


# =====================================================================
# Step 1: GaussianModel basic test
# =====================================================================
def step1_gaussian_model():
    print("\n=== Step 1: GaussianModel basic test ===")

    from gaussian_splatting.model import GaussianModel

    gm = GaussianModel(sh_degree=SH_DEGREE, num_points=N)
    torch.manual_seed(42)
    xyz = torch.randn(N, 3)
    rgb = torch.rand(N, 3)
    gm.initialize_from_point_cloud(xyz, rgb)

    # --- Parameter shapes ---
    print("  -- Parameter shapes --")
    check(gm._xyz.shape == (N, 3), f"_xyz shape = ({N}, 3) (got {tuple(gm._xyz.shape)})")
    check(
        gm._rotation.shape == (N, 4),
        f"_rotation shape = ({N}, 4) (got {tuple(gm._rotation.shape)})",
    )
    check(
        gm._scaling.shape == (N, 3), f"_scaling shape = ({N}, 3) (got {tuple(gm._scaling.shape)})"
    )
    check(
        gm._opacity.shape == (N, 1), f"_opacity shape = ({N}, 1) (got {tuple(gm._opacity.shape)})"
    )
    check(
        gm._features_dc.shape == (N, 1, 3),
        f"_features_dc shape = ({N}, 1, 3) (got {tuple(gm._features_dc.shape)})",
    )

    num_rest = (SH_DEGREE + 1) ** 2 - 1  # 15
    check(
        gm._features_rest.shape == (N, num_rest, 3),
        f"_features_rest shape = ({N}, {num_rest}, 3) (got {tuple(gm._features_rest.shape)})",
    )

    # --- Quaternion unit norm ---
    print("  -- Quaternion norms --")
    q_norms = gm._rotation.norm(dim=-1)
    max_norm_err = (q_norms - 1.0).abs().max().item()
    check(max_norm_err < 1e-5, f"Quaternions ~unit norm (max err={max_norm_err:.6f})")

    # --- get_scaling: all positive ---
    print("  -- get_scaling --")
    scaling = gm.get_scaling()
    check(scaling.shape == (N, 3), f"get_scaling shape = ({N}, 3)")
    check(scaling.min().item() > 0, f"All scales positive (min={scaling.min().item():.6f})")

    # --- get_opacity: in (0, 1) ---
    print("  -- get_opacity --")
    opacity = gm.get_opacity()
    check(opacity.shape == (N, 1), f"get_opacity shape = ({N}, 1)")
    check(
        opacity.min().item() > 0 and opacity.max().item() < 1,
        f"Opacity in (0,1) (min={opacity.min():.4f}, max={opacity.max():.4f})",
    )

    # --- get_covariance: shape, symmetry, PSD ---
    print("  -- get_covariance --")
    cov = gm.get_covariance()
    check(cov.shape == (N, 3, 3), f"Covariance shape = ({N}, 3, 3)")

    sym_err = (cov - cov.transpose(-1, -2)).abs().max().item()
    check(sym_err < 1e-5, f"Covariance symmetric (max err={sym_err:.7f})")

    eigvals = torch.linalg.eigvalsh(cov)  # (N, 3), sorted ascending
    min_eigval = eigvals.min().item()
    check(min_eigval >= -1e-5, f"Covariance PSD (min eigenvalue={min_eigval:.7f})")

    # --- num_points property ---
    check(gm.num_points == N, f"num_points = {N}")

    return gm


# =====================================================================
# Step 2: DeformationNetwork test
# =====================================================================
def step2_deformation_network():
    print("\n=== Step 2: DeformationNetwork test ===")

    from gaussian_splatting.model import DeformationNetwork

    dn = DeformationNetwork()

    total_params = sum(p.numel() for p in dn.parameters())
    print(f"  Total parameter count: {total_params:,}")
    check(total_params > 0, f"Has parameters ({total_params:,})")

    M = 500
    canonical_xyz = torch.randn(M, 3)
    output = dn.forward(canonical_xyz, time=0.5)

    # --- Output shapes ---
    print("  -- Output shapes --")
    check(output["delta_xyz"].shape == (M, 3), f"delta_xyz shape = ({M}, 3)")
    check(output["delta_rotation"].shape == (M, 4), f"delta_rotation shape = ({M}, 4)")
    check(output["delta_scaling"].shape == (M, 3), f"delta_scaling shape = ({M}, 3)")
    check(output["delta_opacity"].shape == (M, 1), f"delta_opacity shape = ({M}, 1)")

    # --- Deltas near zero (small-weight init) ---
    print("  -- Deltas near zero --")
    for key in ["delta_xyz", "delta_rotation", "delta_scaling", "delta_opacity"]:
        max_abs = output[key].abs().max().item()
        # With small-weight init the hidden layers can amplify slightly,
        # but deltas should remain modest.  Use 1.0 as a generous bound.
        check(max_abs < 1.0, f"{key} max |delta| = {max_abs:.6f} (< 1.0)")
        print(f"    {key}: max |delta| = {max_abs:.6f}")

    # --- Gradient flow ---
    print("  -- Gradient flow --")
    total = sum(v.sum() for v in output.values())
    total.backward()

    all_have_grad = True
    for name, p in dn.named_parameters():
        if p.grad is None:
            print(f"    WARNING: {name} has no gradient!")
            all_have_grad = False
    check(all_have_grad, "All DeformationNetwork params have gradients")

    return dn


# =====================================================================
# Step 3: DynamicGaussianModel test
# =====================================================================
def step3_dynamic_model():
    print("\n=== Step 3: DynamicGaussianModel test ===")

    from gaussian_splatting.model import DynamicGaussianModel

    model = DynamicGaussianModel(sh_degree=SH_DEGREE, num_points=N)
    torch.manual_seed(42)
    xyz = torch.randn(N, 3)
    rgb = torch.rand(N, 3)
    model.gaussian_model.initialize_from_point_cloud(xyz, rgb)

    # --- Forward at two different times ---
    out_t0 = model.forward(time=0.0)
    out_t1 = model.forward(time=0.5)

    expected_keys = {"xyz", "rotation", "scaling", "opacity", "features", "covariance"}

    print("  -- Output keys --")
    check(set(out_t0.keys()) == expected_keys, f"t=0.0 has all keys: {sorted(expected_keys)}")
    check(set(out_t1.keys()) == expected_keys, f"t=0.5 has all keys: {sorted(expected_keys)}")

    # --- Shapes ---
    print("  -- Shapes at t=0.0 --")
    check(out_t0["xyz"].shape == (N, 3), f"xyz shape = ({N}, 3)")
    check(out_t0["rotation"].shape == (N, 4), f"rotation shape = ({N}, 4)")
    check(out_t0["scaling"].shape == (N, 3), f"scaling shape = ({N}, 3)")
    check(out_t0["opacity"].shape == (N, 1), f"opacity shape = ({N}, 1)")
    num_sh = (SH_DEGREE + 1) ** 2
    check(out_t0["features"].shape == (N, num_sh, 3), f"features shape = ({N}, {num_sh}, 3)")
    check(out_t0["covariance"].shape == (N, 3, 3), f"covariance shape = ({N}, 3, 3)")

    print("  -- Shapes at t=0.5 --")
    check(out_t1["xyz"].shape == (N, 3), f"xyz shape = ({N}, 3)")

    # --- Different times produce different xyz ---
    print("  -- Temporal variation --")
    diff = (out_t0["xyz"] - out_t1["xyz"]).abs().max().item()
    check(diff > 1e-8, f"xyz differs between t=0 and t=0.5 (max diff={diff:.6e})")

    # --- Gradient flow through the full model ---
    print("  -- Gradient flow --")
    model.zero_grad()
    out = model.forward(time=0.5)
    loss = out["xyz"].sum() + out["opacity"].sum()
    loss.backward()

    check(model.gaussian_model._xyz.grad is not None, "Gradient on _xyz")
    check(model.gaussian_model._opacity.grad is not None, "Gradient on _opacity")

    deform_has_grad = any(p.grad is not None for p in model.deformation_network.parameters())
    check(deform_has_grad, "At least one DeformationNetwork param has gradient")

    return model


# =====================================================================
# Step 4: Utility function tests
# =====================================================================
def step4_utilities():
    print("\n=== Step 4: Utility function tests ===")

    from gaussian_splatting.model import build_rotation_matrix, inverse_sigmoid

    # --- build_rotation_matrix: identity quaternion ---
    print("  -- build_rotation_matrix --")
    q_id = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    R_id = build_rotation_matrix(q_id)  # (1, 3, 3)
    err_id = (R_id[0] - torch.eye(3)).abs().max().item()
    check(err_id < 1e-6, f"Identity quat -> identity matrix (max err={err_id:.7f})")

    # --- build_rotation_matrix: 180-deg rotation around x ---
    q_x = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # 180 deg around x
    R_x = build_rotation_matrix(q_x)  # (1, 3, 3)

    # Valid rotation: orthogonal and det = 1
    RtR = R_x[0] @ R_x[0].T
    orth_err = (RtR - torch.eye(3)).abs().max().item()
    check(orth_err < 1e-5, f"q=[0,1,0,0] -> orthogonal (max err={orth_err:.6f})")
    det_val = torch.linalg.det(R_x[0]).item()
    check(abs(det_val - 1.0) < 1e-5, f"det(R) = 1 (got {det_val:.6f})")

    # Expected: diag(1, -1, -1) for 180-deg rotation around x
    expected_R_x = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    err_R_x = (R_x[0] - expected_R_x).abs().max().item()
    check(err_R_x < 1e-5, f"q=[0,1,0,0] -> diag(1,-1,-1) (max err={err_R_x:.6f})")

    # --- Batched ---
    q_batch = torch.randn(100, 4)
    R_batch = build_rotation_matrix(q_batch)
    check(R_batch.shape == (100, 3, 3), f"Batched: shape = (100, 3, 3)")
    RtR_batch = R_batch @ R_batch.transpose(-1, -2)
    eye_batch = torch.eye(3).unsqueeze(0).expand(100, -1, -1)
    orth_batch_err = (RtR_batch - eye_batch).abs().max().item()
    check(orth_batch_err < 1e-4, f"Batched: all orthogonal (max err={orth_batch_err:.6f})")
    dets = torch.linalg.det(R_batch)
    det_err = (dets - 1.0).abs().max().item()
    check(det_err < 1e-4, f"Batched: all det=1 (max err={det_err:.6f})")

    # --- inverse_sigmoid ---
    print("  -- inverse_sigmoid --")
    torch.manual_seed(123)
    x = torch.randn(200)
    roundtrip = inverse_sigmoid(torch.sigmoid(x))
    rt_err = (roundtrip - x).abs().max().item()
    check(rt_err < 1e-4, f"inverse_sigmoid(sigmoid(x)) ~ x (max err={rt_err:.6f})")


# =====================================================================
# Step 5: Device compatibility
# =====================================================================
def step5_device():
    print("\n=== Step 5: Device compatibility ===")

    from gaussian_splatting.model import DynamicGaussianModel

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print(f"  Using device: {device}")

    model = DynamicGaussianModel(sh_degree=SH_DEGREE, num_points=200)
    xyz = torch.randn(200, 3)
    rgb = torch.rand(200, 3)
    model.gaussian_model.initialize_from_point_cloud(xyz, rgb)
    model = model.to(device)

    out = model.forward(time=0.5)

    check(out["xyz"].device.type == device.type, f"xyz on {device.type}")
    check(out["covariance"].device.type == device.type, f"covariance on {device.type}")
    check(out["opacity"].device.type == device.type, f"opacity on {device.type}")
    check(out["xyz"].shape == (200, 3), f"xyz shape correct on {device}")

    # Backward on device
    loss = out["xyz"].sum() + out["opacity"].sum()
    loss.backward()
    check(model.gaussian_model._xyz.grad is not None, f"Gradient exists on {device}")


# =====================================================================
# Main
# =====================================================================
def main():
    print("=" * 70)
    print("  TEST: gaussian_splatting/model.py")
    print("=" * 70)

    step1_gaussian_model()
    step2_deformation_network()
    step3_dynamic_model()
    step4_utilities()
    step5_device()

    print("\n" + "=" * 70)
    print(f"  Results: {checks_passed}/{checks_total} checks passed")
    print("=" * 70)

    if checks_passed == checks_total:
        print(
            "\nTEST PASSED: model.py"
            " — GaussianModel, DeformationNetwork, DynamicGaussianModel"
            " all verified, gradients flow correctly"
        )
    else:
        print(f"\nTEST FAILED: {checks_total - checks_passed} check(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
