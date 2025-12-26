#!/usr/bin/env python3
"""Compare BitBLAS true INT8 computation vs Fake W8A8 (FP16 simulation).

Uses REAL weights from Pi0.5 model and REAL activations captured from Libero.

Quantization strategy (same for both):
- Weight: per-channel absmax, scale = max(|W[i,:]|) / 127
- Activation: per-tensor dynamic absmax, scale = max(|x|) / 127
"""

import os
import sys

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn as nn
import torch.nn.functional as F


def fake_w8a8_forward_int32_accum(x, W_fp16, weight_scale):
    """Fake W8A8 with INT32 accumulator simulation.

    This more closely matches BitBLAS behavior by using float32 for accumulation.
    """
    # Step 1: Quantize activation
    act_absmax = x.abs().max()
    act_scale = act_absmax / 127.0
    x_q = (x / act_scale).round().clamp(-127, 127)

    # Step 2: Quantize weight
    W_q = (W_fp16 / weight_scale[:, None]).round().clamp(-127, 127)

    # Step 3: Matmul with FP32 accumulator (simulating INT32)
    y = F.linear(x_q.float(), W_q.float())

    # Step 4: Dequantize
    y = y * (act_scale.float() * weight_scale.float())

    return y.half()


def test_bitblas_vs_fake_real():
    """Test MSE between BitBLAS and Fake W8A8 using REAL W and a from Libero."""
    print("=" * 70)
    print("BitBLAS W8A8 vs Fake W8A8 MSE Comparison")
    print("Using REAL weights from Pi0.5 and REAL activations from Libero")
    print("=" * 70)

    device = torch.device("cuda")

    # =========================================================================
    # Step 1: Load Pi0.5 model
    # =========================================================================
    print("\n[1] Loading Pi0.5 model...")

    from openpi.training import config as _config
    from openpi.policies import policy_config as _policy_config

    config = _config.get_config("pi05_libero")
    checkpoint_dir = "/home/jz97/VLM_REPO/openpi/ckpts/pi05_libero_torch"

    policy = _policy_config.create_trained_policy(config, checkpoint_dir)
    model = policy._model

    print(f"    Model loaded: {type(model).__name__}")

    # =========================================================================
    # Step 2: Select a layer and get real W
    # =========================================================================
    LAYER_NAME = "paligemma.language_model.model.layers.0.mlp.gate_proj"

    print(f"\n[2] Extracting layer: {LAYER_NAME}")

    parts = LAYER_NAME.split(".")
    layer = model
    for part in parts:
        layer = getattr(layer, part)

    W_fp16 = layer.weight.data.clone()
    in_features = layer.in_features
    out_features = layer.out_features

    print(f"    Shape: ({out_features}, {in_features})")
    print(f"    W range: [{W_fp16.min().item():.4f}, {W_fp16.max().item():.4f}]")

    # =========================================================================
    # Step 3: Capture real activation from Libero
    # =========================================================================
    print("\n[3] Capturing real activation from Libero...")

    captured_input = None

    def capture_hook(module, input, output):
        nonlocal captured_input
        captured_input = input[0].detach().clone()

    hook = layer.register_forward_hook(capture_hook)

    # Create Libero environment and get observation
    from examples.libero.main import make_libero_env

    env = make_libero_env(
        task_suite_name="libero_spatial",
        task_id=0,
        resolution=224,
    )

    obs, _ = env.reset()
    print(f"    Got observation from Libero")

    # Run inference to capture activation
    sample = {
        "observation/image": obs["agentview_rgb"],
        "observation/wrist_image": obs["robot0_eye_in_hand_rgb"],
        "observation/state": obs["robot0_proprio"],
        "prompt": "pick up the object",
    }

    with torch.no_grad():
        _ = policy.infer(sample)

    hook.remove()
    env.close()

    if captured_input is None:
        print("    ERROR: Failed to capture activation!")
        return

    x = captured_input.to(device)
    print(f"    Captured x: shape={x.shape}, dtype={x.dtype}")
    print(f"    x range: [{x.min().item():.4f}, {x.max().item():.4f}]")

    # Flatten to 2D
    x_2d = x.view(-1, in_features)
    batch_size = x_2d.shape[0]
    print(f"    Reshaped to: ({batch_size}, {in_features})")

    # =========================================================================
    # Step 4: Compute weight scale
    # =========================================================================
    print("\n[4] Computing weight scale...")

    weight_absmax = W_fp16.float().abs().max(dim=1)[0]
    weight_scale = (weight_absmax / 127.0).clamp(min=1e-8).to(device)

    print(f"    Weight scale range: [{weight_scale.min().item():.6f}, {weight_scale.max().item():.6f}]")

    # =========================================================================
    # Step 5: Create BitBLAS W8A8 layer
    # =========================================================================
    print("\n[5] Creating BitBLAS W8A8 layer...")

    from openpi.models_pytorch.bitblas_w8a8_layers import BitBLASW8A8Linear

    # Create a temporary nn.Linear to load from
    temp_linear = nn.Linear(in_features, out_features, bias=False).to(device).half()
    temp_linear.weight.data = W_fp16.to(device)

    bitblas_layer = BitBLASW8A8Linear(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        name=LAYER_NAME,
        enable_tuning=False,
        opt_M=[1, batch_size, 16, 32],
    )
    bitblas_layer.load_from_linear(temp_linear)
    bitblas_layer = bitblas_layer.to(device)

    print(f"    BitBLAS layer created")

    # =========================================================================
    # Step 6: Compute outputs and compare
    # =========================================================================
    print("\n[6] Computing outputs...")

    with torch.no_grad():
        # BitBLAS output (true INT8)
        y_bitblas = bitblas_layer(x_2d)

        # Fake W8A8 with INT32 accumulator simulation
        y_fake_int32 = fake_w8a8_forward_int32_accum(x_2d, W_fp16.to(device), weight_scale)

        # FP16 baseline (no quantization)
        y_fp16 = F.linear(x_2d, W_fp16.to(device))

    print(f"    y_bitblas range: [{y_bitblas.min().item():.4f}, {y_bitblas.max().item():.4f}]")
    print(f"    y_fake range: [{y_fake_int32.min().item():.4f}, {y_fake_int32.max().item():.4f}]")
    print(f"    y_fp16 range: [{y_fp16.min().item():.4f}, {y_fp16.max().item():.4f}]")

    # =========================================================================
    # Step 7: Compare results
    # =========================================================================
    print("\n[7] Comparing results...")
    print("-" * 70)

    # BitBLAS vs Fake W8A8 (core comparison)
    mse_bitblas_vs_fake = ((y_bitblas.float() - y_fake_int32.float()) ** 2).mean().item()
    max_diff_bitblas_vs_fake = (y_bitblas.float() - y_fake_int32.float()).abs().max().item()

    print(f"BitBLAS vs Fake W8A8 (核心比较):")
    print(f"    MSE:      {mse_bitblas_vs_fake:.6e}")
    print(f"    Max diff: {max_diff_bitblas_vs_fake:.6e}")

    # BitBLAS vs FP16 baseline
    mse_bitblas_vs_fp16 = ((y_bitblas.float() - y_fp16.float()) ** 2).mean().item()
    print(f"\nBitBLAS vs FP16 baseline:")
    print(f"    MSE:      {mse_bitblas_vs_fp16:.6e}")

    # Fake W8A8 vs FP16 baseline
    mse_fake_vs_fp16 = ((y_fake_int32.float() - y_fp16.float()) ** 2).mean().item()
    print(f"\nFake W8A8 vs FP16 baseline:")
    print(f"    MSE:      {mse_fake_vs_fp16:.6e}")

    # Relative error
    rel_error = (y_bitblas.float() - y_fake_int32.float()).abs() / (y_fake_int32.float().abs() + 1e-8)
    print(f"\nRelative error (BitBLAS vs Fake):")
    print(f"    Mean: {rel_error.mean().item():.6e}")
    print(f"    Max:  {rel_error.max().item():.6e}")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("Summary:")
    print("=" * 70)
    print(f"Layer: {LAYER_NAME}")
    print(f"Real W from Pi0.5, Real a from Libero")
    print(f"Dimensions: ({batch_size}, {in_features}) x ({out_features}, {in_features})")
    print()

    if mse_bitblas_vs_fake < 1e-10:
        print("✅ SUCCESS: MSE < 1e-10, BitBLAS matches Fake W8A8 exactly!")
    elif mse_bitblas_vs_fake < 1e-5:
        print("✅ SUCCESS: MSE < 1e-5, BitBLAS matches Fake W8A8 closely!")
    else:
        print(f"⚠ WARNING: MSE = {mse_bitblas_vs_fake:.2e}")


if __name__ == "__main__":
    test_bitblas_vs_fake_real()
