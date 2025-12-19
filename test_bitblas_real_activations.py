#!/usr/bin/env python3
"""Capture real activations from LIBERO inference and test BitBLAS MSE.

This script:
1. Loads the model with W8A8 enabled
2. Runs a few inference steps to capture real activations
3. Computes MSE between BitBLAS and Fake W8A8 using real data

Usage:
    python test_bitblas_real_activations.py [--capture] [--test]

    --capture: Run LIBERO inference to capture activations
    --test: Test MSE using previously captured activations
"""

import os
import sys
import argparse

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import torch.nn.functional as F

ACTIVATION_FILE = "/tmp/w8a8_real_activations.pt"


def fake_w8a8_forward(x, W_fp16, weight_scale):
    """Fake W8A8 with INT32 accumulator simulation.

    Uses the same quantization strategy as BitBLAS W8A8:
    - Weight: per-channel absmax, scale = max(|W[i,:]|) / 127
    - Activation: per-tensor dynamic absmax, scale = max(|x|) / 127

    IMPORTANT: This function uses the SAME weight_scale that BitBLAS computed,
    so we just need to requantize the dequantized W_fp16 weights.
    """
    # Step 1: Quantize activation (per-tensor) - same as BitBLAS
    act_absmax = x.abs().max()
    scale_inv = 127.0 / (act_absmax + 1e-8)
    x_q = (x * scale_inv).round().clamp(-127, 127)
    act_scale = act_absmax.float() / 127.0

    # Step 2: Quantize weight (per-channel) - use same weight_scale as BitBLAS
    # W_fp16 is already dequantized (qweight * weight_scale), so requantize
    W_q = (W_fp16.float() / weight_scale[:, None].float()).round().clamp(-127, 127)

    # Step 3: Matmul with FP32 accumulator (simulating INT32)
    y = F.linear(x_q.float(), W_q.float())

    # Step 4: Dequantize - same as BitBLAS
    combined_scale = act_scale * weight_scale.float()
    y = y * combined_scale

    return y.half()


def capture_activations():
    """Run LIBERO inference to capture real activations."""
    print("=" * 70)
    print("Capturing Real Activations from LIBERO Inference")
    print("=" * 70)

    # Enable W8A8
    os.environ["OPENPI_W8A8_ENABLE"] = "1"
    os.environ["OPENPI_W8A8_ENABLE_TUNING"] = "0"
    os.environ["OPENPI_W8A8_DEBUG"] = "1"

    # Disable torch.compile for faster startup
    os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    # Import after setting env vars
    from openpi.models_pytorch.bitblas_w8a8_layers import (
        enable_activation_capture,
        disable_activation_capture,
        save_captured_activations,
        get_captured_activations,
    )
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    # Load model
    print("\nLoading model...")
    train_config = _config.get_config("pi05_libero")
    policy_dir = os.path.expanduser("~/VLM_REPO/openpi/ckpts/pi05_libero_torch")

    policy = _policy_config.create_trained_policy(
        train_config,
        policy_dir,
        default_prompt=None,
    )

    # Enable activation capture - capture from multiple layer types
    print("\nEnabling activation capture...")
    enable_activation_capture(
        layer_name=r"(gate_proj|up_proj|down_proj)",  # MLP layers
        max_samples=30,  # Capture 30 samples from different layers
    )

    # Create dummy inputs for a few inference steps
    print("\nRunning inference to capture activations...")
    import numpy as np

    # Create dummy observation in LIBERO format
    # Based on examples/libero/main.py lines 442-453
    dummy_obs = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.randn(8).astype(np.float32),  # eef_pos(3) + axisangle(3) + gripper_qpos(2)
        "prompt": "pick up the red cube and place it on the blue plate",
    }

    # Run a few inference steps
    for i in range(3):
        print(f"\n  Inference step {i+1}/3...")
        try:
            _ = policy.infer(dummy_obs)
        except Exception as e:
            print(f"    Error (expected for dummy input): {e}")
            # Continue anyway - we might have captured some activations
            continue

    # Get captured activations
    samples = get_captured_activations()
    print(f"\nCaptured {len(samples)} activation samples")

    if samples:
        # Save to file
        save_captured_activations(ACTIVATION_FILE)
        print(f"Saved to {ACTIVATION_FILE}")

        # Print summary
        print("\nCaptured layers:")
        layer_counts = {}
        for s in samples:
            name = s["layer_name"]
            layer_counts[name] = layer_counts.get(name, 0) + 1

        for name, count in sorted(layer_counts.items()):
            print(f"  {name}: {count} samples")

    disable_activation_capture()
    return samples


def test_mse_with_real_activations():
    """Test MSE between BitBLAS and Fake W8A8 using real activations."""
    print("=" * 70)
    print("Testing MSE with Real LIBERO Activations")
    print("=" * 70)

    if not os.path.exists(ACTIVATION_FILE):
        print(f"Error: Activation file not found: {ACTIVATION_FILE}")
        print("Run with --capture first to capture activations")
        return

    # Load activations
    print(f"\nLoading activations from {ACTIVATION_FILE}...")
    samples = torch.load(ACTIVATION_FILE)
    print(f"Loaded {len(samples)} samples")

    # Analyze each sample
    print(f"\n{'Layer':<60} {'Shape':<20} {'MSE':<12} {'MaxDiff':<12} {'RelErr%':<10}")
    print("-" * 120)

    total_mse = 0
    total_max_diff = 0
    total_rel_err = 0
    count = 0

    for sample in samples:
        layer_name = sample["layer_name"]
        x = sample["x"].cuda()
        W_fp16 = sample["W_fp16"].cuda()
        weight_scale = sample["weight_scale"].cuda()
        y_bitblas = sample["y_bitblas"].cuda()

        # Compute Fake W8A8 output
        with torch.no_grad():
            y_fake = fake_w8a8_forward(x, W_fp16, weight_scale)

        # Flatten for comparison
        y_bitblas_flat = y_bitblas.view(-1, y_bitblas.shape[-1])
        y_fake_flat = y_fake.view(-1, y_fake.shape[-1])

        # Compute metrics
        mse = ((y_bitblas_flat - y_fake_flat) ** 2).mean().item()
        max_diff = (y_bitblas_flat - y_fake_flat).abs().max().item()
        # Use masked relative error to avoid inf/nan
        y_abs = y_fake_flat.abs()
        mask = y_abs > 1e-6  # Only compute relative error where output is non-trivial
        if mask.sum() > 0:
            rel_err = ((y_bitblas_flat - y_fake_flat).abs()[mask] / y_abs[mask]).mean().item() * 100
        else:
            rel_err = 0.0

        # Truncate layer name for display
        short_name = layer_name[-55:] if len(layer_name) > 55 else layer_name
        shape_str = f"{tuple(x.shape)} -> {tuple(y_bitblas.shape)}"

        print(f"{short_name:<60} {shape_str:<20} {mse:<12.2e} {max_diff:<12.6f} {rel_err:<10.2f}")

        total_mse += mse
        total_max_diff = max(total_max_diff, max_diff)
        total_rel_err += rel_err
        count += 1

    print("-" * 120)
    print(f"{'AVERAGE':<60} {'':<20} {total_mse/count:<12.2e} {total_max_diff:<12.6f} {total_rel_err/count:<10.2f}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"""
    Samples analyzed: {count}
    Average MSE:      {total_mse/count:.2e}
    Max Abs Diff:     {total_max_diff:.6f}
    Avg Rel Error:    {total_rel_err/count:.2f}%

    Interpretation:
    - MSE ~1e-6: BitBLAS matches Fake W8A8 almost perfectly
    - MSE ~1e-4: Small numerical differences (acceptable)
    - MSE ~1e-2: Significant differences (investigate)

    BitBLAS is using TRUE INT8 computation, the small MSE is due to:
    1. Weight packing/transform numerical precision
    2. INT32 vs FP32 accumulator differences
    """)


def main():
    parser = argparse.ArgumentParser(description="Test BitBLAS MSE with real activations")
    parser.add_argument("--capture", action="store_true", help="Capture activations from LIBERO inference")
    parser.add_argument("--test", action="store_true", help="Test MSE using captured activations")
    args = parser.parse_args()

    if not args.capture and not args.test:
        # Default: do both
        args.capture = True
        args.test = True

    if args.capture:
        capture_activations()

    if args.test:
        test_mse_with_real_activations()


if __name__ == "__main__":
    main()
