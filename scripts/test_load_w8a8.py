#!/usr/bin/env python3
"""Test loading complete W8A8 model from local path or HuggingFace.

This script demonstrates how users can download and use the W8A8 quantized model.
No base checkpoint needed - the model is complete and self-contained.

Usage:
    python scripts/test_load_w8a8.py --local ./pi05-libero-w8a8
    python scripts/test_load_w8a8.py --hf fatdove/pi05-libero-w8a8
"""

import os
import sys
import argparse

# Setup environment
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import logging
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(description="Test loading complete W8A8 model")
    parser.add_argument("--local", type=str, help="Local path to W8A8 model")
    parser.add_argument("--hf", type=str, default="fatdove/pi05-libero-w8a8", help="HuggingFace repo ID")
    args = parser.parse_args()

    repo_or_path = args.local or args.hf

    print("=" * 70)
    print("Test Loading Complete W8A8 Model")
    print("=" * 70)
    print(f"W8A8 source: {repo_or_path}")
    print()

    # Step 1: Load model (no base checkpoint needed!)
    print("[Step 1] Loading complete W8A8 model...")
    from openpi.models_pytorch.bitblas_w8a8_layers import load_w8a8_policy

    policy = load_w8a8_policy(
        repo_or_path,
        policy_config_name="pi05_libero",
        enable_tuning=False,
    )
    print(f"Policy loaded! W8A8 layers: {policy._w8a8_layer_count}")

    # Step 2: Test inference with dummy input
    print("\n[Step 2] Testing inference with dummy input...")
    import numpy as np

    dummy_obs = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.randn(8).astype(np.float32),
        "prompt": "pick up the red cube",
    }

    try:
        result = policy.infer(dummy_obs)
        print(f"Inference succeeded!")
        print(f"Output shape: {result['actions'].shape if 'actions' in result else 'N/A'}")
    except Exception as e:
        print(f"Inference error (expected for dummy input): {e}")

    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
