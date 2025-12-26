#!/usr/bin/env python3
"""Test downloading and running W8A8 model from HuggingFace.

This script simulates a new user downloading the model and running inference.

Usage:
    python scripts/test_download_w8a8.py
"""

import os
import sys

# Setup environment
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import logging
logging.basicConfig(level=logging.INFO)


def main():
    print("=" * 70)
    print("Download and Test W8A8 Model from HuggingFace")
    print("=" * 70)

    REPO_ID = "fatdove/pi05-libero-w8a8"
    print(f"\nHuggingFace repo: {REPO_ID}")

    # Step 1: Download and load model
    print("\n[Step 1] Downloading and loading W8A8 model from HuggingFace...")
    from openpi.models_pytorch.bitblas_w8a8_layers import load_w8a8_policy

    policy = load_w8a8_policy(
        REPO_ID,
        policy_config_name="pi05_libero",
        enable_tuning=False,
    )
    print(f"Model loaded! W8A8 layers: {policy._w8a8_layer_count}")

    # Step 2: Test inference
    print("\n[Step 2] Testing inference with dummy input...")
    import numpy as np

    dummy_obs = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.randn(8).astype(np.float32),
        "prompt": "pick up the red cube and place it on the blue plate",
    }

    result = policy.infer(dummy_obs)
    print(f"Inference succeeded!")
    print(f"Output keys: {list(result.keys())}")
    if "actions" in result:
        print(f"Actions shape: {result['actions'].shape}")

    print("\n" + "=" * 70)
    print("SUCCESS! Model downloaded and inference works.")
    print("=" * 70)
    print(f"""
Usage for users:

    from openpi.models_pytorch.bitblas_w8a8_layers import load_w8a8_policy

    # Download from HuggingFace and load (one line!)
    policy = load_w8a8_policy("{REPO_ID}")

    # Run inference
    result = policy.infer(observation)
""")


if __name__ == "__main__":
    main()
