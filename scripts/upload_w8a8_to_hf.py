#!/usr/bin/env python3
"""Upload W8A8 quantized model to HuggingFace.

Usage:
    python scripts/upload_w8a8_to_hf.py
"""

import os
import sys

# Setup environment
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OPENPI_W8A8_ENABLE"] = "1"
os.environ["OPENPI_W8A8_ENABLE_TUNING"] = "0"
os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import logging
logging.basicConfig(level=logging.INFO)

def main():
    print("=" * 70)
    print("Upload W8A8 Model to HuggingFace")
    print("=" * 70)

    # Step 1: Load model with W8A8 quantization
    print("\n[Step 1] Loading model with W8A8 quantization...")
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    train_config = _config.get_config("pi05_libero")
    checkpoint_dir = os.path.expanduser("~/VLM_REPO/openpi/ckpts/pi05_libero_torch")

    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        default_prompt=None,
    )
    print("Model loaded with W8A8 quantization!")

    # Step 2: Save complete model to local path (including vision encoder + assets)
    print("\n[Step 2] Saving complete W8A8 model to local path...")
    from openpi.models_pytorch.bitblas_w8a8_layers import save_w8a8_to_hf

    save_path = "./pi05-libero-w8a8"
    repo_id = "fatdove/pi05-libero-w8a8"

    save_w8a8_to_hf(
        policy,
        save_path=save_path,
        checkpoint_dir=checkpoint_dir,  # For copying assets
        repo_id=repo_id,
        push_to_hub=True,
    )

    print("\n" + "=" * 70)
    print("Upload Complete!")
    print("=" * 70)
    print(f"\nHuggingFace repo: https://huggingface.co/{repo_id}")
    print(f"\nUsers can now load with:")
    print(f"  from openpi.models_pytorch.bitblas_w8a8_layers import load_w8a8_policy")
    print(f"  policy = load_w8a8_policy('{repo_id}')")


if __name__ == "__main__":
    main()
