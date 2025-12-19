#!/usr/bin/env python3
"""Measure model size and GPU memory usage.

Usage:
    # Test FP16 baseline
    python test_model_memory.py

    # Test W8A8 quantized
    OPENPI_W8A8_ENABLE=1 python test_model_memory.py

Environment variables:
    CKPT: Path to model checkpoint (default: ~/VLM_REPO/openpi/ckpts/pi05_libero_torch)
    OPENPI_W8A8_ENABLE: Set to 1 to enable W8A8 quantization
"""

import gc
import os
import sys
import time

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")

# Disable torch compile for faster loading
os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
import numpy as np


def get_model_size_bytes(model):
    """Get model size in bytes by summing all parameter and buffer sizes."""
    total_bytes = 0

    # Count parameters
    for name, param in model.named_parameters():
        total_bytes += param.numel() * param.element_size()

    # Count buffers (includes quantized weights stored as buffers)
    for name, buf in model.named_buffers():
        if buf is not None:
            total_bytes += buf.numel() * buf.element_size()

    return total_bytes


def count_quantized_layers(model):
    """Count different types of layers."""
    from openpi.models_pytorch.bitblas_w8a8_layers import BitBLASW8A8Linear

    counts = {
        "Linear": 0,
        "BitBLASW8A8Linear": 0,
        "Other": 0,
    }

    for name, module in model.named_modules():
        if isinstance(module, BitBLASW8A8Linear):
            counts["BitBLASW8A8Linear"] += 1
        elif isinstance(module, torch.nn.Linear):
            counts["Linear"] += 1

    return counts


def get_layer_sizes(model):
    """Get size breakdown by layer type."""
    from openpi.models_pytorch.bitblas_w8a8_layers import BitBLASW8A8Linear

    sizes = {
        "Linear": 0,
        "BitBLASW8A8Linear": 0,
        "Embedding": 0,
        "LayerNorm": 0,
        "Other": 0,
    }

    for name, module in model.named_modules():
        module_bytes = 0
        for pname, param in module.named_parameters(recurse=False):
            module_bytes += param.numel() * param.element_size()
        for bname, buf in module.named_buffers(recurse=False):
            if buf is not None:
                module_bytes += buf.numel() * buf.element_size()

        if module_bytes > 0:
            if isinstance(module, BitBLASW8A8Linear):
                sizes["BitBLASW8A8Linear"] += module_bytes
            elif isinstance(module, torch.nn.Linear):
                sizes["Linear"] += module_bytes
            elif isinstance(module, torch.nn.Embedding):
                sizes["Embedding"] += module_bytes
            elif "LayerNorm" in type(module).__name__ or "RMSNorm" in type(module).__name__:
                sizes["LayerNorm"] += module_bytes

    return sizes


def format_bytes(b):
    """Format bytes as human readable string."""
    if b >= 1024 ** 3:
        return f"{b / (1024 ** 3):.2f} GB"
    elif b >= 1024 ** 2:
        return f"{b / (1024 ** 2):.2f} MB"
    elif b >= 1024:
        return f"{b / 1024:.2f} KB"
    else:
        return f"{b} B"


def main():
    # Get settings
    policy_dir = os.environ.get("CKPT", os.path.expanduser("~/VLM_REPO/openpi/ckpts/pi05_libero_torch"))
    policy_config = "pi05_libero"
    w8a8_enabled = os.environ.get("OPENPI_W8A8_ENABLE", "0") == "1"

    mode_str = "W8A8" if w8a8_enabled else "FP16"

    print("=" * 70)
    print(f"Model Memory Analysis: {mode_str} Mode")
    print("=" * 70)
    print(f"\nPolicy config: {policy_config}")
    print(f"Policy dir: {policy_dir}")
    print(f"W8A8 enabled: {w8a8_enabled}")

    # Clear GPU memory
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Record initial memory
    init_alloc = torch.cuda.memory_allocated()
    init_reserved = torch.cuda.memory_reserved()

    print(f"\nInitial GPU memory: {format_bytes(init_alloc)} allocated, {format_bytes(init_reserved)} reserved")

    # Load model
    print(f"\nLoading model...")
    load_start = time.time()

    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    policy_obj = _policy_config.create_trained_policy(
        _config.get_config(policy_config),
        policy_dir,
        default_prompt=None,
    )

    load_time = time.time() - load_start
    print(f"Model loaded in {load_time:.1f}s")

    # Get model
    model = getattr(policy_obj, "_model", None)
    if model is None:
        print("Error: Could not access model")
        return

    # Record post-load memory
    torch.cuda.synchronize()
    post_alloc = torch.cuda.memory_allocated()
    post_reserved = torch.cuda.memory_reserved()

    print(f"Post-load GPU memory: {format_bytes(post_alloc)} allocated, {format_bytes(post_reserved)} reserved")
    print(f"Model GPU footprint: {format_bytes(post_alloc - init_alloc)}")

    # Get model size
    model_size = get_model_size_bytes(model)
    print(f"\nModel size (weights + buffers): {format_bytes(model_size)}")

    # Count layers
    try:
        counts = count_quantized_layers(model)
        print(f"\nLayer counts:")
        print(f"  Linear (FP16):        {counts['Linear']}")
        print(f"  BitBLASW8A8Linear:    {counts['BitBLASW8A8Linear']}")
    except Exception as e:
        print(f"Could not count layers: {e}")

    # Get layer sizes
    try:
        sizes = get_layer_sizes(model)
        print(f"\nSize by layer type:")
        for layer_type, size in sorted(sizes.items(), key=lambda x: -x[1]):
            if size > 0:
                print(f"  {layer_type:<25} {format_bytes(size)}")
    except Exception as e:
        print(f"Could not get layer sizes: {e}")

    # Run inference to measure peak memory
    print(f"\nRunning inference to measure peak memory...")

    from openpi_client import local_policy as _local_policy
    client = _local_policy.LocalPolicy(policy_obj)

    # Create dummy input
    dummy_input = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.randn(9).astype(np.float32),
        "prompt": "pick up the red cube",
    }

    # Warmup
    for _ in range(3):
        _ = client.infer(dummy_input)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    # Benchmark inference
    n_iter = 10
    infer_times = []

    for _ in range(n_iter):
        start = time.perf_counter()
        _ = client.infer(dummy_input)
        torch.cuda.synchronize()
        infer_times.append((time.perf_counter() - start) * 1000)

    peak_alloc = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    avg_infer_ms = sum(infer_times) / len(infer_times)
    throughput = 1000.0 / avg_infer_ms

    # Print summary
    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {mode_str} Mode")
    print(f"{'=' * 70}")
    print(f"\nModel:")
    print(f"  Size (weights + buffers):  {format_bytes(model_size)}")
    print(f"  GPU footprint:             {format_bytes(post_alloc - init_alloc)}")

    print(f"\nGPU Memory:")
    print(f"  After model load:          {format_bytes(post_alloc)} allocated")
    print(f"  Peak during inference:     {format_bytes(peak_alloc)} allocated")
    print(f"  Peak reserved:             {format_bytes(peak_reserved)}")

    print(f"\nInference:")
    print(f"  Average time:              {avg_infer_ms:.1f} ms")
    print(f"  Throughput:                {throughput:.2f} inferences/s")

    print(f"\n{'=' * 70}")

    # Return results for comparison
    return {
        "mode": mode_str,
        "model_size_bytes": model_size,
        "gpu_footprint_bytes": post_alloc - init_alloc,
        "peak_alloc_bytes": peak_alloc,
        "peak_reserved_bytes": peak_reserved,
        "avg_infer_ms": avg_infer_ms,
        "throughput": throughput,
    }


if __name__ == "__main__":
    main()
