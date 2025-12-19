#!/usr/bin/env python3
"""Compare model size and GPU memory between FP16 baseline and W8A8 quantized models.

Usage:
    # Compare FP16 vs W8A8
    python test_memory_comparison.py

    # Only test FP16
    OPENPI_W8A8_ENABLE=0 python test_memory_comparison.py

    # Only test W8A8
    OPENPI_W8A8_ENABLE=1 python test_memory_comparison.py
"""

import gc
import os
import sys

os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch


def get_model_size_mb(model):
    """Get model size in MB by summing all parameter and buffer sizes."""
    total_bytes = 0

    # Count parameters
    for name, param in model.named_parameters():
        total_bytes += param.numel() * param.element_size()

    # Count buffers (includes quantized weights stored as buffers)
    for name, buf in model.named_buffers():
        if buf is not None:
            total_bytes += buf.numel() * buf.element_size()

    return total_bytes / (1024 * 1024)


def get_layer_breakdown(model):
    """Get breakdown of model size by layer type."""
    breakdown = {}

    for name, module in model.named_modules():
        module_bytes = 0

        # Parameters
        for pname, param in module.named_parameters(recurse=False):
            module_bytes += param.numel() * param.element_size()

        # Buffers
        for bname, buf in module.named_buffers(recurse=False):
            if buf is not None:
                module_bytes += buf.numel() * buf.element_size()

        if module_bytes > 0:
            layer_type = type(module).__name__
            if layer_type not in breakdown:
                breakdown[layer_type] = {"count": 0, "bytes": 0}
            breakdown[layer_type]["count"] += 1
            breakdown[layer_type]["bytes"] += module_bytes

    return breakdown


def get_gpu_memory_mb():
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0, 0

    allocated = torch.cuda.memory_allocated() / (1024 * 1024)
    reserved = torch.cuda.memory_reserved() / (1024 * 1024)
    return allocated, reserved


def clear_gpu_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def test_model_memory(enable_w8a8: bool, policy_config: str = "pi05_libero", policy_dir: str = None):
    """Test model memory usage.

    Args:
        enable_w8a8: Whether to enable W8A8 quantization
        policy_config: Policy config name
        policy_dir: Policy checkpoint directory

    Returns:
        dict with memory stats
    """
    # Set environment
    if enable_w8a8:
        os.environ["OPENPI_W8A8_ENABLE"] = "1"
        os.environ["OPENPI_W8A8_ENABLE_TUNING"] = "0"
        os.environ["OPENPI_W8A8_DEBUG"] = "1"
    else:
        os.environ.pop("OPENPI_W8A8_ENABLE", None)
        os.environ.pop("OPENPI_DUQUANT_WBITS_DEFAULT", None)
        os.environ.pop("OPENPI_BITBLAS_ENABLE", None)

    # Disable torch compile for faster loading
    os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    clear_gpu_memory()

    # Record initial memory
    init_alloc, init_reserved = get_gpu_memory_mb()

    # Import and load model
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    mode_str = "W8A8" if enable_w8a8 else "FP16"
    print(f"\n{'=' * 60}")
    print(f"Loading model in {mode_str} mode...")
    print(f"{'=' * 60}")

    policy_obj = _policy_config.create_trained_policy(
        _config.get_config(policy_config),
        policy_dir,
        default_prompt=None,
    )

    # Get model
    model = getattr(policy_obj, "_model", None)
    if model is None:
        print("Warning: Could not access model directly")
        return None

    # Record post-load memory
    post_alloc, post_reserved = get_gpu_memory_mb()

    # Get model size
    model_size_mb = get_model_size_mb(model)

    # Get layer breakdown
    breakdown = get_layer_breakdown(model)

    # Do a forward pass to measure inference memory
    print("Running inference to measure peak memory...")
    import numpy as np
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

    # Measure peak memory during inference
    for _ in range(5):
        _ = client.infer(dummy_input)

    torch.cuda.synchronize()
    peak_alloc = torch.cuda.max_memory_allocated() / (1024 * 1024)
    peak_reserved = torch.cuda.max_memory_reserved() / (1024 * 1024)

    # Print results
    print(f"\n{'=' * 60}")
    print(f"{mode_str} Model Memory Report")
    print(f"{'=' * 60}")
    print(f"\nModel Size (weights + buffers): {model_size_mb:.2f} MB")
    print(f"\nGPU Memory:")
    print(f"  Initial:    {init_alloc:.2f} MB allocated, {init_reserved:.2f} MB reserved")
    print(f"  After load: {post_alloc:.2f} MB allocated, {post_reserved:.2f} MB reserved")
    print(f"  Model load: {post_alloc - init_alloc:.2f} MB")
    print(f"  Peak infer: {peak_alloc:.2f} MB allocated, {peak_reserved:.2f} MB reserved")

    print(f"\nLayer Type Breakdown:")
    print(f"  {'Layer Type':<30} {'Count':>8} {'Size (MB)':>12}")
    print(f"  {'-' * 52}")

    sorted_breakdown = sorted(breakdown.items(), key=lambda x: -x[1]["bytes"])
    for layer_type, info in sorted_breakdown[:15]:
        size_mb = info["bytes"] / (1024 * 1024)
        if size_mb > 0.1:  # Only show layers > 0.1 MB
            print(f"  {layer_type:<30} {info['count']:>8} {size_mb:>12.2f}")

    return {
        "mode": mode_str,
        "model_size_mb": model_size_mb,
        "init_alloc_mb": init_alloc,
        "post_load_alloc_mb": post_alloc,
        "model_load_mb": post_alloc - init_alloc,
        "peak_infer_alloc_mb": peak_alloc,
        "peak_infer_reserved_mb": peak_reserved,
        "breakdown": breakdown,
    }


def main():
    # Get policy directory from environment or default
    policy_dir = os.environ.get("CKPT", os.path.expanduser("~/VLM_REPO/openpi/ckpts/pi05_libero_torch"))
    policy_config = "pi05_libero"

    print("=" * 70)
    print("Model Size and Memory Comparison: FP16 vs W8A8")
    print("=" * 70)
    print(f"\nPolicy config: {policy_config}")
    print(f"Policy dir: {policy_dir}")

    results = {}

    # Test FP16 baseline
    print("\n" + "=" * 70)
    print("PART 1: FP16 Baseline")
    print("=" * 70)
    results["fp16"] = test_model_memory(enable_w8a8=False, policy_config=policy_config, policy_dir=policy_dir)

    # Clear memory before next test
    clear_gpu_memory()

    # Need to reimport modules after changing env vars
    # This is a hack but necessary because the modules cache the env vars
    import importlib
    modules_to_reload = [
        "openpi.models_pytorch.bitblas_w8a8_layers",
        "openpi.policies.policy_config",
    ]
    for mod_name in modules_to_reload:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    # Test W8A8
    print("\n" + "=" * 70)
    print("PART 2: W8A8 Quantized")
    print("=" * 70)
    results["w8a8"] = test_model_memory(enable_w8a8=True, policy_config=policy_config, policy_dir=policy_dir)

    # Print comparison
    if results["fp16"] and results["w8a8"]:
        print("\n" + "=" * 70)
        print("COMPARISON SUMMARY")
        print("=" * 70)

        fp16 = results["fp16"]
        w8a8 = results["w8a8"]

        print(f"\n{'Metric':<35} {'FP16':>12} {'W8A8':>12} {'Savings':>12}")
        print("-" * 73)

        # Model size
        fp16_size = fp16["model_size_mb"]
        w8a8_size = w8a8["model_size_mb"]
        savings = (1 - w8a8_size / fp16_size) * 100 if fp16_size > 0 else 0
        print(f"{'Model Size (MB)':<35} {fp16_size:>12.2f} {w8a8_size:>12.2f} {savings:>11.1f}%")

        # Model load memory
        fp16_load = fp16["model_load_mb"]
        w8a8_load = w8a8["model_load_mb"]
        savings = (1 - w8a8_load / fp16_load) * 100 if fp16_load > 0 else 0
        print(f"{'GPU Memory (model load)':<35} {fp16_load:>12.2f} {w8a8_load:>12.2f} {savings:>11.1f}%")

        # Peak inference memory
        fp16_peak = fp16["peak_infer_alloc_mb"]
        w8a8_peak = w8a8["peak_infer_alloc_mb"]
        savings = (1 - w8a8_peak / fp16_peak) * 100 if fp16_peak > 0 else 0
        print(f"{'Peak Inference Memory (MB)':<35} {fp16_peak:>12.2f} {w8a8_peak:>12.2f} {savings:>11.1f}%")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
