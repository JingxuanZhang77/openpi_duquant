#!/usr/bin/env python3
"""
Simple Linear layer benchmarking by monkey-patching torch.nn.Linear.forward
This avoids environment variable issues and directly measures each layer.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "src"))
sys.path.insert(0, str(Path.cwd() / "third_party" / "libero"))

import time
import collections
import numpy as np
import torch
import torch.nn as nn

# Global statistics storage
LAYER_STATS = collections.defaultdict(lambda: {
    "count": 0,
    "total_time_ms": 0.0,
    "shape": None,
})
PROFILING_ENABLED = False

# Save original forward
_original_linear_forward = nn.Linear.forward


def _profiled_linear_forward(self, input):
    """Wrapped Linear.forward with timing."""
    if not PROFILING_ENABLED:
        return _original_linear_forward(self, input)

    # Get layer name from parent (if available)
    layer_name = getattr(self, '_profiler_name', 'unknown')

    # Skip if not in target scopes
    if not any(scope in layer_name for scope in ['language_model', 'gemma_expert']):
        return _original_linear_forward(self, input)

    # Measure time with CUDA events for GPU
    if input.is_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        output = _original_linear_forward(self, input)
        end_event.record()

        torch.cuda.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
    else:
        # CPU timing
        start = time.perf_counter()
        output = _original_linear_forward(self, input)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    # Record stats
    stats = LAYER_STATS[layer_name]
    stats["count"] += 1
    stats["total_time_ms"] += elapsed_ms
    if stats["shape"] is None:
        stats["shape"] = f"{list(input.shape)} -> {list(output.shape)}"

    return output


def patch_linear_layers():
    """Monkey-patch nn.Linear.forward to add profiling."""
    nn.Linear.forward = _profiled_linear_forward
    print("[PROFILER] Patched nn.Linear.forward")


def unpatch_linear_layers():
    """Restore original nn.Linear.forward."""
    nn.Linear.forward = _original_linear_forward
    print("[PROFILER] Restored original nn.Linear.forward")


def name_all_linear_layers(model, prefix=""):
    """Recursively name all Linear layers in the model."""
    count = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._profiler_name = name
            count += 1
    print(f"[PROFILER] Named {count} Linear layers")


def print_report(top_n=50):
    """Print profiling report."""
    if not LAYER_STATS:
        print("[PROFILER] No statistics collected")
        return

    print("\n" + "=" * 140)
    print("LINEAR LAYER PROFILING REPORT")
    print("=" * 140)

    # Convert to list and sort
    layer_list = []
    for name, stats in LAYER_STATS.items():
        avg_ms = stats["total_time_ms"] / stats["count"] if stats["count"] > 0 else 0
        layer_list.append({
            "name": name,
            "count": stats["count"],
            "total_ms": stats["total_time_ms"],
            "avg_ms": avg_ms,
            "shape": stats["shape"],
        })

    layer_list.sort(key=lambda x: x["total_ms"], reverse=True)

    # Print individual layers
    print(f"\n{'Layer Name':<90} {'Calls':>8} {'Total ms':>12} {'Avg ms':>10} {'Shape':<25}")
    print("-" * 140)

    total_time = sum(x["total_ms"] for x in layer_list)
    for layer in layer_list[:top_n]:
        name = layer["name"]
        if len(name) > 88:
            parts = name.split(".")
            name = f"{'.'.join(parts[:2])}...{'.'.join(parts[-2:])}"

        print(f"{name:<90} {layer['count']:>8} {layer['total_ms']:>12.2f} "
              f"{layer['avg_ms']:>10.4f} {layer['shape']:<25}")

    print("-" * 140)
    total_calls = sum(x["count"] for x in layer_list)
    avg_total = total_time / total_calls if total_calls > 0 else 0
    print(f"{'TOTAL':<90} {total_calls:>8} {total_time:>12.2f} {avg_total:>10.4f}")
    print("=" * 140)

    # Group by scope and type
    print("\n" + "=" * 100)
    print("GROUPED BY SCOPE AND LAYER TYPE")
    print("=" * 100)

    groups = collections.defaultdict(lambda: {"count": 0, "total_ms": 0.0, "layers": 0})

    for layer in layer_list:
        name = layer["name"]

        # Determine scope
        if "language_model" in name:
            scope = "LLM"
        elif "gemma_expert" in name:
            scope = "DiT"
        else:
            scope = "Other"

        # Determine layer type
        if "q_proj" in name:
            layer_type = "q_proj"
        elif "k_proj" in name:
            layer_type = "k_proj"
        elif "v_proj" in name:
            layer_type = "v_proj"
        elif "o_proj" in name:
            layer_type = "o_proj"
        elif "gate_proj" in name:
            layer_type = "gate_proj"
        elif "up_proj" in name:
            layer_type = "up_proj"
        elif "down_proj" in name:
            layer_type = "down_proj"
        else:
            layer_type = "other"

        group_key = f"{scope}.{layer_type}"
        groups[group_key]["count"] += layer["count"]
        groups[group_key]["total_ms"] += layer["total_ms"]
        groups[group_key]["layers"] += 1

    # Sort and print groups
    group_list = [{"name": k, **v} for k, v in groups.items()]
    group_list.sort(key=lambda x: x["total_ms"], reverse=True)

    print(f"\n{'Group':<30} {'Layers':>8} {'Calls':>10} {'Total ms':>12} {'Avg ms/call':>14} {'% of total':>12}")
    print("-" * 100)

    for group in group_list:
        avg_ms = group["total_ms"] / group["count"] if group["count"] > 0 else 0
        percentage = (group["total_ms"] / total_time * 100) if total_time > 0 else 0
        print(f"{group['name']:<30} {group['layers']:>8} {group['count']:>10} "
              f"{group['total_ms']:>12.2f} {avg_ms:>14.4f} {percentage:>11.1f}%")

    print("-" * 100)
    print(f"{'TOTAL':<30} {sum(g['layers'] for g in group_list):>8} "
          f"{sum(g['count'] for g in group_list):>10} {total_time:>12.2f} "
          f"{total_time / sum(g['count'] for g in group_list) if group_list else 0:>14.4f} {100.0:>11.1f}%")
    print("=" * 100)


def main():
    print("=" * 100)
    print("Simple Linear Layer Profiling (No DuQuant)")
    print("=" * 100)
    print()

    # CRITICAL: Disable torch.compile to avoid conflicts with monkey-patching
    import os
    os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = "1"
    os.environ["TORCH_COMPILE_DISABLE"] = "1"
    os.environ["TORCHDYNAMO_DISABLE"] = "1"

    # Import heavy modules here
    from openpi_client import local_policy
    from openpi.policies import policy_config
    from openpi.training import config

    # Load model
    ckpt_path = Path.home() / "VLM_REPO/openpi/ckpts/pi05_libero_torch"
    print(f"Loading model from: {ckpt_path}")
    print("torch.compile: DISABLED (for profiling compatibility)")
    print()

    train_config = config.get_config("pi05_libero")
    policy_obj = policy_config.create_trained_policy(
        train_config,
        str(ckpt_path),
        default_prompt=None,
    )

    print("✓ Model loaded")

    # Get model
    if not hasattr(policy_obj, '_model'):
        print("❌ Not a PyTorch model")
        return

    model = policy_obj._model

    # Name all layers
    name_all_linear_layers(model)

    # Patch Linear.forward
    patch_linear_layers()

    # Create client
    client = local_policy.LocalPolicy(policy_obj)

    # Create dummy input
    dummy_obs = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.randn(7).astype(np.float32),
        "prompt": "pick up the object",
    }

    # Run inference (profiling disabled for warmup)
    print("\nWarming up (1 iteration)...")
    _ = client.infer(dummy_obs)
    print("✓ Warmup complete")

    # Enable profiling
    global PROFILING_ENABLED
    PROFILING_ENABLED = True

    # Run profiling
    num_iters = 5
    print(f"\nProfiling ({num_iters} iterations)...")

    total_start = time.perf_counter()
    for i in range(num_iters):
        iter_start = time.perf_counter()
        _ = client.infer(dummy_obs)
        iter_time = (time.perf_counter() - iter_start) * 1000.0
        print(f"  Iteration {i+1}/{num_iters}: {iter_time:.2f} ms")

    total_time = (time.perf_counter() - total_start) * 1000.0
    avg_time = total_time / num_iters

    print(f"\n✓ Profiling complete")
    print(f"  Total time: {total_time:.2f} ms")
    print(f"  Average per iteration: {avg_time:.2f} ms")

    # Disable profiling
    PROFILING_ENABLED = False

    # Print report
    print_report(top_n=50)

    # Unpatch
    unpatch_linear_layers()

    print("\n✓ Done!")


if __name__ == "__main__":
    main()
