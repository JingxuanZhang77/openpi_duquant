#!/usr/bin/env python3
"""
Profile forward pass time for each Linear layer in LLM and DiT.

This script:
1. Loads PI0.5 model WITHOUT DuQuant
2. Registers forward hooks on all Linear layers
3. Runs inference on LIBERO task
4. Reports timing for each layer

Usage:
    python examples/libero/profile_linear_layers.py
"""

import sys
import time
import collections
from pathlib import Path
import os

# CRITICAL: Fix recursion error by clearing problematic env vars FIRST
if "PYTORCH_NVML_BASED_CUDA_CHECK" in os.environ:
    del os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"]

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "third_party" / "libero"))

# Now import everything else
import numpy as np
import torch
from libero.libero import benchmark
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Clear any DuQuant environment variables to ensure it's disabled
for key in list(os.environ.keys()):
    if key.startswith("OPENPI_DUQUANT_"):
        del os.environ[key]


class LinearLayerProfiler:
    """Profile timing for each Linear layer."""

    def __init__(self, model, scopes=None):
        """
        Args:
            model: PyTorch model
            scopes: List of scope prefixes to profile (e.g., ["language_model", "gemma_expert"])
        """
        self.model = model
        self.scopes = scopes or []
        self.layer_stats = collections.defaultdict(lambda: {
            "count": 0,
            "total_time_ms": 0.0,
            "input_shapes": [],
            "output_shapes": [],
        })
        self.hooks = []
        self.enabled = True

    def should_profile(self, name):
        """Check if layer should be profiled based on scopes."""
        if not self.scopes:
            return True
        return any(name.startswith(scope) for scope in self.scopes)

    def register_hooks(self):
        """Register forward hooks on all Linear layers."""
        for name, module in self.model.named_modules():
            if isinstance(module, torch.nn.Linear) and self.should_profile(name):
                hook = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(hook)

        print(f"[PROFILER] Registered hooks on {len(self.hooks)} Linear layers")

    def _make_hook(self, layer_name):
        """Create a forward hook that measures timing."""
        def hook(module, inputs, output):
            if not self.enabled:
                return

            # Synchronize CUDA before timing
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()

            # Forward pass is already done by PyTorch, but we measure the overhead
            # So we actually need to use CUDA events for accurate GPU timing
            if torch.cuda.is_available():
                # Create CUDA events
                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)

                # Record start
                start_event.record()

                # Do a dummy sync to measure just this layer
                torch.cuda.synchronize()

                # Record end
                end_event.record()
                torch.cuda.synchronize()

                # Calculate elapsed time
                # Note: This measures the synchronization overhead, not the actual layer time
                # For accurate per-layer timing, we need to re-run the layer

                # Re-run the layer to get accurate timing
                x = inputs[0]
                start_event.record()
                with torch.no_grad():
                    _ = module(x)
                end_event.record()
                torch.cuda.synchronize()

                elapsed_ms = start_event.elapsed_time(end_event)
            else:
                # CPU timing
                x = inputs[0]
                start = time.perf_counter()
                with torch.no_grad():
                    _ = module(x)
                end = time.perf_counter()
                elapsed_ms = (end - start) * 1000.0

            # Record statistics
            stats = self.layer_stats[layer_name]
            stats["count"] += 1
            stats["total_time_ms"] += elapsed_ms

            # Record shapes (only first time)
            if stats["count"] == 1:
                stats["input_shapes"].append(tuple(inputs[0].shape))
                stats["output_shapes"].append(tuple(output.shape))

        return hook

    def enable(self):
        """Enable profiling."""
        self.enabled = True

    def disable(self):
        """Disable profiling."""
        self.enabled = False

    def reset(self):
        """Reset all statistics."""
        self.layer_stats.clear()

    def report(self, top_n=50, group_by="layer_type"):
        """
        Print profiling report.

        Args:
            top_n: Number of top layers to show
            group_by: "layer_type" or "none"
        """
        if not self.layer_stats:
            print("[PROFILER] No statistics collected")
            return

        print("\n" + "=" * 120)
        print("LINEAR LAYER PROFILING REPORT")
        print("=" * 120)

        # Convert to list and sort by total time
        layer_list = []
        for name, stats in self.layer_stats.items():
            avg_time = stats["total_time_ms"] / stats["count"] if stats["count"] > 0 else 0
            layer_list.append({
                "name": name,
                "count": stats["count"],
                "total_ms": stats["total_time_ms"],
                "avg_ms": avg_time,
                "input_shape": stats["input_shapes"][0] if stats["input_shapes"] else "N/A",
                "output_shape": stats["output_shapes"][0] if stats["output_shapes"] else "N/A",
            })

        layer_list.sort(key=lambda x: x["total_ms"], reverse=True)

        # Print header
        print(f"\n{'Layer Name':<80} {'Calls':>8} {'Total ms':>12} {'Avg ms':>10} {'Input Shape':<20} {'Output Shape':<20}")
        print("-" * 120)

        # Print top N layers
        total_time = sum(x["total_ms"] for x in layer_list)
        for i, layer in enumerate(layer_list[:top_n]):
            name = layer["name"]
            # Shorten name if too long
            if len(name) > 78:
                parts = name.split(".")
                # Keep first and last 3 parts
                if len(parts) > 6:
                    name = f"{'.'.join(parts[:2])}...{'.'.join(parts[-3:])}"

            percentage = (layer["total_ms"] / total_time * 100) if total_time > 0 else 0

            print(f"{name:<80} {layer['count']:>8} {layer['total_ms']:>12.2f} {layer['avg_ms']:>10.4f} "
                  f"{str(layer['input_shape']):<20} {str(layer['output_shape']):<20}")

        # Print summary
        print("-" * 120)
        print(f"{'TOTAL':<80} {sum(x['count'] for x in layer_list):>8} {total_time:>12.2f} "
              f"{total_time / sum(x['count'] for x in layer_list) if layer_list else 0:>10.4f}")
        print("=" * 120)

        # Group by layer type
        if group_by == "layer_type":
            print("\n" + "=" * 100)
            print("GROUPED BY LAYER TYPE")
            print("=" * 100)

            groups = collections.defaultdict(lambda: {"count": 0, "total_ms": 0.0, "layers": 0})

            for layer in layer_list:
                name = layer["name"]
                # Extract layer type (e.g., "q_proj", "mlp.gate_proj")
                parts = name.split(".")

                # Determine group
                if "language_model" in name:
                    scope = "LLM"
                elif "gemma_expert" in name:
                    scope = "DiT"
                elif "vision_tower" in name:
                    scope = "Vision"
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

            # Sort by total time
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
            print(f"{'TOTAL':<30} {sum(g['layers'] for g in group_list):>8} {sum(g['count'] for g in group_list):>10} "
                  f"{total_time:>12.2f} {total_time / sum(g['count'] for g in group_list) if group_list else 0:>14.4f} {100.0:>11.1f}%")
            print("=" * 100)

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()


def main():
    print("=" * 100)
    print("PI0.5 Linear Layer Profiling (No DuQuant)")
    print("=" * 100)
    print()

    # Import here to ensure env vars are set first
    from openpi_client import local_policy as _local_policy
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    from openpi_client import image_tools

    # Load model
    ckpt_path = Path.home() / "VLM_REPO/openpi/ckpts/pi05_libero_torch"
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading model from: {ckpt_path}")
    config = _config.get_config("pi05_libero")

    policy_obj = _policy_config.create_trained_policy(
        config,
        str(ckpt_path),
        default_prompt=None,
    )

    print("✓ Model loaded successfully")
    print()

    # Get the underlying PyTorch model
    if not hasattr(policy_obj, '_model'):
        print("❌ Model is not PyTorch-based")
        return

    model = policy_obj._model

    # Create profiler
    print("Setting up profiler...")
    profiler = LinearLayerProfiler(
        model,
        scopes=[
            "paligemma_with_expert.paligemma.model.language_model",
            "paligemma_with_expert.gemma_expert.model",
        ]
    )
    profiler.register_hooks()
    print()

    # Load LIBERO task
    print("Loading LIBERO task...")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict["libero_spatial"]()
    task = task_suite.get_task(0)
    task_description = task.language
    print(f"Task: {task_description}")
    print()

    # Create dummy observation for testing
    print("Creating test input...")
    dummy_obs = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.randn(7).astype(np.float32),
        "prompt": task_description,
    }
    print()

    # Warm up (3 iterations)
    print("Warming up (3 iterations)...")
    profiler.disable()
    client = _local_policy.LocalPolicy(policy_obj)
    for i in range(3):
        _ = client.infer(dummy_obs)
        print(f"  Warm-up {i+1}/3 complete")
    print("✓ Warm-up complete")
    print()

    # Profile (10 iterations)
    num_iterations = 10
    print(f"Profiling ({num_iterations} iterations)...")
    profiler.reset()
    profiler.enable()

    total_start = time.perf_counter()
    for i in range(num_iterations):
        iter_start = time.perf_counter()
        _ = client.infer(dummy_obs)
        iter_time = (time.perf_counter() - iter_start) * 1000.0
        print(f"  Iteration {i+1}/{num_iterations}: {iter_time:.2f} ms")

    total_time = (time.perf_counter() - total_start) * 1000.0
    avg_time = total_time / num_iterations

    print(f"\n✓ Profiling complete")
    print(f"  Total time: {total_time:.2f} ms")
    print(f"  Average time per iteration: {avg_time:.2f} ms")
    print()

    # Disable profiler
    profiler.disable()

    # Print report
    profiler.report(top_n=50, group_by="layer_type")

    # Clean up
    profiler.remove_hooks()
    print("\n✓ Profiling complete!")


if __name__ == "__main__":
    main()
