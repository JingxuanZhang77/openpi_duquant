#!/usr/bin/env python3
"""
Fast memory calculator - reads layer shapes directly from safetensors metadata.
"""

import re
from pathlib import Path
from safetensors import safe_open


def format_bytes(bytes_val):
    """Format bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_val < 1024:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.2f} TB"


def parse_layer_name(tensor_name):
    """
    Extract layer name from tensor name.
    Example: "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight"
    Returns: "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj"
    """
    if tensor_name.endswith(".weight"):
        return tensor_name[:-7]
    elif tensor_name.endswith(".bias"):
        return tensor_name[:-5]
    return tensor_name


def match_layer(layer_name, scope_prefix, include_regex, exclude_regex):
    """Check if layer matches the given patterns."""
    # Check scope
    if scope_prefix and not layer_name.startswith(scope_prefix):
        return False

    # Check include
    if include_regex:
        inc = re.compile(include_regex)
        if not inc.search(layer_name):
            return False

    # Check exclude
    if exclude_regex:
        exc = re.compile(exclude_regex)
        if exc.search(layer_name):
            return False

    return True


def calculate_duquant_memory_for_layer(out_features, in_features, has_bias=True, weight_bits=4, block_size=16, enable_permute=True, row_rot_mode="restore"):
    """Calculate DuQuant memory for a single layer."""
    weight_params = out_features * in_features
    bias_params = out_features if has_bias else 0

    memory = {}

    # 1. Quantized weights (packed)
    memory["quantized_weights"] = (weight_params * weight_bits) / 8

    # 2. Weight scales (FP16 per output channel)
    memory["weight_scales"] = out_features * 2

    # 3. Input rotation matrices R_in
    n_in_blocks = (in_features + block_size - 1) // block_size
    memory["R_in_blocks"] = n_in_blocks * block_size * block_size * 4

    # 4. Output rotation matrices R_out (optional)
    if row_rot_mode in ("restore", "propagate"):
        n_out_blocks = (out_features + block_size - 1) // block_size
        memory["R_out_blocks"] = n_out_blocks * block_size * block_size * 4
    else:
        memory["R_out_blocks"] = 0

    # 5. Permutation indices (optional)
    if enable_permute:
        memory["permutation"] = in_features * 4
    else:
        memory["permutation"] = 0

    # 6. Bias
    memory["bias"] = bias_params * 2

    memory["total"] = sum(memory.values())
    return memory


def main():
    print("=" * 100)
    print("DuQuant Memory Usage Calculator (Fast Mode)")
    print("=" * 100)
    print()

    # Path to checkpoint
    ckpt_path = Path.home() / "VLM_REPO/openpi/ckpts/pi05_libero_torch/model.safetensors"
    if not ckpt_path.exists():
        print(f"❌ Checkpoint not found: {ckpt_path}")
        return

    print(f"Loading checkpoint metadata: {ckpt_path}")
    print()

    # Read tensor metadata
    layer_shapes = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for key in f.keys():
            if key.endswith(".weight"):
                layer_name = parse_layer_name(key)
                tensor = f.get_tensor(key)
                shape = tensor.shape

                # Linear layers have 2D weight tensors
                if len(shape) == 2:
                    out_features, in_features = shape
                    has_bias = f"{layer_name}.bias" in f.keys()
                    layer_shapes[layer_name] = {
                        "out_features": out_features,
                        "in_features": in_features,
                        "has_bias": has_bias,
                        "weight_params": out_features * in_features,
                        "bias_params": out_features if has_bias else 0,
                    }

    print(f"✓ Found {len(layer_shapes)} Linear layers in checkpoint")
    print()
    total_params_all = sum(info["weight_params"] + info["bias_params"] for info in layer_shapes.values())
    total_original_bytes_all = total_params_all * 2

    # Define quantization scenarios
    scenarios = {
        "Full Model (all linears)": {
            "scope": "",
            "include": r".*",
            "exclude": r"$^",  # match nothing
        },
        "Full LLM + DiT": {
            "scope": "paligemma_with_expert.",
            "include": r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
            "exclude": r"(?:^|\.)(norm|ln|layernorm|emb|embed|vision_tower|vision|multi_modal_projector|lm_head)(?:\.|$)",
        },
        "LLM Only": {
            "scope": "paligemma_with_expert.paligemma.model.language_model.",
            "include": r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
            "exclude": r"(?:^|\.)(norm|ln|layernorm|emb|embed)(?:\.|$)",
        },
        "DiT Only": {
            "scope": "paligemma_with_expert.gemma_expert.model.",
            "include": r".*(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj).*",
            "exclude": r"(?:^|\.)(norm|ln|layernorm|emb|embed)(?:\.|$)",
        },
        "LLM + DiT MLP": {
            "scope": "",
            "include": r".*(language_model\.(.*\.)?(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)|gemma_expert\.model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)).*",
            "exclude": r"(?:^|\.)(vision_tower|vision|multi_modal_projector|norm|ln|layernorm|embed_tokens|lm_head)(?:\.|$)|gemma_expert\..*\.self_attn\.",
        },
        "DiT MLP Only": {
            "scope": "paligemma_with_expert.gemma_expert.model.",
            "include": r".*(mlp\.(gate_proj|up_proj|down_proj)).*",
            "exclude": r"(?:^|\.)(norm|ln|layernorm|emb|embed|q_proj|k_proj|v_proj|o_proj)(?:\.|$)",
        },
    }

    # DuQuant configuration
    duquant_config = {
        "weight_bits": 4,
        "block_size": 16,
        "enable_permute": True,
        "row_rot_mode": "restore",
    }

    print(f"DuQuant Configuration:")
    print(f"  Weight bits: {duquant_config['weight_bits']}")
    print(f"  Block size: {duquant_config['block_size']}")
    print(f"  Permutation: {duquant_config['enable_permute']}")
    print(f"  Row rotation: {duquant_config['row_rot_mode']}")
    print()
    print("=" * 100)
    print()

    # Calculate for each scenario
    for scenario_name, scenario_config in scenarios.items():
        print(f"📊 Scenario: {scenario_name}")
        print("-" * 100)

        # Filter layers
        matched_layers = {}
        for layer_name, info in layer_shapes.items():
            if match_layer(
                layer_name,
                scenario_config["scope"] if scenario_config["scope"] else None,
                scenario_config["include"],
                scenario_config["exclude"]
            ):
                matched_layers[layer_name] = info

        num_layers = len(matched_layers)
        total_params = sum(info["weight_params"] + info["bias_params"] for info in matched_layers.values())

        print(f"  Matched layers: {num_layers}")
        print(f"  Total parameters: {total_params:,}")
        print()

        if num_layers == 0:
            print("  ⚠️  No layers matched!")
            print()
            print("=" * 100)
            print()
            continue

        # Calculate original memory (BF16: 2 bytes per param)
        subset_original_bytes = total_params * 2
        print(f"  Original Memory (subset BF16):")
        print(f"    Total: {format_bytes(subset_original_bytes)} ({subset_original_bytes:,} bytes)")
        print()

        # Calculate DuQuant memory
        duquant_total = {
            "quantized_weights": 0,
            "weight_scales": 0,
            "R_in_blocks": 0,
            "R_out_blocks": 0,
            "permutation": 0,
            "bias": 0,
        }

        for layer_name, info in matched_layers.items():
            layer_memory = calculate_duquant_memory_for_layer(
                info["out_features"],
                info["in_features"],
                info["has_bias"],
                **duquant_config
            )
            for key in duquant_total:
                duquant_total[key] += layer_memory[key]

        subset_quant_bytes = sum(duquant_total.values())

        print(f"  DuQuant Memory (subset W{duquant_config['weight_bits']}):")
        print(f"    Quantized weights: {format_bytes(duquant_total['quantized_weights'])}")
        print(f"    Weight scales:     {format_bytes(duquant_total['weight_scales'])}")
        print(f"    R_in blocks:       {format_bytes(duquant_total['R_in_blocks'])}")
        print(f"    R_out blocks:      {format_bytes(duquant_total['R_out_blocks'])}")
        print(f"    Permutation:       {format_bytes(duquant_total['permutation'])}")
        print(f"    Bias:              {format_bytes(duquant_total['bias'])}")
        print(f"    {'─' * 40}")
        print(f"    Total:             {format_bytes(subset_quant_bytes)} ({subset_quant_bytes:,} bytes)")
        print()

        subset_compression = subset_original_bytes / subset_quant_bytes if subset_quant_bytes > 0 else 0
        remaining_bytes = total_original_bytes_all - subset_original_bytes
        total_quant_bytes = remaining_bytes + subset_quant_bytes

        # Calculate savings relative to full model
        savings_bytes = total_original_bytes_all - total_quant_bytes
        savings_ratio = (savings_bytes / total_original_bytes_all) * 100 if total_original_bytes_all > 0 else 0
        compression_ratio = total_original_bytes_all / total_quant_bytes if total_quant_bytes > 0 else 0

        print(f"  💾 Memory Savings:")
        print(f"    Absolute (subset): {format_bytes(subset_original_bytes - subset_quant_bytes)}")
        print(f"    Relative (subset): {((subset_original_bytes - subset_quant_bytes) / subset_original_bytes * 100) if subset_original_bytes else 0:.2f}%")
        print(f"    Compression ratio (subset): {subset_compression:.2f}x ({format_bytes(subset_original_bytes)} -> {format_bytes(subset_quant_bytes)})")
        print()
        print(f"    Total model (BF16):       {format_bytes(total_original_bytes_all)}")
        print(f"    After quantizing subset:  {format_bytes(total_quant_bytes)}")
        print(f"    Absolute savings:         {format_bytes(savings_bytes)}")
        print(f"    Relative savings:         {savings_ratio:.2f}%")
        print(f"    Overall compression:      {compression_ratio:.2f}x")
        print()

        # Show some example layers
        if num_layers > 0:
            print(f"  Example layers (first 5):")
            for i, (name, info) in enumerate(list(matched_layers.items())[:5]):
                short_name = ".".join(name.split(".")[-3:])
                print(f"    {i+1}. {short_name}")
                print(f"       Shape: [{info['out_features']}, {info['in_features']}]")
                print(f"       Params: {info['weight_params'] + info['bias_params']:,}")

        print()
        print("=" * 100)
        print()

    print("✓ Memory calculation complete!")


if __name__ == "__main__":
    main()
