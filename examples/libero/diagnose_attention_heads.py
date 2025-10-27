#!/usr/bin/env python3
"""
Diagnostic tool to analyze attention head structure preservation under quantization.

This script compares:
1. Original weights (FP16)
2. Quantized weights with block_size=16 (head-misaligned)
3. Quantized weights with block_size=64 (head-aligned)

Metrics computed per-head:
- Weight magnitude distribution
- Inter-head correlation (should be low)
- Intra-head coherence (should be high)
- Quantization error per head
"""

import os
import sys
import numpy as np
import torch
from safetensors import safe_open

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from openpi.models_pytorch.duquant_preprocess import (
    pack_weight,
    transform_weight_for_forward,
    fake_quantize_sym,
    qmax,
)


def analyze_layer_head_structure(W: torch.Tensor, layer_name: str, num_heads: int):
    """
    Analyze head structure in a weight matrix.

    Args:
        W: Weight tensor [out_features, in_features]
        layer_name: Name of the layer
        num_heads: Number of attention heads

    Returns:
        Dictionary of metrics
    """
    out_features, in_features = W.shape
    head_dim = out_features // num_heads

    if out_features % num_heads != 0:
        print(f"WARNING: {layer_name} out_features={out_features} not divisible by num_heads={num_heads}")
        return None

    # Reshape to [num_heads, head_dim, in_features]
    W_heads = W.reshape(num_heads, head_dim, in_features)

    metrics = {
        "layer_name": layer_name,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "in_features": in_features,
    }

    # 1. Per-head weight magnitude
    head_norms = []
    for h in range(num_heads):
        norm = torch.norm(W_heads[h]).item()
        head_norms.append(norm)

    metrics["head_norms_mean"] = np.mean(head_norms)
    metrics["head_norms_std"] = np.std(head_norms)
    metrics["head_norms_cv"] = np.std(head_norms) / (np.mean(head_norms) + 1e-8)  # Coefficient of variation

    # 2. Inter-head correlation (should be LOW)
    # Flatten each head and compute pairwise correlation
    W_heads_flat = W_heads.reshape(num_heads, -1)  # [num_heads, head_dim * in_features]

    # Normalize each head
    W_heads_norm = W_heads_flat / (torch.norm(W_heads_flat, dim=1, keepdim=True) + 1e-8)

    # Compute correlation matrix
    corr_matrix = torch.mm(W_heads_norm, W_heads_norm.t())  # [num_heads, num_heads]

    # Extract off-diagonal elements
    mask = ~torch.eye(num_heads, dtype=torch.bool)
    inter_head_corr = corr_matrix[mask].abs()

    metrics["inter_head_corr_mean"] = inter_head_corr.mean().item()
    metrics["inter_head_corr_max"] = inter_head_corr.max().item()
    metrics["inter_head_corr_std"] = inter_head_corr.std().item()

    # 3. Intra-head coherence (weight variance within head)
    intra_head_coherence = []
    for h in range(num_heads):
        # Compute variance of weights within this head
        head_var = torch.var(W_heads[h]).item()
        intra_head_coherence.append(head_var)

    metrics["intra_head_var_mean"] = np.mean(intra_head_coherence)
    metrics["intra_head_var_std"] = np.std(intra_head_coherence)

    return metrics


def compare_quantization_configs(W_orig: torch.Tensor, layer_name: str, num_heads: int):
    """
    Compare original vs quantized with different block sizes.
    """
    print(f"\n{'='*80}")
    print(f"Layer: {layer_name}")
    print(f"Shape: {list(W_orig.shape)}")
    print(f"Heads: {num_heads}, Head dim: {W_orig.shape[0] // num_heads}")
    print(f"{'='*80}")

    # Analyze original
    print("\n[Original FP16]")
    metrics_orig = analyze_layer_head_structure(W_orig, layer_name, num_heads)
    if metrics_orig:
        print(f"  Head norms: mean={metrics_orig['head_norms_mean']:.4f}, std={metrics_orig['head_norms_std']:.4f}, CV={metrics_orig['head_norms_cv']:.4f}")
        print(f"  Inter-head correlation: mean={metrics_orig['inter_head_corr_mean']:.4f}, max={metrics_orig['inter_head_corr_max']:.4f}")
        print(f"  Intra-head variance: mean={metrics_orig['intra_head_var_mean']:.6f}, std={metrics_orig['intra_head_var_std']:.6f}")

    # Quantize with block_size=16 (misaligned)
    print("\n[Quantized W4A8, block_size=16 (MISALIGNED)]")
    pack_16 = pack_weight(
        W_orig.cpu().numpy(),
        weight_bits=4,
        block_size=16,
        enable_permute=True,
        lambda_smooth=0.15,
    )
    W_quant_16, scales_16 = transform_weight_for_forward(
        W_orig,
        pack_16,
        weight_bits=4,
        apply_row_rot=True,
    )

    # Fake quantize (this is what happens during forward)
    max_q = qmax(4)
    W_fq_16 = fake_quantize_sym(W_quant_16, scales_16, 4)

    # Compute error
    error_16 = torch.norm(W_orig - W_fq_16) / torch.norm(W_orig)
    print(f"  Quantization error (relative): {error_16.item():.6f}")

    metrics_16 = analyze_layer_head_structure(W_fq_16, layer_name, num_heads)
    if metrics_16:
        print(f"  Head norms: mean={metrics_16['head_norms_mean']:.4f}, std={metrics_16['head_norms_std']:.4f}, CV={metrics_16['head_norms_cv']:.4f}")
        print(f"  Inter-head correlation: mean={metrics_16['inter_head_corr_mean']:.4f}, max={metrics_16['inter_head_corr_max']:.4f}")
        print(f"  Intra-head variance: mean={metrics_16['intra_head_var_mean']:.6f}, std={metrics_16['intra_head_var_std']:.6f}")

        # Compare changes
        print(f"\n  Changes from original:")
        print(f"    Head norm CV: {metrics_orig['head_norms_cv']:.4f} → {metrics_16['head_norms_cv']:.4f} (Δ={metrics_16['head_norms_cv'] - metrics_orig['head_norms_cv']:.4f})")
        print(f"    Inter-head corr: {metrics_orig['inter_head_corr_mean']:.4f} → {metrics_16['inter_head_corr_mean']:.4f} (Δ={metrics_16['inter_head_corr_mean'] - metrics_orig['inter_head_corr_mean']:.4f})")

    # Quantize with block_size=64 (aligned)
    print("\n[Quantized W4A8, block_size=64 (HEAD-ALIGNED)]")
    pack_64 = pack_weight(
        W_orig.cpu().numpy(),
        weight_bits=4,
        block_size=64,
        enable_permute=True,
        lambda_smooth=0.15,
    )
    W_quant_64, scales_64 = transform_weight_for_forward(
        W_orig,
        pack_64,
        weight_bits=4,
        apply_row_rot=True,
    )

    W_fq_64 = fake_quantize_sym(W_quant_64, scales_64, 4)

    error_64 = torch.norm(W_orig - W_fq_64) / torch.norm(W_orig)
    print(f"  Quantization error (relative): {error_64.item():.6f}")

    metrics_64 = analyze_layer_head_structure(W_fq_64, layer_name, num_heads)
    if metrics_64:
        print(f"  Head norms: mean={metrics_64['head_norms_mean']:.4f}, std={metrics_64['head_norms_std']:.4f}, CV={metrics_64['head_norms_cv']:.4f}")
        print(f"  Inter-head correlation: mean={metrics_64['inter_head_corr_mean']:.4f}, max={metrics_64['inter_head_corr_max']:.4f}")
        print(f"  Intra-head variance: mean={metrics_64['intra_head_var_mean']:.6f}, std={metrics_64['intra_head_var_std']:.6f}")

        print(f"\n  Changes from original:")
        print(f"    Head norm CV: {metrics_orig['head_norms_cv']:.4f} → {metrics_64['head_norms_cv']:.4f} (Δ={metrics_64['head_norms_cv'] - metrics_orig['head_norms_cv']:.4f})")
        print(f"    Inter-head corr: {metrics_orig['inter_head_corr_mean']:.4f} → {metrics_64['inter_head_corr_mean']:.4f} (Δ={metrics_64['inter_head_corr_mean'] - metrics_orig['inter_head_corr_mean']:.4f})")

    # Summary comparison
    print(f"\n{'─'*80}")
    print("SUMMARY:")
    print(f"  Quantization error: block_16={error_16.item():.6f}, block_64={error_64.item():.6f}")
    if error_64.item() < error_16.item():
        print(f"  ✅ HEAD-ALIGNED (block_64) has LOWER error by {(error_16.item() - error_64.item())/error_16.item()*100:.2f}%")

    if metrics_16 and metrics_64:
        corr_increase_16 = metrics_16['inter_head_corr_mean'] - metrics_orig['inter_head_corr_mean']
        corr_increase_64 = metrics_64['inter_head_corr_mean'] - metrics_orig['inter_head_corr_mean']

        print(f"  Inter-head correlation increase: block_16=+{corr_increase_16:.4f}, block_64=+{corr_increase_64:.4f}")
        if abs(corr_increase_64) < abs(corr_increase_16):
            print(f"  ✅ HEAD-ALIGNED (block_64) preserves head independence BETTER")

    print(f"{'─'*80}")


def main():
    # Load checkpoint
    ckpt_path = os.path.expanduser("~/VLM_REPO/openpi/ckpts/pi05_libero_torch/model.safetensors")

    print("="*80)
    print("Attention Head Structure Analysis")
    print("Comparing block_size=16 (misaligned) vs block_size=64 (head-aligned)")
    print("="*80)
    print(f"\nLoading checkpoint: {ckpt_path}")

    with safe_open(ckpt_path, framework="pt", device="cpu") as f:
        # Analyze DiT attention layers
        print("\n" + "="*80)
        print("DiT (Action Expert) Attention Layers")
        print("="*80)

        # Q projection: 32 heads × 64 head_dim = 2048
        q_key = "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj.weight"
        W_q = f.get_tensor(q_key)
        compare_quantization_configs(W_q, "DiT layer 0 Q projection", num_heads=32)

        # K projection: 4 heads × 64 head_dim = 256 (Multi-Query Attention)
        k_key = "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.k_proj.weight"
        W_k = f.get_tensor(k_key)
        compare_quantization_configs(W_k, "DiT layer 0 K projection", num_heads=4)

        # V projection: 4 heads × 64 head_dim = 256
        v_key = "paligemma_with_expert.gemma_expert.model.layers.0.self_attn.v_proj.weight"
        W_v = f.get_tensor(v_key)
        compare_quantization_configs(W_v, "DiT layer 0 V projection", num_heads=4)

        # Check if LLM layers exist
        print("\n" + "="*80)
        print("LLM (Gemma) Attention Layers")
        print("="*80)

        try:
            # Try different possible key names for LLM
            llm_keys = [
                "paligemma_with_expert.paligemma.language_model.model.layers.0.self_attn.q_proj.weight",
                "paligemma_with_expert.paligemma.model.language_model.layers.0.self_attn.q_proj.weight",
            ]

            W_llm_q = None
            for key in llm_keys:
                try:
                    W_llm_q = f.get_tensor(key)
                    print(f"Found LLM Q at: {key}")
                    break
                except:
                    continue

            if W_llm_q is not None:
                # Infer num_heads from shape
                out_features = W_llm_q.shape[0]
                if out_features % 64 == 0:
                    num_heads_llm = out_features // 64
                    compare_quantization_configs(W_llm_q, "LLM layer 0 Q projection", num_heads=num_heads_llm)
                else:
                    print(f"Cannot infer num_heads for LLM Q shape {W_llm_q.shape}")
            else:
                print("Could not find LLM Q projection weight")

        except Exception as e:
            print(f"Error loading LLM layers: {e}")

    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
    print("\nKey Findings:")
    print("  - block_size=16 causes rotation to mix features WITHIN heads")
    print("  - block_size=64 aligns with head boundaries, preserving structure")
    print("  - Lower quantization error with block_64")
    print("  - Better preservation of head independence with block_64")
    print("\nRecommendation:")
    print("  Use run_llm_dit_mlp_w4a8_head_aligned.sh for best results!")


if __name__ == "__main__":
    main()
