#!/usr/bin/env python3
"""
Accuracy-only evaluation of BitBLAS W4A8 on GR00T layers using real activations/weights.

This mirrors the measurement style of test_bitblas_real_activations.py:
per-layer MSE, max diff, and masked relative error (%), no latency timing.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from safetensors import safe_open

from bitblas import Matmul, MatmulConfig, auto_detect_nvidia_target


@dataclass
class LayerEntry:
    idx: int
    name: str
    in_features: int
    out_features: int

    @property
    def shape_str(self) -> str:
        return f"{self.in_features}->{self.out_features}"


def parse_layer_sequence(path: Path) -> List[LayerEntry]:
    pattern = re.compile(r"\[[^\]]+\]\[[^\]]+\]\s+(.*?): Linear\((\d+)->(\d+)\)")
    entries: List[LayerEntry] = []
    with path.open() as fin:
        for line in fin:
            match = pattern.search(line)
            if match:
                name, in_feat, out_feat = match.groups()
                entries.append(
                    LayerEntry(
                        idx=len(entries),
                        name=name.strip(),
                        in_features=int(in_feat),
                        out_features=int(out_feat),
                    )
                )
    if not entries:
        raise ValueError(f"No DUQUANT Linear entries found in {path}")
    return entries


def load_weight_map(index_path: Path) -> Dict[str, Path]:
    with index_path.open() as f:
        idx = json.load(f)
    base_dir = index_path.parent
    weight_map = {
        name: base_dir / filename for name, filename in idx["weight_map"].items()
    }
    return weight_map


def load_weight_tensor(name: str, weight_map: Dict[str, Path]) -> torch.Tensor:
    if name not in weight_map:
        raise KeyError(f"Weight {name} not found in index")
    shard_path = weight_map[name]
    with safe_open(shard_path, framework="pt") as f:
        return f.get_tensor(name)


def map_layer_to_activation_key(layer_name: str) -> str | None:
    """
    Convert full layer name like backbone.eagle_model.language_model.model.layers.0.self_attn.q_proj
    to activation cache key like layers.0.self_attn.q_proj.
    """
    m = re.search(
        r"layers\.(\d+)\.(self_attn|mlp)\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)",
        layer_name,
    )
    if not m:
        return None
    idx, block, proj = m.groups()
    return f"layers.{idx}.{block}.{proj}"


def quantize_tensor(x: torch.Tensor, max_q: int) -> Tuple[torch.Tensor, float]:
    scale = torch.max(torch.abs(x)).item() / max_q if torch.max(torch.abs(x)) > 0 else 1.0
    q = torch.clamp(torch.round(x / scale), -max_q, max_q).to(torch.int8)
    return q, scale


def quantize_activation(x: torch.Tensor, max_q: int) -> Tuple[torch.Tensor, float]:
    return quantize_tensor(x, max_q)


def quantize_weight_rowwise(x: torch.Tensor, max_q: int) -> Tuple[torch.Tensor, torch.Tensor]:
    max_abs = torch.max(torch.abs(x), dim=1, keepdim=True).values
    scale = torch.where(max_abs > 0, max_abs / max_q, torch.ones_like(max_abs))
    q = torch.clamp(torch.round(x / scale), -max_q, max_q).to(torch.int8)
    return q, scale.squeeze(1)


def get_bitblas_matmul(
    cache: Dict[Tuple[int, int, int, bool], Matmul],
    *,
    M: int,
    N: int,
    K: int,
    target: str,
    enable_tuning: bool,
) -> Matmul:
    key = (M, N, K, enable_tuning)
    if key not in cache:
        config = MatmulConfig(
            M=M,
            N=N,
            K=K,
            A_dtype="int8",
            W_dtype="int4",
            accum_dtype="int32",
            out_dtype="float16",
            layout="nt",
            with_bias=False,
            group_size=None,
            with_scaling=False,
            with_zeros=False,
            zeros_mode=None,
        )
        cache[key] = Matmul(config, target=target, enable_tuning=enable_tuning)
    return cache[key]


def benchmark_w4a8(
    activation_fp: torch.Tensor,
    weight_fp: torch.Tensor,
    warmup: int,
    runs: int,
    matmul: Matmul,
    *,
    act_max_q: int,
    weight_max_q: int,
) -> torch.Tensor:
    activation_int, s_a = quantize_activation(activation_fp, act_max_q)
    weight_int, s_w = quantize_weight_rowwise(weight_fp, weight_max_q)

    packed_weight = matmul.transform_weight(weight_int)

    def op():
        out = matmul(activation_int, packed_weight)
        return out * (s_a * s_w)

    torch.cuda.synchronize()
    for _ in range(max(0, warmup)):
        op()
    torch.cuda.synchronize()
    for _ in range(runs):
        op()
    torch.cuda.synchronize()
    return op()


def main():
    parser = argparse.ArgumentParser(description="GR00T BitBLAS accuracy check (W4A8, real activations/weights).")
    parser.add_argument(
        "--layer-log",
        type=Path,
        default=Path("reports/QuantVLA/gr00tN1.5_llm_ditmlp_layer.json"),
        help="DuQuant layer log to define layer order/shapes.",
    )
    parser.add_argument(
        "--weight-index",
        type=Path,
        required=True,
        help="Path to model.safetensors.index.json for loading real weights.",
    )
    parser.add_argument(
        "--activation-cache",
        type=Path,
        required=True,
        help="Path to torch file with captured activations (activations/gr00t_gr1.pt).",
    )
    parser.add_argument("--warmup", type=int, default=1, help="Warmup iterations for BitBLAS.")
    parser.add_argument("--runs", type=int, default=1, help="Measured iterations for BitBLAS (accuracy only).")
    parser.add_argument("--act-max-q", type=int, default=127, help="Max int level for activations (int8).")
    parser.add_argument("--weight-max-q", type=int, default=7, help="Max int level for weights (int4).")
    parser.add_argument(
        "--enable-tuning",
        action="store_true",
        help="Enable BitBLAS tuning to search Tensor Core kernels (first run slower).",
    )
    args = parser.parse_args()

    entries = parse_layer_sequence(args.layer_log)
    weight_map = load_weight_map(args.weight_index)

    cache_obj = torch.load(args.activation_cache, map_location="cpu")
    activation_cache = cache_obj.get("activations", {})

    target = auto_detect_nvidia_target()
    matmul_cache: Dict[Tuple[int, int, int, bool], Matmul] = {}

    print(f"{'Layer':<60} {'Shape':<20} {'MSE':<12} {'MaxDiff':<12} {'RelErr%':<10}")
    print("-" * 120)
    total_mse = 0.0
    total_max_diff = 0.0
    total_rel_err = 0.0
    count = 0

    for entry in entries:
        act_key = map_layer_to_activation_key(entry.name)
        if act_key not in activation_cache:
            print(f"{entry.name:<60} missing activation; skipping")
            continue

        activation_fp = activation_cache[act_key].to(device="cuda", dtype=torch.float16)
        orig_shape = tuple(activation_fp.shape)
        activation_2d = activation_fp.reshape(-1, activation_fp.shape[-1])  # flatten batch/seq for matmul
        weight_fp = load_weight_tensor(f"{entry.name}.weight", weight_map).to(torch.float16).cuda()

        ref_out = torch.matmul(activation_2d, weight_fp.t())

        M_dim = activation_2d.shape[0]
        matmul = get_bitblas_matmul(
            matmul_cache,
            M=M_dim,
            N=entry.out_features,
            K=entry.in_features,
            target=target,
            enable_tuning=args.enable_tuning,
        )
        w4a8_out = benchmark_w4a8(
            activation_2d,
            weight_fp,
            args.warmup,
            args.runs,
            matmul,
            act_max_q=args.act_max_q,
            weight_max_q=args.weight_max_q,
        )

        ref_flat = ref_out.view(-1, ref_out.shape[-1]).float()
        w4a8_flat = w4a8_out.view(-1, w4a8_out.shape[-1]).float()
        diff = w4a8_flat - ref_flat
        mse = torch.mean(torch.square(diff)).item()
        max_diff = torch.max(torch.abs(diff)).item()
        ref_abs = ref_flat.abs()
        mask = ref_abs > 1e-6  # match test_bitblas_real_activations style
        rel_err = (torch.mean(torch.abs(diff)[mask] / ref_abs[mask]).item() * 100) if mask.any() else 0.0

        out_shape = (*orig_shape[:-1], entry.out_features)
        shape_str = f"{orig_shape}->{out_shape}"
        short_name = entry.name[-55:] if len(entry.name) > 55 else entry.name
        print(f"{short_name:<60} {shape_str:<20} {mse:<12.2e} {max_diff:<12.6f} {rel_err:<10.2f}")

        total_mse += mse
        total_max_diff = max(total_max_diff, max_diff)
        total_rel_err += rel_err
        count += 1

    if count > 0:
        print("-" * 120)
        print(
            f"{'AVERAGE':<60} {'':<20} {total_mse / count:<12.2e} "
            f"{total_max_diff:<12.6f} {total_rel_err / count:<10.2f}"
        )


if __name__ == "__main__":
    main()
