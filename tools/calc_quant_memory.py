#!/usr/bin/env python
import argparse
import os
from pathlib import Path

import torch

from openpi.policies import policy_config
from openpi.training import config as train_config

TARGET_PREFIX_LLM = "paligemma_with_expert.paligemma.model.language_model"
TARGET_PREFIX_DIT = "paligemma_with_expert.gemma_expert.model"
MLP_SUFFIXES = ("mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
ATTN_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
)


def categorize(name: str) -> str | None:
    if not name.endswith(".weight"):
        return None
    if not name.startswith("paligemma_with_expert"):
        return None
    base = name[:-len(".weight")]
    if base.startswith(TARGET_PREFIX_LLM):
        return "llm"
    if base.startswith(TARGET_PREFIX_DIT):
        if any(base.endswith(s) for s in MLP_SUFFIXES):
            return "dit_mlp"
        if any(base.endswith(s) for s in ATTN_SUFFIXES):
            return "dit_attn"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate Pi0.5 quant memory savings")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--quant-bytes", type=float, default=0.5, help="bytes per weight when quantized (default 4-bit -> 0.5)")
    args = parser.parse_args()

    cfg = train_config.get_config("pi05_libero")
    policy = policy_config.create_trained_policy(cfg, args.checkpoint, pytorch_device="cpu")
    model = policy._model

    counters = {
        "llm": 0,
        "dit_mlp": 0,
        "dit_attn": 0,
    }

    for name, param in model.named_parameters():
        group = categorize(name)
        if group is None:
            continue
        counters[group] += param.numel()

    bf16_bytes = 2.0
    quant_bytes = args.quant_bytes

    def summarize(description: str, numel: int) -> tuple[str, float, float, float]:
        bf = numel * bf16_bytes / (1024 ** 3)
        qt = numel * quant_bytes / (1024 ** 3)
        savings = bf - qt
        return description, bf, qt, savings

    entries = []
    entries.append(summarize("LLM (all linear)", counters["llm"]))
    entries.append(summarize("DiT MLP", counters["dit_mlp"]))
    entries.append(summarize("DiT attention", counters["dit_attn"]))
    entries.append(summarize("DiT all linear", counters["dit_mlp"] + counters["dit_attn"]))
    entries.append(summarize("LLM + DiT all linear", counters["llm"] + counters["dit_mlp"] + counters["dit_attn"]))

    print(f"Quant bytes/param: {quant_bytes}")
    print("")
    print(f"{'Module':<32} {'BF16 GB':>12} {'W4A8 GB':>12} {'Savings GB':>12}")
    print("-" * 70)
    for desc, bf, qt, sv in entries:
        print(f"{desc:<32} {bf:12.3f} {qt:12.3f} {sv:12.3f}")


if __name__ == "__main__":
    main()
