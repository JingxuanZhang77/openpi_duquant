#!/usr/bin/env python3
"""Export the pi0.5 Libero Gemma expert block with DuQuant weights to ONNX."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from scripts.trt._trt_utils import (
    TrtSample,
    build_observation,
    load_calibration_file,
    load_policy,
    prepare_trt_sample,
)


class GemmaExpertWrapper(torch.nn.Module):
    def __init__(self, gemma_model: torch.nn.Module, pkv_structure: tuple[int, ...]):
        super().__init__()
        self.gemma_model = gemma_model
        self.pkv_structure = pkv_structure

    def forward(  # type: ignore[override]
        self,
        suffix_embs: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        adarms_cond: torch.Tensor,
        *flat_pkv: torch.Tensor,
    ) -> torch.Tensor:
        past_key_values = []
        offset = 0
        for count in self.pkv_structure:
            tensors = tuple(flat_pkv[offset + idx] for idx in range(count))
            past_key_values.append(tensors)
            offset += count

        outputs = self.gemma_model(
            inputs_embeds=suffix_embs,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )
        return outputs.last_hidden_state


def export(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    policy = load_policy(args.config, args.checkpoint, device=str(device), duquant_scope=args.duquant_scope)

    calib_path = Path(args.calibration).expanduser()
    batch = load_calibration_file(calib_path, batch_size=args.batch_size, device=device)
    observation = build_observation(batch, device)

    sample: TrtSample = prepare_trt_sample(policy, observation, device=device)

    gemma_model = policy._model.paligemma_with_expert.gemma_expert.model  # noqa: SLF001
    gemma_model.eval()
    gemma_model.cpu()

    wrapper = GemmaExpertWrapper(gemma_model, sample.pkv_structure)
    wrapper.eval()

    input_names = list(sample.tensors.keys())
    dynamic_axes = {
        "suffix_embs": {0: "batch", 1: "suffix_seq"},
        "attention_mask": {0: "batch", 2: "suffix_seq", 3: "total_seq"},
        "position_ids": {0: "batch", 1: "suffix_seq"},
        "adarms_cond": {0: "batch"},
        "hidden_states": {0: "batch", 1: "suffix_seq"},
    }
    for name in input_names:
        if name.startswith("pkv_"):
            dynamic_axes[name] = {0: "batch"}

    inputs = tuple(sample.tensors[name] for name in input_names)

    output_path = Path(args.onnx).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        inputs,
        output_path.as_posix(),
        input_names=input_names,
        output_names=["hidden_states"],
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        do_constant_folding=args.constant_folding,
    )
    print(f"[ONNX] exported Gemma expert -> {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Directory containing model.safetensors")
    parser.add_argument("--config", default="pi05_libero", help="Training config name")
    parser.add_argument("--calibration", required=True, help="Path to a calibration batch (.pt)")
    parser.add_argument("--onnx", default="pi05_duquant_gemma.onnx", help="Output ONNX path")
    parser.add_argument("--batch-size", type=int, default=8, help="Number of samples to use from calibration batch")
    parser.add_argument("--device", default="cuda", help="Device for intermediate tensor preparation")
    parser.add_argument("--duquant-scope", default=None, help="Override OPENPI_DUQUANT_SCOPE if provided")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--constant-folding",
        action="store_true",
        help="Enable constant folding during export (disabled by default)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export(args)


if __name__ == "__main__":
    main()
