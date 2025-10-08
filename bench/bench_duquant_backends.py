from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

if "OPENPI_DUQUANT_PACKDIR" not in os.environ:
    os.environ["OPENPI_DUQUANT_PACKDIR"] = str((Path.cwd() / ".bench_duquant_packs").resolve())

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.openpi.models_pytorch.duquant_layers import enable_duquant_if_configured
from src.openpi.models_pytorch.duquant_preprocess import _DUQUANT_PROFILER


def parse_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"f32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype '{name}'")


class DummyBlock(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        heads: int,
        head_dim: int,
        kv_heads: int,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = head_dim
        self.kv_heads = kv_heads
        q_dim = head_dim * heads
        kv_dim = head_dim * kv_heads
        self.q_proj = nn.Linear(embed_dim, q_dim)
        self.k_proj = nn.Linear(embed_dim, kv_dim)
        self.v_proj = nn.Linear(embed_dim, kv_dim)
        self.o_proj = nn.Linear(q_dim, embed_dim)
        self.gate_proj = nn.Linear(embed_dim, hidden_dim)
        self.up_proj = nn.Linear(embed_dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = torch.relu(self.q_proj(x))
        _ = torch.relu(self.k_proj(x))
        _ = torch.relu(self.v_proj(x))
        attn_out = self.o_proj(q)
        gate = torch.relu(self.gate_proj(x))
        up = torch.relu(self.up_proj(x))
        ffn = self.down_proj(gate * up)
        return attn_out + ffn + x


class DummyTransformerStack(nn.Module):
    def __init__(
        self,
        layers: int,
        *,
        embed_dim: int,
        hidden_dim: int,
        heads: int,
        head_dim: int,
        kv_heads: int,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [DummyBlock(embed_dim, hidden_dim, heads, head_dim, kv_heads) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for block in self.layers:
            out = block(out)
        return out


class DummyPaligemma(nn.Module):
    def __init__(self, layers: int, cfg: Dict[str, int]) -> None:
        super().__init__()
        model = nn.Module()
        model.add_module(
            "language_model",
            DummyTransformerStack(
                layers,
                embed_dim=cfg["embed_dim"],
                hidden_dim=cfg["hidden_dim"],
                heads=cfg["heads"],
                head_dim=cfg["head_dim"],
                kv_heads=cfg["kv_heads"],
            ),
        )
        self.add_module("model", model)


class DummyGemmaExpert(nn.Module):
    def __init__(self, layers: int, cfg: Dict[str, int]) -> None:
        super().__init__()
        model = DummyTransformerStack(
            layers,
            embed_dim=cfg["embed_dim"],
            hidden_dim=cfg["hidden_dim"],
            heads=cfg["heads"],
            head_dim=cfg["head_dim"],
            kv_heads=cfg["kv_heads"],
        )
        self.add_module("model", model)


class DummyPaligemmaWithExpert(nn.Module):
    def __init__(
        self,
        layers: int,
        llm_cfg: Dict[str, int],
        dit_cfg: Dict[str, int],
    ) -> None:
        super().__init__()
        self.add_module("paligemma", DummyPaligemma(layers, llm_cfg))
        self.add_module("gemma_expert", DummyGemmaExpert(layers, dit_cfg))


class DummyPolicy(nn.Module):
    def __init__(
        self,
        layers: int,
        llm_cfg: Dict[str, int],
        dit_cfg: Dict[str, int],
        target: str,
    ) -> None:
        super().__init__()
        if target not in {"dit", "llm"}:
            raise ValueError(f"Unsupported target '{target}', expected 'dit' or 'llm'.")
        self.target = target
        self.add_module(
            "paligemma_with_expert",
            DummyPaligemmaWithExpert(layers, llm_cfg, dit_cfg),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.target == "llm":
            stack = self.paligemma_with_expert.paligemma.model.language_model
        else:
            stack = self.paligemma_with_expert.gemma_expert.model
        return stack(x)


def reset_profiler() -> None:
    if hasattr(_DUQUANT_PROFILER, "_stats") and hasattr(_DUQUANT_PROFILER, "_new_store"):
        _DUQUANT_PROFILER._stats = _DUQUANT_PROFILER._new_store()  # type: ignore[attr-defined]


def collect_profiler_lines() -> List[str]:
    if not getattr(_DUQUANT_PROFILER, "enabled", False):
        return []
    stats = getattr(_DUQUANT_PROFILER, "_stats", {}) or {}
    lines = []
    for label, entry in sorted(stats.items()):
        total_ms = entry.get("time", 0.0) * 1000.0
        lines.append(f"    DUQUANT PROFILE: {label} {total_ms:.1f}ms")
    reset_profiler()
    return lines


def benchmark_backend(
    backend: str,
    base_state: Dict[str, torch.Tensor],
    input_tensor: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    reference: Optional[torch.Tensor],
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, List[str]]:
    os.environ["OPENPI_DUQUANT_BACKEND"] = backend
    os.environ["OPENPI_DUQUANT_SCOPE"] = args.scope
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    model = DummyPolicy(
        layers=args.layers,
        llm_cfg=args.llm_cfg,
        dit_cfg=args.dit_cfg,
        target=args.target,
    )
    model.load_state_dict(base_state, strict=True)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    if backend != "off":
        enable_duquant_if_configured(model)
        model = model.to(device=device, dtype=dtype)

    iters = args.iters
    warmup = args.warmup
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(input_tensor)
    torch.cuda.synchronize()
    reset_profiler()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    outputs: Optional[torch.Tensor] = None
    start.record()
    with torch.inference_mode():
        for _ in range(iters):
            outputs = model(input_tensor)
    end.record()
    torch.cuda.synchronize()

    if outputs is None:
        raise RuntimeError("Model produced no outputs during benchmarking")

    elapsed_ms = start.elapsed_time(end)
    avg_ms = elapsed_ms / iters
    calls_per_s = 1000.0 / avg_ms if avg_ms > 0 else float("inf")
    peak_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)

    lines = [
        f"[BACKEND={backend}]  Avg {avg_ms:.2f} ms/call, {calls_per_s:.1f} calls/s, peak {peak_mem_mb:.0f} MB"
    ]

    if backend != "off" and reference is not None:
        diff = (outputs - reference).to(torch.float32)
        l2 = diff.norm().item()
        mae = diff.abs().mean().item()
        max_abs = diff.abs().max().item()
        lines.append(f"    err vs FP16: L2={l2:.4e}, MAE={mae:.4e}, max_abs={max_abs:.4e}")

    lines.extend(collect_profiler_lines())

    return outputs.detach(), lines


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DuQuant backends (off / fake / bitblas).")
    parser.add_argument("--layers", type=int, default=21)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--heads", type=int, default=None)
    parser.add_argument("--seq", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scope", type=str, default="policy.dit.", help="Prefix scope passed to OPENPI_DUQUANT_SCOPE.")
    parser.add_argument("--head-dim", type=int, default=None)
    parser.add_argument("--kv-heads", type=int, default=None)
    parser.add_argument(
        "--target",
        type=str,
        default="dit",
        choices=["dit", "llm"],
        help="Which module stack the dummy policy executes (controls forward path).",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device is required for benchmarking.")

    torch.manual_seed(args.seed)
    dtype = parse_dtype(args.dtype)
    device = torch.device("cuda")

    llm_defaults = {
        "embed_dim": 2048,
        "hidden_dim": 16384,
        "heads": 8,
        "head_dim": 256,
        "kv_heads": 1,
    }
    dit_defaults = {
        "embed_dim": 1024,
        "hidden_dim": 4096,
        "heads": 8,
        "head_dim": 256,
        "kv_heads": 1,
    }

    llm_cfg = llm_defaults.copy()
    dit_cfg = dit_defaults.copy()

    def apply_overrides(cfg):
        if args.embed_dim is not None:
            cfg["embed_dim"] = args.embed_dim
        if args.hidden_dim is not None:
            cfg["hidden_dim"] = args.hidden_dim
        if args.heads is not None:
            cfg["heads"] = args.heads
        if args.head_dim is not None:
            cfg["head_dim"] = args.head_dim
        if args.kv_heads is not None:
            cfg["kv_heads"] = args.kv_heads

    if args.target == "llm":
        apply_overrides(llm_cfg)
    else:
        apply_overrides(dit_cfg)

    active_cfg = llm_cfg if args.target == "llm" else dit_cfg

    # Persist configs on args for downstream calls.
    args.llm_cfg = llm_cfg
    args.dit_cfg = dit_cfg
    args.active_cfg = active_cfg
    args.embed_dim = active_cfg["embed_dim"]
    args.hidden_dim = active_cfg["hidden_dim"]
    args.heads = active_cfg["heads"]
    args.head_dim = active_cfg["head_dim"]
    args.kv_heads = active_cfg["kv_heads"]

    baseline_model = DummyPolicy(
        layers=args.layers,
        llm_cfg=llm_cfg,
        dit_cfg=dit_cfg,
        target=args.target,
    ).to(device=device, dtype=dtype)
    baseline_model.eval()
    base_state = {k: v.cpu() for k, v in baseline_model.state_dict().items()}

    input_tensor = torch.randn(
        args.batch,
        args.seq,
        active_cfg["embed_dim"],
        device=device,
        dtype=dtype,
    )

    with torch.inference_mode():
        reference = baseline_model(input_tensor).detach()

    reset_profiler()

    outputs, lines = benchmark_backend(
        "off",
        base_state,
        input_tensor,
        device,
        dtype,
        reference=None,
        args=args,
    )
    for line in lines:
        print(line)
    reference = outputs.detach()

    for backend in ("fake", "bitblas"):
        try:
            outputs, lines = benchmark_backend(
                backend,
                base_state,
                input_tensor,
                device,
                dtype,
                reference=reference,
                args=args,
            )
        except RuntimeError as exc:
            print(f"[BACKEND={backend}] Benchmark failed: {exc}")
            continue
        for line in lines:
            print(line)


if __name__ == "__main__":
    main()
