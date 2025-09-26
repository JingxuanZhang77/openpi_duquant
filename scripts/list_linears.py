#!/usr/bin/env python3
import os
import argparse
import torch
from torch import nn

from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config


def iter_linears(root: nn.Module):
    for name, mod in root.named_modules():
        if isinstance(mod, nn.Linear):
            yield name, mod


def main():
    parser = argparse.ArgumentParser(description="List nn.Linear layers in a PyTorch policy model")
    parser.add_argument("config", type=str, help="Train config name (e.g., pi05_libero)")
    parser.add_argument("checkpoint", nargs="?", type=str, help="Checkpoint dir containing model.safetensors (omit with --no-weights)")
    parser.add_argument("--scope", type=str, default=os.environ.get("OPENPI_DUQUANT_SCOPE", ""), help="Scope prefix filter (e.g., policy.dit.)")
    parser.add_argument("--include", type=str, default=os.environ.get("OPENPI_DUQUANT_INCLUDE", ".*"), help="Include regex")
    parser.add_argument("--exclude", type=str, default=os.environ.get("OPENPI_DUQUANT_EXCLUDE", "^$"), help="Exclude regex")
    parser.add_argument("--no-weights", action="store_true", help="Instantiate PyTorch model without loading weights (structure only)")
    args = parser.parse_args()

    if args.no_weights:
        train_cfg = _config.get_config(args.config)
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # lazy import

        model = PI0Pytorch(config=train_cfg.model)
    else:
        if not args.checkpoint:
            raise SystemExit("checkpoint is required unless --no-weights is set")
        policy = _policy_config.create_trained_policy(_config.get_config(args.config), args.checkpoint)
        assert policy._is_pytorch_model, "Expected a PyTorch checkpoint directory containing model.safetensors"
        model = policy._model

    import re

    inc = re.compile(args.include)
    exc = re.compile(args.exclude)
    for name, mod in iter_linears(model):
        if args.scope and not name.startswith(args.scope):
            continue
        if not inc.search(name) or exc.search(name):
            continue
        print(f"{name}: Linear({mod.in_features}, {mod.out_features}) dtype={mod.weight.dtype}")


if __name__ == "__main__":
    main()
