#!/usr/bin/env python3
"""Build a TensorRT INT8 engine for the pi0.5 Libero Gemma expert."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import tensorrt as trt
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit("TensorRT is required to build the engine. Install TensorRT 10.x and retry.") from exc

try:
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
except ModuleNotFoundError as exc:  # pragma: no cover - import guard
    raise SystemExit("pycuda is required for calibration. Install pycuda and retry.") from exc

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))

from scripts.trt._trt_utils import (
    build_observation,
    iter_calibration_files,
    load_calibration_file,
    load_policy,
    prepare_trt_sample,
)
from scripts.trt.inspect_quant_backend import check_engine

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class LiberoCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        *,
        policy,
        calibration_paths: list[Path],
        batch_size: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        self._policy = policy
        self._paths = calibration_paths
        self._batch_size = batch_size
        self._device = device
        self._index = 0
        self._device_buffers: dict[str, cuda.DeviceAllocation] = {}
        self._structure: tuple[int, ...] | None = None

    def get_batch_size(self) -> int:
        return self._batch_size

    def _prepare_numpy(self, path: Path) -> dict[str, np.ndarray]:
        batch = load_calibration_file(path, batch_size=self._batch_size, device=self._device)
        observation = build_observation(batch, self._device)
        sample = prepare_trt_sample(self._policy, observation, device=self._device)
        if self._structure is None:
            self._structure = sample.pkv_structure
        elif self._structure != sample.pkv_structure:
            raise RuntimeError(f"Inconsistent PKV structure in calibration data: {sample.pkv_structure} vs {self._structure}")
        return {name: tensor.numpy() for name, tensor in sample.tensors.items()}

    def get_batch(self, names: list[str]) -> list[int] | None:
        if self._index >= len(self._paths):
            return None

        arrays = self._prepare_numpy(self._paths[self._index])
        if not self._device_buffers:
            for name in names:
                arr = arrays[name]
                self._device_buffers[name] = cuda.mem_alloc(arr.nbytes)

        bindings: list[int] = []
        for name in names:
            if name not in arrays:
                raise KeyError(f"Calibration arrays missing input '{name}'")
            cuda.memcpy_htod(self._device_buffers[name], arrays[name])
            bindings.append(int(self._device_buffers[name]))

        self._index += 1
        return bindings

    def read_calibration_cache(self) -> bytes | None:  # noqa: D401
        return None

    def write_calibration_cache(self, cache: bytes) -> None:  # noqa: D401
        return None


def _set_profile(
    config: trt.IBuilderConfig,
    network: trt.INetworkDefinition,
    sample_tensors: dict[str, torch.Tensor],
    *,
    min_batch: int,
    opt_batch: int,
    max_batch: int,
) -> None:
    profile = config.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        sample = sample_tensors[name]
        shape = list(sample.shape)
        if shape:
            shape[0] = max(shape[0], opt_batch)
            min_shape = [min_batch, *shape[1:]]
            opt_shape = [opt_batch, *shape[1:]]
            max_shape = [max_batch, *shape[1:]]
        else:
            min_shape = opt_shape = max_shape = []
        profile.set_shape(name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)


def build_engine(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    policy = load_policy(args.config, args.checkpoint, device=str(device), duquant_scope=args.duquant_scope)

    calibration_dir = Path(args.calibration).expanduser()
    paths = list(iter_calibration_files(calibration_dir))
    if not paths:
        raise FileNotFoundError(f"No calibration batches found in {calibration_dir}")

    batch = load_calibration_file(paths[0], batch_size=args.batch_size, device=device)
    observation = build_observation(batch, device)
    sample = prepare_trt_sample(policy, observation, device=device)

    builder = trt.Builder(TRT_LOGGER)
    if not builder.platform_has_fast_int8:
        raise RuntimeError("INT8 not supported on this platform")

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    onnx_path = Path(args.onnx).expanduser()
    with onnx_path.open("rb") as f:
        if not parser.parse(f.read()):
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.max_workspace_size = args.workspace

    _set_profile(
        config,
        network,
        sample.tensors,
        min_batch=args.min_batch,
        opt_batch=args.batch_size,
        max_batch=args.max_batch,
    )

    calibrator = LiberoCalibrator(
        policy=policy,
        calibration_paths=paths,
        batch_size=args.batch_size,
        device=device,
    )
    config.int8_calibrator = calibrator

    with builder.build_engine(network, config) as engine:
        engine_path = Path(args.engine).expanduser()
        engine_path.parent.mkdir(parents=True, exist_ok=True)
        with engine_path.open("wb") as f:
            f.write(engine.serialize())
        print(f"[TensorRT] built INT8 engine -> {engine_path}")

    check_engine(engine_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx", required=True, help="Path to the exported ONNX model")
    parser.add_argument("--engine", default="pi05_int8.plan", help="Output TensorRT engine path")
    parser.add_argument("--calibration", default="calibration_batches", help="Calibration directory")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint directory for the policy")
    parser.add_argument("--config", default="pi05_libero", help="Training config name")
    parser.add_argument("--device", default="cuda", help="Device for intermediate tensor prep")
    parser.add_argument("--batch-size", type=int, default=8, help="Calibration batch size")
    parser.add_argument("--min-batch", type=int, default=1, help="Minimum batch size for TensorRT profile")
    parser.add_argument("--max-batch", type=int, default=8, help="Maximum batch size for TensorRT profile")
    parser.add_argument("--workspace", type=int, default=4 << 30, help="Workspace size in bytes")
    parser.add_argument("--duquant-scope", default=None, help="Override OPENPI_DUQUANT_SCOPE if provided")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_engine(args)


if __name__ == "__main__":
    main()
