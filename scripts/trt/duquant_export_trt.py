#!/usr/bin/env python3
"""
Build an INT8 TensorRT engine for a simple matmul representative of DuQuantLinear.
This is an optional utility and does not modify the main code path.

Requires TensorRT installed in the environment.
"""
import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="duquant_linear.onnx")
    parser.add_argument("--engine", type=str, default="duquant_linear_int8.plan")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--in_features", type=int, default=2048)
    parser.add_argument("--out_features", type=int, default=2048)
    args = parser.parse_args()

    try:
        import torch
        from torch import nn
        import tensorrt as trt
        import numpy as np
    except Exception as e:  # noqa: BLE001
        print(f"TensorRT path requires torch and tensorrt installed: {e}", file=sys.stderr)
        sys.exit(1)

    class TinyLinear(nn.Module):
        def __init__(self, in_f, out_f):
            super().__init__()
            self.l = nn.Linear(in_f, out_f, bias=True)

        def forward(self, x):
            return self.l(x)

    model = TinyLinear(args.in_features, args.out_features).eval()
    x = torch.randn(args.batch, args.in_features)
    torch.onnx.export(
        model,
        x,
        args.onnx,
        input_names=["x"],
        output_names=["y"],
        dynamic_axes=None,
        opset_version=17,
    )
    print(f"Exported ONNX to {args.onnx}")

    # Build TensorRT INT8 engine with calibrator
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser_trt = trt.OnnxParser(network, TRT_LOGGER)
    with open(args.onnx, "rb") as f:
        if not parser_trt.parse(f.read()):
            for i in range(parser_trt.num_errors):
                print(parser_trt.get_error(i))
            raise RuntimeError("Failed to parse ONNX")

    config = builder.create_builder_config()
    config.max_workspace_size = 2 << 30
    if not builder.platform_has_fast_int8:
        raise RuntimeError("INT8 not supported on this platform")
    config.set_flag(trt.BuilderFlag.INT8)
    # Simple calibrator providing random data samples
    class RandomCalibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self, shape, n_batches=32):
            super().__init__()
            self.shape = shape
            self.n_batches = n_batches
            self.count = 0
            self.d = None

        def get_batch_size(self):
            return shape[0]

        def get_batch(self, names):
            if self.count >= self.n_batches:
                return None
            data = (np.random.randn(*self.shape)).astype(np.float32)
            import pycuda.driver as cuda  # type: ignore
            import pycuda.autoinit  # noqa: F401  # type: ignore
            if self.d is None:
                self.d = cuda.mem_alloc(data.nbytes)
            cuda.memcpy_htod(self.d, data)
            self.count += 1
            return [int(self.d)]

        def read_calibration_cache(self):
            return None

        def write_calibration_cache(self, cache):
            pass

    # Find input shape
    inp = network.get_input(0)
    shape = tuple(inp.shape)
    calibrator = RandomCalibrator(shape)
    config.int8_calibrator = calibrator

    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("Failed to build engine")
    with open(args.engine, "wb") as f:
        f.write(engine.serialize())
    print(f"Built INT8 engine {args.engine}")


if __name__ == "__main__":
    main()

