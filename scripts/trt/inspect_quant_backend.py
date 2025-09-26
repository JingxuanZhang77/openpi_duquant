#!/usr/bin/env python3
"""
Inspect a TensorRT engine and assert that it contains INT8 precision layers and no obvious QDQ-only simulation.
"""
import argparse
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("engine", type=str)
    args = parser.parse_args()

    try:
        import tensorrt as trt
    except Exception as e:  # noqa: BLE001
        print(f"TensorRT not available: {e}", file=sys.stderr)
        sys.exit(1)

    logger = trt.Logger(trt.Logger.INFO)
    runtime = trt.Runtime(logger)
    with open(args.engine, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize engine")

    int8_layers = 0
    total_layers = engine.num_layers
    for i in range(total_layers):
        layer = engine.get_layer(i)
        if layer.precision == trt.DataType.INT8:
            int8_layers += 1

    print(f"INT8 layers: {int8_layers}/{total_layers}")
    if int8_layers <= 0:
        raise SystemExit("未命中整数核，疑似 QDQ 仿真")


if __name__ == "__main__":
    main()

