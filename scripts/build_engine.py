#!/usr/bin/env python3
"""Build a TensorRT engine from ONNX.

For plugin-mode ONNX, load libcustom_moe_plugin.so before parsing.
"""
from __future__ import annotations

import argparse
import ctypes
from pathlib import Path

import tensorrt as trt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", type=Path, required=True)
    p.add_argument("--engine", type=Path, required=True)
    p.add_argument("--plugin", type=Path, default=None, help="Path to libcustom_moe_plugin.so")
    p.add_argument("--min-batch", type=int, default=1)
    p.add_argument("--opt-batch", type=int, default=8)
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--channels", type=int, default=3)
    p.add_argument("--height", type=int, default=32)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--workspace-gb", type=float, default=4.0)
    p.add_argument("--version-compatible", action="store_true")
    args = p.parse_args()

    logger = trt.Logger(trt.Logger.INFO)
    trt.init_libnvinfer_plugins(logger, "")
    if args.plugin is not None:
        ctypes.CDLL(args.plugin.as_posix(), mode=ctypes.RTLD_GLOBAL)
        print(f"[INFO] loaded plugin: {args.plugin}")

    explicit_batch = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    builder = trt.Builder(logger)
    network = builder.create_network(explicit_batch)
    parser = trt.OnnxParser(network, logger)

    data = args.onnx.read_bytes()
    if not parser.parse(data):
        print("[ERROR] ONNX parse failed")
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit(1)

    config = builder.create_builder_config()
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(args.workspace_gb * (1 << 30)))
    else:
        config.max_workspace_size = int(args.workspace_gb * (1 << 30))

    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("[INFO] enabled FP16")
    elif args.fp16:
        print("[WARN] requested FP16, but platform_has_fast_fp16 is false")

    if args.version_compatible and hasattr(trt.BuilderFlag, "VERSION_COMPATIBLE"):
        config.set_flag(trt.BuilderFlag.VERSION_COMPATIBLE)
        if hasattr(parser, "get_used_vc_plugin_libraries") and hasattr(config, "set_plugins_to_serialize"):
            libs = parser.get_used_vc_plugin_libraries()
            if libs:
                config.set_plugins_to_serialize(libs)
                print(f"[INFO] serializing plugin libraries into engine: {libs}")

    inp = network.get_input(0)
    profile = builder.create_optimization_profile()
    min_shape = (args.min_batch, args.channels, args.height, args.width)
    opt_shape = (args.opt_batch, args.channels, args.height, args.width)
    max_shape = (args.max_batch, args.channels, args.height, args.width)
    profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    print(f"[INFO] optimization profile for {inp.name}: min={min_shape}, opt={opt_shape}, max={max_shape}")

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    args.engine.parent.mkdir(parents=True, exist_ok=True)
    args.engine.write_bytes(bytes(serialized))
    print(f"[OK] wrote {args.engine}")


if __name__ == "__main__":
    main()
