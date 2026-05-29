#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Target:
    name: str
    runner: Path
    engine: Path
    plugin: Path | None = None


METRIC_RE = re.compile(r"avg_latency_ms=([0-9.]+)\s+throughput_images_per_s=([0-9.]+)")


def run_target(target: Target, batch: int, warmup: int, iters: int, cuda_graph: bool) -> tuple[float, float, str]:
    command = [
        str(target.runner),
        "--engine",
        str(target.engine),
        "--batch",
        str(batch),
        "--warmup",
        str(warmup),
        "--iters",
        str(iters),
    ]
    if target.plugin is not None:
        command.extend(["--plugin", str(target.plugin)])
    if not cuda_graph:
        command.append("--no-cuda-graph")

    completed = subprocess.run(command, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = completed.stdout
    if completed.returncode != 0:
        raise RuntimeError(f"{target.name} failed with exit code {completed.returncode}\n{output}")
    match = METRIC_RE.search(output)
    if match is None:
        raise RuntimeError(f"Could not parse benchmark metrics for {target.name}\n{output}")
    return float(match.group(1)), float(match.group(2)), output


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Dense MoE and Sparse Plugin MoE TensorRT engines.")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--show-output", action="store_true")
    args = parser.parse_args()

    root = args.root
    targets = [
        Target(
            name="Dense MoE",
            runner=root / "build" / "moe_trt_infer",
            engine=root / "artifacts" / "bench_b128_dense.engine",
        ),
        Target(
            name="Sparse Plugin MoE",
            runner=root / "build_plugin" / "moe_trt_infer",
            engine=root / "artifacts" / "bench_b128_plugin.engine",
            plugin=root / "build_plugin" / "libcustom_moe_plugin.so",
        ),
        Target(
            name="Sparse Plugin MoE (CUTLASS)",
            runner=root / "build_cutlass" / "moe_trt_infer",
            engine=root / "artifacts" / "bench_b128_plugin_cutlass.engine",
            plugin=root / "build_cutlass" / "libcustom_moe_plugin.so",
        ),
    ]

    print(f"batch={args.batch} warmup={args.warmup} iters={args.iters} cuda_graph={not args.no_cuda_graph}")
    print("| target | avg_latency_ms | throughput_images_per_s |")
    print("| --- | ---: | ---: |")
    for target in targets:
        latency, throughput, output = run_target(target, args.batch, args.warmup, args.iters, not args.no_cuda_graph)
        print(f"| {target.name} | {latency:.6f} | {throughput:.3f} |")
        if args.show_output:
            print(output.rstrip())


if __name__ == "__main__":
    main()