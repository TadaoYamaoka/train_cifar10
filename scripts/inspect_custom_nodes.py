#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import onnx


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("onnx", type=Path)
    args = p.parse_args()
    m = onnx.load(args.onnx.as_posix())
    total = 0
    for n in m.graph.node:
        if n.op_type == "CustomMoE" or n.domain == "trt.plugins":
            total += 1
            attrs = {a.name: onnx.helper.get_attribute_value(a) for a in n.attribute}
            print(f"node={n.name or '<unnamed>'} domain={n.domain} op={n.op_type} inputs={list(n.input)} attrs={attrs}")
    print(f"CustomMoE nodes: {total}")


if __name__ == "__main__":
    main()
