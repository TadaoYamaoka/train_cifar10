#!/usr/bin/env python3
"""Export the attached Swin + Dropless MoE model to ONNX.

Two export modes are provided:
  1. plugin: each DroplessMoEMlp becomes a single ONNX custom node
     trt.plugins::CustomMoE. This is intended for the TensorRT plugin in
     plugins/.
  2. dense: each DroplessMoEMlp becomes a dense-all-experts ONNX subgraph
     using standard operators. This is useful as a correctness/performance
     baseline and avoids custom plugins for small expert counts.

The original model keeps router weights in FP32 and uses expert weights as
3-D tensors [E, D, H] and [E, H, D]. This script preserves that contract.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class CustomMoEFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        x: Tensor,
        router_w: Tensor,
        w1: Tensor,
        b1: Tensor,
        w2: Tensor,
        b2: Tensor,
        top_k: int,
        num_experts: int,
        in_features: int,
        hidden_features: int,
    ) -> Tensor:
        # During ONNX tracing the symbolic() method below emits the actual
        # CustomMoE node. The eager value is needed only for shape propagation,
        # so avoid tracing the expensive dynamic routing reference here.
        return torch.empty_like(x)

    @staticmethod
    def symbolic(
        g: Any,
        x: Any,
        router_w: Any,
        w1: Any,
        b1: Any,
        w2: Any,
        b2: Any,
        top_k: int,
        num_experts: int,
        in_features: int,
        hidden_features: int,
    ) -> Any:
        out = g.op(
            "trt.plugins::CustomMoE",
            x,
            router_w,
            w1,
            b1,
            w2,
            b2,
            top_k_i=int(top_k),
            num_experts_i=int(num_experts),
            in_features_i=int(in_features),
            hidden_features_i=int(hidden_features),
            activation_s="gelu",
            plugin_version_s="1",
            plugin_namespace_s="",
            outputs=1,
        )
        out.setType(x.type())
        return out


class PluginMoEExportWrapper(nn.Module):
    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.in_features = int(src.in_features)
        self.hidden_features = int(src.hidden_features)
        self.num_experts = int(src.num_experts)
        self.top_k = int(src.top_k)
        # Keep these as registered parameters so torch.onnx.export writes them
        # as ONNX initializers connected to the custom node.
        self.router_weight = src.router.weight
        self.w1 = src.w1
        self.b1 = src.b1
        self.w2 = src.w2
        self.b2 = src.b2

    def forward(self, x: Tensor) -> Tensor:
        return CustomMoEFunction.apply(
            x,
            self.router_weight,
            self.w1,
            self.b1,
            self.w2,
            self.b2,
            self.top_k,
            self.num_experts,
            self.in_features,
            self.hidden_features,
        )


class DenseMoEExportWrapper(nn.Module):
    """Dense-all-experts ONNX fallback for small E or plugin bring-up.

    This computes all experts for every token and masks unselected experts
    with the top-k gate. It does more arithmetic than sparse routing, but it
    avoids token packing and custom kernels. For the attached default model
    (E=4, top_k=2, CIFAR-sized token counts), it is a useful baseline.
    """

    def __init__(self, src: nn.Module) -> None:
        super().__init__()
        self.in_features = int(src.in_features)
        self.hidden_features = int(src.hidden_features)
        self.num_experts = int(src.num_experts)
        self.top_k = int(src.top_k)
        self.router = src.router
        self.w1 = src.w1
        self.b1 = src.b1
        self.w2 = src.w2
        self.b2 = src.b2

    def forward(self, x: Tensor) -> Tensor:
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        with torch.autocast(device_type=x.device.type, enabled=False):
            logits = self.router(x_flat.float())
            probs = F.softmax(logits, dim=-1)
            topk_prob, topk_expert = torch.topk(probs, k=self.top_k, dim=-1)
            topk_gate = topk_prob / topk_prob.sum(dim=-1, keepdim=True)

        gates = torch.zeros(x_flat.shape[0], self.num_experts, dtype=x_flat.dtype, device=x_flat.device)
        gates.scatter_add_(1, topk_expert, topk_gate.to(dtype=x_flat.dtype))

        w1 = self.w1.to(dtype=x_flat.dtype)
        b1 = self.b1.to(dtype=x_flat.dtype)
        w2 = self.w2.to(dtype=x_flat.dtype)
        b2 = self.b2.to(dtype=x_flat.dtype)

        # [M, D] x [E, D, H] -> [M, E, H]
        h = torch.einsum("md,edh->meh", x_flat, w1) + b1.unsqueeze(0)
        h = F.gelu(h)
        # [M, E, H] x [E, H, D] -> [M, E, D]
        y_experts = torch.einsum("meh,ehd->med", h, w2) + b2.unsqueeze(0)
        y = (y_experts * gates.unsqueeze(-1)).sum(dim=1)
        return y.reshape(orig_shape)


def install_optional_import_stubs() -> None:
    # The attached file imports Lightning/TorchMetrics for training, but export
    # only needs the plain nn.Module model classes. Provide tiny stubs when those
    # packages are not installed.
    if importlib.util.find_spec("lightning") is None:
        lightning = types.ModuleType("lightning")
        pytorch = types.ModuleType("lightning.pytorch")
        pytorch.LightningModule = nn.Module
        lightning.pytorch = pytorch
        sys.modules.setdefault("lightning", lightning)
        sys.modules.setdefault("lightning.pytorch", pytorch)
    if importlib.util.find_spec("torchmetrics") is None:
        torchmetrics = types.ModuleType("torchmetrics")
        classification = types.ModuleType("torchmetrics.classification")

        class _DummyMetric(nn.Module):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__()
            def forward(self, *args: Any, **kwargs: Any) -> Tensor:
                return torch.tensor(0.0)
            __call__ = forward

        classification.MulticlassAccuracy = _DummyMetric
        torchmetrics.classification = classification
        sys.modules.setdefault("torchmetrics", torchmetrics)
        sys.modules.setdefault("torchmetrics.classification", classification)


def load_module(model_path: Path) -> Any:
    install_optional_import_stubs()
    spec = importlib.util.spec_from_file_location("attached_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import model from {model_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["attached_model"] = module
    spec.loader.exec_module(module)
    return module


def set_child(parent: nn.Module, name: str, new_child: nn.Module) -> None:
    parts = name.split(".")
    obj = parent
    for part in parts[:-1]:
        obj = getattr(obj, part)
    setattr(obj, parts[-1], new_child)


def replace_moe(model: nn.Module, model_module: Any, mode: str, fp16_experts: bool) -> int:
    replaced = 0
    for name, child in list(model.named_modules()):
        if isinstance(child, model_module.DroplessMoEMlp):
            if fp16_experts:
                child.w1.data = child.w1.data.half()
                child.b1.data = child.b1.data.half()
                child.w2.data = child.w2.data.half()
                child.b2.data = child.b2.data.half()
                child.router.float()
            wrapper: nn.Module
            if mode == "plugin":
                wrapper = PluginMoEExportWrapper(child)
            elif mode == "dense":
                wrapper = DenseMoEExportWrapper(child)
            else:
                raise ValueError(f"Unsupported mode: {mode}")
            set_child(model, name, wrapper)
            replaced += 1
    return replaced


def build_model(model_module: Any, args: argparse.Namespace) -> nn.Module:
    model = model_module.SwinTransformer(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        num_classes=args.num_classes,
        embed_dim=args.embed_dim,
        depths=tuple(args.depths),
        num_heads=tuple(args.num_heads),
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        moe_stages=tuple(args.moe_stages),
        moe_num_experts=args.moe_num_experts,
        moe_top_k=args.moe_top_k,
    )
    return model


def load_checkpoint(model: nn.Module, ckpt_path: Path, lightning: bool) -> None:
    obj = torch.load(ckpt_path, map_location="cpu")
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    if lightning:
        fixed = {}
        for k, v in state.items():
            if k.startswith("model."):
                fixed[k[len("model."):]] = v
            else:
                fixed[k] = v
        state = fixed
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] missing keys: {len(missing)}")
        for k in missing[:10]:
            print(f"  missing: {k}")
    if unexpected:
        print(f"[WARN] unexpected keys: {len(unexpected)}")
        for k in unexpected[:10]:
            print(f"  unexpected: {k}")


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-py", type=Path, default=Path(__file__).resolve().parents[1] / "model.py")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--lightning-checkpoint", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("model_moe_plugin.onnx"))
    parser.add_argument("--mode", choices=["plugin", "dense"], default="plugin")
    parser.add_argument("--fp16-experts", action="store_true", help="Export expert weights/biases in FP16; router remains FP32.")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--depths", type=parse_int_list, default=parse_int_list("2,2,2,2"))
    parser.add_argument("--num-heads", type=parse_int_list, default=parse_int_list("2,4,8,16"))
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--moe-stages", type=parse_int_list, default=parse_int_list("0,1"))
    parser.add_argument("--moe-num-experts", type=int, default=4)
    parser.add_argument("--moe-top-k", type=int, default=2)
    args = parser.parse_args()

    model_module = load_module(args.model_py)
    model = build_model(model_module, args).eval()
    if args.checkpoint is not None:
        load_checkpoint(model, args.checkpoint, args.lightning_checkpoint)

    replaced = replace_moe(model, model_module, args.mode, args.fp16_experts)
    if replaced == 0:
        raise RuntimeError("No DroplessMoEMlp modules were found/replaced.")
    print(f"[INFO] replaced {replaced} DroplessMoEMlp modules with {args.mode} export wrappers")

    dummy = torch.randn(args.batch, args.in_channels, args.image_size, args.image_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    custom_opsets = {"trt.plugins": 1} if args.mode == "plugin" else None
    export_kwargs = dict(
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        custom_opsets=custom_opsets,
    )
    with torch.no_grad():
        try:
            torch.onnx.export(model, dummy, args.output.as_posix(), dynamo=False, **export_kwargs)
        except TypeError as exc:
            # Older PyTorch versions do not have the dynamo argument.
            if "dynamo" not in str(exc):
                raise
            torch.onnx.export(model, dummy, args.output.as_posix(), **export_kwargs)
    print(f"[OK] wrote {args.output}")
    if args.mode == "plugin":
        print("[INFO] CustomMoE nodes are in domain trt.plugins. Build with libcustom_moe_plugin.so loaded.")


if __name__ == "__main__":
    main()
