from __future__ import annotations

from typing import Sequence

import lightning.pytorch as pl
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchmetrics.classification import MulticlassAccuracy


def window_partition(x: Tensor, window_size: int) -> Tensor:
    batch_size, height, width, channels = x.shape
    x = x.view(batch_size, height // window_size, window_size, width // window_size, window_size, channels)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return windows.view(-1, window_size, window_size, channels)


def window_reverse(windows: Tensor, window_size: int, height: int, width: int) -> Tensor:
    batch_size = int(windows.shape[0] / (height * width / window_size / window_size))
    x = windows.view(batch_size, height // window_size, width // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(batch_size, height, width, -1)


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: Tensor) -> Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, drop: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DroplessMoEMlp(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        num_experts: int,
        top_k: int = 2,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be greater than 0.")
        if top_k <= 0 or top_k > num_experts:
            raise ValueError("top_k must be in the range [1, num_experts].")

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(in_features, num_experts, bias=False)
        # grouped_mm operand layout avoids per-forward transpose: x @ w1 maps D -> H, h @ w2 maps H -> D.
        self.w1 = nn.Parameter(torch.empty(num_experts, in_features, hidden_features))
        self.b1 = nn.Parameter(torch.zeros(num_experts, hidden_features))
        self.w2 = nn.Parameter(torch.empty(num_experts, hidden_features, in_features))
        self.b2 = nn.Parameter(torch.zeros(num_experts, in_features))
        self.act = nn.GELU()
        self.drop = nn.Dropout(drop)
        self.aux_loss: Tensor | None = None
        self.z_loss: Tensor | None = None

        nn.init.trunc_normal_(self.w1, std=0.02)
        nn.init.trunc_normal_(self.w2, std=0.02)

    def _apply(self, fn):
        module = super()._apply(fn)
        self.router.float()
        return module

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_cuda:
            raise RuntimeError("DroplessMoEMlp requires CUDA tensors because it uses torch.nn.functional.grouped_mm.")
        if not hasattr(F, "grouped_mm"):
            raise RuntimeError("torch.nn.functional.grouped_mm is not available in this PyTorch build.")
        if self.router.weight.dtype != torch.float32:
            raise RuntimeError("Router weights must remain FP32 for stable routing.")

        orig_shape = x.shape
        x = x.reshape(-1, self.in_features)
        tokens = x.shape[0]

        # Keep routing numerics in FP32; expert grouped_mm below still uses the activation dtype.
        with torch.autocast(device_type="cuda", enabled=False):
            logits = self.router(x.float())
            probs = F.softmax(logits, dim=-1)
            topk_prob, topk_expert = torch.topk(probs, k=self.top_k, dim=-1)
            topk_gate = topk_prob / topk_prob.sum(dim=-1, keepdim=True)

        flat_expert = topk_expert.reshape(-1)
        flat_gate = topk_gate.reshape(-1)
        flat_token = torch.arange(tokens, device=x.device).repeat_interleave(self.top_k)

        order = torch.argsort(flat_expert)
        expert_sorted = flat_expert[order]
        token_sorted = flat_token[order]
        gate_sorted = flat_gate[order]
        x_sorted = x.index_select(0, token_sorted)

        counts = torch.bincount(expert_sorted, minlength=self.num_experts)
        offsets = torch.cumsum(counts, dim=0).to(torch.int32)
        tokens_per_expert = counts.to(probs.dtype) / counts.sum().to(probs.dtype)
        prob_per_expert = probs.mean(dim=0)
        self.aux_loss = self.num_experts * torch.sum(tokens_per_expert * prob_per_expert)
        self.z_loss = torch.mean(torch.logsumexp(logits, dim=-1).square())

        w1 = self.w1 if self.w1.dtype == x.dtype else self.w1.to(dtype=x.dtype)

        # grouped_mm requires offs[-1] < mat_a.shape[0], so append one ignored row.
        padding = torch.zeros(1, x_sorted.shape[-1], dtype=x_sorted.dtype, device=x_sorted.device)
        x_grouped = torch.cat((x_sorted, padding), dim=0)
        h = F.grouped_mm(x_grouped, w1, offs=offsets)[: x_sorted.shape[0]]
        h = h + self.b1.index_select(0, expert_sorted).to(dtype=h.dtype)
        h = self.act(h)
        h = self.drop(h)

        w2 = self.w2 if self.w2.dtype == h.dtype else self.w2.to(dtype=h.dtype)

        # grouped_mm requires offs[-1] < mat_a.shape[0], so append one ignored row.
        hidden_padding = torch.zeros(1, h.shape[-1], dtype=h.dtype, device=h.device)
        h_grouped = torch.cat((h, hidden_padding), dim=0)
        y_sorted = F.grouped_mm(h_grouped, w2, offs=offsets)[: h.shape[0]]
        y_sorted = y_sorted + self.b2.index_select(0, expert_sorted).to(dtype=y_sorted.dtype)
        y_sorted = self.drop(y_sorted)
        y_sorted = y_sorted * gate_sorted.unsqueeze(-1).to(dtype=y_sorted.dtype)

        y = torch.zeros_like(x)
        y.index_add_(0, token_sorted, y_sorted.to(dtype=y.dtype))
        return y.reshape(orig_shape)


class WindowAttention(nn.Module):
    def __init__(self, dim: int, window_size: int, num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.0, proj_drop: float = 0.0) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError("num_heads must be greater than 0.")
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads.")
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        table_size = (2 * window_size - 1) * (2 * window_size - 1)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(table_size, num_heads))

        coords = torch.stack(torch.meshgrid(torch.arange(window_size), torch.arange(window_size), indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index, persistent=False)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> Tensor:
        batch_windows, tokens, channels = x.shape
        qkv = self.qkv(x).reshape(batch_windows, tokens, 3, self.num_heads, channels // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv[0], qkv[1], qkv[2]

        query = query * self.scale
        attn = query @ key.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)]
        relative_position_bias = relative_position_bias.view(tokens, tokens, -1).permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            num_windows = mask.shape[0]
            attn = attn.view(batch_windows // num_windows, num_windows, self.num_heads, tokens, tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, tokens, tokens)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ value).transpose(1, 2).reshape(batch_windows, tokens, channels)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        num_heads: int,
        window_size: int = 4,
        shift_size: int = 0,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
        use_moe_mlp: bool = False,
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.window_size = min(window_size, input_resolution[0], input_resolution[1])
        if input_resolution[0] % self.window_size != 0 or input_resolution[1] % self.window_size != 0:
            raise ValueError("input_resolution must be divisible by window_size, or padding is required.")
        self.shift_size = 0 if min(input_resolution) <= self.window_size else shift_size

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, self.window_size, num_heads, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        hidden_features = int(dim * mlp_ratio)
        if use_moe_mlp:
            self.mlp = DroplessMoEMlp(dim, hidden_features, num_experts=moe_num_experts, top_k=moe_top_k, drop=drop)
        else:
            self.mlp = Mlp(dim, hidden_features, drop=drop)

        self.register_buffer("attn_mask", self._create_mask(), persistent=False)

    def _create_mask(self) -> Tensor | None:
        if self.shift_size == 0:
            return None

        height, width = self.input_resolution
        img_mask = torch.zeros((1, height, width, 1))
        height_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        width_slices = (slice(0, -self.window_size), slice(-self.window_size, -self.shift_size), slice(-self.shift_size, None))
        count = 0
        for height_slice in height_slices:
            for width_slice in width_slices:
                img_mask[:, height_slice, width_slice, :] = count
                count += 1

        mask_windows = window_partition(img_mask, self.window_size).view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x: Tensor) -> Tensor:
        height, width = self.input_resolution
        batch_size, length, channels = x.shape
        if length != height * width:
            raise ValueError(f"Expected token length {height * width}, got {length}.")

        shortcut = x
        x = self.norm1(x).view(batch_size, height, width, channels)

        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        x_windows = window_partition(shifted_x, self.window_size).view(-1, self.window_size * self.window_size, channels)
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        shifted_x = window_reverse(attn_windows.view(-1, self.window_size, self.window_size, channels), self.window_size, height, width)

        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x

        x = x.view(batch_size, height * width, channels)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution: tuple[int, int], dim: int) -> None:
        super().__init__()
        self.input_resolution = input_resolution
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: Tensor) -> Tensor:
        height, width = self.input_resolution
        batch_size, length, channels = x.shape
        if length != height * width:
            raise ValueError(f"Expected token length {height * width}, got {length}.")
        if height % 2 != 0 or width % 2 != 0:
            raise ValueError("PatchMerging requires even spatial dimensions.")

        x = x.view(batch_size, height, width, channels)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(batch_size, -1, 4 * channels)
        x = self.norm(x)
        return self.reduction(x)


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        input_resolution: tuple[int, int],
        depth: int,
        num_heads: int,
        window_size: int,
        mlp_ratio: float,
        drop: float,
        attn_drop: float,
        drop_path: Sequence[float],
        downsample: bool,
        use_moe_mlp: bool,
        moe_num_experts: int,
        moe_top_k: int,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                SwinTransformerBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if index % 2 == 0 else window_size // 2,
                    mlp_ratio=mlp_ratio,
                    drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[index],
                    use_moe_mlp=use_moe_mlp,
                    moe_num_experts=moe_num_experts,
                    moe_top_k=moe_top_k,
                )
                for index in range(depth)
            ]
        )
        self.downsample = PatchMerging(input_resolution, dim) if downsample else None

    def moe_loss(self) -> Tensor | None:
        losses = [block.mlp.aux_loss for block in self.blocks if isinstance(block.mlp, DroplessMoEMlp) and block.mlp.aux_loss is not None]
        if not losses:
            return None
        return torch.stack(losses).mean()

    def moe_z_loss(self) -> Tensor | None:
        losses = [block.mlp.z_loss for block in self.blocks if isinstance(block.mlp, DroplessMoEMlp) and block.mlp.z_loss is not None]
        if not losses:
            return None
        return torch.stack(losses).mean()

    def forward(self, x: Tensor) -> Tensor:
        for block in self.blocks:
            x = block(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_channels: int, embed_dim: int) -> None:
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: Tensor) -> Tensor:
        _, _, height, width = x.shape
        if height != self.image_size or width != self.image_size:
            raise ValueError(f"Expected input size {self.image_size}x{self.image_size}, got {height}x{width}.")
        x = self.proj(x).flatten(2).transpose(1, 2)
        return self.norm(x)


class SwinTransformer(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 64,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (2, 4, 8, 16),
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        moe_stages: Sequence[int] = (0, 1),
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size.")
        if len(depths) != len(num_heads):
            raise ValueError("depths and num_heads must have the same length.")
        moe_stage_set = set(moe_stages)
        invalid_moe_stages = moe_stage_set.difference(range(len(depths)))
        if invalid_moe_stages:
            raise ValueError(f"moe_stages contains invalid stage indices: {sorted(invalid_moe_stages)}.")

        self.num_layers = len(depths)
        self.num_features = embed_dim * 2 ** (self.num_layers - 1)
        self.patch_embed = PatchEmbed(image_size, patch_size, in_channels, embed_dim)
        self.pos_drop = nn.Dropout(drop_rate)

        total_depth = sum(depths)
        drop_paths = torch.linspace(0, drop_path_rate, total_depth).tolist()

        self.layers = nn.ModuleList()
        resolution = image_size // patch_size
        depth_offset = 0
        for layer_index in range(self.num_layers):
            dim = embed_dim * 2**layer_index
            input_resolution = (resolution, resolution)
            layer = BasicLayer(
                dim=dim,
                input_resolution=input_resolution,
                depth=depths[layer_index],
                num_heads=num_heads[layer_index],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=drop_paths[depth_offset : depth_offset + depths[layer_index]],
                downsample=layer_index < self.num_layers - 1,
                use_moe_mlp=layer_index in moe_stage_set,
                moe_num_experts=moe_num_experts,
                moe_top_k=moe_top_k,
            )
            self.layers.append(layer)
            depth_offset += depths[layer_index]
            if layer_index < self.num_layers - 1:
                resolution //= 2

        self.norm = nn.LayerNorm(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)

    def moe_loss(self) -> Tensor | None:
        losses = [layer.moe_loss() for layer in self.layers]
        losses = [loss for loss in losses if loss is not None]
        if not losses:
            return None
        return torch.stack(losses).mean()

    def moe_z_loss(self) -> Tensor | None:
        losses = [layer.moe_z_loss() for layer in self.layers]
        losses = [loss for loss in losses if loss is not None]
        if not losses:
            return None
        return torch.stack(losses).mean()


class SwinCIFAR10Classifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int = 10,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dim: int = 64,
        depths: Sequence[int] = (2, 2, 2, 2),
        num_heads: Sequence[int] = (2, 4, 8, 16),
        window_size: int = 4,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
        moe_stages: Sequence[int] = (0, 1),
        moe_num_experts: int = 4,
        moe_top_k: int = 2,
        moe_aux_loss_weight: float = 0.01,
        moe_z_loss_weight: float = 0.001,
        learning_rate: float = 0.001,
        weight_decay: float = 0.05,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = SwinTransformer(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            num_classes=num_classes,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            moe_stages=moe_stages,
            moe_num_experts=moe_num_experts,
            moe_top_k=moe_top_k,
        )
        self.criterion = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes=num_classes)
        self.val_acc = MulticlassAccuracy(num_classes=num_classes)
        self.test_acc = MulticlassAccuracy(num_classes=num_classes)

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _shared_step(self, batch: tuple[Tensor, Tensor], stage: str) -> Tensor:
        images, targets = batch
        logits = self(images)
        loss = self.criterion(logits, targets)
        moe_aux_loss = self.model.moe_loss()
        if moe_aux_loss is not None:
            self.log(f"{stage}_moe_aux_loss", moe_aux_loss, prog_bar=False, on_step=stage == "train", on_epoch=True)
            if stage == "train" and self.hparams.moe_aux_loss_weight > 0.0:
                loss = loss + self.hparams.moe_aux_loss_weight * moe_aux_loss
        moe_z_loss = self.model.moe_z_loss()
        if moe_z_loss is not None:
            self.log(f"{stage}_moe_z_loss", moe_z_loss, prog_bar=False, on_step=stage == "train", on_epoch=True)
            if stage == "train" and self.hparams.moe_z_loss_weight > 0.0:
                loss = loss + self.hparams.moe_z_loss_weight * moe_z_loss
        preds = torch.argmax(logits, dim=1)
        metric = getattr(self, f"{stage}_acc")
        metric(preds, targets)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=stage == "train", on_epoch=True)
        self.log(f"{stage}_acc", metric, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
