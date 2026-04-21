# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import contextlib
import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from .metadinov3.layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from .metadinov3.utils import named_apply

# Not applicable 
# _DINOV3_BASE_URL = "https://dl.fbaipublicfiles.com/dinov3/models"

_DINOV3_BASE_URL = None

_DINOV3_PATH = "./model/backbones/dinov3_weights"

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()


class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: Optional[float] = None,
        pos_embed_rope_max_period: Optional[float] = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: Optional[float] = None,
        pos_embed_rope_jitter_coords: Optional[float] = None,
        pos_embed_rope_rescale_coords: Optional[float] = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: Optional[float] = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Optional[Any] = None,
        output_idx: List[int] | None = None,   # 1-based block ids
        use_norm: bool = False,
        frozen_stages: int = -1,               # -1: freeze nothing, 0: freeze patch_embed only
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.use_norm = use_norm
        self.frozen_stages = frozen_stages

        if output_idx is None:
            if depth >= 4:
                output_idx = [
                    max(1, depth // 4),
                    max(1, depth // 2),
                    max(1, (3 * depth) // 4),
                    depth,
                ]
            else:
                output_idx = list(range(1, depth + 1))
        output_idx = sorted(set(output_idx))
        if not all(1 <= idx <= depth for idx in output_idx):
            raise ValueError(f"output_idx must be in [1, {depth}], got {output_idx}")

        self.depths = output_idx
        self._output_idx = set(output_idx)
        self.embed_dims = [embed_dim] * output_idx[-1]

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(
                torch.empty(1, n_storage_tokens, embed_dim, device=device)
            )
        else:
            self.storage_tokens = None

        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )

        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for _ in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        self.cls_norm = norm_layer_cls(embed_dim) if untie_cls_and_patch_norms else None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        self.local_cls_norm = (
            norm_layer_cls(embed_dim) if untie_global_and_local_cls_norm else None
        )

        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        self.init_weights()

    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.storage_tokens is not None:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def _apply_output_norm(self, x: Tensor) -> Tensor:
        if not self.use_norm:
            return x

        if self.untie_cls_and_patch_norms:
            x_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
            x_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            return torch.cat((x_cls_reg, x_patch), dim=1)

        return self.norm(x)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int, int]]:
        patch_ctx = torch.no_grad() if self.frozen_stages >= 0 else contextlib.nullcontext()
        with patch_ctx:
            x = self.patch_embed(x)

        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            masks = masks.bool().reshape(B, -1, 1)
            x = torch.where(masks, self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token

        if self.storage_tokens is None:
            storage_tokens = x.new_empty(1, 0, self.embed_dim)
        else:
            storage_tokens = self.storage_tokens

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )
        return x, (H, W)

    def forward(self, x: Tensor, masks=None) -> Tuple[List[Tensor], List[Tensor]]:
        batch_size = x.shape[0]
        x, (H, W) = self.prepare_tokens_with_masks(x, masks)

        outputs: List[Tensor] = []
        class_tokens: List[Tensor] = []

        for i, blk in enumerate(self.blocks):
            rope_sincos = self.rope_embed(H=H, W=W) if self.rope_embed is not None else None

            block_ctx = (
                torch.no_grad()
                if self.frozen_stages > 0 and i < self.frozen_stages
                else contextlib.nullcontext()
            )
            with block_ctx:
                x = blk(x, rope_sincos)

            out = self._apply_output_norm(x)
            class_tokens.append(out[:, :1])
            patch_tokens = out[:, self.n_storage_tokens + 1 :]
            outputs.append(patch_tokens.reshape(batch_size, H, W, -1))

        return outputs, class_tokens

    def freeze(self) -> None:
        for module in self.modules():
            module.eval()
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for p in self.patch_embed.parameters():
                p.requires_grad = False
        else:
            for p in self.patch_embed.parameters():
                p.requires_grad = True

        for i, blk in enumerate(self.blocks):
            if i < self.frozen_stages:
                blk.eval()
                for p in blk.parameters():
                    p.requires_grad = False
            else:
                for p in blk.parameters():
                    p.requires_grad = True

        self.cls_token.requires_grad = self.frozen_stages < 1
        self.mask_token.requires_grad = False

        if self.storage_tokens is not None:
            self.storage_tokens.requires_grad = False

        if self.rope_embed is not None:
            for p in self.rope_embed.parameters():
                p.requires_grad = self.frozen_stages < 1

        for norm in (self.norm, self.cls_norm, self.local_cls_norm):
            if norm is not None:
                for p in norm.parameters():
                    p.requires_grad = self.use_norm and (self.frozen_stages < len(self.blocks))


def vit_small(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model

def _make_dinov3_model_name(arch_name: str, patch_size: int) -> str:
    compact_arch_name = arch_name.replace("_", "")[:4]
    return f"dinov3_{compact_arch_name}{patch_size}"

def _make_dinov3_model(
    *,
    arch_name: str = "vit_small",
    img_size: int = 518,
    patch_size: int = 16,
    init_values: float = 1.0,
    ffn_layer: str = "mlp",
    block_chunks: int = 0,
    pretrained: str = "",
    output_idx: Sequence[int] = [],
    num_register_tokens: int = 0,
    drop_path_rate: float = 0.0,
    use_norm: bool = False,
    export: bool = False,
    interpolate_offset: float = 0.0,
    frozen_stages: int = 0,
    **kwargs,
):
    model_name = _make_dinov3_model_name(arch_name, patch_size)
    vit_kwargs = dict(
        img_size=img_size,
        patch_size=patch_size,
        init_values=init_values,
        ffn_layer=ffn_layer,
        block_chunks=block_chunks,
        output_idx=output_idx,
        drop_path_rate=drop_path_rate,
        num_register_tokens=num_register_tokens,
        use_norm=use_norm,
        export=export,
        interpolate_offset=interpolate_offset,
        frozen_stages=frozen_stages,
    )
    vit_kwargs.update(**kwargs)
    model = eval(arch_name)(**vit_kwargs)
    if pretrained == "" and _DINOV3_BASE_URL is not None:
        url = _DINOV3_BASE_URL + f"/{model_name}/{model_name}"
        if num_register_tokens > 0:
            url += "_reg4"
        url += "_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", progress=False
        )
        info = model.load_state_dict(state_dict, strict=False)
        print(info)
    elif pretrained is not None and pretrained != "":
        state_dict = torch.load(pretrained, map_location="cpu")
        info = model.load_state_dict(state_dict, strict=False)
        print(f"loading from {pretrained} with:", info)
    else:
        print("Not loading pretrained weights for backbone")

    return model