import torch
import torch.nn as nn

from model.backbones import ConvNeXt, ConvNeXtV2

# ──────────────────────────────────────────────────────────────────────────────
# Encoder factory functions
# ──────────────────────────────────────────────────────────────────────────────
# Each factory reads ``config.get("output_idx", <default>)`` where <default>
# is the correct value for that specific backbone architecture.
#
# output_idx semantics (ConvNeXt family)
# ───────────────────────────────────────
# ConvNeXt / ConvNeXtV2 with depths=[3,3,27,3] has 36 blocks total:
#   stage 0: blocks  1– 3 (dim=192,  stride 4)
#   stage 1: blocks  4– 6 (dim=384,  stride 8)
#   stage 2: blocks  7–33 (dim=768,  stride 16)
#   stage 3: blocks 34–36 (dim=1536, stride 32)
# output_idx=[3, 6, 33, 36] → cumulative endpoint of each stage.
# The decoder groups encoder outputs by these boundaries and max-pools
# within each group via max_stack().
#
# output_idx semantics (DINOv2 ViT family)
# ─────────────────────────────────────────
# DINOv2 ViT-L/14  (depth=24):  output_idx=[5, 12, 18, 24]
# DINOv2 ViT-B/14  (depth=12):  output_idx=[3,  6,  9, 12]
# DINOv2 ViT-S/14  (depth=12):  output_idx=[3,  6,  9, 12]
# All ViT blocks share the same spatial resolution (H/14 × W/14).
# The decoder handles this via the ``if len(level_shapes) == 1`` branch.
# ──────────────────────────────────────────────────────────────────────────────

class ModelWrap(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.backbone = model

    def forward(self, x, *args, **kwargs):
        features = []
        for layer in self.backbone.features:
            x = layer(x)
            features.append(x)
        return features


def convnextv2_base(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[128, 256, 512, 1024],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_base_22k_384_ema.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_large(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_large_22k_384_ema.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_large_mae(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_large_1k_224_fcmae.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_huge(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[352, 704, 1408, 2816],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/im22k/convnextv2_huge_22k_512_ema.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnextv2_huge_mae(config, **kwargs):
    model = ConvNeXtV2(
        depths=[3, 3, 27, 3],
        dims=[352, 704, 1408, 2816],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    url = "https://dl.fbaipublicfiles.com/convnext/convnextv2/pt_only/convnextv2_huge_1k_224_fcmae.pt"
    state_dict = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", progress=False
    )["model"]
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnext_large_pt(config, **kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        **kwargs,
    )
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import disable_progress_bars

    from model.backbones.convnext import HF_URL, checkpoint_filter_fn

    disable_progress_bars()
    repo_id, filename = HF_URL["convnext_large_pt"]
    state_dict = torch.load(hf_hub_download(repo_id=repo_id, filename=filename))
    state_dict = checkpoint_filter_fn(state_dict, model)
    info = model.load_state_dict(state_dict, strict=False)
    print(info)
    return model


def convnext_large(config, **kwargs):
    model = ConvNeXt(
        depths=[3, 3, 27, 3],
        dims=[192, 384, 768, 1536],
        output_idx=config.get("output_idx", [3, 6, 33, 36]),
        use_checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        **kwargs,
    )
    return model


def dinov3_vits16(config, **kwargs):
    from model.backbones.dinov3 import _make_dinov3_model
    return _make_dinov3_model(
        arch_name="vit_small",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        **kwargs,
    )


def dinov3_vitb16(config, **kwargs):
    from model.backbones.dinov3 import _make_dinov3_model
    return _make_dinov3_model(
        arch_name="vit_base",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        **kwargs,
    )


def dinov3_vitl16(config, **kwargs):
    from model.backbones.dinov3 import _make_dinov3_model
    return _make_dinov3_model(
        arch_name="vit_large",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [5, 12, 18, 24]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        **kwargs,
    )