"""
UniDepth V2 baseline wrapper.

Uses the official HuggingFace model:
    lpiccinelli/unidepth-v2-vitl14

Requires: torch, einops, transformers, huggingface_hub
(already in environment.yml)
"""

import torch
import torch.nn.functional as F

from model.baselines.base import BaseDepthModel
from model.baselines.registry import register_baseline


class UniDepthV2Wrapper(BaseDepthModel):
    """
    Wraps the official UniDepthV2 pretrained model from HuggingFace.

    The model predicts metric depth directly — no scale alignment needed.
    """

    def __init__(self, model_id: str = "lpiccinelli/unidepth-v2-vitl14", device=None):
        super().__init__()
        from unidepth.models import UniDepthV2

        self.model = UniDepthV2.from_pretrained(model_id)
        self._device = device or torch.device("cpu")

    @torch.no_grad()
    def predict_depth(self, rgb: torch.Tensor, intrinsics: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) uint8 or float [0,1].
            intrinsics: (B, 3, 3) or (3, 3) optional camera intrinsics.
        Returns:
            depth: (B, 1, H, W) float32 metres.
        """
        # UniDepth V2 infer() expects uint8-range (0-255) input:
        # it internally does rgb.float() / 255.0 then ImageNet normalisation.
        if rgb.dtype != torch.uint8:
            rgb_255 = (rgb * 255.0).clamp(0, 255).to(torch.uint8)
        else:
            rgb_255 = rgb

        B, _, H, W = rgb_255.shape
        depths = []
        for i in range(B):
            cam = None
            if intrinsics is not None:
                cam = intrinsics[i] if intrinsics.ndim == 3 else intrinsics
                cam = cam.to(self._device)
            pred = self.model.infer(rgb_255[i : i + 1].to(self._device), camera=cam)
            d = pred["depth"]  # (1, 1, h, w)
            if d.shape[-2:] != (H, W):
                d = F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False)
            depths.append(d)
        return torch.cat(depths, dim=0)

    @property
    def device(self):
        return self._device

    def to(self, device, *args, **kwargs):
        self._device = device
        self.model = self.model.to(device)
        return self


@register_baseline("unidepthv2")
def _build_unidepthv2(device, model_id="lpiccinelli/unidepth-v2-vitl14", **kwargs):
    return UniDepthV2Wrapper(model_id=model_id, device=device)
