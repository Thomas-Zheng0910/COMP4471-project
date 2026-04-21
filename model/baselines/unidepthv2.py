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
    def predict_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) uint8 or float [0,1].
        Returns:
            depth: (B, 1, H, W) float32 metres.
        """
        if rgb.dtype == torch.uint8:
            rgb = rgb.float() / 255.0

        B, _, H, W = rgb.shape
        depths = []
        for i in range(B):
            pred = self.model.infer(rgb[i : i + 1].to(self._device))
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
