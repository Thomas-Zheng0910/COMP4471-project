"""
Depth Anything V2 baseline wrapper.

Uses the HuggingFace pipeline:
    depth-anything/Depth-Anything-V2-Large-hf

Requires: torch, transformers, pillow
(already in environment.yml)

NOTE: Depth Anything V2 produces *relative* (affine-invariant) depth.
      The evaluation script handles scale/shift alignment via SSI/SI.
"""

import torch
import torch.nn.functional as F
from PIL import Image

from model.baselines.base import BaseDepthModel
from model.baselines.registry import register_baseline


class DepthAnythingV2Wrapper(BaseDepthModel):
    """
    Wraps Depth Anything V2 via the HuggingFace Transformers pipeline.

    Outputs relative inverse-depth; evaluation uses SSI/SI alignment.
    """

    def __init__(
        self,
        model_id: str = "depth-anything/Depth-Anything-V2-Large-hf",
        device=None,
    ):
        super().__init__()
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self._device = device or torch.device("cpu")

    @torch.no_grad()
    def predict_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) uint8 or float [0,1].
        Returns:
            depth: (B, 1, H, W) float32 relative depth (higher = closer).
        """
        if rgb.dtype != torch.uint8:
            rgb_uint8 = (rgb * 255).clamp(0, 255).to(torch.uint8)
        else:
            rgb_uint8 = rgb

        B, _, H, W = rgb.shape
        depths = []
        for i in range(B):
            # Convert to PIL for the processor
            img_np = rgb_uint8[i].permute(1, 2, 0).cpu().numpy()
            pil_img = Image.fromarray(img_np)

            inputs = self.processor(images=pil_img, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            outputs = self.model(**inputs)
            pred = outputs.predicted_depth  # (1, h, w)

            # Resize to original resolution
            pred = pred.unsqueeze(1)  # (1, 1, h, w)
            if pred.shape[-2:] != (H, W):
                pred = F.interpolate(
                    pred, size=(H, W), mode="bilinear", align_corners=False
                )
            depths.append(pred)

        return torch.cat(depths, dim=0)

    @property
    def device(self):
        return self._device

    def to(self, device, *args, **kwargs):
        self._device = device
        self.model = self.model.to(device)
        return self


@register_baseline("depth_anything_v2")
def _build_depth_anything_v2(
    device,
    model_id="depth-anything/Depth-Anything-V2-Large-hf",
    **kwargs,
):
    return DepthAnythingV2Wrapper(model_id=model_id, device=device)
