"""
Marigold baseline wrapper.

Uses the HuggingFace diffusers pipeline:
    prs-eth/marigold-lcm-v1-0

Requires: torch, diffusers, transformers
(diffusers added to environment.yml)

NOTE: Marigold produces *affine-invariant* depth.
      The evaluation script handles scale/shift alignment via SSI/SI.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from model.baselines.base import BaseDepthModel
from model.baselines.registry import register_baseline


class MarigoldWrapper(BaseDepthModel):
    """
    Wraps the Marigold LCM diffusion depth model via diffusers.

    Uses the fast LCM variant (1-4 denoising steps) for practical speed.
    """

    def __init__(
        self,
        model_id: str = "prs-eth/marigold-lcm-v1-0",
        num_inference_steps: int = 4,
        ensemble_size: int = 1,
        device=None,
    ):
        super().__init__()
        import diffusers

        self.pipe = diffusers.MarigoldDepthPipeline.from_pretrained(
            model_id, torch_dtype=torch.float16
        )
        self.num_inference_steps = num_inference_steps
        self.ensemble_size = ensemble_size
        self._device = device or torch.device("cpu")

    @torch.no_grad()
    def predict_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) uint8 or float [0,1].
        Returns:
            depth: (B, 1, H, W) float32 relative depth.
        """
        if rgb.dtype != torch.uint8:
            rgb_uint8 = (rgb * 255).clamp(0, 255).to(torch.uint8)
        else:
            rgb_uint8 = rgb

        B, _, H, W = rgb.shape
        depths = []
        for i in range(B):
            img_np = rgb_uint8[i].permute(1, 2, 0).cpu().numpy()
            pil_img = Image.fromarray(img_np)

            output = self.pipe(
                pil_img,
                num_inference_steps=self.num_inference_steps,
                ensemble_size=self.ensemble_size,
            )
            # output.prediction is a numpy array (H', W') in [0, 1]
            pred_np = output.prediction.squeeze()
            pred = torch.from_numpy(pred_np).float().unsqueeze(0).unsqueeze(0)

            if pred.shape[-2:] != (H, W):
                pred = F.interpolate(
                    pred, size=(H, W), mode="bilinear", align_corners=False
                )
            depths.append(pred.to(self._device))

        return torch.cat(depths, dim=0)

    @property
    def device(self):
        return self._device

    def to(self, device, *args, **kwargs):
        self._device = device
        self.pipe = self.pipe.to(device)
        return self


@register_baseline("marigold")
def _build_marigold(
    device,
    model_id="prs-eth/marigold-lcm-v1-0",
    num_inference_steps=4,
    ensemble_size=1,
    **kwargs,
):
    return MarigoldWrapper(
        model_id=model_id,
        num_inference_steps=num_inference_steps,
        ensemble_size=ensemble_size,
        device=device,
    )
