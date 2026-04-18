"""
Gradient Matching Loss for depth estimation.

Based on MiDaS (Ranftl et al., TPAMI 2022) and validated by Depth Anything V2
(Yang et al., NeurIPS 2024) as "super beneficial to depth sharpness" when training
with precise synthetic or pseudo-GT labels.

Penalizes differences between spatial gradients of predicted and target depth
at multiple scales, encouraging sharp depth boundaries.
"""

import torch
import torch.nn as nn

from .utils import FNS, masked_mean


class GradientMatching(nn.Module):
    def __init__(
        self,
        weight: float,
        scales: int = 4,
        input_fn: str = "log",
        output_fn: str = "sqrt",
        eps: float = 1e-5,
    ):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight = weight
        self.scales = scales
        self.input_fn = FNS[input_fn]
        self.output_fn = FNS[output_fn]
        self.eps = eps

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        mask = mask.bool()
        input = self.input_fn(input.float())
        target = self.input_fn(target.float())

        total_loss = torch.zeros(input.shape[0], device=input.device)

        for scale in range(self.scales):
            step = 2 ** scale

            # Subsample for multi-scale
            inp = input[:, :, ::step, ::step]
            tgt = target[:, :, ::step, ::step]
            msk = mask[:, :, ::step, ::step]

            # Gradients in x direction (horizontal)
            grad_x_inp = inp[:, :, :, :-1] - inp[:, :, :, 1:]
            grad_x_tgt = tgt[:, :, :, :-1] - tgt[:, :, :, 1:]
            mask_x = msk[:, :, :, :-1] & msk[:, :, :, 1:]

            # Gradients in y direction (vertical)
            grad_y_inp = inp[:, :, :-1, :] - inp[:, :, 1:, :]
            grad_y_tgt = tgt[:, :, :-1, :] - tgt[:, :, 1:, :]
            mask_y = msk[:, :, :-1, :] & msk[:, :, 1:, :]

            # L1 gradient difference
            diff_x = (grad_x_inp - grad_x_tgt).abs()
            diff_y = (grad_y_inp - grad_y_tgt).abs()

            loss_x = masked_mean(diff_x, mask_x, dim=[-3, -2, -1]).reshape(input.shape[0])
            loss_y = masked_mean(diff_y, mask_y, dim=[-3, -2, -1]).reshape(input.shape[0])

            total_loss = total_loss + loss_x + loss_y

        return self.output_fn(total_loss.clamp(min=self.eps))

    @classmethod
    def build(cls, config):
        return cls(
            weight=config["weight"],
            scales=config.get("scales", 4),
            input_fn=config.get("input_fn", "log"),
            output_fn=config.get("output_fn", "sqrt"),
        )
