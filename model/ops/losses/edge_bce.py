import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeBCE(nn.Module):
    """BCE loss for the low-level edge head.

    Generates edge GT from depth gradients (Sobel) and supervises the
    predicted edge logits. Uses balanced BCE since edges are sparse.
    """

    def __init__(self, weight: float):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight: float = weight
        # Sobel kernels (registered as buffers so they move with .to(device))
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _get_depth_edges(self, depth: torch.Tensor, mask: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """Compute binary edge map from depth using Sobel gradients."""
        # depth: [B, 1, H, W], mask: [B, 1, H, W]
        depth_masked = depth * mask.float()
        gx = F.conv2d(depth_masked, self.sobel_x, padding=1)
        gy = F.conv2d(depth_masked, self.sobel_y, padding=1)
        grad_mag = (gx ** 2 + gy ** 2).sqrt()
        # Normalize by local depth to make edges scale-invariant
        edge_map = grad_mag / (depth_masked + 1e-6)
        # Threshold to binary
        edge_binary = (edge_map > threshold).float()
        # Mask out invalid regions
        edge_binary = edge_binary * mask.float()
        return edge_binary

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        image: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            input:  [B, 1, H_pred, W_pred] edge logits from EdgeHead
            target: [B, 1, H, W] depth GT
            mask:   [B, 1, H, W] valid depth mask
            image:  [B, 3, H, W] RGB image (unused, reserved for future)
        """
        input = input.float()
        target = target.float()
        mask = mask.float()

        # Generate edge GT at pred resolution
        pred_h, pred_w = input.shape[2], input.shape[3]
        target_ds = F.interpolate(target, size=(pred_h, pred_w), mode="bilinear", align_corners=False)
        mask_ds = F.interpolate(mask, size=(pred_h, pred_w), mode="nearest")

        edge_gt = self._get_depth_edges(target_ds, mask_ds)

        # Balanced BCE: weight pos/neg according to frequency
        pos_count = edge_gt.sum().clamp(min=1.0)
        neg_count = (mask_ds.sum() - pos_count).clamp(min=1.0)
        pos_weight = neg_count / pos_count

        loss = F.binary_cross_entropy_with_logits(
            input * mask_ds,
            edge_gt,
            pos_weight=pos_weight.expand_as(input),
            reduction="sum",
        ) / mask_ds.sum().clamp(min=1.0)

        return loss

    @classmethod
    def build(cls, config):
        return cls(weight=config["weight"])
