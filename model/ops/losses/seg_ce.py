import torch
import torch.nn as nn
import torch.nn.functional as F


class SegCrossEntropy(nn.Module):
    """Cross-entropy loss for auxiliary segmentation task.

    Downsamples GT labels to match prediction resolution (nearest),
    then applies standard CE loss. Follows the same interface pattern
    as other losses in this package (weight, name, build classmethod).
    """

    def __init__(
        self,
        weight: float,
        num_classes: int = 81,
        ignore_index: int = 255,
    ):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight: float = weight
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            input:  [B, C, H_pred, W_pred] seg logits from SegHead
            target: [B, 1, H_gt, W_gt] int64 class labels
        Returns:
            Scalar CE loss (per-batch mean).
        """
        # Squeeze channel dim if present
        if target.ndim == 4 and target.shape[1] == 1:
            target = target[:, 0]  # [B, H_gt, W_gt]

        # Downsample GT to prediction resolution
        pred_h, pred_w = input.shape[2], input.shape[3]
        if target.shape[-2:] != (pred_h, pred_w):
            target = F.interpolate(
                target.unsqueeze(1).float(),
                size=(pred_h, pred_w),
                mode="nearest",
            ).squeeze(1).long()

        return self.ce(input.float(), target)

    @classmethod
    def build(cls, config):
        return cls(
            weight=config["weight"],
            num_classes=config.get("num_classes", 81),
            ignore_index=config.get("ignore_index", 255),
        )
