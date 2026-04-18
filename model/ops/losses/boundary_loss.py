import torch
import torch.nn as nn
import torch.nn.functional as F


class TransparencyBoundaryLoss(nn.Module):
    """Boundary-aware depth loss for transparent/reflective surfaces.

    Dilates contours from the segmentation mask to create a boundary zone,
    then applies an L1 gradient loss in that zone to enforce sharp depth
    transitions at object edges — critical for glass, mirrors, etc.
    """

    def __init__(self, weight: float, dilation: int = 8):
        super().__init__()
        self.name: str = self.__class__.__name__
        self.weight: float = weight
        self.dilation = dilation
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).reshape(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _get_boundary_zone(self, seg_mask: torch.Tensor) -> torch.Tensor:
        """Create dilated boundary zone from seg mask.

        seg_mask: [B, 1, H, W] float, >0 where object detected.
        Returns: [B, 1, H, W] float boundary zone mask.
        """
        # Find contours: difference between dilated and eroded mask
        k = self.dilation
        kernel_size = 2 * k + 1
        dilated = F.max_pool2d(seg_mask, kernel_size=kernel_size, stride=1, padding=k)
        eroded = -F.max_pool2d(-seg_mask, kernel_size=kernel_size, stride=1, padding=k)
        boundary = (dilated - eroded).clamp(0, 1)
        return boundary

    def _depth_gradients(self, depth: torch.Tensor) -> tuple:
        """Compute Sobel gradients of depth."""
        gx = F.conv2d(depth, self.sobel_x, padding=1)
        gy = F.conv2d(depth, self.sobel_y, padding=1)
        return gx, gy

    @torch.autocast(device_type="cuda", enabled=False, dtype=torch.float32)
    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        seg_labels: torch.Tensor = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
            input:      [B, 1, H, W] predicted depth
            target:     [B, 1, H, W] GT depth
            mask:       [B, 1, H, W] valid depth mask
            seg_labels: [B, 1, H, W] int64 class labels (0=bg, >0 = objects)
        """
        input = input.float()
        target = target.float()
        mask = mask.float()

        if seg_labels is None:
            return torch.tensor(0.0, device=input.device, requires_grad=True)

        # Resize seg_labels to match input resolution
        if seg_labels.shape[-2:] != input.shape[-2:]:
            seg_labels = F.interpolate(
                seg_labels.float(), size=input.shape[-2:], mode="nearest"
            ).long()

        # Create boundary zone from non-background segments
        seg_mask = (seg_labels > 0).float()
        boundary_zone = self._get_boundary_zone(seg_mask)  # [B, 1, H, W]

        # Combined mask: valid depth AND in boundary zone
        zone_mask = boundary_zone * mask

        n_valid = zone_mask.sum().clamp(min=1.0)
        if n_valid < 10:
            return torch.tensor(0.0, device=input.device, requires_grad=True)

        # Compute gradients
        pred_gx, pred_gy = self._depth_gradients(input)
        gt_gx, gt_gy = self._depth_gradients(target)

        # L1 gradient matching in boundary zone
        loss = (
            (pred_gx - gt_gx).abs() * zone_mask +
            (pred_gy - gt_gy).abs() * zone_mask
        ).sum() / n_valid

        return loss

    @classmethod
    def build(cls, config):
        return cls(
            weight=config["weight"],
            dilation=config.get("dilation", 8),
        )
