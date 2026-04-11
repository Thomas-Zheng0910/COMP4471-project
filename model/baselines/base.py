"""Abstract base class for baseline depth models."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn


class BaseDepthModel(ABC, nn.Module):
    """
    Unified interface for baseline depth estimation models.

    Subclasses must implement:
        predict_depth(rgb) -> depth  (B,1,H,W) in metres
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def predict_depth(self, rgb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            rgb: (B, 3, H, W) float32 in [0, 1] or uint8 [0, 255].

        Returns:
            depth: (B, 1, H, W) float32, metric depth in metres.
                   For relative-only models, returns the raw relative
                   prediction (evaluation handles scale alignment).
        """
        ...

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")
