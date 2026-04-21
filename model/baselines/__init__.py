"""
Baseline depth estimation models for comparison.

Each wrapper exposes a unified interface:
    model = build_baseline(name, device, **kwargs)
    depth = model.predict_depth(rgb_tensor)  # (B,1,H,W) metres

Supported baselines:
    - "unidepthv2"       : UniDepthV2 (HuggingFace)
    - "depth_anything_v2": Depth Anything V2 (HuggingFace)
    - "marigold"         : Marigold diffusion-based depth (HuggingFace)
"""

from model.baselines.base import BaseDepthModel
from model.baselines.registry import build_baseline, BASELINE_REGISTRY

__all__ = ["BaseDepthModel", "build_baseline", "BASELINE_REGISTRY"]
