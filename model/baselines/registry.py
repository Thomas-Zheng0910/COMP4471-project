"""Baseline model registry — maps string names to builder functions."""

from typing import Dict, Callable
import torch

BASELINE_REGISTRY: Dict[str, Callable] = {}


def register_baseline(name: str):
    """Decorator to register a baseline builder function."""
    def wrapper(fn: Callable):
        BASELINE_REGISTRY[name] = fn
        return fn
    return wrapper


def build_baseline(name: str, device: torch.device = None, **kwargs):
    """
    Instantiate a baseline model by name.

    Args:
        name:   One of the registered baseline names.
        device: Target device.
        **kwargs: Forwarded to the builder (e.g. model_id overrides).

    Returns:
        A BaseDepthModel instance in eval mode on the given device.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if name not in BASELINE_REGISTRY:
        available = ", ".join(sorted(BASELINE_REGISTRY.keys()))
        raise ValueError(
            f"Unknown baseline '{name}'. Available: {available}"
        )

    model = BASELINE_REGISTRY[name](device=device, **kwargs)
    model.to(device).eval()
    return model


# Import all baseline modules so their @register_baseline decorators run.
import model.baselines.unidepthv2  # noqa: F401, E402
import model.baselines.depth_anything_v2  # noqa: F401, E402
import model.baselines.marigold  # noqa: F401, E402
