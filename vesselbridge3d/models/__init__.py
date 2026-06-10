"""Model registry for vesselbridge3d.

Adding a new model = create a module here that defines the architecture and a
builder decorated with ``@register_model("<name>")``, then import it below so
the decorator runs. The training / inference engines construct models solely
through :func:`build_model`, so no engine changes are needed.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

MODEL_REGISTRY: Dict[str, Callable[..., nn.Module]] = {}


def register_model(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """Decorator registering a model builder under ``name``."""
    def deco(fn: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        if name in MODEL_REGISTRY:
            raise ValueError(f"Model {name!r} is already registered")
        MODEL_REGISTRY[name] = fn
        return fn
    return deco


def build_model(model_type: str, **kwargs) -> nn.Module:
    """Construct a registered model by key."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model_type {model_type!r}; available: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[model_type](**kwargs)


# Import model modules so their @register_model builders run on package import.
from .dinov3_unetr import SegModel3D_UNETRLite  # noqa: E402,F401
# Future: from .medsam_unetr import ...
#         from .medgemma_unetr import ...

__all__ = [
    "MODEL_REGISTRY",
    "register_model",
    "build_model",
    "SegModel3D_UNETRLite",
]
