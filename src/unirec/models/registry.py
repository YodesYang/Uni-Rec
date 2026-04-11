from __future__ import annotations

from typing import Any, Callable

from .base import BaseCVRModel
from .mlp_cvr import MLPCVRModel


def _build_transformer_placeholder(model_config: dict[str, Any], feature_config: dict[str, Any]) -> BaseCVRModel:
    raise NotImplementedError(
        "transformer_cvr is intentionally left as a reserved extension point. "
        "Use model.name=mlp_cvr for the first baseline."
    )


MODEL_REGISTRY: dict[str, Callable[[dict[str, Any], dict[str, Any]], BaseCVRModel]] = {
    "mlp_cvr": MLPCVRModel,
    "transformer_cvr": _build_transformer_placeholder,
}


def build_model(model_config: dict[str, Any], feature_config: dict[str, Any]) -> BaseCVRModel:
    model_name = model_config["name"]
    if model_name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model: {model_name}")
    return MODEL_REGISTRY[model_name](model_config, feature_config)
