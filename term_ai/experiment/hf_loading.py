from __future__ import annotations

from copy import deepcopy
from typing import Any


def _is_missing_remote_module_error(exc: OSError) -> bool:
    text = str(exc)
    return "does not appear to have a file named" in text and (
        "configuration_" in text or "modeling_" in text or text.endswith(".py")
    )


def from_pretrained_with_trust(
    factory: Any,
    model_name_or_path: str,
    trust_remote_code: bool,
    **kwargs: Any,
) -> Any:
    """Call HF from_pretrained with conservative compatibility fallbacks."""

    try:
        return factory.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
    except TypeError as exc:
        if trust_remote_code or "trust_remote_code" not in str(exc):
            raise
        return factory.from_pretrained(model_name_or_path, **kwargs)
    except OSError as exc:
        if not trust_remote_code or not _is_missing_remote_module_error(exc):
            raise
        return factory.from_pretrained(
            model_name_or_path,
            trust_remote_code=False,
            **kwargs,
        )


def is_bitnet_config(config: Any) -> bool:
    if str(getattr(config, "model_type", "") or "").lower() == "bitnet":
        return True
    quant_config = getattr(config, "quantization_config", None)
    if isinstance(quant_config, dict):
        return str(quant_config.get("quant_method") or "").lower() == "bitnet"
    return str(getattr(quant_config, "quant_method", "") or "").lower() == "bitnet"


def bitnet_loading_config(config: Any, *, for_lora: bool) -> Any:
    if not is_bitnet_config(config):
        return config
    cloned = deepcopy(config)
    cloned.quantization_config = {
        "quant_method": "bitnet",
        "linear_class": "autobitlinear" if for_lora else "bitlinear",
        "quantization_mode": "offline",
    }
    return cloned


def _bitnet_target_dtype(model: Any, module: Any) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch import is exercised by callers.
        raise RuntimeError("BitNet weight repair requires torch") from exc

    for candidate in (
        getattr(getattr(module, "weight_scale", None), "dtype", None),
        getattr(model, "dtype", None),
        getattr(getattr(model, "config", None), "dtype", None),
    ):
        if isinstance(candidate, torch.dtype) and candidate.is_floating_point:
            return candidate
    return torch.bfloat16


def repair_bitnet_autobitlinear_weights(model: Any) -> int:
    """Repair Transformers BitNet AutoBitLinear uint8 weights for torch F.linear."""

    if not is_bitnet_config(getattr(model, "config", None)):
        return 0

    try:
        import torch
        from transformers.integrations.bitnet import VALUES_PER_ITEM, unpack_weights
    except ImportError as exc:  # pragma: no cover - imported in train extra environments.
        raise RuntimeError("BitNet weight repair requires torch and transformers BitNet support") from exc

    repaired = 0
    for module in model.modules():
        if module.__class__.__name__ != "AutoBitLinear":
            continue
        weight = getattr(module, "weight", None)
        if weight is None or getattr(weight, "dtype", None) is not torch.uint8:
            continue

        target_dtype = _bitnet_target_dtype(model, module)
        source = weight.detach()
        out_features = int(getattr(module, "out_features", 0) or 0)
        if out_features and source.shape[0] * VALUES_PER_ITEM == out_features:
            converted = unpack_weights(source, dtype=target_dtype)
        elif out_features and source.shape[0] == out_features:
            signed = source.to(torch.int16)
            signed = torch.where(signed == 255, torch.full_like(signed, -1), signed)
            converted = signed.to(target_dtype)
        else:
            continue

        module.weight = torch.nn.Parameter(converted, requires_grad=False)
        repaired += 1
    return repaired


def clear_bitnet_quantization_training_guard(model: Any) -> int:
    """Clear stale BitNet quantization metadata after LoRA weight repair."""

    visited: set[int] = set()
    stack = [model]
    cleared = 0
    while stack:
        current = stack.pop()
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))

        config = getattr(current, "config", None)
        looks_bitnet = is_bitnet_config(config) or str(getattr(current, "quantization_method", "")).lower().endswith(
            "bitnet"
        )
        if looks_bitnet or getattr(current, "is_quantized", False) or getattr(current, "hf_quantizer", None) is not None:
            try:
                current.is_quantized = False
                current.hf_quantizer = None
                current.quantization_method = None
                cleared += 1
            except Exception:
                pass
        if config is not None and is_bitnet_config(config):
            try:
                config.quantization_config = None
            except Exception:
                pass

        for attr in ("base_model", "model"):
            child = getattr(current, attr, None)
            if child is not current:
                stack.append(child)

    return cleared
