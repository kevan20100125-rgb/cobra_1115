from .config import (
    ModelIdResolution,
    ProjectorRotationMode,
    QuantMode,
    QuantRuntimeConfig,
    RuntimeRequestResolution,
    normalize_bits_spec,
    parse_flexible_bits_spec,
    resolve_model_id_base_backend,
    resolve_runtime_request,
)

__all__ = [
    "ModelIdResolution",
    "ProjectorRotationMode",
    "QuantMode",
    "QuantRuntimeConfig",
    "RuntimeRequestResolution",
    "normalize_bits_spec",
    "parse_flexible_bits_spec",
    "resolve_model_id_base_backend",
    "resolve_runtime_request",
    "load_quantized_cobra_vlm",
]


def load_quantized_cobra_vlm(*args, **kwargs):
    """
    Lazy public export to avoid eager runtime loader import during package init.
    This reduces circular-import fragility in Phase 4 decomposition.
    """
    from .load_quantized_vlm import load_quantized_cobra_vlm as _impl

    return _impl(*args, **kwargs)