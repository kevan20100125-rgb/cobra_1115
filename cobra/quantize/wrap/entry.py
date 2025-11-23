# cobra/quantize/wrap/entry.py
from __future__ import annotations
import torch
from typing import Optional, Dict, Any

from cobra.quantize.wrap.registry import build_wrap_registry
from cobra.quantize.wrap.policy import build_policy
from cobra.quantize.wrap.manifest import build_wrap_manifest
from cobra.quantize.wrap.utils import replace_module_inplace


def wrap_model_for_quantization(
    model: torch.nn.Module,
    *,
    int_export_bundle: Optional[Dict[str, Any]] = None,
    bits: str,
    device: torch.device,
    base_dtype: torch.dtype,
):
    """
    High-level wrapper for converting a float Cobra model into runtime-quantized form.

    This function does:

        1. Scan all modules and build a wrap registry
        2. Build policy from registry (select modules to quantize)
        3. Build manifest of (module_path, quantized_module_constructor)
        4. Apply quantized module replacements in-place

    Notes:
        - This does **not** apply activation clipping. That is handled by calibrator.
        - This does **not** load integer-export weights (W8A8 ONNX). 
          That is a separate pipeline in PTQ finalize.
        - This implements Python-runtime quantization (QuantLinear/QuantConv/etc.)
          so that the model can run inside PyTorch for accuracy evaluation.
    """

    # 1. Build registry = scan model for eligible float modules
    registry = build_wrap_registry(model)

    # 2. Build policy = choose which registry entries to quantize
    policy = build_policy(registry)

    # 3. Build manifest = concrete list of modules to replace
    manifest = build_wrap_manifest(
        model=model,
        policy=policy,
        registry=registry,
        device=device,
        dtype=base_dtype,
        bits=bits,
    )

    # 4. Replace modules in-place
    for module_path, new_module in manifest:
        replace_module_inplace(model, module_path, new_module)

    return model

