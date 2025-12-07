# cobra/quantize/wrap/entry.py
from __future__ import annotations

"""
High-level entrypoint for wrapping a float Cobra VLM with Quant* modules.

This module is intentionally thin: it delegates to
`cobra.integration.wrap_replace.wrap_model_for_quantization` so that:

    - All wrap policy / manifest / registry logic stays centralized in
      `integration/wrap_replace.py`.
    - Quantization-aware call sites (e.g. runtime loaders) can import a
      stable API from `cobra.quantize.wrap.entry` without depending on
      integration internals.

Design notes (aligned with the PTQ fake-quant pipeline):

    - Only MambaQuant-style fake quant is used (QuantLinear / QuantConv / QuantMatMul
      + UniformAffineQuantizer).
    - No INT kernels are invoked in this path; integer export is reserved for
      offline / edge use and handled elsewhere.
    - Activation clipping (hi/lo → scale/zero_point, bitwidth selection) is
      handled by `pct.calibrator`, not here.
"""

from typing import Optional, Sequence

import torch.nn as nn

from cobra.integration.wrap_replace import (
    wrap_model_for_quantization as _wrap_model_for_quantization_impl,
)
from cobra.quantize.wrap.manifest import WrapRule
from cobra.quantize.wrap.policy import WrapPolicyConfig
from cobra.quantize.wrap.registry import WrapRegistry
from cobra.quantize.wrap.utils import WrapQuantParams
from cobra.overwatch import initialize_overwatch


overwatch = initialize_overwatch(__name__)


def wrap_model_for_quantization(
    model: nn.Module,
    *,
    policy_cfg: Optional[WrapPolicyConfig] = None,
    manifest: Optional[Sequence[WrapRule]] = None,
    default_params: Optional[WrapQuantParams] = None,
    prefix: str = "",
) -> WrapRegistry:
    """
    Public shim around `integration.wrap_replace.wrap_model_for_quantization`.

    This is the canonical API that other quantization components should call
    when they need to:

        1) Scan a *float* Cobra model for eligible modules.
        2) Wrap those modules with Quant* counterparts in-place.
        3) Obtain the resulting WrapRegistry for inspection / logging.

    Args:
        model:
            The nn.Module to wrap (e.g., Cobra VLM loaded via `cobra.load`).
        policy_cfg:
            WrapPolicyConfig controlling which canonical targets are enabled
            ("vision.dino", "vision.siglip", "llm", "projector") and basic
            inclusion / exclusion rules. If None, the default config from
            `WrapPolicyConfig()` is used.
        manifest:
            Optional sequence of WrapRule objects describing how to turn
            base modules (nn.Linear / nn.ConvNd / ...) into Quant* modules.
            If None, `quantize.wrap.manifest.DEFAULT_WRAP_RULES` is used.
        default_params:
            Optional WrapQuantParams specifying default quantization parameters
            (weight/activation bitwidths, observer settings, etc.) for all
            wrapped modules. If None, a default instance is created in the
            underlying implementation.
        prefix:
            Optional dotted prefix; if non-empty, only modules whose path
            starts with this prefix are considered for wrapping.

    Returns:
        WrapRegistry containing one WrapEntry per wrapped module.
    """
    overwatch.info(
        "[WrapEntry] Delegating wrap_model_for_quantization to "
        "cobra.integration.wrap_replace.wrap_model_for_quantization "
        f"(prefix={prefix!r}, policy_cfg={policy_cfg!r})"
    )

    return _wrap_model_for_quantization_impl(
        model,
        policy_cfg=policy_cfg,
        manifest=manifest,
        default_params=default_params,
        prefix=prefix,
    )
