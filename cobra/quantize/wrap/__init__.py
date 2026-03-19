from __future__ import annotations

"""
Stable entrypoint for wrapping a float Cobra VLM with Quant* modules.

Phase 5:
  - canonical implementation lives in cobra.quantize.wrap.replace
  - callers should import only from cobra.quantize.wrap.entry
  - no direct dependency on cobra.quantize.wrap_replace is needed anymore
"""

from typing import Optional, Sequence

import torch.nn as nn

from cobra.overwatch import initialize_overwatch

from .manifest import WrapRule
from .policy import WrapPolicyConfig
from .replace import wrap_model_for_quantization as _impl
from .utils import WrapQuantParams

overwatch = initialize_overwatch(__name__)


def wrap_model_for_quantization(
    model: nn.Module,
    *,
    policy_cfg: Optional[WrapPolicyConfig] = None,
    manifest: Optional[Sequence[WrapRule]] = None,
    default_params: Optional[WrapQuantParams] = None,
    prefix: str = "",
):
    overwatch.info(
        "[WrapEntry] wrap_model_for_quantization (prefix=%r, policy_cfg=%r)",
        prefix,
        policy_cfg,
    )
    return _impl(
        model,
        policy_cfg=policy_cfg,
        manifest=manifest,
        default_params=default_params,
        prefix=prefix,
    )