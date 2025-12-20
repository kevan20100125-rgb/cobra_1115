# cobra/quantize/wrap/entry.py
from __future__ import annotations

"""
Stable entrypoint for wrapping a float Cobra VLM with Quant* modules.

Historically this codebase delegated to `cobra.integration.wrap_replace`,
but the current repository version keeps the functionality inside
`cobra.quantize.wrap_replace` to avoid missing integration dependencies.

Call sites (e.g. quantize/runtime/load_quantized_vlm.py) should import:
    from cobra.quantize.wrap.entry import wrap_model_for_quantization
"""

from typing import Optional, Sequence

import torch.nn as nn

from cobra.overwatch import initialize_overwatch
from cobra.quantize.wrap.manifest import WrapRule
from cobra.quantize.wrap.policy import WrapPolicyConfig
from cobra.quantize.wrap.utils import WrapQuantParams
from cobra.quantize.wrap_replace import wrap_model_for_quantization as _impl

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
        "[WrapEntry] wrap_model_for_quantization "
        f"(prefix={prefix!r}, policy_cfg={policy_cfg!r})"
    )
    return _impl(
        model,
        policy_cfg=policy_cfg,
        manifest=manifest,
        default_params=default_params,
        prefix=prefix,
    )
