from __future__ import annotations

from typing import Optional, Sequence

import torch.nn as nn

from cobra.overwatch import initialize_overwatch

from .manifest import WrapRule, wrap_module_with_rule
from .policy import WrapPolicyConfig
from .registry import WrapRegistry, build_wrap_registry
from .utils import (
    WrapQuantParams,
    get_module_by_path,
    is_quantized_module,
    replace_module_inplace,
)

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
    Apply Quant* wrappers to a float model in-place.

    Phase 5:
      - the canonical implementation now lives inside cobra.quantize.wrap.replace
      - external call sites should import from cobra.quantize.wrap.entry
      - cobra.quantize.wrap_replace remains only as a compatibility shim
    """
    if default_params is None:
        default_params = WrapQuantParams()

    registry = build_wrap_registry(
        model,
        policy_cfg=policy_cfg,
        manifest=manifest,
        prefix=prefix,
    )

    wrapped = 0
    skipped_already_quantized = 0

    for entry in registry:
        if entry.rule_kind == "pct_only":
            continue

        old_module = get_module_by_path(model, entry.module_path)
        if is_quantized_module(old_module):
            skipped_already_quantized += 1
            continue

        new_module = wrap_module_with_rule(old_module, entry.rule, params=default_params)
        replace_module_inplace(model, entry.module_path, new_module)
        wrapped += 1

    by_target_counts = {
        target: len(paths)
        for target, paths in registry.module_paths_by_target().items()
    }

    overwatch.info(
        "[WrapReplace] Applied wrapping: wrapped=%d skipped_already_quantized=%d planned=%d by_target=%s",
        wrapped,
        skipped_already_quantized,
        len(registry),
        by_target_counts,
    )

    return registry