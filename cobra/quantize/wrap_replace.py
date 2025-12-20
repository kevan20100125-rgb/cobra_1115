# cobra/quantize/wrap_replace.py
from __future__ import annotations

from typing import Optional, Sequence

import torch.nn as nn

from cobra.overwatch import initialize_overwatch
from cobra.quantize.wrap.manifest import WrapRule, wrap_module_with_rule
from cobra.quantize.wrap.policy import WrapPolicyConfig
from cobra.quantize.wrap.registry import WrapRegistry, build_wrap_registry
from cobra.quantize.wrap.utils import (
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

    This is the in-repo replacement for the previously referenced
    `cobra.integration.wrap_replace.wrap_model_for_quantization`.

    Contract:
        - Build a WrapRegistry (planned wraps)
        - For each entry, construct a Quant* module via WrapRule
        - Replace the module at module_path in-place

    Returns:
        WrapRegistry (for logging/inspection/debug).
    """
    if default_params is None:
        default_params = WrapQuantParams()

    registry = build_wrap_registry(
        model,
        policy_cfg=policy_cfg,
        manifest=manifest,
        prefix=prefix,
    )

    total = 0
    skipped = 0

    # Apply in model order (registry entries are collected in traversal order)
    for entry in registry:
        # "pct_only" entries are for activation clipping coverage only.
        # Do not perform in-place replacement, and do not count as wrapped.
        if entry.rule_kind == "pct_only":
            continue

        old = get_module_by_path(model, entry.module_path)

        # Safety: do not double-wrap
        if is_quantized_module(old):
            skipped += 1
            continue

        new = wrap_module_with_rule(old, entry.rule, params=default_params)
        replace_module_inplace(model, entry.module_path, new)
        total += 1

    # Summaries
    by_target = registry.module_paths_by_target()
    by_target_counts = {k: len(v) for k, v in by_target.items()}

    overwatch.info(
        "[WrapReplace] Applied wrapping: "
        f"wrapped={total}, skipped_already_quantized={skipped}, "
        f"planned={len(registry)}, by_target={by_target_counts}"
    )

    return registry

