# cobra/quantize/wrap/registry.py

from __future__ import annotations

"""
Registry for planned module wrapping (float modules -> Quant* wrappers).

This module is responsible for answering:
    "Which modules in this model should be wrapped, and under which target,
     according to a given wrapping policy?"

It does **not** perform in-place replacement itself; that is delegated to
`cobra.integration.wrap_replace`, which consumes the registry and uses
`cobra.quantize.wrap.utils` helpers to actually mutate the model.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

import torch.nn as nn

from cobra.overwatch import initialize_overwatch

from .manifest import WrapRule, find_wrap_rule_for_module, iter_default_wrap_rules
from .policy import DefaultWrapPolicy, WrapPolicyConfig, infer_target_from_module_path
from .utils import is_quantized_module

overwatch = initialize_overwatch(__name__)

def _pct_only_noop_factory(module: nn.Module, params) -> nn.Module:
    """
    "pct_only" entries are for activation clipping calibration/coverage only.

    They must NOT change the module instance, and must NOT trigger in-place
    replacement in wrap_replace().
    """
    return module


PCT_ONLY_RULE: WrapRule = WrapRule(
    source_cls=nn.Module,
    wrap_kind="pct_only",
    factory=_pct_only_noop_factory,
    allow_subclass=True,
)

@dataclass(frozen=True)
class WrapEntry:
    """
    A single wrapping decision.

    Attributes:
        module_path:
            Qualified path from `model.named_modules()`.
        target:
            Canonical target name ("vision.dino", "vision.siglip", "llm", "projector").
        rule_kind:
            WrapRule.wrap_kind (e.g. "linear", "conv2d").
        rule:
            The WrapRule that describes how to construct the Quant* wrapper.
    """

    module_path: str
    target: str
    rule_kind: str
    rule: WrapRule


@dataclass
class WrapRegistry:
    """
    Collection of wrapping decisions for a given model.

    This structure is intentionally small and serializable; it can be:
        - built once on rank 0 and broadcast to other workers,
        - inspected or logged for debugging,
        - consumed by `integration.wrap_replace` to perform replacements.
    """

    entries: List[WrapEntry] = field(default_factory=list)

    def add(self, entry: WrapEntry) -> None:
        self.entries.append(entry)

    # ------------------------------------------------------------------
    # Convenience views
    # ------------------------------------------------------------------

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.entries)

    def __iter__(self) -> Iterator[WrapEntry]:
        return iter(self.entries)

    def by_target(self) -> Mapping[str, List[WrapEntry]]:
        """
        Group entries by canonical target.
        """
        buckets: Dict[str, List[WrapEntry]] = {}
        for e in self.entries:
            buckets.setdefault(e.target, []).append(e)
        return buckets

    def module_paths_by_target(
        self,
        include_targets: Optional[Iterable[str]] = None,
    ) -> Mapping[str, List[str]]:
        """
        Return a mapping {target -> [module_path, ...]} for wrapped modules.

        Args:
            include_targets:
                Optional iterable of canonical target names. If provided,
                only entries whose `target` is in this collection are
                included in the returned mapping.

        This helper is primarily used by calibration / summary code to
        reason about which module paths are expected to be quantized for
        each canonical target according to the current wrapping policy.
        """
        buckets: Dict[str, List[str]] = {}
        for e in self.entries:
            if include_targets is not None and e.target not in include_targets:
                continue
            buckets.setdefault(e.target, []).append(e.module_path)
        return buckets

    def by_module_path(self) -> Mapping[str, WrapEntry]:
        """
        Build a lookup table keyed by module_path.
        """
        return {e.module_path: e for e in self.entries}


# ======================================================================
# Construction
# ======================================================================


def build_wrap_registry(
    model: nn.Module,
    *,
    policy_cfg: Optional[WrapPolicyConfig] = None,
    manifest: Optional[Sequence[WrapRule]] = None,
    prefix: str = "",
) -> WrapRegistry:
    """
    Analyse `model` and build a WrapRegistry that records all modules that
    should be wrapped as Quant* modules according to the provided policy.

    Args:
        model:
            The nn.Module to analyse (e.g. Cobra VLM).
        policy_cfg:
            Configuration for the DefaultWrapPolicy. If None, a default
            instance is used.
        manifest:
            Optional sequence of WrapRule objects describing type-level
            wrapping behavior. If None, the default manifest is used.
        prefix:
            Optional string; if non-empty, only modules whose path starts
            with this prefix will be considered.

    Returns:
        WrapRegistry with one WrapEntry per module to be wrapped.
    """
    policy = DefaultWrapPolicy(policy_cfg)
    if manifest is None:
        manifest = list(iter_default_wrap_rules())

    registry = WrapRegistry()

    for module_path, module in model.named_modules():
        # Skip root module and non-prefix matches
        if module_path == "":
            continue
        if prefix and not module_path.startswith(prefix):
            continue

        # Defensive: skip already-quantized modules
        if is_quantized_module(module):
            continue

        # --------------------------------------------------------------
        # Fusion stage as a formal target:
        # Add a "pct_only" entry for activation clipping calibration/coverage.
        # This entry must not trigger actual weight wrapping.
        # --------------------------------------------------------------
        if module_path == "fusion_stage":
            target = infer_target_from_module_path(module_path)
            if target == "fusion" and policy.cfg.is_target_enabled("fusion"):
                entry = WrapEntry(
                    module_path=module_path,
                    target="fusion",
                    rule_kind=PCT_ONLY_RULE.wrap_kind,
                    rule=PCT_ONLY_RULE,
                )
                registry.add(entry)
            continue

        rule = find_wrap_rule_for_module(module, manifest=manifest)
        if rule is None:
            continue

        target = policy.decide(module_path, module, rule)
        if target is None:
            continue

        entry = WrapEntry(
            module_path=module_path,
            target=target,
            rule_kind=rule.wrap_kind,
            rule=rule,
        )
        registry.add(entry)

    # Logging summary
    buckets = registry.by_target()
    total = len(registry)
    if total == 0:
        overwatch.warning(
            "[WrapRegistry] No modules selected for wrapping by current policy."
        )
    else:
        parts = [f"{total} module(s) selected for wrapping"]
        for tgt, entries in buckets.items():
            parts.append(f"{tgt}: {len(entries)}")
        overwatch.info("[WrapRegistry] " + " | ".join(parts))

    return registry

