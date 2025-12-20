# cobra/quantize/wrap/policy.py

"""
Wrapping policies for mapping model modules to quantized wrappers.

This module answers the question:

    "Given a model, a module path, and a type-level WrapRule,
     should we wrap this module, and if so, under which canonical target?"

Key concepts:
    - Canonical targets (four-way split):
        * "vision.dino"
        * "vision.siglip"
        * "llm"
        * "projector"

    - WrapPolicyConfig:
        * controls which targets are enabled for wrapping
        * controls which op kinds (linear / conv) are included
        * provides simple exclusion knobs (LayerNorm / Embedding / regex)

    - DefaultWrapPolicy:
        * implements a straightforward, path-based policy that:
            - classifies module paths into canonical targets
            - applies per-target enable flags
            - filters out clearly undesirable modules (e.g. LayerNorm)
            - filters by op kind (linear / conv)

Design note:
    The path → target heuristics here are intentionally aligned with:
        - `cobra.switches.quant_calibrate._build_target_module_map`
        - `cobra.quantize.runtime.load_quantized_vlm._classify_target_from_module_name`
    so that:
        - wrap registry coverage
        - pct_stats / pct_hi_lo target fields
        - runtime activation coverage
    all speak the same four-way vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional

import torch.nn as nn

from cobra.overwatch import initialize_overwatch

from .manifest import WrapRule

overwatch = initialize_overwatch(__name__)


# ============================================================================
# Canonical targets
# ============================================================================

_CANONICAL_TARGETS = (
    "vision.dino",
    "vision.siglip",
    "llm",
    "projector",
)


# ============================================================================
# Config
# ============================================================================


@dataclass
@dataclass
class WrapPolicyConfig:
    """
    Configuration for the default wrapping policy.

    Per-target enable switches:
        - enable_vision_dino
        - enable_vision_siglip
        - enable_llm
        - enable_projector
        - enable_fusion

    Op-kind inclusion flags:
        - include_linear:  wrap nn.Linear-like modules
        - include_conv:    wrap nn.ConvNd-like modules

    Exclusions:
        - exclude_layernorm: skip nn.LayerNorm and similar normalization layers
        - exclude_embedding: skip nn.Embedding and similar embedding layers
        - exclude_name_regex: optional regex against module path
    """

    # Per-target enable switches
    enable_vision_dino: bool = True
    enable_vision_siglip: bool = True
    enable_llm: bool = True
    enable_projector: bool = True
    enable_fusion: bool = True

    # Op kinds
    include_linear: bool = True
    include_conv: bool = True

    # Exclusions
    exclude_layernorm: bool = True
    exclude_embedding: bool = True
    exclude_name_regex: Optional[str] = None

    def is_target_enabled(self, target: str) -> bool:
        if target == "vision.dino":
            return self.enable_vision_dino
        if target == "vision.siglip":
            return self.enable_vision_siglip
        if target == "llm":
            return self.enable_llm
        if target == "projector":
            return self.enable_projector
        if target == "fusion":
            return self.enable_fusion
        return False


# ============================================================================
# Target inference
# ============================================================================


def infer_target_from_module_path(module_path: str) -> Optional[str]:
    """
    Infer the canonical target name from a module's qualified path.

    Heuristics are intentionally aligned with:
        - Fusion stage (Point-B):
            * "fusion_stage" or "fusion_stage.*"     -> "fusion"
        - Vision backbones:
            * "vision_backbone.dino_featurizer.*"    -> "vision.dino"
            * "vision_backbone.featurizer.*"         -> "vision.dino"
            * "vision_backbone.siglip_featurizer.*"  -> "vision.siglip"
        - LLM backbone:
            * "llm_backbone.llm.*"                   -> "llm"
        - Projector:
            * "projector.*"                          -> "projector"

    Returns:
        A canonical target string, or None if no mapping applies.
    """
    module_path = (module_path or "").strip()
    if not module_path:
        return None

    # Fusion stage (Point-B)
    if module_path == "fusion_stage" or module_path.startswith("fusion_stage."):
        return "fusion"

    # Vision
    if module_path.startswith("vision_backbone.dino_featurizer"):
        return "vision.dino"
    if module_path.startswith("vision_backbone.featurizer"):
        # Historically, the generic featurizer is DINO in this codebase.
        return "vision.dino"
    if module_path.startswith("vision_backbone.siglip_featurizer"):
        return "vision.siglip"

    # LLM
    if module_path.startswith("llm_backbone.llm"):
        return "llm"

    # Projector
    if module_path.startswith("projector"):
        return "projector"

    return None


# ============================================================================
# Default policy
# ============================================================================


class DefaultWrapPolicy:
    """
    Default wrapping policy used by Cobra quantization.

    The policy is intentionally simple and transparent:

        1) Use the module path to infer a canonical target
           ("vision.dino", "vision.siglip", "llm", "projector").
        2) Apply per-target enable switches from WrapPolicyConfig.
        3) Filter out excluded modules:
             * LayerNorm (if exclude_layernorm=True)
             * Embedding (if exclude_embedding=True)
             * module paths that match exclude_name_regex
        4) Filter by op kind (linear / conv) using WrapRule.wrap_kind.
    """

    def __init__(self, cfg: Optional[WrapPolicyConfig] = None) -> None:
        if cfg is None:
            cfg = WrapPolicyConfig()
        self.cfg = cfg
        self._name_pattern = (
            re.compile(cfg.exclude_name_regex) if cfg.exclude_name_regex else None
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _allow_op_kind(self, rule: WrapRule) -> bool:
        """
        Decide whether the given WrapRule's op kind is allowed under cfg.
        """
        kind = rule.wrap_kind
        if kind == "linear":
            return self.cfg.include_linear
        if kind.startswith("conv"):
            return self.cfg.include_conv
        # For any other kinds (matmul, etc.), default to allowed.
        return True

    def _is_excluded_module(self, module_path: str, module: nn.Module) -> bool:
        """
        Apply name-based and type-based exclusions.
        """
        if self._name_pattern is not None and self._name_pattern.search(module_path):
            return True

        if self.cfg.exclude_layernorm and isinstance(module, nn.LayerNorm):
            return True

        if self.cfg.exclude_embedding and isinstance(module, nn.Embedding):
            return True

        return False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(
        self,
        module_path: str,
        module: nn.Module,
        rule: WrapRule,
    ) -> Optional[str]:
        """
        Decide whether to wrap a specific module with the given WrapRule.

        Args:
            module_path:
                Dotted module path from `model.named_modules()`.
            module:
                The nn.Module instance.
            rule:
                The WrapRule that matched this module type.

        Returns:
            canonical_target if the module should be wrapped, else None.
        """
        # 1) Determine canonical target from path
        target = infer_target_from_module_path(module_path)
        if target is None:
            # Only debug log for vision/LLM-like prefixes to avoid spam.
            if module_path.startswith("vision_backbone") or module_path.startswith(
                "llm_backbone"
            ) or module_path.startswith("projector"):
                overwatch.debug(
                    "[WrapPolicy] No target inferred for module_path=%r (skipping)",
                    module_path,
                    extra={"wrap_module": module_path},
                )
            return None

        # 2) Check per-target enable flags
        if not self.cfg.is_target_enabled(target):
            overwatch.debug(
                "[WrapPolicy] Target disabled by config: target=%r module=%r",
                target,
                module_path,
                extra={"wrap_module": module_path, "wrap_target": target},
            )
            return None

        # 3) Apply simple name-based and type-based exclusions
        if self._is_excluded_module(module_path, module):
            overwatch.debug(
                "[WrapPolicy] Module excluded by name/type: target=%r module=%r",
                target,
                module_path,
                extra={"wrap_module": module_path, "wrap_target": target},
            )
            return None

        # 4) Filter by op kind (linear / conv*)
        if not self._allow_op_kind(rule):
            overwatch.debug(
                "[WrapPolicy] Op kind excluded: target=%r module=%r kind=%r",
                target,
                module_path,
                rule.wrap_kind,
                extra={
                    "wrap_module": module_path,
                    "wrap_target": target,
                    "wrap_kind": rule.wrap_kind,
                },
            )
            return None

        overwatch.debug(
            "[WrapPolicy] Will wrap module=%r as target=%r kind=%r",
            module_path,
            target,
            rule.wrap_kind,
            extra={
                "wrap_module": module_path,
                "wrap_target": target,
                "wrap_kind": rule.wrap_kind,
            },
        )

        return target


# ============================================================================
# Factory
# ============================================================================


def build_default_policy(cfg: Optional[WrapPolicyConfig] = None) -> DefaultWrapPolicy:
    """
    Convenience factory for constructing a DefaultWrapPolicy.
    """
    policy = DefaultWrapPolicy(cfg)
    overwatch.info(
        "[WrapPolicy] Initialized DefaultWrapPolicy with config: %s",
        policy.cfg,
    )
    return policy


