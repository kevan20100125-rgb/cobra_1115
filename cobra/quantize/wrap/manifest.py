# cobra/quantize/wrap/manifest.py

from __future__ import annotations

"""
Mapping from base PyTorch module types to quantized wrapper factories.

This module is purely declarative:
    - It does NOT traverse models.
    - It does NOT decide which specific modules to wrap.
    - It simply exposes a manifest describing how to wrap a given nn.Module
      type (e.g. nn.Linear, nn.Conv2d) into the corresponding Quant* module.

Higher-level components (`policy.py`, `registry.py`, `integration/wrap_replace.py`)
are responsible for:
    - deciding which named modules to wrap,
    - selecting bitwidths / runtime flags,
    - and calling the factories provided here.

Design note:
    - For the current Cobra PTQ stack (including vision backbones), we only
      need to wrap:
          * nn.Linear
          * nn.Conv1d / nn.Conv2d / nn.Conv3d
      because all vision/LLM blocks are built from these.
    - MatMul-style wrappers (`make_quant_matmul`) are kept for completeness
      but deliberately not enabled by default here, to avoid guessing which
      custom classes in your repo should be treated as matmul.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Sequence, Type

import torch.nn as nn

from cobra.overwatch import initialize_overwatch

from .utils import (
    WrapQuantParams,
    make_quant_linear,
    make_quant_conv,
    make_quant_matmul,  # currently unused here but kept for completeness
    is_quantized_module,
)

overwatch = initialize_overwatch(__name__)


# ============================================================================
# Wrap rule abstraction
# ============================================================================

WrapFactory = Callable[[nn.Module, WrapQuantParams], nn.Module]


@dataclass(frozen=True)
class WrapRule:
    """
    A single type-level wrapping rule.

    Attributes:
        source_cls:
            The original nn.Module subclass we want to wrap (e.g. nn.Linear).
        wrap_kind:
            A short string label describing the kind of operation
            ("linear", "conv", "matmul", ...). This is mainly for logging
            and for any policies that want to reason about op categories.
        factory:
            A callable that takes (module, params) and returns a new Quant*
            module wrapping the original.
        allow_subclass:
            If True, this rule applies to subclasses of `source_cls` as well.
    """

    source_cls: Type[nn.Module]
    wrap_kind: str
    factory: WrapFactory
    allow_subclass: bool = True


# ============================================================================
# Default manifest
# ============================================================================


def _linear_factory(module: nn.Module, params: WrapQuantParams) -> nn.Module:
    if not isinstance(module, nn.Linear):
        raise TypeError(f"_linear_factory expects nn.Linear, got {type(module)!r}")
    return make_quant_linear(module, params=params)


def _conv_factory(module: nn.Module, params: WrapQuantParams) -> nn.Module:
    if not isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        raise TypeError(f"_conv_factory expects ConvNd, got {type(module)!r}")
    return make_quant_conv(module, params=params)


def _noop_factory(module: nn.Module, params: WrapQuantParams) -> nn.Module:
    """
    Fallback factory that simply returns the original module.

    This is not used in the default manifest, but may be convenient for
    downstream policies that want to "opt-out" of wrapping while still
    using a uniform interface.
    """
    return module


DEFAULT_WRAP_RULES: Sequence[WrapRule] = (
    WrapRule(
        source_cls=nn.Linear,
        wrap_kind="linear",
        factory=_linear_factory,
        allow_subclass=True,
    ),
    WrapRule(
        source_cls=nn.Conv1d,
        wrap_kind="conv1d",
        factory=_conv_factory,
        allow_subclass=True,
    ),
    WrapRule(
        source_cls=nn.Conv2d,
        wrap_kind="conv2d",
        factory=_conv_factory,
        allow_subclass=True,
    ),
    WrapRule(
        source_cls=nn.Conv3d,
        wrap_kind="conv3d",
        factory=_conv_factory,
        allow_subclass=True,
    ),
    # NOTE:
    #   If, in the future, you introduce a dedicated MatMul module (e.g.
    #   cobra.layers.MatMul or similar) and want to use QuantMatMul for it,
    #   you can extend DEFAULT_WRAP_RULES here with an appropriate rule.
    #   For now we keep make_quant_matmul imported but unused to avoid
    #   making assumptions about your repository's custom modules.
)


# Build a lookup map keyed by source_cls for faster exact matches.
_DEFAULT_MANIFEST_BY_TYPE: Dict[Type[nn.Module], WrapRule] = {
    rule.source_cls: rule for rule in DEFAULT_WRAP_RULES
}


# ============================================================================
# Public helpers
# ============================================================================


def iter_default_wrap_rules() -> Iterable[WrapRule]:
    """
    Iterate over all default wrap rules.
    """
    return iter(DEFAULT_WRAP_RULES)


def find_wrap_rule_for_module(
    module: nn.Module,
    *,
    manifest: Optional[Sequence[WrapRule]] = None,
) -> Optional[WrapRule]:
    """
    Find the first WrapRule that applies to the given module.

    Resolution order:
        1. If manifest is provided, scan its rules in order.
        2. Otherwise, try an exact type match against the default manifest.
        3. If no exact match, fall back to the first rule whose source_cls
           is a superclass of module's type and `allow_subclass` is True.

    Returns:
        A WrapRule if a match is found, else None.
    """
    if is_quantized_module(module):
        # Already wrapped; do not apply any additional rule.
        return None

    rules: Sequence[WrapRule]
    if manifest is not None:
        rules = manifest
    else:
        rules = DEFAULT_WRAP_RULES

    # Fast path: exact-type lookup when using default manifest
    if manifest is None:
        rule = _DEFAULT_MANIFEST_BY_TYPE.get(type(module))
        if rule is not None:
            return rule

    # Fallback: linear scan with subclass checks
    for rule in rules:
        if rule.allow_subclass and isinstance(module, rule.source_cls):
            return rule

    return None


def wrap_module_with_rule(
    module: nn.Module,
    rule: WrapRule,
    params: Optional[WrapQuantParams] = None,
) -> nn.Module:
    """
    Apply a WrapRule to a single module.

    Args:
        module:
            The original nn.Module instance to wrap.
        rule:
            The WrapRule describing how to wrap this module.
        params:
            Optional WrapQuantParams; if None, a default instance is used.

    Returns:
        A new nn.Module (typically a Quant* wrapper) that should replace
        the original module in the model tree.
    """
    if params is None:
        params = WrapQuantParams()

    wrapped = rule.factory(module, params)

    overwatch.debug(
        "[WrapManifest] Wrapped %s as %s (kind=%r)",
        module.__class__.__name__,
        wrapped.__class__.__name__,
        rule.wrap_kind,
        extra={"wrap_kind": rule.wrap_kind},
    )

    return wrapped

