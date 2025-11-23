# cobra/quantize/wrap/utils.py

"""
Utility helpers for wrapping PyTorch modules with quantized counterparts.

This module is intentionally "dumb":
    - It does **not** decide which modules to wrap (that's policy/manifest/registry).
    - It does **not** pick bitwidths or clipping strategies (that's pct/calibrator/finalize).
    - It only provides:
        * safe traversal / replacement helpers for nn.Module trees
        * thin constructors for Quant* wrappers that reuse existing int_* implementations
        * predicates to detect already-quantized modules

Dependencies:
    - Uses the quantized modules defined in `cobra.quantize.int_linear`,
      `cobra.quantize.int_conv`, `cobra.quantize.int_matmul`, and `cobra.quantize.int_others`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn

from cobra.overwatch import initialize_overwatch

from cobra.quantize.int_linear import QuantLinear
from cobra.quantize.int_conv import QuantConv1d, QuantConv2d, QuantConv3d
from cobra.quantize.int_matmul import QuantMatMul
from cobra.quantize.int_others import (
    QuantAdd,
    QuantSoftmax,
    # Others (Mul/Sub/Div/etc.) can be added here when/if needed.
)

overwatch = initialize_overwatch(__name__)


# ============================================================================
# Module tree traversal / replacement
# ============================================================================


def resolve_parent_and_attr(root: nn.Module, module_path: str) -> Tuple[nn.Module, str]:
    """
    Resolve the parent module and the attribute name for a dotted module path.

    Example:
        root:  model
        path:  "encoder.layers.0.self_attn.out_proj"
        returns: (parent=<Module encoder.layers.0.self_attn>, attr_name="out_proj")

    Raises:
        AttributeError: if any intermediate attribute does not exist.
    """
    if not module_path:
        raise ValueError("module_path must be a non-empty dotted string")

    parts = module_path.split(".")
    parent = root
    for part in parts[:-1]:
        if not hasattr(parent, part):
            raise AttributeError(f"Parent module has no attribute {part!r} while resolving {module_path!r}")
        parent = getattr(parent, part)

    return parent, parts[-1]


def get_module_by_path(root: nn.Module, module_path: str) -> nn.Module:
    """
    Resolve the module at a dotted path from the given root module.

    This is a thin helper around getattr-chaining, mainly to keep wrap_replace logic clean.
    """
    parent, attr = resolve_parent_and_attr(root, module_path)
    if not hasattr(parent, attr):
        raise AttributeError(f"Module path {module_path!r} does not exist on parent")
    return getattr(parent, attr)


def replace_module_inplace(root: nn.Module, module_path: str, new_module: nn.Module) -> None:
    """
    Replace the module at `module_path` under `root` with `new_module`.

    This is a safe, in-place mutation used by wrap_replace:
        - We resolve the parent + attr
        - We set the new module on the parent
    """
    parent, attr = resolve_parent_and_attr(root, module_path)
    old = getattr(parent, attr, None)
    setattr(parent, attr, new_module)

    overwatch.debug(
        f"[WrapUtils] Replaced module at {module_path!r} "
        f"({old.__class__.__name__ if old is not None else 'None'}) "
        f"-> {new_module.__class__.__name__}",
        extra={"module_path": module_path},
    )


def iter_leaf_modules(
    root: nn.Module,
    prefix: str = "",
) -> Iterator[Tuple[str, nn.Module]]:
    """
    Yield (module_path, module) pairs for all **leaf** modules under root.

    A "leaf" is defined as a module with no child modules (len(list(m.children())) == 0).

    Example:
        for name, mod in iter_leaf_modules(model):
            ...

    This is useful when a policy wants to consider all lowest-level ops for wrapping.
    """
    for name, module in root.named_modules():
        # Skip the root itself if prefix is specified
        if prefix and not name.startswith(prefix):
            continue
        # Leaf check
        if len(list(module.children())) == 0:
            yield name, module


# ============================================================================
# Quantized-module predicates
# ============================================================================


_QUANTIZED_TYPES = (
    QuantLinear,
    QuantConv1d,
    QuantConv2d,
    QuantConv3d,
    QuantMatMul,
    QuantAdd,
    QuantSoftmax,
)


def is_quantized_module(module: nn.Module) -> bool:
    """
    Return True if the module is already a known quantized type.

    This is used by wrap policies to avoid double-wrapping.
    """
    return isinstance(module, _QUANTIZED_TYPES)


# ============================================================================
# Thin wrappers for creating Quant* modules
# ============================================================================


@dataclass
class WrapQuantParams:
    """
    Small struct to carry default quantization params for wrapping.

    This does NOT encode bitwidths; bitwidths are configured later via:
        - activation calibration (pct/calibrator.py)
        - finalize/int_export.py for weight quantization

    Fields:
        weight_quant_params: kwargs passed into `UniformAffineQuantizer` for weights
        act_quant_params:    kwargs passed into `UniformAffineQuantizer` for activations
        disable_input_quant: whether to disable input activation quantization
        observe:             observer type (e.g. "minmax", "percentile")
    """

    weight_quant_params: Dict[str, Any] = None
    act_quant_params: Dict[str, Any] = None
    disable_input_quant: bool = False
    observe: str = "minmax"

    def __post_init__(self) -> None:
        if self.weight_quant_params is None:
            self.weight_quant_params = {"dynamic_method": "per_tensor"}
        if self.act_quant_params is None:
            self.act_quant_params = {"dynamic_method": "per_tensor"}


def make_quant_linear(
    org_module: nn.Linear,
    params: Optional[WrapQuantParams] = None,
) -> QuantLinear:
    """
    Wrap an nn.Linear into a QuantLinear, preserving its weights/bias and basic structure.

    The created QuantLinear starts with quantization disabled:
    - `use_weight_quant = False`
    - `use_act_quant = False`
    and can later be toggled via `set_quant_state(...)`.
    """
    if params is None:
        params = WrapQuantParams()

    qmod = QuantLinear(
        org_module=org_module,
        weight_quant_params=params.weight_quant_params,
        act_quant_params=params.act_quant_params,
        disable_input_quant=params.disable_input_quant,
        observe=params.observe,
    )
    return qmod


def make_quant_conv(
    org_module: Union[nn.Conv1d, nn.Conv2d, nn.Conv3d],
    params: Optional[WrapQuantParams] = None,
) -> nn.Module:
    """
    Wrap an nn.ConvNd into its corresponding QuantConvNd wrapper.

    Supported:
        - nn.Conv1d -> QuantConv1d
        - nn.Conv2d -> QuantConv2d
        - nn.Conv3d -> QuantConv3d
    """
    if params is None:
        params = WrapQuantParams()

    if isinstance(org_module, nn.Conv1d):
        return QuantConv1d(
            org_module=org_module,
            weight_quant_params=params.weight_quant_params,
            act_quant_params=params.act_quant_params,
            disable_input_quant=params.disable_input_quant,
            observe=params.observe,
        )
    elif isinstance(org_module, nn.Conv2d):
        return QuantConv2d(
            org_module=org_module,
            weight_quant_params=params.weight_quant_params,
            act_quant_params=params.act_quant_params,
            disable_input_quant=params.disable_input_quant,
            observe=params.observe,
        )
    elif isinstance(org_module, nn.Conv3d):
        return QuantConv3d(
            org_module=org_module,
            weight_quant_params=params.weight_quant_params,
            act_quant_params=params.act_quant_params,
            disable_input_quant=params.disable_input_quant,
            observe=params.observe,
        )
    else:
        raise TypeError(f"Unsupported conv module type for quantization: {type(org_module)!r}")


def make_quant_matmul(
    disable_act_quant: bool = False,
    observe: str = "minmax",
) -> QuantMatMul:
    """
    Construct a QuantMatMul module with default quantization params.

    This wrapper does not take an org_module because matmul is typically functional
    (torch.matmul) rather than a pre-existing nn.Module. Policies are responsible for
    deciding **where** to insert QuantMatMul in the model.
    """
    x_quant_params = {"dynamic_method": "per_tensor"}

    return QuantMatMul(
        x1_quant_params=x_quant_params,
        x2_quant_params=x_quant_params,
        disable_act_quant=disable_act_quant,
        observe=observe,
    )


def enable_module_quantization(
    module: nn.Module,
    weight_quant: bool,
    act_quant: bool,
) -> None:
    """
    Best-effort toggle for enabling quantization on wrapped modules.

    For QuantLinear / QuantConvNd:
        - uses `set_quant_state(weight_quant, act_quant)`

    For QuantMatMul / QuantAdd / QuantSoftmax:
        - toggles `.use_act_quant` if available.
    """
    # QuantLinear / QuantConv family
    if hasattr(module, "set_quant_state") and callable(getattr(module, "set_quant_state")):
        try:
            module.set_quant_state(weight_quant=weight_quant, act_quant=act_quant)
            return
        except TypeError:
            # Some implementations may use positional arguments
            module.set_quant_state(weight_quant, act_quant)
            return

    # QuantMatMul / QuantAdd / QuantSoftmax style (act-only)
    if hasattr(module, "use_act_quant"):
        module.use_act_quant = bool(act_quant)


def get_module_device(module: nn.Module) -> torch.device:
    """
    Infer the device of a module by inspecting its parameters / buffers.
    """
    for p in module.parameters(recurse=True):
        return p.device
    for b in module.buffers(recurse=True):
        return b.device
    return torch.device("cpu")
