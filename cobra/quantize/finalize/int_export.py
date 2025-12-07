# cobra/quantize/finalize/int_export.py 

"""
Integer export utilities for Cobra PTQ.

This module is the final step in the PTQ pipeline:

    pct(schema/best/collect/apply) -> calibrator -> wrap/rotate -> finalize(int_export)

Responsibilities:
    - Given a **calibrated & wrapped** model (QuantLinear/Conv/MatMul, etc.),
      and a desired (weight_bits, act_bits) configuration, produce:
        * integer weights (W_int) for all quantized modules
        * corresponding (scale, zero_point) metadata for weights & activations
    - Support a family of combinations with
        * weight_bits, act_bits ∈ {2, 4, 8, 16}
      (1-bit quantization is intentionally not supported here).

Non-responsibilities:
    - Does NOT perform percentile collection or best-percentile selection.
    - Does NOT perform projector KLT/Hadamard rotation (handled by
      `cobra.quantize.rotate.projector`).
    - Does NOT deal with any CLI / argparse; that is owned by
      `cobra/switches/quant_finalize.py`.

Export format (in-memory):

    {
      "config": { ... IntExportConfig ... },
      "weights": {
        "<module_path>.weight": {
          "target":       "vision.dino" | "vision.siglip" | "llm" | "projector",
          "bits":         weight_bits,
          "signed":       True/False,
          "int_weight":   torch.Tensor[int8 or int16],
          "scale":        torch.Tensor[float] (broadcastable to weight),
          "zero_point":   torch.Tensor[int32] or None,
        },
        ...
      },
      "activations": {
        "<module_path>": {
          "target":       canonical target,
          "bits":         act_bits,
          "signed":       True/False,
          "scale":        torch.Tensor[float],
          "zero_point":   torch.Tensor[int32] or None,
        },
        ...
      },
    }

Typical usage (from switches/quant_finalize.py):

    from cobra.quantize.finalize.int_export import (
        IntExportConfig,
        export_int_quant_state,
        save_int_export,
    )

    cfg = IntExportConfig(weight_bits=4, act_bits=8, ...)
    export_blob = export_int_quant_state(vlm, cfg)
    save_int_export(export_blob, out_path="outputs/quantize/int_export_w4a8.pt")

Phase 2 note:
    - 新增 `int_export_config_from_quant_cfg(quant_cfg, ...)`，由
      QuantRuntimeConfig 提供 weight_bits/act_bits + symmetric_*/targets，
      讓 quant_finalize 不再手動組 IntExportConfig。
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from cobra.overwatch import initialize_overwatch
from cobra.quantize.pct.schema import compute_affine_params
from cobra.quantize.wrap.policy import infer_target_from_module_path
from cobra.quantize.quantizer import UniformAffineQuantizer
from cobra.quantize.runtime.config import QuantRuntimeConfig  # Phase 2 helper import

# We only need the types; logic is generic and checks attributes instead of
# doing heavy isinstance chains. Importing them helps type-checkers & IDEs.
from cobra.quantize.int_linear import QuantLinear
from cobra.quantize.int_conv import QuantConv1d, QuantConv2d, QuantConv3d
from cobra.quantize.int_matmul import QuantMatMul

overwatch = initialize_overwatch(__name__)


# ============================================================================
# Config
# ============================================================================


@dataclass
class IntExportConfig:
    """
    Configuration for integer export.

    weight_bits:
        Number of bits for **weights** (supported: 2, 4, 8, or 16).
    act_bits:
        Number of bits for **activations** (supported: 2, 4, 8, or 16).
    signed_weights:
        Whether weights are symmetrically quantized around zero.
    signed_activations:
        Whether activations are symmetrically quantized. Note: the actual
        clipping range for activations comes from `pct.calibrator`, we only
        expose the sign here.
    include_vision_dino / include_vision_siglip / include_llm / include_projector:
        Per-target on/off switches.
    """

    weight_bits: int = 8
    act_bits: int = 8
    signed_weights: bool = True
    signed_activations: bool = True

    include_vision_dino: bool = True
    include_vision_siglip: bool = True
    include_llm: bool = True
    include_projector: bool = True

    def __post_init__(self) -> None:
        valid_bits = (2, 4, 8, 16)

        # Explicitly reject 1-bit configs to avoid implying binary support.
        if self.weight_bits == 1 or self.act_bits == 1:
            raise ValueError(
                f"1-bit quantization is not supported in IntExportConfig "
                f"(got W{self.weight_bits}A{self.act_bits}). "
                f"Use one of {valid_bits} instead."
            )

        if self.weight_bits not in valid_bits:
            raise ValueError(
                f"weight_bits must be one of {valid_bits}, got {self.weight_bits}"
            )
        if self.act_bits not in valid_bits:
            raise ValueError(
                f"act_bits must be one of {valid_bits}, got {self.act_bits}"
            )

    def is_target_included(self, target: Optional[str]) -> bool:
        """
        Check whether a canonical target should be exported.
        Unknown targets are excluded by default.
        """
        if target is None:
            return False
        if target == "vision.dino":
            return self.include_vision_dino
        if target == "vision.siglip":
            return self.include_vision_siglip
        if target == "llm":
            return self.include_llm
        if target == "projector":
            return self.include_projector
        return False


def int_export_config_from_quant_cfg(
    quant_cfg: QuantRuntimeConfig,
    *,
    include_vision_dino: Optional[bool] = None,
    include_vision_siglip: Optional[bool] = None,
    include_llm: Optional[bool] = None,
    include_projector: Optional[bool] = None,
) -> IntExportConfig:
    """
    Helper: derive IntExportConfig from QuantRuntimeConfig.

    Phase 2 goal:
        - bits/backend/targets 的唯一來源是 QuantRuntimeConfig。
        - int_export 僅負責把這些資訊轉成 IntExportConfig，避免 quant_finalize
          再自己決定一次 include_* / bits。

    Args:
        quant_cfg:
            QuantRuntimeConfig produced by `QuantRuntimeConfig.from_bits_backend`.
        include_* (optional overrides):
            若為 None，則預設「由 quant_cfg.use_pct_for 是否包含該 target」決定；
            若不為 None，則以傳入值為準。

    Returns:
        IntExportConfig with:
            - weight_bits / act_bits      ← quant_cfg.weight_bits / quant_cfg.act_bits
            - signed_weights              ← quant_cfg.symmetric_weights
            - signed_activations          ← quant_cfg.symmetric_acts
            - include_*                   ← intersection(override, use_pct_for)
    """
    use_pct_for = set(quant_cfg.use_pct_for)

    def _resolve_flag(name: str, override: Optional[bool]) -> bool:
        if override is not None:
            return override
        return name in use_pct_for

    return IntExportConfig(
        weight_bits=quant_cfg.weight_bits,
        act_bits=quant_cfg.act_bits,
        signed_weights=quant_cfg.symmetric_weights,
        signed_activations=quant_cfg.symmetric_acts,
        include_vision_dino=_resolve_flag("vision.dino", include_vision_dino),
        include_vision_siglip=_resolve_flag("vision.siglip", include_vision_siglip),
        include_llm=_resolve_flag("llm", include_llm),
        include_projector=_resolve_flag("projector", include_projector),
    )


# ============================================================================
# Low-level helpers
# ============================================================================


def _qrange(bits: int, signed: bool) -> Tuple[int, int]:
    if signed:
        qmin = -(1 << (bits - 1))
        qmax = (1 << (bits - 1)) - 1
    else:
        qmin = 0
        qmax = (1 << bits) - 1
    return qmin, qmax


def _affine_quantize_tensor(
    x: torch.Tensor,
    *,
    bits: int,
    signed: bool,
    existing_scale: Optional[torch.Tensor] = None,
    existing_zero_point: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Min-max affine quantization for a single tensor, with optional reuse of
    precomputed (scale, zero_point).

    If `existing_scale` is provided (e.g. from a UniformAffineQuantizer on the
    module), we directly use it (and its zero_point) to quantize `x`. This
    keeps integer weights consistent with how the module was trained/calibrated.

    Otherwise, we fall back to computing affine params from x_min/x_max via
    `compute_affine_params`.

    Returns:
        (x_int, scale_tensor, zero_point_tensor_or_None)
    """
    if x.numel() == 0:
        raise ValueError("Cannot quantize empty tensor")

    # ------------------------------------------------------------------
    # Path 1: reuse provided scale / zero_point (preferred)
    # ------------------------------------------------------------------
    scale_t: Optional[torch.Tensor] = None
    if existing_scale is not None:
        scale_t = existing_scale.detach().clone().to(device=x.device, dtype=x.dtype)
        # guard against degenerate scale
        if torch.all(scale_t == 0):
            scale_t = None

    if scale_t is not None:
        zp_t: Optional[torch.Tensor] = None
        if existing_zero_point is not None:
            zp_t = existing_zero_point.detach().clone().to(device=x.device, dtype=torch.int32)

        qmin, qmax = _qrange(bits, signed)
        dtype_int = torch.int8 if bits <= 8 else torch.int16

        # broadcast-friendly quantization
        if zp_t is not None:
            zp_math = zp_t.to(dtype=x.dtype)
            x_int = torch.round(x / scale_t) + zp_math
        else:
            x_int = torch.round(x / scale_t)

        x_int = x_int.clamp(qmin, qmax).to(dtype=dtype_int)
        return x_int, scale_t, zp_t

    # ------------------------------------------------------------------
    # Path 2: min-max based affine params (fallback)
    # ------------------------------------------------------------------
    x_min = x.min().item()
    x_max = x.max().item()

    # Degenerate case: constant tensor or NaN/Inf
    if (
        not torch.isfinite(torch.tensor(x_min))
        or not torch.isfinite(torch.tensor(x_max))
        or x_min == x_max
    ):
        scale = torch.tensor(1.0, dtype=x.dtype, device=x.device)
        zp = None
        x_int = torch.zeros_like(x, dtype=torch.int8 if bits <= 8 else torch.int16)
        return x_int, scale, zp

    scale_val, zp_int = compute_affine_params(
        x_min=x_min,
        x_max=x_max,
        bits=bits,
        signed=signed,
    )
    scale = torch.tensor(scale_val, dtype=x.dtype, device=x.device)
    zp = torch.tensor(zp_int, dtype=torch.int32, device=x.device) if zp_int is not None else None

    qmin, qmax = _qrange(bits, signed)
    dtype_int = torch.int8 if bits <= 8 else torch.int16

    if zp is not None:
        x_int = torch.round(x / scale) + zp.to(dtype=x.dtype)
    else:
        x_int = torch.round(x / scale)

    x_int = x_int.clamp(qmin, qmax).to(dtype=dtype_int)

    return x_int, scale, zp


def _is_weight_export_candidate(module: nn.Module) -> bool:
    """
    Decide whether a module likely has an exportable weight.

    We intentionally keep this predicate simple: we look for a `weight`
    attribute with Tensor type, and we avoid double-quantized modules.
    """
    if not hasattr(module, "weight"):
        return False
    w = getattr(module, "weight")
    if not isinstance(w, torch.Tensor):
        return False

    # int_* modules already use fake-quant / dequant in forward; we still
    # want to export their underlying float weights, so we allow them.
    return True


def _extract_weight_quantizer(module: nn.Module):
    """
    Pull out the weight quantizer (if any) from a module.

    This is a soft contract: QuantLinear/QuantConv* are expected to expose
    a `weight_quantizer` attribute of type UniformAffineQuantizer when
    weight quantization is enabled.
    """
    q = getattr(module, "weight_quantizer", None)
    if isinstance(q, UniformAffineQuantizer):
        return q
    return None


def _extract_act_quantizer(module: nn.Module):
    """
    Pull out the activation quantizer (if any) from a module.

    We intentionally rely only on a few attribute names instead of class
    type checks, to keep this generic.
    """
    # For QuantLinear / QuantConv*, we expect `act_quantizer`.
    if hasattr(module, "act_quantizer"):
        return getattr(module, "act_quantizer")
    # For other Quant* modules (e.g., QuantMatMul / QuantAdd / QuantSoftmax),
    # different attribute names may exist; extend here if needed.
    return None


# ============================================================================
# Core export routine
# ============================================================================


def export_int_quant_state(
    model: nn.Module,
    cfg: IntExportConfig,
) -> Dict[str, Dict]:
    """
    Analyse a **calibrated & wrapped** model and export integer weights +
    activation quantization parameters according to IntExportConfig.

    Pre-conditions:
        - The model has been wrapped via `integration.wrap_replace.wrap_model_for_quantization`.
        - Activation quantizers have already been calibrated via
          `cobra.quantize.pct.calibrator.calibrate_model_from_stats`, so that
          their `.scale` and `.round_zero_point` fields are populated.
        - If you need projector rotation (R = H K), it should be applied
          **before** calling this function.

    Returns:
        A nested dict with "config", "weights", and "activations" keys.
    """
    export_weights: Dict[str, Dict] = {}
    export_acts: Dict[str, Dict] = {}

    for module_path, module in model.named_modules():
        if module_path == "":
            continue

        target = infer_target_from_module_path(module_path)
        if not cfg.is_target_included(target):
            continue

        # -----------------------------
        # 1) Weight export (W_int)
        # -----------------------------
        if _is_weight_export_candidate(module):
            W = getattr(module, "weight")
            if isinstance(W, torch.Tensor) and W.numel() > 0:
                w_q = _extract_weight_quantizer(module)

                if isinstance(w_q, UniformAffineQuantizer) and getattr(w_q, "scale", None) is not None:
                    # Prefer reusing weight_quantizer's scale / zero_point
                    W_int, W_scale, W_zp = _affine_quantize_tensor(
                        W,
                        bits=cfg.weight_bits,
                        signed=cfg.signed_weights,
                        existing_scale=w_q.scale,
                        existing_zero_point=getattr(w_q, "round_zero_point", None),
                    )
                else:
                    # Fallback: derive params from weight tensor itself
                    W_int, W_scale, W_zp = _affine_quantize_tensor(
                        W,
                        bits=cfg.weight_bits,
                        signed=cfg.signed_weights,
                    )

                export_weights[f"{module_path}.weight"] = {
                    "target": target,
                    "bits": cfg.weight_bits,
                    "signed": cfg.signed_weights,
                    "int_weight": W_int,
                    "scale": W_scale,
                    "zero_point": W_zp,
                }

        # -----------------------------
        # 2) Activation export (scale/zp)
        # -----------------------------
        act_q = _extract_act_quantizer(module)
        if act_q is not None:
            scale = getattr(act_q, "scale", None)
            zp = getattr(act_q, "round_zero_point", None)

            # Only export if calibration has populated scale
            if scale is not None:
                export_acts[module_path] = {
                    "target": target,
                    "bits": cfg.act_bits,
                    "signed": cfg.signed_activations,
                    "scale": scale.detach().clone(),
                    "zero_point": None if zp is None else zp.detach().clone(),
                }

    overwatch.info(
        "[IntExport] Exported quant state | "
        f"weights={len(export_weights)} | activations={len(export_acts)} | "
        f"weight_bits={cfg.weight_bits}, act_bits={cfg.act_bits}",
        extra={
            "weight_bits": cfg.weight_bits,
            "act_bits": cfg.act_bits,
            "signed_weights": cfg.signed_weights,
            "signed_activations": cfg.signed_activations,
        },
    )

    return {
        "config": asdict(cfg),
        "weights": export_weights,
        "activations": export_acts,
    }


# ============================================================================
# Persistence helpers
# ============================================================================


def save_int_export(
    export_blob: Dict[str, Dict],
    out_path: Path | str,
) -> None:
    """
    Save the export blob (produced by `export_int_quant_state`) to disk.

    We use `torch.save` to allow mixed dtypes (int tensors, float tensors,
    Python dicts) without worrying about serialization details.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(export_blob, out_path)

    overwatch.info(
        "[IntExport] Saved integer export blob to %s", str(out_path),
        extra={"out_path": str(out_path)},
    )

