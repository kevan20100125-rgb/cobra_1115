# cobra/quantize/pct/calibrator.py 

"""
Calibration utilities: apply (hi, lo) clipping bounds to quantized modules.

Responsibilities:
    - Take per-(target, module) clipping bounds (hi, lo) produced by apply.py.
    - Write (hi, lo) into activation quantizers in a bit-agnostic fashion.
    - Honor the configured activation bitwidth by setting quantizer.n_bits.
    - Traverse a model, find quantized modules (e.g. QuantLinear / QuantConv / QuantMatMul),
      and write the calibrated parameters into their activation quantizers.

Design for W4/W8 fake-quant study:
    - pct_stats -> pct_hi_lo: bit-agnostic clipping map.
    - calibrator: injects (hi, lo) into activation quantizers via
          UniformAffineQuantizer.set_clipping_range(xmin=lo, xmax=hi)
      and sets act_bits via change_n_bits(act_bits).
    - Actual fake quantization is performed at runtime inside the quantizers.

This module does NOT:
    - Attach observers or run data through the model (that's collect.py).
    - Decide which percentile to use (that's best_percentile.py + apply.py).
    - Export integer weights / kernels (that's finalize/int_export.py).

Typical usage (inside a CLI script):

    stats = torch.load(pct_stats_path)
    # hi/lo map per record
    hi_lo_map = build_hi_lo_map(stats, best_percent_map=None, symmetric=True)
    # apply to model
    summary = calibrate_model_from_hi_lo(
        model,
        hi_lo_map,
        act_bits=args.act_bits,
        signed=True,
    )

"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn

from cobra.overwatch import initialize_overwatch

from cobra.quantize.quantizer import UniformAffineQuantizer

from .schema import compute_affine_params, normalize_target
from .apply import HiLoRecord, build_hi_lo_map

overwatch = initialize_overwatch(__name__)


# Canonical target vocabulary used throughout the PTQ stack.
_CANONICAL_TARGETS: Tuple[str, ...] = (
    "fusion",
    "vision.dino",
    "vision.siglip",
    "llm",
    "projector",
)

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _infer_module_device(module: nn.Module) -> torch.device:
    """
    Best-effort inference of the device on which a module lives.

    We prefer a parameter device, then a buffer device, and finally fall back
    to CPU.
    """
    for p in module.parameters(recurse=False):
        return p.device
    for b in module.buffers(recurse=False):
        return b.device
    return torch.device("cpu")


def _apply_hi_lo_to_quantizer(
    quant: UniformAffineQuantizer,
    hi: float,
    lo: float,
    *,
    act_bits: int,
    signed: bool,  # kept for API symmetry; sign is handled by quantizer.symmetric
) -> None:
    """
    Write calibrated (hi, lo) bounds into a UniformAffineQuantizer.

    This is the core bridge between the bit-agnostic pct_hi_lo map and the
    bit-specific fake-quant quantizer:

        - Set n_bits for this quantizer.
        - Install xmin/xmax via set_clipping_range(xmin=lo, xmax=hi).
        - let the quantizer compute its own scale / zero_point from these.
    """
    device = _infer_module_device(quant)

    # Configure bitwidth first so that recomputed scales use the correct bits
    quant.change_n_bits(act_bits)

    # Move bounds onto the same device as the quantizer
    xmin = torch.tensor(float(lo), dtype=torch.float32, device=device)
    xmax = torch.tensor(float(hi), dtype=torch.float32, device=device)

    # This call freezes the quantizer into static mode and recomputes
    # (scale, zero_point) internally.
    quant.set_clipping_range(xmin=xmin, xmax=xmax)

    # Ensure we are not in dynamic / observing mode anymore
    if hasattr(quant, "set_quant_state"):
        quant.set_quant_state(enable=True, is_dynamic=False)
    else:
        if hasattr(quant, "is_dynamic_quant"):
            quant.is_dynamic_quant = False
        if hasattr(quant, "is_observing"):
            quant.is_observing = False


def _iter_child_activation_quantizers(module: nn.Module):
    """
    Yield all UniformAffineQuantizer instances that appear as activation
    quantizers on child modules of `module`.

    This is used as a fallback when `module` itself is a container
    (e.g. a whole Mamba layer or a vision backbone block) and the actual
    Quant* modules live inside.
    """
    for child in module.modules():
        # Skip the module itself; we only want strict children
        if child is module:
            continue

        # Pattern A: child exposing a single activation quantizer as act_quantizer
        q = getattr(child, "act_quantizer", None)
        if isinstance(q, UniformAffineQuantizer):
            yield q

        # Pattern B: QuantMatMul-style child modules with x1/x2 activation quantizers
        for attr in ("x1_quantizer", "x2_quantizer"):
            q = getattr(child, attr, None)
            if isinstance(q, UniformAffineQuantizer):
                yield q


def _calibrate_single_module(
    module: nn.Module,
    module_name: str,
    hi: float,
    lo: float,
    *,
    act_bits: int,
    signed: bool,
) -> bool:
    """
    Apply calibration to a single module if it exposes known activation quantizers.

    The calibration strategy is:

        - First, try quantizers attached directly to this module
          (act_quantizer, x1_quantizer, x2_quantizer).
        - If none are found, fall back to scanning child modules for
          activation quantizers, and apply the same (hi, lo, act_bits)
          to all of them.

    Returns:
        True if any quantizer on this module or its children was updated,
        False otherwise.
    """
    applied = False
    direct_applied = False
    fallback_count = 0

    # Pattern 1: modules exposing a single activation quantizer as act_quantizer
    q = getattr(module, "act_quantizer", None)
    if isinstance(q, UniformAffineQuantizer):
        _apply_hi_lo_to_quantizer(q, hi=hi, lo=lo, act_bits=act_bits, signed=signed)
        applied = True
        direct_applied = True

    # Pattern 2: QuantMatMul-style modules with x1/x2 activation quantizers
    for attr in ("x1_quantizer", "x2_quantizer"):
        q = getattr(module, attr, None)
        if isinstance(q, UniformAffineQuantizer):
            _apply_hi_lo_to_quantizer(q, hi=hi, lo=lo, act_bits=act_bits, signed=signed)
            applied = True
            direct_applied = True

    # Pattern 3 (fallback): scan child modules for activation quantizers
    if not applied:
        for q in _iter_child_activation_quantizers(module):
            _apply_hi_lo_to_quantizer(q, hi=hi, lo=lo, act_bits=act_bits, signed=signed)
            applied = True
            fallback_count += 1

        if fallback_count > 0:
            overwatch.debug(
                "[PctCalib] Fallback container calibration on module=%r: "
                "applied to %d child activation quantizer(s)",
                module_name,
                fallback_count,
                extra={
                    "pct_module": module_name,
                    "pct_fallback_children": fallback_count,
                },
            )

    # In rare cases both direct + fallback could trigger (e.g. mixed patterns),
    # but for our current Quant* design this should be either-or.
    if applied and not direct_applied and fallback_count == 0:
        # Sanity log if we ever flip this logic in the future.
        overwatch.debug(
            "[PctCalib] Applied hi/lo to module=%r without discovering any "
            "known activation quantizer attributes; this should not happen.",
            module_name,
            extra={"pct_module": module_name},
        )

    return applied


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def calibrate_model_from_hi_lo(
    model: nn.Module,
    hi_lo_map: Mapping[str, HiLoRecord],
    *,
    act_bits: int = 8,
    signed: bool = True,
    include_targets: Optional[Sequence[str]] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Apply (hi, lo) clipping bounds to a model's quantized activation paths.

    Args:
        model:
            A Cobra model (or any nn.Module) that has already been wrapped with
            quantized modules (e.g. QuantLinear / QuantConv / QuantMatMul).
        hi_lo_map:
            Mapping from record-key -> HiLoRecord, as produced by
            apply.build_hi_lo_map(...). Each HiLoRecord contains:
                - "target": canonical target name
                - "module": module qualified name (from model.named_modules())
                - "percent": selected clipping percentile (float)
                - "hi": upper bound (>= 0)
                - "lo": lower bound (<= 0)
        act_bits:
            Bitwidth for activation quantization.
            Supported: {2, 4, 8, 16}. (1-bit is intentionally not supported in this PTQ stack.)
            This bitwidth is applied uniformly to all calibrated activation quantizers.
        signed:
            If True, we treat activations as signed (typical for centered
            features) when *reporting* scale/zero in the summary. The actual
            quantizer behavior is governed by its own `symmetric` flag.
        include_targets:
            Optional list/tuple of target names to include. Each entry can be
            any alias accepted by normalize_target. If None, all targets in
            hi_lo_map are processed.

    Returns:
        summary:
            Mapping from module_name -> dict with fields:
                - "target"
                - "percent"
                - "hi"
                - "lo"
                - "scale"
                - "zero_point"
                - "act_bits"
    """
    # Normalize include_targets if provided
    allowed_targets: Optional[Sequence[str]] = None
    if include_targets is not None:
        allowed_targets = [normalize_target(t) for t in include_targets]

    name_to_module: Dict[str, nn.Module] = dict(model.named_modules())
    summary: Dict[str, Dict[str, float]] = {}

    total_records = 0
    applied_records = 0

    # Per-target coverage stats
    coverage: Dict[str, Dict[str, int]] = {}

    for record_key, rec in hi_lo_map.items():
        total_records += 1

        target_raw = rec.get("target", "")
        module_name = rec.get("module", None)

        # Make missing/empty module field debuggable (avoid silent "" / confusing logs)
        if module_name is None or (isinstance(module_name, str) and not module_name.strip()):
            module_name = "<unknown>"
            overwatch.warning(
                f"[PctCalib] Record={record_key!r} has empty module field (target={target_raw!r}); "
                "cannot match model.named_modules().",
                extra={"record": record_key, "target": target_raw, "module": module_name},
            )
        percent = float(rec.get("percent", 0.0))
        hi = float(rec.get("hi", 0.0))
        lo = float(rec.get("lo", 0.0))

        try:
            target = normalize_target(str(target_raw))
        except KeyError:
            overwatch.warning(
                f"[PctCalib] Skip record={record_key!r}: unknown target={target_raw!r}",
                extra={"target": target_raw, "record": record_key},
            )
            continue

        if allowed_targets is not None and target not in allowed_targets:
            continue

        cov = coverage.setdefault(
            target,
            {
                "records": 0,
                "found_modules": 0,
                "calibrated_modules": 0,
            },
        )
        cov["records"] += 1

        if module_name not in name_to_module:
            overwatch.warning(
                f"[PctCalib] Skip record={record_key!r}: module={module_name!r} not found in model "
                f"(target={target!r}).",
                extra={"record": record_key, "target": target, "module": module_name},
            )
            continue

        cov["found_modules"] += 1
        module = name_to_module[module_name]

        applied = _calibrate_single_module(
            module,
            module_name,
            hi=hi,
            lo=lo,
            act_bits=act_bits,
            signed=signed,
        )

        if not applied:
            overwatch.warning(
                f"[PctCalib] No known activation quantizer found on module={module_name!r} "
                f"(target={target!r}); record={record_key!r} ignored.",
                extra={"target": target, "pct_module": module_name},
            )
            continue

        applied_records += 1
        cov["calibrated_modules"] += 1

        # For summary/reporting purposes, compute scale/zero with the
        # same analytic helper used elsewhere. This does not affect the
        # internal quantizer state, which already computed its own scale/zp
        # from (hi, lo) and act_bits.
        x_min = lo
        x_max = hi
        scale, zero = compute_affine_params(x_min=x_min, x_max=x_max, bits=act_bits, signed=signed)

        summary[module_name] = {
            "target": target,
            "percent": percent,
            "hi": hi,
            "lo": lo,
            "scale": float(scale),
            "zero_point": float(zero),
            "act_bits": float(act_bits),
        }

        overwatch.info(
            (
                "[PctCalib] target=%r module=%r percentile=%7.3f "
                "hi=%+.6f lo=%+.6f scale=%e zero_point=%+.1f bits=%d"
            )
            % (target, module_name, percent, hi, lo, scale, zero, act_bits),
            extra={"target": target, "pct_module": module_name},
        )

    # Per-target coverage summary (esp. for vision targets)
    for tgt, cov in coverage.items():
        records = cov.get("records", 0)
        found = cov.get("found_modules", 0)
        calibrated = cov.get("calibrated_modules", 0)

        coverage_ratio = float(calibrated) / float(found) if found > 0 else 0.0

        overwatch.info(
            "[PctCalib] Coverage | target=%r records=%d found_modules=%d "
            "calibrated_modules=%d coverage_ratio=%.3f",
            tgt,
            records,
            found,
            calibrated,
            coverage_ratio,
            extra={
                "target": tgt,
                "pct_records": records,
                "pct_found_modules": found,
                "pct_calibrated_modules": calibrated,
                "pct_coverage_ratio": coverage_ratio,
            },
        )

    overwatch.info(
        "[PctCalib] Finished calibration from hi/lo: total_records=%d applied_records=%d "
        "summary_modules=%d",
        total_records,
        applied_records,
        len(summary),
        extra={
            "pct_total_records": total_records,
            "pct_applied_records": applied_records,
            "pct_summary_modules": len(summary),
        },
    )

    return summary


def calibrate_model_from_stats(
    model: nn.Module,
    stats: Mapping[str, Mapping[str, Any]],
    *,
    best_percent_map: Optional[Mapping[str, float]] = None,
    tau_growth: float = 5.0,
    symmetric: bool = True,
    act_bits: int = 8,
    signed: bool = True,
    include_targets: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, HiLoRecord]]:
    """
    High-level convenience API: stats → hi/lo → write into model.

    Args:
        model:
            Cobra model already wrapped with quantized modules.
        stats:
            Percentile stats payload produced by collect.build_activation_stats.
        best_percent_map:
            Optional {target -> percentile} map. If None, this function will
            derive one using best_percentile.get_best_percentile_map.
        tau_growth:
            Growth-ratio threshold for best-percentile selection.
        symmetric:
            Whether build_hi_lo_map enforces symmetric hi/lo around 0.
        act_bits:
            Activation bitwidth to apply to all calibrated quantizers.
        signed:
            Whether to treat activations as signed when computing scale/zp for
            reporting purposes.
        include_targets:
            Optional list of targets to include (aliases accepted).

    Returns:
        (summary, hi_lo_map):
            - summary: same as calibrate_model_from_hi_lo return value.
            - hi_lo_map: the intermediate {record_key -> HiLoRecord} used.
    """
    hi_lo_map = build_hi_lo_map(
        stats=stats,
        best_percent_map=best_percent_map,
        tau_growth=tau_growth,
        symmetric=symmetric,
        include_targets=include_targets,
    )

    summary = calibrate_model_from_hi_lo(
        model,
        hi_lo_map=hi_lo_map,
        act_bits=act_bits,
        signed=signed,
        include_targets=include_targets,
    )

    return summary, hi_lo_map


