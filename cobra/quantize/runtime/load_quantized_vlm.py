# cobra/quantize/runtime/load_quantized_vlm.py
"""
load_quantized_vlm.py

Unified entrypoint for reconstructing a quantized Cobra VLM from:
    • float checkpoint (loaded by cobra.load)
    • pct_hi_lo  (activation clipping ranges, produced by pct/apply)
    • (optionally) int_export (offline integer export blob; currently not
      used in the PyTorch runtime path)

This file is used ONLY at inference time (eval), not for calibration.

Design aligned with your PTQ pipeline:
    - Use `integration.wrap_replace.wrap_model_for_quantization` to wrap the
      float Cobra model with Quant* modules (int_linear / int_conv / ...).
    - Load (hi, lo) clipping bounds from `pct_hi_lo_path` (torch.save dict).
    - Use `pct.calibrator.calibrate_model_from_hi_lo` to convert (hi, lo) to
      affine activation quantization parameters (scale, zero_point, n_bits)
      and write them into activation quantizers.
    - Activation bitwidth (A) is parsed from `bits` (e.g. "W8A4" -> act_bits=4).

Note:
    - The integer export blob (`int_export_path`) is currently **not** used
      in this runtime loader. It is intended for offline export (ONNX / edge
      deployment). We still load & log its presence for sanity checking.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import torch
from torch import nn

from cobra import load as cobra_load
from cobra.quantize.wrap.entry import wrap_model_for_quantization
from cobra.quantize.pct.calibrator import calibrate_model_from_hi_lo


def _parse_bits(bits: str) -> tuple[int, int]:
    """
    Parse a bitwidth string like "W8A8", "w4a8", etc. into (weight_bits, act_bits).

    Supported bitwidths are {2, 4, 8, 16} for both weights and activations.
    1-bit quantization is intentionally not supported in this PTQ runtime.
    """
    m = re.fullmatch(r"[Ww](\d+)[Aa](\d+)", bits.strip())
    if not m:
        raise ValueError(f"Invalid bits spec {bits!r}; expected like 'W8A8', 'W4A8', ...")
    w_bits = int(m.group(1))
    a_bits = int(m.group(2))
    valid_bits = (2, 4, 8, 16)

    # Explicitly reject 1-bit to avoid giving the illusion of binary support.
    if w_bits == 1 or a_bits == 1:
        raise ValueError(
            f"Unsupported bitwidth combo W{w_bits}A{a_bits}; "
            "1-bit quantization is not supported in this PTQ runtime. "
            f"Use one of W{valid_bits}A{valid_bits}."
        )

    if w_bits not in valid_bits or a_bits not in valid_bits:
        raise ValueError(
            f"Unsupported bitwidth combo W{w_bits}A{a_bits}; "
            f"supported bitwidths are {valid_bits} for both W and A."
        )
    return w_bits, a_bits


def _resolve_model_id_or_path() -> str:
    """
    Resolve which float Cobra checkpoint to load.

    This MUST match what you would normally pass into `cobra.load(...)`
    as `model_id_or_path`. For now we read from an environment variable to
    avoid changing the function signature:

        export COBRA_MODEL_ID_OR_PATH="cobra+3b"
        # or
        export COBRA_MODEL_ID_OR_PATH="/work/.../cobra_1115/run_cobra_3b"

    If unset, we fail loudly.
    """
    env_val = os.environ.get("COBRA_MODEL_ID_OR_PATH", "").strip()
    if not env_val:
        raise ValueError(
            "[load_quantized_cobra_vlm] Missing COBRA_MODEL_ID_OR_PATH; "
            "set it to a valid Cobra model id or run_dir used by cobra.load(...)."
        )
    return env_val


def _load_hi_lo_map(pct_hi_lo_path: Path) -> dict:
    """
    Load the hi/lo clipping map produced by `quant_pct_apply.py`.

    `quant_pct_apply.py` saves a plain dict (hi_lo_map) via `torch.save`.
    """
    pct_hi_lo_path = Path(pct_hi_lo_path)
    if not pct_hi_lo_path.is_file():
        raise FileNotFoundError(
            f"[load_quantized_cobra_vlm] pct_hi_lo_path does not exist: {pct_hi_lo_path}"
        )
    hi_lo_map = torch.load(pct_hi_lo_path, map_location="cpu")
    if not isinstance(hi_lo_map, dict):
        raise TypeError(
            f"[load_quantized_cobra_vlm] Expected hi_lo_map to be a dict, got {type(hi_lo_map)!r}"
        )
    return hi_lo_map


def _maybe_check_int_export(int_export_path: Path) -> None:
    """
    Best-effort sanity check that an int_export blob exists.

    Current runtime path does not consume the blob, but we verify existence and
    basic loadability so that a broken export is caught early.
    """
    int_export_path = Path(int_export_path)
    if not int_export_path:
        return
    if not int_export_path.exists():
        print(f"[load_quantized_cobra_vlm] WARNING: int_export_path={int_export_path} does not exist; "
              f"runtime will proceed without using integer export.")
        return

    try:
        _ = torch.load(int_export_path, map_location="cpu")
        print(f"[load_quantized_cobra_vlm] Found int_export blob at {int_export_path} (not used at runtime).")
    except Exception as e:  # noqa: BLE001
        print(f"[load_quantized_cobra_vlm] WARNING: Failed to load int_export blob at {int_export_path}: {e}")


# ================================================================
#   Main Unified API
# ================================================================
def load_quantized_cobra_vlm(
    *,
    bits: str,
    pct_hi_lo_path: Path,
    int_export_path: Path,
    hf_token: Optional[str],
    base_dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """
    Load a quantized Cobra VLM according to the current PTQ stack.

    Args:
        bits:
            String of the form "W{w_bits}A{a_bits}", e.g. "W8A8", "W4A4",
            "W2A2", "W16A16", "W8A4", "W4A8".
            Supported bitwidths for both weights and activations are
            {2, 4, 8, 16}. 1-bit ("W1A*", "W*A1") configurations are rejected.
            Controls activation bitwidth via calibrator; weight bitwidth is
            currently implicit in the Quant* modules and/or offline int_export.    
        pct_hi_lo_path:
            Path to the hi/lo clipping map produced by `quant_pct_apply.py`.
        int_export_path:
            Path to the integer export blob produced by `quant_finalize.py`.
            Currently only checked for existence; not consumed in this runtime.
        hf_token:
            HF token passed to `cobra.load()`.
        base_dtype:
            Final inference dtype (e.g., torch.bfloat16 or torch.float16).
        device:
            Device placement from accelerate PartialState.

    Returns:
        Quantized Cobra VLM model (wrapped + calibrated), ready for inference.
    """

    # ------------------------------------------------------------------
    # 0. Parse bit configuration
    # ------------------------------------------------------------------
    weight_bits, act_bits = _parse_bits(bits)
    print(f"[load_quantized_cobra_vlm] Requested configuration: W{weight_bits}A{act_bits}")

    # ------------------------------------------------------------------
    # 1. Load FLOAT Cobra model
    # ------------------------------------------------------------------
    model_id_or_path = _resolve_model_id_or_path()
    print(f"[load_quantized_cobra_vlm] Loading float Cobra from {model_id_or_path!r} ...")

    float_vlm = cobra_load(
        model_id_or_path,
        hf_token=hf_token,
    )
    float_vlm.to(device=device, dtype=base_dtype)

    # ------------------------------------------------------------------
    # 2. Wrap model with Quant* modules
    # ------------------------------------------------------------------
    print("[load_quantized_cobra_vlm] Wrapping model with Quant* modules via wrap_model_for_quantization(...)")
    wrap_model_for_quantization(
        float_vlm,
        # For now we use default policy / manifest / params; they already
        # route modules into the four canonical targets.
        policy_cfg=None,
        manifest=None,
        default_params=None,
        prefix="",
    )

    # ------------------------------------------------------------------
    # 3. Load activation clipping map (pct_hi_lo) and calibrate activations
    # ------------------------------------------------------------------
    print(f"[load_quantized_cobra_vlm] Loading activation hi/lo map from {pct_hi_lo_path}")
    hi_lo_map = _load_hi_lo_map(pct_hi_lo_path)

    print(f"[load_quantized_cobra_vlm] Calibrating activation quantizers (act_bits={act_bits}) ...")
    _ = calibrate_model_from_hi_lo(
        model=float_vlm,
        hi_lo_map=hi_lo_map,
        act_bits=act_bits,
        signed=True,
        include_targets=None,  # all four canonical targets
    )

    # ------------------------------------------------------------------
    # 4. (Optional) Sanity-check integer export blob
    # ------------------------------------------------------------------
    _maybe_check_int_export(int_export_path)

    print("[load_quantized_cobra_vlm] Quantized Cobra ready (wrapped + calibrated).")
    return float_vlm


