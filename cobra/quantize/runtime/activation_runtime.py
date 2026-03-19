from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from cobra.quantize.runtime.act_policy import normalize_llm_act_mode
from cobra.quantize.runtime.types import RuntimeActivationResult


def apply_activation_calibration(
    vlm: nn.Module,
    *,
    requested: bool,
    pct_hi_lo_path,
    act_bits: Optional[int],
    enabled_targets,
    output_dir,
    llm_act_mode: str,
) -> RuntimeActivationResult:
    act_calib_summary = None
    act_calib_enabled = bool(requested and pct_hi_lo_path is not None)
    error = None

    normalized_mode = normalize_llm_act_mode(llm_act_mode)

    if act_calib_enabled:
        from cobra.quantize.pct.calibrator import calibrate_model_from_hi_lo

        try:
            hi_lo_map = torch.load(pct_hi_lo_path, map_location="cpu")
        except Exception as e:
            hi_lo_map = None
            error = repr(e)
            print(
                f"[WARN] Failed to load pct_hi_lo_path={str(pct_hi_lo_path)!r} "
                f"({repr(e)})"
            )

        if hi_lo_map is not None:
            try:
                act_calib_summary = calibrate_model_from_hi_lo(
                    vlm,
                    hi_lo_map,
                    act_bits=int(act_bits),
                    signed=True,
                    include_targets=sorted(enabled_targets) if enabled_targets else None,
                )
                print(
                    f"[Info] Activation hi/lo calibrated for A{int(act_bits)} "
                    f"(pct_hi_lo_path={str(pct_hi_lo_path)!r}, llm_act_mode={normalized_mode!r})."
                )
            except Exception as e:
                import traceback

                print("[WARN] calibrate_model_from_hi_lo crashed; act_quant will be OFF.")
                print(f"[WARN] Exception: {repr(e)}")
                traceback.print_exc()
                act_calib_enabled = False
                error = repr(e)
        else:
            act_calib_enabled = False

        if output_dir is not None and act_calib_summary is not None:
            try:
                import json

                out_path = output_dir / f"act_calib_A{int(act_bits)}_{normalized_mode}.json"
                out_path.write_text(json.dumps(act_calib_summary, indent=2))
            except Exception:
                pass

    return RuntimeActivationResult(
        requested=bool(requested),
        enabled=bool(act_calib_enabled),
        summary=act_calib_summary,
        pct_hi_lo_path=pct_hi_lo_path,
        act_bits=int(act_bits) if act_bits is not None else None,
        error=error,
    )
