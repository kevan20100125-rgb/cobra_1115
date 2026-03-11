# cobra/quantize/runtime/load_quantized_vlm.py
"""
Runtime loader for Cobra PTQ baseline.

Current behavior (kept stable):
  - activation remains float (no pct_hi_lo dependency, no activation calibration)
  - weights can be fake-quantized when W-bits are provided (W8/W4/W2)
  - fusion wrapping disabled (avoid coverage/latency contamination)

New behavior (interface only; no A-quant enabled yet):
  - Accept richer BITS specs (pass-through from upstream), e.g.:
      W8, W4, W2, W4A8, W8A8, A8
  - Parse into (w_bits, a_bits) but keep a_bits as a no-op for now.
  - Do NOT fail fast or warn for unsupported A bits (per user request).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import nn

from cobra import load as cobra_load
from cobra.quantize.utils import set_quant_state
from cobra.quantize.wrap.entry import wrap_model_for_quantization
from cobra.quantize.wrap.policy import WrapPolicyConfig


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def _resolve_model_id_or_path() -> str:
    """
    Resolve the float checkpoint id/path for cobra.load(...).

    Priority:
      1) COBRA_MODEL_ID_OR_PATH (explicit)
      2) COBRA_MODEL_BASE_ID    (script-provided)
      3) fallback: "cobra+3b"
    """
    v = os.environ.get("COBRA_MODEL_ID_OR_PATH")
    if v and v.strip():
        return v.strip()

    v = os.environ.get("COBRA_MODEL_BASE_ID")
    if v and v.strip():
        return v.strip()

    return "cobra+3b"


def parse_bits_spec(bits: Optional[str]) -> Tuple[Optional[int], Optional[int]]:
    """
    Parse a flexible BITS spec into (w_bits, a_bits).

    Accepted (case-insensitive; whitespace ignored):
      - "W8" / "W4" / "W2"
      - "W4A8" / "W8A8" / "W2A2"  (A part parsed but is NO-OP currently)
      - "A8" / "A16"             (activation-only request; NO-OP currently)

    Returns:
      (w_bits, a_bits), each Optional[int].
      If parsing fails, returns (None, None) without raising.
    """
    if bits is None:
        return None, None

    s = str(bits).strip().upper()
    if not s:
        return None, None

    # W-only: W{n}
    if s.startswith("W") and "A" not in s:
        try:
            w = int(s[1:])
        except Exception:
            return None, None
        return w, None

    # WA: W{n}A{m}
    if s.startswith("W") and "A" in s:
        w_part, a_part = s.split("A", 1)
        if not w_part.startswith("W"):
            return None, None
        try:
            w = int(w_part[1:])
        except Exception:
            w = None
        try:
            a = int(a_part) if a_part else None
        except Exception:
            a = None
        return w, a

    # A-only: A{m}
    if s.startswith("A"):
        try:
            a = int(s[1:])
        except Exception:
            return None, None
        return None, a

    return None, None


def _apply_runtime_weight_bits(vlm: nn.Module, w_bits: int) -> None:
    """
    Propagate the desired weight bit-width to all Quant* modules (weights only).
    """
    # Local imports to avoid heavier import costs and circular deps.
    from cobra.quantize.int_linear import QuantLinear
    from cobra.quantize.int_conv import QuantConv1d, QuantConv2d
    from cobra.quantize.int_matmul import QuantMatMul

    num_modules = 0
    for _, module in vlm.named_modules():
        if isinstance(module, (QuantLinear, QuantConv1d, QuantConv2d, QuantMatMul)):
            module.change_bits(weight_bits=w_bits, act_bits=None)
            num_modules += 1

    print(f"[Info] Runtime weight_bits applied to {num_modules} Quant* modules (W{w_bits}).")


def _iter_wrap_registry_entries(registry):
    """
    Extract (target, module_path) from WrapRegistry in a version-tolerant way.
    """
    if registry is None:
        return
    if hasattr(registry, "items") and callable(registry.items):
        for module_path, entry in registry.items():
            target = getattr(entry, "target", None) or getattr(entry, "kind", None) or "unknown"
            yield str(target), str(module_path)
        return

    entries = getattr(registry, "entries", None)
    if entries is None:
        return

    if isinstance(entries, dict):
        for module_path, entry in entries.items():
            target = getattr(entry, "target", None) or getattr(entry, "kind", None) or "unknown"
            yield str(target), str(module_path)
        return

    if isinstance(entries, (list, tuple)):
        for e in entries:
            if isinstance(e, (list, tuple)) and len(e) == 2:
                yield str(e[0]), str(e[1])
            else:
                target = getattr(e, "target", None) or getattr(e, "kind", None) or "unknown"
                module_path = getattr(e, "module_path", None) or getattr(e, "path", None) or "<unknown>"
                yield str(target), str(module_path)
        return


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on", "enable", "enabled", "hadamard"):
        return True
    if s in ("0", "false", "no", "n", "off", "disable", "disabled", ""):
        return False
    return default


def _extract_llm_for_mixer_rotation(vlm: nn.Module) -> nn.Module:
    """
    Robustly extract the underlying LLM module from a VLM wrapper.
    """
    lb = getattr(vlm, "llm_backbone", None)
    if lb is not None:
        cand = getattr(lb, "llm", None)
        if isinstance(cand, nn.Module):
            return cand
        cand = getattr(lb, "model", None)
        if isinstance(cand, nn.Module):
            return cand
        cand = getattr(lb, "backbone", None)
        if isinstance(cand, nn.Module):
            return cand

    cand = getattr(vlm, "llm", None)
    if isinstance(cand, nn.Module):
        return cand

    return vlm


def _maybe_enable_llm_mixer_hadamard(llm, *, ptq_stage: int = 0):
    """
    Enable Quamba-structured mixer rotation (input + optional output transform) with KLT+Hadamard.
    """
    import os
    from cobra.quantize.rotate.mixer import MixerHadamardRotationConfig, rotate_llm_mamba_mixers_hadamard_inplace

    enabled = os.environ.get("COBRA_LLM_MIXER_HADAMARD", "0").strip() == "1"
    if not enabled:
        return

    block = int(os.environ.get("COBRA_LLM_MIXER_BLOCK", "512").strip() or "512")
    targets_s = os.environ.get("COBRA_LLM_MIXER_TARGETS", "out_proj").strip()
    targets = tuple([t.strip() for t in targets_s.split(",") if t.strip()]) or ("out_proj",)

    max_layers_s = os.environ.get("COBRA_LLM_MIXER_MAX_LAYERS", "").strip()
    max_layers = int(max_layers_s) if max_layers_s else None

    dry_run = os.environ.get("COBRA_LLM_MIXER_DRY_RUN", "0").strip() == "1"

    act_klt = os.environ.get("COBRA_LLM_MIXER_ACT_KLT", "0").strip() == "1"
    strict = os.environ.get("COBRA_LLM_MIXER_ACT_KLT_STRICT", "0").strip() == "1"

    out_tx = os.environ.get("COBRA_LLM_MIXER_OUT_TRANSFORM", "0").strip() == "1"

    in_path = os.environ.get("ACT_KLT_OUTPROJ_IN", "").strip()
    if not in_path:
        in_path = os.environ.get("ACT_KLT_OUT", "").strip()  # legacy

    out_path = os.environ.get("ACT_KLT_OUTPROJ_OUT", "").strip()

    cfg = MixerHadamardRotationConfig(
        enabled=True,
        block_size=block,
        targets=targets,
        max_layers=max_layers,
        dry_run=dry_run,
        act_klt_enabled=act_klt,
        act_klt_strict=strict,
    )

    setattr(cfg, "out_transform_enabled", bool(out_tx))
    if in_path:
        setattr(cfg, "act_klt_in_path", in_path)
    if out_path:
        setattr(cfg, "act_klt_out_path", out_path)

    report = rotate_llm_mamba_mixers_hadamard_inplace(llm, cfg=cfg)

    applied = len(report.get("applied", []) or [])
    skipped = len(report.get("skipped", []) or [])
    print(
        f"[MixerRotation] enabled=1 mode=quamba_kh block={block} "
        f"targets={targets} max_layers={max_layers} dry_run={dry_run} "
        f"act_klt={int(act_klt)} out_tx={int(out_tx)} strict={int(strict)} "
        f"applied={applied} skipped={skipped}"
    )

def _restrict_llm_act_quant_to_out_proj(vlm: nn.Module) -> None:
    """
    Keep activation quantization enabled ONLY for:
        llm_backbone.llm.backbone.layers.*.mixer.out_proj

    All other LLM-side Quant* modules keep their wrapper structure and can still
    participate in weight quantization, but their activation quantizers are disabled.

    IMPORTANT:
      - This helper must be called AFTER the global set_quant_state(...)
        because set_quant_state(...) will otherwise re-enable act_quant broadly.
      - We only touch modules under llm_backbone.llm.*.
    """
    from cobra.quantize.int_linear import QuantLinear
    from cobra.quantize.int_conv import QuantConv1d, QuantConv2d
    from cobra.quantize.int_matmul import QuantMatMul

    keep_suffix = ".mixer.out_proj"

    num_keep = 0
    num_disable = 0

    for module_path, module in vlm.named_modules():
        if not module_path.startswith("llm_backbone.llm."):
            continue

        if not isinstance(module, (QuantLinear, QuantConv1d, QuantConv2d, QuantMatMul)):
            continue

        keep = module_path.endswith(keep_suffix)

        # Wrapper-level gate
        if hasattr(module, "use_act_quant"):
            module.use_act_quant = bool(keep)

        # Underlying quantizer-level gates
        for attr in ("act_quantizer", "x1_quantizer", "x2_quantizer"):
            q = getattr(module, attr, None)
            if q is None:
                continue
            q.set_quant_state(enable=keep, is_dynamic=False)

            if not keep:
                q.is_observing = False
                q.observered = False

        if keep:
            num_keep += 1
        else:
            num_disable += 1

    print(
        f"[Info] Restricted llm activation quant to out_proj only: "
        f"keep={num_keep} disable={num_disable} suffix={keep_suffix!r}"
    )

# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------

def load_quantized_cobra_vlm(
    *,
    bits: Optional[str],
    pct_hi_lo_path,  # now used when A-bits are requested
    hf_token: str,
    base_dtype,
    device,
    enabled_targets=None,
    run_dir=None,
    output_dir=None,
):
    """
    Runtime loader for Cobra fake-quant PTQ.

    Supported:
      - W-only:  W8 / W4 / W2
      - WA:      W8A8 / W4A8 / ...
      - A-only:  A8 / A16

    New behavior:
      - If env COBRA_LLM_ACT_ONLY=out_proj, then AFTER global quant-state enable,
        we restrict LLM activation quantization to:
            llm_backbone.llm.backbone.layers.*.mixer.out_proj
        while leaving all other LLM Quant* modules wrapped and eligible for W-quant.
    """
    import json
    from collections import defaultdict

    # -----------------------------
    # 0) Entry diagnostics
    # -----------------------------
    try:
        this_file = __file__
    except Exception:
        this_file = "<unknown>"

    llm_act_only = os.environ.get("COBRA_LLM_ACT_ONLY", "").strip().lower()

    print(f"[INFO] load_quantized_cobra_vlm ENTER  file={this_file}")
    print(
        "[INFO] env "
        f"COBRA_LLM_MIXER_HADAMARD={os.environ.get('COBRA_LLM_MIXER_HADAMARD', '')!r} "
        f"COBRA_LLM_MIXER_ACT_KLT={os.environ.get('COBRA_LLM_MIXER_ACT_KLT', '')!r} "
        f"COBRA_DISABLE_MAMBA_FAST_PATH={os.environ.get('COBRA_DISABLE_MAMBA_FAST_PATH', '')!r} "
        f"COBRA_LLM_ACT_ONLY={llm_act_only!r}"
    )
    print(
        f"[INFO] args bits={bits!r} pct_hi_lo_path={pct_hi_lo_path!r} "
        f"enabled_targets={enabled_targets!r}"
    )

    # -----------------------------
    # 0.1) Normalize enabled_targets
    # -----------------------------
    if enabled_targets is None:
        enabled_targets_set = {"vision.dino", "vision.siglip", "llm", "projector"}
    else:
        enabled_targets_set = set(enabled_targets)

    # -----------------------------
    # 0.2) Parse bits
    # -----------------------------
    w_bits, a_bits = parse_bits_spec(bits)
    do_weight = w_bits is not None
    do_act = a_bits is not None

    if not do_weight and not do_act:
        model_id_or_path = _resolve_model_id_or_path()
        print(f"[load_quantized_cobra_vlm] bits={bits!r} -> loading FLOAT Cobra from {model_id_or_path!r} ...")
        vlm = cobra_load(model_id_or_path, hf_token=hf_token)
        vlm.to(device=device, dtype=base_dtype)
        return vlm

    # -----------------------------
    # 0.3) Output dir best-effort
    # -----------------------------
    if output_dir is None:
        if run_dir is not None:
            output_dir = Path(run_dir) / "outputs" / "quantize"
        elif pct_hi_lo_path is not None:
            try:
                output_dir = Path(pct_hi_lo_path).parent
            except Exception:
                output_dir = None

    if output_dir is not None:
        try:
            Path(output_dir).mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"[WARN] output_dir mkdir failed: {output_dir!r} ({repr(e)})")
            output_dir = None

    # -----------------------------
    # 0.4) A-quant requires pct_hi_lo_path
    # -----------------------------
    if do_act and pct_hi_lo_path is None:
        print(
            f"[WARN] A-bits requested (A{a_bits}) but pct_hi_lo_path is None -> act_quant will stay OFF."
        )

    # -----------------------------
    # 0.5) If mixer rotation enabled, force Mamba slow path
    # -----------------------------
    mixer_rot_enabled = _env_flag("COBRA_LLM_MIXER_HADAMARD", default=False)
    if mixer_rot_enabled:
        if os.environ.get("COBRA_DISABLE_MAMBA_FAST_PATH", "").strip() == "":
            os.environ["COBRA_DISABLE_MAMBA_FAST_PATH"] = "1"
            print("[MixerRotation] Auto-set COBRA_DISABLE_MAMBA_FAST_PATH=1 (required for out_proj hooks).")
    else:
        print("[MixerRotation] gate OFF (COBRA_LLM_MIXER_HADAMARD is not enabled).")

    # -----------------------------
    # 1) Load float Cobra VLM
    # -----------------------------
    model_id_or_path = _resolve_model_id_or_path()
    print(f"[load_quantized_cobra_vlm] loading FLOAT Cobra from {model_id_or_path!r} ...")
    vlm = cobra_load(model_id_or_path, hf_token=hf_token)
    vlm.to(device=device, dtype=base_dtype)

    # -----------------------------
    # 2) Wrap model into Quant* modules
    # -----------------------------
    wrap_policy_cfg = WrapPolicyConfig(
        enable_vision_dino="vision.dino" in enabled_targets_set,
        enable_vision_siglip="vision.siglip" in enabled_targets_set,
        enable_llm="llm" in enabled_targets_set,
        enable_projector="projector" in enabled_targets_set,
        enable_fusion=False,  # hard lock
        include_linear=True,
        include_conv=True,
    )

    registry = wrap_model_for_quantization(
        vlm,
        policy_cfg=wrap_policy_cfg,
        manifest=None,
        default_params=None,
        prefix="",
    )

    # -----------------------------
    # 2.5) Enable LLM mixer rotation AFTER wrapping
    # -----------------------------
    try:
        llm = _extract_llm_for_mixer_rotation(vlm)
        llm_type = type(llm).__name__
        has_layers = bool(
            (hasattr(llm, "backbone") and hasattr(llm.backbone, "layers")) or hasattr(llm, "layers")
        )
        print(f"[MixerRotation] resolved_llm={llm_type} has_layers={int(has_layers)}")

        _maybe_enable_llm_mixer_hadamard(llm)
    except Exception as e:
        import traceback
        print("[MixerRotation] ERROR: mixer rotation crashed.")
        print(f"[MixerRotation] Exception: {repr(e)}")
        traceback.print_exc()

    # -----------------------------
    # 3) Apply runtime weight bits (optional)
    # -----------------------------
    if do_weight:
        _apply_runtime_weight_bits(vlm, w_bits=int(w_bits))
    else:
        print("[Info] W-bits not requested -> weight quant will stay OFF (weights remain float).")

    # -----------------------------
    # 4) Apply activation hi/lo calibration (optional; requires pct_hi_lo_path)
    # -----------------------------
    act_calib_summary = None
    act_calib_enabled = bool(do_act and pct_hi_lo_path is not None)

    if act_calib_enabled:
        from cobra.quantize.pct.calibrator import calibrate_model_from_hi_lo

        try:
            hi_lo_map = torch.load(pct_hi_lo_path, map_location="cpu")
        except Exception as e:
            hi_lo_map = None
            print(f"[WARN] Failed to load pct_hi_lo_path={pct_hi_lo_path!r} ({repr(e)})")

        if hi_lo_map is not None:
            try:
                act_calib_summary = calibrate_model_from_hi_lo(
                    vlm,
                    hi_lo_map,
                    act_bits=int(a_bits),
                    signed=True,
                    include_targets=sorted(enabled_targets_set) if enabled_targets_set else None,
                )
                print(f"[Info] Activation hi/lo calibrated for A{int(a_bits)} (pct_hi_lo_path={pct_hi_lo_path!r}).")
            except Exception as e:
                import traceback
                print("[WARN] calibrate_model_from_hi_lo crashed; act_quant will be OFF.")
                print(f"[WARN] Exception: {repr(e)}")
                traceback.print_exc()
                act_calib_enabled = False
        else:
            act_calib_enabled = False

        if output_dir is not None and act_calib_summary is not None:
            try:
                out_path = Path(output_dir) / f"act_calib_A{int(a_bits)}.json"
                out_path.write_text(json.dumps(act_calib_summary, indent=2))
            except Exception:
                pass

    # -----------------------------
    # 5) Enable quant flags globally
    # -----------------------------
    set_quant_state(vlm, weight_quant=bool(do_weight), act_quant=bool(act_calib_enabled))

    # -----------------------------
    # 5.1) Optional post-gate:
    #      keep LLM activation quant ONLY on *.mixer.out_proj
    # -----------------------------
    if act_calib_enabled and llm_act_only == "out_proj":
        _restrict_llm_act_quant_to_out_proj(vlm)

    # -----------------------------
    # 6) Emit coverage
    # -----------------------------
    if output_dir is not None and registry is not None:
        by_target = defaultdict(list)
        for target, module_path in _iter_wrap_registry_entries(registry):
            by_target[target].append(module_path)

        bits_effective = []
        if do_weight:
            bits_effective.append(f"W{int(w_bits)}")
        if do_act and act_calib_enabled:
            bits_effective.append(f"A{int(a_bits)}")
        elif do_act and not act_calib_enabled:
            bits_effective.append(f"A{int(a_bits)}(OFF)")
        bits_effective = "".join(bits_effective) if bits_effective else "FLOAT"

        coverage_payload = {
            "stage": "runtime_WA" if (do_weight and do_act) else ("runtime_W_only" if do_weight else "runtime_A_only"),
            "bits_raw": bits,
            "bits_effective": bits_effective,
            "backend": "fake",
            "act_pct_hi_lo_path": str(pct_hi_lo_path) if pct_hi_lo_path is not None else None,
            "llm_act_only": llm_act_only if llm_act_only else None,
            "counts": {k: len(v) for k, v in by_target.items()},
            "module_paths": dict(by_target),
        }

        out_name = f"coverage_{bits_effective}.json".replace("/", "_")
        out_path = Path(output_dir) / out_name
        try:
            out_path.write_text(json.dumps(coverage_payload, indent=2))
        except Exception:
            pass

    return vlm
