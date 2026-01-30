# cobra/quantize/runtime/load_quantized_vlm.py
"""
Runtime loader for Cobra PTQ baseline (weight-only fake-quant).

Guarantees:
  - activation remains float (no pct_hi_lo dependency, no activation calibration)
  - weights are fake-quantized (W8/W4/W2)
  - fusion wrapping disabled (avoid coverage/latency contamination)

This file is used at inference time (eval). 
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

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


def _parse_weight_bits(bits: Optional[str]) -> Optional[int]:
    """
    Accepted:
      - "W8" / "W4" / "W2"
      - legacy: "W8A8" / "W4A4" / "W2A2" (A part ignored)

    Returns:
      2/4/8 or None (None => float path)
    """
    if bits is None:
        return None

    b = str(bits).strip().upper()
    if not b:
        return None

    if b.startswith("W") and "A" not in b:
        try:
            w = int(b[1:])
        except ValueError:
            raise ValueError(f"Invalid bits spec {bits!r}; expected W8/W4/W2.")
    else:
        # legacy W8A8 forms
        if "A" not in b:
            raise ValueError(f"Invalid bits spec {bits!r}; expected W8/W4/W2 (legacy W8A8 accepted).")
        w_part = b.split("A", 1)[0]
        if not w_part.startswith("W"):
            raise ValueError(f"Invalid bits spec {bits!r}; expected W8/W4/W2 (legacy W8A8 accepted).")
        try:
            w = int(w_part[1:])
        except ValueError:
            raise ValueError(f"Invalid bits spec {bits!r}; expected W8/W4/W2.")

    if w not in (2, 4, 8):
        raise ValueError(f"Unsupported weight bits W{w}; Only supports W8/W4/W2.")
    return w


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
    # Common patterns observed across revisions:
    #   - registry.items(): yields (module_path, entry) where entry has `.target`
    #   - registry.entries: list/dict with (target, module_path)
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

    We want the module that owns `backbone.layers` (Mamba) or `layers`.
    Common CobraVLM layouts observed:
      - vlm.llm_backbone.llm.backbone.layers
      - vlm.llm_backbone.llm.layers
      - vlm.llm_backbone.backbone.layers
      - vlm.llm (already a bare llm)

    Returns:
      A best-effort LLM module. If extraction fails, returns `vlm` (safe fallback).
    """
    # 1) Preferred: vlm.llm_backbone.llm
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

    # 2) Common: vlm.llm
    cand = getattr(vlm, "llm", None)
    if isinstance(cand, nn.Module):
        return cand

    # 3) Last resort: maybe caller already passed LLM
    return vlm

def _maybe_enable_llm_mixer_hadamard(llm, *, ptq_stage: int = 0):
    """
    Enable Quamba-structured mixer rotation (input + optional output transform) with KLT+Hadamard.

    Env:
      COBRA_LLM_MIXER_HADAMARD=1            enable mixer rotation pipeline
      COBRA_LLM_MIXER_ACT_KLT=1             enable act-KLT (in/out)
      COBRA_LLM_MIXER_OUT_TRANSFORM=1       enable output_transform (needs OUT KLT blocks)
      ACT_KLT_OUTPROJ_IN=/path/to/in.pt     in-side KLT payload (fallback to ACT_KLT_OUT)
      ACT_KLT_OUT=/path/to/in.pt            legacy in-side KLT payload
      ACT_KLT_OUTPROJ_OUT=/path/to/out.pt   out-side KLT payload (required if OUT_TRANSFORM=1)
      COBRA_LLM_MIXER_BLOCK=512             block size
      COBRA_LLM_MIXER_TARGETS=out_proj      targets (comma sep)
      COBRA_LLM_MIXER_MAX_LAYERS=...        optional cap
      COBRA_LLM_MIXER_DRY_RUN=1             no-op but reports what would be applied
      COBRA_LLM_MIXER_ACT_KLT_STRICT=1      strict missing/mismatch handling
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

    # Add extra fields dynamically for backward compat if dataclass hasn't been updated
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

# -----------------------------------------------------------------------------
# Public entrypoint
# -----------------------------------------------------------------------------

def load_quantized_cobra_vlm(
    *,
    bits: Optional[str],
    pct_hi_lo_path,  # kept for interface compatibility
    hf_token: str,
    base_dtype,
    device,
    enabled_targets=None,
    run_dir=None,
    output_dir=None,
):
    """

    Adds optional LLM mixer rotation:
      - Hadamard-only: COBRA_LLM_MIXER_HADAMARD=1
      - Hadamard + act-KLT: COBRA_LLM_MIXER_HADAMARD=1 and COBRA_LLM_MIXER_ACT_KLT=1

    IMPORTANT:
      Mamba fast path can bypass nn.Linear(out_proj) calls, which prevents forward_pre_hook from firing.
      When mixer rotation is enabled, we automatically set COBRA_DISABLE_MAMBA_FAST_PATH=1 unless user already set it.
    """
    import json
    from collections import defaultdict

    # -----------------------------
    # 0) Entry diagnostics (so we can prove this loader is used)
    # -----------------------------
    try:
        this_file = __file__
    except Exception:
        this_file = "<unknown>"
    print(f"[INFO] load_quantized_cobra_vlm ENTER  file={this_file}")
    print(
        "[INFO] env "
        f"COBRA_LLM_MIXER_HADAMARD={os.environ.get('COBRA_LLM_MIXER_HADAMARD', '')!r} "
        f"COBRA_LLM_MIXER_ACT_KLT={os.environ.get('COBRA_LLM_MIXER_ACT_KLT', '')!r} "
        f"COBRA_LLM_MIXER_OUT_TRANSFORM={os.environ.get('COBRA_LLM_MIXER_OUT_TRANSFORM', '')!r} "
        f"ACT_KLT_OUTPROJ_IN={os.environ.get('ACT_KLT_OUTPROJ_IN', '')!r} "
        f"ACT_KLT_OUTPROJ_OUT={os.environ.get('ACT_KLT_OUTPROJ_OUT', '')!r} "
        f"COBRA_DISABLE_MAMBA_FAST_PATH={os.environ.get('COBRA_DISABLE_MAMBA_FAST_PATH', '')!r}"
    )

    w_bits = _parse_weight_bits(bits)

    # Resolve output_dir
    if output_dir is None:
        if run_dir is not None:
            output_dir = Path(run_dir) / "outputs" / "quantize"
        elif pct_hi_lo_path is not None:
            try:
                output_dir = Path(pct_hi_lo_path).parent
            except Exception:
                output_dir = None
    if output_dir is not None:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Float fallback (no quant)
    if w_bits is None:
        model_id_or_path = _resolve_model_id_or_path()
        print(f"[load_quantized_cobra_vlm] bits=None → loading FLOAT Cobra from {model_id_or_path!r} ...")
        vlm = cobra_load(model_id_or_path, hf_token=hf_token)
        vlm.to(device=device, dtype=base_dtype)
        return vlm

    if pct_hi_lo_path is not None:
        print("[INFO] NOTE: pct_hi_lo_path is ignored (activation remains float).")

    # -----------------------------
    # 0.5) If mixer rotation enabled, force Mamba slow path (before model construction)
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
    print(f"[load_quantized_cobra_vlm] Loading FLOAT Cobra from {model_id_or_path!r} ...")

    vlm = cobra_load(model_id_or_path, hf_token=hf_token)
    vlm.to(device=device, dtype=base_dtype)

    # -----------------------------
    # 2) Wrap model (Quant* modules)
    # -----------------------------
    enabled_targets_set = set(enabled_targets) if enabled_targets else {"vision.dino", "vision.siglip", "llm", "projector"}

    wrap_policy_cfg = WrapPolicyConfig(
        enable_vision_dino="vision.dino" in enabled_targets_set,
        enable_vision_siglip="vision.siglip" in enabled_targets_set,
        enable_llm="llm" in enabled_targets_set,
        enable_projector="projector" in enabled_targets_set,
        enable_fusion=False,   # hard lock
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
    #     CRITICAL: apply to LLM module (not the VLM wrapper) so we can find backbone.layers.
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
    # 3) Apply runtime weight bits
    # -----------------------------
    _apply_runtime_weight_bits(vlm, w_bits=w_bits)

    # -----------------------------
    # 4) Enable fake quant flags
    # -----------------------------
    set_quant_state(vlm, weight_quant=True, act_quant=False)

    # -----------------------------
    # 5) Emit coverage
    # -----------------------------
    if output_dir is not None and registry is not None:
        by_target = defaultdict(list)
        for target, module_path in _iter_wrap_registry_entries(registry):
            by_target[target].append(module_path)

        coverage_payload = {
            "stage": "runtime_weight_only",
            "bits": f"W{w_bits}",
            "backend": "fake",
            "counts": {k: len(v) for k, v in by_target.items()},
            "module_paths": dict(by_target),
        }

        out_path = Path(output_dir) / f"coverage_W{w_bits}.json"
        try:
            out_path.write_text(json.dumps(coverage_payload, indent=2))
        except Exception:
            pass

    return vlm

