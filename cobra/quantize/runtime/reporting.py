from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from cobra.quantize.runtime.act_policy import (
    LLM_ACT_MODE_DEFAULT,
    LLM_ACT_MODE_MAMBA_SENSITIVE,
    LLM_ACT_MODE_OUT_PROJ_ONLY,
    normalize_llm_act_mode,
)
from cobra.quantize.runtime.types import RuntimeBehaviorArtifacts


def _safe_str_path(path_obj) -> Optional[str]:
    if path_obj is None:
        return None
    try:
        return str(path_obj)
    except Exception:
        return None


def _registry_summary(registry) -> Dict[str, Any]:
    if registry is None:
        return {
            "num_entries": 0,
            "targets": {},
            "module_paths_sample": [],
        }

    target_counts: Dict[str, int] = {}
    module_paths = []

    for entry in getattr(registry, "entries", []):
        target = getattr(entry, "target", None)
        module_path = getattr(entry, "module_path", None)

        if target is not None:
            target_counts[target] = target_counts.get(target, 0) + 1
        if module_path:
            module_paths.append(str(module_path))

    module_paths = sorted(set(module_paths))

    return {
        "num_entries": int(len(getattr(registry, "entries", []))),
        "targets": target_counts,
        "module_paths_sample": module_paths[:64],
    }


def _rotation_summary(mixer_rotation_report) -> Dict[str, Any]:
    if mixer_rotation_report is None:
        return {
            "enabled": False,
            "summary": None,
        }

    if isinstance(mixer_rotation_report, dict):
        return {
            "enabled": True,
            "summary": mixer_rotation_report,
        }

    try:
        payload = dict(mixer_rotation_report)
    except Exception:
        payload = {"repr": repr(mixer_rotation_report)}

    return {
        "enabled": True,
        "summary": payload,
    }


def _llm_policy_summary(resolved_inputs) -> Dict[str, Any]:
    llm_act_only = getattr(resolved_inputs, "llm_act_only", "")
    llm_act_mode = normalize_llm_act_mode(
        getattr(resolved_inputs, "llm_act_mode", ""),
        fallback_llm_act_only=llm_act_only,
    )

    coarse_llm_tap_expected = llm_act_mode == LLM_ACT_MODE_DEFAULT

    return {
        "llm_act_only": llm_act_only if llm_act_only else None,
        "llm_act_mode": llm_act_mode,
        "coarse_llm_tap_expected": bool(coarse_llm_tap_expected),
        "llm_collector_mode": (
            "coarse_plus_module_paths"
            if llm_act_mode == LLM_ACT_MODE_DEFAULT
            else "module_path_filtered"
        ),
        "llm_runtime_gate_mode": llm_act_mode,
        "mode_notes": {
            LLM_ACT_MODE_DEFAULT: "Enable activation quant for all wrapped llm modules; coarse llm tap is expected during calibration.",
            LLM_ACT_MODE_OUT_PROJ_ONLY: "Restrict llm activation quant to mixer.out_proj paths only.",
            LLM_ACT_MODE_MAMBA_SENSITIVE: "Restrict llm activation quant to policy-selected sensitive mixer paths.",
        }.get(llm_act_mode, "Unknown mode."),
    }


def build_coverage_payload(
    *,
    req,
    quant_cfg,
    do_weight: bool,
    do_act_requested: bool,
    act_calib_enabled: bool,
    resolved_inputs,
    registry,
) -> Dict[str, Any]:
    """
    Coverage-oriented payload:
      - what was requested
      - what artifacts were found
      - what wrapping coverage exists

    This payload should stay stable and compact because it is useful for
    behavior-lock testing and quick diffing across runs.
    """
    requested_bits = getattr(req, "bits", None)
    requested_backend = getattr(req, "backend", None)
    base_model_id = getattr(req, "base_model_id", None)

    llm_act_only = getattr(resolved_inputs, "llm_act_only", "")
    llm_act_mode = normalize_llm_act_mode(
        getattr(resolved_inputs, "llm_act_mode", ""),
        fallback_llm_act_only=llm_act_only,
    )

    resolved_pct_hi_lo_path = getattr(resolved_inputs, "pct_hi_lo_path", None)

    coverage_payload: Dict[str, Any] = {
        "request": {
            "bits": requested_bits,
            "backend": requested_backend,
            "base_model_id": base_model_id,
        },
        "quant_cfg": {
            "mode": getattr(getattr(quant_cfg, "mode", None), "value", None),
            "requested_weight_bits": getattr(quant_cfg, "requested_weight_bits", None),
            "requested_act_bits": getattr(quant_cfg, "requested_act_bits", None),
            "weight_bits": getattr(quant_cfg, "weight_bits", None),
            "act_bits": getattr(quant_cfg, "act_bits", None),
        },
        "execution": {
            "do_weight": bool(do_weight),
            "do_act_requested": bool(do_act_requested),
            "act_calib_enabled": bool(act_calib_enabled),
        },
        "artifacts": {
            "pct_hi_lo_path": _safe_str_path(resolved_pct_hi_lo_path),
            "pct_hi_lo_resolved": resolved_pct_hi_lo_path is not None,
        },
        "llm_policy": {
            "llm_act_only": llm_act_only if llm_act_only else None,
            "llm_act_mode": llm_act_mode,
        },
        "wrap_registry": _registry_summary(registry),
    }
    return coverage_payload


def build_behavior_payload(
    *,
    this_file: str,
    req,
    quant_cfg,
    do_weight: bool,
    do_act_requested: bool,
    act_calib_enabled: bool,
    resolved_inputs,
    registry,
    mixer_rotation_report,
) -> Dict[str, Any]:
    """
    Behavior-oriented payload:
      - resolved runtime inputs
      - quantization toggles
      - wrap / rotation / activation policy context

    This payload is intentionally richer than coverage_payload.
    """
    policy_summary = _llm_policy_summary(resolved_inputs)

    behavior_payload: Dict[str, Any] = {
        "source": {
            "file": this_file,
        },
        "request": {
            "bits": getattr(req, "bits", None),
            "backend": getattr(req, "backend", None),
            "base_model_id": getattr(req, "base_model_id", None),
        },
        "resolved_inputs": {
            "raw_model_id": getattr(resolved_inputs, "raw_model_id", None),
            "run_dir": _safe_str_path(getattr(resolved_inputs, "run_dir", None)),
            "output_dir": _safe_str_path(getattr(resolved_inputs, "output_dir", None)),
            "pct_hi_lo_path": _safe_str_path(getattr(resolved_inputs, "pct_hi_lo_path", None)),
            "enabled_targets": sorted(list(getattr(resolved_inputs, "enabled_targets", []))),
            "llm_act_only": getattr(resolved_inputs, "llm_act_only", "") or None,
            "llm_act_mode": getattr(resolved_inputs, "llm_act_mode", None),
        },
        "quant_cfg": {
            "mode": getattr(getattr(quant_cfg, "mode", None), "value", None),
            "requested_weight_bits": getattr(quant_cfg, "requested_weight_bits", None),
            "requested_act_bits": getattr(quant_cfg, "requested_act_bits", None),
            "weight_bits": getattr(quant_cfg, "weight_bits", None),
            "act_bits": getattr(quant_cfg, "act_bits", None),
            "enable_act_quant": getattr(quant_cfg, "enable_act_quant", None),
            "enable_llm": getattr(quant_cfg, "enable_llm", None),
            "enable_projector": getattr(quant_cfg, "enable_projector", None),
            "enable_vision_dino": getattr(quant_cfg, "enable_vision_dino", None),
            "enable_vision_siglip": getattr(quant_cfg, "enable_vision_siglip", None),
        },
        "execution": {
            "do_weight": bool(do_weight),
            "do_act_requested": bool(do_act_requested),
            "act_calib_enabled": bool(act_calib_enabled),
        },
        "llm_policy": policy_summary,
        "wrap_registry": _registry_summary(registry),
        "mixer_rotation": _rotation_summary(mixer_rotation_report),
    }
    return behavior_payload


def write_runtime_artifacts(
    *,
    output_dir: Path,
    coverage_payload: Dict[str, Any],
    behavior_payload: Dict[str, Any],
) -> RuntimeBehaviorArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)

    llm_policy = behavior_payload.get("llm_policy", {}) if isinstance(behavior_payload, dict) else {}
    llm_act_mode = normalize_llm_act_mode(
        llm_policy.get("llm_act_mode", ""),
        fallback_llm_act_only=llm_policy.get("llm_act_only", ""),
    )
    mode_tag = str(llm_act_mode).strip().lower() or "default"

    coverage_path = output_dir / f"runtime_coverage_{mode_tag}.json"
    behavior_path = output_dir / f"runtime_behavior_{mode_tag}.json"

    coverage_path.write_text(
        json.dumps(coverage_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    behavior_path.write_text(
        json.dumps(behavior_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Keep stable "latest snapshot" aliases for backward compatibility.
    latest_coverage_path = output_dir / "runtime_coverage.json"
    latest_behavior_path = output_dir / "runtime_behavior.json"

    latest_coverage_path.write_text(
        json.dumps(coverage_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    latest_behavior_path.write_text(
        json.dumps(behavior_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"[INFO] wrote runtime coverage artifact: {coverage_path}")
    print(f"[INFO] wrote runtime behavior artifact: {behavior_path}")
    print(f"[INFO] updated runtime coverage alias: {latest_coverage_path}")
    print(f"[INFO] updated runtime behavior alias: {latest_behavior_path}")

    return RuntimeBehaviorArtifacts(
        coverage_payload=coverage_payload,
        behavior_payload=behavior_payload,
        coverage_path=coverage_path,
        behavior_path=behavior_path,
    )
