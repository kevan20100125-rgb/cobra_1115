from __future__ import annotations

from typing import Any, Dict


def _extract_llm_for_mixer_rotation(vlm) -> Any:
    """
    Local, dependency-light extractor.
    Avoid importing helpers.py at module import time.
    """
    lb = getattr(vlm, "llm_backbone", None)
    if lb is not None:
        cand = getattr(lb, "llm", None)
        if cand is not None:
            return cand
        cand = getattr(lb, "model", None)
        if cand is not None:
            return cand
        cand = getattr(lb, "backbone", None)
        if cand is not None:
            return cand

    cand = getattr(vlm, "llm", None)
    if cand is not None:
        return cand

    return vlm


def _disabled_report(rotation_spec) -> Dict[str, object]:
    return {
        "enabled": False,
        "mode": "mixer_kh",
        "block_size": getattr(rotation_spec, "block_size", None),
        "targets": list(getattr(rotation_spec, "targets", ()) or []),
        "max_layers": getattr(rotation_spec, "max_layers", None),
        "dry_run": bool(getattr(rotation_spec, "dry_run", False)),
        "in_transform_enabled": bool(getattr(rotation_spec, "in_transform_enabled", False)),
        "out_transform_enabled": bool(getattr(rotation_spec, "out_transform_enabled", False)),
        "act_klt_enabled": bool(getattr(rotation_spec, "act_klt_enabled", False)),
        "in_act_klt_enabled": bool(getattr(rotation_spec, "in_act_klt_enabled", False)),
        "out_act_klt_enabled": bool(getattr(rotation_spec, "out_act_klt_enabled", False)),
        "act_klt_strict": bool(getattr(rotation_spec, "act_klt_strict", False)),
        "act_klt_in_path": (
            str(getattr(rotation_spec, "act_klt_in_path", None))
            if getattr(rotation_spec, "act_klt_in_path", None) is not None
            else None
        ),
        "act_klt_out_path": (
            str(getattr(rotation_spec, "act_klt_out_path", None))
            if getattr(rotation_spec, "act_klt_out_path", None) is not None
            else None
        ),
        "applied_count": 0,
        "skipped_count": 0,
        "applied": [],
        "skipped": [],
        "error": None,
        "resolved_llm_type": None,
        "resolved_has_layers": False,
    }


def apply_mixer_rotation(vlm, rotation_spec) -> Dict[str, object]:
    """
    Runtime shim for mixer Hadamard / act-KLT integration.

    Design choice for Phase 4:
    - keep this file import-light
    - avoid importing repo-internal symbols at module load time
    - push all fragile imports inside this function
    """
    import os
    import traceback

    enabled = bool(getattr(rotation_spec, "enabled", False))
    if not enabled:
        print("[MixerRotation] gate OFF (resolver says mixer rotation is disabled).")
        return _disabled_report(rotation_spec)

    if os.environ.get("COBRA_DISABLE_MAMBA_FAST_PATH", "").strip() == "":
        os.environ["COBRA_DISABLE_MAMBA_FAST_PATH"] = "1"
        print("[MixerRotation] Auto-set COBRA_DISABLE_MAMBA_FAST_PATH=1 (required for out_proj hooks).")

    report = _disabled_report(rotation_spec)
    report["enabled"] = True

    try:
        llm = _extract_llm_for_mixer_rotation(vlm)
        llm_type = type(llm).__name__
        has_layers = bool(
            (hasattr(llm, "backbone") and hasattr(llm.backbone, "layers"))
            or hasattr(llm, "layers")
        )

        report["resolved_llm_type"] = llm_type
        report["resolved_has_layers"] = has_layers

        print(f"[MixerRotation] resolved_llm={llm_type} has_layers={int(has_layers)}")

        from cobra.quantize.rotate.mixer import (
            MixerHadamardRotationConfig,
            rotate_llm_mamba_mixers_hadamard_inplace,
        )

        cfg = MixerHadamardRotationConfig(
            enabled=True,
            block_size=int(getattr(rotation_spec, "block_size", 512)),
            targets=tuple(getattr(rotation_spec, "targets", ("out_proj",))),
            max_layers=getattr(rotation_spec, "max_layers", None),
            dry_run=bool(getattr(rotation_spec, "dry_run", False)),
            in_transform_enabled=bool(getattr(rotation_spec, "in_transform_enabled", False)),
            out_transform_enabled=bool(getattr(rotation_spec, "out_transform_enabled", False)),
            in_act_klt_enabled=bool(getattr(rotation_spec, "in_act_klt_enabled", False)),
            out_act_klt_enabled=bool(getattr(rotation_spec, "out_act_klt_enabled", False)),
            act_klt_in_path=(
                str(getattr(rotation_spec, "act_klt_in_path", None))
                if getattr(rotation_spec, "act_klt_in_path", None) is not None
                else None
            ),
            act_klt_out_path=(
                str(getattr(rotation_spec, "act_klt_out_path", None))
                if getattr(rotation_spec, "act_klt_out_path", None) is not None
                else None
            ),
            act_klt_strict=bool(getattr(rotation_spec, "act_klt_strict", False)),
        )

        rotate_report = rotate_llm_mamba_mixers_hadamard_inplace(llm, cfg=cfg)
        applied = rotate_report.get("applied", []) or []
        skipped = rotate_report.get("skipped", []) or []

        report.update(
            {
                "mode": "mixer_kh",
                "block_size": int(getattr(rotation_spec, "block_size", 512)),
                "targets": list(getattr(rotation_spec, "targets", ("out_proj",))),
                "max_layers": getattr(rotation_spec, "max_layers", None),
                "dry_run": bool(getattr(rotation_spec, "dry_run", False)),
                "in_transform_enabled": bool(getattr(rotation_spec, "in_transform_enabled", False)),
                "out_transform_enabled": bool(getattr(rotation_spec, "out_transform_enabled", False)),
                "act_klt_enabled": bool(getattr(rotation_spec, "act_klt_enabled", False)),
                "in_act_klt_enabled": bool(getattr(rotation_spec, "in_act_klt_enabled", False)),
                "out_act_klt_enabled": bool(getattr(rotation_spec, "out_act_klt_enabled", False)),
                "act_klt_strict": bool(getattr(rotation_spec, "act_klt_strict", False)),
                "act_klt_in_path": (
                    str(getattr(rotation_spec, "act_klt_in_path", None))
                    if getattr(rotation_spec, "act_klt_in_path", None) is not None
                    else None
                ),
                "act_klt_out_path": (
                    str(getattr(rotation_spec, "act_klt_out_path", None))
                    if getattr(rotation_spec, "act_klt_out_path", None) is not None
                    else None
                ),
                "applied_count": len(applied),
                "skipped_count": len(skipped),
                "applied": applied,
                "skipped": skipped,
                "error": None,
            }
        )

        print(
            f"[MixerRotation] enabled=1 mode=mixer_kh "
            f"block={report['block_size']} targets={report['targets']} "
            f"max_layers={report['max_layers']} dry_run={int(report['dry_run'])} "
            f"in_tx={int(report['in_transform_enabled'])} "
            f"out_tx={int(report['out_transform_enabled'])} "
            f"in_klt={int(report['in_act_klt_enabled'])} "
            f"out_klt={int(report['out_act_klt_enabled'])} "
            f"strict={int(report['act_klt_strict'])} "
            f"applied={report['applied_count']} skipped={report['skipped_count']}"
        )
        return report

    except Exception as e:
        print("[MixerRotation] ERROR: mixer rotation crashed.")
        print(f"[MixerRotation] Exception: {repr(e)}")
        traceback.print_exc()

        report["error"] = repr(e)
        report["applied_count"] = 0
        report["skipped_count"] = 0
        report["applied"] = []
        report["skipped"] = []
        return report
