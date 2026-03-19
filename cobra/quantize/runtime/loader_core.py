from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

from cobra import load as cobra_load


def _log_runtime_entry(
    *,
    this_file: str,
    bits,
    pct_hi_lo_path,
    enabled_targets,
    model_id_or_path,
) -> None:
    print(f"[INFO] load_quantized_cobra_vlm ENTER  file={this_file}")
    print(
        f"[INFO] args bits={bits!r} pct_hi_lo_path={pct_hi_lo_path!r} "
        f"enabled_targets={enabled_targets!r} model_id_or_path={model_id_or_path!r}"
    )


def _load_base_vlm(*, base_model_id: str, hf_token: str, base_dtype, device):
    print(f"[load_quantized_cobra_vlm] loading FLOAT Cobra from {base_model_id!r} ...")
    vlm = cobra_load(base_model_id, hf_token=hf_token)
    vlm.to(device=device, dtype=base_dtype)
    return vlm


def _load_float_passthrough(
    *,
    base_model_id: str,
    hf_token: str,
    base_dtype,
    device,
):
    print(
        "[load_quantized_cobra_vlm] runtime request resolves to FLOAT path; "
        f"loading {base_model_id!r}"
    )
    return _load_base_vlm(
        base_model_id=base_model_id,
        hf_token=hf_token,
        base_dtype=base_dtype,
        device=device,
    )


def _prepare_output_dir(output_dir: Path) -> Optional[Path]:
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    except Exception as e:
        print(f"[WARN] output_dir mkdir failed: {str(output_dir)!r} ({repr(e)})")
        return None


def _apply_wrapping(vlm, *, wrap_policy_cfg):
    from cobra.quantize.wrap.entry import wrap_model_for_quantization

    return wrap_model_for_quantization(
        vlm,
        policy_cfg=wrap_policy_cfg,
        manifest=None,
        default_params=None,
        prefix="",
    )


def _apply_rotation(vlm, *, rotation_spec):
    from cobra.quantize.runtime.mixer_runtime import apply_mixer_rotation

    return apply_mixer_rotation(vlm, rotation_spec)


def _apply_weight_quant(vlm, *, do_weight: bool, requested_weight_bits: Optional[int]) -> None:
    if not do_weight:
        print("[Info] W-bits not requested -> weight quant will stay OFF (weights remain float).")
        return

    from cobra.quantize.runtime.helpers import apply_runtime_weight_bits

    apply_runtime_weight_bits(vlm, w_bits=int(requested_weight_bits))


def _apply_act_quant(
    vlm,
    *,
    do_act_requested: bool,
    pct_hi_lo_path,
    act_bits: Optional[int],
    enabled_targets,
    output_dir: Optional[Path],
    llm_act_mode: str,
):
    from cobra.quantize.runtime.activation_runtime import apply_activation_calibration

    return apply_activation_calibration(
        vlm,
        requested=bool(do_act_requested),
        pct_hi_lo_path=pct_hi_lo_path,
        act_bits=act_bits,
        enabled_targets=enabled_targets,
        output_dir=output_dir,
        llm_act_mode=llm_act_mode,
    )


def _extract_calibrated_llm_module_paths(act_summary) -> set[str]:
    """
    Derive the set of LLM module paths that were ACTUALLY calibrated from
    the activation calibration summary returned by calibrate_model_from_hi_lo().

    The summary is expected to be keyed by module qualified names.
    """
    if not isinstance(act_summary, dict):
        return set()

    out = set()
    for module_path in act_summary.keys():
        if module_path is None:
            continue
        module_path = str(module_path)
        if module_path.startswith("llm_backbone.llm."):
            out.add(module_path)
    return out


def _finalize_quant_state(
    vlm,
    *,
    do_weight: bool,
    act_enabled: bool,
    llm_act_mode: str,
    calibrated_llm_module_paths,
) -> None:
    from cobra.quantize.runtime.helpers import restrict_llm_act_quant_by_mode
    from cobra.quantize.utils import set_quant_state

    set_quant_state(vlm, weight_quant=bool(do_weight), act_quant=bool(act_enabled))

    if act_enabled:
        restrict_llm_act_quant_by_mode(
            vlm,
            mode=llm_act_mode,
            calibrated_module_paths=calibrated_llm_module_paths,
        )


def _emit_runtime_reports(
    *,
    output_dir: Optional[Path],
    this_file: str,
    resolved,
    registry,
    mixer_rotation_report,
    do_weight: bool,
    do_act_requested: bool,
    activation_result,
) -> None:
    if output_dir is None or registry is None:
        return

    from cobra.quantize.runtime.reporting import (
        build_behavior_payload,
        build_coverage_payload,
        write_runtime_artifacts,
    )

    req = resolved.runtime_request
    quant_cfg = resolved.quant_cfg

    coverage_payload = build_coverage_payload(
        req=req,
        quant_cfg=quant_cfg,
        do_weight=do_weight,
        do_act_requested=do_act_requested,
        act_calib_enabled=activation_result.enabled,
        resolved_inputs=resolved,
        registry=registry,
    )
    behavior_payload = build_behavior_payload(
        this_file=this_file,
        req=req,
        quant_cfg=quant_cfg,
        do_weight=do_weight,
        do_act_requested=do_act_requested,
        act_calib_enabled=activation_result.enabled,
        resolved_inputs=resolved,
        registry=registry,
        mixer_rotation_report=mixer_rotation_report,
    )
    write_runtime_artifacts(
        output_dir=output_dir,
        coverage_payload=coverage_payload,
        behavior_payload=behavior_payload,
    )


def _resolve_runtime(bits, pct_hi_lo_path, enabled_targets, run_dir, output_dir, model_id_or_path, backend):
    from cobra.quantize.runtime.bootstrap import resolve_runtime_inputs

    resolved = resolve_runtime_inputs(
        bits=bits,
        pct_hi_lo_path=pct_hi_lo_path,
        enabled_targets=enabled_targets,
        run_dir=run_dir,
        output_dir=output_dir,
        model_id_or_path=model_id_or_path,
        backend=backend,
    )

    print(
        "[INFO] resolved runtime inputs: "
        f"model_id={resolved.raw_model_id!r} "
        f"llm_act_only={resolved.llm_act_only!r} "
        f"llm_act_mode={resolved.llm_act_mode!r} "
        f"pct_hi_lo_path={str(resolved.pct_hi_lo_path) if resolved.pct_hi_lo_path is not None else None!r}"
    )
    return resolved


def load_quantized_cobra_vlm_impl(
    *,
    bits: Optional[str],
    pct_hi_lo_path,
    hf_token: str,
    base_dtype,
    device,
    enabled_targets=None,
    run_dir=None,
    output_dir=None,
    model_id_or_path: Optional[str] = None,
    backend: str = "fake",
):
    """
    Phase 5 runtime loader core.

    Responsibilities:
      1. resolve runtime request / artifacts
      2. load float base model
      3. wrap / rotate / calibrate
      4. emit runtime artifacts

    The function body is intentionally orchestration-only; heavy logic is pushed
    into small helpers so later phases can evolve pieces independently.
    """
    try:
        this_file = __file__
    except Exception:
        this_file = "<unknown>"

    _log_runtime_entry(
        this_file=this_file,
        bits=bits,
        pct_hi_lo_path=pct_hi_lo_path,
        enabled_targets=enabled_targets,
        model_id_or_path=model_id_or_path,
    )

    resolved = _resolve_runtime(
        bits=bits,
        pct_hi_lo_path=pct_hi_lo_path,
        enabled_targets=enabled_targets,
        run_dir=run_dir,
        output_dir=output_dir,
        model_id_or_path=model_id_or_path,
        backend=backend,
    )

    req = resolved.runtime_request
    quant_cfg = resolved.quant_cfg

    do_weight = quant_cfg.should_apply_weight_quant()
    do_act_requested = quant_cfg.requested_act_bits is not None

    print(
        "[INFO] runtime quant request: "
        f"bits_raw={req.bits!r} backend={req.backend!r} "
        f"do_weight={bool(do_weight)} do_act_requested={bool(do_act_requested)} "
        f"llm_act_mode={resolved.llm_act_mode!r}"
    )

    if not do_weight and not do_act_requested:
        return _load_float_passthrough(
            base_model_id=req.base_model_id,
            hf_token=hf_token,
            base_dtype=base_dtype,
            device=device,
        )

    resolved_output_dir = _prepare_output_dir(resolved.output_dir)

    if do_act_requested and resolved.pct_hi_lo_path is None:
        print(
            f"[WARN] A-bits requested (A{quant_cfg.requested_act_bits}) but pct_hi_lo_path "
            "cannot be resolved -> act_quant will stay OFF."
        )

    vlm = _load_base_vlm(
        base_model_id=req.base_model_id,
        hf_token=hf_token,
        base_dtype=base_dtype,
        device=device,
    )

    registry = _apply_wrapping(vlm, wrap_policy_cfg=resolved.wrap_policy_cfg)
    mixer_rotation_report = _apply_rotation(vlm, rotation_spec=resolved.rotation_spec)

    _apply_weight_quant(
        vlm,
        do_weight=do_weight,
        requested_weight_bits=quant_cfg.requested_weight_bits,
    )

    activation_result = _apply_act_quant(
        vlm,
        do_act_requested=do_act_requested,
        pct_hi_lo_path=resolved.pct_hi_lo_path,
        act_bits=quant_cfg.requested_act_bits,
        enabled_targets=resolved.enabled_targets,
        output_dir=resolved_output_dir,
        llm_act_mode=resolved.llm_act_mode,
    )

    calibrated_llm_module_paths = _extract_calibrated_llm_module_paths(
        activation_result.summary
    )

    print(
        "[INFO] activation calibration result: "
        f"requested={bool(activation_result.requested)} "
        f"enabled={bool(activation_result.enabled)} "
        f"act_bits={activation_result.act_bits!r} "
        f"llm_act_mode={resolved.llm_act_mode!r} "
        f"calibrated_llm_modules={len(calibrated_llm_module_paths)}"
    )

    if activation_result.enabled and len(calibrated_llm_module_paths) == 0:
        print(
            "[WARN] Activation calibration is globally enabled, but no LLM modules "
            f"were calibrated for llm_act_mode={resolved.llm_act_mode!r}; "
            "all LLM activation quant will be disabled conservatively."
        )

    _finalize_quant_state(
        vlm,
        do_weight=do_weight,
        act_enabled=activation_result.enabled,
        llm_act_mode=resolved.llm_act_mode,
        calibrated_llm_module_paths=calibrated_llm_module_paths,
    )

    _emit_runtime_reports(
        output_dir=resolved_output_dir,
        this_file=this_file,
        resolved=resolved,
        registry=registry,
        mixer_rotation_report=mixer_rotation_report,
        do_weight=do_weight,
        do_act_requested=do_act_requested,
        activation_result=activation_result,
    )

    return vlm
