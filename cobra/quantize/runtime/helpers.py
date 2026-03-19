from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import nn


def apply_runtime_weight_bits(vlm: nn.Module, w_bits: int) -> None:
    """
    Propagate the desired weight bit-width to all Quant* modules (weights only).
    """
    from cobra.quantize.int_conv import QuantConv1d, QuantConv2d
    from cobra.quantize.int_linear import QuantLinear
    from cobra.quantize.int_matmul import QuantMatMul

    num_modules = 0
    for _, module in vlm.named_modules():
        if isinstance(module, (QuantLinear, QuantConv1d, QuantConv2d, QuantMatMul)):
            module.change_bits(weight_bits=w_bits, act_bits=None)
            num_modules += 1

    print(f"[Info] Runtime weight_bits applied to {num_modules} Quant* modules (W{w_bits}).")


def iter_wrap_registry_entries(registry):
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


def extract_llm_for_mixer_rotation(vlm: nn.Module) -> nn.Module:
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


def restrict_llm_act_quant_by_mode(
    vlm: nn.Module,
    mode: str,
    calibrated_module_paths=None,
):
    """
    Keep or disable activation quantization for LLM-side Quant* modules
    according to BOTH:
      1) the resolved activation policy mode
      2) the set of modules that were actually calibrated from pct_hi_lo

    Final rule:
        final_keep = policy_keep AND calibrated_keep

    Notes
    -----
    - Non-LLM modules are left untouched.
    - If `calibrated_module_paths` is None or empty, all LLM activation
      quantization will be disabled conservatively.
    """
    from cobra.quantize.int_conv import QuantConv1d, QuantConv2d
    from cobra.quantize.int_linear import QuantLinear
    from cobra.quantize.int_matmul import QuantMatMul
    from cobra.quantize.runtime.act_policy import (
        normalize_llm_act_mode,
        should_enable_llm_module_act_quant,
    )

    normalized_mode = normalize_llm_act_mode(mode)

    calibrated_set = {
        str(p)
        for p in (calibrated_module_paths or [])
        if p is not None and str(p).strip()
    }

    num_llm_quant_modules = 0
    num_policy_keep = 0
    num_calibrated_keep = 0
    num_final_keep = 0
    num_disable = 0
    num_disabled_due_to_missing_calibration = 0

    kept_paths = []
    disabled_paths = []
    missing_calibration_paths = []

    for module_path, module in vlm.named_modules():
        if not module_path.startswith("llm_backbone.llm."):
            continue

        if not isinstance(module, (QuantLinear, QuantConv1d, QuantConv2d, QuantMatMul)):
            continue

        num_llm_quant_modules += 1

        policy_keep = should_enable_llm_module_act_quant(module_path, mode=normalized_mode)
        calibrated_keep = module_path in calibrated_set
        final_keep = bool(policy_keep and calibrated_keep)

        if policy_keep:
            num_policy_keep += 1
        if calibrated_keep:
            num_calibrated_keep += 1

        if hasattr(module, "use_act_quant"):
            module.use_act_quant = final_keep

        for attr in ("act_quantizer", "x1_quantizer", "x2_quantizer"):
            q = getattr(module, attr, None)
            if q is None:
                continue

            # Attach a stable debug name for one-shot quantizer warnings.
            try:
                q.debug_name = f"{module_path}.{attr}"
            except Exception:
                pass

            q.set_quant_state(enable=final_keep, is_dynamic=False)

            if not final_keep:
                q.is_observing = False
                q.observered = False

        if final_keep:
            num_final_keep += 1
            kept_paths.append(module_path)
        else:
            num_disable += 1
            disabled_paths.append(module_path)
            if policy_keep and not calibrated_keep:
                num_disabled_due_to_missing_calibration += 1
                missing_calibration_paths.append(module_path)

    print(
        "[Info] Applied llm activation runtime policy: "
        f"mode={normalized_mode!r} llm_quant_modules={num_llm_quant_modules} "
        f"policy_keep={num_policy_keep} calibrated_keep={num_calibrated_keep} "
        f"final_keep={num_final_keep} disable={num_disable} "
        f"disabled_due_to_missing_calibration={num_disabled_due_to_missing_calibration}"
    )

    if kept_paths:
        print(
            "[Info] LLM act-quant keep sample: "
            + ", ".join(repr(p) for p in kept_paths[:8])
            + (" ..." if len(kept_paths) > 8 else "")
        )

    if missing_calibration_paths:
        print(
            "[WARN] LLM act-quant disabled due to missing calibration sample: "
            + ", ".join(repr(p) for p in missing_calibration_paths[:8])
            + (" ..." if len(missing_calibration_paths) > 8 else "")
        )

    if disabled_paths:
        print(
            "[Info] LLM act-quant disable sample: "
            + ", ".join(repr(p) for p in disabled_paths[:8])
            + (" ..." if len(disabled_paths) > 8 else "")
        )


def restrict_llm_act_quant_to_out_proj(vlm: nn.Module, calibrated_module_paths=None) -> None:
    """
    Backward-compatible wrapper that delegates to the generalized mode-aware
    runtime policy with the same calibration gate.
    """
    restrict_llm_act_quant_by_mode(
        vlm,
        mode="out_proj_only",
        calibrated_module_paths=calibrated_module_paths,
    )

    
def safe_jsonable(obj: Any):
    """
    Convert arbitrary objects into JSON-safe values for behavior-lock dumps.
    """
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): safe_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [safe_jsonable(v) for v in obj]
    return repr(obj)


def write_json_best_effort(path: Path, payload: dict) -> None:
    import json

    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(safe_jsonable(payload), indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"[WARN] Failed to write JSON to {str(path)!r}: {repr(e)}")


def summarize_wrap_registry(registry) -> dict:
    from collections import defaultdict

    by_target = defaultdict(list)
    for target, module_path in iter_wrap_registry_entries(registry):
        by_target[str(target)].append(str(module_path))

    return {
        "counts_by_target": {k: len(v) for k, v in by_target.items()},
        "module_paths_by_target": dict(by_target),
        "wrapped_module_count": sum(len(v) for v in by_target.values()),
    }
