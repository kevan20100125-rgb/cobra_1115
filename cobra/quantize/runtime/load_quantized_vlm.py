# cobra/quantize/runtime/load_quantized_vlm.py
"""
load_quantized_vlm.py

Unified entrypoint for reconstructing a *fake-quantized* Cobra VLM from:
    • float checkpoint (loaded by cobra.load)
    • pct_hi_lo  (activation clipping ranges, produced by quant_calibrate.py
                  or quant_pct_apply.py)

This file is used ONLY at inference time (eval), not for calibration.

Design aligned with the W4/W8 fake-quant PTQ pipeline:
    - Use `quantize.wrap.wrap_model_for_quantization` to wrap the
      float Cobra model with Quant* modules (MambaQuant-style fake quant).
      All kernels remain PyTorch float kernels with quantize/dequant around them.
    - Load (hi, lo) clipping bounds from `pct_hi_lo_path` (torch.save dict).
    - Use `pct.calibrator.calibrate_model_from_hi_lo` to convert (hi, lo) to
      affine activation quantization parameters (scale, zero_point, n_bits)
      and write them into activation quantizers.
    - Activation bitwidth (A) is derived from QuantRuntimeConfig.act_bits
      (parsed from `bits`, e.g. "W8A4" -> act_bits=4).

Notes (Phase 1, runtime unification):
    - Quantization設定（bits/backend/哪些 target 進 percentile pipeline）交給
      `QuantRuntimeConfig.from_bits_backend` 處理：
        * bits    -> (weight_bits, act_bits)
        * backend -> mode ∈ {FLOAT, FAKE}
        * enable_* / vision_in_pct_pipeline -> use_pct_for（哪些 target 會被校正）
    - 本檔只處理「推論時的 fake quant 重建」：
        * bits/backend/targets 的解析：QuantRuntimeConfig
        * wrap：wrap_model_for_quantization + WrapPolicyConfig
        * pct_hi_lo 應用：calibrate_model_from_hi_lo(include_targets=use_pct_for)

Notes (Phase 2, config single-source):
    - calibrate_model_from_hi_lo 的 include_targets 與 signed 參數均來自
      QuantRuntimeConfig（use_pct_for / symmetric_acts），確保與
      quant_calibrate / quant_finalize 完全一致。

Notes (Phase 2+, projector rotation):
    - 是否允許 projector rotation 由 QuantRuntimeConfig.use_rotation_for /
      QuantRuntimeConfig.should_rotate_projector() 決定，確保：
        * runtime (fake / 未來 INT) 使用同一 gating 策略。
    - projector rotation 模式（"hk" / "hadamard" / "none"）由
      QuantRuntimeConfig.projector_rotation_mode 管理；來源可以是：
        * CLI（quant_finalize）或
        * 環境變數 COBRA_PROJECTOR_ROTATION_MODE（runtime）
    - 實際旋轉邏輯（KLT + Hadamard）集中在
        `cobra.quantize.rotate.projector`：
        * SHARED_KLT_PATH
        * ProjectorRotationConfig
        * rotate_cobra_vlm_output_projector_from_path_inplace(...)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Dict

import torch
from torch import nn

from cobra import load as cobra_load
from cobra.quantize.utils import set_quant_state
from cobra.quantize.wrap.entry import wrap_model_for_quantization
from cobra.quantize.wrap.policy import WrapPolicyConfig
from cobra.quantize.pct.calibrator import calibrate_model_from_hi_lo
from cobra.quantize.quantizer import UniformAffineQuantizer
from cobra.quantize.runtime.config import QuantRuntimeConfig, QuantMode
from cobra.quantize.rotate.projector import (
    ProjectorRotationConfig,
    SHARED_KLT_PATH,
    rotate_cobra_vlm_output_projector_from_path_inplace,
)

# 環境變數：控制 projector rotation 的模式
#   - "hk" / "klt+hadamard" / "klt_hadamard" : KLT + Hadamard（預設）
#   - "hadamard" / "h"                      : 只有 Hadamard
#   - "none" / "off" / "disable"/"disabled": 完全不旋轉
_ROTATION_MODE_ENV = "COBRA_PROJECTOR_ROTATION_MODE"

def _resolve_rotation_mode_from_env() -> Optional[str]:
    """
    解析環境變數中的 projector rotation mode，並做穩健 fallback。

    行為：
        - 若未設置，預設為 "hk"。
        - 若設置為合法值（含別名），回傳標準化後的字串：
              "hk" / "hadamard" / "none"
        - 若設置為非法值，印出警告並 fallback 為 "hk"。
    """
    raw = os.environ.get(_ROTATION_MODE_ENV, None)
    if raw is None or not raw.strip():
        return "hk"

    try:
        # 利用 QuantRuntimeConfig 自己的 parser 做 mapping，然後取 enum.value
        mode_enum = QuantRuntimeConfig._parse_projector_rotation_mode(raw)
        return mode_enum.value  # "hk" / "hadamard" / "none"
    except ValueError:
        print(
            "[load_quantized_cobra_vlm] WARNING: Unknown "
            f"{_ROTATION_MODE_ENV}={raw!r}; falling back to 'hk'."
        )
        return "hk"

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
    Load the hi/lo clipping map produced by `quant_calibrate.py` or `quant_pct_apply.py`.

    These scripts save a plain dict (hi_lo_map) via `torch.save`.
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

def _classify_target_from_module_name(module_name: str) -> str:
    """
    Heuristic mapping from module qualified name to canonical target.

    This mirrors the logic used by wrap policy / percentile calibration so that we can
    group activation quantizers by:
        - "fusion"
        - "vision.dino"
        - "vision.siglip"
        - "llm"
        - "projector"
    """
    module_name = (module_name or "").strip()
    if not module_name:
        return "<unknown>"

    # Fusion stage (Point-B)
    # Wrap policy uses "fusion_stage" / "fusion_stage.*"
    if module_name == "fusion_stage" or module_name.startswith("fusion_stage."):
        return "fusion"
    # Additional permissive fallback (covers possible refactors)
    if ".fusion_stage" in module_name or module_name.startswith("fusion"):
        return "fusion"

    # Vision
    if module_name.startswith("vision_backbone.dino_featurizer") or module_name.startswith(
        "vision_backbone.featurizer"
    ):
        return "vision.dino"
    if module_name.startswith("vision_backbone.siglip_featurizer"):
        return "vision.siglip"

    # LLM
    if module_name.startswith("llm_backbone.llm"):
        return "llm"

    # Projector
    if module_name.startswith("projector"):
        return "projector"

    return "<unknown>"

def _summarize_activation_quantizers(model: nn.Module, *, header: str) -> None:
    """
    Inspect all activation quantizers in the wrapped model and print coverage
    statistics per canonical target, focusing on whether `scale` has been
    populated (i.e., calibration actually wrote parameters into them).

    This is a runtime-only diagnostic and does not mutate the model.
    """
    coverage: Dict[str, Dict[str, int]] = {}
    total_quantizers = 0
    total_calibrated = 0

    for module_name, module in model.named_modules():
        # Pattern A: standard modules with a single act_quantizer
        for attr in ("act_quantizer", "x1_quantizer", "x2_quantizer"):
            q = getattr(module, attr, None)
            if not isinstance(q, UniformAffineQuantizer):
                continue

            target = _classify_target_from_module_name(module_name)
            bucket = coverage.setdefault(
                target,
                {"total": 0, "calibrated": 0},
            )

            bucket["total"] += 1
            total_quantizers += 1

            # After calibration, UniformAffineQuantizer.scale should be a tensor
            # (and clip_min/clip_max should be non-None). We use scale as the
            # primary indicator here.
            if getattr(q, "scale", None) is not None:
                bucket["calibrated"] += 1
                total_calibrated += 1

    print(f"[load_quantized_cobra_vlm] Activation quantizer coverage {header}:")
    if total_quantizers == 0:
        print("  (no UniformAffineQuantizer instances found; is the model wrapped?)")
        return

    for tgt, stats in coverage.items():
        total = stats["total"]
        calibrated = stats["calibrated"]
        ratio = float(calibrated) / float(total) if total > 0 else 0.0
        print(
            f"  - target={tgt:12s} calibrated={calibrated:4d} / {total:4d} "
            f"(coverage={ratio:6.3f})"
        )

    global_ratio = float(total_calibrated) / float(total_quantizers)
    print(
        f"  -> global calibrated={total_calibrated} / {total_quantizers} "
        f"(coverage={global_ratio:6.3f})"
    )

def _apply_runtime_weight_bits(vlm: nn.Module, quant_cfg: QuantRuntimeConfig) -> None:
    """
    Propagate QuantRuntimeConfig.weight_bits to all Quant* modules
    (weights only).

    This ensures that configurations like W2A2 / W4A4 / W8A8 truly differ
    in their *weight* bit-width at runtime, instead of always using the
    construction-time default (typically 8 bits).

    Args:
        vlm:
            The already-wrapped Cobra VLM (after wrap_model_for_quantization).
        quant_cfg:
            QuantRuntimeConfig for the current run. Only effective when
            quant_cfg.mode is QuantMode.FAKE.
    """
    if quant_cfg.mode is not QuantMode.FAKE:
        # Only meaningful for FAKE runtime; FLOAT is handled elsewhere.
        print(
            "[load_quantized_cobra_vlm] _apply_runtime_weight_bits called with "
            f"mode={quant_cfg.mode.value}; skipping."
        )
        return

    w_bits = quant_cfg.weight_bits
    if w_bits is None:
        print("[load_quantized_cobra_vlm] weight_bits is None; skipping weight bit propagation.")
        return

    # Local imports to avoid circular dependencies at module import time.
    from cobra.quantize.int_linear import QuantLinear
    from cobra.quantize.int_conv import QuantConv1d, QuantConv2d
    from cobra.quantize.int_matmul import QuantMatMul

    print(
        "[load_quantized_cobra_vlm] Applying runtime weight_bits to Quant* modules "
        f"(weight_bits={w_bits}) ..."
    )

    num_modules = 0
    for module_name, module in vlm.named_modules():
        if isinstance(module, (QuantLinear, QuantConv1d, QuantConv2d, QuantMatMul)):
            # Only touch weight bits here; activation bits are driven by calibrator.
            module.change_bits(weight_bits=w_bits, act_bits=None)
            num_modules += 1

    print(
        "[load_quantized_cobra_vlm] Runtime weight_bits applied to "
        f"{num_modules} Quant* modules (W{w_bits})."
    )

def _iter_wrap_registry_entries(registry):
    """
    Extract (target, module_path) from WrapRegistry in a version-tolerant way.

    This is intentionally defensive: cobra_1115_20251230's WrapRegistry does not
    expose a stable public iterator API, so we introspect instance attributes.

    Expected entry has attributes:
    - target (str)
    - module_path (str)
    """
    # 1) Try common attribute names that store list/tuple of records
    candidate_attr_names = [
        "_wrapped",
        "_records",
        "_entries",
        "_items",
        "_wrapped_records",
        "wrapped",
        "records",
        "entries",
        "items",
    ]

    for name in candidate_attr_names:
        if hasattr(registry, name):
            obj = getattr(registry, name)
            if isinstance(obj, (list, tuple)):
                for rec in obj:
                    target = getattr(rec, "target", None)
                    module_path = getattr(rec, "module_path", None)
                    if isinstance(target, str) and isinstance(module_path, str):
                        yield target, module_path
                return  # Found a plausible container; stop searching

    # 2) Fallback: scan registry.__dict__ for list/tuple that contains record-like objects
    if hasattr(registry, "__dict__"):
        for name, obj in registry.__dict__.items():
            if isinstance(obj, (list, tuple)) and len(obj) > 0:
                # Heuristic: does this container hold objects with target/module_path?
                rec0 = obj[0]
                if hasattr(rec0, "target") and hasattr(rec0, "module_path"):
                    for rec in obj:
                        target = getattr(rec, "target", None)
                        module_path = getattr(rec, "module_path", None)
                        if isinstance(target, str) and isinstance(module_path, str):
                            yield target, module_path
                    return

    # 3) If still nothing, raise with actionable debug info
    keys = []
    try:
        keys = sorted(list(getattr(registry, "__dict__", {}).keys()))
    except Exception:
        keys = []
    raise RuntimeError(
        "[Stage-1] Cannot extract wrap coverage from WrapRegistry. "
        f"WrapRegistry type={type(registry)!r}, __dict__ keys={keys}. "
        "Please inspect registry.__dict__ to locate the record container."
    )

def _enable_model_fake_quant(vlm: nn.Module, quant_cfg: QuantRuntimeConfig) -> None:
    """
    Enable MambaQuant-style fake quantization on the wrapped Cobra VLM.

    This function is ONLY used in QuantMode.FAKE:
        - It toggles use_weight_quant / use_act_quant on all Quant* modules
          via cobra.quantize.utils.set_quant_state.
        - The actual quantization behavior is implemented inside each module
          (e.g. QuantLinear / QuantConv / QuantMatMul), which will:
              * fake-quantize weights once (and cache them),
              * fake-quantize activations on every forward pass.

    Args:
        vlm:
            The already-wrapped Cobra VLM (after wrap_model_for_quantization).
        quant_cfg:
            QuantRuntimeConfig used for this run. Currently we always enable
            both weight and activation fake quantization when mode=FAKE.
    """
    # Local import to avoid creating a hard dependency at module import time.
    from cobra.quantize.utils import set_quant_state as _set_model_quant_state

    if quant_cfg.mode is not QuantMode.FAKE:
        # Safety guard: this helper is only meaningful in FAKE mode.
        print(
            "[load_quantized_cobra_vlm] _enable_model_fake_quant called with "
            f"mode={quant_cfg.mode.value}; skipping."
        )
        return

    # For now we always enable both weight and activation fake quant in FAKE mode.
    weight_quant = True
    act_quant = True

    print(
        "[load_quantized_cobra_vlm] Enabling fake quantization on wrapped modules "
        f"(weight_quant={weight_quant}, act_quant={act_quant}) ..."
    )
    _set_model_quant_state(vlm, weight_quant=weight_quant, act_quant=act_quant)
    print("[load_quantized_cobra_vlm] Fake quantization flags set on Quant* modules.")

# ================================================================
#   Main Unified API
# ================================================================
def load_quantized_cobra_vlm(
    *,
    bits: Optional[str],
    pct_hi_lo_path,
    hf_token: str,
    base_dtype,
    device,
    enabled_targets=None,
    run_dir=None,
    output_dir=None,
):
    """
    Stage 1 (W-only baseline) compliant loader.

    Key guarantees for Stage 1:
      - activation remains float (no pct_hi_lo dependency, no calibration)
      - weights are fake-quantized (W8/W4/W2)
      - fusion wrapping disabled (avoid coverage/latency contamination)
      - projector rotation disabled by ROTATION_MODE=none (handled by QuantRuntimeConfig)
      - coverage.json is emitted
    """
    from pathlib import Path
    import json

    # -----------------------------
    # 1) Build QuantRuntimeConfig
    # -----------------------------
    quant_cfg = QuantRuntimeConfig.from_bits_backend(
        bits=bits,
        backend="fake" if bits is not None else "float",
        enable_vision_dino=True,
        enable_vision_siglip=True,
        enable_llm=True,
        enable_projector=True,
        projector_rotation_mode=os.environ.get("COBRA_PROJECTOR_ROTATION_MODE", "hk"),
        enable_act_quant=False,  # Stage 1 hard lock
    )

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

    # -----------------------------
    # 2) Load float Cobra VLM 
    # -----------------------------
    model_id_or_path = _resolve_model_id_or_path()
    print(f"[load_quantized_cobra_vlm] Loading float Cobra from {model_id_or_path!r} ...")

    vlm = cobra_load(
        model_id_or_path,
        hf_token=hf_token,
    )
    vlm.to(device=device, dtype=base_dtype)


    # -----------------------------
    # 3) Wrap model (Quant* modules)
    #    IMPORTANT: disable fusion for Stage 1 baseline
    # -----------------------------
    enabled_targets_set = set(enabled_targets) if enabled_targets else set(quant_cfg.enabled_targets())

    wrap_policy_cfg = WrapPolicyConfig(
        enable_vision_dino="vision.dino" in enabled_targets_set,
        enable_vision_siglip="vision.siglip" in enabled_targets_set,
        enable_llm="llm" in enabled_targets_set,
        enable_projector="projector" in enabled_targets_set,
        enable_fusion=False,          # Stage 1 hard lock
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
    # 4) Apply runtime weight bits
    # -----------------------------
    _apply_runtime_weight_bits(vlm, quant_cfg=quant_cfg)

    # -----------------------------
    # 5) Enable fake quant flags
    #    Stage 1: weight_quant=True, act_quant=False
    # -----------------------------
    weight_quant = True
    act_quant = False
    set_quant_state(vlm, weight_quant=weight_quant, act_quant=act_quant)

    # -----------------------------
    # 6) Emit coverage.json (Stage 1 required output)
    # -----------------------------
    if output_dir is not None and registry is not None:
        from collections import defaultdict
        import json
        from pathlib import Path

        by_target = defaultdict(list)

        for target, module_path in _iter_wrap_registry_entries(registry):
            by_target[target].append(module_path)

        coverage_payload = {
            "stage": "stage1_weight_only",
            "bits": bits,
            "backend": quant_cfg.backend,
            "counts": {k: len(v) for k, v in by_target.items()},
            "module_paths": dict(by_target),
        }

        cov_path = Path(output_dir) / f"coverage_{bits}.json"
        cov_path.write_text(
            json.dumps(coverage_payload, indent=2, ensure_ascii=False)
        )

        print(f"[Stage-1] coverage written to: {cov_path}")

    # -----------------------------
    # 7) Activation calibration (SKIPPED in Stage 1)
    # -----------------------------
    if quant_cfg.should_calibrate_activations():
        # Keep the original behavior for later stages
        if pct_hi_lo_path is None:
            raise ValueError(
                "[load_quantized_cobra_vlm] pct_hi_lo_path is required when enable_act_quant=True"
            )
        hi_lo_map = _load_hi_lo_map(pct_hi_lo_path)
        calibrate_model_from_hi_lo(
            vlm,
            hi_lo_map=hi_lo_map,
            act_bits=quant_cfg.act_bits,
            signed=quant_cfg.symmetric_acts,
            enabled_targets=tuple(sorted(enabled_targets_set)),
        )

    # -----------------------------
    # 8) Projector rotation (Stage 1 expects NONE)
    # -----------------------------
    # Keep your existing rotation block as-is; Stage 1 forces env ROTATION_MODE=none in vlm_eval.sh,
    # so quant_cfg.should_rotate_projector() will be False.

    return vlm

