# cobra/quantize/runtime/load_quantized_vlm.py
"""
Unified entrypoint for reconstructing a *runtime* Cobra VLM for evaluation.

This module is used ONLY at inference time (eval), not for calibration.

What this loader does:
  1) Load the float Cobra checkpoint (via cobra.load)
  2) Wrap the model with fake-quant modules (MambaQuant-style: float kernels +
     quant/dequant stubs)
  3) Apply activation clipping ranges (pct hi/lo) produced by quant_calibrate.py
  4) Enable fake-quant at runtime (QuantRuntimeConfig.mode == FAKE)
  5) Optional rotations:
       - Output projector rotation (existing path): lm_head / output projection
       - Fusion rotation at Point B (Milestone 2): fused_embeddings rotation

Design principles:
  - QuantRuntimeConfig is the single source of truth for runtime behavior.
  - This file performs *plumbing* only (load + wrap + apply stats + configure
    optional rotations). It does not compute calibration stats.
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
from typing import Optional, Tuple

import torch
from torch import nn

from cobra import load as cobra_load
from cobra.quantize.pct.calibrator import calibrate_model_from_hi_lo
from cobra.quantize.runtime.config import QuantRuntimeConfig, QuantMode
from cobra.quantize.rotate.projector import (
    ProjectorRotationConfig,
    SHARED_KLT_PATH,
    rotate_cobra_vlm_output_projector_from_path_inplace,
)
from cobra.quantize.wrap.entry import wrap_model_for_quantization
from cobra.quantize.wrap.policy import WrapPolicyConfig

from cobra.quantize.runtime.pipeline_spec import (
    ENV_PROJECTOR_ROTATION_MODE,
    ENV_FUSION_ROTATION_MODE,
    ENV_FUSION_ROTATION_ABSORB,
    LLM_XT_ACT_QUANT_ALLOW_SUBSTRINGS,
)

# -----------------------------------------------------------------------------
# Env controls (runtime only)
# -----------------------------------------------------------------------------

def _resolve_rotation_mode_from_env() -> str:
    """
    Resolve projector rotation mode from env, with robust fallback.

    Behavior:
      - If unset: default to "hk"
      - If set: normalize via QuantRuntimeConfig parser
      - If invalid: warn and fallback to "hk"
    """
    raw = os.environ.get(ENV_PROJECTOR_ROTATION_MODE)
    if raw is None or not raw.strip():
        return "hk"

    try:
        mode_enum = QuantRuntimeConfig._parse_projector_rotation_mode(raw)
        return mode_enum.value  # "hk" / "hadamard" / "none"
    except Exception as e:
        print(
            f"[load_quantized_cobra_vlm] WARNING: invalid {ENV_PROJECTOR_ROTATION_MODE}={raw!r} "
            f"({type(e).__name__}: {e}); fallback to 'hk'."
        )
        return "hk"

def _resolve_fusion_rotation_mode_from_env() -> str:
    """
    Resolve fusion rotation mode (Point B) from env, with robust fallback.

    Behavior:
      - If unset: default to "none"  (safety: do not enable silently)
      - If set: normalize via QuantRuntimeConfig parser
      - If invalid: warn and fallback to "none"
    """
    raw = os.environ.get(ENV_FUSION_ROTATION_MODE, None)
    if raw is None or not raw.strip():
        return "none"

    try:
        mode_enum = QuantRuntimeConfig._parse_fusion_rotation_mode(raw)
        return mode_enum.value  # "hk" / "hadamard" / "none"
    except Exception as e:
        print(
            f"[load_quantized_cobra_vlm] WARNING: invalid {ENV_FUSION_ROTATION_MODE}={raw!r} "
            f"({type(e).__name__}: {e}); fallback to 'none'."
        )
        return "none"

def _resolve_absorb_fusion_rotation_from_env() -> bool:
    """
    Whether to absorb fusion rotation inverse into the first LLM linear.

    Default: False (off)
    Enable via: COBRA_FUSION_ROTATION_ABSORB=1/true/on/yes
    """
    raw = os.environ.get(ENV_FUSION_ROTATION_ABSORB, "")
    s = str(raw).strip().lower()
    return s in ("1", "true", "yes", "y", "on", "enable", "enabled")

def _resolve_model_id_or_path() -> str:
    """
    Resolve Cobra model id or run_dir from env COBRA_MODEL_ID_OR_PATH.
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
    Load the hi/lo clipping map produced by `quant_calibrate.py` (torch.save dict).
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

def _load_shared_klt_matrix(*, klt_path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Load shared KLT matrix from a torch.load dict containing key 'K'.
    """
    obj = torch.load(klt_path, map_location="cpu")
    if not isinstance(obj, dict) or "K" not in obj:
        raise ValueError(
            f"[load_quantized_cobra_vlm] Invalid KLT file format at {str(klt_path)}; "
            "expected dict with key 'K'."
        )

    K = obj["K"]
    if not isinstance(K, torch.Tensor):
        raise TypeError(f"[load_quantized_cobra_vlm] KLT 'K' must be torch.Tensor, got {type(K)}")
    if K.ndim != 2 or K.shape[0] != K.shape[1]:
        raise ValueError(f"[load_quantized_cobra_vlm] KLT 'K' must be square [D,D], got shape={tuple(K.shape)}")

    return K.to(device=device, dtype=dtype)

def _configure_llm_xt_only_activation_quant(vlm: nn.Module, quant_cfg: QuantRuntimeConfig) -> None:
    """
    Milestone 5:
      - Keep activation quantization ONLY at x_t entry points in each LLM block.
      - Disable activation quant on all other LLM QuantLinear modules.

    Implementation details:
      - We gate by module *name* substring allowlist (LLM_XT_ACT_QUANT_ALLOW_SUBSTRINGS).
      - We explicitly call `set_quant_state(..., act_quant=...)` to keep wrapper flags
        and underlying quantizer enable-state consistent.
      - We also force act bitwidth for enabled modules to match QuantRuntimeConfig.act_bits.
    """
    from cobra.quantize.int_linear import QuantLinear

    if quant_cfg.mode is not QuantMode.FAKE:
        return
    if not hasattr(vlm, "llm_backbone"):
        return

    llm = getattr(vlm, "llm_backbone")
    allow_substrings = LLM_XT_ACT_QUANT_ALLOW_SUBSTRINGS

    act_bits = quant_cfg.act_bits
    # activation fake quant is only meaningful when A<16
    want_act_quant = (act_bits is not None) and (int(act_bits) < 16)

    n_total = 0
    n_enabled = 0
    n_missing_quantizer = 0
def _classify_target_from_module_name(module_name: str) -> str:
    """
    Heuristic mapping from module qualified name to canonical target.

    This mirrors the logic used in percentile calibration so that we can
    group activation quantizers by:
        - "vision.dino"
        - "vision.siglip"
        - "llm"
        - "projector"
    """
    if module_name.startswith("vision_backbone.dino_featurizer") or module_name.startswith(
        "vision_backbone.featurizer"
    ):
        return "vision.dino"
    if module_name.startswith("vision_backbone.siglip_featurizer"):
        return "vision.siglip"
    if module_name.startswith("llm_backbone.llm"):
        return "llm"
    if module_name.startswith("projector"):
        return "projector"
    return "<unknown>"


def _summarize_activation_quantizers(model: nn.Module, *, header: str) -> None:
    """
    Inspect all activation quantizers in the wrapped model and print coverage
    statistics per canonical target, focusing on whether `scale` has been
    populated (i.e., calibration actually wrote parameters into them).

    for name, mod in llm.named_modules():
        if not isinstance(mod, QuantLinear):
            continue

        n_total += 1

        lname = (name or "").lower()
        is_xt_entry = any(s in lname for s in allow_substrings)

        # Wrapper-level flags used in forward()
        mod.use_act_quant = bool(want_act_quant and is_xt_entry)
        mod.disable_input_quant = not mod.use_act_quant

        # Keep underlying quantizer state consistent
        try:
            mod.set_quant_state(weight_quant=mod.use_weight_quant, act_quant=mod.use_act_quant)
        except Exception:
            pass

        # Ensure enabled modules have correct A-bit set
        if mod.use_act_quant and mod.act_quantizer is not None:
            try:
                mod.change_bits(weight_bits=None, act_bits=int(act_bits))
            except Exception:
                pass
            n_enabled += 1
        elif mod.use_act_quant and mod.act_quantizer is None:
            # Should be rare because wrappers are created with disable_input_quant=False by default,
            # but keep a counter for debugging.
            n_missing_quantizer += 1

    print(
        f"[load_quantized_cobra_vlm] LLM x_t-only act-quant configured: enabled={n_enabled}/{n_total} QuantLinear modules "
        f"(A{act_bits}, missing_act_quantizer={n_missing_quantizer})."
    )


def _apply_runtime_weight_bits(vlm: nn.Module, quant_cfg: QuantRuntimeConfig) -> None:
    """
    Propagate QuantRuntimeConfig.weight_bits to QuantLinear / QuantConv* modules.

    Notes:
      - QuantLinear/QuantConv store bitwidth inside their quantizers
        (weight_quantizer.n_bits), not as a top-level `w_bits` attribute.
      - These modules cache quantized weights via `weight_quantized`; if we change
        bits at runtime we must reset this flag to force re-quantization on the
        next forward.
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
        return

    from cobra.quantize.int_linear import QuantLinear
    from cobra.quantize.int_conv import QuantConv1d, QuantConv2d, QuantConv3d

    n_updated = 0
    n_reset = 0

    for _, module in vlm.named_modules():
        if isinstance(module, (QuantLinear, QuantConv1d, QuantConv2d, QuantConv3d)):
            try:
                module.change_bits(weight_bits=int(w_bits), act_bits=None)
                n_updated += 1
            except Exception:
                # Be conservative: if a module refuses, skip it.
                continue

            # Ensure the next forward re-quantizes weights under the new bitwidth.
            if hasattr(module, "weight_quantized"):
                try:
                    module.weight_quantized = False
                    n_reset += 1
                except Exception:
                    pass

    print(
        f"[load_quantized_cobra_vlm] Applied runtime weight_bits={w_bits} to {n_updated} Quant* modules "
        f"(reset_weight_cache={n_reset})."
    )

def _enable_model_fake_quant(vlm: nn.Module, quant_cfg: QuantRuntimeConfig) -> None:
    """
    Enable fake-quant runtime behavior for QuantLinear / QuantConv* modules.

    Refactor note:
      - Older code toggled `module.enabled` when present, but the main wrappers in
        this repo (QuantLinear/QuantConv*) control behavior via `set_quant_state`.
      - We intentionally enable *weight* fake quant broadly (when W<16), while
        leaving activation fake quant to be configured separately (e.g., LLM x_t-only).
    """
    if quant_cfg.mode is not QuantMode.FAKE:
        return

    from cobra.quantize.int_linear import QuantLinear
    from cobra.quantize.int_conv import QuantConv1d, QuantConv2d, QuantConv3d

    weight_quant = (quant_cfg.weight_bits is not None) and (int(quant_cfg.weight_bits) < 16)

    n_touched = 0
    n_failed = 0
    
    for _, module in vlm.named_modules():
        if isinstance(module, (QuantLinear, QuantConv1d, QuantConv2d, QuantConv3d)):
            try:
                module.set_quant_state(weight_quant=weight_quant, act_quant=False)
                n_touched += 1
            except Exception:
                pass

    print(
        f"[load_quantized_cobra_vlm] Enabled fake-quant wrappers: touched={n_touched} "
        f"(weight_quant={weight_quant}, act_quant=False, failed={n_failed})."
    )

def _disable_vision_activation_quant(vlm: nn.Module) -> None:
    """
    Disable activation fake quant in the vision backbone (clip-only behavior), without
    affecting weight fake quant.

    This function also syncs the underlying quantizer enable-state via set_quant_state.
    """
    from cobra.quantize.int_linear import QuantLinear

    if not hasattr(vlm, "vision_backbone"):
        return

    vb = getattr(vlm, "vision_backbone")
    n = 0
    for _, mod in vb.named_modules():
        if isinstance(mod, QuantLinear):
            mod.use_act_quant = False
            mod.disable_input_quant = True
            try:
                mod.set_quant_state(weight_quant=mod.use_weight_quant, act_quant=False)
            except Exception:
                pass
            n += 1

    print(f"[load_quantized_cobra_vlm] Disabled vision activation quant on {n} QuantLinear modules (clip-only).")

def _find_first_llm_linear(vlm: nn.Module) -> Tuple[str, nn.Linear]:
    """
    Best-effort locator for the first LLM nn.Linear that consumes embeddings.

    This is intentionally conservative and used only for optional absorption.
    """
    if not hasattr(vlm, "llm_backbone"):
        raise AttributeError("[load_quantized_cobra_vlm] vlm has no attribute 'llm_backbone'")

    llm = getattr(vlm, "llm_backbone")
    for name, mod in llm.named_modules():
        if not isinstance(mod, nn.Linear):
            continue

        n = (name or "").lower()
        if "lm_head" in n:
            continue
        if "embed" in n or "embedding" in n:
            continue

        return name, mod

    raise RuntimeError("[load_quantized_cobra_vlm] Could not locate a suitable first nn.Linear in llm_backbone")

def _absorb_fusion_inverse_into_first_linear(
    *,
    vlm: nn.Module,
    fusion_mode: str,
    K: Optional[torch.Tensor],
) -> None:
    """
    Absorb R^{-1} into the first LLM linear weight (W_first := W_first · R^{-1}).

    Assumptions:
      - Hadamard and K are orthogonal (so inverse equals transpose).
      - Our fused activation rotation is:
          "hadamard": x := x·H
          "hk":       x := x·K then x := x·H    (x := x·(H·K))
        Thus R^{-1} = (H·K)^{-1} = K^T · H^T  (and H^T = H)

    Implementation avoids materializing full R.
    """
    mode = (fusion_mode or "none").strip().lower()
    if mode == "none":
        return

    linear_path, linear = _find_first_llm_linear(vlm)

    from cobra.quantize.rotate.hadamard import apply_hadamard_transform
    from cobra.quantize.rotate.klt import apply_klt_rotation

    W = linear.weight.data

    if mode == "hadamard":
        linear.weight.data = apply_hadamard_transform(W, dim=-1, normalize=True)
        print(f"[load_quantized_cobra_vlm] Absorbed fusion inverse (hadamard) into llm_backbone.{linear_path}")
        return

    if mode != "hk":
        raise ValueError(f"[load_quantized_cobra_vlm] Unsupported fusion_mode for absorption: {fusion_mode!r}")

    if K is None:
        raise ValueError("[load_quantized_cobra_vlm] Absorption for mode='hk' requires K")

    W_h = apply_hadamard_transform(W, dim=-1, normalize=True)     # W := W·H
    W_new = apply_klt_rotation(W_h, K.T, right_multiply=True)     # W := W·K^T
    linear.weight.data = W_new

    print(f"[load_quantized_cobra_vlm] Absorbed fusion inverse (hk) into llm_backbone.{linear_path}")

def load_quantized_cobra_vlm(
    *,
    bits: str,
    pct_hi_lo_path: Path,
    hf_token: Optional[str],
    base_dtype: torch.dtype,
    device: torch.device,
) -> nn.Module:
    """
    Load a Cobra VLM according to the current PTQ stack.

    Args:
        bits:
            String like "W8A8", "W4A4", "W2A2", "W16A16", "W8A4", "W4A8".
        pct_hi_lo_path:
            Path to the hi/lo map produced by quant_calibrate.py.
            Path to the hi/lo clipping map produced by `quant_calibrate.py`
            or `quant_pct_apply.py`.
        hf_token:
            HF token passed to cobra.load().
        base_dtype:
            Inference dtype (e.g., torch.bfloat16 / torch.float16).
        device:
            Target device.

    Returns:
        Wrapped Cobra VLM (float or fake-quant) ready for inference.
    """
    # ------------------------------------------------------------------
    # 0) Build QuantRuntimeConfig from bits + backend + rotation env
    # ------------------------------------------------------------------
    backend = os.environ.get("BACKEND", "fake")
    rotation_mode_raw = _resolve_rotation_mode_from_env()
    fusion_rotation_mode_raw = _resolve_fusion_rotation_mode_from_env()
    absorb_fusion_rotation = _resolve_absorb_fusion_rotation_from_env()

    quant_cfg = QuantRuntimeConfig.from_bits_backend(
        bits=bits,
        backend=backend,
        enable_vision_dino=True,
        enable_vision_siglip=True,
        enable_llm=True,
        enable_projector=True,
        enable_fusion=True,
        vision_in_pct_pipeline=True,
        symmetric_acts=True,
        symmetric_weights=True,
        config_name=f"load_quantized_vlm::{bits}::{backend}",
        projector_rotation_mode=rotation_mode_raw,
        fusion_rotation_mode=fusion_rotation_mode_raw,
        absorb_fusion_rotation=absorb_fusion_rotation,
    )

    enabled_targets_set = set(quant_cfg.use_pct_for)
    rotation_mode = quant_cfg.projector_rotation_mode.value
    fusion_mode = quant_cfg.fusion_rotation_mode.value

    print(
        "[load_quantized_cobra_vlm] Requested configuration: "
        f"bits={quant_cfg.bits} (W{quant_cfg.weight_bits}A{quant_cfg.act_bits}), "
        f"mode={quant_cfg.mode.value}, backend={quant_cfg.backend}, "
        f"use_pct_for={sorted(quant_cfg.use_pct_for)}, "
        f"projector_rotation_mode={rotation_mode}, "
        f"fusion_rotation_mode={fusion_mode}, absorb_fusion_rotation={quant_cfg.absorb_fusion_rotation}"
    )

    # ------------------------------------------------------------------
    # 1) Load float model
    # ------------------------------------------------------------------
    model_id_or_path = _resolve_model_id_or_path()
    print(f"[load_quantized_cobra_vlm] Loading float Cobra model: {model_id_or_path}")
    vlm = cobra_load(model_id_or_path, hf_token=hf_token).to(dtype=base_dtype, device=device)
    vlm.eval()

    # ------------------------------------------------------------------
    # 2) Wrap model for fake-quant runtime
    # ------------------------------------------------------------------
    print("[load_quantized_cobra_vlm] Wrapping model with fake-quant modules via wrap_model_for_quantization(...)")

    if quant_cfg.mode is QuantMode.FLOAT:
        # Pure float: no wrap, no calibration, no pct consumed.
        print("[load_quantized_cobra_vlm] QuantMode=FLOAT; returning float Cobra without wrapping.")
        return vlm

    # From here on we are in FAKE mode (current main path).
    assert quant_cfg.mode is QuantMode.FAKE

    # ------------------------------------------------------------------
    # 3. Wrap model with Quant* modules (fake quant)
    # ------------------------------------------------------------------
    print(
        "[load_quantized_cobra_vlm] Wrapping model with Quant* modules via "
        "wrap_model_for_quantization(...)"
    )

    # Wrap policy aligned with QuantRuntimeConfig.use_pct_for: we only wrap
    # targets that participate in the percentile pipeline.
    wrap_policy_cfg = WrapPolicyConfig(
        enable_vision_dino="vision.dino" in enabled_targets_set,
        enable_vision_siglip="vision.siglip" in enabled_targets_set,
        enable_llm="llm" in enabled_targets_set,
        enable_projector="projector" in enabled_targets_set,
        include_linear=True,
        include_conv=True,
    )

    wrap_model_for_quantization(
        vlm,
        policy_cfg=wrap_policy_cfg,
        manifest=None,
        default_params=None,
        prefix="",
    )

    _apply_runtime_weight_bits(vlm, quant_cfg=quant_cfg)

    # ------------------------------------------------------------------
    # 3) Apply activation hi/lo map (pct) => calibrate activations
    # ------------------------------------------------------------------
    print(f"[load_quantized_cobra_vlm] Loading activation hi/lo map from {pct_hi_lo_path}")
    hi_lo_map = _load_hi_lo_map(pct_hi_lo_path)

    calibrate_model_from_hi_lo(
        vlm,
        hi_lo_map,
        include_targets=sorted(quant_cfg.use_pct_for),
        signed=quant_cfg.symmetric_acts,
    )

    if hasattr(vlm, "fusion_stage") and hasattr(vlm.fusion_stage, "mark_clipping_ready"):
        vlm.fusion_stage.mark_clipping_ready()
    # --- Milestone 4: Global low-bit mapping @ Point B ---
    if hasattr(vlm, "fusion_stage") and hasattr(vlm.fusion_stage, "configure_quant"):
        if quant_cfg.should_quantize_fusion():
            vlm.fusion_stage.configure_quant(n_bits=int(quant_cfg.act_bits), enable=True)
            print(f"[load_quantized_cobra_vlm] Enabled fusion quant-dequant at B with act_bits={quant_cfg.act_bits}")
        else:
            # explicit disable (keeps Milestone 3 clipping-only behavior)
            vlm.fusion_stage.configure_quant(n_bits=16, enable=False)
            print("[load_quantized_cobra_vlm] Disabled fusion quant-dequant at B")

    # ------------------------------------------------------------------
    # 4) Enable fake-quant runtime behavior
    # ------------------------------------------------------------------
    _enable_model_fake_quant(vlm, quant_cfg=quant_cfg)

    _disable_vision_activation_quant(vlm)
    _configure_llm_xt_only_activation_quant(vlm, quant_cfg=quant_cfg)

    # ------------------------------------------------------------------
    # 5) Optional: configure fusion rotation @ Point B
    # ------------------------------------------------------------------
    K_for_fusion: Optional[torch.Tensor] = None
    if quant_cfg.should_rotate_fusion():
        if not hasattr(vlm, "fusion_stage"):
            raise AttributeError(
                "[load_quantized_cobra_vlm] vlm has no attribute 'fusion_stage'. "
                "Milestone 1 (FusionStage hook) may be missing."
            )

        if fusion_mode == "hk":
            klt_path = SHARED_KLT_PATH
            if not Path(klt_path).exists():
                raise FileNotFoundError(f"[load_quantized_cobra_vlm] Fusion rotation needs KLT file at {str(klt_path)}")
            K_for_fusion = _load_shared_klt_matrix(klt_path=Path(klt_path), device=device, dtype=base_dtype)

        vlm.fusion_stage.configure_rotation(mode=fusion_mode, klt_matrix=K_for_fusion)
        print(
            "[load_quantized_cobra_vlm] Configured fusion rotation "
            f"(mode={fusion_mode}, has_klt={K_for_fusion is not None})"
        )

        if quant_cfg.should_absorb_fusion_rotation():
            _absorb_fusion_inverse_into_first_linear(vlm=vlm, fusion_mode=fusion_mode, K=K_for_fusion)

    # ------------------------------------------------------------------
    # 6) Optional: apply output projector rotation (existing)
    # ------------------------------------------------------------------
    if quant_cfg.should_rotate_projector():
        effective_use_klt = quant_cfg.projector_rotation_uses_klt()
        effective_use_hadamard = quant_cfg.projector_rotation_uses_hadamard()

        proj_rot_cfg = ProjectorRotationConfig(
            use_klt=effective_use_klt,
            use_hadamard=effective_use_hadamard,
            shared_klt=True,
        )

        module_path, lm_head = rotate_cobra_vlm_output_projector_from_path_inplace(
            vlm=vlm,
            klt_path=SHARED_KLT_PATH,
            cfg=proj_rot_cfg,
        )

        print(
            "[load_quantized_cobra_vlm] Applied projector rotation on "
            f"module_path={module_path}, lm_head_shape={tuple(lm_head.weight.shape)}"
        )
    else:
        print(
            "[load_quantized_cobra_vlm] Projector rotation skipped "
            f"(mode={quant_cfg.mode.value}, use_rotation_for={sorted(quant_cfg.use_rotation_for)}, "
            f"projector_rotation_mode={rotation_mode})"
        )

    return vlm

