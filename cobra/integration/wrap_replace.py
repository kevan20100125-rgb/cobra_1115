# cobra/integration/wrap_replace.py

"""
wrap_replace.py

Utilities for applying quantization wrappers to a Cobra model.

Responsibilities:
    - Build a WrapRegistry that records which modules should be wrapped
      with Quant* counterparts (using `quantize.wrap.registry`).
    - Apply that registry in-place to a given model:
        * replace float modules (nn.Linear / nn.ConvNd / ...) with Quant*
        * keep track of which canonical target each wrapped module belongs to
          ("vision.dino", "vision.siglip", "llm", "projector").
    - Restore a quantized integer state exported by
      `cobra.quantize.finalize.int_export.export_int_quant_state`, via:
        * apply_int_quant_state(model, blob)

This module does NOT:
    - Decide bitwidths or clipping strategies (handled by pct/* and finalize/int_export.py).
    - Perform percentile collection or calibration.
    - Perform any logging beyond using Overwatch.

Typical usage (inside a CLI or training script):

    from cobra.integration.wrap_replace import (
        wrap_model_for_quantization,
        apply_int_quant_state,
    )
    from cobra.quantize.wrap.policy import WrapPolicyConfig

    # 1) Float model → wrapped Quant* 模組
    policy_cfg = WrapPolicyConfig(
        enable_vision_dino=True,
        enable_vision_siglip=True,
        enable_llm=True,
        enable_projector=True,
    )
    wrap_registry = wrap_model_for_quantization(vlm, policy_cfg=policy_cfg)

    # 2) 推論環境：從 int_export_W8A8.pt 還原量化狀態
    blob = torch.load("outputs/quantize/int_export_W8A8.pt", map_location="cpu")
    apply_int_quant_state(vlm, blob)

    # `vlm` 會被就地替換成 Quant* 模組並安裝對應的
    # integer weights / scales / zero-points / activation quantizers。
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from cobra.overwatch import initialize_overwatch

from cobra.quantize.wrap.manifest import WrapRule, iter_default_wrap_rules
from cobra.quantize.wrap.policy import (
    DefaultWrapPolicy,
    WrapPolicyConfig,
    infer_target_from_module_path,
)
from cobra.quantize.wrap.registry import WrapEntry, WrapRegistry, build_wrap_registry
from cobra.quantize.wrap.utils import (
    WrapQuantParams,
    get_module_by_path,
    replace_module_inplace,
    is_quantized_module,
)

overwatch = initialize_overwatch(__name__)


# ======================================================================
# Core application logic
# ======================================================================


def apply_wrap_registry(
    model: nn.Module,
    registry: WrapRegistry,
    *,
    default_params: Optional[WrapQuantParams] = None,
) -> nn.Module:
    """
    Apply a WrapRegistry to `model` in-place.

    For each WrapEntry in the registry:
        - resolve the float module at `entry.module_path`,
        - construct the corresponding Quant* wrapper using `entry.rule`,
        - replace the module on the parent with the wrapper.

    Args:
        model:
            The nn.Module to mutate (e.g., Cobra VLM).
        registry:
            WrapRegistry built via `build_wrap_registry(...)`.
        default_params:
            Optional WrapQuantParams controlling default quantizer kwargs.
            If None, a fresh default instance is created for each wrap.

    Returns:
        The same `model` instance (mutated in-place), returned for convenience.
    """
    if default_params is None:
        # Use a single shared instance for all modules when None is provided.
        default_params = WrapQuantParams()

    replaced_count = 0

    for entry in registry:
        module_path = entry.module_path

        try:
            float_module = get_module_by_path(model, module_path)
        except AttributeError:
            overwatch.warning(
                f"[WrapReplace] Module path {module_path!r} not found on model; "
                f"skipping WrapEntry(target={entry.target!r}, kind={entry.rule_kind!r})",
                extra={"module_path": module_path, "target": entry.target},
            )
            continue

        # Defensive: skip if already Quant* (e.g., registry re-applied accidentally)
        if is_quantized_module(float_module):
            overwatch.info(
                f"[WrapReplace] Module at {module_path!r} is already quantized; skipping.",
                extra={"module_path": module_path, "target": entry.target},
            )
            continue

        # Construct wrapped module
        params = default_params
        wrapped = entry.rule.factory(float_module, params)

        # Replace on parent
        replace_module_inplace(model, module_path, wrapped)
        replaced_count += 1

    overwatch.info(
        f"[WrapReplace] Applied wrap registry: replaced {replaced_count} module(s) "
        f"out of {len(registry)} planned entries."
    )

    return model


# ======================================================================
# High-level convenience API
# ======================================================================


def wrap_model_for_quantization(
    model: nn.Module,
    *,
    policy_cfg: Optional[WrapPolicyConfig] = None,
    manifest: Optional[Sequence[WrapRule]] = None,
    default_params: Optional[WrapQuantParams] = None,
    prefix: str = "",
) -> WrapRegistry:
    """
    High-level helper to:
        1) Build a WrapRegistry for `model` using a DefaultWrapPolicy.
        2) Apply that registry in-place to `model`.

    Args:
        model:
            The nn.Module to wrap (e.g., Cobra VLM).
        policy_cfg:
            WrapPolicyConfig controlling per-target enable switches and
            basic inclusion/exclusion rules. If None, a default config
            is used.
        manifest:
            Optional manifest of WrapRule objects. If None, the default
            manifest from `quantize.wrap.manifest` is used.
        default_params:
            Optional WrapQuantParams controlling default quantization
            parameters for all wrapped modules. If None, a default instance
            is created internally.
        prefix:
            Optional dotted prefix; if provided, only modules whose path
            starts with this prefix will be considered for wrapping.

    Returns:
        The WrapRegistry that was constructed and applied.
    """
    # Build a registry according to the policy + manifest
    registry = build_wrap_registry(
        model,
        policy_cfg=policy_cfg,
        manifest=manifest,
        prefix=prefix,
    )

    if len(registry) == 0:
        overwatch.warning("[WrapReplace] No modules selected for wrapping; model unchanged.")
        return registry

    overwatch.info(
        "[WrapReplace] Applying wrap registry to model "
        f"(entries={len(registry)}, policy={policy_cfg!r})"
    )

    # Apply it in-place
    apply_wrap_registry(
        model,
        registry,
        default_params=default_params,
    )

    # Log a compact per-target summary
    by_target = registry.by_target()
    parts = []
    for tgt, entries in by_target.items():
        parts.append(f"{tgt}: {len(entries)}")
    overwatch.info(
        "[WrapReplace] Wrap summary by target: " + " | ".join(parts)
    )

    return registry


# ======================================================================
# Integer export restore logic
# ======================================================================


def _build_policy_cfg_from_export_config(cfg_dict: Mapping[str, Any]) -> WrapPolicyConfig:
    """
    Convert an IntExportConfig-like dict into a WrapPolicyConfig.

    We rely on the same canonical four-way split:
        - vision.dino
        - vision.siglip
        - llm
        - projector

    Unknown fields in cfg_dict are ignored.
    """
    enable_vision_dino = bool(cfg_dict.get("include_vision_dino", True))
    enable_vision_siglip = bool(cfg_dict.get("include_vision_siglip", True))
    enable_llm = bool(cfg_dict.get("include_llm", True))
    enable_projector = bool(cfg_dict.get("include_projector", True))

    policy_cfg = WrapPolicyConfig(
        enable_vision_dino=enable_vision_dino,
        enable_vision_siglip=enable_vision_siglip,
        enable_llm=enable_llm,
        enable_projector=enable_projector,
    )

    overwatch.info(
        "[WrapReplace] Derived WrapPolicyConfig from export config: %s",
        policy_cfg,
        extra={
            "enable_vision_dino": enable_vision_dino,
            "enable_vision_siglip": enable_vision_siglip,
            "enable_llm": enable_llm,
            "enable_projector": enable_projector,
        },
    )

    return policy_cfg


def _tensor_like_ref(t: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """
    Move tensor `t` to the same device as `ref`.

    We intentionally keep dtype of `t` (int weights / float scales),
    only align device。
    """
    if not isinstance(t, torch.Tensor):
        return t
    return t.to(device=ref.device)


def apply_int_quant_state(
    model: nn.Module,
    export_blob: Mapping[str, Any],
    *,
    strict: bool = True,
) -> nn.Module:
    """
    Restore a quantized integer state exported by `export_int_quant_state`.

    This function is the bridge from offline PTQ finalize → runtime inference:

        1) Use export_blob["config"] to derive a WrapPolicyConfig
           (which targets are included).
        2) Wrap the float model in-place using Quant* modules
           via `wrap_model_for_quantization`.
        3) For each entry in export_blob["weights"]:
             - find the corresponding module by path
             - install integer weights and (scale, zero_point) metadata
               on the Quant* module.
        4) For each entry in export_blob["activations"]:
             - find the module
             - install activation quantizer scale / zero_point
               (if `act_quantizer` is present).

    Pre-conditions:
        - `export_blob` was produced by `cobra.quantize.finalize.int_export.export_int_quant_state`.
        - `model` is a *float* Cobra VLM (e.g., loaded via `cobra.load`),
          not already heavily mutated by other quantization flows.

    Post-conditions:
        - `model` is mutated in-place:
            * Linear/Conv/MatMul modules replaced by Quant* wrappers
              according to the export config.
            * Quant* modules carry integer weights + quantization metadata.
        - The same model instance is returned for convenience.
    """
    if not isinstance(export_blob, Mapping):
        raise TypeError(
            f"apply_int_quant_state expected a Mapping export_blob, got {type(export_blob)!r}"
        )

    cfg_dict = export_blob.get("config", {})
    if not isinstance(cfg_dict, Mapping):
        overwatch.warning(
            "[WrapReplace] export_blob['config'] missing or not a mapping; "
            "falling back to default WrapPolicyConfig (all targets enabled).",
        )
        policy_cfg = WrapPolicyConfig()
    else:
        policy_cfg = _build_policy_cfg_from_export_config(cfg_dict)

    weights_blob = export_blob.get("weights", {})
    acts_blob = export_blob.get("activations", {})

    if not isinstance(weights_blob, Mapping):
        raise TypeError(
            f"apply_int_quant_state expected 'weights' to be a mapping, got {type(weights_blob)!r}"
        )
    if not isinstance(acts_blob, Mapping):
        raise TypeError(
            f"apply_int_quant_state expected 'activations' to be a mapping, got {type(acts_blob)!r}"
        )

    overwatch.info(
        "[WrapReplace] Applying integer export: "
        f"{len(weights_blob)} weight entries, {len(acts_blob)} activation entries.",
        extra={
            "num_weight_entries": len(weights_blob),
            "num_activation_entries": len(acts_blob),
        },
    )

    # ------------------------------------------------------------------
    # 1) Wrap float model according to export config
    # ------------------------------------------------------------------
    # We rely on the same target vocabulary as IntExportConfig:
    #   - include_vision_dino / include_vision_siglip / include_llm / include_projector
    #
    # wrap_model_for_quantization internally uses:
    #   - build_wrap_registry + apply_wrap_registry
    wrap_model_for_quantization(
        model,
        policy_cfg=policy_cfg,
        manifest=None,
        default_params=None,
        prefix="",
    )

    # After wrapping, rebuild name → module map
    name_to_module: Dict[str, nn.Module] = dict(model.named_modules())

    # ------------------------------------------------------------------
    # 2) Install integer weights
    # ------------------------------------------------------------------
    installed_weights = 0
    skipped_weights = 0

    for full_name, w_record in weights_blob.items():
        # export_int_quant_state 使用 key: "<module_path>.weight"
        # 這裡我們拆出 module_path 與 param_name，以便找回對應 module。
        if not isinstance(full_name, str) or "." not in full_name:
            overwatch.warning(
                "[WrapReplace] Malformed weight key in export blob: %r",
                full_name,
            )
            skipped_weights += 1
            continue

        module_path, param_name = full_name.rsplit(".", 1)
        if param_name != "weight":
            # 目前僅支援 ".weight"，其他 parameter 先直接略過。
            overwatch.debug(
                "[WrapReplace] Skipping non-weight parameter in export blob: %r",
                full_name,
            )
            skipped_weights += 1
            continue

        module = name_to_module.get(module_path, None)
        if module is None:
            msg = (
                f"[WrapReplace] No module found at path {module_path!r} "
                f"for exported weight key {full_name!r}; skipping."
            )
            if strict:
                overwatch.error(msg)
                raise KeyError(msg)
            else:
                overwatch.warning(msg)
                skipped_weights += 1
                continue

        if not hasattr(module, "weight"):
            msg = (
                f"[WrapReplace] Module {module_path!r} has no 'weight' attribute "
                f"but weight entry exists in export blob; skipping."
            )
            if strict:
                overwatch.error(msg)
                raise AttributeError(msg)
            else:
                overwatch.warning(msg)
                skipped_weights += 1
                continue

        weight_ref: torch.Tensor = getattr(module, "weight")
        if not isinstance(weight_ref, torch.Tensor):
            msg = (
                f"[WrapReplace] Module {module_path!r}.weight is not a Tensor "
                f"(type={type(weight_ref)!r}); skipping."
            )
            if strict:
                overwatch.error(msg)
                raise TypeError(msg)
            else:
                overwatch.warning(msg)
                skipped_weights += 1
                continue

        if not isinstance(w_record, Mapping):
            msg = (
                f"[WrapReplace] Weight record for {full_name!r} is not a mapping; "
                f"got {type(w_record)!r}."
            )
            if strict:
                overwatch.error(msg)
                raise TypeError(msg)
            else:
                overwatch.warning(msg)
                skipped_weights += 1
                continue

        # 取出 integer weight / scale / zero_point
        int_weight = w_record.get("int_weight", None)
        scale = w_record.get("scale", None)
        zero_point = w_record.get("zero_point", None)

        if not isinstance(int_weight, torch.Tensor):
            msg = (
                f"[WrapReplace] Weight record for {full_name!r} missing 'int_weight' Tensor."
            )
            if strict:
                overwatch.error(msg)
                raise ValueError(msg)
            else:
                overwatch.warning(msg)
                skipped_weights += 1
                continue

        # Align devices
        int_weight = _tensor_like_ref(int_weight, weight_ref)
        if isinstance(scale, torch.Tensor):
            scale = _tensor_like_ref(scale, weight_ref)
        if isinstance(zero_point, torch.Tensor):
            zero_point = _tensor_like_ref(zero_point, weight_ref)

        # 安裝到 module 上：
        # 1) 通用字段：weight_int / weight_scale / weight_zero_point
        # 2) 若存在 weight_quantizer，則同步其 scale / round_zero_point
        setattr(module, "weight_int", int_weight)
        setattr(module, "weight_scale", scale)
        setattr(module, "weight_zero_point", zero_point)

        w_q = getattr(module, "weight_quantizer", None)
        # 僅在 quantizer 存在時才同步參數，避免 NoneType 問題
        if w_q is not None and isinstance(w_q, nn.Module):
            if isinstance(scale, torch.Tensor):
                setattr(w_q, "scale", scale)
            if isinstance(zero_point, torch.Tensor):
                setattr(w_q, "round_zero_point", zero_point)


        installed_weights += 1

        overwatch.debug(
            "[WrapReplace] Installed integer weight for %r (module=%r, shape=%s)",
            full_name,
            module_path,
            tuple(int_weight.shape),
            extra={
                "module_path": module_path,
                "param_name": param_name,
                "weight_shape": tuple(int_weight.shape),
            },
        )

    overwatch.info(
        "[WrapReplace] Finished installing integer weights: "
        f"installed={installed_weights}, skipped={skipped_weights}.",
        extra={
            "installed_weights": installed_weights,
            "skipped_weights": skipped_weights,
        },
    )

    # ------------------------------------------------------------------
    # 3) Install activation quantizer parameters
    # ------------------------------------------------------------------
    installed_acts = 0
    skipped_acts = 0

    for module_path, a_record in acts_blob.items():
        module = name_to_module.get(module_path, None)
        if module is None:
            msg = (
                f"[WrapReplace] No module found at path {module_path!r} "
                f"for exported activation entry; skipping."
            )
            if strict:
                overwatch.error(msg)
                raise KeyError(msg)
            else:
                overwatch.warning(msg)
                skipped_acts += 1
                continue

        act_q = getattr(module, "act_quantizer", None)
        if act_q is None:
            # 某些模組可能只有 weight quantizer，沒有 act_quantizer。
            overwatch.debug(
                "[WrapReplace] Module %r has no 'act_quantizer'; "
                "activation entry will be ignored.",
                module_path,
            )
            skipped_acts += 1
            continue

        if not isinstance(a_record, Mapping):
            msg = (
                f"[WrapReplace] Activation record for {module_path!r} is not a mapping; "
                f"got {type(a_record)!r}."
            )
            if strict:
                overwatch.error(msg)
                raise TypeError(msg)
            else:
                overwatch.warning(msg)
                skipped_acts += 1
                continue

        scale = a_record.get("scale", None)
        zp = a_record.get("zero_point", None)

        # act_quantizer 本身未必有參考 tensor，這裡直接沿用原本裝在 CPU 的 scale/zp。
        if isinstance(scale, torch.Tensor):
            setattr(act_q, "scale", scale)
        else:
            overwatch.debug(
                "[WrapReplace] Activation record for %r has no Tensor 'scale'; skipping.",
                module_path,
            )
            skipped_acts += 1
            continue

        if isinstance(zp, torch.Tensor):
            setattr(act_q, "round_zero_point", zp)
        else:
            # 允許 zero_point 為 None（symmetric quantization）
            setattr(act_q, "round_zero_point", None)

        installed_acts += 1

        overwatch.debug(
            "[WrapReplace] Installed activation quantizer params for module %r",
            module_path,
            extra={"module_path": module_path},
        )

    overwatch.info(
        "[WrapReplace] Finished installing activation quantizer params: "
        f"installed={installed_acts}, skipped={skipped_acts}.",
        extra={
            "installed_activations": installed_acts,
            "skipped_activations": skipped_acts,
        },
    )

    overwatch.info(
        "[WrapReplace] apply_int_quant_state() completed.",
        extra={
            "num_weight_entries": len(weights_blob),
            "num_activation_entries": len(acts_blob),
            "installed_weights": installed_weights,
            "installed_activations": installed_acts,
        },
    )

    return model
