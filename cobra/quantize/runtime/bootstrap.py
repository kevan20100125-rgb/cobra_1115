from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Sequence, Set

from cobra.quantize.resolver import (
    resolve_mixer_rotation,
    resolve_pct_hi_lo_path,
    resolve_quant_output_dir,
    resolve_runtime_model_id,
)
from cobra.quantize.runtime.act_policy import resolve_llm_act_mode_from_env
from cobra.quantize.runtime.config import QuantRuntimeConfig, resolve_runtime_request
from cobra.quantize.runtime.types import ResolvedRuntimeInputs
from cobra.quantize.targets import CANONICAL_TARGETS, normalize_target
from cobra.quantize.wrap.policy import WrapPolicyConfig


def _normalize_enabled_targets(enabled_targets: Optional[Sequence[str]]) -> Set[str]:
    """
    Phase 3-compatible target normalization.

    Your repo's stable API is normalize_target(...), not normalize_targets(...).
    """
    if enabled_targets is None:
        return set(CANONICAL_TARGETS)

    out: Set[str] = set()
    for item in enabled_targets:
        if item is None:
            continue
        out.add(normalize_target(str(item)))
    return out


def resolve_runtime_inputs(
    *,
    bits: Optional[str],
    pct_hi_lo_path,
    enabled_targets=None,
    run_dir=None,
    output_dir=None,
    model_id_or_path: Optional[str] = None,
    backend: str = "fake",
) -> ResolvedRuntimeInputs:
    enabled_targets_set = _normalize_enabled_targets(enabled_targets)

    run_dir_path = Path(run_dir) if run_dir is not None else None

    resolved_output_dir = resolve_quant_output_dir(
        run_dir=run_dir_path,
        output_dir=output_dir,
        pct_hi_lo_path=pct_hi_lo_path,
    )

    resolved_pct_hi_lo_path = resolve_pct_hi_lo_path(
        run_dir=run_dir_path,
        output_dir=resolved_output_dir,
        explicit_path=pct_hi_lo_path,
        env=os.environ,
    )

    rotation_spec = resolve_mixer_rotation(
        run_dir=run_dir_path,
        output_dir=resolved_output_dir,
        env=os.environ,
    )

    raw_model_id = resolve_runtime_model_id(
        model_id_or_path=model_id_or_path,
        env=os.environ,
    )

    req = resolve_runtime_request(
        raw_model_id=raw_model_id,
        env_bits=bits,
        env_backend=backend,
    )

    quant_cfg = QuantRuntimeConfig.from_bits_backend(
        bits=req.bits,
        backend=req.backend,
        enable_vision_dino=("vision.dino" in enabled_targets_set),
        enable_vision_siglip=("vision.siglip" in enabled_targets_set),
        enable_llm=("llm" in enabled_targets_set),
        enable_projector=("projector" in enabled_targets_set),
        enable_act_quant=(resolved_pct_hi_lo_path is not None),
        config_name="runtime::load_quantized_cobra_vlm",
        strict_bits=False,
    )

    wrap_policy_cfg = WrapPolicyConfig(
        enable_vision_dino=("vision.dino" in enabled_targets_set),
        enable_vision_siglip=("vision.siglip" in enabled_targets_set),
        enable_llm=("llm" in enabled_targets_set),
        enable_projector=("projector" in enabled_targets_set),
        include_linear=True,
        include_conv=True,
    )

    llm_act_only = os.environ.get("COBRA_LLM_ACT_ONLY", "").strip().lower()
    llm_act_mode = resolve_llm_act_mode_from_env(os.environ)

    return ResolvedRuntimeInputs(
        raw_model_id=raw_model_id,
        runtime_request=req,
        quant_cfg=quant_cfg,
        enabled_targets=enabled_targets_set,
        llm_act_only=llm_act_only,
        llm_act_mode=llm_act_mode,
        run_dir=run_dir_path,
        output_dir=resolved_output_dir,
        pct_hi_lo_path=resolved_pct_hi_lo_path,
        rotation_spec=rotation_spec,
        wrap_policy_cfg=wrap_policy_cfg,
    )
