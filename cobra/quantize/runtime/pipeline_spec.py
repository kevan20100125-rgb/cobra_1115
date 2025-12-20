# cobra/quantize/runtime/pipeline_spec.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch.nn as nn


# -----------------------------------------------------------------------------
# Canonical targets used by Cobra_1115 PTQ pipeline
# -----------------------------------------------------------------------------
CANONICAL_TARGETS: Tuple[str, ...] = (
    "vision.dino",
    "vision.siglip",
    "llm",
    "projector",
    "fusion",
)

# Module path on CobraVLM for point-B fusion stage
FUSION_STAGE_MODULE_PATH: str = "fusion_stage"

# -----------------------------------------------------------------------------
# Environment variables used by runtime loader
# -----------------------------------------------------------------------------
ENV_PROJECTOR_ROTATION_MODE: str = "COBRA_PROJECTOR_ROTATION_MODE"
ENV_FUSION_ROTATION_MODE: str = "COBRA_FUSION_ROTATION_MODE"
ENV_FUSION_ROTATION_ABSORB: str = "COBRA_FUSION_ROTATION_ABSORB"
ENV_DEBUG_FLOW: str = "COBRA_DEBUG_FLOW"


# -----------------------------------------------------------------------------
# LLM x_t-only activation quantization allowlist (Milestone 5)
#   - Keep activation quant ONLY at x_t entry points in each LLM block.
#   - The exact substrings reflect this repo’s LLM module naming conventions.
# -----------------------------------------------------------------------------
LLM_XT_ACT_QUANT_ALLOW_SUBSTRINGS: Tuple[str, ...] = (
    ".mixer.in_proj",
    ".mixer.in_proj_states",
    ".mixer.in_proj_gates",
    ".mlp.up_proj",
    ".mlp.fc1",
)


@dataclass(frozen=True)
class PipelineHookSpec:
    """
    A minimal spec object to group key hook locations.
    This is intentionally small to avoid making runtime code rigid.
    """
    fusion_stage_path: str = FUSION_STAGE_MODULE_PATH
    canonical_targets: Tuple[str, ...] = CANONICAL_TARGETS
    llm_xt_allow_substrings: Tuple[str, ...] = LLM_XT_ACT_QUANT_ALLOW_SUBSTRINGS


PIPELINE_SPEC = PipelineHookSpec()


def add_fusion_stage_to_target_map(
    *,
    model: nn.Module,
    enabled_targets: List[str],
    target_to_modules: Dict[str, List[str]],
) -> None:
    """
    Ensure Point-B fusion stage is included in pct collection/apply when target "fusion" is enabled.

    This is the canonical (single) place to define how "fusion" maps to actual model modules.
    """
    if "fusion" not in enabled_targets:
        return
    if not hasattr(model, FUSION_STAGE_MODULE_PATH):
        return

    target_to_modules.setdefault("fusion", [])
    if FUSION_STAGE_MODULE_PATH not in target_to_modules["fusion"]:
        target_to_modules["fusion"].append(FUSION_STAGE_MODULE_PATH)
    target_to_modules["fusion"] = sorted(target_to_modules["fusion"])
