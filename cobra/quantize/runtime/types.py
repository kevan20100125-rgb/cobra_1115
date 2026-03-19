from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set

from cobra.quantize.resolver import MixerRotationResolution
from cobra.quantize.runtime.config import QuantRuntimeConfig, RuntimeRequestResolution
from cobra.quantize.wrap.policy import WrapPolicyConfig


@dataclass(frozen=True)
class ResolvedRuntimeInputs:
    raw_model_id: str
    runtime_request: RuntimeRequestResolution
    quant_cfg: QuantRuntimeConfig
    enabled_targets: Set[str]
    llm_act_only: str
    llm_act_mode: str
    run_dir: Optional[Path]
    output_dir: Path
    pct_hi_lo_path: Optional[Path]
    rotation_spec: MixerRotationResolution
    wrap_policy_cfg: WrapPolicyConfig


@dataclass(frozen=True)
class RuntimeWrapResult:
    registry: Any
    wrap_summary: Dict[str, Any]


@dataclass(frozen=True)
class RuntimeActivationResult:
    requested: bool
    enabled: bool
    summary: Optional[Dict[str, Dict[str, float]]]
    pct_hi_lo_path: Optional[Path]
    act_bits: Optional[int]
    error: Optional[str]


@dataclass(frozen=True)
class RuntimeBehaviorArtifacts:
    coverage_payload: Dict[str, Any]
    behavior_payload: Dict[str, Any]
    coverage_path: Optional[Path]
    behavior_path: Optional[Path]
