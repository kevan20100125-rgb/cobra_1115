from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple, Union

PathLike = Union[str, Path]

# -----------------------------------------------------------------------------
# Official env contract 
# -----------------------------------------------------------------------------
ENV_MODEL_ID = "COBRA_MODEL_ID_OR_PATH"
ENV_MODEL_BASE_ID = "COBRA_MODEL_BASE_ID"

ENV_PCT_HILO = "PCT_HI_LO_PATH"
LEGACY_ENV_PCT_HILO = ("PCT_HI_LO",)

ENV_ACT_KLT_IN = "ACT_KLT_OUTPROJ_IN"
LEGACY_ENV_ACT_KLT_IN = ("ACT_KLT_OUT_IN", "ACT_KLT_OUT")

ENV_ACT_KLT_OUT = "ACT_KLT_OUTPROJ_OUT"
LEGACY_ENV_ACT_KLT_OUT = ("ACT_KLT_OUT_OUT",)

ENV_MIXER_HADAMARD = "COBRA_LLM_MIXER_HADAMARD"
ENV_MIXER_BLOCK = "COBRA_LLM_MIXER_BLOCK"
ENV_MIXER_TARGETS = "COBRA_LLM_MIXER_TARGETS"
ENV_MIXER_MAX_LAYERS = "COBRA_LLM_MIXER_MAX_LAYERS"
ENV_MIXER_DRY_RUN = "COBRA_LLM_MIXER_DRY_RUN"
ENV_MIXER_IN_TRANSFORM = "COBRA_LLM_MIXER_IN_TRANSFORM"
ENV_MIXER_OUT_TRANSFORM = "COBRA_LLM_MIXER_OUT_TRANSFORM"
ENV_MIXER_ACT_KLT = "COBRA_LLM_MIXER_ACT_KLT"  # legacy aggregate gate
ENV_MIXER_IN_ACT_KLT = "COBRA_LLM_MIXER_IN_ACT_KLT"
ENV_MIXER_OUT_ACT_KLT = "COBRA_LLM_MIXER_OUT_ACT_KLT"
ENV_MIXER_ACT_KLT_STRICT = "COBRA_LLM_MIXER_ACT_KLT_STRICT"
ENV_MIXER_KLT_DTYPE = "COBRA_LLM_MIXER_KLT_DTYPE"

ENV_LLM_ACT_MODE = "COBRA_LLM_ACT_MODE"
ENV_LLM_ACT_ONLY = "COBRA_LLM_ACT_ONLY"
ENV_DISABLE_MAMBA_FAST_PATH = "COBRA_DISABLE_MAMBA_FAST_PATH"

ENV_LLM_MAMBA_SENSITIVE_IN_PROJ = "COBRA_LLM_MAMBA_SENSITIVE_IN_PROJ"
ENV_LLM_MAMBA_SENSITIVE_X_PROJ = "COBRA_LLM_MAMBA_SENSITIVE_X_PROJ"
ENV_LLM_MAMBA_SENSITIVE_DT_PROJ = "COBRA_LLM_MAMBA_SENSITIVE_DT_PROJ"
ENV_LLM_MAMBA_SENSITIVE_OUT_PROJ = "COBRA_LLM_MAMBA_SENSITIVE_OUT_PROJ"


def _clean_str(v: object) -> str:
    return str(v).strip() if v is not None else ""


def _env_str(env: Mapping[str, str], key: str) -> str:
    return _clean_str(env.get(key))


def _env_bool(env: Mapping[str, str], key: str, default: bool = False) -> bool:
    raw = _env_str(env, key).lower()
    if raw in ("1", "true", "yes", "y", "on", "enable", "enabled", "hadamard"):
        return True
    if raw in ("0", "false", "no", "n", "off", "disable", "disabled", ""):
        return False
    return default


def _env_int(env: Mapping[str, str], key: str, default: Optional[int] = None) -> Optional[int]:
    raw = _env_str(env, key)
    if not raw:
        return default
    return int(raw)


def _as_path(v: Optional[PathLike]) -> Optional[Path]:
    if v is None:
        return None
    s = _clean_str(v)
    if not s:
        return None
    return Path(s)


def _warn_legacy_env(legacy_key: str, official_key: str) -> None:
    warnings.warn(
        f"[artifact_resolver] Legacy env {legacy_key!r} is deprecated; "
        f"use {official_key!r} instead.",
        FutureWarning,
        stacklevel=3,
    )


def _resolve_env_path(
    *,
    env: Mapping[str, str],
    official_key: str,
    legacy_keys: Sequence[str] = (),
) -> Optional[Path]:
    official = _env_str(env, official_key)
    if official:
        return Path(official)

    for key in legacy_keys:
        raw = _env_str(env, key)
        if raw:
            _warn_legacy_env(key, official_key)
            return Path(raw)
    return None


def resolve_runtime_model_id(
    *,
    model_id_or_path: Optional[str] = None,
    env: Mapping[str, str] = os.environ,
) -> str:
    explicit = _clean_str(model_id_or_path)
    if explicit:
        return explicit

    env_model = _env_str(env, ENV_MODEL_ID)
    if env_model:
        return env_model

    env_base = _env_str(env, ENV_MODEL_BASE_ID)
    if env_base:
        return env_base

    return "cobra+3b"


def resolve_quant_output_dir(
    *,
    run_dir: Optional[PathLike] = None,
    output_dir: Optional[PathLike] = None,
    pct_hi_lo_path: Optional[PathLike] = None,
) -> Path:
    explicit = _as_path(output_dir)
    if explicit is not None:
        return explicit

    run_dir_p = _as_path(run_dir)
    if run_dir_p is not None:
        return run_dir_p / "outputs" / "quantize"

    pct_p = _as_path(pct_hi_lo_path)
    if pct_p is not None:
        return pct_p.parent

    return Path("outputs") / "quantize"


def resolve_pct_hi_lo_path(
    *,
    run_dir: Optional[PathLike] = None,
    output_dir: Optional[PathLike] = None,
    explicit_path: Optional[PathLike] = None,
    env: Mapping[str, str] = os.environ,
) -> Optional[Path]:
    explicit = _as_path(explicit_path)
    if explicit is not None:
        return explicit

    env_path = _resolve_env_path(
        env=env,
        official_key=ENV_PCT_HILO,
        legacy_keys=LEGACY_ENV_PCT_HILO,
    )
    if env_path is not None:
        return env_path

    default_path = resolve_quant_output_dir(
        run_dir=run_dir,
        output_dir=output_dir,
        pct_hi_lo_path=None,
    ) / "pct_hi_lo.pt"

    return default_path if default_path.exists() else None


def resolve_mixer_act_klt_in_path(
    *,
    block_size: int,
    run_dir: Optional[PathLike] = None,
    output_dir: Optional[PathLike] = None,
    explicit_path: Optional[PathLike] = None,
    env: Mapping[str, str] = os.environ,
) -> Path:
    explicit = _as_path(explicit_path)
    if explicit is not None:
        return explicit

    env_path = _resolve_env_path(
        env=env,
        official_key=ENV_ACT_KLT_IN,
        legacy_keys=LEGACY_ENV_ACT_KLT_IN,
    )
    if env_path is not None:
        return env_path

    return (
        resolve_quant_output_dir(run_dir=run_dir, output_dir=output_dir)
        / f"act_klt_outproj_in_bs{int(block_size)}"
        / "act_klt_outproj_in.pt"
    )


def resolve_mixer_act_klt_out_path(
    *,
    block_size: int,
    run_dir: Optional[PathLike] = None,
    output_dir: Optional[PathLike] = None,
    explicit_path: Optional[PathLike] = None,
    env: Mapping[str, str] = os.environ,
) -> Path:
    explicit = _as_path(explicit_path)
    if explicit is not None:
        return explicit

    env_path = _resolve_env_path(
        env=env,
        official_key=ENV_ACT_KLT_OUT,
        legacy_keys=LEGACY_ENV_ACT_KLT_OUT,
    )
    if env_path is not None:
        return env_path

    return (
        resolve_quant_output_dir(run_dir=run_dir, output_dir=output_dir)
        / f"act_klt_outproj_out_bs{int(block_size)}"
        / "act_klt_outproj_out.pt"
    )


@dataclass(frozen=True)
class MixerRotationResolution:
    enabled: bool
    block_size: int
    targets: Tuple[str, ...]
    max_layers: Optional[int]
    dry_run: bool

    in_transform_enabled: bool
    out_transform_enabled: bool

    act_klt_enabled: bool
    in_act_klt_enabled: bool
    out_act_klt_enabled: bool
    act_klt_strict: bool

    act_klt_in_path: Optional[Path]
    act_klt_out_path: Optional[Path]
    klt_dtype: str


@dataclass(frozen=True)
class MambaSensitiveProjectionResolution:
    enable_in_proj: bool
    enable_x_proj: bool
    enable_dt_proj: bool
    enable_out_proj: bool

    @property
    def enabled_suffixes(self) -> Tuple[str, ...]:
        suffixes = []
        if self.enable_in_proj:
            suffixes.append(".mixer.in_proj")
        if self.enable_x_proj:
            suffixes.append(".mixer.x_proj")
        if self.enable_dt_proj:
            suffixes.append(".mixer.dt_proj")
        if self.enable_out_proj:
            suffixes.append(".mixer.out_proj")
        return tuple(suffixes)

    def as_dict(self) -> dict:
        return {
            "enable_in_proj": bool(self.enable_in_proj),
            "enable_x_proj": bool(self.enable_x_proj),
            "enable_dt_proj": bool(self.enable_dt_proj),
            "enable_out_proj": bool(self.enable_out_proj),
            "enabled_suffixes": list(self.enabled_suffixes),
        }


def resolve_mamba_sensitive_projection_gates(
    *,
    env: Mapping[str, str] = os.environ,
) -> MambaSensitiveProjectionResolution:
    """
    Resolve sub-path gates for mamba_sensitive mode.

    Default:
      - in_proj   : ON
      - x_proj    : ON
      - dt_proj   : ON
      - out_proj  : ON
    """
    return MambaSensitiveProjectionResolution(
        enable_in_proj=_env_bool(env, ENV_LLM_MAMBA_SENSITIVE_IN_PROJ, default=True),
        enable_x_proj=_env_bool(env, ENV_LLM_MAMBA_SENSITIVE_X_PROJ, default=True),
        enable_dt_proj=_env_bool(env, ENV_LLM_MAMBA_SENSITIVE_DT_PROJ, default=True),
        enable_out_proj=_env_bool(env, ENV_LLM_MAMBA_SENSITIVE_OUT_PROJ, default=True),
    )


def resolve_mixer_rotation(
    *,
    run_dir: Optional[PathLike] = None,
    output_dir: Optional[PathLike] = None,
    env: Mapping[str, str] = os.environ,
) -> MixerRotationResolution:
    legacy_hadamard_enabled = _env_bool(env, ENV_MIXER_HADAMARD, default=False)
    block_size = _env_int(env, ENV_MIXER_BLOCK, default=512) or 512

    targets_raw = _env_str(env, ENV_MIXER_TARGETS)
    if targets_raw:
        targets = tuple(t.strip() for t in targets_raw.split(",") if t.strip())
    else:
        targets = ("out_proj",)

    max_layers = _env_int(env, ENV_MIXER_MAX_LAYERS, default=None)
    dry_run = _env_bool(env, ENV_MIXER_DRY_RUN, default=False)
    act_klt_strict = _env_bool(env, ENV_MIXER_ACT_KLT_STRICT, default=False)
    klt_dtype = _env_str(env, ENV_MIXER_KLT_DTYPE).lower() or "fp32"

    # Side transform gates
    in_transform_enabled = _env_bool(
        env,
        ENV_MIXER_IN_TRANSFORM,
        default=legacy_hadamard_enabled,
    )
    out_transform_enabled = _env_bool(
        env,
        ENV_MIXER_OUT_TRANSFORM,
        default=False,
    )

    # Legacy aggregate KLT gate for backward compatibility
    legacy_act_klt_enabled = _env_bool(env, ENV_MIXER_ACT_KLT, default=False)

    in_act_klt_enabled = _env_bool(
        env,
        ENV_MIXER_IN_ACT_KLT,
        default=(legacy_act_klt_enabled and in_transform_enabled),
    )
    out_act_klt_enabled = _env_bool(
        env,
        ENV_MIXER_OUT_ACT_KLT,
        default=(legacy_act_klt_enabled and out_transform_enabled),
    )

    enabled = bool(in_transform_enabled or out_transform_enabled)
    act_klt_enabled = bool(in_act_klt_enabled or out_act_klt_enabled)

    act_klt_in_path = None
    act_klt_out_path = None

    if in_act_klt_enabled:
        act_klt_in_path = resolve_mixer_act_klt_in_path(
            block_size=block_size,
            run_dir=run_dir,
            output_dir=output_dir,
            env=env,
        )

    if out_act_klt_enabled:
        act_klt_out_path = resolve_mixer_act_klt_out_path(
            block_size=block_size,
            run_dir=run_dir,
            output_dir=output_dir,
            env=env,
        )

    return MixerRotationResolution(
        enabled=enabled,
        block_size=block_size,
        targets=targets,
        max_layers=max_layers,
        dry_run=dry_run,
        in_transform_enabled=in_transform_enabled,
        out_transform_enabled=out_transform_enabled,
        act_klt_enabled=act_klt_enabled,
        in_act_klt_enabled=in_act_klt_enabled,
        out_act_klt_enabled=out_act_klt_enabled,
        act_klt_strict=act_klt_strict,
        act_klt_in_path=act_klt_in_path,
        act_klt_out_path=act_klt_out_path,
        klt_dtype=klt_dtype,
    )


def collect_resolver_env_snapshot(env: Mapping[str, str] = os.environ) -> dict:
    keys = [
        "BITS",
        "BACKEND",
        ENV_MODEL_ID,
        ENV_MODEL_BASE_ID,
        ENV_PCT_HILO,
        *LEGACY_ENV_PCT_HILO,
        ENV_LLM_ACT_ONLY,
        ENV_LLM_ACT_MODE,
        ENV_DISABLE_MAMBA_FAST_PATH,
        ENV_LLM_MAMBA_SENSITIVE_IN_PROJ,
        ENV_LLM_MAMBA_SENSITIVE_X_PROJ,
        ENV_LLM_MAMBA_SENSITIVE_DT_PROJ,
        ENV_LLM_MAMBA_SENSITIVE_OUT_PROJ,
        ENV_MIXER_HADAMARD,
        ENV_MIXER_BLOCK,
        ENV_MIXER_TARGETS,
        ENV_MIXER_MAX_LAYERS,
        ENV_MIXER_DRY_RUN,
        ENV_MIXER_IN_TRANSFORM,
        ENV_MIXER_OUT_TRANSFORM,
        ENV_MIXER_ACT_KLT,
        ENV_MIXER_IN_ACT_KLT,
        ENV_MIXER_OUT_ACT_KLT,
        ENV_MIXER_ACT_KLT_STRICT,
        ENV_MIXER_KLT_DTYPE,
        ENV_ACT_KLT_IN,
        *LEGACY_ENV_ACT_KLT_IN,
        ENV_ACT_KLT_OUT,
        *LEGACY_ENV_ACT_KLT_OUT,
    ]
    return {k: env.get(k) for k in keys}
