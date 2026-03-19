from __future__ import annotations

import os
from typing import Iterable, List, Optional, Sequence, Tuple

LLM_ACT_MODE_DEFAULT = "default"
LLM_ACT_MODE_OUT_PROJ_ONLY = "out_proj_only"
LLM_ACT_MODE_MAMBA_SENSITIVE = "mamba_sensitive"

_VALID_LLM_ACT_MODES = {
    LLM_ACT_MODE_DEFAULT,
    LLM_ACT_MODE_OUT_PROJ_ONLY,
    LLM_ACT_MODE_MAMBA_SENSITIVE,
}

LLM_MAMBA_SENSITIVE_SUFFIX_IN_PROJ = ".mixer.in_proj"
LLM_MAMBA_SENSITIVE_SUFFIX_X_PROJ = ".mixer.x_proj"
LLM_MAMBA_SENSITIVE_SUFFIX_DT_PROJ = ".mixer.dt_proj"
LLM_MAMBA_SENSITIVE_SUFFIX_OUT_PROJ = ".mixer.out_proj"

# All suffixes that the user-facing mamba_sensitive mode may refer to.
_ALL_MAMBA_SENSITIVE_SUFFIXES: Tuple[str, ...] = (
    LLM_MAMBA_SENSITIVE_SUFFIX_IN_PROJ,
    LLM_MAMBA_SENSITIVE_SUFFIX_X_PROJ,
    LLM_MAMBA_SENSITIVE_SUFFIX_DT_PROJ,
    LLM_MAMBA_SENSITIVE_SUFFIX_OUT_PROJ,
)

# IMPORTANT:
# For path-aware calibration in this Cobra snapshot, only suffixes that are
# reliably observable through module-level hooks should be considered
# "effective" targets. in_proj / dt_proj can exist as named modules but may be
# consumed via weight-level / functional paths instead of module __call__,
# which makes forward_pre_hook collection incomplete by construction.
#
# Therefore, the stable hook-visible subset is intentionally limited to:
#   - x_proj
#   - out_proj
#
# This keeps strict completeness gating valid and prevents emitting partial
# calibration artifacts under llm_act_mode='mamba_sensitive'.
_HOOK_VISIBLE_MAMBA_SENSITIVE_SUFFIXES: Tuple[str, ...] = (
    LLM_MAMBA_SENSITIVE_SUFFIX_X_PROJ,
    LLM_MAMBA_SENSITIVE_SUFFIX_OUT_PROJ,
)


def normalize_llm_act_mode(
    value: Optional[str],
    *,
    fallback_llm_act_only: Optional[str] = None,
) -> str:
    """
    Normalize runtime/calibration LLM activation policy mode.

    Priority:
      1) explicit value via COBRA_LLM_ACT_MODE
      2) backward-compat alias via COBRA_LLM_ACT_ONLY=out_proj
      3) default
    """
    raw = (value or "").strip().lower()
    if raw in _VALID_LLM_ACT_MODES:
        return raw

    alias = (fallback_llm_act_only or "").strip().lower()
    if alias == "out_proj":
        return LLM_ACT_MODE_OUT_PROJ_ONLY

    return LLM_ACT_MODE_DEFAULT


def resolve_llm_act_mode_from_env(env: Optional[dict] = None) -> str:
    source = os.environ if env is None else env
    return normalize_llm_act_mode(
        source.get("COBRA_LLM_ACT_MODE", ""),
        fallback_llm_act_only=source.get("COBRA_LLM_ACT_ONLY", ""),
    )


def _dedupe_preserve_order(values: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    seen = set()
    for v in values:
        s = str(v or "").strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return tuple(out)


def _normalize_mamba_sensitive_suffixes(
    suffixes: Optional[Sequence[str]],
) -> Tuple[str, ...]:
    """
    Normalize a suffix selection while preserving canonical ordering.
    Unknown suffixes are ignored conservatively.
    """
    if suffixes is None:
        return ()

    requested = set(_dedupe_preserve_order(suffixes))
    return tuple(s for s in _ALL_MAMBA_SENSITIVE_SUFFIXES if s in requested)


def resolve_effective_mamba_sensitive_suffixes(
    *,
    requested_suffixes: Optional[Sequence[str]] = None,
    hook_visible_only: bool = True,
) -> Tuple[str, ...]:
    """
    Resolve the effective suffixes for mamba_sensitive mode.

    Behavior:
      - If `requested_suffixes` is omitted, we treat all user-facing
        mamba_sensitive suffixes as requested.
      - If `hook_visible_only=True`, the result is intersected with the stable
        hook-visible subset for this snapshot.
      - Canonical ordering is preserved.
    """
    if requested_suffixes is None:
        requested = _ALL_MAMBA_SENSITIVE_SUFFIXES
    else:
        requested = _normalize_mamba_sensitive_suffixes(requested_suffixes)

    allowed = (
        _HOOK_VISIBLE_MAMBA_SENSITIVE_SUFFIXES
        if hook_visible_only
        else _ALL_MAMBA_SENSITIVE_SUFFIXES
    )
    allowed_set = set(allowed)

    return tuple(s for s in requested if s in allowed_set)


def resolve_effective_mamba_sensitive_suffixes_from_env() -> Tuple[str, ...]:
    """
    Resolve env-gated mamba_sensitive suffixes and intersect with the stable
    hook-visible subset. This keeps runtime behavior aligned with calibration.

    Local import avoids a top-level dependency cycle.
    """
    from cobra.quantize.resolver.artifact_resolver import (
        resolve_mamba_sensitive_projection_gates,
    )

    gates = resolve_mamba_sensitive_projection_gates()
    return resolve_effective_mamba_sensitive_suffixes(
        requested_suffixes=gates.enabled_suffixes,
        hook_visible_only=True,
    )


def is_llm_out_proj_path(module_path: str) -> bool:
    path = str(module_path or "")
    return path.startswith("llm_backbone.llm.") and path.endswith(".mixer.out_proj")


def is_llm_mamba_sensitive_path(
    module_path: str,
    *,
    enabled_suffixes: Optional[Sequence[str]] = None,
) -> bool:
    path = str(module_path or "")
    if not path.startswith("llm_backbone.llm."):
        return False

    suffixes = (
        resolve_effective_mamba_sensitive_suffixes_from_env()
        if enabled_suffixes is None
        else resolve_effective_mamba_sensitive_suffixes(
            requested_suffixes=enabled_suffixes,
            hook_visible_only=True,
        )
    )
    return any(path.endswith(sfx) for sfx in suffixes)


def filter_llm_module_paths(
    module_paths: Sequence[str],
    *,
    mode: str,
    mamba_sensitive_suffixes: Optional[Sequence[str]] = None,
) -> List[str]:
    """
    Filter already-discovered LLM module paths according to activation mode.

    This function is intentionally path-based and conservative. It does not try
    to infer graph-internal tensors that are not represented in named_modules().
    """
    normalized_mode = normalize_llm_act_mode(mode)
    paths = [str(p) for p in module_paths if p]

    if normalized_mode == LLM_ACT_MODE_DEFAULT:
        return list(paths)

    if normalized_mode == LLM_ACT_MODE_OUT_PROJ_ONLY:
        return [p for p in paths if is_llm_out_proj_path(p)]

    if normalized_mode == LLM_ACT_MODE_MAMBA_SENSITIVE:
        return [
            p
            for p in paths
            if is_llm_mamba_sensitive_path(
                p,
                enabled_suffixes=mamba_sensitive_suffixes,
            )
        ]

    return list(paths)


def filter_target_module_map_for_llm_mode(
    target_to_module_paths: dict[str, Sequence[str]],
    *,
    mode: str,
    mamba_sensitive_suffixes: Optional[Sequence[str]] = None,
) -> dict[str, List[str]]:
    """
    Apply LLM-only module filtering while preserving other targets unchanged.
    """
    out: dict[str, List[str]] = {}
    normalized_mode = normalize_llm_act_mode(mode)

    for target, paths in target_to_module_paths.items():
        if target == "llm":
            out[target] = filter_llm_module_paths(
                paths,
                mode=normalized_mode,
                mamba_sensitive_suffixes=mamba_sensitive_suffixes,
            )
        else:
            out[target] = [str(p) for p in paths if p]
    return out


def should_enable_llm_module_act_quant(module_path: str, *, mode: str) -> bool:
    """
    Runtime-side predicate used later by loader/helpers to keep or disable
    activation quant per LLM Quant* module.

    For mamba_sensitive mode, runtime behavior intentionally shares the same
    effective suffix resolution as calibration:
      env gates
        -> requested suffixes
        -> intersect stable hook-visible subset
        -> final keep predicate
    """
    normalized_mode = normalize_llm_act_mode(mode)

    if normalized_mode == LLM_ACT_MODE_DEFAULT:
        return str(module_path or "").startswith("llm_backbone.llm.")

    if normalized_mode == LLM_ACT_MODE_OUT_PROJ_ONLY:
        return is_llm_out_proj_path(module_path)

    if normalized_mode == LLM_ACT_MODE_MAMBA_SENSITIVE:
        return is_llm_mamba_sensitive_path(module_path)

    return str(module_path or "").startswith("llm_backbone.llm.")


def summarize_llm_module_paths(module_paths: Iterable[str]) -> dict[str, int]:
    """
    Small helper for logging/reporting/debugging.
    """
    total = 0
    out_proj = 0
    x_proj = 0
    in_proj = 0
    dt_proj = 0
    other = 0

    for raw_path in module_paths:
        path = str(raw_path or "")
        if not path:
            continue
        total += 1
        if path.endswith(".mixer.out_proj"):
            out_proj += 1
        elif path.endswith(".mixer.x_proj"):
            x_proj += 1
        elif path.endswith(".mixer.in_proj"):
            in_proj += 1
        elif path.endswith(".mixer.dt_proj"):
            dt_proj += 1
        else:
            other += 1

    return {
        "total": total,
        "out_proj": out_proj,
        "x_proj": x_proj,
        "in_proj": in_proj,
        "dt_proj": dt_proj,
        "other": other,
    }
