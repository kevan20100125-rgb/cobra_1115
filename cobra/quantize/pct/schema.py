# cobra/quantize/pct/schema.py

"""
Percentile clipping schema utilities.

Provides:
- normalize_target(name) -> canonical target name in {"vision.dino","vision.siglip","llm","projector"}
- normalize_stage(name)  -> alias of normalize_target for backwards-compat
- is_percentile_key(key) / parse_percentile_key(key) helpers
- compute_affine_params(x_min, x_max, bits, signed) -> (scale: float, zero_point: int)

Design notes:
- "Percentile keys" are those starting with 'p' (e.g., "p99.9", "p99_99").
- Targets are normalized to the canonical four-class vocabulary to keep collect/apply aligned.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Mapping, Tuple, TypedDict

import torch

from cobra.quantize.quantizer import UniformAffineQuantizer

# ---------------------------------------------------------------------------
# Target normalization
# ---------------------------------------------------------------------------

ALLOWED_TARGETS = {"fusion", "vision.dino", "vision.siglip", "llm", "projector"}

# Legacy / alias → canonical target mapping
LEGACY_TARGET_MAP: Dict[str, str] = {
    # Fusion stage (Point-B)
    "fusion": "fusion",
    "fusion_stage": "fusion",
    "fusionstage": "fusion",

    # Older naming patterns (from previous prototypes / scripts)
    "vision_backbone.dino": "vision.dino",
    "vision_backbone.siglip": "vision.siglip",
    "vision.dinov2": "vision.dino",
    "vision.siglip_vit": "vision.siglip",
    "llm_backbone": "llm",
    "lm_backbone": "llm",
    "language": "llm",
    "projector.out": "projector",
    "proj.out": "projector",
    "encoder.out": "projector",
}

def normalize_target(name: str) -> str:
    """
    Map various user / legacy target names to the canonical vocabulary.

    Canonical targets:
        - "fusion"
        - "vision.dino"
        - "vision.siglip"
        - "llm"
        - "projector"

    Raises:
        KeyError: if the name cannot be normalized.
    """
    raw = (name or "").strip()

    # tolerate "target::foo" style prefixes
    if "::" in raw:
        raw = raw.split("::", 1)[-1]

    # strip whitespace & lower for matching
    raw_stripped = raw.replace(" ", "")
    lowered = raw_stripped.lower()

    # exact canonical
    if raw_stripped in ALLOWED_TARGETS:
        return raw_stripped

    # exact legacy mapping
    if raw_stripped in LEGACY_TARGET_MAP:
        return LEGACY_TARGET_MAP[raw_stripped]
    if lowered in LEGACY_TARGET_MAP:
        return LEGACY_TARGET_MAP[lowered]

    # heuristic fallbacks based on substrings
    if "fusion" in lowered:
        return "fusion"
    if "dino" in lowered:
        return "vision.dino"
    if "siglip" in lowered:
        return "vision.siglip"
    if "projector" in lowered or lowered.endswith(".out"):
        return "projector"
    if "llm" in lowered or "gpt" in lowered or "language" in lowered:
        return "llm"

    raise KeyError(f"Unrecognized percentile target name: {name!r}")

def normalize_stage(name: str) -> str:
    """
    Alias kept for clarity with earlier drafts where we used "stage" terminology.
    """
    return normalize_target(name)


# ---------------------------------------------------------------------------
# Percentile key helpers
# ---------------------------------------------------------------------------

# Accept both dot and underscore decimals, e.g., p99.9 or p99_9 or p99_99.
_PCT_KEY_RE = re.compile(r"^p(?P<main>\d{1,3})(?:[._](?P<frac>\d+))?$")


def is_percentile_key(key: str) -> bool:
    """
    Return True if key looks like a percentile entry, e.g. "p99.9" or "p99_9".
    """
    return bool(_PCT_KEY_RE.match(key))


def parse_percentile_key(key: str) -> float:
    """
    Convert a percentile key like "p99.9" or "p99_9" into a float 99.9.

    Raises:
        ValueError: if the key does not match the percentile pattern.
    """
    m = _PCT_KEY_RE.match(key)
    if not m:
        raise ValueError(f"Key {key!r} is not a valid percentile key")

    main = int(m.group("main"))
    frac = m.group("frac")
    if frac:
        return float(f"{main}.{frac}")
    return float(main)


def iter_percentile_items(stats: Mapping[str, Any]) -> Iterable[Tuple[str, float]]:
    """
    Yield (key, value) pairs for all percentile-like keys in the given mapping.
    """
    for k, v in stats.items():
        if is_percentile_key(k):
            try:
                value = float(v)
            except (TypeError, ValueError):
                continue
            yield k, value


# ---------------------------------------------------------------------------
# Simple schema validation
# ---------------------------------------------------------------------------


class PercentileRecord(TypedDict, total=False):
    mode: str
    percent: float
    numel: int
    target: str
    module: str
    min: float
    max: float
    # plus dynamic percentile keys such as "p99.9", "p99_99", ...


def validate_record_schema(d: Mapping[str, Any]) -> None:
    """
    Lightweight validation for a single stats record produced by collect.py.

    Requirements:
      - has a 'target' and 'module' field (strings)
      - any 'p*' keys must be convertible to float
      - if present, 'min' and 'max' must be numeric
    """
    if "target" not in d:
        raise AssertionError("missing 'target' field in percentile record")
    if "module" not in d:
        raise AssertionError("missing 'module' field in percentile record")

    if not isinstance(d["target"], str):
        raise AssertionError("'target' field must be a str")
    if not isinstance(d["module"], str):
        raise AssertionError("'module' field must be a str")

    for k, v in d.items():
        if is_percentile_key(k):
            try:
                float(v)
            except (TypeError, ValueError) as exc:
                raise AssertionError(f"percentile key {k!r} must be numeric") from exc

    for bound in ("min", "max"):
        if bound in d and not isinstance(d[bound], (int, float)):
            raise AssertionError(f"{bound!r} must be numeric if present")


def validate_stats_payload(payload: Mapping[str, Any]) -> None:
    """
    Validate a top-level payload, typically loaded from a .pt file.

    Expectation:
      - either a mapping of name -> record (dict-like objects)
      - or a single record itself.
    """
    # Single-record case
    if "target" in payload and "module" in payload:
        validate_record_schema(payload)
        return

    # Mapping of records
    for key, value in payload.items():
        if not isinstance(value, Mapping):
            raise AssertionError(f"stats[{key!r}] must be a mapping, got {type(value)}")
        validate_record_schema(value)


# ---------------------------------------------------------------------------
# Affine quantization helpers (thin wrapper around UniformAffineQuantizer)
# ---------------------------------------------------------------------------


def compute_affine_params(
    x_min: float,
    x_max: float,
    bits: int,
    signed: bool,
) -> Tuple[float, int]:
    """
    Compute (scale, zero_point) for a simple per-tensor affine quantizer.

    This is intentionally a thin wrapper around the existing
    `UniformAffineQuantizer` implementation in `quantizer.py` so that
    the math stays perfectly aligned with the rest of the project.

    Args:
        x_min: minimum real value to represent
        x_max: maximum real value to represent
        bits: bitwidth in [2, 16]
        signed: if True, use a symmetric signed range; otherwise
                use an asymmetric unsigned range.

    Returns:
        (scale, zero_point) as Python scalars.
    """
    if bits < 2 or bits > 16:
        raise ValueError(f"Unsupported bitwidth: {bits}")

    # `symmetric=True` + `disable_zero_point=True` gives a symmetric, signed
    # quantizer; the zero_point is effectively 0 in that case.
    quant = UniformAffineQuantizer(
        n_bits=bits,
        symmetric=signed,
        disable_zero_point=signed,
        observe="minmax",
        is_weight=False,
    )

    xmin_t = torch.tensor(float(x_min))
    xmax_t = torch.tensor(float(x_max))

    if signed:
        quant.symmetric_cal_scale(xmin_t, xmax_t)
    else:
        quant.assymmetric_cal_scale(xmin_t, xmax_t)

    scale = float(quant.scale.item())
    if quant.round_zero_point is None:
        zero_point = 0
    else:
        zero_point = int(quant.round_zero_point.item())

    return scale, zero_point

