# cobra/quantize/pct/apply.py

"""
Mapping from best percentile → numeric (hi, lo) clipping bounds.

Responsibilities:
    - Consume activation stats produced by `collect.py`.
    - Consume a per-target best-percentile map (either provided or computed via
      `best_percentile.get_best_percentile_map`).
    - For each record (target, module), compute a pair of numeric bounds:
          hi >= 0, lo <= 0
      typically using symmetric clipping around 0.

This module does NOT:
    - Touch any quantizer implementation or model layers.
    - Write scales/zero-points; that is delegated to `calibrator.py`.

Typical usage (inside a CLI script):

    stats = torch.load("pct_stats.pt")
    # Case 1: heuristic best-percentile (apply mode)
    hi_lo_map = build_hi_lo_map(
        stats,
        best_percent_map=None,
        tau_growth=5.0,
        symmetric=True,
        default_percentile=None,
        targets=("vision.dino", "vision.siglip", "llm", "projector"),
    )

    # Case 2: best-percentile disabled (off mode), use default_percentile for all
    hi_lo_map = build_hi_lo_map(
        stats,
        best_percent_map=None,
        tau_growth=5.0,
        symmetric=True,
        default_percentile=99.9,
        targets=("vision.dino", "vision.siglip", "llm", "projector"),
    )

The resulting hi_lo_map is then consumed by `calibrator.py` to register
scale/zero-point per module.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, TypedDict

from cobra.overwatch import initialize_overwatch

from .schema import (
    PercentileRecord,
    normalize_target,
    is_percentile_key,
    parse_percentile_key,
    iter_percentile_items,
    validate_stats_payload,
)
from .best_percentile import get_best_percentile_map


overwatch = initialize_overwatch(__name__)


class HiLoRecord(TypedDict, total=False):
    """
    Schema for hi/lo clipping results per (target, module).
    """

    target: str
    module: str
    percent: float
    hi: float
    lo: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _select_percentile_value(
    rec: Mapping[str, Any],
    percent: float,
) -> float:
    """
    Select an approximate value for the requested percentile from a stats record.

    Strategy:
        - Inspect all percentile-like keys (e.g., "p99.9", "p99_9") available
          in the record.
        - Choose the key whose numeric percentile is closest to `percent`.
        - If no percentile keys are present, or if the chosen key is missing,
          fall back to max(|min|, |max|) of the record.

    This assumes that `collect.py` has already populated a decent coverage of
    percentiles (e.g., 25, 50, 75, 90, 95, 99, 99.9, ...).
    """
    candidates: List[Tuple[float, str]] = []
    for k in rec.keys():
        if is_percentile_key(k):
            try:
                p = parse_percentile_key(k)
            except ValueError:
                continue
            candidates.append((p, k))

    if not candidates:
        # Fallback: no percentile keys at all; use max abs from min/max.
        x_min = float(rec.get("min", 0.0))
        x_max = float(rec.get("max", 0.0))
        return max(abs(x_min), abs(x_max))

    # Find nearest available percentile key.
    target = float(percent)
    best_p, best_key = min(candidates, key=lambda t: abs(t[0] - target))
    value = rec.get(best_key, None)
    if value is None:
        x_min = float(rec.get("min", 0.0))
        x_max = float(rec.get("max", 0.0))
        return max(abs(x_min), abs(x_max))

    return float(value)


def compute_hi_lo_for_record(
    rec: Mapping[str, Any],
    percent: float,
    *,
    symmetric: bool = True,
) -> Tuple[float, float]:
    """
    Compute (hi, lo) for a single stats record at the given percentile.

    Args:
        rec:      A single PercentileRecord-like mapping.
        percent:  Desired clipping percentile (e.g. 99.0, 99.9).
        symmetric:
            If True (default), we perform symmetric clipping around 0:
                hi = |v|
                lo = -|v|
            where v is the selected percentile value.
            If False, hi = v, lo = min(rec["min"], -hi) as a conservative bound.

    Returns:
        (hi, lo) as Python floats.
    """
    v = _select_percentile_value(rec, percent)
    if symmetric:
        hi = float(abs(v))
        lo = -hi
    else:
        hi = float(v)
        lo = float(rec.get("min", -hi))
    return hi, lo


def _log_hi_lo_stats(
    hi_lo_map: Mapping[str, HiLoRecord],
    *,
    tail_hi_threshold: float = 20.0,
) -> None:
    """
    Log aggregate hi/lo statistics and long-tail counts per target.

    This is a logging-only helper to aid analysis of the percentile → hi/lo
    mapping. It does not mutate the input map or depend on any wrap/calibrator
    logic.

    A module is considered long-tail if its hi exceeds `tail_hi_threshold`.
    """
    if not hi_lo_map:
        overwatch.info("[PctApply] hi/lo map is empty; no stats to summarize")
        return

    per_target: Dict[str, Dict[str, List[float]]] = {}
    long_tail_by_target: Dict[str, List[Tuple[str, float]]] = {}

    for name, rec in hi_lo_map.items():
        tgt = str(rec.get("target", "<unknown>"))
        pct = rec.get("percent", None)
        hi = rec.get("hi", None)
        lo = rec.get("lo", None)

        bucket = per_target.setdefault(
            tgt, {"percentile": [], "hi": [], "lo": []}
        )
        if isinstance(pct, (int, float)):
            bucket["percentile"].append(float(pct))
        if isinstance(hi, (int, float)):
            h = float(hi)
            bucket["hi"].append(h)
            if h > tail_hi_threshold:
                long_tail_by_target.setdefault(tgt, []).append((name, h))
        if isinstance(lo, (int, float)):
            bucket["lo"].append(float(lo))

    all_pct: List[float] = []
    all_hi: List[float] = []
    all_lo: List[float] = []

    for tgt, vals in per_target.items():
        pct_vals = vals["percentile"]
        hi_vals = vals["hi"]
        lo_vals = vals["lo"]

        count = max(len(pct_vals), len(hi_vals), len(lo_vals))
        if count == 0:
            continue

        if pct_vals:
            pct_min = min(pct_vals)
            pct_max = max(pct_vals)
            pct_mean = sum(pct_vals) / float(len(pct_vals))
            all_pct.extend(pct_vals)
        else:
            pct_min = pct_max = pct_mean = None

        if hi_vals:
            hi_min = min(hi_vals)
            hi_max = max(hi_vals)
            hi_mean = sum(hi_vals) / float(len(hi_vals))
            all_hi.extend(hi_vals)
        else:
            hi_min = hi_max = hi_mean = None

        if lo_vals:
            lo_min = min(lo_vals)
            lo_max = max(lo_vals)
            lo_mean = sum(lo_vals) / float(len(lo_vals))
            all_lo.extend(lo_vals)
        else:
            lo_min = lo_max = lo_mean = None

        long_tail = long_tail_by_target.get(tgt, [])
        long_tail_count = len(long_tail)

        overwatch.info(
            (
                "[PctApply] Stats | target=%r count=%d "
                "pct[min=%.3f max=%.3f mean=%.3f] "
                "hi[min=%.6f max=%.6f mean=%.6f] "
                "lo[min=%.6f max=%.6f mean=%.6f] "
                "long_tail_hi>%.1f=%d"
            )
            % (
                tgt,
                count,
                pct_min if pct_min is not None else float("nan"),
                pct_max if pct_max is not None else float("nan"),
                pct_mean if pct_mean is not None else float("nan"),
                hi_min if hi_min is not None else float("nan"),
                hi_max if hi_max is not None else float("nan"),
                hi_mean if hi_mean is not None else float("nan"),
                lo_min if lo_min is not None else float("nan"),
                lo_max if lo_max is not None else float("nan"),
                lo_mean if lo_mean is not None else float("nan"),
                tail_hi_threshold,
                long_tail_count,
            ),
            extra={
                "target": tgt,
                "count": count,
                "pct_min": pct_min,
                "pct_max": pct_max,
                "pct_mean": pct_mean,
                "hi_min": hi_min,
                "hi_max": hi_max,
                "hi_mean": hi_mean,
                "lo_min": lo_min,
                "lo_max": lo_max,
                "lo_mean": lo_mean,
                "tail_hi_threshold": tail_hi_threshold,
                "long_tail_count": long_tail_count,
            },
        )

        if long_tail_count > 0:
            sample = long_tail[:5]
            overwatch.debug(
                "[PctApply] Long-tail hi modules for target=%r (sample): %s",
                tgt,
                [n for (n, _h) in sample],
                extra={
                    "target": tgt,
                    "tail_hi_threshold": tail_hi_threshold,
                    "num_long_tail": long_tail_count,
                },
            )

    num_entries = len(hi_lo_map)
    pct_mean_global = (
        sum(all_pct) / float(len(all_pct)) if all_pct else float("nan")
    )
    hi_min_global = min(all_hi) if all_hi else float("nan")
    hi_max_global = max(all_hi) if all_hi else float("nan")
    hi_mean_global = (
        sum(all_hi) / float(len(all_hi)) if all_hi else float("nan")
    )
    lo_min_global = min(all_lo) if all_lo else float("nan")
    lo_max_global = max(all_lo) if all_lo else float("nan")
    lo_mean_global = (
        sum(all_lo) / float(len(all_lo)) if all_lo else float("nan")
    )

    overwatch.info(
        (
            "[PctApply] Global hi/lo stats | num_entries=%d "
            "pct_mean=%.3f hi[min=%.6f max=%.6f mean=%.6f] "
            "lo[min=%.6f max=%.6f mean=%.6f]"
        )
        % (
            num_entries,
            pct_mean_global,
            hi_min_global,
            hi_max_global,
            hi_mean_global,
            lo_min_global,
            lo_max_global,
            lo_mean_global,
        ),
        extra={
            "num_entries": num_entries,
            "pct_mean": pct_mean_global,
            "hi_min": hi_min_global,
            "hi_max": hi_max_global,
            "hi_mean": hi_mean_global,
            "lo_min": lo_min_global,
            "lo_max": lo_max_global,
            "lo_mean": lo_mean_global,
        },
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_hi_lo_map(
    stats: Mapping[str, Mapping[str, Any]],
    *,
    best_percent_map: Optional[Mapping[str, float]] = None,
    tau_growth: float = 5.0,
    symmetric: bool = True,
    default_percentile: Optional[float] = None,
    include_targets: Optional[Sequence[str]] = None,
    targets: Optional[Sequence[str]] = None,
) -> Dict[str, HiLoRecord]:
    """
    Build a hi/lo clipping map per record from collected stats.

    Args:
        stats:
            Mapping of record-key -> stats record, typically produced by
            `collect.build_activation_stats(...)`. Each record should follow
            the PercentileRecord schema (contains "target", "module", "min",
            "max", and various "pXX[.YY]" percentile keys).
        best_percent_map:
            Optional mapping {target -> percentile}. If provided, this mapping
            is used as-is and `get_best_percentile_map` is not invoked.
        tau_growth:
            Growth-ratio threshold passed through to `get_best_percentile_map`
            when `best_percent_map` is None and `default_percentile` is None.
            Larger values lead to more aggressive (higher) percentiles.
        symmetric:
            Whether to enforce symmetric clipping hi = -lo around 0.
        default_percentile:
            If not None and `best_percent_map` is None, this value is used as
            the clipping percentile for all targets (i.e., "best percentile"
            heuristic is effectively disabled, matching `"best_percentile"="off"`
            semantics in the CLI).
            If both `best_percent_map` and `default_percentile` are provided,
            `best_percent_map` takes precedence per target, and
            `default_percentile` is only used as a fallback for targets not
            appearing in the map.
        include_targets:
            Optional list/tuple of targets to include. Each entry can be any
            alias accepted by `normalize_target`. This is the preferred new
            name used by other modules (e.g., calibrator).
        targets:
            Backwards-compatible alias for `include_targets`, used by older
            CLI code (e.g., `quant_pct_apply.py`).

    Returns:
        hi_lo_map:
            Mapping from record-key (e.g. "vision.dino::backbone.blocks.0")
            to HiLoRecord:
                {
                    "target":  "vision.dino",
                    "module":  "backbone.blocks.0",
                    "percent": 99.9,
                    "hi":      0.42,
                    "lo":     -0.42,
                }
    """
    validate_stats_payload(stats)

    # -----------------------------------------------------------------------
    # Normalize the target filters (include_targets / targets)
    # -----------------------------------------------------------------------
    effective_targets: Optional[Sequence[str]] = None
    if include_targets is not None:
        effective_targets = include_targets
    elif targets is not None:
        effective_targets = targets

    allowed_targets: Optional[list[str]] = None
    if effective_targets is not None:
        allowed_targets = [normalize_target(t) for t in effective_targets]

    # -----------------------------------------------------------------------
    # Derive best_percent_map if not provided AND we have no default_percentile
    # -----------------------------------------------------------------------
    if best_percent_map is None and default_percentile is None:
        # "apply" mode: run heuristic selection per target
        best_percent_map = get_best_percentile_map(
            stats=stats,
            tau_growth=tau_growth,
            include_targets=effective_targets,
        )

    # Normalize keys of best_percent_map to canonical targets.
    normalized_best: Dict[str, float] = {}
    if best_percent_map is not None:
        for t, p in best_percent_map.items():
            try:
                nt = normalize_target(t)
            except KeyError:
                continue
            normalized_best[nt] = float(p)

    hi_lo_map: Dict[str, HiLoRecord] = {}

    # -----------------------------------------------------------------------
    # Per-record hi/lo computation
    # -----------------------------------------------------------------------
    for name, rec in stats.items():
        target_raw = rec.get("target", "")
        try:
            target = normalize_target(str(target_raw))
        except KeyError:
            overwatch.warning(
                f"[PctApply] Skipping unknown target={target_raw!r} for record={name!r}",
                extra={"target": target_raw, "record": name},
            )
            continue

        if allowed_targets is not None and target not in allowed_targets:
            continue

        # Prefer target-specific best-percentile, fall back to default_percentile.
        percent: Optional[float] = normalized_best.get(target, default_percentile)

        if percent is None:
            overwatch.warning(
                f"[PctApply] No percentile available for target={target!r}; "
                f"skipping record={name!r}",
                extra={"target": target, "record": name},
            )
            continue

        hi, lo = compute_hi_lo_for_record(rec, percent, symmetric=symmetric)

        module_name = str(rec.get("module", name))

        hi_lo_map[name] = HiLoRecord(
            target=target,
            module=module_name,
            percent=float(percent),
            hi=float(hi),
            lo=float(lo),
        )

        overwatch.info(
            (
                "[PctApply] target=%r module=%r percentile=%7.3f "
                "hi=%+.6f lo=%+.6f"
            )
            % (target, module_name, percent, hi, lo),
            extra={"target": target, "pct_module": module_name},
        )

    # Log aggregate statistics and long-tail counts for analysis.
    _log_hi_lo_stats(hi_lo_map)

    return hi_lo_map

