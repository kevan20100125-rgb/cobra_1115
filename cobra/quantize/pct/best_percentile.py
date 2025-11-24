# cobra/quantize/pct/best_percentile.py

"""
Heuristics for selecting a "best" clipping percentile per target.

This module consumes activation statistics collected by `collect.py` and
produces, for each canonical target:

    - "vision.dino"
    - "vision.siglip"
    - "llm"
    - "projector"

a single scalar percentile such as 99.0, 99.9, 99.99, ...

The intent is:
    - Use robust statistics (IQR-based scale) to avoid over-reacting to outliers.
    - Use growth-ratio tests between successive high-percentile intervals to
      detect when the tail becomes dominated by a few extreme values.
    - Keep the logic simple and debuggable so that callers (e.g. CLI switches)
      can print rule hits and maps.

This file does *not* decide the actual hi/lo numeric bounds; that is handled
by `apply.py`, which maps percentile → (hi, lo).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .schema import (
    normalize_target,
    is_percentile_key,
    parse_percentile_key,
    iter_percentile_items,
    validate_stats_payload,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RecordView:
    """
    Lightweight view over a single stats record.

    Attributes:
        name:   arbitrary key/name used in the top-level stats dict
        target: canonical target name ("vision.dino", "vision.siglip", "llm", "projector")
        module: module path / repr (for debug only)
        pcts:   sorted list of (percent, value) pairs, percent in [0, 100]
    """

    name: str
    target: str
    module: str
    pcts: List[Tuple[float, float]]


@dataclass
class RuleTrace:
    """
    Captures how a particular best-percentile decision was made.

    This is kept intentionally small; callers that want richer debugging
    can extend or pretty-print as they see fit.
    """

    rule: str
    chosen_percent: float
    candidates: List[float]

    # Optional metadata for richer analysis (record-level context)
    record_name: Optional[str] = None
    target: Optional[str] = None
    module: Optional[str] = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_record_view(name: str, rec: Mapping[str, Any]) -> Optional[RecordView]:
    """
    Convert a raw stats record dict into a RecordView.

    Returns None if the record does not contain any usable percentile keys.
    """
    try:
        target = normalize_target(str(rec.get("target", "")))
    except KeyError:
        return None

    module = str(rec.get("module", name))

    items: List[Tuple[float, float]] = []
    for key, value in iter_percentile_items(rec):
        try:
            pct = parse_percentile_key(key)
            val = float(value)
        except (ValueError, TypeError):
            continue
        items.append((pct, val))

    if not items:
        return None

    # sort by percentile ascending
    items.sort(key=lambda t: t[0])
    return RecordView(name=name, target=target, module=module, pcts=items)


def _compute_robust_scale(pcts: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """
    Compute IQR-based robust scale S = 0.7413 * (p75 - p25), plus key quantiles.

    Returns:
        (p25, p50, p75, S)
    """
    # Build a simple mapping percent -> value
    lookup: Dict[float, float] = {p: v for p, v in pcts}

    def nearest(q: float) -> float:
        # If exact percentile not available, use nearest available.
        if q in lookup:
            return lookup[q]
        # Find closest in absolute difference.
        closest = min(pcts, key=lambda t: abs(t[0] - q))
        return closest[1]

    p25 = nearest(25.0)
    p50 = nearest(50.0)
    p75 = nearest(75.0)

    S = 0.7413 * (p75 - p25)
    if S <= 0:
        S = 1.0
    return p25, p50, p75, S


def _select_from_tail(
    pcts: Sequence[Tuple[float, float]],
    min_base: float = 90.0,
    tau_growth: float = 5.0,
) -> Tuple[float, RuleTrace]:
    """
    Core heuristic: scan high-percentile tail and detect when growth between
    successive intervals becomes too aggressive.

    Args:
        pcts: sorted list of (percent, value) pairs.
        min_base: smallest percentile we consider as "tail" (default 90.0).
        tau_growth: growth ratio threshold; if
            (delta_current / (delta_prev + eps)) > tau_growth,
            we consider current percentile as dominated by outliers and choose
            the previous percentile.

    Returns:
        (best_percentile, RuleTrace)
    """
    # Filter to percentiles >= min_base
    tail = [(p, v) for (p, v) in pcts if p >= min_base]
    if len(tail) < 2:
        # Not enough high-percentile info; fall back to max available.
        best = tail[-1][0] if tail else pcts[-1][0]
        return best, RuleTrace(
            rule="fallback_max",
            chosen_percent=best,
            candidates=[t[0] for t in (tail or pcts)],
        )

    eps = 1e-6
    candidates = [p for (p, _) in tail]

    # delta_i = v_i - v_{i-1}
    deltas = [tail[i][1] - tail[i - 1][1] for i in range(1, len(tail))]

    # Iterate over growth ratios: g_i = delta_i / (delta_{i-1} + eps)
    # If we detect a large jump, we choose the previous percentile.
    for i in range(1, len(deltas)):
        prev_delta = deltas[i - 1]
        curr_delta = deltas[i]
        growth = curr_delta / (abs(prev_delta) + eps)

        if growth > tau_growth:
            best = tail[i - 1][0]
            return best, RuleTrace(
                rule=f"g{i}_gt_tau",
                chosen_percent=best,
                candidates=candidates,
            )

    # If no explosive growth detected, use the highest available.
    best = tail[-1][0]
    return best, RuleTrace(rule="max_tail", chosen_percent=best, candidates=candidates)


def _compute_best_for_record(
    view: RecordView,
    tau_growth: float = 5.0,
) -> Tuple[float, RuleTrace]:
    """
    Determine best percentile for a single RecordView.
    """
    _, p50, _, S = _compute_robust_scale(view.pcts)
    best, trace = _select_from_tail(view.pcts, min_base=90.0, tau_growth=tau_growth)

    # Attach record-level metadata (helps downstream analysis).
    trace.record_name = view.name
    trace.target = view.target
    trace.module = view.module

    # Attach a simple sanity check via z-score; if the chosen percentile
    # is extremely far from the median, we can clamp to a slightly lower one.
    lookup = {p: v for p, v in view.pcts}
    val = lookup.get(best, view.pcts[-1][1])
    z = (val - p50) / S

    if abs(z) > 12.0 and len(view.pcts) >= 2:
        # pick second-highest percentile as a conservative fallback
        alt = sorted([p for p, _ in view.pcts])[-2]
        best = alt
        trace = RuleTrace(
            rule="zscore_clamp",
            chosen_percent=best,
            candidates=trace.candidates,
            record_name=view.name,
            target=view.target,
            module=view.module,
        )

    return best, trace


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_best_percentile_for_target(
    records: Iterable[Mapping[str, Any]],
    tau_growth: float = 5.0,
) -> Tuple[float, List[RuleTrace]]:
    """
    Compute a single best percentile for a given target, aggregating over
    multiple module-level records.

    Strategy:
        - Compute best percentile per record (module).
        - Aggregate via median across records to reduce sensitivity to outliers.

    Returns:
        (best_percentile, traces)

        where `traces` is a list of RuleTrace for each contributing record.
    """
    views: List[RecordView] = []
    for idx, rec in enumerate(records):
        rv = _extract_record_view(name=str(idx), rec=rec)
        if rv is not None:
            views.append(rv)

    if not views:
        # No usable records; fall back to a conservative default.
        return 99.0, []

    per_record_best: List[Tuple[float, RuleTrace]] = []
    for v in views:
        bp, trace = _compute_best_for_record(v, tau_growth=tau_growth)
        per_record_best.append((bp, trace))

    # Median across records
    percents = sorted(bp for bp, _ in per_record_best)
    mid = len(percents) // 2
    if len(percents) % 2 == 1:
        best_overall = percents[mid]
    else:
        best_overall = 0.5 * (percents[mid - 1] + percents[mid])

    traces = [t for _, t in per_record_best]
    return best_overall, traces


def get_best_percentile_map(
    stats: Mapping[str, Mapping[str, Any]],
    tau_growth: float = 5.0,
    include_targets: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Compute the best percentile for each canonical target given a stats payload.

    Args:
        stats:
            Mapping of arbitrary key -> record dict. Each record is typically
            produced by `collect.py` and contains percentile keys (e.g. "p99.9")
            along with metadata such as "target" and "module".
        tau_growth:
            Growth-ratio threshold; higher values produce more aggressive
            (higher) percentiles.
        include_targets:
            Optional list/sequence of canonical targets to include
            (e.g. ["vision.dino", "vision.siglip", "llm", "projector"]).
            If None, all targets present in `stats` are considered.

    Returns:
        A dict {target_name: best_percentile_float}.
    """
    validate_stats_payload(stats)

    # Normalize include_targets to canonical names if provided
    allowed: Optional[set[str]] = None
    if include_targets is not None:
        allowed = {normalize_target(t) for t in include_targets}

    # Group records by canonical target name
    by_target: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for name, rec in stats.items():
        rv = _extract_record_view(name, rec)
        if rv is None:
            continue
        if allowed is not None and rv.target not in allowed:
            continue
        by_target[rv.target].append(rec)

    out: Dict[str, float] = {}
    for target, recs in by_target.items():
        best, _ = compute_best_percentile_for_target(recs, tau_growth=tau_growth)
        out[target] = float(best)

    return out


def get_best_percentile_map_with_traces(
    stats: Mapping[str, Mapping[str, Any]],
    tau_growth: float = 5.0,
    include_targets: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, float], Dict[str, List[RuleTrace]]]:
    """
    Variant of `get_best_percentile_map` that also returns per-target rule traces.

    This is intended for CLI frontends such as `quant_pct_apply.py` which want
    to print, for each target, how many records hit each rule.

    Returns:
        (best_percent_map, trace_map)

        best_percent_map:
            {target_name: best_percentile_float}

        trace_map:
            {target_name: [RuleTrace, ...]}
    """
    validate_stats_payload(stats)

    allowed: Optional[set[str]] = None
    if include_targets is not None:
        allowed = {normalize_target(t) for t in include_targets}

    by_target: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for name, rec in stats.items():
        rv = _extract_record_view(name, rec)
        if rv is None:
            continue
        if allowed is not None and rv.target not in allowed:
            continue
        by_target[rv.target].append(rec)

    best_map: Dict[str, float] = {}
    traces_map: Dict[str, List[RuleTrace]] = {}

    for target, recs in by_target.items():
        best, traces = compute_best_percentile_for_target(recs, tau_growth=tau_growth)
        best_map[target] = float(best)
        traces_map[target] = traces

    return best_map, traces_map


# Backwards-compatible alias used in some earlier scripts / discussions.
def get_optimal_percent_map(
    stats: Mapping[str, Mapping[str, Any]],
    tau_growth: float = 5.0,
    include_targets: Optional[Sequence[str]] = None,
    stages: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    """
    Legacy alias to `get_best_percentile_map`.

    Args:
        stats:
            Percentile stats payload produced by `collect.py`.
        tau_growth:
            Growth-ratio threshold for best-percentile selection.
        include_targets:
            Canonical targets to include; preferred new-style argument.
        stages:
            Backwards-compatible alias for `include_targets`, as used by
            older CLI code (e.g. `quant_pct_apply.py`).
    """
    # Backwards-compat: `stages` is an alias for `include_targets`.
    if stages is not None and include_targets is None:
        include_targets = stages

    return get_best_percentile_map(
        stats=stats,
        tau_growth=tau_growth,
        include_targets=include_targets,
    )

