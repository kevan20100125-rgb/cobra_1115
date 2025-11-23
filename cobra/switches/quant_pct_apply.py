# cobra/switches/quant_pct_apply.py

"""
quant_pct_apply.py

CLI entrypoint: "Percentile Apply Only"

This script performs the **offline percentile-apply step**:

    1) Load previously collected activation percentile stats from disk
       (e.g., outputs/quantize/pct_stats.pt).
    2) Optionally compute a "best percentile" map using heuristic rules
       (growth ratios, robust scale, etc.) via `best_percentile.py`.
    3) Convert the chosen percentiles into per-hook hi/lo clipping bounds
       via `pct.apply`.
    4) Save:
           - hi/lo map       -> pct_hi_lo.pt
           - human-readable  -> pct_apply_summary.json

It does NOT:
    - Touch any model or run a dataloader.
    - Attach scales/zeros to quantizers (that is the calibrator's job).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

import draccus
import torch

from cobra.overwatch import initialize_overwatch
from cobra.quantize.pct.best_percentile import get_optimal_percent_map
from cobra.quantize.pct.apply import build_hi_lo_map

overwatch = initialize_overwatch(__name__)


# =============================================================================
# Config
# =============================================================================


_CANONICAL_TARGETS: Tuple[str, ...] = ("vision.dino", "vision.siglip", "llm", "projector")


@dataclass
class PctApplyConfig:
    """
    Configuration for percentile-apply-only CLI.

    Paths
    -----
    pct_stats_in:
        Path to the percentile stats file produced by the **collect** step
        (e.g., outputs/quantize/pct_stats.pt).
    pct_hi_lo_out:
        Where to save the hi/lo clipping map (torch.save dict).
    pct_summary_out:
        Where to save a human-readable JSON summary of chosen percentiles
        and hi/lo ranges.

    Best-percentile logic
    ---------------------
    best_percentile:
        Mode in < "off" | "apply" >:
            - "off"   : do not use heuristic best percentiles; instead use
                        `default_percentile` for all hooks.
            - "apply" : compute a best-percentile map using
                        `get_optimal_percent_map(...)`.
    default_percentile:
        Fallback percentile when best-percentile is disabled or when a hook
        has no explicit override.

    Targets
    -------
    targets:
        Optional subset of canonical targets in
            {"vision.dino","vision.siglip","llm","projector"}.
        If empty, all four are considered.

    Other
    -----
    symmetric:
        Whether to enforce symmetric clipping [-hi, +hi] when computing hi/lo.
        This is forwarded to `build_hi_lo_map`.
    """

    # fmt: off
    pct_stats_in: Path = Path("outputs/quantize/pct_stats.pt")
    pct_hi_lo_out: Path = Path("outputs/quantize/pct_hi_lo.pt")
    pct_summary_out: Path = Path("outputs/quantize/pct_apply_summary.json")

    best_percentile: str = "apply"          # in {"off", "apply"}
    default_percentile: float = 99.9

    targets: Tuple[str, ...] = field(default_factory=lambda: _CANONICAL_TARGETS)
    symmetric: bool = True
    # fmt: on

    def resolved_targets(self) -> Tuple[str, ...]:
        if not self.targets:
            return _CANONICAL_TARGETS
        return self.targets


# =============================================================================
# Helpers
# =============================================================================


def _load_pct_stats(path: Path) -> Mapping[str, Mapping[str, float]]:
    if not path.is_file():
        raise FileNotFoundError(f"Percentile stats file not found at: {path}")
    stats = torch.load(path, map_location="cpu")
    if not isinstance(stats, Mapping):
        raise TypeError(f"Loaded stats must be a Mapping, got {type(stats)}")
    return stats


def _summarize_hi_lo_map(
    hi_lo_map: Mapping[str, Mapping[str, float]],
    ) -> Dict[str, Dict[str, float]]:
    """
    Build a compact, JSON-serializable summary from hi_lo_map.

    hi_lo_map is expected to be:
        {
          "<hook_name>": {
              "target": "vision.dino" | "vision.siglip" | "llm" | "projector",
              "percentile": 99.9,
              "hi": <float>,
              "lo": <float>,
              ...
          },
          ...
        }

    We keep only the core fields for readability.
    """
    summary: Dict[str, Dict[str, float]] = {}
    for hook, info in hi_lo_map.items():
        if not isinstance(info, Mapping):
            continue

        tgt = info.get("target", None)
        pct = info.get("percent", None)
        hi = info.get("hi", None)
        lo = info.get("lo", None)

        entry: Dict[str, float] = {}
        if tgt is not None:
            entry["target"] = str(tgt)
        if pct is not None:
            entry["percentile"] = float(pct)
        if hi is not None:
            entry["hi"] = float(hi)
        if lo is not None:
            entry["lo"] = float(lo)

        summary[hook] = entry

    return summary


def _save_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


# =============================================================================
# Main
# =============================================================================


@draccus.wrap()
def quant_pct_apply(cfg: PctApplyConfig) -> None:
    overwatch.info(
        "[quant_pct_apply] Starting percentile-apply phase",
        extra={
            "stats_in": str(cfg.pct_stats_in),
            "hi_lo_out": str(cfg.pct_hi_lo_out),
            "summary_out": str(cfg.pct_summary_out),
            "best_percentile": cfg.best_percentile,
            "default_percentile": cfg.default_percentile,
            "targets": cfg.resolved_targets(),
            "symmetric": cfg.symmetric,
        },
    )

    # -------------------------------------------------------------------------
    # 1) Load percentile stats
    # -------------------------------------------------------------------------
    stats = _load_pct_stats(cfg.pct_stats_in)

    # -------------------------------------------------------------------------
    # 2) Compute best-percentile map (optional)
    # -------------------------------------------------------------------------
    best_percent_map: Optional[Dict[str, float]] = None
    if cfg.best_percentile == "apply":
        stages: Optional[Tuple[str, ...]] = cfg.resolved_targets()
        best_percent_map = get_optimal_percent_map(
            stats=stats,
            stages=stages,
        )

        overwatch.info("[quant_pct_apply] Best-percentile map computed")
        if best_percent_map:
            for stage, value in sorted(best_percent_map.items()):
                overwatch.info(
                    f"[BestPercentile] {stage}: {value:.3f} applied:true",
                    extra={"stage": stage, "percentile": float(value)},
                )
        else:
            overwatch.info("[BestPercentile] (no overrides) applied:true")

    elif cfg.best_percentile not in {"off"}:
        raise ValueError(f"Unsupported best_percentile mode: {cfg.best_percentile!r}")

    # -------------------------------------------------------------------------
    # 3) Build hi/lo map from stats + best-percentile decisions
    # -------------------------------------------------------------------------
    hi_lo_map = build_hi_lo_map(
        stats=stats,
        best_percent_map=best_percent_map,
        default_percentile=cfg.default_percentile,
        symmetric=cfg.symmetric,
        targets=cfg.resolved_targets(),
    )

    # Save hi/lo map as a torch blob
    cfg.pct_hi_lo_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hi_lo_map, cfg.pct_hi_lo_out)

    # -------------------------------------------------------------------------
    # 4) Human-readable summary
    # -------------------------------------------------------------------------
    summary = {
        "config": {
            "pct_stats_in": str(cfg.pct_stats_in),
            "pct_hi_lo_out": str(cfg.pct_hi_lo_out),
            "best_percentile": cfg.best_percentile,
            "default_percentile": cfg.default_percentile,
            "targets": cfg.resolved_targets(),
            "symmetric": cfg.symmetric,
        },
        "num_entries": len(hi_lo_map),
        "by_target": {},
        "entries": _summarize_hi_lo_map(hi_lo_map),
    }

    # Aggregate counts by target
    by_target: Dict[str, int] = {}
    for hook, info in hi_lo_map.items():
        tgt = info.get("target", "unknown") if isinstance(info, Mapping) else "unknown"
        by_target[tgt] = by_target.get(tgt, 0) + 1
    summary["by_target"] = by_target

    _save_json(summary, cfg.pct_summary_out)

    overwatch.info(
        "[quant_pct_apply] Finished percentile-apply phase",
        extra={
            "num_entries": len(hi_lo_map),
            "by_target": by_target,
        },
    )


if __name__ == "__main__":
    quant_pct_apply()
