# cobra/quantize/pct/collect.py

"""
Activation collection utilities for percentile clipping.

This module provides a minimal, model-agnostic API for collecting activation
statistics from a Cobra model (or any nn.Module) without re-implementing any
observer logic.

Key responsibilities:
    - Given a mapping from canonical percentile targets:
          {"vision.dino", "vision.siglip", "llm", "projector"}
      to a list of module *qualified names* in `model.named_modules()`,
      attach lightweight forward hooks.
    - Accumulate a bounded reservoir of activation samples per (target, module).
    - After a calibration run (dataloader loop), convert the collected samples
      into a schema-compatible stats payload with keys:
          - "mode", "numel", "target", "module", "min", "max"
          - dynamic percentile keys such as "p25", "p50", "p75", "p99.9", ...

The actual *choice* of "best percentile" is delegated to `best_percentile.py`;
this module only produces the raw empirical distribution summaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from cobra.overwatch import initialize_overwatch

from .schema import (
    PercentileRecord,
    normalize_target,
)

overwatch = initialize_overwatch(__name__)


# ---------------------------------------------------------------------------
# Activation buffer + hook handle
# ---------------------------------------------------------------------------


@dataclass
class ActivationBuffer:
    """
    Bounded reservoir of activation samples for a single (target, module).

    Notes:
        - We keep all samples up to `max_samples`; above that, we perform a
          simple reservoir-style downsampling via random permutation.
        - Samples are stored as a 1-D float32 tensor on the given device.
    """

    target: str
    module_name: str
    max_samples: int = 2_000_000
    device: torch.device = torch.device("cpu")

    _storage: Optional[torch.Tensor] = None

    @property
    def numel(self) -> int:
        return int(self._storage.numel()) if self._storage is not None else 0

    def update(self, tensor: torch.Tensor) -> None:
        """
        Ingest a batch of activations.

        Args:
            tensor: arbitrary-shaped activation tensor; will be flattened,
                    converted to float32, and moved to `self.device`.
        """
        if tensor is None:
            return

        if isinstance(tensor, (tuple, list)):
            # e.g., some modules return (output, aux)
            tensor = tensor[0]

        if not isinstance(tensor, torch.Tensor):
            return

        if tensor.is_sparse:
            tensor = tensor.to_dense()

        x = tensor.detach()
        # Move to CPU (or configured device) and flatten
        x = x.to(self.device, dtype=torch.float32).reshape(-1)

        if x.numel() == 0:
            return

        if self._storage is None:
            # First batch
            max_n = min(x.numel(), self.max_samples)
            self._storage = x[:max_n].clone()
            return

        # Concatenate and downsample if needed
        cat = torch.cat([self._storage, x], dim=0)
        if cat.numel() <= self.max_samples:
            self._storage = cat
            return

        # Reservoir: keep a random subset of size max_samples
        idx = torch.randperm(cat.numel(), device=cat.device)[: self.max_samples]
        self._storage = cat[idx]


@dataclass
class ActivationCollector:
    """
    Lightweight wrapper around an ActivationBuffer and its hook handle.
    """

    target: str
    module_name: str
    buffer: ActivationBuffer
    handle: RemovableHandle

    @property
    def numel(self) -> int:
        return self.buffer.numel

    def remove(self) -> None:
        self.handle.remove()


# ---------------------------------------------------------------------------
# Hook registration
# ---------------------------------------------------------------------------


def _make_hook(buffer: ActivationBuffer):
    def hook(_module: nn.Module, _inputs: Tuple[Any, ...], output: Any) -> None:
        buffer.update(output)

    return hook


def register_activation_collectors(
    model: nn.Module,
    target_to_module_names: Mapping[str, Sequence[str]],
    *,
    max_samples_per_module: int = 2_000_000,
    device: Optional[torch.device] = None,
) -> Dict[str, ActivationCollector]:
    """
    Attach forward hooks to selected modules and return a dict of collectors.

    Args:
        model: nn.Module in which to register hooks (e.g., CobraVLM instance).
        target_to_module_names:
            Mapping from *target name* (any alias accepted by `normalize_target`)
            to a list of module qualified names as returned by
            `dict(model.named_modules()).keys()`.
        max_samples_per_module:
            Upper bound on the number of activation samples stored per module.
        device:
            Device on which to keep the activation reservoir; default is CPU.

    Returns:
        A dict mapping a stable record key:
            f"{target}::{module_name}"
        to an ActivationCollector instance.

    Raises:
        KeyError: if any requested module name cannot be found.
    """
    if device is None:
        device = torch.device("cpu")

    name_to_module = dict(model.named_modules())
    collectors: Dict[str, ActivationCollector] = {}

    for raw_target, module_names in target_to_module_names.items():
        target = normalize_target(raw_target)

        for module_name in module_names:
            if module_name not in name_to_module:
                raise KeyError(f"Module name {module_name!r} not found in model.named_modules()")

            module = name_to_module[module_name]

            buffer = ActivationBuffer(
                target=target,
                module_name=module_name,
                max_samples=max_samples_per_module,
                device=device,
            )
            hook = module.register_forward_hook(_make_hook(buffer))

            key = f"{target}::{module_name}"
            collectors[key] = ActivationCollector(
                target=target,
                module_name=module_name,
                buffer=buffer,
                handle=hook,
            )

            overwatch.debug(
                f"[PctCollect] Registered hook for target={target!r}, module={module_name!r}",
                extra={"target": target, "module": module_name},
            )

    return collectors


def remove_activation_collectors(collectors: Mapping[str, ActivationCollector]) -> None:
    """
    Remove all forward hooks associated with the given collectors.
    """
    for col in collectors.values():
        col.remove()


# ---------------------------------------------------------------------------
# Stats construction
# ---------------------------------------------------------------------------


_DEFAULT_PERCENTILES: Tuple[float, ...] = (
    25.0,
    50.0,
    75.0,
    90.0,
    95.0,
    99.0,
    99.9,
    99.99,
    99.999,
)


def _format_percent_key(q: float) -> str:
    """
    Turn a percentile value (e.g. 99.9) into a schema key like "p99.9".
    """
    if float(q).is_integer():
        return f"p{int(q)}"
    # Normalize to trimmed decimal string
    s = f"{q:.6f}".rstrip("0").rstrip(".")
    return f"p{s}"


def build_activation_stats(
    collectors: Mapping[str, ActivationCollector],
    *,
    percentiles: Optional[Sequence[float]] = None,
    mode: str = "activation",
) -> Dict[str, PercentileRecord]:
    """
    Convert collected activation samples into a stats payload.

    Args:
        collectors: mapping of record-key -> ActivationCollector.
        percentiles: optional custom percentile list; defaults to a preset that
            covers both robust statistics (p25/50/75) and high-tail behavior
            (p90, p95, p99, p99.9, p99.99, p99.999).
        mode: an opaque string stored in each record's "mode" field.

    Returns:
        stats: mapping from record-key to PercentileRecord, ready to be saved
               (e.g. via torch.save) and later consumed by `best_percentile.py`
               and `apply.py`.
    """
    if percentiles is None:
        percentiles = list(_DEFAULT_PERCENTILES)

    # Ensure sorted & unique for deterministic behavior
    percentiles = sorted(set(float(q) for q in percentiles))

    stats: Dict[str, PercentileRecord] = {}

    for key, col in collectors.items():
        buf = col.buffer
        if buf._storage is None or buf.numel == 0:
            overwatch.warning(
                f"[PctCollect] No activations collected for target={col.target!r}, "
                f"module={col.module_name!r}; skipping.",
                extra={"target": col.target, "pct_module": col.module_name},
            )
            continue

        x = buf._storage
        # For safety, move to CPU and ensure float32
        x = x.detach().to("cpu", dtype=torch.float32)

        record: PercentileRecord = {
            "mode": mode,
            "numel": int(x.numel()),
            "target": col.target,
            "module": col.module_name,
            "min": float(x.min().item()),
            "max": float(x.max().item()),
        }

        # Compute requested percentiles
        qs = torch.tensor([q / 100.0 for q in percentiles], dtype=torch.float32, device=x.device)
        values = torch.quantile(x, qs)

        for q, v in zip(percentiles, values.tolist()):
            key_q = _format_percent_key(q)
            record[key_q] = float(v)

        stats[key] = record

        overwatch.info(
            f"[PctCollect] target={col.target!r}, module={col.module_name!r}, "
            f"numel={record['numel']}, min={record['min']:.6f}, max={record['max']:.6f}",
            extra={"target": col.target, "pct_module": col.module_name},
        )

    return stats

