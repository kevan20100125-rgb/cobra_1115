# cobra/quantize/rotate/hadamard.py

"""
Hadamard rotation utilities.

This module is intentionally a thin wrapper around the existing
`cobra.quantize.hadamard_utils` helper functions.

Responsibilities:
    - Provide a stable API for applying Hadamard transforms to weight matrices.
    - Avoid re-implementing math logic (reuse hadamard_utils fully).
    - Ensure integration with projector rotation (in projector.py).
"""

from __future__ import annotations

from typing import Optional

import torch
from cobra.overwatch import initialize_overwatch

# === Reuse existing implementation ===
from cobra.quantize.hadamard_utils import matmul_hadU

overwatch = initialize_overwatch(__name__)


# =====================================================================
# Public API
# =====================================================================

def apply_hadamard_transform(
    W: torch.Tensor,
    *,
    dim: int = -1,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Apply Hadamard transform to weight matrix W along specified dimension.

    Args:
        W:
            Weight tensor. Must be float dtype; Hadamard is linear so types
            are preserved.
        dim:
            Dimension along which the Hadamard matrix is applied.
            Default = -1 (last dimension).
        normalize:
            If True (recommended), scale by 1/sqrt(N) to preserve variance.

    Returns:
        W_rot:
            Rotated tensor, same shape as W.

    Notes:
        - This delegates directly to `_hadamard_transform` from hadamard_utils.
        - It does NOT change dtype or quantize the tensor.
        - It performs no in-place modifications.
    """
    if not isinstance(W, torch.Tensor):
        raise TypeError(f"Expected W to be torch.Tensor, got {type(W)}")

    W_rot = _hadamard_transform(W, dim=dim, normalize=normalize)

    overwatch.debug(
        f"[Hadamard] Applied Hadamard transform on dim={dim}, normalize={normalize}",
        extra={"shape": tuple(W.shape)},
    )
    return W_rot