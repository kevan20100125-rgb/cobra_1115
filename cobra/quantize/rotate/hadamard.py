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
def _hadamard_transform(
    X: torch.Tensor,
    *,
    dim: int = -1,
    normalize: bool = True,
) -> torch.Tensor:
    """
    Internal Hadamard transform wrapper.

    Uses cobra.quantize.hadamard_utils.matmul_hadU which applies a normalized
    Hadamard transform along the last dimension (i.e., scales by 1/sqrt(N)).

    This helper supports applying the transform along an arbitrary dimension by
    permuting the tensor, and supports normalize=False by undoing the 1/sqrt(N)
    scaling (i.e., multiplying by sqrt(N)).
    """
    if not isinstance(X, torch.Tensor):
        raise TypeError(f"Expected X to be torch.Tensor, got {type(X)}")

    if dim < 0:
        dim = X.ndim + dim
    if dim < 0 or dim >= X.ndim:
        raise ValueError(f"Invalid dim={dim} for X.ndim={X.ndim}")

    # matmul_hadU operates on the last dimension
    if dim != X.ndim - 1:
        perm = list(range(X.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        Xp = X.permute(*perm).contiguous()
        Yp = matmul_hadU(Xp)
        if not normalize:
            n = Xp.shape[-1]
            Yp = Yp * torch.tensor(n, device=Yp.device, dtype=Yp.dtype).sqrt()
        # invert permutation
        inv = [0] * len(perm)
        for i, p in enumerate(perm):
            inv[p] = i
        return Yp.permute(*inv).contiguous()

    Y = matmul_hadU(X)
    if not normalize:
        n = X.shape[-1]
        Y = Y * torch.tensor(n, device=Y.device, dtype=Y.dtype).sqrt()
    return Y

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
