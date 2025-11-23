# cobra/quantize/rotate/klt.py

"""
KLT (Karhunen–Loève Transform) rotation utilities.

This module is a thin wrapper around `cobra.quantize.get_klt_matrix`:

- It does NOT重新實作任何 KLT 演算法。
- 它只提供幾個穩定、乾淨的介面，用於：
    * 對權重矩陣套用 KLT 旋轉 (right-multiply by K)
    * 對給定的 K 做正交化 (project to orthogonal)
    * 針對給定權重做小幅 KLT 優化 (optimize_klt_matrix_3)
    * 由現有 LLM 權重統計推估每層的 KLT (get_llm_weight_klt)

設計目的：
    - 給 `rotate/projector.py` 和 `finalize/int_export.py` 使用，
      作為「shared KLT」的統一入口。
    - 避免在各處直接 import `get_klt_matrix.py`，集中依賴，降低耦合。
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn

from cobra.overwatch import initialize_overwatch

# === 重用既有 get_klt_matrix.py ===
from cobra.quantize.get_klt_matrix import (
    project_to_orthogonal as _project_to_orthogonal,
    optimize_klt_matrix_3 as _optimize_klt_matrix_3,
    get_llm_weight_klt as _get_llm_weight_klt,
)

overwatch = initialize_overwatch(__name__)


# =====================================================================
# 基本 KLT 旋轉
# =====================================================================

def apply_klt_rotation(
    W: torch.Tensor,
    K: torch.Tensor,
    *,
    right_multiply: bool = True,
) -> torch.Tensor:
    """
    Apply KLT rotation to a weight matrix.

    通常用於 projector output weight：
        - 若 W 形狀為 (out_dim, in_dim)，K 為 (in_dim, in_dim)：
            right_multiply=True → W_rot = W @ K

    Args:
        W:
            權重張量，通常為 2D (out_dim, in_dim)，float dtype。
        K:
            KLT 矩陣，維度需對得上被乘的 axis。
        right_multiply:
            若 True (預設)，計算 W @ K；
            若 False，計算 K @ W。

    Returns:
        W_rot: 旋轉後的權重張量，shape 與 W 相同。
    """
    if not isinstance(W, torch.Tensor) or not isinstance(K, torch.Tensor):
        raise TypeError("W and K must be torch.Tensor")

    if right_multiply:
        if W.shape[-1] != K.shape[0]:
            raise ValueError(
                f"apply_klt_rotation (right): W.shape={tuple(W.shape)}, K.shape={tuple(K.shape)} "
                "expected W[..., in_dim] and K[in_dim, in_dim]"
            )
        W_rot = W @ K.to(W.device, W.dtype)
    else:
        if K.shape[-1] != W.shape[0]:
            raise ValueError(
                f"apply_klt_rotation (left): W.shape={tuple(W.shape)}, K.shape={tuple(K.shape)} "
                "expected K[out_dim, out_dim] and W[out_dim, ...]"
            )
        W_rot = K.to(W.device, W.dtype) @ W

    overwatch.debug(
        "[KLT] Applied KLT rotation",
        extra={
            "right_multiply": right_multiply,
            "W_shape": tuple(W.shape),
            "K_shape": tuple(K.shape),
        },
    )
    return W_rot


# =====================================================================
# KLT 矩陣處理 (正交化 / 微調)
# =====================================================================

def orthogonalize_klt_matrix(K: torch.Tensor) -> torch.Tensor:
    """
    將任意矩陣 K 投影到正交矩陣空間。

    直接呼叫 get_klt_matrix.project_to_orthogonal(K)，使用 QR 分解。

    Args:
        K: 任意形狀為 (d, d) 的方陣，float tensor。

    Returns:
        K_orth: 正交矩陣，K_orth.T @ K_orth = I。
    """
    if not isinstance(K, torch.Tensor):
        raise TypeError("K must be torch.Tensor")

    K_orth = _project_to_orthogonal(K)

    overwatch.debug(
        "[KLT] Orthogonalized KLT matrix via QR",
        extra={"shape": tuple(K.shape)},
    )
    return K_orth


def refine_klt_matrix(
    K: torch.Tensor,
    W: torch.Tensor,
) -> torch.Tensor:
    """
    針對給定權重 W，對 K 做一輪小幅度的優化調整。

    直接呼叫 get_klt_matrix.optimize_klt_matrix_3(K, W)。

    使用情境：
        - 你已經有一個初始 K (例如 from eigenvectors / act KLT)，
        - 想在不大改方向的前提下，針對特定權重矩陣做 fine-tuning。

    Args:
        K:
            初始 KLT 矩陣，shape (d, d)。
        W:
            權重矩陣，例如 projector 的 weight，通常 shape (out_dim, d)。

    Returns:
        K_refined: 優化後的 KLT 矩陣，shape 與 K 相同。
    """
    if not isinstance(K, torch.Tensor) or not isinstance(W, torch.Tensor):
        raise TypeError("K and W must be torch.Tensor")

    K_refined = _optimize_klt_matrix_3(K, W)

    overwatch.debug(
        "[KLT] Refined KLT matrix using optimize_klt_matrix_3",
        extra={"K_shape": tuple(K.shape), "W_shape": tuple(W.shape)},
    )
    return K_refined


# =====================================================================
# 由 LLM 權重 / 激活估計 KLT
# =====================================================================

def compute_llm_weight_klt(
    model: nn.Module,
    dataloader,
    *,
    num_samples: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    使用現有 LLM 權重分布估計每層的 KLT 矩陣。

    這是一個「橋接函式」，直接呼叫：
        get_klt_matrix.get_llm_weight_klt(model, dataloader, num_samples)

    Args:
        model:
            一個 LLM 模型（或包含 LLM block 的模組），和 get_klt_matrix 中預期的一致。
        dataloader:
            用於蒐集統計的 DataLoader。實際上 get_llm_weight_klt 目前只跑少量 batch，
            並於內部掛勾各層權重。
        num_samples:
            理論上的樣本數參考值，實際行為依 get_llm_weight_klt 實作而定。

    Returns:
        weight_klt:
            dict[str, torch.Tensor]，key 為層名，value 為對應層的 KLT 矩陣。
    """
    weight_klt = _get_llm_weight_klt(model, dataloader, num_samples=num_samples)

    # 這裡只做輕微型別保證，不改變結構
    for name, K in list(weight_klt.items()):
        if not isinstance(K, torch.Tensor):
            overwatch.warning(
                "[KLT] get_llm_weight_klt returned non-tensor for layer=%r; dropping entry",
                extra={"layer": name},
            )
            weight_klt.pop(name, None)

    overwatch.info(
        "[KLT] Computed LLM weight KLTs for %d layer(s)",
        extra={"num_layers": len(weight_klt)},
    )
    return weight_klt
