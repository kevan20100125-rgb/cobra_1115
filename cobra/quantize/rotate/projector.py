# cobra/quantize/rotate/projector.py

"""
Projector rotation utilities for the LLM output head (W_out).

This module is responsible for applying the shared rotation

    R = H K

to the LLM's output projection weight matrix (typically `lm_head.weight`),
where:

    - K is a (shared) KLT matrix (from activation/weight statistics).
    - H is a (randomized) Hadamard transform.

Design goals:
    - Only operate on the **output projector** of the LLM
      (MambaForCausalLM.lm_head).
    - Reuse existing implementations:
        * KLT logic from `cobra.quantize.get_klt_matrix` via
          `cobra.quantize.rotate.klt`.
        * Hadamard transform from `cobra.quantize.hadamard_utils`.
    - Keep this file free from percentile / quantizer / finalize logic.
      It only computes and applies rotations on weights.

Typical usage (inside a CLI like `quant_finalize.py`):

    from cobra.quantize.rotate.projector import (
        ProjectorRotationConfig,
        rotate_llm_output_projector_inplace,
    )

    cfg = ProjectorRotationConfig(
        use_klt=True,
        use_hadamard=True,
        shared_klt=True,
    )

    # K: (d_model, d_model) precomputed shared KLT matrix, or None
    rotate_llm_output_projector_inplace(
        llm_backbone=llm_backbone,  # e.g. MambaLLMBackbone
        K=K,
        cfg=cfg,
    )

After this call, `lm_head.weight` is replaced by the rotated weight
W_out_rot, while activations y_t remain in float.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn

from cobra.overwatch import initialize_overwatch
from cobra.quantize.rotate.klt import apply_klt_rotation
from cobra.quantize.hadamard_utils import matmul_hadU

overwatch = initialize_overwatch(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ProjectorRotationConfig:
    """
    Configuration for output-projector rotation.

    Flags are intentionally high-level. Bitwidth / quantization mode is
    handled in `finalize/int_export.py`, not here.

    Attributes
    ----------
    use_klt:
        If True, apply KLT rotation using a shared K matrix (if provided).
    use_hadamard:
        If True, apply a normalized Hadamard transform on the last dimension
        after KLT rotation.
    shared_klt:
        For documentation / sanity only: indicates that the caller intends
        to use a single shared KLT matrix across layers. This module does
        not enforce it, but logs the assumption.
    """

    use_klt: bool = True
    use_hadamard: bool = True
    shared_klt: bool = True


# ============================================================================
# Internal helpers
# ============================================================================


def _locate_lm_head_linear(llm_backbone: nn.Module) -> Tuple[str, nn.Linear]:
    """
    Locate the LLM output projector (lm_head) as an nn.Linear.

    Expected structure (for MambaLLMBackbone):

        llm_backbone.llm: MambaForCausalLM
        llm_backbone.llm.lm_head: nn.Linear(d_model, vocab_size)

    Fallbacks:
        - llm_backbone.lm_head
        - named_modules() entry whose name endswith "lm_head"

    Returns
    -------
    (module_path, module)
        module_path: string describing how we found it (`"llm.lm_head"`, `"lm_head"`, or full name)
        module: nn.Linear instance for the output projector.

    Raises
    ------
    RuntimeError:
        If a suitable lm_head nn.Linear cannot be found.
    """
    # 1) Canonical MambaLLMBackbone.llm.lm_head
    if hasattr(llm_backbone, "llm") and hasattr(llm_backbone.llm, "lm_head"):
        mod = llm_backbone.llm.lm_head
        if isinstance(mod, nn.Linear):
            return "llm.lm_head", mod

    # 2) Direct lm_head attribute
    if hasattr(llm_backbone, "lm_head"):
        mod = getattr(llm_backbone, "lm_head")
        if isinstance(mod, nn.Linear):
            return "lm_head", mod

    # 3) Fallback: search named_modules
    for name, module in llm_backbone.named_modules():
        if name.endswith("lm_head") and isinstance(module, nn.Linear):
            return name, module

    raise RuntimeError(
        "Failed to locate LLM output projector `lm_head` as nn.Linear on backbone. "
        "Expected one of: `.llm.lm_head`, `.lm_head`, or a named submodule ending with 'lm_head'."
    )


def _apply_klt_hadamard_to_weight(
    W: torch.Tensor,
    K: Optional[torch.Tensor],
    cfg: ProjectorRotationConfig,
) -> torch.Tensor:
    """
    Apply KLT + Hadamard rotation to a weight matrix W (usually lm_head.weight).

    Conceptually:
        W_out_rot = (W_out K) H  if both KLT and Hadamard are enabled,
        W_out_rot = W_out K      if only KLT is enabled,
        W_out_rot = W_out H      if only Hadamard is enabled,
        W_out_rot = W_out        if both disabled.

    Implementation detail:
        - KLT: reuses `apply_klt_rotation` from `rotate.klt`, performing W @ K.
        - Hadamard: reuses `matmul_hadU` from `hadamard_utils`, which applies
          a normalized Hadamard transform on the last dimension.

    Args
    ----
    W:
        Weight tensor with shape (vocab_size, d_model) or similar; rotation
        is applied along the last dimension.
    K:
        KLT matrix of shape (d_model, d_model), or None.
    cfg:
        ProjectorRotationConfig.

    Returns
    -------
    W_rot:
        Rotated weight tensor, same shape as W.
    """
    if not isinstance(W, torch.Tensor):
        raise TypeError(f"Expected W to be torch.Tensor, got {type(W)}")

    W_rot = W

    # --- KLT rotation (W @ K) ---
    if cfg.use_klt:
        if K is None:
            overwatch.warning(
                "[ProjectorRotate] KLT enabled but no K matrix provided; "
                "skipping KLT rotation for output projector."
            )
        else:
            if K.ndim != 2 or K.shape[0] != K.shape[1]:
                raise ValueError(
                    f"[ProjectorRotate] Expected square KLT matrix, got shape={tuple(K.shape)}"
                )
            if K.shape[0] != W.shape[-1]:
                raise ValueError(
                    f"[ProjectorRotate] KLT dimension mismatch: "
                    f"K.shape={tuple(K.shape)}, W.shape={tuple(W.shape)} "
                    f"(last dim must match)."
                )
            # Ensure device/dtype alignment
            K_use = K.to(device=W.device, dtype=W.dtype)
            W_rot = apply_klt_rotation(W_rot, K_use, right_multiply=True)

    # --- Hadamard rotation (on last dimension) ---
    if cfg.use_hadamard:
        # matmul_hadU applies a normalized Hadamard transform on the last dim.
        W_rot = matmul_hadU(W_rot)
        overwatch.debug(
            "[ProjectorRotate] Applied Hadamard transform to output projector weight.",
            extra={"shape": tuple(W.shape)},
        )

    return W_rot


# ============================================================================
# Public API
# ============================================================================


def rotate_llm_output_projector_inplace(
    llm_backbone: nn.Module,
    *,
    K: Optional[torch.Tensor] = None,
    cfg: Optional[ProjectorRotationConfig] = None,
) -> Tuple[str, nn.Linear]:
    """
    Apply shared R = H K rotation to the LLM output projector (lm_head) in-place.

    This function:
        1. Locates the nn.Linear lm_head on the given LLM backbone.
        2. Computes W_out_rot via `_apply_klt_hadamard_to_weight`.
        3. Overwrites `lm_head.weight` with the rotated weight.

    Args
    ----
    llm_backbone:
        LLM backbone wrapper, e.g. `MambaLLMBackbone`. Must expose an
        underlying `MambaForCausalLM` whose `lm_head` is an nn.Linear, or
        else have its own `lm_head` attribute, or a named submodule ending
        with "lm_head".
    K:
        Shared KLT matrix of shape (d_model, d_model), or None if KLT is
        disabled / not yet available.
    cfg:
        ProjectorRotationConfig. If None, a default instance is used.

    Returns
    -------
    (module_path, lm_head_module)
        module_path:
            String path used to locate the lm_head (for logging/debugging).
        lm_head_module:
            The nn.Linear module whose weight has been rotated in-place.

    Notes
    -----
    - This function ONLY modifies `lm_head.weight`. Activations y_t remain
      float and unrotated, matching the design "quantized x_t → Mamba →
      float y_t → quantized, rotated W_out".
    - It does not attach any quantization modules; those are handled by the
      quantization / finalize pipeline.
    """
    if cfg is None:
        cfg = ProjectorRotationConfig()

    module_path, lm_head = _locate_lm_head_linear(llm_backbone)

    W = lm_head.weight.data
    W_rot = _apply_klt_hadamard_to_weight(W, K, cfg)

    lm_head.weight.data.copy_(W_rot)

    overwatch.info(
        "[ProjectorRotate] Rotated LLM output projector in-place.",
        extra={
            "module_path": module_path,
            "use_klt": cfg.use_klt,
            "use_hadamard": cfg.use_hadamard,
            "shared_klt": cfg.shared_klt,
            "orig_shape": tuple(W.shape),
        },
    )

    return module_path, lm_head


def rotate_cobra_vlm_output_projector_inplace(
    vlm: nn.Module,
    *,
    K: Optional[torch.Tensor] = None,
    cfg: Optional[ProjectorRotationConfig] = None,
) -> Tuple[str, nn.Linear]:
    """
    Convenience wrapper for rotating the LLM output projector of a CobraVLM.

    Expected structure:

        vlm.llm_backbone: MambaLLMBackbone
        vlm.llm_backbone.llm.lm_head: nn.Linear

    Args
    ----
    vlm:
        CobraVLM instance (`cobra.models.vlms.cobra.CobraVLM` compatible).
    K:
        Shared KLT matrix (d_model, d_model), or None.
    cfg:
        ProjectorRotationConfig. If None, uses default.

    Returns
    -------
    (module_path, lm_head_module)
        As in `rotate_llm_output_projector_inplace`.

    Raises
    ------
    AttributeError:
        If `vlm` does not expose `llm_backbone` attribute.
    """
    if not hasattr(vlm, "llm_backbone"):
        raise AttributeError(
            "[ProjectorRotate] Expected CobraVLM-like module with `llm_backbone` attribute."
        )

    llm_backbone = vlm.llm_backbone
    return rotate_llm_output_projector_inplace(llm_backbone=llm_backbone, K=K, cfg=cfg)
