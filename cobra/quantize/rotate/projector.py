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
    - Centralize shared-KLT path and loading logic so that both:
        * offline finalize (INT export), and
        * online runtimes (fake / INT backends)
      can use the exact same convention.

Typical usage (inside a CLI like `quant_finalize.py`):

    from cobra.quantize.rotate.projector import (
        ProjectorRotationConfig,
        SHARED_KLT_PATH,
        load_klt_matrix,
        rotate_cobra_vlm_output_projector_inplace,
        rotate_cobra_vlm_output_projector_from_path_inplace,
    )

    cfg = ProjectorRotationConfig(
        use_klt=True,
        use_hadamard=True,
        shared_klt=True,
    )

    # Option A: pass a K matrix explicitly
    rotate_cobra_vlm_output_projector_inplace(
        vlm,
        K=K,
        cfg=cfg,
    )

    # Option B: let this module load K from disk
    rotate_cobra_vlm_output_projector_from_path_inplace(
        vlm,
        klt_path=SHARED_KLT_PATH,
        cfg=cfg,
    )

After these calls, `lm_head.weight` is replaced by the rotated weight
W_out_rot, while activations y_t remain in float.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn

from cobra.overwatch import initialize_overwatch
from cobra.quantize.rotate.klt import apply_klt_rotation
from cobra.quantize.hadamard_utils import matmul_hadU

overwatch = initialize_overwatch(__name__)


# ============================================================================
# Shared KLT path & loader
# ============================================================================

#: Default shared-KLT matrix path.
#:
#: Centralized here so that:
#:   - `quant_klt.py` (estimation),
#:   - `quant_finalize.py` (offline INT export),
#:   - `runtime/load_quantized_vlm.py` (online fake/INT backends)
#: all agree on the same convention.
SHARED_KLT_PATH: Path = Path("/work/asdf1234/cobra_1115/outputs/quantize/shared_klt.pt")


def load_klt_matrix(path: Optional[Path]) -> Optional[torch.Tensor]:
    """
    Load a KLT matrix from disk.

    The file may contain either:
        - a bare Tensor, or
        - a dict with one of the keys {"K", "klt", "KLT"} mapping to a Tensor.

    The matrix is always loaded onto CPU; callers are responsible for moving
    it to the desired device/dtype.

    Parameters
    ----------
    path:
        Path to the KLT file. If None, returns None.

    Returns
    -------
    K:
        Tensor containing the KLT matrix, or None if `path` is None.

    Raises
    ------
    FileNotFoundError:
        If `path` is not None and does not exist.
    TypeError:
        If the file exists but does not contain a suitable Tensor.
    """
    if path is None:
        return None

    if not path.is_file():
        raise FileNotFoundError(f"KLT file not found at: {path}")

    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, dict):
        for key in ("K", "klt", "KLT"):
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key]

    raise TypeError(
        f"KLT file {path} must contain a Tensor or a dict with key 'K' or 'klt'/'KLT'; "
        f"got {type(obj)}"
    )


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ProjectorRotationConfig:
    """
    Configuration for output-projector rotation.

    Flags are intentionally high-level. Bitwidth / quantization mode is
    handled in `finalize/int_export.py` or runtime config modules, not here.

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
      quantization / finalize / runtime pipeline.
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


def rotate_llm_output_projector_from_path_inplace(
    llm_backbone: nn.Module,
    *,
    klt_path: Optional[Path],
    cfg: Optional[ProjectorRotationConfig] = None,
) -> Tuple[str, nn.Linear]:
    """
    Variant of `rotate_llm_output_projector_inplace` that loads K from disk.

    This is a convenience entrypoint for callers that only know the KLT
    file path (e.g., CLIs or runtimes) and do not want to manually handle
    `torch.load`.

    Args
    ----
    llm_backbone:
        LLM backbone wrapper, e.g. `MambaLLMBackbone`.
    klt_path:
        Path to the KLT file (or None to skip KLT).
    cfg:
        ProjectorRotationConfig. If None, a default instance is used.

    Returns
    -------
    (module_path, lm_head_module)
        As in `rotate_llm_output_projector_inplace`.
    """
    if cfg is None:
        cfg = ProjectorRotationConfig()

    K = load_klt_matrix(klt_path) if klt_path is not None else None
    if cfg.use_klt and K is None:
        overwatch.warning(
            "[ProjectorRotate] use_klt=True but loaded KLT matrix is None; "
            "KLT part of rotation will be skipped."
        )

    return rotate_llm_output_projector_inplace(llm_backbone=llm_backbone, K=K, cfg=cfg)


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


def rotate_cobra_vlm_output_projector_from_path_inplace(
    vlm: nn.Module,
    *,
    klt_path: Optional[Path],
    cfg: Optional[ProjectorRotationConfig] = None,
) -> Tuple[str, nn.Linear]:
    """
    Variant of `rotate_cobra_vlm_output_projector_inplace` that loads K from disk.

    This is the natural entrypoint for:
        - `quant_klt.py` + `quant_finalize.py` (offline INT export), and
        - `runtime/load_quantized_vlm.py` (fake / INT runtime),

    when they want to apply projector rotation using a shared KLT matrix saved
    at a well-known location (e.g., `SHARED_KLT_PATH`).

    This function is responsible for:
        1) Loading the KLT matrix from `klt_path` (or deciding to skip KLT),
        2) Calling `rotate_cobra_vlm_output_projector_inplace`,
        3) Handling path / loading errors at this layer, so callers do not
           need to duplicate error-handling logic.

    Args
    ----
    vlm:
        CobraVLM instance (`cobra.models.vlms.cobra.CobraVLM` compatible).
    klt_path:
        Path to the KLT file. If None, KLT will be skipped (depending on cfg).
    cfg:
        ProjectorRotationConfig. If None, a default instance is used.

    Returns
    -------
    (module_path, lm_head_module)
        As in `rotate_llm_output_projector_inplace`.

    Notes
    -----
    - If `cfg.use_klt` is True but the KLT matrix cannot be loaded (missing
      file / invalid contents), this function will log a warning and proceed
      with K=None, so that Hadamard-only rotation can still be applied.
    - Unexpected errors (e.g., torch.load internal failures) are re-raised
      to avoid silently masking serious issues.
    """
    if not hasattr(vlm, "llm_backbone"):
        raise AttributeError(
            "[ProjectorRotate] Expected CobraVLM-like module with `llm_backbone` attribute."
        )

    if cfg is None:
        cfg = ProjectorRotationConfig()

    K: Optional[torch.Tensor] = None

    # Handle KLT loading here so that callers do not need to worry about
    # FileNotFoundError / TypeError, etc.
    if klt_path is not None:
        try:
            K = load_klt_matrix(klt_path)
            overwatch.info(
                "[ProjectorRotate] Loaded shared KLT matrix for projector rotation.",
                extra={
                    "klt_path": str(klt_path),
                    "use_klt": cfg.use_klt,
                },
            )
        except FileNotFoundError:
            if cfg.use_klt:
                overwatch.warning(
                    "[ProjectorRotate] use_klt=True but KLT file not found; "
                    "KLT part of projector rotation will be skipped.",
                    extra={"klt_path": str(klt_path)},
                )
            else:
                overwatch.info(
                    "[ProjectorRotate] KLT file not found but use_klt=False; "
                    "this is expected, proceeding without K.",
                    extra={"klt_path": str(klt_path)},
                )
            K = None
        except TypeError as e:
            # The file exists but contents are not usable as KLT matrix.
            overwatch.warning(
                "[ProjectorRotate] Failed to interpret KLT file contents; "
                "KLT part of projector rotation will be skipped.",
                extra={
                    "klt_path": str(klt_path),
                    "error": str(e),
                },
            )
            K = None
        except Exception as e:
            # Unexpected failure: surface the error instead of silently skipping.
            overwatch.error(
                "[ProjectorRotate] Unexpected error while loading KLT matrix.",
                extra={
                    "klt_path": str(klt_path),
                    "error": str(e),
                },
            )
            raise

    # Delegate to the core in-place rotation on the CobraVLM.
    return rotate_cobra_vlm_output_projector_inplace(
        vlm=vlm,
        K=K,
        cfg=cfg,
    )

