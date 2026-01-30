"""
Block-wise Hadamard rotation utilities for Mamba mixer linear projections.

Hadamard-only:
  - Replace selected mixer nn.Linear modules with a wrapper that:
      (1) applies block-wise Hadamard to input activations (last dim)
      (2) calls the underlying linear (which may later be wrapped into QuantLinear)
  - Rotate the underlying linear weights with the same block-wise Hadamard on
    the in_features dimension (last dim of weight).

Design goals:
  - Keep this module independent from percentile/KLT logic.
  - Be robust to minor differences in mamba_ssm mixer attribute naming by using
    runtime introspection and pattern matching on submodule names.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import math
import functools

import torch
from torch import nn
import torch.nn.functional as F

from cobra.overwatch import initialize_overwatch
from cobra.quantize.hadamard_utils import matmul_hadU
from cobra.quantize.wrap.utils import replace_module_inplace

overwatch = initialize_overwatch(__name__)


@dataclass
class MixerHadamardRotationConfig:
    enabled: bool = False

    # What to rotate
    block_size: int = 512
    targets: Tuple[str, ...] = ("out_proj",)
    max_layers: Optional[int] = None

    # Control
    dry_run: bool = False

    # === act-KLT integration ===
    # If enabled, apply layer-wise block KLT in addition to Hadamard:
    #   Quamba-style hooks:
    #     input_transform : x <- x @ K_in ; then x <- x @ H
    #     output_transform: y <- y @ K_out; then y <- y @ H
    #
    # Weight update (to preserve function):
    #     W' = R_out @ W @ R_in^T
    #     R_in  = K_in  @ H
    #     R_out = K_out @ H
    # Therefore:
    #     right-multiply:  @ (R_in^T) = @ (H @ K_in^T)
    #     left-multiply :  (R_out @)  = (K_out @ H) @
    act_klt_enabled: bool = False

    # Optional explicit paths (override env/default)
    act_klt_in_path: Optional[str] = None
    act_klt_out_path: Optional[str] = None

    # If True: missing layer/block in act-klt payload is a hard error (except layer0 OUT-KLT skip)
    act_klt_strict: bool = False

    # Enable Quamba-like output_transform (OUT side). If False, OUT KLT/H is not applied.
    out_transform_enabled: bool = False

    # Required behavior: skip OUT-KLT on layer 0 (still allow OUT-Hadamard if out_transform_enabled).
    out_klt_skip_layer0: bool = True

    # Parity / sanity
    parity_check: bool = False
    parity_num_samples: int = 2
    parity_batch: int = 2

    # Route-A: tolerate FWHT float32 reduction-tree differences.
    # Gate is used to label status ("ok"/"warn"), NOT to stop execution.
    parity_cos_tol: float = 0.999
    parity_rel_tol: float = 3.0e-3
    parity_max_tol: float = 5.0e-2

    # To keep runtime manageable when scaling to many layers:
    # If parity_check=True, only run parity on first N applied modules (typically 1~4).
    parity_check_first_n: int = 1


def _parse_targets(targets: Sequence[str]) -> Tuple[str, ...]:
    out: List[str] = []
    for t in targets:
        s = str(t).strip()
        if not s:
            continue
        out.append(s)
    return tuple(out) if out else ("out_proj",)


def blockwise_hadamard(x: torch.Tensor, *, block_size: int) -> torch.Tensor:
    """Apply normalized Hadamard on last dim in block-wise fashion."""
    if x is None:
        raise ValueError("blockwise_hadamard: x is None")
    if x.ndim == 0:
        return x
    n = x.shape[-1]
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if n % block_size != 0:
        raise ValueError(f"Last dim {n} is not divisible by block_size {block_size}")
    if n == block_size:
        return matmul_hadU(x)

    # reshape (..., nblocks, block) and apply hadamard on last dim
    nblocks = n // block_size
    y = x.view(*x.shape[:-1], nblocks, block_size)
    y = matmul_hadU(y)
    return y.view_as(x)


@torch.no_grad()
def apply_blockwise_hadamard_to_weight_inplace(
    W: torch.Tensor,
    *,
    block_size: int,
    normalized: bool = True,
    compute_dtype: torch.dtype = torch.float32,
) -> None:
    """
    In-place right-rotation on Linear weight:
        W <- W @ H
    where H is block-diagonal Hadamard (block_size x block_size), repeated over in_features.

    Key properties:
      - Rotation is applied along the *in_features* axis (right-multiply), which matches x_rot = x @ H in forward.
      - Compute is done in fp32 by default to reduce FWHT numerical error.
      - Result is cast back to W.dtype and copied into W in-place.

    Assumptions:
      - W is shaped [out_features, in_features] (PyTorch nn.Linear weight convention).
      - in_features divisible by block_size.
      - Hadamard transform kernel matmul_hadU applies FWHT over the last dimension, and returns normalized FWHT
        (i.e. includes 1/sqrt(n)). If normalized=False, we undo that normalization.
    """
    import math
    import torch

    from cobra.quantize.hadamard_utils import matmul_hadU

    if W.ndim != 2:
        raise ValueError(f"Expected 2D weight [out,in], got shape={tuple(W.shape)}")

    out_f, in_f = W.shape
    B = int(block_size)
    if B <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if in_f % B != 0:
        raise ValueError(f"in_features={in_f} not divisible by block_size={B}")

    # Compute in fp32 for stability (very important for weight-match sanity).
    W_comp = W.to(dtype=compute_dtype)
    nb = in_f // B

    # Right-multiply by block-diag(H): reshape to [out, nb, B] and apply FWHT on last dim (B).
    # matmul_hadU is normalized by 1/sqrt(B) (per your hadamard_utils implementation).
    W3 = W_comp.reshape(out_f, nb, B).contiguous()
    W3r = matmul_hadU(W3)

    # If caller wants unnormalized Hadamard, scale back by sqrt(B).
    if not bool(normalized):
        W3r = W3r * math.sqrt(B)

    W_rot = W3r.reshape(out_f, in_f)

    # Copy back to original dtype/device in-place.
    if W_rot.dtype != W.dtype:
        W_rot = W_rot.to(dtype=W.dtype)
    W.copy_(W_rot)

class BlockwiseHadamardInputWrapper(nn.Module):
    """A thin wrapper that applies block-wise Hadamard to input then calls `linear`."""

    def __init__(self, linear: nn.Module, *, block_size: int):
        super().__init__()
        self.linear = linear
        self.block_size = int(block_size)
        self._cobra_hadamard_wrapped = True

        # Best-effort metadata for repr and downstream code
        self.in_features = getattr(linear, "in_features", None)
        self.out_features = getattr(linear, "out_features", None)

    @property
    def weight(self):
        return getattr(self.linear, "weight", None)

    @property
    def bias(self):
        return getattr(self.linear, "bias", None)

    def extra_repr(self) -> str:
        return f"block_size={self.block_size}, linear={self.linear.__class__.__name__}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rot = blockwise_hadamard(x, block_size=self.block_size)
        return self.linear(x_rot)


def _iter_mamba_blocks(llm: nn.Module) -> Iterable[Tuple[int, nn.Module]]:
    """Yield (layer_idx, block_module) for llm.backbone.layers."""
    backbone = getattr(llm, "backbone", None)
    if backbone is None:
        return
    layers = getattr(backbone, "layers", None)
    if layers is None:
        return
    for i, blk in enumerate(layers):
        yield i, blk


def find_mixer_linears(
    llm: nn.Module,
    *,
    targets: Sequence[str],
    max_layers: Optional[int] = None,
    strict_out_proj_top_level: bool = True,
) -> List[Tuple[str, nn.Linear]]:
    """
    Find mixer submodule linears by name match.

    IMPORTANT (robustness):
      - mixer.named_modules() will recursively return nested modules, so matching by leaf
        name (e.g. "out_proj") can accidentally pick mixer.<something>.out_proj.
      - In Cobra's Mamba mixer, runtime rotation is intended to apply to the *top-level*
        mixer.out_proj (i.e. exactly mixer.out_proj), which may have in_features=2*hidden_size (e.g. 5120).
      - If we accidentally rotate a nested out_proj (often 2560), remember/export paths and runtime paths diverge.

    Args:
      targets: module leaf names to match (e.g. ("out_proj",))
      max_layers: optional layer cap
      strict_out_proj_top_level:
        - If True, only match the top-level "out_proj" under mixer:
          either (name == "out_proj") OR (module is getattr(mixer, "out_proj")).
        - This prevents accidental matches to nested "...out_proj".

    Returns:
      List of (module_path_under_llm, linear_module).
    """
    wanted = set(_parse_targets(targets))
    found: List[Tuple[str, nn.Linear]] = []

    for layer_idx, blk in _iter_mamba_blocks(llm):
        if max_layers is not None and layer_idx >= max_layers:
            break
        mixer = getattr(blk, "mixer", None)
        if mixer is None:
            continue

        mixer_out_proj = getattr(mixer, "out_proj", None)

        for name, mod in mixer.named_modules():
            if not isinstance(mod, nn.Linear):
                continue

            leaf = name.split(".")[-1] if name else ""
            if leaf not in wanted:
                continue

            # Robustly restrict out_proj to the runtime-equivalent top-level module
            if strict_out_proj_top_level and leaf == "out_proj":
                is_top_level_by_name = (name == "out_proj")
                is_top_level_by_identity = (mixer_out_proj is not None and mod is mixer_out_proj)
                if not (is_top_level_by_name or is_top_level_by_identity):
                    continue

            # Build absolute path under llm
            if name:
                module_path = f"backbone.layers.{layer_idx}.mixer.{name}"
            else:
                module_path = f"backbone.layers.{layer_idx}.mixer"

            found.append((module_path, mod))

    return found

def _parity_metrics(y_ref: torch.Tensor, y_rot: torch.Tensor) -> Dict[str, float]:
    """
    Compute simple parity metrics in fp32 for stability.
    Returns: cos_sim, rel_err, max_abs
    """
    a = y_ref.detach().float().reshape(-1)
    b = y_rot.detach().float().reshape(-1)

    denom = (a.norm() * b.norm()).clamp_min(1e-12)
    cos_sim = float((a @ b) / denom)

    diff = (a - b).abs()
    max_abs = float(diff.max().item()) if diff.numel() else 0.0

    ref_max = float(a.abs().max().item()) if a.numel() else 0.0
    rel_err = float(max_abs / (ref_max + 1e-12))

    return {"cos_sim": cos_sim, "rel_err": rel_err, "max_abs": max_abs}


def _hadamard_base2(dtype: torch.dtype = torch.float32, device: Optional[torch.device] = None) -> torch.Tensor:
    # Sylvester base matrix [[1, 1], [1, -1]]
    return torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=dtype, device=device)


@functools.lru_cache(maxsize=32)
def _get_hadamard_matrix_cpu(n: int, normalized: bool) -> torch.Tensor:
    """
    Build Hadamard matrix H_n on CPU in float32 using Sylvester construction.
    n must be a power of 2.
    """
    if n <= 0 or (n & (n - 1)) != 0:
        raise ValueError(f"Hadamard size must be power-of-2, got n={n}")

    H = _hadamard_base2(dtype=torch.float32, device=torch.device("cpu"))
    # If n == 2, done; else kron up.
    k = int(math.log2(n))
    for _ in range(1, k):
        H = torch.kron(H, _hadamard_base2(dtype=torch.float32, device=torch.device("cpu")))

    if normalized:
        H = H * (1.0 / math.sqrt(n))
    return H


def _get_hadamard_matrix(n: int, *, normalized: bool, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Fetch cached CPU H and move/cast to requested device/dtype.
    """
    H_cpu = _get_hadamard_matrix_cpu(int(n), bool(normalized))
    # Keep parity math stable: fp32 is fine; but caller may request dtype.
    return H_cpu.to(device=device, dtype=dtype)


def _apply_blockdiag_right(W: torch.Tensor, H_block: torch.Tensor) -> torch.Tensor:
    """
    Compute W @ blockdiag(H_block, ..., H_block) where H_block is [B, B] and
    W is [out_features, in_features], with in_features divisible by B.
    Returns a new tensor.
    """
    if W.ndim != 2:
        raise ValueError(f"W must be 2D [out,in], got shape={tuple(W.shape)}")
    out_f, in_f = W.shape
    B = H_block.shape[0]
    if H_block.shape != (B, B):
        raise ValueError(f"H_block must be square, got shape={tuple(H_block.shape)}")
    if in_f % B != 0:
        raise ValueError(f"in_features={in_f} not divisible by block={B}")

    nb = in_f // B
    # [out, nb, B] @ [B, B] -> [out, nb, B]
    W3 = W.reshape(out_f, nb, B)
    W3r = torch.matmul(W3, H_block)
    return W3r.reshape(out_f, in_f)


def _cosine_sim_flat(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    a1 = a.reshape(-1)
    b1 = b.reshape(-1)
    num = torch.dot(a1, b1)
    den = (a1.norm() * b1.norm()).clamp_min(eps)
    return float((num / den).item())


def _relative_fro_err(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> float:
    # ||a-b||_F / (||b||_F + eps)
    diff = (a - b).reshape(-1)
    ref = b.reshape(-1)
    num = diff.norm()
    den = ref.norm().clamp_min(eps)
    return float((num / den).item())


def _parity_diagnostic_wrong_side(
    *,
    W0: torch.Tensor,
    b0: Optional[torch.Tensor],
    Wr: torch.Tensor,
    br: Optional[torch.Tensor],
    block_size: int,
    num_samples: int,
    batch: int,
    device: torch.device,
    dtype: torch.dtype,
    gen: torch.Generator,
) -> Dict[str, float]:
    """
    REDEFINED (v2 sanity):
    The previous implementation attempted to synthesize a left-rotated candidate weight and was flawed.

    New diagnostic:
      Compare Wr against the analytically expected right-rotation:
        - W0 @ H_norm   (H normalized by 1/sqrt(B))
        - W0 @ H_unnorm (no normalization)
      This diagnoses normalization mismatches and/or axis mistakes (at least for right-rotation hypothesis).

    Returns fields compatible with existing log formatter:
      - diag_best_case: 0.0 => norm closer, 1.0 => unnorm closer
      - diag_best_rel_err: relative Fro error of best match
      - diag_best_cos_sim: cosine similarity of best match
    Also returns extra debug fields:
      - wmatch_rel_norm / wmatch_cos_norm / wmatch_max_norm
      - wmatch_rel_unnorm / wmatch_cos_unnorm / wmatch_max_unnorm
    """
    # Weight-match sanity is independent of b0/br (bias unaffected by input rotation).
    out_f, in_f = W0.shape
    B = int(block_size)
    if in_f % B != 0:
        return {
            "diag_best_case": -1.0,
            "diag_best_rel_err": 1e9,
            "diag_best_cos_sim": -1.0,
        }

    # Build H on the same device/dtype as parity math (usually fp32 on GPU).
    H_norm = _get_hadamard_matrix(B, normalized=True, device=device, dtype=dtype)
    H_unnorm = _get_hadamard_matrix(B, normalized=False, device=device, dtype=dtype)

    Wexp_norm = _apply_blockdiag_right(W0, H_norm)
    Wexp_unnorm = _apply_blockdiag_right(W0, H_unnorm)

    rel_norm = _relative_fro_err(Wr, Wexp_norm)
    rel_unnorm = _relative_fro_err(Wr, Wexp_unnorm)

    cos_norm = _cosine_sim_flat(Wr, Wexp_norm)
    cos_unnorm = _cosine_sim_flat(Wr, Wexp_unnorm)

    max_norm = float((Wr - Wexp_norm).abs().max().item())
    max_unnorm = float((Wr - Wexp_unnorm).abs().max().item())

    if rel_norm <= rel_unnorm:
        best_case = 0.0
        best_rel = rel_norm
        best_cos = cos_norm
    else:
        best_case = 1.0
        best_rel = rel_unnorm
        best_cos = cos_unnorm

    return {
        "diag_best_case": float(best_case),
        "diag_best_rel_err": float(best_rel),
        "diag_best_cos_sim": float(best_cos),
        "wmatch_rel_norm": float(rel_norm),
        "wmatch_cos_norm": float(cos_norm),
        "wmatch_max_norm": float(max_norm),
        "wmatch_rel_unnorm": float(rel_unnorm),
        "wmatch_cos_unnorm": float(cos_unnorm),
        "wmatch_max_unnorm": float(max_unnorm),
    }

def _run_linear_parity_sanity(
    *,
    W_orig: torch.Tensor,
    b_orig: Optional[torch.Tensor],
    W_rot: torch.Tensor,
    b_rot: Optional[torch.Tensor],
    block_size: int,
    num_samples: int,
    batch: int,
) -> Dict[str, float]:
    """
    Local Hadamard-only sanity check on a single Linear weight.

    y_ref = linear(x, W_orig, b_orig)
    y_rot = linear(H(x), W_rot, b_rot)  where ideally W_rot = W_orig @ H  (right-rotation on in_features)

    Parity math is executed in fp32 to avoid fp16/bf16 false alarms.
    Additionally, we perform a *weight-match* diagnostic:
      Compare W_rot against (W_orig @ H_norm) and (W_orig @ H_unnorm) to detect normalization mismatch.
    """
    device = W_rot.device
    dtype = torch.float32
    in_features = int(W_rot.shape[1])

    gen = torch.Generator(device=device)
    gen.manual_seed(0)

    metrics_accum = {"cos_sim": 0.0, "rel_err": 0.0, "max_abs": 0.0}
    ns = max(int(num_samples), 1)
    bs = max(int(batch), 1)

    W0 = W_orig.to(device=device, dtype=dtype)
    b0 = b_orig.to(device=device, dtype=dtype) if b_orig is not None else None
    Wr = W_rot.to(device=device, dtype=dtype)
    br = b_rot.to(device=device, dtype=dtype) if b_rot is not None else None

    for _ in range(ns):
        x = torch.randn(bs, in_features, device=device, dtype=dtype, generator=gen)
        y_ref = F.linear(x, W0, b0)
        x_rot = blockwise_hadamard(x, block_size=block_size)
        y_rot = F.linear(x_rot, Wr, br)

        m = _parity_metrics(y_ref, y_rot)
        for k in ("cos_sim", "rel_err", "max_abs"):
            metrics_accum[k] += float(m[k])

    for k in ("cos_sim", "rel_err", "max_abs"):
        metrics_accum[k] /= float(ns)

    # Weight-match diagnostic: replaces the previous flawed "wrong-side" synthesis.
    diag = _parity_diagnostic_wrong_side(
        W0=W0,
        b0=b0,
        Wr=Wr,
        br=br,
        block_size=int(block_size),
        num_samples=ns,
        batch=bs,
        device=device,
        dtype=dtype,
        gen=gen,
    )
    # Keep backward-compatible fields (diag_best_*) and add richer debug info.
    metrics_accum.update(diag)
    return metrics_accum


def _emit_mixer_hadamard_summary(
    report: Dict[str, object],
    *,
    targets: Tuple[str, ...],
    block_size: int,
    max_layers: Optional[int],
    max_list: int = 8,
) -> None:
    """
    Emit a log summary that is self-diagnosing.

    Prints:
      - applied/skipped counts and config
      - for applied modules: parity (cos/rel/max) plus diag_best_* and (if present) weight-match fields
        wmatch_rel_norm/unnorm and wmatch_max_norm/unnorm.
    """
    applied = report.get("applied", []) or []
    skipped = report.get("skipped", []) or []

    overwatch.info(
        f"[MixerHadamard] Applied={len(applied)}, Skipped={len(skipped)}, "
        f"targets={targets}, block={block_size}, max_layers={max_layers}, "
        f"dry_run={bool(report.get('dry_run'))}, parity_check={bool(report.get('parity_check'))}"
    )

    for i, rec in enumerate(applied[:max_list]):
        mp = rec.get("module_path", "<unknown>")
        note = rec.get("note")
        in_f = rec.get("in_features")
        out_f = rec.get("out_features")

        if note:
            overwatch.info(f"[MixerHadamard][applied:{i}] {mp} (in={in_f}, out={out_f}) note={note}")
            continue

        parity = rec.get("parity")
        if not isinstance(parity, dict):
            overwatch.info(f"[MixerHadamard][applied:{i}] {mp} (in={in_f}, out={out_f})")
            continue

        msg = (
            f"[MixerHadamard][applied:{i}] {mp} (in={in_f}, out={out_f}) "
            f"parity cos={parity.get('cos_sim'):.6f} rel={parity.get('rel_err'):.6e} max={parity.get('max_abs'):.6e}"
        )

        # diag_best_* (always printed if present)
        if "diag_best_case" in parity:
            msg += (
                f" diag_best_case={parity.get('diag_best_case')} "
                f"diag_best_rel={parity.get('diag_best_rel_err'):.6e} "
                f"diag_best_cos={parity.get('diag_best_cos_sim'):.6f}"
            )

        # weight-match debug fields (printed if present)
        if "wmatch_rel_norm" in parity:
            msg += (
                f" wmatch_rel_norm={parity.get('wmatch_rel_norm'):.6e}"
                f" wmatch_max_norm={parity.get('wmatch_max_norm'):.6e}"
            )
        if "wmatch_rel_unnorm" in parity:
            msg += (
                f" wmatch_rel_unnorm={parity.get('wmatch_rel_unnorm'):.6e}"
                f" wmatch_max_unnorm={parity.get('wmatch_max_unnorm'):.6e}"
            )

        overwatch.info(msg)

    if len(applied) > max_list:
        overwatch.info(f"[MixerHadamard] ... {len(applied) - max_list} more applied entries omitted")

    for i, rec in enumerate(skipped[:max_list]):
        mp = rec.get("module_path", "<unknown>")
        rsn = rec.get("reason", "<unknown>")
        overwatch.info(f"[MixerHadamard][skipped:{i}] {mp} reason={rsn}")

    if len(skipped) > max_list:
        overwatch.info(f"[MixerHadamard] ... {len(skipped) - max_list} more skipped entries omitted")


@torch.no_grad()
def rotate_llm_mamba_mixers_hadamard_inplace(llm: nn.Module, *, cfg: MixerHadamardRotationConfig) -> Dict[str, object]:
    """
    Apply Quamba-style KLT+Hadamard to Mamba mixer out_proj, with mathematically closed-loop HK-IN and HK-OUT.

    This implementation is **self-contained** and uses only repo-existing imports/helpers:
      - initialize_overwatch (module-level)
      - matmul_hadU (module-level import)
      - blockwise_hadamard / apply_blockwise_hadamard_to_weight_inplace (defined in this file)

    Closed-loop convention used here matches this file's hooks:

      pre-hook (IN):
        x' = x @ R_in
        R_in = K_in @ H

      post-hook (OUT) [if enabled]:
        y  = y' @ R_out^T
        R_out = K_out @ H
        => R_out^T = H @ K_out^T   (Hadamard symmetric)

      PyTorch Linear: y' = x' @ W_rot^T

      To preserve function: y = x @ W^T
        Choose: W_rot = R_out^T @ W @ R_in

      Therefore weight transforms must be:
        Right (IN):   W <- W @ R_in     = (W @ K_in) then ( @ H)
        Left  (OUT):  W <- R_out^T @ W  = (K_out^T @ W) then (H @ W)

    Robustness goals:
      - CUDA-graph friendly: no `.to()` inside hooks (materialize buffers at install)
      - Fallback: on mismatch, disable that KLT buffer and continue Hadamard-only
    """
    # NOTE: do NOT import repo-nonexistent symbols here.
    # We rely on module-level imports (torch/nn/overwatch/matmul_hadU) and local helpers in this file.

    applied: List[Dict[str, object]] = []
    skipped: List[Dict[str, object]] = []

    if not bool(getattr(cfg, "enabled", False)):
        return {"applied": [], "skipped": [{"module_path": "<llm>", "reason": "cfg.enabled=0"}]}

    # -----------------------------
    # Resolve layers
    # -----------------------------
    layers = None
    if hasattr(llm, "backbone") and hasattr(llm.backbone, "layers"):
        layers = llm.backbone.layers
    elif hasattr(llm, "layers"):
        layers = llm.layers

    if layers is None:
        return {"applied": [], "skipped": [{"module_path": "<llm>", "reason": "cannot locate llm layers"}]}

    max_layers = cfg.max_layers if cfg.max_layers is not None else len(layers)
    max_layers = min(max_layers, len(layers))
    B = int(cfg.block_size)

    # Targets
    targets = tuple(getattr(cfg, "targets", ("out_proj",)) or ("out_proj",))
    targets = tuple([t.strip() for t in targets if str(t).strip()])

    # OUT transform settings
    out_tx = bool(getattr(cfg, "out_transform_enabled", False))
    out_skip0 = bool(getattr(cfg, "out_klt_skip_layer0", True))

    # KLT dtype policy (buffer dtype for runtime hooks)
    klt_dtype_env = os.environ.get("COBRA_LLM_MIXER_KLT_DTYPE", "fp32").strip().lower()
    if klt_dtype_env in ("fp16", "float16", "half"):
        klt_dtype = torch.float16
    else:
        klt_dtype = torch.float32

    # -----------------------------
    # Load KLT payloads (cpu fp32)
    # -----------------------------
    def _resolve_in_klt_path() -> Optional[str]:
        p = getattr(cfg, "act_klt_in_path", None) or getattr(cfg, "act_klt_path", None)
        if p and str(p).strip():
            return str(p).strip()

        env_p = os.environ.get("ACT_KLT_OUT_IN", "").strip()
        if env_p:
            return env_p

        env_p2 = os.environ.get("ACT_KLT_OUTPROJ_IN", "").strip()
        if env_p2:
            return env_p2

        # legacy
        env_legacy = os.environ.get("ACT_KLT_OUT", "").strip()
        if env_legacy:
            return env_legacy

        # default
        B0 = int(cfg.block_size)
        return f"outputs/quantize/act_klt_outproj_in_bs{B0}/act_klt_outproj_in.pt"

    def _resolve_out_klt_path() -> Optional[str]:
        p = getattr(cfg, "act_klt_out_path", None)
        if p and str(p).strip():
            return str(p).strip()

        env_p = os.environ.get("ACT_KLT_OUT_OUT", "").strip()
        if env_p:
            return env_p

        env_p2 = os.environ.get("ACT_KLT_OUTPROJ_OUT", "").strip()
        if env_p2:
            return env_p2

        # default
        B0 = int(cfg.block_size)
        return f"outputs/quantize/act_klt_outproj_out_bs{B0}/act_klt_outproj_out.pt"

    k_in_layers: Dict[int, torch.Tensor] = {}
    k_out_layers: Dict[int, torch.Tensor] = {}

    def _load_klt_payload(path: str) -> Dict[int, torch.Tensor]:
        payload = torch.load(path, map_location="cpu")
        layers_d = payload.get("layers", {}) if isinstance(payload, dict) else {}
        norm_layers: Dict[int, torch.Tensor] = {}
        if isinstance(layers_d, dict):
            for k, v in layers_d.items():
                try:
                    li = int(k)
                except Exception:
                    continue
                if not torch.is_tensor(v):
                    continue
                norm_layers[li] = v.detach().to(dtype=torch.float32, device=torch.device("cpu"))
        return norm_layers

    if bool(cfg.act_klt_enabled):
        in_path = _resolve_in_klt_path()
        if not in_path:
            msg = "[MixerHK] act_klt_enabled=1 but IN-KLT path cannot be resolved"
            if cfg.act_klt_strict:
                raise FileNotFoundError(msg)
            overwatch.warning(msg)
        else:
            k_in_layers = _load_klt_payload(in_path)
            overwatch.info(f"[MixerHK] Loaded IN act-KLT: path={in_path!r}, layers={len(k_in_layers)}")

        if out_tx:
            out_path = _resolve_out_klt_path()
            if not out_path:
                msg = "[MixerHK] out_transform_enabled=1 but OUT-KLT path cannot be resolved"
                if cfg.act_klt_strict:
                    raise FileNotFoundError(msg)
                overwatch.warning(msg)
            else:
                k_out_layers = _load_klt_payload(out_path)
                overwatch.info(f"[MixerHK] Loaded OUT act-KLT: path={out_path!r}, layers={len(k_out_layers)}")

    def _validate_k_blocks(
        kb: torch.Tensor,
        *,
        B: int,
        dim: int,
        layer_idx: int,
        module_path: str,
        tag: str,
    ) -> Optional[torch.Tensor]:
        if kb.ndim != 3 or tuple(kb.shape[-2:]) != (B, B):
            overwatch.warning(
                f"[MixerHK] {tag} KLT shape invalid layer={layer_idx} module={module_path}: "
                f"expected [nb,{B},{B}] got {tuple(kb.shape)}"
            )
            return None
        nb = int(kb.shape[0])
        rot_dim = nb * B
        if dim < rot_dim:
            overwatch.warning(
                f"[MixerHK] {tag} DIM MISMATCH layer={layer_idx} module={module_path}: "
                f"dim={dim} < rot_dim={rot_dim} (nb={nb}, B={B})"
            )
            return None
        return kb

    # -----------------------------
    # Blockwise ops (activation / weight)
    # -----------------------------
    def _blockwise_apply_klt_prefix(x: torch.Tensor, *, k_blocks: torch.Tensor, transpose: bool) -> torch.Tensor:
        D = int(x.shape[-1])
        nb, Bb, Bb2 = map(int, k_blocks.shape)
        if Bb != Bb2:
            raise ValueError(f"k_blocks must be [nb,B,B], got {tuple(k_blocks.shape)}")
        if D != nb * Bb:
            raise ValueError(f"x last dim={D} mismatched with k_blocks nb*B={nb*Bb}")
        x3 = x.view(*x.shape[:-1], nb, Bb)
        K = k_blocks.transpose(-2, -1) if transpose else k_blocks
        y3 = torch.einsum("...ib,ibc->...ic", x3, K)
        return y3.reshape_as(x)

    @torch.no_grad()
    def _apply_klt_right_prefix_inplace(W: torch.Tensor, *, k_blocks: torch.Tensor, transpose: bool) -> None:
        out_f, in_f = map(int, W.shape)
        nb, Bb, Bb2 = map(int, k_blocks.shape)
        if Bb != Bb2:
            raise ValueError(f"k_blocks must be [nb,B,B], got {tuple(k_blocks.shape)}")
        rot_dim = nb * Bb
        if in_f < rot_dim:
            raise ValueError(f"W in_features={in_f} < rot_dim={rot_dim}")

        Wf = W.to(dtype=torch.float32)
        K = k_blocks.to(device=Wf.device, dtype=torch.float32)
        if transpose:
            K = K.transpose(-2, -1)

        Wp = Wf[:, :rot_dim].contiguous().reshape(out_f, nb, Bb)
        Wp_r = torch.einsum("oib,ibc->oic", Wp, K).reshape(out_f, rot_dim)
        Wf[:, :rot_dim] = Wp_r
        W.copy_(Wf.to(dtype=W.dtype))

    @torch.no_grad()
    def _apply_klt_left_prefix_inplace(W: torch.Tensor, *, k_blocks: torch.Tensor, transpose: bool) -> None:
        out_f, in_f = map(int, W.shape)
        nb, Bb, Bb2 = map(int, k_blocks.shape)
        if Bb != Bb2:
            raise ValueError(f"k_blocks must be [nb,B,B], got {tuple(k_blocks.shape)}")
        rot_dim = nb * Bb
        if out_f < rot_dim:
            raise ValueError(f"W out_features={out_f} < rot_dim={rot_dim}")

        Wf = W.to(dtype=torch.float32)
        K = k_blocks.to(device=Wf.device, dtype=torch.float32)
        if transpose:
            K = K.transpose(-2, -1)

        Wr = Wf[:rot_dim, :].contiguous().reshape(nb, Bb, in_f)
        Wr2 = torch.einsum("ibc,icn->ibn", K, Wr)
        Wf[:rot_dim, :] = Wr2.reshape(rot_dim, in_f)
        W.copy_(Wf.to(dtype=W.dtype))

    def _apply_hadamard_right_prefix_inplace(W: torch.Tensor, *, rot_dim: int, block_size: int) -> None:
        out_f, in_f = map(int, W.shape)
        Bb = int(block_size)
        if rot_dim % Bb != 0:
            raise ValueError(f"rot_dim={rot_dim} not divisible by block={Bb}")
        if in_f < rot_dim:
            raise ValueError(f"in_features={in_f} < rot_dim={rot_dim}")
        if rot_dim == in_f:
            apply_blockwise_hadamard_to_weight_inplace(W, block_size=Bb, normalized=True, compute_dtype=torch.float32)
            return
        Wp = W[:, :rot_dim].contiguous()
        apply_blockwise_hadamard_to_weight_inplace(Wp, block_size=Bb, normalized=True, compute_dtype=torch.float32)
        W[:, :rot_dim].copy_(Wp)

    @torch.no_grad()
    def _apply_hadamard_left_prefix_inplace(W: torch.Tensor, *, rot_dim: int, block_size: int) -> None:
        out_f, in_f = map(int, W.shape)
        Bb = int(block_size)
        if rot_dim % Bb != 0:
            raise ValueError(f"rot_dim={rot_dim} not divisible by block={Bb}")
        if out_f < rot_dim:
            raise ValueError(f"out_features={out_f} < rot_dim={rot_dim}")
        nb = rot_dim // Bb

        Wf = W.to(dtype=torch.float32)
        Wr = Wf[:rot_dim, :].contiguous().reshape(nb, Bb, in_f)
        Wr = Wr.permute(0, 2, 1).contiguous()
        Wr = matmul_hadU(Wr)
        Wr = Wr.permute(0, 2, 1).contiguous().reshape(rot_dim, in_f)
        Wf[:rot_dim, :] = Wr
        W.copy_(Wf.to(dtype=W.dtype))

    # -----------------------------
    # Hooks (pre: IN, post: OUT^T)
    # -----------------------------
    def _install_quamba_hooks(
        mod: nn.Module,
        *,
        block_size: int,
        rot_in_dim: int,
        rot_out_dim: int,
        k_in_cpu: Optional[torch.Tensor],
        k_out_cpu: Optional[torch.Tensor],
        enable_out: bool,
        module_path: str,
        layer_idx: int,
    ) -> None:
        if getattr(mod, "_cobra_quamba_hooks_installed", False):
            return

        dev = None
        w = getattr(mod, "weight", None)
        if torch.is_tensor(w):
            dev = w.device
        else:
            for p in mod.parameters(recurse=False):
                dev = p.device
                break
        if dev is None:
            dev = torch.device("cpu")

        def _materialize_klt(name: str, kb: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if kb is None or (not torch.is_tensor(kb)):
                return None
            try:
                return kb.detach().to(device=dev, dtype=klt_dtype).contiguous()
            except Exception as e:
                overwatch.warning(
                    f"[MixerHK] {name} KLT materialization failed; disabling KLT for module={module_path} "
                    f"layer={layer_idx}: {repr(e)}"
                )
                return None

        k_in_dev = _materialize_klt("IN", k_in_cpu)
        k_out_dev = _materialize_klt("OUT", k_out_cpu)

        if k_in_dev is not None:
            mod.register_buffer("_cobra_klt_in_blocks", k_in_dev, persistent=False)
        else:
            setattr(mod, "_cobra_klt_in_blocks", None)

        if k_out_dev is not None:
            mod.register_buffer("_cobra_klt_out_blocks", k_out_dev, persistent=False)
        else:
            setattr(mod, "_cobra_klt_out_blocks", None)

        B0 = int(block_size)
        RDI = int(rot_in_dim)
        RDO = int(rot_out_dim)
        do_out = bool(enable_out)

        def _warn_once(flag_name: str, msg: str) -> None:
            if getattr(mod, flag_name, False):
                return
            setattr(mod, flag_name, True)
            overwatch.warning(msg)

        def _pre_hook(_module, inputs):
            if not inputs:
                return inputs
            x = inputs[0]
            if not torch.is_tensor(x):
                return inputs

            x_dtype = x.dtype
            x = x.to(dtype=torch.float32)

            if RDI < int(x.shape[-1]):
                xp = x[..., :RDI]
                xt = x[..., RDI:]
            else:
                xp = x
                xt = None

            # R_in = K_in @ H => activation applies K then H
            Kin = getattr(_module, "_cobra_klt_in_blocks", None)
            if torch.is_tensor(Kin):
                if Kin.device != xp.device:
                    _warn_once(
                        "_cobra_klt_warned_pre",
                        f"[MixerHK] IN-KLT device mismatch; disabling KLT for module={module_path} layer={layer_idx} "
                        f"(Kin.device={Kin.device}, x.device={xp.device})"
                    )
                    setattr(_module, "_cobra_klt_in_blocks", None)
                else:
                    try:
                        xp = _blockwise_apply_klt_prefix(xp, k_blocks=Kin, transpose=False)
                    except Exception as e:
                        _warn_once(
                            "_cobra_klt_warned_pre",
                            f"[MixerHK] IN-KLT apply failed; disabling KLT for module={module_path} layer={layer_idx}: {repr(e)}"
                        )
                        setattr(_module, "_cobra_klt_in_blocks", None)

            xp = blockwise_hadamard(xp, block_size=B0)

            x = torch.cat([xp, xt], dim=-1) if xt is not None else xp
            if x.dtype != x_dtype:
                x = x.to(dtype=x_dtype)
            return (x,) + tuple(inputs[1:])

        def _post_hook(_module, inputs, output):
            if not do_out:
                return output
            y = output
            if not torch.is_tensor(y):
                return output

            y_dtype = y.dtype
            y = y.to(dtype=torch.float32)

            if RDO < int(y.shape[-1]):
                yp = y[..., :RDO]
                yt = y[..., RDO:]
            else:
                yp = y
                yt = None

            # post-hook applies R_out^T = H @ K_out^T:
            yp = blockwise_hadamard(yp, block_size=B0)

            Kout = getattr(_module, "_cobra_klt_out_blocks", None)
            if torch.is_tensor(Kout):
                if Kout.device != yp.device:
                    _warn_once(
                        "_cobra_klt_warned_post",
                        f"[MixerHK] OUT-KLT device mismatch; disabling KLT for module={module_path} layer={layer_idx} "
                        f"(Kout.device={Kout.device}, y.device={yp.device})"
                    )
                    setattr(_module, "_cobra_klt_out_blocks", None)
                else:
                    try:
                        yp = _blockwise_apply_klt_prefix(yp, k_blocks=Kout, transpose=True)
                    except Exception as e:
                        _warn_once(
                            "_cobra_klt_warned_post",
                            f"[MixerHK] OUT-KLT apply failed; disabling KLT for module={module_path} layer={layer_idx}: {repr(e)}"
                        )
                        setattr(_module, "_cobra_klt_out_blocks", None)

            y = torch.cat([yp, yt], dim=-1) if yt is not None else yp
            if y.dtype != y_dtype:
                y = y.to(dtype=y_dtype)
            return y

        mod.register_forward_pre_hook(_pre_hook, with_kwargs=False)
        if do_out:
            mod.register_forward_hook(_post_hook, with_kwargs=False)
        setattr(mod, "_cobra_quamba_hooks_installed", True)

    # -----------------------------
    # Main loop over layers
    # -----------------------------
    for layer_idx in range(max_layers):
        layer = layers[layer_idx]

        mixer = getattr(layer, "mixer", None)
        if mixer is None:
            skipped.append({"module_path": f"backbone.layers.{layer_idx}.mixer", "reason": "no mixer"})
            continue

        if "out_proj" not in targets:
            skipped.append({"module_path": f"backbone.layers.{layer_idx}.mixer", "reason": f"targets={targets} not supported"})
            continue

        mod = getattr(mixer, "out_proj", None)
        module_path = f"backbone.layers.{layer_idx}.mixer.out_proj"
        if mod is None or not isinstance(mod, nn.Module):
            skipped.append({"module_path": module_path, "reason": "missing out_proj"})
            continue

        if bool(getattr(cfg, "dry_run", False)):
            applied.append({"module_path": module_path, "layer": layer_idx, "status": "dry_run"})
            continue

        W = getattr(mod, "weight", None)
        if not torch.is_tensor(W) or W.ndim != 2:
            skipped.append({"module_path": module_path, "reason": "no 2D weight"})
            continue

        out_f, in_f = map(int, W.shape)

        # ---- resolve per-layer KLT blocks ----
        k_in = None
        if bool(cfg.act_klt_enabled):
            kb = k_in_layers.get(layer_idx, None)
            if torch.is_tensor(kb):
                kb2 = _validate_k_blocks(kb, B=B, dim=in_f, layer_idx=layer_idx, module_path=module_path, tag="IN")
                if kb2 is not None:
                    k_in = kb2

        enable_out = bool(out_tx)
        k_out = None
        if enable_out:
            # required safety: skip OUT-KLT on layer0, but still allow OUT-Hadamard
            if out_skip0 and layer_idx == 0:
                k_out = None
            else:
                kb = k_out_layers.get(layer_idx, None)
                if torch.is_tensor(kb):
                    kb2 = _validate_k_blocks(kb, B=B, dim=out_f, layer_idx=layer_idx, module_path=module_path, tag="OUT")
                    if kb2 is not None:
                        k_out = kb2

        # rot dims
        if torch.is_tensor(k_in):
            rot_in_dim = int(k_in.shape[0]) * B
        else:
            rot_in_dim = (in_f // B) * B

        if enable_out:
            if torch.is_tensor(k_out):
                rot_out_dim = int(k_out.shape[0]) * B
            else:
                rot_out_dim = (out_f // B) * B
        else:
            rot_out_dim = 0

        # -----------------------------
        # Weight rotation (in-place): CLOSED-LOOP HK-IN + HK-OUT
        #
        # W_rot = R_out^T @ W @ R_in
        #   R_in     = K_in @ H
        #   R_out^T  = H @ K_out^T
        #
        # Right (IN):   W <- W @ K_in ; then W <- W @ H
        # Left  (OUT):  W <- K_out^T @ W ; then W <- H @ W
        # -----------------------------
        try:
            # LEFT (OUT): K_out^T then H
            if enable_out and rot_out_dim > 0:
                if torch.is_tensor(k_out):
                    _apply_klt_left_prefix_inplace(W, k_blocks=k_out, transpose=True)
                _apply_hadamard_left_prefix_inplace(W, rot_dim=rot_out_dim, block_size=B)

            # RIGHT (IN): K_in then H
            if rot_in_dim > 0:
                if torch.is_tensor(k_in):
                    _apply_klt_right_prefix_inplace(W, k_blocks=k_in, transpose=False)
                _apply_hadamard_right_prefix_inplace(W, rot_dim=rot_in_dim, block_size=B)
        except Exception as e:
            skipped.append({"module_path": module_path, "reason": f"weight_rotation_failed: {repr(e)}"})
            continue

        # Install hooks (activation-side transforms)
        try:
            _install_quamba_hooks(
                mod,
                block_size=B,
                rot_in_dim=rot_in_dim,
                rot_out_dim=rot_out_dim if enable_out else 0,
                k_in_cpu=k_in,
                k_out_cpu=k_out if enable_out else None,
                enable_out=bool(enable_out),
                module_path=module_path,
                layer_idx=layer_idx,
            )
        except Exception as e:
            skipped.append({"module_path": module_path, "reason": f"hook_install_failed: {repr(e)}"})
            continue

        applied.append(
            {
                "module_path": module_path,
                "layer": layer_idx,
                "mode": "hk" if bool(cfg.act_klt_enabled) else "h",
                "block": B,
                "rot_in_dim": rot_in_dim,
                "rot_out_dim": rot_out_dim if enable_out else 0,
                "has_k_in": int(torch.is_tensor(k_in)),
                "has_k_out": int(torch.is_tensor(k_out)) if enable_out else 0,
                "out_transform": int(enable_out),
                "klt_dtype": str(klt_dtype).replace("torch.", ""),
            }
        )

    return {"applied": applied, "skipped": skipped}

