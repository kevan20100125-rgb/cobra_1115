# cobra/switches/verify_act_klt_outproj.py
"""
Sanity-check layer-wise act-KLT payload for Mamba mixer out_proj.

This is a read-only verifier:
  - Load torch payload from act_klt_outproj.pt
  - Validate schema/meta/layer keys
  - Validate tensor shapes per layer: [n_blocks, B, B]
  - Validate orthonormality per block: K^T K ~ I
  - Report per-layer max errors and global worst-case

Exit code:
  0: pass
  1: fail
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import torch


@dataclass
class VerifyActKLTOutProjConfig:
    act_klt_path: Path = Path("outputs/quantize/act_klt_outproj_bs512/act_klt_outproj.pt")

    # If provided, enforce these expected values; otherwise infer from meta and tensors.
    expected_num_layers: Optional[int] = None
    expected_block_size: Optional[int] = None
    expected_hidden_size: Optional[int] = None

    # Orthonormal check thresholds (float64 check)
    # max_abs(K^T K - I) should be tiny; allow some slack for numerical issues.
    max_abs_tol: float = 3e-3
    fro_norm_tol: float = 5e-2  # Frobenius norm of (K^T K - I)

    # Report details
    report_topk_worst_blocks: int = 5


def _as_int_layer_keys(layers: Dict[Any, Any]) -> Dict[int, torch.Tensor]:
    out: Dict[int, torch.Tensor] = {}
    for k, v in layers.items():
        if isinstance(k, int):
            out[int(k)] = v
            continue
        # torch.save sometimes keeps keys as strings depending on writer
        if isinstance(k, str) and k.isdigit():
            out[int(k)] = v
            continue
        raise RuntimeError(f"[verify_act_klt_outproj] Invalid layer key type/value: {type(k)} {k}")
    return out


@torch.no_grad()
def _check_layer_tensor(
    layer_idx: int,
    K: torch.Tensor,
    *,
    expect_n_blocks: Optional[int],
    expect_B: Optional[int],
    cfg: VerifyActKLTOutProjConfig,
) -> Tuple[float, float, Tuple[int, int], Tuple[float, float, int]]:
    """
    Returns:
      layer_max_abs_err, layer_max_fro_err,
      worst_block_idx=(bi, ???) but we keep (bi, bi) placeholder,
      worst_block_stats=(worst_abs_err, worst_fro_err, worst_bi)
    """
    if not torch.is_tensor(K):
        raise RuntimeError(f"[verify_act_klt_outproj] Layer {layer_idx} value is not a tensor: {type(K)}")

    if K.ndim != 3:
        raise RuntimeError(f"[verify_act_klt_outproj] Layer {layer_idx} tensor must be 3D [n_blocks,B,B], got {tuple(K.shape)}")

    n_blocks, B1, B2 = map(int, K.shape)
    if B1 != B2:
        raise RuntimeError(f"[verify_act_klt_outproj] Layer {layer_idx} has non-square blocks: {B1}x{B2}")

    if expect_B is not None and B1 != int(expect_B):
        raise RuntimeError(
            f"[verify_act_klt_outproj] Layer {layer_idx} block_size mismatch: got {B1}, expect {int(expect_B)}"
        )

    if expect_n_blocks is not None and n_blocks != int(expect_n_blocks):
        raise RuntimeError(
            f"[verify_act_klt_outproj] Layer {layer_idx} n_blocks mismatch: got {n_blocks}, expect {int(expect_n_blocks)}"
        )

    # Orthonormal check in float64
    K64 = K.to(dtype=torch.float64, device="cpu")
    I = torch.eye(B1, dtype=torch.float64)

    layer_max_abs = 0.0
    layer_max_fro = 0.0
    worst_abs = -1.0
    worst_fro = -1.0
    worst_bi = -1

    for bi in range(n_blocks):
        Qi = K64[bi]  # [B,B]
        # Q^T Q
        M = Qi.t().matmul(Qi)
        E = M - I

        abs_err = float(E.abs().max().item())
        fro_err = float(torch.linalg.norm(E).item())

        if abs_err > layer_max_abs:
            layer_max_abs = abs_err
        if fro_err > layer_max_fro:
            layer_max_fro = fro_err

        if abs_err > worst_abs:
            worst_abs = abs_err
            worst_fro = fro_err
            worst_bi = bi

    return layer_max_abs, layer_max_fro, (worst_bi, worst_bi), (worst_abs, worst_fro, worst_bi)


def verify_act_klt_outproj(cfg: VerifyActKLTOutProjConfig) -> None:
    path = cfg.act_klt_path
    if not path.is_file():
        raise RuntimeError(f"[verify_act_klt_outproj] File not found: {path}")

    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise RuntimeError(f"[verify_act_klt_outproj] Payload must be dict, got {type(payload)}")

    if "layers" not in payload:
        raise RuntimeError("[verify_act_klt_outproj] Missing key: 'layers'")
    layers_raw = payload["layers"]
    if not isinstance(layers_raw, dict):
        raise RuntimeError(f"[verify_act_klt_outproj] payload['layers'] must be dict, got {type(layers_raw)}")

    meta = payload.get("meta", {})
    if meta and not isinstance(meta, dict):
        raise RuntimeError(f"[verify_act_klt_outproj] payload['meta'] must be dict if present, got {type(meta)}")

    layers = _as_int_layer_keys(layers_raw)
    layer_indices = sorted(layers.keys())
    if not layer_indices:
        raise RuntimeError("[verify_act_klt_outproj] Empty layers dict")

    # Infer expected sizes from meta if not specified
    meta_num_layers = int(meta.get("num_layers", len(layer_indices))) if isinstance(meta, dict) else len(layer_indices)
    meta_B = int(meta.get("block_size", 0)) if isinstance(meta, dict) else 0
    meta_D = int(meta.get("hidden_size", 0)) if isinstance(meta, dict) else 0
    meta_n_blocks = int(meta.get("n_blocks", 0)) if isinstance(meta, dict) else 0

    expected_num_layers = cfg.expected_num_layers if cfg.expected_num_layers is not None else meta_num_layers
    expected_B = cfg.expected_block_size if cfg.expected_block_size is not None else (meta_B if meta_B > 0 else None)
    expected_D = cfg.expected_hidden_size if cfg.expected_hidden_size is not None else (meta_D if meta_D > 0 else None)

    # Determine expected n_blocks
    expected_n_blocks: Optional[int] = None
    if meta_n_blocks > 0:
        expected_n_blocks = meta_n_blocks
    elif expected_D is not None and expected_B is not None:
        if int(expected_D) % int(expected_B) != 0:
            raise RuntimeError(
                f"[verify_act_klt_outproj] hidden_size D={expected_D} not divisible by block_size B={expected_B}"
            )
        expected_n_blocks = int(expected_D) // int(expected_B)

    # Basic key checks: contiguous layers?
    # In Cobra 3B Mamba, you expect 0..63 for 64 layers.
    if expected_num_layers is not None and len(layer_indices) != int(expected_num_layers):
        raise RuntimeError(
            f"[verify_act_klt_outproj] num_layers mismatch: file has {len(layer_indices)}, expected {int(expected_num_layers)}"
        )

    # If indices look like a full range, enforce contiguity
    if layer_indices[0] == 0 and len(layer_indices) >= 2:
        expected_range = list(range(layer_indices[0], layer_indices[0] + len(layer_indices)))
        if layer_indices != expected_range:
            raise RuntimeError(
                f"[verify_act_klt_outproj] Layer indices not contiguous. Head={layer_indices[:10]} Tail={layer_indices[-10:]}"
            )

    print(f"[A1] verify_act_klt_outproj: file={path}")
    if isinstance(meta, dict) and meta:
        print(f"[A1] meta: model_id={meta.get('model_id')} stage={meta.get('stage')} dataset_id={meta.get('dataset_id')}")
        print(f"[A1] meta: hidden_size={meta.get('hidden_size')} block_size={meta.get('block_size')} n_blocks={meta.get('n_blocks')} num_layers={meta.get('num_layers')}")
    print(f"[A1] layers: num_layers={len(layer_indices)} layer_range=[{layer_indices[0]}..{layer_indices[-1]}]")

    worst_global_abs = -1.0
    worst_global_fro = -1.0
    worst_global_layer = -1
    worst_global_block = -1

    per_layer_max: Dict[int, Tuple[float, float, int]] = {}

    for li in layer_indices:
        K = layers[li]
        layer_max_abs, layer_max_fro, _, worst_block_stats = _check_layer_tensor(
            li, K, expect_n_blocks=expected_n_blocks, expect_B=expected_B, cfg=cfg
        )
        worst_abs, worst_fro, worst_bi = worst_block_stats

        per_layer_max[li] = (layer_max_abs, layer_max_fro, worst_bi)

        if layer_max_abs > worst_global_abs:
            worst_global_abs = layer_max_abs
            worst_global_fro = layer_max_fro
            worst_global_layer = li
            worst_global_block = worst_bi

    # Report summary
    print(f"[A1] Orthonormality thresholds: max_abs_tol={cfg.max_abs_tol} fro_norm_tol={cfg.fro_norm_tol}")
    print(f"[A1] Worst global: layer={worst_global_layer} block={worst_global_block} max_abs={worst_global_abs:.6g} fro={worst_global_fro:.6g}")

    # Top-k worst layers by max_abs
    topk = sorted(per_layer_max.items(), key=lambda x: x[1][0], reverse=True)[: max(1, cfg.report_topk_worst_blocks)]
    print("[A1] Top worst layers (by max_abs):")
    for li, (ma, mf, bi) in topk:
        print(f"  layer={li:>2d} worst_block={bi:>2d} max_abs={ma:.6g} fro={mf:.6g}")

    # Enforce thresholds
    failed_layers = []
    for li, (ma, mf, bi) in per_layer_max.items():
        if (ma > cfg.max_abs_tol) or (mf > cfg.fro_norm_tol):
            failed_layers.append((li, ma, mf, bi))

    if failed_layers:
        print("[A1][FAIL] Layers exceeding orthonormal tolerances:")
        for li, ma, mf, bi in failed_layers[:20]:
            print(f"  layer={li} worst_block={bi} max_abs={ma:.6g} fro={mf:.6g}")
        raise RuntimeError(f"[verify_act_klt_outproj] FAIL: {len(failed_layers)} layers exceed tolerances.")

    print("[A1][PASS] act-KLT payload sanity checks passed.")


def main() -> int:
    # Allow override by env (useful on HPC)
    path = os.environ.get("ACT_KLT_OUT", "").strip()
    if path:
        cfg = VerifyActKLTOutProjConfig(act_klt_path=Path(path))
    else:
        cfg = VerifyActKLTOutProjConfig()

    try:
        verify_act_klt_outproj(cfg)
        return 0
    except Exception as e:
        print(f"[A1][ERROR] {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

