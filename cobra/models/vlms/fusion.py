from __future__ import annotations

import os
import torch
import torch.nn as nn

from cobra.quantize.quantizer import UniformAffineQuantizer

class FusionStage(nn.Module):
    """
    FusionStage (Point B)

    This module is the explicit hook right after:
      concat([img_embeddings, text_embeddings]) -> fused_embeddings

    Milestone 1 behavior:
      - Identity (no-op).

    Future milestones will extend this module to support:
      - Global rotation: fused_embeddings := fused_embeddings @ R
      - Global best clipping
      - Global activation quant-dequant
    """

    def __init__(self) -> None:
        super().__init__()
        self._rotation_mode: str = "none"
        self._has_klt: bool = False
        self.register_buffer("klt_matrix", torch.empty(0), persistent=False)
        self.act_quantizer = UniformAffineQuantizer(
            n_bits=8,
            symmetric=True,
            per_channel_axes=[],
            metric="minmax",
            dynamic=False,
        )
        self._enable_quant: bool = False
        self._has_clip_range: bool = False


    def forward(self, fused_embeddings: torch.Tensor) -> torch.Tensor:
        if os.environ.get("COBRA_DEBUG_FLOW", "").strip().lower() in ("1", "true", "yes", "on"):
            try:
                shp = tuple(fused_embeddings.shape)
            except Exception:
                shp = "<unknown>"
            print(
                f"[FusionStage.forward] called: mode={getattr(self, '_rotation_mode', 'unknown')}, "
                f"shape={shp}, dtype={getattr(fused_embeddings, 'dtype', None)}, device={getattr(fused_embeddings, 'device', None)}"
            )
            
        m = self._rotation_mode

        if m == "none":
            x = fused_embeddings
        else:
            from cobra.quantize.rotate.hadamard import apply_hadamard_transform

            if m == "hadamard":
                x = apply_hadamard_transform(fused_embeddings, dim=-1, normalize=True)
            else:
                # m == "hk"
                if not self._has_klt or self.klt_matrix.numel() == 0:
                    AA = tuple(fused_embeddings.shape)
                    KK = tuple(self.klt_matrix.shape)
                    raise RuntimeError(
                        f"[FusionStage] rotation_mode='hk' but klt_matrix is not configured; "
                        f"fused_shape={AA}, klt_shape={KK}"
                    )
                x = torch.matmul(fused_embeddings, self.klt_matrix)
                x = apply_hadamard_transform(x, dim=-1, normalize=True)

        # Milestone 3: Global best clipping @ B (range comes from pct calibrator)
        if getattr(self, "_has_clip_range", False):
            xmin = getattr(self.act_quantizer, "clip_min", None)
            xmax = getattr(self.act_quantizer, "clip_max", None)
            if isinstance(xmin, torch.Tensor) and isinstance(xmax, torch.Tensor) and xmin.numel() > 0 and xmax.numel() > 0:
                lo = xmin.to(device=x.device, dtype=x.dtype)
                hi = xmax.to(device=x.device, dtype=x.dtype)
                # If quant is disabled, we still need pure clipping behavior (Milestone 3)
                if not getattr(self, "_enable_quant", False):
                    x = x.clamp(lo, hi)

        # Milestone 4: Global low-bit mapping @ B (quant/dequant)
        if getattr(self, "_enable_quant", False):
            x = self.act_quantizer(x)

        return x

    def configure_rotation(self, *, mode: str, klt_matrix: torch.Tensor | None) -> None:
        m = (mode or "none").strip().lower()
        if m not in ("none", "hadamard", "hk"):
            raise ValueError(f"[FusionStage] Unsupported rotation mode: {mode!r}")

        self._rotation_mode = m

        if m == "hk":
            if klt_matrix is None:
                raise ValueError("[FusionStage] mode='hk' requires klt_matrix")
            if not isinstance(klt_matrix, torch.Tensor):
                raise TypeError("[FusionStage] klt_matrix must be a torch.Tensor")
            if klt_matrix.ndim != 2 or klt_matrix.shape[0] != klt_matrix.shape[1]:
                raise ValueError(
                    f"[FusionStage] klt_matrix must be square [D,D], got shape={tuple(klt_matrix.shape)}"
                )
            # store as buffer (device-aware)
            self.klt_matrix = klt_matrix
            self._has_klt = True
        else:
            # clear KLT
            self.klt_matrix = torch.empty(0, device=self.klt_matrix.device)
            self._has_klt = False

    def configure_quant(self, *, n_bits: int, enable: bool) -> None:
        nb = int(n_bits)
        if nb < 2 or nb > 16:
            raise ValueError(f"[FusionStage] n_bits out of range: {n_bits!r} (expected 2..16)")

        self._enable_quant = bool(enable)
        # UniformAffineQuantizer uses `n_bits` + `enable` to decide no-op vs fake-quant
        self.act_quantizer.n_bits = nb
        self.act_quantizer.enable = bool(enable)

    def mark_clipping_ready(self) -> None:
        self._has_clip_range = True
