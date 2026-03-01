# -*- coding: utf-8 -*-
"""
Custom Mamba block with percentile clipping, numeric guards, and activation capture.
This version forces FULL-SEQUENCE path in forward() (no incremental step during inference),
to avoid fused LayerNorm residual-shape and CUDA graph capture issues.
"""
from __future__ import annotations

import contextlib
from typing import Iterator, List, Optional, Sequence, Dict, Tuple

import math
import os, builtins, atexit

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

from mamba_ssm.modules.mamba_simple import Mamba as _BaseMamba
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn


# ===================== logging (tee to file) =====================
builtins.print(f"[mamba_cfzo] module loaded from: {__file__}")
_LOG_PATH = os.path.abspath(os.path.join(os.getcwd(), "mamba_cfzo_log.txt"))
builtins.print(f"[mamba_cfzo] logging to: {_LOG_PATH}")

_original_print = builtins.print
_log_file = None
try:
    _log_file = open(_LOG_PATH, "w", encoding="utf-8")  # overwrite per run
except Exception as e:
    _original_print(f"[mamba_cfzo] ERROR opening log file: {e}")

def _tee_print(*args, **kwargs):
    _original_print(*args, **kwargs)  # console
    if _log_file is None:
        return
    sep = kwargs.get("sep", " ")
    end = kwargs.get("end", "\n")
    try:
        _log_file.write(sep.join(str(a) for a in args) + end)
        _log_file.flush()
    except Exception as e:
        _original_print(f"[mamba_cfzo] ERROR writing log file: {e}")

_DEBUG_TENSORS = os.environ.get("MAMBA_CFZO_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def _debug_print(*args, **kwargs):
    if not _DEBUG_TENSORS:
        return
    _tee_print(*args, **kwargs)


builtins.print = _tee_print

@atexit.register
def _close_mamba_log_file():
    try:
        if _log_file is not None:
            _log_file.close()
    except Exception as e:
        _original_print(f"[mamba_cfzo] ERROR closing log file: {e}")
# ================================================================


# ----------------- helpers -----------------
def _compute_percentile_bounds(
    channel_values: Tensor,
    lower: float,
    upper: float,
    bins: int,
) -> tuple[float, float]:
    """Return percentile clip bounds for a flattened 1D tensor."""
    if channel_values.numel() == 0:
        return 0.0, 0.0
    channel_cpu = channel_values.detach().to(dtype=torch.float32, device="cpu")
    min_val = float(channel_cpu.min())
    max_val = float(channel_cpu.max())
    if not torch.isfinite(channel_cpu).all() or min_val == max_val:
        return min_val, max_val
    hist = torch.histc(channel_cpu, bins=bins, min=min_val, max=max_val)
    cdf = torch.cumsum(hist, dim=0)
    total = float(cdf[-1]) or 1.0
    lower_idx = torch.searchsorted(cdf, torch.tensor(lower * total))
    upper_idx = torch.searchsorted(cdf, torch.tensor(upper * total))
    bin_width = (max_val - min_val) / bins
    clip_min = min_val + float(lower_idx.item()) * bin_width
    clip_max = min_val + float(upper_idx.item()) * bin_width
    return clip_min, clip_max

def _num(x: torch.Tensor) -> int:
    return int(x.numel())

def _safe_stats(t: torch.Tensor):
    t = t.detach()
    finite = torch.isfinite(t)
    if finite.any():
        t = t[finite]
        return float(t.min().item()), float(t.max().item()), float(t.mean().item()), float(t.std(unbiased=False).item()), int(finite.sum().item())
    return float("nan"), float("nan"), float("nan"), float("nan"), 0

def print_tensor_stats(t: torch.Tensor, name: str = "tensor") -> None:
    mn, mx, mu, sd, fin = _safe_stats(t)
    _debug_print(
        f"[STAT] {name:>18} | shape={tuple(t.shape)} dtype={t.dtype} dev={t.device} "
        f"finite={fin}/{_num(t)} min={mn:.6g} max={mx:.6g} mean={mu:.6g} std={sd:.6g}"
    )

def ascii_hist(t: torch.Tensor, name: str = "tensor", bins: int = 30, width: int = 50) -> None:
    x = t.detach().reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        _debug_print(f"[HIST] {name}: (no finite values)")
        return
    lo, hi = float(x.min()), float(x.max())
    if lo == hi:
        _debug_print(f"[HIST] {name}: all {lo}")
        return
    edges = torch.linspace(lo, hi, steps=bins + 1, device=x.device)
    hist = torch.histc(x, bins=bins, min=lo, max=hi).to(torch.int64)
    m = int(hist.max().item()) or 1
    _debug_print(f"[HIST] {name} range=[{lo:.6g},{hi:.6g}] bins={bins}")
    for i, h in enumerate(hist):
        bar = "█" * max(1, math.floor(width * int(h) / m))
        e0, e1 = edges[i].item(), edges[i + 1].item()
        _debug_print(f"{e0:>10.4g}–{e1:<10.4g} | {bar}")

def channel_variance(x_blD: torch.Tensor) -> torch.Tensor:
    """Input (B,L,D) or (B,D,L) -> return per-channel variance (D,)."""
    if x_blD.dim() != 3:
        raise ValueError("channel_variance expects a 3D tensor")
    B, A, C = x_blD.shape
    if A < C:   # (B, L, D)
        X = x_blD.reshape(B * A, C)
    else:       # (B, D, L) -> (B*L, D)
        X = x_blD.permute(0, 2, 1).reshape(B * C, A)
    return X.to(torch.float32).var(dim=0, unbiased=False)

def compare_channel_variance(x_t: torch.Tensor, y_t: torch.Tensor, topk: int = 8, name: str = "layer") -> None:
    vx = channel_variance(x_t)
    vy = channel_variance(y_t)
    delta = (vy - vx).abs()
    D = vx.numel()
    k = min(topk, D)
    vals, idx = torch.topk(delta, k)
    _debug_print(f"[VAR] {name}: D={D}, top{ k } channels with largest |var(y)-var(x)|")
    _debug_print(" ch |   var(x)    var(y)    diff   ")
    _debug_print("----+------------------------------")
    for i, d in zip(idx.tolist(), vals.tolist()):
        _debug_print(f"{i:>3d} | {float(vx[i]):>9.4g}  {float(vy[i]):>9.4g}  {float(d):>9.4g}")

def _finite_ratio(t: Tensor, name: str):
    if not _DEBUG_TENSORS:
        return
    f = torch.isfinite(t)
    pct = float(f.float().mean().item() * 100.0)
    _debug_print(f"[FINITE] {name}: {int(f.sum().item())}/{t.numel()} = {pct:.4f}%")

# ================================================================


class Mamba(_BaseMamba):
    """
    Wrap the base Mamba block with CF-ZO specific tooling.

    Key behavioral change:
    - forward() always runs FULL-SEQUENCE path (even when inference_params is provided).
      This avoids residual-shape assertions in fused LayerNorm and CUDA graph capture errors.
    """

    def __init__(
        self,
        *args,
        percentile_lower: float = 0.01,
        percentile_upper: float = 0.99,
        histogram_bins: int = 2048,
        **kwargs
    ):
        # Safer default: disable fast path unless you explicitly re-enable after stability check
        kwargs.setdefault("use_fast_path", False)
        super().__init__(*args, **kwargs)
        self.histogram_bins = histogram_bins
        # Trainable scale on delta in fp32 softplus path (initialized to 1)
        self.mamba_scale = nn.Parameter(
            torch.ones(self.d_inner, device=self.out_proj.weight.device, dtype=self.out_proj.weight.dtype)
        )
        self._captured_activations: Optional[List[dict[str, Tensor]]] = None

        # === CF-ZO percentile clipping switch (edit here: 0.99 / 0.999 / 0.9999 / 1.0) ===
        self.use_percentile_clip = False  # flip to False to disable clipping entirely
        self.clip_percentile = float(percentile_upper)  # e.g. 0.99 / 0.999 / 0.9999 / 1.0
        self.clip_percentile = max(0.0, min(1.0, self.clip_percentile))
        if not self.use_percentile_clip or self.clip_percentile >= 1.0:
            self.percentile_lower = 0.0
            self.percentile_upper = 1.0
        else:
            self.percentile_lower = max(0.0, 1.0 - self.clip_percentile)
            self.percentile_upper = self.clip_percentile

    # ---------- percentile clipping ----------
    def _clip_percentiles(self, x: Tensor) -> Tensor:
        if not self.use_percentile_clip or self.percentile_upper >= 1.0:
            return x
        if x.numel() == 0:
            return x
        mins: List[float] = []
        maxs: List[float] = []
        _, channels, _ = x.shape
        for idx in range(channels):
            channel = x[:, idx, :].reshape(-1)
            clip_min, clip_max = _compute_percentile_bounds(
                channel,
                self.percentile_lower,
                self.percentile_upper,
                self.histogram_bins,
            )
            mins.append(clip_min)
            maxs.append(clip_max)
        device = x.device
        dtype = x.dtype
        min_tensor = torch.tensor(mins, device=device, dtype=dtype)
        max_tensor = torch.tensor(maxs, device=device, dtype=dtype)
        return torch.max(torch.min(x, max_tensor.view(1, -1, 1)), min_tensor.view(1, -1, 1))

    # ---------- FULL-SEQUENCE forward ----------
    def forward(self, hidden_states: Tensor, inference_params=None):
        """
        Always full sequence path:
          Input : (B, L, D)
          Output: (B, L, D)
        """
        batch, seqlen, _ = hidden_states.shape

        # ==== in-proj (matmul path) ====
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)  # x,z: (B, Din, L)

        # ==== 1D causal conv ====
        # No per-step conv_state usage here (full sequence path)
        if causal_conv1d_fn is None:
            x = self.act(self.conv1d(x)[..., :seqlen])
        else:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation,
            )
            x = x[..., :seqlen]

        # numeric sanitation on conv output
        if not torch.isfinite(x).all():
            bad = ~torch.isfinite(x)
            print(f"[WARN] conv output has {int(bad.sum().item())}/{x.numel()} non-finite; sanitizing.")
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)
        _debug_print("xt shape", x.shape)
        _debug_print("xt ", x)
        _finite_ratio(x, "xt-before-clip")

        # percentile clip (visualization & stability)
        x = self._clip_percentiles(x)
        _debug_print("xt clipped shape", x.shape)
        _debug_print("xt clipped ", x)
        _finite_ratio(x, "xt-after-clip")

        # ==== x_proj and split ====
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, Bm, Cm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        # ==== delta (fp32 softplus + trainable scale) ====
        dt = self.dt_proj.weight @ dt.t()
        dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
        dt = dt + self.dt_proj.bias.view(1, -1, 1).to(dtype=dt.dtype)

        orig_dtype = dt.dtype
        dt_fp32 = dt.to(torch.float32)
        dt_fp32 = torch.clamp(dt_fp32, -20.0, 20.0)  # avoid overflow in softplus
        delta_fp32 = F.softplus(dt_fp32)
        scale_fp32 = self.mamba_scale.view(1, -1, 1).to(torch.float32)
        delta_fp32 = delta_fp32 * scale_fp32
        delta = delta_fp32.to(orig_dtype)
        if not torch.isfinite(delta).all():
            print("[WARN] delta non-finite after fp32 softplus; sanitizing.")
            delta = torch.nan_to_num(delta, nan=0.0, posinf=1e4, neginf=-1e4)
        tiny = torch.finfo(delta.dtype).tiny
        delta = torch.clamp(delta, min=tiny)

        # ==== A, B, C ====
        A = -torch.exp(self.A_log.float())  # (d_state,)
        Bm = rearrange(Bm, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        Cm = rearrange(Cm, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        # final guards before SSM
        def _sanitize_(name: str, t: Tensor) -> Tensor:
            if not torch.isfinite(t).all():
                bad = ~torch.isfinite(t)
                print(f"[WARN] {name} has {int(bad.sum().item())}/{t.numel()} non-finite; sanitizing.")
                t = torch.nan_to_num(t, nan=0.0, posinf=1e4, neginf=-1e4)
            return t

        x = _sanitize_("x", x)
        delta = _sanitize_("delta", delta)
        A = _sanitize_("A", A)
        Bm = _sanitize_("B", Bm)
        Cm = _sanitize_("C", Cm)
        z = _sanitize_("z", z)

        # ==== selective scan (full sequence) ====
        need_last_state = False
        ssm_state_ref = None
        if inference_params is not None:
            # For compatibility with upstream cache structure: if the caller provided a cache,
            # we can return last_state and let the wrapper manage it. We DO NOT run step().
            conv_state_ref, ssm_state_ref = self._get_states_from_cache(inference_params, batch)
            need_last_state = ssm_state_ref is not None

        y = selective_scan_fn(
            x,
            delta,
            A,
            Bm,
            Cm,
            self.D.float(),
            z=z,
            delta_bias=None,
            delta_softplus=False,
            return_last_state=need_last_state,
        )
        _debug_print("yt shape", (y[0].shape if isinstance(y, tuple) else y.shape))
        _debug_print("yt ", y)

        if need_last_state:
            y, last_state = y  # y:(B,D,L)
            # keep cache up-to-date for callers, even though we didn't step()
            if ssm_state_ref is not None:
                ssm_state_ref.copy_(last_state)
            # conv_state maintenance (optional best-effort): fill with last W window of x
            if conv_state_ref is not None:
                W = conv_state_ref.shape[-1]
                tail = x[..., -W:] if x.shape[-1] >= W else F.pad(x, (W - x.shape[-1], 0))
                conv_state_ref.copy_(tail)

        y = rearrange(y if isinstance(y, torch.Tensor) else y[0], "b d l -> b l d")

        if self._captured_activations is not None:
            self._captured_activations.append(
                {
                    "layer_idx": getattr(self, "layer_idx", None),
                    "x_t": rearrange(x, "b d l -> b l d").detach().cpu(),
                    "y_t": y.detach().cpu(),
                }
            )

        out = self.out_proj(y)
        init_layer_scale = getattr(self, "init_layer_scale", None)
        if init_layer_scale is not None and hasattr(self, "gamma"):
            out = out * self.gamma
        return out  # (B, L, D)

    # ---------- (kept for completeness; not used by forward in this file) ----------
    def step(self, hidden_states: Tensor, conv_state: Tensor, ssm_state: Tensor):
        """Single-token step (kept for reference; full-seq forward() doesn’t call this)."""
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1
        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=1e4, neginf=-1e4)

        x_db = self.x_proj(x)
        dt, Bm, Cm = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dt = F.linear(dt, self.dt_proj.weight)
        dt = dt + self.dt_proj.bias.to(dtype=dt.dtype)

        orig_dtype = dt.dtype
        dt_fp32 = torch.clamp(dt.to(torch.float32), -20.0, 20.0)
        delta_fp32 = F.softplus(dt_fp32)
        scale_fp32 = self.mamba_scale.view(1, -1).to(torch.float32)
        delta_fp32 = delta_fp32 * scale_fp32
        delta = delta_fp32.to(orig_dtype)
        if not torch.isfinite(delta).all():
            print("[WARN] delta(non-step) non-finite after fp32 softplus; sanitizing.")
            delta = torch.nan_to_num(delta, nan=0.0, posinf=1e4, neginf=-1e4)
        tiny = torch.finfo(delta.dtype).tiny
        delta = torch.clamp(delta, min=tiny)

        A = -torch.exp(self.A_log.float())
        dA = torch.exp(torch.einsum("bd,dn->bdn", delta, A))
        dB = torch.einsum("bd,bn->bdn", delta, Bm)
        ssm_state.copy_(ssm_state * dA + x.unsqueeze(-1) * dB)
        y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), Cm) + self.D.to(dtype) * x
        y = y * self.act(z)
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    # ---------- activation capture ----------
    def set_activation_capture(self, store: Optional[List[dict[str, Tensor]]]):
        self._captured_activations = store


@contextlib.contextmanager
def capture_mamba_activations(model: nn.Module) -> Iterator[List[dict[str, Tensor]]]:
    store: List[dict[str, Tensor]] = []
    modules: List[Mamba] = []
    for module in model.modules():
        if isinstance(module, Mamba):
            modules.append(module)
            module.set_activation_capture(store)
    try:
        yield store
    finally:
        for module in modules:
            module.set_activation_capture(None)


def show_activation_summary(entry: Dict[str, torch.Tensor], layer_name: str = None) -> None:
    lid = entry.get("layer_idx", None)
    nm = layer_name or (f"layer{lid}" if lid is not None else "layer")
    x_t = entry["x_t"]  # (B,L,D)
    y_t = entry["y_t"]  # (B,L,D)
    print_tensor_stats(x_t, f"{nm}.x_t")
    print_tensor_stats(y_t, f"{nm}.y_t")
    ascii_hist(x_t, f"{nm}.x_t", bins=24, width=40)
    ascii_hist(y_t, f"{nm}.y_t", bins=24, width=40)
    compare_channel_variance(x_t, y_t, topk=8, name=nm)

def show_model_mamba_scales(model: nn.Module, topk: int = 5) -> None:
    for m in model.modules():
        if isinstance(m, Mamba):
            s = m.mamba_scale.detach().float().cpu()
            mn, mx, mu, sd, _ = _safe_stats(s)
            print(f"[SCALE] layer{getattr(m, 'layer_idx', '?')}: len={s.numel()} "
                  f"min={mn:.6g} max={mx:.6g} mean={mu:.6g} std={sd:.6g}")
            k = min(topk, s.numel())
            if k > 0:
                maxv, maxi = torch.topk(s, k)
                minv, mini = torch.topk(-s, k)
                print("   top-max idx,val:", [(int(i), float(v)) for v, i in zip(maxv.tolist(), maxi.tolist())])
                print("   top-min idx,val:", [(int(i), float(-v)) for v, i in zip(minv.tolist(), mini.tolist())])

def show_captured_activations_summary(activations: Sequence[Dict[str, torch.Tensor]], limit: int = 3) -> None:
    n = len(activations)
    print(f"[CAPSUM] captured layers: {n}")
    for i, entry in enumerate(activations[:limit]):
        show_activation_summary(entry, layer_name=f"layer{entry.get('layer_idx', i)}")
        print("-" * 60)
    if n > limit:
        print(f"... ({n - limit} more layers omitted)")
