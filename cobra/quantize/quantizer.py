from re import U
import math
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from .observers.hist_observers import PercentileObserver, KLObserver, MSEObserver
from .observers.minmax_observers import MinMaxObserver

CLIPMIN = 1e-5


class ClampSte(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, min_, max_):
        return x.clamp(min_, max_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.clone(), None, None


def round_ste(x: torch.Tensor):
    """
    Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


class UniformAffineQuantizer(nn.Module):
    """
    Core fake quantizer used for both weights and activations.

    This module is MambaQuant-style:
        - Fake quant only (float kernels, quant/dequant in PyTorch).
        - Supports per-tensor / per-channel / grouped quantization.
        - Can run in:
            * observer mode (collect stats),
            * dynamic mode (per-batch ranges),
            * static mode with pre-computed clipping ranges (pct_hi_lo).

    For our W4/W8 fake-quant PTQ pipeline, percentile-based clipping will
    write pre-computed (xmin, xmax) into each activation quantizer via
    `set_clipping_range`, and `n_bits` can be changed at runtime.
    """

    def __init__(
        self,
        n_bits: int = 8,
        symmetric: bool = False,
        per_channel_axes=[],
        metric: str = "minmax",
        dynamic: bool = False,
        dynamic_method: str = "per_cluster",
        group_size=None,
        shape=None,
        lwc: bool = False,
        disable_zero_point: bool = False,
        rescale: bool = False,
        rescale_limit: bool = False,
        has_batch_dim: bool = False,
        is_weight: bool = False,
        observe: str = "minmax",
        percent: float = 0.999999,
    ):
        """
        Parameters
        ----------
        n_bits:
            Bit-width for quantization (2–16).
        symmetric:
            Whether to use symmetric quantization (zero_point = 0).
        per_channel_axes:
            List of axes used for per-channel quantization.
        metric:
            Quantization metric ("minmax", "fix0to1", etc.).
        dynamic:
            Whether to use dynamic quantization for activations.
        dynamic_method:
            "per_token", "per_channel", or "per_tensor".
        group_size:
            Group size for grouped quantization (e.g., weight-only).
        shape:
            Original tensor shape for grouped quantization.
        lwc:
            Whether to use learnable weight clipping.
        disable_zero_point:
            Force zero_point to be 0 (symmetric).
        rescale / rescale_limit:
            Optional post-dequant rescaling parameters.
        has_batch_dim:
            For dynamic per-tensor calibration with batch dimension.
        is_weight:
            Flag to distinguish weight quant vs activation quant.
        observe:
            "minmax" or "percentile" for observer choice.
        percent:
            Percentile used when observe == "percentile".
        """
        super().__init__()

        self.symmetric = symmetric
        self.disable_zero_point = disable_zero_point
        assert 2 <= n_bits <= 16, "bitwidth not supported"
        self.n_bits = n_bits

        if self.disable_zero_point or self.symmetric:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**n_bits - 1

        self.per_channel_axes = per_channel_axes
        self.metric = metric
        self.cluster_counts = None
        self.cluster_dim = None

        self.scale = None
        self.zero_point = None
        self.round_zero_point = None

        # For static percentile-based clipping: store raw ranges
        self.clip_min = None
        self.clip_max = None

        self.cached_xmin = None
        self.cached_xmax = None
        self.dynamic = dynamic
        self.dynamic_method = dynamic_method
        self.deficiency = 0
        self.lwc = lwc
        self.rescale = rescale  # for channel-rescale
        self.rescale_limit = rescale_limit

        init_value = 4.0  # init value of learnable weight clipping

        if lwc:
            if group_size:
                dim1 = int(shape[0] * math.ceil(shape[1] / group_size))
                self.deficiency = shape[-1] % group_size
                if self.deficiency > 0:
                    self.deficiency = group_size - self.deficiency
                    # support for mlc-llm symmetric quantization
                    assert self.symmetric
            else:
                dim1 = shape[0]
            self.upbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)
            self.lowbound_factor = nn.Parameter(torch.ones((dim1, 1)) * init_value)

        if rescale:
            if rescale_limit:
                self.rescale_param = nn.Parameter(torch.zeros(dim1, 1))
            else:
                self.rescale_param = nn.Parameter(torch.ones(dim1, 1))

        self.sigmoid = nn.Sigmoid()

        # Percentile clipping / quantization enable switch
        self.enable = True
        self.group_size = group_size

        self.has_batch_dim = has_batch_dim
        self.is_observing = False

        # Activation dynamic quantization flag
        self.is_dynamic_quant = True

        granularity = f"dim{per_channel_axes[0]}" if len(per_channel_axes) > 0 else "tensor"

        if observe == "percentile":
            self.observer = PercentileObserver(percent=percent, granularity=granularity)
        else:
            self.observer = MinMaxObserver(granularity=granularity)

        self.observered = False
        self.is_weight = is_weight

    # ------------------------------------------------------------------
    # Public API helpers for PTQ pipeline
    # ------------------------------------------------------------------
    def change_n_bits(self, n_bits: int):
        """
        Change bit-width at runtime. If a clipping range has already been
        loaded (clip_min / clip_max), we recompute scale / zero_point based
        on the new bit-width.

        This is important for our W4/W8 fake quant experiments:
            - pct_hi_lo is bit-agnostic,
            - act_bits can be changed at calibrate or load time.
        """
        self.n_bits = n_bits
        if self.disable_zero_point or self.symmetric:
            self.qmin = -(2 ** (n_bits - 1))
            self.qmax = 2 ** (n_bits - 1) - 1
        else:
            self.qmin = 0
            self.qmax = 2**n_bits - 1

        self._recompute_scale_from_clipping()

    def set_quant_state(self, enable: bool, is_dynamic: bool):
        """
        Convenience setter used by wrappers / calibrator to toggle
        fake-quant behavior.
        """
        self.enable = enable
        self.is_dynamic_quant = is_dynamic

    def set_clipping_range(self, xmin, xmax):
        """
        Load a pre-computed clipping range (e.g. from percentile stats).

        This is the main entry point for the percentile-based calibration
        pipeline (pct_stats -> pct_hi_lo -> calibrator):

            - stores (xmin, xmax) as tensors
            - recomputes scale / zero_point from the current bit-width
            - disables observers and dynamic quantization
        """
        if not torch.is_tensor(xmin):
            xmin = torch.as_tensor(xmin, dtype=torch.float32)
        if not torch.is_tensor(xmax):
            xmax = torch.as_tensor(xmax, dtype=torch.float32)

        # Prefer to keep whatever device the caller passes; just align them
        if xmin.device != xmax.device:
            xmax = xmax.to(xmin.device)

        self.clip_min = xmin
        self.clip_max = xmax

        self._recompute_scale_from_clipping()

        # Turn into static fake-quant mode
        self.is_dynamic_quant = False
        self.is_observing = False
        self.observer = None
        self.observered = True

    # ------------------------------------------------------------------
    # Core fake quant logic
    # ------------------------------------------------------------------
    def fake_quant(self, x: torch.Tensor, scale: torch.Tensor, round_zero_point: torch.Tensor):
        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)

        x_int = round_ste(x / scale)

        if round_zero_point is not None:
            x_int = x_int.add(round_zero_point)

        x_int = x_int.clamp(self.qmin, self.qmax)
        x_dequant = x_int

        if round_zero_point is not None:
            x_dequant = x_dequant.sub(round_zero_point)

        x_dequant = x_dequant.mul(scale)

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)

        if self.deficiency > 0:
            x_dequant = x_dequant[:, : -self.deficiency]

        if self.rescale:
            rescale_param = self.rescale_param
            if self.rescale_limit:
                rescale_param = 0.5 + F.sigmoid(rescale_param)
            if len(rescale_param.shape) == 2 and len(x_dequant.shape) == 3:
                rescale_param = rescale_param.unsqueeze(-1)
            x_dequant = x_dequant * rescale_param.to(x_dequant.device)

        return x_dequant

    def forward(self, x: torch.Tensor):
        # No-op for high precision or disabled quantization
        if self.n_bits >= 16 or not self.enable:
            return x

        # Special case metric
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)

        # ------------------------------
        # Weight fake quant (PTQ)
        # ------------------------------
        if self.is_weight:
            # We currently only use static observer-based PTQ for weights.
            if True:  # not self.is_dynamic_quant:
                if self.is_observing:
                    # Stat collection phase: just feed data to observer
                    return x
                if self.observer is not None:
                    self.observer.update(x)
                    xmin, xmax = self.observer.cal_min_max()
                    self.assymmetric_cal_scale(xmin, xmax)
                    self.scale = self.expand_scale_shape_2_x(x, self.scale)
                    self.round_zero_point = self.expand_scale_shape_2_x(x, self.round_zero_point)
                    self.observer = None
                x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
                return x_dequant.type_as(x)

        # ------------------------------
        # Activation fake quant
        # ------------------------------
        if not self.is_dynamic_quant:
            # Static mode:
            #   - either observer-based (legacy),
            #   - or percentile-based via set_clipping_range (preferred).
            if self.is_observing and self.observer is not None:
                # Calibration pass for observer-based static act quant
                self.observer.update(x)
                return x.type_as(x)
            else:
                if (not self.observered) and (self.observer is not None):
                    # Legacy path: compute min/max from observer once
                    xmin, xmax = self.observer.cal_min_max()
                    self.assymmetric_cal_scale(xmin, xmax)
                    self.scale = self.expand_scale_shape_2_x(x, self.scale)
                    self.round_zero_point = self.expand_scale_shape_2_x(x, self.round_zero_point)
                    self.observered = True
                    self.observer = None

                x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
                return x_dequant.type_as(x)

        # Dynamic activation quant (per-token / per-channel / per-tensor)
        if self.dynamic_method == "per_token" or self.dynamic_method == "per_channel":
            self.per_token_dynamic_calibration(x)
        else:
            self.dynamic_per_tensor_calibration(x)

        x_dequant = self.fake_quant(x, self.scale, self.round_zero_point)
        return x_dequant.type_as(x)

    # ------------------------------------------------------------------
    # Calibration helpers
    # ------------------------------------------------------------------
    def expand_scale_shape_2_x(self, x: torch.Tensor, scale: torch.Tensor):
        if self.per_channel_axes:
            dim = self.per_channel_axes[0]
            for i in range(len(x.shape)):
                if i != dim:
                    scale = scale.unsqueeze(i)
        return scale

    def per_token_dynamic_calibration(self, x: torch.Tensor):
        if self.group_size:
            if self.deficiency == 0:
                x = x.reshape(-1, self.group_size)
            else:
                pad_zeros = torch.zeros(
                    (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
                )
                x = torch.cat((x, pad_zeros), dim=1)
                x = x.reshape(-1, self.group_size)

        if self.dynamic_method == "per_channel":
            if len(self.per_channel_axes):
                assert len(self.per_channel_axes) == 1, "must be one"
                reduce_shape = list(range(x.dim()))
                reduce_shape.remove(self.per_channel_axes[0])
            else:
                reduce_shape = list(range(x.dim() - 1))
        else:
            reduce_shape = [-1]

        xmin = x.amin(reduce_shape, keepdim=True)
        xmax = x.amax(reduce_shape, keepdim=True)

        if self.lwc:
            xmax = self.sigmoid(self.upbound_factor) * xmax
            xmin = self.sigmoid(self.lowbound_factor) * xmin

        self.xmin_tmp = xmin.detach()
        self.xmax_tmp = xmax.detach()

        if self.symmetric:
            abs_max = torch.max(xmax.abs(), xmin.abs())
            scale = abs_max / (2 ** (self.n_bits - 1) - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = (2 ** (self.n_bits - 1) - 1) * torch.ones_like(self.scale)
        else:
            dynamic_range = xmax - xmin
            scale = dynamic_range / (2**self.n_bits - 1)
            self.scale = scale.clamp(min=CLIPMIN, max=1e4)
            zero_point = -xmin / self.scale

        if self.disable_zero_point:
            self.round_zero_point = None
        else:
            self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()

    def MaxMin_except_first_dim(self, tensor: torch.Tensor, func):
        # reduce over all dims except the first
        dims = list(range(1, tensor.dim()))
        for dim in dims:
            tensor, _ = func(tensor, dim=dim, keepdim=True)
        return tensor

    def dynamic_per_tensor_calibration(self, x: torch.Tensor):
        if not self.has_batch_dim:
            xmin = x.min()
            xmax = x.max()
        else:
            shape = [1] * len(x.shape)
            shape[0] = -1
            xmin = self.MaxMin_except_first_dim(x, torch.min).view(shape)
            xmax = self.MaxMin_except_first_dim(x, torch.max).view(shape)

        if self.symmetric or self.disable_zero_point:
            self.symmetric_cal_scale(xmin, xmax)
        else:
            self.assymmetric_cal_scale(xmin, xmax)

    def symmetric_cal_scale(self, xmin: torch.Tensor, xmax: torch.Tensor):
        abs_max = torch.max(xmax.abs(), xmin.abs())
        scale = abs_max / (2 ** (self.n_bits - 1) - 1)
        self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        self.round_zero_point = None

    def assymmetric_cal_scale(self, xmin: torch.Tensor, xmax: torch.Tensor):
        dynamic_range = xmax - xmin
        scale = dynamic_range / (2**self.n_bits - 1)
        self.scale = scale.clamp(min=CLIPMIN, max=1e4)
        zero_point = -xmin / self.scale
        self.round_zero_point = zero_point.clamp(min=-1e4, max=1e4).round()

    def _recompute_scale_from_clipping(self):
        """
        Recompute scale / zero_point from stored clipping range and
        current bit-width. No-op if clipping range is not set.
        """
        if self.clip_min is None or self.clip_max is None:
            return

        xmin = self.clip_min
        xmax = self.clip_max

        # Support both symmetric and asymmetric in a single helper
        if self.symmetric or self.disable_zero_point:
            self.symmetric_cal_scale(xmin, xmax)
        else:
            self.assymmetric_cal_scale(xmin, xmax)

    # ------------------------------------------------------------------
    # Helpers for integer export (kept for compatibility, but not used
    # in the W4/W8 fake-quant accuracy study).
    # ------------------------------------------------------------------
    def normal_quantize(self, x, scales: torch.Tensor, mig_cof: torch.Tensor):
        s = (scales / mig_cof).max()
        s = s / (2**self.n_bits - 1)
        self.scale = s
        # only support symmetric quantization
        self.round_zero_point = None

    def scale_frexp(self):
        k = 16
        m = (self.scale * (2**k)).round()
        self.scale = m * (2**(-k))
        return self.scale

    def register_scales_and_zeros(self):
        self.register_buffer("scales", self.scale)
        self.register_buffer("zeros", self.round_zero_point)
        del self.scale
        del self.round_zero_point

    def quant2int(self, x: torch.Tensor):
        if self.n_bits >= 16 or not self.enable:
            return x
        if self.metric == "fix0to1":
            return x.mul_(2**self.n_bits - 1).round_().div_(2**self.n_bits - 1)

        if self.deficiency > 0:
            pad_zeros = torch.zeros(
                (x.shape[0], self.deficiency), dtype=x.dtype, device=x.device
            )
            x = torch.cat((x, pad_zeros), dim=1)

        if self.group_size:
            assert len(x.shape) == 2, "only support linear layer now"
            dim1, dim2 = x.shape
            x = x.reshape(-1, self.group_size)

        x_int = round_ste(x / self.scale)
        if self.round_zero_point is not None:
            x_int = x_int.add(self.round_zero_point)
        x_int = x_int.clamp(self.qmin, self.qmax)

        if self.group_size:
            x_int = x_int.reshape(dim1, dim2)
        return x_int

    def dequant(self, x_int: torch.Tensor):
        if self.group_size:
            assert len(x_int.shape) == 2, "only support linear layer now"
            dim1, dim2 = x_int.shape
            x_int = x_int.reshape(-1, self.group_size)

        x_dequant = x_int
        if self.round_zero_point is not None:
            x_dequant = x_dequant.sub(self.round_zero_point)
        x_dequant = x_dequant.mul(self.scale)

        if self.group_size:
            x_dequant = x_dequant.reshape(dim1, dim2)
        if self.deficiency > 0:
            x_dequant = x_dequant[:, : -self.deficiency]

        if self.rescale:
            rescale_param = self.rescale_param
            if self.rescale_limit:
                rescale_param = F.sigmoid(rescale_param) + 0.5
            x_dequant = x_dequant * self.rescale_param
        return x_dequant


class ActQuantizer(nn.Module):
    """
    Thin activation quantizer wrapper around UniformAffineQuantizer.

    Kept mainly for compatibility with any existing code that expects an
    `ActQuantizer` type. New code should directly use `UniformAffineQuantizer`
    with `is_weight=False` and control it via `set_clipping_range` and
    `set_quant_state`.
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, **kwargs):
        super().__init__()
        kwargs.setdefault("is_weight", False)
        self.quantizer = UniformAffineQuantizer(
            n_bits=n_bits,
            symmetric=symmetric,
            **kwargs,
        )
        # 0: not calibrated (identity), 1: apply fake quant
        self.register_buffer("calibed_enabled", torch.tensor([1], dtype=torch.uint8))

    def set_clipping_range(self, xmin, xmax):
        self.quantizer.set_clipping_range(xmin, xmax)
        self.calibed_enabled.fill_(1)

    def change_n_bits(self, n_bits: int):
        self.quantizer.change_n_bits(n_bits)

    def set_quant_state(self, enable: bool, is_dynamic: bool):
        self.quantizer.set_quant_state(enable=enable, is_dynamic=is_dynamic)

    def forward(self, x: torch.Tensor):
        if self.calibed_enabled.item() == 0:
            return x
        return self.quantizer(x)


if __name__ == "__main__":
    cfg = {"dynamic_method": "per_tensor", "n_bits": 8, "symmetric": True}
    weight = torch.randn(100, 100)
    quantizer = UniformAffineQuantizer(**cfg, is_weight=True)
    quantizer.is_observing = True
    _ = quantizer(weight)  # observe
    quantizer.is_observing = False
    weight_quant = quantizer(weight)
    diff = weight - weight_quant
    print(diff.sum())
