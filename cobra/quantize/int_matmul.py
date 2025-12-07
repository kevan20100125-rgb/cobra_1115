import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import UniformAffineQuantizer


class QuantMatMul(nn.Module):
    """
    Quantized MatMul (MambaQuant-style fake quant).

    - Wraps a generic matmul-like function (default: torch.matmul).
    - Uses UniformAffineQuantizer for:
        * x1 fake quant (activation)
        * x2 fake quant (activation)
    - No INT kernel: all math is still float, with quant/dequant around matmul.

    Typical usage in Cobra:
        - x1 / x2 are activations (e.g., queries / keys / values).
        - Percentile-based clipping can be injected via
          set_x1_clipping_range / set_x2_clipping_range.
    """

    def __init__(
        self,
        x1_quant_params: dict = {"dynamic_method": "per_tensor"},
        x2_quant_params: dict = {"dynamic_method": "per_tensor"},
        disable_act_quant: bool = False,
        observe: str = "minmax",
        matmul_func=torch.matmul,
    ):
        super().__init__()

        # De-activate quantized forward by default
        self.use_act_quant = False
        self.use_weight_quant = False  # kept for API compatibility

        # Initialize quantizers (both are activations)
        self.i_cluster_counts = None
        self.x1_quantizer = UniformAffineQuantizer(
            **x1_quant_params,
            has_batch_dim=True,
            observe=observe,
        )
        self.x2_quantizer = UniformAffineQuantizer(
            **x2_quant_params,
            has_batch_dim=True,
            observe=observe,
        )

        self.matmul_func = matmul_func
        self.disable_act_quant = disable_act_quant

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        """
        Toggle fake quantization for activations.

        `weight_quant` is accepted for API compatibility but has no effect
        (QuantMatMul has no learned weights).
        """
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

        if self.x1_quantizer is not None:
            self.x1_quantizer.set_quant_state(
                enable=act_quant,
                is_dynamic=self.x1_quantizer.dynamic,
            )
        if self.x2_quantizer is not None:
            self.x2_quantizer.set_quant_state(
                enable=act_quant,
                is_dynamic=self.x2_quantizer.dynamic,
            )

    def quant_x1(self, x1: torch.Tensor):
        if (
            self.use_act_quant
            and not self.disable_act_quant
            and self.x1_quantizer is not None
        ):
            x1 = self.x1_quantizer(x1)
        return x1

    def quant_x2(self, x2: torch.Tensor):
        if (
            self.use_act_quant
            and not self.disable_act_quant
            and self.x2_quantizer is not None
        ):
            x2 = self.x2_quantizer(x2)
        return x2

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        # Special per-token reshape path (kept for compatibility)
        if hasattr(self, "pertoken"):
            B, L, ED, N = x1.shape
            x1_flat = x1.reshape(B, L * ED, N)
            x1_flat = self.quant_x1(x1_flat)
            x1 = x1_flat.reshape(B, L, ED, N)
            x2 = self.quant_x2(x2)
            out = self.matmul_func(x1, x2)
        else:
            x1 = self.quant_x1(x1)
            x2 = self.quant_x2(x2)
            out = self.matmul_func(x1, x2)
        return out

    # ------------------------------------------------------------------
    # Helpers for percentile-based calibration and bit-width sweeps
    # ------------------------------------------------------------------
    def set_x1_clipping_range(self, xmin, xmax):
        """
        Load pre-computed clipping range (xmin, xmax) for x1.

        Typically called by the calibrator after pct_stats -> pct_hi_lo
        to inject bit-agnostic activation clipping into the quantizer.
        """
        if self.x1_quantizer is not None:
            self.x1_quantizer.set_clipping_range(xmin, xmax)

    def set_x2_clipping_range(self, xmin, xmax):
        """
        Load pre-computed clipping range (xmin, xmax) for x2.
        """
        if self.x2_quantizer is not None:
            self.x2_quantizer.set_clipping_range(xmin, xmax)

    def change_bits(self, x1_bits: int = None, x2_bits: int = None):
        """
        Change bit-widths at runtime for x1 / x2 quantizers.

        Useful for toggling between 4/8-bit experiments with the same
        pct_hi_lo statistics.
        """
        if x1_bits is not None and self.x1_quantizer is not None:
            self.x1_quantizer.change_n_bits(x1_bits)
        if x2_bits is not None and self.x2_quantizer is not None:
            self.x2_quantizer.change_n_bits(x2_bits)
