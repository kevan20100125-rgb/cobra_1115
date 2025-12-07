import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantizer import UniformAffineQuantizer


class QuantConv1d(nn.Conv1d):
    """
    Quantized 1D convolution (MambaQuant-style fake quant).

    - Wraps an existing nn.Conv1d.
    - Uses UniformAffineQuantizer for:
        * weight fake quant (PTQ)
        * activation fake quant (runtime, optional)
    - No INT kernel: all math is still float, with quant/dequant around F.conv1d.
    """

    def __init__(
        self,
        org_module: nn.Conv1d,
        weight_quant_params: dict = {"dynamic_method": "per_tensor"},
        act_quant_params: dict = {"dynamic_method": "per_tensor"},
        observe: str = "minmax",
        disable_input_quant: bool = False,
    ):
        super().__init__(
            org_module.in_channels,
            org_module.out_channels,
            kernel_size=org_module.kernel_size,
            stride=org_module.stride,
            padding=org_module.padding,
            dilation=org_module.dilation,
            bias=org_module.bias is not None,
            groups=org_module.groups,
        )

        self.fwd_kwargs = dict()
        self.fwd_func = F.conv1d

        # Reuse original parameters
        self.weight = org_module.weight
        if org_module.bias is not None:
            self.bias = org_module.bias
        else:
            self.bias = None

        # De-activate the quantized forward by default
        self.use_weight_quant = False
        self.use_act_quant = False

        # Initialize weight quantizer (PTQ / fake quant)
        self.weight_quantizer = UniformAffineQuantizer(
            **weight_quant_params,
            shape=org_module.weight.shape,
            is_weight=True,
            observe=observe,
        )

        # Activation quantizer (fake quant, with batch dimension awareness)
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(
                **act_quant_params,
                has_batch_dim=True,
                observe=observe,
            )
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

        # Convolution hyperparameters
        self.stride = org_module.stride
        self.padding = org_module.padding
        self.dilation = org_module.dilation
        self.groups = org_module.groups

        # Flag to avoid repeatedly re-quantizing already-quantized weights
        self.weight_quantized = False

    def forward(self, input: torch.Tensor):
        # ------------------------------
        # Select weight / bias
        # ------------------------------
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias

        elif self.use_weight_quant:
            # Weight fake quantization path
            if self.weight_quantizer.is_observing:
                # Collecting stats only, do not quantize yet
                weight = self.weight
            elif not self.weight_quantized:
                # First time: quantize and cache back into self.weight
                self.weight = torch.nn.Parameter(self.weight_quantizer(self.weight))
                weight = self.weight
                self.weight_quantized = True
            else:
                # Already quantized weights
                weight = self.weight
            bias = self.bias

        else:
            # No weight quantization
            weight = self.weight
            bias = self.bias

        # ------------------------------
        # Activation fake quant
        # ------------------------------
        if self.use_act_quant and not self.disable_input_quant and self.act_quantizer is not None:
            input = self.act_quantizer(input)

        # Align dtype/device
        if bias is not None:
            bias = bias.to(weight)
        out = self.fwd_func(
            input.to(weight),
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            **self.fwd_kwargs,
        )
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        """
        Toggle fake quantization for weights and activations.

        This only controls the wrapper-level flags. More fine-grained control
        (e.g., static vs dynamic activation quant) is handled inside
        UniformAffineQuantizer via its own `set_quant_state` / `set_clipping_range`.
        """
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

        # Propagate to underlying quantizers
        if self.weight_quantizer is not None:
            # Weight quant is always static PTQ in our pipeline
            self.weight_quantizer.set_quant_state(enable=weight_quant, is_dynamic=False)

        if self.act_quantizer is not None:
            # Dynamic vs static is determined by the quantizer config; here we toggle enable only.
            self.act_quantizer.set_quant_state(
                enable=act_quant, is_dynamic=self.act_quantizer.dynamic
            )

    def set_act_clipping_range(self, xmin, xmax):
        """
        Convenience wrapper for percentile-based activation clipping.

        Typically called by the calibrator after pct_stats -> pct_hi_lo to
        load bit-agnostic (xmin, xmax) into the activation quantizer.
        """
        if self.act_quantizer is not None:
            self.act_quantizer.set_clipping_range(xmin, xmax)

    def change_bits(self, weight_bits: int = None, act_bits: int = None):
        """
        Change bit-widths at runtime for weight / activation quantizers.
        Useful for toggling between W4/W8 experiments with the same
        pct_hi_lo files.
        """
        if weight_bits is not None and self.weight_quantizer is not None:
            self.weight_quantizer.change_n_bits(weight_bits)
        if act_bits is not None and self.act_quantizer is not None:
            self.act_quantizer.change_n_bits(act_bits)


class QuantConv2d(nn.Conv2d):
    """
    Quantized 2D convolution (MambaQuant-style fake quant).

    Same design as QuantConv1d, but for nn.Conv2d.
    """

    def __init__(
        self,
        org_module: nn.Conv2d,
        weight_quant_params: dict = {"dynamic_method": "per_tensor"},
        act_quant_params: dict = {"dynamic_method": "per_tensor"},
        disable_input_quant: bool = False,
        observe: str = "minmax",
    ):
        super().__init__(
            org_module.in_channels,
            org_module.out_channels,
            org_module.kernel_size,
            stride=org_module.stride,
            padding=org_module.padding,
            dilation=org_module.dilation,
            groups=org_module.groups,
            bias=org_module.bias is not None,
        )

        self.fwd_kwargs = dict()
        self.fwd_func = F.conv2d

        # Reuse original parameters
        self.weight = org_module.weight
        if org_module.bias is not None:
            self.bias = org_module.bias
        else:
            self.bias = None

        # De-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False

        # Initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(
            **weight_quant_params,
            shape=org_module.weight.shape,
            is_weight=True,
            observe=observe,
        )
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(
                **act_quant_params,
                has_batch_dim=True,
                observe=observe,
            )
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

        self.in_channels = org_module.in_channels
        self.out_channels = org_module.out_channels
        self.kernel_size = org_module.kernel_size
        self.stride = org_module.stride
        self.padding = org_module.padding
        self.dilation = org_module.dilation
        self.groups = org_module.groups

        self.weight_quantized = False

    def forward(self, input: torch.Tensor):
        # ------------------------------
        # Select weight / bias
        # ------------------------------
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias

        elif self.use_weight_quant:
            # Weight fake quantization path
            if self.weight_quantizer.is_observing:
                # Collecting stats only, do not quantize yet
                weight = self.weight
            elif not self.weight_quantized:
                # First time: quantize and cache back into self.weight
                self.weight = torch.nn.Parameter(self.weight_quantizer(self.weight))
                weight = self.weight
                self.weight_quantized = True
            else:
                # Already quantized weights
                weight = self.weight
            bias = self.bias

        else:
            # No weight quantization
            weight = self.weight
            bias = self.bias

        # ------------------------------
        # Activation fake quant
        # ------------------------------
        if self.use_act_quant and not self.disable_input_quant and self.act_quantizer is not None:
            input = self.act_quantizer(input)

        # Align dtype/device
        if bias is not None:
            bias = bias.to(weight)
        out = self.fwd_func(
            input.to(weight),
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            **self.fwd_kwargs,
        )
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

        if self.weight_quantizer is not None:
            self.weight_quantizer.set_quant_state(enable=weight_quant, is_dynamic=False)

        if self.act_quantizer is not None:
            self.act_quantizer.set_quant_state(
                enable=act_quant, is_dynamic=self.act_quantizer.dynamic
            )

    def set_act_clipping_range(self, xmin, xmax):
        if self.act_quantizer is not None:
            self.act_quantizer.set_clipping_range(xmin, xmax)

    def change_bits(self, weight_bits: int = None, act_bits: int = None):
        if weight_bits is not None and self.weight_quantizer is not None:
            self.weight_quantizer.change_n_bits(weight_bits)
        if act_bits is not None and self.act_quantizer is not None:
            self.act_quantizer.change_n_bits(act_bits)


class QuantConv3d(nn.Conv3d):
    """
    Quantized 3D convolution (MambaQuant-style fake quant).

    Same design as QuantConv1d/2d, but for nn.Conv3d.
    """

    def __init__(
        self,
        org_module: nn.Conv3d,
        weight_quant_params: dict = {"dynamic_method": "per_tensor"},
        act_quant_params: dict = {"dynamic_method": "per_tensor"},
        disable_input_quant: bool = False,
        observe: str = "minmax",
    ):
        super().__init__(
            org_module.in_channels,
            org_module.out_channels,
            org_module.kernel_size,
            stride=org_module.stride,
            padding=org_module.padding,
            dilation=org_module.dilation,
            groups=org_module.groups,
            bias=org_module.bias is not None,
        )

        self.fwd_kwargs = dict()
        self.fwd_func = F.conv3d

        # Reuse original parameters
        self.weight = org_module.weight
        if org_module.bias is not None:
            self.bias = org_module.bias
        else:
            self.bias = None

        # De-activate the quantized forward default
        self.use_weight_quant = False
        self.use_act_quant = False

        # Initialize quantizer
        self.weight_quantizer = UniformAffineQuantizer(
            **weight_quant_params,
            shape=org_module.weight.shape,
            is_weight=True,
            observe=observe,
        )
        if not disable_input_quant:
            self.act_quantizer = UniformAffineQuantizer(
                **act_quant_params,
                has_batch_dim=True,
                observe=observe,
            )
        else:
            self.act_quantizer = None

        self.disable_input_quant = disable_input_quant
        self.use_temporary_parameter = False

        self.in_channels = org_module.in_channels
        self.out_channels = org_module.out_channels
        self.kernel_size = org_module.kernel_size
        self.stride = org_module.stride
        self.padding = org_module.padding
        self.dilation = org_module.dilation
        self.groups = org_module.groups

        self.weight_quantized = False

    def forward(self, input: torch.Tensor):
        # ------------------------------
        # Select weight / bias
        # ------------------------------
        if self.use_temporary_parameter:
            weight = self.temp_weight
            bias = self.temp_bias

        elif self.use_weight_quant:
            # Weight fake quantization path
            if self.weight_quantizer.is_observing:
                # Collecting stats only, do not quantize yet
                weight = self.weight
            elif not self.weight_quantized:
                # First time: quantize and cache back into self.weight
                self.weight = torch.nn.Parameter(self.weight_quantizer(self.weight))
                weight = self.weight
                self.weight_quantized = True
            else:
                # Already quantized weights
                weight = self.weight
            bias = self.bias

        else:
            # No weight quantization
            weight = self.weight
            bias = self.bias

        # ------------------------------
        # Activation fake quant
        # ------------------------------
        if self.use_act_quant and not self.disable_input_quant and self.act_quantizer is not None:
            input = self.act_quantizer(input)

        # Align dtype/device
        if bias is not None:
            bias = bias.to(weight)
        out = self.fwd_func(
            input.to(weight),
            weight,
            bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
            **self.fwd_kwargs,
        )
        return out

    def set_quant_state(self, weight_quant: bool = False, act_quant: bool = False):
        self.use_weight_quant = weight_quant
        self.use_act_quant = act_quant

        if self.weight_quantizer is not None:
            self.weight_quantizer.set_quant_state(enable=weight_quant, is_dynamic=False)

        if self.act_quantizer is not None:
            self.act_quantizer.set_quant_state(
                enable=act_quant, is_dynamic=self.act_quantizer.dynamic
            )

    def set_act_clipping_range(self, xmin, xmax):
        if self.act_quantizer is not None:
            self.act_quantizer.set_clipping_range(xmin, xmax)

    def change_bits(self, weight_bits: int = None, act_bits: int = None):
        if weight_bits is not None and self.weight_quantizer is not None:
            self.weight_quantizer.change_n_bits(weight_bits)
        if act_bits is not None and self.act_quantizer is not None:
            self.act_quantizer.change_n_bits(act_bits)
