# cobra/switches/quant_finalize.py

"""
quant_finalize.py

CLI entrypoint: "Finalize PTQ & Integer Export"

End-to-end responsibilities (最後一哩路):

    1) Load a pretrained Cobra VLM checkpoint.
    2) Wrap selected modules with Quant* wrappers (linear/conv) according
       to a wrapping policy.
    3) Load precomputed percentile clipping bounds (hi/lo map) from disk
       and apply them to activation quantizers via `pct.calibrator`.
    4) Apply the shared output-projector rotation R = H K on the LLM
       lm_head (if enabled).
    5) Export integer weights and activation quantization parameters in
       a single blob (torch.save) for runtime backends to consume.

This script does NOT:
    - Run any dataloader or collect activation statistics (that is
      handled by `quant_calibrate.py` + `quant_pct_apply.py`).
    - Recompute best percentiles (that is `pct/best_percentile.py`).
"""

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import draccus
import torch
from torch import nn

from cobra.conf import ModelConfig, ModelRegistry
from cobra.models import (
    get_llm_backbone_and_tokenizer,
    get_vision_backbone_and_transform,
    get_vlm,
)
from cobra.overwatch import initialize_overwatch
from cobra.quantize.pct.calibrator import calibrate_model_from_hi_lo
from cobra.quantize.wrap.policy import WrapPolicyConfig
from cobra.quantize.finalize.int_export import (
    IntExportConfig,
    export_int_quant_state,
    save_int_export,
)
from cobra.quantize.rotate.projector import (
    ProjectorRotationConfig,
    rotate_cobra_vlm_output_projector_inplace,
)
from cobra.integration.wrap_replace import wrap_model_for_quantization
from cobra.util import set_global_seed

overwatch = initialize_overwatch(__name__)


_CANONICAL_TARGETS: Tuple[str, ...] = ("vision.dino", "vision.siglip", "llm", "projector")

# Default shared-KLT matrix path (user requested: hard-coded in quant_finalize.py)
_SHARED_KLT_PATH = Path("outputs/quantize/shared_klt.pt")

# =====================================================================
# Config
# =====================================================================


@dataclass
class QuantFinalizeConfig:
    """
    Configuration for the PTQ finalization + integer export step.

    Model / Checkpoint
    ------------------
    model:
        ModelConfig from `cobra/conf/models.py`.
    stage:
        Pretraining stage in {"align", "finetune", "full-finetune"}; used to
        determine which checkpoint to load (same semantics as training).
    pretrained_checkpoint_root:
        Optional run directory; if None, defaults to `runs/<model_id>`.
    hf_token:
        Either:
            - environment variable name containing a HF token, or
            - Path to a file storing the token.

    Percentile calibration
    ----------------------
    pct_hi_lo_in:
        Path to the hi/lo map produced by `quant_pct_apply.py` or
        `quant_calibrate.py` (`torch.save` dict).

    Quantization mode (four combos)
    --------------------------------
    weight_bits:
        Bits for weights (4 or 8).
    act_bits:
        Bits for activations (4 or 8).
    signed_weights:
        Whether to use symmetric weight quantization.
    signed_activations:
        Whether to treat activations as signed. Actual ranges come from
        calibration; this only records the logical sign.

    Targets included in this export
    -------------------------------
    include_vision_dino / include_vision_siglip / include_llm / include_projector:
        Per-target boolean switches.

    Output-projector rotation
    -------------------------
    use_klt:
        Whether to apply KLT part of R = H K to W_out.
    use_hadamard:
        Whether to apply Hadamard part of R = H K to W_out.
    shared_klt:
        For documentation/sanity only; indicates we expect a shared K matrix.
    klt_path:
        Optional path to a file containing KLT matrix. If provided and
        `use_klt=True`, we load:
            - If object is a Tensor: use as K.
            - If object is a dict, we look for key "K" or "klt".

    Export
    ------
    out_path:
        Destination of the integer export blob (.pt).
    device:
        "cuda" or "cpu" for finalization.
    seed:
        Global seed for reproducibility (mostly for any randomness inside K).
    """

    # --- Model / checkpoint ---
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.COBRA_3B.model_id)
    )
    stage: str = "finetune"
    pretrained_checkpoint_root: Optional[Path] = None
    hf_token: Union[str, Path] = Path(".hf_token")

    # --- Percentile calibration ---
    pct_hi_lo_in: Path = Path("outputs/quantize/pct_hi_lo.pt")

    # --- Quantization bits ---
    weight_bits: int = 8 # now supports {2,4,8,16};
    act_bits: int = 8 # same as above
    signed_weights: bool = True
    signed_activations: bool = True

    # --- Targets ---
    include_vision_dino: bool = True
    include_vision_siglip: bool = True
    include_llm: bool = True
    include_projector: bool = True

    # --- Rotation ---
    use_klt: bool = True
    use_hadamard: bool = True
    shared_klt: bool = True
    klt_path: Optional[Path] = _SHARED_KLT_PATH

    # --- Export & runtime ---
    out_path: Path = Path("outputs/quantize/int_export.pt")
    device: str = "cuda"
    seed: int = 7

    def __post_init__(self) -> None:
        valid_bits = (2, 4, 8, 16)

        # Explicitly reject 1-bit to avoid implying binary quant support
        if self.weight_bits == 1 or self.act_bits == 1:
            raise ValueError(
                f"1-bit quantization is not supported in quant_finalize "
                f"(got W{self.weight_bits}A{self.act_bits}). "
                f"Use one of {valid_bits}."
            )


        if self.weight_bits not in valid_bits:
            raise ValueError(
                f"weight_bits must be one of {valid_bits}, got {self.weight_bits}"
            )
        if self.act_bits not in valid_bits:
            raise ValueError(
                f"act_bits must be one of {valid_bits}, got {self.act_bits}"
            )

        if self.device == "cuda" and not torch.cuda.is_available():
            overwatch.warning("CUDA not available; falling back to CPU for finalization")
            self.device = "cpu"

        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convenience for passing into calibrator / export
    def enabled_targets(self) -> Sequence[str]:
        targets: list[str] = []
        if self.include_vision_dino:
            targets.append("vision.dino")
        if self.include_vision_siglip:
            targets.append("vision.siglip")
        if self.include_llm:
            targets.append("llm")
        if self.include_projector:
            targets.append("projector")
        return targets


# =====================================================================
# Helpers
# =====================================================================


def _load_hi_lo_map(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"hi/lo map not found at: {path}")
    hi_lo_map = torch.load(path, map_location="cpu")
    if not isinstance(hi_lo_map, dict):
        raise TypeError(f"hi/lo map must be a dict, got {type(hi_lo_map)}")
    return hi_lo_map


def _load_klt_matrix(path: Optional[Path]) -> Optional[torch.Tensor]:
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
        f"KLT file {path} must contain a Tensor or a dict with key 'K' or 'klt'; got {type(obj)}"
    )


def _load_hf_token(spec: Union[str, Path]) -> Optional[str]:
    """
    Resolve HF token in a minimal, explicit way without hidden side effects.

    - If `spec` is a Path: read the file if it exists; otherwise warn and return None.
    - If `spec` is a str: treat as an environment variable name; if unset, warn and return None.

    Returning None is allowed; the caller may pass it through to HF APIs that
    can operate with public checkpoints.
    """
    if isinstance(spec, Path):
        if not spec.is_file():
            overwatch.warning(
                "[quant_finalize] HF token file does not exist; proceeding without explicit token",
                extra={"hf_token_path": str(spec)},
            )
            return None
        token = spec.read_text().strip()
        if not token:
            overwatch.warning(
                "[quant_finalize] HF token file is empty; proceeding without explicit token",
                extra={"hf_token_path": str(spec)},
            )
            return None
        return token

    # Treat as env var name
    env_var = spec
    token = os.environ.get(env_var)
    if token is None or not token.strip():
        overwatch.warning(
            "[quant_finalize] HF token env var missing or empty; proceeding without explicit token",
            extra={"env_var": env_var},
        )
        return None
    return token.strip()


# =====================================================================
# Main CLI
# =====================================================================


@draccus.wrap()
def quant_finalize(cfg: QuantFinalizeConfig) -> None:
    """
    Orchestrate:
        - model load
        - wrapping
        - percentile hi/lo apply into quantizers
        - projector rotation
        - integer export (supports W{2,4,8,16}A{2,4,8,16})
    """
    set_global_seed(cfg.seed)
    device = torch.device(cfg.device)
    dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    # ------------------------------------------------------------------
    # Model / backbone construction
    # ------------------------------------------------------------------
    model_id = cfg.model.model_id

    # Resolve HF token (no hidden _resolve_hf_token; explicit, flag-driven)
    hf_token = _load_hf_token(cfg.hf_token)

    overwatch.info(
        "[quant_finalize] Loading model backbones",
        extra={"model_id": model_id, "stage": cfg.stage},
    )

    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id,
        image_resize_strategy=cfg.model.image_resize_strategy,
    )

    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id,
        llm_max_length=cfg.model.llm_max_length,
        hf_token=hf_token,
        inference_mode=True,
    )

    overwatch.info(
        "[quant_finalize] Instantiating CobraVLM",
        extra={"arch_specifier": cfg.model.arch_specifier},
    )
    vlm = get_vlm(
        model_id,
        cfg.model.arch_specifier,
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
    )

    vlm.freeze_backbones(cfg.stage)

    if cfg.pretrained_checkpoint_root is not None:
        run_dir = cfg.pretrained_checkpoint_root
    else:
        run_dir = Path("runs") / model_id

    overwatch.info(
        "[quant_finalize] Loading checkpoint",
        extra={"run_dir": str(run_dir), "stage": cfg.stage},
    )
    vlm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=None)

    vlm.to(device=device, dtype=dtype)
    vlm.eval()

    # ------------------------------------------------------------------
    # Wrap model with Quant* modules
    # ------------------------------------------------------------------
    wrap_cfg = WrapPolicyConfig(
        enable_vision_dino=cfg.include_vision_dino,
        enable_vision_siglip=cfg.include_vision_siglip,
        enable_llm=cfg.include_llm,
        enable_projector=cfg.include_projector,
        include_linear=True,
        include_conv=True,
    )

    overwatch.info(
        "[quant_finalize] Wrapping model for quantization",
        extra=asdict(wrap_cfg),
    )
    wrap_registry = wrap_model_for_quantization(
        vlm,
        policy_cfg=wrap_cfg,
    )

    # ------------------------------------------------------------------
    # Load hi/lo map and calibrate activation quantizers
    # ------------------------------------------------------------------
    overwatch.info(
        "[quant_finalize] Loading hi/lo map",
        extra={"pct_hi_lo_in": str(cfg.pct_hi_lo_in)},
    )
    hi_lo_map = _load_hi_lo_map(cfg.pct_hi_lo_in)

    overwatch.info(
        "[quant_finalize] Applying hi/lo map to activation quantizers",
        extra={
            "act_bits": cfg.act_bits,
            "signed_activations": cfg.signed_activations,
            "targets": cfg.enabled_targets(),
        },
    )

    calibrate_model_from_hi_lo(
        vlm,
        hi_lo_map=hi_lo_map,
        act_bits=cfg.act_bits,
        signed=cfg.signed_activations,
        include_targets=cfg.enabled_targets(),
    )

    # ------------------------------------------------------------------
    # Apply projector rotation (R = H K)
    # ------------------------------------------------------------------
    K: Optional[torch.Tensor] = None
    if cfg.use_klt:

        overwatch.info(
            "[quant_finalize] Loading shared KLT matrix",
            extra={"klt_path": str(cfg.klt_path)},
        )

        K = _load_klt_matrix(cfg.klt_path)
        if K is None:
            overwatch.warning(
                "[quant_finalize] use_klt=True but KLT matrix is None; "
                "KLT part of rotation will be skipped."
            )

    proj_rot_cfg = ProjectorRotationConfig(
        use_klt=cfg.use_klt,
        use_hadamard=cfg.use_hadamard,
        shared_klt=cfg.shared_klt,
    )

    try:
        module_path, lm_head = rotate_cobra_vlm_output_projector_inplace(
            vlm,
            K=K,
            cfg=proj_rot_cfg,
        )
        overwatch.info(
            "[quant_finalize] Applied projector rotation",
            extra={"module_path": module_path},
        )
    except Exception as exc:
        overwatch.warning(
            f"[quant_finalize] Failed to rotate output projector: {exc!r}; "
            "continuing without rotation."
        )

    # ------------------------------------------------------------------
    # Integer export
    # ------------------------------------------------------------------
    int_cfg = IntExportConfig(
        weight_bits=cfg.weight_bits,
        act_bits=cfg.act_bits,
        signed_weights=cfg.signed_weights,
        signed_activations=cfg.signed_activations,
        include_vision_dino=cfg.include_vision_dino,
        include_vision_siglip=cfg.include_vision_siglip,
        include_llm=cfg.include_llm,
        include_projector=cfg.include_projector,
    )

    overwatch.info(
        "[quant_finalize] Exporting integer quant state",
        extra=asdict(int_cfg),
    )

    export_blob = export_int_quant_state(
        vlm,
        cfg=int_cfg,
    )

    save_int_export(
        export_blob,
        out_path=cfg.out_path,
    )

    overwatch.info(
        "[quant_finalize] Done.",
        extra={
            "out_path": str(cfg.out_path),
            "weight_bits": cfg.weight_bits,
            "act_bits": cfg.act_bits,
        },
    )


if __name__ == "__main__":
    quant_finalize()


