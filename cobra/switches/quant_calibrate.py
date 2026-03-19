# cobra/switches/quant_calibrate.py 

"""
quant_calibrate.py

CLI entrypoint for percentile-based activation calibration.

Pipeline role (three-stage PTQ):
    quant_calibrate.py
         - Run model + dataloader once
         - Collect activation statistics and build percentile stats
         - (Convenience mode) Optionally convert stats → hi/lo directly
           and write pct_hi_lo_out + pct_summary_out

Responsibilities of this script:
    - Initialize a Cobra VLM and its pretraining dataset (align / finetune)
      using the existing ModelConfig / DatasetConfig infrastructure.
    - Run a (single-node) DataLoader to stream calibration batches through the model.
    - Use `cobra.quantize.pct.collect` to register activation collectors
      and build percentile stats.
    - Use `cobra.quantize.pct.apply.build_hi_lo_map` to:
          stats -> best-percentile map (internal) -> (hi, lo)
      and save the resulting hi/lo clipping map + JSON summary to disk.

Design notes in this variant:
    - Quantization-related knobs（bits/backend/哪些 target 進 percentile pipeline）
      由 QuantRuntimeConfig 統一管理：
          * quant_bits + backend -> (weight_bits, act_bits, mode, use_pct_for...)
      這樣 quant_calibrate / load_quantized_vlm 共享同一套
      bits/backend/targets 決策邏輯。
    - Vision backbones (DINO / SigLIP) participation in percentile pipeline
      仍由 enable_vision_* + vision_in_pct_pipeline 控制，但實際啟用與否
      會反映在 QuantRuntimeConfig.use_pct_for。
    - Convenience mode:同時產出 pct_hi_lo_out + pct_summary_out，可直接給 load_quantized_vlm 使用
"""

import json
import os
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import draccus
import torch
from torch import nn
from torch.utils.data import DataLoader

from cobra.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from cobra.models import (
    get_llm_backbone_and_tokenizer,
    get_vision_backbone_and_transform,
    get_vlm,
)
from cobra.overwatch import initialize_overwatch
from cobra.preprocessing import get_dataset_and_collator
from cobra.quantize.pct.collect import (
    build_activation_stats,
    register_activation_collectors,
    remove_activation_collectors,
    LLMActivationTapContext,
    set_global_llm_tap_context,
    get_global_llm_tap_context,
)
from cobra.quantize.pct.apply import build_hi_lo_map
from cobra.quantize.runtime.act_policy import (
    LLM_ACT_MODE_DEFAULT,
    LLM_ACT_MODE_MAMBA_SENSITIVE,
    LLM_ACT_MODE_OUT_PROJ_ONLY,
    filter_target_module_map_for_llm_mode,
    normalize_llm_act_mode,
    summarize_llm_module_paths,
)
from cobra.quantize.runtime.config import QuantRuntimeConfig
from cobra.quantize.wrap.policy import WrapPolicyConfig
from cobra.quantize.wrap.registry import build_wrap_registry
from cobra.quantize.targets import CANONICAL_TARGETS
from cobra.quantize.resolver.artifact_resolver import (
    ENV_DISABLE_MAMBA_FAST_PATH,
    resolve_mamba_sensitive_projection_gates,
)
from cobra.util import set_global_seed

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch => Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# =====================================================================
# Config
# =====================================================================


@dataclass
class QuantCalibrateConfig:
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.COBRA_3B.model_id)
    )
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(
            DatasetRegistry.TEXTVQA_100_CALIB.dataset_id
        )
    )

    stage: str = "align"
    pretrained_checkpoint_root: Optional[Path] = None
    hf_token: Union[str, Path] = Path(".hf_token")

    per_device_batch_size: int = 8
    num_workers: int = 4
    max_calib_batches: int = 0

    act_bits: Optional[int] = None
    signed_activations: bool = True

    quant_bits: str = "W8A8"
    backend: str = "fake"

    enable_vision_dino: bool = True
    enable_vision_siglip: bool = True
    enable_llm: bool = True
    enable_projector: bool = True
    vision_in_pct_pipeline: bool = True

    tau_growth: float = 5.0
    symmetric_clipping: bool = True
    max_samples_per_module: int = 5_000_000

    pct_stats_out: Path = Path("outputs/quantize/pct_stats.pt")
    pct_hi_lo_out: Path = Path("outputs/quantize/pct_hi_lo.pt")
    pct_summary_out: Path = Path("outputs/quantize/pct_calibrate_summary.json")

    seed: int = 7
    device: str = "cuda"

    quant_cfg: QuantRuntimeConfig = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.quant_cfg = QuantRuntimeConfig.from_bits_backend(
            bits=self.quant_bits,
            backend=self.backend,
            enable_vision_dino=self.enable_vision_dino,
            enable_vision_siglip=self.enable_vision_siglip,
            enable_llm=self.enable_llm,
            enable_projector=self.enable_projector,
            vision_in_pct_pipeline=self.vision_in_pct_pipeline,
            symmetric_acts=self.signed_activations,
            symmetric_weights=True,
            config_name=f"quant_calibrate::{self.quant_bits}::{self.backend}",
        )

        valid_bits = (2, 4, 8, 16)

        if self.act_bits is None:
            self.act_bits = self.quant_cfg.act_bits
        else:
            if self.act_bits not in valid_bits:
                overwatch.warning(
                    "[QuantCalibrate] act_bits override is invalid; "
                    "using quant_bits as source of truth.",
                    extra={
                        "stage": "config",
                        "cli_act_bits": self.act_bits,
                        "quant_bits": self.quant_bits,
                        "resolved_act_bits": self.quant_cfg.act_bits,
                        "valid_bits": valid_bits,
                    },
                )
                self.act_bits = self.quant_cfg.act_bits
            elif self.act_bits != self.quant_cfg.act_bits:
                overwatch.warning(
                    "[QuantCalibrate] act_bits override disagrees with quant_bits; "
                    "using quant_bits as source of truth.",
                    extra={
                        "stage": "config",
                        "cli_act_bits": self.act_bits,
                        "quant_bits": self.quant_bits,
                        "resolved_act_bits": self.quant_cfg.act_bits,
                    },
                )
                self.act_bits = self.quant_cfg.act_bits

        valid_bits = (2, 4, 8, 16)

        # Explicitly reject 1-bit to avoid implying binary support in this PTQ stack.
        if self.act_bits == 1:
            raise ValueError(
                f"1-bit activation quantization is not supported in QuantCalibrateConfig "
                f"(got act_bits={self.act_bits}). Use one of {valid_bits}."
            )

        if self.act_bits not in valid_bits:
            raise ValueError(
                f"act_bits must be one of {valid_bits}, got {self.act_bits}"
            )

        # ------------------------------------------------------------------
        # 3) Normalize device
        # ------------------------------------------------------------------
        if self.device == "cuda" and not torch.cuda.is_available():
            overwatch.warning(
                "[QuantCalibrate] CUDA not available; falling back to CPU",
                extra={
                    "stage": "device",
                    "requested_device": "cuda",
                    "fallback_device": "cpu",
                },
            )
            self.device = "cpu"

        # ------------------------------------------------------------------
        # 4) Ensure output directories exist
        # ------------------------------------------------------------------
        for path in (
            self.pct_stats_out,
            self.pct_hi_lo_out,
            self.pct_summary_out,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Target → module mapping helpers
# =====================================================================
def _move_to_device(obj: Any, device: torch.device) -> Any:
    """
    Recursively move tensors in nested structures (dict/list/tuple) onto `device`.

    This is required because cobra collators may produce nested payloads such as:
        batch["pixel_values"] = {"dino": Tensor, "siglip": Tensor, ...}

    Moving only top-level tensors leaves nested `pixel_values` tensors on CPU, while
    `multimodal_indices` is moved to CUDA. That later crashes in Cobra forward when
    indexing CPU tensors with CUDA indices.
    """
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)

    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}

    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]

    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)

    return obj


def _cast_pixel_values_to_dtype(pixel_values, dtype: torch.dtype):
    if isinstance(pixel_values, dict):
        out = {}
        for k, v in pixel_values.items():
            if torch.is_tensor(v) and torch.is_floating_point(v):
                out[k] = v.to(dtype=dtype)
            else:
                out[k] = v
        return out
    if torch.is_tensor(pixel_values) and torch.is_floating_point(pixel_values):
        return pixel_values.to(dtype=dtype)
    return pixel_values


def _build_target_module_map_from_wrap_registry(
    *,
    model: nn.Module,
    registry,
    cfg: QuantCalibrateConfig,
) -> Dict[str, List[str]]:
    """
    Build target -> wrapped module paths for activation collection.

    Phase 3 + activation-policy aware behavior:
      - target taxonomy remains centralized/canonical
      - only canonical targets are emitted
      - LLM collection can be filtered by COBRA_LLM_ACT_MODE

    Robustness rule for llm_act_mode='mamba_sensitive':
      - Respect user-facing env gates
      - But intersect them with the stable hook-visible suffix subset for this
        Cobra snapshot before building the requested path set
      - This keeps strict completeness gating meaningful and prevents impossible
        requests such as in_proj / dt_proj from being treated as required
        path-aware module-hook targets
    """
    del model  # registry already provides the module paths we need

    from cobra.quantize.runtime.act_policy import (
        resolve_effective_mamba_sensitive_suffixes,
    )

    target_to_modules: Dict[str, List[str]] = {t: [] for t in CANONICAL_TARGETS}
    enabled_targets = set(cfg.quant_cfg.use_pct_for)

    llm_act_only = os.environ.get("COBRA_LLM_ACT_ONLY", "").strip().lower()
    llm_act_mode = normalize_llm_act_mode(
        os.environ.get("COBRA_LLM_ACT_MODE", ""),
        fallback_llm_act_only=llm_act_only,
    )

    mamba_sensitive_requested_suffixes: Tuple[str, ...] = ()
    mamba_sensitive_effective_suffixes: Tuple[str, ...] = ()
    mamba_sensitive_ignored_suffixes: Tuple[str, ...] = ()

    if llm_act_mode == LLM_ACT_MODE_MAMBA_SENSITIVE:
        gates = resolve_mamba_sensitive_projection_gates()
        mamba_sensitive_requested_suffixes = tuple(gates.enabled_suffixes)
        mamba_sensitive_effective_suffixes = resolve_effective_mamba_sensitive_suffixes(
            requested_suffixes=mamba_sensitive_requested_suffixes,
            hook_visible_only=True,
        )
        effective_set = set(mamba_sensitive_effective_suffixes)
        mamba_sensitive_ignored_suffixes = tuple(
            s for s in mamba_sensitive_requested_suffixes if s not in effective_set
        )

    raw_llm_paths: List[str] = []

    for entry in registry.entries:
        target = entry.target
        if target not in enabled_targets:
            continue

        module_path = entry.module_path
        if not module_path:
            continue

        target_to_modules[target].append(module_path)
        if target == "llm":
            raw_llm_paths.append(module_path)

    for target in list(target_to_modules.keys()):
        mods = sorted(set(target_to_modules[target]))
        if mods:
            target_to_modules[target] = mods
        else:
            del target_to_modules[target]

    pre_filter_counts = {k: len(v) for k, v in target_to_modules.items()}
    raw_llm_summary = summarize_llm_module_paths(sorted(set(raw_llm_paths)))

    target_to_modules = filter_target_module_map_for_llm_mode(
        target_to_modules,
        mode=llm_act_mode,
        mamba_sensitive_suffixes=(
            mamba_sensitive_effective_suffixes
            if llm_act_mode == LLM_ACT_MODE_MAMBA_SENSITIVE
            else None
        ),
    )

    for target in list(target_to_modules.keys()):
        mods = sorted(set(target_to_modules[target]))
        if mods:
            target_to_modules[target] = mods
        else:
            del target_to_modules[target]

    post_filter_counts = {k: len(v) for k, v in target_to_modules.items()}
    filtered_llm_summary = summarize_llm_module_paths(target_to_modules.get("llm", []))

    overwatch.info(
        "[QuantCalibrate] Built target module map.",
        extra={
            "enabled_targets": sorted(enabled_targets),
            "llm_act_only": llm_act_only if llm_act_only else None,
            "llm_act_mode": llm_act_mode,
            "target_to_count_before_llm_filter": pre_filter_counts,
            "target_to_count_after_llm_filter": post_filter_counts,
            "llm_paths_before_filter": raw_llm_summary,
            "llm_paths_after_filter": filtered_llm_summary,
            "mamba_sensitive_requested_suffixes": list(mamba_sensitive_requested_suffixes),
            "mamba_sensitive_effective_suffixes": list(mamba_sensitive_effective_suffixes),
            "mamba_sensitive_ignored_suffixes": list(mamba_sensitive_ignored_suffixes),
        },
    )
    return target_to_modules


def _summarize_hi_lo_map(
    hi_lo_map: Mapping[str, Mapping[str, float]],
) -> Dict[str, Dict[str, float]]:
    """
    Build a compact, JSON-serializable summary from hi_lo_map.

    hi_lo_map is expected to be:
        {
          "<hook_name>": {
              "target": "vision.dino" | "vision.siglip" | "llm" | "projector",
              "percent": 99.9,
              "hi": <float>,
              "lo": <float>,
              ...
          },
          ...
        }
    """
    summary: Dict[str, Dict[str, float]] = {}

    for hook, record in hi_lo_map.items():
        entry: Dict[str, float] = {}

        tgt = record.get("target")
        pct = record.get("percent")
        hi = record.get("hi")
        lo = record.get("lo")

        if tgt is not None:
            entry["target"] = str(tgt)
        if pct is not None:
            entry["percentile"] = float(pct)
        if hi is not None:
            entry["hi"] = float(hi)
        if lo is not None:
            entry["lo"] = float(lo)

        summary[hook] = entry

    return summary


# =====================================================================
# Main Calibration Routine
# =====================================================================
@draccus.wrap()
def quant_calibrate(cfg: QuantCalibrateConfig) -> None:
    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    set_global_seed(cfg.seed)

    device = torch.device(cfg.device)
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    if isinstance(cfg.hf_token, Path):
        hf_token = cfg.hf_token.read_text().strip()
    else:
        hf_token = str(cfg.hf_token).strip()

    llm_act_only = os.environ.get("COBRA_LLM_ACT_ONLY", "").strip().lower()
    llm_act_mode = normalize_llm_act_mode(
        os.environ.get("COBRA_LLM_ACT_MODE", ""),
        fallback_llm_act_only=llm_act_only,
    )
    mamba_sensitive_gates = resolve_mamba_sensitive_projection_gates()

    overwatch.info(
        "[QuantCalibrate] QuantRuntimeConfig resolved",
        extra={
            "quant_bits": cfg.quant_bits,
            "backend": cfg.backend,
            "mode": cfg.quant_cfg.mode.value,
            "weight_bits": cfg.quant_cfg.weight_bits,
            "act_bits": cfg.quant_cfg.act_bits,
            "use_pct_for": sorted(cfg.quant_cfg.use_pct_for),
            "llm_act_only": llm_act_only if llm_act_only else None,
            "llm_act_mode": llm_act_mode,
            "mamba_sensitive_projection_gates": mamba_sensitive_gates.as_dict(),
        },
    )

    # ------------------------------------------------------------------
    # Instantiate VLM (Vision + LLM Backbones)
    #
    # For path-aware LLM calibration modes, force Mamba slow path during model
    # construction so the internal nn.Linear modules are actually invoked.
    # This is required for input-side forward_pre_hook collectors to see
    # in_proj / x_proj / dt_proj / out_proj activations.
    # ------------------------------------------------------------------
    model_id = cfg.model.model_id

    force_mamba_slow_path = bool(
        "llm" in cfg.quant_cfg.use_pct_for
        and llm_act_mode in (LLM_ACT_MODE_OUT_PROJ_ONLY, LLM_ACT_MODE_MAMBA_SENSITIVE)
    )

    prev_disable_fast = os.environ.get(ENV_DISABLE_MAMBA_FAST_PATH)
    if force_mamba_slow_path:
        os.environ[ENV_DISABLE_MAMBA_FAST_PATH] = "1"
        overwatch.info(
            "[QuantCalibrate] Forcing Mamba slow path for path-aware LLM activation collection.",
            extra={
                "env_key": ENV_DISABLE_MAMBA_FAST_PATH,
                "env_value": "1",
                "llm_act_mode": llm_act_mode,
            },
        )

    try:
        overwatch.info(
            f"[QuantCalibrate] Loading Vision Backbone `{cfg.model.vision_backbone_id}` via TIMM"
        )
        vision_backbone, image_transform = get_vision_backbone_and_transform(
            cfg.model.vision_backbone_id,
            image_resize_strategy=cfg.model.image_resize_strategy,
        )

        overwatch.info(
            f"[QuantCalibrate] Loading LLM Backbone `{cfg.model.llm_backbone_id}` via HF Transformers"
        )
        llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
            cfg.model.llm_backbone_id,
            llm_max_length=cfg.model.llm_max_length,
            hf_token=hf_token,
            inference_mode=True,
        )

        overwatch.info(
            f"[QuantCalibrate] Instantiating CobraVLM `{model_id}` for Stage = `{cfg.stage}`"
        )
        vlm = get_vlm(
            model_id,
            cfg.model.arch_specifier,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        )
    finally:
        if force_mamba_slow_path:
            if prev_disable_fast is None:
                os.environ.pop(ENV_DISABLE_MAMBA_FAST_PATH, None)
            else:
                os.environ[ENV_DISABLE_MAMBA_FAST_PATH] = prev_disable_fast

    # For calibration, we treat everything as frozen; load from checkpoint if provided
    vlm.freeze_backbones(cfg.stage)

    if cfg.pretrained_checkpoint_root is not None:
        run_dir = cfg.pretrained_checkpoint_root
    else:
        run_dir = Path("runs") / model_id

    overwatch.info(
        f"[QuantCalibrate] Loading checkpoint for `{model_id}` from run_dir = `{run_dir}` "
        f"(stage = `{cfg.stage}`)"
    )
    vlm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=None)

    vlm.to(device=device, dtype=dtype)
    vlm.eval()

    # ------------------------------------------------------------------
    # Build wrap registry for coverage analysis AND activation collection
    # ------------------------------------------------------------------
    wrap_policy_cfg = WrapPolicyConfig(
        enable_vision_dino=cfg.enable_vision_dino,
        enable_vision_siglip=cfg.enable_vision_siglip,
        enable_llm=cfg.enable_llm,
        enable_projector=cfg.enable_projector,
        include_linear=True,
        include_conv=True,
    )
    wrap_registry = build_wrap_registry(
        vlm,
        policy_cfg=wrap_policy_cfg,
        manifest=None,
        prefix="",
    )

    # ------------------------------------------------------------------
    # Dataset + DataLoader
    # ------------------------------------------------------------------
    overwatch.info(
        f"[QuantCalibrate] Creating Dataset `{cfg.dataset.dataset_id}` for Stage = `{cfg.stage}` "
        f"at root_dir = `{cfg.dataset.dataset_root_dir}`"
    )

    train_dataset, collator = get_dataset_and_collator(
        stage=cfg.stage,
        dataset_cfg=cfg.dataset,
        image_transform=image_transform,
        tokenizer=tokenizer,
        prompt_builder_fn=llm_backbone.prompt_builder_fn,
        default_image_resolution=vision_backbone.default_image_resolution,
        padding_side=tokenizer.padding_side,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.per_device_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collator,
    )

    overwatch.info(
        f"[QuantCalibrate] Initialized DataLoader: "
        f"num_samples={len(train_dataset)}, "
        f"batch_size={cfg.per_device_batch_size}, "
        f"num_workers={cfg.num_workers}"
    )

    # ------------------------------------------------------------------
    # Register Activation Collectors
    # ------------------------------------------------------------------
    target_to_module_names = _build_target_module_map_from_wrap_registry(
        model=vlm,
        cfg=cfg,
        registry=wrap_registry,
    )

    llm_target_paths = target_to_module_names.get("llm", [])
    llm_path_summary = summarize_llm_module_paths(llm_target_paths)

    use_llm_input_pre_hooks = bool(
        llm_target_paths
        and llm_act_mode in (LLM_ACT_MODE_OUT_PROJ_ONLY, LLM_ACT_MODE_MAMBA_SENSITIVE)
    )
    llm_collection_mode = (
        "module_input_pre_hook_slow_path"
        if use_llm_input_pre_hooks
        else "coarse_llm_tap"
    )

    overwatch.info(
        "[QuantCalibrate] LLM collector selection summary",
        extra={
            "llm_act_mode": llm_act_mode,
            "llm_collection_mode": llm_collection_mode,
            "llm_target_path_count": len(llm_target_paths),
            "llm_target_path_summary": llm_path_summary,
            "mamba_sensitive_projection_gates": mamba_sensitive_gates.as_dict(),
            "force_mamba_slow_path": bool(force_mamba_slow_path),
        },
    )

    module_name_to_hook_kind: Dict[str, str] = {}
    if use_llm_input_pre_hooks:
        for module_path in llm_target_paths:
            module_name_to_hook_kind[module_path] = "pre"

    collectors = register_activation_collectors(
        model=vlm,
        target_to_module_names=target_to_module_names,
        max_samples_per_module=cfg.max_samples_per_module,
        device=torch.device("cpu"),  # store activation buffers on CPU by default
        allow_missing=True,
        module_name_to_hook_kind=module_name_to_hook_kind,
    )

    num_pre_hooks = sum(1 for c in collectors.values() if c.hook_kind == "pre")
    num_forward_hooks = sum(1 for c in collectors.values() if c.hook_kind == "forward")

    overwatch.info(
        f"[QuantCalibrate] Registered activation collectors for "
        f"{len(collectors)} module(s) across {len(target_to_module_names)} target(s)",
        extra={
            "llm_act_mode": llm_act_mode,
            "llm_collection_mode": llm_collection_mode,
            "llm_target_path_summary": llm_path_summary,
            "num_pre_hooks": num_pre_hooks,
            "num_forward_hooks": num_forward_hooks,
        },
    )

    # ------------------------------------------------------------------
    # Optional coarse LLM tap context (default mode only)
    #
    # Policy:
    #   - default         : coarse tap allowed
    #   - out_proj_only   : coarse tap disabled
    #   - mamba_sensitive : coarse tap disabled
    #
    # Non-default modes use path-aware module-input pre-hooks above.
    # ------------------------------------------------------------------
    enable_llm_tap = (
        "llm" in cfg.quant_cfg.use_pct_for and llm_act_mode == LLM_ACT_MODE_DEFAULT
    )
    llm_tap_ctx: Optional[LLMActivationTapContext] = None

    if enable_llm_tap:
        llm_tap_ctx = LLMActivationTapContext(
            enabled=True,
            max_samples_per_module=cfg.max_samples_per_module,
            device=torch.device("cpu"),
        )
        set_global_llm_tap_context(llm_tap_ctx)
        overwatch.info(
            "[QuantCalibrate] Enabled coarse LLM tap context.",
            extra={
                "max_samples_per_module": cfg.max_samples_per_module,
                "llm_act_only": llm_act_only if llm_act_only else None,
                "llm_act_mode": llm_act_mode,
                "coarse_llm_tap_enabled": True,
            },
        )
    else:
        set_global_llm_tap_context(None)
        if "llm" in cfg.quant_cfg.use_pct_for:
            overwatch.info(
                "[QuantCalibrate] Disabled coarse LLM tap context; using collector-based LLM activation collection.",
                extra={
                    "llm_act_only": llm_act_only if llm_act_only else None,
                    "llm_act_mode": llm_act_mode,
                    "coarse_llm_tap_enabled": False,
                    "llm_collection_mode": llm_collection_mode,
                },
            )

    # ------------------------------------------------------------------
    # Calibration Loop: Run Batches through the Model
    # ------------------------------------------------------------------
    num_batches_processed = 0
    overwatch.info("[QuantCalibrate] Starting calibration loop...")

    with torch.inference_mode():
        for batch in train_dataloader:
            if cfg.max_calib_batches > 0 and num_batches_processed >= cfg.max_calib_batches:
                break

            overwatch.info(
                f"[QuantCalibrate] >>> Forward batch {num_batches_processed+1}"
            )

            batch_on_device = _move_to_device(batch, device)
            if "pixel_values" in batch_on_device:
                batch_on_device["pixel_values"] = _cast_pixel_values_to_dtype(
                    batch_on_device["pixel_values"],
                    dtype=dtype,
                )

            _ = vlm(
                input_ids=batch_on_device["input_ids"],
                attention_mask=batch_on_device["attention_mask"],
                pixel_values=batch_on_device["pixel_values"],
                labels=batch_on_device.get("labels"),
                multimodal_indices=batch_on_device.get("multimodal_indices"),
            )

            num_batches_processed += 1

    # ------------------------------------------------------------------
    # Build percentile stats
    # ------------------------------------------------------------------
    try:
        stats = build_activation_stats(collectors, mode="activation")

        if llm_tap_ctx is not None and llm_tap_ctx.enabled:
            llm_tap_buffer_count = len(llm_tap_ctx.buffers_by_key)
        else:
            llm_tap_buffer_count = 0

        llm_stat_paths = sorted(
            str(record.get("module"))
            for record in stats.values()
            if str(record.get("target")) == "llm" and record.get("module")
        )
        llm_stat_path_summary = summarize_llm_module_paths(llm_stat_paths)

        # Strict completeness gate for path-aware LLM modes.
        # If requested LLM paths are not all represented in the resulting stats,
        # do not emit a partial artifact.
        if use_llm_input_pre_hooks and llm_target_paths:
            requested_llm = set(llm_target_paths)
            observed_llm = set(llm_stat_paths)
            missing_llm = sorted(requested_llm - observed_llm)
            unexpected_llm = sorted(observed_llm - requested_llm)

            if missing_llm:
                raise RuntimeError(
                    "[QuantCalibrate] Path-aware LLM activation collection is incomplete: "
                    f"llm_act_mode={llm_act_mode!r} requested={len(requested_llm)} "
                    f"observed={len(observed_llm)} missing={len(missing_llm)} "
                    f"sample_missing={missing_llm[:8]!r} "
                    f"unexpected={len(unexpected_llm)} "
                    f"sample_unexpected={unexpected_llm[:8]!r}"
                )

        torch.save(stats, cfg.pct_stats_out)
        overwatch.info(
            f"[QuantCalibrate] Saved percentile stats to `{cfg.pct_stats_out}`",
            extra={
                "num_records": len(stats),
                "num_batches_processed": num_batches_processed,
                "llm_act_mode": llm_act_mode,
                "llm_collection_mode": llm_collection_mode,
                "coarse_llm_tap_enabled": enable_llm_tap,
                "llm_tap_buffer_count": llm_tap_buffer_count,
                "llm_stat_path_summary": llm_stat_path_summary,
            },
        )

        # --------------------------------------------------------------
        # Convenience path: directly build hi/lo map + summary
        # --------------------------------------------------------------
        hi_lo_map = build_hi_lo_map(
            stats=stats,
            symmetric=cfg.symmetric_clipping,
            tau_growth=cfg.tau_growth,
            include_targets=cfg.quant_cfg.use_pct_for,
        )

        torch.save(hi_lo_map, cfg.pct_hi_lo_out)
        overwatch.info(
            f"[QuantCalibrate] Saved hi/lo map to `{cfg.pct_hi_lo_out}`",
            extra={
                "num_records": len(hi_lo_map),
                "llm_act_mode": llm_act_mode,
                "llm_collection_mode": llm_collection_mode,
            },
        )

        hi_lo_summary = _summarize_hi_lo_map(hi_lo_map)

        summary_payload: Dict[str, Any] = {
            "quant_bits": cfg.quant_bits,
            "backend": cfg.backend,
            "weight_bits": cfg.quant_cfg.weight_bits,
            "act_bits": cfg.act_bits,
            "symmetric_clipping": bool(cfg.symmetric_clipping),
            "tau_growth": float(cfg.tau_growth),
            "signed_activations": bool(cfg.signed_activations),
            "use_pct_for": sorted(cfg.quant_cfg.use_pct_for),
            "llm_act_only": llm_act_only if llm_act_only else None,
            "llm_act_mode": llm_act_mode,
            "llm_collection_mode": llm_collection_mode,
            "coarse_llm_tap_enabled": bool(enable_llm_tap),
            "force_mamba_slow_path": bool(force_mamba_slow_path),
            "mamba_sensitive_projection_gates": mamba_sensitive_gates.as_dict(),
            "num_batches_processed": int(num_batches_processed),
            "num_stat_records": int(len(stats)),
            "num_hi_lo_records": int(len(hi_lo_map)),
            "target_to_module_names": {
                k: list(v) for k, v in target_to_module_names.items()
            },
            "llm_target_path_summary": llm_path_summary,
            "llm_stat_path_summary": llm_stat_path_summary,
            "num_pre_hooks": int(num_pre_hooks),
            "num_forward_hooks": int(num_forward_hooks),
            "hi_lo_summary": hi_lo_summary,
        }

        with cfg.pct_summary_out.open("w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)

        overwatch.info(
            f"[QuantCalibrate] Saved summary JSON to `{cfg.pct_summary_out}`",
            extra={
                "llm_act_mode": llm_act_mode,
                "llm_collection_mode": llm_collection_mode,
                "coarse_llm_tap_enabled": enable_llm_tap,
                "llm_target_path_summary": llm_path_summary,
                "llm_stat_path_summary": llm_stat_path_summary,
                "mamba_sensitive_projection_gates": mamba_sensitive_gates.as_dict(),
            },
        )

    finally:
        remove_activation_collectors(collectors)
        set_global_llm_tap_context(None)

     
if __name__ == "__main__":
    quant_calibrate()



