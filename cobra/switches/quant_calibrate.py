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
from cobra.models.vlms.fusion import FusionStage
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
from cobra.quantize.runtime.config import QuantRuntimeConfig
from cobra.quantize.runtime.pipeline_spec import CANONICAL_TARGETS, add_fusion_stage_to_target_map
from cobra.quantize.wrap.policy import WrapPolicyConfig
from cobra.quantize.wrap.registry import build_wrap_registry
from cobra.quantize.rotate.projector import SHARED_KLT_PATH, load_klt_matrix
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
    # === Model / Dataset Selection ===

    # ModelConfig (`cobra/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.COBRA_3B.model_id)
    )

    # DatasetConfig (`cobra/conf/datasets.py`); override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(
            DatasetRegistry.TEXTVQA_100_CALIB.dataset_id
        )
    )

    # Pretraining Stage in < align (projector-only) | finetune (projector + LLM) | full-finetune (all) >
    stage: str = "align"

    # Optional pretrained checkpoint root (for multi-run setups); otherwise derived from model_id
    pretrained_checkpoint_root: Optional[Path] = None

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")  # Env var name or path to token file

    # === Dataloader / Calibration Budget ===

    per_device_batch_size: int = 8
    num_workers: int = 4

    # Upper bound on how many batches to pass through for calibration.
    #   - If <= 0, iterate over the entire dataset once.
    max_calib_batches: int = 0

    # === Quantization Settings (centralized via QuantRuntimeConfig) ===
    # act_bits 保留給 CLI 相容（legacy override）。
    # 根因修正：預設為 None，表示「不指定」，完全由 quant_bits 推導得到。
    # 只有在使用者顯式指定 act_bits 且與 quant_bits 推導不一致時才發警告。
    act_bits: Optional[int] = None
    signed_activations: bool = True

    # 統一的 bits/backend 入口（交由 QuantRuntimeConfig 解析）
    quant_bits: str = "W8A8"
    backend: str = "fake"

    # 是否允許進 percentile pipeline：會傳入 QuantRuntimeConfig
    enable_vision_dino: bool = True
    enable_vision_siglip: bool = True
    enable_llm: bool = True
    enable_projector: bool = True
    enable_fusion: bool = True
    vision_in_pct_pipeline: bool = True

    # Best-percentile selection hyperparameter (see `best_percentile.py`)
    tau_growth: float = 5.0
    symmetric_clipping: bool = True

    # How many activation samples to store per module before subsampling (see `collect.py`)
    max_samples_per_module: int = 5_000_000

    # === Outputs ===
    #   - quant_calibrate: producer of pct_stats_out + pct_hi_lo_out / pct_summary_out
    pct_stats_out: Path = Path("outputs/quantize/pct_stats.pt")
    pct_hi_lo_out: Path = Path("outputs/quantize/pct_hi_lo.pt")
    pct_summary_out: Path = Path("outputs/quantize/pct_calibrate_summary.json")

    # === Misc ===

    seed: int = 7
    device: str = "cuda"  # "cuda" or "cpu"

    # QuantRuntimeConfig（由 __post_init__ 建立；不透過 CLI 直接設）
    quant_cfg: QuantRuntimeConfig = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # ------------------------------------------------------------------
        # 1) 建立 QuantRuntimeConfig（集中解析 bits/backend/enable_*）
        # ------------------------------------------------------------------
        self.quant_cfg = QuantRuntimeConfig.from_bits_backend(
            bits=self.quant_bits,
            backend=self.backend,
            enable_vision_dino=self.enable_vision_dino,
            enable_vision_siglip=self.enable_vision_siglip,
            enable_llm=self.enable_llm,
            enable_projector=self.enable_projector,
            vision_in_pct_pipeline=self.vision_in_pct_pipeline,
            symmetric_acts=self.signed_activations,
            symmetric_weights=True,  # 校正只看 activation，weights 無實際影響
            config_name=f"quant_calibrate::{self.quant_bits}::{self.backend}",
        )

        # ------------------------------------------------------------------
        # act_bits root-cause fix:
        # - act_bits defaults to None (meaning: "not specified").
        # - quant_bits is the single source of truth for resolved act_bits.
        # - Only warn when the user explicitly provides act_bits and it disagrees.
        # ------------------------------------------------------------------
        valid_bits = (2, 4, 8, 16)

        if self.act_bits is None:
            # Default: no user override; derive from quant_bits.
            self.act_bits = self.quant_cfg.act_bits
        else:
            # User explicitly supplied act_bits: validate + reconcile with quant_bits.
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
                    "[QuantCalibrate] act_bits override differs from quant_bits-derived act_bits; "
                    "using quant_bits as source of truth.",
                    extra={
                        "stage": "config",
                        "cli_act_bits": self.act_bits,
                        "quant_bits": self.quant_bits,
                        "resolved_act_bits": self.quant_cfg.act_bits,
                    },
                )
                self.act_bits = self.quant_cfg.act_bits
            # else: user override matches derived; keep it (already consistent).


        # ------------------------------------------------------------------
        # 2) Sanity checks for act_bits（與 QuantRuntimeConfig 對齊後）
        # ------------------------------------------------------------------
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

def _build_target_module_map_from_wrap_registry(
    model: nn.Module,
    cfg: QuantCalibrateConfig,
    wrap_registry,
) -> Dict[str, List[str]]:
    # Modules considered "wrapped" (i.e., eligible for quantization) under the current policy.
    modules_by_target: Mapping[str, Sequence[str]] = wrap_registry.module_paths_by_target(
        include_targets = CANONICAL_TARGETS
    )

    enabled_targets = cfg.quant_cfg.use_pct_for

    target_to_modules: Dict[str, List[str]] = {}

    for target in CANONICAL_TARGETS:
        if target not in enabled_targets:
            continue

        mod_names = list(modules_by_target.get(target, []))
        if not mod_names:
            continue

        mod_names = sorted(mod_names)
        target_to_modules[target] = mod_names

    if not target_to_modules:
        overwatch.warning(
            "[QuantCalibrate] No target modules found for activation collection; "
            "please check quant_bits/backend, enable_* flags, wrap policy, and model architecture.",
            extra={
                "stage": "collect",
                "quant_bits": cfg.quant_bits,
                "backend": cfg.backend,
                "use_pct_for": sorted(cfg.quant_cfg.use_pct_for),
            },
        )

    # Logging summary
    for target in CANONICAL_TARGETS:
        mods = target_to_modules.get(target, [])
        if mods:
            overwatch.info(
                f"[QuantCalibrate] Target={target!r} (from WrapRegistry) → "
                f"{len(mods)} module(s); example: {mods[0]!r}"
            )
        else:
            overwatch.info(
                f"[QuantCalibrate] Target={target!r} (from WrapRegistry) → 0 modules"
            )

    add_fusion_stage_to_target_map(
        model=model,
        enabled_targets=enabled_targets,
        target_to_modules=target_to_modules,
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

def _configure_fusion_rotation_for_calibration(
    vlm: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """
    Configure FusionStage rotation mode BEFORE activation collection.

    This aligns calibration stats with the Paper-main Point-B behavior:
        fused_embeddings := fused_embeddings · R
        fused_embeddings := clip_best(fused_embeddings)
        fused_embeddings := quant_dequant_A(fused_embeddings)

    Env keys:
        - COBRA_FUSION_ROTATION_MODE: none|hadamard|hk
        - KLT_OUT (optional): preferred KLT path produced by cobra_1115_ptq.sh
          fallback to SHARED_KLT_PATH for backward compatibility.
    """
    # Find fusion_stage
    fusion_stage = getattr(vlm, "fusion_stage", None)
    if fusion_stage is None:
        return
    if not isinstance(fusion_stage, FusionStage):
        return

    mode = os.environ.get("COBRA_FUSION_ROTATION_MODE", "none").strip().lower()
    if mode not in ("none", "hadamard", "hk"):
        overwatch.warning(
            "[QuantCalibrate] Unknown COBRA_FUSION_ROTATION_MODE; falling back to 'none'.",
            extra={"COBRA_FUSION_ROTATION_MODE": mode},
        )
        mode = "none"

    K = None
    klt_loaded = False

    if mode == "hk":
        # Prefer KLT_OUT (script export), then fallback to SHARED_KLT_PATH.
        klt_out_env = os.environ.get("KLT_OUT", "").strip()
        klt_path = Path(klt_out_env) if klt_out_env else SHARED_KLT_PATH

        if not klt_path.is_file():
            raise FileNotFoundError(
                f"[QuantCalibrate] FusionStage mode='hk' requires KLT file, but not found: {klt_path}"
            )

        K = load_klt_matrix(klt_path)
        if K is None:
            raise RuntimeError(
                f"[QuantCalibrate] Failed to load KLT matrix from: {klt_path}"
            )

        # Move to calibration device/dtype
        K = K.to(device=device, dtype=dtype)
        klt_loaded = True

    fusion_stage.configure_rotation(mode=mode, klt_matrix=K)

    overwatch.info(
        "[QuantCalibrate] Configured fusion_stage rotation for calibration.",
        extra={
            "fusion_rotation_mode": mode,
            "klt_loaded": klt_loaded,
        },
    )

# =====================================================================
# Main Calibration Routine
# =====================================================================


@draccus.wrap()
def quant_calibrate(cfg: QuantCalibrateConfig) -> None:
    # ------------------------------------------------------------------
    # Basic Setup
    # ------------------------------------------------------------------
    set_global_seed(cfg.seed)

    device = torch.device(cfg.device)
    dtype = torch.float32

    # Resolve HF token
    if isinstance(cfg.hf_token, Path):
        hf_token = cfg.hf_token.read_text().strip()
    else:
        hf_token = os.environ[cfg.hf_token]

    overwatch.info(
        "[QuantCalibrate] Effective QuantRuntimeConfig",
        extra={
            "quant_bits": cfg.quant_bits,
            "backend": cfg.backend,
            "mode": cfg.quant_cfg.mode.value,
            "weight_bits": cfg.quant_cfg.weight_bits,
            "act_bits": cfg.quant_cfg.act_bits,
            "use_pct_for": sorted(cfg.quant_cfg.use_pct_for),
        },
    )

    # ------------------------------------------------------------------
    # Instantiate VLM (Vision + LLM Backbones)
    # ------------------------------------------------------------------
    model_id = cfg.model.model_id
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

    # For calibration, we treat everything as frozen; load from checkpoint if provided
    vlm.freeze_backbones(cfg.stage)

    run_dir: Optional[Path] = None
    if cfg.pretrained_checkpoint_root is not None:
        run_dir = cfg.pretrained_checkpoint_root
    else:
        # Default to runs/<model_id> (same convention as `scripts/pretrain.py`)
        run_dir = Path("runs") / model_id

    overwatch.info(
        f"[QuantCalibrate] Loading checkpoint for `{model_id}` from run_dir = `{run_dir}` "
        f"(stage = `{cfg.stage}`)"
    )
    vlm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=None)

    vlm.to(device=device, dtype=dtype)
    vlm.eval()
   
    # ------------------------------------------------------------------
    # Configure FusionStage rotation BEFORE activation collection.
    # This aligns calibration stats with Point-B global rotation/clipping/quant.
    # ------------------------------------------------------------------
    _configure_fusion_rotation_for_calibration(
        vlm,
        device=device,
        dtype=dtype,
    )

    # ------------------------------------------------------------------
    # Build wrap registry for coverage analysis AND activation collection
    # ------------------------------------------------------------------
    wrap_policy_cfg = WrapPolicyConfig(
        enable_vision_dino=cfg.enable_vision_dino,
        enable_vision_siglip=cfg.enable_vision_siglip,
        enable_llm=cfg.enable_llm,
        enable_projector=cfg.enable_projector,
        enable_fusion=cfg.enable_fusion,
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
        wrap_registry=wrap_registry,
    )

    collectors = register_activation_collectors(
        model=vlm,
        target_to_module_names=target_to_module_names,
        max_samples_per_module=cfg.max_samples_per_module,
        device=torch.device("cpu"),  # store activation buffers on CPU by default
    )

    overwatch.info(
        f"[QuantCalibrate] Registered activation collectors for "
        f"{len(collectors)} module(s) across {len(target_to_module_names)} target(s)"
    )

    # ------------------------------------------------------------------
    # Optional LLM tap context (for Mamba backbone activations)
    # ------------------------------------------------------------------
    llm_tap_ctx: Optional[LLMActivationTapContext] = None
    if "llm" in cfg.quant_cfg.use_pct_for:
        llm_tap_ctx = LLMActivationTapContext(
            enabled=True,
            max_samples_per_module=cfg.max_samples_per_module,
            device=torch.device("cpu"),
        )
        set_global_llm_tap_context(llm_tap_ctx)
        overwatch.info(
            "[QuantCalibrate] Enabled global LLM tap context for activation collection",
            extra={
                "stage": "collect",
                "max_samples_per_module": cfg.max_samples_per_module,
            },
        )
    else:
        # Ensure no stale context leaks from previous runs.
        set_global_llm_tap_context(None)

    # ------------------------------------------------------------------
    # Calibration Loop: Run Batches through the Model
    # ------------------------------------------------------------------
    num_batches_processed = 0
    overwatch.info("[QuantCalibrate] Starting calibration loop...")

    with torch.inference_mode():
        for batch in train_dataloader:
            # Respect calibration budget
            if cfg.max_calib_batches > 0 and num_batches_processed >= cfg.max_calib_batches:
                break

            overwatch.info(
                f"[QuantCalibrate] >>> Forward batch {num_batches_processed}"
            )

            # Move batch to device (support both tensor and nested dict(pixel_values))
            batch_on_device: Dict[str, Any] = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_on_device[k] = v.to(device)
                elif isinstance(v, dict):
                    batch_on_device[k] = {
                        kk: vv.to(device) if isinstance(vv, torch.Tensor) else vv
                        for kk, vv in v.items()
                    }
                else:
                    batch_on_device[k] = v

            _ = vlm(**batch_on_device)

            num_batches_processed += 1
            overwatch.info(
                f"[QuantCalibrate] <<< Done batch {num_batches_processed}"
            )

            if num_batches_processed % 10 == 0:
                overwatch.info(
                    f"[QuantCalibrate] Processed {num_batches_processed} batches for calibration"
                )

    overwatch.info(
        f"[QuantCalibrate] Finished calibration run: total_batches={num_batches_processed}"
    )

    # Simple diagnostic on LLM tap buffers (if enabled)
    ctx = get_global_llm_tap_context()
    if ctx is not None and ctx.enabled:
        overwatch.info(
            "[QuantCalibrate] LLM tap context summary",
            extra={
                "stage": "collect",
                "num_llm_buffers": len(ctx.buffers_by_key),
            },
        )

    # ------------------------------------------------------------------
    # Build Percentile Stats and Persist (Minimal pipeline 的必要輸出)
    # ------------------------------------------------------------------
    stats = build_activation_stats(
        collectors,
        mode="activation",
    )

    # Once stats have been built, clear the global tap context to avoid
    # leaking state into other scripts or subsequent runs.
    set_global_llm_tap_context(None)

    overwatch.info(
        f"[QuantCalibrate] Built percentile stats for {len(stats)} record(s); "
        f"saving to `{cfg.pct_stats_out}`"
    )
    torch.save(stats, cfg.pct_stats_out)

    # Clean up hooks
    remove_activation_collectors(collectors)

    # ------------------------------------------------------------------
    # Convenience mode: Run Best-Percentile → hi/lo（不動 model，本地只產生 hi/lo）
    # ------------------------------------------------------------------
    overwatch.info(
        "[QuantCalibrate] Running best-percentile selection + hi/lo mapping (no model calibration)",
        extra={
            "tau_growth": cfg.tau_growth,
            "symmetric_clipping": cfg.symmetric_clipping,
        },
    )

    enabled_targets: List[str] = sorted(cfg.quant_cfg.use_pct_for)

    overwatch.info(
        "[QuantCalibrate] Enabled targets for percentile pipeline: "
        + (", ".join(enabled_targets) if enabled_targets else "<none>"),
        extra={"use_pct_for": enabled_targets},
    )

    # Build hi/lo map directly from stats; let best_percentile.py choose per-target percentiles.
    hi_lo_map = build_hi_lo_map(
        stats=stats,
        best_percent_map=None,
        tau_growth=cfg.tau_growth,
        symmetric=cfg.symmetric_clipping,
        default_percentile=None,
        targets=enabled_targets,
    )

    # Save hi/lo map as a simple torch file
    cfg.pct_hi_lo_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(hi_lo_map, cfg.pct_hi_lo_out)

    # ------------------------------------------------------------------
    # Build JSON summary 
    # ------------------------------------------------------------------
    entries = _summarize_hi_lo_map(hi_lo_map)

    summary: Dict[str, Any] = {
        "config": {
            "pct_stats_out": str(cfg.pct_stats_out),
            "pct_hi_lo_out": str(cfg.pct_hi_lo_out),
            "pct_summary_out": str(cfg.pct_summary_out),
            "tau_growth": cfg.tau_growth,
            "symmetric_clipping": cfg.symmetric_clipping,
            "act_bits": cfg.act_bits,
            "signed_activations": cfg.signed_activations,
            "stage": cfg.stage,
            "dataset_id": cfg.dataset.dataset_id,
            "enabled_targets": enabled_targets,
            "quant_bits": cfg.quant_bits,
            "backend": cfg.backend,
            "quant_use_pct_for": sorted(cfg.quant_cfg.use_pct_for),
            "targets_effective": enabled_targets,
        },
        "num_entries": len(hi_lo_map),
        "by_target": {},
        "per_target_stats": {},
        "global_stats": {},
        "entries": entries,
    }

    # ------------------------------------------------------------------
    # Aggregate statistics (per-target + global)
    # ------------------------------------------------------------------
    by_target: Dict[str, int] = {}
    per_target_values: Dict[str, Dict[str, List[float]]] = {}

    for hook, entry in entries.items():
        tgt = entry.get("target", "<unknown>")
        by_target[tgt] = by_target.get(tgt, 0) + 1

        # Only aggregate stats for known targets
        if tgt == "<unknown>":
            continue

        bucket = per_target_values.setdefault(
            tgt,
            {"percentile": [], "hi": [], "lo": []},
        )

        pct = entry.get("percentile")
        hi = entry.get("hi")
        lo = entry.get("lo")

        if isinstance(pct, (int, float)):
            bucket["percentile"].append(float(pct))
        if isinstance(hi, (int, float)):
            bucket["hi"].append(float(hi))
        if isinstance(lo, (int, float)):
            bucket["lo"].append(float(lo))

    summary["by_target"] = by_target

    # Per-target stats: count + min/max/mean for percentile/hi/lo
    per_target_stats: Dict[str, Dict[str, float]] = {}
    for tgt, values in per_target_values.items():
        pct_vals = values["percentile"]
        hi_vals = values["hi"]
        lo_vals = values["lo"]

        if not pct_vals and not hi_vals and not lo_vals:
            continue

        count = max(len(pct_vals), len(hi_vals), len(lo_vals))

        per_target_stats[tgt] = {
            "count": float(count),
            "percentile_min": min(pct_vals) if pct_vals else None,
            "percentile_max": max(pct_vals) if pct_vals else None,
            "percentile_mean": statistics.mean(pct_vals) if pct_vals else None,
            "hi_min": min(hi_vals) if hi_vals else None,
            "hi_max": max(hi_vals) if hi_vals else None,
            "hi_mean": statistics.mean(hi_vals) if hi_vals else None,
            "lo_min": min(lo_vals) if lo_vals else None,
            "lo_max": max(lo_vals) if lo_vals else None,
            "lo_mean": statistics.mean(lo_vals) if lo_vals else None,
        }

    summary["per_target_stats"] = per_target_stats

    # Global stats across all targets
    all_pct: List[float] = []
    all_hi: List[float] = []
    all_lo: List[float] = []

    for values in per_target_values.values():
        all_pct.extend(values["percentile"])
        all_hi.extend(values["hi"])
        all_lo.extend(values["lo"])

    global_stats: Dict[str, Any] = {
        "count": float(len(entries)),
        "percentile_min": min(all_pct) if all_pct else None,
        "percentile_max": max(all_pct) if all_pct else None,
        "percentile_mean": statistics.mean(all_pct) if all_pct else None,
        "hi_min": min(all_hi) if all_hi else None,
        "hi_max": max(all_hi) if all_hi else None,
        "hi_mean": statistics.mean(all_hi) if all_hi else None,
        "lo_min": min(all_lo) if all_lo else None,
        "lo_max": max(all_lo) if all_lo else None,
        "lo_mean": statistics.mean(all_lo) if all_lo else None,
    }

    summary["global_stats"] = global_stats

    # ------------------------------------------------------------------
    # Coverage: wrapped vs observed vs calibrated
    # ------------------------------------------------------------------
    coverage: Dict[str, Any] = {}

    # Wrapped modules according to wrap policy/manifest
    wrapped_modules_by_target = wrap_registry.module_paths_by_target(
        include_targets = CANONICAL_TARGETS
    )

    # Activation coverage from stats (collectors)
    observed_by_target: Dict[str, set] = {t: set() for t in CANONICAL_TARGETS}
    for rec in stats.values():
        tgt = str(rec.get("target", ""))
        mod = str(rec.get("module", ""))
        if tgt in CANONICAL_TARGETS and mod:
            observed_by_target[tgt].add(mod)

    # Calibration coverage from hi/lo map
    calibrated_by_target: Dict[str, set] = {t: set() for t in CANONICAL_TARGETS}
    for _, rec in hi_lo_map.items():
        tgt = str(rec.get("target", ""))
        mod = str(rec.get("module", ""))
        if tgt in CANONICAL_TARGETS and mod:
            calibrated_by_target[tgt].add(mod)

    for tgt in CANONICAL_TARGETS:
        wrapped = set(wrapped_modules_by_target.get(tgt, []))
        num_wrapped = len(wrapped)

        observed = observed_by_target.get(tgt, set())
        calibrated = calibrated_by_target.get(tgt, set())

        num_observed = len(observed)
        num_calibrated = len(calibrated)

        activation_coverage_ratio = (
            float(num_observed) / float(num_wrapped) if num_wrapped > 0 else 0.0
        )
        calibration_coverage_ratio = (
            float(num_calibrated) / float(num_wrapped) if num_wrapped > 0 else 0.0
        )

        missing_modules = sorted(wrapped - observed)

        coverage[tgt] = {
            # Backwards-compatible field name; now interpreted as "wrapped modules under policy"
            "configured_modules": int(num_wrapped),
            "activation_observed_modules": int(num_observed),
            "activation_coverage_ratio": activation_coverage_ratio,
            "calibrated_modules": int(num_calibrated),
            "calibration_coverage_ratio": calibration_coverage_ratio,
            "missing_modules": missing_modules,
        }

    summary["coverage"] = coverage

    cfg.pct_summary_out.parent.mkdir(parents=True, exist_ok=True)
    with cfg.pct_summary_out.open("w") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    overwatch.info(
        "[QuantCalibrate] Finished collect-only percentile calibration",
        extra={
            "num_entries": summary["num_entries"],
            "by_target": summary["by_target"],
            "coverage": {
                tgt: f"{cov['calibrated_modules']}/{cov['configured_modules']}"
                for tgt, cov in summary["coverage"].items()
            },
        },
    )


if __name__ == "__main__":
    quant_calibrate()


