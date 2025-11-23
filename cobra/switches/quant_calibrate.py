# cobra/switches/quant_calibrate.py

"""
quant_calibrate.py

CLI entrypoint for percentile-based activation calibration.

Responsibilities:
    - Initialize a Cobra VLM and its pretraining dataset (align / finetune) using the existing
      ModelConfig / DatasetConfig infrastructure.
    - Run a (single-node) DataLoader to stream calibration batches through the model.
    - Use `cobra.quantize.pct.collect` to register activation collectors and build percentile stats.
    - Use `cobra.quantize.pct.apply` to:
        stats -> best-percentile map -> (hi, lo)
      and save the resulting hi/lo clipping map + JSON summary to disk.

This script does NOT:
    - Perform weight quantization or integer kernel export (handled by `quantize/finalize/int_export.py`).
    - Perform module wrapping (handled by `integration/wrap_replace.py`).
    - Handle distributed training (FSDP / DDP); it is intended for single-process calibration.
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
)
from cobra.quantize.pct.apply import build_hi_lo_map
from cobra.quantize.wrap.policy import WrapPolicyConfig
from cobra.quantize.wrap.registry import build_wrap_registry
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
    """
    Configuration for percentile-based activation calibration.

    Note:
    This CLI is now **collect-only**:
    it builds activation percentile stats and hi/lo clipping bounds,
    but does not modify the model in-place. Downstream scripts
    (`quant_pct_apply.py`, `quant_finalize.py`) are responsible for
    actually applying these parameters to wrapped modules.

    Key knobs (旗標):
        - model / dataset / stage: which pretrained Cobra VLM + dataset split to use.
        - act_bits: activation bitwidth (supported: 2 / 4 / 8 / 16; 1-bit is not supported).
        - tau_growth: growth-ratio threshold for best-percentile selection.
        - symmetric_clipping: whether to enforce symmetric hi/lo around 0.
        - enable_*: which of the four canonical targets to calibrate.
        - max_calib_steps / max_calib_batches: limit how much data to stream.
        - max_samples_per_module: cap how many activation samples each collector stores.
        - pct_*_out: output paths for stats / hi-lo map / calibration summary.
    """

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

    # === Percentile / Quantization Settings ===

    # Activation bitwidth (2 / 4 / 8 / 16); weight_bits are handled later by `finalize/int_export.py`
    act_bits: int = 8
    signed_activations: bool = True

    # Best-percentile selection hyperparameter (see `best_percentile.py`)
    tau_growth: float = 5.0
    symmetric_clipping: bool = True

    # How many activation samples to store per module before subsampling (see `collect.py`)
    max_samples_per_module: int = 5_000_000

    # Enable / disable percentile-based clipping per canonical target
    enable_vision_dino: bool = True
    enable_vision_siglip: bool = True
    enable_llm: bool = True
    enable_projector: bool = True

    # === Outputs ===

    pct_stats_out: Path = Path("outputs/quantize/pct_stats.pt")
    pct_hi_lo_out: Path = Path("outputs/quantize/pct_hi_lo.pt")
    pct_summary_out: Path = Path("outputs/quantize/pct_calibrate_summary.json")

    # === Misc ===

    seed: int = 7
    device: str = "cuda"  # "cuda" or "cpu"

    def __post_init__(self) -> None:
        # Sanity checks for bits
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

        # Normalize device
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

        # Ensure output directories exist
        for path in (
            self.pct_stats_out,
            self.pct_hi_lo_out,
            self.pct_summary_out,
        ):
            path.parent.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Target → module mapping helpers
# =====================================================================


_CANONICAL_TARGETS: Tuple[str, ...] = (
    "vision.dino",
    "vision.siglip",
    "llm",
    "projector",
)


def _build_target_module_map(
    model: nn.Module, cfg: QuantCalibrateConfig
) -> Dict[str, List[str]]:
    """
    Build a default mapping {canonical_target -> [module_qualified_name, ...]} from the VLM structure.

    We follow the canonical four-target vocabulary:
        - "vision.dino":   modules inside EITHER a DINOv2-only backbone (`vision_backbone.featurizer`) OR
                           the `vision_backbone.dino_featurizer` when using DinoSigLIP.
        - "vision.siglip": modules inside `vision_backbone.siglip_featurizer` (if present).
        - "llm":           modules inside `llm_backbone.llm`.
        - "projector":     modules inside `projector`.

    Notes:
        - If a backbone does not expose a given featurizer (e.g., no SigLIP), that target will simply be empty.
        - We only keep non-empty targets in the returned dict.
        - Per-target enable_* flags from cfg further filter which targets are active.
    """
    name_to_module: Dict[str, nn.Module] = dict(model.named_modules())

    target_to_modules: Dict[str, List[str]] = {
        "vision.dino": [],
        "vision.siglip": [],
        "llm": [],
        "projector": [],
    }

    for name in name_to_module.keys():
        # Vision (DINOv2 or DinoSigLIP)
        if name.startswith("vision_backbone.dino_featurizer") or name.startswith(
            "vision_backbone.featurizer"
        ):
            target_to_modules["vision.dino"].append(name)
        # Vision (SigLIP in DinoSigLIP backbone)
        elif name.startswith("vision_backbone.siglip_featurizer"):
            target_to_modules["vision.siglip"].append(name)
        # LLM
        elif name.startswith("llm_backbone.llm"):
            target_to_modules["llm"].append(name)
        # Projector
        elif name.startswith("projector"):
            target_to_modules["projector"].append(name)

    # Apply enable_* flags
    enabled_mask = {
        "vision.dino": cfg.enable_vision_dino,
        "vision.siglip": cfg.enable_vision_siglip,
        "llm": cfg.enable_llm,
        "projector": cfg.enable_projector,
    }

    target_to_modules = {
        target: sorted(mod_names)
        for target, mod_names in target_to_modules.items()
        if enabled_mask.get(target, True) and len(mod_names) > 0
    }

    if not target_to_modules:
        overwatch.warning(
            "[QuantCalibrate] No target modules found for activation collection; "
            "please check enable_* flags and model architecture.",
            extra={
                "stage": "collect",
                "enable_vision_dino": cfg.enable_vision_dino,
                "enable_vision_siglip": cfg.enable_vision_siglip,
                "enable_llm": cfg.enable_llm,
                "enable_projector": cfg.enable_projector,
            },
        )

    # Logging summary
    for target, mods in target_to_modules.items():
        overwatch.info(
            f"[QuantCalibrate] Target={target!r} → {len(mods)} module(s); "
            f"example: {mods[0]!r}"
            if mods
            else f"[QuantCalibrate] Target={target!r} → 0 modules"
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
    """
    End-to-end percentile-based activation calibration (collect-only):

        1. Initialize Cobra VLM + Dataset + DataLoader.
        2. Register activation collectors on selected module groups.
        3. Stream calibration batches through the model, filling activation buffers.
        4. Build percentile stats and persist them.
        5. Run best-percentile selection + hi/lo mapping from stats (no in-place model calibration).
        6. Persist hi/lo map and calibration summary for downstream steps (rotate / finalize).
    """

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
    # Build wrap registry for coverage analysis (no in-place wrapping here)
    # ------------------------------------------------------------------
    wrap_policy_cfg = WrapPolicyConfig(
        enable_vision_dino=cfg.enable_vision_dino,
        enable_vision_siglip=cfg.enable_vision_siglip,
        enable_llm=cfg.enable_llm,
        enable_projector=cfg.enable_projector,
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
    target_to_module_names = _build_target_module_map(vlm, cfg)

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
            batch_on_device = {}
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

    # ------------------------------------------------------------------
    # Build Percentile Stats and Persist
    # ------------------------------------------------------------------
    stats = build_activation_stats(
        collectors,
        mode="activation",
    )

    overwatch.info(
        f"[QuantCalibrate] Built percentile stats for {len(stats)} record(s); "
        f"saving to `{cfg.pct_stats_out}`"
    )
    torch.save(stats, cfg.pct_stats_out)

    # Clean up hooks
    remove_activation_collectors(collectors)

    # ------------------------------------------------------------------
    # Run Best-Percentile → hi/lo (collect-only; no model calibration)
    # ------------------------------------------------------------------
    overwatch.info(
        "[QuantCalibrate] Running best-percentile selection + hi/lo mapping (no model calibration)",
        extra={
            "tau_growth": cfg.tau_growth,
            "symmetric_clipping": cfg.symmetric_clipping,
        },
    )

    enabled_targets: List[str] = [
        t
        for t, enabled in (
            ("vision.dino", cfg.enable_vision_dino),
            ("vision.siglip", cfg.enable_vision_siglip),
            ("llm", cfg.enable_llm),
            ("projector", cfg.enable_projector),
        )
        if enabled
    ]

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
    # Build JSON summary (similar to quant_pct_apply)
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
        include_targets=_CANONICAL_TARGETS
    )

    # Activation coverage from stats (collectors)
    observed_by_target: Dict[str, set] = {t: set() for t in _CANONICAL_TARGETS}
    for rec in stats.values():
        tgt = str(rec.get("target", ""))
        mod = str(rec.get("module", ""))
        if tgt in _CANONICAL_TARGETS and mod:
            observed_by_target[tgt].add(mod)

    # Calibration coverage from hi/lo map
    calibrated_by_target: Dict[str, set] = {t: set() for t in _CANONICAL_TARGETS}
    for name, rec in hi_lo_map.items():
        tgt = str(rec.get("target", ""))
        mod = str(rec.get("module", ""))
        if tgt in _CANONICAL_TARGETS and mod:
            calibrated_by_target[tgt].add(mod)

    for tgt in _CANONICAL_TARGETS:
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


