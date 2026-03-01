"""Offline CF-ZO calibration utility for Mamba-based Cobra models."""
# ==== Ensure repo root on sys.path ====
import sys
from pathlib import Path

_THIS = Path(__file__).resolve()
_REPO_ROOT = _THIS.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
# ======================================

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import draccus
import torch
from torch.utils.data import DataLoader, Subset

from cobra.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from cobra.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform
from cobra.overwatch import initialize_overwatch
from cobra.preprocessing import get_dataset_and_collator
from cobra.training.optimizer import CFZOParams, calibrate_mamba_scales
from cobra.util import set_global_seed

overwatch = initialize_overwatch(__name__)


def _resolve_hf_token(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    potential_path = Path(token)
    if potential_path.is_file():
        return potential_path.read_text().strip()
    return os.environ.get(token, token)


@dataclass
class OfflineCalibrationConfig:
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.COBRA_3B.model_id)
    )
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15.dataset_id)
    )
    dataset_stage: str = "finetune"
    subset_size: int = 256
    batch_size: int = 4
    num_workers: int = 2
    seed: int = 7
    device: Optional[str] = None
    hf_token: Optional[str] = None
    checkpoint_path: Optional[Path] = None
    output_path: Path = Path("mamba_calibration.pt")
    cfzo: CFZOParams = field(default_factory=CFZOParams)


def _load_checkpoint(llm_module: torch.nn.Module, checkpoint_path: Path) -> None:
    overwatch.info(f"Loading checkpoint from `{checkpoint_path}` for calibration", ctx_level=1)
    state = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(state, dict):
        if "model" in state:
            state = state["model"]
        elif "state_dict" in state:
            state = state["state_dict"]
    missing, unexpected = llm_module.load_state_dict(state, strict=False)
    print(f"[CKPT] missing={len(missing)} unexpected={len(unexpected)}")  #testprint
    if missing:
        overwatch.warning(f"Missing keys when loading checkpoint: {missing}", ctx_level=2)
    if unexpected:
        overwatch.warning(f"Unexpected keys when loading checkpoint: {unexpected}", ctx_level=2)


def _prepare_dataloader(cfg: OfflineCalibrationConfig, vision_backbone, image_transform, tokenizer, prompt_builder_fn) -> DataLoader:
    dataset, collator = get_dataset_and_collator(
        cfg.dataset_stage,
        cfg.dataset,
        image_transform,
        tokenizer,
        prompt_builder_fn=prompt_builder_fn,
        default_image_resolution=vision_backbone.default_image_resolution,
    )

    subset_size = min(cfg.subset_size, len(dataset))
    if subset_size < len(dataset):
        indices = torch.randperm(len(dataset))[:subset_size]
        dataset = Subset(dataset, indices.tolist())
    overwatch.info(f"Calibrating with subset of {len(dataset)} samples", ctx_level=1)
    print(f"[DL] stage={cfg.dataset_stage} subset={len(dataset)} bs={cfg.batch_size} num_workers={cfg.num_workers}")  #testprint

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collator,
    )


@draccus.wrap()
def offline_calibration(cfg: OfflineCalibrationConfig) -> None:
    set_global_seed(cfg.seed)
    device = torch.device(cfg.device) if cfg.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    overwatch.info(f"Running CF-ZO calibration on device `{device}`", ctx_level=1)

    hf_token = _resolve_hf_token(cfg.hf_token)

    # Build backbones/tokenizer
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        cfg.model.vision_backbone_id, cfg.model.image_resize_strategy
    )
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        cfg.model.llm_backbone_id, cfg.model.llm_max_length, hf_token=hf_token, inference_mode=False
    )

    # Optional: load a fine-tuned checkpoint into LLM
    if cfg.checkpoint_path:
        _load_checkpoint(llm_backbone.llm, cfg.checkpoint_path)

    # Data
    dataloader = _prepare_dataloader(
        cfg,
        vision_backbone,
        image_transform,
        tokenizer,
        llm_backbone.prompt_builder_fn,
    )

    # Calibrate
    llm_backbone.llm.to(device)
    state_dict = calibrate_mamba_scales(llm_backbone.llm, dataloader, cfg.cfzo, device=device)

    # Save
    cfg.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, cfg.output_path)
    overwatch.info(f"Saved calibrated mamba_scale state dict to `{cfg.output_path}`", ctx_level=1)


if __name__ == "__main__":
    offline_calibration()
