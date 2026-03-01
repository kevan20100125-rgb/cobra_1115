"""CF-ZO calibration utilities for Mamba blocks."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, MutableMapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from cobra.models.mamba.mamba_cfzo import Mamba, capture_mamba_activations

logger = logging.getLogger(__name__)


@dataclass
class CFZOParams:
    """Hyper-parameters for CF-ZO calibration."""

    steps: int = 10
    perturb_radius: float = 1e-2
    step_size: float = 1e-2
    log_frequency: int = 10
    loss_type: str = "ppl"  # "ppl" (cross-entropy) or "entropy" (original entropy-align)


class _ScaleManager:
    """Utility to handle flattened access to all Mamba scaling vectors."""

    def __init__(self, modules: Sequence[Mamba]):
        self._modules: List[Mamba] = list(modules)
        self._sizes: List[int] = [m.mamba_scale.numel() for m in self._modules]
        self._numel = sum(self._sizes)

    @classmethod
    def from_model(cls, model: nn.Module) -> "_ScaleManager":
        modules = [module for module in model.modules() if isinstance(module, Mamba)]
        return cls(modules)

    @property
    def numel(self) -> int:
        return self._numel

    def flatten(self) -> torch.Tensor:
        if not self._modules:
            return torch.zeros(0)
        return torch.cat([module.mamba_scale.detach().reshape(-1) for module in self._modules], dim=0)

    def load_vector(self, vector: torch.Tensor) -> None:
        assert vector.numel() == self._numel, "Vector length mismatch for mamba_scale calibration"
        offset = 0
        for module, size in zip(self._modules, self._sizes):
            chunk = vector[offset : offset + size]
            module.mamba_scale.data.copy_(chunk.view_as(module.mamba_scale).to(module.mamba_scale.device))
            offset += size

    def clamp_(self, min_value: float = 1e-3, max_value: float = 1e3) -> None:
        for module in self._modules:
            module.mamba_scale.data.clamp_(min_value, max_value)

    def reset(self) -> None:
        for module in self._modules:
            module.mamba_scale.data.fill_(1.0)


def _to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {
            key: (value if key == "pixel_values" else _to_device(value, device))
            for key, value in batch.items()
        }
    if isinstance(batch, list):
        return [_to_device(item, device) for item in batch]
    if isinstance(batch, tuple):
        return tuple(_to_device(item, device) for item in batch)
    return batch


def _run_model(model: nn.Module, batch: Any) -> None:
    if isinstance(batch, dict):
        allowed_keys = {
            "input_ids",
            "attention_mask",
            "labels",
            "position_ids",
            "inputs_embeds",
            "past_key_values",
            "use_cache",
        }
        filtered = {key: value for key, value in batch.items() if key in allowed_keys}
        model(**filtered)
        return
    if isinstance(batch, (list, tuple)):
        try:
            model(*batch)
            return
        except TypeError:
            model(batch[0])
            return
    model(batch)


def _entropy_from_logits(logits: torch.Tensor, eps: float = 1e-6, temperature: float = 1.5) -> torch.Tensor:
    """Compute per-token entropy with temperature to increase sensitivity."""
    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1).clamp_min(eps)
    return -(probs * probs.log()).sum(dim=-1)


def _activation_entropy_align_loss(
    activations: Sequence[Dict[str, torch.Tensor]], device: torch.device
) -> torch.Tensor:
    # align entropy loss: match entropy of y_t to x_t for stability under quantization
    if not activations:
        return torch.zeros((), device=device)
    losses = []
    eps = 1e-6
    temperature = 1.5  # align entropy loss
    for entry in activations:
        x = entry["x_t"].to(device=device, dtype=torch.float32)
        y = entry["y_t"].to(device=device, dtype=torch.float32)
        # light normalization to make entropy more sensitive to scale
        x = F.layer_norm(x, x.shape[-1:])
        y = F.layer_norm(y, y.shape[-1:])
        hx = _entropy_from_logits(x, eps=eps, temperature=temperature).mean()
        hy = _entropy_from_logits(y, eps=eps, temperature=temperature).mean()
        losses.append(torch.abs(hy - hx))
    return torch.stack(losses).mean()


def _evaluate_entropy_align_loss(model: nn.Module, batch: Any, device: torch.device) -> torch.Tensor:
    batch_on_device = _to_device(batch, device)
    with torch.no_grad():
        with capture_mamba_activations(model) as activations:
            _run_model(model, batch_on_device)
    return _activation_entropy_align_loss(activations, device=device)


def _evaluate_ppl_loss(model: nn.Module, batch: Any, device: torch.device) -> torch.Tensor:
    """Cross-entropy (perplexity) loss using provided labels; mirrors MambaExtend PPL-style calibration."""
    batch_on_device = _to_device(batch, device)
    allowed_keys = {
        "input_ids",
        "attention_mask",
        "labels",
        "position_ids",
        "inputs_embeds",
        "past_key_values",
        "use_cache",
    }
    if isinstance(batch_on_device, dict):
        inputs = {k: v for k, v in batch_on_device.items() if k in allowed_keys}
    elif isinstance(batch_on_device, (list, tuple)):
        # assume first element is dict from collator
        inputs = batch_on_device[0] if isinstance(batch_on_device[0], dict) else {"input_ids": batch_on_device[0]}
    else:
        inputs = {"input_ids": batch_on_device}
    with torch.no_grad():
        outputs = model(**inputs)
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
       
        logits = outputs.logits if hasattr(outputs, "logits") else None
        labels = inputs.get("labels", None)
        if logits is None or labels is None:
            return torch.zeros((), device=device)
      
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        vocab_size = shift_logits.size(-1)
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss


def calibrate_mamba_scales(
    model: nn.Module,
    dataloader: Iterable[Any],
    params: CFZOParams,
    device: Optional[torch.device] = None,
) -> MutableMapping[str, torch.Tensor]:
    """Run CF-ZO calibration across the provided dataloader."""

    scale_manager = _ScaleManager.from_model(model)
    if scale_manager.numel == 0:
        logger.warning("No Mamba blocks detected for CF-ZO calibration.")
        return {}

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    vector = scale_manager.flatten().to(device=device, dtype=torch.float32)
    was_training = model.training
    model.eval()
    batch_iter = iter(dataloader)
    # choose loss function
    if params.loss_type == "ppl":
        eval_loss_fn = _evaluate_ppl_loss
        loss_tag = "ppl"
    else:
        eval_loss_fn = _evaluate_entropy_align_loss
        loss_tag = "entropy-align"

    try:
        for step in range(params.steps):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(dataloader)
                batch = next(batch_iter)
            perturb = torch.randint_like(vector, low=0, high=2, dtype=torch.float32)
            perturb.mul_(2.0).sub_(1.0)
            scale_manager.load_vector(vector + params.perturb_radius * perturb)
            loss_plus = eval_loss_fn(model, batch, device)
            scale_manager.load_vector(vector - params.perturb_radius * perturb)
            loss_minus = eval_loss_fn(model, batch, device)
            grad = ((loss_plus - loss_minus) / (2.0 * params.perturb_radius)) * perturb
            vector = vector - params.step_size * grad
            scale_manager.load_vector(vector)
            scale_manager.clamp_()
            vector = scale_manager.flatten().to(device=device, dtype=torch.float32)
            if params.log_frequency and step % params.log_frequency == 0:
                logger.info(
                    "[CF-ZO][%s] step %d loss+=%.6f loss-=%.6f", loss_tag, step, loss_plus.item(), loss_minus.item()
                )
                base_loss = eval_loss_fn(model, batch, device)
                logger.info("[CF-ZO][%s] step %d base=%.6f |scale|_mean=%.6f", loss_tag, step, base_loss.item(), vector.abs().mean().item())
    finally:
        scale_manager.load_vector(vector)
        scale_manager.clamp_()
        model.train(was_training)

    state_dict: Dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        if name.endswith("mamba_scale"):
            state_dict[name] = parameter.detach().cpu().clone()
    return state_dict
