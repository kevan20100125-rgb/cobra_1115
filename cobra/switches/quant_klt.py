# cobra/switches/quant_klt.py

"""
quant_klt.py

CLI entrypoint: "Estimate shared KLT matrix (K) for projector rotation".

Version B: Pure weight-based KLT from the LLM final output head (lm_head).

Responsibilities:
    1) Load a pretrained Cobra VLM checkpoint (same ModelConfig / stage as training).
    2) Locate the LLM output head (lm_head), assumed to be a Linear layer.
    3) Take its weight matrix W (shape: [vocab_size, d_model]).
    4) Compute feature covariance C = (W_centered^T @ W_centered) / (N - 1).
    5) Compute eigen-decomposition C = Q Λ Q^T, and take K = Q (d_model x d_model).
    6) Save shared K to disk (e.g., outputs/quantize/shared_klt.pt), to be consumed by
         - runtime loaders (e.g. `quantize/runtime/load_quantized_vlm.py`) for
           fake backend that want to apply the same projector rotation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import torch

from cobra.conf import ModelConfig, ModelRegistry
from cobra.models import (
    get_llm_backbone_and_tokenizer,
    get_vision_backbone_and_transform,
    get_vlm,
)
from cobra.overwatch import initialize_overwatch
from cobra.quantize.rotate.projector import SHARED_KLT_PATH
from cobra.util import set_global_seed

overwatch = initialize_overwatch(__name__)


# =====================================================================
# Config
# =====================================================================


@dataclass
class QuantKLTConfig:
    """
    Configuration for estimating a shared KLT matrix from the LLM final output head.

    Model / Checkpoint
    ------------------
    model:
        ModelConfig from `cobra/conf/models.py`.
    stage:
        Pretraining stage in {"align", "finetune", "full-finetune"}.
    pretrained_checkpoint_root:
        Optional run directory; if None, defaults to `runs/<model_id>`.
    hf_token:
        Either env-var name or Path to a token file.

    Output / Runtime
    ----------------
    klt_out:
        Where to save the shared K matrix (as dict{"K": Tensor}).
        Default is `cobra.quantize.rotate.projector.SHARED_KLT_PATH`, so that
        finalize + runtime backends can consume the same file without having
        to re-agree on a hard-coded path here.
    device:
        "cuda" or "cpu" device for loading the model.
    seed:
        Global seed for reproducibility.
    """

    # --- Model / checkpoint ---
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.COBRA_3B.model_id)
    )
    stage: str = "finetune"
    pretrained_checkpoint_root: Optional[Path] = None
    hf_token: Union[str, Path] = Path(".hf_token")

    # --- Output / runtime ---
    klt_out: Path = SHARED_KLT_PATH
    device: str = "cuda"
    seed: int = 7

    def __post_init__(self) -> None:
        if self.device == "cuda" and not torch.cuda.is_available():
            overwatch.warning("[quant_klt] CUDA not available; falling back to CPU")
            self.device = "cpu"

        self.klt_out.parent.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Helpers
# =====================================================================


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
                "[quant_klt] HF token file does not exist; proceeding without explicit token",
                extra={"hf_token_path": str(spec)},
            )
            return None
        token = spec.read_text().strip()
        if not token:
            overwatch.warning(
                "[quant_klt] HF token file is empty; proceeding without explicit token",
                extra={"hf_token_path": str(spec)},
            )
            return None
        return token

    # Treat as env var name
    env_var = spec
    token = os.environ.get(env_var)
    if token is None or not token.strip():
        overwatch.warning(
            "[quant_klt] HF token env var missing or empty; proceeding without explicit token",
            extra={"env_var": env_var},
        )
        return None
    return token.strip()


def _resolve_lm_head(vlm) -> torch.nn.Linear:
    """
    Locate the LLM output head (lm_head), assumed to be torch.nn.Linear.

    Preferred path:
        vlm.llm_backbone.llm.lm_head

    Fallback:
        vlm.llm_backbone.lm_head

    Raises:
        RuntimeError if no suitable Linear head is found.
    """
    from torch import nn

    llm_backbone = vlm.llm_backbone

    # Preferred path
    if hasattr(llm_backbone, "llm") and hasattr(llm_backbone.llm, "lm_head"):
        cand = llm_backbone.llm.lm_head
        if isinstance(cand, nn.Linear):
            return cand

    # Fallback path
    if hasattr(llm_backbone, "lm_head"):
        cand = llm_backbone.lm_head
        if isinstance(cand, nn.Linear):
            return cand

    raise RuntimeError(
        "[quant_klt] Failed to locate LLM output head (lm_head as Linear); "
        "cannot infer KLT dimension."
    )


def _compute_klt_from_output_head(W: torch.Tensor) -> torch.Tensor:
    """
    Given the weight matrix W of the LLM output head (vocab_size x d_model),
    compute a KLT matrix K in R^{d_model x d_model} via eigen-decomposition
    of the feature covariance.

        - Let W ∈ R^{N x d}, N = vocab_size, d = d_model.
        - Center columns: W_c = W - mean(W, dim=0).
        - Compute covariance: C = (W_c^T @ W_c) / (N - 1).
        - Eigen-decomposition: C = Q Λ Q^T, and take K = Q as the KLT matrix.

    Args:
        W: weight matrix of shape (vocab_size, d_model), float tensor on CPU.

    Returns:
        K: KLT matrix of shape (d_model, d_model), float32.
    """
    if W.ndim != 2:
        raise ValueError(f"[quant_klt] Expected 2D weight matrix, got shape={tuple(W.shape)}")

    vocab_size, d_model = W.shape

    # Use float64 for numerical stability in covariance + eigh.
    W = W.to(dtype=torch.float64)

    # Center columns
    W_mean = W.mean(dim=0, keepdim=True)
    W_centered = W - W_mean

    if vocab_size > 1:
        C = (W_centered.T @ W_centered) / (vocab_size - 1)
    else:
        # Degenerate case; just use outer product
        C = W_centered.T @ W_centered

    # Symmetrize just in case of tiny numerical asymmetry
    C = 0.5 * (C + C.T)

    # Eigen-decomposition: C = Q Λ Q^T
    eigvals, eigvecs = torch.linalg.eigh(C)  # eigvecs: (d_model, d_model)

    # eigvecs columns are orthonormal; use them directly as K
    K = eigvecs.to(dtype=torch.float32)

    # Optional sanity check (no exception; just log)
    with torch.no_grad():
        I_approx = K.T @ K
        diag_dev = (I_approx.diag() - 1.0).abs().max().item()
        off_diag_max = (I_approx - torch.eye(d_model, dtype=I_approx.dtype)).abs().max().item()
        overwatch.info(
            "[quant_klt] KLT orthogonality diagnostics",
            extra={
                "diag_max_abs_error": float(diag_dev),
                "off_diag_max_abs": float(off_diag_max),
            },
        )

    return K


# =====================================================================
# Main logic
# =====================================================================


def quant_klt(cfg: QuantKLTConfig) -> None:
    """
    Orchestrate:
        - model load
        - resolve LLM output head (lm_head)
        - compute KLT from its weight matrix
        - save shared K to disk
    """
    set_global_seed(cfg.seed)
    device = torch.device(cfg.device)

    # ------------------------------------------------------------------
    # Model / backbone construction
    # ------------------------------------------------------------------
    model_id = cfg.model.model_id
    hf_token = _load_hf_token(cfg.hf_token)

    overwatch.info(
        "[quant_klt] Loading model backbones",
        extra={"model_id": model_id, "stage": cfg.stage},
    )

    # We still build full vision + LLM backbones to respect get_vlm interface,
    # though KLT only uses the LLM head weights.
    vision_backbone, _image_transform = get_vision_backbone_and_transform(
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
        "[quant_klt] Instantiating CobraVLM",
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

    # Resolve checkpoint directory
    if cfg.pretrained_checkpoint_root is not None:
        run_dir = cfg.pretrained_checkpoint_root
    else:
        run_dir = Path("runs") / model_id

    overwatch.info(
        "[quant_klt] Loading checkpoint",
        extra={"run_dir": str(run_dir), "stage": cfg.stage},
    )
    vlm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=None)

    # Device / dtype for holding the model; KLT itself will be computed on CPU
    # from lm_head weights.
    dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )
    vlm.to(device=device, dtype=dtype)
    vlm.eval()

    # ------------------------------------------------------------------
    # Resolve LLM output head & inspect weight shape
    # ------------------------------------------------------------------
    try:
        lm_head = _resolve_lm_head(vlm)
    except RuntimeError as exc:
        overwatch.error(str(exc))
        return

    W = lm_head.weight.detach().cpu()
    vocab_size, d_model = W.shape

    overwatch.info(
        "[quant_klt] Resolved output head",
        extra={
            "lm_head_shape": tuple(W.shape),
            "vocab_size": vocab_size,
            "d_model": d_model,
        },
    )

    # ------------------------------------------------------------------
    # Compute KLT matrix from lm_head weight
    # ------------------------------------------------------------------
    overwatch.info(
        "[quant_klt] Computing KLT from LLM final output head weight",
        extra={"method": "covariance_eigh", "device": "cpu"},
    )

    K = _compute_klt_from_output_head(W)

    overwatch.info(
        "[quant_klt] Computed shared KLT matrix",
        extra={"K_shape": tuple(K.shape)},
    )

    # ------------------------------------------------------------------
    # Save shared K
    # ------------------------------------------------------------------
    torch.save({"K": K}, cfg.klt_out)

    overwatch.info(
        "[quant_klt] Saved shared KLT matrix",
        extra={"klt_out": str(cfg.klt_out)},
    )


if __name__ == "__main__":
    # No draccus: simple instantiation with default config.
    cfg = QuantKLTConfig()
    quant_klt(cfg)

