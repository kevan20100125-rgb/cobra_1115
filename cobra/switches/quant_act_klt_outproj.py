# cobra/switches/quant_act_klt_outproj.py

"""
Export layer-wise activation KLT (act-KLT) for Mamba mixer out_proj.

Scope:
  - Only Mamba mixer out_proj linears (per layer).
  - Layer-wise K (NOT shared).
  - Blockwise KLT with block_size aligned to Hadamard block size (default 512).

Output:
  outputs/quantize/act_klt_outproj_bs{B}/act_klt_outproj.pt

Format:
  {
    "meta": {...},
    "layers": {
      <layer_idx:int>: Tensor[n_blocks, B, B]   # K blocks (orthogonal)
    }
  }

Notes:
  - Collection uses forward_pre_hook on out_proj to capture input activations.
  - Accumulators are on CPU float32; eig uses float64 for stability.
  - This module mirrors cobra.switches.quant_calibrate.py materialization patterns.
"""

from __future__ import annotations

import os
import time
import math
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader

from cobra.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from cobra.models.materialize import (
    get_llm_backbone_and_tokenizer,
    get_vlm,
    get_vision_backbone_and_transform,
)
from cobra.overwatch import initialize_overwatch
from cobra.preprocessing.materialize import get_dataset_and_collator
from cobra.util import set_global_seed

# Disable Tokenizers Parallelism to avoid worker contention
os.environ["TOKENIZERS_PARALLELISM"] = "false"

overwatch = initialize_overwatch(__name__)

def _move_to_device(obj, device: torch.device):
    """
    Recursively move tensors in nested structures (dict/list/tuple) onto `device`.

    This is required because cobra collators often produce nested dicts such as:
      batch["pixel_values"] = { "global": Tensor, "local": Tensor, ... }

    If only top-level tensors are moved, indexing in cobra forward will crash due to
    CPU tensor indexed by CUDA indices.
    """
    if torch.is_tensor(obj):
        return obj.to(device, non_blocking=True)

    if isinstance(obj, dict):
        for k, v in list(obj.items()):
            obj[k] = _move_to_device(v, device)
        return obj

    if isinstance(obj, (list, tuple)):
        moved = [_move_to_device(v, device) for v in obj]
        return type(obj)(moved)

    # leave other types unchanged (ints/strings/None/custom objects)
    return obj

# =====================================================================
# Config
# =====================================================================

@dataclass
class QuantActKLTOutProjConfig:
    # === Model / Dataset Selection ===
    model: ModelConfig = field(
        default_factory=ModelConfig.get_choice_class(ModelRegistry.COBRA_3B.model_id)
    )
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(
            DatasetRegistry.TEXTVQA_100_CALIB.dataset_id
        )
    )

    stage: str = "finetune"
    pretrained_checkpoint_root: Optional[Path] = None
    hf_token: Union[str, Path] = Path(".hf_token")

    device: str = "cuda"

    per_device_batch_size: int = 8
    num_workers: int = 4
    max_calib_batches: int = 0
    max_tokens_per_sample: int = 128

    block_size: int = 512
    eps: float = 1e-4
    compute_dtype: torch.dtype = torch.float32
    eig_dtype: torch.dtype = torch.float64

    seed: int = 7

    act_klt_in_out: Path = Path("outputs/quantize/act_klt_outproj_in_bs512/act_klt_outproj_in.pt")

    export_out_feature: bool = True
    act_klt_out_out: Path = Path("outputs/quantize/act_klt_outproj_out_bs512/act_klt_outproj_out.pt")

    # === Reporting ===
    eig_topk: int = 8  # top-k eigen spectrum summary per block (aggregated per-layer)
    report_head_layers: int = 8  # include first N layers in "head_layers" to keep logs readable

    def __post_init__(self) -> None:
        if int(self.block_size) <= 0:
            raise ValueError(
                f"[quant_act_klt_outproj] block_size must be positive, got {self.block_size}"
            )

        dev = str(self.device).lower().strip()
        if dev == "cuda" and not torch.cuda.is_available():
            overwatch.warning(
                "[quant_act_klt_outproj] CUDA requested but not available; falling back to CPU"
            )
            dev = "cpu"
        if dev not in ("cuda", "cpu"):
            overwatch.warning(
                "[quant_act_klt_outproj] Unknown device; falling back to CPU",
                extra={"device": self.device},
            )
            dev = "cpu"
        self.device = dev

        if int(self.eig_topk) <= 0:
            raise ValueError(f"[quant_act_klt_outproj] eig_topk must be positive, got {self.eig_topk}")
        if int(self.report_head_layers) < 0:
            raise ValueError(
                f"[quant_act_klt_outproj] report_head_layers must be >=0, got {self.report_head_layers}"
            )

        # ---- Robust output path auto-alignment with block_size ----
        B = int(self.block_size)

        def _is_default_bs512_path(p: Path, stem: str) -> bool:
            s = str(p).replace("\\", "/")
            return (f"act_klt_outproj_in_bs512/{stem}" in s) or (f"act_klt_outproj_out_bs512/{stem}" in s)

        # If user didn't override paths (still default bs512), rewrite to current block_size.
        in_default = _is_default_bs512_path(self.act_klt_in_out, "act_klt_outproj_in.pt")
        out_default = _is_default_bs512_path(self.act_klt_out_out, "act_klt_outproj_out.pt")

        if in_default:
            self.act_klt_in_out = Path(f"outputs/quantize/act_klt_outproj_in_bs{B}/act_klt_outproj_in.pt")
        if bool(self.export_out_feature) and out_default:
            self.act_klt_out_out = Path(f"outputs/quantize/act_klt_outproj_out_bs{B}/act_klt_outproj_out.pt")

        self.act_klt_in_out.parent.mkdir(parents=True, exist_ok=True)
        if bool(self.export_out_feature):
            self.act_klt_out_out.parent.mkdir(parents=True, exist_ok=True)

def _load_hf_token(spec: Union[str, Path]) -> Optional[str]:
    """
    Resolve HF token:
      - Path => read file
      - str  => treat as env var name
    """
    if isinstance(spec, Path):
        if not spec.is_file():
            overwatch.warning(
                "[quant_act_klt_outproj] HF token file missing; proceeding without explicit token",
                extra={"hf_token_path": str(spec)},
            )
            return None
        token = spec.read_text().strip()
        return token if token else None

    env_var = str(spec)
    token = os.environ.get(env_var, "").strip()
    return token if token else None

# =====================================================================
# JSON report sink (collect log-body JSON into a single file)
# =====================================================================

_REPORT_SINK = None  # type: Optional["_JsonReportSink"]


class _JsonReportSink:
    """
    Collect JSON payloads that are printed into log *body* and flush them into a file.

    Output format: a single JSON object:
      {
        "meta": {...},
        "events": [ {...}, {...}, ... ]
      }
    """

    def __init__(self, out_path: Path) -> None:
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = {}
        self.events = []

    def set_meta(self, meta: dict) -> None:
        if isinstance(meta, dict):
            self.meta = meta

    def add(self, obj: dict) -> None:
        if obj is None or not isinstance(obj, dict):
            return

        # Optional lightweight de-dup key
        ev = obj.get("event", None)
        tag = obj.get("tag", None)
        if not hasattr(self, "_seen"):
            self._seen = set()  # type: ignore[attr-defined]

        key = None
        if isinstance(ev, str):
            key = (ev, str(tag) if tag is not None else None)

        if key is not None:
            if key in self._seen:  # type: ignore[attr-defined]
                # allow duplicates only if explicitly requested
                return
            self._seen.add(key)  # type: ignore[attr-defined]

        self.events.append(dict(obj))

    def flush(self) -> None:
        tmp = self.out_path.with_suffix(self.out_path.suffix + ".tmp")
        payload = {"meta": self.meta, "events": self.events}
        tmp.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
        tmp.replace(self.out_path)

def _log_json_line(obj: dict) -> None:
    """
    Record JSON payloads to the aggregated report file only.
    Keep stdout log volume low (no JSON dump in log body).
    """
    global _REPORT_SINK
    if _REPORT_SINK is not None and isinstance(obj, dict):
        try:
            _REPORT_SINK.add(obj)
        except Exception:
            pass
    # Intentionally no overwatch.info(json.dumps(...)) here.

def _pct(sorted_vals, q: float) -> float:
    # q in [0,1]
    n = len(sorted_vals)
    if n == 0:
        return float("nan")
    if n == 1:
        return float(sorted_vals[0])
    x = q * (n - 1)
    lo = int(math.floor(x))
    hi = int(math.ceil(x))
    if lo == hi:
        return float(sorted_vals[lo])
    a = float(sorted_vals[lo])
    b = float(sorted_vals[hi])
    t = x - lo
    return a * (1.0 - t) + b * t

def _dist_stats(vals: list) -> dict:
    s = sorted([float(v) for v in vals if v is not None and not math.isnan(float(v))])
    if not s:
        return {"count": 0}
    return {
        "count": len(s),
        "min": float(s[0]),
        "p50": _pct(s, 0.50),
        "p90": _pct(s, 0.90),
        "p95": _pct(s, 0.95),
        "p99": _pct(s, 0.99),
        "max": float(s[-1]),
    }

@torch.no_grad()
def _offdiag_ratio_frob(cov: torch.Tensor) -> float:
    """
    offdiag_ratio = ||offdiag(C)||_F / ||diag(C)||_F
    cov: [B,B] float64 or float32 on CPU
    """
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return float("nan")
    d = torch.diag(cov)
    diag_energy = torch.sum(d * d).item()
    if diag_energy <= 0.0:
        return float("inf")
    total_energy = torch.sum(cov * cov).item()
    offdiag_energy = max(total_energy - diag_energy, 0.0)
    return float(math.sqrt(offdiag_energy / diag_energy))

@torch.no_grad()
def _topk_list(evals_desc: torch.Tensor, k: int) -> list:
    k = int(k)
    if k <= 0:
        return []
    kk = min(k, int(evals_desc.numel()))
    return [float(x) for x in evals_desc[:kk].tolist()]


@torch.no_grad()
def _accumulate_blockwise_cov_inplace(
    sum_z: torch.Tensor,
    sum_zzT: torch.Tensor,
    count: torch.Tensor,
    x_flat: torch.Tensor,
) -> None:
    """
    Update blockwise sums in-place.

    Args:
      sum_z:   [n_blocks, B]
      sum_zzT: [n_blocks, B, B]
      count:   [n_blocks] int64
      x_flat:  [N, D] CPU float32
    """
    if x_flat.ndim != 2:
        return

    N, D = x_flat.shape
    n_blocks, B = sum_z.shape
    if D != n_blocks * B:
        return

    for bi in range(n_blocks):
        s = bi * B
        e = s + B
        z = x_flat[:, s:e]  # [N,B]
        sum_z[bi] += z.sum(dim=0)
        sum_zzT[bi] += z.t().matmul(z)
        count[bi] += N


# =====================================================================
# Main
# =====================================================================

@torch.no_grad()
def quant_act_klt_outproj(cfg: QuantActKLTOutProjConfig) -> None:
    """
    Export layer-wise act-KLT for Mamba mixer *top-level* out_proj, for BOTH:
      - IN side  (out_proj input activations):  K_in  [nb_in, B, B]
      - OUT side (out_proj output activations): K_out [nb_out, B, B]

    Also logs (JSON line, no overwatch extra):
      (1) whitening/decorrelation proxy via offdiag Frobenius ratio (before/after eig-rotation)
      (2) top-k eigen spectrum summary (per layer aggregated over blocks)
      (3) 64-layer distributions for key metrics

    Additionally:
      - Collects all JSON payloads emitted in log body into a single JSON file:
        /work/asdf1234/cobra_1115/outputs/quantize/act_klt_outproj_report.json
    """
    global _REPORT_SINK

    # ----------------------------
    # Enable report sink
    # ----------------------------
    report_path = Path("/work/asdf1234/cobra_1115/outputs/quantize/act_klt_outproj_report.json")
    _REPORT_SINK = _JsonReportSink(report_path)

    run_t0 = time.time()
    try:
        set_global_seed(cfg.seed)

        device = torch.device(cfg.device)
        dtype = torch.float32

        hf_token = _load_hf_token(cfg.hf_token)
        model_id = cfg.model.model_id

        # record meta early (so even failures produce useful report)
        _REPORT_SINK.set_meta(
            {
                "timestamp_start": float(run_t0),
                "model_id": str(cfg.model.model_id),
                "llm_backbone_id": str(cfg.model.llm_backbone_id),
                "vision_backbone_id": str(cfg.model.vision_backbone_id),
                "dataset_id": str(cfg.dataset.dataset_id),
                "stage": str(cfg.stage),
                "device": str(cfg.device),
                "dtype": str(dtype).replace("torch.", ""),
                "compute_dtype": str(cfg.compute_dtype).replace("torch.", ""),
                "eig_dtype": str(cfg.eig_dtype).replace("torch.", ""),
                "block_size": int(cfg.block_size),
                "max_tokens_per_sample": int(cfg.max_tokens_per_sample),
                "max_calib_batches": int(cfg.max_calib_batches),
                "per_device_batch_size": int(cfg.per_device_batch_size),
                "num_workers": int(cfg.num_workers),
                "eig_topk": int(cfg.eig_topk),
                "report_path": str(report_path),
            }
        )
        _REPORT_SINK.add({"event": "run_start", "timestamp": float(run_t0)})

        overwatch.info(
            f"[QuantActKLTOutProj] Loading Vision Backbone `{cfg.model.vision_backbone_id}` via TIMM"
        )
        vision_backbone, image_transform = get_vision_backbone_and_transform(
            cfg.model.vision_backbone_id,
            image_resize_strategy=cfg.model.image_resize_strategy,
        )

        overwatch.info(
            f"[QuantActKLTOutProj] Loading LLM Backbone `{cfg.model.llm_backbone_id}` via HF Transformers"
        )
        llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
            cfg.model.llm_backbone_id,
            llm_max_length=cfg.model.llm_max_length,
            hf_token=hf_token,
            inference_mode=True,
        )

        overwatch.info(
            f"[QuantActKLTOutProj] Instantiating CobraVLM `{model_id}` for Stage = `{cfg.stage}`"
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
            f"[QuantActKLTOutProj] Loading checkpoint for `{model_id}` from run_dir = `{run_dir}` "
            f"(stage = `{cfg.stage}`)"
        )
        vlm.load_from_checkpoint(cfg.stage, run_dir, pretrained_checkpoint=None)

        vlm.to(device=device, dtype=dtype)
        vlm.eval()

        overwatch.info(
            f"[QuantActKLTOutProj] Creating Dataset `{cfg.dataset.dataset_id}` for Stage = `{cfg.stage}` "
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
            drop_last=False,
        )

        overwatch.info(
            f"[QuantActKLTOutProj] Initialized DataLoader: "
            f"num_samples={len(train_dataset)}, batch_size={cfg.per_device_batch_size}, num_workers={cfg.num_workers}"
        )

        # ------------------------------------------------------------------
        # Locate llm + layers + mixer.out_proj (top-level only)
        # ------------------------------------------------------------------
        llm = None
        if hasattr(vlm, "llm_backbone"):
            llm = getattr(vlm.llm_backbone, "llm", None)
        if llm is None:
            llm = getattr(llm_backbone, "llm", None)
        if llm is None:
            raise RuntimeError("[QuantActKLTOutProj] Cannot locate llm module for mixer traversal.")

        layers = None
        if hasattr(llm, "backbone") and hasattr(llm.backbone, "layers"):
            layers = llm.backbone.layers
        elif hasattr(llm, "layers"):
            layers = llm.layers
        if layers is None:
            raise RuntimeError(
                "[QuantActKLTOutProj] Cannot locate llm layers container (expected llm.backbone.layers or llm.layers)."
            )

        expected_hidden = None
        cfg_obj = getattr(llm, "config", None)
        if cfg_obj is not None:
            for key in ("d_model", "hidden_size", "n_embd", "model_dim"):
                v = getattr(cfg_obj, key, None)
                if isinstance(v, int) and v > 0:
                    expected_hidden = int(v)
                    break

        layer_to_mod: Dict[int, torch.nn.Module] = {}
        bad_layers: Dict[int, str] = {}

        for li in range(len(layers)):
            layer = layers[li]
            mixer = getattr(layer, "mixer", None)
            if mixer is None:
                bad_layers[li] = "no mixer attr"
                continue

            out_proj = getattr(mixer, "out_proj", None)
            if out_proj is None:
                bad_layers[li] = "mixer.out_proj missing"
                continue

            if not hasattr(out_proj, "weight") or not hasattr(out_proj, "in_features") or not hasattr(out_proj, "out_features"):
                bad_layers[li] = "mixer.out_proj missing weight/in/out_features"
                continue

            in_f = int(getattr(out_proj, "in_features", -1))
            out_f = int(getattr(out_proj, "out_features", -1))
            if in_f <= 0 or out_f <= 0:
                bad_layers[li] = f"mixer.out_proj invalid shapes (in={in_f}, out={out_f})"
                continue

            if expected_hidden is not None and in_f != expected_hidden:
                if in_f == 2 * expected_hidden:
                    overwatch.warning(
                        "[QuantActKLTOutProj] out_proj.in_features is 2*hidden_size; IN-KLT will be exported in 2*hidden space (expected).",
                        extra={"layer": li, "hidden_size": expected_hidden, "out_proj_in_features": in_f},
                    )
                else:
                    overwatch.warning(
                        "[QuantActKLTOutProj] out_proj.in_features differs from hidden_size; exporting IN-KLT in out_proj input space.",
                        extra={"layer": li, "hidden_size": expected_hidden, "out_proj_in_features": in_f},
                    )

            layer_to_mod[li] = out_proj

        if not layer_to_mod:
            raise RuntimeError(
                "[QuantActKLTOutProj] Failed to locate any usable mixer.out_proj modules. "
                f"bad_layers_head={list(bad_layers.items())[:10]}"
            )

        layer_indices = sorted(layer_to_mod.keys())

        first_mod = layer_to_mod[layer_indices[0]]
        D_in = int(getattr(first_mod, "in_features", 0))
        D_out = int(getattr(first_mod, "out_features", 0))
        if D_in <= 0 or D_out <= 0:
            W = getattr(first_mod, "weight", None)
            if W is None or W.ndim != 2:
                raise RuntimeError("[QuantActKLTOutProj] Cannot infer dims for mixer.out_proj.")
            D_out = int(W.shape[0])
            D_in = int(W.shape[1])

        inconsistent = []
        for li in list(layer_indices):
            m = layer_to_mod[li]
            di = int(getattr(m, "in_features", -1))
            do = int(getattr(m, "out_features", -1))
            if di != D_in or do != D_out:
                inconsistent.append((li, di, do))
        if inconsistent:
            head = inconsistent[:10]
            overwatch.warning(
                "[QuantActKLTOutProj] Some layers have different out_proj dims; skipping those layers.",
                extra={"reference_D_in": D_in, "reference_D_out": D_out, "num_inconsistent": len(inconsistent), "head": head},
            )
            for li, _di, _do in inconsistent:
                layer_to_mod.pop(li, None)
            layer_indices = sorted(layer_to_mod.keys())
            if not layer_indices:
                raise RuntimeError("[QuantActKLTOutProj] All layers were inconsistent; no valid out_proj left.")

        B = int(cfg.block_size)
        if D_in % B != 0:
            raise ValueError(f"[QuantActKLTOutProj] out_proj input dim D_in={D_in} not divisible by block_size B={B}")
        n_blocks_in = D_in // B

        export_out = bool(cfg.export_out_feature)
        if export_out and (D_out % B != 0):
            overwatch.warning(
                "[QuantActKLTOutProj] out_proj out_features not divisible by block_size; disabling OUT-KLT export.",
                extra={"D_out": D_out, "block_size": B},
            )
            export_out = False
        n_blocks_out = (D_out // B) if export_out else 0

        # Add meta details discovered after model materialization
        _REPORT_SINK.add(
            {
                "event": "model_dims",
                "num_layers_found": int(len(layer_indices)),
                "num_layers_total": int(len(layers)),
                "out_proj_in_features": int(D_in),
                "out_proj_out_features": int(D_out),
                "block_size": int(B),
                "n_blocks_in": int(n_blocks_in),
                "n_blocks_out": int(n_blocks_out) if export_out else None,
                "hidden_size_from_config": int(expected_hidden) if expected_hidden is not None else None,
            }
        )

        overwatch.info(
            "[QuantActKLTOutProj] Located runtime-equivalent mixer.out_proj modules",
            extra={
                "num_layers_found": len(layer_indices),
                "num_layers_total": len(layers),
                "out_proj_in_features": int(D_in),
                "out_proj_out_features": int(D_out),
                "block_size": B,
                "n_blocks_in": int(n_blocks_in),
                "n_blocks_out": int(n_blocks_out) if export_out else None,
                "hidden_size_from_config": int(expected_hidden) if expected_hidden is not None else None,
            },
        )

        # ------------------------------------------------------------------
        # Allocate CPU accumulators per layer (IN + OUT)
        # ------------------------------------------------------------------
        sum_z_in: Dict[int, torch.Tensor] = {}
        sum_zzT_in: Dict[int, torch.Tensor] = {}
        count_in: Dict[int, torch.Tensor] = {}

        sum_z_out: Dict[int, torch.Tensor] = {}
        sum_zzT_out: Dict[int, torch.Tensor] = {}
        count_out: Dict[int, torch.Tensor] = {}

        for li in layer_indices:
            sum_z_in[li] = torch.zeros((n_blocks_in, B), dtype=cfg.compute_dtype, device="cpu")
            sum_zzT_in[li] = torch.zeros((n_blocks_in, B, B), dtype=cfg.compute_dtype, device="cpu")
            count_in[li] = torch.zeros((n_blocks_in,), dtype=torch.int64, device="cpu")

            if export_out:
                sum_z_out[li] = torch.zeros((n_blocks_out, B), dtype=cfg.compute_dtype, device="cpu")
                sum_zzT_out[li] = torch.zeros((n_blocks_out, B, B), dtype=cfg.compute_dtype, device="cpu")
                count_out[li] = torch.zeros((n_blocks_out,), dtype=torch.int64, device="cpu")

        # ------------------------------------------------------------------
        # Hooks
        # ------------------------------------------------------------------
        hooks = []

        hook_hits_in: Dict[int, int] = {int(li): 0 for li in layer_indices}
        acc_hits_in: Dict[int, int] = {int(li): 0 for li in layer_indices}
        dim_mismatch_in: Dict[int, int] = {int(li): 0 for li in layer_indices}

        hook_hits_out: Dict[int, int] = {int(li): 0 for li in layer_indices}
        acc_hits_out: Dict[int, int] = {int(li): 0 for li in layer_indices}
        dim_mismatch_out: Dict[int, int] = {int(li): 0 for li in layer_indices}

        def _cap_and_flatten(x: torch.Tensor) -> Optional[torch.Tensor]:
            if not torch.is_tensor(x):
                return None
            if x.ndim == 3 and cfg.max_tokens_per_sample and cfg.max_tokens_per_sample > 0:
                x = x[:, : int(cfg.max_tokens_per_sample), :]
            x = x.detach()
            if x.ndim == 3:
                x = x.reshape(-1, x.shape[-1])
            elif x.ndim == 2:
                pass
            else:
                return None
            return x

        def _make_in_pre_hook(layer_idx: int):
            def _pre_hook(module, inputs):
                hook_hits_in[layer_idx] += 1
                if not inputs:
                    return
                x = inputs[0]
                x = _cap_and_flatten(x)
                if x is None:
                    return
                if int(x.shape[-1]) != D_in:
                    dim_mismatch_in[layer_idx] += 1
                    return
                x_cpu = x.to(device="cpu", dtype=cfg.compute_dtype, non_blocking=True)
                _accumulate_blockwise_cov_inplace(sum_z_in[layer_idx], sum_zzT_in[layer_idx], count_in[layer_idx], x_cpu)
                acc_hits_in[layer_idx] += 1
            return _pre_hook

        def _make_out_fwd_hook(layer_idx: int):
            def _fwd_hook(module, inputs, output):
                if not export_out:
                    return
                hook_hits_out[layer_idx] += 1
                y = output
                y = _cap_and_flatten(y)
                if y is None:
                    return
                if int(y.shape[-1]) != D_out:
                    dim_mismatch_out[layer_idx] += 1
                    return
                y_cpu = y.to(device="cpu", dtype=cfg.compute_dtype, non_blocking=True)
                _accumulate_blockwise_cov_inplace(sum_z_out[layer_idx], sum_zzT_out[layer_idx], count_out[layer_idx], y_cpu)
                acc_hits_out[layer_idx] += 1
            return _fwd_hook

        for li in layer_indices:
            mod = layer_to_mod[li]
            hooks.append(mod.register_forward_pre_hook(_make_in_pre_hook(int(li))))
            hooks.append(mod.register_forward_hook(_make_out_fwd_hook(int(li))))

        overwatch.info(
            "[QuantActKLTOutProj] Registered hooks",
            extra={
                "num_layers": len(layer_indices),
                "num_hooks_total": len(hooks),
                "export_out_feature": bool(export_out),
                "D_in": int(D_in),
                "D_out": int(D_out),
                "block_size": int(B),
            },
        )

        # ------------------------------------------------------------------
        # Run forward passes
        # ------------------------------------------------------------------
        num_batches = 0
        t0 = time.time()

        for batch in train_dataloader:
            num_batches += 1
            if cfg.max_calib_batches and cfg.max_calib_batches > 0 and num_batches > int(cfg.max_calib_batches):
                break

            batch = _move_to_device(batch, device)
            _ = vlm(**batch)

            if num_batches % 10 == 0:
                overwatch.info(
                    "[QuantActKLTOutProj] Progress",
                    extra={"batches": num_batches, "elapsed_sec": round(time.time() - t0, 2)},
                )

        for h in hooks:
            try:
                h.remove()
            except Exception:
                pass

        overwatch.info(
            "[QuantActKLTOutProj] Finished activation collection",
            extra={"batches": num_batches, "elapsed_sec": round(time.time() - t0, 2)},
        )

        # ------------------------------------------------------------------
        # Compute KLT per layer/block (IN + OUT) + JSON reports
        # ------------------------------------------------------------------
        eps = float(cfg.eps)
        topk = int(cfg.eig_topk)
        headN = int(cfg.report_head_layers)

        layers_in: Dict[int, torch.Tensor] = {}
        layers_out: Dict[int, torch.Tensor] = {}

        report_in: Dict[int, dict] = {}
        report_out: Dict[int, dict] = {}

        def _compute_side(
            *,
            tag: str,
            layer_indices_use: list,
            n_blocks: int,
            sum_z: Dict[int, torch.Tensor],
            sum_zzT: Dict[int, torch.Tensor],
            count: Dict[int, torch.Tensor],
            out_dict: Dict[int, torch.Tensor],
            report_dict: Dict[int, dict],
        ) -> None:
            eye = torch.eye(B, dtype=cfg.eig_dtype, device="cpu")

            layer_off_before = []
            layer_off_after = []
            layer_top1 = []
            layer_trace = []
            layer_cond = []

            for li in layer_indices_use:
                K_side = torch.empty((n_blocks, B, B), dtype=torch.float32, device="cpu")

                block_off_before = []
                block_off_after = []
                block_topk_lists = []
                block_traces = []
                block_conds = []

                for bi in range(n_blocks):
                    n = int(count[li][bi].item())
                    if n <= 0:
                        K_side[bi] = torch.eye(B, dtype=torch.float32)
                        continue

                    mu = (sum_z[li][bi] / float(n)).to(dtype=cfg.eig_dtype)
                    EzzT = (sum_zzT[li][bi] / float(n)).to(dtype=cfg.eig_dtype)
                    cov = EzzT - torch.outer(mu, mu)
                    cov = 0.5 * (cov + cov.t())
                    if eps > 0:
                        cov = cov + (eps * eye)

                    off_b = _offdiag_ratio_frob(cov)
                    block_off_before.append(off_b)

                    evals, evecs = torch.linalg.eigh(cov)
                    idx = torch.argsort(evals, descending=True)
                    evals = evals[idx]
                    evecs = evecs[:, idx]

                    K_side[bi] = evecs.to(dtype=torch.float32)

                    topk_list = _topk_list(evals, topk)
                    block_topk_lists.append(topk_list)

                    tr = float(torch.sum(evals).item())
                    block_traces.append(tr)

                    ev_max = float(evals[0].item()) if evals.numel() > 0 else float("nan")
                    ev_min = float(evals[-1].item()) if evals.numel() > 0 else float("nan")
                    ev_min_safe = max(ev_min, 1e-30)
                    block_conds.append(float(ev_max / ev_min_safe))

                    cov_rot = evecs.t().matmul(cov).matmul(evecs)
                    off_a = _offdiag_ratio_frob(cov_rot)
                    block_off_after.append(off_a)

                out_dict[int(li)] = K_side

                # Use global _dist_stats for per-layer block summaries (consistent definition)
                off_b_sum = _dist_stats(block_off_before)
                off_a_sum = _dist_stats(block_off_after)
                tr_sum = _dist_stats(block_traces)
                cond_sum = _dist_stats(block_conds)

                # Top-k aggregation across blocks:
                topk_mean = []
                topk_med = []
                if block_topk_lists:
                    Kk = topk
                    cols = [[] for _ in range(Kk)]
                    for lst in block_topk_lists:
                        for ki in range(Kk):
                            cols[ki].append(float(lst[ki]) if ki < len(lst) else float("nan"))
                    for ki in range(Kk):
                        col = [x for x in cols[ki] if not math.isnan(x)]
                        if not col:
                            topk_mean.append(float("nan"))
                            topk_med.append(float("nan"))
                        else:
                            col_s = sorted(col)
                            topk_mean.append(float(sum(col) / len(col)))
                            topk_med.append(_pct(col_s, 0.50))

                layer_summary = {
                    "n_blocks_used": int(tr_sum.get("count", 0)),
                    "offdiag_ratio_before": {
                        "count": int(off_b_sum.get("count", 0)),
                        "min": float(off_b_sum.get("min", float("nan"))),
                        "p50": float(off_b_sum.get("p50", float("nan"))),
                        "p90": float(off_b_sum.get("p90", float("nan"))),
                        "max": float(off_b_sum.get("max", float("nan"))),
                    },
                    "offdiag_ratio_after": {
                        "count": int(off_a_sum.get("count", 0)),
                        "min": float(off_a_sum.get("min", float("nan"))),
                        "p50": float(off_a_sum.get("p50", float("nan"))),
                        "p90": float(off_a_sum.get("p90", float("nan"))),
                        "max": float(off_a_sum.get("max", float("nan"))),
                    },
                    "eig_topk_mean": topk_mean,
                    "eig_topk_median": topk_med,
                    "trace": {
                        "count": int(tr_sum.get("count", 0)),
                        "min": float(tr_sum.get("min", float("nan"))),
                        "p50": float(tr_sum.get("p50", float("nan"))),
                        "p90": float(tr_sum.get("p90", float("nan"))),
                        "max": float(tr_sum.get("max", float("nan"))),
                    },
                    "cond": {
                        "count": int(cond_sum.get("count", 0)),
                        "min": float(cond_sum.get("min", float("nan"))),
                        "p50": float(cond_sum.get("p50", float("nan"))),
                        "p90": float(cond_sum.get("p90", float("nan"))),
                        "max": float(cond_sum.get("max", float("nan"))),
                    },
                }
                report_dict[int(li)] = layer_summary

                # Per-layer scalars for 64-layer distributions (use p50)
                if int(layer_summary["offdiag_ratio_before"].get("count", 0)) > 0:
                    layer_off_before.append(float(layer_summary["offdiag_ratio_before"]["p50"]))
                if int(layer_summary["offdiag_ratio_after"].get("count", 0)) > 0:
                    layer_off_after.append(float(layer_summary["offdiag_ratio_after"]["p50"]))
                if int(layer_summary["trace"].get("count", 0)) > 0:
                    layer_trace.append(float(layer_summary["trace"]["p50"]))
                if int(layer_summary["cond"].get("count", 0)) > 0:
                    layer_cond.append(float(layer_summary["cond"]["p50"]))
                if topk_mean:
                    v = float(topk_mean[0]) if not math.isnan(float(topk_mean[0])) else float("nan")
                    layer_top1.append(v)

            head_layers = {}
            for li in layer_indices_use[:headN]:
                head_layers[str(li)] = report_dict[int(li)]

            _log_json_line(
                {
                    "event": "act_klt_whiten_spectrum_summary",
                    "tag": tag,
                    "block_size": int(B),
                    "eps": float(eps),
                    "topk": int(topk),
                    "n_layers": int(len(layer_indices_use)),
                    "n_blocks": int(n_blocks),
                    "head_layers": head_layers,
                    "layer_offdiag_before_p50_dist": _dist_stats(layer_off_before),
                    "layer_offdiag_after_p50_dist": _dist_stats(layer_off_after),
                    "layer_trace_p50_dist": _dist_stats(layer_trace),
                    "layer_cond_p50_dist": _dist_stats(layer_cond),
                    "layer_top1_eig_mean_dist": _dist_stats(layer_top1),
                }
            )

        _compute_side(
            tag="IN",
            layer_indices_use=layer_indices,
            n_blocks=n_blocks_in,
            sum_z=sum_z_in,
            sum_zzT=sum_zzT_in,
            count=count_in,
            out_dict=layers_in,
            report_dict=report_in,
        )

        if export_out:
            _compute_side(
                tag="OUT",
                layer_indices_use=layer_indices,
                n_blocks=n_blocks_out,
                sum_z=sum_z_out,
                sum_zzT=sum_zzT_out,
                count=count_out,
                out_dict=layers_out,
                report_dict=report_out,
            )

        # ------------------------------------------------------------------
        # Save payloads (two files)
        # ------------------------------------------------------------------
        meta_common = {
            "timestamp": float(time.time()),
            "model_id": str(cfg.model.model_id),
            "llm_backbone_id": str(cfg.model.llm_backbone_id),
            "vision_backbone_id": str(cfg.model.vision_backbone_id),
            "dataset_id": str(cfg.dataset.dataset_id),
            "stage": str(cfg.stage),
            "device": str(cfg.device),
            "dtype": str(dtype).replace("torch.", ""),
            "compute_dtype": str(cfg.compute_dtype).replace("torch.", ""),
            "eig_dtype": str(cfg.eig_dtype).replace("torch.", ""),
            "block_size": int(cfg.block_size),
            "hidden_size_from_config": int(expected_hidden) if expected_hidden is not None else None,
            "max_tokens_per_sample": int(cfg.max_tokens_per_sample),
            "max_calib_batches": int(cfg.max_calib_batches),
            "per_device_batch_size": int(cfg.per_device_batch_size),
            "num_workers": int(cfg.num_workers),
            "eig_topk": int(cfg.eig_topk),
        }

        payload_in = {
            "meta": {
                **meta_common,
                "side": "in",
                "out_proj_in_features": int(D_in),
                "n_blocks": int(n_blocks_in),
            },
            "layers": layers_in,
        }
        torch.save(payload_in, cfg.act_klt_in_out)
        overwatch.info(
            "[QuantActKLTOutProj] Saved IN act-KLT payload",
            extra={
                "act_klt_in_out": str(cfg.act_klt_in_out),
                "num_layers": len(layer_indices),
                "D_in": int(D_in),
                "n_blocks_in": int(n_blocks_in),
            },
        )

        if export_out:
            payload_out = {
                "meta": {
                    **meta_common,
                    "side": "out",
                    "out_proj_out_features": int(D_out),
                    "n_blocks": int(n_blocks_out),
                },
                "layers": layers_out,
            }
            torch.save(payload_out, cfg.act_klt_out_out)
            overwatch.info(
                "[QuantActKLTOutProj] Saved OUT act-KLT payload",
                extra={
                    "act_klt_out_out": str(cfg.act_klt_out_out),
                    "num_layers": len(layer_indices),
                    "D_out": int(D_out),
                    "n_blocks_out": int(n_blocks_out),
                },
            )

        t1 = time.time()
        _REPORT_SINK.add({"event": "run_end", "timestamp": float(t1), "elapsed_sec": float(t1 - run_t0)})

    finally:
        # Always flush the collected report, even if exceptions happen.
        if _REPORT_SINK is not None:
            try:
                _REPORT_SINK.flush()
                overwatch.info(
                    "[QuantActKLTOutProj] Wrote aggregated JSON report",
                    extra={"report_path": str(getattr(_REPORT_SINK, "out_path", "<?>"))},
                )
            except Exception as e:
                overwatch.warning(
                    "[QuantActKLTOutProj] Failed to write aggregated JSON report",
                    extra={"error": repr(e)},
                )
        _REPORT_SINK = None

if __name__ == "__main__":
    cfg = QuantActKLTOutProjConfig()
    quant_act_klt_outproj(cfg)

