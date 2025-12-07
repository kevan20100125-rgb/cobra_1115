# cobra/switches/quant_finalize.py

"""
quant_finalize.py

CLI entrypoint: "Finalize PTQ & Integer Export"

Pipeline role (Phase 2, three-stage PTQ):
    1) quant_calibrate.py
         - Run VLM + calibration dataset
         - Collect activation stats and produce percentile stats
         - (Optionally) also generate hi/lo + summary
    2) quant_pct_apply.py
         - Pure offline step: pct_stats_in → pct_hi_lo_out (+ summary)
         - Does not touch the model or checkpoint
    3) quant_finalize.py  ← 本檔
         - Load a pretrained Cobra VLM checkpoint
         - Wrap selected modules with Quant* wrappers (linear/conv)
         - Load precomputed hi/lo map and apply to activation quantizers
         - Apply shared output-projector rotation R = H K (if enabled)
         - Export integer weights and activation quant parameters as a blob
           for runtime backends to consume

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
    - Implement the online INT runtime; it only produces the offline
      export blob consumed by future backends.

Design note (Phase 2+):
    - Quantization相關設定（bits/backend/哪些 target 量化 / 哪些 target 允許 rotation /
      projector rotation 模式）統一交給
      `cobra.quantize.runtime.config.QuantRuntimeConfig` 解析：
        * bits    -> (weight_bits, act_bits)
        * backend -> mode ∈ {FLOAT, FAKE, INT_EXPORT}
        * enable_* / vision_in_pct_pipeline -> use_pct_for / use_rotation_for
        * projector_rotation_mode -> HK / HADAMARD / NONE
      quant_finalize.py 只負責：
        - 呼叫 QuantRuntimeConfig.from_bits_backend(...) 取得 quant_cfg
        - 依 quant_cfg.use_pct_for 決定 wrap / calibrate / export 的 target
        - 依 quant_cfg.should_rotate_projector() 決定是否嘗試 projector rotation
        - 依 quant_cfg.projector_rotation_mode 決定 HK / HADAMARD / NONE 的行為
        - 把對應設定轉成 IntExportConfig，避免重複手動拼 include_*。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import draccus
import torch

from cobra.conf import ModelConfig, ModelRegistry
from cobra.models import (
    get_llm_backbone_and_tokenizer,
    get_vision_backbone_and_transform,
    get_vlm,
)
from cobra.overwatch import initialize_overwatch
from cobra.quantize.pct.calibrator import calibrate_model_from_hi_lo
from cobra.quantize.runtime.config import (
    QuantRuntimeConfig,
    ProjectorRotationMode,
)
from cobra.quantize.wrap.policy import WrapPolicyConfig
from cobra.quantize.wrap.entry import wrap_model_for_quantization
from cobra.quantize.finalize.int_export import (
    export_int_quant_state,
    save_int_export,
    int_export_config_from_quant_cfg,
)
from cobra.quantize.rotate.projector import (
    ProjectorRotationConfig,
    SHARED_KLT_PATH,
    rotate_cobra_vlm_output_projector_from_path_inplace,
)
from cobra.util import set_global_seed

overwatch = initialize_overwatch(__name__)


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
        ModelConfig from `cobra/conf/models.py`。
    stage:
        Pretraining stage in {"align", "finetune", "full-finetune"}；用來決定要載入哪一個 checkpoint。
    pretrained_checkpoint_root:
        Optional run directory；若為 None，預設為 `runs/<model_id>`。
    hf_token:
        Either:
            - environment variable name containing a HF token, or
            - Path to a file storing the token。

    Percentile calibration
    ----------------------
    pct_hi_lo_in:
        Path to the hi/lo map produced by `quant_pct_apply.py` or
        `quant_calibrate.py` (`torch.save` dict)。

    Quantization mode / bits
    ------------------------
    weight_bits / act_bits:
        Legacy CLI knobs；若 quant_bits=None，會組合成 "W{weight_bits}A{act_bits}"。
    quant_bits:
        統一 bits 字串，例如 "W4A4" / "W8A8" / "W2A4" / "W16A8"。
        若為 None，則由 weight_bits / act_bits 推出。
        實際解析與合法性檢查交給 QuantRuntimeConfig。
    backend:
        QuantRuntime backend 標記，交給 QuantRuntimeConfig：
            - "float" -> FLOAT
            - "fake"  -> FAKE
            - "int"   -> INT_EXPORT（本檔典型設定）

    signed_weights / signed_activations:
        記錄是否使用對稱量化；實際範圍仍由 hi/lo map 決定。

    Targets included in this export
    -------------------------------
    include_vision_dino / include_vision_siglip / include_llm / include_projector:
        Per-target boolean switches。會同時傳入 QuantRuntimeConfig.from_bits_backend，
        並透過 quant_cfg.use_pct_for 反映哪幾個 target 會被 calibrate / export。

    Output-projector rotation
    -------------------------
    projector_rotation_mode:
        高階旋轉模式（交給 QuantRuntimeConfig 管理）：
            - "hk"       : KLT + Hadamard
            - "hadamard" : 只有 Hadamard
            - "none"     : 完全不旋轉
        實際行為由 QuantRuntimeConfig.projector_rotation_mode 決定。

    use_klt / use_hadamard / shared_klt / klt_path:
        細部旗標（在高階模式允許旋轉的前提下，進一步決定是否啟用 KLT / Hadamard）：
            - 實際 use_klt = (projector_rotation_mode 需要 KLT) AND cfg.use_klt
            - 實際 use_hadamard = (projector_rotation_mode 需要 H) AND cfg.use_hadamard
        klt_path:
            - 若實際 use_klt=False，會在主流程中傳入 None，僅做 Hadamard。

    Export
    ------
    out_path:
        Destination of the integer export blob (.pt)。
    device:
        "cuda" or "cpu" for finalization。
    seed:
        Global seed for reproducibility。
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

    # --- Quantization bits / backend ---
    weight_bits: int = 8
    act_bits: int = 8
    quant_bits: Optional[str] = None
    backend: str = "int"  # 典型為 "int"（INT_EXPORT 模式）

    signed_weights: bool = True
    signed_activations: bool = True

    # --- Targets ---
    include_vision_dino: bool = True
    include_vision_siglip: bool = True
    include_llm: bool = True
    include_projector: bool = True

    # --- Rotation（高階模式 + 細部旗標） ---
    # 高階模式：交給 QuantRuntimeConfig 管理 HK / HADAMARD / NONE
    projector_rotation_mode: str = "hk"  # "hk" / "hadamard" / "none"
    # 細部旗標：在高階模式允許的前提下，再決定是否啟用 KLT / Hadamard
    use_klt: bool = True
    use_hadamard: bool = True
    shared_klt: bool = True
    klt_path: Optional[Path] = SHARED_KLT_PATH

    # --- Export & runtime ---
    out_path: Path = Path("outputs/quantize/int_export.pt")
    device: str = "cuda"
    seed: int = 7

    # QuantRuntimeConfig（不透過 CLI 直接設，於 __post_init__ 產生）
    quant_cfg: QuantRuntimeConfig = field(init=False, repr=False)

    def __post_init__(self) -> None:
        # --------------------------------------------------------------
        # 1) 決定 quant_bits：若未指定，從 weight_bits / act_bits 組合
        # --------------------------------------------------------------
        if self.quant_bits is None:
            self.quant_bits = f"W{self.weight_bits}A{self.act_bits}"

        # --------------------------------------------------------------
        # 2) 建立 QuantRuntimeConfig（集中 bits/backend/targets/rotation 模式 校驗）
        # --------------------------------------------------------------
        self.quant_cfg = QuantRuntimeConfig.from_bits_backend(
            bits=self.quant_bits,
            backend=self.backend,
            enable_vision_dino=self.include_vision_dino,
            enable_vision_siglip=self.include_vision_siglip,
            enable_llm=self.include_llm,
            enable_projector=self.include_projector,
            vision_in_pct_pipeline=True,
            symmetric_acts=self.signed_activations,
            symmetric_weights=self.signed_weights,
            config_name=f"quant_finalize::{self.quant_bits}::{self.backend}",
            projector_rotation_mode=self.projector_rotation_mode,
        )

        # 讓 legacy weight_bits / act_bits 與 quant_cfg 對齊，避免兩套來源互相矛盾
        if (self.weight_bits, self.act_bits) != (
            self.quant_cfg.weight_bits,
            self.quant_cfg.act_bits,
        ):
            overwatch.warning(
                "[quant_finalize] (weight_bits, act_bits) differ from quant_bits-derived values; "
                "using quant_bits as source of truth.",
                extra={
                    "cli_weight_bits": self.weight_bits,
                    "cli_act_bits": self.act_bits,
                    "quant_bits": self.quant_bits,
                    "resolved_weight_bits": self.quant_cfg.weight_bits,
                    "resolved_act_bits": self.quant_cfg.act_bits,
                },
            )
        self.weight_bits = self.quant_cfg.weight_bits
        self.act_bits = self.quant_cfg.act_bits

        # --------------------------------------------------------------
        # 3) 裝置檢查
        # --------------------------------------------------------------
        if self.device == "cuda" and not torch.cuda.is_available():
            overwatch.warning("CUDA not available; falling back to CPU for finalization")
            self.device = "cpu"

        # --------------------------------------------------------------
        # 4) 準備輸出目錄
        # --------------------------------------------------------------
        self.out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convenience for passing into calibrator / export
    def enabled_targets(self) -> Sequence[str]:
        """
        實際啟用的 target 集合，直接反映 QuantRuntimeConfig.use_pct_for。
        """
        return sorted(self.quant_cfg.use_pct_for)


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


def _load_hf_token(spec: Union[str, Path]) -> Optional[str]:
    """
    Resolve HF token in a minimal, explicit way without hidden side effects.

    - If `spec` is a Path: read the file if it exists; otherwise warn and return None.
    - If `spec` is a str: treat as an environment variable name; if unset, warn and return None.
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

def run_quant_finalize(cfg: QuantFinalizeConfig) -> None:
    """
    Orchestrate:
        - model load
        - wrapping
        - percentile hi/lo apply into quantizers
        - projector rotation
        - integer export (supports W{2,4,8,16}A{2,4,8,16} via QuantRuntimeConfig)

    This is the third step of the PTQ pipeline:
        1) quant_calibrate.py   -> pct_stats_out (+ optional pct_hi_lo_out)
        2) quant_pct_apply.py   -> pct_stats_in → pct_hi_lo_out (+ summary)
        3) quant_finalize.py    -> pct_hi_lo_in → wrapped + calibrated model
                                   → projector rotation → int_export blob

    Notes:
        - 此函式不經由 draccus.wrap 包裝，適合在程式內部或 HPC inline Python
          直接呼叫（run_quant_finalize(cfg)），避免 CLI 參數解析介入。
        - CLI 入口請使用下方的 quant_finalize（有 draccus.wrap）。
    """
    set_global_seed(cfg.seed)
    device = torch.device(cfg.device)
    dtype = (
        torch.bfloat16
        if (device.type == "cuda" and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    enabled_targets = cfg.enabled_targets()
    enabled_targets_set = set(enabled_targets)

    overwatch.info(
        "[quant_finalize] Effective QuantRuntimeConfig",
        extra={
            "quant_bits": cfg.quant_bits,
            "backend": cfg.backend,
            "mode": cfg.quant_cfg.mode.value,
            "weight_bits": cfg.quant_cfg.weight_bits,
            "act_bits": cfg.quant_cfg.act_bits,
            "use_pct_for": enabled_targets,
            "use_rotation_for": sorted(cfg.quant_cfg.use_rotation_for),
            "projector_rotation_mode": cfg.quant_cfg.projector_rotation_mode.value,
        },
    )

    # ------------------------------------------------------------------
    # Model / backbone construction
    # ------------------------------------------------------------------
    model_id = cfg.model.model_id

    # Resolve HF token
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
        enable_vision_dino="vision.dino" in enabled_targets_set,
        enable_vision_siglip="vision.siglip" in enabled_targets_set,
        enable_llm="llm" in enabled_targets_set,
        enable_projector="projector" in enabled_targets_set,
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
        manifest=None,
        default_params=None,
        prefix="",
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
            "targets": enabled_targets,
        },
    )

    calibrate_model_from_hi_lo(
        vlm,
        hi_lo_map=hi_lo_map,
        act_bits=cfg.act_bits,
        signed=cfg.signed_activations,
        include_targets=enabled_targets,
    )

    # ------------------------------------------------------------------
    # Apply projector rotation (R = H K) — INT export path
    # ------------------------------------------------------------------
    if cfg.quant_cfg.should_rotate_projector():
        # 根據 QuantRuntimeConfig 的高階模式，再 AND 上 CLI 細部旗標
        effective_use_klt = (
            cfg.use_klt and cfg.quant_cfg.projector_rotation_uses_klt()
        )
        effective_use_hadamard = (
            cfg.use_hadamard and cfg.quant_cfg.projector_rotation_uses_hadamard()
        )

        proj_rot_cfg = ProjectorRotationConfig(
            use_klt=effective_use_klt,
            use_hadamard=effective_use_hadamard,
            shared_klt=cfg.shared_klt,
        )

        # 若不使用 KLT，klt_path=None，只會做 Hadamard。
        klt_path = cfg.klt_path if effective_use_klt else None

        overwatch.info(
            "[quant_finalize] Applying projector rotation",
            extra={
                "projector_rotation_mode": cfg.quant_cfg.projector_rotation_mode.value,
                "use_klt_cli": cfg.use_klt,
                "use_hadamard_cli": cfg.use_hadamard,
                "effective_use_klt": effective_use_klt,
                "effective_use_hadamard": effective_use_hadamard,
                "shared_klt": cfg.shared_klt,
                "klt_path": str(klt_path) if klt_path is not None else None,
            },
        )

        module_path, lm_head = rotate_cobra_vlm_output_projector_from_path_inplace(
            vlm,
            klt_path=klt_path,
            cfg=proj_rot_cfg,
        )

        overwatch.info(
            "[quant_finalize] Applied projector rotation",
            extra={"module_path": module_path},
        )
    else:
        overwatch.info(
            "[quant_finalize] Projector rotation disabled by QuantRuntimeConfig; skipping rotation.",
            extra={
                "mode": cfg.quant_cfg.mode.value,
                "use_rotation_for": sorted(cfg.quant_cfg.use_rotation_for),
                "projector_rotation_mode": cfg.quant_cfg.projector_rotation_mode.value,
            },
        )

    # ------------------------------------------------------------------
    # Integer export (now derived from QuantRuntimeConfig)
    # ------------------------------------------------------------------
    int_cfg = int_export_config_from_quant_cfg(
        cfg.quant_cfg,
        include_vision_dino="vision.dino" in enabled_targets_set,
        include_vision_siglip="vision.siglip" in enabled_targets_set,
        include_llm="llm" in enabled_targets_set,
        include_projector="projector" in enabled_targets_set,
    )

    overwatch.info(
        "[quant_finalize] Exporting integer quant state",
        extra={
            "int_export_config": asdict(int_cfg),
            "quant_bits": cfg.quant_bits,
            "backend": cfg.backend,
            "quant_use_pct_for": enabled_targets,
        },
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

@draccus.wrap()
def quant_finalize(cfg: QuantFinalizeConfig) -> None:
    """
    CLI wrapper around `run_quant_finalize`.

    用法：
        python -m cobra.switches.quant_finalize --quant_bits W4A4 --backend int ...

    說明：
        - 由 draccus.wrap() 負責解析 CLI 參數並建立 QuantFinalizeConfig。
        - 實際核心流程委派給 run_quant_finalize(cfg)，
          以便在程式內部（例如 HPC inline Python）可以直接呼叫
          run_quant_finalize(cfg) 而不經過 draccus。
    """
    run_quant_finalize(cfg)


if __name__ == "__main__":
    quant_finalize()

