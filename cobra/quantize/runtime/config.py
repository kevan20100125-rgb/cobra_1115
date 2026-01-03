# cobra/quantize/runtime/config.py

"""
runtime/config.py

Centralized runtime quantization configuration for Cobra PTQ.

Design goal (Phase 2+):
    - All decisions about bits / backend / targets / modes should come from
      QuantRuntimeConfig, and only from here.
    - The following entrypoints must treat QuantRuntimeConfig as the single
      source of truth:
          * switches/quant_calibrate.py
          * switches/quant_finalize.py
          * quantize/runtime/load_quantized_vlm.py
    - Scripts and CLIs should NOT re-parse bits or re-decide target sets
      on their own; instead they should:
          1) build QuantRuntimeConfig via from_bits_backend(...)
          2) read weight_bits / act_bits / mode / use_pct_for / use_rotation_for
             / projector_rotation_mode from quant_cfg
          3) pass those settings into wrap / calibrator / rotate / export.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Optional, Iterable, Set, Tuple


_CANONICAL_TARGETS = ("vision.dino", "vision.siglip", "llm", "projector")
_SUPPORTED_BITS = (2, 4, 8, 16)


@unique
class QuantMode(Enum):
    FLOAT = "float"         # 完全不量化（baseline）
    FAKE = "fake"           # fake quant（MambaQuant-style，仍用 float kernel）


@unique
class ProjectorRotationMode(Enum):
    """
    控制 LLM output projector（lm_head）的旋轉模式。

    - HK       : KLT + Hadamard
    - HADAMARD : 只有 Hadamard
    - NONE     : 完全不旋轉（即使 use_rotation_for 包含 projector）
    """
    HK = "hk"
    HADAMARD = "hadamard"
    NONE = "none"


@dataclass
class QuantRuntimeConfig:
    """
    集中管理 Cobra PTQ runtime 相關設定的物件。

    設計目標：
        - bits/backend/vision_in_pct_pipeline 等不要散落在各個 script。
        - 任何需要量化設定的地方（wrap/pct/finalize/runtime）統一吃這個 config。
    """

    # 使用者視角
    bits: Optional[str]                      # 原始字串："W4A4" / "W8A8" / None (表示 float)
    weight_bits: int                         # 內部解析後的 W bits
    act_bits: int                            # 內部解析後的 A bits
    mode: QuantMode                          # FLOAT / FAKE

    # 哪些 target 進 percentile / quant 流程
    use_pct_for: Set[str] = field(default_factory=set)
    # 哪些 target 允許「旋轉」；目前實務上只支援 "projector"
    use_rotation_for: Set[str] = field(default_factory=set)

    # projector rotation 模式（HK / HADAMARD / NONE）
    projector_rotation_mode: ProjectorRotationMode = ProjectorRotationMode.HK

    # 額外旗標
    vision_in_pct_pipeline: bool = True
    enable_act_quant: bool = False
    symmetric_acts: bool = True
    symmetric_weights: bool = True

    # 方便 logging / debug
    config_name: Optional[str] = None

    # 原始 backend 字串，可選（"fake"/"int"/"float"）
    backend: Optional[str] = None

    # ------------------------------------------------------------------
    # 建構 helper
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_bits(bits: Optional[str]) -> Tuple[int, int]:
        """
        將 "W8A4" 等字串轉成 (w_bits, a_bits)。

        若 bits 為 None，代表 float 路徑（W16A16 當作預設）。
        """
        if bits is None:
            # 對 float 路徑統一視作 W16A16，方便下游簡單判斷
            return 16, 16

        s = bits.strip()
        # 支援大小寫混用
        # 形式：W8A8 / w4a8 等
        import re

        m = re.fullmatch(r"[Ww](\d+)[Aa](\d+)", s)
        if m is None:
            raise ValueError(
                f"[QuantRuntimeConfig] Invalid bits spec {bits!r}; "
                f"expected like 'W8A8', 'W4A4'."
            )

        w_bits = int(m.group(1))
        a_bits = int(m.group(2))

        if w_bits not in _SUPPORTED_BITS or a_bits not in _SUPPORTED_BITS:
            raise ValueError(
                f"[QuantRuntimeConfig] Unsupported bitwidth combo W{w_bits}A{a_bits}; "
                f"supported bitwidths are {_SUPPORTED_BITS} for both W and A."
            )
        return w_bits, a_bits

    @staticmethod
    def _normalize_targets(targets: Optional[Iterable[str]]) -> Set[str]:
        if not targets:
            return set()
        out: Set[str] = set()
        for t in targets:
            t = (t or "").strip()
            if not t:
                continue
            if t not in _CANONICAL_TARGETS:
                raise KeyError(
                    f"[QuantRuntimeConfig] Unknown canonical target {t!r}. "
                    f"Expected one of {_CANONICAL_TARGETS}."
                )
            out.add(t)
        return out

    @staticmethod
    def _parse_projector_rotation_mode(
        mode: Optional[str],
    ) -> ProjectorRotationMode:
        """
        將使用者提供的 rotation mode 字串（或 None）轉成 enum。

        合法字串：
            - "hk", "klt+hadamard", "klt_hadamard"
            - "hadamard", "h"
            - "none", "off", "disable", "disabled"

        若為 None，視為 "hk"。
        """
        if mode is None:
            return ProjectorRotationMode.HK

        raw = mode.strip().lower()
        if raw in ("hk", "klt+hadamard", "klt_hadamard"):
            return ProjectorRotationMode.HK
        if raw in ("hadamard", "h"):
            return ProjectorRotationMode.HADAMARD
        if raw in ("none", "off", "disable", "disabled"):
            return ProjectorRotationMode.NONE

        raise ValueError(
            f"[QuantRuntimeConfig] Unknown projector_rotation_mode={mode!r}; "
            "expected one of ['hk', 'hadamard', 'none'] (with a few aliases)."
        )

    @classmethod
    def from_bits_backend(
        cls,
        *,
        bits: Optional[str],
        backend: Optional[str],
        enable_vision_dino: bool = True,
        enable_vision_siglip: bool = True,
        enable_llm: bool = True,
        enable_projector: bool = True,
        vision_in_pct_pipeline: bool = True,
        symmetric_acts: bool = True,
        symmetric_weights: bool = True,
        config_name: Optional[str] = None,
        projector_rotation_mode: Optional[str] = "hk",
        enable_act_quant: bool = False,
    ) -> "QuantRuntimeConfig":
        """
        Build QuantRuntimeConfig from bits/backend and switches.

        Stage 1 requirement:
            - activation must remain float
            - pct_hi_lo must NOT be required
            - calibration must be skipped

        Control knob:
            enable_act_quant:
                - False => Stage 1 W-only (skip activation calibration)
                - True  => later stages (allow calibration)
        """
        backend_norm = (backend or "float").lower()
        if backend_norm not in ("float", "fake"):
            raise ValueError(
                f"[QuantRuntimeConfig] Unsupported backend={backend!r}; "
                f"expected 'float', 'fake'."
            )

        w_bits, a_bits = cls._parse_bits(bits)
        proj_rot_mode_enum = cls._parse_projector_rotation_mode(projector_rotation_mode)

        if backend_norm == "float":
            mode = QuantMode.FLOAT
        else:
            mode = QuantMode.FAKE

        # Which targets enter quant pipeline (wrapping decision still uses this set)
        use_pct_for: Set[str] = set()
        if mode is not QuantMode.FLOAT:
            if enable_vision_dino:
                use_pct_for.add("vision.dino")
            if enable_vision_siglip:
                use_pct_for.add("vision.siglip")
            if enable_llm:
                use_pct_for.add("llm")
            if enable_projector:
                use_pct_for.add("projector")

        if not vision_in_pct_pipeline:
            use_pct_for.discard("vision.dino")
            use_pct_for.discard("vision.siglip")

        # Rotation gating (Stage 1 will set projector_rotation_mode="none" via script)
        use_rotation_for: Set[str] = set()
        if (
            mode is not QuantMode.FLOAT
            and enable_projector
            and proj_rot_mode_enum is not ProjectorRotationMode.NONE
        ):
            use_rotation_for.add("projector")

        return cls(
            bits=bits,
            weight_bits=w_bits,
            act_bits=a_bits,
            mode=mode,
            use_pct_for=use_pct_for,
            use_rotation_for=use_rotation_for,
            projector_rotation_mode=proj_rot_mode_enum,
            vision_in_pct_pipeline=vision_in_pct_pipeline,
            symmetric_acts=symmetric_acts,
            symmetric_weights=symmetric_weights,
            config_name=config_name,
            backend=backend_norm,
            enable_act_quant=enable_act_quant,
        )


    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------
    def enabled_targets(self) -> Tuple[str, ...]:
        """回傳會進 percentile / quant 流程的 target（排序後）。"""
        return tuple(sorted(self.use_pct_for))

    def should_quantize_target(self, target: str) -> bool:
        """該 target 是否會進 percentile / quant pipeline。"""
        return target in self.use_pct_for

    def should_rotate_projector(self) -> bool:
        """
        回傳是否允許對 projector（LLM output head）套用 rotation。

        條件（集中在這裡統一判斷）：
            - mode 為 FAKE 或 INT_EXPORT（非 FLOAT）
            - use_rotation_for 包含 "projector"
            - projector_rotation_mode 不是 NONE

        後續若要讓某些 bits/組合禁用 projector rotation，也應該透過
        from_bits_backend(...) 或額外 flag 來影響：
            - self.mode
            - self.use_rotation_for
            - self.projector_rotation_mode
        而不是讓下游自行再判斷。
        """
        if self.mode is QuantMode.FLOAT:
            return False

        return (
            "projector" in self.use_rotation_for
            and self.projector_rotation_mode is not ProjectorRotationMode.NONE
        )
    def should_calibrate_activations(self) -> bool:
        """
        Stage-gated activation calibration switch.

        Stage 1 (W-only baseline):
            - enable_act_quant must be False
            - calibration MUST be skipped
        """
        return (self.mode is not QuantMode.FLOAT) and bool(self.enable_act_quant)

    def projector_rotation_uses_klt(self) -> bool:
        """
        目前設定下，projector rotation 是否會使用 KLT。

        僅在「本來就會旋轉 projector」的前提下才有意義，否則一律視為 False。
        """
        if not self.should_rotate_projector():
            return False
        return self.projector_rotation_mode is ProjectorRotationMode.HK

    def projector_rotation_uses_hadamard(self) -> bool:
        """
        目前設定下，projector rotation 是否會使用 Hadamard。

        僅在「本來就會旋轉 projector」的前提下才有意義，否則一律視為 False。
        """
        if not self.should_rotate_projector():
            return False
        return self.projector_rotation_mode in (
            ProjectorRotationMode.HK,
            ProjectorRotationMode.HADAMARD,
        )



