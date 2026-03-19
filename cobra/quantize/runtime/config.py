from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Iterable, Optional, Set, Tuple

from cobra.quantize.targets import CANONICAL_TARGETS, normalize_targets

_SUPPORTED_BITS = (2, 4, 8, 16)

@unique
class QuantMode(Enum):
    FLOAT = "float"
    FAKE = "fake"


@unique
class ProjectorRotationMode(Enum):
    HK = "hk"
    HADAMARD = "hadamard"
    NONE = "none"


@dataclass(frozen=True)
class ModelIdResolution:
    raw_model_id: str
    base_model_id: str
    backend: str
    bits_hint: Optional[str]


@dataclass(frozen=True)
class RuntimeRequestResolution:
    raw_model_id: str
    base_model_id: str
    backend: str
    bits: Optional[str]
    bits_hint: Optional[str]


def normalize_bits_spec(bits: Optional[str]) -> Optional[str]:
    if bits is None:
        return None
    s = str(bits).strip().upper()
    return s or None


def parse_flexible_bits_spec(bits: Optional[str], *, strict: bool = False) -> Tuple[Optional[int], Optional[int]]:
    """
    Supported:
      - None
      - W8 / W4 / W2 / W16
      - W8A8 / W4A8 / W2A2 / ...
      - A8 / A16
    """
    import re

    s = normalize_bits_spec(bits)
    if s is None:
        return None, None

    m = re.fullmatch(r"W(\d+)A(\d+)", s)
    if m is not None:
        w_bits = int(m.group(1))
        a_bits = int(m.group(2))
        if w_bits in _SUPPORTED_BITS and a_bits in _SUPPORTED_BITS:
            return w_bits, a_bits
        if strict:
            raise ValueError(
                f"[parse_flexible_bits_spec] Unsupported bits spec {bits!r}; "
                f"supported bitwidths are {_SUPPORTED_BITS}."
            )
        return None, None

    m = re.fullmatch(r"W(\d+)", s)
    if m is not None:
        w_bits = int(m.group(1))
        if w_bits in _SUPPORTED_BITS:
            return w_bits, None
        if strict:
            raise ValueError(
                f"[parse_flexible_bits_spec] Unsupported weight bits {bits!r}; "
                f"supported bitwidths are {_SUPPORTED_BITS}."
            )
        return None, None

    m = re.fullmatch(r"A(\d+)", s)
    if m is not None:
        a_bits = int(m.group(1))
        if a_bits in _SUPPORTED_BITS:
            return None, a_bits
        if strict:
            raise ValueError(
                f"[parse_flexible_bits_spec] Unsupported activation bits {bits!r}; "
                f"supported bitwidths are {_SUPPORTED_BITS}."
            )
        return None, None

    if strict:
        raise ValueError(
            f"[parse_flexible_bits_spec] Invalid bits spec {bits!r}; "
            "expected one of: W8, W4A8, A8, etc."
        )
    return None, None


def resolve_model_id_base_backend(model_id: str) -> ModelIdResolution:
    """
    Examples:
      cobra+3b
      cobra+3b-ptq-w8-fake
      cobra+3b-ptq-w8a8-fake
      cobra+3b-ptq-a8-fake
    """
    raw = str(model_id).strip()
    lower = raw.lower()

    backend = "float"
    base_for_bits = raw

    if lower.endswith("-fake"):
        backend = "fake"
        base_for_bits = raw[:-5]

    lower2 = base_for_bits.lower()
    if "-ptq-" not in lower2:
        return ModelIdResolution(
            raw_model_id=raw,
            base_model_id=base_for_bits,
            backend=backend,
            bits_hint=None,
        )

    idx = lower2.index("-ptq-")
    base_id = base_for_bits[:idx]
    bits_hint = normalize_bits_spec(base_for_bits[idx + len("-ptq-") :])

    return ModelIdResolution(
        raw_model_id=raw,
        base_model_id=base_id,
        backend=backend,
        bits_hint=bits_hint,
    )


def resolve_runtime_request(
    *,
    raw_model_id: str,
    env_bits: Optional[str] = None,
    env_backend: Optional[str] = None,
) -> RuntimeRequestResolution:
    """
    Precedence:
      backend: env BACKEND > model_id suffix > float
      bits:    env BITS > model_id -ptq-<bits> suffix > None
    """
    model_res = resolve_model_id_base_backend(raw_model_id)

    backend = (env_backend or "").strip().lower()
    if backend not in ("float", "fake"):
        backend = model_res.backend

    bits = normalize_bits_spec(env_bits)
    if bits is None:
        bits = model_res.bits_hint

    return RuntimeRequestResolution(
        raw_model_id=model_res.raw_model_id,
        base_model_id=model_res.base_model_id,
        backend=backend,
        bits=bits,
        bits_hint=model_res.bits_hint,
    )


@dataclass
class QuantRuntimeConfig:
    """
    Single source of truth for resolved quant runtime configuration.

    Notes:
      - requested_* means the user actually asked for that quant branch.
      - weight_bits / act_bits remain "effective" bit placeholders for compatibility.
        If a branch is not requested, the effective value is 16 (float-like).
    """

    bits: Optional[str]
    requested_weight_bits: Optional[int]
    requested_act_bits: Optional[int]

    weight_bits: int
    act_bits: int

    mode: QuantMode

    use_pct_for: Set[str] = field(default_factory=set)
    use_rotation_for: Set[str] = field(default_factory=set)

    projector_rotation_mode: ProjectorRotationMode = ProjectorRotationMode.HK

    vision_in_pct_pipeline: bool = True
    enable_act_quant: bool = False
    symmetric_acts: bool = True
    symmetric_weights: bool = True

    config_name: Optional[str] = None
    backend: Optional[str] = None

    @staticmethod
    def _normalize_targets(targets: Optional[Iterable[str]]) -> Set[str]:
        try:
            return set(normalize_targets(targets))
        except KeyError as e:
            raise KeyError(
                f"[QuantRuntimeConfig] Invalid canonical target in {targets!r}. "
                f"Expected one of {CANONICAL_TARGETS}."
            ) from e

    @staticmethod
    def _parse_projector_rotation_mode(mode: Optional[str]) -> ProjectorRotationMode:
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
            "expected one of ['hk', 'hadamard', 'none']."
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
        strict_bits: bool = True,
    ) -> "QuantRuntimeConfig":
        backend_norm = (backend or "float").strip().lower()
        if backend_norm not in ("float", "fake"):
            raise ValueError(
                f"[QuantRuntimeConfig] Unsupported backend={backend!r}; "
                "expected 'float' or 'fake'."
            )

        bits_norm = normalize_bits_spec(bits)
        req_w_bits, req_a_bits = parse_flexible_bits_spec(bits_norm, strict=strict_bits)
        proj_rot_mode_enum = cls._parse_projector_rotation_mode(projector_rotation_mode)

        mode = QuantMode.FLOAT if backend_norm == "float" else QuantMode.FAKE

        effective_w_bits = req_w_bits if req_w_bits is not None else 16
        effective_a_bits = req_a_bits if req_a_bits is not None else 16

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

        use_rotation_for: Set[str] = set()
        if (
            mode is not QuantMode.FLOAT
            and enable_projector
            and proj_rot_mode_enum is not ProjectorRotationMode.NONE
        ):
            use_rotation_for.add("projector")

        return cls(
            bits=bits_norm,
            requested_weight_bits=req_w_bits,
            requested_act_bits=req_a_bits,
            weight_bits=effective_w_bits,
            act_bits=effective_a_bits,
            mode=mode,
            use_pct_for=use_pct_for,
            use_rotation_for=use_rotation_for,
            projector_rotation_mode=proj_rot_mode_enum,
            vision_in_pct_pipeline=vision_in_pct_pipeline,
            enable_act_quant=enable_act_quant,
            symmetric_acts=symmetric_acts,
            symmetric_weights=symmetric_weights,
            config_name=config_name,
            backend=backend_norm,
        )

    def enabled_targets(self) -> Tuple[str, ...]:
        return tuple(sorted(self.use_pct_for))

    def should_quantize_target(self, target: str) -> bool:
        return target in self.use_pct_for

    def should_rotate_projector(self) -> bool:
        if self.mode is QuantMode.FLOAT:
            return False
        return (
            "projector" in self.use_rotation_for
            and self.projector_rotation_mode is not ProjectorRotationMode.NONE
        )

    def should_apply_weight_quant(self) -> bool:
        return (self.mode is not QuantMode.FLOAT) and (self.requested_weight_bits is not None)

    def should_calibrate_activations(self) -> bool:
        return (
            (self.mode is not QuantMode.FLOAT)
            and bool(self.enable_act_quant)
            and (self.requested_act_bits is not None)
        )

    def projector_rotation_uses_klt(self) -> bool:
        if not self.should_rotate_projector():
            return False
        return self.projector_rotation_mode is ProjectorRotationMode.HK

    def projector_rotation_uses_hadamard(self) -> bool:
        if not self.should_rotate_projector():
            return False
        return self.projector_rotation_mode in (
            ProjectorRotationMode.HK,
            ProjectorRotationMode.HADAMARD,
        )

    def requested_bits_label(self) -> str:
        if self.requested_weight_bits is not None and self.requested_act_bits is not None:
            return f"W{self.requested_weight_bits}A{self.requested_act_bits}"
        if self.requested_weight_bits is not None:
            return f"W{self.requested_weight_bits}"
        if self.requested_act_bits is not None:
            return f"A{self.requested_act_bits}"
        return "FLOAT"

    def effective_bits_label(self, *, act_calib_enabled: Optional[bool] = None) -> str:
        parts = []
        if self.requested_weight_bits is not None:
            parts.append(f"W{self.requested_weight_bits}")
        if self.requested_act_bits is not None:
            if act_calib_enabled is None:
                parts.append(f"A{self.requested_act_bits}")
            elif act_calib_enabled:
                parts.append(f"A{self.requested_act_bits}")
            else:
                parts.append(f"A{self.requested_act_bits}(OFF)")
        return "".join(parts) if parts else "FLOAT"