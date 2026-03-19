from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Set, Tuple


TARGET_VISION_DINO = "vision.dino"
TARGET_VISION_SIGLIP = "vision.siglip"
TARGET_LLM = "llm"
TARGET_PROJECTOR = "projector"

CANONICAL_TARGETS: Tuple[str, ...] = (
    TARGET_VISION_DINO,
    TARGET_VISION_SIGLIP,
    TARGET_LLM,
    TARGET_PROJECTOR,
)

CANONICAL_TARGET_SET: Set[str] = set(CANONICAL_TARGETS)

LEGACY_TARGET_ALIASES = {
    "vision_backbone.dino": TARGET_VISION_DINO,
    "vision.dinov2": TARGET_VISION_DINO,
    "vision_backbone.featurizer": TARGET_VISION_DINO,

    "vision_backbone.siglip": TARGET_VISION_SIGLIP,
    "vision.siglip_vit": TARGET_VISION_SIGLIP,
    "vision_backbone.siglip_featurizer": TARGET_VISION_SIGLIP,

    "llm_backbone": TARGET_LLM,
    "lm_backbone": TARGET_LLM,
    "language": TARGET_LLM,

    "projector.out": TARGET_PROJECTOR,
    "proj.out": TARGET_PROJECTOR,
    "encoder.out": TARGET_PROJECTOR,
}


def is_canonical_target(name: str) -> bool:
    return (name or "").strip() in CANONICAL_TARGET_SET


def normalize_target(name: str) -> str:
    raw = (name or "").strip()
    if not raw:
        raise KeyError("Empty target name is not allowed.")

    if "::" in raw:
        raw = raw.split("::", 1)[-1].strip()

    raw_no_space = raw.replace(" ", "")
    lowered = raw_no_space.lower()

    if raw_no_space in CANONICAL_TARGET_SET:
        return raw_no_space

    if raw_no_space in LEGACY_TARGET_ALIASES:
        return LEGACY_TARGET_ALIASES[raw_no_space]

    if lowered in LEGACY_TARGET_ALIASES:
        return LEGACY_TARGET_ALIASES[lowered]

    if "dino" in lowered:
        return TARGET_VISION_DINO
    if "siglip" in lowered:
        return TARGET_VISION_SIGLIP
    if "projector" in lowered or lowered.endswith(".out"):
        return TARGET_PROJECTOR
    if "llm" in lowered or "gpt" in lowered or "language" in lowered:
        return TARGET_LLM

    raise KeyError(
        f"Unrecognized target name: {name!r}. "
        f"Expected one of {CANONICAL_TARGETS} or a supported alias."
    )


def normalize_targets(targets: Optional[Iterable[str]]) -> Set[str]:
    if not targets:
        return set()
    return {normalize_target(t) for t in targets if (t or "").strip()}


def validate_targets(targets: Sequence[str]) -> Tuple[str, ...]:
    out = []
    for t in targets:
        nt = normalize_target(t)
        out.append(nt)
    return tuple(out)


def infer_target_from_module_path(module_path: str) -> Optional[str]:
    module_path = (module_path or "").strip()
    if not module_path:
        return None

    if module_path.startswith("vision_backbone.dino_featurizer"):
        return TARGET_VISION_DINO

    if module_path.startswith("vision_backbone.featurizer"):
        return TARGET_VISION_DINO

    if module_path.startswith("vision_backbone.siglip_featurizer"):
        return TARGET_VISION_SIGLIP

    if module_path.startswith("llm_backbone.llm"):
        return TARGET_LLM

    if module_path.startswith("projector"):
        return TARGET_PROJECTOR

    return None


@dataclass(frozen=True)
class TargetFlags:
    enable_vision_dino: bool = True
    enable_vision_siglip: bool = True
    enable_llm: bool = True
    enable_projector: bool = True

    def is_enabled(self, target: str) -> bool:
        t = normalize_target(target)
        if t == TARGET_VISION_DINO:
            return self.enable_vision_dino
        if t == TARGET_VISION_SIGLIP:
            return self.enable_vision_siglip
        if t == TARGET_LLM:
            return self.enable_llm
        if t == TARGET_PROJECTOR:
            return self.enable_projector
        return False

    def enabled_targets(self) -> Tuple[str, ...]:
        out = []
        for t in CANONICAL_TARGETS:
            if self.is_enabled(t):
                out.append(t)
        return tuple(out)