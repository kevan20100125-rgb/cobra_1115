from __future__ import annotations

from typing import Optional

from cobra.quantize.runtime.loader_core import load_quantized_cobra_vlm_impl


def load_quantized_cobra_vlm(
    *,
    bits: Optional[str],
    pct_hi_lo_path,
    hf_token: str,
    base_dtype,
    device,
    enabled_targets=None,
    run_dir=None,
    output_dir=None,
    model_id_or_path: Optional[str] = None,
    backend: str = "fake",
):
    """
    Public compatibility facade for the Cobra PTQ runtime loader.

    Phase 4 keeps this import path stable while moving implementation details
    into cobra.quantize.runtime.loader_core.
    """
    return load_quantized_cobra_vlm_impl(
        bits=bits,
        pct_hi_lo_path=pct_hi_lo_path,
        hf_token=hf_token,
        base_dtype=base_dtype,
        device=device,
        enabled_targets=enabled_targets,
        run_dir=run_dir,
        output_dir=output_dir,
        model_id_or_path=model_id_or_path,
        backend=backend,
    )