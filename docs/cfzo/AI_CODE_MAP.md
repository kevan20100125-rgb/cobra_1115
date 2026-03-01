# CF-ZO Code Map

This note explains where CF-ZO logic lives in `cobra_1115`.

## Core files
- `cobra/models/mamba/mamba_cfzo.py`
  - CF-ZO Mamba block implementation.
  - Adds per-layer `mamba_scale` and activation capture helpers.

- `cobra/models/mamba/modeling_mamba.py`
  - Integrates CF-ZO Mamba into HF/Cobra Mamba model path.
  - Runtime switch: `COBRA_USE_CFZO_MAMBA=1`.

- `cobra/models/backbones/llm/mamba.py`
  - Backbone-level wrappers and embed dimension compatibility.

## Checkpoint compatibility
- `cobra/models/vlms/cobra.py`
  - Allows missing CF-ZO-only keys (`*.mixer.mamba_scale`) when loading legacy checkpoints.

## Optimizer/calibration
- `cobra/training/optimizer/cf_zo.py`
  - CF-ZO calibration implementation (`CFZOParams`, `calibrate_mamba_scales`).
- `cobra/training/optimizer/__init__.py`
  - Public optimizer export entry.

## Entrypoints
- `scripts/offline_calibration.py`
  - Offline CF-ZO calibration script.
- `scripts/slurm_cfzo_grid.sh`
  - Slurm sweep launcher for CF-ZO hyperparameters.

## Minimal enable steps
```bash
export COBRA_USE_CFZO_MAMBA=1
python scripts/offline_calibration.py ...
```
