#!/bin/bash
#SBATCH --job-name=cobra_outproj_dim
#SBATCH --account=MST114205
#SBATCH --partition=normal2
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

set -euo pipefail

# ============================================================
# 0) Runtime environment
# ============================================================
module load cuda/12.4

set +u
source /work/asdf1234/miniconda3/etc/profile.d/conda.sh
conda activate cobra
set -u

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

# ============================================================
# 1) Project roots
# ============================================================
export COBRA_1115_ROOT="/work/asdf1234/cobra_1115"
export VLM_EVAL_ROOT="/work/asdf1234/vlm-evaluation"

# nounset-safe PYTHONPATH chaining
export PYTHONPATH="${COBRA_1115_ROOT}:${PYTHONPATH:-}"

# ============================================================
# 2) Match eval-like env (Stage-1 fake-quant)
# ============================================================
export BACKEND="fake"
export BITS="W8"
export PTQ_STAGE="2"

# Base model id (this v2 loader infers from env)
export COBRA_MODEL_BASE_ID="cobra+3b"

# Explicitly disable rotation for a clean dimension check
unset COBRA_LLM_MIXER_HADAMARD || true
unset COBRA_LLM_MIXER_ACT_KLT || true
unset ACT_KLT_OUT || true

# HF token (same as your eval)
export HF_TOKEN_PATH="${VLM_EVAL_ROOT}/.hf_token"

# pct_hi_lo_path:
# - Loader signature requires it; many codepaths skip it in Stage-1 fake-quant.
# - If your loader enforces existence, replace with a real file path.
export PCT_HI_LO_PATH=""

# ============================================================
# 3) Run a small python probe
# ============================================================
python - <<'PY'
import os
import torch
import inspect

from cobra.quantize.runtime.load_quantized_vlm import load_quantized_cobra_vlm

bits = os.environ.get("BITS", "W8")
hf_token = os.environ["HF_TOKEN_PATH"]
pct_hi_lo_path = os.environ.get("PCT_HI_LO_PATH", "")
run_dir = os.environ.get("COBRA_1115_ROOT") or None

print("[Probe] env:",
      "BACKEND=", os.environ.get("BACKEND"),
      "BITS=", bits,
      "PTQ_STAGE=", os.environ.get("PTQ_STAGE"),
      "COBRA_MODEL_BASE_ID=", os.environ.get("COBRA_MODEL_BASE_ID"),
      "run_dir=", run_dir)

vlm = load_quantized_cobra_vlm(
    bits=bits,
    pct_hi_lo_path=pct_hi_lo_path,
    hf_token=hf_token,
    base_dtype=torch.float16,
    device="cuda",
    run_dir=run_dir,
)

# ---- 1) quick introspection: what does CobraVLM expose? ----
top_attrs = [a for a in dir(vlm) if not a.startswith("_")]
print("[Probe] CobraVLM type:", type(vlm))
print("[Probe] CobraVLM attrs (sample):", ", ".join(top_attrs[:40]), " ...")

# ---- 2) Directly locate all mixer.out_proj modules without needing vlm.llm ----
hits = []
for name, mod in vlm.named_modules():
    if name.endswith("mixer.out_proj") or (".mixer." in name and name.endswith("out_proj")):
        w = getattr(mod, "weight", None)
        wshape = tuple(w.shape) if w is not None else None
        in_f = getattr(mod, "in_features", None)
        out_f = getattr(mod, "out_features", None)
        hits.append((name, in_f, out_f, wshape, type(mod).__name__))

print(f"[Probe] Found {len(hits)} candidate out_proj modules:")
for (name, in_f, out_f, wshape, cls) in hits[:200]:
    print(f"  {name:90s}  {cls:18s}  in/out={in_f}/{out_f}  w={wshape}")

if not hits:
    raise RuntimeError("No mixer.out_proj modules found via named_modules().")

# Optional: show a few top-level children modules to reveal naming
print("[Probe] Top-level children:")
for n, c in list(vlm.named_children())[:30]:
    print(" ", n, "->", type(c))
PY

