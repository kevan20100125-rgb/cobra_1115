#!/bin/bash
#SBATCH --job-name=cobra1115_ptq
#SBATCH --account=MST114205
#SBATCH --partition=dev
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -o outputs/slurm/%x_%j.out
#SBATCH -e outputs/slurm/%x_%j.err

set -euo pipefail

# ==============================
# 1. 環境設定
# ==============================
module load cuda/12.4

export ADDR2LINE=${ADDR2LINE:-$(command -v addr2line || true)}
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-6}"

set +u
source /work/asdf1234/miniconda3/etc/profile.d/conda.sh
conda activate cobra
set -u

export DATASETS_DIR=${DATASETS_DIR:-/work/asdf1234/samples}
export HF_HOME=${HF_HOME:-/work/asdf1234/.cache/huggingface}

# ==============================
# 2. 專案位置
# ==============================
PROJECT_ROOT="/work/asdf1234/cobra_1115"
cd "${PROJECT_ROOT}"

mkdir -p outputs/quantize outputs/slurm

# ==============================
# 3. 控制參數：MODE & BITS
# ==============================
MODE="${MODE:-full}"

# 允許 {2,4,8,16} 組合
BITS="${BITS:-W2A2}"

# 使用正規表示式解析 WxxAyy
if [[ "$BITS" =~ ^W([0-9]+)A([0-9]+)$ ]]; then
  W_BITS="${BASH_REMATCH[1]}"
  A_BITS="${BASH_REMATCH[2]}"
else
  echo "[ERROR] Invalid BITS format '$BITS'. Expected W{num}A{num}." >&2
  exit 1
fi

# 允許的合法 bit 值集合
VALID_BITS=("2" "4" "8" "16")

# 拒絕 1-bit
if [[ "$W_BITS" == "1" ]] || [[ "$A_BITS" == "1" ]]; then
  echo "[ERROR] 1-bit quantization is NOT supported. Got BITS='${BITS}'." >&2
  exit 1
fi

# 檢查合法 bit 值
# --- weight_bits ---
is_valid=false
for b in "${VALID_BITS[@]}"; do
  if [[ "$W_BITS" == "$b" ]]; then
    is_valid=true
    break
  fi
done

if [[ "$is_valid" != true ]]; then
  echo "[ERROR] weight_bits=$W_BITS not in {2,4,8,16}" >&2
  exit 1
fi

# --- act_bits ---
is_valid=false
for b in "${VALID_BITS[@]}"; do
  if [[ "$A_BITS" == "$b" ]]; then
    is_valid=true
    break
  fi
done

if [[ "$is_valid" != true ]]; then
  echo "[ERROR] act_bits=$A_BITS not in {2,4,8,16}" >&2
  exit 1
fi

echo "[INFO] MODE=${MODE}, BITS=${BITS}  (W_BITS=${W_BITS}, A_BITS=${A_BITS})"

# 依 bit 組合分檔
PCT_STATS="outputs/quantize/pct_stats_${BITS}.pt"
PCT_HI_LO="outputs/quantize/pct_hi_lo_${BITS}.pt"
PCT_SUMMARY="outputs/quantize/pct_calibrate_summary_${BITS}.json"
INT_EXPORT="outputs/quantize/int_export_${BITS}.pt"

export BITS W_BITS A_BITS
export PCT_STATS PCT_HI_LO PCT_SUMMARY INT_EXPORT

# ==============================
# 4. Calibration（quant_calibrate）
# ==============================
if [[ "${MODE}" == "calibrate" || "${MODE}" == "full" ]]; then
  echo "[STEP] Running cobra_1115 quant_calibrate ..."

  python - << 'PY'
import os
from pathlib import Path

from cobra.switches.quant_calibrate import QuantCalibrateConfig, quant_calibrate
from cobra.conf.datasets import DatasetConfig, DatasetRegistry

BITS = os.environ.get("BITS", "W8A8")
A_BITS = int(os.environ.get("A_BITS", "8"))

pct_stats_out = Path(os.environ["PCT_STATS"])
pct_hi_lo_out = Path(os.environ["PCT_HI_LO"])
pct_summary_out = Path(os.environ["PCT_SUMMARY"])

calib_cfg_cls = DatasetConfig.get_choice_class(
    DatasetRegistry.TEXTVQA_100_CALIB.dataset_id
)
calib_dataset_cfg = calib_cfg_cls()

cfg = QuantCalibrateConfig(
    act_bits=A_BITS,
    pct_stats_out=pct_stats_out,
    pct_hi_lo_out=pct_hi_lo_out,
    pct_summary_out=pct_summary_out,
    dataset=calib_dataset_cfg,
    stage="align",

    ###Smoke test
    per_device_batch_size=2,
    num_workers=0,
    max_calib_batches=2,
    max_samples_per_module=200_000,
)

quant_calibrate(cfg)
PY

  echo "[STEP] Calibration finished."
  echo "       stats   -> ${PCT_STATS}"
  echo "       hi/lo   -> ${PCT_HI_LO}"
  echo "       summary -> ${PCT_SUMMARY}"
fi

# ==============================
# 5. Finalize（quant_finalize）
# ==============================
if [[ "${MODE}" == "finalize" || "${MODE}" == "full" ]]; then
  echo "[STEP] Running cobra_1115 quant_finalize ..."

  python - << 'PY'
import os
from pathlib import Path

from cobra.switches.quant_finalize import QuantFinalizeConfig, quant_finalize

W_BITS = int(os.environ.get("W_BITS", "8"))
A_BITS = int(os.environ.get("A_BITS", "8"))

pct_hi_lo_in = Path(os.environ["PCT_HI_LO"])
out_path = Path(os.environ["INT_EXPORT"])

cfg = QuantFinalizeConfig(
    pct_hi_lo_in=pct_hi_lo_in,
    weight_bits=W_BITS,
    act_bits=A_BITS,
    signed_weights=True,
    signed_activations=True,
    include_vision_dino=True,
    include_vision_siglip=True,
    include_llm=True,
    include_projector=True,
    use_klt=True,
    use_hadamard=True,
    shared_klt=True,
    out_path=out_path,
    device="cuda",
)

quant_finalize(cfg)
PY

  echo "[STEP] Finalize finished. Integer export -> ${INT_EXPORT}"
fi

echo "[DONE] cobra_1115 PTQ pipeline complete (MODE=${MODE}, BITS=${BITS})."

