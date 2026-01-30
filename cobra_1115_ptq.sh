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

# conda activate under nounset-safe wrapper
set +u
source /work/asdf1234/miniconda3/etc/profile.d/conda.sh
conda activate cobra
set -u

# ==============================
# 2. 專案位址 / PYTHONPATH
# ==============================
# 預設以當前工作目錄為 cobra_1115 根目錄；可由外部覆寫
export COBRA_1115_ROOT="${COBRA_1115_ROOT:-$(pwd)}"
cd "${COBRA_1115_ROOT}"

export PYTHONPATH="${COBRA_1115_ROOT}:${PYTHONPATH:-}"

# Cache 位置（可視環境調整）
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"

# ==============================
# 3. 入口參數
# ==============================
# MODE:
#   klt        -> 只跑 quant_klt（產生 shared_klt.pt）
#   calibrate  -> 只跑 quant_calibrate（產生 pct_stats / pct_hi_lo / summary）
#   full       -> calibrate + klt
#   act_klt    -> 只跑 quant_act_klt_outproj（產生 layer-wise act-KLT for out_proj）
MODE="${MODE:-act_klt}"

# BITS: 例如 W8A8 / W4A4 / W16A16（quant_calibrate 用）
BITS="${BITS:-W8A8}"

# SMOKE:
#   1 -> smoke test（極少 batch 驗證流程）
#   0 -> 正式校正（使用 QuantCalibrateConfig 的預設為主）
SMOKE="${SMOKE:-0}"

# quant_calibrate 的 backend：你目前 repo 只保留 float/fake runtime；校正應走 fake
BACKEND="${BACKEND:-fake}"
case "${BACKEND}" in
  fake) ;;
  *)
    echo "[ERROR] BACKEND must be 'fake' for quant_calibrate. Got: ${BACKEND}"
    exit 1
    ;;
esac

# stage（QuantKLTConfig / QuantActKLTOutProjConfig）
STAGE="${STAGE:-finetune}"

# HF token（quant_klt / act_klt 會用到）
HF_TOKEN_PATH="${HF_TOKEN_PATH:-${COBRA_1115_ROOT}/.hf_token}"

mkdir -p outputs/slurm outputs/quantize

# ==============================
# 4. 輸出路徑
# ==============================
PCT_STATS="${PCT_STATS:-outputs/quantize/pct_stats_${BITS}.pt}"
PCT_HI_LO="${PCT_HI_LO:-outputs/quantize/pct_hi_lo_${BITS}.pt}"
PCT_SUMMARY="${PCT_SUMMARY:-outputs/quantize/pct_calibrate_summary_${BITS}.json}"

# shared KLT
KLT_OUT="${KLT_OUT:-outputs/quantize/shared_klt.pt}"

# act-KLT (out_proj, layer-wise)
ACT_KLT_OUT_IN="${ACT_KLT_OUT_IN:-outputs/quantize/act_klt_outproj_in_bs512/act_klt_outproj_in.pt}"
ACT_KLT_OUT_OUT="${ACT_KLT_OUT_OUT:-outputs/quantize/act_klt_outproj_out_bs512/act_klt_outproj_out.pt}"
ACT_KLT_EXPORT_OUT_FEATURE="${ACT_KLT_EXPORT_OUT_FEATURE:-1}"   # 1=同時輸出 out-feature K

ACT_KLT_BLOCK_SIZE="${ACT_KLT_BLOCK_SIZE:-512}"
ACT_KLT_MAX_BATCHES="${ACT_KLT_MAX_BATCHES:-0}"
ACT_KLT_MAX_TOKENS="${ACT_KLT_MAX_TOKENS:-128}"
COBRA_ACT_KLT_EXPORT="${COBRA_ACT_KLT_EXPORT:-1}"

# ==============================
# 5. Export env vars
# ==============================
export MODE
export BITS
export SMOKE
export BACKEND
export STAGE
export HF_TOKEN_PATH
export PCT_STATS
export PCT_HI_LO
export PCT_SUMMARY
export KLT_OUT

export ACT_KLT_OUT_IN
export ACT_KLT_OUT_OUT
export ACT_KLT_EXPORT_OUT_FEATURE

export ACT_KLT_BLOCK_SIZE
export ACT_KLT_MAX_BATCHES
export ACT_KLT_MAX_TOKENS

# IMPORTANT:
# Mamba fast path can bypass out_proj module calls, preventing forward_pre_hook collection.
# For MODE=act_klt, force slow path via COBRA_ACT_KLT_EXPORT=1 unless user explicitly overrides.
if [[ "${MODE}" == "act_klt" ]]; then
  export COBRA_ACT_KLT_EXPORT="${COBRA_ACT_KLT_EXPORT:-1}"
else
  export COBRA_ACT_KLT_EXPORT="${COBRA_ACT_KLT_EXPORT:-0}"
fi

# ==============================
# 6. Print
# ==============================
echo "[INFO] COBRA_1115_ROOT=${COBRA_1115_ROOT}"
echo "[INFO] MODE=${MODE}, BITS=${BITS}, BACKEND=${BACKEND}, SMOKE=${SMOKE}, STAGE=${STAGE}"
echo "[INFO] PCT_STATS=${PCT_STATS}"
echo "[INFO] PCT_HI_LO=${PCT_HI_LO}"
echo "[INFO] PCT_SUMMARY=${PCT_SUMMARY}"
echo "[INFO] KLT_OUT=${KLT_OUT}"
echo "[INFO] ACT_KLT_OUT_IN=${ACT_KLT_OUT_IN}"
echo "[INFO] ACT_KLT_OUT_OUT=${ACT_KLT_OUT_OUT}"
echo "[INFO] ACT_KLT_BLOCK_SIZE=${ACT_KLT_BLOCK_SIZE}, ACT_KLT_MAX_BATCHES=${ACT_KLT_MAX_BATCHES}, ACT_KLT_MAX_TOKENS=${ACT_KLT_MAX_TOKENS}"
echo "[INFO] HF_TOKEN_PATH=${HF_TOKEN_PATH}"
echo "[INFO] COBRA_ACT_KLT_EXPORT=${COBRA_ACT_KLT_EXPORT}"

# ==============================
# 7. KLT（quant_klt）
# ==============================
if [[ "${MODE}" == "klt" || "${MODE}" == "full" ]]; then
  echo "[STEP] Running cobra_1115 quant_klt ..."

  python - << '__PY__'
import os
from pathlib import Path

from cobra.switches.quant_klt import QuantKLTConfig, quant_klt

STAGE = os.environ.get("STAGE", "finetune")
HF_TOKEN_PATH = Path(os.environ.get("HF_TOKEN_PATH", ".hf_token"))
KLT_OUT = Path(os.environ.get("KLT_OUT", "outputs/quantize/shared_klt.pt"))

cfg = QuantKLTConfig(
    stage=STAGE,
    hf_token=HF_TOKEN_PATH,
    klt_out=KLT_OUT,
)

print(
    f"[QuantKLT] Running with stage={cfg.stage}, device={cfg.device}, "
    f"klt_out={cfg.klt_out}, hf_token={cfg.hf_token}"
)
quant_klt(cfg)
__PY__

  echo "[STEP] KLT finished. Saved -> ${KLT_OUT}"
fi

# ==============================
# 8. Calibration（quant_calibrate）
# ==============================
if [[ "${MODE}" == "calibrate" || "${MODE}" == "full" ]]; then
  echo "[STEP] Running cobra_1115 quant_calibrate ..."

  python - << '__PY__'
import os
from pathlib import Path

from cobra.switches.quant_calibrate import QuantCalibrateConfig, quant_calibrate
from cobra.conf.datasets import DatasetConfig, DatasetRegistry

BITS = os.environ.get("BITS", "W8A8")
BACKEND = os.environ.get("BACKEND", "fake")
SMOKE = int(os.environ.get("SMOKE", "0"))

pct_stats_out = Path(os.environ["PCT_STATS"])
pct_hi_lo_out = Path(os.environ["PCT_HI_LO"])
pct_summary_out = Path(os.environ["PCT_SUMMARY"])

# 預設 calibration dataset：TEXTVQA_100_CALIB
calib_cfg_cls = DatasetConfig.get_choice_class(
    DatasetRegistry.TEXTVQA_100_CALIB.dataset_id
)
calib_dataset_cfg = calib_cfg_cls()

base_cfg_kwargs = dict(
    quant_bits=BITS,
    backend=BACKEND,
    dataset=calib_dataset_cfg,
    pct_stats_out=pct_stats_out,
    pct_hi_lo_out=pct_hi_lo_out,
    pct_summary_out=pct_summary_out,
)

if SMOKE == 1:
    cfg = QuantCalibrateConfig(
        **base_cfg_kwargs,
        per_device_batch_size=2,
        num_workers=0,
        max_calib_batches=2,
        max_samples_per_module=200_000,
    )
else:
    cfg = QuantCalibrateConfig(
        **base_cfg_kwargs,
        # 正式校正：其餘超參數用 QuantCalibrateConfig 預設
    )

print(
    f"[QuantCalibrate] Running with quant_bits={cfg.quant_bits}, backend={cfg.backend}, "
    f"resolved act_bits={cfg.act_bits}, smoke={SMOKE}, "
    f"pct_stats_out={cfg.pct_stats_out}, pct_hi_lo_out={cfg.pct_hi_lo_out}"
)
quant_calibrate(cfg)
__PY__

  echo "[STEP] Calibration finished."
fi

# ==============================
# 9. act-KLT（quant_act_klt_outproj）
# ==============================
if [[ "${MODE}" == "act_klt" ]]; then
  echo "[STEP] Running cobra_1115 quant_act_klt_outproj (layer-wise) ..."

  python - << '__PY__'
import os
from pathlib import Path

from cobra.switches.quant_act_klt_outproj import QuantActKLTOutProjConfig, quant_act_klt_outproj

STAGE = os.environ.get("STAGE", "finetune")
HF_TOKEN_PATH = Path(os.environ.get("HF_TOKEN_PATH", ".hf_token"))

ACT_KLT_OUT_IN = Path(os.environ.get("ACT_KLT_OUT_IN", "outputs/quantize/act_klt_outproj_in_bs512/act_klt_outproj_in.pt"))
ACT_KLT_OUT_OUT = Path(os.environ.get("ACT_KLT_OUT_OUT", "outputs/quantize/act_klt_outproj_out_bs512/act_klt_outproj_out.pt"))
EXPORT_OUT = int(os.environ.get("ACT_KLT_EXPORT_OUT_FEATURE", "1")) != 0

BLOCK_SIZE = int(os.environ.get("ACT_KLT_BLOCK_SIZE", "512"))
MAX_BATCHES = int(os.environ.get("ACT_KLT_MAX_BATCHES", "0"))
MAX_TOKENS = int(os.environ.get("ACT_KLT_MAX_TOKENS", "128"))

cfg = QuantActKLTOutProjConfig(
    stage=STAGE,
    hf_token=HF_TOKEN_PATH,
    device="cuda",
    block_size=BLOCK_SIZE,
    max_calib_batches=MAX_BATCHES,
    max_tokens_per_sample=MAX_TOKENS,
    act_klt_in_out=ACT_KLT_OUT_IN,
    export_out_feature=EXPORT_OUT,
    act_klt_out_out=ACT_KLT_OUT_OUT,
)

print(
    f"[QuantActKLTOutProj] Running with stage={cfg.stage}, device={cfg.device}, dataset={cfg.dataset.dataset_id}, "
    f"block_size={cfg.block_size}, max_batches={cfg.max_calib_batches}, max_tokens={cfg.max_tokens_per_sample}, "
    f"act_klt_in_out={cfg.act_klt_in_out}, act_klt_out_out={cfg.act_klt_out_out}"
)
quant_act_klt_outproj(cfg)
__PY__

  echo "[STEP] act-KLT finished. Saved -> ${ACT_KLT_OUT_IN} , ${ACT_KLT_OUT_OUT}"
fi

echo "[DONE] cobra_1115 PTQ script complete (MODE=${MODE}, BITS=${BITS}, SMOKE=${SMOKE})."

