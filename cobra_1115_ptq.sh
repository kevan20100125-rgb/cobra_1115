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

module load cuda/12.4

set +u
source /work/asdf1234/miniconda3/etc/profile.d/conda.sh
conda activate cobra
set -u

normalize_bool_01() {
  local raw="${1:-}"
  raw="$(echo "${raw}" | tr 'A-Z' 'a-z')"
  case "${raw}" in
    1|true|yes|y|on) echo "1" ;;
    0|false|no|n|off) echo "0" ;;
    *)
      echo "[ERROR] Invalid boolean value: ${1}" >&2
      exit 1
      ;;
  esac
}

export COBRA_1115_ROOT="${COBRA_1115_ROOT:-$(pwd)}"
cd "${COBRA_1115_ROOT}"
export PYTHONPATH="${COBRA_1115_ROOT}:${PYTHONPATH:-}"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}}"

MODE="${MODE:-calibrate}"          # calibrate | act_klt
BITS="${BITS:-W8A8}"
SMOKE="${SMOKE:-1}"
BACKEND="${BACKEND:-fake}"
STAGE="${STAGE:-finetune}"
HF_TOKEN_PATH="${HF_TOKEN_PATH:-.hf_token}"

export COBRA_LLM_ACT_MODE="${COBRA_LLM_ACT_MODE:-mamba_sensitive}"
LLM_ACT_MODE_TAG="$(echo "${COBRA_LLM_ACT_MODE}" | tr 'A-Z' 'a-z')"

MAMBA_SENSITIVE_IN_PROJ_RAW="${MAMBA_SENSITIVE_IN_PROJ:-0}"
MAMBA_SENSITIVE_X_PROJ_RAW="${MAMBA_SENSITIVE_X_PROJ:-0}"
MAMBA_SENSITIVE_DT_PROJ_RAW="${MAMBA_SENSITIVE_DT_PROJ:-0}"
MAMBA_SENSITIVE_OUT_PROJ_RAW="${MAMBA_SENSITIVE_OUT_PROJ:-1}"

MAMBA_SENSITIVE_IN_PROJ="$(normalize_bool_01 "${MAMBA_SENSITIVE_IN_PROJ_RAW}")"
MAMBA_SENSITIVE_X_PROJ="$(normalize_bool_01 "${MAMBA_SENSITIVE_X_PROJ_RAW}")"
MAMBA_SENSITIVE_DT_PROJ="$(normalize_bool_01 "${MAMBA_SENSITIVE_DT_PROJ_RAW}")"
MAMBA_SENSITIVE_OUT_PROJ="$(normalize_bool_01 "${MAMBA_SENSITIVE_OUT_PROJ_RAW}")"

export COBRA_LLM_MAMBA_SENSITIVE_IN_PROJ="${MAMBA_SENSITIVE_IN_PROJ}"
export COBRA_LLM_MAMBA_SENSITIVE_X_PROJ="${MAMBA_SENSITIVE_X_PROJ}"
export COBRA_LLM_MAMBA_SENSITIVE_DT_PROJ="${MAMBA_SENSITIVE_DT_PROJ}"
export COBRA_LLM_MAMBA_SENSITIVE_OUT_PROJ="${MAMBA_SENSITIVE_OUT_PROJ}"

MAMBA_SENSITIVE_PROJ_TAG="in${MAMBA_SENSITIVE_IN_PROJ}_x${MAMBA_SENSITIVE_X_PROJ}_dt${MAMBA_SENSITIVE_DT_PROJ}_out${MAMBA_SENSITIVE_OUT_PROJ}"
if [[ "${LLM_ACT_MODE_TAG}" == "mamba_sensitive" ]]; then
  LLM_ACT_CONFIG_TAG="${LLM_ACT_MODE_TAG}_${MAMBA_SENSITIVE_PROJ_TAG}"
else
  LLM_ACT_CONFIG_TAG="${LLM_ACT_MODE_TAG}"
fi

case "${BACKEND}" in
  fake) ;;
  *)
    echo "[ERROR] BACKEND must be 'fake' for cobra_1115_ptq.sh. Got: ${BACKEND}" >&2
    exit 1
    ;;
esac

case "${MODE}" in
  calibrate|act_klt) ;;
  *)
    echo "[ERROR] MODE must be 'calibrate' or 'act_klt'. Got: ${MODE}" >&2
    exit 1
    ;;
esac

if [[ ! -f "${HF_TOKEN_PATH}" ]]; then
  echo "[ERROR] HF_TOKEN_PATH not found: ${HF_TOKEN_PATH}" >&2
  exit 1
fi

mkdir -p outputs/slurm outputs/quantize

PCT_STATS_OUT="${PCT_STATS_OUT:-outputs/quantize/pct_stats_${BITS}_${LLM_ACT_CONFIG_TAG}.pt}"
PCT_HI_LO_PATH="${PCT_HI_LO_PATH:-outputs/quantize/pct_hi_lo_${BITS}_${LLM_ACT_CONFIG_TAG}.pt}"
PCT_SUMMARY_OUT="${PCT_SUMMARY_OUT:-outputs/quantize/pct_calibrate_summary_${BITS}_${LLM_ACT_CONFIG_TAG}.json}"

ACT_KLT_BLOCK_SIZE="${ACT_KLT_BLOCK_SIZE:-512}"
ACT_KLT_OUTPROJ_IN="${ACT_KLT_OUTPROJ_IN:-outputs/quantize/act_klt_outproj_in_bs${ACT_KLT_BLOCK_SIZE}/act_klt_outproj_in.pt}"
ACT_KLT_OUTPROJ_OUT="${ACT_KLT_OUTPROJ_OUT:-outputs/quantize/act_klt_outproj_out_bs${ACT_KLT_BLOCK_SIZE}/act_klt_outproj_out.pt}"
ACT_KLT_EXPORT_OUT_FEATURE="${ACT_KLT_EXPORT_OUT_FEATURE:-1}"
ACT_KLT_MAX_BATCHES="${ACT_KLT_MAX_BATCHES:-0}"
ACT_KLT_MAX_TOKENS="${ACT_KLT_MAX_TOKENS:-128}"

if [[ "${MODE}" == "act_klt" ]]; then
  export COBRA_ACT_KLT_EXPORT="${COBRA_ACT_KLT_EXPORT:-1}"
else
  export COBRA_ACT_KLT_EXPORT="${COBRA_ACT_KLT_EXPORT:-0}"
fi

export MODE BITS SMOKE BACKEND STAGE HF_TOKEN_PATH
export PCT_STATS_OUT PCT_HI_LO_PATH PCT_SUMMARY_OUT
export ACT_KLT_OUTPROJ_IN ACT_KLT_OUTPROJ_OUT
export ACT_KLT_EXPORT_OUT_FEATURE
export ACT_KLT_BLOCK_SIZE ACT_KLT_MAX_BATCHES ACT_KLT_MAX_TOKENS
export LLM_ACT_MODE_TAG LLM_ACT_CONFIG_TAG
export MAMBA_SENSITIVE_IN_PROJ MAMBA_SENSITIVE_X_PROJ
export MAMBA_SENSITIVE_DT_PROJ MAMBA_SENSITIVE_OUT_PROJ
export MAMBA_SENSITIVE_PROJ_TAG

echo "[INFO] COBRA_1115_ROOT=${COBRA_1115_ROOT}"
echo "[INFO] MODE=${MODE}, BITS=${BITS}, BACKEND=${BACKEND}, SMOKE=${SMOKE}, STAGE=${STAGE}"
echo "[INFO] COBRA_LLM_ACT_MODE=${COBRA_LLM_ACT_MODE}"
echo "[INFO] LLM_ACT_MODE_TAG=${LLM_ACT_MODE_TAG}"
echo "[INFO] LLM_ACT_CONFIG_TAG=${LLM_ACT_CONFIG_TAG}"
echo "[INFO] COBRA_LLM_MAMBA_SENSITIVE_IN_PROJ=${COBRA_LLM_MAMBA_SENSITIVE_IN_PROJ}"
echo "[INFO] COBRA_LLM_MAMBA_SENSITIVE_X_PROJ=${COBRA_LLM_MAMBA_SENSITIVE_X_PROJ}"
echo "[INFO] COBRA_LLM_MAMBA_SENSITIVE_DT_PROJ=${COBRA_LLM_MAMBA_SENSITIVE_DT_PROJ}"
echo "[INFO] COBRA_LLM_MAMBA_SENSITIVE_OUT_PROJ=${COBRA_LLM_MAMBA_SENSITIVE_OUT_PROJ}"
echo "[INFO] MAMBA_SENSITIVE_PROJ_TAG=${MAMBA_SENSITIVE_PROJ_TAG}"
echo "[INFO] PCT_STATS_OUT=${PCT_STATS_OUT}"
echo "[INFO] PCT_HI_LO_PATH=${PCT_HI_LO_PATH}"
echo "[INFO] PCT_SUMMARY_OUT=${PCT_SUMMARY_OUT}"
echo "[INFO] ACT_KLT_OUTPROJ_IN=${ACT_KLT_OUTPROJ_IN}"
echo "[INFO] ACT_KLT_OUTPROJ_OUT=${ACT_KLT_OUTPROJ_OUT}"
echo "[INFO] ACT_KLT_BLOCK_SIZE=${ACT_KLT_BLOCK_SIZE}"
echo "[INFO] ACT_KLT_MAX_BATCHES=${ACT_KLT_MAX_BATCHES}"
echo "[INFO] ACT_KLT_MAX_TOKENS=${ACT_KLT_MAX_TOKENS}"
echo "[INFO] HF_TOKEN_PATH=${HF_TOKEN_PATH}"
echo "[INFO] COBRA_ACT_KLT_EXPORT=${COBRA_ACT_KLT_EXPORT}"

if [[ "${MODE}" == "calibrate" ]]; then
  python - <<'PY'
import os
from pathlib import Path

from cobra.conf.datasets import DatasetConfig, DatasetRegistry
from cobra.switches.quant_calibrate import QuantCalibrateConfig, quant_calibrate

bits = os.environ.get("BITS", "W8A8")
backend = os.environ.get("BACKEND", "fake")
smoke = int(os.environ.get("SMOKE", "1"))
llm_act_mode = os.environ.get("COBRA_LLM_ACT_MODE", "default")
llm_act_config_tag = os.environ.get("LLM_ACT_CONFIG_TAG", llm_act_mode)

calib_cfg_cls = DatasetConfig.get_choice_class(
    DatasetRegistry.TEXTVQA_100_CALIB.dataset_id
)
calib_dataset_cfg = calib_cfg_cls()

base_cfg_kwargs = dict(
    quant_bits=bits,
    backend=backend,
    dataset=calib_dataset_cfg,
    pct_stats_out=Path(os.environ["PCT_STATS_OUT"]),
    pct_hi_lo_out=Path(os.environ["PCT_HI_LO_PATH"]),
    pct_summary_out=Path(os.environ["PCT_SUMMARY_OUT"]),
)

if smoke == 1:
    cfg = QuantCalibrateConfig(
        **base_cfg_kwargs,
        per_device_batch_size=2,
        num_workers=0,
        max_calib_batches=2,
        max_samples_per_module=200_000,
    )
else:
    cfg = QuantCalibrateConfig(**base_cfg_kwargs)

print(
    f"[QuantCalibrate] bits={cfg.quant_bits} backend={cfg.backend} "
    f"act_bits={cfg.act_bits} llm_act_mode={llm_act_mode} "
    f"llm_act_config_tag={llm_act_config_tag} "
    f"pct_hi_lo_out={cfg.pct_hi_lo_out}"
)
quant_calibrate(cfg)
PY
fi

if [[ "${MODE}" == "act_klt" ]]; then
  python - <<'PY'
import os
from pathlib import Path

from cobra.switches.quant_act_klt_outproj import (
    QuantActKLTOutProjConfig,
    quant_act_klt_outproj,
)

cfg = QuantActKLTOutProjConfig(
    stage=os.environ.get("STAGE", "finetune"),
    hf_token=Path(os.environ.get("HF_TOKEN_PATH", ".hf_token")),
    act_klt_in_out=Path(os.environ.get("ACT_KLT_OUTPROJ_IN")),
    act_klt_out_out=Path(os.environ.get("ACT_KLT_OUTPROJ_OUT")),
    export_out_feature=(int(os.environ.get("ACT_KLT_EXPORT_OUT_FEATURE", "1")) != 0),
    block_size=int(os.environ.get("ACT_KLT_BLOCK_SIZE", "512")),
    max_calib_batches=int(os.environ.get("ACT_KLT_MAX_BATCHES", "0")),
    max_tokens_per_sample=int(os.environ.get("ACT_KLT_MAX_TOKENS", "128")),
)

print(
    f"[QuantActKLTOutProj] block_size={cfg.block_size} "
    f"in={cfg.act_klt_in_out} out={cfg.act_klt_out_out}"
)
quant_act_klt_outproj(cfg)
PY
fi

echo "[DONE] cobra_1115_ptq.sh finished."
