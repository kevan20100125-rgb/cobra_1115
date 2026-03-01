#!/usr/bin/env bash
#
# Submit with:
#   sbatch scripts/slurm_cfzo_grid.sh
#
# Runs CF-ZO calibration grid over perturb_radius / step_size / steps.
# Outputs calibration checkpoints + a CSV summary under outputs/cfzo_grid.
#
#SBATCH -J cobra_offcal_grid
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH -p dev
#SBATCH -A MST114205
#SBATCH -o outputs/slurm/%x.%j.out
#SBATCH -e outputs/slurm/%x.%j.err

set -eo pipefail
IFS=$'\n\t'

# ---------------- Env / modules ----------------
module load miniconda3 2>/dev/null || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /work/hsuan1007/miniconda3/envs/cobra

export MKL_INTERFACE_LAYER=LP64
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

# ---------------- Paths ----------------
export REPO_ROOT=${REPO_ROOT:-/work/hsuan1007/cobra_1230/cobra_1115}
export WORKDIR=${WORKDIR:-${REPO_ROOT}/scripts}
export HF_HOME=${HF_HOME:-"${REPO_ROOT}/.hf_cache"}
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-"${REPO_ROOT}/.hf_token"}

mkdir -p "${HF_HOME}" "${REPO_ROOT}/outputs/slurm"

cd "${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# ---------------- Static args ----------------
MODEL_ID=${MODEL_ID:-"cobra+3b"}
OUT_DIR=${OUT_DIR:-"${REPO_ROOT}/outputs/cfzo_grid"}
OUT_BASENAME=${OUT_BASENAME:-"textvqa_calibration"}  # 可自訂檔名前綴
SUBSET_SIZE=${SUBSET_SIZE:-256}
BATCH_SIZE=${BATCH_SIZE:-4}
NUM_WORKERS=${NUM_WORKERS:-2}
SEED=${SEED:-7}
DEVICE=${DEVICE:-""}

mkdir -p "${OUT_DIR}"
cd "${WORKDIR}"

# ---------------- 校準超參數（可改） ----------------
# 想掃多組就直接寫：PERTURB_LIST=(0.01 0.05) 等
PERTURB_LIST=(${PERTURB_LIST:-0.1})
STEP_SIZE_LIST=(${STEP_SIZE_LIST:-0.01 0.02})
STEPS_LIST=(${STEPS_LIST:-1000})
LOSS=${LOSS:-ppl}                 # ppl (cross-entropy) 或 entropy
DATASET_TYPE=${DATASET_TYPE:-llava-v15}
LOG_FREQ=${LOG_FREQ:-10}

RESULT_CSV="${OUT_DIR}/grid_results.csv"
echo "perturb_radius,step_size,steps,loss_plus,loss_minus" > "${RESULT_CSV}"

# ---------------- Run combinations ----------------
for pr in "${PERTURB_LIST[@]}"; do
  for ss in "${STEP_SIZE_LIST[@]}"; do
    for st in "${STEPS_LIST[@]}"; do
      echo "[INFO] Running CF-ZO calibration with perturb_radius=${pr}, step_size=${ss}, steps=${st}"

      JOB_DIR="${OUT_DIR}/pr_${pr}_ss_${ss}_st_${st}"
      mkdir -p "${JOB_DIR}"
      CAL_PT="${JOB_DIR}/${OUT_BASENAME}_pr_${pr}_ss_${ss}_st_${st}.pt"
      LOG_FILE="${JOB_DIR}/log.txt"

      ARGS=(
        scripts/offline_calibration.py
        --hf_token "${HF_TOKEN_PATH}"
        --subset_size "${SUBSET_SIZE}"
        --batch_size "${BATCH_SIZE}"
        --num_workers "${NUM_WORKERS}"
        --seed "${SEED}"
        --dataset.type "${DATASET_TYPE}"
        --dataset_stage finetune
        --output_path "${CAL_PT}"
        --cfzo.steps "${st}"
        --cfzo.perturb_radius "${pr}"
        --cfzo.step_size "${ss}"
        --cfzo.loss_type "${LOSS}"
        --cfzo.log_frequency "${LOG_FREQ}"
      )
      if [[ -n "${DEVICE}" ]]; then
        ARGS+=(--device "${DEVICE}")
      fi

      # Run calibration
      set -x
      srun -u python "${ARGS[@]}" >"${LOG_FILE}" 2>&1
      set +x

      # Extract final loss values if available
      FINAL_LINE=$(grep -E "\[CF-ZO\].*loss" "${LOG_FILE}" | tail -1 || true)
      LOSS_PLUS="NaN"
      LOSS_MINUS="NaN"
      if [[ -n "${FINAL_LINE}" ]]; then
        LOSS_PLUS=${FINAL_LINE#*loss+=}
        LOSS_PLUS=${LOSS_PLUS%% *}
        LOSS_MINUS=${FINAL_LINE#*loss-=}
        LOSS_MINUS=${LOSS_MINUS%% *}
      fi
      echo "${pr},${ss},${st},${LOSS_PLUS},${LOSS_MINUS}" >> "${RESULT_CSV}"

      echo "[DONE] Saved: ${CAL_PT}  (loss+=${LOSS_PLUS} loss-=${LOSS_MINUS})"
    done
  done
done

echo "[SUMMARY] All combinations complete. Results stored in ${RESULT_CSV}"
