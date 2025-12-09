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
# 3. 控制參數：MODE / BITS / SMOKE / ROTATION_MODE
# ==============================
# MODE:
#   calibrate  -> 只跑 quant_calibrate（產生 pct_stats_*, pct_hi_lo_*）
#   finalize   -> 只跑 quant_finalize（產生 int_export_*）
#   full       -> 先 calibrate 再 finalize
MODE="${MODE:-calibrate}"

# BITS: W{2,4,8,16}A{2,4,8,16}，例如 W8A8 / W4A4
#   目前 fake quant accuracy study 主力會用 W8A8 / W4A4
BITS="${BITS:-W16A16}"

# SMOKE:
#   0 -> 正式校正（使用 QuantCalibrateConfig 的預設為主）
#   1 -> smoke test，只跑極少 batch 做流程驗證
SMOKE="${SMOKE:-1}"

# ROTATION_MODE:
#   hk       -> KLT + Hadamard
#   hadamard -> 只有 Hadamard
#   none     -> 完全不旋轉
ROTATION_MODE="${ROTATION_MODE:-hk}"

# 基本合法值檢查；不合法時 fallback 為 hk
case "${ROTATION_MODE}" in
  hk|hadamard|none)
    ;;
  *)
    echo "[WARN] Unknown ROTATION_MODE='${ROTATION_MODE}', falling back to 'hk'." >&2
    ROTATION_MODE="hk"
    ;;
esac

# 使用正規表示式解析 WxxAyy
if [[ "$BITS" =~ ^W([0-9]+)A([0-9]+)$ ]]; then
  W_BITS="${BASH_REMATCH[1]}"
  A_BITS="${BASH_REMATCH[2]}"
else
  echo "[ERROR] Invalid BITS format '$BITS'. Expected W{num}A{num}." >&2
  exit 1
fi

VALID_BITS=("2" "4" "8" "16")

# 拒絕 1-bit
if [[ "$W_BITS" == "1" ]] || [[ "$A_BITS" == "1" ]]; then
  echo "[ERROR] 1-bit quantization is NOT supported. Got BITS='${BITS}'." >&2
  exit 1
fi

# --- 檢查 weight_bits ---
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

# --- 檢查 act_bits ---
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

echo "[INFO] MODE=${MODE}, BITS=${BITS}  (W_BITS=${W_BITS}, A_BITS=${A_BITS}, SMOKE=${SMOKE}, ROTATION_MODE=${ROTATION_MODE})"

# 依 bit 組合分檔（fake quant / int_export 都共用這套命名）
PCT_STATS="outputs/quantize/pct_stats_${BITS}.pt"
PCT_HI_LO="outputs/quantize/pct_hi_lo_${BITS}.pt"
PCT_SUMMARY="outputs/quantize/pct_calibrate_summary_${BITS}.json"
INT_EXPORT="outputs/quantize/int_export_${BITS}.pt"

export BITS W_BITS A_BITS
export PCT_STATS PCT_HI_LO PCT_SUMMARY INT_EXPORT SMOKE
# 讓後續 Python / runtime 都看到 ROTATION_MODE，並且把它同步到
# load_quantized_vlm.py 用的 COBRA_PROJECTOR_ROTATION_MODE。
export ROTATION_MODE
export COBRA_PROJECTOR_ROTATION_MODE="${ROTATION_MODE}"


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
SMOKE = int(os.environ.get("SMOKE", "0"))

pct_stats_out = Path(os.environ["PCT_STATS"])
pct_hi_lo_out = Path(os.environ["PCT_HI_LO"])
pct_summary_out = Path(os.environ["PCT_SUMMARY"])

# 使用 TEXTVQA_100_CALIB 作為預設 calibration dataset
calib_cfg_cls = DatasetConfig.get_choice_class(
    DatasetRegistry.TEXTVQA_100_CALIB.dataset_id
)
calib_dataset_cfg = calib_cfg_cls()

# 基本設定：bit / dataset / 輸出路徑
base_cfg_kwargs = dict(
    act_bits=A_BITS,
    pct_stats_out=pct_stats_out,
    pct_hi_lo_out=pct_hi_lo_out,
    pct_summary_out=pct_summary_out,
    dataset=calib_dataset_cfg,
    stage="align",
    enable_vision_dino=True,
    enable_vision_siglip=True,
    enable_llm=True,
    enable_projector=True,
    vision_in_pct_pipeline=True,
)

if SMOKE == 1:
    # 極小預算：只驗證 pipeline 是否能跑通
    cfg = QuantCalibrateConfig(
        **base_cfg_kwargs,
        per_device_batch_size=2,
        num_workers=0,
        max_calib_batches=2,
        max_samples_per_module=200_000,
    )
else:
    # 正式校正：採用 QuantCalibrateConfig 預設為主（可視需要再微調）
    cfg = QuantCalibrateConfig(
        **base_cfg_kwargs,
        # 你可以視需要在這裡覆寫 per_device_batch_size / num_workers / max_calib_batches
        # 不寫就用 QuantCalibrateConfig 的 defaults：
        # per_device_batch_size=8, num_workers=4, max_calib_batches=0 ...
    )

print(f"[QuantCalibrate] Running with BITS={BITS}, act_bits={cfg.act_bits}, smoke={SMOKE}")
quant_calibrate(cfg)
PY

  echo "[STEP] Calibration finished."
  echo "       stats   -> ${PCT_STATS}"
  echo "       hi/lo   -> ${PCT_HI_LO}"
  echo "       summary -> ${PCT_SUMMARY}"
fi

# ==============================
# 5. Finalize（quant_finalize，預留未來 INT 用）
# ==============================
if [[ "${MODE}" == "finalize" || "${MODE}" == "full" ]]; then
  echo "[STEP] Running cobra_1115 quant_finalize ..."

  python - << 'PY'
import os
from pathlib import Path

from cobra.switches.quant_finalize import QuantFinalizeConfig, run_quant_finalize

W_BITS = int(os.environ.get("W_BITS", "8"))
A_BITS = int(os.environ.get("A_BITS", "8"))
ROTATION_MODE = os.environ.get("ROTATION_MODE", "hk")

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
    # projector rotation 設定：
    #   - 高階模式由 QuantRuntimeConfig 解析 ROTATION_MODE（hk/hadamard/none）
    #   - 細部旗標預設全開，交由 QuantRuntimeConfig 決定實際是否使用 KLT/Hadamard
    projector_rotation_mode=ROTATION_MODE,
    use_klt=True,
    use_hadamard=True,
    shared_klt=True,
    out_path=out_path,
    device="cuda",
)

print(
    f"[QuantFinalize] Running with W_BITS={W_BITS}, A_BITS={A_BITS}, "
    f"projector_rotation_mode={ROTATION_MODE}"
)
run_quant_finalize(cfg)
PY

  echo "[STEP] Finalize finished. Integer export -> ${INT_EXPORT}"
fi

echo "[DONE] cobra_1115 PTQ pipeline complete (MODE=${MODE}, BITS=${BITS}, SMOKE=${SMOKE})."


