#!/usr/bin/env bash
# Serial run of all open-source vLLM models on UNREAL_Benchmark.
# Each model resumes from results/open/eval_<name>.jsonl.
#
# Usage:
#   bash run_all_open.sh                       # 8 models, default 8 GPUs
#   GPUS=4 bash run_all_open.sh                # override TP size
#   MODELS="InternVL3_5-8B Qwen3-VL-8B-Instruct" bash run_all_open.sh

set -u
cd "$(dirname "$0")"

GPUS="${GPUS:-8}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_MEM="${GPU_MEM:-0.9}"

DEFAULT_MODELS=(
  "Qwen3-VL-4B-Instruct"
  "Qwen3-VL-8B-Instruct"
  "Qwen3-VL-32B-Instruct"
  "InternVL3_5-8B"
  "InternVL3_5-38B"
  "Qwen3.5-4B"
  "Qwen3.5-9B"
  "Qwen3.5-27B"
)

if [[ -n "${MODELS:-}" ]]; then
  # shellcheck disable=SC2206
  MODEL_LIST=($MODELS)
else
  MODEL_LIST=("${DEFAULT_MODELS[@]}")
fi

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for model in "${MODEL_LIST[@]}"; do
  log="$LOG_DIR/${model//\//_}_$(date +%Y%m%d_%H%M%S).log"
  echo "===================================================================="
  echo "[$(date +%H:%M:%S)] Running model: $model  (GPUS=$GPUS, max_len=$MAX_LEN)"
  echo "  log: $log"
  echo "===================================================================="
  python3 eval_vllm.py \
    --models "$model" \
    --vllm-gpus "$GPUS" \
    --vllm-max-model-len "$MAX_LEN" \
    --vllm-gpu-mem "$GPU_MEM" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "[!] $model exited with code $rc — continuing with next model"
  fi
done

echo
echo "All models finished. Results in: results/open/"
