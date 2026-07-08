#!/bin/bash
# Usage: bash eval.sh <ckpt_dir_name>
# Example: bash /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/eval/eval.sh MM-Bench-CN_stage1
#
# Checkpoint -> Benchmark mapping:
#   MM-Bench-CN_stage1   -> MMBench_DEV_CN_V11
#   MM-Bench-CN_stage2   -> MMBench_DEV_CN_V11
#   MM-Bench-EN_stage1   -> MMBench_DEV_EN_V11
#   MM-Bench-EN_stage2   -> MMBench_DEV_EN_V11
#   mmstar-stage1        -> MMStar
#   mmstar-stage2        -> MMStar
#   realwordQA-stage1    -> RealWorldQA
#   realwordQA-stage2    -> RealWorldQA

VLMEVAL="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/VLMEvalKit/run.py"
WORK_DIR="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/eval_results"

export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
export OPENAI_API_BASE="https://runway.devops.xiaohongshu.com/openai/qwen/v1/chat/completions"
export no_proxy="127.0.0.1,localhost"

declare -A BENCH_MAP=(
  ["MM-Bench-CN_stage1"]="MMBench_DEV_CN_V11"
  ["MM-Bench-CN_stage2"]="MMBench_DEV_CN_V11"
  ["MM-Bench-EN_stage1"]="MMBench_DEV_EN_V11"
  ["MM-Bench-EN_stage2"]="MMBench_DEV_EN_V11"
  ["mmstar-stage1"]="MMStar"
  ["mmstar-stage2"]="MMStar"
  ["realwordQA-stage1"]="RealWorldQA"
  ["realwordQA-stage2"]="RealWorldQA"
)

CKPT_NAME="${1:?Usage: $0 <ckpt_dir_name>}"
BENCH="${BENCH_MAP[$CKPT_NAME]:?Unknown checkpoint: $CKPT_NAME}"
MODEL="yangfan-qwen3vl-${CKPT_NAME}"

LOG_DIR="${WORK_DIR}/yangfan-qwen3vl-${CKPT_NAME}"
mkdir -p "${LOG_DIR}"

echo "Checkpoint : ${CKPT_NAME}"
echo "Benchmark  : ${BENCH}"
echo "Model      : ${MODEL}"
echo "eval log   : ${LOG_DIR}/eval.log"
echo ""

python "${VLMEVAL}" \
  --data "${BENCH}" \
  --model "${MODEL}" \
  --work-dir "${WORK_DIR}" \
  --api-nproc 128 \
  --judge qwen3.6-plus 2>&1 | tee "${LOG_DIR}/eval.log"
