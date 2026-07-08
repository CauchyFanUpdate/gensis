#!/bin/bash
# Usage: bash serve_vllm_yangfan.sh <ckpt_dir_name>
# Example: bash /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/eval/serve_vllm_yangfan.sh MM-Bench-CN_stage1
#
# The served-model-name starts with "qwen3vl-" so VLMEvalKit's
# LMDeployAPI._detect_adapter() picks up the 'qwen3' adapter automatically.
#
# Available checkpoints:
#   MM-Bench-CN_stage1   -> eval on MMBench_DEV_CN_V11
#   MM-Bench-CN_stage2   -> eval on MMBench_DEV_CN_V11
#   MM-Bench-EN_stage1   -> eval on MMBench_DEV_EN_V11
#   MM-Bench-EN_stage2   -> eval on MMBench_DEV_EN_V11
#   mmstar-stage1        -> eval on MMStar
#   mmstar-stage2        -> eval on MMStar
#   realwordQA-stage1    -> eval on RealWorldQA
#   realwordQA-stage2    -> eval on RealWorldQA
#
# After the server starts and shows "Application startup complete.",
# run the eval command printed below.

CKPT_BASE="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/Research/llamafactory-ego/output"

CKPT_NAME="${1:?Usage: $0 <ckpt_dir_name>}"
CKPT_PATH="${CKPT_BASE}/${CKPT_NAME}"
SERVED_MODEL_NAME="qwen3vl-${CKPT_NAME}"

WORK_DIR="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/eval_results"
LOG_DIR="${WORK_DIR}/yangfan-qwen3vl-${CKPT_NAME}"

if [ ! -d "${CKPT_PATH}" ]; then
  echo "ERROR: checkpoint not found: ${CKPT_PATH}"
  exit 1
fi

mkdir -p "${LOG_DIR}"
echo "vllm log: ${LOG_DIR}/vllm.log"

python -m vllm.entrypoints.cli.main serve "${CKPT_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size 1 \
  --data-parallel-size 8 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.8 \
  --max-model-len 32768 \
  --trust-remote-code \
  --host 0.0.0.0 \
  --port 9006 2>&1 | tee "${LOG_DIR}/vllm.log"

