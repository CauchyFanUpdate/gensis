#!/bin/bash
# 启动远端 vLLM judge server (Qwen3-32B)
# 用法: bash start_judge_server.sh

set -x

MODEL_PATH=${MODEL_PATH:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3-32B"}
PORT=${PORT:-31001}
TP=${TP:-8}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-12288}

echo "Starting vLLM judge server"
echo "  Model: $MODEL_PATH"
echo "  Port:  $PORT"
echo "  TP:    $TP"
echo "  GPU memory util: $GPU_MEMORY_UTIL"

LOG_DIR="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/checkpoints"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/vllm_judge_${PORT}.log"
echo "Log: $LOG_FILE"

NCCL_CUMEM_ENABLE=0 vllm serve "$MODEL_PATH" \
    --port "$PORT" \
    --tensor-parallel-size "$TP" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    --enforce-eager \
    --dtype bfloat16 \
    --api-key "qwen" \
    --served-model-name "qwen" \
    2>&1 | tee "$LOG_FILE"
