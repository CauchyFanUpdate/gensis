#!/bin/bash
# SAPO training with LLM judge reward manager for Qwen3.5-4B
# Smoothed adaptive preference optimization: soft tau-based PPO
# Usage: bash run_sapo_qwen35_llm_judge.sh
# Environment variables: MODEL_PATH, TRAIN_FILE, VAL_FILE, JUDGE_MODEL_PATH, N_GPUS, ROLLOUT_TP, etc.

set -x
set -o pipefail

# ── Configuration ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT=${VERL_ROOT:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/naive_cross_vlm/workspace/verl"}
export VERL_ROOT

MODEL_PATH=${MODEL_PATH:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3.5-4B"}
export MODEL_PATH

TRAIN_FILE=${TRAIN_FILE:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/dataset/charxiv_babyvision_mmmu_zerobench.parquet"}
export TRAIN_FILE

VAL_FILE=${VAL_FILE:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/dataset/charxiv_babyvision.parquet"}
export VAL_FILE

# Judge server (Qwen3-32B)
JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3-32B"}
JUDGE_PORT=${JUDGE_PORT:-51002}
JUDGE_TP=${JUDGE_TP:-4}
JUDGE_GPU_MEMORY_UTIL=${JUDGE_GPU_MEMORY_UTIL:-0.35}
START_JUDGE_SERVER=${START_JUDGE_SERVER:-1}

# Training config
DATE_TIME="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="qwen35_charxiv_babyvision_mmmu_zerobench_sapo_rubric_${DATE_TIME}"
export EXPERIMENT_NAME
CKPT_ROOT="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/checkpoints"
CKPTS_DIR="${CKPT_ROOT}/${EXPERIMENT_NAME}"
export CKPTS_DIR

N_GPUS=${N_GPUS:-8}
export N_GPUS

ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_TP

# Judge environment
export JUDGE_BASE_URL="http://localhost:${JUDGE_PORT}/v1"
export JUDGE_API_KEY="qwen"
export JUDGE_MODEL_NAME="qwen"
export JUDGE_PROMPT_PATH="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/llm_judge_prompt.txt"

# ── Cleanup stale Ray / vLLM processes from previous runs ─────────────────────
pkill -TERM -f "verl.trainer.main_ppo" 2>/dev/null || true
pkill -TERM -f "sglang.srt"            2>/dev/null || true
pkill -TERM -f "raylet"                2>/dev/null || true
pkill -TERM -f "gcs_server"            2>/dev/null || true
sleep 15

ray stop --force 2>/dev/null || true
pkill -9 -f "verl.trainer.main_ppo" 2>/dev/null || true
pkill -9 -f "vllm.entrypoints"      2>/dev/null || true
pkill -9 -f "VLLM::Worker"          2>/dev/null || true
pkill -9 -f "sglang.srt"            2>/dev/null || true
pkill -9 -f "raylet"                2>/dev/null || true
pkill -9 -f "gcs_server"            2>/dev/null || true
pkill -9 -f "ray::"                 2>/dev/null || true
rm -rf /tmp/ray
sleep 5

# ── Start LLM judge server ─────────────────────────────────────────
if [ "$START_JUDGE_SERVER" -eq 1 ]; then
  echo "=========================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting vLLM judge server"
  echo "  Model: $JUDGE_MODEL_PATH"
  echo "  Port: $JUDGE_PORT"
  echo "  TP: $JUDGE_TP"
  echo "=========================================="

  pkill -f "vllm serve.*--port $JUDGE_PORT" 2>/dev/null || true
  sleep 2

  NCCL_CUMEM_ENABLE=0 vllm serve "$JUDGE_MODEL_PATH" \
    --port "$JUDGE_PORT" \
    --gpu-memory-utilization "0.35" \
    --tensor-parallel-size "$JUDGE_TP" \
    --max-model-len 12288 \
    --enable-sleep-mode \
    --enforce-eager \
    --dtype bfloat16 \
    --api-key "qwen" \
    --served-model-name "qwen" \
    > /tmp/vllm_judge_${JUDGE_PORT}.log 2>&1 &

  JUDGE_PID=$!
  echo "[INFO] vLLM judge server started (PID: $JUDGE_PID)"
  echo "[INFO] Log file: /tmp/vllm_judge_${JUDGE_PORT}.log"

  echo "[INFO] Waiting for judge server to be ready..."
  JUDGE_READY=0
  for i in {1..60}; do
    if curl -s "http://127.0.0.1:$JUDGE_PORT/v1/models" >/dev/null 2>&1; then
      JUDGE_READY=1
      echo "[INFO] vLLM judge server is ready!"
      break
    fi
    echo "  Attempt $i/60..."
    sleep 60
  done

  if [ "$JUDGE_READY" -ne 1 ]; then
    echo "[ERROR] vLLM judge server did not start in time"
    echo "[ERROR] Check log: /tmp/vllm_judge_${JUDGE_PORT}.log"
    exit 1
  fi

  echo "[INFO] Putting judge server to sleep (release GPUs)..."
  curl -s "http://127.0.0.1:$JUDGE_PORT/sleep?level=1" >/dev/null 2>&1 || true
  sleep 2

  for i in {1..30}; do
    status=$(curl -s "http://127.0.0.1:$JUDGE_PORT/is_sleeping" 2>/dev/null | grep -o "true\|false" || echo "unknown")
    if [ "$status" = "true" ]; then
      echo "[INFO] Judge server is now sleeping (GPUs released)"
      break
    fi
    if [ $i -eq 30 ]; then
      echo "[WARN] Judge server sleep status uncertain, continuing anyway..."
    fi
    sleep 1
  done
fi

# ── Prepare directories ────────────────────────────────────────────
mkdir -p "$CKPTS_DIR"
LOG_FILE="$CKPTS_DIR/train.log"

# ── Environment ────────────────────────────────────────────────────
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_MODE="online"
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export RAY_raylet_start_wait_time_s=120
export RAY_num_server_call_thread=4
export RAY_core_worker_num_server_call_thread=4
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.9;9.0+PTX}"

# ── Launch training ────────────────────────────────────────────────
echo "[INFO] Starting SAPO training with LLM judge..."
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Checkpoint dir: $CKPTS_DIR"
echo "[INFO] Log file: $LOG_FILE"
cd "$VERL_ROOT"

python3 -u -m verl.trainer.main_ppo \
    --config-path="$SCRIPT_DIR/../config" \
    --config-name=sapo_qwen35_4b_llm_judge \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    reward.reward_kwargs.llm_judge_kwargs.prompt_path=/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/llm_judge_prompt.txt \
    reward.reward_kwargs.llm_judge_kwargs.rubric_prompt_path=/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/rubric_prompt.txt \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
echo "[INFO] Training completed with exit code: $EXIT_CODE"

# ── Convert checkpoint to HuggingFace format ──────────────────────
if [ "$EXIT_CODE" -eq 0 ]; then
  echo "[INFO] Converting FSDP checkpoint to HuggingFace format..."
  bash "${SCRIPT_DIR}/../convert/convert_fsdp_to_hf.sh" "$CKPTS_DIR" \
    || echo "[WARN] Checkpoint conversion failed (non-fatal)"
fi

# ── Cleanup judge server ───────────────────────────────────────────
if [ "$START_JUDGE_SERVER" -eq 1 ]; then
  echo "[INFO] Stopping judge server (port $JUDGE_PORT)..."
  pkill -f "vllm serve.*--port $JUDGE_PORT" 2>/dev/null || true
  sleep 2
fi

exit $EXIT_CODE
