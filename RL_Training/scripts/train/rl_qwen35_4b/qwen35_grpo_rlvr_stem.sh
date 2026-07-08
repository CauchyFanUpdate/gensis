#!/bin/bash
# GRPO training for Qwen3.5-4B (multimodal VLM), single node 8 GPU.
# Based on verl examples/grpo_trainer/run_qwen3_5_27b_vllm_fsdp.sh
#
# Prerequisites:
#   conda activate rl
#   python workspace/grpo_datagen/prepare_parquet.py   # generate train/val.parquet
#
# Override any variable on the command line:
#   MODEL_PATH=... N_GPUS=4 bash run_grpo_qwen35_4b.sh
set -x
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ── Cleanup stale Ray / vLLM processes from previous runs ─────────────────────
# Phase 1: SIGTERM only to parent daemons (raylet/gcs_server).
# They will wait() their own worker children before exiting → no new zombies.
# Do NOT send SIGTERM directly to workers here — if workers die before raylet
# calls wait(), they become orphans adopted by PID 1 (sleep 365d, never reaps).
pkill -TERM -f "verl.trainer.main_ppo" 2>/dev/null || true
pkill -TERM -f "sglang.srt"            2>/dev/null || true
pkill -TERM -f "raylet"                2>/dev/null || true
pkill -TERM -f "gcs_server"            2>/dev/null || true
sleep 30  # give raylet/gcs_server time to wait() their children before escalating

# Phase 2: force-kill anything still alive (workers should already be reaped)
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

# ── WandB ─────────────────────────────────────────────────────────────────────
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_MODE="online"

# ── Ray stability ──────────────────────────────────────────────────────────────
export NO_PROXY="localhost,127.0.0.1"
export no_proxy="localhost,127.0.0.1"
export RAY_raylet_start_wait_time_s=120
export RAY_num_server_call_thread=4
export RAY_core_worker_num_server_call_thread=4
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.9;9.0+PTX}"

# ── HF datasets cache (makes overlong-prompt filter cacheable across runs) ─────
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/.hf_cache}"

# ── Paths ──────────────────────────────────────────────────────────────────────
VERL_ROOT=${VERL_ROOT:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/naive_cross_vlm/workspace/verl"}

MODEL_PATH=${MODEL_PATH:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3.5-4B"}
export MODEL_PATH

TRAIN_FILE=${TRAIN_FILE:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/dataset/parquets/stem.parquet"}
export TRAIN_FILE

VAL_FILE=${VAL_FILE:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/dataset/parquets/stem.parquet"}
export VAL_FILE

EXPERIMENT_NAME="qwen35_grpo_rlvr_stem"
export EXPERIMENT_NAME
CKPT_ROOT="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/checkpoints/qwen35_4b"
CKPTS_DIR=${CKPTS_DIR:-"${CKPT_ROOT}/${EXPERIMENT_NAME}"}
export CKPTS_DIR

PROJECT_NAME=${PROJECT_NAME:-"zerovision"}

N_GPUS=${N_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ENGINE=${1:-vllm}

# Judge server (Qwen3-32B)
JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3-32B"}
JUDGE_PORT=${JUDGE_PORT:-51002}
JUDGE_TP=${JUDGE_TP:-4}
JUDGE_GPU_MEMORY_UTIL=${JUDGE_GPU_MEMORY_UTIL:-0.35}
START_JUDGE_SERVER=${START_JUDGE_SERVER:-1}

# ── Log file ───────────────────────────────────────────────────────────────────
mkdir -p "${CKPTS_DIR}"
LOG_FILE="${CKPTS_DIR}/train.log"
echo "Logging to: ${LOG_FILE}"

# ── Start LLM judge server ─────────────────────────────────────────
if [ "$START_JUDGE_SERVER" -eq 1 ]; then
  echo "=========================================="
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting vLLM judge server"
  echo "  Model: $JUDGE_MODEL_PATH"
  echo "  Port:  $JUDGE_PORT"
  echo "  TP:    $JUDGE_TP"
  echo "  Util:  $JUDGE_GPU_MEMORY_UTIL"
  echo "=========================================="

  pkill -f "vllm serve.*--port $JUDGE_PORT" 2>/dev/null || true
  sleep 2

  NCCL_CUMEM_ENABLE=0 vllm serve "$JUDGE_MODEL_PATH" \
    --port "$JUDGE_PORT" \
    --gpu-memory-utilization "$JUDGE_GPU_MEMORY_UTIL" \
    --tensor-parallel-size "$JUDGE_TP" \
    --max-model-len 12288 \
    --enable-sleep-mode \
    --enforce-eager \
    --dtype bfloat16 \
    --api-key "qwen" \
    --served-model-name "qwen" \
    > "${CKPTS_DIR}/vllm_judge_${JUDGE_PORT}.log" 2>&1 &

  JUDGE_PID=$!
  echo "[INFO] vLLM judge server started (PID: $JUDGE_PID)"
  echo "[INFO] Log: ${CKPTS_DIR}/vllm_judge_${JUDGE_PORT}.log"

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
    echo "[ERROR] Check log: ${CKPTS_DIR}/vllm_judge_${JUDGE_PORT}.log"
    exit 1
  fi

  curl -s "http://127.0.0.1:$JUDGE_PORT/sleep?level=1" >/dev/null 2>&1 || true
  sleep 2
  for i in {1..30}; do
    status=$(curl -s "http://127.0.0.1:$JUDGE_PORT/is_sleeping" 2>/dev/null | grep -o "true\|false" || echo "unknown")
    if [ "$status" = "true" ]; then
      echo "[INFO] Judge server is now sleeping (GPUs released)"
      break
    fi
    [ $i -eq 30 ] && echo "[WARN] Judge server sleep status uncertain, continuing anyway..."
    sleep 1
  done
fi

cd "${VERL_ROOT}"

python3 -u -m verl.trainer.main_ppo \
    --config-path="${SCRIPTS_DIR}/config" \
    --config-name=grpo_qwen35_4b_llm_judge \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.rollout_data_dir="${CKPTS_DIR}/rollout" \
    trainer.save_freq=40 \
    trainer.resume_mode=auto \
    data.max_prompt_length=8192 \
    data.max_response_length=8192 \
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=4096 \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False \
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False \
    reward.reward_kwargs.llm_judge_kwargs.base_url="http://localhost:${JUDGE_PORT}/v1" \
    reward.reward_kwargs.llm_judge_kwargs.prompt_path="${SCRIPTS_DIR}/llm_judge_prompt.txt" \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo "[INFO] Training completed with exit code: $EXIT_CODE"

# ── Convert checkpoint to HuggingFace format ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$EXIT_CODE" -eq 0 ]; then
  echo "[INFO] Converting FSDP checkpoint to HuggingFace format..."
  bash "${SCRIPT_DIR}/../../convert/convert_fsdp_to_hf.sh" "${CKPTS_DIR}" \
    || echo "[WARN] Checkpoint conversion failed (non-fatal)"
fi

if [ "$START_JUDGE_SERVER" -eq 1 ]; then
  echo "[INFO] Stopping judge server..."
  pkill -f "vllm serve.*--port $JUDGE_PORT" 2>/dev/null || true
  sleep 2
fi

exit $EXIT_CODE
