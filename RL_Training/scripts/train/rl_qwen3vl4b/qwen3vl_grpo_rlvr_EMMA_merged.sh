#!/bin/bash
# GRPO RLVR training — rule-based reward, LLM judge for complex zerobench answers only (rubric disabled)

set -x
set -o pipefail

# ── Configuration ──────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERL_ROOT=${VERL_ROOT:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/naive_cross_vlm/workspace/verl"}
export VERL_ROOT

MODEL_PATH=${MODEL_PATH:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3-VL-4B-Instruct"}
export MODEL_PATH

TRAIN_FILE=${TRAIN_FILE:-"/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench_output_v5/parquet/emma_all_merged_think.parquet"}
export TRAIN_FILE

VAL_FILE=${VAL_FILE:-"/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench_output_v5/parquet/emma_all_merged_think.parquet"}
export VAL_FILE

# Training config
EXPERIMENT_NAME="emma_merged"
export EXPERIMENT_NAME
CKPT_ROOT="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/checkpoints/qwen3vl4b"
CKPTS_DIR="${CKPT_ROOT}/${EXPERIMENT_NAME}"
export CKPTS_DIR

N_GPUS=${N_GPUS:-8}
export N_GPUS

ROLLOUT_TP=${ROLLOUT_TP:-1}
export ROLLOUT_TP

# Judge server (for complex zerobench answer evaluation, rubric disabled)
JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3-32B"}
JUDGE_PORT=${JUDGE_PORT:-31001}
JUDGE_TP=${JUDGE_TP:-4}
JUDGE_GPU_MEMORY_UTIL=${JUDGE_GPU_MEMORY_UTIL:-0.35}
START_JUDGE_SERVER=${START_JUDGE_SERVER:-0}
JUDGE_HOST=${JUDGE_HOST:-"10.144.205.26"}

# ── Cleanup ────────────────────────────────────────────────────────
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


mkdir -p "$CKPTS_DIR"
LOG_FILE="$CKPTS_DIR/train.log"

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
    > "$CKPTS_DIR/vllm_judge_${JUDGE_PORT}.log" 2>&1 &

  JUDGE_PID=$!
  echo "[INFO] vLLM judge server started (PID: $JUDGE_PID)"
  echo "[INFO] Log: $CKPTS_DIR/vllm_judge_${JUDGE_PORT}.log"

  JUDGE_READY=0
  for i in {1..60}; do
    if curl -s "http://10.144.206.71:$JUDGE_PORT/v1/models" >/dev/null 2>&1; then
      JUDGE_READY=1
      echo "[INFO] vLLM judge server is ready!"
      break
    fi
    echo "  Attempt $i/60..."
    sleep 60
  done

  if [ "$JUDGE_READY" -ne 1 ]; then
    echo "[ERROR] vLLM judge server did not start in time"
    echo "[ERROR] Check log: $CKPTS_DIR/vllm_judge_${JUDGE_PORT}.log"
    exit 1
  fi

  curl -s "http://10.144.206.71:$JUDGE_PORT/sleep?level=1" >/dev/null 2>&1 || true
  sleep 2
  for i in {1..30}; do
    status=$(curl -s "http://10.144.206.71:$JUDGE_PORT/is_sleeping" 2>/dev/null | grep -o "true\|false" || echo "unknown")
    if [ "$status" = "true" ]; then
      echo "[INFO] Judge server is now sleeping (GPUs released)"
      break
    fi
    [ $i -eq 30 ] && echo "[WARN] Judge server sleep status uncertain, continuing anyway..."
    sleep 1
  done
fi


# ── Environment ────────────────────────────────────────────────────
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_MODE="online"
export NO_PROXY="localhost,127.0.0.1,10."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export no_proxy="localhost,127.0.0.1,10."
export RAY_raylet_start_wait_time_s=120
export RAY_num_server_call_thread=4
export RAY_core_worker_num_server_call_thread=4
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.9;9.0+PTX}"

echo "[INFO] Starting GRPO RLVR training (rubric disabled, judge for complex answers)"
echo "[INFO] Experiment: $EXPERIMENT_NAME"
echo "[INFO] Checkpoint dir: $CKPTS_DIR"
echo "[INFO] Judge URL: http://${JUDGE_HOST}:${JUDGE_PORT}/v1"
echo "[INFO] Log file: $LOG_FILE"

cd "$VERL_ROOT"

python3 -u -m verl.trainer.main_ppo \
    --config-path="$SCRIPT_DIR/../../config" \
    --config-name=grpo_qwen3vl_4b_llm_judge \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.project_name='zerovision' \
    trainer.rollout_data_dir="$CKPTS_DIR/rollout" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    reward.reward_kwargs.rubric_weight=0 \
    reward.reward_kwargs.llm_judge_kwargs.base_url="http://${JUDGE_HOST}:${JUDGE_PORT}/v1" \
    reward.reward_kwargs.llm_judge_kwargs.prompt_path="${SCRIPTS_DIR}/llm_judge_prompt.txt" \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=${PIPESTATUS[0]}
echo "[INFO] Training completed with exit code: $EXIT_CODE"

if [ "$EXIT_CODE" -eq 0 ]; then
  echo "[INFO] Converting FSDP checkpoint to HuggingFace format..."
  bash "${SCRIPTS_DIR}/convert/convert_fsdp_to_hf.sh" "$CKPTS_DIR" \
    || echo "[WARN] Checkpoint conversion failed (non-fatal)"
fi

if [ "$START_JUDGE_SERVER" -eq 1 ]; then
  echo "[INFO] Stopping judge server..."
  pkill -f "vllm serve.*--port $JUDGE_PORT" 2>/dev/null || true
  sleep 2
fi

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

exit $EXIT_CODE
