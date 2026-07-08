#!/bin/bash
# SAPO training for Qwen3.5-4B (without LLM judge, rule-based reward only)
# Smoothed adaptive preference optimization: soft tau-based PPO
# Usage: bash run_sapo_qwen35.sh
# Environment variables: MODEL_PATH, TRAIN_FILE, VAL_FILE, N_GPUS, ROLLOUT_TP, etc.

set -x
set -o pipefail

# ── Cleanup stale Ray / vLLM processes from previous runs ─────────────────────
pkill -TERM -f "verl.trainer.main_ppo" 2>/dev/null || true
pkill -TERM -f "sglang.srt"            2>/dev/null || true
pkill -TERM -f "raylet"                2>/dev/null || true
pkill -TERM -f "gcs_server"            2>/dev/null || true
sleep 30

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

# ── HF datasets cache ─────
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/.hf_cache}"

# ── Paths ──────────────────────────────────────────────────────────────────────
DATE_TIME="$(date +%Y%m%d_%H%M%S)"
VERL_ROOT=${VERL_ROOT:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/naive_cross_vlm/workspace/verl"}
DATASET_ROOT=${DATASET_ROOT:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/naive_cross_vlm/dataset/stage3_grpo"}

MODEL_PATH=${MODEL_PATH:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3.5-4B"}
TRAIN_FILE=${TRAIN_FILE:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/dataset/charxiv_babyvision_mmmu_zerobench.parquet"}
VAL_FILE=${VAL_FILE:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/dataset/charxiv_babyvision.parquet"}

EXPERIMENT_NAME="qwen35_charxiv_babyvision_mmmu_zerobench_sapo_${DATE_TIME}"
CKPT_ROOT="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/checkpoints"
CKPTS_DIR=${CKPTS_DIR:-"${CKPT_ROOT}/${EXPERIMENT_NAME}"}

PROJECT_NAME=${PROJECT_NAME:-"zerovision"}

N_GPUS=${N_GPUS:-8}
ROLLOUT_TP=${ROLLOUT_TP:-1}
ENGINE=${1:-vllm}

# SAPO hyperparameters
TAU_POS=${TAU_POS:-1.0}
TAU_NEG=${TAU_NEG:-1.05}

# ── Log file ───────────────────────────────────────────────────────────────────
mkdir -p "${CKPTS_DIR}"
LOG_FILE="${CKPTS_DIR}/train.log"
echo "Logging to: ${LOG_FILE}"

cd "${VERL_ROOT}"

echo "[INFO] Starting SAPO training (rule-based reward)..."
echo "[INFO] Model: $MODEL_PATH"
echo "[INFO] Checkpoint dir: $CKPTS_DIR"
echo "[INFO] Log file: $LOG_FILE"
echo "[INFO] SAPO tau_pos=$TAU_POS tau_neg=$TAU_NEG"

python3 -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    actor_rollout_ref.actor.policy_loss.loss_mode=sapo \
    actor_rollout_ref.actor.tau_pos=$TAU_POS \
    actor_rollout_ref.actor.tau_neg=$TAU_NEG \
    \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=4096 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=true \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$ROLLOUT_TP \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.45 \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.max_num_batched_tokens=20000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.rollout.enforce_eager=false \
    actor_rollout_ref.rollout.free_cache_engine=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    \
    reward.reward_manager.name=naive \
    reward.num_workers=2 \
    reward.reward_model.enable=false \
    \
    trainer.critic_warmup=0 \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.default_local_dir=$CKPTS_DIR \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.project_name=$PROJECT_NAME \
    trainer.total_epochs=1 \
    trainer.save_freq=60 \
    trainer.test_freq=-1 \
    trainer.val_before_train=false \
    trainer.logger=['console','wandb'] \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
echo "[INFO] Training completed with exit code: $EXIT_CODE"

# ── Convert checkpoint to HuggingFace format ──────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$EXIT_CODE" -eq 0 ]; then
  echo "[INFO] Converting FSDP checkpoint to HuggingFace format..."
  bash "${SCRIPT_DIR}/../convert/convert_fsdp_to_hf.sh" "$CKPTS_DIR" \
    || echo "[WARN] Checkpoint conversion failed (non-fatal)"
fi

exit $EXIT_CODE
