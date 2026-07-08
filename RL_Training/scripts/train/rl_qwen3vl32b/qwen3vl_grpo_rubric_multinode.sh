#!/bin/bash
# Multi-node GRPO training with LLM judge for Qwen3-VL-32B-Instruct
#
# ===== 启动方式 =====
#
#   K8s 平台在每台机器上同时运行本脚本，通过环境变量区分角色：
#     - POD_NAME 含 "master" → Ray HEAD，提交训练
#     - POD_NAME 不含 "master" → Ray WORKER，加入集群后阻塞等待
#
#   平台必须注入以下环境变量：
#     MASTER_ADDR  - master 节点 IP
#     HOST_IP      - 当前节点 IP
#     POD_NAME     - Pod 名称（含 "master" 为 head）
#
# ===== 手动启动（非 K8s）=====
#
#   judge server（10.144.204.172 上提前启动）:
#PORT=31001 TP=4 bash /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/train/rl_qwen3vl32b/start_judge_server.sh
#
#   主机 10.144.206.207 (head):
#env POD_NAME=master MASTER_ADDR=10.144.206.207 HOST_IP=10.144.206.207 WORLD_SIZE=6 bash /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/train/rl_qwen3vl32b/qwen3vl_grpo_rubric_multinode.sh
#
#   worker 10.144.206.218:
#env POD_NAME=worker MASTER_ADDR=10.144.206.207 HOST_IP=10.144.206.218 WORLD_SIZE=6 bash /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/train/rl_qwen3vl32b/qwen3vl_grpo_rubric_multinode.sh
#
#   worker 10.144.204.220:
#env POD_NAME=worker MASTER_ADDR=10.144.206.207 HOST_IP=10.144.204.220 WORLD_SIZE=6 bash /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/train/rl_qwen3vl32b/qwen3vl_grpo_rubric_multinode.sh
#
#   worker 10.144.206.214:
#env POD_NAME=worker MASTER_ADDR=10.144.206.207 HOST_IP=10.144.206.214 WORLD_SIZE=6 bash /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/train/rl_qwen3vl32b/qwen3vl_grpo_rubric_multinode.sh
#
#   worker 10.144.204.212:
#env POD_NAME=worker MASTER_ADDR=10.144.206.207 HOST_IP=10.144.204.212 WORLD_SIZE=6 bash /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/train/rl_qwen3vl32b/qwen3vl_grpo_rubric_multinode.sh
#
#   worker 10.144.206.211:
#env POD_NAME=worker MASTER_ADDR=10.144.206.207 HOST_IP=10.144.206.211 WORLD_SIZE=6 bash /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/train/rl_qwen3vl32b/qwen3vl_grpo_rubric_multinode.sh

# ── 清理残留进程 ──────────────────────────────────────────────────
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

set -x
set -o pipefail

# ── 基础配置 ──────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPTS_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
VERL_ROOT=${VERL_ROOT:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/naive_cross_vlm/workspace/verl"}
export VERL_ROOT

export WORLD_SIZE="${WORLD_SIZE:-6}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

MODEL_PATH=${MODEL_PATH:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3-VL-32B-Instruct"}
export MODEL_PATH

TRAIN_FILE=${TRAIN_FILE:-"/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench_output_v5/parquet/merged_dataset.parquet"}
export TRAIN_FILE

VAL_FILE=${VAL_FILE:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/dataset/parquets/stem.parquet"}
export VAL_FILE

# Judge server 配置
JUDGE_MODEL_PATH=${JUDGE_MODEL_PATH:-"/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3-32B"}
JUDGE_PORT=${JUDGE_PORT:-31001}
JUDGE_TP=${JUDGE_TP:-4}
START_JUDGE_SERVER=${START_JUDGE_SERVER:-0}
JUDGE_HOST=${JUDGE_HOST:-"10.144.204.172"}

# 训练配置
DATE_TIME="$(date +%Y%m%d_%H%M%S)"
EXPERIMENT_NAME="stem_final_grpo_rubric_general"
export EXPERIMENT_NAME
CKPT_ROOT="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/checkpoints/qwen3vl32b"
CKPTS_DIR="${CKPT_ROOT}/${EXPERIMENT_NAME}"
export CKPTS_DIR

# [perf-tune] 原值 4，改为 2：H200 141GB 显存充裕，TP=2 让 rollout 实例数翻倍（48 卡 12→24 实例），缓解长尾。OOM 时改回 4
# ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_TP=${ROLLOUT_TP:-2}
export ROLLOUT_TP

FILTER_OVERLONG=${FILTER_OVERLONG:-true}
export FILTER_OVERLONG

# Judge URL 指向外部专用机器
LLM_JUDGE_HOST="${JUDGE_HOST}"
export JUDGE_BASE_URL="http://${LLM_JUDGE_HOST}:${JUDGE_PORT}/v1"
export JUDGE_API_KEY="qwen"
export JUDGE_MODEL_NAME="qwen"
export JUDGE_PROMPT_PATH="${SCRIPTS_DIR}/llm_judge_prompt.txt"

# ── 环境变量 ──────────────────────────────────────────────────────
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.9;9.0+PTX}"
export WANDB_API_KEY="YOUR_WANDB_API_KEY"
export WANDB_MODE="online"
export NO_PROXY="localhost,127.0.0.1,10."
export no_proxy="localhost,127.0.0.1,10."
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY
export RAY_raylet_start_wait_time_s=120
export RAY_num_server_call_thread=4
export RAY_core_worker_num_server_call_thread=4


# ── 启动 Ray 集群 ────────────────────────────────────────────────
# Worker 节点会在这里阻塞（sleep inf），不会执行后续代码
# 只有 HEAD 节点（POD_NAME 含 master）会继续往下走
bash "${SCRIPTS_DIR}/shared/ray_start.sh"

# ================================================================
# 以下仅在 HEAD 节点执行
# ================================================================

# ── 启动 LLM Judge server（仅 head 节点）─────────────────────────
if [ "$START_JUDGE_SERVER" -eq 1 ]; then
    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting vLLM judge server"
    echo "  Model: $JUDGE_MODEL_PATH"
    echo "  Port: $JUDGE_PORT"
    echo "  TP: $JUDGE_TP"
    echo "  Access URL: http://${LLM_JUDGE_HOST}:${JUDGE_PORT}/v1"
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
        > "$CKPTS_DIR/vllm_judge_${JUDGE_PORT}.log" 2>&1 &

    JUDGE_PID=$!
    echo "[INFO] vLLM judge server started (PID: $JUDGE_PID)"

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
        echo "[ERROR] Check log: $CKPTS_DIR/vllm_judge_${JUDGE_PORT}.log"
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

# ── 准备目录和日志 ────────────────────────────────────────────────
mkdir -p "$CKPTS_DIR"
LOG_FILE="$CKPTS_DIR/train.log"

echo "[INFO] Starting multi-node GRPO training"
echo "[INFO] Nodes: ${WORLD_SIZE} x ${N_GPUS_PER_NODE} GPUs = $(( WORLD_SIZE * N_GPUS_PER_NODE )) total"
echo "[INFO] Judge URL: http://${LLM_JUDGE_HOST}:${JUDGE_PORT}/v1"
echo "[INFO] Checkpoint dir: $CKPTS_DIR"
echo "[INFO] Log file: $LOG_FILE"

# ── runtime-env：把环境变量下发到所有节点的 Ray worker ────────────
RUNTIME_ENV_JSON=$(cat <<EOF
{
  "env_vars": {
    "PYTHONUNBUFFERED": "1",
    "TORCH_CUDA_ARCH_LIST": "${TORCH_CUDA_ARCH_LIST}",
    "NCCL_TIMEOUT": "1800",
    "RAY_OVERRIDE_JOB_RUNTIME_ENV": "1",
    "MASTER_ADDR": "${MASTER_ADDR}",
    "MODEL_PATH": "${MODEL_PATH}",
    "TRAIN_FILE": "${TRAIN_FILE}",
    "VAL_FILE": "${VAL_FILE}",
    "CKPTS_DIR": "${CKPTS_DIR}",
    "EXPERIMENT_NAME": "${EXPERIMENT_NAME}",
    "WANDB_API_KEY": "${WANDB_API_KEY}",
    "WANDB_MODE": "${WANDB_MODE}",
    "NO_PROXY": "localhost,127.0.0.1,10.",
    "no_proxy": "localhost,127.0.0.1,10.",
    "HTTP_PROXY": "",
    "HTTPS_PROXY": "",
    "http_proxy": "",
    "https_proxy": "",
    "all_proxy": "",
    "ALL_PROXY": ""
  }
}
EOF
)

# ── 通过 ray job submit 提交训练 ─────────────────────────────────
cd "$VERL_ROOT"

ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 -u -m verl.trainer.main_ppo \
        --config-path="$SCRIPTS_DIR/config" \
        --config-name=grpo_qwen3vl_4b_llm_judge \
        trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
        trainer.nnodes="$WORLD_SIZE" \
        trainer.experiment_name="$EXPERIMENT_NAME" \
        trainer.project_name='zerovision' \
        data.filter_overlong_prompts="$FILTER_OVERLONG" \
        data.filter_overlong_prompts_workers=128 \
        actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
        actor_rollout_ref.rollout.enforce_eager=false \
        `# [perf-tune] 新增：启用 vLLM CUDA Graph，rollout 提速 30-50%。OOM 或编译报错时删除此行` \
        actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0 \
        actor_rollout_ref.model.use_fused_kernels=false \
        actor_rollout_ref.actor.fsdp_config.fsdp_size=8 \
        actor_rollout_ref.ref.fsdp_config.fsdp_size=8 \
        `# [perf-tune] batch 48→96, mini 24→48：48 卡每卡 prompt 数从 1→2，缓解欠载。OOM 时改回 48/24` \
        data.train_batch_size=96 \
        actor_rollout_ref.actor.ppo_mini_batch_size=48 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        `# [perf-tune] max_token_len 16384→32768：H200 141GB 显存充裕，每 micro-batch 装 2x token。OOM 时改回 16384` \
        actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
        "reward.reward_kwargs.llm_judge_kwargs.base_url=http://${LLM_JUDGE_HOST}:${JUDGE_PORT}/v1" \
        reward.reward_kwargs.llm_judge_kwargs.timeout=300 \
        reward.reward_kwargs.llm_judge_kwargs.prompt_path="${SCRIPTS_DIR}/llm_judge_prompt.txt" \
        reward.reward_kwargs.llm_judge_kwargs.rubric_prompt_path="${SCRIPTS_DIR}/rubric_prompt.txt" \
        reward.reward_kwargs.llm_judge_kwargs.rubric_mode=general \
        trainer.save_freq=40 \
        "$@" \
    2>&1 | tee "${LOG_FILE}"

EXIT_CODE=$?
echo "[INFO] Training completed with exit code: $EXIT_CODE"

# ── Convert checkpoint to HuggingFace format ──────────────────────
if [ "$EXIT_CODE" -eq 0 ]; then
  echo "[INFO] Converting FSDP checkpoint to HuggingFace format..."
  bash "${SCRIPT_DIR}/../convert/convert_fsdp_to_hf.sh" "$CKPTS_DIR" \
    || echo "[WARN] Checkpoint conversion failed (non-fatal)"
fi

# ── 清理 ─────────────────────────────────────────────────────────
if [ "$START_JUDGE_SERVER" -eq 1 ]; then
    echo "[INFO] Stopping judge server..."
    pkill -f "vllm serve.*--port $JUDGE_PORT" 2>/dev/null || true
    sleep 2
fi

ray stop --force 2>/dev/null || true
exit $EXIT_CODE
