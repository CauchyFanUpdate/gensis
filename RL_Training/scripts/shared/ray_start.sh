#!/bin/bash
# Ray 集群启动脚本（K8s 模式）
#
# 通过 POD_NAME 判断角色：
#   - 含 "master" → Ray HEAD
#   - 不含 "master" → Ray WORKER（阻塞等待）
#
# 必须注入的环境变量：
#   MASTER_ADDR      - head 节点 IP
#   HOST_IP          - 当前节点 IP
#   POD_NAME         - 当前 Pod 名称
#   WORLD_SIZE       - 节点总数
#   N_GPUS_PER_NODE  - 每节点 GPU 数（默认 8）

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export RAY_LOG_TO_DRIVER=true

# ── 把 conda NCCL 置顶到 LD_LIBRARY_PATH，防止系统 torch 路径里的旧 NCCL 被优先加载 ──
# 机器B的 LD_LIBRARY_PATH 包含 /usr/local/lib/python3.12/dist-packages/torch/lib，
# 其中可能有不同版本的 libnccl.so.2，必须确保 conda 的版本优先。
_CONDA_PREFIX=${CONDA_PREFIX:-/root/miniconda3/envs/verl_qwen3}
_CONDA_NCCL="${_CONDA_PREFIX}/lib/python3.12/site-packages/nvidia/nccl/lib"
_CONDA_TORCH="${_CONDA_PREFIX}/lib/python3.12/site-packages/torch/lib"
export LD_LIBRARY_PATH="${_CONDA_NCCL}:${_CONDA_TORCH}:${LD_LIBRARY_PATH:-}"
echo "[ray_start] LD_LIBRARY_PATH (first 2 entries): ${_CONDA_NCCL}:${_CONDA_TORCH}"
unset _CONDA_PREFIX _CONDA_NCCL _CONDA_TORCH

# NCCL/GLOO 网络配置 — 在 ray start 之前 export，让 Ray daemon 继承，
# 避免被 verl ray.init() 的 runtime_env 覆盖
export NCCL_TIMEOUT=1800
export CUDA_DEVICE_MAX_CONNECTIONS=1

N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

if echo "${POD_NAME}" | grep -q master; then
    echo "[ray_start] HEAD node: ${HOST_IP:-${MASTER_ADDR}}"
    ray start --head --port=6379 \
        --node-ip-address="${HOST_IP:-${MASTER_ADDR}}"
else
    echo "[ray_start] WORKER node: ${HOST_IP} -> HEAD ${MASTER_ADDR}:6379"
    ray start --block \
        --address="${MASTER_ADDR}:6379" \
        --node-ip-address="${HOST_IP}"
fi

if echo "${POD_NAME}" | grep -q master; then
    target_gpus=$(( WORLD_SIZE * N_GPUS_PER_NODE ))
    echo "[ray_start] Waiting for ${target_gpus} GPUs (${WORLD_SIZE} nodes x ${N_GPUS_PER_NODE} GPUs)..."
    while true; do
        gpu_info=$(ray status 2>/dev/null | grep GPU || true)
        current_gpus=$(echo "${gpu_info}" | awk -F '[./ ]' '{ print $4 }')
        echo "[ray_start] GPUs ready: ${current_gpus:-0}/${target_gpus}"
        if [ "${current_gpus:-0}" -ge "${target_gpus}" ] 2>/dev/null; then
            echo "[ray_start] All nodes joined!"
            break
        fi
        sleep 5
    done
else
    unset WORLD_SIZE
    echo "[ray_start] Worker blocking forever (managed by K8s)..."
    sleep inf
fi
