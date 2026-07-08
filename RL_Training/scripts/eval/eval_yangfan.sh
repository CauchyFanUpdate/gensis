#!/bin/bash
set -e

VLLM_PYTHON="/root/miniconda3/envs/vllm_ov/bin/python"
PYTHON="/mnt/tidal-alsh01/dataset/perceptionVLM/code_chenlifeng/miniforge3/envs/VlmEvalKit/bin/python"
VLMEVAL="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/VLMEvalKit/run.py"
WORK_DIR="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/eval_results"
LOG_DIR="${WORK_DIR}/logs"
CKPT_BASE="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/Research/llamafactory-ego/output"
PORT=9006

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

mkdir -p "${WORK_DIR}" "${LOG_DIR}"

# 主脚本自身的输出也 tee 到总日志
MAIN_LOG="${LOG_DIR}/main.log"
exec > >(tee -a "${MAIN_LOG}") 2>&1

# 杀掉 vllm 主进程及其所有子进程（DP worker、EngineCore 等）
stop_vllm() {
  local pid=${1:-}
  local pgid=${2:-}
  [ -z "${pid}" ] && return
  echo "[$(date)] Stopping vllm (pid=${pid}, pgid=${pgid}) and all child processes..."
  if [ -n "${pgid}" ] && [ "${pgid}" != "0" ]; then
    kill -9 -- -"${pgid}" 2>/dev/null || true
  fi
  kill -9 "${pid}" 2>/dev/null || true
  # 兜底：杀掉所有还占用该端口的进程
  fuser -k ${PORT}/tcp 2>/dev/null || true
  # 兜底：按 vllm_ov env 路径匹配，覆盖所有 DP worker（cmdline 不含 vllm.entrypoints）
  pkill -9 -f "vllm_ov" 2>/dev/null || true
  wait "${pid}" 2>/dev/null || true
  sleep 15
  echo "[$(date)] vllm stopped."
}

VLLM_PID=""
VLLM_PGID=""
trap 'stop_vllm "${VLLM_PID}" "${VLLM_PGID}"' EXIT INT TERM

# 清理上次可能残留的 vllm 进程
echo "[$(date)] Cleaning up any leftover processes on port ${PORT}..."
fuser -k ${PORT}/tcp 2>/dev/null || true
pkill -9 -f "vllm_ov" 2>/dev/null || true
sleep 5

echo "[$(date)] Starting eval_yangfan"

for CKPT_NAME in "${!BENCH_MAP[@]}"; do
  BENCH="${BENCH_MAP[$CKPT_NAME]}"
  CKPT_PATH="${CKPT_BASE}/${CKPT_NAME}"
  SERVED_MODEL="qwen3vl-${CKPT_NAME}"
  VLMEVAL_MODEL="yangfan-${SERVED_MODEL}"
  VLLM_LOG="${LOG_DIR}/vllm_${CKPT_NAME}.log"
  EVAL_LOG="${LOG_DIR}/eval_${CKPT_NAME}.log"

  echo "========================================"
  echo "[$(date)] Checkpoint : ${CKPT_NAME}"
  echo "           Benchmark  : ${BENCH}"
  echo "           vllm log   : ${VLLM_LOG}"
  echo "           eval log   : ${EVAL_LOG}"
  echo "========================================"

  # setsid 让 vllm 成为新 session leader，PGID = 自身 PID
  # 这样 kill -- -PGID 能杀掉所有 DP worker / EngineCore 子进程，不会误杀本脚本
  setsid "${VLLM_PYTHON}" -m vllm.entrypoints.cli.main serve "${CKPT_PATH}" \
    --served-model-name "${SERVED_MODEL}" \
    --tensor-parallel-size 1 \
    --data-parallel-size 8 \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --trust-remote-code \
    --host 0.0.0.0 \
    --port ${PORT} > "${VLLM_LOG}" 2>&1 &
  VLLM_PID=$!
  VLLM_PGID=$VLLM_PID  # setsid 后 PGID = PID

  echo "[$(date)] Waiting for server (pid=${VLLM_PID}, pgid=${VLLM_PGID}) on port ${PORT} ..."
  until curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; do
    sleep 5
  done
  echo "[$(date)] Server up. Sending warmup request (with image) to trigger Triton compilation..."
  # 1x1 红色像素 PNG，触发图片 encoder 的 Triton kernel 编译
  WARMUP_IMG="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADklEQVQI12P4z8BQDwAEgAF/QualIQAAAABJRU5ErkJggg=="
  WARMUP_BODY="{\"model\":\"${SERVED_MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,${WARMUP_IMG}\"}},{\"type\":\"text\",\"text\":\"hi\"}]}],\"max_tokens\":5}"
  until curl -s -X POST "http://127.0.0.1:${PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer sk-123456" \
    -d "${WARMUP_BODY}" \
    --max-time 600 | grep -q "choices"; do
    echo "[$(date)] Warmup not ready yet, retrying..."
    sleep 10
  done
  echo "[$(date)] Warmup done. Server ready for eval."

  # eval 日志单独写文件，同时打到终端
  "${PYTHON}" "${VLMEVAL}" \
    --data "${BENCH}" \
    --model "${VLMEVAL_MODEL}" \
    --work-dir "${WORK_DIR}" \
    --api-nproc 32 \
    --judge qwen3.6-plus 2>&1 | tee "${EVAL_LOG}"

  echo "[$(date)] Done: ${CKPT_NAME}. Stopping server..."
  stop_vllm "${VLLM_PID}" "${VLLM_PGID}"
  VLLM_PID=""
  VLLM_PGID=""
done

echo "[$(date)] All done. Results in ${WORK_DIR}"
