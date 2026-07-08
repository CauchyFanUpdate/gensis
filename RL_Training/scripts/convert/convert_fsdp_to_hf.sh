#!/bin/bash
# 将 verl FSDP checkpoint 转换为 HuggingFace 格式
#
# 用法：
#   bash convert_fsdp_to_hf.sh <checkpoint_dir> [step]
#
#   checkpoint_dir: 实验目录，如 checkpoints/qwen3vl_grpo_xxx
#   step:           要导出的 step，不填则自动取最新的
#
# 示例：
#   bash convert_fsdp_to_hf.sh checkpoints/qwen3vl_grpo_rubric_20260501_204048
#   bash /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/scripts/convert/convert_fsdp_to_hf.sh /mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/checkpoints/qwen35_4b/qwen35_grpo_rlvr_stem

set -e

CKPT_ROOT="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/checkpoints"
VERL_ROOT="/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/naive_cross_vlm/workspace/verl"

# ── 参数解析 ──────────────────────────────────────────────────────
if [ -z "$1" ]; then
    echo "[ERROR] Usage: bash convert_fsdp_to_hf.sh <checkpoint_dir> [step]"
    exit 1
fi
CKPT_DIR="$1"

# 如果传的是相对路径，补全为绝对路径
if [[ "$CKPT_DIR" != /* ]]; then
    CKPT_DIR="${CKPT_ROOT}/${CKPT_DIR}"
fi

if [ ! -d "$CKPT_DIR" ]; then
    echo "[ERROR] checkpoint 目录不存在: $CKPT_DIR"
    exit 1
fi

# 确定 step
if [ -n "$2" ]; then
    STEP="$2"
    LOCAL_DIR="${CKPT_DIR}/global_step_${STEP}/actor"
else
    # 自动取最新的 global_step_xxx
    LOCAL_DIR=$(ls -d "${CKPT_DIR}"/global_step_*/actor 2>/dev/null | sort -V | tail -1)
    STEP=$(basename "$(dirname "$LOCAL_DIR")" | sed 's/global_step_//')
fi

if [ ! -d "$LOCAL_DIR" ]; then
    echo "[ERROR] actor 目录不存在: $LOCAL_DIR"
    echo "可用的 step:"
    ls -d "${CKPT_DIR}"/global_step_* 2>/dev/null || echo "  (无)"
    exit 1
fi

TARGET_DIR="${CKPT_DIR}/hf_model_step${STEP}"

echo "=================================================="
echo "[INFO] checkpoint 目录 : $CKPT_DIR"
echo "[INFO] actor 目录      : $LOCAL_DIR"
echo "[INFO] 输出目录        : $TARGET_DIR"
echo "[INFO] step            : $STEP"
echo "=================================================="

mkdir -p "$TARGET_DIR"

cd "$VERL_ROOT"
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR"

echo ""
echo "[INFO] 转换完成！HF 模型已保存至: $TARGET_DIR"
