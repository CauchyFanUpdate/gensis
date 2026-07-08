#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# BabyVision v2 Pipeline — Stage 1 only
#
# Stage 1: surgical 答案变化编辑（每 source 5 edits × 1 seed，1:5 ratio）
#          + 双路 Qwen VL 交叉验证 actual_new_answer
#
# Usage:
#   bash run_babyvision_v2.sh
#   LIMIT=10 bash run_babyvision_v2.sh   # 冒烟（10 题）
#
# Resume: 自动启用，已写入 jsonl 的 sample_key 会被跳过。
#         (强制重跑请删除 output_babyvision_v2/stage1_samples.jsonl 或加 --no_resume)
#
# 可选环境变量：
#   PARQUET, OUTPUT_DIR, MAX_CONCURRENT, MAX_RETRIES, LIMIT
#   GEMINI_VL_ENDPOINT, GEMINI_VL_MODEL, GEMINI_VL_API_KEY
#   QWEN_API_BASE, QWEN_MODEL, QWEN_API_KEY
#   GPT_IMAGE_ENDPOINT, GPT_IMAGE_API_KEY
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PARQUET="${PARQUET:-/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/BabyVision/data/train-00000-of-00001.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output_babyvision_v2}"

MAX_CONCURRENT="${MAX_CONCURRENT:-64}"

export API_TIMEOUT="${API_TIMEOUT:-1200}"
export MAX_RETRIES="${MAX_RETRIES:-20}"

LIMIT="${LIMIT:-0}"

# ── 1. Gemini VL（Call 1 分析 / Modifier）─────────────────────────────────
export GEMINI_VL_ENDPOINT="${GEMINI_VL_ENDPOINT:-https://runway.devops.rednote.life/openai/google/v1:generateContent}"
export GEMINI_VL_MODEL="${GEMINI_VL_MODEL:-gemini-3.1-pro-preview}"
export GEMINI_VL_API_KEY="${GEMINI_VL_API_KEY:-YOUR_GEMINI_API_KEY}"

# ── 2. Qwen3.6-plus（Call 2a text + Modifier VL）──────────────────────────
export QWEN_API_BASE="${QWEN_API_BASE:-https://runway.devops.xiaohongshu.com/openai/qwen/v1}"
export QWEN_MODEL="${QWEN_MODEL:-qwen3.6-plus}"
export QWEN_API_KEY="${QWEN_API_KEY:-${OPENAI_API_KEY:-YOUR_API_KEY}}"
export OPENAI_API_KEY="$QWEN_API_KEY"
if [ -z "$QWEN_API_KEY" ]; then
    echo "[run_babyvision_v2] ERROR: QWEN_API_KEY (or OPENAI_API_KEY) env var required"
    exit 1
fi

# ── 3. GPT-Image-2（Call 3a stage1 编辑）──────────────────────────────────
# 当前用 api.gpt.ge 外网端点 + model=gpt-image-2-c（cheap variant，速率配额更宽）
# Helper 自动识别 endpoint 风格：以 /v1 结尾 → Bearer auth；否则 → Azure 风格 api-key 头 + ?api-version=...
export GPT_IMAGE_ENDPOINT="${GPT_IMAGE_ENDPOINT:-https://api.gpt.ge/v1}"
export GPT_IMAGE_API_KEY="${GPT_IMAGE_API_KEY:-YOUR_GPT_IMAGE_API_KEY}"

echo "[run_babyvision_v2] Parquet:            $PARQUET"
echo "[run_babyvision_v2] Output dir:         $OUTPUT_DIR"
echo "[run_babyvision_v2] Max concurrent:     $MAX_CONCURRENT"
echo "[run_babyvision_v2] Max retries:        $MAX_RETRIES"
echo "[run_babyvision_v2] API timeout:        ${API_TIMEOUT}s"
echo "[run_babyvision_v2] Limit:              ${LIMIT} (0=全量)"
echo "[run_babyvision_v2] Gemini VL:  $GEMINI_VL_MODEL @ $GEMINI_VL_ENDPOINT  (key=${GEMINI_VL_API_KEY:0:6}…)"
echo "[run_babyvision_v2] Qwen:       $QWEN_MODEL @ $QWEN_API_BASE  (key=${QWEN_API_KEY:0:6}…)"
echo "[run_babyvision_v2] GPT-Image:  gpt-image-2 @ $GPT_IMAGE_ENDPOINT  (key=${GPT_IMAGE_API_KEY:0:6}…)"
echo ""

cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════════"
echo "  STAGE 1: surgical answer-changing edit (5 edits/source)"
echo "═══════════════════════════════════════════════════════"

python3 generate_babyvision_v2_stage1.py \
    --parquet         "$PARQUET" \
    --output_dir      "$OUTPUT_DIR" \
    --max_concurrent  "$MAX_CONCURRENT" \
    --max_retries     "$MAX_RETRIES" \
    --limit           "$LIMIT"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  STAGE 1 完成"
echo "═══════════════════════════════════════════════════════"
echo "  输出: $OUTPUT_DIR/stage1_samples.jsonl"
echo "        $OUTPUT_DIR/stage1_images/"
