#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# BabyVision v2 — Original-Image Diversification (Stage 2 only)
#
# 对原图进行 cosmetic 多样化，答案保持不变，每道题生成 5 张（1:5）。
#
# 流程：
#   Call 1  [Gemini VL]  → frozen_predicate + axis_table 分析
#   Selector [Qwen VL]   → 5 套 axis 方案（看原图）
#   Call 2b [Qwen text]  → 多样化编辑 prose
#   Call 3b [GPT-Image-2]→ 从原图生成多样化图
#   Verifier [Qwen VL]   → 4 yes/no 考点验证
#   Call 4  [Gemini VL]  → CoT（答案锚定原始答案）
#
# Usage:
#   bash run_babyvision_v2_orig.sh
#   LIMIT=10 bash run_babyvision_v2_orig.sh   # 冒烟
#
# Resume: 自动启用。强制重跑请删除 output_babyvision_v2_orig/stage2_samples.jsonl
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
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output_babyvision_v2_orig}"

MAX_CONCURRENT="${MAX_CONCURRENT:-64}"

export API_TIMEOUT="${API_TIMEOUT:-1200}"
export MAX_RETRIES="${MAX_RETRIES:-20}"

LIMIT="${LIMIT:-0}"

# ── 1. Gemini VL（Call 1 分析 + Call 4 CoT）───────────────────────────────
export GEMINI_VL_ENDPOINT="${GEMINI_VL_ENDPOINT:-https://runway.devops.rednote.life/openai/google/v1:generateContent}"
export GEMINI_VL_MODEL="${GEMINI_VL_MODEL:-gemini-3.1-pro-preview}"
export GEMINI_VL_API_KEY="${GEMINI_VL_API_KEY:-YOUR_GEMINI_API_KEY}"

# ── 2. Qwen3.6-plus（Call 2b text + Selector VL + Verifier VL）────────────
export QWEN_API_BASE="${QWEN_API_BASE:-https://runway.devops.xiaohongshu.com/openai/qwen/v1}"
export QWEN_MODEL="${QWEN_MODEL:-qwen3.6-plus}"
export QWEN_API_KEY="${QWEN_API_KEY:-${OPENAI_API_KEY:-YOUR_API_KEY}}"
export OPENAI_API_KEY="$QWEN_API_KEY"
if [ -z "$QWEN_API_KEY" ]; then
    echo "[run_babyvision_v2_orig] ERROR: QWEN_API_KEY (or OPENAI_API_KEY) env var required"
    exit 1
fi

# ── 3. GPT-Image-2（Call 3b 从原图生成多样化图）──────────────────────────
# 当前用 api.gpt.ge 外网端点 + model=gpt-image-2-c（cheap variant，速率配额更宽）
# Helper 自动识别 endpoint 风格：以 /v1 结尾 → Bearer auth；否则 → Azure 风格 api-key 头 + ?api-version=...
export GPT_IMAGE_ENDPOINT="${GPT_IMAGE_ENDPOINT:-https://api.gpt.ge/v1}"
export GPT_IMAGE_API_KEY="${GPT_IMAGE_API_KEY:-YOUR_GPT_IMAGE_API_KEY}"

echo "[run_babyvision_v2_orig] Parquet:        $PARQUET"
echo "[run_babyvision_v2_orig] Output dir:     $OUTPUT_DIR"
echo "[run_babyvision_v2_orig] Max concurrent: $MAX_CONCURRENT"
echo "[run_babyvision_v2_orig] Max retries:    $MAX_RETRIES"
echo "[run_babyvision_v2_orig] API timeout:    ${API_TIMEOUT}s"
echo "[run_babyvision_v2_orig] Limit:          ${LIMIT} (0=全量)"
echo "[run_babyvision_v2_orig] Gemini VL:  $GEMINI_VL_MODEL @ $GEMINI_VL_ENDPOINT  (key=${GEMINI_VL_API_KEY:0:6}…)"
echo "[run_babyvision_v2_orig] Qwen:       $QWEN_MODEL @ $QWEN_API_BASE  (key=${QWEN_API_KEY:0:6}…)"
echo "[run_babyvision_v2_orig] GPT-Image:  gpt-image-2 @ $GPT_IMAGE_ENDPOINT  (key=${GPT_IMAGE_API_KEY:0:6}…)"
echo ""

cd "$SCRIPT_DIR"

echo "═══════════════════════════════════════════════════════"
echo "  原图多样化 Stage 2（5 张/source，答案不变）"
echo "═══════════════════════════════════════════════════════"

python3 generate_babyvision_v2_orig_stage2.py \
    --parquet         "$PARQUET" \
    --output_dir      "$OUTPUT_DIR" \
    --max_concurrent  "$MAX_CONCURRENT" \
    --max_retries     "$MAX_RETRIES" \
    --limit           "$LIMIT"

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  完成"
echo "═══════════════════════════════════════════════════════"
echo "  输出: $OUTPUT_DIR/stage2_samples.jsonl"
echo "        $OUTPUT_DIR/stage2_images/"
