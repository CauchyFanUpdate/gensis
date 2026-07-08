#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# MMMU Pipeline — Stage 1 (emma_v3)
#
# 1 original image → 5 generated images (1:5 ratio)
# max_concurrent=64  max_retries=20
#
# Usage:
#   bash run_stage1_emma_v3.sh                    # 全量跑
#   LIMIT=10 bash run_stage1_emma_v3.sh           # 冒烟（10 题）
#   NO_RESUME=1 bash run_stage1_emma_v3.sh        # 强制重跑（忽略已有结果）
#
# Resume: 自动启用，已写入 jsonl 的 sample_key 会跳过。
#         强制重跑: NO_RESUME=1 bash run_stage1_emma_v3.sh
#
# 可选环境变量覆盖:
#   MANIFEST, OUTPUT_DIR, MAX_CONCURRENT, LIMIT
#   QWEN_API_BASE, QWEN_MODEL, QWEN_API_KEY
#   GPT_IMAGE_ENDPOINT, GPT_IMAGE_API_KEY
#   GEMINI_VL_ENDPOINT, GEMINI_VL_MODEL, GEMINI_VL_API_KEY
#   QWEN_MAX_CONCURRENT, GPT_IMAGE_MAX_CONCURRENT
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 数据与输出路径 ─────────────────────────────────────────────
MANIFEST="${MANIFEST:-/mnt/tidal-alsh01/dataset/perceptionVLM/wanghan/edit_mmmu/selected_data/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/output_emma_v3}"

# ── 并发与重试 ────────────────────────────────────────────────
MAX_CONCURRENT="${MAX_CONCURRENT:-64}"
export MAX_RETRIES="${MAX_RETRIES:-20}"
export API_TIMEOUT="${API_TIMEOUT:-1200}"

export QWEN_MAX_CONCURRENT="${QWEN_MAX_CONCURRENT:-256}"
export GPT_IMAGE_MAX_CONCURRENT="${GPT_IMAGE_MAX_CONCURRENT:-16}"

LIMIT="${LIMIT:-0}"
NO_RESUME="${NO_RESUME:-0}"

# ── API 凭据 ──────────────────────────────────────────────────

# 1. Qwen3.6-plus（Call 1 VL 分析 + Call 2a 编辑文本 + Modifier 双路验证）
export QWEN_API_BASE="${QWEN_API_BASE:-https://runway.devops.xiaohongshu.com/openai/qwen/v1}"
export QWEN_MODEL="${QWEN_MODEL:-qwen3.6-plus}"
export QWEN_API_KEY="${QWEN_API_KEY:-YOUR_QWEN_API_KEY}"
export OPENAI_API_KEY="$QWEN_API_KEY"

# 2. GPT-Image-2（Call 3a 图像编辑）
export GPT_IMAGE_ENDPOINT="${GPT_IMAGE_ENDPOINT:-https://runway.devops.rednote.life}"
export GPT_IMAGE_API_KEY="${GPT_IMAGE_API_KEY:-YOUR_GPT_IMAGE_API_KEY}"

# 3. Gemini 3.0 Pro（预留，当前 Stage 1 未使用）
export GEMINI_VL_ENDPOINT="${GEMINI_VL_ENDPOINT:-https://runway.devops.rednote.life/openai/google/v1:generateContent}"
export GEMINI_VL_MODEL="${GEMINI_VL_MODEL:-gemini-3.0-pro}"
export GEMINI_VL_API_KEY="${GEMINI_VL_API_KEY:-YOUR_GEMINI_API_KEY}"

# ── 启动信息 ──────────────────────────────────────────────────
echo "[run_stage1_emma_v3] Manifest:          $MANIFEST"
echo "[run_stage1_emma_v3] Output dir:        $OUTPUT_DIR"
echo "[run_stage1_emma_v3] Source concurrent: $MAX_CONCURRENT  (qwen_cap=$QWEN_MAX_CONCURRENT  gpt_img_cap=$GPT_IMAGE_MAX_CONCURRENT)"
echo "[run_stage1_emma_v3] Retries:           $MAX_RETRIES  timeout: ${API_TIMEOUT}s"
echo "[run_stage1_emma_v3] Limit:             ${LIMIT} (0=全量)"
echo "[run_stage1_emma_v3] Ratio:             1:5 (n_edits_per_source=5)"
echo "[run_stage1_emma_v3] Qwen:              $QWEN_MODEL @ $QWEN_API_BASE  (key=${QWEN_API_KEY:0:6}…)"
echo "[run_stage1_emma_v3] GPT-Image-2:       @ $GPT_IMAGE_ENDPOINT         (key=${GPT_IMAGE_API_KEY:0:6}…)"
echo "[run_stage1_emma_v3] Gemini (reserved): $GEMINI_VL_MODEL              (key=${GEMINI_VL_API_KEY:0:6}…)"

MMMU_PRO_DIR="$(cd "$SCRIPT_DIR/../gen_pipeline_mmmu_pro" && pwd)"

cd "$MMMU_PRO_DIR"

# ── 构造 --no_resume 参数 ──────────────────────────────────────
RESUME_FLAG=""
if [ "$NO_RESUME" = "1" ]; then
    RESUME_FLAG="--no_resume"
    echo "[run_stage1_emma_v3] NO_RESUME=1: 强制重跑，忽略已有结果"
fi

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  STAGE 1: surgical answer-changing edit (1:5 ratio)"
echo "═══════════════════════════════════════════════════════"

python3 stage1_edit_emma_v3.py \
    --manifest        "$MANIFEST" \
    --output_dir      "$OUTPUT_DIR" \
    --max_concurrent  "$MAX_CONCURRENT" \
    --max_retries     "$MAX_RETRIES" \
    --api_timeout     "$API_TIMEOUT" \
    --n_edits_per_source 5 \
    --limit           "$LIMIT" \
    $RESUME_FLAG

echo ""
echo "═══════════════════════════════════════════════════════"
echo "  STAGE 1 完成"
echo "═══════════════════════════════════════════════════════"
echo "  输出 JSONL:  $OUTPUT_DIR/stage1_samples.jsonl"
echo "  输出图像:    $OUTPUT_DIR/stage1_images/"
echo ""
