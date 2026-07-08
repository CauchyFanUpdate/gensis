#!/bin/bash
# 方案二 pipeline 运行脚本
# Stage 2 Direct (5 cosmetic variants of original images, 1:5 ratio) → Stage 3 Lite (clean + format)
#
# 用法:
#   bash run_pipeline.sh                    # 全量跑
#   LIMIT=10 bash run_pipeline.sh           # 冒烟（10 题）
#   NO_RESUME=1 bash run_pipeline.sh        # 强制重跑
#
# 可选环境变量覆盖:
#   MANIFEST, OUTPUT_DIR, N_MIN_PLANS, N_MAX_PLANS, MAX_CONCURRENT, LIMIT
#   GPT_IMAGE_ENDPOINT, GPT_IMAGE_API_KEY
#   QWEN_API_BASE, QWEN_API_KEY, QWEN_MODEL
#   GEMINI_ENDPOINT, GEMINI_API_KEY, GEMINI_MODEL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 数据与输出路径 ─────────────────────────────────────────────
MANIFEST="${MANIFEST:-/mnt/tidal-alsh01/dataset/perceptionVLM/wanghan/edit_mmmu/selected_data/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/_archive/output_fangan2}"

# ── 并发与重试 ────────────────────────────────────────────────
N_MIN_PLANS="${N_MIN_PLANS:-5}"
N_MAX_PLANS="${N_MAX_PLANS:-5}"
MAX_CONCURRENT="${MAX_CONCURRENT:-64}"
LIMIT="${LIMIT:-0}"

# ── 1. Qwen3.6-plus（Call 2b text + Selector VL + Verifier VL + Stage 3 CoT）─
QWEN_API_BASE="${QWEN_API_BASE:-https://runway.devops.xiaohongshu.com/openai/qwen/v1}"
QWEN_MODEL="${QWEN_MODEL:-qwen3.6-plus}"
QWEN_API_KEY="${QWEN_API_KEY:-${OPENAI_API_KEY:-YOUR_QWEN_API_KEY}}"
export OPENAI_API_KEY="$QWEN_API_KEY"

# ── 2. GPT-Image-2（Call 3b 从原图生成多样化图）──────────────────────────────
# 当前用 api.gpt.ge 外网端点 + model=gpt-image-2-c（cheap variant，速率配额更宽）
# Helper 自动识别 endpoint 风格：以 /v1 结尾 → Bearer auth；否则 → Azure 风格 api-key 头 + ?api-version=...
GPT_IMAGE_ENDPOINT="${GPT_IMAGE_ENDPOINT:-https://api.gpt.ge/v1}"
GPT_IMAGE_API_KEY="${GPT_IMAGE_API_KEY:-YOUR_GPT_IMAGE_API_KEY}"

# ── 3. Gemini VL（Call 1 分析，预留）─────────────────────────────────────────
GEMINI_ENDPOINT="${GEMINI_ENDPOINT:-https://runway.devops.rednote.life/openai/google/v1:generateContent}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.1-pro-preview}"
GEMINI_API_KEY="${GEMINI_API_KEY:-YOUR_GEMINI_API_KEY}"

# ── 启动信息 ──────────────────────────────────────────────────
echo "======================================================================"
echo "方案二 Pipeline"
echo "  MANIFEST:             $MANIFEST"
echo "  OUTPUT_DIR:           $OUTPUT_DIR"
echo "  n_plans:              $N_MIN_PLANS - $N_MAX_PLANS"
echo "  max_concurrent:       $MAX_CONCURRENT"
echo "  limit:                ${LIMIT} (0=全量)"
echo "  Qwen:                 $QWEN_MODEL @ $QWEN_API_BASE  (key=${QWEN_API_KEY:0:6}…)"
echo "  GPT-Image-2:          gpt-image-2-c @ $GPT_IMAGE_ENDPOINT  (key=${GPT_IMAGE_API_KEY:0:6}…)"
echo "  Gemini (预留):        $GEMINI_MODEL  (key=${GEMINI_API_KEY:0:6}…)"
echo "======================================================================"

STAGE2_JSONL="$OUTPUT_DIR/stage2_samples.jsonl"
STAGE3_OUTPUT="$OUTPUT_DIR/stage3_output.jsonl"
STAGE3_REJECTED="$OUTPUT_DIR/stage3_rejected.jsonl"

mkdir -p "$OUTPUT_DIR"

# ── Stage 2 Direct: cosmetic diversification on original images ───────────────
echo ""
echo "[Stage 2 Direct] 对原图生成 $N_MIN_PLANS-$N_MAX_PLANS 个外观多样性变体..."
python3 "$SCRIPT_DIR/generate_stage2_direct.py" \
    --manifest              "$MANIFEST" \
    --output_dir            "$OUTPUT_DIR" \
    --n_min_plans           "$N_MIN_PLANS" \
    --n_max_plans           "$N_MAX_PLANS" \
    --max_concurrent        "$MAX_CONCURRENT" \
    --limit                 "$LIMIT" \
    --gpt_image_endpoint    "$GPT_IMAGE_ENDPOINT" \
    --gpt_image_api_key     "$GPT_IMAGE_API_KEY" \
    --qwen_api_base         "$QWEN_API_BASE" \
    --qwen_api_key          "$QWEN_API_KEY" \
    --qwen_model            "$QWEN_MODEL" \
    --gemini_endpoint       "$GEMINI_ENDPOINT" \
    --gemini_api_key        "$GEMINI_API_KEY" \
    --gemini_model          "$GEMINI_MODEL"
echo "[Stage 2 Direct] 完成 → $STAGE2_JSONL"

# ── Stage 3 Lite: clean CoT + verify + format ─────────────────────────────────
echo ""
echo "[Stage 3 Lite] 清洗 CoT + 答案验证 + 训练格式化..."
python3 "$SCRIPT_DIR/generate_stage3_lite.py" \
    --stage2_input      "$STAGE2_JSONL" \
    --stage3_output     "$STAGE3_OUTPUT" \
    --stage3_rejected   "$STAGE3_REJECTED" \
    --qwen_api_base     "$QWEN_API_BASE" \
    --qwen_api_key      "$QWEN_API_KEY" \
    --qwen_model        "$QWEN_MODEL" \
    --max_concurrent    "$MAX_CONCURRENT"
echo "[Stage 3 Lite] 完成 → $STAGE3_OUTPUT"

echo ""
echo "======================================================================"
echo "方案二 Pipeline 全部完成！"
echo "  训练数据: $STAGE3_OUTPUT"
echo "  拒绝记录: $STAGE3_REJECTED"
echo "======================================================================"
