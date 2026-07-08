#!/bin/bash
# 方案一 pipeline 运行脚本
# Stage 1 (5 answer-changing edits, 1:5 ratio) → Stage 1.5 (option fixup) → Stage 3 (CoT + format)
#
# 用法:
#   bash run_pipeline.sh                    # 全量跑
#   LIMIT=10 bash run_pipeline.sh           # 冒烟（10 题）
#   NO_RESUME=1 bash run_pipeline.sh        # 强制重跑
#
# 可选环境变量覆盖:
#   MANIFEST, OUTPUT_DIR, N_MIN_EDITS, N_MAX_EDITS, MAX_CONCURRENT, LIMIT
#   GPT_IMAGE_ENDPOINT, GPT_IMAGE_API_KEY
#   QWEN_API_BASE, QWEN_API_KEY, QWEN_MODEL
#   GEMINI_ENDPOINT, GEMINI_API_KEY, GEMINI_MODEL

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── 数据与输出路径 ─────────────────────────────────────────────
MANIFEST="${MANIFEST:-/mnt/tidal-alsh01/dataset/perceptionVLM/wanghan/edit_mmmu/selected_data/manifest.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/_archive/output_fangan1}"

# ── 并发与重试 ────────────────────────────────────────────────
N_MIN_EDITS="${N_MIN_EDITS:-5}"
N_MAX_EDITS="${N_MAX_EDITS:-5}"
MAX_CONCURRENT="${MAX_CONCURRENT:-64}"
LIMIT="${LIMIT:-0}"

# ── 1. Qwen3.6-plus（Call 2a text + Modifier VL + Stage 3 CoT）───────────────
QWEN_API_BASE="${QWEN_API_BASE:-https://runway.devops.xiaohongshu.com/openai/qwen/v1}"
QWEN_MODEL="${QWEN_MODEL:-qwen3.6-plus}"
QWEN_API_KEY="${QWEN_API_KEY:-${OPENAI_API_KEY:-YOUR_QWEN_API_KEY}}"
export OPENAI_API_KEY="$QWEN_API_KEY"

# ── 2. GPT-Image-2（Call 3a stage1 编辑）──────────────────────────────────────
# 当前用 api.gpt.ge 外网端点 + model=gpt-image-2-c（cheap variant，速率配额更宽）
# Helper 自动识别 endpoint 风格：以 /v1 结尾 → Bearer auth；否则 → Azure 风格 api-key 头 + ?api-version=...
GPT_IMAGE_ENDPOINT="${GPT_IMAGE_ENDPOINT:-https://api.gpt.ge/v1}"
GPT_IMAGE_API_KEY="${GPT_IMAGE_API_KEY:-YOUR_GPT_IMAGE_API_KEY}"

# ── 3. Gemini VL（Call 1 分析 / Modifier，预留）──────────────────────────────
GEMINI_ENDPOINT="${GEMINI_ENDPOINT:-https://runway.devops.rednote.life/openai/google/v1:generateContent}"
GEMINI_MODEL="${GEMINI_MODEL:-gemini-3.1-pro-preview}"
GEMINI_API_KEY="${GEMINI_API_KEY:-YOUR_GEMINI_API_KEY}"

# ── 启动信息 ──────────────────────────────────────────────────
echo "======================================================================"
echo "方案一 Pipeline"
echo "  MANIFEST:             $MANIFEST"
echo "  OUTPUT_DIR:           $OUTPUT_DIR"
echo "  n_edits:              $N_MIN_EDITS - $N_MAX_EDITS"
echo "  max_concurrent:       $MAX_CONCURRENT"
echo "  limit:                ${LIMIT} (0=全量)"
echo "  Qwen:                 $QWEN_MODEL @ $QWEN_API_BASE  (key=${QWEN_API_KEY:0:6}…)"
echo "  GPT-Image-2:          gpt-image-2-c @ $GPT_IMAGE_ENDPOINT  (key=${GPT_IMAGE_API_KEY:0:6}…)"
echo "  Gemini (预留):        $GEMINI_MODEL  (key=${GEMINI_API_KEY:0:6}…)"
echo "======================================================================"

STAGE1_JSONL="$OUTPUT_DIR/stage1_samples.jsonl"
STAGE1P5_JSONL="$OUTPUT_DIR/stage1p5_samples.jsonl"
STAGE3_OUTPUT="$OUTPUT_DIR/stage3_output.jsonl"
STAGE3_REJECTED="$OUTPUT_DIR/stage3_rejected.jsonl"

mkdir -p "$OUTPUT_DIR"

# ── Stage 1: answer-changing edits (5 per source, 1:5 ratio) ─────────────────
echo ""
echo "[Stage 1] 生成每题 $N_MIN_EDITS-$N_MAX_EDITS 个答案变化编辑变体..."
python3 "$SCRIPT_DIR/generate_stage1.py" \
    --manifest              "$MANIFEST" \
    --output_dir            "$OUTPUT_DIR" \
    --n_min_edits           "$N_MIN_EDITS" \
    --n_max_edits           "$N_MAX_EDITS" \
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
echo "[Stage 1] 完成 → $STAGE1_JSONL"

# ── Stage 1.5: MC option fixup (纯 Python，无 API 调用) ──────────────────────
echo ""
echo "[Stage 1.5] MC 选项修正..."
python3 "$SCRIPT_DIR/generate_stage1p5.py" \
    --stage1_jsonl  "$STAGE1_JSONL" \
    --output_jsonl  "$STAGE1P5_JSONL"
echo "[Stage 1.5] 完成 → $STAGE1P5_JSONL"

# ── Stage 3: CoT 生成 + 验证 + 格式化 ────────────────────────────────────────
echo ""
echo "[Stage 3] CoT 生成 + 答案验证 + 训练格式化..."
python3 "$SCRIPT_DIR/generate_stage3.py" \
    --stage1p5_input    "$STAGE1P5_JSONL" \
    --stage3_output     "$STAGE3_OUTPUT" \
    --stage3_rejected   "$STAGE3_REJECTED" \
    --qwen_api_base     "$QWEN_API_BASE" \
    --qwen_api_key      "$QWEN_API_KEY" \
    --qwen_model        "$QWEN_MODEL" \
    --max_concurrent    "$MAX_CONCURRENT"
echo "[Stage 3] 完成 → $STAGE3_OUTPUT"

echo ""
echo "======================================================================"
echo "方案一 Pipeline 全部完成！"
echo "  训练数据: $STAGE3_OUTPUT"
echo "  拒绝记录: $STAGE3_REJECTED"
echo "======================================================================"
