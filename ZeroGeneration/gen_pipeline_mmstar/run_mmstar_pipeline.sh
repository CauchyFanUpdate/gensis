#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# ZeroBench Pipeline v5.1 — MMStar 两阶段串行入口
#
# Stage 1: surgical 答案变化编辑（每 source 2 edits × 1 seed）
#          + Call 1 产 frozen_predicate(slots) + question_template + axis_table(19)
# Stage 2: axis_table 驱动的考点保持多样化
#          - Selector (Qwen-VL): 看 stage1 图，输出 4 套 axis valuation + slot 实例化
#          - 每 stage1 输出 4 prompts × 1 seed = 4 张训练图
#          - Verifier (Qwen-VL, 4 sub-Q): 核对考点是否还在（不验答案数值）
#
# MMStar 数据集说明：
#   - 1500 题，6 大类各 250 题（coarse/fine-grained perception, instance/logical
#     reasoning, math, science & technology）
#   - 单图 + 4 选 1 文字选项（A/B/C/D），选项已内嵌于 question_text
#   - 无子图（choice 以文字形式呈现，非独立图像）
#
# Usage:
#   STAGE=1   bash run_mmstar_pipeline.sh          # 仅跑 Stage 1（推荐先用此 + LIMIT=N 冒烟）
#   STAGE=2   bash run_mmstar_pipeline.sh          # 仅跑 Stage 2（必须先有 Stage 1 输出）
#   STAGE=all bash run_mmstar_pipeline.sh          # 自动 Stage 1 → Stage 2
#   LIMIT=10 STAGE=1 bash run_mmstar_pipeline.sh  # 冒烟（10 题）
#
# Resume: 自动启用，已写入 jsonl 的 sample_key 会被跳过。
#         （强制重跑请删除 stage1_samples.jsonl / stage2_samples.jsonl 或加 --no_resume）
#
# 可选环境变量：
#   MMSTAR_PARQUET, CONVERTED_PARQUET, OUTPUT_DIR, MAX_CONCURRENT, LIMIT
#   GEMINI_MAX_CONCURRENT, QWEN_MAX_CONCURRENT, GPT_IMAGE_MAX_CONCURRENT
#   GEMINI_VL_ENDPOINT, GPT_IMAGE_ENDPOINT, QWEN_API_BASE
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MMSTAR_PARQUET="${MMSTAR_PARQUET:-/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/MMStar/mmstar.parquet}"
CONVERTED_PARQUET="${CONVERTED_PARQUET:-$SCRIPT_DIR/_archive/mmstar_input.parquet}"
PARQUET="${PARQUET:-$CONVERTED_PARQUET}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/_archive/output_mmstar}"

# Source-level concurrency (number of source questions processed in parallel)
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"

export GEMINI_MAX_CONCURRENT="${GEMINI_MAX_CONCURRENT:-16}"
export QWEN_MAX_CONCURRENT="${QWEN_MAX_CONCURRENT:-64}"
export GPT_IMAGE_MAX_CONCURRENT="${GPT_IMAGE_MAX_CONCURRENT:-16}"

export API_TIMEOUT="${API_TIMEOUT:-1200}"
export MAX_RETRIES="${MAX_RETRIES:-6}"

LIMIT="${LIMIT:-0}"

# STAGE 必须显式指定为 1 / 2 / all
case "${STAGE:-}" in
    1|2|all) ;;
    *)
        echo "[run_mmstar] ERROR: STAGE='${STAGE:-<unset>}' invalid. 必须显式指定为 1 / 2 / all。"
        echo ""
        echo "  STAGE=1   bash run_mmstar_pipeline.sh          # 仅跑 Stage 1 (推荐先冒烟)"
        echo "  STAGE=2   bash run_mmstar_pipeline.sh          # 仅跑 Stage 2 (Stage 1 完成后)"
        echo "  STAGE=all bash run_mmstar_pipeline.sh          # 自动 Stage 1 → Stage 2"
        echo "  LIMIT=10 STAGE=1 bash run_mmstar_pipeline.sh  # 冒烟测试 10 题"
        exit 1
        ;;
esac

if [ "$STAGE" = "all" ]; then
    echo ""
    echo "⚠⚠⚠  STAGE=all  ⚠⚠⚠"
    echo "Stage 1 跑完后将自动启动 Stage 2，无人工核验环节。"
    echo "前提：你已经通过 LIMIT=N 冒烟确认 Stage 1 质量。"
    echo ""
fi

# ─────────────────────────────────────────────────────────────────────────
# 三种 API：URL + KEY + MODEL 三件套
# ─────────────────────────────────────────────────────────────────────────

# ── 1. Gemini VL（Stage 2 Call 4 CoT 专用）──────────────────────────────
export GEMINI_VL_ENDPOINT="${GEMINI_VL_ENDPOINT:-https://runway.devops.rednote.life/openai/google/v1:generateContent}"
export GEMINI_VL_MODEL="${GEMINI_VL_MODEL:-gemini-3.1-pro-preview}"
export GEMINI_VL_API_KEY="${GEMINI_VL_API_KEY:-YOUR_API_KEY}"

# ── 2. Qwen3.6-plus（Call 2a/2b text + Modifier + Selector + Verifier VL）
export QWEN_API_BASE="${QWEN_API_BASE:-https://runway.devops.xiaohongshu.com/openai/qwen/v1}"
export QWEN_MODEL="${QWEN_MODEL:-qwen3.6-plus}"
export QWEN_API_KEY="${QWEN_API_KEY:-${OPENAI_API_KEY:-YOUR_API_KEY}}"
export OPENAI_API_KEY="$QWEN_API_KEY"
if [ -z "$QWEN_API_KEY" ]; then
    echo "[run_mmstar] ERROR: QWEN_API_KEY (or OPENAI_API_KEY) env var required for Qwen calls"
    echo "             export QWEN_API_KEY=sk-xxx 后重试"
    exit 1
fi

# ── 3. GPT-Image-2（Call 3a stage1 编辑 / Call 3b stage2 编辑）────────────
# 使用外部 api.gpt.ge/v1 端点（标准 OpenAI Bearer 风格），参考 gen_pipeline_emma
export GPT_IMAGE_ENDPOINT="${GPT_IMAGE_ENDPOINT:-https://api.gpt.ge/v1}"
export GPT_IMAGE_API_KEY="${GPT_IMAGE_API_KEY:-YOUR_GPT_IMAGE_API_KEY}"

echo "[run_mmstar] STAGE:                 $STAGE"
echo "[run_mmstar] MMStar parquet:        $MMSTAR_PARQUET"
echo "[run_mmstar] Converted parquet:     $CONVERTED_PARQUET"
echo "[run_mmstar] Parquet:               $PARQUET"
echo "[run_mmstar] Output dir:            $OUTPUT_DIR"
echo "[run_mmstar] Source concurrency:    $MAX_CONCURRENT"
echo "[run_mmstar] API caps: Gemini=$GEMINI_MAX_CONCURRENT  Qwen=$QWEN_MAX_CONCURRENT  GPT-I-2=$GPT_IMAGE_MAX_CONCURRENT"
echo "[run_mmstar] API timeout:           ${API_TIMEOUT}s  retries: $MAX_RETRIES"
echo "[run_mmstar] Limit:                 ${LIMIT} (0=全量)"
echo "[run_mmstar] Gemini VL:    $GEMINI_VL_MODEL  @ $GEMINI_VL_ENDPOINT  (key=${GEMINI_VL_API_KEY:0:6}…)"
echo "[run_mmstar] Qwen:         $QWEN_MODEL  @ $QWEN_API_BASE  (key=${QWEN_API_KEY:0:6}…)"
echo "[run_mmstar] GPT-Image-2:  gpt-image-2-c  @ $GPT_IMAGE_ENDPOINT  (key=${GPT_IMAGE_API_KEY:0:6}…)"

cd "$SCRIPT_DIR"

# ── Step 0: convert MMStar parquet (skip if already done) ────────────────
if [ ! -f "$CONVERTED_PARQUET" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STEP 0: Converting MMStar to pipeline format"
    echo "═══════════════════════════════════════════════════════"
    python3 prepare_mmstar_input.py \
        --mmstar_parquet "$MMSTAR_PARQUET" \
        --output         "$CONVERTED_PARQUET"
else
    echo "[run_mmstar] Converted parquet already exists, skipping conversion."
fi

run_stage1() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 1: surgical answer-changing edit (单 seed)"
    echo "═══════════════════════════════════════════════════════"
    python3 generate_samples_v5_stage1.py \
        --parquet         "$PARQUET" \
        --output_dir      "$OUTPUT_DIR" \
        --max_concurrent  "$MAX_CONCURRENT" \
        --limit           "$LIMIT"

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 1 完成"
    echo "═══════════════════════════════════════════════════════"
    echo "  输出: $OUTPUT_DIR/stage1_samples.jsonl"
    echo "        $OUTPUT_DIR/stage1_images/"
    echo ""
    echo "  ⚠ 请人工核验 stage1 输出后，再手动启动 Stage 2:"
    echo ""
    echo "      STAGE=2 bash run_mmstar_pipeline.sh"
    echo ""
}

run_stage2() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 2: axis_table-driven diversification (4 prompts × 1 seed per stage1 image)"
    echo "═══════════════════════════════════════════════════════"
    STAGE1_JSONL="$OUTPUT_DIR/stage1_samples.jsonl"
    if [ ! -f "$STAGE1_JSONL" ]; then
        echo "[run_mmstar] ERROR: stage1 jsonl not found: $STAGE1_JSONL"
        echo "             请先运行 STAGE=1 bash run_mmstar_pipeline.sh"
        exit 1
    fi
    python3 generate_samples_v5_stage2.py \
        --stage1_jsonl    "$STAGE1_JSONL" \
        --output_dir      "$OUTPUT_DIR" \
        --max_concurrent  "$MAX_CONCURRENT" \
        --limit           "$LIMIT"

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 2 完成"
    echo "═══════════════════════════════════════════════════════"
    echo "  输出: $OUTPUT_DIR/stage2_samples.jsonl"
    echo "        $OUTPUT_DIR/stage2_images/"
}

case "$STAGE" in
    1)    run_stage1 ;;
    2)    run_stage2 ;;
    all)  run_stage1 && run_stage2 ;;
esac
