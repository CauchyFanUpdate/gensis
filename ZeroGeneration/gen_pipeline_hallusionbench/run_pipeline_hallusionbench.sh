#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# ZeroBench Pipeline v5.1 — HallusionBench 适配入口
#
# 数据集: HallusionBench (yes/no 二分类视觉幻觉 benchmark)
#   - 951 张含图像样本 (visual_input=1/2)，跳过 178 张无图样本
#   - gt_answer 0→"no" / 1→"yes"，Stage 1 主要产生 flip_boolean 类型编辑
#
# Stage 1: surgical 答案变化编辑（yes→no 或 no→yes，每 source 2 edits × 1 seed）
# Stage 2: axis_table 驱动的考点保持多样化（4 prompts × 1 seed per stage1 image）
#
# Usage:
#   STAGE=1   bash run_pipeline_hallusionbench.sh   # 仅跑 Stage 1（推荐先冒烟）
#   STAGE=2   bash run_pipeline_hallusionbench.sh   # 仅跑 Stage 2（Stage 1 完成后）
#   STAGE=all bash run_pipeline_hallusionbench.sh   # 自动 Stage 1 → Stage 2
#   LIMIT=10 STAGE=1 bash run_pipeline_hallusionbench.sh   # 冒烟（10 题）
#
# Resume: 自动启用，已写入 jsonl 的 sample_key 会被跳过。
#
# 可覆盖的环境变量：
#   HALLUSIONBENCH_DIR, CONVERTED_PARQUET, OUTPUT_DIR, LIMIT
#   GEMINI_MAX_CONCURRENT, QWEN_MAX_CONCURRENT, GPT_IMAGE_MAX_CONCURRENT
#   GEMINI_VL_ENDPOINT, GEMINI_VL_MODEL, GEMINI_VL_API_KEY
#   QWEN_API_BASE, QWEN_MODEL, QWEN_API_KEY
#   GPT_IMAGE_ENDPOINT, GPT_IMAGE_API_KEY
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

HALLUSIONBENCH_DIR="${HALLUSIONBENCH_DIR:-/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/HallusionBench}"
CONVERTED_PARQUET="${CONVERTED_PARQUET:-$SCRIPT_DIR/_archive/hallusionbench_input.parquet}"
PARQUET="${PARQUET:-$CONVERTED_PARQUET}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/_archive/output_hallusionbench}"

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
        echo "[run_pipeline_hallusionbench] ERROR: STAGE='${STAGE:-<unset>}' 无效，必须为 1 / 2 / all"
        echo ""
        echo "  STAGE=1   bash run_pipeline_hallusionbench.sh   # 仅跑 Stage 1（推荐先冒烟）"
        echo "  STAGE=2   bash run_pipeline_hallusionbench.sh   # 仅跑 Stage 2"
        echo "  STAGE=all bash run_pipeline_hallusionbench.sh   # 自动 Stage 1 → Stage 2"
        exit 1
        ;;
esac

if [ "$STAGE" = "all" ]; then
    echo ""
    echo "⚠⚠⚠  STAGE=all  ⚠⚠⚠"
    echo "Stage 1 跑完后将自动启动 Stage 2，无人工核验环节。"
    echo "前提：已通过 LIMIT=N 冒烟确认 Stage 1 质量。"
    echo ""
fi

# ── API 端点配置 ───────────────────────────────────────────────
export GEMINI_VL_ENDPOINT="${GEMINI_VL_ENDPOINT:-https://runway.devops.rednote.life/openai/google/v1:generateContent}"
export GEMINI_VL_MODEL="${GEMINI_VL_MODEL:-gemini-3.1-pro-preview}"
export GEMINI_VL_API_KEY="${GEMINI_VL_API_KEY:-YOUR_API_KEY}"

export QWEN_API_BASE="${QWEN_API_BASE:-https://runway.devops.xiaohongshu.com/openai/qwen/v1}"
export QWEN_MODEL="${QWEN_MODEL:-qwen3.6-plus}"
export QWEN_API_KEY="${QWEN_API_KEY:-${OPENAI_API_KEY:-YOUR_API_KEY}}"
export OPENAI_API_KEY="$QWEN_API_KEY"
if [ -z "$QWEN_API_KEY" ]; then
    echo "[run_pipeline_hallusionbench] ERROR: QWEN_API_KEY (or OPENAI_API_KEY) env var required"
    echo "                              export QWEN_API_KEY=sk-xxx 后重试"
    exit 1
fi

export GPT_IMAGE_ENDPOINT="${GPT_IMAGE_ENDPOINT:-https://api.gpt.ge/v1}"
export GPT_IMAGE_API_KEY="${GPT_IMAGE_API_KEY:-YOUR_GPT_IMAGE_API_KEY}"

echo "[run_pipeline_hallusionbench] STAGE:               $STAGE"
echo "[run_pipeline_hallusionbench] HallusionBench dir:  $HALLUSIONBENCH_DIR"
echo "[run_pipeline_hallusionbench] Converted parquet:   $CONVERTED_PARQUET"
echo "[run_pipeline_hallusionbench] Output dir:          $OUTPUT_DIR"
echo "[run_pipeline_hallusionbench] Source concurrency:  $MAX_CONCURRENT"
echo "[run_pipeline_hallusionbench] API caps: Gemini=$GEMINI_MAX_CONCURRENT  Qwen=$QWEN_MAX_CONCURRENT  GPT-I-2=$GPT_IMAGE_MAX_CONCURRENT"
echo "[run_pipeline_hallusionbench] API timeout:         ${API_TIMEOUT}s  retries: $MAX_RETRIES"
echo "[run_pipeline_hallusionbench] Limit:               ${LIMIT} (0=全量)"
echo "[run_pipeline_hallusionbench] Gemini VL: $GEMINI_VL_MODEL  @ $GEMINI_VL_ENDPOINT  (key=${GEMINI_VL_API_KEY:0:6}…)"
echo "[run_pipeline_hallusionbench] Qwen:      $QWEN_MODEL  @ $QWEN_API_BASE  (key=${QWEN_API_KEY:0:6}…)"
echo "[run_pipeline_hallusionbench] GPT-I-2:   gpt-image-2-c  @ $GPT_IMAGE_ENDPOINT  (key=${GPT_IMAGE_API_KEY:0:6}…)"

cd "$SCRIPT_DIR"

# ── Step 0: 转换 HallusionBench parquet（已存在则跳过）────────────────────
if [ ! -f "$CONVERTED_PARQUET" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STEP 0: Converting HallusionBench to pipeline format"
    echo "═══════════════════════════════════════════════════════"
    python3 prepare_hallusionbench_input.py \
        --hallusionbench_dir "$HALLUSIONBENCH_DIR" \
        --output             "$CONVERTED_PARQUET"
else
    echo "[run_pipeline_hallusionbench] Converted parquet already exists, skipping conversion."
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
    echo "      STAGE=2 bash run_pipeline_hallusionbench.sh"
    echo ""
}

run_stage2() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 2: axis_table-driven diversification (4 prompts × 1 seed)"
    echo "═══════════════════════════════════════════════════════"
    STAGE1_JSONL="$OUTPUT_DIR/stage1_samples.jsonl"
    if [ ! -f "$STAGE1_JSONL" ]; then
        echo "[run_pipeline_hallusionbench] ERROR: stage1 jsonl not found: $STAGE1_JSONL"
        echo "                              请先运行 STAGE=1 bash run_pipeline_hallusionbench.sh"
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
