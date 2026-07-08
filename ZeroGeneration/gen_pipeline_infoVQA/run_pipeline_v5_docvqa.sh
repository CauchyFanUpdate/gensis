#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# ZeroBench Pipeline v5.1 — DocVQA 两阶段串行入口
#
# 数据集: /mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/DocVQA/DocVQA/
#   train-00000-of-00012.parquet … train-00011-of-00012.parquet
#   (validation / test 同目录，按需通过 SPLITS 指定)
#
# 用法:
#   STAGE=1   bash run_pipeline_v5_docvqa.sh          # 仅跑 Stage 1 (推荐先 LIMIT=N 冒烟)
#   STAGE=2   bash run_pipeline_v5_docvqa.sh          # 仅跑 Stage 2 (Stage 1 完成后)
#   STAGE=all bash run_pipeline_v5_docvqa.sh          # 自动 Stage 1 → Stage 2
#   LIMIT=10 STAGE=1 bash run_pipeline_v5_docvqa.sh  # 冒烟
#
# 可选环境变量:
#   DOCVQA_DATA_DIR   DocVQA parquet 目录 (默认见下)
#   SPLITS            加载哪些 split, 空格分隔 (默认 "train")
#   OUTPUT_DIR        输出目录 (默认 _archive/output_v5_docvqa)
#   MAX_CONCURRENT    source 级并发数 (默认 16)
#   LIMIT             每次运行最多处理多少题 (0=全量)
#   GEMINI_MAX_CONCURRENT / QWEN_MAX_CONCURRENT / GPT_IMAGE_MAX_CONCURRENT
#   GEMINI_VL_ENDPOINT / GEMINI_VL_MODEL / GEMINI_VL_API_KEY
#   QWEN_API_BASE / QWEN_MODEL / QWEN_API_KEY
#   GPT_IMAGE_ENDPOINT / GPT_IMAGE_API_KEY
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DOCVQA_DATA_DIR="${DOCVQA_DATA_DIR:-/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/DocVQA/DocVQA}"
SPLITS="${SPLITS:-train}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/_archive/output_v5_docvqa}"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
SAMPLE="${SAMPLE:-0}"

export GEMINI_MAX_CONCURRENT="${GEMINI_MAX_CONCURRENT:-16}"
export QWEN_MAX_CONCURRENT="${QWEN_MAX_CONCURRENT:-64}"
export GPT_IMAGE_MAX_CONCURRENT="${GPT_IMAGE_MAX_CONCURRENT:-16}"
export API_TIMEOUT="${API_TIMEOUT:-1200}"
export MAX_RETRIES="${MAX_RETRIES:-6}"

LIMIT="${LIMIT:-0}"

case "${STAGE:-}" in
    1|2|all) ;;
    *)
        echo "[run_pipeline_v5_docvqa] ERROR: STAGE='${STAGE:-<unset>}' 无效，必须是 1 / 2 / all。"
        echo ""
        echo "  STAGE=1   bash run_pipeline_v5_docvqa.sh   # 仅 Stage 1"
        echo "  STAGE=2   bash run_pipeline_v5_docvqa.sh   # 仅 Stage 2"
        echo "  STAGE=all bash run_pipeline_v5_docvqa.sh   # 全量 Stage 1 → Stage 2"
        exit 1
        ;;
esac

# ── API 配置 ──────────────────────────────────────────────────────────────────
export GEMINI_VL_ENDPOINT="${GEMINI_VL_ENDPOINT:-https://runway.devops.rednote.life/openai/google/v1:generateContent}"
export GEMINI_VL_MODEL="${GEMINI_VL_MODEL:-gemini-3.1-pro-preview}"
export GEMINI_VL_API_KEY="${GEMINI_VL_API_KEY:-YOUR_API_KEY}"

export QWEN_API_BASE="${QWEN_API_BASE:-https://runway.devops.xiaohongshu.com/openai/qwen/v1}"
export QWEN_MODEL="${QWEN_MODEL:-qwen3.6-plus}"
export QWEN_API_KEY="${QWEN_API_KEY:-${OPENAI_API_KEY:-YOUR_API_KEY}}"
export OPENAI_API_KEY="$QWEN_API_KEY"
if [ -z "$QWEN_API_KEY" ]; then
    echo "[run_pipeline_v5_docvqa] ERROR: QWEN_API_KEY (or OPENAI_API_KEY) required"
    exit 1
fi

export GPT_IMAGE_ENDPOINT="${GPT_IMAGE_ENDPOINT:-https://api.gpt.ge/v1}"
export GPT_IMAGE_API_KEY="${GPT_IMAGE_API_KEY:-YOUR_GPT_IMAGE_API_KEY}"

echo "[run_pipeline_v5_docvqa] STAGE:              $STAGE"
echo "[run_pipeline_v5_docvqa] DocVQA data dir:    $DOCVQA_DATA_DIR"
echo "[run_pipeline_v5_docvqa] Splits:             $SPLITS"
echo "[run_pipeline_v5_docvqa] Output dir:         $OUTPUT_DIR"
echo "[run_pipeline_v5_docvqa] Concurrency:        $MAX_CONCURRENT"
echo "[run_pipeline_v5_docvqa] API caps: Gemini=$GEMINI_MAX_CONCURRENT  Qwen=$QWEN_MAX_CONCURRENT  GPT-I-2=$GPT_IMAGE_MAX_CONCURRENT"
echo "[run_pipeline_v5_docvqa] Sample:             ${SAMPLE} (0=不采样)"
echo "[run_pipeline_v5_docvqa] Limit:              ${LIMIT} (0=全量)"
echo "[run_pipeline_v5_docvqa] Gemini VL:   $GEMINI_VL_MODEL"
echo "[run_pipeline_v5_docvqa] Qwen:        $QWEN_MODEL @ $QWEN_API_BASE"
echo "[run_pipeline_v5_docvqa] GPT-Image-2: @ $GPT_IMAGE_ENDPOINT"

cd "$SCRIPT_DIR"

run_stage1() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 1: DocVQA surgical answer-changing edit"
    echo "═══════════════════════════════════════════════════════"
    # SPLITS may be space-separated; pass each as a separate --splits token
    # shellcheck disable=SC2086
    python3 generate_samples_v5_stage1.py \
        --data_dir       "$DOCVQA_DATA_DIR" \
        --splits         $SPLITS \
        --output_dir     "$OUTPUT_DIR" \
        --max_concurrent "$MAX_CONCURRENT" \
        --sample         "$SAMPLE" \
        --limit          "$LIMIT"

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 1 完成"
    echo "═══════════════════════════════════════════════════════"
    echo "  输出: $OUTPUT_DIR/stage1_samples.jsonl"
    echo "        $OUTPUT_DIR/stage1_images/"
    echo ""
    echo "  ⚠ 请人工核验 stage1 输出后，再手动启动 Stage 2:"
    echo "      STAGE=2 bash run_pipeline_v5_docvqa.sh"
    echo ""
}

run_stage2() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 2: DocVQA axis-table diversification"
    echo "═══════════════════════════════════════════════════════"
    STAGE1_JSONL="$OUTPUT_DIR/stage1_samples.jsonl"
    if [ ! -f "$STAGE1_JSONL" ]; then
        echo "[run_pipeline_v5_docvqa] ERROR: stage1 jsonl not found: $STAGE1_JSONL"
        echo "                     请先运行 STAGE=1 bash run_pipeline_v5_docvqa.sh"
        exit 1
    fi
    python3 generate_samples_v5_stage2.py \
        --stage1_jsonl   "$STAGE1_JSONL" \
        --output_dir     "$OUTPUT_DIR" \
        --max_concurrent "$MAX_CONCURRENT" \
        --limit          "$LIMIT"

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
