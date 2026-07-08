#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# ZeroBench Pipeline v5.1 — InfoVQA Stage 1 入口
#
# 数据集: /mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/InfoVQA/data/
#   val-00000-of-00003.parquet … val-00002-of-00003.parquet   (2801 条, 含 GT answer)
#   test-00000-of-00003.parquet … test-00002-of-00003.parquet (3288 条, answer 为空, 自动跳过)
#
# 默认行为: 从 val+test 合并后过滤掉无 answer 的行(即 test 全部跳过),
#           从剩余的 val 2801 条中随机采样 1000 条, 仅跑 Stage 1.
#
# 用法:
#   STAGE=1   bash run_pipeline_v5_infovqa.sh          # 仅跑 Stage 1 (默认)
#   STAGE=2   bash run_pipeline_v5_infovqa.sh          # 仅跑 Stage 2 (Stage 1 完成后)
#   STAGE=all bash run_pipeline_v5_infovqa.sh          # Stage 1 → Stage 2
#   LIMIT=10 STAGE=1 bash run_pipeline_v5_infovqa.sh  # 冒烟测试
#
# 可选环境变量:
#   INFOVQA_DATA_DIR  InfoVQA parquet 目录 (默认见下)
#   SPLITS            加载哪些 split, 空格分隔 (默认 "val test")
#   SAMPLE            随机采样条数 (默认 1000; 0=不采样)
#   OUTPUT_DIR        输出目录 (默认 _archive/output_v5_infovqa)
#   MAX_CONCURRENT    source 级并发数 (默认 16)
#   LIMIT             每次运行最多处理多少题 (0=全量)
#   QWEN_API_BASE / QWEN_MODEL / QWEN_API_KEY
#   GPT_IMAGE_ENDPOINT / GPT_IMAGE_API_KEY
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INFOVQA_DATA_DIR="${INFOVQA_DATA_DIR:-/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/InfoVQA/data}"
SPLITS="${SPLITS:-val test}"
SAMPLE="${SAMPLE:-1000}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/_archive/output_v5_infovqa}"
MAX_CONCURRENT="${MAX_CONCURRENT:-16}"
LIMIT="${LIMIT:-0}"

export GEMINI_MAX_CONCURRENT="${GEMINI_MAX_CONCURRENT:-16}"
export QWEN_MAX_CONCURRENT="${QWEN_MAX_CONCURRENT:-64}"
export GPT_IMAGE_MAX_CONCURRENT="${GPT_IMAGE_MAX_CONCURRENT:-16}"
export API_TIMEOUT="${API_TIMEOUT:-1200}"
export MAX_RETRIES="${MAX_RETRIES:-6}"

case "${STAGE:-1}" in
    1|2|all) ;;
    *)
        echo "[run_pipeline_v5_infovqa] ERROR: STAGE='${STAGE}' 无效，必须是 1 / 2 / all。"
        echo ""
        echo "  STAGE=1   bash run_pipeline_v5_infovqa.sh"
        echo "  STAGE=2   bash run_pipeline_v5_infovqa.sh"
        echo "  STAGE=all bash run_pipeline_v5_infovqa.sh"
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
    echo "[run_pipeline_v5_infovqa] ERROR: QWEN_API_KEY (or OPENAI_API_KEY) required"
    exit 1
fi

export GPT_IMAGE_ENDPOINT="${GPT_IMAGE_ENDPOINT:-https://api.gpt.ge/v1}"
export GPT_IMAGE_API_KEY="${GPT_IMAGE_API_KEY:-YOUR_GPT_IMAGE_API_KEY}"

echo "[run_pipeline_v5_infovqa] STAGE:              ${STAGE:-1}"
echo "[run_pipeline_v5_infovqa] InfoVQA data dir:   $INFOVQA_DATA_DIR"
echo "[run_pipeline_v5_infovqa] Splits:             $SPLITS"
echo "[run_pipeline_v5_infovqa] Sample:             $SAMPLE (0=不采样; test split 无 GT 自动跳过)"
echo "[run_pipeline_v5_infovqa] Output dir:         $OUTPUT_DIR"
echo "[run_pipeline_v5_infovqa] Concurrency:        $MAX_CONCURRENT"
echo "[run_pipeline_v5_infovqa] Limit:              $LIMIT (0=全量)"
echo "[run_pipeline_v5_infovqa] Qwen:        $QWEN_MODEL @ $QWEN_API_BASE"
echo "[run_pipeline_v5_infovqa] GPT-Image-2: @ $GPT_IMAGE_ENDPOINT"

cd "$SCRIPT_DIR"

run_stage1() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 1: InfoVQA surgical answer-changing edit"
    echo "═══════════════════════════════════════════════════════"
    # shellcheck disable=SC2086
    python3 generate_samples_v5_stage1.py \
        --data_dir       "$INFOVQA_DATA_DIR" \
        --splits         $SPLITS \
        --data_format    infovqa \
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
    echo "      STAGE=2 bash run_pipeline_v5_infovqa.sh"
    echo ""
}

run_stage2() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 2: InfoVQA axis-table diversification"
    echo "═══════════════════════════════════════════════════════"
    STAGE1_JSONL="$OUTPUT_DIR/stage1_samples.jsonl"
    if [ ! -f "$STAGE1_JSONL" ]; then
        echo "[run_pipeline_v5_infovqa] ERROR: stage1 jsonl not found: $STAGE1_JSONL"
        echo "                     请先运行 STAGE=1 bash run_pipeline_v5_infovqa.sh"
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

case "${STAGE:-1}" in
    1)    run_stage1 ;;
    2)    run_stage2 ;;
    all)  run_stage1 && run_stage2 ;;
esac
