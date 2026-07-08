#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────
# LogicVista Pipeline Runner
#
# Step 0: convert LogicVista parquet → pipeline-compatible parquet
# Step 1: Stage 1 (surgical answer-changing edit)
# Step 2: Stage 2 (axis_table-driven diversification)
#
# Usage:
#   STAGE=1   bash run_logicvista_pipeline.sh
#   STAGE=2   bash run_logicvista_pipeline.sh
#   STAGE=all bash run_logicvista_pipeline.sh
#   LIMIT=2 STAGE=1 bash run_logicvista_pipeline.sh   # smoke test
#
# Resume: automatic (already-written sample_keys are skipped).
# To force re-run: delete the output jsonl file or add --no_resume.
#
# Optional env overrides (same as run_pipeline_v5.sh):
#   LOGICVISTA_DIR, OUTPUT_DIR, MAX_CONCURRENT, LIMIT
#   GEMINI_MAX_CONCURRENT, QWEN_MAX_CONCURRENT, GPT_IMAGE_MAX_CONCURRENT
#   GEMINI_VL_ENDPOINT, GPT_IMAGE_ENDPOINT, QWEN_API_BASE
# ─────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

LOGICVISTA_DIR="${LOGICVISTA_DIR:-/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/LogicVista}"
CONVERTED_PARQUET="${CONVERTED_PARQUET:-$SCRIPT_DIR/_archive/logicvista_input.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/_archive/output_logicvista}"

MAX_CONCURRENT="${MAX_CONCURRENT:-16}"

export GEMINI_MAX_CONCURRENT="${GEMINI_MAX_CONCURRENT:-16}"
export QWEN_MAX_CONCURRENT="${QWEN_MAX_CONCURRENT:-64}"
export GPT_IMAGE_MAX_CONCURRENT="${GPT_IMAGE_MAX_CONCURRENT:-16}"

export API_TIMEOUT="${API_TIMEOUT:-1200}"
export MAX_RETRIES="${MAX_RETRIES:-6}"

LIMIT="${LIMIT:-0}"

# STAGE must be explicitly specified
case "${STAGE:-}" in
    1|2|all) ;;
    *)
        echo "[run_logicvista] ERROR: STAGE='${STAGE:-<unset>}' invalid. Must be 1 / 2 / all."
        echo ""
        echo "  STAGE=1   bash run_logicvista_pipeline.sh   # Stage 1 only (recommend smoke test first)"
        echo "  STAGE=2   bash run_logicvista_pipeline.sh   # Stage 2 only (after Stage 1 complete)"
        echo "  STAGE=all bash run_logicvista_pipeline.sh   # Stage 1 then Stage 2 automatically"
        exit 1
        ;;
esac

# ── API credentials ───────────────────────────────────────────────────────────
export GEMINI_VL_ENDPOINT="${GEMINI_VL_ENDPOINT:-https://runway.devops.rednote.life/openai/google/v1:generateContent}"
export GEMINI_VL_MODEL="${GEMINI_VL_MODEL:-gemini-3.1-pro-preview}"
export GEMINI_VL_API_KEY="${GEMINI_VL_API_KEY:-YOUR_API_KEY}"

export QWEN_API_BASE="${QWEN_API_BASE:-https://runway.devops.xiaohongshu.com/openai/qwen/v1}"
export QWEN_MODEL="${QWEN_MODEL:-qwen3.6-plus}"
export QWEN_API_KEY="${QWEN_API_KEY:-${OPENAI_API_KEY:-YOUR_API_KEY}}"
export OPENAI_API_KEY="$QWEN_API_KEY"

if [ -z "$QWEN_API_KEY" ]; then
    echo "[run_logicvista] ERROR: QWEN_API_KEY (or OPENAI_API_KEY) env var required"
    exit 1
fi

export GPT_IMAGE_ENDPOINT="${GPT_IMAGE_ENDPOINT:-https://api.gpt.ge/v1}"
export GPT_IMAGE_API_KEY="${GPT_IMAGE_API_KEY:-YOUR_GPT_IMAGE_API_KEY}"

echo "[run_logicvista] STAGE:              $STAGE"
echo "[run_logicvista] LogicVista dir:     $LOGICVISTA_DIR"
echo "[run_logicvista] Converted parquet:  $CONVERTED_PARQUET"
echo "[run_logicvista] Output dir:         $OUTPUT_DIR"
echo "[run_logicvista] Source concurrency: $MAX_CONCURRENT"
echo "[run_logicvista] API caps: Gemini=$GEMINI_MAX_CONCURRENT  Qwen=$QWEN_MAX_CONCURRENT  GPT-I-2=$GPT_IMAGE_MAX_CONCURRENT"
echo "[run_logicvista] API timeout:        ${API_TIMEOUT}s  retries: $MAX_RETRIES"
echo "[run_logicvista] Limit:              ${LIMIT} (0=all)"

cd "$SCRIPT_DIR"

# ── Step 0: convert LogicVista parquet (skip if already done) ─────────────────
if [ ! -f "$CONVERTED_PARQUET" ]; then
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STEP 0: Converting LogicVista to pipeline format"
    echo "═══════════════════════════════════════════════════════"
    python3 prepare_logicvista_input.py \
        --logicvista_dir "$LOGICVISTA_DIR" \
        --output         "$CONVERTED_PARQUET"
else
    echo "[run_logicvista] Converted parquet already exists, skipping conversion."
fi

run_stage1() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 1: surgical answer-changing edit"
    echo "═══════════════════════════════════════════════════════"
    python3 generate_samples_v5_stage1.py \
        --parquet        "$CONVERTED_PARQUET" \
        --output_dir     "$OUTPUT_DIR" \
        --max_concurrent "$MAX_CONCURRENT" \
        --limit          "$LIMIT"

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 1 complete"
    echo "═══════════════════════════════════════════════════════"
    echo "  Output: $OUTPUT_DIR/stage1_samples.jsonl"
    echo "          $OUTPUT_DIR/stage1_images/"
    echo ""
    echo "  Review stage1 output, then run:"
    echo "      STAGE=2 bash run_logicvista_pipeline.sh"
    echo ""
}

run_stage2() {
    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 2: axis_table-driven diversification"
    echo "═══════════════════════════════════════════════════════"
    STAGE1_JSONL="$OUTPUT_DIR/stage1_samples.jsonl"
    if [ ! -f "$STAGE1_JSONL" ]; then
        echo "[run_logicvista] ERROR: stage1 jsonl not found: $STAGE1_JSONL"
        echo "                 Run STAGE=1 first."
        exit 1
    fi
    python3 generate_samples_v5_stage2.py \
        --stage1_jsonl   "$STAGE1_JSONL" \
        --output_dir     "$OUTPUT_DIR" \
        --max_concurrent "$MAX_CONCURRENT" \
        --limit          "$LIMIT"

    echo ""
    echo "═══════════════════════════════════════════════════════"
    echo "  STAGE 2 complete"
    echo "═══════════════════════════════════════════════════════"
    echo "  Output: $OUTPUT_DIR/stage2_samples.jsonl"
    echo "          $OUTPUT_DIR/stage2_images/"
}

case "$STAGE" in
    1)    run_stage1 ;;
    2)    run_stage2 ;;
    all)  run_stage1 && run_stage2 ;;
esac
