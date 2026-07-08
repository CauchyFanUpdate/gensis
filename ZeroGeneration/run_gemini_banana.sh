#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# run_gemini_banana.sh — Run Gemini (gemini-3-pro-image-preview) Stage 1 and
# Stage 2 for every pipeline listed in gen_banana_list.json.
#
# Usage:
#   STAGE=1   bash run_gemini_banana.sh              # Stage 1 only (all pipelines)
#   STAGE=2   bash run_gemini_banana.sh              # Stage 2 only (Stage 1 must be done)
#   STAGE=all bash run_gemini_banana.sh              # Stage 1 → Stage 2 sequentially
#
#   # Single pipeline:
#   PIPELINE=gen_pipeline_emma STAGE=1 bash run_gemini_banana.sh
#
#   # Smoke-test (10 samples per pipeline):
#   LIMIT=10 STAGE=1 bash run_gemini_banana.sh
#
# Output: each pipeline's images and jsonl go to
#   {pipeline_dir}/_archive/output_gemini/stage1_images/
#   {pipeline_dir}/_archive/output_gemini/stage2_images/
#   {pipeline_dir}/_archive/output_gemini/stage1_samples.jsonl
#   {pipeline_dir}/_archive/output_gemini/stage2_samples.jsonl
#
# Resume: automatic — already-written sample_keys are skipped.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BANANA_LIST="$SCRIPT_DIR/gen_banana_list.json"

# ── API configuration ─────────────────────────────────────────────────────────

# Gemini VL (Call 1 analysis / Modifier / Call 4 CoT) — NOT the image model
export GEMINI_VL_ENDPOINT="${GEMINI_VL_ENDPOINT:-https://runway.devops.rednote.life/openai/google/v1:generateContent}"
export GEMINI_VL_MODEL="${GEMINI_VL_MODEL:-gemini-3.1-pro-preview}"
export GEMINI_VL_API_KEY="${GEMINI_VL_API_KEY:-YOUR_GEMINI_API_KEY}"

# Qwen (Call 2a/2b text + Selector + Verifier VL)
export QWEN_API_BASE="${QWEN_API_BASE:-https://runway.devops.xiaohongshu.com/openai/qwen/v1}"
export QWEN_MODEL="${QWEN_MODEL:-qwen3.6-plus}"
export QWEN_API_KEY="${QWEN_API_KEY:-${OPENAI_API_KEY:-YOUR_API_KEY}}"
export OPENAI_API_KEY="$QWEN_API_KEY"

# Gemini image generation (Call 3a/3b — replaces GPT-Image-2)
export GEMINI_IMAGE_ENDPOINT="${GEMINI_IMAGE_ENDPOINT:-https://runway.devops.rednote.life/openai/google/v1:generateContent}"
export GEMINI_IMAGE_API_KEY="${GEMINI_IMAGE_API_KEY:-YOUR_GEMINI_IMAGE_API_KEY}"

# Concurrency and retry settings
export MAX_CONCURRENT="${MAX_CONCURRENT:-64}"
export MAX_RETRIES="${MAX_RETRIES:-20}"
export GEMINI_MAX_CONCURRENT="${GEMINI_MAX_CONCURRENT:-16}"
export GEMINI_IMG_MAX_CONCURRENT="${GEMINI_IMG_MAX_CONCURRENT:-64}"
export QWEN_MAX_CONCURRENT="${QWEN_MAX_CONCURRENT:-64}"
export API_TIMEOUT="${API_TIMEOUT:-1200}"

LIMIT="${LIMIT:-0}"

# ── Input validation ───────────────────────────────────────────────────────────

case "${STAGE:-}" in
    1|2|all) ;;
    *)
        echo "[gemini_banana] ERROR: STAGE='${STAGE:-<unset>}' invalid."
        echo ""
        echo "  STAGE=1   bash run_gemini_banana.sh"
        echo "  STAGE=2   bash run_gemini_banana.sh"
        echo "  STAGE=all bash run_gemini_banana.sh"
        exit 1
        ;;
esac

if [ ! -f "$BANANA_LIST" ]; then
    echo "[gemini_banana] ERROR: gen_banana_list.json not found: $BANANA_LIST"
    exit 1
fi

# ── Extract pipeline paths from banana list ───────────────────────────────────

mapfile -t PIPELINE_PATHS < <(python3 -c "
import json, sys
d = json.loads(open('$BANANA_LIST').read())
for k, v in d.items():
    print(v)
")

echo "[gemini_banana] STAGE=$STAGE  LIMIT=$LIMIT  MAX_CONCURRENT=$MAX_CONCURRENT  MAX_RETRIES=$MAX_RETRIES"
echo "[gemini_banana] Gemini image: $GEMINI_IMAGE_ENDPOINT  (key=${GEMINI_IMAGE_API_KEY:0:8}…)"
echo "[gemini_banana] Pipelines: ${#PIPELINE_PATHS[@]}"
echo ""

# ── Per-pipeline parquet/manifest defaults ────────────────────────────────────
# Each pipeline reads from a different data source. Override any of these via env.

ZEROBENCH_PARQUET="/mnt/tidal-alsh01/dataset/perceptionVLM/code_chenlifeng/datasets/zerobench/data/zerobench_subquestions-00000-of-00001.parquet"
BABYVISION_PARQUET="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/BabyVision/data/train-00000-of-00001.parquet"
MMMU_MANIFEST="/mnt/tidal-alsh01/dataset/perceptionVLM/wanghan/edit_mmmu/selected_data/manifest.jsonl"

_get_parquet_for() {
    local pipeline_dir="$1"
    local name
    name="$(basename "$pipeline_dir")"
    case "$name" in
        gen_pipeline_babyvision)
            echo "$BABYVISION_PARQUET" ;;
        gen_pipeline_DocVQA|gen_pipeline_charxiv|gen_pipeline_infoVQA|gen_pipeline_logitVista)
            echo "$ZEROBENCH_PARQUET" ;;
        gen_pipeline_emma)
            echo "$pipeline_dir/_archive/emma_mini_input.parquet" ;;
        gen_pipeline_hallusionbench)
            echo "$pipeline_dir/_archive/hallusionbench_input.parquet" ;;
        gen_pipeline_mmstar)
            echo "$pipeline_dir/_archive/mmstar_input.parquet" ;;
        gen_pipeline_realwordQA)
            echo "$pipeline_dir/_archive/realworldqa_input.parquet" ;;
        *)
            echo "" ;;
    esac
}

# ── Per-pipeline stage functions ──────────────────────────────────────────────

run_stage1_for() {
    local pipeline_dir="$1"
    local name
    name="$(basename "$pipeline_dir")"
    local output_dir="$pipeline_dir/_archive/output_gemini"

    # Determine which stage1 script to use
    local stage1_script
    if [ "$name" = "gen_pipeline_mmmu_pro" ]; then
        stage1_script="stage1_edit_gemini.py"
    elif [ "$name" = "gen_zerobench_output_v5" ]; then
        echo "  [skip] $name: no generate scripts"
        return 0
    else
        stage1_script="generate_samples_gemini_stage1.py"
    fi

    if [ ! -f "$pipeline_dir/$stage1_script" ]; then
        echo "  [skip] $name/$stage1_script — not found"
        return 0
    fi

    echo "  ─── Stage 1: $name ───"
    local extra_args=()

    if [ "$LIMIT" -gt 0 ]; then
        extra_args+=(--limit "$LIMIT")
    fi

    cd "$pipeline_dir"

    if [ "$name" = "gen_pipeline_mmmu_pro" ]; then
        # mmmu_pro uses --manifest instead of --parquet
        local manifest="${MANIFEST:-$MMMU_MANIFEST}"
        python3 "$stage1_script" \
            --manifest        "$manifest" \
            --output_dir      "$output_dir" \
            --max_concurrent  "$MAX_CONCURRENT" \
            --max_retries     "$MAX_RETRIES" \
            --gemini_image_endpoint "$GEMINI_IMAGE_ENDPOINT" \
            --gemini_image_api_key  "$GEMINI_IMAGE_API_KEY" \
            "${extra_args[@]}"
    else
        # All other pipelines use --parquet
        local parquet="${PARQUET:-$(_get_parquet_for "$pipeline_dir")}"
        if [ -z "$parquet" ]; then
            echo "  [warn] $name: no parquet path configured, skipping"
            cd "$SCRIPT_DIR"
            return 0
        fi
        python3 "$stage1_script" \
            --parquet         "$parquet" \
            --output_dir      "$output_dir" \
            --max_concurrent  "$MAX_CONCURRENT" \
            --max_retries     "$MAX_RETRIES" \
            --gemini_image_endpoint "$GEMINI_IMAGE_ENDPOINT" \
            --gemini_image_api_key  "$GEMINI_IMAGE_API_KEY" \
            "${extra_args[@]}"
    fi

    cd "$SCRIPT_DIR"
    echo "  Stage 1 done: $output_dir/stage1_samples.jsonl"
}

run_stage2_for() {
    local pipeline_dir="$1"
    local name
    name="$(basename "$pipeline_dir")"
    local output_dir="$pipeline_dir/_archive/output_gemini"

    local stage2_script
    if [ "$name" = "gen_pipeline_mmmu_pro" ]; then
        stage2_script="stage2_diversify_gemini.py"
    elif [ "$name" = "gen_zerobench_output_v5" ]; then
        echo "  [skip] $name: no generate scripts"
        return 0
    else
        stage2_script="generate_samples_gemini_stage2.py"
    fi

    if [ ! -f "$pipeline_dir/$stage2_script" ]; then
        echo "  [skip] $name/$stage2_script — not found"
        return 0
    fi

    local stage1_jsonl="$output_dir/stage1_samples.jsonl"
    if [ ! -f "$stage1_jsonl" ]; then
        echo "  [warn] $name: stage1_samples.jsonl not found at $stage1_jsonl — skipping Stage 2"
        return 0
    fi

    echo "  ─── Stage 2: $name ───"
    local extra_args=()
    if [ "$LIMIT" -gt 0 ]; then
        extra_args+=(--limit "$LIMIT")
    fi

    cd "$pipeline_dir"
    python3 "$stage2_script" \
        --stage1_jsonl    "$stage1_jsonl" \
        --output_dir      "$output_dir" \
        --max_concurrent  "$MAX_CONCURRENT" \
        --max_retries     "$MAX_RETRIES" \
        --gemini_image_endpoint "$GEMINI_IMAGE_ENDPOINT" \
        --gemini_image_api_key  "$GEMINI_IMAGE_API_KEY" \
        "${extra_args[@]}"
    cd "$SCRIPT_DIR"
    echo "  Stage 2 done: $output_dir/stage2_samples.jsonl"
}

# ── Main loop ─────────────────────────────────────────────────────────────────

FILTER="${PIPELINE:-}"  # optional: run only one pipeline by name

for pipeline_dir in "${PIPELINE_PATHS[@]}"; do
    name="$(basename "$pipeline_dir")"
    if [ -n "$FILTER" ] && [ "$name" != "$FILTER" ]; then
        continue
    fi
    if [ ! -d "$pipeline_dir" ]; then
        echo "[gemini_banana] warn: directory not found: $pipeline_dir"
        continue
    fi

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Pipeline: $name"
    echo "══════════════════════════════════════════════════════"

    case "$STAGE" in
        1)
            run_stage1_for "$pipeline_dir"
            ;;
        2)
            run_stage2_for "$pipeline_dir"
            ;;
        all)
            run_stage1_for "$pipeline_dir"
            run_stage2_for "$pipeline_dir"
            ;;
    esac
done

echo ""
echo "[gemini_banana] All done."
