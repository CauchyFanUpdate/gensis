#!/bin/bash
# ZeroBench generation pipeline launcher
# Usage: bash run.sh [track_b|track_a|hopchain|all]

set -e
cd "$(dirname "$0")"

# Load env
set -a; source .env; set +a

TARGET="${1:-all}"

run_track_b() {
    echo "=== Track B: programmatic puzzles ==="
    python -m track_b.run_track_b \
        --puzzle all \
        --n 50 \
        --stage all \
        --prefix zerobench \
        --seed 2025
}

run_track_a() {
    echo "=== Track A Stage 1: image editing + verification ==="
    python track_a/generate_stage1.py \
        --gpt_image2_api_key "$GPT_IMAGE2_API_KEY" \
        --gemini_api_key     "$GEMINI_API_KEY" \
        --qwen_api_key       "$QWEN_API_KEY"

    echo "=== Track A Stage 2: style diversification + CoT ==="
    python track_a/generate_stage2.py \
        --input              outputs/rl/zerobench_track_a_stage1.jsonl \
        --n_styles           3 \
        --gpt_image2_api_key "$GPT_IMAGE2_API_KEY" \
        --gemini_api_key     "$GEMINI_API_KEY"
}

run_hopchain() {
    echo "=== HopChain (auxiliary) ==="
    HOPCHAIN_IMAGE_DIR="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/zerobench/images"
    HOPCHAIN_CATEGORIES="outputs/hopchain/categories.json"
    HOPCHAIN_OUTPUT="outputs/hopchain/verified_queries.jsonl"
    mkdir -p outputs/hopchain

    python hopchain/step4_verify_answers.py \
        --image_dir       "$HOPCHAIN_IMAGE_DIR" \
        --categories_file "$HOPCHAIN_CATEGORIES" \
        --output_file     "$HOPCHAIN_OUTPUT" \
        --gemini_api_key  "$GEMINI_API_KEY" \
        --qwen_api_key    "$QWEN_API_KEY" \
        --concurrency     8 \
        --n_queries       3 \
        --combo_size      4 \
        --max_combos      10 \
        --device          cuda
}

run_convert() {
    echo "=== Convert outputs ==="
    # Track B
    for f in outputs/rl/zerobench_*_stage1.jsonl; do
        python convert/convert_track_b.py --input "$f"
    done
    # Track A RL
    [ -f outputs/rl/zerobench_track_a_stage1.jsonl ] && \
        python convert/convert_track_a.py \
            --input outputs/rl/zerobench_track_a_stage1.jsonl --split rl
    # Track A SFT
    [ -f outputs/sft/zerobench_track_a_stage2.jsonl ] && \
        python convert/convert_track_a.py \
            --input outputs/sft/zerobench_track_a_stage2.jsonl --split sft
    # HopChain
    [ -f outputs/hopchain/verified_queries.jsonl ] && \
        python convert/convert_hopchain.py \
            --input outputs/hopchain/verified_queries.jsonl
    # Filter CoT loops in SFT outputs
    for f in outputs/sft/clear_*.jsonl; do
        [ -f "$f" ] && python filter_loop_think.py --input "$f"
    done
}

case "$TARGET" in
    track_b)  run_track_b ;;
    track_a)  run_track_a ;;
    hopchain) run_hopchain ;;
    all)
        run_track_b
        run_track_a
        run_hopchain
        run_convert
        ;;
    convert)  run_convert ;;
    *)
        echo "Usage: bash run.sh [track_b|track_a|hopchain|convert|all]"
        exit 1
        ;;
esac

echo "Done."
