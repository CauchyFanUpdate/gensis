#!/usr/bin/env bash
# Serial run of all closed-source API models on UNREAL_Benchmark.
# Each model resumes from results/close/eval_<slug>.jsonl.

set -u
cd "$(dirname "$0")"

MODELS=(
  "gemini-3.1-pro-preview"
  "qwen3.6-plus"
  "gpt-5.5"
  "claude-opus-4-7"
  "claude-sonnet-4-6"
  "doubao-seed-1-8"
  "kimi-k2.6"
)

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for model in "${MODELS[@]}"; do
  slug=$(echo "$model" | sed 's/[\/.\-]/_/g')
  log="$LOG_DIR/${slug}_$(date +%Y%m%d_%H%M%S).log"
  echo "===================================================================="
  echo "[$(date +%H:%M:%S)] Running model: $model"
  echo "  log: $log"
  echo "===================================================================="
  python3 eval_api.py --model "$model" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  if [[ $rc -ne 0 ]]; then
    echo "[!] $model exited with code $rc — continuing with next model"
  fi
done

echo
echo "All models finished. Results in: results/close/"
