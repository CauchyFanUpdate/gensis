#!/bin/bash
# Sequential runner for all think-data experiments

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXPERIMENTS=(
    # "qwen3vl_grpo_rlvr_ZEROBench_old"
    # "qwen3vl_grpo_rlvr_Babyvision_new"
    # "qwen3vl_grpo_rlvr_Babyvision_old"
    # "qwen3vl_grpo_rlvr_Babyvision_merged"
    # "qwen3vl_grpo_rlvr_MMMU_merged"
    # "qwen3vl_grpo_rlvr_EMMA_merged"
     "qwen3vl_grpo_rlvr_ZEROBench_merged"
)

TOTAL=${#EXPERIMENTS[@]}
FAILED=()

for i in "${!EXPERIMENTS[@]}"; do
    name="${EXPERIMENTS[$i]}"
    script="${SCRIPT_DIR}/${name}.sh"
    echo ""
    echo "=========================================="
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running experiment $((i+1))/${TOTAL}: ${name}"
    echo "=========================================="

    bash "$script"
    EXIT_CODE=$?

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "[WARN] Experiment ${name} failed with exit code ${EXIT_CODE}, continuing..."
        FAILED+=("$name")
    else
        echo "[INFO] Experiment ${name} completed successfully."
    fi
done

echo ""
echo "=========================================="
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All experiments finished."
echo "  Total:  ${TOTAL}"
echo "  Failed: ${#FAILED[@]}"
for name in "${FAILED[@]}"; do
    echo "    - $name"
done
echo "=========================================="

[ ${#FAILED[@]} -eq 0 ] && exit 0 || exit 1
