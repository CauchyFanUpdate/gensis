"""
Convert Track B Stage1 (RL) and Stage2 (SFT) JSONL outputs to LLaMA-Factory messages format.

Input records already have the messages format from run_track_b.py.
This script validates, re-normalises, and optionally merges outputs.

Usage:
    python convert/convert_track_b.py --input outputs/rl/zerobench_sokoban_stage1.jsonl
    python convert/convert_track_b.py --input outputs/sft/zerobench_sudoku_stage2.jsonl --split sft
"""

import argparse
import json
import os
import re
from pathlib import Path

SYSTEM_PROMPT = (
    "You are an expert visual puzzle solver. "
    "Output your reasoning inside <think>...</think> "
    "and your final answer inside <answer>...</answer>."
)


def _fix_assistant_content(content: str, stage: str) -> str:
    """Ensure assistant content has correct <think>/<answer> wrapper."""
    has_think  = "<think>" in content
    has_answer = "<answer>" in content

    if stage == "stage1_rl":
        # RL records: no CoT, just <answer>
        if has_answer:
            return content
        return f"<answer>\n{content.strip()}</answer>"

    # SFT records: must have both
    if has_think and has_answer:
        return content
    if has_answer and not has_think:
        ans_m = re.search(r"<answer>(.*?)</answer>", content, re.DOTALL)
        ans = ans_m.group(1).strip() if ans_m else content.strip()
        rest = re.sub(r"<answer>.*?</answer>", "", content, flags=re.DOTALL).strip()
        think = rest if rest else "Let me solve this step by step."
        return f"<think>\n{think}\n</think>\n<answer>\n{ans}</answer>"

    return content


def convert_file(input_path: str, output_path: str = None):
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"clear_{input_path.name}"
    output_path = Path(output_path)

    records_in, records_out = 0, 0
    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            records_in += 1
            rec = json.loads(line)

            # Validate required fields
            if "messages" not in rec or "images" not in rec:
                continue

            # Ensure system message exists
            if rec["messages"][0].get("role") != "system":
                rec["messages"].insert(0, {"role": "system", "content": SYSTEM_PROMPT})

            # Fix assistant content format
            stage = rec.get("stage", "stage2_sft")
            for msg in rec["messages"]:
                if msg["role"] == "assistant":
                    msg["content"] = _fix_assistant_content(msg["content"], stage)

            # Validate images exist
            valid = all(os.path.exists(p) for p in rec.get("images", []))
            if not valid:
                continue

            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            records_out += 1

    print(f"{input_path.name}: {records_in} in → {records_out} out → {output_path}")
    return records_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+", help="Input JSONL file(s)")
    parser.add_argument("--output", default=None, help="Output path (auto if not given)")
    args = parser.parse_args()

    total = 0
    for inp in args.input:
        total += convert_file(inp, args.output if len(args.input) == 1 else None)
    print(f"Total valid records: {total}")


if __name__ == "__main__":
    main()
