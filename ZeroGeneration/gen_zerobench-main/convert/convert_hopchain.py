"""
Convert HopChain verified JSONL outputs to LLaMA-Factory messages format.

Input records (from hopchain/step4_verify_answers.py):
  {
    image_path: str,
    query: str,           # the multi-hop question
    answer: str,          # numeric ground-truth
    hops: [...],          # intermediate reasoning steps
    verification: {...},  # vote results
    stage: "hopchain"
  }

Output RL record (no CoT — numeric answer only):
  {
    messages: [system, user(<image>query), assistant(<answer>...)],
    images:   [image_path],
    sample_key: str,
    stage:    "hopchain"
  }

HopChain data is used as RL training only (Stage 1 equivalent).
CoT generation for HopChain is left for future work since the hop
annotations already provide structured reasoning.

Usage:
    python convert/convert_hopchain.py \
        --input outputs/hopchain/verified_queries.jsonl
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


def _normalise_answer(text: str) -> str:
    if "<answer>" in text:
        return text
    return f"<answer>\n{text.strip()}\n</answer>"


def _make_hopchain_record(rec: dict, idx: int) -> dict | None:
    """Convert a verified HopChain record to LLaMA-Factory format."""
    image_path = rec.get("image_path", "")
    if not image_path or not os.path.exists(image_path):
        return None

    query  = rec.get("query", "")
    answer = str(rec.get("answer", "")).strip()
    if not query or not answer:
        return None

    sample_key = rec.get("sample_key") or f"hopchain_{idx:04d}"

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"<image>{query}"},
            {"role": "assistant", "content": _normalise_answer(answer)},
        ],
        "images":     [image_path],
        "sample_key": sample_key,
        "stage":      "hopchain",
        # Preserve hop structure for downstream analysis
        "hops":       rec.get("hops", []),
    }


def convert_file(input_path: str, output_path: str | None = None) -> int:
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.parent / f"clear_{input_path.name}"
    output_path = Path(output_path)

    records_in = records_out = 0

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            records_in += 1
            rec = json.loads(line)

            out_rec = _make_hopchain_record(rec, records_in)
            if out_rec is None:
                continue

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            records_out += 1

    print(f"{input_path.name}: {records_in} in → {records_out} out → {output_path}")
    return records_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, nargs="+", help="Input JSONL file(s)")
    parser.add_argument("--output", default=None, help="Output path (auto if not given)")
    args = parser.parse_args()

    total = 0
    for inp in args.input:
        total += convert_file(inp, args.output if len(args.input) == 1 else None)
    print(f"Total valid records: {total}")


if __name__ == "__main__":
    main()
