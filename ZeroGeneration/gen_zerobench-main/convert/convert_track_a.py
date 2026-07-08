"""
Convert Track A Stage1 (RL) and Stage2 (SFT) outputs to LLaMA-Factory messages format.

Stage 1 input records (from track_a/generate_stage1.py):
  {
    sample_key, question_id, original_question, original_answer,
    edit_description, new_answer, quality_pass, subq_verification,
    image_path, stage: "track_a_stage1"
  }
  → Converted to RL record: no CoT, just <answer>.

Stage 2 input records (from track_a/generate_stage2.py):
  {
    messages: [...],  # already in LLaMA-Factory format with <think>/<answer>
    images: [...],
    sample_key, stage: "track_a_stage2"
  }
  → Validated and passed through.

Usage:
    # Convert Stage 1 to RL records
    python convert/convert_track_a.py \
        --input outputs/rl/zerobench_track_a_stage1.jsonl --split rl

    # Validate Stage 2 SFT records
    python convert/convert_track_a.py \
        --input outputs/sft/zerobench_track_a_stage2.jsonl --split sft
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
    """Ensure content is wrapped in <answer>...</answer>."""
    if "<answer>" in text:
        return text
    return f"<answer>\n{text.strip()}\n</answer>"


def _normalise_sft_content(text: str) -> str:
    """Ensure SFT content has both <think> and <answer> wrappers."""
    has_think  = "<think>"  in text
    has_answer = "<answer>" in text

    if has_think and has_answer:
        return text

    if has_answer and not has_think:
        ans_m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
        ans   = ans_m.group(1).strip() if ans_m else text.strip()
        rest  = re.sub(r"<answer>.*?</answer>", "", text, flags=re.DOTALL).strip()
        think = rest if rest else "Let me solve this step by step."
        return f"<think>\n{think}\n</think>\n<answer>\n{ans}\n</answer>"

    return text


def _make_rl_record(rec: dict) -> dict | None:
    """
    Build an RL record from a Stage-1 output dict.
    Returns None if the record should be skipped.
    """
    if not rec.get("quality_pass"):
        return None
    image_path = rec.get("image_path", "")
    if not image_path or not os.path.exists(image_path):
        return None

    question = rec.get("original_question", "")
    answer   = rec.get("new_answer") or rec.get("original_answer", "")
    sample_key = rec.get("sample_key", f"main_{rec.get('question_id', '?')}")

    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"<image>{question}"},
            {"role": "assistant", "content": _normalise_answer(answer)},
        ],
        "images":     [image_path],
        "sample_key": sample_key,
        "stage":      "track_a_stage1",
    }


def convert_file(input_path: str, split: str, output_path: str | None = None) -> int:
    input_path  = Path(input_path)
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

            if split == "rl":
                # Stage-1 raw record → build RL messages format
                out_rec = _make_rl_record(rec)
                if out_rec is None:
                    continue
            else:
                # Stage-2: already in messages format; validate and normalise
                if "messages" not in rec or "images" not in rec:
                    continue
                # Ensure system message present
                if rec["messages"][0].get("role") != "system":
                    rec["messages"].insert(0, {"role": "system", "content": SYSTEM_PROMPT})
                # Fix assistant content
                for msg in rec["messages"]:
                    if msg["role"] == "assistant":
                        msg["content"] = _normalise_sft_content(msg["content"])
                # Validate images exist
                if not all(os.path.exists(p) for p in rec.get("images", [])):
                    continue
                out_rec = rec

            fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
            records_out += 1

    print(f"{input_path.name}: {records_in} in → {records_out} out → {output_path}")
    return records_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  required=True, help="Input JSONL file")
    parser.add_argument("--split",  choices=["rl", "sft"], default="rl",
                        help="rl=Stage1 raw records, sft=Stage2 messages records")
    parser.add_argument("--output", default=None,  help="Output path (auto if not given)")
    args = parser.parse_args()

    total = convert_file(args.input, args.split, args.output)
    print(f"Total valid records: {total}")


if __name__ == "__main__":
    main()
