#!/usr/bin/env python3
"""
Convert stage2_samples.jsonl → LLaMA-Factory conversations JSONL.

Output format:
{
  "conversations": [
    {"from": "human", "value": "<image>question text"},
    {"from": "gpt",   "value": "<think>\n...\n</think>\n<answer>\n...</answer>"}
  ],
  "images": ["path/to/stage2_main.png"]
}

HallusionBench questions have no <image_N> placeholders; <image> is prepended.
CoT cleanup: strip trailing **Final Answer: ...** lines (already in <answer> tag).
Only quality_pass=True records are included.
"""

import argparse
import json
import os
import re

_FINAL_ANSWER_RE = re.compile(r'\n?\s*\*\*Final Answer:.*?\*\*\s*\n?', re.DOTALL)


def clean_cot(cot: str) -> str:
    return _FINAL_ANSWER_RE.sub("\n", cot).strip()


def convert(input_jsonl: str, output_jsonl: str) -> None:
    included = skipped_rejected = skipped_no_img = skipped_missing = 0

    with open(input_jsonl, encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            if not d.get("quality_pass"):
                skipped_rejected += 1
                continue

            question = (d.get("final_question") or "").strip()
            cot      = clean_cot(d.get("cot") or "")
            answer   = (d.get("final_answer") or "").strip()
            img_path = (d.get("stage2_image_path") or "").strip()

            if not question or not answer or not img_path:
                skipped_missing += 1
                continue

            if not os.path.exists(img_path):
                skipped_no_img += 1
                continue

            # HallusionBench has no sub-images; prepend single <image> token
            human_text = "<image>" + question

            if cot:
                gpt_value = f"<think>\n{cot}\n</think>\n<answer>\n{answer}</answer>"
            else:
                gpt_value = f"<answer>\n{answer}</answer>"

            record = {
                "conversations": [
                    {"from": "human", "value": human_text},
                    {"from": "gpt",   "value": gpt_value},
                ],
                "images": [img_path],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            included += 1

    print(f"输出: {included} 条")
    print(f"跳过 rejected/failed: {skipped_rejected}")
    print(f"跳过 无图片:          {skipped_no_img}")
    print(f"跳过 字段缺失:        {skipped_missing}")
    print(f"文件: {output_jsonl}")


def main():
    base = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_hallusionbench/_archive/output_hallusionbench"
    parser = argparse.ArgumentParser(description="Convert HallusionBench stage2 JSONL to LLaMA-Factory format")
    parser.add_argument("--input",  default=f"{base}/stage2_samples.jsonl")
    parser.add_argument("--output", default=f"{base}/hallusionbench_stage2_think.jsonl")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
