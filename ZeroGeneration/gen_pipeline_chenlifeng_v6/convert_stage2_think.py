#!/usr/bin/env python3
"""
Convert stage2_samples.jsonl (clf_use/output_v6) → LLaMA-Factory conversations JSONL.

Output format:
{
  "conversations": [
    {"from": "human", "value": "<image><image>...question text"},
    {"from": "gpt",   "value": "<think>\n...\n</think>\n<answer>\n...</answer>"}
  ],
  "images": ["path/img1.png", "path/img2.png", ...]
}

Rules:
- quality_pass=True  → include
- quality_pass=None/False → skip
- Reasoning: cot with "**Final Answer: ...**" suffix stripped
- Images: stage2_all_image_paths (supports multi-image; all paths valid as-is)
  Prepend one <image> per image to the question text (no <image_N> placeholders in text)
"""

import json
import os
import re

INPUT_JSONL  = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/zerobench_pipeline_chenlifeng/clf_use/_archive/output_v6/stage2_samples.jsonl"
OUTPUT_JSONL = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_chenlifeng_v6/chenlifeng_v6_stage2_think.jsonl"

_FINAL_ANSWER_RE = re.compile(r'\n?\s*\*\*Final Answer:.*?\*\*\s*\n?', re.DOTALL)


def clean_cot(cot: str) -> str:
    return _FINAL_ANSWER_RE.sub("\n", cot).strip()


included = skipped_no_pass = skipped_no_img = skipped_missing = 0

with open(INPUT_JSONL, encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)

        if d.get("quality_pass") is not True:
            skipped_no_pass += 1
            continue

        question = (d.get("final_question") or "").strip()
        answer   = (d.get("final_answer") or "").strip()
        cot      = clean_cot(d.get("cot") or "")

        if not question or not answer:
            skipped_missing += 1
            continue

        # collect images from all_image_paths (single or multi)
        all_paths = [p for p in (d.get("stage2_all_image_paths") or []) if os.path.exists(p)]
        if not all_paths:
            p = (d.get("stage2_image_path") or "").strip()
            if p and os.path.exists(p):
                all_paths = [p]
        if not all_paths:
            skipped_no_img += 1
            continue

        human_text = "".join("<image>" for _ in all_paths) + question

        if cot:
            gpt_value = f"<think>\n{cot}\n</think>\n<answer>\n{answer}</answer>"
        else:
            gpt_value = f"<answer>\n{answer}</answer>"

        record = {
            "conversations": [
                {"from": "human", "value": human_text},
                {"from": "gpt",   "value": gpt_value},
            ],
            "images": all_paths,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        included += 1

print(f"输出: {included} 条")
print(f"跳过 quality_pass非True: {skipped_no_pass}")
print(f"跳过 无图片:              {skipped_no_img}")
print(f"跳过 字段缺失:            {skipped_missing}")
print(f"文件: {OUTPUT_JSONL}")
