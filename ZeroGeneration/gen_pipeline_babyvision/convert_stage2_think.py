#!/usr/bin/env python3
"""
Convert stage2_samples.jsonl → LLaMA-Factory JSONL with <think>/<answer> format.

Fields used:
  final_question  → user question
  cot             → <think> content
  final_answer    → <answer> content
  stage2_image_path → image path

Only quality_pass=True records are included.
"""

import json
import os

INPUT_JSONL  = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_babyvision/_archive/output_babyvision_v5/stage2_samples.jsonl"
OUTPUT_JSONL = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_babyvision/babyvision_stage2_think.jsonl"

included = skipped_rejected = skipped_no_img = skipped_missing = 0

with open(INPUT_JSONL, encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)

        if not d.get("quality_pass"):
            skipped_rejected += 1
            continue

        question = (d.get("final_question") or "").strip()
        cot      = (d.get("cot") or "").strip()
        answer   = (d.get("final_answer") or "").strip()
        img_path = (d.get("stage2_image_path") or "").strip()

        if not question or not answer or not img_path:
            skipped_missing += 1
            continue

        if not os.path.exists(img_path):
            skipped_no_img += 1
            continue

        if cot:
            assistant = f"<think>\n{cot}\n</think>\n<answer>\n{answer}</answer>"
        else:
            assistant = f"<answer>\n{answer}</answer>"

        record = {
            "messages": [
                {"role": "user",      "content": f"<image>{question}"},
                {"role": "assistant", "content": assistant},
            ],
            "images": [img_path],
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        included += 1

print(f"输出: {included} 条")
print(f"跳过 rejected:   {skipped_rejected}")
print(f"跳过 无图片:     {skipped_no_img}")
print(f"跳过 字段缺失:   {skipped_missing}")
print(f"文件: {OUTPUT_JSONL}")
