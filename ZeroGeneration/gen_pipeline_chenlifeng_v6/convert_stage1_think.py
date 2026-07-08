#!/usr/bin/env python3
"""
Convert stage1_samples.jsonl (clf_use/output_v6) → LLaMA-Factory conversations JSONL.

Output format:
{
  "conversations": [
    {"from": "human", "value": "<image><image>...question text"},
    {"from": "gpt",   "value": "<answer>\n...</answer>"}
  ],
  "images": ["path/img1.png", "path/img2.png", ...]
}

Rules:
- quality_pass=True  → include, answer from actual_new_answer
- quality_pass=False/None → skip
- No reasoning (modifier_qwen/gemini are None for all records) → answer-only output
- Images: stage1_all_image_paths (supports multi-image; all paths valid as-is)
  Prepend one <image> per image to the question text (no <image_N> placeholders in text)
"""

import json
import os

INPUT_JSONL  = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/zerobench_pipeline_chenlifeng/clf_use/_archive/output_v6/stage1_samples.jsonl"
OUTPUT_JSONL = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_chenlifeng_v6/chenlifeng_v6_stage1_think.jsonl"

included = skipped_no_pass = skipped_no_img = skipped_no_answer = 0

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

        answer = d.get("actual_new_answer", "").strip()
        question = (d.get("new_question") or d.get("original_question", "")).strip()
        if not answer or not question:
            skipped_no_answer += 1
            continue

        # collect images from all_image_paths (single or multi)
        all_paths = [p for p in (d.get("stage1_all_image_paths") or []) if os.path.exists(p)]
        if not all_paths:
            # fallback to stage1_image_path
            p = d.get("stage1_image_path", "").strip()
            if p and os.path.exists(p):
                all_paths = [p]
        if not all_paths:
            skipped_no_img += 1
            continue

        human_text = "".join("<image>" for _ in all_paths) + question

        record = {
            "conversations": [
                {"from": "human", "value": human_text},
                {"from": "gpt",   "value": f"<answer>\n{answer}</answer>"},
            ],
            "images": all_paths,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        included += 1

print(f"输出: {included} 条")
print(f"跳过 quality_pass非True: {skipped_no_pass}")
print(f"跳过 无图片:              {skipped_no_img}")
print(f"跳过 无答案/问题:         {skipped_no_answer}")
print(f"文件: {OUTPUT_JSONL}")
