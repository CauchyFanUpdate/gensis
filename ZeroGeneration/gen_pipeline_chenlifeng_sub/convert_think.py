#!/usr/bin/env python3
"""
Convert samples.jsonl (output_v2_subject_subsititution) → LLaMA-Factory conversations JSONL.

Output format:
{
  "conversations": [
    {"from": "human", "value": "<image>question text"},
    {"from": "gpt",   "value": "<think>\n...\n</think>\n<answer>\n...</answer>"}
  ],
  "images": ["path/to/edited_image.png"]
}

Rules:
- quality_pass=True  → include, answer from new_answer, reasoning from generated_answer
- quality_pass=False → skip
- Image path: local images/ dir + basename of edited_image_path
"""

import json
import os

INPUT_JSONL  = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/zerobench_pipeline_chenlifeng/_archive/output_v2_subject_subsititution/samples.jsonl"
IMAGES_DIR   = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/zerobench_pipeline_chenlifeng/_archive/output_v2_subject_subsititution/images"
OUTPUT_JSONL = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_chenlifeng_sub/chenlifeng_sub_think.jsonl"

included = skipped_no_pass = skipped_no_img = skipped_no_answer = 0

with open(INPUT_JSONL, encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)

        # 1. only include quality_pass=True
        if not d.get("quality_pass"):
            skipped_no_pass += 1
            continue

        # 2. answer and question
        answer = d.get("new_answer", "").strip()
        question = d.get("new_question", "").strip()
        if not answer or not question:
            skipped_no_answer += 1
            continue

        # 3. reasoning from generated_answer
        reasoning = d.get("generated_answer", "").strip()

        # 4. image path (local images/ dir takes priority over edited_image_path)
        edited_path = d.get("edited_image_path", "")
        img_path = os.path.join(IMAGES_DIR, os.path.basename(edited_path))
        if not os.path.exists(img_path):
            skipped_no_img += 1
            continue

        # 5. build gpt response
        if reasoning:
            gpt_value = f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}</answer>"
        else:
            gpt_value = f"<answer>\n{answer}</answer>"

        record = {
            "conversations": [
                {"from": "human", "value": f"<image>{question}"},
                {"from": "gpt",   "value": gpt_value},
            ],
            "images": [img_path],
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        included += 1

print(f"输出: {included} 条")
print(f"跳过 quality_pass=False: {skipped_no_pass}")
print(f"跳过 无图片:              {skipped_no_img}")
print(f"跳过 无答案/问题:         {skipped_no_answer}")
print(f"文件: {OUTPUT_JSONL}")
