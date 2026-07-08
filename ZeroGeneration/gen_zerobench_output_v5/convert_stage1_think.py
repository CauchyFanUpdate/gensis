#!/usr/bin/env python3
"""
Convert stage1_samples.jsonl → LLaMA-Factory JSONL with <think>/<answer> format.

Rules:
- quality_pass=True  → answer = actual_new_answer,   reasoning = modifier_gemini.reasoning
- quality_pass=None  with designed_new_answer + valid image → answer = designed_new_answer
- Otherwise → skip (no reliable answer or image)
"""

import json
import os

INPUT_JSONL  = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/zerobench_pipeline_chenlifeng/_archive/output_v5/stage1_samples.jsonl"
OUTPUT_JSONL = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench_output_v5/zerobench_stage1_think.jsonl"

included = skipped_no_img = skipped_no_answer = 0

with open(INPUT_JSONL, encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)

        # determine answer
        if d.get("quality_pass"):
            answer = (d.get("actual_new_answer") or "").strip()
        elif d.get("designed_new_answer"):
            answer = d.get("designed_new_answer").strip()
        else:
            skipped_no_answer += 1
            continue

        if not answer:
            skipped_no_answer += 1
            continue

        question = (d.get("new_question") or d.get("original_question", "")).strip()
        if not question:
            skipped_no_answer += 1
            continue

        mg = d.get("modifier_gemini") or {}
        mq = d.get("modifier_qwen") or {}
        reasoning = (mg.get("reasoning") or mq.get("reasoning") or "").strip()

        img_path = (d.get("stage1_image_path") or "").strip()
        if not img_path or not os.path.exists(img_path):
            skipped_no_img += 1
            continue

        if reasoning:
            assistant = f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}</answer>"
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
print(f"跳过 无图片:     {skipped_no_img}")
print(f"跳过 无答案/问题: {skipped_no_answer}")
print(f"文件: {OUTPUT_JSONL}")
