#!/usr/bin/env python3
"""
Convert stage1_samples.jsonl → LLaMA-Factory JSONL with <think>/<answer> format.

Rules:
- Skip status=skipped records
- Skip records with no image file on disk
- quality_pass=True  → answer = actual_new_answer,   reasoning = modifier_gemini.reasoning
- status=rejected    → answer = designed_new_answer,  reasoning = modifier_gemini.reasoning
- status=failed      → skip (no reliable answer or image)
"""

import json
import os
from pathlib import Path

INPUT_JSONL  = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_babyvision/_archive/output_babyvision_v5/stage1_samples.jsonl"
OUTPUT_JSONL = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_babyvision/babyvision_stage1_think.jsonl"
IMAGES_DIR   = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_babyvision/_archive/output_babyvision_v5/stage1_images"

included = skipped_status = skipped_no_img = skipped_failed = skipped_no_answer = 0

with open(INPUT_JSONL, encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        status = d.get("status", "")

        # 1. skip explicitly skipped
        if status == "skipped":
            skipped_status += 1
            continue

        # 2. skip failed (no reliable answer / image)
        if status == "failed":
            skipped_failed += 1
            continue

        # 3. determine answer
        if d.get("quality_pass"):
            answer = d.get("actual_new_answer", "").strip()
        elif status == "rejected":
            answer = d.get("designed_new_answer", "").strip()
        else:
            skipped_no_answer += 1
            continue

        if not answer:
            skipped_no_answer += 1
            continue

        # 4. question
        question = (d.get("new_question") or d.get("original_question", "")).strip()
        if not question:
            skipped_no_answer += 1
            continue

        # 5. reasoning from modifier_gemini (fallback to modifier_qwen)
        mg = d.get("modifier_gemini") or {}
        mq = d.get("modifier_qwen") or {}
        reasoning = (mg.get("reasoning") or mq.get("reasoning") or "").strip()

        # 6. image path
        img_path = d.get("stage1_image_path", "")
        if not img_path:
            sample_key = d.get("sample_key", "")
            img_path = os.path.join(IMAGES_DIR, f"{sample_key}.png")
        if not os.path.exists(img_path):
            skipped_no_img += 1
            continue

        # 7. build assistant content
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
print(f"跳过 skipped:    {skipped_status}")
print(f"跳过 failed:     {skipped_failed}")
print(f"跳过 无图片:     {skipped_no_img}")
print(f"跳过 无答案/问题: {skipped_no_answer}")
print(f"文件: {OUTPUT_JSONL}")
