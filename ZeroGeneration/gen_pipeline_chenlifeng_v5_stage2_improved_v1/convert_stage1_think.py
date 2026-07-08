#!/usr/bin/env python3
"""
Convert stage1_samples.jsonl (clf_use/output_v5_stage2_improved_v1) → LLaMA-Factory conversations JSONL.

Rules:
- quality_pass=True → include, answer from actual_new_answer
- No reasoning (modifier_qwen/gemini are None) → answer-only output
- Image: stage1_image_path (paths valid as-is)
"""

import json
import os
import re

INPUT_JSONL  = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/zerobench_pipeline_chenlifeng/clf_use/_archive/output_v5_stage2_improved_v1/stage1_samples.jsonl"
OUTPUT_JSONL = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_chenlifeng_v5_stage2_improved_v1/chenlifeng_v5_s2imp_v1_stage1_think.jsonl"

_IMAGE_N_RE = re.compile(r'<image_(\d+)>')


def replace_image_placeholders(text: str, image_paths: list) -> tuple:
    matches = list(_IMAGE_N_RE.finditer(text))
    if not matches:
        prefix = "".join("<image>" for _ in image_paths)
        return prefix + text, list(image_paths)
    ordered_images = []
    for m in matches:
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(image_paths):
            ordered_images.append(image_paths[idx])
    new_text = _IMAGE_N_RE.sub("<image>", text)
    return new_text, ordered_images


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

        img_path = d.get("stage1_image_path", "").strip()
        if not img_path or not os.path.exists(img_path):
            skipped_no_img += 1
            continue

        human_text, ordered_images = replace_image_placeholders(question, [img_path])
        if not ordered_images:
            skipped_no_img += 1
            continue

        record = {
            "conversations": [
                {"from": "human", "value": human_text},
                {"from": "gpt",   "value": f"<answer>\n{answer}</answer>"},
            ],
            "images": ordered_images,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        included += 1

print(f"输出: {included} 条")
print(f"跳过 quality_pass非True: {skipped_no_pass}")
print(f"跳过 无图片:              {skipped_no_img}")
print(f"跳过 无答案/问题:         {skipped_no_answer}")
print(f"文件: {OUTPUT_JSONL}")
