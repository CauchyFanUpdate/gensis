#!/usr/bin/env python3
"""
Convert stage1_samples.jsonl (output_v5) → LLaMA-Factory conversations JSONL.

Output format:
{
  "conversations": [
    {"from": "human", "value": "<image>question text"},
    {"from": "gpt",   "value": "<think>\n...\n</think>\n<answer>\n...</answer>"}
  ],
  "images": ["path/to/stage1_image.png"]
}

Rules:
- quality_pass=True  → include, answer from actual_new_answer
- quality_pass=None/False → skip
- Reasoning: modifier_qwen.reasoning, fallback to modifier_gemini.reasoning
- Image: stage1_image_path
"""

import json
import os
import re

INPUT_JSONL  = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/zerobench_pipeline_chenlifeng/_archive/output_v5/stage1_samples.jsonl"
OUTPUT_JSONL = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_chenlifeng_v5/chenlifeng_v5_stage1_think.jsonl"

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

        # 1. only quality_pass=True
        if not d.get("quality_pass"):
            skipped_no_pass += 1
            continue

        # 2. answer and question
        answer = d.get("actual_new_answer", "").strip()
        question = (d.get("new_question") or d.get("original_question", "")).strip()
        if not answer or not question:
            skipped_no_answer += 1
            continue

        # 3. reasoning: prefer modifier_qwen, fallback to modifier_gemini
        mq  = d.get("modifier_qwen") or {}
        mg  = d.get("modifier_gemini") or {}
        reasoning = (mq.get("reasoning") or mg.get("reasoning") or "").strip()

        # 4. image path
        img_path = d.get("stage1_image_path", "").strip()
        if not img_path or not os.path.exists(img_path):
            skipped_no_img += 1
            continue

        # 5. handle image placeholders
        human_text, ordered_images = replace_image_placeholders(question, [img_path])
        if not ordered_images:
            skipped_no_img += 1
            continue

        # 6. build gpt response
        if reasoning:
            gpt_value = f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}</answer>"
        else:
            gpt_value = f"<answer>\n{answer}</answer>"

        record = {
            "conversations": [
                {"from": "human", "value": human_text},
                {"from": "gpt",   "value": gpt_value},
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
