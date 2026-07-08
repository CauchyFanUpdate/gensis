#!/usr/bin/env python3
"""
Convert stage1_samples.jsonl → LLaMA-Factory conversations JSONL.

Output format:
{
  "conversations": [
    {"from": "human", "value": "...<image>...question text..."},
    {"from": "gpt",   "value": "<think>\n...\n</think>\n<answer>\n...</answer>"}
  ],
  "images": ["path/to/stage1_main.png", "path/to/sub_2.png", ...]
}

Answer rules:
- quality_pass=True  → actual_new_answer,  reasoning from modifier_qwen_a
- status=rejected    → designed_new_answer, reasoning from modifier_qwen_a
- status=skipped / failed → skip
"""

import json
import os
import re

INPUT_JSONL  = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_logitVista/_archive/output_logicvista/stage1_samples.jsonl"
OUTPUT_JSONL = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_logitVista/_archive/output_logicvista/logicvista_stage1_think.jsonl"
IMAGES_DIR   = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_logitVista/_archive/output_logicvista/stage1_images"


def replace_image_placeholders(text: str, image_paths: list) -> tuple:
    pattern = re.compile(r'<image_(\d+)>')
    matches = list(pattern.finditer(text))

    if not matches:
        prefix = "".join("<image>" for _ in image_paths)
        return prefix + text, list(image_paths)

    ordered_images = []
    for m in matches:
        idx = int(m.group(1)) - 1  # 0-based
        if 0 <= idx < len(image_paths):
            ordered_images.append(image_paths[idx])

    new_text = pattern.sub("<image>", text)
    return new_text, ordered_images


included = skipped_status = skipped_no_img = skipped_failed = skipped_no_answer = 0

with open(INPUT_JSONL, encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        status = d.get("status", "")

        if status == "skipped":
            skipped_status += 1
            continue

        if status == "failed":
            skipped_failed += 1
            continue

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

        question = (d.get("new_question") or d.get("original_question", "")).strip()
        if not question:
            skipped_no_answer += 1
            continue

        mqa = d.get("modifier_qwen_a") or {}
        mqb = d.get("modifier_qwen_b") or {}
        reasoning = (mqa.get("reasoning") or mqb.get("reasoning") or "").strip()

        main_path = d.get("stage1_image_path", "")
        if not main_path:
            main_path = os.path.join(IMAGES_DIR, f"{d.get('sample_key', '')}.png")
        if not os.path.exists(main_path):
            skipped_no_img += 1
            continue

        sub_paths = [sp for sp in (d.get("sub_image_paths") or []) if os.path.exists(sp)]
        all_image_paths = [main_path] + sub_paths

        human_text, ordered_images = replace_image_placeholders(question, all_image_paths)

        if not ordered_images:
            skipped_no_img += 1
            continue

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
print(f"跳过 skipped:     {skipped_status}")
print(f"跳过 failed:      {skipped_failed}")
print(f"跳过 无图片:      {skipped_no_img}")
print(f"跳过 无答案/问题: {skipped_no_answer}")
print(f"文件: {OUTPUT_JSONL}")
