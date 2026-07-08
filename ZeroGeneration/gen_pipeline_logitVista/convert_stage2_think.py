#!/usr/bin/env python3
"""
Convert stage2_samples.jsonl → LLaMA-Factory conversations JSONL.

Output format:
{
  "conversations": [
    {"from": "human", "value": "<image>...question text..."},
    {"from": "gpt",   "value": "<think>\n...\n</think>\n<answer>\n...</answer>"}
  ],
  "images": ["path/to/stage2_main.png", "path/to/sub_2.png", ...]
}

Only quality_pass=True records are included.
"""

import json
import os
import re

INPUT_JSONL  = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_logitVista/_archive/output_logicvista/stage2_samples.jsonl"
OUTPUT_JSONL = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_logitVista/_archive/output_logicvista/logicvista_stage2_think.jsonl"

_FINAL_ANSWER_RE = re.compile(r'\n?\s*\*\*Final Answer:.*?\*\*\s*\n?', re.DOTALL)
_IMAGE_N_RE      = re.compile(r'<image_(\d+)>')


def replace_image_placeholders(text: str, image_paths: list) -> tuple:
    matches = list(_IMAGE_N_RE.finditer(text))
    if not matches:
        prefix = "".join("<image>" for _ in image_paths)
        return prefix + text, list(image_paths)

    ordered_images = []
    for m in matches:
        idx = int(m.group(1)) - 1  # 0-based
        if 0 <= idx < len(image_paths):
            ordered_images.append(image_paths[idx])

    new_text = _IMAGE_N_RE.sub("<image>", text)
    return new_text, ordered_images


def clean_cot(cot: str) -> str:
    return _FINAL_ANSWER_RE.sub("\n", cot).strip()


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
        cot      = clean_cot(d.get("cot") or "")
        answer   = (d.get("final_answer") or "").strip()
        img_path = (d.get("stage2_image_path") or "").strip()

        if not question or not answer or not img_path:
            skipped_missing += 1
            continue

        if not os.path.exists(img_path):
            skipped_no_img += 1
            continue

        sub_paths = [sp for sp in (d.get("sub_image_paths") or []) if os.path.exists(sp)]
        all_image_paths = [img_path] + sub_paths

        human_text, ordered_images = replace_image_placeholders(question, all_image_paths)

        if not ordered_images:
            skipped_no_img += 1
            continue

        if cot:
            gpt_value = f"<think>\n{cot}\n</think>\n<answer>\n{answer}</answer>"
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
print(f"跳过 rejected/failed: {skipped_rejected}")
print(f"跳过 无图片:          {skipped_no_img}")
print(f"跳过 字段缺失:        {skipped_missing}")
print(f"文件: {OUTPUT_JSONL}")
