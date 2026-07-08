#!/usr/bin/env python3
"""
Convert stage1_samples.jsonl + stage1_images → LLaMA-Factory JSONL format.

Output schema (one JSON object per line):
{
  "messages": [
    {"role": "user",      "content": "<image>\n{question}"},
    {"role": "assistant", "content": "{answer}"}
  ],
  "images": ["_archive/output_babyvision_v5/stage1_images/{sample_key}.png"]
}

Only quality_pass=true records are included.
Image paths are relative to gen_pipeline_babyvision/ (the directory containing this script).
"""

import json
import os
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
INPUT_JSONL = SCRIPT_DIR / "_archive/output_babyvision_v5/stage1_samples.jsonl"
OUTPUT_JSONL = SCRIPT_DIR / "babyvision_stage1_llamafactory.jsonl"
IMAGES_DIR   = SCRIPT_DIR / "_archive/output_babyvision_v5/stage1_images"


def main():
    records = []
    with open(INPUT_JSONL, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if not d.get("quality_pass"):
                continue

            question = d.get("new_question") or d.get("original_question", "")
            answer   = d.get("actual_new_answer", "")
            img_path = Path(d.get("stage1_image_path", ""))

            # Use path relative to SCRIPT_DIR so LLaMA-Factory can find it
            # via dataset_info image_dir = gen_pipeline_babyvision/
            try:
                rel_img = img_path.relative_to(SCRIPT_DIR)
            except ValueError:
                # Fallback: reconstruct from sample_key
                sample_key = d.get("sample_key", "")
                rel_img = Path("_archive/output_babyvision_v5/stage1_images") / f"{sample_key}.png"

            # Skip if image file doesn't exist
            abs_img = SCRIPT_DIR / rel_img
            if not abs_img.exists():
                print(f"  [WARN] image not found, skip: {abs_img}")
                continue

            records.append({
                "messages": [
                    {"role": "user",      "content": f"<image>\n{question}"},
                    {"role": "assistant", "content": answer},
                ],
                "images": [str(rel_img)],
            })

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Done. {len(records)} records → {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
