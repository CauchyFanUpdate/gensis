#!/usr/bin/env python3
"""
Convert DocVQA stage1_samples.jsonl → LLaMA-Factory conversations JSONL.

Output format:
{
  "conversations": [
    {"from": "human", "value": "<image>question text"},
    {"from": "gpt",   "value": "<think>\n...\n</think>\n<answer>\n...</answer>"}
  ],
  "images": ["path/to/stage1_image.png"]
}

Include rules:
- quality_pass=True  → actual_new_answer + modifier_qwen_a.reasoning
- status=rejected    → designed_new_answer + modifier_qwen_a.reasoning
- status=skipped / failed → skip

Image handling:
- DocVQA records have no <image_N> placeholders in question text.
- Prepend one <image> tag before the question text.
- sub_image_paths is None for DocVQA; handled gracefully.
"""

import argparse
import json
import os
import re

_IMAGE_N_RE = re.compile(r'<image_(\d+)>')


def replace_image_placeholders(text: str, image_paths: list) -> tuple:
    """
    Replace <image_N> tokens in text with <image> in-place.
    If no <image_N> found, prepend one <image> per path.
    Returns (new_text, ordered_images_list).
    """
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default=os.path.join(
        os.path.dirname(__file__),
        "_archive/output_v5_docvqa/stage1_samples.jsonl"))
    parser.add_argument("--output", default=os.path.join(
        os.path.dirname(__file__),
        "_archive/output_v5_docvqa/docvqa_stage1_think.jsonl"))
    args = parser.parse_args()

    included = skipped_status = skipped_failed = skipped_no_img = skipped_no_answer = 0

    with open(args.input, encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

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

            # Determine answer and whether to include
            if d.get("quality_pass"):
                answer = (d.get("actual_new_answer") or "").strip()
            elif status == "rejected":
                answer = (d.get("designed_new_answer") or "").strip()
            else:
                skipped_no_answer += 1
                continue

            if not answer:
                skipped_no_answer += 1
                continue

            question = (d.get("new_question") or d.get("original_question") or "").strip()
            if not question:
                skipped_no_answer += 1
                continue

            # Reasoning from modifier_qwen_a, fallback to modifier_qwen_b
            mqa = d.get("modifier_qwen_a") or {}
            mqb = d.get("modifier_qwen_b") or {}
            reasoning = (mqa.get("reasoning") or mqb.get("reasoning") or "").strip()

            # Image paths: main stage1 image + any sub-images
            main_path = (d.get("stage1_image_path") or "").strip()
            if not main_path or not os.path.exists(main_path):
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

    print(f"输出:             {included} 条")
    print(f"跳过 skipped:     {skipped_status}")
    print(f"跳过 failed:      {skipped_failed}")
    print(f"跳过 无图片:      {skipped_no_img}")
    print(f"跳过 无答案/问题: {skipped_no_answer}")
    print(f"文件: {args.output}")


if __name__ == "__main__":
    main()
