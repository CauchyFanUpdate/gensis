#!/usr/bin/env python3
"""
Convert EMMA stage1 and stage2 samples to ShareGPT format.

Reference format:
  {
    "messages": [
      {"role": "user",      "content": "<image>...question..."},
      {"role": "assistant", "content": "<think>\n...\n</think>\n<answer>\n...\n</answer>"}
    ],
    "images": ["/path/to/image.png"],
    "sample_key": "..."
  }

Stage1 rules (output_emma_v3/stage1_samples.jsonl):
  - quality_pass=True  → answer = actual_new_answer,  reasoning = modifier_qwen_1/2.reasoning
  - status=rejected    → answer = designed_new_answer, reasoning = modifier_qwen_1/2.reasoning
  - status=skipped / failed → skip
  - options_fixed=True  → options already in new_question
  - options_fixed=False → append options from mmmu_meta.options
  - Replace "<image 1>" (and similar) with "<image>" in question text

Stage2 rules (output_emma_v3_stage2_only/stage2_train.jsonl):
  - Already in ShareGPT messages format; fix wrong image path prefix
  - Wrong: .../gen_pipeline_mmmu_pro/output_emma_v3_stage2_only/stage2_images/...
  - Right:  .../gen_pipeline_emma/output_emma_v3_stage2_only/stage2_images/...
"""

import json
import os
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STAGE1_INPUT  = os.path.join(BASE_DIR, "output_emma_v3/stage1_samples.jsonl")
STAGE2_INPUT  = os.path.join(BASE_DIR, "output_emma_v3_stage2_only/stage2_train.jsonl")
STAGE1_OUTPUT = os.path.join(BASE_DIR, "emma_v3_stage1_sharegpt.jsonl")
STAGE2_OUTPUT = os.path.join(BASE_DIR, "emma_v3_stage2_sharegpt.jsonl")
COMBINED_OUTPUT = os.path.join(BASE_DIR, "emma_v3_sharegpt.jsonl")

# Wrong image prefix in stage2 (typo in pipeline output)
WRONG_IMG_PREFIX = (
    "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/"
    "ZeroGeneration/gen_pipeline_mmmu_pro/"
)
RIGHT_IMG_PREFIX = (
    "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/"
    "ZeroGeneration/gen_pipeline_emma/"
)

LETTER_LABELS = "ABCDEFGHIJ"


def normalize_question(question):
    """Replace <image N> / <image_N> / <imageN> with <image>."""
    return re.sub(r"<image[_ ]?\d+>", "<image>", question).strip()


def build_question_with_options(question, options_fixed, mmmu_meta):
    """Return final user question string (options appended if not already fixed)."""
    q = normalize_question(question)
    if options_fixed:
        return q
    options = (mmmu_meta or {}).get("options", [])
    if options:
        opts_str = "\n".join(
            f"{LETTER_LABELS[i]}) {opt}" for i, opt in enumerate(options)
        )
        q = f"{q}\n{opts_str}"
    return q


def build_record(question, answer, reasoning, img_path, sample_key):
    user_content = f"<image>{question}"
    if reasoning:
        assistant_content = f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}\n</answer>"
    else:
        assistant_content = f"<answer>\n{answer}\n</answer>"
    return {
        "messages": [
            {"role": "user",      "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [img_path],
        "sample_key": sample_key,
    }


def convert_stage1(input_path, output_path):
    included = 0
    skipped_status = 0
    skipped_failed = 0
    skipped_no_img = 0
    skipped_no_answer = 0

    records = []
    with open(input_path, encoding="utf-8") as fin:
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

            # Determine answer
            if d.get("quality_pass"):
                answer = (d.get("actual_new_answer") or "").strip()
            elif status == "rejected":
                answer = (d.get("designed_new_answer") or "").strip()
            else:
                skipped_no_answer += 1
                continue

            question_raw = (d.get("new_question") or d.get("original_question") or "").strip()
            if not question_raw or not answer:
                skipped_no_answer += 1
                continue

            question = build_question_with_options(
                question_raw,
                d.get("options_fixed"),
                d.get("mmmu_meta"),
            )

            # Reasoning: modifier_qwen_1 preferred, fallback modifier_qwen_2
            mq1 = d.get("modifier_qwen_1") or {}
            mq2 = d.get("modifier_qwen_2") or {}
            reasoning = (mq1.get("reasoning") or mq2.get("reasoning") or "").strip()

            img_path = (d.get("stage1_image_path") or "").strip()
            if not img_path or not os.path.exists(img_path):
                skipped_no_img += 1
                continue

            sample_key = d.get("sample_key", "")
            records.append(build_record(question, answer, reasoning, img_path, sample_key))
            included += 1

    with open(output_path, "w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Stage1] 输出: {included} 条")
    print(f"         跳过 skipped:    {skipped_status}")
    print(f"         跳过 failed:     {skipped_failed}")
    print(f"         跳过 无图片:     {skipped_no_img}")
    print(f"         跳过 无答案/问题: {skipped_no_answer}")
    print(f"         文件: {output_path}")
    return records


def convert_stage2(input_path, output_path):
    included = 0
    skipped_no_img = 0

    records = []
    with open(input_path, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            # Fix wrong image prefix
            images = [
                img.replace(WRONG_IMG_PREFIX, RIGHT_IMG_PREFIX)
                for img in d.get("images", [])
            ]

            if not images or not os.path.exists(images[0]):
                skipped_no_img += 1
                continue

            record = {
                "messages":   d["messages"],
                "images":     images,
                "sample_key": d.get("sample_key", ""),
            }
            records.append(record)
            included += 1

    with open(output_path, "w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Stage2] 输出: {included} 条")
    print(f"         跳过 无图片: {skipped_no_img}")
    print(f"         文件: {output_path}")
    return records


if __name__ == "__main__":
    print("=" * 60)
    s1_records = convert_stage1(STAGE1_INPUT, STAGE1_OUTPUT)
    print()
    s2_records = convert_stage2(STAGE2_INPUT, STAGE2_OUTPUT)
    print()

    all_records = s1_records + s2_records
    with open(COMBINED_OUTPUT, "w", encoding="utf-8") as fout:
        for r in all_records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[合并]   总计: {len(all_records)} 条  →  {COMBINED_OUTPUT}")
    print("=" * 60)
