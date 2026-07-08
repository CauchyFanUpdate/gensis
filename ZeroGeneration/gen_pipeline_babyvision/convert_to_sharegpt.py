#!/usr/bin/env python3
"""
Convert BabyVision stage1 and stage2 samples to ShareGPT format.

Reference format (stage1_train_fixed.jsonl from mmmu_pro):
  {
    "messages": [
      {"role": "user",      "content": "<image>...question..."},
      {"role": "assistant", "content": "<think>\n...\n</think>\n<answer>\n...\n</answer>"}
    ],
    "images": ["/path/to/image.png"],
    "sample_key": "..."
  }

Stage1 rules (output_babyvision_v2/stage1_samples.jsonl):
  - quality_pass=True  → answer = actual_new_answer,  reasoning = modifier_gemini/qwen.reasoning
  - status=rejected    → answer = designed_new_answer, reasoning = modifier_gemini/qwen.reasoning
  - status=skipped / failed → skip

Stage2 rules (output_babyvision_v2_orig/stage2_samples.jsonl):
  - quality_pass=True  → answer = final_answer, reasoning = cot
  - all others         → skip
"""

import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

STAGE1_INPUT  = os.path.join(BASE_DIR, "output_babyvision_v2/stage1_samples.jsonl")
STAGE2_INPUT  = os.path.join(BASE_DIR, "output_babyvision_v2_orig/stage2_samples.jsonl")
STAGE1_OUTPUT = os.path.join(BASE_DIR, "babyvision_v2_stage1_sharegpt.jsonl")
STAGE2_OUTPUT = os.path.join(BASE_DIR, "babyvision_v2_stage2_sharegpt.jsonl")
COMBINED_OUTPUT = os.path.join(BASE_DIR, "babyvision_v2_sharegpt.jsonl")


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

            question = (d.get("new_question") or d.get("original_question") or "").strip()
            if not question or not answer:
                skipped_no_answer += 1
                continue

            # Reasoning from modifier_gemini, fallback to modifier_qwen
            mg = d.get("modifier_gemini") or {}
            mq = d.get("modifier_qwen") or {}
            reasoning = (mg.get("reasoning") or mq.get("reasoning") or "").strip()

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
    skipped_rejected = 0
    skipped_no_img = 0
    skipped_missing = 0

    records = []
    with open(input_path, encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)

            if not d.get("quality_pass"):
                skipped_rejected += 1
                continue

            question = (d.get("final_question") or "").strip()
            cot      = (d.get("cot") or "").strip()
            answer   = (d.get("final_answer") or "").strip()
            img_path = (d.get("stage2_image_path") or "").strip()

            if not question or not answer or not img_path:
                skipped_missing += 1
                continue
            if not os.path.exists(img_path):
                skipped_no_img += 1
                continue

            sample_key = d.get("sample_key", "")
            records.append(build_record(question, answer, cot, img_path, sample_key))
            included += 1

    with open(output_path, "w", encoding="utf-8") as fout:
        for r in records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[Stage2] 输出: {included} 条")
    print(f"         跳过 quality_pass!=True: {skipped_rejected}")
    print(f"         跳过 无图片:             {skipped_no_img}")
    print(f"         跳过 字段缺失:           {skipped_missing}")
    print(f"         文件: {output_path}")
    return records


if __name__ == "__main__":
    print("=" * 60)
    s1_records = convert_stage1(STAGE1_INPUT, STAGE1_OUTPUT)
    print()
    s2_records = convert_stage2(STAGE2_INPUT, STAGE2_OUTPUT)
    print()

    # Write combined file
    all_records = s1_records + s2_records
    with open(COMBINED_OUTPUT, "w", encoding="utf-8") as fout:
        for r in all_records:
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[合并]   总计: {len(all_records)} 条  →  {COMBINED_OUTPUT}")
    print("=" * 60)
