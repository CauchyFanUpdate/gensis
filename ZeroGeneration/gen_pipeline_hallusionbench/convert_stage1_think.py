#!/usr/bin/env python3
"""
Convert stage1_samples.jsonl → LLaMA-Factory conversations JSONL.

Output format:
{
  "conversations": [
    {"from": "human", "value": "<image>question text"},
    {"from": "gpt",   "value": "<think>\n...\n</think>\n<answer>\n...</answer>"}
  ],
  "images": ["path/to/stage1_main.png"]
}

HallusionBench questions are plain text (no <image_N> placeholders),
so <image> is always prepended. Answers are "yes" / "no".

Inclusion rules:
  quality_pass=True  → actual_new_answer,  reasoning from modifier_qwen_a
  status=rejected    → designed_new_answer, reasoning from modifier_qwen_a
  status=skipped / failed → skip
"""

import argparse
import json
import os


def convert(input_jsonl: str, output_jsonl: str, images_dir: str) -> None:
    included = skipped_status = skipped_failed = skipped_no_img = skipped_no_answer = 0

    with open(input_jsonl, encoding="utf-8") as fin, \
         open(output_jsonl, "w", encoding="utf-8") as fout:

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

            # determine answer
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

            # question text
            question = (d.get("new_question") or d.get("original_question", "")).strip()
            if not question:
                skipped_no_answer += 1
                continue

            # reasoning from modifier_qwen_a (fallback to modifier_qwen_b)
            mqa = d.get("modifier_qwen_a") or {}
            mqb = d.get("modifier_qwen_b") or {}
            reasoning = (mqa.get("reasoning") or mqb.get("reasoning") or "").strip()

            # main stage1 image
            main_path = d.get("stage1_image_path", "")
            if not main_path:
                main_path = os.path.join(images_dir, f"{d.get('sample_key', '')}.png")
            if not os.path.exists(main_path):
                skipped_no_img += 1
                continue

            # HallusionBench has no sub-images; prepend single <image> token
            human_text = "<image>" + question

            # gpt response
            if reasoning:
                gpt_value = f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}</answer>"
            else:
                gpt_value = f"<answer>\n{answer}</answer>"

            record = {
                "conversations": [
                    {"from": "human", "value": human_text},
                    {"from": "gpt",   "value": gpt_value},
                ],
                "images": [main_path],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            included += 1

    print(f"输出: {included} 条")
    print(f"跳过 skipped:     {skipped_status}")
    print(f"跳过 failed:      {skipped_failed}")
    print(f"跳过 无图片:      {skipped_no_img}")
    print(f"跳过 无答案/问题: {skipped_no_answer}")
    print(f"文件: {output_jsonl}")


def main():
    base = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_hallusionbench/_archive/output_hallusionbench"
    parser = argparse.ArgumentParser(description="Convert HallusionBench stage1 JSONL to LLaMA-Factory format")
    parser.add_argument("--input",      default=f"{base}/stage1_samples.jsonl")
    parser.add_argument("--output",     default=f"{base}/hallusionbench_stage1_think.jsonl")
    parser.add_argument("--images_dir", default=f"{base}/stage1_images")
    args = parser.parse_args()
    convert(args.input, args.output, args.images_dir)


if __name__ == "__main__":
    main()
