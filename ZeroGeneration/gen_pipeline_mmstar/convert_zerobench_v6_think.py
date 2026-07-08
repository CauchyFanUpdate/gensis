#!/usr/bin/env python3
"""
Convert sharegpt_zerobenchv6.json → LLaMA-Factory conversations JSONL.

Input format (sharegpt):
  [{"conversations": [{"from": "human", "value": "<image>\n...\nquestion"},
                      {"from": "gpt",   "value": "...reasoning...\n**Final Answer: X**"}],
    "images": ["/abs/path/img1.png", ...]}, ...]

Output format:
  {"conversations": [{"from": "human", "value": "<image>...question"},
                     {"from": "gpt",   "value": "<think>\n{cot}\n</think>\n<answer>\n{answer}</answer>"}],
   "images": ["/abs/path/img1.png", ...]}

- human value and images are kept as-is (already correct format)
- gpt value is split: CoT = everything before **Final Answer: X**, answer = X
- Records without **Final Answer:** are skipped
"""

import json
import os
import re

INPUT_JSON   = "/mnt/tidal-alsh01/dataset/perceptionVLM/code_chenlifeng/AMath_train_dataset/zerobench_pipeline/clf_use/v6/sharegpt_zerobenchv6.json"
OUTPUT_JSONL = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench_output_v5/zerobench_v6_think.jsonl"

_FINAL_ANSWER_RE = re.compile(r'\*\*Final Answer:\s*(.+?)\*\*\s*$', re.DOTALL)


def split_cot_and_answer(gpt_value: str):
    """Extract (cot, answer) from gpt response ending with **Final Answer: X**."""
    m = _FINAL_ANSWER_RE.search(gpt_value)
    if not m:
        return None, None
    answer = m.group(1).strip()
    cot = gpt_value[:m.start()].strip()
    return cot, answer


data = json.load(open(INPUT_JSON, encoding="utf-8"))

included = skipped_no_answer = skipped_no_img = 0

with open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:
    for d in data:
        convs = d.get("conversations", [])
        images = d.get("images", [])

        human_val = convs[0]["value"] if len(convs) > 0 else ""
        gpt_val   = convs[1]["value"] if len(convs) > 1 else ""

        # Split CoT and answer
        cot, answer = split_cot_and_answer(gpt_val)
        if not answer:
            skipped_no_answer += 1
            continue

        # Verify images exist
        missing = [p for p in images if not os.path.exists(p)]
        if missing:
            skipped_no_img += 1
            continue

        gpt_out = f"<think>\n{cot}\n</think>\n<answer>\n{answer}</answer>"

        record = {
            "conversations": [
                {"from": "human", "value": human_val},
                {"from": "gpt",   "value": gpt_out},
            ],
            "images": images,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        included += 1

print(f"输出: {included} 条")
print(f"跳过 无 Final Answer: {skipped_no_answer}")
print(f"跳过 图片缺失:        {skipped_no_img}")
print(f"文件: {OUTPUT_JSONL}")
