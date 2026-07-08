#!/usr/bin/env python3
"""
Convert zerobench_train_converted.jsonl → LLaMA-Factory conversations JSONL.

Input format:
{
  "messages": [
    {"role": "user",      "content": "<image>question text"},
    {"role": "assistant", "content": "1. step...\n...\n**Final Answer: X**"}
  ],
  "images": ["path/to/image.png"]
}

Output format:
{
  "conversations": [
    {"from": "human", "value": "<image>question text"},
    {"from": "gpt",   "value": "<think>\n...\n</think>\n<answer>\nX</answer>"}
  ],
  "images": ["path/to/image.png"]
}

Splitting rules for assistant content:
- If "**Final Answer:" is present, everything before it → <think>, the answer value → <answer>
- Otherwise, wrap the entire content in <answer>
"""

import json
import re

INPUT_JSONL  = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/ZeroBench_Dataset/zerobench_train_converted.jsonl"
OUTPUT_JSONL = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/zerobench_train_think.jsonl"

FINAL_ANSWER_RE = re.compile(r'\*\*Final Answer:\s*(.*?)\*\*\s*$', re.DOTALL)


def split_reasoning_answer(content: str) -> tuple[str, str]:
    """
    Split assistant content into (reasoning, answer).
    Returns (reasoning, answer) where reasoning may be empty.
    """
    m = FINAL_ANSWER_RE.search(content)
    if m:
        reasoning = content[:m.start()].rstrip()
        answer = m.group(1).strip()
        return reasoning, answer
    return "", content.strip()


included = skipped_no_msg = skipped_no_img = 0

with open(INPUT_JSONL, encoding="utf-8") as fin, \
     open(OUTPUT_JSONL, "w", encoding="utf-8") as fout:

    for line in fin:
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)

        messages = d.get("messages", [])
        images   = d.get("images", [])

        # Extract user and assistant messages
        user_content      = next((m["content"] for m in messages if m["role"] == "user"),      None)
        assistant_content = next((m["content"] for m in messages if m["role"] == "assistant"), None)

        if not user_content or not assistant_content:
            skipped_no_msg += 1
            continue

        if not images:
            skipped_no_img += 1
            continue

        # Build gpt value with <think>/<answer> structure
        reasoning, answer = split_reasoning_answer(assistant_content)
        if reasoning:
            gpt_value = f"<think>\n{reasoning}\n</think>\n<answer>\n{answer}</answer>"
        else:
            gpt_value = f"<answer>\n{answer}</answer>"

        record = {
            "conversations": [
                {"from": "human", "value": user_content},
                {"from": "gpt",   "value": gpt_value},
            ],
            "images": images,
        }
        fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        included += 1

print(f"输出: {included} 条")
print(f"跳过 无消息:  {skipped_no_msg}")
print(f"跳过 无图片:  {skipped_no_img}")
print(f"文件: {OUTPUT_JSONL}")
