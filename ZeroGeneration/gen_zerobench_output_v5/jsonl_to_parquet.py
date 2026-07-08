"""
Convert ZeroBench ShareGPT JSON/JSONL to verl RL Parquet format.

Supports:
  - JSON array files (.json)  with conversations[{from,value}] format
  - JSONL files (.jsonl)      with conversations[{from,value}] format

Answer extraction (tried in order):
  1. <answer>...</answer> tags
  2. **Final Answer: ...** pattern

Usage:
  python jsonl_to_parquet.py --inputs FILE [FILE ...] --output OUT.parquet
"""

import argparse
import json
import re
from pathlib import Path

import datasets
from datasets import Features, Value, Image as HFImage

DATA_SOURCE = "zerobench"

SYSTEM_PROMPT = (
    "You are a helpful, conversational assistant tasked with answering a question about an image. "
    "First provide concise reasoning inside <think></think>, then give the final response inside <answer></answer>."
)


def extract_answer(text: str) -> str | None:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    m = re.search(r"\*\*Final Answer:\s*(.*?)\*\*", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return None


def extract_question(user_content: str) -> str:
    return re.sub(r"^<image>\s*", "", user_content).strip()


def _make_row(user_content: str, asst_content: str, images: list[str],
              source_label: str, lineno: int) -> dict | None:
    question = extract_question(user_content)
    answer = extract_answer(asst_content)

    if not question:
        print(f"[WARN] {source_label}:{lineno}: empty question, skipping")
        return None
    if answer is None:
        print(f"[WARN] {source_label}:{lineno}: no answer found, skipping")
        return None
    if not images:
        print(f"[WARN] {source_label}:{lineno}: no images, skipping")
        return None

    img_records = []
    for img_path_str in images:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"[WARN] {source_label}:{lineno}: image not found: {img_path}, skipping row")
            return None
        try:
            img_records.append({"bytes": img_path.read_bytes(), "path": str(img_path)})
        except Exception as e:
            print(f"[WARN] {source_label}:{lineno}: failed to read {img_path}: {e}, skipping row")
            return None

    n_img_tokens = question.count("<image>")
    n_images = len(img_records)
    if n_img_tokens < n_images:
        question = "<image>\n" * (n_images - n_img_tokens) + question
    elif n_img_tokens > n_images:
        removed = 0
        def _drop_extra(m):
            nonlocal removed
            if removed < n_img_tokens - n_images:
                removed += 1
                return ""
            return m.group(0)
        question = re.sub(r"<image>\s*", _drop_extra, question)

    img_match = re.match(r"(<image>\s*)+", question)
    if img_match:
        # Leading <image> tags: place them before SYSTEM_PROMPT
        img_prefix = img_match.group(0)
        q_body = question[img_match.end():]
        full_question = f"{img_prefix}{SYSTEM_PROMPT}\n{q_body}"
    elif "<image>" in question:
        # Embedded <image> tags (not at start): keep them in place
        full_question = f"{SYSTEM_PROMPT}\n{question}"
    else:
        # No <image> tags: add one prefix before SYSTEM_PROMPT
        full_question = f"<image>\n{SYSTEM_PROMPT}\n{question}"

    return {
        "images":       img_records,
        "data_source":  DATA_SOURCE,
        "prompt":       [{"role": "user", "content": full_question}],
        "reward_model": {"ground_truth": answer, "style": "rule"},
    }


def _parse_item(item: dict, source_label: str, lineno: int) -> dict | None:
    # Support both 'conversations' (from/value) and 'messages' (role/content)
    convs = item.get("conversations")
    if convs is not None:
        user_msg = next((m for m in convs if m.get("from") == "human"), None)
        asst_msg = next((m for m in convs if m.get("from") == "gpt"), None)
        if not user_msg or not asst_msg:
            print(f"[WARN] {source_label}:{lineno}: missing human/gpt turn, skipping")
            return None
        user_content = user_msg.get("value", "")
        asst_content = asst_msg.get("value", "")
    else:
        msgs = item.get("messages", [])
        user_msg = next((m for m in msgs if m.get("role") == "user"), None)
        asst_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
        if not user_msg or not asst_msg:
            print(f"[WARN] {source_label}:{lineno}: missing user/assistant turn, skipping")
            return None
        user_content = user_msg.get("content", "")
        asst_content = asst_msg.get("content", "")

    return _make_row(user_content, asst_content, item.get("images", []),
                     source_label, lineno)


def load_file(path: Path) -> list[dict]:
    rows, skipped = [], 0
    suffix = path.suffix.lower()

    if suffix == ".json":
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, list):
            data = [data]
        for lineno, item in enumerate(data, 1):
            row = _parse_item(item, path.name, lineno)
            if row is None:
                skipped += 1
            else:
                rows.append(row)
    else:
        # JSONL
        with open(path) as f:
            for lineno, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    print(f"[WARN] {path.name}:{lineno}: JSON decode error, skipping")
                    skipped += 1
                    continue
                row = _parse_item(item, path.name, lineno)
                if row is None:
                    skipped += 1
                else:
                    rows.append(row)

    print(f"  {path.name}: {len(rows)} loaded, {skipped} skipped")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Input JSON/JSONL files")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    features = Features({
        "images":       [HFImage()],
        "data_source":  Value("string"),
        "prompt":       [{"role": Value("string"), "content": Value("string")}],
        "reward_model": {"ground_truth": Value("string"), "style": Value("string")},
    })

    all_rows = []
    for inp in args.inputs:
        p = Path(inp)
        print(f"Loading: {p}")
        all_rows.extend(load_file(p))

    if not all_rows:
        print("[ERROR] No valid rows found.")
        return

    ds = datasets.Dataset.from_list(all_rows, features=features)
    ds.to_parquet(str(output_path))
    print(f"\n→ {len(ds)} rows saved: {output_path}")


if __name__ == "__main__":
    main()
