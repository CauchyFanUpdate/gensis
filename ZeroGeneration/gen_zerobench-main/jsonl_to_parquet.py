"""
Convert ZeroBench ShareGPT JSONL to verl RL Parquet format.

Input:  outputs/sharegpt/all.jsonl
Output: outputs/parquet/ZeroBench.parquet
"""

INPUT_JSONL = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench-main/outputs/sharegpt/all.jsonl"
OUTPUT_DIR  = "/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_zerobench-main/outputs/parquet"
DATA_SOURCE = "zerobench"

SYSTEM_PROMPT = (
    "You are a helpful, conversational assistant tasked with answering a question about an image. "
    "First provide concise reasoning inside <think></think>, then give the final response inside <answer></answer>."
)

import json
import re
from pathlib import Path

import datasets
from datasets import Features, Value, Image as HFImage


def extract_answer(text: str) -> str | None:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
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
        print(f"[WARN] {source_label}:{lineno}: no <answer> tag found, skipping")
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

    # Align <image> tokens in question with actual image count
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

    # Prepend system prompt after any leading <image> tokens
    img_match = re.match(r"(<image>\s*)+", question)
    if img_match:
        img_prefix = img_match.group(0)
        q_body = question[img_match.end():]
    else:
        img_prefix = "<image>\n"
        q_body = question
    full_question = f"{img_prefix}{SYSTEM_PROMPT}\n{q_body}"

    return {
        "images":       img_records,
        "data_source":  DATA_SOURCE,
        "prompt":       [{"role": "user", "content": full_question}],
        "reward_model": {"ground_truth": answer, "style": "rule"},
    }


def load_jsonl(path: Path) -> list[dict]:
    rows, skipped = [], 0
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

            messages = item.get("messages", [])
            user_msg = next((m for m in messages if m.get("role") == "user"), None)
            asst_msg = next((m for m in messages if m.get("role") == "assistant"), None)
            if not user_msg or not asst_msg:
                print(f"[WARN] {path.name}:{lineno}: missing user/assistant, skipping")
                skipped += 1
                continue

            row = _make_row(
                user_msg.get("content", ""),
                asst_msg.get("content", ""),
                item.get("images", []),
                path.name, lineno,
            )
            if row is None:
                skipped += 1
            else:
                rows.append(row)

    print(f"  {path.name}: {len(rows)} loaded, {skipped} skipped")
    return rows


def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    features = Features({
        "images":       [HFImage()],
        "data_source":  Value("string"),
        "prompt":       [{"role": Value("string"), "content": Value("string")}],
        "reward_model": {"ground_truth": Value("string"), "style": Value("string")},
    })

    print(f"Input:  {INPUT_JSONL}")
    print(f"Output: {output_dir}\n")

    rows = load_jsonl(Path(INPUT_JSONL))
    if not rows:
        print("[ERROR] No valid rows found.")
        return

    out_path = output_dir / "ZeroBench.parquet"
    ds = datasets.Dataset.from_list(rows, features=features)
    ds.to_parquet(str(out_path))
    print(f"\n→ {len(ds)} rows saved: {out_path}")


if __name__ == "__main__":
    main()
