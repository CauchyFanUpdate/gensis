"""Merge per-sample rubrics from a JSONL file into a training parquet.

Matching key: (first_image_path, question_text) — unique for 99.9% of rows.
Rows with no JSONL match get rubric=None and fall back to the general judge prompt.

Usage:
    python3 add_rubric_to_parquet.py \
        --parquet  dataset/merged_rl_train.parquet \
        --jsonl    dataset/all_rl_rubric.jsonl \
        --output   dataset/merged_rl_train_rubric.parquet
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def extract_key(record: dict) -> tuple[str, str]:
    """Return (first_image_path, question_text) as the composite match key."""
    images = record.get("images", [])
    img_path = images[0].get("path", "") if images else ""

    question = ""
    for turn in record.get("prompt", []):
        if isinstance(turn, dict) and turn.get("role") == "user":
            content = turn.get("content", "")
            if isinstance(content, str):
                question = content.strip()
                break

    return img_path, question


def build_lookup(jsonl_path: str) -> dict[tuple[str, str], object]:
    lookup: dict[tuple[str, str], object] = {}
    skipped = 0
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            rubric = record.get("rubric")
            if rubric is None:
                skipped += 1
                continue
            key = extract_key(record)
            if key not in lookup:  # first match wins for the 14 composite-key duplicates
                lookup[key] = rubric
    print(f"[INFO] JSONL loaded: {len(lookup)} unique (image_path, question) entries, {skipped} skipped")
    return lookup


def parquet_key(row) -> tuple[str, str]:
    """Extract composite key from a parquet row (images column is a list of dicts)."""
    images = row.get("images", [])
    img_path = ""
    if images:
        first = images[0]
        if isinstance(first, dict):
            img_path = first.get("path", "")

    question = ""
    for turn in row.get("prompt", []):
        if isinstance(turn, dict) and turn.get("role") == "user":
            content = turn.get("content", "")
            if isinstance(content, str):
                question = content.strip()
                break

    return img_path, question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print(f"[INFO] Loading parquet: {args.parquet}")
    df = pd.read_parquet(args.parquet)
    print(f"[INFO] Parquet rows: {len(df)}, columns: {df.columns.tolist()}")

    lookup = build_lookup(args.jsonl)

    matched = 0
    rubrics = []
    for _, row in df.iterrows():
        key = parquet_key(row)
        rubric = lookup.get(key)
        rubrics.append(rubric)
        if rubric is not None:
            matched += 1

    df["rubric"] = rubrics
    unmatched = len(df) - matched
    print(f"[INFO] Matched: {matched}/{len(df)} rows  (unmatched={unmatched} → rubric=None, will use general judge)")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(str(out), index=False)
    print(f"[INFO] Saved enriched parquet: {out}")
    print(f"[INFO] Columns: {df.columns.tolist()}")


if __name__ == "__main__":
    main()
