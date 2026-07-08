#!/usr/bin/env python3
"""
Convert CharXiv val.parquet → pipeline-compatible parquet.

CharXiv columns used:
  image.bytes      → question_images_decoded[0].bytes
  reasoning_q      → question_text
  reasoning_a      → question_answer
  figure_path      → question_id  (e.g. "images/42.jpg" → "charxiv_42")

Output schema (what generate_samples_v5_stage1.py expects):
  question_id               string
  question_text             string
  question_answer           string
  question_images_decoded   list<struct<bytes: binary>>
"""

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def convert(src: str, dst: str) -> None:
    src_table = pq.read_table(src)
    n = len(src_table)
    print(f"[convert] loaded {n} rows from {src}")

    ids, texts, answers, images = [], [], [], []
    skipped = 0

    img_col = src_table.column("image")
    rq_col  = src_table.column("reasoning_q")
    ra_col  = src_table.column("reasoning_a")
    fp_col  = src_table.column("figure_path")

    for i in range(n):
        rq = rq_col[i].as_py()
        ra = ra_col[i].as_py()
        img_struct = img_col[i].as_py()
        fp = fp_col[i].as_py() or f"images/{i}.jpg"

        if not rq or not ra or not img_struct or not img_struct.get("bytes"):
            skipped += 1
            continue

        stem = Path(fp).stem          # "images/42.jpg" → "42"
        qid  = f"charxiv_{stem}"

        ids.append(qid)
        texts.append(rq)
        answers.append(str(ra))
        images.append([{"bytes": img_struct["bytes"]}])

    print(f"[convert] kept {len(ids)} rows, skipped {skipped}")

    img_type = pa.list_(pa.struct([("bytes", pa.binary())]))
    out_table = pa.table({
        "question_id":             pa.array(ids,     type=pa.string()),
        "question_text":           pa.array(texts,   type=pa.string()),
        "question_answer":         pa.array(answers, type=pa.string()),
        "question_images_decoded": pa.array(images,  type=img_type),
    })

    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, dst)
    print(f"[convert] written → {dst}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="CharXiv val.parquet path")
    parser.add_argument("--dst", required=True, help="Output parquet path")
    args = parser.parse_args()
    convert(args.src, args.dst)
