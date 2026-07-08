#!/usr/bin/env python3
"""
Convert MMStar parquet to the ZeroBench pipeline v5 expected schema.

MMStar schema:
  index (int64), question (str), answer (str), category (str),
  l2_category (str), image (bytes), meta_info (dict)

  Note: `image` is raw bytes (not a struct). `question` already embeds
  options inline ("Options: A: ... B: ... C: ... D: ..."). `answer` is
  a single letter (A/B/C/D).

Pipeline expected schema:
  question_id             <- "mmstar_{index:05d}"
  question_text           <- question  (includes embedded options)
  question_answer         <- answer    (single letter)
  question_images_decoded <- [{"bytes": image_bytes, "path": image_path}]

Extra columns preserved (useful for filtering / stratified analysis):
  category, l2_category, source
"""

import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def convert(mmstar_parquet: str, output_path: str) -> None:
    table = pq.read_table(mmstar_parquet)
    n = len(table)
    print(f"[prepare_mmstar] loaded {n} rows from {mmstar_parquet}")

    img_struct = pa.struct([
        pa.field("bytes", pa.large_binary()),
        pa.field("path",  pa.string()),
    ])
    schema = pa.schema([
        pa.field("question_id",             pa.string()),
        pa.field("question_text",           pa.string()),
        pa.field("question_answer",         pa.string()),
        pa.field("question_images_decoded", pa.list_(img_struct)),
        pa.field("category",               pa.string()),
        pa.field("l2_category",            pa.string()),
        pa.field("source",                 pa.string()),
    ])

    records = []
    skipped = 0
    for i in range(n):
        row = {col: table.column(col)[i].as_py() for col in table.schema.names}

        img_bytes = row.get("image")
        if not img_bytes:
            print(f"[prepare_mmstar]   row {i}: no image bytes, skipping")
            skipped += 1
            continue

        meta = row.get("meta_info") or {}
        image_path = meta.get("image_path") or ""
        source = meta.get("source") or ""

        records.append({
            "question_id":             f"mmstar_{row['index']:05d}",
            "question_text":           str(row["question"]).strip(),
            "question_answer":         str(row["answer"]).strip(),
            "question_images_decoded": [{"bytes": img_bytes, "path": image_path}],
            "category":               str(row.get("category") or ""),
            "l2_category":            str(row.get("l2_category") or ""),
            "source":                 source,
        })

    print(f"[prepare_mmstar] converted: {len(records)} rows  (skipped {skipped} without images)")

    arrays = {
        "question_id":     pa.array([r["question_id"]     for r in records], type=pa.string()),
        "question_text":   pa.array([r["question_text"]   for r in records], type=pa.string()),
        "question_answer": pa.array([r["question_answer"] for r in records], type=pa.string()),
        "question_images_decoded": pa.array(
            [r["question_images_decoded"] for r in records],
            type=pa.list_(img_struct),
        ),
        "category":    pa.array([r["category"]    for r in records], type=pa.string()),
        "l2_category": pa.array([r["l2_category"] for r in records], type=pa.string()),
        "source":      pa.array([r["source"]      for r in records], type=pa.string()),
    }

    out_table = pa.table(arrays, schema=schema)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, output_path, compression="snappy")
    print(f"[prepare_mmstar] written: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert MMStar to pipeline input parquet")
    parser.add_argument(
        "--mmstar_parquet",
        default="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/MMStar/mmstar.parquet",
    )
    parser.add_argument(
        "--output",
        default="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_mmstar/_archive/mmstar_input.parquet",
    )
    args = parser.parse_args()
    convert(args.mmstar_parquet, args.output)


if __name__ == "__main__":
    main()
