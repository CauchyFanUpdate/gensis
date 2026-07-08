#!/usr/bin/env python3
"""
Convert RealworldQA parquet to the ZeroBench pipeline v5 expected schema.

RealworldQA schema:
  image (struct<bytes: binary, path: string>), question, answer

Pipeline expected schema:
  question_id             <- "rw_{i}" (row index, zero-padded)
  question_text           <- question
  question_answer         <- answer
  question_images_decoded <- [{"bytes": image.bytes, "path": image.path or ""}]
"""

import argparse
import glob
import os
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def convert(realworldqa_dir: str, output_path: str) -> None:
    data_dir = Path(realworldqa_dir) / "data"
    parquet_files = sorted(glob.glob(str(data_dir / "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {data_dir}")

    tables = [pq.read_table(f) for f in parquet_files]
    import pyarrow as pa
    table = pa.concat_tables(tables)
    n = len(table)
    print(f"[prepare_realworldqa] loaded {n} rows from {len(parquet_files)} file(s) in {data_dir}")

    records = []
    skipped = 0
    for i in range(n):
        row = {col: table.column(col)[i].as_py() for col in table.schema.names}

        img = row.get("image")
        if not (isinstance(img, dict) and img.get("bytes")):
            print(f"[prepare_realworldqa]   row {i}: no image bytes, skipping")
            skipped += 1
            continue

        records.append({
            "question_id":             f"rw_{i:05d}",
            "question_text":           str(row["question"]).strip(),
            "question_answer":         str(row["answer"]).strip(),
            "question_images_decoded": [{"bytes": img["bytes"], "path": img.get("path") or ""}],
        })

    print(f"[prepare_realworldqa] converted: {len(records)} rows  (skipped {skipped} without images)")

    img_struct = pa.struct([
        pa.field("bytes", pa.large_binary()),
        pa.field("path",  pa.string()),
    ])
    schema = pa.schema([
        pa.field("question_id",             pa.string()),
        pa.field("question_text",           pa.string()),
        pa.field("question_answer",         pa.string()),
        pa.field("question_images_decoded", pa.list_(img_struct)),
    ])

    arrays = {
        "question_id":     pa.array([r["question_id"]     for r in records], type=pa.string()),
        "question_text":   pa.array([r["question_text"]   for r in records], type=pa.string()),
        "question_answer": pa.array([r["question_answer"] for r in records], type=pa.string()),
        "question_images_decoded": pa.array(
            [r["question_images_decoded"] for r in records],
            type=pa.list_(img_struct),
        ),
    }

    out_table = pa.table(arrays, schema=schema)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, output_path, compression="snappy")
    print(f"[prepare_realworldqa] written: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert RealworldQA to pipeline input parquet")
    parser.add_argument(
        "--realworldqa_dir",
        default="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/RealworldQA",
    )
    parser.add_argument(
        "--output",
        default="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_realwordQA/_archive/realworldqa_input.parquet",
    )
    args = parser.parse_args()
    convert(args.realworldqa_dir, args.output)


if __name__ == "__main__":
    main()
