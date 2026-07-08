#!/usr/bin/env python3
"""
Convert HallusionBench parquet to the ZeroBench pipeline v5 expected schema.

HallusionBench schema (per row):
  category, subcategory, visual_input (0/1/2), set_id, figure_id,
  sample_note, question_id, question, gt_answer_details, gt_answer (0/1),
  filename, image ({"bytes": ..., "path": ...} | None)

Pipeline expected schema:
  question_id             <- "{category}_{subcategory}_s{set_id}_f{figure_id}_q{question_id}"
                             (globally unique composite key)
  question_text           <- question
  question_answer         <- "yes" if gt_answer==1 else "no"
  question_images_decoded <- [{"bytes": image["bytes"], "path": filename or image["path"]}]

Skipped rows:
  - visual_input == 0  (no image; 178 rows in non_image-*.parquet)
  - image is None or missing bytes

For visual_input == 2 (two-image comparison questions): only the single stored
image is available in the parquet, so it is used as-is.
"""

import argparse
import glob

import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def convert(hallusionbench_dir: str, output_path: str) -> None:
    data_dir = Path(hallusionbench_dir) / "data"
    parquet_files = sorted(glob.glob(str(data_dir / "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in: {data_dir}")

    tables = [pq.read_table(f) for f in parquet_files]
    table = pa.concat_tables(tables)
    n = len(table)
    print(f"[prepare_hallusionbench] loaded {n} rows from {len(parquet_files)} file(s) in {data_dir}")

    records = []
    skipped_no_image = 0
    skipped_no_bytes = 0

    for i in range(n):
        row = {col: table.column(col)[i].as_py() for col in table.schema.names}

        if row.get("visual_input") == 0 or row.get("visual_input") == "0":
            skipped_no_image += 1
            continue

        img = row.get("image")
        if not (isinstance(img, dict) and img.get("bytes")):
            skipped_no_bytes += 1
            continue

        category = str(row.get("category", "")).strip()
        subcategory = str(row.get("subcategory", "")).strip()
        set_id = str(row.get("set_id", "")).strip()
        figure_id = str(row.get("figure_id", "")).strip()
        local_qid = str(row.get("question_id", "")).strip()
        unique_id = f"{category}_{subcategory}_s{set_id}_f{figure_id}_q{local_qid}"

        gt_answer = row.get("gt_answer", 0)
        answer_str = "yes" if (gt_answer == 1 or gt_answer == "1") else "no"

        filename = row.get("filename") or img.get("path") or ""

        records.append({
            "question_id":             unique_id,
            "question_text":           str(row.get("question", "")).strip(),
            "question_answer":         answer_str,
            "question_images_decoded": [{"bytes": img["bytes"], "path": str(filename)}],
        })

    print(
        f"[prepare_hallusionbench] converted: {len(records)} rows  "
        f"(skipped {skipped_no_image} no-image visual_input=0, "
        f"{skipped_no_bytes} missing image bytes)"
    )

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
    print(f"[prepare_hallusionbench] written: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert HallusionBench to pipeline input parquet")
    parser.add_argument(
        "--hallusionbench_dir",
        default="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/HallusionBench",
    )
    parser.add_argument(
        "--output",
        default="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_hallusionbench/_archive/hallusionbench_input.parquet",
    )
    args = parser.parse_args()
    convert(args.hallusionbench_dir, args.output)


if __name__ == "__main__":
    main()
