#!/usr/bin/env python3
"""
Convert LogicVista parquet to the ZeroBench pipeline v5 expected schema.

LogicVista schema:
  id, question, answer, reasoning, skill, broad_capability, specific_capability,
  imagesource, sourcelink, liscenced, image (struct<bytes, path>)

Pipeline expected schema:
  question_id             <- id
  question_text           <- question
  question_answer         <- answer
  question_images_decoded <- [{"bytes": image.bytes, "path": image.path or ""}]
"""

import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


def convert(logicvista_dir: str, output_path: str) -> None:
    parquet_path = Path(logicvista_dir) / "data" / "test-00000-of-00001.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet not found: {parquet_path}")

    table = pq.read_table(str(parquet_path))
    n = len(table)
    print(f"[prepare_logicvista] loaded {n} rows from {parquet_path}")

    records = []
    skipped = 0
    for i in range(n):
        row = {col: table.column(col)[i].as_py() for col in table.schema.names}

        img = row.get("image")
        if not (isinstance(img, dict) and img.get("bytes")):
            print(f"[prepare_logicvista]   {row.get('id', i)}: no image bytes, skipping")
            skipped += 1
            continue

        # List columns → comma-joined string for passthrough metadata
        skill = ", ".join(row.get("skill") or [])
        broad_cap = ", ".join(row.get("broad_capability") or [])
        specific_cap = ", ".join(row.get("specific_capability") or [])

        records.append({
            "question_id":             str(row["id"]),
            "question_text":           str(row["question"]).strip(),
            "question_answer":         str(row["answer"]).strip(),
            "question_images_decoded": [{"bytes": img["bytes"], "path": img.get("path") or ""}],
            "lv_skill":                skill,
            "lv_broad_capability":     broad_cap,
            "lv_specific_capability":  specific_cap,
            "lv_imagesource":          str(row.get("imagesource") or ""),
            "lv_liscenced":            str(row.get("liscenced") or ""),
        })

    print(f"[prepare_logicvista] converted: {len(records)} rows  (skipped {skipped} without images)")

    img_struct = pa.struct([
        pa.field("bytes", pa.large_binary()),
        pa.field("path",  pa.string()),
    ])
    schema = pa.schema([
        pa.field("question_id",              pa.string()),
        pa.field("question_text",            pa.string()),
        pa.field("question_answer",          pa.string()),
        pa.field("question_images_decoded",  pa.list_(img_struct)),
        pa.field("lv_skill",                 pa.string()),
        pa.field("lv_broad_capability",      pa.string()),
        pa.field("lv_specific_capability",   pa.string()),
        pa.field("lv_imagesource",           pa.string()),
        pa.field("lv_liscenced",             pa.string()),
    ])

    arrays = {
        "question_id":     pa.array([r["question_id"]     for r in records], type=pa.string()),
        "question_text":   pa.array([r["question_text"]   for r in records], type=pa.string()),
        "question_answer": pa.array([r["question_answer"] for r in records], type=pa.string()),
        "question_images_decoded": pa.array(
            [r["question_images_decoded"] for r in records],
            type=pa.list_(img_struct),
        ),
        "lv_skill":              pa.array([r["lv_skill"]              for r in records], type=pa.string()),
        "lv_broad_capability":   pa.array([r["lv_broad_capability"]   for r in records], type=pa.string()),
        "lv_specific_capability":pa.array([r["lv_specific_capability"]for r in records], type=pa.string()),
        "lv_imagesource":        pa.array([r["lv_imagesource"]        for r in records], type=pa.string()),
        "lv_liscenced":          pa.array([r["lv_liscenced"]          for r in records], type=pa.string()),
    }

    out_table = pa.table(arrays, schema=schema)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, output_path, compression="snappy")
    print(f"[prepare_logicvista] written: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert LogicVista to pipeline input parquet")
    parser.add_argument(
        "--logicvista_dir",
        default="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/LogicVista",
    )
    parser.add_argument(
        "--output",
        default="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_logitVista/_archive/logicvista_input.parquet",
    )
    args = parser.parse_args()
    convert(args.logicvista_dir, args.output)


if __name__ == "__main__":
    main()
