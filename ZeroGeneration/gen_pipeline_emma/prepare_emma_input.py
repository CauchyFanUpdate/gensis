#!/usr/bin/env python3
"""
Convert EMMA-mini parquet files to the ZeroBench pipeline v5 expected schema.

EMMA-mini schema:
  pid, question, options, answer, image_1..image_5, solution,
  subject, task, category, source, type, context

Pipeline expected schema:
  question_id           ← pid (prefixed with subject for uniqueness)
  question_text         ← question (+ options list for multiple-choice)
  question_answer       ← answer
  question_images_decoded ← [{"bytes": ..., "path": ...}] from image_1..image_5

Only the first non-None image is used by the pipeline (get_image_bytes reads [0]).
All subjects are combined into a single output parquet.
"""

import argparse
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path


SUBJECTS = ["Chemistry", "Coding", "Math", "Physics"]
IMAGE_COLS = ["image_1", "image_2", "image_3", "image_4", "image_5"]


def _option_suffix(options) -> str:
    """Build an options line for multiple-choice questions."""
    if options is None:
        return ""
    if hasattr(options, "tolist"):
        options = options.tolist()
    opts = [str(o).strip() for o in options if str(o).strip()]
    if not opts:
        return ""
    return "\nOptions: " + " / ".join(opts)


def convert(emma_dir: str, output_path: str) -> None:
    emma_root = Path(emma_dir)
    records = []

    for subject in SUBJECTS:
        parquet_path = emma_root / subject / "test-00000-of-00001.parquet"
        if not parquet_path.exists():
            print(f"[prepare_emma] WARNING: {parquet_path} not found, skipping")
            continue

        table = pq.read_table(str(parquet_path))
        n = len(table)
        print(f"[prepare_emma] {subject}: {n} rows")

        for i in range(n):
            row = {col: table.column(col)[i].as_py() for col in table.schema.names}

            pid = str(row.get("pid", f"{subject}_{i}"))
            question = str(row.get("question", "")).strip()
            answer = str(row.get("answer", "")).strip()
            qtype = str(row.get("type", "")).strip()
            options = row.get("options")

            # Append options for multiple-choice so the LLM has full context
            if "choice" in qtype.lower() and options is not None:
                question = question + _option_suffix(options)

            # Collect all non-None images in order
            images_decoded = []
            for col in IMAGE_COLS:
                img = row.get(col)
                if img is not None and isinstance(img, dict) and img.get("bytes"):
                    images_decoded.append({"bytes": img["bytes"], "path": img.get("path", "")})

            if not images_decoded:
                print(f"[prepare_emma]   {pid}: no images, skipping")
                continue

            records.append({
                "question_id": pid,
                "question_text": question,
                "question_answer": answer,
                "question_images_decoded": images_decoded,
                # passthrough metadata (useful for filtering / analysis)
                "emma_subject": subject,
                "emma_type": qtype,
                "emma_category": str(row.get("category", "")),
                "emma_task": str(row.get("task", "")),
            })

    print(f"[prepare_emma] total: {len(records)} rows with images")

    # Build PyArrow table with explicit schema for question_images_decoded
    img_struct = pa.struct([
        pa.field("bytes", pa.large_binary()),
        pa.field("path",  pa.string()),
    ])
    schema = pa.schema([
        pa.field("question_id",              pa.string()),
        pa.field("question_text",            pa.string()),
        pa.field("question_answer",          pa.string()),
        pa.field("question_images_decoded",  pa.list_(img_struct)),
        pa.field("emma_subject",             pa.string()),
        pa.field("emma_type",                pa.string()),
        pa.field("emma_category",            pa.string()),
        pa.field("emma_task",                pa.string()),
    ])

    arrays = {
        "question_id":     pa.array([r["question_id"]     for r in records], type=pa.string()),
        "question_text":   pa.array([r["question_text"]   for r in records], type=pa.string()),
        "question_answer": pa.array([r["question_answer"] for r in records], type=pa.string()),
        "question_images_decoded": pa.array(
            [r["question_images_decoded"] for r in records],
            type=pa.list_(img_struct),
        ),
        "emma_subject":  pa.array([r["emma_subject"]  for r in records], type=pa.string()),
        "emma_type":     pa.array([r["emma_type"]     for r in records], type=pa.string()),
        "emma_category": pa.array([r["emma_category"] for r in records], type=pa.string()),
        "emma_task":     pa.array([r["emma_task"]     for r in records], type=pa.string()),
    }

    out_table = pa.table(arrays, schema=schema)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(out_table, output_path, compression="snappy")
    print(f"[prepare_emma] written: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert EMMA-mini to pipeline input parquet")
    parser.add_argument(
        "--emma_dir",
        default="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/EMMA-mini",
    )
    parser.add_argument(
        "--output",
        default="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/project/benchmark/ZeroGeneration/gen_pipeline_emma/_archive/emma_mini_input.parquet",
    )
    args = parser.parse_args()
    convert(args.emma_dir, args.output)


if __name__ == "__main__":
    main()
