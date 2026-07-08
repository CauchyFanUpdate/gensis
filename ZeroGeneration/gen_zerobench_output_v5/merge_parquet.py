"""
Merge two ZeroBench parquet files into one.

Usage:
  python merge_parquet.py --inputs A.parquet B.parquet --output merged.parquet
"""

import argparse
from pathlib import Path

import datasets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True, help="Input parquet files")
    parser.add_argument("--output", required=True, help="Output parquet file path")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    parts = []
    for inp in args.inputs:
        p = Path(inp)
        ds = datasets.Dataset.from_parquet(str(p))
        print(f"  {p.name}: {len(ds)} rows")
        parts.append(ds)

    merged = datasets.concatenate_datasets(parts)
    merged.to_parquet(str(output_path))
    print(f"\n→ {len(merged)} rows saved: {output_path}")


if __name__ == "__main__":
    main()
