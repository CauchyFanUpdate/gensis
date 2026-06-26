"""Build new UNREAL_Benchmark with 600 reviewed=True samples.

Reads UNREAL_Benchmark_old633/Unreal_Benchmark.jsonl (English) plus the
reviewed flag from Unreal_Benchmark_zh.jsonl. Renumbers IDs within each
family (e.g., MATH_0001..MATH_0259), copies images with new names, and
writes UNREAL_Benchmark/Unreal_Benchmark.jsonl.
"""
import json, shutil
from pathlib import Path
from collections import defaultdict

ROOT = Path(__file__).parent
OLD = ROOT / "UNREAL_Benchmark_old633"
NEW = ROOT / "UNREAL_Benchmark"
NEW_IMG = NEW / "images"
NEW_IMG.mkdir(parents=True, exist_ok=True)

en = {}
with (OLD / "Unreal_Benchmark.jsonl").open() as f:
    for line in f:
        r = json.loads(line)
        en[r["id"]] = r

reviewed_ids = []
with (OLD / "Unreal_Benchmark_zh.jsonl").open() as f:
    for line in f:
        r = json.loads(line)
        if r.get("reviewed") is True:
            reviewed_ids.append(r["id"])

assert len(reviewed_ids) == 600, len(reviewed_ids)

# Preserve original ordering; renumber within family
family_counters = defaultdict(int)
out_records = []
for new_index, old_id in enumerate(reviewed_ids, start=1):
    rec = dict(en[old_id])  # copy
    family = old_id.split("_")[0]
    family_counters[family] += 1
    new_id = f"{family}_{family_counters[family]:04d}"
    new_image = f"{new_id}.png"
    old_image_path = OLD / "images" / rec["image"]
    new_image_path = NEW_IMG / new_image
    shutil.copy2(old_image_path, new_image_path)
    rec["index"] = new_index
    rec["id"] = new_id
    rec["image"] = new_image
    out_records.append(rec)

with (NEW / "Unreal_Benchmark.jsonl").open("w") as f:
    for r in out_records:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"wrote {len(out_records)} records")
print("family counts:", dict(family_counters))
print("images copied:", len(list(NEW_IMG.iterdir())))
