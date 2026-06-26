"""Remap old eval_*.jsonl results to the new 600-sample UNREAL_Benchmark.

Match by:
  1. image md5 (old image → new image must be identical bytes)
  2. question + answer strings (must be unchanged in the new release)

Surviving records keep all original eval fields but with `id` rewritten to
the new id. Output: UNREAL_Benchmark/eval/results/eval_*.jsonl
"""
import hashlib, json, sys
from pathlib import Path

ROOT = Path("/mnt/tidal-alsh01/dataset/perceptionVLM/wanghan/zerogen")
NEW_DIR = ROOT / "UNREAL_Benchmark"
NEW_IMG = NEW_DIR / "images"
NEW_JSONL = NEW_DIR / "Unreal_Benchmark.jsonl"

OLD_BASE = Path("/mnt/tidal-alsh01/dataset/perceptionVLM/code_chenlifeng/new_benchmark/unrealbench")
EVAL_DIR = OLD_BASE / "export_vlmevalkit"
UNIFIED = EVAL_DIR / "unified_benchmark.jsonl"

UB_IMG = OLD_BASE / "images"
UB_MATH_DIRS = [OLD_BASE / "math_pipeline" / "output" / "images",
                OLD_BASE / "math_pipeline" / "output_v2" / "images"]
ZG_IMG = ROOT / "release" / "images"

OUT_DIR = NEW_DIR / "eval" / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_old_image(rec) -> Path | None:
    rid = rec["id"]
    family = rec.get("family", "")
    if rid.startswith("UB-"):
        raw = rid[3:]
        if family == "MATH":
            for d in UB_MATH_DIRS:
                for ext in (".png", ".jpg"):
                    p = d / (raw + ext)
                    if p.exists():
                        return p
            for step4 in [OLD_BASE / "math_pipeline" / "output" / "step4_final.jsonl",
                          OLD_BASE / "math_pipeline" / "output_v2" / "step4_final.jsonl"]:
                if not step4.exists():
                    continue
                with step4.open() as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except Exception:
                            continue
                        if r.get("id") == raw and r.get("status") == "ok":
                            rel = r.get("edited_image_path", "")
                            if rel:
                                p = OLD_BASE / "math_pipeline" / rel
                                if p.exists():
                                    return p
            return None
        for ext in (".png", ".jpg"):
            p = UB_IMG / (raw + ext)
            if p.exists():
                return p
        return None
    if rid.startswith("ZG-"):
        raw = rid[3:]
        for ext in (".png", ".jpg"):
            p = ZG_IMG / (raw + ext)
            if p.exists():
                return p
        base = raw.rsplit("-", 1)[0] if "-" in raw else raw
        for ext in (".png", ".jpg"):
            p = ZG_IMG / (base + ext)
            if p.exists():
                return p
    return None


def norm(s):
    return (s or "").strip()


print("[1/3] hashing new release images ...")
md5_to_new = {}
new_records = {}
with NEW_JSONL.open() as f:
    for line in f:
        r = json.loads(line)
        new_records[r["id"]] = r
for r in new_records.values():
    p = NEW_IMG / r["image"]
    md5_to_new[md5_file(p)] = r["id"]
print(f"  new images: {len(md5_to_new)} (expect 600)")

print("[2/3] hashing old unified_benchmark images ...")
old_records = {}
md5_to_old = {}
missing = 0
with UNIFIED.open() as f:
    for line in f:
        r = json.loads(line)
        old_records[r["id"]] = r
        p = resolve_old_image(r)
        if p is None:
            missing += 1
            continue
        try:
            md5_to_old.setdefault(md5_file(p), []).append(r["id"])
        except Exception as e:
            missing += 1
print(f"  old records: {len(old_records)}, hashed: {sum(len(v) for v in md5_to_old.values())}, missing: {missing}")

# Build old_id -> new_id (only when question + answer also match)
old_to_new = {}
qa_mismatch = 0
md5_no_new = 0
md5_no_old = 0
for md5_hex, new_id in md5_to_new.items():
    old_ids = md5_to_old.get(md5_hex)
    if not old_ids:
        md5_no_old += 1
        continue
    new_r = new_records[new_id]
    matched = None
    for oid in old_ids:
        old_r = old_records[oid]
        if norm(new_r["question"]) == norm(old_r["question"]) and norm(new_r["answer"]) == norm(old_r["answer"]):
            matched = oid
            break
    if matched is None:
        qa_mismatch += 1
        continue
    old_to_new[matched] = new_id

print(f"  matched (md5+qa): {len(old_to_new)} / 600")
print(f"  new images without old md5 match: {md5_no_old}")
print(f"  new images with md5 match but qa mismatch: {qa_mismatch}")

# Save mapping for inspection
(NEW_DIR / "eval" / "old_to_new_id.json").write_text(
    json.dumps(old_to_new, indent=2, ensure_ascii=False))

print("[3/3] remapping eval files ...")
total_in = total_out = 0
for src in sorted(EVAL_DIR.glob("eval_*.jsonl")):
    out_path = OUT_DIR / src.name
    n_in = n_out = 0
    with src.open() as fin, out_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_in += 1
            new_id = old_to_new.get(r["id"])
            if new_id is None:
                continue
            r["id"] = new_id
            fout.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1
    total_in += n_in
    total_out += n_out
    print(f"  {src.name}: {n_in} -> {n_out}")
print(f"total: {total_in} -> {total_out}")
