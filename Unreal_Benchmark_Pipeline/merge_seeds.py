#!/usr/bin/env python3
"""Merge the original seed_bank.json with all per-family batch files in seeds/.

Input:
  - seed_bank.json (top-level: benchmark_name, version, language, schema, families)
  - seeds/{FAMILY}_b{N}.json (top-level: family_id, family_name, seeds)

Output:
  - seed_bank_merged.json with the same top-level structure as seed_bank.json,
    but each family.seeds extended by the matching batch files.

For each family, the original 10 seeds keep their IDs (-001..-010); batch seeds
keep the IDs they were authored with. Brand-new families (no entry in
seed_bank.json) are appended to the families list in the order they first appear.

Re-runs are idempotent: writes seed_bank_merged.json fresh from sources every time.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT / "seed_bank.json"
SEEDS_DIR = ROOT / "seeds"
OUT = ROOT / "seed_bank_merged.json"


def load_original() -> dict:
    with ORIGINAL.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_batches() -> "OrderedDict[str, list[dict]]":
    """Return mapping family_id -> {family_name, seeds (concatenated)}."""
    files = sorted(glob.glob(str(SEEDS_DIR / "*_b*.json")))
    by_family: "OrderedDict[str, dict]" = OrderedDict()
    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fid = data["family_id"]
        fname = data["family_name"]
        seeds = data["seeds"]
        if fid not in by_family:
            by_family[fid] = {"family_name": fname, "seeds": []}
        by_family[fid]["seeds"].extend(seeds)
    return by_family


def merge() -> dict:
    base = load_original()
    batches = load_batches()

    families_out = []
    seen_family_ids = set()
    for fam in base["families"]:
        fid = fam["family_id"]
        seen_family_ids.add(fid)
        merged = {
            "family_id": fid,
            "family_name": fam["family_name"],
            "seeds": list(fam["seeds"]),
        }
        seen_seed_ids = {s["id"] for s in merged["seeds"]}
        if fid in batches:
            for s in batches[fid]["seeds"]:
                if s["id"] not in seen_seed_ids:
                    merged["seeds"].append(s)
                    seen_seed_ids.add(s["id"])
        families_out.append(merged)

    for fid, fdata in batches.items():
        if fid in seen_family_ids:
            continue
        families_out.append({
            "family_id": fid,
            "family_name": fdata["family_name"],
            "seeds": fdata["seeds"],
        })

    total = sum(len(f["seeds"]) for f in families_out)
    out = {
        "benchmark_name": "ZeroGen",
        "version": "v2",
        "language": "en",
        "schema": base.get("schema"),
        "families": families_out,
        "stats": {
            "num_families": len(families_out),
            "num_seeds": total,
        },
    }
    return out


def report(merged: dict) -> None:
    print(f"Families: {len(merged['families'])}")
    print(f"Total seeds: {merged['stats']['num_seeds']}")
    print()
    print(f"{'family':<8} {'count':>5}")
    print("-" * 16)
    dup_ids: dict[str, int] = {}
    for fam in merged["families"]:
        print(f"{fam['family_id']:<8} {len(fam['seeds']):>5}")
        for s in fam["seeds"]:
            dup_ids[s["id"]] = dup_ids.get(s["id"], 0) + 1
    dups = {k: v for k, v in dup_ids.items() if v > 1}
    if dups:
        print()
        print("WARNING: duplicate seed IDs:", dups)
        sys.exit(1)


def main() -> None:
    merged = merge()
    with OUT.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Wrote {OUT}")
    report(merged)


if __name__ == "__main__":
    main()
