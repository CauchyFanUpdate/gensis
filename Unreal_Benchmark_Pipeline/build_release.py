#!/usr/bin/env python3
"""Extract manually-verified samples and package them as a release dataset.

Reads manual_review.jsonl (latest-wins by sample_id), keeps records with
human_verified=True AND valid=True, applies any human corrections, and writes:
  release/dataset.json   — pretty-printed JSON array, one entry per sample
  release/images/*.png   — copied from images/<sample_id>.png

The release/ directory is wiped clean before writing so stale files from a
previous build do not leak through.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).parent
REVIEW_PATH = HERE / "manual_review.jsonl"
SEED_BANK_PATH = HERE / "seed_bank_merged.json"
SRC_IMAGES_DIR = HERE / "images"
RELEASE_DIR = HERE / "release"
RELEASE_IMAGES_DIR = RELEASE_DIR / "images"
DATASET_PATH = RELEASE_DIR / "dataset.json"


def load_family_names(path: Path) -> dict[str, str]:
    sb = json.loads(path.read_text(encoding="utf-8"))
    return {f["family_id"]: f["family_name"] for f in sb["families"]}


def _dumps(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def format_entry(entry: dict, indent: str = "  ") -> str:
    """Pretty-print one entry with indent=2 BUT collapse each item of `options`
    onto a single line."""
    lines = [indent + "{"]
    keys = list(entry.keys())
    for i, k in enumerate(keys):
        tail = "," if i < len(keys) - 1 else ""
        v = entry[k]
        if k == "options" and isinstance(v, list):
            lines.append(f'{indent*2}"options": [')
            for j, opt in enumerate(v):
                opt_tail = "," if j < len(v) - 1 else ""
                lines.append(f"{indent*3}{_dumps(opt)}{opt_tail}")
            lines.append(f"{indent*2}]{tail}")
        else:
            lines.append(f"{indent*2}{_dumps(k)}: {_dumps(v)}{tail}")
    lines.append(indent + "}")
    return "\n".join(lines)


def format_dataset(entries: list[dict]) -> str:
    parts = ["["]
    for i, e in enumerate(entries):
        tail = "," if i < len(entries) - 1 else ""
        parts.append(format_entry(e) + tail)
    parts.append("]")
    return "\n".join(parts) + "\n"


def load_latest_by_sample_id(path: Path) -> dict[str, dict]:
    by_sid: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sample_id")
            if sid:
                by_sid[sid] = rec
    return by_sid


def build_entry(rec: dict, family_names: dict[str, str]) -> dict:
    sid = rec["sample_id"]
    qtype = rec["qtype"]
    seed = rec.get("seed", {})
    s1 = rec.get("stage1", {})

    question = rec.get("question_corrected") or s1.get("question")
    fid = rec.get("family_id")

    entry: dict = {
        "sample_id": sid,
        "family": family_names.get(fid, fid),
        "qtype": qtype,
        "domain": seed.get("domain"),
        "question": question,
    }

    if qtype == "mc":
        options = rec.get("options_corrected") or s1.get("options")
        correct_label = rec.get("correct_label_corrected") or s1.get("correct_label")
        entry["options"] = options
        entry["correct_label"] = correct_label
    elif qtype == "oe":
        answer = rec.get("answer_corrected") or s1.get("answer")
        entry["answer"] = answer
    else:
        raise ValueError(f"unknown qtype {qtype!r} on {sid}")

    entry["image_path"] = f"images/{sid}.png"
    return entry


def main() -> None:
    if not REVIEW_PATH.exists():
        sys.exit(f"missing {REVIEW_PATH}")

    by_sid = load_latest_by_sample_id(REVIEW_PATH)
    selected = [r for r in by_sid.values()
                if r.get("human_verified") and r.get("valid")]
    selected.sort(key=lambda r: r["sample_id"])
    print(f"found {len(selected)} verified+valid samples "
          f"(total {len(by_sid)} unique sample_ids)")

    family_names = load_family_names(SEED_BANK_PATH)
    entries = [build_entry(r, family_names) for r in selected]

    missing_imgs: list[str] = []
    for e in entries:
        src = SRC_IMAGES_DIR / f"{e['sample_id']}.png"
        if not src.exists():
            missing_imgs.append(e["sample_id"])
    if missing_imgs:
        sys.exit(f"missing {len(missing_imgs)} image(s): {missing_imgs[:10]}")

    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
        print(f"wiped existing {RELEASE_DIR}")
    RELEASE_DIR.mkdir(parents=True)
    RELEASE_IMAGES_DIR.mkdir(parents=True)

    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        f.write(format_dataset(entries))
    print(f"wrote {DATASET_PATH} ({DATASET_PATH.stat().st_size / 1024:.1f} KB)")

    for e in entries:
        sid = e["sample_id"]
        shutil.copy2(SRC_IMAGES_DIR / f"{sid}.png",
                     RELEASE_IMAGES_DIR / f"{sid}.png")
    print(f"images: copied={len(entries)} → {RELEASE_IMAGES_DIR}")

    n_mc = sum(1 for e in entries if e["qtype"] == "mc")
    n_oe = sum(1 for e in entries if e["qtype"] == "oe")
    print(f"breakdown: mc={n_mc} oe={n_oe}")


if __name__ == "__main__":
    main()
