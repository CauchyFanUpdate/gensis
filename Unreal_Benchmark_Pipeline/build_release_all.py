#!/usr/bin/env python3
"""Convert unified_benchmark.jsonl (UnrealBench + ZeroGen merged dataset) into
the zerogen release format and copy images into release_all/images/.

Source: /mnt/tidal-alsh01/dataset/perceptionVLM/code_chenlifeng/new_benchmark/
        unrealbench/export_vlmevalkit/unified_benchmark.jsonl
        (633 rows: PHYS/BIO/CHEM/MATH (UB-*) + ZEROGen (ZG-*))

Output:
  release_all/dataset.json   — same shape as release/dataset.json so
                               eval_models_api.py / eval_models_vllm.py can
                               consume it via --dataset release_all/dataset.json
  release_all/images/*.png   — images copied from the various source dirs

Schema mapping:
  id       → sample_id              (UB-/ZG- prefix kept to avoid collisions)
  family   → family                 (codes: PHYS/BIO/CHEM/MATH/ZEROGen)
  qtype    → qtype                  (oe(mc) → oe, options dropped)
  subdomain→ domain                 (more specific than top-level domain)
  question → question
  answer   → correct_label (mc) | answer (oe)
  options  → [{label, parts:[text]}, ...] for mc only
  image    → image_path = release_all/images/<id>.png
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

# ── paths ────────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
SRC_JSONL = Path(
    "/mnt/tidal-alsh01/dataset/perceptionVLM/code_chenlifeng/new_benchmark/"
    "unrealbench/export_vlmevalkit/unified_benchmark.jsonl"
)
UNREALBENCH = Path("/mnt/tidal-alsh01/dataset/perceptionVLM/code_chenlifeng/new_benchmark/unrealbench")
UB_IMAGES_DIR = UNREALBENCH / "images"
UB_MATH_IMAGE_DIRS = [
    UNREALBENCH / "math_pipeline" / "output" / "images",
    UNREALBENCH / "math_pipeline" / "output_v2" / "images",
]
UB_MATH_STEP4 = [
    UNREALBENCH / "math_pipeline" / "output" / "step4_final.jsonl",
    UNREALBENCH / "math_pipeline" / "output_v2" / "step4_final.jsonl",
]
ZG_IMAGES_DIR = HERE / "release" / "images"

RELEASE_DIR = HERE / "release_all"
RELEASE_IMAGES_DIR = RELEASE_DIR / "images"
DATASET_PATH = RELEASE_DIR / "dataset.json"

# release_all/images/<id>.png  →  image_path = "release_all/images/<id>.png"
# this string is what eval scripts resolve via HERE / s["image_path"]
IMAGE_PATH_PREFIX = "release_all/images"


# ── helpers ──────────────────────────────────────────────────────────────


def _dumps(v) -> str:
    return json.dumps(v, ensure_ascii=False)


def format_entry(entry: dict, indent: str = "  ") -> str:
    """Pretty-print one entry: indent=2 BUT each option on a single line."""
    lines = [indent + "{"]
    keys = list(entry.keys())
    for i, k in enumerate(keys):
        tail = "," if i < len(keys) - 1 else ""
        v = entry[k]
        if k == "options" and isinstance(v, list):
            lines.append(f'{indent * 2}"options": [')
            for j, opt in enumerate(v):
                opt_tail = "," if j < len(v) - 1 else ""
                lines.append(f"{indent * 3}{_dumps(opt)}{opt_tail}")
            lines.append(f"{indent * 2}]{tail}")
        else:
            lines.append(f"{indent * 2}{_dumps(k)}: {_dumps(v)}{tail}")
    lines.append(indent + "}")
    return "\n".join(lines)


def format_dataset(entries: list[dict]) -> str:
    parts = ["["]
    for i, e in enumerate(entries):
        tail = "," if i < len(entries) - 1 else ""
        parts.append(format_entry(e) + tail)
    parts.append("]")
    return "\n".join(parts) + "\n"


# ── image resolution (port of step5_export_vlmevalkit.resolve_image_src) ──

_step4_cache: dict[str, Path] | None = None


def _load_step4_cache() -> dict[str, Path]:
    global _step4_cache
    if _step4_cache is not None:
        return _step4_cache
    cache: dict[str, Path] = {}
    for step4_path in UB_MATH_STEP4:
        if not step4_path.exists():
            continue
        with open(step4_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("status") != "ok":
                    continue
                raw_id = r.get("id")
                rel = r.get("edited_image_path", "")
                if not raw_id or not rel:
                    continue
                p = UNREALBENCH / "math_pipeline" / rel
                if p.exists():
                    cache.setdefault(raw_id, p)
    _step4_cache = cache
    return cache


def resolve_image_src(rid: str, family: str) -> Path | None:
    if rid.startswith(("UB-", "ADD-")):
        raw_sid = rid.split("-", 1)[1]
        if family == "MATH":
            for img_dir in UB_MATH_IMAGE_DIRS:
                for ext in (".png", ".jpg"):
                    p = img_dir / (raw_sid + ext)
                    if p.exists():
                        return p
            return _load_step4_cache().get(raw_sid)
        else:  # PHYS / BIO / CHEM
            for ext in (".png", ".jpg"):
                p = UB_IMAGES_DIR / (raw_sid + ext)
                if p.exists():
                    return p
            return None
    elif rid.startswith("ZG-"):
        raw_sid = rid[3:]
        for ext in (".png", ".jpg"):
            p = ZG_IMAGES_DIR / (raw_sid + ext)
            if p.exists():
                return p
        return None
    return None


# ── conversion ───────────────────────────────────────────────────────────


def convert_options(opts: dict | list) -> list[dict]:
    """Source MC options are dict {A: text, ...}. Convert to release format
    list [{label, parts:[text]}, ...]."""
    if isinstance(opts, dict):
        return [
            {"label": label, "parts": [str(text)]}
            for label, text in sorted(opts.items())
        ]
    if isinstance(opts, list):
        # already in release-ish format (defensive)
        out = []
        for o in opts:
            label = o.get("label", "")
            parts = o.get("parts")
            if not parts:
                txt = o.get("text", "")
                parts = [str(txt)] if txt else []
            out.append({"label": label, "parts": [str(p) for p in parts]})
        return out
    raise ValueError(f"unknown options shape: {type(opts).__name__}")


def build_entry(rec: dict) -> dict:
    rid = rec["id"]
    src_qtype = rec["qtype"]
    family = rec.get("family", "")

    # oe(mc) → oe: keep answer text, drop options
    qtype = "oe" if src_qtype == "oe(mc)" else src_qtype
    if qtype not in ("mc", "oe"):
        raise ValueError(f"unknown qtype {src_qtype!r} on {rid}")

    entry: dict = {
        "sample_id": rid,
        "family": family,
        "qtype": qtype,
        "domain": rec.get("subdomain") or rec.get("domain"),
        "question": rec.get("question", ""),
    }

    if qtype == "mc":
        opts = rec.get("options")
        if not opts:
            raise ValueError(f"mc sample {rid} has no options")
        entry["options"] = convert_options(opts)
        entry["correct_label"] = rec.get("answer")
    else:
        entry["answer"] = str(rec.get("answer", ""))

    entry["image_path"] = f"{IMAGE_PATH_PREFIX}/{rid}.png"
    return entry


def load_source() -> list[dict]:
    records: list[dict] = []
    with SRC_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def main() -> None:
    if not SRC_JSONL.exists():
        sys.exit(f"missing source: {SRC_JSONL}")

    records = load_source()
    print(f"loaded {len(records)} records from {SRC_JSONL.name}")

    # resolve image sources; skip records whose image we cannot find
    resolved: list[tuple[dict, Path]] = []
    missing: list[str] = []
    for r in records:
        src = resolve_image_src(r["id"], r.get("family", ""))
        if src is None:
            missing.append(r["id"])
        else:
            resolved.append((r, src))
    if missing:
        print(f"[warn] skipping {len(missing)} record(s) with no image:")
        for sid in missing[:20]:
            print(f"  - {sid}")
        if len(missing) > 20:
            print(f"  ... +{len(missing) - 20} more")
    print(f"resolved {len(resolved)} image source paths")

    # build entries
    entries = [build_entry(r) for r, _ in resolved]
    entries.sort(key=lambda e: e["sample_id"])

    # wipe + recreate release_all/
    if RELEASE_DIR.exists():
        shutil.rmtree(RELEASE_DIR)
        print(f"wiped existing {RELEASE_DIR}")
    RELEASE_DIR.mkdir(parents=True)
    RELEASE_IMAGES_DIR.mkdir(parents=True)

    # write dataset.json
    DATASET_PATH.write_text(format_dataset(entries), encoding="utf-8")
    size_kb = DATASET_PATH.stat().st_size / 1024
    print(f"wrote {DATASET_PATH} ({size_kb:.1f} KB)")

    # copy images (rename to <id>.png to match image_path)
    n_copied = 0
    for r, src in resolved:
        dst = RELEASE_IMAGES_DIR / f"{r['id']}.png"
        shutil.copy2(src, dst)
        n_copied += 1
    print(f"images: copied={n_copied} → {RELEASE_IMAGES_DIR}")

    # breakdown
    n_mc = sum(1 for e in entries if e["qtype"] == "mc")
    n_oe = sum(1 for e in entries if e["qtype"] == "oe")
    print(f"breakdown: mc={n_mc} oe={n_oe}")

    by_family: dict[str, int] = {}
    for e in entries:
        by_family[e["family"]] = by_family.get(e["family"], 0) + 1
    print("by family:")
    for fam, n in sorted(by_family.items(), key=lambda kv: -kv[1]):
        print(f"  {fam}: {n}")


if __name__ == "__main__":
    main()
