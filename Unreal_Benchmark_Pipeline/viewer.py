#!/usr/bin/env python3
"""
viewer.py — Flask web viewer for the zerogen pipeline (per-sample variant).

Each line in stage1/stage2/stage3 is now ONE SAMPLE (mc or oe), keyed by
`sample_id = "<seed_id>-<qtype>"`. The viewer treats every sample as an
independent reviewable item.

For each sample stage 3 produces:
    keep=true  ⇔ ANY verifier disagrees with gold (HARD sample, default review)
    keep=false ⇔ both verifiers agree with gold (easy, default discard)

Original stage1/stage2/stage3 outputs are READ-ONLY here. All human edits +
flags are written to one separate file:

    manual_review.jsonl

Schema of one manual_review.jsonl record (append-only; latest line per
sample_id wins on load):

    {
      "index":                   1,                // 1-based global ordinal
      "sample_id":               "COUNT-001-mc",
      "seed_id":                 "COUNT-001",
      "family_id":               "COUNT",
      "qtype":                   "mc" | "oe",
      "seed":                    { ... original seed ... },
      "stage1":                  { ... per-sample stage1 payload ... },
      "stage2":                  { ... stage2 record ... },
      "stage3":                  { ... stage3 record ... } | null,

      "human_verified":          true | false,
      "valid":                   true | false,    # default = stage3.keep

      # Per-sample human overrides (do not modify stage1; persisted here).
      "question_corrected":      null | "<text>",
      "options_corrected":       null | [ {label, parts: [..]}, ... ],   # mc only
      "correct_label_corrected": null | "<letter>",                       # mc only
      "answer_corrected":        null | "<short text>",                   # oe only

      # Human-W annotation fields (imported from human_correct.xlsx; editable).
      # null = not yet annotated. true/false carry the Chinese answer:
      #   humanw_image_broken:     true="有崩坏", false="无崩坏"
      #   humanw_question_ok:      true="合理",   false="不合理"
      #   humanw_correct_label_ok: true="正确",   false="不正确"
      #   humanw_keep:             true="保留",   false="不保留"
      "humanw_image_broken":     null | true | false,
      "humanw_question_ok":      null | true | false,
      "humanw_correct_label_ok": null | true | false,
      "humanw_keep":             null | true | false,

      "notes":                   "",
      "edited_at":               <unix epoch seconds>,
      "source":                  "init" | "human" | "humanw_import"
    }
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from threading import Lock

from flask import Flask, jsonify, render_template_string, request, send_file, abort

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
STAGE1_PATH = Path(os.environ.get("STAGE1_JSONL", str(HERE / "stage1_output.jsonl")))
STAGE2_PATH = Path(os.environ.get("STAGE2_JSONL", str(HERE / "stage2_output.jsonl")))
STAGE3_PATH = Path(os.environ.get("STAGE3_JSONL", str(HERE / "stage3_output.jsonl")))
MANUAL_PATH = Path(os.environ.get("MANUAL_JSONL", str(HERE / "manual_review.jsonl")))
IMAGES_DIR  = Path(os.environ.get("IMAGES_DIR",   str(HERE / "images")))

PORT = int(os.environ.get("PORT", "7987"))
HOST = os.environ.get("HOST", "0.0.0.0")

FAMILY_NAMES = {
    "COUNT": "Counting",
    "SPAT":  "Spatial",
    "READ":  "Reading text",
    "ATTR":  "Attribute & state",
    "COMP":  "Comparison",
    "EXIS":  "Existence & verification",
    "PHYS":  "Physics intuition",
    "GEOM":  "Geometry",
    "DOM":   "Domain reading",
    "TEMP":  "Temporal cues",
    "DOCU":  "Document reading",
    "MAPS":  "Maps / charts",
    "VIEW":  "Viewpoint / projection",
    "OPTI":  "Optics",
    "CONS":  "Consistency",
}

_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# ── In-memory cache ───────────────────────────────────────────────────────
_DATA: dict[str, dict] = {}
_DATA_LOCK = Lock()
_FILE_LOCK = Lock()


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def _load_originals() -> dict[str, dict]:
    """Read stage1+stage2+stage3 (read-only) → sample_id → bundle dict."""
    s1_records = _read_jsonl(STAGE1_PATH)
    s2_records = _read_jsonl(STAGE2_PATH)
    s3_records = _read_jsonl(STAGE3_PATH)

    by_sid: dict[str, dict] = {}
    # latest-wins per sample_id, only ok records used as origins
    for rec in s1_records:
        sid = rec.get("sample_id")
        if not sid or rec.get("status") != "ok":
            continue
        by_sid[sid] = {
            "sample_id": sid,
            "seed_id":   rec.get("seed_id"),
            "family_id": rec.get("family_id"),
            "qtype":     rec.get("qtype"),
            "seed":      rec.get("seed", {}),
            "stage1":    rec.get("stage1", {}),
            "stage2":    None,
            "stage3":    None,
        }

    s2_latest: dict[str, dict] = {}
    for rec in s2_records:
        sid = rec.get("sample_id")
        if sid:
            s2_latest[sid] = rec
    for sid, rec in s2_latest.items():
        if sid in by_sid:
            by_sid[sid]["stage2"] = rec

    s3_latest: dict[str, dict] = {}
    for rec in s3_records:
        sid = rec.get("sample_id")
        if sid:
            s3_latest[sid] = rec
    for sid, rec in s3_latest.items():
        if sid in by_sid:
            by_sid[sid]["stage3"] = rec

    return by_sid


def _read_manual_latest() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for rec in _read_jsonl(MANUAL_PATH):
        sid = rec.get("sample_id")
        if not sid:
            continue
        out[sid] = rec
    return out


def _make_init_record(orig: dict) -> dict:
    s3 = orig.get("stage3") or {}
    keep_default = bool(s3.get("keep", True)) if s3 else True
    return {
        "index":                   None,           # assigned in _refresh
        "sample_id":               orig["sample_id"],
        "seed_id":                 orig["seed_id"],
        "family_id":               orig["family_id"],
        "qtype":                   orig["qtype"],
        "seed":                    orig["seed"],
        "stage1":                  orig["stage1"],
        "stage2":                  orig["stage2"],
        "stage3":                  orig["stage3"],
        "human_verified":          False,
        "valid":                   keep_default,
        "question_corrected":      None,
        "options_corrected":       None,
        "correct_label_corrected": None,
        "answer_corrected":        None,
        "humanw_image_broken":     None,
        "humanw_question_ok":      None,
        "humanw_correct_label_ok": None,
        "humanw_keep":             None,
        "notes":                   "",
        "edited_at":               None,
        "source":                  "init",
    }


def _append_records(records: list[dict]) -> None:
    if not records:
        return
    MANUAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MANUAL_PATH, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _is_invalid(rec: dict) -> bool:
    """Invalid = both verifiers agreed with gold (any_disagree=False & stage3 ok).
    Pending stage3 (no verdict yet) counts as 'hard' so it stays in the front group.
    """
    s3 = rec.get("stage3") or {}
    if s3.get("status") != "ok":
        return False
    return not bool(s3.get("any_disagree"))


def _global_sort_key(rec: dict) -> tuple:
    # group 0 = hard / pending; group 1 = invalid (all-agree) → end of list
    group = 1 if _is_invalid(rec) else 0
    return (group,
            rec.get("family_id") or "",
            rec.get("seed_id") or "",
            0 if rec.get("qtype") == "mc" else 1,
            rec.get("sample_id") or "")


def _refresh() -> dict:
    originals = _load_originals()
    manual_latest = _read_manual_latest()

    merged: dict[str, dict] = {}
    bootstrapped_sids: set[str] = set()
    n_from_manual = 0
    n_bootstrapped = 0

    for sid, orig in originals.items():
        if sid in manual_latest:
            m = dict(manual_latest[sid])
            # always refresh upstream artifacts from disk
            m["seed_id"]   = orig["seed_id"]
            m["family_id"] = orig["family_id"]
            m["qtype"]     = orig["qtype"]
            m["seed"]      = orig["seed"]
            m["stage1"]    = orig["stage1"]
            m["stage2"]    = orig["stage2"]
            m["stage3"]    = orig["stage3"]
            s3 = orig.get("stage3") or {}
            keep_default = bool(s3.get("keep", True)) if s3 else True
            if "valid" not in m:
                m["valid"] = keep_default
            m.setdefault("index", None)
            m.setdefault("question_corrected", None)
            m.setdefault("options_corrected", None)
            m.setdefault("correct_label_corrected", None)
            m.setdefault("answer_corrected", None)
            m.setdefault("notes", "")
            m.setdefault("human_verified", False)
            m.setdefault("humanw_image_broken", None)
            m.setdefault("humanw_question_ok", None)
            m.setdefault("humanw_correct_label_ok", None)
            m.setdefault("humanw_keep", None)
            merged[sid] = m
            n_from_manual += 1
        else:
            init = _make_init_record(orig)
            merged[sid] = init
            bootstrapped_sids.add(sid)
            n_bootstrapped += 1

    # keep stale manual records (sample no longer in stage1) so edits aren't lost
    for sid, m in manual_latest.items():
        if sid not in merged:
            m.setdefault("index", None)
            merged[sid] = m

    # Assign global 1-based indices: hard/pending first, invalid (all-agree) last.
    sorted_sids = sorted(merged.keys(), key=lambda s: _global_sort_key(merged[s]))
    to_append: list[dict] = []
    n_reindexed = 0
    for i, sid in enumerate(sorted_sids, start=1):
        rec = merged[sid]
        old_idx = rec.get("index")
        if old_idx != i:
            rec["index"] = i
            n_reindexed += 1
        # Persist if newly bootstrapped, OR if existing record's index changed
        if sid in bootstrapped_sids or (sid in manual_latest and old_idx != i):
            to_append.append(rec)

    if to_append:
        with _FILE_LOCK:
            _append_records(to_append)

    with _DATA_LOCK:
        global _DATA
        _DATA = merged

    return {
        "total":        len(merged),
        "from_manual":  n_from_manual,
        "bootstrapped": n_bootstrapped,
        "reindexed":    n_reindexed,
    }


# ── Sidebar summary ───────────────────────────────────────────────────────

def _summary_for(rec: dict) -> dict:
    s2 = rec.get("stage2") or {}
    s3 = rec.get("stage3") or {}
    s3_ok = s3.get("status") == "ok"
    any_dis = bool(s3.get("any_disagree")) if s3_ok else False
    return {
        "index":          rec.get("index"),
        "sample_id":      rec["sample_id"],
        "seed_id":        rec.get("seed_id"),
        "family_id":      rec.get("family_id"),
        "qtype":          rec.get("qtype"),
        "sub_skill":      rec.get("seed", {}).get("sub_skill", ""),
        "domain":         rec.get("seed", {}).get("domain", ""),
        "stage2_status":  s2.get("status") if s2 else "pending",
        "stage3_status":  s3.get("status") if s3 else "pending",
        "any_disagree":   any_dis,
        "valid":          bool(rec.get("valid", True)),
        "human_verified": bool(rec.get("human_verified")),
        "humanw_image_broken":     rec.get("humanw_image_broken"),
        "humanw_question_ok":      rec.get("humanw_question_ok"),
        "humanw_correct_label_ok": rec.get("humanw_correct_label_ok"),
        "humanw_keep":             rec.get("humanw_keep"),
    }


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(_INDEX_HTML, family_names=FAMILY_NAMES)


@app.route("/api/samples")
def api_samples():
    """
    Sidebar list. Filters via query string (CSV):
      family=COUNT,SPAT
      qtype=mc,oe
      disagree=any|none
      verified=yes|no
      valid=yes|no
    """
    with _DATA_LOCK:
        items = [_summary_for(r) for r in _DATA.values()]
    # Sort by global index (assigned in _refresh): hard first, invalid last.
    items.sort(key=lambda x: (x["index"] if x["index"] is not None else 10**9,
                              x["sample_id"] or ""))

    fam = request.args.get("family", "").strip()
    if fam:
        wanted = {s.strip() for s in fam.split(",") if s.strip()}
        items = [x for x in items if x["family_id"] in wanted]

    qt = request.args.get("qtype", "").strip()
    if qt:
        wanted_q = {s.strip() for s in qt.split(",") if s.strip()}
        items = [x for x in items if x["qtype"] in wanted_q]

    dis = request.args.get("disagree", "").strip().lower()
    if dis == "any":
        items = [x for x in items if x["any_disagree"]]
    elif dis == "none":
        items = [x for x in items if not x["any_disagree"] and x["stage3_status"] == "ok"]

    # Human-W filters: each is yes (true) / no (false) / unset (null)
    for fld in ("humanw_image_broken", "humanw_question_ok",
                "humanw_correct_label_ok", "humanw_keep"):
        f = request.args.get(fld, "").strip().lower()
        if f in ("yes", "true", "1"):
            items = [x for x in items if x.get(fld) is True]
        elif f in ("no", "false", "0"):
            items = [x for x in items if x.get(fld) is False]
        elif f in ("unset", "null", "none"):
            items = [x for x in items if x.get(fld) is None]

    ver = request.args.get("verified", "").strip().lower()
    if ver in ("yes", "true", "1"):
        items = [x for x in items if x["human_verified"]]
    elif ver in ("no", "false", "0"):
        items = [x for x in items if not x["human_verified"]]

    val = request.args.get("valid", "").strip().lower()
    if val in ("yes", "true", "1"):
        items = [x for x in items if x["valid"]]
    elif val in ("no", "false", "0"):
        items = [x for x in items if not x["valid"]]

    counts = defaultdict(int)
    for x in items:
        counts["qtype_" + (x["qtype"] or "?")] += 1
        if x["any_disagree"]:
            counts["disagree"] += 1
        if x["human_verified"]:
            counts["verified"] += 1
        if x["valid"]:
            counts["valid"] += 1
        if x.get("humanw_keep") is True:
            counts["humanw_keep"] += 1
        if x.get("humanw_keep") is False:
            counts["humanw_drop"] += 1
        if x.get("humanw_image_broken") is True:
            counts["humanw_broken"] += 1
    counts["total"] = len(items)
    return jsonify({"items": items, "counts": dict(counts)})


@app.route("/api/sample/<sample_id>")
def api_sample(sample_id: str):
    with _DATA_LOCK:
        rec = _DATA.get(sample_id)
    if not rec:
        return jsonify({"error": f"sample_id {sample_id} not found"}), 404
    return jsonify(rec)


@app.route("/img/<sample_id>")
def img(sample_id: str):
    with _DATA_LOCK:
        rec = _DATA.get(sample_id)
    if not rec:
        abort(404)
    s2 = rec.get("stage2") or {}
    rel = s2.get("image_path")
    if not rel:
        abort(404)
    p = (HERE / rel).resolve()
    if not p.exists():
        abort(404)
    return send_file(str(p), mimetype="image/png")


def _normalize_options(raw) -> list[dict] | None:
    """Accepts a list of {label, parts:[...]} or a list of strings/list-of-parts.

    Returns sanitized options or None if invalid.
    """
    if not isinstance(raw, list) or not raw:
        return None
    out = []
    for i, opt in enumerate(raw):
        if i >= len(_LABELS):
            return None
        expected_label = _LABELS[i]
        if isinstance(opt, dict):
            parts = opt.get("parts")
        elif isinstance(opt, list):
            parts = opt
        elif isinstance(opt, str):
            parts = [opt]
        else:
            return None
        if not isinstance(parts, list) or not parts:
            return None
        clean: list[str] = []
        for p in parts:
            if p is None:
                continue
            ps = str(p).strip()
            if ps:
                clean.append(ps)
        if not clean:
            return None
        out.append({"label": expected_label, "parts": clean})
    return out


@app.route("/api/edit/<sample_id>", methods=["POST"])
def api_edit(sample_id: str):
    """
    Body fields (all optional):
      valid:                    bool
      human_verified:           bool
      notes:                    str
      question_corrected:       str | null
      answer_corrected:         str | null    (oe only)
      options_corrected:        list | null   (mc only)
      correct_label_corrected:  str | null    (mc only)
    """
    body = request.get_json(silent=True) or {}

    with _DATA_LOCK:
        cur = _DATA.get(sample_id)
        if not cur:
            return jsonify({"error": f"sample_id {sample_id} not found"}), 404
        new_rec = json.loads(json.dumps(cur))

    qtype = new_rec.get("qtype")
    changed = False

    if "valid" in body:
        v = bool(body["valid"])
        if v != bool(new_rec.get("valid", True)):
            new_rec["valid"] = v
            changed = True

    if "human_verified" in body:
        hv = bool(body["human_verified"])
        if hv != bool(new_rec.get("human_verified")):
            new_rec["human_verified"] = hv
            changed = True

    if "notes" in body and body["notes"] is not None:
        nt = str(body["notes"])
        if nt != (new_rec.get("notes") or ""):
            new_rec["notes"] = nt
            changed = True

    if "question_corrected" in body:
        v = body["question_corrected"]
        v_norm = None if v in (None, "") else str(v).strip()
        if v_norm != new_rec.get("question_corrected"):
            new_rec["question_corrected"] = v_norm
            changed = True

    if qtype == "oe" and "answer_corrected" in body:
        v = body["answer_corrected"]
        v_norm = None if v in (None, "") else str(v).strip()
        if v_norm != new_rec.get("answer_corrected"):
            new_rec["answer_corrected"] = v_norm
            changed = True

    if qtype == "mc" and "options_corrected" in body:
        v = body["options_corrected"]
        if v in (None, []):
            v_norm = None
        else:
            v_norm = _normalize_options(v)
            if v_norm is None:
                return jsonify({"error": "options_corrected: invalid shape"}), 400
        if v_norm != new_rec.get("options_corrected"):
            new_rec["options_corrected"] = v_norm
            changed = True

    # Human-W annotation fields (tri-state: true / false / null)
    for fld in ("humanw_image_broken", "humanw_question_ok",
                "humanw_correct_label_ok", "humanw_keep"):
        if fld in body:
            v = body[fld]
            if v is None or v == "":
                v_norm = None
            elif isinstance(v, bool):
                v_norm = v
            elif isinstance(v, str):
                s = v.strip().lower()
                if s in ("true", "1", "yes", "有", "合理", "正确", "保留"):
                    v_norm = True
                elif s in ("false", "0", "no", "无", "不合理", "不正确", "不保留"):
                    v_norm = False
                else:
                    v_norm = None
            else:
                v_norm = bool(v)
            if v_norm != new_rec.get(fld):
                new_rec[fld] = v_norm
                changed = True

    if qtype == "mc" and "correct_label_corrected" in body:
        v = body["correct_label_corrected"]
        v_norm = None if v in (None, "") else str(v).strip().upper() or None
        # validate against options (corrected if present, else stage1)
        if v_norm is not None:
            opts = (new_rec.get("options_corrected")
                    or (new_rec.get("stage1") or {}).get("options")
                    or [])
            valid_labels = {o.get("label") for o in opts if isinstance(o, dict)}
            if v_norm not in valid_labels:
                return jsonify({"error":
                    f"correct_label_corrected={v_norm!r} not in option labels {sorted(valid_labels)}"
                }), 400
        if v_norm != new_rec.get("correct_label_corrected"):
            new_rec["correct_label_corrected"] = v_norm
            changed = True

    if not changed:
        return jsonify({"ok": True, "no_change": True})

    new_rec["edited_at"] = int(time.time())
    new_rec["source"] = "human"

    with _FILE_LOCK:
        _append_records([new_rec])

    with _DATA_LOCK:
        _DATA[sample_id] = new_rec

    return jsonify({"ok": True, "record": new_rec})


@app.route("/api/reload", methods=["POST"])
def api_reload():
    info = _refresh()
    return jsonify({"ok": True, **info})


# ── HTML (single-page app) ────────────────────────────────────────────────

_INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>zerogen viewer (per-sample)</title>
  <style>
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
    body { display: flex; flex-direction: column; }

    header {
      flex: 0 0 auto; background: #20232a; color: #fff;
      padding: 8px 14px; display: flex; align-items: center; gap: 12px;
      border-bottom: 1px solid #444; flex-wrap: wrap;
    }
    header h1 { font-size: 14px; margin: 0; font-weight: 600; }
    header .stats { font-size: 12px; opacity: 0.85; }
    header label { font-size: 12px; display: flex; align-items: center; gap: 4px; }
    header select {
      font-size: 12px; padding: 3px 6px; border-radius: 3px; border: 1px solid #555;
      background: #2c303a; color: #eee;
    }
    header button {
      font-size: 12px; padding: 4px 10px; border-radius: 3px; border: 1px solid #555;
      background: #3a3f4b; color: #eee; cursor: pointer;
    }
    header button:hover { background: #4a505d; }

    .main { display: flex; flex: 1 1 auto; overflow: hidden; }
    .sidebar {
      flex: 0 0 320px; border-right: 1px solid #ddd; overflow-y: auto;
      background: #fafafa;
    }
    .sidebar .group-header {
      position: sticky; top: 0; background: #eee; padding: 4px 10px;
      font-size: 12px; font-weight: 600; color: #333;
      border-top: 1px solid #ccc; border-bottom: 1px solid #ccc; z-index: 1;
    }
    .sidebar .item {
      padding: 5px 10px; font-size: 12px; cursor: pointer;
      border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 6px;
    }
    .sidebar .item:hover { background: #e8eef7; }
    .sidebar .item.active { background: #d6e4f6; }
    .sidebar .item.hard { background: #fff3cd; }
    .sidebar .item.hard.active { background: #ffe580; }
    .sidebar .item .idx { font-family: ui-monospace, monospace; font-weight: 700;
                            color: #888; min-width: 32px; text-align: right; }
    .sidebar .item .sid { font-family: ui-monospace, monospace; font-weight: 600; }
    .sidebar .item .sub { color: #555; font-size: 11px; flex: 1; min-width: 0;
                          overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .sidebar .item .marks { display: flex; gap: 3px; flex: 0 0 auto; }
    .badge {
      font-size: 10px; padding: 1px 5px; border-radius: 8px; color: #fff;
      font-weight: 600; text-transform: uppercase;
    }
    .badge.mc      { background: #1a73e8; }
    .badge.oe      { background: #c43d8a; }
    .badge.hard    { background: #d18000; }
    .badge.agree   { background: #1e8e3e; }
    .badge.unknown { background: #888; }
    .mark { font-size: 11px; padding: 0 4px; border-radius: 8px; font-weight: 700; }
    .mark.verified  { background: #2962ff; color: #fff; }
    .mark.invalid   { background: #b00020; color: #fff; }
    .mark.hw-broken { background: #d93025; color: #fff; }
    .mark.hw-drop   { background: #777;    color: #fff; }
    .mark.hw-keep   { background: #1e8e3e; color: #fff; }
    .mark.hw-badlbl { background: #f29900; color: #fff; }

    .humanw-panel {
      margin-top: 14px; padding: 10px 14px; background: #fff3e0;
      border: 1px solid #f4b863; border-radius: 4px;
    }
    .humanw-panel h3 {
      margin: 0 0 8px 0; font-size: 13px; color: #7a4c00;
    }
    .humanw-panel .hw-row {
      display: flex; flex-wrap: wrap; gap: 14px 22px; align-items: center;
    }
    .humanw-panel .hw-row label {
      display: flex; align-items: center; gap: 6px; font-size: 13px; color: #333;
    }
    .humanw-panel select {
      font-size: 13px; padding: 3px 6px; border-radius: 3px; border: 1px solid #bbb;
      background: #fff;
    }
    .humanw-panel .hw-key {
      font-weight: 600; color: #5a3a00; min-width: 84px; display: inline-block;
    }

    .content {
      flex: 1 1 auto; overflow-y: auto; padding: 16px 24px; background: #fff;
    }
    .placeholder { color: #888; padding: 40px; text-align: center; }

    .sample-head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 6px; flex-wrap: wrap; }
    .sample-head h2 { font-size: 18px; margin: 0; font-family: ui-monospace, monospace; }
    .sample-head .meta { color: #666; font-size: 13px; }

    .qpanel {
      border: 1px solid #ddd; border-radius: 6px; padding: 14px 16px;
      background: #fcfcfc; margin-top: 12px;
    }
    .qpanel.disagree { border: 2px solid #d18000; background: #fff8e8; }
    .qpanel h3 { font-size: 14px; margin: 0 0 8px 0; display: flex;
                 justify-content: space-between; align-items: center; }
    .qpanel .row {
      display: flex; gap: 16px; align-items: flex-start;
    }
    .qpanel .col-img {
      flex: 0 0 360px; max-width: 50%;
    }
    .qpanel .col-img img {
      width: 100%; max-height: 480px; object-fit: contain; border: 1px solid #ddd;
      border-radius: 3px; background: #f0f0f0; display: block;
    }
    .qpanel .no-img {
      width: 100%; aspect-ratio: 1; display: flex; align-items: center; justify-content: center;
      background: #f3f3f3; color: #888; font-size: 12px; border: 1px dashed #bbb; border-radius: 3px;
    }
    .qpanel .col-q { flex: 1 1 0; min-width: 0; }
    .qpanel .qlabel { font-size: 11px; font-weight: 600; text-transform: uppercase;
                       color: #555; margin-bottom: 3px; }
    .qpanel .qtext { font-size: 14px; line-height: 1.4; padding: 6px 8px; background: #fff;
                      border: 1px solid #ddd; border-radius: 3px; }
    .qpanel .qtext.editable { background: #fff; }
    .qpanel textarea.qtext-edit {
      width: 100%; padding: 6px 8px; font-size: 14px; font-family: inherit;
      border: 1px solid #ccc; border-radius: 3px; background: #fcfcfc;
      resize: vertical; min-height: 50px;
    }

    .options-edit { margin-top: 8px; }
    .options-edit .opt-row {
      display: flex; align-items: center; gap: 8px; margin-bottom: 4px;
    }
    .options-edit .opt-row.gold { background: #e6f4ea; padding: 2px 4px; border-radius: 3px; }
    .options-edit .opt-label {
      width: 22px; text-align: center; font-family: ui-monospace, monospace;
      font-weight: 700; color: #444;
    }
    .options-edit .opt-input {
      flex: 1; padding: 4px 8px; font-size: 13px; border: 1px solid #ccc; border-radius: 3px;
    }
    .options-edit .gold-radio { cursor: pointer; }
    .options-edit .opt-del {
      font-size: 14px; padding: 0 8px; cursor: pointer; border: 1px solid #aaa;
      background: #f5f5f5; border-radius: 3px; line-height: 1.6;
    }
    .options-edit .opt-add {
      font-size: 12px; padding: 4px 12px; margin-top: 4px; cursor: pointer;
      border: 1px solid #aaa; background: #f5f5f5; border-radius: 3px;
    }
    .options-edit .hint { font-size: 11px; color: #666; margin-top: 4px; }
    .options-edit .hint .warn { color: #b00020; font-weight: 600; }

    .answer-edit { margin-top: 8px; }
    .answer-edit input {
      width: 100%; padding: 4px 8px; font-size: 14px; border: 1px solid #ccc; border-radius: 3px;
      font-family: ui-monospace, monospace;
    }

    .field { margin: 12px 0; }
    .field label.field-label {
      display: block; font-size: 11px; font-weight: 600;
      text-transform: uppercase; color: #555; margin-bottom: 3px;
    }

    .verifier-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
    .verifier-card {
      border: 1px solid #ddd; border-radius: 4px; padding: 8px 10px;
      background: #fcfcfc; font-size: 12px;
    }
    .verifier-card.pass { border-left: 4px solid #1e8e3e; }
    .verifier-card.fail { border-left: 4px solid #d93025; }
    .verifier-card.error { border-left: 4px solid #ff8c00; }
    .verifier-card h4 { font-size: 12px; margin: 0 0 4px 0; display: flex;
                         gap: 6px; align-items: center; justify-content: space-between; }
    .verifier-card .row2 { font-size: 12px; color: #444; margin: 2px 0; }
    .verifier-card .row2 b { display: inline-block; min-width: 80px; color: #222; }
    .verifier-card .reasoning {
      margin-top: 4px; font-size: 11px; color: #333; white-space: pre-wrap;
      max-height: 130px; overflow-y: auto; padding: 4px 6px; background: #f7f7f7;
      border-radius: 3px;
    }
    .verifier-card .judgement-true  { color: #1e8e3e; font-weight: 600; }
    .verifier-card .judgement-false { color: #d93025; font-weight: 600; }
    .verifier-card .judgement-null  { color: #888; font-weight: 600; }

    .global-controls {
      margin-top: 14px; display: flex; gap: 14px; align-items: center; flex-wrap: wrap;
      padding: 10px 14px; background: #f3f7fb; border: 1px solid #d3dde8; border-radius: 4px;
    }
    .global-controls label { display: flex; align-items: center; gap: 6px; cursor: pointer;
                              font-size: 13px; }
    .global-controls label.verified-on { color: #2962ff; font-weight: 600; }
    .global-controls label.invalid-on { color: #b00020; font-weight: 600; }

    .field textarea {
      width: 100%; padding: 6px 8px; font-size: 13px; font-family: inherit;
      border: 1px solid #ccc; border-radius: 3px; background: #fcfcfc;
      resize: vertical; min-height: 60px;
    }

    .actions { margin-top: 6px; display: flex; gap: 8px; align-items: center; }
    .actions button {
      font-size: 13px; padding: 6px 14px; border-radius: 3px;
      border: 1px solid #aaa; background: #f5f5f5; cursor: pointer;
    }
    .actions button.primary { background: #1a73e8; color: #fff; border-color: #1a73e8; }
    .actions button.primary:hover { background: #1762c4; }
    .actions .save-status { font-size: 12px; color: #555; margin-left: 6px; }
    .actions .save-status.ok    { color: #1e8e3e; }
    .actions .save-status.error { color: #d93025; }

    details { margin-top: 14px; border-top: 1px solid #eee; padding-top: 8px; }
    details > summary { cursor: pointer; font-size: 13px; font-weight: 600; color: #333; }
    details .body {
      padding: 8px 4px; font-size: 13px; color: #333; white-space: pre-wrap;
      max-height: 360px; overflow-y: auto;
    }
    details pre { background: #f5f5f5; padding: 8px; border-radius: 3px;
                   font-size: 12px; max-height: 320px; overflow: auto; margin: 0; }

    .small { font-size: 11px; color: #888; margin-top: 6px; }
    code { font-family: ui-monospace, monospace; }
  </style>
</head>
<body>

<header>
  <h1>zerogen viewer · per-sample</h1>
  <span class="stats" id="stats">loading…</span>
  <label>family
    <select id="filter-family" multiple size="1">
      {% for fid, fname in family_names.items() %}
      <option value="{{ fid }}">{{ fid }}</option>
      {% endfor %}
    </select>
  </label>
  <label>qtype
    <select id="filter-qtype">
      <option value="">all</option>
      <option value="mc">mc only</option>
      <option value="oe">oe only</option>
    </select>
  </label>
  <label>disagree
    <select id="filter-disagree">
      <option value="">all</option>
      <option value="any">HARD (any disagree)</option>
      <option value="none">all-agree</option>
    </select>
  </label>
  <label>verified
    <select id="filter-verified">
      <option value="">all</option>
      <option value="yes">yes</option>
      <option value="no">no</option>
    </select>
  </label>
  <label>valid
    <select id="filter-valid">
      <option value="">all</option>
      <option value="yes">valid</option>
      <option value="no">invalid</option>
    </select>
  </label>
  <label title="人工：图片有无崩坏">img崩坏
    <select id="filter-humanw_image_broken">
      <option value="">all</option>
      <option value="yes">有</option>
      <option value="no">无</option>
      <option value="unset">未标</option>
    </select>
  </label>
  <label title="人工：问题是否合理">题合理
    <select id="filter-humanw_question_ok">
      <option value="">all</option>
      <option value="yes">合理</option>
      <option value="no">不合理</option>
      <option value="unset">未标</option>
    </select>
  </label>
  <label title="人工：correct_label 是否正确">label正确
    <select id="filter-humanw_correct_label_ok">
      <option value="">all</option>
      <option value="yes">正确</option>
      <option value="no">不正确</option>
      <option value="unset">未标</option>
    </select>
  </label>
  <label title="人工：是否保留（两模型都对=不保留）">保留
    <select id="filter-humanw_keep">
      <option value="">all</option>
      <option value="yes">保留</option>
      <option value="no">不保留</option>
      <option value="unset">未标</option>
    </select>
  </label>
  <button id="btn-reload" title="re-read all JSONL files from disk">reload</button>
</header>

<div class="main">
  <aside class="sidebar" id="sidebar">loading…</aside>
  <section class="content" id="content">
    <div class="placeholder">Select a sample from the left.</div>
  </section>
</div>

<script>
const FAMILY_NAMES = {{ family_names | tojson }};
const LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
let SAMPLE_LIST = [];
let CURRENT_SID = null;
let CURRENT_REC = null;
let DIRTY = false;
let MC_OPT_STATE = [];     // editable copy: [{label, parts:[..]}]
let MC_GOLD_LABEL = null;  // editable correct label

function el(tag, props = {}, children = []) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === 'class') e.className = v;
    else if (k === 'html') e.innerHTML = v;
    else if (k === 'text') e.textContent = v;
    else if (k.startsWith('on')) e.addEventListener(k.slice(2), v);
    else if (k === 'checked') { if (v) e.checked = true; }
    else e.setAttribute(k, v);
  }
  for (const c of children) {
    if (c == null) continue;
    e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  }
  return e;
}

function escapeHtml(s) {
  if (s == null) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function markDirty() { DIRTY = true; updateSaveStatus(''); }
function updateSaveStatus(msg, cls) {
  const e = document.getElementById('save-status');
  if (!e) return;
  e.textContent = msg || (DIRTY ? 'unsaved changes' : '');
  e.className = 'save-status' + (cls ? ' ' + cls : '');
}

async function loadList() {
  const params = new URLSearchParams();
  const fams = Array.from(document.getElementById('filter-family').selectedOptions)
                    .map(o => o.value).filter(v => v);
  if (fams.length) params.set('family', fams.join(','));
  const qt = document.getElementById('filter-qtype').value;
  if (qt) params.set('qtype', qt);
  const dis = document.getElementById('filter-disagree').value;
  if (dis) params.set('disagree', dis);
  const ver = document.getElementById('filter-verified').value;
  if (ver) params.set('verified', ver);
  const val = document.getElementById('filter-valid').value;
  if (val) params.set('valid', val);
  for (const fld of ['humanw_image_broken', 'humanw_question_ok',
                     'humanw_correct_label_ok', 'humanw_keep']) {
    const v = document.getElementById('filter-' + fld).value;
    if (v) params.set(fld, v);
  }

  const r = await fetch('/api/samples?' + params.toString());
  const data = await r.json();
  SAMPLE_LIST = data.items;
  const c = data.counts;
  document.getElementById('stats').textContent =
    `total=${c.total||0} | mc=${c.qtype_mc||0} oe=${c.qtype_oe||0} | ` +
    `hard=${c.disagree||0} | verified=${c.verified||0} valid=${c.valid||0} | ` +
    `留=${c.humanw_keep||0} 丢=${c.humanw_drop||0} 崩=${c.humanw_broken||0}`;

  const sb = document.getElementById('sidebar');
  sb.innerHTML = '';
  let lastSeed = null;
  for (const item of SAMPLE_LIST) {
    if (item.seed_id !== lastSeed) {
      lastSeed = item.seed_id;
      const fname = FAMILY_NAMES[item.family_id] || item.family_id;
      sb.appendChild(el('div', { class: 'group-header',
        text: `${item.seed_id} — ${fname} · ${item.sub_skill || ''}` }));
    }
    const isHard = item.any_disagree;
    const marks = el('span', { class: 'marks' }, [
      el('span', { class: 'badge ' + (item.qtype || 'unknown'), text: item.qtype || '?' }),
      isHard ? el('span', { class: 'badge hard', text: 'HARD' })
             : (item.stage3_status === 'ok'
                 ? el('span', { class: 'badge agree', text: 'AGREE' })
                 : el('span', { class: 'badge unknown', text: '?' })),
      item.human_verified ? el('span', { class: 'mark verified', text: '✓' }) : null,
      !item.valid ? el('span', { class: 'mark invalid', text: '✗' }) : null,
      item.humanw_image_broken === true ? el('span', { class: 'mark hw-broken', text: '崩' }) : null,
      item.humanw_correct_label_ok === false ? el('span', { class: 'mark hw-badlbl', text: 'L?' }) : null,
      item.humanw_keep === false ? el('span', { class: 'mark hw-drop', text: '丢' }) :
        (item.humanw_keep === true ? el('span', { class: 'mark hw-keep', text: '留' }) : null),
    ]);
    const cls = 'item' +
      (item.sample_id === CURRENT_SID ? ' active' : '') +
      (isHard ? ' hard' : '');
    const div = el('div', {
      class: cls,
      'data-sid': item.sample_id,
      onclick: () => selectSample(item.sample_id),
    }, [
      el('span', { class: 'idx', text: '#' + (item.index != null ? item.index : '?') }),
      el('span', { class: 'sid', text: (item.qtype || '?').toUpperCase() }),
      el('span', { class: 'sub', text: item.domain }),
      marks,
    ]);
    sb.appendChild(div);
  }
  if (!SAMPLE_LIST.length) {
    sb.appendChild(el('div', { class: 'placeholder', text: '(no samples match filters)' }));
  }
}

async function selectSample(sid) {
  if (DIRTY) {
    if (!confirm('You have unsaved edits. Discard them?')) return;
  }
  CURRENT_SID = sid;
  DIRTY = false;
  document.querySelectorAll('.sidebar .item').forEach(e => {
    e.classList.toggle('active', e.dataset.sid === sid);
  });

  const r = await fetch('/api/sample/' + encodeURIComponent(sid));
  if (!r.ok) {
    document.getElementById('content').innerHTML =
      '<div class="placeholder">Failed to load: ' + escapeHtml(sid) + '</div>';
    return;
  }
  const rec = await r.json();
  CURRENT_REC = rec;
  renderSample(rec);
}

function deepCopyOptions(opts) {
  if (!Array.isArray(opts)) return [];
  return opts.map(o => ({
    label: o.label,
    parts: Array.isArray(o.parts) ? o.parts.slice() : [],
  }));
}

function effectiveOptions(rec) {
  if (rec.options_corrected && rec.options_corrected.length) return rec.options_corrected;
  return (rec.stage1 || {}).options || [];
}

function effectiveCorrectLabel(rec) {
  if (rec.correct_label_corrected) return rec.correct_label_corrected;
  return (rec.stage1 || {}).correct_label || null;
}

function effectiveQuestion(rec) {
  if (rec.question_corrected != null) return rec.question_corrected;
  return (rec.stage1 || {}).question || '';
}

function effectiveAnswer(rec) {
  if (rec.answer_corrected != null) return rec.answer_corrected;
  return (rec.stage1 || {}).answer || '';
}

function renderSample(rec) {
  const seed = rec.seed || {};
  const stage1 = rec.stage1 || {};
  const stage2 = rec.stage2 || {};
  const stage3 = rec.stage3 || {};
  const qtype = rec.qtype;

  const c = document.getElementById('content');
  c.innerHTML = '';

  const idxLabel = rec.index != null ? `#${rec.index}` : '#?';
  const head = el('div', { class: 'sample-head' }, [
    el('h2', { text: `${idxLabel} · ${rec.sample_id}` }),
    el('span', { class: 'meta', text:
      `${rec.family_id} · ${qtype} · ${seed.sub_skill || ''} · ${seed.domain || ''}` }),
    rec.human_verified ? el('span', { class: 'mark verified', text: '✓ verified' }) : null,
    !rec.valid ? el('span', { class: 'mark invalid', text: '✗ invalid' }) : null,
  ]);
  c.appendChild(head);

  c.appendChild(buildPanel(rec));

  c.appendChild(buildHumanWPanel(rec));

  // Global controls
  const verifiedCb = el('input', { type: 'checkbox', id: 'flag-verified',
                                     checked: !!rec.human_verified });
  verifiedCb.addEventListener('change', () => {
    document.querySelector('label.verified-toggle')
            .classList.toggle('verified-on', verifiedCb.checked);
    markDirty();
  });

  const validCb = el('input', { type: 'checkbox', id: 'flag-valid',
                                  checked: rec.valid !== false });
  validCb.addEventListener('change', () => {
    document.querySelector('label.valid-toggle')
            .classList.toggle('invalid-on', !validCb.checked);
    markDirty();
  });

  const keep = stage3 && stage3.keep;
  c.appendChild(el('div', { class: 'global-controls' }, [
    el('label', { class: 'verified-toggle' + (rec.human_verified ? ' verified-on' : '') }, [
      verifiedCb, document.createTextNode('Human verified ✓'),
    ]),
    el('label', { class: 'valid-toggle' + (rec.valid === false ? ' invalid-on' : '') }, [
      validCb, document.createTextNode(`Sample is valid (default keep=${keep})`),
    ]),
  ]));

  const notesField = el('div', { class: 'field' }, [
    el('label', { class: 'field-label', text: 'Notes (optional, free text)' }),
    el('textarea', { id: 'notes-text', rows: '2' }),
  ]);
  c.appendChild(notesField);
  notesField.querySelector('textarea').value = rec.notes || '';
  notesField.querySelector('textarea').addEventListener('input', markDirty);

  c.appendChild(el('div', { class: 'actions' }, [
    el('button', { class: 'primary', onclick: doSave }, ['Save']),
    el('span', { class: 'save-status', id: 'save-status' }),
  ]));

  if (rec.edited_at) {
    const dt = new Date(rec.edited_at * 1000).toLocaleString();
    c.appendChild(el('div', { class: 'small',
      text: `last human edit: ${dt} (source=${rec.source||'?'})` }));
  }

  c.appendChild(detailsBlockJSON('Test point analysis (shared)', stage1.test_point_analysis));
  c.appendChild(detailsBlock('Image prompt', stage1.image_prompt || '(none)'));
  c.appendChild(detailsBlock('Elaborated scene', stage1.elaborated_scene || '(none)'));
  c.appendChild(detailsBlock('Answer derivation', stage1.answer_derivation || '(none)'));
  c.appendChild(detailsBlockJSON('Decision notes', stage1.decision_notes));
  c.appendChild(detailsBlockJSON('Original seed', seed));
}

function buildPanel(rec) {
  const qtype = rec.qtype;
  const stage1 = rec.stage1 || {};
  const stage2 = rec.stage2 || {};
  const stage3 = rec.stage3 || {};
  const anyDis = !!(stage3 && stage3.any_disagree);
  const isMC = qtype === 'mc';

  const panel = el('div', { class: 'qpanel' + (anyDis ? ' disagree' : '') });
  panel.appendChild(el('h3', {}, [
    document.createTextNode(`${qtype.toUpperCase()} sample`),
    el('span', { class: 'badge ' + (anyDis ? 'hard' : 'agree'),
                 text: anyDis ? 'HARD (any_disagree)' : 'all-agree' }),
  ]));

  const row = el('div', { class: 'row' });

  // Image column
  const colImg = el('div', { class: 'col-img' });
  if (stage2 && stage2.image_path) {
    colImg.appendChild(el('img', {
      src: '/img/' + encodeURIComponent(rec.sample_id) + '?t=' + Date.now(),
      alt: rec.sample_id,
    }));
  } else {
    colImg.appendChild(el('div', { class: 'no-img',
      text: stage2 ? `(image missing — status: ${stage2.status||'?'})` : '(no image)' }));
  }
  row.appendChild(colImg);

  // Question column (editable)
  const colQ = el('div', { class: 'col-q' });
  colQ.appendChild(el('div', { class: 'qlabel', text: 'Question (editable)' }));
  const qta = el('textarea', { id: 'edit-question', class: 'qtext-edit', rows: '3' });
  qta.value = effectiveQuestion(rec);
  qta.addEventListener('input', markDirty);
  colQ.appendChild(qta);

  if (isMC) {
    colQ.appendChild(el('div', { class: 'qlabel', text: 'Options (editable; pick gold)' ,
                                   style: 'margin-top:10px;' }));
    MC_OPT_STATE = deepCopyOptions(effectiveOptions(rec));
    MC_GOLD_LABEL = effectiveCorrectLabel(rec);
    colQ.appendChild(buildOptionsEditor());
  } else {
    colQ.appendChild(el('div', { class: 'qlabel', text: 'Gold answer (editable)',
                                   style: 'margin-top:10px;' }));
    const ainp = el('input', {
      type: 'text', id: 'edit-answer',
      placeholder: 'short gold answer (1-4 words)',
    });
    ainp.value = effectiveAnswer(rec);
    ainp.addEventListener('input', markDirty);
    colQ.appendChild(el('div', { class: 'answer-edit' }, [ainp]));
  }
  row.appendChild(colQ);
  panel.appendChild(row);

  // Verifier cards
  const grid = el('div', { class: 'verifier-grid' });
  if (isMC) {
    grid.appendChild(mcVerifierCard('Qwen 3.6 Plus', stage3 && stage3.qwen, stage3 && stage3.gold_label));
    grid.appendChild(mcVerifierCard('Gemini 3.1 Pro', stage3 && stage3.gemini, stage3 && stage3.gold_label));
  } else {
    grid.appendChild(oeVerifierCard('Qwen 3.6 Plus', stage3 && stage3.qwen));
    grid.appendChild(oeVerifierCard('Gemini 3.1 Pro', stage3 && stage3.gemini));
  }
  panel.appendChild(grid);

  return panel;
}

function buildHumanWPanel(rec) {
  const wrap = el('div', { class: 'humanw-panel' });
  wrap.appendChild(el('h3', { text: '人工标注（humanw）' }));
  const row = el('div', { class: 'hw-row' });

  const mkSel = (id, current, choices) => {
    const sel = el('select', { id });
    for (const [val, label] of choices) {
      const opt = el('option', { value: val, text: label });
      // null/undefined → '' to match unset option
      const cur = current == null ? '' : String(current);
      if (cur === val) opt.selected = true;
      sel.appendChild(opt);
    }
    sel.addEventListener('change', markDirty);
    return sel;
  };

  const fields = [
    ['hw-image_broken',     '图片有无崩坏',        rec.humanw_image_broken,
        [['', '— 未标'], ['true', '有崩坏'], ['false', '无崩坏']]],
    ['hw-question_ok',      '问题是否合理',        rec.humanw_question_ok,
        [['', '— 未标'], ['true', '合理'],   ['false', '不合理']]],
    ['hw-correct_label_ok', 'correct_label正确?', rec.humanw_correct_label_ok,
        [['', '— 未标'], ['true', '正确'],   ['false', '不正确']]],
    ['hw-keep',             '是否保留',           rec.humanw_keep,
        [['', '— 未标'], ['true', '保留'],   ['false', '不保留']]],
  ];
  for (const [id, label, cur, choices] of fields) {
    row.appendChild(el('label', {}, [
      el('span', { class: 'hw-key', text: label }),
      mkSel(id, cur, choices),
    ]));
  }
  wrap.appendChild(row);
  return wrap;
}

function readHumanWFromUI() {
  const get = (id) => {
    const el = document.getElementById(id);
    if (!el) return undefined;
    const v = el.value;
    if (v === '') return null;
    if (v === 'true') return true;
    if (v === 'false') return false;
    return null;
  };
  return {
    humanw_image_broken:     get('hw-image_broken'),
    humanw_question_ok:      get('hw-question_ok'),
    humanw_correct_label_ok: get('hw-correct_label_ok'),
    humanw_keep:             get('hw-keep'),
  };
}

function buildOptionsEditor() {
  const wrap = el('div', { class: 'options-edit', id: 'options-edit' });
  renderOptionsEditor(wrap);
  return wrap;
}

function renderOptionsEditor(wrap) {
  if (!wrap) wrap = document.getElementById('options-edit');
  if (!wrap) return;
  wrap.innerHTML = '';
  // Re-label sequentially A,B,C... as state may have been mutated
  MC_OPT_STATE.forEach((opt, i) => {
    opt.label = LABELS[i];
  });
  // If gold label refers to a label that no longer exists, clear it
  if (MC_GOLD_LABEL && !MC_OPT_STATE.find(o => o.label === MC_GOLD_LABEL)) {
    MC_GOLD_LABEL = null;
  }

  for (let i = 0; i < MC_OPT_STATE.length; i++) {
    const opt = MC_OPT_STATE[i];
    const isGold = opt.label === MC_GOLD_LABEL;
    const radio = el('input', {
      type: 'radio', name: 'mc-gold', class: 'gold-radio',
      'data-label': opt.label,
      checked: isGold,
    });
    radio.addEventListener('change', () => {
      MC_GOLD_LABEL = opt.label;
      markDirty();
      renderOptionsEditor(wrap);
    });
    const inp = el('input', {
      type: 'text', class: 'opt-input',
      'data-i': String(i),
      placeholder: 'option text — use " / " between blanks for fill-in (max 3 parts)',
    });
    inp.value = (opt.parts || []).join(' / ');
    inp.addEventListener('input', () => {
      const v = inp.value;
      MC_OPT_STATE[i].parts = v.split('/').map(s => s.trim()).filter(s => s.length);
      markDirty();
    });
    const del = el('button', {
      type: 'button', class: 'opt-del', title: 'remove this option',
      onclick: () => {
        MC_OPT_STATE.splice(i, 1);
        markDirty();
        renderOptionsEditor(wrap);
      },
    }, ['−']);
    const row = el('div', { class: 'opt-row' + (isGold ? ' gold' : '') }, [
      radio,
      el('span', { class: 'opt-label', text: opt.label + '.' }),
      inp,
      del,
    ]);
    wrap.appendChild(row);
  }
  const add = el('button', {
    type: 'button', class: 'opt-add',
    onclick: () => {
      if (MC_OPT_STATE.length >= LABELS.length) return;
      MC_OPT_STATE.push({ label: LABELS[MC_OPT_STATE.length], parts: [''] });
      markDirty();
      renderOptionsEditor(wrap);
    },
  }, ['+ add option']);
  wrap.appendChild(add);

  const n = MC_OPT_STATE.length;
  const standard = (n === 4 || n === 6 || n === 8);
  const hint = el('div', { class: 'hint' });
  hint.appendChild(document.createTextNode(`${n} option(s)`));
  if (!standard) {
    hint.appendChild(document.createTextNode(' · '));
    hint.appendChild(el('span', { class: 'warn',
      text: 'non-standard count (canonical: 4/6/8)' }));
  }
  if (!MC_GOLD_LABEL) {
    hint.appendChild(document.createTextNode(' · '));
    hint.appendChild(el('span', { class: 'warn', text: 'no correct label selected' }));
  }
  wrap.appendChild(hint);
}

function mcVerifierCard(title, v, gold) {
  const card = el('div', { class: 'verifier-card' });
  if (!v) {
    card.classList.add('error');
    card.appendChild(el('h4', { text: title }));
    card.appendChild(el('div', { class: 'row2', text: '(no verdict)' }));
    return card;
  }
  if (v.error) {
    card.classList.add('error');
    card.appendChild(el('h4', { text: title }));
    card.appendChild(el('div', { class: 'row2', text: 'Error: ' + v.error }));
    return card;
  }
  const correct = v.correct === true;
  card.classList.add(correct ? 'pass' : (v.parsed_label === null ? 'error' : 'fail'));
  card.appendChild(el('h4', {}, [
    el('span', { text: title }),
    el('span', { class: 'badge ' + (correct ? 'agree' : 'hard'),
                 text: correct ? 'agree' : 'disagree' }),
  ]));
  card.appendChild(el('div', { class: 'row2', html:
    `<b>parsed:</b> <code>${escapeHtml(v.parsed_label == null ? '(unparsed)' : v.parsed_label)}</code> ` +
    `vs gold <code>${escapeHtml(gold || '?')}</code>` }));
  if (v.raw && v.raw !== v.parsed_label) {
    card.appendChild(el('div', { class: 'reasoning', text: 'raw reply: ' + v.raw }));
  }
  return card;
}

function oeVerifierCard(title, v) {
  const card = el('div', { class: 'verifier-card' });
  if (!v) {
    card.classList.add('error');
    card.appendChild(el('h4', { text: title }));
    card.appendChild(el('div', { class: 'row2', text: '(no verdict)' }));
    return card;
  }
  if (v.error) {
    card.classList.add('error');
    card.appendChild(el('h4', { text: title }));
    card.appendChild(el('div', { class: 'row2', text: 'Error: ' + v.error }));
    return card;
  }
  const jc = v.judge_correct;
  card.classList.add(jc === true ? 'pass' : (jc === null ? 'error' : 'fail'));
  card.appendChild(el('h4', {}, [
    el('span', { text: title }),
    el('span', { class: 'badge ' + (jc === true ? 'agree' : 'hard'),
                 text: jc === true ? 'agree' : (jc === null ? '?' : 'disagree') }),
  ]));
  card.appendChild(el('div', { class: 'row2', html:
    `<b>vl_answer:</b> <code>${escapeHtml(v.vl_answer == null ? '' : v.vl_answer)}</code>` }));
  card.appendChild(el('div', { class: 'row2', html:
    `<b>judge:</b> <span class="judgement-${jc===true?'true':(jc===false?'false':'null')}">${jc}</span>` }));
  if (v.judge_reasoning) {
    card.appendChild(el('div', { class: 'reasoning', text: v.judge_reasoning }));
  }
  if (v.judge_error) {
    card.appendChild(el('div', { class: 'reasoning', text: 'judge error: ' + v.judge_error }));
  }
  return card;
}

function detailsBlock(title, body) {
  return el('details', {}, [
    el('summary', { text: title }),
    el('div', { class: 'body', text: body }),
  ]);
}

function detailsBlockJSON(title, obj) {
  const d = el('details', {}, [el('summary', { text: title })]);
  const pre = el('pre', { text: obj == null ? '(none)' : JSON.stringify(obj, null, 2) });
  d.appendChild(pre);
  return d;
}

async function doSave() {
  if (!CURRENT_SID || !CURRENT_REC) return;
  const qtype = CURRENT_REC.qtype;
  const body = {
    valid: document.getElementById('flag-valid').checked,
    human_verified: document.getElementById('flag-verified').checked,
    notes: document.getElementById('notes-text').value,
    ...readHumanWFromUI(),
  };

  // Question — always saved as override (null if blank or matches stage1)
  const newQ = document.getElementById('edit-question').value.trim();
  const baseQ = (CURRENT_REC.stage1 || {}).question || '';
  body.question_corrected = (newQ && newQ !== baseQ) ? newQ : null;

  if (qtype === 'mc') {
    // Sanitize options state
    const opts = MC_OPT_STATE.map((o, i) => ({
      label: LABELS[i],
      parts: (o.parts || []).map(p => String(p).trim()).filter(s => s.length),
    })).filter(o => o.parts.length);
    if (!opts.length) {
      updateSaveStatus('error: at least one option required', 'error');
      return;
    }
    if (!MC_GOLD_LABEL) {
      updateSaveStatus('error: pick a correct label', 'error');
      return;
    }
    if (!opts.find(o => o.label === MC_GOLD_LABEL)) {
      updateSaveStatus('error: correct label not in options', 'error');
      return;
    }
    // Compare with stage1 to decide whether to record an override
    const baseOpts = (CURRENT_REC.stage1 || {}).options || [];
    const baseLabel = (CURRENT_REC.stage1 || {}).correct_label || null;
    const sameOpts = JSON.stringify(opts) === JSON.stringify(baseOpts);
    body.options_corrected = sameOpts ? null : opts;
    body.correct_label_corrected = (MC_GOLD_LABEL === baseLabel) ? null : MC_GOLD_LABEL;
  } else {
    const newA = document.getElementById('edit-answer').value.trim();
    const baseA = (CURRENT_REC.stage1 || {}).answer || '';
    body.answer_corrected = (newA && newA !== baseA) ? newA : null;
  }

  updateSaveStatus('saving…');
  const r = await fetch('/api/edit/' + encodeURIComponent(CURRENT_SID), {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok || data.error) {
    updateSaveStatus('error: ' + (data.error || r.status), 'error');
    return;
  }
  if (data.no_change) {
    updateSaveStatus('no changes', 'ok');
    DIRTY = false;
    return;
  }
  updateSaveStatus('saved ✓', 'ok');
  DIRTY = false;
  CURRENT_REC = data.record;
  await loadList();
  await selectSample(CURRENT_SID);
}

async function reloadAll() {
  if (DIRTY && !confirm('You have unsaved edits. Reload anyway?')) return;
  await fetch('/api/reload', { method: 'POST' });
  DIRTY = false;
  await loadList();
  if (CURRENT_SID) await selectSample(CURRENT_SID);
}

document.getElementById('btn-reload').addEventListener('click', reloadAll);
document.getElementById('filter-family').addEventListener('change', loadList);
document.getElementById('filter-qtype').addEventListener('change', loadList);
document.getElementById('filter-disagree').addEventListener('change', loadList);
document.getElementById('filter-verified').addEventListener('change', loadList);
document.getElementById('filter-valid').addEventListener('change', loadList);
for (const fld of ['humanw_image_broken', 'humanw_question_ok',
                   'humanw_correct_label_ok', 'humanw_keep']) {
  document.getElementById('filter-' + fld).addEventListener('change', loadList);
}

const famSel = document.getElementById('filter-family');
famSel.size = Math.min(8, Math.max(3, famSel.options.length));
famSel.style.minWidth = '90px';

loadList();

window.addEventListener('beforeunload', (e) => {
  if (DIRTY) { e.preventDefault(); e.returnValue = ''; }
});
</script>

</body>
</html>
"""


# ── Entrypoint ────────────────────────────────────────────────────────────

def main() -> None:
    info = _refresh()
    print(f"[viewer] manual_review.jsonl: {MANUAL_PATH} "
          f"({'exists' if MANUAL_PATH.exists() else 'will be created'})", flush=True)
    print(f"[viewer] stage1: {STAGE1_PATH} (read-only)", flush=True)
    print(f"[viewer] stage2: {STAGE2_PATH} (read-only)", flush=True)
    print(f"[viewer] stage3: {STAGE3_PATH} (read-only)", flush=True)
    print(f"[viewer] images: {IMAGES_DIR}", flush=True)
    print(f"[viewer] loaded {info['total']} samples "
          f"(from manual={info['from_manual']}, bootstrapped={info['bootstrapped']})",
          flush=True)
    print(f"[viewer] http://{HOST}:{PORT}/", flush=True)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
