#!/usr/bin/env python3
"""
viewer_release_all.py — Read-only Flask viewer for release_all/dataset.json.

A simpler sibling of viewer.py: no human edits, no manual_review. Just browse
the 632 samples (UB-PHYS/BIO/CHEM/MATH + ZG-*) with image, question, options
(MC) or gold answer (OE), and — if eval_results_release_all/<model>.jsonl
exists — each model's verdict for the current sample.

ENV overrides (optional):
  DATASET_JSON   default: release_all/dataset.json
  IMAGES_DIR     default: release_all/images   (also any path used inside image_path is honoured)
  EVAL_DIR       default: eval_results_release_all
  PORT           default: 7988
  HOST           default: 0.0.0.0
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

from flask import Flask, abort, jsonify, render_template_string, request, send_file

app = Flask(__name__)

# ── Config ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
DATASET_PATH = Path(os.environ.get("DATASET_JSON", str(HERE / "release_all" / "dataset.json")))
IMAGES_DIR = Path(os.environ.get("IMAGES_DIR", str(HERE / "release_all" / "images")))
EVAL_DIR = Path(os.environ.get("EVAL_DIR", str(HERE / "eval_results_release_all")))

PORT = int(os.environ.get("PORT", "7988"))
HOST = os.environ.get("HOST", "0.0.0.0")

# ── In-memory state ───────────────────────────────────────────────────────
_DATA: list[dict] = []                  # ordered list of entries
_BY_SID: dict[str, dict] = {}           # sample_id → entry
_EVAL: dict[str, dict[str, dict]] = {}  # model → sample_id → record


def _load_dataset() -> None:
    global _DATA, _BY_SID
    if not DATASET_PATH.exists():
        raise SystemExit(f"dataset not found: {DATASET_PATH}")
    with open(DATASET_PATH, encoding="utf-8") as f:
        _DATA = json.load(f)
    _BY_SID = {e["sample_id"]: e for e in _DATA}
    print(f"[viewer] loaded {len(_DATA)} samples from {DATASET_PATH}", flush=True)


def _load_evals() -> None:
    """Best-effort load of per-model eval JSONLs. Latest line per sample_id wins."""
    global _EVAL
    _EVAL = {}
    if not EVAL_DIR.exists():
        return
    for p in sorted(EVAL_DIR.glob("*.jsonl")):
        model = p.stem
        by_sid: dict[str, dict] = {}
        with open(p, encoding="utf-8") as f:
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
        _EVAL[model] = by_sid
        print(f"[viewer] eval/{model}: {len(by_sid)} records", flush=True)


def _refresh() -> dict:
    _load_dataset()
    _load_evals()
    return {"total": len(_DATA), "models": list(_EVAL.keys())}


# ── Helpers ───────────────────────────────────────────────────────────────


def _id_prefix(sid: str) -> str:
    """Return UB / ZG / ADD / OTHER — used for sidebar grouping."""
    if "-" not in sid:
        return "OTHER"
    head = sid.split("-", 1)[0]
    return head if head in ("UB", "ZG", "ADD") else "OTHER"


def _verdict_for(model: str, sid: str, qtype: str) -> str | None:
    """Compact correctness label per model for sidebar marks: 'ok' / 'bad' / None."""
    rec = _EVAL.get(model, {}).get(sid)
    if not rec or rec.get("status") != "ok":
        return None
    if qtype == "mc":
        c = rec.get("correct")
    else:
        c = rec.get("judge_correct")
    if c is True:
        return "ok"
    if c is False:
        return "bad"
    return None


def _summary_for(entry: dict) -> dict:
    sid = entry["sample_id"]
    qtype = entry.get("qtype")
    verdicts = {m: _verdict_for(m, sid, qtype) for m in _EVAL.keys()}
    any_bad = any(v == "bad" for v in verdicts.values())
    all_ok = bool(verdicts) and all(v == "ok" for v in verdicts.values())
    return {
        "sample_id": sid,
        "id_prefix": _id_prefix(sid),
        "family": entry.get("family"),
        "qtype": qtype,
        "domain": entry.get("domain"),
        "question_head": (entry.get("question") or "")[:90],
        "verdicts": verdicts,
        "any_bad": any_bad,
        "all_ok": all_ok,
    }


# ── Routes ────────────────────────────────────────────────────────────────


@app.route("/")
def index() -> str:
    families = sorted({e.get("family", "?") for e in _DATA})
    return render_template_string(_INDEX_HTML,
                                  families=families,
                                  models=sorted(_EVAL.keys()))


@app.route("/api/samples")
def api_samples():
    """
    Filters via query string (CSV):
      family=PHYS,MATH
      qtype=mc|oe
      prefix=UB|ZG|ADD
      verdict=any_bad|all_ok|none      (only meaningful if eval data loaded)
      q=<substring>                     (case-insensitive substring in question)
    """
    items = [_summary_for(e) for e in _DATA]

    fam = request.args.get("family", "").strip()
    if fam:
        wanted = {s.strip() for s in fam.split(",") if s.strip()}
        items = [x for x in items if x["family"] in wanted]

    qt = request.args.get("qtype", "").strip()
    if qt:
        wanted_q = {s.strip() for s in qt.split(",") if s.strip()}
        items = [x for x in items if x["qtype"] in wanted_q]

    pref = request.args.get("prefix", "").strip()
    if pref:
        wanted_p = {s.strip() for s in pref.split(",") if s.strip()}
        items = [x for x in items if x["id_prefix"] in wanted_p]

    ver = request.args.get("verdict", "").strip().lower()
    if ver == "any_bad":
        items = [x for x in items if x["any_bad"]]
    elif ver == "all_ok":
        items = [x for x in items if x["all_ok"]]
    elif ver == "none":
        # at least one model with no verdict (unevaluated)
        items = [x for x in items
                 if any(v is None for v in x["verdicts"].values())
                 or not x["verdicts"]]

    q = request.args.get("q", "").strip().lower()
    if q:
        items = [x for x in items if q in (x["question_head"] or "").lower()
                 or q in (x["sample_id"] or "").lower()]

    counts: dict[str, int] = defaultdict(int)
    for x in items:
        counts["qtype_" + (x["qtype"] or "?")] += 1
        counts["prefix_" + (x["id_prefix"] or "?")] += 1
        if x["any_bad"]:
            counts["any_bad"] += 1
        if x["all_ok"]:
            counts["all_ok"] += 1
    counts["total"] = len(items)

    return jsonify({"items": items, "counts": dict(counts)})


@app.route("/api/sample/<path:sample_id>")
def api_sample(sample_id: str):
    rec = _BY_SID.get(sample_id)
    if not rec:
        return jsonify({"error": f"sample_id {sample_id} not found"}), 404
    eval_for_sid = {m: _EVAL[m].get(sample_id) for m in _EVAL}
    return jsonify({"sample": rec, "eval": eval_for_sid})


@app.route("/img/<path:sample_id>")
def img(sample_id: str):
    rec = _BY_SID.get(sample_id)
    if not rec:
        abort(404)
    rel = rec.get("image_path")
    if not rel:
        abort(404)
    # image_path is stored as "release_all/images/X.png" — relative to project root
    p = (HERE / rel).resolve()
    if not p.exists():
        # fallback: look in IMAGES_DIR by basename
        alt = (IMAGES_DIR / Path(rel).name).resolve()
        if not alt.exists():
            abort(404)
        p = alt
    return send_file(str(p), mimetype="image/png")


@app.route("/api/reload", methods=["POST"])
def api_reload():
    info = _refresh()
    return jsonify({"ok": True, **info})


# ── HTML ──────────────────────────────────────────────────────────────────

_INDEX_HTML = r"""
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>release_all viewer</title>
  <style>
    * { box-sizing: border-box; }
    html, body { height: 100%; margin: 0; font-family: -apple-system, "Segoe UI", Roboto, sans-serif; }
    body { display: flex; flex-direction: column; }

    header {
      flex: 0 0 auto; background: #1f2937; color: #fff;
      padding: 8px 14px; display: flex; align-items: center; gap: 12px;
      border-bottom: 1px solid #444; flex-wrap: wrap;
    }
    header h1 { font-size: 14px; margin: 0; font-weight: 600; }
    header .stats { font-size: 12px; opacity: 0.9; }
    header label { font-size: 12px; display: flex; align-items: center; gap: 4px; }
    header select, header input[type=search] {
      font-size: 12px; padding: 3px 6px; border-radius: 3px; border: 1px solid #555;
      background: #2c303a; color: #eee;
    }
    header input[type=search] { width: 180px; }
    header button {
      font-size: 12px; padding: 4px 10px; border-radius: 3px; border: 1px solid #555;
      background: #3a3f4b; color: #eee; cursor: pointer;
    }
    header button:hover { background: #4a505d; }

    .main { display: flex; flex: 1 1 auto; overflow: hidden; }

    .sidebar {
      flex: 0 0 340px; border-right: 1px solid #ddd; overflow-y: auto;
      background: #fafafa;
    }
    .sidebar .group-header {
      position: sticky; top: 0; background: #eef2f7; padding: 5px 10px;
      font-size: 12px; font-weight: 700; color: #1f2937;
      border-top: 1px solid #ccc; border-bottom: 1px solid #ccc; z-index: 1;
    }
    .sidebar .item {
      padding: 6px 10px; font-size: 12px; cursor: pointer;
      border-bottom: 1px solid #eee; display: flex; align-items: center; gap: 6px;
    }
    .sidebar .item:hover { background: #e8eef7; }
    .sidebar .item.active { background: #d6e4f6; }
    .sidebar .item.any-bad { background: #fff3cd; }
    .sidebar .item.any-bad.active { background: #ffe580; }
    .sidebar .item.all-ok { background: #e8f6ec; }
    .sidebar .item.all-ok.active { background: #b6e2c2; }
    .sidebar .item .sid {
      font-family: ui-monospace, monospace; font-weight: 700; color: #333;
      flex: 0 0 auto;
    }
    .sidebar .item .sub {
      color: #555; font-size: 11px; flex: 1; min-width: 0;
      overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .sidebar .item .marks { display: flex; gap: 3px; flex: 0 0 auto; }

    .badge {
      font-size: 10px; padding: 1px 6px; border-radius: 8px; color: #fff;
      font-weight: 700; text-transform: uppercase;
    }
    .badge.mc      { background: #1a73e8; }
    .badge.oe      { background: #c43d8a; }
    .badge.ub      { background: #1e8e3e; }
    .badge.zg      { background: #7b3eb1; }
    .badge.add     { background: #d18000; }
    .badge.other   { background: #888; }

    .vmark {
      width: 12px; height: 12px; border-radius: 50%; display: inline-block;
      border: 1px solid rgba(0,0,0,0.15);
    }
    .vmark.ok   { background: #1e8e3e; }
    .vmark.bad  { background: #d93025; }
    .vmark.none { background: #ddd; }

    .content {
      flex: 1 1 auto; overflow-y: auto; padding: 16px 24px; background: #fff;
    }
    .placeholder { color: #888; padding: 40px; text-align: center; }

    .sample-head {
      display: flex; align-items: baseline; gap: 12px; margin-bottom: 6px;
      flex-wrap: wrap;
    }
    .sample-head h2 { font-size: 18px; margin: 0; font-family: ui-monospace, monospace; }
    .sample-head .meta { color: #666; font-size: 13px; }

    .qpanel {
      border: 1px solid #ddd; border-radius: 6px; padding: 14px 16px;
      background: #fcfcfc; margin-top: 12px;
    }
    .qpanel .row {
      display: flex; gap: 16px; align-items: flex-start;
    }
    .qpanel .col-img { flex: 0 0 420px; max-width: 55%; }
    .qpanel .col-img img {
      width: 100%; max-height: 560px; object-fit: contain; border: 1px solid #ddd;
      border-radius: 3px; background: #f0f0f0; display: block;
    }
    .qpanel .col-img .no-img {
      width: 100%; aspect-ratio: 1; display: flex; align-items: center;
      justify-content: center; background: #f3f3f3; color: #888;
      font-size: 12px; border: 1px dashed #bbb; border-radius: 3px;
    }
    .qpanel .col-q { flex: 1 1 0; min-width: 0; }
    .qlabel {
      font-size: 11px; font-weight: 700; text-transform: uppercase;
      color: #555; margin-bottom: 4px;
    }
    .qtext {
      font-size: 14px; line-height: 1.5; padding: 8px 10px; background: #fff;
      border: 1px solid #ddd; border-radius: 3px; white-space: pre-wrap;
    }

    .opts { margin-top: 12px; }
    .opts .opt-row {
      display: flex; align-items: flex-start; gap: 8px; padding: 5px 8px;
      border-radius: 3px;
    }
    .opts .opt-row.gold { background: #e6f4ea; border-left: 3px solid #1e8e3e; }
    .opts .opt-label {
      width: 22px; text-align: center; font-family: ui-monospace, monospace;
      font-weight: 700; color: #444; flex: 0 0 auto;
    }
    .opts .opt-text { flex: 1; font-size: 13px; line-height: 1.5; }
    .opts .gold-mark { color: #1e8e3e; font-weight: 700; font-size: 11px;
                        align-self: center; }

    .answer-box {
      margin-top: 12px; padding: 8px 10px; background: #e6f4ea;
      border: 1px solid #cde9d3; border-left: 4px solid #1e8e3e;
      border-radius: 3px; font-size: 14px;
    }
    .answer-box .label {
      font-size: 11px; font-weight: 700; text-transform: uppercase;
      color: #1e8e3e; margin-bottom: 3px;
    }

    .model-grid {
      display: grid; gap: 10px; margin-top: 16px;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
    }
    .model-card {
      border: 1px solid #ddd; border-radius: 4px; padding: 8px 10px;
      background: #fcfcfc; font-size: 12px;
    }
    .model-card.pass { border-left: 4px solid #1e8e3e; }
    .model-card.fail { border-left: 4px solid #d93025; }
    .model-card.error { border-left: 4px solid #ff8c00; }
    .model-card.none { border-left: 4px solid #ccc; }
    .model-card h4 {
      font-size: 12px; margin: 0 0 4px 0; display: flex; gap: 6px;
      align-items: center; justify-content: space-between;
    }
    .model-card .row2 { font-size: 12px; color: #444; margin: 2px 0; }
    .model-card .row2 b { display: inline-block; min-width: 70px; color: #222; }
    .model-card .reasoning {
      margin-top: 4px; font-size: 11px; color: #333; white-space: pre-wrap;
      max-height: 130px; overflow-y: auto; padding: 4px 6px; background: #f7f7f7;
      border-radius: 3px;
    }

    .small { font-size: 11px; color: #888; margin-top: 10px; }
    code { font-family: ui-monospace, monospace; }

    details { margin-top: 14px; border-top: 1px solid #eee; padding-top: 8px; }
    details > summary { cursor: pointer; font-size: 13px; font-weight: 600; color: #333; }
    details pre {
      background: #f5f5f5; padding: 8px; border-radius: 3px;
      font-size: 12px; max-height: 320px; overflow: auto; margin: 0;
    }
  </style>
</head>
<body>

<header>
  <h1>release_all viewer</h1>
  <span class="stats" id="stats">loading…</span>

  <label>family
    <select id="filter-family" multiple>
      {% for f in families %}<option value="{{ f }}">{{ f }}</option>{% endfor %}
    </select>
  </label>
  <label>qtype
    <select id="filter-qtype">
      <option value="">all</option>
      <option value="mc">mc</option>
      <option value="oe">oe</option>
    </select>
  </label>
  <label>prefix
    <select id="filter-prefix">
      <option value="">all</option>
      <option value="UB">UB</option>
      <option value="ZG">ZG</option>
      <option value="ADD">ADD</option>
    </select>
  </label>
  {% if models %}
  <label>verdict
    <select id="filter-verdict">
      <option value="">all</option>
      <option value="any_bad">any model wrong</option>
      <option value="all_ok">all models correct</option>
      <option value="none">missing eval</option>
    </select>
  </label>
  {% endif %}
  <label>search
    <input id="filter-q" type="search" placeholder="question / id substring…" />
  </label>

  <button id="btn-reload" title="re-read dataset + eval files">reload</button>
</header>

<div class="main">
  <aside class="sidebar" id="sidebar">loading…</aside>
  <section class="content" id="content">
    <div class="placeholder">Select a sample from the left.</div>
  </section>
</div>

<script>
const MODELS = {{ models | tojson }};
const LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
let SAMPLE_LIST = [];
let CURRENT_SID = null;

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

function debounce(fn, ms) {
  let t = null;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

async function loadList() {
  const params = new URLSearchParams();
  const fams = Array.from(document.getElementById('filter-family').selectedOptions)
                    .map(o => o.value).filter(v => v);
  if (fams.length) params.set('family', fams.join(','));
  const qt = document.getElementById('filter-qtype').value;
  if (qt) params.set('qtype', qt);
  const pref = document.getElementById('filter-prefix').value;
  if (pref) params.set('prefix', pref);
  const ver = document.getElementById('filter-verdict');
  if (ver && ver.value) params.set('verdict', ver.value);
  const q = document.getElementById('filter-q').value.trim();
  if (q) params.set('q', q);

  const r = await fetch('/api/samples?' + params.toString());
  const data = await r.json();
  SAMPLE_LIST = data.items;
  const c = data.counts;
  const statBits = [
    `total=${c.total||0}`,
    `mc=${c.qtype_mc||0} oe=${c.qtype_oe||0}`,
    `UB=${c.prefix_UB||0} ZG=${c.prefix_ZG||0} ADD=${c.prefix_ADD||0}`,
  ];
  if (MODELS.length) {
    statBits.push(`bad=${c.any_bad||0} all-ok=${c.all_ok||0}`);
  }
  document.getElementById('stats').textContent = statBits.join(' | ');

  const sb = document.getElementById('sidebar');
  sb.innerHTML = '';
  let lastGroup = null;
  for (const item of SAMPLE_LIST) {
    const groupKey = `${item.id_prefix} · ${item.family}`;
    if (groupKey !== lastGroup) {
      lastGroup = groupKey;
      sb.appendChild(el('div', { class: 'group-header', text: groupKey }));
    }
    const marks = el('span', { class: 'marks' }, [
      el('span', { class: 'badge ' + (item.qtype || 'other'), text: item.qtype || '?' }),
      ...MODELS.map(m => {
        const v = item.verdicts[m];
        const cls = v === 'ok' ? 'ok' : (v === 'bad' ? 'bad' : 'none');
        return el('span', { class: 'vmark ' + cls, title: m + ': ' + (v || 'no verdict') });
      }),
    ]);
    const cls = 'item' +
      (item.sample_id === CURRENT_SID ? ' active' : '') +
      (item.any_bad ? ' any-bad' : '') +
      (item.all_ok ? ' all-ok' : '');
    const div = el('div', {
      class: cls,
      'data-sid': item.sample_id,
      onclick: () => selectSample(item.sample_id),
      title: item.question_head || '',
    }, [
      el('span', { class: 'sid', text: item.sample_id }),
      el('span', { class: 'sub', text: item.domain || '' }),
      marks,
    ]);
    sb.appendChild(div);
  }
  if (!SAMPLE_LIST.length) {
    sb.appendChild(el('div', { class: 'placeholder', text: '(no samples match filters)' }));
  }
}

async function selectSample(sid) {
  CURRENT_SID = sid;
  document.querySelectorAll('.sidebar .item').forEach(e => {
    e.classList.toggle('active', e.dataset.sid === sid);
  });
  const r = await fetch('/api/sample/' + encodeURIComponent(sid));
  if (!r.ok) {
    document.getElementById('content').innerHTML =
      '<div class="placeholder">Failed to load: ' + escapeHtml(sid) + '</div>';
    return;
  }
  const data = await r.json();
  renderSample(data.sample, data.eval || {});
}

function renderSample(rec, evalMap) {
  const c = document.getElementById('content');
  c.innerHTML = '';

  const head = el('div', { class: 'sample-head' }, [
    el('h2', { text: rec.sample_id }),
    el('span', { class: 'meta', text:
      `${rec.family} · ${rec.qtype} · ${rec.domain || ''}` }),
  ]);
  c.appendChild(head);

  const panel = el('div', { class: 'qpanel' });
  const row = el('div', { class: 'row' });

  // Image column
  const colImg = el('div', { class: 'col-img' });
  if (rec.image_path) {
    colImg.appendChild(el('img', {
      src: '/img/' + encodeURIComponent(rec.sample_id) + '?t=' + Date.now(),
      alt: rec.sample_id,
    }));
  } else {
    colImg.appendChild(el('div', { class: 'no-img', text: '(no image)' }));
  }
  row.appendChild(colImg);

  // Question + options/answer
  const colQ = el('div', { class: 'col-q' });
  colQ.appendChild(el('div', { class: 'qlabel', text: 'Question' }));
  colQ.appendChild(el('div', { class: 'qtext', text: rec.question || '' }));

  if (rec.qtype === 'mc') {
    const gold = rec.correct_label;
    colQ.appendChild(el('div', { class: 'qlabel', text: `Options (gold=${gold || '?'})`,
                                   style: 'margin-top:12px;' }));
    const optsBox = el('div', { class: 'opts' });
    for (const opt of (rec.options || [])) {
      const isGold = opt.label === gold;
      const text = (opt.parts || []).join(' / ');
      optsBox.appendChild(el('div', { class: 'opt-row' + (isGold ? ' gold' : '') }, [
        el('span', { class: 'opt-label', text: opt.label + '.' }),
        el('span', { class: 'opt-text', text: text }),
        isGold ? el('span', { class: 'gold-mark', text: '✓ GOLD' }) : null,
      ]));
    }
    colQ.appendChild(optsBox);
  } else {
    colQ.appendChild(el('div', { class: 'answer-box' }, [
      el('div', { class: 'label', text: 'Gold answer' }),
      document.createTextNode(rec.answer || '(empty)'),
    ]));
  }
  row.appendChild(colQ);
  panel.appendChild(row);
  c.appendChild(panel);

  // Model eval cards (if any)
  if (MODELS.length) {
    const grid = el('div', { class: 'model-grid' });
    for (const m of MODELS) {
      grid.appendChild(modelCard(m, evalMap[m], rec.qtype));
    }
    c.appendChild(grid);
  }

  c.appendChild(el('div', { class: 'small',
    text: `image_path = ${rec.image_path}` }));

  c.appendChild(detailsBlockJSON('Raw dataset entry', rec));
}

function modelCard(model, rec, qtype) {
  const card = el('div', { class: 'model-card' });
  if (!rec) {
    card.classList.add('none');
    card.appendChild(el('h4', { text: model }));
    card.appendChild(el('div', { class: 'row2', text: '(no verdict)' }));
    return card;
  }
  if (rec.status !== 'ok') {
    card.classList.add('error');
    card.appendChild(el('h4', { text: model }));
    card.appendChild(el('div', { class: 'row2', text: 'status: ' + (rec.status || '?') }));
    if (rec.error) card.appendChild(el('div', { class: 'reasoning', text: rec.error }));
    return card;
  }
  if (qtype === 'mc') {
    const ok = rec.correct === true;
    card.classList.add(ok ? 'pass' : 'fail');
    card.appendChild(el('h4', {}, [
      el('span', { text: model }),
      el('span', { class: 'badge ' + (ok ? 'oe' : 'add'),
                   text: ok ? 'CORRECT' : 'WRONG' }),
    ]));
    card.appendChild(el('div', { class: 'row2', html:
      `<b>parsed:</b> <code>${escapeHtml(rec.parsed_label ?? '(unparsed)')}</code> ` +
      `vs gold <code>${escapeHtml(rec.gold_label || '?')}</code>` }));
    if (rec.raw_answer && rec.raw_answer !== rec.parsed_label) {
      card.appendChild(el('div', { class: 'reasoning', text: 'raw: ' + rec.raw_answer }));
    }
  } else {
    const ok = rec.judge_correct === true;
    const unknown = rec.judge_correct == null;
    card.classList.add(unknown ? 'error' : (ok ? 'pass' : 'fail'));
    card.appendChild(el('h4', {}, [
      el('span', { text: model }),
      el('span', { class: 'badge ' + (ok ? 'oe' : 'add'),
                   text: ok ? 'CORRECT' : (unknown ? '?' : 'WRONG') }),
    ]));
    card.appendChild(el('div', { class: 'row2', html:
      `<b>vl_answer:</b> <code>${escapeHtml(rec.vl_answer || '')}</code>` }));
    card.appendChild(el('div', { class: 'row2', html:
      `<b>gold:</b> <code>${escapeHtml(rec.gold || '')}</code>` }));
    if (rec.judge_reasoning) {
      card.appendChild(el('div', { class: 'reasoning', text: rec.judge_reasoning }));
    }
  }
  return card;
}

function detailsBlockJSON(title, obj) {
  const d = el('details', {}, [el('summary', { text: title })]);
  d.appendChild(el('pre', { text: obj == null ? '(none)' : JSON.stringify(obj, null, 2) }));
  return d;
}

async function reloadAll() {
  await fetch('/api/reload', { method: 'POST' });
  await loadList();
  if (CURRENT_SID) await selectSample(CURRENT_SID);
}

document.getElementById('btn-reload').addEventListener('click', reloadAll);
for (const id of ['filter-family', 'filter-qtype', 'filter-prefix']) {
  document.getElementById(id).addEventListener('change', loadList);
}
const ver = document.getElementById('filter-verdict');
if (ver) ver.addEventListener('change', loadList);
document.getElementById('filter-q').addEventListener('input', debounce(loadList, 250));

const famSel = document.getElementById('filter-family');
famSel.size = Math.min(6, Math.max(3, famSel.options.length));
famSel.style.minWidth = '90px';

loadList();
</script>

</body>
</html>
"""


def main() -> None:
    info = _refresh()
    print(f"[viewer] dataset: {DATASET_PATH}", flush=True)
    print(f"[viewer] images:  {IMAGES_DIR}", flush=True)
    print(f"[viewer] eval:    {EVAL_DIR} "
          f"({'exists' if EVAL_DIR.exists() else 'not found — eval cards hidden'})",
          flush=True)
    print(f"[viewer] http://{HOST}:{PORT}/", flush=True)
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
