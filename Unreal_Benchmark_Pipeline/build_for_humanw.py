#!/usr/bin/env python3
"""Extract unverified + hard samples for human review.

- "Unverified": manual_review.jsonl record (latest by sample_id) has human_verified=False AND valid=True.
- "Hard": at least one of {qwen, gemini} got the answer wrong on stage3.

For each selected sample, copy the image into for_humanw/images/ and append a JSONL
record with question / options / answer / image (relative path) / two model raw responses
+ correctness flags. Each text field is also translated to Chinese via Qwen 3.6 Plus
(English original kept side-by-side under *_zh keys).

Resume-aware: re-running skips sample_ids already present in for_humanw/for_review.jsonl.
"""
from __future__ import annotations

import json
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent
MANUAL_REVIEW = ROOT / "manual_review.jsonl"
IMAGES_DIR = ROOT / "images"
OUT_DIR = ROOT / "for_humanw"
OUT_IMAGES = OUT_DIR / "images"
OUT_JSONL = OUT_DIR / "for_review.jsonl"

QWEN_BASE = "<QWEN_ENDPOINT>"
QWEN_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_MODEL = "qwen3.6-plus"

CONCURRENCY = 64
TRANSLATE_TIMEOUT = 180.0
TRANSLATE_RETRIES = 5


def load_latest_reviews() -> dict:
    latest: dict[str, dict] = {}
    with MANUAL_REVIEW.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            sid = r.get("sample_id")
            if sid:
                latest[sid] = r
    return latest


def is_hard_unverified(r: dict) -> bool:
    if r.get("human_verified"):
        return False
    if r.get("valid") is not True:
        return False
    s3 = r.get("stage3")
    if not isinstance(s3, dict):
        return False
    qtype = r.get("qtype")
    if qtype == "mc":
        q_ok = (s3.get("qwen") or {}).get("correct")
        g_ok = (s3.get("gemini") or {}).get("correct")
    else:
        q_ok = (s3.get("qwen") or {}).get("judge_correct")
        g_ok = (s3.get("gemini") or {}).get("judge_correct")
    return q_ok is False or g_ok is False


def extract_payload(r: dict) -> dict:
    """Build the English payload (pre-translation) for one sample."""
    sid = r["sample_id"]
    qtype = r["qtype"]
    s1 = r.get("stage1") or {}
    s2 = r.get("stage2") or {}
    s3 = r.get("stage3") or {}

    src_image = s2.get("image_path")
    image_filename = Path(src_image).name if src_image else f"{sid}.png"

    base = {
        "sample_id": sid,
        "family_id": r.get("family_id"),
        "qtype": qtype,
        "image_path": f"images/{image_filename}",
        "_src_image": src_image,
        "question": s1.get("question") or "",
    }

    if qtype == "mc":
        opts = s1.get("options") or []
        # render options as readable strings: "A: part1 / part2"
        options = []
        for o in opts:
            label = o.get("label", "")
            parts = o.get("parts") or []
            text = " / ".join(str(p) for p in parts) if parts else ""
            options.append({"label": label, "text": text})
        base["options"] = options
        base["correct_label"] = s3.get("gold_label") or s1.get("correct_label")
        qwen = s3.get("qwen") or {}
        gemini = s3.get("gemini") or {}
        base["qwen"] = {
            "answer": qwen.get("raw") or "",
            "parsed_label": qwen.get("parsed_label"),
            "correct": qwen.get("correct"),
        }
        base["gemini"] = {
            "answer": gemini.get("raw") or "",
            "parsed_label": gemini.get("parsed_label"),
            "correct": gemini.get("correct"),
        }
    else:  # oe
        base["answer"] = s1.get("answer") or s3.get("gold") or ""
        qwen = s3.get("qwen") or {}
        gemini = s3.get("gemini") or {}
        base["qwen"] = {
            "answer": qwen.get("vl_answer") or "",
            "judge_correct": qwen.get("judge_correct"),
            "judge_reasoning": qwen.get("judge_reasoning") or "",
        }
        base["gemini"] = {
            "answer": gemini.get("vl_answer") or "",
            "judge_correct": gemini.get("judge_correct"),
            "judge_reasoning": gemini.get("judge_reasoning") or "",
        }
    return base


TRANSLATE_SYSTEM = (
    "You are a precise English-to-Chinese translator for a visual-reasoning benchmark. "
    "Translate each input string to natural, faithful simplified Chinese. "
    "Preserve numbers, option labels (A/B/C...), and proper nouns verbatim. "
    "Return ONLY a JSON object that maps each input key to its Chinese translation. "
    "No prose, no markdown fences."
)


def build_translate_payload(payload: dict) -> dict[str, str]:
    """Pick fields that need translation, return key->english_text mapping."""
    fields: dict[str, str] = {"question": payload["question"]}
    if payload["qtype"] == "mc":
        for i, opt in enumerate(payload["options"]):
            if opt["text"]:
                fields[f"option_{opt['label']}"] = opt["text"]
        fields["qwen_answer"] = payload["qwen"]["answer"]
        fields["gemini_answer"] = payload["gemini"]["answer"]
    else:
        fields["answer"] = payload["answer"]
        fields["qwen_answer"] = payload["qwen"]["answer"]
        fields["qwen_judge_reasoning"] = payload["qwen"]["judge_reasoning"]
        fields["gemini_answer"] = payload["gemini"]["answer"]
        fields["gemini_judge_reasoning"] = payload["gemini"]["judge_reasoning"]
    # drop empties
    return {k: v for k, v in fields.items() if v and str(v).strip()}


def call_qwen_translate(client: httpx.Client, fields: dict[str, str]) -> dict[str, str]:
    if not fields:
        return {}
    user_msg = (
        "Translate each value to simplified Chinese. Return a JSON object with the same keys.\n\n"
        + json.dumps(fields, ensure_ascii=False, indent=2)
    )
    body = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": TRANSLATE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {QWEN_KEY}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(TRANSLATE_RETRIES):
        try:
            resp = client.post(
                f"{QWEN_BASE}/chat/completions",
                headers=headers,
                json=body,
                timeout=TRANSLATE_TIMEOUT,
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            data = json.loads(content)
            return {k: str(v) for k, v in data.items()}
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"qwen translate failed: {last_err}")


def attach_translations(payload: dict, zh: dict[str, str]) -> dict:
    """Insert _zh fields next to their English counterparts."""
    payload["question_zh"] = zh.get("question", "")
    if payload["qtype"] == "mc":
        for opt in payload["options"]:
            opt["text_zh"] = zh.get(f"option_{opt['label']}", "")
        payload["qwen"]["answer_zh"] = zh.get("qwen_answer", "")
        payload["gemini"]["answer_zh"] = zh.get("gemini_answer", "")
    else:
        payload["answer_zh"] = zh.get("answer", "")
        payload["qwen"]["answer_zh"] = zh.get("qwen_answer", "")
        payload["qwen"]["judge_reasoning_zh"] = zh.get("qwen_judge_reasoning", "")
        payload["gemini"]["answer_zh"] = zh.get("gemini_answer", "")
        payload["gemini"]["judge_reasoning_zh"] = zh.get("gemini_judge_reasoning", "")
    return payload


def copy_image(payload: dict) -> bool:
    src = payload.pop("_src_image", None)
    if not src:
        return False
    src_path = ROOT / src
    if not src_path.exists():
        return False
    dst = OUT_IMAGES / Path(src).name
    if not dst.exists():
        shutil.copy2(src_path, dst)
    return True


def already_done() -> set[str]:
    if not OUT_JSONL.exists():
        return set()
    done: set[str] = set()
    with OUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                done.add(json.loads(line)["sample_id"])
            except Exception:
                continue
    return done


def process_one(client: httpx.Client, payload: dict) -> dict | None:
    if not copy_image(payload):
        return {"_error": "image_missing", "sample_id": payload["sample_id"]}
    fields = build_translate_payload(payload)
    try:
        zh = call_qwen_translate(client, fields)
    except Exception as e:  # noqa: BLE001
        return {"_error": f"translate_failed: {e}", "sample_id": payload["sample_id"]}
    return attach_translations(payload, zh)


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    OUT_IMAGES.mkdir(exist_ok=True)

    print("Loading manual_review.jsonl ...", flush=True)
    latest = load_latest_reviews()
    print(f"  {len(latest)} unique sample_ids", flush=True)

    selected = [r for r in latest.values() if is_hard_unverified(r)]
    print(f"Selected {len(selected)} unverified+hard samples", flush=True)

    done = already_done()
    if done:
        print(f"Resuming: {len(done)} already in {OUT_JSONL.name}", flush=True)
    todo = [extract_payload(r) for r in selected if r["sample_id"] not in done]
    print(f"To process: {len(todo)}", flush=True)
    if not todo:
        return

    ok = 0
    failed = 0
    started = time.time()
    with OUT_JSONL.open("a", encoding="utf-8") as out_f, \
         httpx.Client(http2=False) as client, \
         ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(process_one, client, p): p["sample_id"] for p in todo}
        for i, fut in enumerate(as_completed(futures), 1):
            sid = futures[fut]
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001
                result = {"_error": f"unhandled: {e}", "sample_id": sid}
            if result and "_error" not in result:
                out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
                out_f.flush()
                ok += 1
            else:
                failed += 1
                print(f"  [fail] {sid}: {result.get('_error') if result else 'None'}", flush=True)
            if i % 25 == 0 or i == len(todo):
                elapsed = time.time() - started
                rate = i / elapsed if elapsed > 0 else 0
                print(f"  progress {i}/{len(todo)} ok={ok} failed={failed} ({rate:.1f}/s)", flush=True)

    print(f"\nDone. ok={ok} failed={failed}")
    print(f"Output: {OUT_JSONL}")
    print(f"Images: {OUT_IMAGES}")


if __name__ == "__main__":
    main()
