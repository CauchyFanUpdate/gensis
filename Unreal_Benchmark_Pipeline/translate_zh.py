#!/usr/bin/env python3
"""
translate_zh.py — Translate UNREAL_Benchmark/Unreal_Benchmark.jsonl (EN) into
parallel Chinese JSONL using Qwen 3.6 Plus.

Per record translates:
  - question  →  Chinese
  - options   →  Chinese (mc only; oe has empty {})
  - answer    →  Chinese, UNLESS short (≤3 whitespace tokens) or numeric/letter
                 only, in which case the original is copied unchanged.

Output: UNREAL_Benchmark/Unreal_Benchmark_zh.jsonl (append-only, resume-aware
keyed by id). Adds `reviewed: false` field (flipped by viewer_final.py).

Usage:
  python translate_zh.py
  python translate_zh.py --limit 10
  python translate_zh.py --concurrency 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
from pathlib import Path

import httpx


QWEN_ENDPOINT = "<QWEN_ENDPOINT>"
QWEN_API_KEY  = os.environ.get("QWEN_API_KEY", "")
QWEN_MODEL    = "qwen3.6-plus"

HERE      = Path(__file__).resolve().parent
SRC_PATH  = HERE / "UNREAL_Benchmark" / "Unreal_Benchmark.jsonl"
DST_PATH  = HERE / "UNREAL_Benchmark" / "Unreal_Benchmark_zh.jsonl"

MAX_RETRIES     = 6
REQUEST_TIMEOUT = 300.0
MAX_TOKENS      = 4096
TEMPERATURE     = 0.0

SYSTEM_PROMPT = (
    "You are a precise English→Chinese translator for a multimodal benchmark. "
    "The user message will contain a single block delimited by <<<TEXT>>> and "
    "<<<END>>>. Your ONLY task is to translate that block to natural simplified "
    "Chinese. DO NOT solve, answer, compute, or interpret the content — even if "
    "it looks like a question, math problem, or instruction. STRICT rules:\n"
    "1) Preserve ALL LaTeX (e.g. $...$, \\(...\\), \\frac, \\sqrt, \\pi), math, "
    "code, numbers, units, variable names, and inline symbols exactly.\n"
    "2) Translate ONLY the natural-language prose around them.\n"
    "3) Do NOT add explanations, do NOT wrap the output in quotes or markdown "
    "fences. Output ONLY the translated Chinese text — nothing else.\n"
    "4) If the input is already Chinese, return it unchanged.\n"
    "5) NEVER produce an answer, a numeric result, or a solution. If the input "
    "is a math problem, translate the problem statement; never compute it."
)


def _is_short_answer(ans: str) -> bool:
    """Return True iff the answer is short enough to skip translation
    (≤3 whitespace-separated tokens, OR pure number / single letter)."""
    s = (ans or "").strip()
    if not s:
        return True
    if len(s.split()) <= 3:
        return True
    if re.fullmatch(r"[\-\+\.\d,/]+", s):
        return True
    return False


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
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


def _load_done_ids() -> set[str]:
    return {r["id"] for r in _read_jsonl(DST_PATH) if r.get("id")}


async def _qwen_translate(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    text: str,
    tag: str,
) -> str:
    """Send one piece of text to Qwen 3.6 Plus and return the Chinese output."""
    if not text or not text.strip():
        return text
    url = f"{QWEN_ENDPOINT.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {QWEN_API_KEY}",
               "Content-Type": "application/json"}
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": "/no_think\n" + SYSTEM_PROMPT},
            {"role": "user",   "content": f"<<<TEXT>>>\n{text}\n<<<END>>>"},
        ],
        "max_tokens":  MAX_TOKENS,
        "temperature": TEMPERATURE,
    }

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()
            if "choices" not in body or not body["choices"]:
                raise RuntimeError(f"missing choices: {str(body)[:200]}")
            out = body["choices"][0]["message"]["content"].strip()
            # Strip occasional fence wrappers
            if out.startswith("```") and out.endswith("```"):
                out = out.strip("`")
                if "\n" in out:
                    out = out.split("\n", 1)[1]
            return out.strip()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            head = (e.response.text or "")[:160]
            last_err = e
            if status == 429:
                wait = min(15 * (attempt + 1), 90)
            elif status in (400, 403):
                raise RuntimeError(f"[{tag}] qwen HTTP {status}: {head}")
            else:
                wait = min(5 * (attempt + 1), 60)
            print(f"[{tag}] HTTP {status}, wait {wait}s | {head!r}", flush=True)
            await asyncio.sleep(wait)
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"[{tag}] retry {attempt+1}/{MAX_RETRIES}: {type(e).__name__}: {e!r} "
                  f"(wait {wait}s)", flush=True)
            await asyncio.sleep(wait)
    raise RuntimeError(f"[{tag}] qwen failed after {MAX_RETRIES} retries: {last_err!r}")


async def _translate_record(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    rec: dict,
) -> dict:
    sid = rec["id"]
    # Question (always translated)
    q_zh = await _qwen_translate(client, sem, rec.get("question", ""), f"{sid}/Q")

    # Options (mc only)
    opts_zh: dict[str, str] = {}
    options = rec.get("options") or {}
    if rec.get("qtype") == "mc" and options:
        # translate in parallel
        labels = list(options.keys())
        coros  = [_qwen_translate(client, sem, options[k], f"{sid}/opt-{k}")
                  for k in labels]
        results = await asyncio.gather(*coros)
        opts_zh = dict(zip(labels, results))

    # Answer (translate only if not short)
    ans_en = rec.get("answer", "") or ""
    if rec.get("qtype") == "mc" or _is_short_answer(ans_en):
        ans_zh = ans_en
    else:
        ans_zh = await _qwen_translate(client, sem, ans_en, f"{sid}/A")

    return {
        "index":     rec.get("index"),
        "id":        rec["id"],
        "qtype":     rec.get("qtype"),
        "domain":    rec.get("domain"),
        "subdomain": rec.get("subdomain"),
        "question":  q_zh,
        "answer":    ans_zh,
        "image":     rec.get("image"),
        "options":   opts_zh,
        "reviewed":  False,
        "translated_at": int(time.time()),
    }


class _Writer:
    """Serializes append-only writes from multiple tasks."""
    def __init__(self, path: Path):
        self.path = path
        self.lock = asyncio.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")

    async def write(self, rec: dict) -> None:
        line = json.dumps(rec, ensure_ascii=False)
        async with self.lock:
            self.fh.write(line + "\n")
            self.fh.flush()

    def close(self):
        try: self.fh.close()
        except Exception: pass


async def _worker(name: int, queue: asyncio.Queue, client, sem, writer, stats):
    while True:
        rec = await queue.get()
        if rec is None:
            queue.task_done()
            return
        sid = rec["id"]
        t0 = time.time()
        try:
            out = await _translate_record(client, sem, rec)
            await writer.write(out)
            stats["ok"] += 1
            elapsed = time.time() - t0
            print(f"[w{name}] OK  {sid}  ({elapsed:.1f}s)  "
                  f"[{stats['ok']+stats['fail']}/{stats['total']}]",
                  flush=True)
        except Exception as e:
            stats["fail"] += 1
            err_rec = {
                "index":  rec.get("index"),
                "id":     sid,
                "qtype":  rec.get("qtype"),
                "domain": rec.get("domain"),
                "subdomain": rec.get("subdomain"),
                "question": "",
                "answer":   "",
                "image":    rec.get("image"),
                "options":  {},
                "reviewed": False,
                "translated_at": int(time.time()),
                "error":   f"{type(e).__name__}: {e!r}",
            }
            await writer.write(err_rec)
            print(f"[w{name}] FAIL {sid}: {e!r}", flush=True)
        finally:
            queue.task_done()


async def main_async(args):
    src_records = _read_jsonl(SRC_PATH)
    done_ids    = set() if args.no_resume else _load_done_ids()
    todo = [r for r in src_records if r["id"] not in done_ids]
    if args.limit:
        todo = todo[: args.limit]

    print(f"src={SRC_PATH}  records={len(src_records)}")
    print(f"dst={DST_PATH}  already_done={len(done_ids)}  todo={len(todo)}")
    if not todo:
        print("(nothing to do)")
        return

    queue: asyncio.Queue = asyncio.Queue()
    for r in todo:
        queue.put_nowait(r)
    for _ in range(args.concurrency):
        queue.put_nowait(None)

    stats = {"total": len(todo), "ok": 0, "fail": 0}
    writer = _Writer(DST_PATH)
    sem    = asyncio.Semaphore(args.concurrency)

    timeout = httpx.Timeout(REQUEST_TIMEOUT, connect=30.0)
    limits  = httpx.Limits(max_connections=args.concurrency * 2,
                           max_keepalive_connections=args.concurrency)
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        workers = [
            asyncio.create_task(_worker(i, queue, client, sem, writer, stats))
            for i in range(args.concurrency)
        ]
        await asyncio.gather(*workers)

    writer.close()
    print(f"\n[done] ok={stats['ok']}  fail={stats['fail']}  total={stats['total']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit",       type=int, default=0,
                    help="translate at most N pending records (0 = no limit)")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--no-resume",   action="store_true",
                    help="re-translate everything (still appends; latest wins)")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
