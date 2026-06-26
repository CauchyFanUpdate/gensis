#!/usr/bin/env python3
"""
stage3_verify.py — Stage 3 of the zerogen pipeline (closed-book verification).

For every SAMPLE that has both stage1 (ok) and stage2 (ok), this script runs
TWO independent VL calls — Qwen-VL + Gemini-VL — closed-book (gold answer
NOT shown to the verifier model) and grades each result based on `qtype`:

  * qtype == "mc": rule-based parse of the model's free text → first valid
                   letter → compared to gold correct_label.
  * qtype == "oe": Qwen 3.6 Plus (text-only) is invoked as a strict grader,
                   separately for the Qwen-VL candidate and the Gemini-VL
                   candidate. Judge sees question + gold + candidate →
                   {correct, reasoning}.

A sample is `keep=true` when ANY verifier disagrees with gold (HARD sample
worth human review). When BOTH verifiers agree with gold the sample is
`keep=false` (default-discard).

Output: stage3_output.jsonl (append-only, one record per SAMPLE). Resume by
skipping sample_ids already ok.

Run:
  python stage3_verify.py
  python stage3_verify.py --limit 5
  python stage3_verify.py --no-resume
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx


# ── API constants (hardcoded by project convention) ───────────────────────
QWEN_ENDPOINT = "<QWEN_ENDPOINT>"
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")
QWEN_MODEL = "qwen3.6-plus"

GEMINI_ENDPOINT = "<GEMINI_ENDPOINT>"
GEMINI_API_KEYS: list[str] = [
    # os.environ.get("GEMINI_API_KEY_1", ""),
    os.environ.get("GEMINI_API_KEY_2", ""),
]
GEMINI_MODEL = "gemini-3.1-pro-preview"

# ── Run config ────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
STAGE1_PATH = HERE / "stage1_output.jsonl"
STAGE2_PATH = HERE / "stage2_output.jsonl"
OUTPUT_PATH = HERE / "stage3_output.jsonl"

QWEN_CONCURRENCY = 256          # shared by VL answering + text-only judging
GEMINI_PER_KEY_LIMIT = 64       # × 2 keys = 128 in flight
DEFAULT_PARALLEL_SAMPLES = 256

MAX_RETRIES = 6
REQUEST_TIMEOUT = 1200.0

VL_MAX_TOKENS = 1024
VL_TEMPERATURE = 0.0

JUDGE_MAX_TOKENS = 512
JUDGE_TEMPERATURE = 0.0

GEMINI_SAFETY = [
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "OFF"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "OFF"},
]

_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ── Verifier prompts ──────────────────────────────────────────────────────

MC_VERIFIER_SYSTEM = (
    "You are answering a single multiple-choice visual question about ONE image.\n"
    "Inspect the image carefully. Then output ONLY the single letter of the correct option "
    "(e.g. `A`, `B`, `C`, ...). No explanation, no punctuation, no extra words. "
    "Just one capital letter."
)


def format_options_for_prompt(options: list[dict]) -> str:
    """Render structured options as a printable list. Multi-part options use ' / '."""
    lines = []
    for opt in options:
        label = opt["label"]
        parts = opt["parts"]
        if len(parts) == 1:
            lines.append(f"{label}. {parts[0]}")
        else:
            lines.append(f"{label}. " + " / ".join(parts))
    return "\n".join(lines)


def build_mc_user_text(question: str, options: list[dict]) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Options:\n{format_options_for_prompt(options)}\n\n"
        "Reply with ONLY the single capital letter of the correct option."
    )


OE_VERIFIER_SYSTEM = (
    "You are answering a single open-ended visual question about ONE image.\n"
    "Inspect the image carefully and answer in AT MOST FOUR words. "
    "No preamble, no explanation, no trailing punctuation beyond what's strictly necessary. "
    "Just the answer."
)


def build_oe_user_text(question: str) -> str:
    return f"Question:\n{question}\n\nAnswer in at most 4 words."


JUDGE_SYSTEM = (
    "You are a strict grader for short visual-QA answers. "
    "Decide whether a candidate answer matches the gold answer. "
    "A candidate is CORRECT iff it conveys exactly the same information as the gold "
    "(synonyms, equivalent numerals, equivalent units are OK; "
    "missing parts, extra incorrect information, off-by-one numbers, or contradictions are NOT OK). "
    "Output STRICT JSON ONLY (no prose outside the JSON, no markdown fences) with this schema:\n"
    "{\n"
    '  "correct":  <true | false>,\n'
    '  "reasoning": <one short sentence>\n'
    "}"
)


def build_judge_user_text(question: str, gold: str, candidate: str) -> str:
    return (
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Candidate answer:\n{candidate}\n\n"
        "Output the JSON judgment."
    )


# ── JSON extraction (for OE judge) ────────────────────────────────────────

def extract_json(text: str) -> Any:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("no '{' found", text, 0)
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise json.JSONDecodeError("unbalanced braces", text, start)


# ── MC label parser (rule-based) ──────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:[a-zA-Z]+)?\s*|\s*```")


def parse_mc_label(raw_text: str, n_options: int) -> str | None:
    if not isinstance(raw_text, str):
        return None
    text = _THINK_RE.sub(" ", raw_text)
    text = _FENCE_RE.sub(" ", text).strip()
    if not text:
        return None
    valid = set(_LABELS[:n_options])
    upper = text.upper()
    pattern = re.compile(r"(?<![A-Z0-9])([A-Z])(?![A-Z0-9])")
    for m in pattern.finditer(upper):
        letter = m.group(1)
        if letter in valid:
            return letter
    return None


# ── Gemini key pool ───────────────────────────────────────────────────────

class GeminiKeyPool:
    def __init__(self, keys: list[str], per_key_limit: int):
        if not keys:
            raise ValueError("GeminiKeyPool requires at least one key")
        self._keys = list(keys)
        self._sems = [asyncio.Semaphore(per_key_limit) for _ in keys]
        self._idx = 0
        self._idx_lock = asyncio.Lock()

    @asynccontextmanager
    async def use_key(self):
        async with self._idx_lock:
            i = self._idx % len(self._keys)
            self._idx += 1
        async with self._sems[i]:
            yield self._keys[i]


# ── HTTP retryable error tuple ────────────────────────────────────────────

_RETRYABLE = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ProxyError,
)


# ── Qwen VL call (closed-book answer) ─────────────────────────────────────

async def call_qwen_vl(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    image_b64: str,
    system_prompt: str,
    user_text: str,
    sample_id: str,
    label: str,
) -> str:
    """Returns the raw text reply from Qwen VL."""
    url = f"{QWEN_ENDPOINT.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}
    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": user_text},
    ]
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": "/no_think\n" + system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": VL_MAX_TOKENS,
        "temperature": VL_TEMPERATURE,
    }

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()
            if "choices" not in body or not body["choices"]:
                raise RuntimeError(f"qwen response missing choices: {str(body)[:200]}")
            return body["choices"][0]["message"]["content"].strip()
        except _RETRYABLE as e:
            last_err = e
            wait = min(2 ** attempt + attempt, 30)
            print(f"[stage3/{sample_id}/{label}] qwen-vl network retry {attempt + 1}/{MAX_RETRIES}: "
                  f"{type(e).__name__}: {e!r} (wait {wait}s)", flush=True)
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body_head = (e.response.text or "")[:200]
            if status == 429:
                wait = min(15 * (attempt + 1), 90)
                print(f"[stage3/{sample_id}/{label}] qwen-vl 429, wait {wait}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
                await asyncio.sleep(wait)
                last_err = e
            elif status in (400, 403):
                raise RuntimeError(f"qwen-vl HTTP {status}: {body_head}")
            else:
                wait = min(5 * (attempt + 1), 60)
                print(f"[stage3/{sample_id}/{label}] qwen-vl HTTP {status}, wait {wait}s | {body_head!r}",
                      flush=True)
                await asyncio.sleep(wait)
                last_err = e
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"[stage3/{sample_id}/{label}] qwen-vl retry {attempt + 1}/{MAX_RETRIES}: "
                  f"{type(e).__name__}: {e!r} (wait {wait}s)", flush=True)
            await asyncio.sleep(wait)
    raise RuntimeError(f"qwen-vl failed after {MAX_RETRIES} retries: {last_err!r}")


# ── Gemini VL call (closed-book answer) ───────────────────────────────────

async def call_gemini_vl(
    client: httpx.AsyncClient,
    key_pool: GeminiKeyPool,
    image_b64: str,
    system_prompt: str,
    user_text: str,
    sample_id: str,
    label: str,
) -> str:
    parts = [
        {"inlineData": {"mimeType": "image/png", "data": image_b64}},
        {"text": user_text},
    ]
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": VL_TEMPERATURE,
            "maxOutputTokens": 8192,
            "topP": 0.9,
        },
        "safetySettings": GEMINI_SAFETY,
    }

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with key_pool.use_key() as api_key:
                headers = {"api-key": api_key, "Content-Type": "application/json"}
                resp = await client.post(GEMINI_ENDPOINT, headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()
            if "Code" in body and "Error" in body and "candidates" not in body:
                err_str = str(body.get("Error", ""))
                low = err_str.lower()
                if any(s in low for s in ("429", "resource_exhausted", "quota", "exhausted")):
                    wait = min(25 * (attempt + 1), 180)
                    print(f"[stage3/{sample_id}/{label}] gemini-vl proxy quota, wait {wait}s "
                          f"(attempt {attempt + 1}/{MAX_RETRIES}) | {err_str[:120]!r}",
                          flush=True)
                    await asyncio.sleep(wait)
                    continue
                raise RuntimeError(f"gemini-vl proxy error Code={body.get('Code')} | {err_str[:200]}")
            candidates = body.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"gemini-vl response no candidates: {str(body)[:200]}")
            cand = candidates[0]
            parts_out = cand.get("content", {}).get("parts", [])
            text_parts = [p.get("text", "") for p in parts_out if "text" in p and not p.get("thought")]
            full = "".join(text_parts).strip()
            if not full:
                finish = cand.get("finishReason", "")
                raise RuntimeError(f"gemini-vl empty visible-text (finishReason={finish!r})")
            return full
        except _RETRYABLE as e:
            last_err = e
            wait = min(2 ** attempt + attempt, 30)
            print(f"[stage3/{sample_id}/{label}] gemini-vl network retry {attempt + 1}/{MAX_RETRIES}: "
                  f"{type(e).__name__}: {e!r} (wait {wait}s)", flush=True)
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body_head = (e.response.text or "")[:200]
            if status == 429:
                wait = min(20 * (attempt + 1), 120)
                print(f"[stage3/{sample_id}/{label}] gemini-vl 429, wait {wait}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
                await asyncio.sleep(wait)
                last_err = e
            elif status in (400, 403):
                raise RuntimeError(f"gemini-vl HTTP {status}: {body_head}")
            else:
                wait = min(5 * (attempt + 1), 60)
                print(f"[stage3/{sample_id}/{label}] gemini-vl HTTP {status}, wait {wait}s | {body_head!r}",
                      flush=True)
                await asyncio.sleep(wait)
                last_err = e
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"[stage3/{sample_id}/{label}] gemini-vl retry {attempt + 1}/{MAX_RETRIES}: "
                  f"{type(e).__name__}: {e!r} (wait {wait}s)", flush=True)
            await asyncio.sleep(wait)
    raise RuntimeError(f"gemini-vl failed after {MAX_RETRIES} retries: {last_err!r}")


# ── Qwen text-only judge (for OE grading) ─────────────────────────────────

async def judge_oe(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    question: str,
    gold: str,
    candidate: str,
    sample_id: str,
    label: str,
) -> dict:
    url = f"{QWEN_ENDPOINT.rstrip('/')}/chat/completions"
    headers = {"Authorization": f"Bearer {QWEN_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": "/no_think\n" + JUDGE_SYSTEM},
            {"role": "user", "content": build_judge_user_text(question, gold, candidate)},
        ],
        "max_tokens": JUDGE_MAX_TOKENS,
        "temperature": JUDGE_TEMPERATURE,
    }

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()
            if "choices" not in body or not body["choices"]:
                raise RuntimeError(f"judge response missing choices: {str(body)[:200]}")
            raw = body["choices"][0]["message"]["content"].strip()
            try:
                data = extract_json(raw)
            except Exception as e:
                return {"correct": None, "reasoning": None, "raw": raw,
                        "parse_error": f"{type(e).__name__}: {e!r}"}
            cor = data.get("correct")
            if isinstance(cor, str):
                cor = cor.strip().lower() in ("true", "yes", "y", "1")
            if cor is not None:
                cor = bool(cor)
            return {"correct": cor, "reasoning": data.get("reasoning"), "raw": raw}
        except _RETRYABLE as e:
            last_err = e
            wait = min(2 ** attempt + attempt, 30)
            print(f"[stage3/{sample_id}/{label}] judge network retry {attempt + 1}/{MAX_RETRIES}: "
                  f"{type(e).__name__}: {e!r} (wait {wait}s)", flush=True)
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body_head = (e.response.text or "")[:200]
            if status == 429:
                wait = min(15 * (attempt + 1), 90)
                print(f"[stage3/{sample_id}/{label}] judge 429, wait {wait}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES})", flush=True)
                await asyncio.sleep(wait)
                last_err = e
            elif status in (400, 403):
                raise RuntimeError(f"judge HTTP {status}: {body_head}")
            else:
                wait = min(5 * (attempt + 1), 60)
                print(f"[stage3/{sample_id}/{label}] judge HTTP {status}, wait {wait}s | {body_head!r}",
                      flush=True)
                await asyncio.sleep(wait)
                last_err = e
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"[stage3/{sample_id}/{label}] judge retry {attempt + 1}/{MAX_RETRIES}: "
                  f"{type(e).__name__}: {e!r} (wait {wait}s)", flush=True)
            await asyncio.sleep(wait)
    raise RuntimeError(f"judge failed after {MAX_RETRIES} retries: {last_err!r}")


# ── JSONL helpers ─────────────────────────────────────────────────────────

class JsonlAppender:
    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()

    async def write(self, record: dict) -> None:
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_sample_ids(path: Path) -> set[str]:
    by_sid: dict[str, dict] = {}
    if not path.exists():
        return set()
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
    return {sid for sid, rec in by_sid.items() if rec.get("status") == "ok"}


def load_jsonl_by_sample(path: Path, ok_only: bool = True) -> dict[str, dict]:
    by_sid: dict[str, dict] = {}
    if not path.exists():
        return by_sid
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ok_only and rec.get("status") != "ok":
                continue
            sid = rec.get("sample_id")
            if sid:
                by_sid[sid] = rec
    return by_sid


# ── Per-sample MC / OE verifiers ──────────────────────────────────────────

async def verify_mc_sample(
    qwen_client, qwen_sem,
    gemini_client, gemini_pool,
    image_b64: str,
    question: str,
    options: list[dict],
    correct_label: str,
    sample_id: str,
) -> dict:
    n_opts = len(options)
    user_text = build_mc_user_text(question, options)
    qwen_task = asyncio.create_task(
        call_qwen_vl(qwen_client, qwen_sem, image_b64,
                     MC_VERIFIER_SYSTEM, user_text, sample_id, "qwen")
    )
    gem_task = asyncio.create_task(
        call_gemini_vl(gemini_client, gemini_pool, image_b64,
                       MC_VERIFIER_SYSTEM, user_text, sample_id, "gemini")
    )
    qwen_res, gem_res = await asyncio.gather(qwen_task, gem_task, return_exceptions=True)

    def pack(res) -> dict:
        if isinstance(res, BaseException):
            return {"raw": None, "parsed_label": None, "correct": None,
                    "error": f"{type(res).__name__}: {res!r}"}
        parsed = parse_mc_label(res, n_opts)
        return {"raw": res, "parsed_label": parsed,
                "correct": (parsed == correct_label) if parsed else False}

    q = pack(qwen_res)
    g = pack(gem_res)
    both_correct = bool(q.get("correct")) and bool(g.get("correct"))
    any_disagree = (not q.get("correct")) or (not g.get("correct"))
    return {
        "gold_label": correct_label,
        "n_options": n_opts,
        "qwen": q,
        "gemini": g,
        "both_correct": both_correct,
        "any_disagree": any_disagree,
        "keep": any_disagree,
    }


async def verify_oe_sample(
    qwen_client, qwen_sem,
    gemini_client, gemini_pool,
    image_b64: str,
    question: str,
    gold: str,
    sample_id: str,
) -> dict:
    user_text = build_oe_user_text(question)
    qwen_task = asyncio.create_task(
        call_qwen_vl(qwen_client, qwen_sem, image_b64,
                     OE_VERIFIER_SYSTEM, user_text, sample_id, "qwen")
    )
    gem_task = asyncio.create_task(
        call_gemini_vl(gemini_client, gemini_pool, image_b64,
                       OE_VERIFIER_SYSTEM, user_text, sample_id, "gemini")
    )
    qwen_vl, gem_vl = await asyncio.gather(qwen_task, gem_task, return_exceptions=True)

    qwen_failed = isinstance(qwen_vl, BaseException)
    gem_failed = isinstance(gem_vl, BaseException)

    qwen_judge: dict = {}
    gem_judge: dict = {}
    judge_tasks = []
    if not qwen_failed:
        judge_tasks.append(("qwen", asyncio.create_task(
            judge_oe(qwen_client, qwen_sem, question, gold, qwen_vl, sample_id, "judge-qwen")
        )))
    if not gem_failed:
        judge_tasks.append(("gemini", asyncio.create_task(
            judge_oe(qwen_client, qwen_sem, question, gold, gem_vl, sample_id, "judge-gemini")
        )))
    for which, task in judge_tasks:
        try:
            res = await task
        except Exception as e:
            res = {"correct": None, "reasoning": None, "raw": None,
                   "judge_error": f"{type(e).__name__}: {e!r}"}
        if which == "qwen":
            qwen_judge = res
        else:
            gem_judge = res

    def pack(vl_res, judge: dict) -> dict:
        if isinstance(vl_res, BaseException):
            return {"vl_answer": None, "judge_correct": None, "judge_reasoning": None,
                    "error": f"{type(vl_res).__name__}: {vl_res!r}"}
        out = {"vl_answer": vl_res,
               "judge_correct": judge.get("correct"),
               "judge_reasoning": judge.get("reasoning")}
        if judge.get("parse_error"):
            out["judge_parse_error"] = judge["parse_error"]
        if judge.get("judge_error"):
            out["judge_error"] = judge["judge_error"]
        return out

    q = pack(qwen_vl, qwen_judge)
    g = pack(gem_vl, gem_judge)
    both_correct = bool(q.get("judge_correct")) and bool(g.get("judge_correct"))
    any_disagree = (not q.get("judge_correct")) or (not g.get("judge_correct"))
    return {
        "gold": gold,
        "qwen": q,
        "gemini": g,
        "both_correct": both_correct,
        "any_disagree": any_disagree,
        "keep": any_disagree,
    }


async def process_one(
    sample_sem: asyncio.Semaphore,
    qwen_client, qwen_sem,
    gemini_client, gemini_pool,
    writer: JsonlAppender,
    s1: dict,
    s2: dict,
) -> tuple[str, bool]:
    sample_id = s1["sample_id"]
    seed_id   = s1["seed_id"]
    family_id = s1["family_id"]
    qtype     = s1["qtype"]
    t0 = time.time()

    img_path = HERE / s2["image_path"]
    record: dict[str, Any] = {
        "sample_id": sample_id,
        "seed_id":   seed_id,
        "family_id": family_id,
        "qtype":     qtype,
        "status":    "failed",
        "elapsed_s": None,
        "created_at": int(time.time()),
    }

    try:
        img_b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    except Exception as e:
        record["error"] = f"image read failed: {type(e).__name__}: {e!r}"
        record["elapsed_s"] = round(time.time() - t0, 2)
        await writer.write(record)
        print(f"[stage3/{sample_id}] FAILED reading image: {e!r}", flush=True)
        return sample_id, False

    s1_payload = s1["stage1"]
    async with sample_sem:
        try:
            if qtype == "mc":
                verdict = await verify_mc_sample(
                    qwen_client, qwen_sem, gemini_client, gemini_pool,
                    img_b64,
                    s1_payload["question"],
                    s1_payload["options"],
                    s1_payload["correct_label"],
                    sample_id,
                )
            else:  # oe
                verdict = await verify_oe_sample(
                    qwen_client, qwen_sem, gemini_client, gemini_pool,
                    img_b64,
                    s1_payload["question"],
                    str(s1_payload["answer"]),
                    sample_id,
                )
        except Exception as e:
            record["error"] = f"verify crashed: {type(e).__name__}: {e!r}"
            record["elapsed_s"] = round(time.time() - t0, 2)
            await writer.write(record)
            print(f"[stage3/{sample_id}] FAILED verify: {e!r}", flush=True)
            return sample_id, False

    record.update(verdict)
    record["status"] = "ok"
    record["elapsed_s"] = round(time.time() - t0, 2)
    await writer.write(record)

    if qtype == "mc":
        q_label = (verdict["qwen"] or {}).get("parsed_label")
        g_label = (verdict["gemini"] or {}).get("parsed_label")
        print(
            f"[stage3/{sample_id}] ok in {record['elapsed_s']}s | mc | "
            f"qwen={q_label} gem={g_label} gold={verdict['gold_label']} keep={verdict['keep']}",
            flush=True,
        )
    else:
        q_jc = (verdict["qwen"] or {}).get("judge_correct")
        g_jc = (verdict["gemini"] or {}).get("judge_correct")
        print(
            f"[stage3/{sample_id}] ok in {record['elapsed_s']}s | oe | "
            f"qwen_jc={q_jc} gem_jc={g_jc} keep={verdict['keep']}",
            flush=True,
        )
    return sample_id, True


# ── Main ──────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    stage1_path = Path(args.stage1)
    stage2_path = Path(args.stage2)
    output_path = Path(args.output)

    stage1_by_sid = load_jsonl_by_sample(stage1_path, ok_only=True)
    stage2_by_sid = load_jsonl_by_sample(stage2_path, ok_only=True)
    print(f"Loaded {len(stage1_by_sid)} stage1 / {len(stage2_by_sid)} stage2 ok samples.",
          flush=True)

    candidates = []
    for sid, s1 in stage1_by_sid.items():
        s2 = stage2_by_sid.get(sid)
        if not s2:
            continue
        candidates.append((sid, s1, s2))
    candidates.sort(key=lambda t: t[0])
    print(f"{len(candidates)} samples have both stage1 and stage2.", flush=True)

    if args.limit > 0:
        candidates = candidates[: args.limit]
        print(f"--limit {args.limit} → processing first {len(candidates)} samples", flush=True)

    if args.no_resume and output_path.exists():
        backup = output_path.with_suffix(output_path.suffix + ".bak")
        output_path.rename(backup)
        print(f"--no-resume: moved existing {output_path} → {backup}", flush=True)

    done = load_done_sample_ids(output_path)
    if done:
        print(f"resume: {len(done)} sample_ids already ok in {output_path} — will skip", flush=True)

    pending = [(sid, s1, s2) for sid, s1, s2 in candidates if sid not in done]
    if not pending:
        print("nothing to do — all samples already processed.", flush=True)
        return

    n_keys = len(GEMINI_API_KEYS)
    gem_total = GEMINI_PER_KEY_LIMIT * n_keys
    print(
        f"submitting {len(pending)} samples | parallel_samples={args.parallel_samples} | "
        f"qwen_concurrency={args.qwen_concurrency} | "
        f"gemini_concurrency={gem_total} ({n_keys} keys × {GEMINI_PER_KEY_LIMIT}/key)",
        flush=True,
    )

    sample_sem = asyncio.Semaphore(args.parallel_samples)
    qwen_sem = asyncio.Semaphore(args.qwen_concurrency)
    gemini_pool = GeminiKeyPool(GEMINI_API_KEYS, GEMINI_PER_KEY_LIMIT)

    timeout = httpx.Timeout(REQUEST_TIMEOUT, read=REQUEST_TIMEOUT, pool=60)
    qwen_limits = httpx.Limits(
        max_connections=args.qwen_concurrency * 2,
        max_keepalive_connections=args.qwen_concurrency,
    )
    gemini_limits = httpx.Limits(
        max_connections=gem_total * 2,
        max_keepalive_connections=gem_total,
    )

    writer = JsonlAppender(output_path)

    t_start = time.time()
    async with httpx.AsyncClient(timeout=timeout, limits=qwen_limits, trust_env=False) as qwen_client, \
               httpx.AsyncClient(timeout=timeout, limits=gemini_limits, trust_env=False) as gemini_client:
        tasks = [
            process_one(sample_sem, qwen_client, qwen_sem,
                        gemini_client, gemini_pool, writer, s1, s2)
            for _, s1, s2 in pending
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    n_ok = sum(1 for r in results if isinstance(r, tuple) and r[1])
    n_fail = len(results) - n_ok
    print(
        f"\nDone in {time.time() - t_start:.1f}s. "
        f"ok={n_ok} failed={n_fail} → {output_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 3 (verify): per-sample closed-book Qwen+Gemini answering"
    )
    parser.add_argument("--stage1", default=str(STAGE1_PATH))
    parser.add_argument("--stage2", default=str(STAGE2_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--parallel-samples", type=int, default=DEFAULT_PARALLEL_SAMPLES)
    parser.add_argument("--qwen-concurrency", type=int, default=QWEN_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\ninterrupted — partial results are in the output JSONL; rerun to resume.",
              flush=True)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
