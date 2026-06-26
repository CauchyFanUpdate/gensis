#!/usr/bin/env python3
"""
eval_models_api.py — Evaluate remote API VLMs on the released zerogen benchmark.

Reads release/dataset.json (200 samples; mc + oe). For each (model, sample):
  * MC: VL model answers with a single letter; rule-based parse → compare to
        correct_label.
  * OE: VL model answers in <=4 words; qwen3.6-plus (text-only) judges
        candidate vs. gold and returns {correct, reasoning}.

Per-model results land in eval_results/<model>.jsonl (append-only, resume-aware).
A summary (accuracy per model + qtype) lands in eval_results/summary.json after
each run.

Run:
  python eval_models_api.py                                # all registered models
  python eval_models_api.py --models qwen3.6-plus gemini-3.1-pro
  python eval_models_api.py --models qwen3-vl-32b-thinking --limit 10
  python eval_models_api.py --no-resume                    # rebuild from scratch
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
from tqdm.asyncio import tqdm


# ── API constants ─────────────────────────────────────────────────────────
QWEN_ENDPOINT = "<QWEN_ENDPOINT>"
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", "")

GEMINI_ENDPOINT = "<GEMINI_ENDPOINT>"
GEMINI_API_KEYS: list[str] = [
    os.environ.get("GEMINI_API_KEY_1", ""),
    os.environ.get("GEMINI_API_KEY_2", ""),
]

JUDGE_MODEL = "qwen3.6-plus"

# ── Model registry ────────────────────────────────────────────────────────
# api_model: the exact model id sent to the API.
# provider:  which call shape to use ("qwen" OpenAI-compat | "gemini" Google native).
# thinking:  hint that this is a "thinking" variant (longer max tokens, allow CoT).
MODELS: dict[str, dict] = {
    "qwen3.6-plus-thinking":        {"provider": "qwen",   "api_model": "qwen3.6-plus",                 "thinking": True},
    # "qwen3.5-plus":                 {"provider": "qwen",   "api_model": "qwen3.5-plus",                 "thinking": False},
    # "qwen3-vl-8b-instruct":           {"provider": "qwen",   "api_model": "qwen3-vl-8b-instruct",         "thinking": False},
    # "qwen3-vl-8b-thinking":         {"provider": "qwen",   "api_model": "qwen3-vl-8b-thinking",         "thinking": True},
    # "qwen3-vl-32b-thinking":        {"provider": "qwen",   "api_model": "qwen3-vl-32b-thinking",        "thinking": True},
    # "qwen3-vl-235b-a22b-instruct":  {"provider": "qwen",   "api_model": "qwen3-vl-235b-a22b-instruct",  "thinking": False},
    # The runway /openai/google/v1:generateContent proxy is hard-pinned to
    # gemini-3.1-pro-preview regardless of payload `model`. Cannot reach
    # gemini-3-pro from here; add a separate endpoint to support it.
    "gemini-3.1-pro":               {"provider": "gemini", "api_model": "gemini-3.1-pro-preview",       "thinking": True},
}

# ── Run config ────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
DATASET_PATH = HERE / "release" / "dataset.json"
IMAGES_DIR = HERE / "release" / "images"
RESULTS_DIR = HERE / "eval_results"
SUMMARY_PATH = RESULTS_DIR / "summary.json"

QWEN_CONCURRENCY = 128  # shared by every Qwen VL + judge call
GEMINI_PER_KEY_LIMIT = 64  # × 2 keys = 128 in flight
DEFAULT_PARALLEL_SAMPLES = 128

MAX_RETRIES = 6
REQUEST_TIMEOUT = 1200.0

VL_TEMPERATURE = 0.0
VL_MAX_TOKENS_INSTRUCT = 1024  # plenty for a single letter / a few words
VL_MAX_TOKENS_THINKING = 8192  # allow CoT to expand before final answer

JUDGE_MAX_TOKENS = 512
JUDGE_TEMPERATURE = 0.0

GEMINI_SAFETY = [
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "OFF"},
]

_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ── Prompts ───────────────────────────────────────────────────────────────

MC_SYSTEM = (
    "You are answering a single multiple-choice visual question about ONE image.\n"
    "Inspect the image carefully. Then output ONLY the single letter of the correct option "
    "(e.g. `A`, `B`, `C`, ...). No explanation, no punctuation, no extra words. "
    "Just one capital letter."
)


def format_options_for_prompt(options: list[dict]) -> str:
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


OE_SYSTEM = (
    "You are answering a single open-ended visual question about ONE image.\n"
    "No preamble, no explanation, no trailing punctuation beyond what's strictly necessary. "
    "Just the answer."
)


def build_oe_user_text(question: str) -> str:
    return f"Question:\n{question}\n\nAnswer in at most 4 words."


JUDGE_SYSTEM = (
    "You are a strict grader for short visual-QA answers.\n"
    "You will be shown a Question, the Gold answer, and a Candidate answer "
    "produced by a vision-language model. Decide whether the Candidate is correct.\n\n"
    "Rules:\n"
    "- A Candidate is CORRECT if and only if it conveys EXACTLY the same factual "
    "content as the Gold answer for the asked question.\n"
    '- Synonyms, paraphrases, equivalent numerals ("3" == "three"), and equivalent '
    'units ("5 m" == "5 meters") are acceptable.\n'
    "- Off-by-one numbers, contradictions, missing required parts, or extra incorrect "
    "information make the Candidate INCORRECT.\n"
    "- If the Gold contains multiple required pieces (e.g. counts of two object types), "
    "the Candidate must include ALL of them to be CORRECT.\n"
    "- Capitalization, punctuation, and surrounding filler words are ignored.\n"
    "- Refusals, non-answers, or 'I cannot determine' are INCORRECT.\n\n"
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


# ── JSON / label parsers ──────────────────────────────────────────────────


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
    # Prefer the LAST stand-alone capital letter — for thinking models the
    # final answer comes after the chain of thought.
    pattern = re.compile(r"(?<![A-Z0-9])([A-Z])(?![A-Z0-9])")
    last: str | None = None
    for m in pattern.finditer(upper):
        letter = m.group(1)
        if letter in valid:
            last = letter
    return last


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


_RETRYABLE = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ProxyError,
)


# ── Qwen VL / text call ───────────────────────────────────────────────────


async def call_qwen(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    api_model: str,
    system_prompt: str,
    user_content: Any,
    max_tokens: int,
    sample_id: str,
    label: str,
    temperature: float = VL_TEMPERATURE,
) -> str:
    """Returns the raw assistant content from Qwen OpenAI-compat endpoint."""
    url = f"{QWEN_ENDPOINT.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {QWEN_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
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
            content = body["choices"][0]["message"].get("content")
            if content is None:
                raise RuntimeError(f"qwen response missing content: {str(body)[:200]}")
            return content.strip()
        except _RETRYABLE as e:
            last_err = e
            wait = min(2**attempt + attempt, 30)
            print(
                f"[{label}/{sample_id}] qwen network retry {attempt + 1}/{MAX_RETRIES}: "
                f"{type(e).__name__}: {e!r} (wait {wait}s)",
                flush=True,
            )
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body_head = (e.response.text or "")[:200]
            if status == 429:
                wait = min(15 * (attempt + 1), 90)
                print(
                    f"[{label}/{sample_id}] qwen 429, wait {wait}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})",
                    flush=True,
                )
                await asyncio.sleep(wait)
                last_err = e
            elif status in (400, 403):
                raise RuntimeError(f"qwen HTTP {status}: {body_head}")
            else:
                wait = min(5 * (attempt + 1), 60)
                print(
                    f"[{label}/{sample_id}] qwen HTTP {status}, wait {wait}s | {body_head!r}",
                    flush=True,
                )
                await asyncio.sleep(wait)
                last_err = e
        except Exception as e:
            last_err = e
            wait = min(2**attempt, 30)
            print(
                f"[{label}/{sample_id}] qwen retry {attempt + 1}/{MAX_RETRIES}: "
                f"{type(e).__name__}: {e!r} (wait {wait}s)",
                flush=True,
            )
            await asyncio.sleep(wait)
    raise RuntimeError(f"qwen failed after {MAX_RETRIES} retries: {last_err!r}")


# ── Gemini VL call ────────────────────────────────────────────────────────


async def call_gemini_vl(
    client: httpx.AsyncClient,
    key_pool: GeminiKeyPool,
    image_b64: str,
    system_prompt: str,
    user_text: str,
    max_tokens: int,
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
            "maxOutputTokens": max_tokens,
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
                if any(
                    s in low
                    for s in ("429", "resource_exhausted", "quota", "exhausted")
                ):
                    wait = min(25 * (attempt + 1), 180)
                    print(
                        f"[{label}/{sample_id}] gemini proxy quota, wait {wait}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES}) | {err_str[:120]!r}",
                        flush=True,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise RuntimeError(
                    f"gemini proxy error Code={body.get('Code')} | {err_str[:200]}"
                )
            candidates = body.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"gemini response no candidates: {str(body)[:200]}")
            cand = candidates[0]
            parts_out = cand.get("content", {}).get("parts", [])
            text_parts = [
                p.get("text", "")
                for p in parts_out
                if "text" in p and not p.get("thought")
            ]
            full = "".join(text_parts).strip()
            if not full:
                finish = cand.get("finishReason", "")
                raise RuntimeError(
                    f"gemini empty visible-text (finishReason={finish!r})"
                )
            return full
        except _RETRYABLE as e:
            last_err = e
            wait = min(2**attempt + attempt, 30)
            print(
                f"[{label}/{sample_id}] gemini network retry {attempt + 1}/{MAX_RETRIES}: "
                f"{type(e).__name__}: {e!r} (wait {wait}s)",
                flush=True,
            )
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body_head = (e.response.text or "")[:200]
            if status == 429:
                wait = min(20 * (attempt + 1), 120)
                print(
                    f"[{label}/{sample_id}] gemini 429, wait {wait}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})",
                    flush=True,
                )
                await asyncio.sleep(wait)
                last_err = e
            elif status in (400, 403):
                raise RuntimeError(f"gemini HTTP {status}: {body_head}")
            else:
                wait = min(5 * (attempt + 1), 60)
                print(
                    f"[{label}/{sample_id}] gemini HTTP {status}, wait {wait}s | {body_head!r}",
                    flush=True,
                )
                await asyncio.sleep(wait)
                last_err = e
        except Exception as e:
            last_err = e
            wait = min(2**attempt, 30)
            print(
                f"[{label}/{sample_id}] gemini retry {attempt + 1}/{MAX_RETRIES}: "
                f"{type(e).__name__}: {e!r} (wait {wait}s)",
                flush=True,
            )
            await asyncio.sleep(wait)
    raise RuntimeError(f"gemini failed after {MAX_RETRIES} retries: {last_err!r}")


# ── Higher-level provider call ────────────────────────────────────────────


async def call_vl(
    spec: dict,
    qwen_client,
    qwen_sem,
    gemini_client,
    gemini_pool,
    image_b64: str,
    system_prompt: str,
    user_text: str,
    sample_id: str,
    model_name: str,
) -> str:
    max_tokens = (
        VL_MAX_TOKENS_THINKING if spec.get("thinking") else VL_MAX_TOKENS_INSTRUCT
    )
    if spec["provider"] == "qwen":
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_b64}"},
            },
            {"type": "text", "text": user_text},
        ]
        return await call_qwen(
            qwen_client,
            qwen_sem,
            api_model=spec["api_model"],
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=max_tokens,
            sample_id=sample_id,
            label=model_name,
        )
    elif spec["provider"] == "gemini":
        return await call_gemini_vl(
            gemini_client,
            gemini_pool,
            image_b64=image_b64,
            system_prompt=system_prompt,
            user_text=user_text,
            max_tokens=max_tokens,
            sample_id=sample_id,
            label=model_name,
        )
    else:
        raise ValueError(f"unknown provider {spec['provider']!r}")


# ── Qwen text-only judge ──────────────────────────────────────────────────


async def judge_oe(
    qwen_client,
    qwen_sem,
    question: str,
    gold: str,
    candidate: str,
    sample_id: str,
    label: str,
) -> dict:
    raw = await call_qwen(
        qwen_client,
        qwen_sem,
        api_model=JUDGE_MODEL,
        system_prompt="/no_think\n" + JUDGE_SYSTEM,
        user_content=build_judge_user_text(question, gold, candidate),
        max_tokens=JUDGE_MAX_TOKENS,
        sample_id=sample_id,
        label=label,
        temperature=JUDGE_TEMPERATURE,
    )
    try:
        data = extract_json(raw)
    except Exception as e:
        return {
            "correct": None,
            "reasoning": None,
            "raw": raw,
            "parse_error": f"{type(e).__name__}: {e!r}",
        }
    cor = data.get("correct")
    if isinstance(cor, str):
        cor = cor.strip().lower() in ("true", "yes", "y", "1")
    if cor is not None:
        cor = bool(cor)
    return {"correct": cor, "reasoning": data.get("reasoning"), "raw": raw}


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


# ── Per-sample evaluation ─────────────────────────────────────────────────


async def eval_one(
    sample_sem: asyncio.Semaphore,
    qwen_client,
    qwen_sem,
    gemini_client,
    gemini_pool,
    writer: JsonlAppender,
    model_name: str,
    spec: dict,
    sample: dict,
    image_b64: str,
) -> tuple[str, bool]:
    sample_id = sample["sample_id"]
    qtype = sample["qtype"]
    family = sample.get("family")
    t0 = time.time()

    record: dict[str, Any] = {
        "sample_id": sample_id,
        "model": model_name,
        "qtype": qtype,
        "family": family,
        "status": "failed",
        "elapsed_s": None,
        "created_at": int(time.time()),
    }

    async with sample_sem:
        try:
            if qtype == "mc":
                user_text = build_mc_user_text(sample["question"], sample["options"])
                raw = await call_vl(
                    spec,
                    qwen_client,
                    qwen_sem,
                    gemini_client,
                    gemini_pool,
                    image_b64,
                    MC_SYSTEM,
                    user_text,
                    sample_id,
                    model_name,
                )
                parsed = parse_mc_label(raw, len(sample["options"]))
                gold = sample["correct_label"]
                record.update(
                    {
                        "status": "ok",
                        "raw_answer": raw,
                        "parsed_label": parsed,
                        "gold_label": gold,
                        "correct": (parsed == gold) if parsed else False,
                    }
                )
            elif qtype == "oe":
                user_text = build_oe_user_text(sample["question"])
                raw = await call_vl(
                    spec,
                    qwen_client,
                    qwen_sem,
                    gemini_client,
                    gemini_pool,
                    image_b64,
                    OE_SYSTEM,
                    user_text,
                    sample_id,
                    model_name,
                )
                gold = str(sample["answer"])
                judge = await judge_oe(
                    qwen_client,
                    qwen_sem,
                    sample["question"],
                    gold,
                    raw,
                    sample_id,
                    f"{model_name}-judge",
                )
                record.update(
                    {
                        "status": "ok",
                        "vl_answer": raw,
                        "gold": gold,
                        "judge_correct": judge.get("correct"),
                        "judge_reasoning": judge.get("reasoning"),
                        "judge_raw": judge.get("raw"),
                    }
                )
                if "parse_error" in judge:
                    record["judge_parse_error"] = judge["parse_error"]
            else:
                raise ValueError(f"unknown qtype {qtype!r}")
        except Exception as e:
            record["error"] = f"{type(e).__name__}: {e!r}"
            record["elapsed_s"] = round(time.time() - t0, 2)
            await writer.write(record)
            return sample_id, False

    record["elapsed_s"] = round(time.time() - t0, 2)
    await writer.write(record)
    return sample_id, True


# ── Per-model orchestration ───────────────────────────────────────────────


async def eval_model(
    model_name: str,
    samples: list[dict],
    image_cache: dict[str, str],
    args: argparse.Namespace,
    qwen_client,
    qwen_sem,
    gemini_client,
    gemini_pool,
) -> dict:
    if model_name not in MODELS:
        raise SystemExit(f"unknown model: {model_name}. Known: {sorted(MODELS)}")
    spec = MODELS[model_name]
    output_path = RESULTS_DIR / f"{model_name}.jsonl"

    if args.no_resume and output_path.exists():
        backup = output_path.with_suffix(output_path.suffix + ".bak")
        output_path.rename(backup)

    done = load_done_sample_ids(output_path)

    pending = [s for s in samples if s["sample_id"] not in done]
    if not pending:
        # Load and print results for already-done models
        _print_model_results(model_name)
        return {"model": model_name, "skipped": True}

    print(f"\n[{model_name}] evaluating {len(pending)}/{len(samples)} samples...")
    sample_sem = asyncio.Semaphore(args.parallel_samples)
    writer = JsonlAppender(output_path)

    t0 = time.time()
    tasks = [
        eval_one(
            sample_sem,
            qwen_client,
            qwen_sem,
            gemini_client,
            gemini_pool,
            writer,
            model_name,
            spec,
            s,
            image_cache[s["sample_id"]],
        )
        for s in pending
    ]
    results = await tqdm.gather(
        *tasks, total=len(tasks), desc=model_name, unit="sample"
    )
    n_ok = sum(1 for r in results if isinstance(r, tuple) and r[1])
    n_fail = len(results) - n_ok
    elapsed = time.time() - t0
    print(f"✓ {model_name}: {n_ok}/{len(pending)} ok in {elapsed:.1f}s")

    # Print detailed results
    _print_model_results(model_name)

    return {"model": model_name, "ok": n_ok, "failed": n_fail}


# ── Model results printing ───────────────────────────────────────────


def _print_model_results(model_name: str) -> None:
    """Load model results and print formatted accuracy report."""
    output_path = RESULTS_DIR / f"{model_name}.jsonl"
    if not output_path.exists():
        return

    per_sid: dict[str, dict] = {}
    with open(output_path, encoding="utf-8") as f:
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
                per_sid[sid] = rec

    ok_records = [r for r in per_sid.values() if r.get("status") == "ok"]
    if not ok_records:
        return

    mc_recs = [r for r in ok_records if r.get("qtype") == "mc"]
    oe_recs = [r for r in ok_records if r.get("qtype") == "oe"]
    mc_correct = sum(1 for r in mc_recs if r.get("correct"))
    oe_correct = sum(1 for r in oe_recs if r.get("judge_correct"))
    n_total = len(ok_records)
    n_correct = mc_correct + oe_correct

    # Per-family stats
    per_family: dict[str, dict] = {}
    for r in ok_records:
        fam = r.get("family", "unknown")
        d = per_family.setdefault(fam, {"n": 0, "correct": 0})
        d["n"] += 1
        if r.get("correct") or r.get("judge_correct"):
            d["correct"] += 1

    # Print results
    overall_acc = (n_correct / n_total * 100) if n_total else 0
    mc_acc = (mc_correct / len(mc_recs) * 100) if mc_recs else 0
    oe_acc = (oe_correct / len(oe_recs) * 100) if oe_recs else 0

    print(f"\n{'=' * 80}")
    print(f"Model: {model_name}")
    print(f"{'=' * 80}")
    print(f"\nOVERALL ACCURACY: {n_correct}/{n_total} = {overall_acc:.1f}%")
    if mc_recs:
        print(f"  MC (multiple choice): {mc_correct}/{len(mc_recs)} = {mc_acc:.1f}%")
    if oe_recs:
        print(f"  OE (open-ended):      {oe_correct}/{len(oe_recs)} = {oe_acc:.1f}%")

    print(f"\nPER-FAMILY ACCURACY:")
    print(f"{'-' * 80}")
    print(f"{'Family':<50s} {'Correct':>10s} {'Accuracy':>10s}")
    print(f"{'-' * 80}")

    for fam in sorted(per_family.keys()):
        d = per_family[fam]
        acc = (d["correct"] / d["n"] * 100) if d["n"] else 0
        print(f"{fam:<50s} {d['correct']:>4d}/{d['n']:<4d} {acc:>9.1f}%")

    print(f"{'-' * 80}\n")


# ── Summary ───────────────────────────────────────────────────────────────


def summarize() -> dict:
    out: dict = {"models": {}}
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        name = path.stem
        per_sid: dict[str, dict] = {}
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
                    per_sid[sid] = rec
        ok_records = [r for r in per_sid.values() if r.get("status") == "ok"]
        mc_recs = [r for r in ok_records if r.get("qtype") == "mc"]
        oe_recs = [r for r in ok_records if r.get("qtype") == "oe"]
        mc_correct = sum(1 for r in mc_recs if r.get("correct"))
        oe_correct = sum(1 for r in oe_recs if r.get("judge_correct"))
        n_mc, n_oe = len(mc_recs), len(oe_recs)
        n_total = n_mc + n_oe
        n_correct = mc_correct + oe_correct
        per_family: dict[str, dict] = {}
        for r in ok_records:
            fam = r.get("family", "?")
            d = per_family.setdefault(fam, {"n": 0, "correct": 0})
            d["n"] += 1
            if r.get("correct") or r.get("judge_correct"):
                d["correct"] += 1
        out["models"][name] = {
            "n_total": n_total,
            "n_correct": n_correct,
            "accuracy": round(n_correct / n_total, 4) if n_total else None,
            "mc": {
                "n": n_mc,
                "correct": mc_correct,
                "accuracy": round(mc_correct / n_mc, 4) if n_mc else None,
            },
            "oe": {
                "n": n_oe,
                "correct": oe_correct,
                "accuracy": round(oe_correct / n_oe, 4) if n_oe else None,
            },
            "by_family": {
                fam: {
                    "n": d["n"],
                    "correct": d["correct"],
                    "accuracy": round(d["correct"] / d["n"], 4) if d["n"] else None,
                }
                for fam, d in sorted(per_family.items())
            },
        }
    return out


# ── Main ──────────────────────────────────────────────────────────────────


async def main_async(args: argparse.Namespace) -> None:
    samples: list[dict] = json.loads(Path(args.dataset).read_text(encoding="utf-8"))
    samples.sort(key=lambda s: s["sample_id"])
    if args.limit > 0:
        samples = samples[: args.limit]
    print(f"loaded {len(samples)} samples from {args.dataset}", flush=True)

    # Pre-encode every image once.
    image_cache: dict[str, str] = {}
    for s in samples:
        sid = s["sample_id"]
        path = HERE / s["image_path"]
        image_cache[sid] = base64.b64encode(path.read_bytes()).decode("ascii")
    print(f"cached {len(image_cache)} images base64-encoded", flush=True)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = args.models or list(MODELS)
    for m in models:
        if m not in MODELS:
            raise SystemExit(f"unknown model: {m}. Known: {sorted(MODELS)}")
    print(f"will evaluate {len(models)} model(s): {models}", flush=True)

    qwen_sem = asyncio.Semaphore(args.qwen_concurrency)
    gemini_pool = GeminiKeyPool(GEMINI_API_KEYS, GEMINI_PER_KEY_LIMIT)
    timeout = httpx.Timeout(REQUEST_TIMEOUT, read=REQUEST_TIMEOUT, pool=60)
    qwen_limits = httpx.Limits(
        max_connections=args.qwen_concurrency * 2,
        max_keepalive_connections=args.qwen_concurrency,
    )
    gemini_total = GEMINI_PER_KEY_LIMIT * len(GEMINI_API_KEYS)
    gemini_limits = httpx.Limits(
        max_connections=gemini_total * 2,
        max_keepalive_connections=gemini_total,
    )

    async with (
        httpx.AsyncClient(
            timeout=timeout, limits=qwen_limits, trust_env=False
        ) as qwen_client,
        httpx.AsyncClient(
            timeout=timeout, limits=gemini_limits, trust_env=False
        ) as gemini_client,
    ):
        for m in models:
            await eval_model(
                m,
                samples,
                image_cache,
                args,
                qwen_client,
                qwen_sem,
                gemini_client,
                gemini_pool,
            )

    summary = summarize()
    SUMMARY_PATH.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"\n✓ Summary written to {SUMMARY_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-model VLM eval on the released dataset"
    )
    parser.add_argument("--dataset", default=str(DATASET_PATH))
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Subset of registered models to evaluate. Default = all.",
    )
    parser.add_argument(
        "--parallel-samples", type=int, default=DEFAULT_PARALLEL_SAMPLES
    )
    parser.add_argument("--qwen-concurrency", type=int, default=QWEN_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print(
            "\ninterrupted — partial results in eval_results/<model>.jsonl; rerun to resume.",
            flush=True,
        )
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
