#!/usr/bin/env python3
"""
eval_api.py — Multi-model VLM evaluation on the unified UnrealBench (633 samples).

Evaluation methodology copied verbatim from
  /mnt/tidal-alsh01/.../export_vlmevalkit/eval_gemini.py
i.e.:
  - VL inference with MathVision-style Hint embedded in the user text (NO system prompt).
  - MC (qtype=="mc"):       heuristic letter extraction → GPT fallback → random "Z".
  - OE/oe(mc):              Qwen verifier one-step (extract+judge), tail-3000 of CoT.
  - Empty VL response       → auto wrong (extracted_letter="Z", correct=False).

Model registry mirrors zerogen/eval_models_api.py (qwen + gemini for now;
extra GPT/Claude/etc. left commented out for the user to enable).

Phase 1 (VL inference) and Phase 2 (extract + judge) are both resume-aware:
re-running picks up where it left off. Each model writes to
  evaluation/eval_results/<model_slug>.jsonl

Usage:
  python evaluation/eval_api.py
  python evaluation/eval_api.py --models gemini-3.1-pro
  python evaluation/eval_api.py --models qwen3.6-plus-thinking gemini-3.1-pro
  python evaluation/eval_api.py --phase 1
  python evaluation/eval_api.py --phase 2 --rejudge
  python evaluation/eval_api.py --retry-empty
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import json
import random
import re
import string
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx

# ── paths ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_PATH = Path(
    "/mnt/tidal-alsh01/dataset/perceptionVLM/code_chenlifeng/new_benchmark/"
    "unrealbench/export_vlmevalkit/unified_benchmark.jsonl"
)
DEFAULT_IMAGE_DIR = REPO_ROOT / "release_all" / "images"
RESULTS_DIR = HERE / "eval_results"


# ── API constants (verbatim from eval_gemini.py) ─────────────────────────
GEMINI_ENDPOINT = "<GEMINI_ENDPOINT>"
GEMINI_KEYS = [
    os.environ.get("GEMINI_API_KEY_1", ""),
    os.environ.get("GEMINI_API_KEY_2", ""),
]

QWEN_ENDPOINT = "<QWEN_ENDPOINT>"
QWEN_KEYS = [
    os.environ.get("QWEN_API_KEY", ""),
    os.environ.get("QWEN_API_KEY_2", ""),
]

GENERIC_ENDPOINT = "https://api.gpt.ge"
GENERIC_KEY = os.environ.get("GENERIC_API_KEY", "")

QWEN_JUDGE_MODEL = "qwen3.6-plus"


# ── Model registry (aligned with zerogen/eval_models_api.py) ─────────────
# provider:  "qwen" (OpenAI-compat) | "gemini" (Google native) | "generic" (OpenAI-compat via api.gpt.ge)
# api_model: exact id sent to the provider
# thinking:  hint that VL inference may produce CoT (longer max tokens)
MODELS: dict[str, dict] = {
    # "qwen3-vl-8b-thinking":    {"provider": "qwen",   "api_model": "qwen3-vl-8b-thinking",  "thinking": True},
    # "qwen3-vl-32b-thinking":   {"provider": "qwen",   "api_model": "qwen3-vl-32b-thinking", "thinking": True},
    "qwen3-vl-235b-a22b-thinking": {"provider": "qwen", "api_model": "qwen3-vl-235b-a22b-thinking", "thinking": True},
}


# ── concurrency / token budgets (aligned with eval_gemini.py) ────────────
VL_CONCURRENCY = 64
QWEN_CONCURRENCY = 64
REQUEST_TIMEOUT = 60.0
REQUEST_TIMEOUT_VL = 60.0
MAX_RETRIES = 6

MAX_OUTPUT_TOKENS_GEMINI = 16384
MAX_OUTPUT_TOKENS_QWEN = 8192
MAX_OUTPUT_TOKENS_GENERIC = 8192
MAX_OUTPUT_TOKENS_EXTRACT = 256
MAX_OUTPUT_TOKENS_VERIFIER = 512


# ── verifier system prompt (verbatim from eval_gemini.py) ────────────────
VERIFIER_SYSTEM_EN = (
    "You are a strict grader for visual reasoning answers. "
    "Decide whether a candidate answer matches the gold answer. "
    "Compare ONLY the candidate's final answer — IGNORE any errors in reasoning steps. "
    "A candidate is CORRECT iff it conveys exactly the same information as the gold "
    "(synonyms, equivalent numerals, equivalent units, equivalent mathematical expressions are OK; "
    "missing parts, extra incorrect information, off-by-one numbers, or contradictions are NOT OK). "
    "If the candidate's answer is invalid (incomplete, repetitive, refusal to answer, or irrelevant), output false. "
    "Output STRICT JSON ONLY (no prose outside the JSON, no markdown fences):\n"
    '{"correct": <true|false>, "reasoning": "<one short sentence>"}'
)


# ── helpers ──────────────────────────────────────────────────────────────


def read_jsonl(path: Path) -> list[dict]:
    out = []
    if not path.exists():
        return out
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


def load_image_b64(image_dir: Path, filename: str) -> str | None:
    p = image_dir / filename
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def extract_json_strict(text: str) -> Any:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("no {", text, 0)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise json.JSONDecodeError("unbalanced", text, 0)


class QwenKeyPool:
    """Per-key semaphore pool, used for both VL Qwen calls and text judge/extract."""

    def __init__(self, keys: list[str], per_key_limit: int):
        self._keys = keys
        self._sems = [asyncio.Semaphore(per_key_limit) for _ in keys]

    @asynccontextmanager
    async def use_key(self):
        while True:
            for i, sem in enumerate(self._sems):
                if not sem.locked():
                    async with sem:
                        yield self._keys[i]
                    return
            await asyncio.sleep(0.1)


# ── Question formatting with MathVision-style Hint (verbatim) ────────────


def _format_options(options):
    if not options:
        return ""
    if isinstance(options, dict):
        return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    if isinstance(options, list):
        lines = []
        for opt in options:
            label = opt.get("label", "?")
            parts = opt.get("parts", [])
            lines.append(f"{label}. " + " / ".join(parts))
        return "\n".join(lines)
    return ""


def format_question_with_hint(record: dict) -> str:
    family = record.get("family", "")
    qtype = record.get("qtype", "oe")
    question = record.get("question", "")
    options = record.get("options")

    is_phys_bio_chem_zg = family in ("PHYS", "BIO", "CHEM", "ZEROGen")
    is_mc = qtype == "mc"

    if is_mc:
        opts_text = _format_options(options)
        if is_phys_bio_chem_zg:
            hint = (
                "Hint: Please reason through the question and provide ONLY the single "
                "capital letter of the correct option at the end."
            )
        else:
            hint = (
                "Hint: Please reason through the question and provide only the letter "
                "of the correct option at the end."
            )
        return f"{hint}\nQuestion: {question}\n\nOptions:\n{opts_text}"

    if is_phys_bio_chem_zg:
        hint = (
            "Hint: Please reason through the question and provide the final answer "
            "in at most 4 words at the end."
        )
    elif family == "MATH":
        hint = (
            "Hint: Please reason through the question and provide only the final "
            "answer at the end."
        )
    else:
        hint = (
            "Hint: Please reason through the question and provide the final "
            "answer concisely at the end."
        )
    return f"{hint}\nQuestion: {question}"


# ── MC extraction: heuristic + GPT fallback (verbatim from eval_gemini.py)


def can_infer_option(answer: str, choices):
    if "Failed to obtain answer via API" in answer:
        return False
    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    def count_choice(splits, choices, prefix="", suffix=""):
        cnt = 0
        for c in choices:
            if prefix + c + suffix in splits:
                cnt += 1
        return cnt

    answer_mod = copy.copy(answer)
    chars = ".()[],:;!*#{}"
    for c in chars:
        answer_mod = answer_mod.replace(c, " ")
    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)
    if count == 1:
        for ch in choices:
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    return False


def can_infer_text(answer: str, choices: dict):
    answer = answer.lower()
    if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = []
    for k in choices:
        if choices[k] in answer:
            cands.append(k)
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def build_extract_prompt(question, options_text, prediction):
    return (
        "You are an AI assistant who will help me to match "
        "an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, "
        "and you need to find which option is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output Z. "
        "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z.\n"
        "Example 1:\n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: a cute teddy bear\nYour output: A\n"
        "Example 2:\n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: Spider\nYour output: Z\n"
        "Example 3:\n"
        f"Question: {question}\nOptions: {options_text}\n"
        f"Answer: {prediction}\nYour output: "
    )


def build_choices_dict(options) -> dict[str, str]:
    if isinstance(options, dict):
        return {k: str(v) for k, v in options.items()}
    if isinstance(options, list):
        return {
            opt.get("label", "?"): " / ".join(opt.get("parts", [])) for opt in options
        }
    return {}


# ── Qwen text call (for VL Qwen models AND text judge/extract) ───────────


async def call_qwen_text(
    client: httpx.AsyncClient,
    key_pool: QwenKeyPool,
    system_prompt: str,
    user_text: str,
    max_tokens: int = 512,
) -> str:
    url = f"{QWEN_ENDPOINT.rstrip('/')}/chat/completions"
    payload = {
        "model": QWEN_JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": "/no_think\n" + system_prompt},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    for attempt in range(MAX_RETRIES):
        try:
            async with key_pool.use_key() as api_key:
                resp = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
            body = resp.json()
            if "choices" not in body or not body["choices"]:
                raise RuntimeError(f"no choices: {str(body)[:200]}")
            return body["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            wait = (
                min(5 * (attempt + 1), 30)
                if e.response.status_code == 429
                else min(2**attempt, 3)
            )
            await asyncio.sleep(wait)
        except Exception:
            await asyncio.sleep(min(2**attempt, 10))
    return ""


async def extract_answer_from_item(
    client: httpx.AsyncClient,
    key_pool: QwenKeyPool,
    record: dict,
    response: str,
) -> str:
    """MC extraction: heuristic → GPT fallback → random."""
    question = record.get("question", "")
    options = record.get("options")
    choices = build_choices_dict(options)
    option_str = _format_options(options)

    ret = can_infer(response, choices)
    if ret:
        return ret

    prompt = build_extract_prompt(question, option_str, response)
    extract_system = (
        "/no_think\n"
        "You are an AI assistant that matches an answer to multiple-choice options. "
        "Output ONLY a single uppercase letter (A-F) or Z."
    )
    ans = await call_qwen_text(
        client, key_pool, extract_system, prompt, max_tokens=MAX_OUTPUT_TOKENS_EXTRACT
    )
    # call_qwen_text adds "/no_think\n" once; we already prefixed extract_system so
    # the actual content reaching the API is "/no_think\n/no_think\n..." — same as
    # eval_gemini.py, kept verbatim to preserve behaviour.
    ret = can_infer(ans, choices)
    if ret:
        return ret

    all_choices = list(choices.keys()) + ["Z"]
    return random.choice(all_choices)


async def judge_oe_verifier(
    client: httpx.AsyncClient,
    key_pool: QwenKeyPool,
    question: str,
    gold: str,
    response: str,
) -> dict:
    """OE/oe(mc) one-step verifier — sends TAIL of response (final answer at end of CoT)."""
    resp_excerpt = response[-3000:] if len(response) > 3000 else response
    user_text = (
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Candidate response:\n{resp_excerpt}\n\n"
        "Output the JSON judgment."
    )
    raw = await call_qwen_text(
        client, key_pool, VERIFIER_SYSTEM_EN, user_text,
        max_tokens=MAX_OUTPUT_TOKENS_VERIFIER,
    )
    if not raw:
        return {"correct": None, "verifier_raw": raw, "judge_error": "empty verifier response"}
    try:
        data = extract_json_strict(raw)
    except Exception as e:
        return {"correct": None, "verifier_raw": raw, "judge_error": str(e)}
    cor = data.get("correct")
    if isinstance(cor, str):
        cor = cor.strip().lower() in ("true", "yes", "y", "1")
    return {
        "correct": bool(cor) if cor is not None else None,
        "verifier_raw": raw,
        "verifier_reasoning": data.get("reasoning"),
    }


# ── VL inference dispatchers ─────────────────────────────────────────────


def _gemini_text(body: dict) -> str:
    cands = body.get("candidates", [])
    if not cands:
        return ""
    parts = cands[0].get("content", {}).get("parts", [])
    return "".join(
        p.get("text", "") for p in parts if "text" in p and not p.get("thought")
    ).strip()


async def vl_infer(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    record: dict,
    spec: dict,
    image_b64: str,
    image_filename: str,
    key_idx: int,
) -> tuple[str, str | None, str | None]:
    """Returns (id, response_text_or_none, error_or_none)."""
    question = format_question_with_hint(record)
    provider = spec["provider"]
    api_model = spec["api_model"]
    thinking = spec.get("thinking", False)

    ext = image_filename.rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"

    if provider == "gemini":
        max_tok = MAX_OUTPUT_TOKENS_GEMINI
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"inlineData": {"mimeType": mime, "data": image_b64}},
                        {"text": question},
                    ],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": max_tok,
                "topP": 0.95,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "api-key": GEMINI_KEYS[key_idx % len(GEMINI_KEYS)],
        }
        endpoint = GEMINI_ENDPOINT
        is_openai_shape = False
    elif provider == "generic":
        max_tok = MAX_OUTPUT_TOKENS_GENERIC
        payload = {
            "model": api_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": max_tok,
            "temperature": 0.0,
            "enable_thinking": True,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
        }
        if "gpt" in api_model.lower():
            payload["temperature"] = 1.0
            payload["max_completion_tokens"] = max_tok
            payload.pop("max_tokens", None)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GENERIC_KEY}",
        }
        endpoint = f"{GENERIC_ENDPOINT.rstrip('/')}/v1/chat/completions"
        is_openai_shape = True
    elif provider == "qwen":
        max_tok = MAX_OUTPUT_TOKENS_QWEN if thinking else 1024
        max_tok = max(max_tok, 8192) if thinking else max_tok
        payload = {
            "model": api_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "max_tokens": max_tok,
            "temperature": 0.0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {QWEN_KEYS[key_idx % len(QWEN_KEYS)]}",
        }
        endpoint = f"{QWEN_ENDPOINT.rstrip('/')}/chat/completions"
        is_openai_shape = True
    else:
        return record["id"], None, f"unknown provider {provider!r}"

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    endpoint, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_VL
                )
                if not is_openai_shape:  # gemini
                    if resp.status_code != 200:
                        body = (resp.text or "")[:300]
                        raise RuntimeError(f"HTTP {resp.status_code}: {body}")
                    return record["id"], _gemini_text(resp.json()), None
                resp.raise_for_status()
                return (
                    record["id"],
                    resp.json()["choices"][0]["message"]["content"],
                    None,
                )
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return record["id"], None, f"{type(e).__name__}: {str(e)[:200]}"
                await asyncio.sleep(min(2**attempt, 3))


# ── Phase 1: VL inference per model ──────────────────────────────────────


async def phase1_vl(
    model_name: str,
    spec: dict,
    valid_recs: list[dict],
    image_dir: Path,
    results_path: Path,
    retry_empty: bool,
) -> None:
    existing = read_jsonl(results_path)
    done_ids = {r["id"] for r in existing if r.get("vl_response") or r.get("vl_error")}
    empty_ids = {
        r["id"] for r in existing
        if not r.get("vl_response") and not r.get("vl_error")
    }
    if retry_empty:
        print(f"  --retry-empty: {len(empty_ids)} empty responses will be re-inferred")
        done_ids -= empty_ids
    pending = [r for r in valid_recs if r["id"] not in done_ids]
    print(f"[Phase 1] {model_name}: {len(pending)} pending, {len(done_ids)} done")
    if not pending:
        return

    sem = asyncio.Semaphore(VL_CONCURRENCY)
    done_count = [0]
    lock = asyncio.Lock()

    is_generic_provider = spec["provider"] == "generic"

    async def infer(client, idx, rec):
        img_b64 = load_image_b64(image_dir, rec["image"])
        if not img_b64:
            result = {
                "id": rec["id"], "vl_response": None, "vl_error": "NO_IMAGE",
                "gold": rec.get("answer", ""), "qtype": rec.get("qtype", "oe"),
                "family": rec.get("family", ""), "domain": rec.get("domain", ""),
                "subdomain": rec.get("subdomain", ""),
                "correct": None, "extracted_letter": None,
                "verifier_raw": None, "verifier_reasoning": None,
            }
        else:
            rid, response, error = await vl_infer(
                client, sem, rec, spec, img_b64, rec["image"], idx
            )
            result = {
                "id": rid, "vl_response": response, "vl_error": error,
                "gold": rec.get("answer", ""), "qtype": rec.get("qtype", "oe"),
                "family": rec.get("family", ""), "domain": rec.get("domain", ""),
                "subdomain": rec.get("subdomain", ""),
                "correct": None, "extracted_letter": None,
                "verifier_raw": None, "verifier_reasoning": None,
            }
        async with lock:
            done_count[0] += 1
            if done_count[0] % 20 == 0:
                print(f"    {done_count[0]}/{len(pending)}", flush=True)
        with open(results_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT_VL,
        trust_env=is_generic_provider,
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
    ) as client:
        for i in range(0, len(pending), 50):
            batch = pending[i : i + 50]
            await asyncio.gather(
                *[infer(client, i + j, item) for j, item in enumerate(batch)]
            )
    print(f"  done {len(pending)}")


# ── Phase 2: extract (MC) + judge (OE/oe(mc)) ────────────────────────────


def _dedup_results(results: list[dict]) -> list[dict]:
    seen = {}
    for r in results:
        seen[r["id"]] = r
    return list(seen.values())


async def phase2_judge(
    model_name: str,
    valid_recs: list[dict],
    results_path: Path,
    rejudge: bool,
) -> None:
    results = _dedup_results(read_jsonl(results_path))
    src_map = {r["id"]: r for r in valid_recs}

    empty_count = 0
    for r in results:
        if r.get("correct") is not None and not rejudge:
            continue
        if not r.get("vl_response") and not r.get("vl_error"):
            r["extracted_letter"] = "Z"
            r["correct"] = False
            empty_count += 1

    mc_pending = [
        r for r in results
        if r.get("qtype") == "mc"
        and r.get("vl_response")
        and (rejudge or r.get("correct") is None)
    ]
    oe_pending = [
        r for r in results
        if r.get("qtype") in ("oe", "oe(mc)")
        and r.get("vl_response")
        and (rejudge or r.get("correct") is None)
    ]
    print(f"[Phase 2] {model_name}: {empty_count} empty, {len(mc_pending)} MC + {len(oe_pending)} OE/oe(mc)")
    if not mc_pending and not oe_pending and empty_count == 0:
        return

    kp = QwenKeyPool(QWEN_KEYS, QWEN_CONCURRENCY // max(len(QWEN_KEYS), 1))

    async with httpx.AsyncClient(
        timeout=REQUEST_TIMEOUT,
        trust_env=False,
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
    ) as client:
        if mc_pending:
            done_mc = [0]

            async def mc_worker(rec):
                src = src_map.get(rec["id"], rec)
                letter = await extract_answer_from_item(client, kp, src, rec["vl_response"])
                rec["extracted_letter"] = letter
                if letter and letter != "Z":
                    gold_letter = (rec.get("gold") or "").strip().upper()
                    rec["correct"] = letter.upper() == gold_letter
                else:
                    rec["correct"] = False
                done_mc[0] += 1
                if done_mc[0] % 20 == 0:
                    print(f"    MC {done_mc[0]}/{len(mc_pending)}", flush=True)

            for i in range(0, len(mc_pending), 100):
                batch = mc_pending[i : i + 100]
                await asyncio.gather(*[mc_worker(r) for r in batch])
            print(f"  MC done: {len(mc_pending)}")

        if oe_pending:
            done_oe = [0]

            async def oe_worker(rec):
                src = src_map.get(rec["id"], rec)
                # clear stale MC artifact (oe(mc) may have been routed to MC earlier)
                if rec.get("qtype") == "oe(mc)":
                    rec["extracted_letter"] = None
                result = await judge_oe_verifier(
                    client, kp,
                    src.get("question", ""),
                    rec.get("gold", ""),
                    rec["vl_response"],
                )
                rec["correct"] = result.get("correct")
                rec["verifier_raw"] = result.get("verifier_raw")
                rec["verifier_reasoning"] = result.get("verifier_reasoning")
                if result.get("judge_error"):
                    rec["judge_error"] = result["judge_error"]
                done_oe[0] += 1
                if done_oe[0] % 20 == 0:
                    print(f"    OE {done_oe[0]}/{len(oe_pending)}", flush=True)

            for i in range(0, len(oe_pending), 100):
                batch = oe_pending[i : i + 100]
                await asyncio.gather(*[oe_worker(r) for r in batch])
            print(f"  OE done: {len(oe_pending)}")

    with open(results_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ── Phase 3: stats ───────────────────────────────────────────────────────


def phase3_stats(model_name: str, results_path: Path) -> None:
    results = _dedup_results(read_jsonl(results_path))
    if not results:
        return

    errors = sum(1 for r in results if r.get("vl_error"))
    total = len(results)
    valid_count = total - errors
    correct_count = sum(
        1 for r in results if r.get("correct") and not r.get("vl_error")
    )
    acc = correct_count / max(valid_count, 1) * 100

    print(f"\n{'=' * 70}")
    print(f"  {model_name}: {acc:.1f}% ({correct_count}/{valid_count})  errors={errors}")
    print(f"{'=' * 70}")

    # qtype breakdown
    print("  By qtype:")
    for q in ("mc", "oe", "oe(mc)"):
        items = [r for r in results if r.get("qtype") == q and not r.get("vl_error")]
        if not items:
            continue
        ok = sum(1 for r in items if r.get("correct"))
        print(f"    {q:8s}: {ok / len(items) * 100:5.1f}% ({ok}/{len(items)})")

    # family × qtype breakdown
    print("  By family × qtype:")
    fam_q: dict = {}
    for r in results:
        if r.get("vl_error"):
            continue
        key = (r.get("family", "?"), r.get("qtype", "?"))
        d = fam_q.setdefault(key, [0, 0])
        d[0] += 1
        if r.get("correct"):
            d[1] += 1
    for key in sorted(fam_q.keys()):
        n, ok = fam_q[key]
        bar = "█" * int(ok / n * 20)
        print(f"    {key[0]:8s} {key[1]:8s}: {ok / n * 100:5.1f}% ({ok}/{n}) {bar}")

    # subdomain breakdown
    print("  By subdomain:")
    sd: dict = {}
    for r in results:
        if r.get("vl_error"):
            continue
        key = r.get("subdomain", "?")
        d = sd.setdefault(key, [0, 0])
        d[0] += 1
        if r.get("correct"):
            d[1] += 1
    for key in sorted(sd.keys(), key=lambda k: -sd[k][0]):
        n, ok = sd[key]
        bar = "█" * int(ok / n * 20)
        print(f"    {key:35s}: {ok / n * 100:5.1f}% ({ok}/{n}) {bar}")


# ── main ─────────────────────────────────────────────────────────────────


async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of registered models. Default = all in MODELS dict.",
    )
    parser.add_argument(
        "--data", default=str(DATA_PATH),
        help="Path to unified_benchmark.jsonl",
    )
    parser.add_argument(
        "--image-dir", default=str(DEFAULT_IMAGE_DIR),
        help="Directory containing images named <record.image>",
    )
    parser.add_argument(
        "--results-dir", default=str(RESULTS_DIR),
        help="Output directory for eval_<model>.jsonl",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=0,
                        help="0 = run all phases (default)")
    parser.add_argument("--rejudge", action="store_true",
                        help="Re-run Phase 2 judging on all rows (overwrite correct).")
    parser.add_argument("--retry-empty", action="store_true",
                        help="Re-run Phase 1 for rows with empty vl_response.")
    args = parser.parse_args()

    run_all = args.phase == 0
    data_path = Path(args.data)
    image_dir = Path(args.image_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data:       {data_path}")
    print(f"Images:     {image_dir}")
    print(f"Results:    {results_dir}")

    all_recs = read_jsonl(data_path)
    print(f"Loaded {len(all_recs)} records")
    if args.limit > 0:
        all_recs = all_recs[: args.limit]
        print(f"  --limit {args.limit}")

    # filter to records whose image actually exists locally
    valid = [r for r in all_recs if load_image_b64(image_dir, r.get("image", ""))]
    skipped = len(all_recs) - len(valid)
    print(f"  {len(valid)} with image  (skipped {skipped} missing)")

    model_names = args.models or list(MODELS)
    for m in model_names:
        if m not in MODELS:
            raise SystemExit(f"unknown model {m!r}. known: {sorted(MODELS)}")
    print(f"Will evaluate {len(model_names)} model(s): {model_names}\n")

    for model_name in model_names:
        spec = MODELS[model_name]
        slug = model_name.replace("/", "_")
        results_path = results_dir / f"eval_{slug}.jsonl"
        print(f"\n{'#' * 70}\n# {model_name}  ({spec['provider']}/{spec['api_model']})\n{'#' * 70}")

        t0 = time.time()
        if run_all or args.phase == 1:
            await phase1_vl(model_name, spec, valid, image_dir, results_path, args.retry_empty)
        if run_all or args.phase == 2:
            await phase2_judge(model_name, valid, results_path, args.rejudge)
        if run_all or args.phase == 3:
            phase3_stats(model_name, results_path)
        print(f"  [{model_name}] elapsed: {time.time() - t0:.1f}s")


def main() -> None:
    try:
        asyncio.run(async_main())
    except KeyboardInterrupt:
        print("\ninterrupted — rerun to resume (results are append-only).", flush=True)


if __name__ == "__main__":
    main()
