#!/usr/bin/env python3
"""
eval_vllm_v2.py — Multi-model VLM evaluation via in-process vLLM batched chat
on the unified UnrealBench (633 samples).

Same on-disk output schema as evaluation/eval_vllm.py (one record per sample
with id / vl_response / extracted_letter / correct / verifier_raw / ...),
but with the per-sample async retry/judge pipeline from
zerogen/eval_models_vllm.py — i.e.:
  * robust call_gemini_text with differentiated waits for network / 429 /
    HTTP-{400,403} / other HTTP / other Exception (mirrors call_qwen);
  * per-sample async handle() inside an `atqdm.gather`;
  * each finalised record append-written under a `write_lock` (resume-safe);
  * re-runs skip any sample id whose record on disk is already "done".

Pipeline:
  1) vLLM batched chat (no system prompt, MathVision-style Hint in user text).
  2) Per sample: extract (MC) / judge (OE/oe(mc)) via Gemini; append to
     evaluation/eval_results/eval_<slug>.jsonl.
  3) Print per-model stats (overall / qtype / family×qtype / subdomain).

Usage:
  python evaluation/eval_vllm_v2.py
  python evaluation/eval_vllm_v2.py --models InternVL3_5-8B --limit 5
  python evaluation/eval_vllm_v2.py --no-resume
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import gc
import json
import random
import re
import string
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from tqdm.asyncio import tqdm as atqdm


# ── paths ────────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DATA_PATH = Path(
    "/mnt/tidal-alsh01/dataset/perceptionVLM/code_chenlifeng/new_benchmark/"
    "unrealbench/export_vlmevalkit/unified_benchmark.jsonl"
)
DEFAULT_IMAGE_DIR = REPO_ROOT / "release_all" / "images"
RESULTS_DIR = HERE / "eval_results"


# ── Gemini text-API constants (judge: MC fallback + OE verifier) ─────────
GEMINI_ENDPOINT = "<GEMINI_ENDPOINT>"
GEMINI_KEYS = [
    os.environ.get("GEMINI_API_KEY_1", ""),
    os.environ.get("GEMINI_API_KEY_2", ""),
]
GEMINI_JUDGE_MODEL = "gemini-3.1-pro-preview"

JUDGE_CONCURRENCY = 128         # split across keys → ~64 in-flight per key
REQUEST_TIMEOUT = 1200.0
MAX_RETRIES = 6

MAX_OUTPUT_TOKENS_EXTRACT = 256
MAX_OUTPUT_TOKENS_VERIFIER = 512


# ── vLLM defaults ────────────────────────────────────────────────────────
VLLM_DEFAULT_GPUS = 8
VLLM_DEFAULT_GPU_MEM = 0.9
VLLM_DEFAULT_MAX_MODEL_LEN = 8192
VL_TEMPERATURE = 0.0
VL_MAX_TOKENS = 8192


# ── Model registry (aligned with zerogen/eval_models_vllm.py) ────────────
MODELS: dict[str, dict] = {
    # "Qwen3.5-4B": {
    #     "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3.5-4B",
    # },
    # "Qwen3.5-9B": {
    #     "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3.5-9B",
    # },
    # "Qwen3.5-27B": {
    #     "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3.5-27B",
    # },
    "Ours-4B": {
        "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/checkpoints/qwen3vl4b/qwen3vl4b_multinode_grpo_rubric_general/hf_model_step173/",
    },
    "Ours-8B": {
        "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/checkpoints/qwen3vl8b/stem_final_grpo_rubric_general/hf_model_step524/",
    },
    "Ours-32B": {
        "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/checkpoints/qwen3vl32b/stem_final_grpo_rubric_general/hf_model_step173/",
    },
}


# ── Verifier system prompt (verbatim from eval_gemini.py) ────────────────
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


def append_jsonl(path: Path, record: dict) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_ids(path: Path) -> set[str]:
    """Skip ids whose record is already final on disk.

    A record is "done" if it has a vl_error set, OR vl_response present AND
    correct already judged. Partial records (vl_response but correct=None)
    are NOT considered done — they get re-judged.
    """
    by_id: dict[str, dict] = {}
    for rec in read_jsonl(path):
        rid = rec.get("id")
        if rid:
            by_id[rid] = rec
    done: set[str] = set()
    for rid, rec in by_id.items():
        if rec.get("vl_error"):
            done.add(rid)
        elif rec.get("vl_response") and rec.get("correct") is not None:
            done.add(rid)
    return done


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


class GeminiKeyPool:
    """Round-robin Gemini key pool, per-key semaphore."""

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


# ── MC extraction: heuristic + GPT fallback (verbatim from eval_gemini.py) ──


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


# ── Gemini text call with eval_models_vllm.py-style robust retry ─────────

_RETRYABLE = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ProxyError,
)


async def call_gemini_text(
    client: httpx.AsyncClient,
    key_pool: GeminiKeyPool,
    system_prompt: str,
    user_text: str,
    max_tokens: int,
    sample_id: str,
    label: str,
) -> str:
    """Gemini text-only call (Google native generateContent format) with full
    retry. Mirrors call_qwen() in eval_models_vllm.py: differentiated waits
    for network / 429 / HTTP-{400,403} / other HTTP / other Exception.

    Thinking is disabled (thinkingBudget=0) so the judge returns JSON fast.
    """
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "temperature": 0.0,
            "maxOutputTokens": max_tokens,
            "topP": 0.95,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with key_pool.use_key() as api_key:
                resp = await client.post(
                    GEMINI_ENDPOINT,
                    headers={
                        "api-key": api_key,
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
                body = resp.json()
            cands = body.get("candidates", [])
            if not cands:
                raise RuntimeError(
                    f"gemini response missing candidates: {str(body)[:200]}"
                )
            parts = cands[0].get("content", {}).get("parts", []) or []
            text = "".join(
                p.get("text", "")
                for p in parts
                if "text" in p and not p.get("thought")
            ).strip()
            if not text:
                raise RuntimeError(f"gemini response empty text: {str(body)[:200]}")
            return text
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
                wait = min(15 * (attempt + 1), 90)
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


async def extract_answer_from_item(
    client: httpx.AsyncClient,
    key_pool: GeminiKeyPool,
    record: dict,
    response: str,
    sample_id: str,
    label: str,
) -> str:
    question = record.get("question", "")
    options = record.get("options")
    choices = build_choices_dict(options)
    option_str = _format_options(options)

    ret = can_infer(response, choices)
    if ret:
        return ret

    prompt = build_extract_prompt(question, option_str, response)
    extract_system = (
        "You are an AI assistant that matches an answer to multiple-choice options. "
        "Output ONLY a single uppercase letter (A-F) or Z."
    )
    try:
        ans = await call_gemini_text(
            client, key_pool, extract_system, prompt,
            max_tokens=MAX_OUTPUT_TOKENS_EXTRACT,
            sample_id=sample_id, label=f"{label}-extract",
        )
    except Exception:
        ans = ""
    ret = can_infer(ans, choices)
    if ret:
        return ret

    all_choices = list(choices.keys()) + ["Z"]
    return random.choice(all_choices)


async def judge_oe_verifier(
    client: httpx.AsyncClient,
    key_pool: GeminiKeyPool,
    question: str,
    gold: str,
    response: str,
    sample_id: str,
    label: str,
) -> dict:
    resp_excerpt = response[-3000:] if len(response) > 3000 else response
    user_text = (
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Candidate response:\n{resp_excerpt}\n\n"
        "Output the JSON judgment."
    )
    try:
        raw = await call_gemini_text(
            client, key_pool, VERIFIER_SYSTEM_EN, user_text,
            max_tokens=MAX_OUTPUT_TOKENS_VERIFIER,
            sample_id=sample_id, label=f"{label}-verifier",
        )
    except Exception as e:
        return {
            "correct": None, "verifier_raw": None,
            "judge_error": f"{type(e).__name__}: {e!r}",
        }
    try:
        data = extract_json_strict(raw)
    except Exception as e:
        return {
            "correct": None, "verifier_raw": raw,
            "judge_error": f"{type(e).__name__}: {e!r}",
        }
    cor = data.get("correct")
    if isinstance(cor, str):
        cor = cor.strip().lower() in ("true", "yes", "y", "1")
    return {
        "correct": bool(cor) if cor is not None else None,
        "verifier_raw": raw,
        "verifier_reasoning": data.get("reasoning"),
    }


# ── vLLM batched inference ───────────────────────────────────────────────


def build_chat_messages(record: dict, image_url: str) -> list[dict]:
    """No system prompt — Hint is embedded in the user text (eval_gemini.py style)."""
    user_text = format_question_with_hint(record)
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def run_vllm_batch(
    model_path: str,
    pending_recs: list[dict],
    image_urls: dict[str, str],
    args: argparse.Namespace,
) -> dict[str, str]:
    """Load vLLM model, run batched chat, return {id: raw_text}."""
    from vllm import LLM, SamplingParams

    print(f"  loading vLLM model: {model_path}", flush=True)
    t0 = time.time()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.vllm_gpus,
        gpu_memory_utilization=args.vllm_gpu_mem,
        trust_remote_code=True,
        max_model_len=args.vllm_max_model_len,
        limit_mm_per_prompt={"image": 1},
        dtype="auto",
    )
    print(f"  vLLM loaded in {time.time() - t0:.1f}s", flush=True)

    sampling = SamplingParams(temperature=VL_TEMPERATURE, max_tokens=args.vl_max_tokens)

    sample_ids = [r["id"] for r in pending_recs]
    conversations = [build_chat_messages(r, image_urls[r["id"]]) for r in pending_recs]

    print(f"  running batched chat on {len(conversations)} samples ...", flush=True)
    t0 = time.time()
    outputs = llm.chat(conversations, sampling_params=sampling, use_tqdm=True)
    print(f"  vLLM inference done in {time.time() - t0:.1f}s", flush=True)

    raw_by_id: dict[str, str] = {}
    for sid, out in zip(sample_ids, outputs):
        text = out.outputs[0].text if out.outputs else ""
        raw_by_id[sid] = text.strip()

    del llm, outputs
    try:
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return raw_by_id


# ── Per-sample async judge + append (eval_models_vllm.py pattern) ────────


async def judge_and_write(
    pending_recs: list[dict],
    raw_by_id: dict[str, str],
    results_path: Path,
    model_name: str,
) -> tuple[int, int]:
    kp = GeminiKeyPool(GEMINI_KEYS, JUDGE_CONCURRENCY // max(len(GEMINI_KEYS), 1))
    limits = httpx.Limits(
        max_connections=JUDGE_CONCURRENCY * 2,
        max_keepalive_connections=JUDGE_CONCURRENCY,
    )
    timeout = httpx.Timeout(REQUEST_TIMEOUT, read=REQUEST_TIMEOUT, pool=60)

    n_ok = 0
    n_fail = 0
    write_lock = asyncio.Lock()

    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, trust_env=False
    ) as client:

        async def handle(rec: dict) -> bool:
            nonlocal n_ok, n_fail
            sid = rec["id"]
            qtype = rec.get("qtype", "oe")
            t0 = time.time()
            out: dict[str, Any] = {
                "id": sid,
                "vl_response": None,
                "vl_error": None,
                "gold": rec.get("answer", ""),
                "qtype": qtype,
                "family": rec.get("family", ""),
                "domain": rec.get("domain", ""),
                "subdomain": rec.get("subdomain", ""),
                "correct": None,
                "extracted_letter": None,
                "verifier_raw": None,
                "verifier_reasoning": None,
            }
            ok = False
            try:
                raw = raw_by_id.get(sid)
                if raw is None:
                    # vLLM produced no output object for this id.
                    out["vl_error"] = "MISSING_VLLM_OUTPUT"
                    out["correct"] = False
                    out["extracted_letter"] = "Z"
                elif not raw:
                    # Empty VL response → auto wrong (matches eval_gemini.py).
                    out["vl_error"] = "EMPTY"
                    out["correct"] = False
                    out["extracted_letter"] = "Z"
                else:
                    out["vl_response"] = raw
                    if qtype == "mc":
                        letter = await extract_answer_from_item(
                            client, kp, rec, raw, sid, model_name
                        )
                        out["extracted_letter"] = letter
                        if letter and letter != "Z":
                            gold_letter = (rec.get("answer") or "").strip().upper()
                            out["correct"] = letter.upper() == gold_letter
                        else:
                            out["correct"] = False
                    elif qtype in ("oe", "oe(mc)"):
                        judge = await judge_oe_verifier(
                            client, kp,
                            rec.get("question", ""),
                            rec.get("answer", ""),
                            raw,
                            sid, model_name,
                        )
                        out["correct"] = judge.get("correct")
                        out["verifier_raw"] = judge.get("verifier_raw")
                        out["verifier_reasoning"] = judge.get("verifier_reasoning")
                        if judge.get("judge_error"):
                            out["judge_error"] = judge["judge_error"]
                    else:
                        out["vl_error"] = f"UNKNOWN_QTYPE:{qtype}"
                        out["correct"] = False
                ok = True
            except Exception as e:
                out["judge_error"] = f"{type(e).__name__}: {e!r}"
            out["elapsed_s"] = round(time.time() - t0, 2)
            async with write_lock:
                append_jsonl(results_path, out)
            if ok:
                n_ok += 1
            else:
                n_fail += 1
            return ok

        tasks = [handle(r) for r in pending_recs]
        await atqdm.gather(*tasks, total=len(tasks), desc=f"{model_name}-judge", unit="s")

    return n_ok, n_fail


# ── Stats ────────────────────────────────────────────────────────────────


def _dedup_results(results: list[dict]) -> list[dict]:
    """Latest-wins by id — Phase 2 reruns will produce duplicate lines and we
    want the most recent record per id."""
    seen: dict[str, dict] = {}
    for r in results:
        rid = r.get("id")
        if rid:
            seen[rid] = r
    return list(seen.values())


def print_stats(model_name: str, results_path: Path) -> None:
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

    print("  By qtype:")
    for q in ("mc", "oe", "oe(mc)"):
        items = [r for r in results if r.get("qtype") == q and not r.get("vl_error")]
        if not items:
            continue
        ok = sum(1 for r in items if r.get("correct"))
        print(f"    {q:8s}: {ok / len(items) * 100:5.1f}% ({ok}/{len(items)})")

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


# ── Per-model orchestration ──────────────────────────────────────────────


def eval_model(
    model_name: str,
    spec: dict,
    valid_recs: list[dict],
    image_dir: Path,
    results_dir: Path,
    args: argparse.Namespace,
) -> None:
    results_path = results_dir / f"eval_{model_name.replace('/', '_')}.jsonl"

    if args.no_resume and results_path.exists():
        backup = results_path.with_suffix(results_path.suffix + ".bak")
        results_path.rename(backup)
        print(f"  [{model_name}] --no-resume: moved old jsonl to {backup.name}")

    done = load_done_ids(results_path)
    pending = [r for r in valid_recs if r["id"] not in done]
    print(f"\n[{model_name}] {len(pending)} pending, {len(done)} done")
    if not pending:
        print_stats(model_name, results_path)
        return

    # Pre-encode all pending images as base64 data URLs (vLLM accepts directly).
    image_urls: dict[str, str] = {}
    for r in pending:
        b64 = load_image_b64(image_dir, r["image"])
        if b64 is None:
            continue
        ext = r["image"].rsplit(".", 1)[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"
        image_urls[r["id"]] = f"data:{mime};base64,{b64}"

    no_image = [r for r in pending if r["id"] not in image_urls]
    pending_with_img = [r for r in pending if r["id"] in image_urls]
    if no_image:
        print(f"  writing {len(no_image)} NO_IMAGE records and skipping", flush=True)
        for r in no_image:
            append_jsonl(
                results_path,
                {
                    "id": r["id"], "vl_response": None, "vl_error": "NO_IMAGE",
                    "gold": r.get("answer", ""), "qtype": r.get("qtype", "oe"),
                    "family": r.get("family", ""), "domain": r.get("domain", ""),
                    "subdomain": r.get("subdomain", ""),
                    "correct": False, "extracted_letter": "Z",
                    "verifier_raw": None, "verifier_reasoning": None,
                },
            )

    if not pending_with_img:
        print_stats(model_name, results_path)
        return

    raw_by_id = run_vllm_batch(spec["model_path"], pending_with_img, image_urls, args)

    n_ok, n_fail = asyncio.run(
        judge_and_write(pending_with_img, raw_by_id, results_path, model_name)
    )
    print(f"  [{model_name}] judge: {n_ok} ok, {n_fail} failed")

    print_stats(model_name, results_path)


# ── main ─────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Subset of registered models. Default = all in MODELS dict.",
    )
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--image-dir", default=str(DEFAULT_IMAGE_DIR))
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true",
                        help="Move existing eval_<slug>.jsonl to .bak and start fresh.")
    parser.add_argument("--vllm-gpus", type=int, default=VLLM_DEFAULT_GPUS)
    parser.add_argument("--vllm-gpu-mem", type=float, default=VLLM_DEFAULT_GPU_MEM)
    parser.add_argument("--vllm-max-model-len", type=int, default=VLLM_DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--vl-max-tokens", type=int, default=VL_MAX_TOKENS)
    args = parser.parse_args()

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

    valid = [r for r in all_recs if (image_dir / r.get("image", "")).exists()]
    skipped = len(all_recs) - len(valid)
    print(f"  {len(valid)} with image  (skipped {skipped} missing)")

    model_names = args.models or list(MODELS)
    for m in model_names:
        if m not in MODELS:
            raise SystemExit(f"unknown model {m!r}. known: {sorted(MODELS)}")
    print(f"Will evaluate {len(model_names)} model(s): {model_names}\n")

    try:
        for model_name in model_names:
            spec = MODELS[model_name]
            print(f"\n{'#' * 70}\n# {model_name}  ({spec['model_path']})\n{'#' * 70}")
            t0 = time.time()
            eval_model(model_name, spec, valid, image_dir, results_dir, args)
            print(f"  [{model_name}] elapsed: {time.time() - t0:.1f}s")
    except KeyboardInterrupt:
        print("\ninterrupted — rerun to resume (results are append-only).", flush=True)
        return
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
