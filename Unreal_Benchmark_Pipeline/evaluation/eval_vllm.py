#!/usr/bin/env python3
"""
eval_vllm.py — Multi-model VLM evaluation via in-process vLLM batched chat
on the unified UnrealBench (633 samples).

Evaluation methodology copied verbatim from
  /mnt/tidal-alsh01/.../export_vlmevalkit/eval_gemini.py
i.e.:
  - VL inference with MathVision-style Hint embedded in the user text (NO system prompt).
  - MC (qtype=="mc"):       heuristic letter extraction → Gemini fallback → random "Z".
  - OE/oe(mc):              Gemini verifier one-step (extract+judge), tail-3000 of CoT.
  - Empty VL response       → auto wrong (extracted_letter="Z", correct=False).

VL inference is `vllm.LLM.chat([...], use_tqdm=True)` — batched in-process.
MC extract fallback and OE verifier use Gemini (thinkingBudget=0 for speed).

Each model writes to evaluation/eval_results/eval_<model_slug>.jsonl.
Resume-aware: re-runs skip already-processed sample ids.

Usage:
  python evaluation/eval_vllm.py
  python evaluation/eval_vllm.py --models qwen3vl4b_step582_rubric
  python evaluation/eval_vllm.py --vllm-gpus 8 --vllm-gpu-mem 0.5
  python evaluation/eval_vllm.py --phase 2 --rejudge
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


# ── Gemini text-API constants (for MC fallback + OE verifier) ────────────
# Same proxy/endpoint as eval_gemini.py and eval_api.py; thinking is disabled
# for judge calls so JSON output is fast.
GEMINI_ENDPOINT = "<GEMINI_ENDPOINT>"
GEMINI_KEYS = [
    os.environ.get("GEMINI_API_KEY_1", ""),
    os.environ.get("GEMINI_API_KEY_2", ""),
]
GEMINI_JUDGE_MODEL = "gemini-3.1-pro-preview"  # the only model the proxy reaches

JUDGE_CONCURRENCY = 128        # split across keys → ~32 in-flight per key
REQUEST_TIMEOUT = 120.0
MAX_RETRIES = 6

MAX_OUTPUT_TOKENS_EXTRACT = 256
MAX_OUTPUT_TOKENS_VERIFIER = 512


# ── vLLM defaults ────────────────────────────────────────────────────────
VLLM_DEFAULT_GPUS = 8
VLLM_DEFAULT_GPU_MEM = 0.9
VLLM_DEFAULT_MAX_MODEL_LEN = 8192
VL_TEMPERATURE = 0.0
VL_MAX_TOKENS = 8192  # allow long CoT for thinking models (math/reasoning)


# ── Model registry (aligned with zerogen/eval_models_vllm.py) ────────────
MODELS: dict[str, dict] = {
    # "qwen3vl4b_step540": {
    #     "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/checkpoints/new_sft_qwen3vl4b_grpo_rubric_sample_20260506_094403/hf_model_step540",
    # },
    # "qwen3vl4b_step582": {
    #     "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/zerovision/checkpoints/new_sft_qwen3vl4b_grpo_rlvr_20260506_085025/hf_model_step582",
    # },
    # "qwen3vl4b_instruct": {
    #     "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3-VL-4B-Instruct",
    # },
    # "Qwen3.5-4B": {
    #     "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3.5-4B",
    # },
    # "Qwen3.5-9B": {
    #     "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3.5-9B",
    # },
    # "Qwen3.5-27B": {
    #     "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3.5-27B",
    # },
    # "Qwen3-VL-4B-Instruct": {
    #     "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3-VL-4B-Instruct",
    # },
    # "Qwen3-VL-8B-Instruct": {
    #     "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3-VL-8B-Instruct",
    # },
    # "Qwen3-VL-32B-Instruct": {
    #     "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wangyipu/model/Qwen3-VL-32B-Instruct",
    # },
    # "InternVL3_5-8B": {
    #     "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wanghan/models/InternVL3_5-8B/",
    # },
    "InternVL3_5-38B": {
        "model_path": "/mnt/tidal-alsh01/dataset/perceptionVLM/wanghan/models/InternVL3_5-38B/",
    },
}


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


# ── MC extraction: heuristic + GPT fallback (verbatim) ───────────────────


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


# ── Gemini text call (for MC fallback + OE verifier) ─────────────────────


async def call_gemini_text(
    client: httpx.AsyncClient,
    key_pool: GeminiKeyPool,
    system_prompt: str,
    user_text: str,
    max_tokens: int = 512,
) -> str:
    """Gemini text-only call (Google native generateContent format).

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
                raise RuntimeError(f"no candidates: {str(body)[:200]}")
            parts = cands[0].get("content", {}).get("parts", [])
            return "".join(
                p.get("text", "")
                for p in parts
                if "text" in p and not p.get("thought")
            ).strip()
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
    key_pool: GeminiKeyPool,
    record: dict,
    response: str,
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
    ans = await call_gemini_text(
        client, key_pool, extract_system, prompt, max_tokens=MAX_OUTPUT_TOKENS_EXTRACT
    )
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
) -> dict:
    resp_excerpt = response[-3000:] if len(response) > 3000 else response
    user_text = (
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Candidate response:\n{resp_excerpt}\n\n"
        "Output the JSON judgment."
    )
    raw = await call_gemini_text(
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


# ── vLLM batched inference (Phase 1) ─────────────────────────────────────


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


def phase1_vllm(
    model_name: str,
    spec: dict,
    valid_recs: list[dict],
    image_dir: Path,
    results_path: Path,
    args: argparse.Namespace,
) -> None:
    existing = read_jsonl(results_path)
    done_ids = {r["id"] for r in existing if r.get("vl_response") or r.get("vl_error")}
    empty_ids = {
        r["id"] for r in existing
        if not r.get("vl_response") and not r.get("vl_error")
    }
    if args.retry_empty:
        print(f"  --retry-empty: {len(empty_ids)} empty responses will be re-inferred")
        done_ids -= empty_ids
    pending = [r for r in valid_recs if r["id"] not in done_ids]
    print(f"[Phase 1] {model_name}: {len(pending)} pending, {len(done_ids)} done")
    if not pending:
        return

    # Pre-encode all pending images as base64 data URLs (vLLM accepts these directly)
    image_urls: dict[str, str] = {}
    for r in pending:
        b64 = load_image_b64(image_dir, r["image"])
        if b64 is None:
            continue
        ext = r["image"].rsplit(".", 1)[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"
        image_urls[r["id"]] = f"data:{mime};base64,{b64}"
    no_image = [r for r in pending if r["id"] not in image_urls]
    pending = [r for r in pending if r["id"] in image_urls]
    if no_image:
        print(f"  skipping {len(no_image)} records with missing image", flush=True)
        for r in no_image:
            result = {
                "id": r["id"], "vl_response": None, "vl_error": "NO_IMAGE",
                "gold": r.get("answer", ""), "qtype": r.get("qtype", "oe"),
                "family": r.get("family", ""), "domain": r.get("domain", ""),
                "subdomain": r.get("subdomain", ""),
                "correct": None, "extracted_letter": None,
                "verifier_raw": None, "verifier_reasoning": None,
            }
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

    raw_by_id = run_vllm_batch(spec["model_path"], pending, image_urls, args)

    with open(results_path, "a", encoding="utf-8") as f:
        for r in pending:
            raw = raw_by_id.get(r["id"], "")
            result = {
                "id": r["id"],
                "vl_response": raw if raw else None,
                "vl_error": None if raw is not None else "EMPTY",
                "gold": r.get("answer", ""),
                "qtype": r.get("qtype", "oe"),
                "family": r.get("family", ""),
                "domain": r.get("domain", ""),
                "subdomain": r.get("subdomain", ""),
                "correct": None,
                "extracted_letter": None,
                "verifier_raw": None,
                "verifier_reasoning": None,
            }
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
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

    kp = GeminiKeyPool(GEMINI_KEYS, JUDGE_CONCURRENCY // max(len(GEMINI_KEYS), 1))

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
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=0,
                        help="0 = run all phases (default)")
    parser.add_argument("--rejudge", action="store_true")
    parser.add_argument("--retry-empty", action="store_true")
    parser.add_argument("--vllm-gpus", type=int, default=VLLM_DEFAULT_GPUS)
    parser.add_argument("--vllm-gpu-mem", type=float, default=VLLM_DEFAULT_GPU_MEM)
    parser.add_argument("--vllm-max-model-len", type=int, default=VLLM_DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--vl-max-tokens", type=int, default=VL_MAX_TOKENS)
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
        print(f"\n{'#' * 70}\n# {model_name}  ({spec['model_path']})\n{'#' * 70}")

        t0 = time.time()
        if run_all or args.phase == 1:
            phase1_vllm(model_name, spec, valid, image_dir, results_path, args)
        if run_all or args.phase == 2:
            asyncio.run(phase2_judge(model_name, valid, results_path, args.rejudge))
        if run_all or args.phase == 3:
            phase3_stats(model_name, results_path)
        print(f"  [{model_name}] elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted — rerun to resume (results are append-only).", flush=True)
