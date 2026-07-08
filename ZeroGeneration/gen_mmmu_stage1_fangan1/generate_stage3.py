#!/usr/bin/env python3
"""
方案一 — Stage 3: CoT 生成 + 答案验证 + 选项改写 → 训练 JSONL

输入：stage1p5_samples.jsonl（quality_pass=True 的记录）
输出：stage3_output.jsonl（LLaMA-Factory JSONL 格式）

处理流程（每条 quality_pass 记录）：
  1. 用 Qwen VL（enable_thinking=True）生成 CoT（使用 reasoning_content）
  2. 用 Qwen VL 验证答案是否与 CoT 一致
  3. MC 题：用 Qwen text 改写干扰项（生成 3 个新干扰项）
  4. 格式化为 <think>…</think><answer>…</answer> 训练格式
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Optional

import httpx

# ── Config ─────────────────────────────────────────────────────────────────────

QWEN_API_BASE = os.environ.get("QWEN_API_BASE", "https://runway.devops.xiaohongshu.com/openai/qwen/v1")
QWEN_API_KEY = os.environ.get("OPENAI_API_KEY", "YOUR_QWEN_API_KEY")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.6-plus")
MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "128"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "5"))

STAGE1P5_INPUT = os.environ.get(
    "STAGE1P5_INPUT",
    str(Path(__file__).resolve().parent / "_archive/output_fangan1/stage1p5_samples.jsonl"),
)
STAGE3_OUTPUT = os.environ.get(
    "STAGE3_OUTPUT",
    str(Path(__file__).resolve().parent / "_archive/output_fangan1/stage3_output.jsonl"),
)
STAGE3_REJECTED = os.environ.get(
    "STAGE3_REJECTED",
    str(Path(__file__).resolve().parent / "_archive/output_fangan1/stage3_rejected.jsonl"),
)

# ── Helpers ────────────────────────────────────────────────────────────────────

_sem: Optional[asyncio.Semaphore] = None


def _get_sem() -> asyncio.Semaphore:
    global _sem
    if _sem is None:
        _sem = asyncio.Semaphore(MAX_CONCURRENT)
    return _sem


def log(msg: str) -> None:
    print(f"[stage3] {msg}", flush=True)


def _backoff(attempt: int, base: float = 2.0) -> float:
    wait = base * (attempt + 1)
    jitter = wait * 0.4 * (2 * random.random() - 1)
    return max(1.0, wait + jitter)


RETRYABLE: tuple = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ProxyError,
)


# ── JSONL I/O ──────────────────────────────────────────────────────────────────

class JsonlWriter:
    def __init__(self, path: str):
        self.path = path
        self._lock = asyncio.Lock()

    async def write(self, record: dict) -> None:
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_keys(path: str) -> set[str]:
    done: set[str] = set()
    if not os.path.isfile(path):
        return done
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                k = rec.get("sample_key", "")
                if k:
                    done.add(k)
            except json.JSONDecodeError:
                pass
    return done


def load_samples(path: str) -> list[dict]:
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return samples


# ── Image utils ────────────────────────────────────────────────────────────────

def load_image_b64(path: str) -> Optional[tuple[str, str]]:
    p = Path(path)
    if not p.is_file():
        return None
    data = p.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    ext = p.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return b64, mime


# ── Qwen API ───────────────────────────────────────────────────────────────────

async def call_qwen(
    client: httpx.AsyncClient,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 0.7,
    enable_thinking: bool = False,
    log_prefix: str = "qwen",
) -> Optional[dict]:
    payload: dict[str, Any] = {
        "model": QWEN_MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 1.0 if enable_thinking else temperature,
    }
    if enable_thinking:
        payload["enable_thinking"] = True

    sem = _get_sem()
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.post(
                    f"{QWEN_API_BASE.rstrip('/')}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                )
                resp.raise_for_status()
                body = resp.json()

            if "choices" not in body or not body["choices"]:
                wait = _backoff(attempt, base=10.0)
                log(f"  [{log_prefix}] no choices, wait {wait:.1f}s")
                await asyncio.sleep(wait)
                continue

            msg = body["choices"][0]["message"]
            return {
                "content": (msg.get("content") or "").strip(),
                "thinking": (msg.get("reasoning_content") or "").strip(),
            }

        except RETRYABLE as e:
            wait = _backoff(attempt)
            log(f"  [{log_prefix}] retry {attempt+1}/{MAX_RETRIES}: {type(e).__name__} (wait {wait:.1f}s)")
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429:
                wait = _backoff(attempt, base=15.0)
                log(f"  [{log_prefix}] HTTP 429, wait {wait:.1f}s")
                await asyncio.sleep(wait)
            elif status == 400:
                log(f"  [{log_prefix}] HTTP 400: {e.response.text[:300]}")
                return None
            else:
                wait = _backoff(attempt, base=5.0)
                log(f"  [{log_prefix}] HTTP {status}, wait {wait:.1f}s")
                await asyncio.sleep(wait)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = _backoff(attempt)
                log(f"  [{log_prefix}] error {type(e).__name__}: {e!r}, wait {wait:.1f}s")
                await asyncio.sleep(wait)
            else:
                log(f"  [{log_prefix}] failed after {MAX_RETRIES} retries: {e!r}")
                return None
    return None


def build_vl_messages(
    system: str, image_b64: str, image_mime: str, user_text: str, no_think: bool = False
) -> list[dict]:
    if no_think:
        system = "/no_think\n" + system
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}},
                {"type": "text", "text": user_text},
            ],
        },
    ]


def build_text_messages(system: str, user_text: str, no_think: bool = False) -> list[dict]:
    if no_think:
        system = "/no_think\n" + system
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user_text},
    ]


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_COT_GEN = (
    "You are an expert visual question answering assistant. You will be shown an image and a question about the image. "
    "Reason step by step about the image content to arrive at the correct answer. "
    "Focus on what you actually observe in the image. Be concise but thorough."
)

PROMPT_COT_GEN = """\
Look at the image and answer the following question step by step.

Question: {question}

Correct answer: {answer}

Reason carefully about the image to explain how you arrive at this answer. \
Focus on visual details from the image that support the answer."""

SYSTEM_VERIFY = (
    "You are a rigorous visual question answering verifier. You will be shown an image, a question, "
    "a chain-of-thought reasoning, and a claimed answer. "
    "Your job is to determine whether the answer is correct based on the image content. Be strict and accurate."
)

PROMPT_VERIFY = """\
Look at the image and verify whether the following answer is correct.

Question: {question}

Chain-of-thought reasoning:
{cot}

Claimed answer: {answer}

Based on the image, is the claimed answer correct? Respond in strict JSON format:
{{
  "correct": true or false,
  "reason": "brief explanation"
}}"""

SYSTEM_OPTION_REWRITE = (
    "You are an expert question designer. Your task is to generate new challenging multiple-choice distractors for a question. "
    "The distractors must be plausible but clearly incorrect. They should represent common misconceptions or close-but-wrong answers. "
    "Return ONLY a valid JSON object."
)

PROMPT_OPTION_REWRITE = """\
Given the following question, correct answer, and original options, generate 3 NEW distractor options.

Question: {question}
Correct answer: {answer}
Original options: {options}

Requirements:
1. Keep the correct answer EXACTLY as-is (do not modify it)
2. Generate exactly 3 NEW distractor options that are DIFFERENT from the original distractors
3. Distractors must be plausible, challenging, and similar in format/length to the correct answer
4. Distractors must be clearly incorrect
5. Do NOT include trivially wrong or absurd options

Return a JSON object:
{{
  "distractors": ["distractor1", "distractor2", "distractor3"]
}}"""


# ── JSON extraction ────────────────────────────────────────────────────────────

def extract_json(text: str) -> Any:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start_obj = text.find("{")
        start_arr = text.find("[")
        candidates = [c for c in (start_obj, start_arr) if c >= 0]
        if not candidates:
            return None
        start = min(candidates)
        end = max(text.rfind("}"), text.rfind("]"))
        if end <= start:
            return None
        try:
            return json.loads(text[start: end + 1])
        except json.JSONDecodeError:
            return None


# ── CoT cleaning ───────────────────────────────────────────────────────────────

def clean_cot(cot: str) -> str:
    cot = re.sub(r"\n*\*\*Final Answer[:\s].*$", "", cot, flags=re.DOTALL)
    cot = re.sub(
        r"\n*(?:This|The answer|Therefore,?\s*the answer)\s+corresponds?\s+to\s+option\s+[A-Z]\.?\s*$",
        "", cot, flags=re.IGNORECASE,
    )
    cot = re.sub(r"\n*The correct (?:answer|option) is [A-Z]\.?\s*$", "", cot, flags=re.IGNORECASE)
    return cot.rstrip()


# ── Answer resolution ──────────────────────────────────────────────────────────

def resolve_answer_text(answer: str, options: list[str]) -> str:
    stripped = answer.strip()
    if re.match(r"^[A-Ja-j]$", stripped) and options:
        idx = ord(stripped.upper()) - ord("A")
        if 0 <= idx < len(options):
            return options[idx]
    return answer


def format_options_str(options: list[str]) -> str:
    letters = "ABCDEFGHIJ"
    parts = []
    for i, opt in enumerate(options):
        if i < len(letters):
            parts.append(f"{letters[i]}) {opt}")
    return "\n".join(parts)


# ── Multi-image check ──────────────────────────────────────────────────────────

def is_multi_image(sample: dict) -> bool:
    q = sample.get("new_question") or sample.get("final_question") or ""
    refs = re.findall(r"<image\s+(\d+)>", q)
    return bool(refs) and max(int(r) for r in refs) > 1


def normalize_question(q: str) -> str:
    q = re.sub(r"<image\s*\d*\s*>", "<image>", q)
    q = re.sub(r"  +", " ", q)
    return q.strip()


# ── Per-sample processing ──────────────────────────────────────────────────────

async def generate_cot(
    client: httpx.AsyncClient,
    image_path: str,
    question: str,
    answer: str,
    sample_key: str,
) -> Optional[str]:
    img = load_image_b64(image_path)
    if img is None:
        log(f"  [{sample_key}] image not found: {image_path}")
        return None

    b64, mime = img
    prompt = PROMPT_COT_GEN.format(question=question, answer=answer)
    messages = build_vl_messages(SYSTEM_COT_GEN, b64, mime, prompt)

    result = await call_qwen(
        client, messages, max_tokens=8000, enable_thinking=True, log_prefix=f"cot:{sample_key}",
    )
    if result is None:
        return None

    thinking = result["thinking"]
    if thinking:
        return thinking
    return result["content"] or None


async def verify_answer(
    client: httpx.AsyncClient,
    image_path: str,
    question: str,
    cot: str,
    answer: str,
    sample_key: str,
) -> Optional[bool]:
    img = load_image_b64(image_path)
    if img is None:
        log(f"  [{sample_key}] image not found for verify: {image_path}")
        return None

    b64, mime = img
    prompt = PROMPT_VERIFY.format(question=question, cot=cot, answer=answer)
    messages = build_vl_messages(SYSTEM_VERIFY, b64, mime, prompt, no_think=True)

    result = await call_qwen(client, messages, max_tokens=512, temperature=0.1, log_prefix=f"verify:{sample_key}")
    if result is None:
        return None

    parsed = extract_json(result["content"])
    if not isinstance(parsed, dict):
        return None
    return bool(parsed.get("correct", False))


async def rewrite_options(
    client: httpx.AsyncClient,
    question: str,
    correct_answer: str,
    old_options: list[str],
    sample_key: str,
) -> Optional[tuple[list[str], int]]:
    prompt = PROMPT_OPTION_REWRITE.format(
        question=question,
        answer=correct_answer,
        options=json.dumps(old_options, ensure_ascii=False),
    )
    messages = build_text_messages(SYSTEM_OPTION_REWRITE, prompt, no_think=True)

    result = await call_qwen(client, messages, max_tokens=1024, temperature=0.7, log_prefix=f"opts:{sample_key}")
    if result is None:
        return None

    parsed = extract_json(result["content"])
    if not isinstance(parsed, dict) or "distractors" not in parsed:
        log(f"  [{sample_key}] bad option rewrite response")
        return None

    distractors = parsed["distractors"]
    if not isinstance(distractors, list) or len(distractors) < 3:
        log(f"  [{sample_key}] not enough distractors")
        return None

    distractors = [str(d) for d in distractors[:3]]
    new_options = [correct_answer] + distractors
    random.shuffle(new_options)
    answer_idx = new_options.index(correct_answer)
    return new_options, answer_idx


async def process_sample(
    client: httpx.AsyncClient,
    sample: dict,
    writer: JsonlWriter,
    rejected_writer: JsonlWriter,
    stats: dict,
) -> None:
    sample_key = sample.get("sample_key", "unknown")

    image_path = sample.get("stage1_image_path", "")
    question_raw = sample.get("new_question", "")
    answer_raw = sample.get("actual_new_answer", "")

    meta = sample.get("mmmu_meta", {})
    q_type = meta.get("question_type", "open")
    options = meta.get("options", [])
    is_mc = q_type == "multiple-choice" and len(options) >= 2

    question = normalize_question(question_raw)
    if "<image>" not in question:
        question = "<image>" + question
    answer_text = resolve_answer_text(answer_raw, options) if is_mc else answer_raw

    # Validate answer is among options (MC)
    if is_mc and answer_text not in options:
        best = None
        best_score = 0
        at_lower = answer_text.lower().strip()
        for opt in options:
            ol = opt.lower().strip()
            if at_lower == ol:
                best = opt
                break
            if at_lower in ol or ol in at_lower:
                score = min(len(at_lower), len(ol)) / max(len(at_lower), len(ol))
                if score > best_score:
                    best_score = score
                    best = opt
        if best:
            answer_text = best

    # Generate CoT (thinking mode)
    cot = await generate_cot(client, image_path, question, answer_text, sample_key)
    if not cot:
        stats["cot_missing"] += 1
        await rejected_writer.write({"sample_key": sample_key, "reason": "cot_generation_failed"})
        return

    cot = clean_cot(cot)

    # Parallel: verify + option rewrite
    tasks: dict[str, asyncio.Task] = {}
    tasks["verify"] = asyncio.create_task(
        verify_answer(client, image_path, question, cot, answer_text, sample_key)
    )
    if is_mc:
        tasks["options"] = asyncio.create_task(
            rewrite_options(client, question, answer_text, options, sample_key)
        )

    results = {}
    for name, task in tasks.items():
        results[name] = await task

    verified = results.get("verify")
    if verified is False:
        stats["answer_incorrect"] += 1
        await rejected_writer.write({"sample_key": sample_key, "reason": "answer_incorrect"})
        return

    if is_mc:
        opt_result = results.get("options")
        if opt_result is not None:
            new_options, answer_idx = opt_result
            answer_letter = chr(ord("A") + answer_idx)
            options_str = format_options_str(new_options)
            user_content = f"{question}\n{options_str}"
            assistant_answer = answer_letter
            stats["options_rewritten"] += 1
        else:
            answer_idx = None
            for i, opt in enumerate(options):
                if opt == answer_text:
                    answer_idx = i
                    break
            if answer_idx is None:
                answer_idx = 0
            answer_letter = chr(ord("A") + answer_idx)
            options_str = format_options_str(options)
            user_content = f"{question}\n{options_str}"
            assistant_answer = answer_letter
    else:
        user_content = question
        assistant_answer = answer_text

    assistant_content = f"<think>\n{cot}\n</think>\n<answer>\n{assistant_answer}\n</answer>"

    record = {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [image_path],
        "sample_key": sample_key,
        "source_stage": "stage1.5",
    }

    await writer.write(record)
    stats["written"] += 1


# ── Main ───────────────────────────────────────────────────────────────────────

class ProgressTracker:
    def __init__(self, total: int, label: str):
        self.total = total
        self.label = label
        self._done = 0
        self._lock = asyncio.Lock()
        self._start = time.time()

    async def tick(self) -> None:
        async with self._lock:
            self._done += 1
            if self._done % 50 == 0 or self._done == self.total:
                elapsed = time.time() - self._start
                rate = self._done / elapsed if elapsed > 0 else 0
                eta = (self.total - self._done) / rate if rate > 0 else 0
                log(f"[{self.label}] {self._done}/{self.total} ({rate:.1f}/s, ETA {eta:.0f}s)")


async def main_async(
    stage1p5_input: str,
    stage3_output: str,
    stage3_rejected: str,
) -> None:
    log(f"Loading stage1.5 from {stage1p5_input}")
    all_samples = load_samples(stage1p5_input)
    good_samples = [
        s for s in all_samples
        if s.get("quality_pass") and not is_multi_image(s)
    ]
    multi_skipped = sum(1 for s in all_samples if s.get("quality_pass") and is_multi_image(s))
    log(f"  Total: {len(all_samples)}, quality_pass (single-image): {len(good_samples)}, multi-image skipped: {multi_skipped}")

    done_keys = load_done_keys(stage3_output) | load_done_keys(stage3_rejected)
    todo = [s for s in good_samples if s.get("sample_key", "") not in done_keys]
    log(f"  Already done: {len(done_keys)}, remaining: {len(todo)}")

    if not todo:
        log("Nothing to do")
        return

    Path(stage3_output).parent.mkdir(parents=True, exist_ok=True)
    Path(stage3_rejected).parent.mkdir(parents=True, exist_ok=True)

    writer = JsonlWriter(stage3_output)
    rejected_writer = JsonlWriter(stage3_rejected)
    stats = {"written": 0, "cot_missing": 0, "answer_incorrect": 0, "options_rewritten": 0}

    progress = ProgressTracker(len(todo), "stage3")

    timeout = httpx.Timeout(600.0, read=600.0, pool=60.0)
    limits = httpx.Limits(max_connections=512, max_keepalive_connections=256)
    client = httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False)

    async def _wrapper(sample: dict) -> None:
        try:
            await process_sample(client, sample, writer, rejected_writer, stats)
        except Exception as e:
            sk = sample.get("sample_key", "?")
            log(f"  [{sk}] unexpected error: {type(e).__name__}: {e!r}")
        await progress.tick()

    batch_size = MAX_CONCURRENT * 2
    try:
        for i in range(0, len(todo), batch_size):
            batch = todo[i: i + batch_size]
            await asyncio.gather(*[_wrapper(s) for s in batch])
    finally:
        await client.aclose()

    log(f"Done. Stats: {stats}")


def main():
    import argparse
    global QWEN_API_BASE, QWEN_API_KEY, QWEN_MODEL, MAX_CONCURRENT

    parser = argparse.ArgumentParser(description="方案一 Stage 3: CoT + verify + format to training JSONL")
    parser.add_argument("--stage1p5_input", default=STAGE1P5_INPUT)
    parser.add_argument("--stage3_output", default=STAGE3_OUTPUT)
    parser.add_argument("--stage3_rejected", default=STAGE3_REJECTED)
    parser.add_argument("--qwen_api_base", default=QWEN_API_BASE)
    parser.add_argument("--qwen_api_key", default=QWEN_API_KEY)
    parser.add_argument("--qwen_model", default=QWEN_MODEL)
    parser.add_argument("--max_concurrent", type=int, default=MAX_CONCURRENT)
    args = parser.parse_args()

    QWEN_API_BASE = args.qwen_api_base
    QWEN_API_KEY = args.qwen_api_key
    QWEN_MODEL = args.qwen_model
    MAX_CONCURRENT = args.max_concurrent

    asyncio.run(main_async(args.stage1p5_input, args.stage3_output, args.stage3_rejected))


if __name__ == "__main__":
    main()
