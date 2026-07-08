#!/usr/bin/env python3
"""
MMMU Pipeline — Stage 1 Fix: CoT Generation + Option Rewriting → Training Format

For each quality_pass stage1 record:
  1. Find actual_new_answer in options (reject if not found)
  2. Generate CoT via Qwen VL (thinking mode): image + question + options + correct answer
  3. Clean CoT (remove trailing "**Final Answer: X**" etc.)
  4. Rewrite options: keep correct answer, generate N-1 new distractors via Qwen text
  5. Output ShareGPT format: {messages, images}

Output:
  output/stage1_train_fixed.jsonl   (training-ready records)
  output/stage1_fix_rejected.jsonl  (rejected records with reason)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import re
import time
import traceback
from pathlib import Path
from typing import Optional

import httpx

from _common import (
    QWEN_API_BASE,
    QWEN_API_KEY,
    QWEN_MODEL,
    LETTERS,
    chat_qwen,
    qwen_text_messages,
    qwen_vl_messages,
    extract_json,
    make_clients,
    JsonlWriter,
    load_done_keys,
    log,
    format_options_multiline,
    normalize_question,
    _get_qwen_sem,
    _backoff_with_jitter,
    RETRYABLE,
    OUTPUT_DIR,
)


# ── CoT generation ─────────────────────────────────────────────────────────────

SYSTEM_COT_GEN = (
    "You are an expert visual question answering assistant. You will be shown an image "
    "and a question about the image. Reason step by step about the image content to arrive "
    "at the correct answer. Focus on what you actually observe in the image. Be concise but thorough."
)

PROMPT_COT_GEN = (
    "Look at the image and answer the following question step by step.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Correct answer: {answer}\n\n"
    "Reason carefully about the image to explain how you arrive at this answer. "
    "Focus on visual details from the image that support the answer."
)


async def _call_qwen_thinking(
    client: httpx.AsyncClient,
    messages: list[dict],
    max_tokens: int = 8000,
    max_retries: int = 6,
    log_prefix: str = "cot",
) -> Optional[str]:
    sem = _get_qwen_sem()
    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.post(
                    f"{QWEN_API_BASE.rstrip('/')}/chat/completions",
                    json={
                        "model": QWEN_MODEL,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 1.0,
                        "enable_thinking": True,
                    },
                    headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                )
                resp.raise_for_status()
                body = resp.json()

            if "choices" not in body or not body["choices"]:
                await asyncio.sleep(_backoff_with_jitter(10, attempt))
                continue

            msg = body["choices"][0]["message"]
            reasoning = (msg.get("reasoning_content") or "").strip()
            if reasoning:
                return reasoning
            content = (msg.get("content") or "").strip()
            return content or None

        except RETRYABLE as e:
            log(f"  [{log_prefix}] retry {attempt+1}/{max_retries}: {type(e).__name__}", "s1fix")
            await asyncio.sleep(_backoff_with_jitter(2 ** attempt, 0, jitter_frac=0.5))
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await asyncio.sleep(_backoff_with_jitter(15, attempt))
            elif e.response.status_code == 400:
                log(f"  [{log_prefix}] HTTP 400, skip", "s1fix")
                return None
            else:
                await asyncio.sleep(_backoff_with_jitter(5, attempt))
        except Exception:
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(_backoff_with_jitter(2, attempt))
    return None


def clean_cot(cot: str) -> str:
    cot = re.sub(r"\n*\*\*Final Answer[:\s].*$", "", cot, flags=re.DOTALL)
    cot = re.sub(
        r"\n*(?:This|The answer|Therefore,?\s*the answer)\s+corresponds?\s+to\s+option\s+[A-Z]\.?\s*$",
        "", cot, flags=re.IGNORECASE,
    )
    cot = re.sub(r"\n*The correct (?:answer|option) is [A-Z]\.?\s*$", "", cot, flags=re.IGNORECASE)
    return cot.rstrip()


# ── Option rewriting ────────────────────────────────────────────────────────────

SYSTEM_OPTION_REWRITE = (
    "You are an expert question designer. Your task is to generate new challenging "
    "multiple-choice distractors for a question. The distractors must be plausible but "
    "clearly incorrect. They should represent common misconceptions or close-but-wrong "
    "answers. Return ONLY a valid JSON object."
)

PROMPT_OPTION_REWRITE = (
    "Given the following question, correct answer, and original options, "
    "generate {num_distractors} NEW distractor options.\n\n"
    "Question: {question}\n"
    "Correct answer: {answer}\n"
    "Original options: {options}\n\n"
    "Requirements:\n"
    "1. Keep the correct answer EXACTLY as-is (do not modify it)\n"
    "2. Generate exactly {num_distractors} NEW distractor options that are DIFFERENT from the original\n"
    "3. Distractors must be plausible, challenging, and similar in format/length to the correct answer\n"
    "4. Distractors must be clearly incorrect\n"
    "5. Do NOT include trivially wrong or absurd options\n\n"
    'Return a JSON object:\n{{\n  "distractors": ["distractor1", "distractor2", ...]\n}}'
)


# ── Helpers ─────────────────────────────────────────────────────────────────────

def find_answer_letter(answer_str: str, options: list[str]) -> str:
    """Find which letter index matches answer_str in options (loose normalize)."""
    norm = lambda s: re.sub(r"[^\w]", "", s.lower().strip())
    # Strip "A) " prefix if present
    cleaned = re.sub(r"^[A-Z]\)\s*", "", answer_str.strip())
    target = norm(cleaned)
    for i, opt in enumerate(options):
        if norm(opt) == target:
            return LETTERS[i]
    # Fallback: try raw
    target2 = norm(answer_str)
    for i, opt in enumerate(options):
        if norm(opt) == target2:
            return LETTERS[i]
    return ""


# ── Pipeline ─────────────────────────────────────────────────────────────────────

class Stage1FixPipeline:
    def __init__(
        self,
        output_dir: str,
        max_concurrent: int = 128,
        max_retries: int = 6,
        api_timeout: float = 1200.0,
        resume: bool = True,
        limit: int = 0,
    ):
        self.output_dir = Path(output_dir)
        self.train_out = self.output_dir / "stage1_train_fixed.jsonl"
        self.rejected_out = self.output_dir / "stage1_fix_rejected.jsonl"

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.api_timeout = api_timeout
        self.resume = resume
        self.limit = limit

        self._train_writer: Optional[JsonlWriter] = None
        self._rejected_writer: Optional[JsonlWriter] = None
        self._done_keys: set[str] = set()
        self._client: Optional[httpx.AsyncClient] = None

    async def _generate_cot(
        self,
        image_b64: str,
        question: str,
        options: list[str],
        answer_display: str,
        sample_key: str,
    ) -> Optional[str]:
        prompt = PROMPT_COT_GEN.format(
            question=question,
            options=format_options_multiline(options),
            answer=answer_display,
        )
        messages = qwen_vl_messages(SYSTEM_COT_GEN, [(image_b64, "image/png")], prompt)
        return await _call_qwen_thinking(
            self._client, messages, max_tokens=8000,
            max_retries=self.max_retries,
            log_prefix=f"cot/{sample_key}",
        )

    async def _rewrite_options(
        self,
        question: str,
        correct_answer_text: str,
        old_options: list[str],
        sample_key: str,
    ) -> Optional[tuple[list[str], str]]:
        num_distractors = max(1, len(old_options) - 1)
        prompt = PROMPT_OPTION_REWRITE.format(
            question=question,
            answer=correct_answer_text,
            options=json.dumps(old_options, ensure_ascii=False),
            num_distractors=num_distractors,
        )
        messages = qwen_text_messages(SYSTEM_OPTION_REWRITE, prompt, no_think=True)

        try:
            text = await chat_qwen(
                self._client, QWEN_API_BASE, QWEN_API_KEY, QWEN_MODEL,
                messages, max_tokens=2048, temperature=0.7,
                max_retries=self.max_retries,
                log_prefix=f"opts/{sample_key}",
            )
        except Exception as e:
            log(f"  [{sample_key}] rewrite_options failed: {e!r}", "s1fix")
            return None

        parsed = extract_json(text)
        if not isinstance(parsed, dict) or "distractors" not in parsed:
            log(f"  [{sample_key}] bad option rewrite response", "s1fix")
            return None

        distractors = parsed["distractors"]
        if not isinstance(distractors, list) or len(distractors) < num_distractors:
            log(f"  [{sample_key}] not enough distractors", "s1fix")
            return None

        distractors = [str(d) for d in distractors[:num_distractors]]
        new_options = [correct_answer_text] + distractors
        random.shuffle(new_options)
        answer_idx = new_options.index(correct_answer_text)
        return new_options, LETTERS[answer_idx]

    async def _process_one(self, rec: dict) -> None:
        sample_key = rec["sample_key"]
        async with self.semaphore:
            try:
                meta = rec.get("mmmu_meta", {})
                options = meta.get("options", [])
                if not isinstance(options, list) or len(options) < 2:
                    await self._rejected_writer.write({
                        "sample_key": sample_key, "reject_reason": "no_options",
                    })
                    return

                actual_new_answer = rec.get("actual_new_answer", "")
                answer_letter = find_answer_letter(actual_new_answer, options)
                if not answer_letter:
                    await self._rejected_writer.write({
                        "sample_key": sample_key, "reject_reason": "answer_not_in_options",
                        "actual_new_answer": actual_new_answer, "options": options,
                    })
                    return

                answer_text_str = options[ord(answer_letter) - ord("A")]
                question = rec.get("new_question") or rec.get("original_question", "")

                img_path = rec.get("stage1_image_path", "")
                if not os.path.isfile(img_path):
                    await self._rejected_writer.write({
                        "sample_key": sample_key, "reject_reason": "image_not_found",
                    })
                    return
                img_b64 = base64.b64encode(Path(img_path).read_bytes()).decode()

                # Generate CoT
                answer_display = f"{answer_letter}) {answer_text_str}"
                cot_raw = await self._generate_cot(
                    img_b64, question, options, answer_display, sample_key,
                )
                if not cot_raw:
                    await self._rejected_writer.write({
                        "sample_key": sample_key, "reject_reason": "cot_generation_failed",
                    })
                    return

                cot = clean_cot(cot_raw)
                if not cot.strip():
                    await self._rejected_writer.write({
                        "sample_key": sample_key, "reject_reason": "cot_empty_after_cleaning",
                    })
                    return

                # Rewrite options
                opt_result = await self._rewrite_options(
                    question, answer_text_str, options, sample_key,
                )
                if opt_result is not None:
                    new_options, new_letter = opt_result
                else:
                    new_options = list(options)
                    new_letter = answer_letter

                # Build ShareGPT record
                q_norm = normalize_question(question)
                if "<image>" not in q_norm:
                    q_norm = "<image>" + q_norm

                options_str = format_options_multiline(new_options)
                user_content = f"{q_norm}\n{options_str}"
                assistant_content = (
                    f"<think>\n{cot}\n</think>\n"
                    f"<answer>\n{new_letter}\n</answer>"
                )

                record = {
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": assistant_content},
                    ],
                    "images": [img_path],
                    "sample_key": sample_key,
                }
                await self._train_writer.write(record)
                log(f"  [{sample_key}] done | answer={new_letter}", "s1fix")

            except Exception as e:
                log(f"  [{sample_key}] ERROR: {type(e).__name__}: {e!r}", "s1fix")
                log(traceback.format_exc(), "s1fix")

    async def run(self) -> None:
        log("Loading stage1 quality_pass records...", "s1fix")
        s1_path = self.output_dir / "stage1_samples.jsonl"
        if not s1_path.exists():
            log(f"ERROR: {s1_path} not found. Run stage1_edit.py first.", "s1fix")
            return

        records = []
        with open(s1_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("quality_pass"):
                    records.append(rec)
        log(f"  quality_pass records: {len(records)}", "s1fix")

        if self.resume:
            self._done_keys = load_done_keys(self.train_out)
            self._done_keys |= load_done_keys(self.rejected_out)
            log(f"  already done: {len(self._done_keys)}", "s1fix")

        todo = [r for r in records if r["sample_key"] not in self._done_keys]
        if self.limit > 0:
            todo = todo[:self.limit]
        log(f"  to process: {len(todo)}", "s1fix")

        if not todo:
            log("Nothing to do.", "s1fix")
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._train_writer = JsonlWriter(str(self.train_out))
        self._rejected_writer = JsonlWriter(str(self.rejected_out))
        inner, _ = make_clients(api_timeout=self.api_timeout)
        self._client = inner

        try:
            await asyncio.gather(
                *[self._process_one(r) for r in todo], return_exceptions=True
            )
        finally:
            await inner.aclose()

        log(f"Done. Processed {len(todo)} records.", "s1fix")
        log(f"  train output:    {self.train_out}", "s1fix")
        log(f"  rejected output: {self.rejected_out}", "s1fix")


def main():
    parser = argparse.ArgumentParser(
        description="Convert MMMU stage1 samples to ShareGPT training format"
    )
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--max_concurrent", type=int, default=256)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--api_timeout", type=float, default=1200.0)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    pipe = Stage1FixPipeline(
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        api_timeout=args.api_timeout,
        resume=not args.no_resume,
        limit=args.limit,
    )
    asyncio.run(pipe.run())


if __name__ == "__main__":
    main()
