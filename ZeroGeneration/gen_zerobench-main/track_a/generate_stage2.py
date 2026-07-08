#!/usr/bin/env python3
"""
ZeroBench MAIN — Track A, Stage 2: Visual Diversification + CoT Generation (SFT)

Input:  outputs/rl/zerobench_track_a_stage1.jsonl  (quality_pass=True records)
Output: outputs/sft/zerobench_track_a_stage2.jsonl

Pipeline per Stage-1 record:
  Step 1 [GPT-Image-2]  → generate STYLE_NAMES diverse re-stylings of the edited image
  Step 2 [Gemini VL]    → generate <think>...</think><answer>...</answer> CoT trace
                          for each styled variant

SFT record format (LLaMA-Factory messages):
  {
    "messages": [
      {"role": "system",    "content": SYSTEM_PROMPT},
      {"role": "user",      "content": "<image>{question}"},
      {"role": "assistant", "content": "<think>\n...\n</think>\n<answer>\n...\n</answer>"}
    ],
    "images":     [absolute_image_path],
    "sample_key": "main_{question_id}_s{style_idx}",
    "stage":      "track_a_stage2"
  }

Usage:
    python track_a/generate_stage2.py \
        --input  outputs/rl/zerobench_track_a_stage1.jsonl \
        --n_styles 3 \
        --output zerobench_track_a_stage2.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import concurrent.futures
import io
import json
import os
import random
import re
import traceback
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

import requests as _requests
from PIL import Image
from utils.gemini import call_gemini as _call_gemini

# ── Constants ─────────────────────────────────────────────────────────────────

GEMINI_ENDPOINT = "https://runway.devops.rednote.life/openai/google/v1:generateContent"
GEMINI_MODEL    = "gemini-3.0-pro"
GEMINI_API_KEY  = "YOUR_GEMINI_API_KEY"

GPT_IMAGE2_URL     = "https://runway.devops.rednote.life"
GPT_IMAGE2_API_VER = "2025-04-01-preview"
GPT_IMAGE2_MODEL   = "gpt-image-2-c"
GPT_IMAGE2_SIZE    = "auto"
GPT_IMAGE2_QUALITY = "auto"

GEMINI_SEM = asyncio.Semaphore(8)
GPT_SEM    = asyncio.Semaphore(2)

OUTPUT_DIR   = Path("outputs")
SFT_DIR      = OUTPUT_DIR / "sft"
IMG_SFT_DIR  = SFT_DIR / "track_a_stage2_images"

SYSTEM_PROMPT = (
    "You are an expert visual puzzle solver. "
    "Output your reasoning inside <think>...</think> "
    "and your final answer inside <answer>...</answer>."
)

# Visual style prompts for GPT-Image-2 re-styling
STYLE_PROMPTS = {
    "vivid":    (
        "Re-render this image with vivid, high-contrast colors and punchy saturation, "
        "as if taken on a professional mirrorless camera in excellent lighting. "
        "Preserve ALL objects, text, numbers, and spatial layout exactly. "
        "Do not add or remove any items."
    ),
    "vintage":  (
        "Apply a warm vintage film photography look with subtle grain and slightly faded colors, "
        "as if taken on a 35mm film camera from the 1980s. "
        "Keep ALL objects, text, numbers, and spatial arrangement identical. "
        "Do not add or remove anything."
    ),
    "cool":     (
        "Adjust the image to have cool, desaturated blue-green tones with soft shadows, "
        "like a cinematic photograph taken under overcast sky light. "
        "Every object, piece of text, and number must remain exactly the same. "
        "Do not add or remove any elements."
    ),
    "bright":   (
        "Make this image bright and airy with lifted shadows, soft highlights, and warm white tones, "
        "like a product photograph taken in a light studio. "
        "The content — all objects, text, and numbers — must remain completely unchanged. "
        "Do not add or remove anything."
    ),
}

STYLE_NAMES = list(STYLE_PROMPTS.keys())   # ["vivid", "vintage", "cool", "bright"]


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[track_a_s2] {msg}", flush=True)


# ── Backoff ───────────────────────────────────────────────────────────────────

def _backoff(attempt: int, base: float = 5.0) -> float:
    wait = base * (2 ** attempt)
    jitter = wait * 0.3 * (2 * random.random() - 1)
    return max(2.0, wait + jitter)


# ── Image helpers ─────────────────────────────────────────────────────────────

def _load_image(path: str) -> Optional[Image.Image]:
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        _log(f"  load_image error: {e!r}")
        return None


def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> tuple[str, str]:
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return base64.b64encode(buf.getvalue()).decode(), mime


def _resize_if_needed(img: Image.Image, max_side: int = 1024) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ── Step 1: Style diversification via GPT-Image-2 ────────────────────────────

def _gpt_image2_style_sync(
    image_bytes: bytes,
    style_prompt: str,
    api_key: str,
    timeout: float = 300.0,
) -> Optional[bytes]:
    """Blocking call to GPT-Image-2 edit endpoint (runway internal) for style transfer."""
    url = f"{GPT_IMAGE2_URL}/openai/images/edits?api-version={GPT_IMAGE2_API_VER}"
    headers = {"api-key": api_key}
    data = {
        "model":   GPT_IMAGE2_MODEL,
        "prompt":  style_prompt,
        "n":       "1",
        "size":    GPT_IMAGE2_SIZE,
        "quality": GPT_IMAGE2_QUALITY,
    }
    try:
        resp = _requests.post(
            url,
            headers=headers,
            files={"image": ("image.png", image_bytes, "image/png")},
            data=data,
            timeout=timeout,
            proxies={"https": None, "http": None},
        )
        resp.raise_for_status()
        body = resp.json()

        if isinstance(body, dict) and "Code" in body and "data" not in body:
            _log(f"  gpt2 proxy error Code={body['Code']}: {str(body.get('Error',''))[:120]!r}")
            return None

        items = body.get("data", [])
        if not items:
            return None
        b64_json = items[0].get("b64_json", "")
        if not b64_json:
            return None
        return base64.b64decode(b64_json)

    except _requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        _log(f"  gpt2 HTTP {status}: {(e.response.text[:150] if e.response else '')!r}")
        raise
    except _requests.exceptions.RequestException as e:
        _log(f"  gpt2 network error: {type(e).__name__}: {e!r}")
        raise


async def style_image(
    img: Image.Image,
    style_name: str,
    api_key: str,
    sem: asyncio.Semaphore,
    loop: asyncio.AbstractEventLoop,
    executor: concurrent.futures.Executor,
    max_retries: int = 4,
) -> Optional[Image.Image]:
    """Return a style-transferred PIL image, or None on failure."""
    import functools

    img_resized = _resize_if_needed(img, max_side=1024)
    buf = io.BytesIO()
    img_resized.save(buf, format="PNG", optimize=True)
    image_bytes = buf.getvalue()

    style_prompt = STYLE_PROMPTS[style_name]

    for attempt in range(max_retries):
        try:
            fn = functools.partial(_gpt_image2_style_sync, image_bytes, style_prompt, api_key)
            async with sem:
                result_bytes = await loop.run_in_executor(executor, fn)

            if result_bytes is None:
                return None
            return Image.open(io.BytesIO(result_bytes)).convert("RGB")

        except _requests.exceptions.Timeout:
            _log(f"  style_image timeout (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff(attempt, base=10.0))
            else:
                return None
        except _requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 429:
                await asyncio.sleep(_backoff(attempt, base=30.0))
            elif status in (400, 403):
                return None
            else:
                if attempt < max_retries - 1:
                    await asyncio.sleep(_backoff(attempt, base=5.0))
                else:
                    return None
        except Exception as e:
            _log(f"  style_image error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e!r}")
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff(attempt, base=5.0))
            else:
                return None

    return None


# ── Step 2: CoT generation via Gemini VL ─────────────────────────────────────

_COT_SYSTEM = (
    "You are an expert at visual question answering. "
    "Given an image and a question with its correct answer, write a detailed chain-of-thought "
    "reasoning trace that leads to the answer.\n\n"
    "Requirements:\n"
    "1. Carefully observe and describe the relevant visual details in the image.\n"
    "2. Reason step-by-step, showing how the observations lead to the answer.\n"
    "3. Be specific about what you see (counts, measurements, positions, values).\n"
    "4. End with the exact answer.\n\n"
    "Format: wrap your reasoning in <think>...</think> and the final answer in <answer>...</answer>.\n"
    "The content inside <answer> must match the correct answer exactly."
)


async def generate_cot(
    img: Image.Image,
    question: str,
    answer: str,
    sem: asyncio.Semaphore,
    max_retries: int = 4,
) -> dict:
    """
    Generate a CoT trace for the given image+question+answer.
    Returns dict with keys: think, answer (both strings).
    Falls back to a minimal CoT on failure.
    """
    img_resized = _resize_if_needed(img, max_side=1568)

    prompt = (
        f"Question: {question}\n\n"
        f"Correct answer: {answer}\n\n"
        "Now write the chain-of-thought reasoning trace for this question and answer, "
        "based on what you observe in the image."
    )

    for attempt in range(max_retries):
        try:
            async with sem:
                raw = await _call_gemini(
                    prompt=prompt,
                    images=[img_resized],
                    system=_COT_SYSTEM,
                    temperature=0.3,
                    max_tokens=2048,
                    max_retries=1,
                )

            think_m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
            ans_m   = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)

            think = think_m.group(1).strip() if think_m else raw
            cot_answer = ans_m.group(1).strip() if ans_m else answer

            return {"think": think, "answer": cot_answer}

        except Exception as e:
            _log(f"  generate_cot error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e!r}")
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff(attempt, base=5.0))
            else:
                break

    # Fallback: minimal CoT
    return {
        "think": f"Looking at the image carefully to answer: {question}\n\nThe answer is {answer}.",
        "answer": answer,
    }


# ── Record builders ───────────────────────────────────────────────────────────

def make_sft_record(
    sample_key: str,
    image_path: str,
    question: str,
    think: str,
    answer: str,
) -> dict:
    assistant_content = f"<think>\n{think}\n</think>\n<answer>\n{answer}\n</answer>"
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": f"<image>{question}"},
            {"role": "assistant", "content": assistant_content},
        ],
        "images":     [image_path],
        "sample_key": sample_key,
        "stage":      "track_a_stage2",
    }


# ── JSONL helpers ─────────────────────────────────────────────────────────────

def load_stage1(path: str) -> list[dict]:
    """Load Stage 1 JSONL; keep only quality_pass=True records."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("quality_pass") and rec.get("image_path") and rec.get("original_question"):
                records.append(rec)
    return records


class _JsonlWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    async def write(self, record: dict) -> None:
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Per-record processing ─────────────────────────────────────────────────────

async def process_record(
    rec: dict,
    n_styles: int,
    api_key: str,
    loop: asyncio.AbstractEventLoop,
    executor: concurrent.futures.Executor,
    writer: _JsonlWriter,
) -> int:
    """
    Process one Stage-1 record: generate n_styles styled variants + CoT each.
    Returns count of successfully written SFT records.
    """
    qid      = rec["question_id"]
    question = rec["original_question"]
    answer   = rec["new_answer"] or rec["original_answer"]
    img_path = rec["image_path"]

    img = _load_image(img_path)
    if img is None:
        _log(f"[{qid}] image load failed: {img_path}")
        return 0

    # Select styles
    styles = STYLE_NAMES[:n_styles]

    written = 0
    for si, style_name in enumerate(styles):
        style_key = f"main_{qid}_s{si}"

        # Step 1: style transfer
        styled_img = await style_image(img, style_name, api_key, GPT_SEM, loop, executor)
        if styled_img is None:
            _log(f"[{qid}] style '{style_name}' failed, using original")
            styled_img = img

        # Save styled image
        IMG_SFT_DIR.mkdir(parents=True, exist_ok=True)
        save_path = IMG_SFT_DIR / f"{qid}_{style_name}.png"
        try:
            styled_img.save(str(save_path), format="PNG", optimize=True)
        except Exception as e:
            _log(f"[{qid}] save error for {style_name}: {e!r}")
            continue

        # Step 2: CoT generation
        cot = await generate_cot(styled_img, question, answer, GEMINI_SEM)

        # Build and write SFT record
        sft_rec = make_sft_record(
            style_key,
            str(save_path.resolve()),
            question,
            cot["think"],
            cot["answer"],
        )
        await writer.write(sft_rec)
        written += 1
        _log(f"[{qid}] style '{style_name}' → {save_path.name} written")

    return written


# ── Main ──────────────────────────────────────────────────────────────────────

async def main(
    input_path: str,
    output_file: str,
    n_styles: int,
    api_key: str,
    gemini_api_key: str = "",  # unused; Gemini key is baked into utils/gemini.py
) -> None:
    stage1_records = load_stage1(input_path)
    _log(f"Loaded {len(stage1_records)} quality_pass=True records from {input_path}")

    if not stage1_records:
        _log("No qualifying records found. Exiting.")
        return

    SFT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SFT_DIR / output_file
    writer = _JsonlWriter(output_path)

    loop     = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    total_written = 0

    async def _process_and_count(rec: dict) -> int:
        try:
            return await process_record(rec, n_styles, api_key, loop, executor, writer)
        except Exception as e:
            _log(f"[{rec.get('question_id','?')}] UNHANDLED: {traceback.format_exc()}")
            return 0

    tasks = [_process_and_count(rec) for rec in stage1_records]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    for r in results:
        if isinstance(r, int):
            total_written += r

    executor.shutdown(wait=False)

    _log(f"Done. Wrote {total_written} SFT records → {output_path}")
    _log(f"Images saved to: {IMG_SFT_DIR}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Track A Stage 2: visual diversification + CoT generation (SFT)"
    )
    parser.add_argument(
        "--input", default="outputs/rl/zerobench_track_a_stage1.jsonl",
        help="Stage 1 JSONL (quality_pass records)",
    )
    parser.add_argument(
        "--output", default="zerobench_track_a_stage2.jsonl",
        help="Output filename (written to outputs/sft/)",
    )
    parser.add_argument(
        "--n_styles", type=int, default=3,
        help="Number of style variants per question (1-4, default 3)",
    )
    parser.add_argument(
        "--gpt_image2_api_key",
        default=os.environ.get("GPT_IMAGE2_API_KEY", ""),
        help="API key for GPT-Image-2 (or set GPT_IMAGE2_API_KEY env var)",
    )
    parser.add_argument(
        "--gemini_api_key",
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="API key for Gemini (or set GEMINI_API_KEY env var)",
    )
    args = parser.parse_args()

    asyncio.run(main(
        input_path=args.input,
        output_file=args.output,
        n_styles=min(max(1, args.n_styles), 4),
        api_key=args.gpt_image2_api_key,
        gemini_api_key=args.gemini_api_key,
    ))
