#!/usr/bin/env python3
"""
ZeroBench MAIN — Track A, Stage 1: Surgical Answer-Changing Edit with Subquestion Verification

Pipeline per main question:
  Call 1 [Gemini VL]     → scene analysis JSON: scene_description, current_answer,
                            feasible_edit, new_answer, edit_instructions
  Call 2 [GPT-Image-2]   → edited image (minimal change that alters the answer)
  Call 3 [Qwen VL × N]   → subquestion verification on the new image
                            (ALL subquestions must pass → quality_pass=True)

Key difference from v5 zerobench_pipeline_chenlifeng:
  - Input: ZeroBench MAIN questions (not sub-questions)
  - Verification: subquestion-based (not double-blind cross-VLM agreement)
    For each subquestion of the parent question, call Qwen VL and check the
    answer matches the expected subquestion answer. All must pass.

Output JSONL (outputs/rl/zerobench_track_a_stage1.jsonl):
  {
    "sample_key": "main_{question_id}",
    "question_id": str,
    "original_question": str,
    "original_answer": str,
    "edit_description": str,
    "new_answer": str,
    "quality_pass": bool,
    "subq_verification": [...],
    "image_path": str,
    "stage": "track_a_stage1"
  }

Images saved to: outputs/rl/track_a_stage1_images/{question_id}.png
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import traceback
from pathlib import Path
from typing import Optional

import sys
import os as _os
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx as _httpx
import requests as _requests
from PIL import Image
from utils.gemini import call_gemini as _call_gemini


# ── Constants ─────────────────────────────────────────────────────────────────

GEMINI_ENDPOINT = "https://runway.devops.rednote.life/openai/google/v1:generateContent"
GEMINI_MODEL    = "gemini-3.0-pro"
GEMINI_API_KEY  = "YOUR_GEMINI_API_KEY"

QWEN_BASE_URL   = "https://runway.devops.xiaohongshu.com/openai/qwen/v1"
QWEN_MODEL      = "qwen3-vl-plus"

GPT_IMAGE2_URL      = "https://runway.devops.rednote.life"
GPT_IMAGE2_API_VER  = "2025-04-01-preview"
GPT_IMAGE2_MODEL    = "gpt-image-2-c"
GPT_IMAGE2_SIZE     = "auto"
GPT_IMAGE2_QUALITY  = "auto"

GEMINI_SEM = asyncio.Semaphore(16)
QWEN_SEM   = asyncio.Semaphore(64)
GPT_SEM    = asyncio.Semaphore(2)

DATA_DIR   = Path("/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/zerobench")
OUTPUT_DIR = Path("outputs/rl")
IMAGE_DIR  = OUTPUT_DIR / "track_a_stage1_images"


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg: str, prefix: str = "track_a") -> None:
    print(f"[{prefix}] {msg}", flush=True)


# ── Retry / backoff ───────────────────────────────────────────────────────────

import random as _random

def _backoff(attempt: int, base: float = 5.0) -> float:
    """Exponential backoff with jitter: base * 2^attempt ± 30%."""
    wait = base * (2 ** attempt)
    jitter = wait * 0.3 * (2 * _random.random() - 1)
    return max(2.0, wait + jitter)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_zerobench_main() -> list[dict]:
    """
    Load ZeroBench MAIN questions from parquet.

    Parquet columns:
      question_id, question_text, question_answer,
      question_images_decoded (list of struct{bytes, path}),
      question_images (list of relative str paths), image_attribution

    Returns list of dicts with:
      question_id (str), question_text (str), question_answer (str),
      images (list[PIL.Image]), question_images (list[str])
    """
    import pyarrow.parquet as pq

    parquet_path = DATA_DIR / "data" / "zerobench-00000-of-00001.parquet"
    table = pq.read_table(str(parquet_path))

    records = []
    for i in range(len(table)):
        row = {col: table.column(col)[i].as_py() for col in table.schema.names}

        # Decode images from bytes stored in question_images_decoded
        pil_images: list[Image.Image] = []
        for img_struct in (row.get("question_images_decoded") or []):
            img_bytes: Optional[bytes] = None
            if isinstance(img_struct, dict):
                img_bytes = img_struct.get("bytes")
            elif isinstance(img_struct, bytes):
                img_bytes = img_struct
            if img_bytes:
                try:
                    pil_images.append(Image.open(io.BytesIO(img_bytes)).convert("RGB"))
                except Exception as e:
                    _log(f"  warning: could not decode image for q{row.get('question_id')}: {e}")

        records.append({
            "question_id":      str(row.get("question_id", "")),
            "question_text":    row.get("question_text", ""),
            "question_answer":  str(row.get("question_answer", "")),
            "images":           pil_images,
            "question_images":  row.get("question_images") or [],
        })

    _log(f"Loaded {len(records)} main questions from {parquet_path}")
    return records


def load_zerobench_subquestions() -> dict[str, list[dict]]:
    """
    Load ZeroBench subquestions from parquet.

    Subquestion IDs have the form "{parent_id}_{index}" (e.g., "1_1", "1_2").
    Parent ID is extracted by splitting on the last underscore.

    Returns dict: parent_id (str) → list of {question_id, question_text, question_answer}
    """
    import pyarrow.parquet as pq

    parquet_path = DATA_DIR / "data" / "zerobench_subquestions-00000-of-00001.parquet"
    table = pq.read_table(str(parquet_path))

    subq_map: dict[str, list[dict]] = {}
    for i in range(len(table)):
        row = {col: table.column(col)[i].as_py() for col in table.schema.names}
        qid = str(row.get("question_id", ""))
        # Extract parent ID: "1_2" → "1", "10_3" → "10"
        parts = qid.rsplit("_", 1)
        parent_id = parts[0] if len(parts) == 2 else qid

        subq_map.setdefault(parent_id, []).append({
            "question_id":    qid,
            "question_text":  row.get("question_text", ""),
            "question_answer": str(row.get("question_answer", "")),
        })

    _log(f"Loaded subquestions for {len(subq_map)} parent IDs from {parquet_path}")
    return subq_map


# ── Image helpers ─────────────────────────────────────────────────────────────

def _pil_to_b64(img: Image.Image, fmt: str = "PNG") -> tuple[str, str]:
    """Convert PIL image to base64 string. Returns (b64_str, mime_type)."""
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return base64.b64encode(buf.getvalue()).decode(), mime


def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> tuple[bytes, str]:
    """Convert PIL image to bytes. Returns (raw_bytes, mime_type)."""
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return buf.getvalue(), mime


def _resize_if_needed(img: Image.Image, max_side: int = 1024) -> Image.Image:
    """Resize image so its longest side is at most max_side pixels."""
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    scale = max_side / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)


# ── Call 1: Scene analysis (Gemini VL) ───────────────────────────────────────

_CALL1_SYSTEM = (
    "You are an image-editing analysis assistant for a data generation pipeline. "
    "Given an image and a visual question with its ground-truth answer, your task is to:\n"
    "1. Describe the scene in the image.\n"
    "2. Identify the visual element(s) that directly determine the answer.\n"
    "3. Propose ONE minimal, realistic image edit that changes the answer to the question.\n"
    "   - The edit should be as small as possible while still changing the answer.\n"
    "   - The rest of the image should remain as close to the original as possible.\n"
    "   - The edited image should look like a natural, real photograph.\n"
    "   - Do NOT propose edits that make the image look fake, surreal, or heavily altered.\n\n"
    "Return ONLY a JSON object with exactly these fields:\n"
    "{\n"
    '  "scene_description": "<1-2 sentences describing the scene, focusing on the answer-relevant region>",\n'
    '  "current_answer": "<echo the given answer verbatim>",\n'
    '  "answer_region": "<where in the image the answer comes from>",\n'
    '  "feasible_edit": "<one sentence: what single minimal change to make, and why it changes the answer>",\n'
    '  "new_answer": "<what the answer would be after the edit, as a short string>",\n'
    '  "edit_instructions": "<2-4 sentences of concrete, spatial instructions for GPT-Image-2 to execute the edit. '
    'Describe exactly what to change, where, and what to keep unchanged. '
    'Write as natural prose an art director would use. No bullet points.>"\n'
    "}\n\n"
    "Constraints:\n"
    "- The edit MUST change the answer. Identity-preserving changes are not allowed.\n"
    "- Prefer counting changes (remove/add one object), color changes, or text value changes.\n"
    "- The new_answer must be mechanically derivable from the edit (e.g., count-1, new color name).\n"
    "- Output ONLY the JSON object. No markdown, no commentary, no fences."
)


async def call1_scene_analysis(
    question: str,
    images: list[Image.Image],
    sem: asyncio.Semaphore,
    max_retries: int = 5,
) -> Optional[dict]:
    """
    Call Gemini VL to analyze the image and propose a minimal answer-changing edit.

    Returns a dict with keys:
      scene_description, current_answer, answer_region,
      feasible_edit, new_answer, edit_instructions
    Returns None on failure.
    """
    if not images:
        _log("  call1: no images provided")
        return None

    resized_images = [_resize_if_needed(img, max_side=1568) for img in images]
    prompt = (
        f"Question: {question}\n"
        "Answer: (provided below — use this as the ground truth)\n\n"
        "Analyze this image and propose ONE minimal edit that changes the answer."
    )

    for attempt in range(max_retries):
        try:
            async with sem:
                raw = await _call_gemini(
                    prompt=prompt,
                    images=resized_images,
                    system=_CALL1_SYSTEM,
                    temperature=0.3,
                    max_tokens=8192,
                    max_retries=1,
                )
            # Strip markdown fences if present
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError(f"Expected dict, got {type(result).__name__}")
            required = ("scene_description", "current_answer", "feasible_edit", "new_answer", "edit_instructions")
            missing = [k for k in required if not result.get(k)]
            if missing:
                raise ValueError(f"Missing required keys: {missing}")
            return result
        except json.JSONDecodeError as e:
            _log(f"  call1 JSON parse error (attempt {attempt+1}/{max_retries}): {e} | raw={raw[:200]!r}")
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(_backoff(attempt, base=3.0))
        except Exception as e:
            _log(f"  call1 error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e!r}")
            if attempt == max_retries - 1:
                return None
            await asyncio.sleep(_backoff(attempt, base=5.0))

    return None


# ── Call 2: Edit image (GPT-Image-2) ─────────────────────────────────────────

def _gpt_image2_edit_sync(
    image_bytes: bytes,
    image_mime: str,
    prompt: str,
    api_key: str,
    size: str = GPT_IMAGE2_SIZE,
    quality: str = GPT_IMAGE2_QUALITY,
    timeout: float = 300.0,
) -> Optional[bytes]:
    """
    Blocking call to GPT-Image-2 edit endpoint via requests (runway internal proxy).
    Returns raw PNG bytes on success, None on failure.
    """
    ext = "png" if image_mime == "image/png" else "jpg"
    url = f"{GPT_IMAGE2_URL}/openai/images/edits?api-version={GPT_IMAGE2_API_VER}"
    headers = {"api-key": api_key}
    data = {
        "model":   GPT_IMAGE2_MODEL,
        "prompt":  prompt,
        "n":       "1",
        "size":    size,
        "quality": quality,
    }
    try:
        resp = _requests.post(
            url,
            headers=headers,
            files={"image": (f"image.{ext}", image_bytes, image_mime)},
            data=data,
            timeout=timeout,
            proxies={"https": None, "http": None},
        )
        resp.raise_for_status()
        body = resp.json()

        # Handle proxy-wrapper error (HTTP 200 but body has Code/Error fields)
        if isinstance(body, dict) and "Code" in body and "Error" in body and "data" not in body:
            err_str = str(body.get("Error", ""))
            _log(f"  call2 proxy error Code={body['Code']}: {err_str[:150]!r}")
            return None

        items = body.get("data", [])
        if not items:
            _log(f"  call2: empty data array | body keys={list(body.keys())}")
            return None
        b64_json = items[0].get("b64_json", "")
        if not b64_json:
            _log(f"  call2: no b64_json in response item | item keys={list(items[0].keys())}")
            return None
        return base64.b64decode(b64_json)

    except _requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 0
        body_text = (e.response.text[:200] if e.response is not None else "")
        _log(f"  call2 HTTP {status}: {body_text!r}")
        raise
    except _requests.exceptions.RequestException as e:
        _log(f"  call2 network error: {type(e).__name__}: {e!r}")
        raise


async def call2_edit_image(
    original_image_b64: str,
    edit_instructions: str,
    sem: asyncio.Semaphore,
    api_key: str,
    max_retries: int = 5,
) -> Optional[Image.Image]:
    """
    Call GPT-Image-2 edit endpoint to generate an edited image.
    Returns a PIL.Image on success, None on failure.
    """
    import concurrent.futures
    import functools

    image_bytes = base64.b64decode(original_image_b64)

    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)

    for attempt in range(max_retries):
        try:
            fn = functools.partial(
                _gpt_image2_edit_sync,
                image_bytes,
                "image/png",
                edit_instructions,
                api_key,
            )
            async with sem:
                result_bytes = await loop.run_in_executor(executor, fn)

            if result_bytes is None:
                return None
            img = Image.open(io.BytesIO(result_bytes)).convert("RGB")
            return img

        except _requests.exceptions.Timeout:
            _log(f"  call2 timeout (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff(attempt, base=10.0))
            else:
                return None
        except _requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 429:
                wait = _backoff(attempt, base=30.0)
                _log(f"  call2 HTTP 429, wait {wait:.1f}s (attempt {attempt+1}/{max_retries})")
                await asyncio.sleep(wait)
            elif status in (400, 403):
                _log(f"  call2 HTTP {status}: unrecoverable, giving up")
                return None
            else:
                if attempt < max_retries - 1:
                    await asyncio.sleep(_backoff(attempt, base=5.0))
                else:
                    return None
        except Exception as e:
            _log(f"  call2 error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e!r}")
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff(attempt, base=5.0))
            else:
                return None

    return None


# ── Call 3: Subquestion verification (Qwen VL) ───────────────────────────────

_VERIFY_SYSTEM = (
    "/no_think\n"
    "You are a visual question answering assistant. "
    "Answer the given question based ONLY on what you can directly observe in the image. "
    "Give a short, direct answer. Do not explain or add caveats. "
    "Your answer should be a single number, word, or short phrase — match the expected format of the question."
)


async def _verify_single_subquestion(
    new_image_b64: str,
    question: str,
    expected_answer: str,
    qwen_api_key: str,
    sem: asyncio.Semaphore,
    max_retries: int = 4,
) -> dict:
    """
    Call Qwen VL on new_image with a single subquestion via httpx directly.
    Returns {question, expected, got, pass}.
    """
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{new_image_b64}"},
        },
        {"type": "text", "text": question},
    ]
    messages = [
        {"role": "system", "content": _VERIFY_SYSTEM},
        {"role": "user",   "content": content},
    ]
    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "max_tokens": 256,
        "temperature": 0.0,
    }
    headers = {"Authorization": f"Bearer {qwen_api_key}"}

    for attempt in range(max_retries):
        try:
            async with sem:
                async with _httpx.AsyncClient(trust_env=False, timeout=120.0) as client:
                    resp = await client.post(
                        f"{QWEN_BASE_URL}/chat/completions",
                        json=payload,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    body = resp.json()

            if "choices" not in body:
                _log(f"  verify subq: no 'choices' in response (attempt {attempt+1}/{max_retries}): {str(body)[:200]}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(_backoff(attempt, base=5.0))
                    continue
                return {"question": question, "expected": expected_answer, "got": "NO_CHOICES", "pass": False}

            got = body["choices"][0]["message"]["content"].strip()
            passed = (got.strip().lower() == expected_answer.strip().lower())
            return {
                "question": question,
                "expected": expected_answer,
                "got": got,
                "pass": passed,
            }
        except Exception as e:
            _log(f"  verify subq error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e!r}")
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff(attempt, base=3.0))
            else:
                return {
                    "question": question,
                    "expected": expected_answer,
                    "got": f"ERROR: {type(e).__name__}: {e!r}",
                    "pass": False,
                }

    return {"question": question, "expected": expected_answer, "got": "TIMEOUT", "pass": False}


async def call3_verify_subquestions(
    new_image: Image.Image,
    subquestions: list[dict],
    qwen_api_key: str,
    sem: asyncio.Semaphore,
) -> dict:
    """
    Verify all subquestions of the parent question on the new (edited) image.

    For each subquestion, call Qwen VL and compare the answer (case-insensitive strip).
    Returns:
      {
        "all_pass": bool,   # True if ALL subquestions answered correctly
        "results": [        # per-subquestion results
          {"question": str, "expected": str, "got": str, "pass": bool}
        ]
      }
    """
    if not subquestions:
        return {"all_pass": True, "results": []}

    # Encode new image once
    img_resized = _resize_if_needed(new_image, max_side=1024)
    new_image_b64, _ = _pil_to_b64(img_resized, fmt="PNG")

    # Run all subquestion verifications concurrently
    tasks = [
        _verify_single_subquestion(
            new_image_b64,
            sq["question_text"],
            sq["question_answer"],
            qwen_api_key,
            sem,
        )
        for sq in subquestions
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Normalise results (exceptions become failed entries)
    clean_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            clean_results.append({
                "question": subquestions[i]["question_text"],
                "expected": subquestions[i]["question_answer"],
                "got": f"EXCEPTION: {r!r}",
                "pass": False,
            })
        else:
            clean_results.append(r)

    all_pass = all(r["pass"] for r in clean_results)
    return {"all_pass": all_pass, "results": clean_results}


# ── Main per-question processing ──────────────────────────────────────────────

async def process_question(
    item: dict,
    subq_map: dict[str, list[dict]],
    qwen_api_key: str,
    gpt_image2_api_key: str,
) -> dict:
    """
    Process one ZeroBench main question through the 3-call pipeline.

    Returns a record dict with all output fields.
    quality_pass=False is returned (not raised) on any failure.
    """
    question_id   = item["question_id"]
    question_text = item["question_text"]
    question_answer = item["question_answer"]
    images        = item["images"]

    _log(f"[{question_id}] starting (q={question_text[:60]!r}, ans={question_answer!r})")

    base_record = {
        "sample_key":        f"main_{question_id}",
        "question_id":       question_id,
        "original_question": question_text,
        "original_answer":   question_answer,
        "edit_description":  "",
        "new_answer":        "",
        "quality_pass":      False,
        "subq_verification": [],
        "image_path":        "",
        "stage":             "track_a_stage1",
    }

    # ── Call 1: Scene analysis (Gemini VL) ────────────────────────────────
    try:
        analysis = await call1_scene_analysis(
            question_text, images, GEMINI_SEM
        )
    except Exception as e:
        _log(f"[{question_id}] call1 exception: {type(e).__name__}: {e!r}")
        base_record["error"] = f"call1_exception: {type(e).__name__}: {e!r}"
        return base_record

    if analysis is None:
        _log(f"[{question_id}] call1 failed (None)")
        base_record["error"] = "call1_failed"
        return base_record

    edit_instructions = analysis.get("edit_instructions", "")
    new_answer_designed = analysis.get("new_answer", "")
    edit_description = analysis.get("feasible_edit", "")
    _log(f"[{question_id}] call1 ok | edit={edit_description[:80]!r} | new_ans={new_answer_designed!r}")

    # ── Call 2: Generate edited image (GPT-Image-2) ───────────────────────
    if not images:
        _log(f"[{question_id}] no images, skip call2")
        base_record["error"] = "no_images"
        return base_record

    # Use the first image (main image)
    src_img = images[0]
    src_img_for_gpt = _resize_if_needed(src_img, max_side=1024)
    src_b64, _ = _pil_to_b64(src_img_for_gpt, fmt="PNG")

    try:
        new_image = await call2_edit_image(
            src_b64, edit_instructions, GPT_SEM, api_key=gpt_image2_api_key
        )
    except Exception as e:
        _log(f"[{question_id}] call2 exception: {type(e).__name__}: {e!r}")
        base_record["error"] = f"call2_exception: {type(e).__name__}: {e!r}"
        base_record["edit_description"] = edit_description
        base_record["new_answer"] = new_answer_designed
        return base_record

    if new_image is None:
        _log(f"[{question_id}] call2 failed (None)")
        base_record["error"] = "call2_failed"
        base_record["edit_description"] = edit_description
        base_record["new_answer"] = new_answer_designed
        return base_record

    # Save edited image
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    image_path = IMAGE_DIR / f"{question_id}.png"
    try:
        new_image.save(str(image_path), format="PNG", optimize=True)
    except Exception as e:
        _log(f"[{question_id}] image save error: {e!r}")
        base_record["error"] = f"image_save_error: {e!r}"
        base_record["edit_description"] = edit_description
        base_record["new_answer"] = new_answer_designed
        return base_record

    _log(f"[{question_id}] call2 ok | saved to {image_path}")

    # ── Call 3: Subquestion verification (Qwen VL) ────────────────────────
    subquestions = subq_map.get(question_id, [])
    if not subquestions:
        _log(f"[{question_id}] no subquestions found for parent_id={question_id!r}")

    try:
        verification = await call3_verify_subquestions(
            new_image, subquestions, qwen_api_key, QWEN_SEM
        )
    except Exception as e:
        _log(f"[{question_id}] call3 exception: {type(e).__name__}: {e!r}")
        verification = {"all_pass": False, "results": [], "error": str(e)}

    all_pass = verification.get("all_pass", False)
    results  = verification.get("results", [])

    n_pass  = sum(1 for r in results if r.get("pass"))
    n_total = len(results)
    _log(
        f"[{question_id}] call3 done | subq pass={n_pass}/{n_total} | "
        f"quality_pass={all_pass}"
    )

    return {
        "sample_key":        f"main_{question_id}",
        "question_id":       question_id,
        "original_question": question_text,
        "original_answer":   question_answer,
        "edit_description":  edit_description,
        "new_answer":        new_answer_designed,
        "quality_pass":      all_pass,
        "subq_verification": results,
        "image_path":        str(image_path),
        "stage":             "track_a_stage1",
        # Extra detail fields for debugging
        "scene_description": analysis.get("scene_description", ""),
        "edit_instructions": edit_instructions,
    }


# ── JSONL writer ──────────────────────────────────────────────────────────────

class _JsonlWriter:
    """Async-safe append-only JSONL writer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()

    async def write(self, record: dict) -> None:
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── Main entry point ──────────────────────────────────────────────────────────

async def main(
    n_questions: Optional[int] = None,
    output_file: str = "zerobench_track_a_stage1.jsonl",
    gpt_image2_api_key: str = "",
    gemini_api_key: str = "",
    qwen_api_key: str = "",
) -> None:
    """
    Main async pipeline entry point.

    Loads ZeroBench MAIN questions and subquestions, processes each question
    through the 3-call pipeline, writes results to OUTPUT_DIR/output_file as JSONL,
    and prints statistics.

    Args:
        n_questions:      If set, only process the first N questions.
        output_file:      Output JSONL filename (relative to OUTPUT_DIR).
        gpt_image2_api_key: API key for GPT-Image-2.
        gemini_api_key:   API key for Gemini VL.
        qwen_api_key:     API key for Qwen VL.
    """
    # Resolve API keys from env if not provided
    if not gpt_image2_api_key:
        gpt_image2_api_key = os.environ.get("GPT_IMAGE2_API_KEY", "")
    if not gemini_api_key:
        gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not qwen_api_key:
        qwen_api_key = os.environ.get("QWEN_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    # Gemini uses utils.gemini.call_gemini (httpx direct, trust_env=False)
    # Qwen uses httpx directly in _verify_single_subquestion
    # GPT-Image-2 uses requests — no client needed here

    # Load data
    main_questions = load_zerobench_main()
    subq_map       = load_zerobench_subquestions()

    # Optionally limit to first N questions
    if n_questions is not None and n_questions > 0:
        main_questions = main_questions[:n_questions]
        _log(f"Limiting to first {n_questions} questions")

    _log(f"Processing {len(main_questions)} questions")
    if not gpt_image2_api_key:
        _log("WARNING: GPT_IMAGE2_API_KEY is empty")

    # Prepare output
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / output_file
    writer = _JsonlWriter(output_path)

    # Process all questions concurrently (semaphores control per-API concurrency)
    async def _process_and_write(item: dict) -> dict:
        try:
            record = await process_question(
                item, subq_map, qwen_api_key, gpt_image2_api_key
            )
        except Exception as e:
            _log(f"[{item['question_id']}] UNHANDLED EXCEPTION: {traceback.format_exc()}")
            record = {
                "sample_key":        f"main_{item['question_id']}",
                "question_id":       item["question_id"],
                "original_question": item["question_text"],
                "original_answer":   item["question_answer"],
                "edit_description":  "",
                "new_answer":        "",
                "quality_pass":      False,
                "subq_verification": [],
                "image_path":        "",
                "stage":             "track_a_stage1",
                "error":             f"unhandled: {type(e).__name__}: {e!r}",
            }
        await writer.write(record)
        return record

    tasks = [_process_and_write(item) for item in main_questions]
    all_records = await asyncio.gather(*tasks, return_exceptions=True)

    # Print stats
    total       = len(main_questions)
    pass_count  = sum(
        1 for r in all_records
        if isinstance(r, dict) and r.get("quality_pass")
    )
    fail_count  = total - pass_count

    _log(f"Done! Processed {total} questions.")
    _log(f"  quality_pass: {pass_count}/{total} ({100*pass_count/max(total,1):.1f}%)")
    _log(f"  quality_fail: {fail_count}/{total}")
    _log(f"  Output JSONL: {output_path}")
    _log(f"  Output images: {IMAGE_DIR}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ZeroBench Track A Stage 1: Surgical edit + subquestion verification"
    )
    parser.add_argument(
        "--n", type=int, default=None,
        help="Process only the first N questions (default: all 100)",
    )
    parser.add_argument(
        "--output", type=str, default="zerobench_track_a_stage1.jsonl",
        help="Output JSONL filename (written to outputs/rl/). Default: zerobench_track_a_stage1.jsonl",
    )
    parser.add_argument(
        "--gpt_image2_api_key", type=str,
        default=os.environ.get("GPT_IMAGE2_API_KEY", ""),
        help="API key for GPT-Image-2 (or set GPT_IMAGE2_API_KEY env var)",
    )
    parser.add_argument(
        "--gemini_api_key", type=str,
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="API key for Gemini VL (or set GEMINI_API_KEY env var)",
    )
    parser.add_argument(
        "--qwen_api_key", type=str,
        default=os.environ.get("QWEN_API_KEY", ""),
        help="API key for Qwen VL (or set QWEN_API_KEY env var)",
    )
    args = parser.parse_args()

    asyncio.run(main(
        n_questions=args.n,
        output_file=args.output,
        gpt_image2_api_key=args.gpt_image2_api_key,
        gemini_api_key=args.gemini_api_key,
        qwen_api_key=args.qwen_api_key,
    ))
