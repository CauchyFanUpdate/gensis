"""
Shared infrastructure for ZeroBench Pipeline v5 (Stage 1 + Stage 2).

Includes:
- API client functions: Gemini VL, Qwen text, Qwen VL, GPT-Image-2 (with n>=1 support)
- Image preparation (resize/format conversion)
- Parquet loader
- JSONL utilities
- AsyncClient builders (separate inner-net + proxy clients)
- Common retry / logging helpers
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Optional

import concurrent.futures
import functools

import httpx
import requests as _requests


# ── Per-API global semaphores (limit concurrent calls per upstream API) ────
#
# Configurable via env. v5.1 defaults pulled back after observing 429 / NoCapacity
# in production. Internal proxy quotas are tighter than nominal:
#   Qwen 64 / Gemini 16 / GPT-Image-2 2
# Override via env if your quota tier is higher.

_GEMINI_LIMIT = int(os.environ.get("GEMINI_MAX_CONCURRENT", "16"))
_QWEN_LIMIT   = int(os.environ.get("QWEN_MAX_CONCURRENT",   "64"))
_GPT_IMG_LIMIT = int(os.environ.get("GPT_IMAGE_MAX_CONCURRENT", "2"))

# Lazy init to avoid creating semaphores at import time before event loop exists
_gemini_sem: Optional[asyncio.Semaphore] = None
_qwen_sem:   Optional[asyncio.Semaphore] = None
_gpt_img_sem: Optional[asyncio.Semaphore] = None


def _get_gemini_sem() -> asyncio.Semaphore:
    global _gemini_sem
    if _gemini_sem is None:
        _gemini_sem = asyncio.Semaphore(_GEMINI_LIMIT)
    return _gemini_sem


def _get_qwen_sem() -> asyncio.Semaphore:
    global _qwen_sem
    if _qwen_sem is None:
        _qwen_sem = asyncio.Semaphore(_QWEN_LIMIT)
    return _qwen_sem


def _get_gpt_img_sem() -> asyncio.Semaphore:
    global _gpt_img_sem
    if _gpt_img_sem is None:
        _gpt_img_sem = asyncio.Semaphore(_GPT_IMG_LIMIT)
    return _gpt_img_sem


def _backoff_with_jitter(base: float, attempt: int, jitter_frac: float = 0.4) -> float:
    """Linear backoff with random jitter to desynchronize retries."""
    wait = base * (attempt + 1)
    jitter = wait * jitter_frac * (2 * random.random() - 1)  # ±jitter_frac of wait
    return max(1.0, wait + jitter)


# ── Retry exceptions ───────────────────────────────────────────────────────

RETRYABLE: tuple = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ProxyError,   # proxy 503 (transient) — retry with backoff
)
if hasattr(httpx, "ReadTimeout"):
    RETRYABLE = RETRYABLE + (httpx.ReadTimeout,)  # type: ignore


# ── Logging ────────────────────────────────────────────────────────────────

def log(msg: str, prefix: str = "v5") -> None:
    print(f"[{prefix}] {msg}", flush=True)


# ── JSON / JSONL utilities ─────────────────────────────────────────────────

def jsonl_iter(path: str):
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    pass


def extract_json(text: str) -> Any:
    """Extract JSON from model output, stripping markdown fences and stray text."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    # Some models prepend explanation; try to find the first balanced JSON object/array
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # find first { or [ and last matching brace
        start_obj = text.find("{")
        start_arr = text.find("[")
        candidates = [c for c in (start_obj, start_arr) if c >= 0]
        if not candidates:
            raise
        start = min(candidates)
        # naive but works most of the time: take from start to last } or ]
        end = max(text.rfind("}"), text.rfind("]"))
        if end <= start:
            raise
        return json.loads(text[start:end + 1])


# ── Parquet / image utilities ──────────────────────────────────────────────

def load_parquet(path: str) -> list[dict]:
    """Load ZeroBench subquestions parquet, return list of records."""
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    records = []
    for i in range(len(table)):
        row = {col: table.column(col)[i].as_py() for col in table.schema.names}
        records.append(row)
    return records


def load_docvqa_parquets(
    data_dir: str,
    splits: Optional[list[str]] = None,
) -> list[dict]:
    """
    Load DocVQA HuggingFace-format parquet files and normalize to ZeroBench-compatible schema.

    DocVQA fields:
      questionId (str), question (str), answers (list[str]),
      image (struct{bytes: binary, path: str}), question_types (list[str]),
      docId, ucsf_document_id, ucsf_document_page_no, data_split

    Normalised output per record:
      question_id, question_text, question_answer  ← required by Stage 1 pipeline
      question_images_decoded: [{bytes: ...}]      ← compatible with get_image_bytes()
      all_answers, question_types, docId, …        ← DocVQA extras preserved in output
    """
    import pyarrow.parquet as pq
    from collections import Counter as _Counter

    if splits is None:
        splits = ["train"]

    data_dir_path = Path(data_dir)
    records: list[dict] = []
    for split in splits:
        parquet_files = sorted(data_dir_path.glob(f"{split}-*.parquet"))
        if not parquet_files:
            log(f"load_docvqa_parquets: no files for split={split!r} in {data_dir}", "docvqa")
            continue
        for pfile in parquet_files:
            log(f"load_docvqa_parquets: reading {pfile.name}", "docvqa")
            table = pq.read_table(str(pfile))
            for i in range(len(table)):
                row = {col: table.column(col)[i].as_py() for col in table.schema.names}

                # Canonical answer: most-frequent among annotators, fallback to first.
                raw_answers: list[str] = row.get("answers") or []
                stripped = [a.strip() for a in raw_answers if a and a.strip()]
                canonical = _Counter(stripped).most_common(1)[0][0] if stripped else ""

                # Image bytes from DocVQA's struct{bytes, path}.
                img_struct = row.get("image") or {}
                img_bytes: Optional[bytes] = (
                    img_struct.get("bytes") if isinstance(img_struct, dict) else None
                )

                records.append({
                    # ── ZeroBench-compatible fields ────────────────────────────────────
                    "question_id":   str(row.get("questionId", "")),
                    "question_text": row.get("question", ""),
                    "question_answer": canonical,
                    # get_image_bytes() reads question_images_decoded[0]["bytes"]
                    "question_images_decoded": [{"bytes": img_bytes}] if img_bytes else [],
                    # ── DocVQA extras (preserved in stage1 output records) ─────────────
                    "all_answers":            raw_answers,
                    "question_types":         row.get("question_types", []),
                    "data_split":             row.get("data_split", ""),
                    "docId":                  row.get("docId"),
                    "ucsf_document_id":       row.get("ucsf_document_id", ""),
                    "ucsf_document_page_no":  row.get("ucsf_document_page_no", ""),
                })
    log(f"load_docvqa_parquets: {len(records)} records from splits={splits}", "docvqa")
    return records


def get_image_bytes(row: dict) -> Optional[bytes]:
    """Extract first image bytes from parquet row."""
    images = row.get("question_images_decoded") or []
    if not images:
        return None
    first = images[0]
    if isinstance(first, dict):
        return first.get("bytes")
    return None


def prepare_image(img_bytes: bytes, max_side: int, fmt: str) -> tuple[bytes, str]:
    """Resize and convert image format. fmt: 'JPEG' | 'PNG'. Returns (bytes, mime_type)."""
    from PIL import Image
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue(), "image/jpeg"
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue(), "image/png"


# ── Gemini VL (Google native API) ──────────────────────────────────────────

GEMINI_SAFETY = [
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "OFF"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "OFF"},
]


async def chat_gemini_vl(
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    system: str,
    image_b64s: list[tuple[str, str]],   # [(b64, mime_type), ...]
    user_text: str,
    max_tokens: int = 4096,
    temperature: float = 0.3,
    disable_thinking: bool = False,
    max_retries: int = 4,
    log_prefix: str = "gemini_vl",
) -> str:
    """
    Single call to Gemini VL with one or more images.
    Returns concatenated non-thought text parts.
    """
    gen_config: dict = {
        "temperature": temperature,
        "maxOutputTokens": max_tokens,
        "topP": 0.95,
    }
    if disable_thinking:
        gen_config["thinkingConfig"] = {"thinkingBudget": 0}

    parts: list[dict] = []
    for b64, mime in image_b64s:
        parts.append({"inlineData": {"mimeType": mime, "data": b64}})
    parts.append({"text": user_text})

    payload = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": gen_config,
        "safetySettings": GEMINI_SAFETY,
    }

    sem = _get_gemini_sem()
    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.post(
                    endpoint,
                    headers={"api-key": api_key, "Content-Type": "application/json"},
                    json=payload,
                )
                resp.raise_for_status()
                body = resp.json()

            # Detect proxy-side error wrapper (200 OK but body has Code/Error)
            if "Code" in body and "Error" in body and "candidates" not in body:
                err_str = str(body.get("Error", ""))
                log(f"  Gemini VL proxy error: Code={body['Code']} | {err_str[:150]!r}", log_prefix)
                if any(s in err_str for s in ("429", "RESOURCE_EXHAUSTED")) or "quota" in err_str.lower():
                    wait = _backoff_with_jitter(25, attempt)
                    log(f"  Gemini quota, wait {wait:.1f}s (attempt {attempt + 1}/{max_retries})", log_prefix)
                    await asyncio.sleep(wait)
                    continue
                raise RuntimeError(f"Gemini proxy error: {err_str}")

            candidate = body.get("candidates", [{}])[0]
            parts_out = candidate.get("content", {}).get("parts", [])
            text_parts = [p.get("text", "") for p in parts_out if "text" in p and not p.get("thought")]
            return " ".join(text_parts).strip()

        except RETRYABLE as e:
            wait = _backoff_with_jitter(2 ** attempt, 0, jitter_frac=0.5)
            log(f"  retry {attempt + 1}/{max_retries}: {type(e).__name__}: {e!r} (wait {wait:.1f}s)", log_prefix)
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429:
                wait = _backoff_with_jitter(20, attempt)
                log(f"  HTTP 429, wait {wait:.1f}s (attempt {attempt + 1}/{max_retries})", log_prefix)
                await asyncio.sleep(wait)
            elif status in (400, 403):
                log(f"  HTTP {status}: {e.response.text[:200]}", log_prefix)
                raise
            else:
                wait = _backoff_with_jitter(5, attempt)
                log(f"  HTTP {status}, wait {wait:.1f}s", log_prefix)
                await asyncio.sleep(wait)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = _backoff_with_jitter(2 ** attempt, 0, jitter_frac=0.5)
                log(f"  error retry {attempt + 1}/{max_retries}: {type(e).__name__}: {e!r} (wait {wait:.1f}s)", log_prefix)
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Gemini VL failed after {max_retries} retries")


# ── Qwen (text + VL via OpenAI-compatible endpoint) ─────────────────────────

async def chat_qwen(
    client: httpx.AsyncClient,
    api_base: str,
    api_key: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 4096,
    temperature: float = 0.7,
    max_retries: int = 4,
    log_prefix: str = "qwen",
) -> str:
    """
    Generic Qwen chat completion call (works for both text-only and VL messages).
    For VL: messages should contain content with image_url entries.
    """
    api_base = api_base.rstrip("/")
    sem = _get_qwen_sem()
    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.post(
                    f"{api_base}/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                    },
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                resp.raise_for_status()
                body = resp.json()
            if "choices" not in body:
                wait = _backoff_with_jitter(10, attempt)
                log(f"  bad response (no 'choices'), wait {wait:.1f}s", log_prefix)
                await asyncio.sleep(wait)
                continue
            return body["choices"][0]["message"]["content"].strip()
        except RETRYABLE as e:
            wait = _backoff_with_jitter(2 ** attempt, 0, jitter_frac=0.5)
            log(f"  retry {attempt + 1}/{max_retries}: {type(e).__name__}: {e!r} (wait {wait:.1f}s)", log_prefix)
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = _backoff_with_jitter(15, attempt)
                log(f"  HTTP 429, wait {wait:.1f}s (attempt {attempt + 1}/{max_retries})", log_prefix)
                await asyncio.sleep(wait)
            else:
                log(f"  HTTP {e.response.status_code}: {e.response.text[:200]}", log_prefix)
                raise
    raise RuntimeError(f"Qwen failed after {max_retries} retries")


def qwen_text_messages(system: str, user_text: str, no_think: bool = False) -> list[dict]:
    """
    Build Qwen text messages. If no_think=True, prepend `/no_think` to the system
    prompt so Qwen 3.6-plus skips its reasoning preamble (saves 200-500 tokens,
    ~30% latency on schema-driven tasks like prose writing / verification).
    """
    if no_think:
        system = "/no_think\n" + system
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user_text},
    ]


def qwen_vl_messages(
    system: str,
    image_b64s: list[tuple[str, str]],   # [(b64, mime), ...]
    user_text: str,
    no_think: bool = False,
) -> list[dict]:
    """Build Qwen VL messages. no_think same semantics as qwen_text_messages."""
    if no_think:
        system = "/no_think\n" + system
    content: list[dict] = []
    for b64, mime in image_b64s:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{b64}"},
        })
    content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": content},
    ]


# ── GPT-Image-2 image edit (with n>=1 multi-seed) ──────────────────────────

_GPT_IMG_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=16, thread_name_prefix="gpt_img")


def _gpt_image_edit_sync(
    endpoint: str,
    api_key: str,
    image_bytes: bytes,
    prompt: str,
    n: int,
    image_mime: str,
    size: str,
    quality: str,
    timeout: float,
    api_version: str,
) -> dict:
    """Blocking requests call — run in thread executor."""
    endpoint = endpoint.rstrip("/")
    ext = "png" if image_mime == "image/png" else "jpg"

    is_standard_openai = endpoint.endswith("/v1")
    if is_standard_openai:
        # External endpoint (e.g. api.gpt.ge) — go through system proxy (reads https_proxy env)
        url = f"{endpoint}/images/edits"
        headers = {"Authorization": f"Bearer {api_key}"}
        data = {"model": "gpt-image-2-c", "prompt": prompt, "n": str(n), "size": size, "quality": quality}
        proxies = None  # let requests use env proxy automatically
    else:
        # Internal Rednote/Azure endpoint — direct connection, bypass system proxy
        url = f"{endpoint}/openai/images/edits?api-version={api_version}"
        headers = {"api-key": api_key}
        data = {"prompt": prompt, "n": str(n), "size": size, "quality": quality}
        proxies = {"https": None, "http": None}

    resp = _requests.post(
        url,
        headers=headers,
        files={"image": (f"image.{ext}", image_bytes, image_mime)},
        data=data,
        timeout=timeout,
        proxies=proxies,
    )
    resp.raise_for_status()
    return resp.json()


async def gpt_image_edit(
    client: httpx.AsyncClient,   # kept for call-site compatibility, unused
    endpoint: str,
    api_key: str,
    image_bytes: bytes,
    prompt: str,
    n: int = 1,
    image_mime: str = "image/png",
    size: str = "1024x1024",
    quality: str = "high",
    max_retries: int = 6,
    log_prefix: str = "gpt_image",
    api_version: str = "2025-04-01-preview",
    _timeout: float = 300.0,
) -> Optional[list[bytes]]:
    """
    Call GPT-Image-2 /images/edits via requests in a thread executor.
    Uses requests (not httpx) so proxy env vars (https_proxy etc.) are picked up
    automatically — matching the behaviour of test_api.py.
    Supports two endpoint styles (auto-detected by /v1 suffix):
      - Standard OpenAI: https://api.gpt.ge/v1  → Bearer auth, model=gpt-image-2-c
      - Azure/Rednote:   https://runway.devops.rednote.life → api-key header
    Returns list of image bytes (length == n) on success, or None on failure.
    """
    sem = _get_gpt_img_sem()
    loop = asyncio.get_running_loop()
    fn = functools.partial(
        _gpt_image_edit_sync,
        endpoint, api_key, image_bytes, prompt, n,
        image_mime, size, quality, _timeout, api_version,
    )

    for attempt in range(max_retries):
        try:
            async with sem:
                body = await loop.run_in_executor(_GPT_IMG_EXECUTOR, fn)

            # Proxy-wrapper error: HTTP 200 but body has {Code, Error}
            if isinstance(body, dict) and "Code" in body and "Error" in body and "data" not in body:
                err_str = str(body.get("Error", ""))
                err_low = err_str.lower()
                if any(s in err_low for s in (
                    "engineoverloaded", "too many requests", "429",
                    "resource_exhausted", "quota",
                    "nocapacity", "high demand", "exceeds the maximum",
                )):
                    wait = _backoff_with_jitter(30, attempt)
                    log(f"  proxy overload, wait {wait:.1f}s (attempt {attempt + 1}/{max_retries}) | {err_str[:120]!r}", log_prefix)
                    await asyncio.sleep(wait)
                    continue
                log(f"  proxy error Code={body['Code']} | {err_str[:200]!r}", log_prefix)
                return None

            items = body.get("data", []) if isinstance(body, dict) else []
            out: list[bytes] = []
            for item in items:
                if "b64_json" in item:
                    out.append(base64.b64decode(item["b64_json"]))
                else:
                    log(f"  unexpected response keys: {list(item.keys())}", log_prefix)
            if len(out) == 0:
                log(f"  empty data array | body keys: {list(body.keys()) if isinstance(body, dict) else type(body).__name__}", log_prefix)
                return None
            return out

        except _requests.exceptions.Timeout as e:
            wait = _backoff_with_jitter(2 ** attempt, 0, jitter_frac=0.5)
            log(f"  timeout retry {attempt + 1}/{max_retries} (wait {wait:.1f}s)", log_prefix)
            await asyncio.sleep(wait)
        except _requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            body_text = e.response.text[:300] if e.response is not None else ""
            if status == 429:
                wait = _backoff_with_jitter(25, attempt)
                log(f"  HTTP 429, wait {wait:.1f}s (attempt {attempt + 1}/{max_retries})", log_prefix)
                await asyncio.sleep(wait)
            elif status in (400, 403):
                log(f"  HTTP {status}: {body_text}", log_prefix)
                if n > 1 and "n" in body_text.lower():
                    log(f"  fallback: trying n=1 × {n} sequential", log_prefix)
                    out = []
                    for _ in range(n):
                        single = await gpt_image_edit(
                            client, endpoint, api_key, image_bytes, prompt,
                            n=1, image_mime=image_mime, size=size, quality=quality,
                            max_retries=max_retries, log_prefix=log_prefix,
                            api_version=api_version, _timeout=_timeout,
                        )
                        if single is None:
                            return None
                        out.extend(single)
                    return out
                return None
            else:
                wait = _backoff_with_jitter(5, attempt)
                log(f"  HTTP {status}, wait {wait:.1f}s | {body_text!r}", log_prefix)
                await asyncio.sleep(wait)
        except _requests.exceptions.RequestException as e:
            wait = _backoff_with_jitter(2 ** attempt, 0, jitter_frac=0.5)
            log(f"  network retry {attempt + 1}/{max_retries}: {type(e).__name__}: {e!r} (wait {wait:.1f}s)", log_prefix)
            await asyncio.sleep(wait)
        except Exception as e:
            log(f"  unexpected: {type(e).__name__}: {e!r}", log_prefix)
            return None

    log(f"  failed after {max_retries} retries", log_prefix)
    return None


# ── HTTP client builders ───────────────────────────────────────────────────

def make_clients(
    api_timeout: float = 1200.0,   # v5.1: bumped 480→1200; quality-first, avoid retries on long Gemini-thinking calls
    max_connections: int = 512,
    max_keepalive: int = 256,
) -> tuple[httpx.AsyncClient, httpx.AsyncClient]:
    """
    Build two httpx.AsyncClient instances:
    - inner_client: no proxy (Gemini / Qwen via internal endpoints)
    - proxy_client: with HTTP(S)_PROXY env (GPT-Image-2 via external endpoint)
    """
    timeout = httpx.Timeout(api_timeout, read=api_timeout, pool=60)
    limits = httpx.Limits(max_connections=max_connections, max_keepalive_connections=max_keepalive)
    proxy_url = (
        os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("HTTP_PROXY")
    )
    inner_client = httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False)
    proxy_client = httpx.AsyncClient(timeout=timeout, limits=limits, proxy=proxy_url, trust_env=False)
    return inner_client, proxy_client


# ── Async file write lock helper ───────────────────────────────────────────

class JsonlWriter:
    """Async-safe append-only JSONL writer."""
    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()

    async def write(self, record: dict) -> None:
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── v5.1: Universal axis taxonomy (19 axes, locked) ──────────────────────
#
# Stage 1's Call 1 outputs a per-image `axis_table` classifying each axis as
# free / frozen / depends. Stage 2's selector samples 4 axis valuations from
# the free axes (all aggressive, capped to natural-photo distribution).

UNIVERSAL_AXES: tuple[str, ...] = (
    # SUBJECT (5)
    "subject_count",
    "subject_class",
    "subject_color",
    "subject_material",
    "subject_specific_appearance",
    # ARRANGEMENT (5)
    "subject_to_surface_relation",
    "subject_to_subject_arrangement",
    "subject_position_in_frame",
    "subject_orientation",
    "subject_size_relative",
    # SCENE (3) — Tier-A, main visual diversity drivers
    "background_setting",
    "background_clutter",
    "non_subject_objects",
    # SURFACE (2) — Tier-A
    "surface_class",
    "surface_appearance",
    # CAMERA (1, yaw+pitch merged)
    "camera_distance",
    "camera_angle",
    # LIGHTING (1, direction+intensity+temp merged, natural-capped)
    "lighting_quality",
    # TEXT (1)
    "visible_text_content",
)

# Default safety per axis (Stage 1 LLM uses as prior, then per-image adjusts).
# `depends`: must be classified by LLM based on test_point.
AXIS_TIER_DEFAULTS: dict[str, str] = {
    # Tier-A: almost always free
    "background_setting":      "free",
    "background_clutter":      "free",
    "non_subject_objects":     "free",
    "surface_class":           "free",
    "surface_appearance":      "free",
    # Tier-B: depends on test_point
    "camera_distance":         "depends",
    "camera_angle":            "depends",
    "lighting_quality":        "free",         # natural-capped, see NATURAL_CAPS
    "subject_color":           "depends",
    "subject_material":        "depends",
    "subject_specific_appearance": "depends",
    "subject_size_relative":   "depends",
    "subject_position_in_frame": "depends",
    # Tier-C: default frozen, only free with strong reason
    "subject_count":           "depends",      # frozen for counting; free else
    "subject_class":           "depends",      # via slot mechanism
    "subject_to_surface_relation":    "depends",
    "subject_to_subject_arrangement": "depends",
    "subject_orientation":     "depends",
    "visible_text_content":    "depends",      # frozen for text_reading
}

# Natural-photo cap for axes where LLM tends to push out of distribution.
# These strings are inlined into the Call 1 prompt so the LLM knows the
# allowable aggressive_value range.
NATURAL_CAPS: dict[str, str] = {
    "lighting_quality": (
        "natural range only: indoor daylight / indoor warm lamp / golden hour / overcast / morning sun. "
        "FORBIDDEN: extreme dim, blown-out, dramatic high contrast, single-color stage light, "
        "deep blue / orange / purple color filters, surreal weather (heavy fog/rain/blizzard)."
    ),
    "camera_distance": (
        "photographer-natural range: close-up / medium / wide. No extreme macro, no fish-eye-wide."
    ),
    "camera_angle": (
        "natural range: frontal / 3/4 / side / mild high (~15°) / mild low (~15°). "
        "FORBIDDEN: extreme top-down, extreme worm's-eye, dutch tilt > 20°."
    ),
    "background_setting": (
        "any real-world location (warehouse, kitchen, market, studio, store, office, garden). "
        "FORBIDDEN: surreal scenes, fantasy environments, illustrative backdrops."
    ),
}

# Question-type → axis safety prior. Stage 1 LLM uses as starting point then refines.
# Maps test_point.structural_pattern → which axes should be FROZEN (rest default).
TEST_POINT_FROZEN_PRIORS: dict[str, tuple[str, ...]] = {
    "spatial_ordering":          ("subject_orientation", "subject_position_in_frame",
                                  "camera_angle", "subject_to_subject_arrangement"),
    "category_filtering":        ("subject_class", "subject_color"),
    "real_vs_depicted":          ("subject_class", "subject_material"),
    "lookalike_confusion":       ("subject_class", "subject_specific_appearance"),
    "partial_visibility":        ("subject_position_in_frame", "camera_angle", "camera_distance"),
    "dual_value_disambiguation": ("visible_text_content",),
    "composition_state":         ("subject_specific_appearance", "subject_size_relative"),
    "text_in_context":           ("visible_text_content", "camera_distance"),
}

# Test-point → readability check template. Stage 1 fills in slot names.
# Used by Stage 2 Verifier as the "is the question still answerable" check.
READABILITY_TEMPLATES: dict[str, str] = {
    "counting":          "Is each <S> cleanly visually separable from the others (no overlap into a blob, no items hidden behind others)?",
    "spatial_ordering":  "Is the left-to-right (or specified) order of the <S_plural> clearly visible from this viewpoint?",
    "spatial_relation":  "Are both subjects of the relation fully visible such that their relative position is unambiguous?",
    "comparison":        "Are the compared subjects both shown at a scale that allows direct comparison of the queried attribute?",
    "text_reading":      "Is the relevant text in the image clearly legible and unaltered from the previous edit?",
    "existence":         "Is there a clear, unambiguous answer to whether the queried subject is present in the scene?",
    "attribute":         "Is the <S>'s queried attribute clearly visible and unambiguous in this image?",
    "default":           "Is the test-point structure of this question clearly preserved and answerable from this image?",
}


# ── Resume helper ─────────────────────────────────────────────────────────

def load_done_keys(jsonl_path: Path, key_field: str = "sample_key") -> set[str]:
    """Scan a jsonl file and return the set of sample_keys already written (for resume)."""
    done: set[str] = set()
    if not jsonl_path.is_file():
        return done
    for rec in jsonl_iter(str(jsonl_path)):
        k = rec.get(key_field, "")
        if k:
            done.add(k)
    return done
