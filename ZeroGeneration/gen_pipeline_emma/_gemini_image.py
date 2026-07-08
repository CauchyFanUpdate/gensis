"""
Gemini image generation/editing utility — drop-in replacement for gpt_image_edit.

Model: gemini-3-pro-image-preview
Endpoint: https://runway.devops.rednote.life/openai/google/v1:generateContent
Auth: api-key header
"""

from __future__ import annotations

import asyncio
import base64
import os
import random
from typing import Optional

import httpx


GEMINI_IMAGE_ENDPOINT = os.environ.get(
    "GEMINI_IMAGE_ENDPOINT",
    "https://runway.devops.rednote.life/openai/google/v1:generateContent",
)
GEMINI_IMAGE_API_KEY = os.environ.get("GEMINI_IMAGE_API_KEY", "YOUR_GEMINI_IMAGE_API_KEY")

_GEMINI_IMG_LIMIT = int(os.environ.get("GEMINI_IMG_MAX_CONCURRENT", "64"))
_gemini_img_sem: Optional[asyncio.Semaphore] = None


def _get_gemini_img_sem() -> asyncio.Semaphore:
    global _gemini_img_sem
    if _gemini_img_sem is None:
        _gemini_img_sem = asyncio.Semaphore(_GEMINI_IMG_LIMIT)
    return _gemini_img_sem


def _backoff(base: float, attempt: int) -> float:
    wait = base * (attempt + 1)
    jitter = wait * 0.4 * (2 * random.random() - 1)
    return max(1.0, wait + jitter)


SAFETY_OFF = [
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "OFF"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "OFF"},
]


def _build_payload(image_bytes: bytes, image_mime: str, prompt: str) -> dict:
    b64 = base64.b64encode(image_bytes).decode()
    return {
        "contents": [{
            "role": "user",
            "parts": [
                {"inlineData": {"mimeType": image_mime, "data": b64}},
                {"text": prompt},
            ],
        }],
        "generationConfig": {
            "temperature": 1,
            "maxOutputTokens": 32768,
            "responseModalities": ["TEXT", "IMAGE"],
            "topP": 0.95,
            "imageConfig": {
                "aspectRatio": "1:1",
                "imageSize": "1K",
                "imageOutputOptions": {"mimeType": "image/png"},
                "personGeneration": "ALLOW_ALL",
            },
        },
        "safetySettings": SAFETY_OFF,
    }


def _extract_images(body: dict) -> list[bytes]:
    out: list[bytes] = []
    for candidate in body.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            inline = part.get("inlineData")
            if inline and "data" in inline:
                out.append(base64.b64decode(inline["data"]))
    return out


def log(msg: str, prefix: str = "gemini_img") -> None:
    print(f"[{prefix}] {msg}", flush=True)


async def gemini_image_edit(
    client: httpx.AsyncClient,
    endpoint: str,
    api_key: str,
    image_bytes: bytes,
    prompt: str,
    n: int = 1,
    image_mime: str = "image/png",
    max_retries: int = 20,
    log_prefix: str = "gemini_img",
    **_kwargs,
) -> Optional[list[bytes]]:
    """
    Gemini image edit — same call signature as gpt_image_edit.

    Sends the source image + prompt to the Gemini generateContent endpoint and
    returns a list of edited image bytes (length == n).  The API generates one
    image per call, so n > 1 is satisfied by n sequential calls.
    Returns None on permanent failure.
    """
    sem = _get_gemini_img_sem()
    payload = _build_payload(image_bytes, image_mime, prompt)

    async def _single_call(attempt_offset: int) -> Optional[bytes]:
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

                # Proxy-level error (HTTP 200 but body has Code/Error)
                if "Code" in body and "Error" in body and "candidates" not in body:
                    err_str = str(body.get("Error", ""))
                    if any(s in err_str.lower() for s in (
                        "429", "resource_exhausted", "quota", "overload",
                        "too many requests", "nocapacity", "high demand",
                    )):
                        wait = _backoff(25, attempt)
                        log(f"proxy overload, wait {wait:.1f}s (attempt {attempt+1}/{max_retries}) | {err_str[:100]!r}", log_prefix)
                        await asyncio.sleep(wait)
                        continue
                    log(f"proxy error Code={body['Code']} | {err_str[:200]!r}", log_prefix)
                    return None

                imgs = _extract_images(body)
                if imgs:
                    return imgs[0]

                # No image in response — may be a transient issue
                finish = body.get("candidates", [{}])[0].get("finishReason", "")
                log(f"no image in response (finishReason={finish!r}), attempt {attempt+1}/{max_retries}", log_prefix)
                await asyncio.sleep(_backoff(5, attempt))

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                if status == 429:
                    wait = _backoff(25, attempt)
                    log(f"HTTP 429, wait {wait:.1f}s (attempt {attempt+1}/{max_retries})", log_prefix)
                    await asyncio.sleep(wait)
                elif status in (400, 403):
                    log(f"HTTP {status}: {e.response.text[:300]}", log_prefix)
                    return None
                else:
                    wait = _backoff(5, attempt)
                    log(f"HTTP {status}, wait {wait:.1f}s", log_prefix)
                    await asyncio.sleep(wait)
            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError,
                    httpx.ReadError, httpx.ProxyError) as e:
                wait = _backoff(2 ** min(attempt, 4), 0, )
                log(f"network retry {attempt+1}/{max_retries}: {type(e).__name__} (wait {wait:.1f}s)", log_prefix)
                await asyncio.sleep(wait)
            except Exception as e:
                log(f"unexpected: {type(e).__name__}: {e!r}", log_prefix)
                if attempt < max_retries - 1:
                    await asyncio.sleep(_backoff(3, attempt))
                else:
                    return None

        log(f"failed after {max_retries} retries", log_prefix)
        return None

    if n == 1:
        img = await _single_call(0)
        return [img] if img is not None else None

    # n > 1: run sequentially (Gemini generates one image per call)
    results: list[bytes] = []
    for i in range(n):
        img = await _single_call(i)
        if img is None:
            return None
        results.append(img)
    return results
