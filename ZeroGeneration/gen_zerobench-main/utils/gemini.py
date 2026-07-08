"""
Gemini API helper (runway internal endpoint, native Gemini format).

Supports both text-only and vision-language (VL) calls.
Uses httpx directly with trust_env=False to bypass system proxy.
"""

from __future__ import annotations

import asyncio
import base64
import io
from typing import Optional

import httpx
from PIL import Image

GEMINI_ENDPOINT = "https://runway.devops.rednote.life/openai/google/v1:generateContent"
GEMINI_MODEL    = "gemini-3.0-pro"
GEMINI_API_KEY  = "YOUR_GEMINI_API_KEY"

_SAFETY = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in (
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


def _pil_to_b64(img: Image.Image, max_side: int = 1568) -> tuple[str, str]:
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode(), "image/png"


async def call_gemini(
    prompt: str,
    images: Optional[list[Image.Image]] = None,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 4096,
    api_key: str = GEMINI_API_KEY,
    endpoint: str = GEMINI_ENDPOINT,
    max_retries: int = 5,
) -> str:
    """
    Call Gemini (text-only or VL). Returns the response text.

    Args:
        prompt:      User prompt text.
        images:      Optional list of PIL images for VL calls.
        system:      Optional system instruction (prepended to user turn).
        temperature: Sampling temperature.
        max_tokens:  Max output tokens.
    """
    parts: list[dict] = []
    if images:
        for img in images:
            b64, mime = _pil_to_b64(img)
            parts.append({"inlineData": {"mimeType": mime, "data": b64}})
    parts.append({"text": prompt})

    payload: dict = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": 0.95,
        },
        "safetySettings": _SAFETY,
    }
    if system:
        payload["systemInstruction"] = {"parts": [{"text": system}]}

    headers = {"api-key": api_key, "Content-Type": "application/json"}

    async with httpx.AsyncClient(trust_env=False, timeout=180.0) as client:
        for attempt in range(max_retries):
            try:
                resp = await client.post(endpoint, headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()

                # Proxy error wrapper (HTTP 200 but Code/Error instead of candidates)
                if "Code" in body and "Error" in body and "candidates" not in body:
                    err = str(body.get("Error", ""))
                    if any(s in err for s in ("429", "RESOURCE_EXHAUSTED", "quota", "overload")):
                        wait = min(30 * (attempt + 1), 180)
                        await asyncio.sleep(wait)
                        continue
                    raise RuntimeError(f"Gemini proxy error: {err[:300]}")

                candidates = body.get("candidates", [])
                if not candidates:
                    raise RuntimeError(f"Gemini returned no candidates: {str(body)[:300]}")
                content = candidates[0].get("content", {})
                parts_out = content.get("parts", [])
                if not parts_out:
                    # Thinking model consumed all tokens before output — retry
                    if attempt < max_retries - 1:
                        await asyncio.sleep(5)
                        continue
                    return ""
                return " ".join(
                    p.get("text", "") for p in parts_out
                    if "text" in p and not p.get("thought")
                ).strip()

            except (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError):
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (2 ** attempt))
                else:
                    raise

    return ""
