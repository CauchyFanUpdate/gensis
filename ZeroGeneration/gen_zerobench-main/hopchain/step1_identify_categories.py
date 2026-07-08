#!/usr/bin/env python3
"""
HopChain Step 1 — Identify Semantic Object Categories

For each image, call Gemini VL to list all distinct semantic object categories
visible in the scene. Returns a clean list of category strings suitable for
SAM3 text-prompted segmentation in Step 2.

Output shape:
    list[str]  e.g. ["person", "car", "bicycle", "traffic light"]
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import random
import re
import sys
import traceback
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.gemini import call_gemini as _call_gemini

from PIL import Image


# ── Constants ─────────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-3.0-pro"

MAX_CATEGORIES = 12

# Categories that are scene-level / not meaningfully segmentable
_UNSEGMENTABLE = frozenset({
    "sky", "background", "ground", "floor", "ceiling",
    "wall", "air", "atmosphere", "surface", "pavement",
    "road surface", "landscape", "scenery", "environment",
    "outdoor", "indoor", "space", "area",
})

_SYSTEM_PROMPT = "You are an expert visual analyst."

_USER_PROMPT = (
    "List all distinct semantic object categories visible in this image. "
    "Return ONLY a JSON array of category name strings. "
    'Example: ["person", "car", "tree"]. '
    "Be specific but use common English nouns."
)


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[hopchain/step1] {msg}", flush=True)


# ── Retry / back-off ──────────────────────────────────────────────────────────

def _backoff(attempt: int, base: float = 5.0) -> float:
    """Exponential back-off with ±30 % jitter."""
    wait = base * (2 ** attempt)
    jitter = wait * 0.3 * (2 * random.random() - 1)
    return max(2.0, wait + jitter)


# ── Image helper ──────────────────────────────────────────────────────────────

def _pil_to_b64(img: Image.Image, max_side: int = 1568) -> tuple[str, str]:
    """Resize if needed, encode as PNG base64. Returns (b64_str, mime)."""
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), "image/png"


# ── Core async function ───────────────────────────────────────────────────────

async def identify_categories(
    image: Image.Image,
    sem: asyncio.Semaphore,
    max_retries: int = 4,
) -> list[str]:
    """
    Call Gemini VL to identify semantic object categories in *image*.

    Returns a list of up to MAX_CATEGORIES category name strings, filtered to
    exclude scene-level tokens that cannot be segmented with SAM3.
    Returns an empty list on total failure.
    """
    for attempt in range(max_retries):
        try:
            async with sem:
                raw = await _call_gemini(
                    prompt=_USER_PROMPT,
                    images=[image],
                    system=_SYSTEM_PROMPT,
                    temperature=0.2,
                    max_tokens=4096,
                    max_retries=1,
                )

            # Strip optional markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)

            # Extract JSON array (may be embedded in prose)
            m = re.search(r"\[.*?\]", raw, flags=re.DOTALL)
            if m:
                raw = m.group(0)

            categories: list = json.loads(raw)
            if not isinstance(categories, list):
                raise ValueError(f"Expected JSON list, got {type(categories).__name__}")

            # Normalise and filter
            cleaned: list[str] = []
            seen: set[str] = set()
            for item in categories:
                if not isinstance(item, str):
                    continue
                cat = item.strip().lower()
                if not cat:
                    continue
                if cat in _UNSEGMENTABLE:
                    continue
                # Skip multi-word phrases that look like scene descriptors
                if len(cat.split()) > 3:
                    continue
                if cat not in seen:
                    seen.add(cat)
                    cleaned.append(cat)

            result = cleaned[:MAX_CATEGORIES]
            _log(f"identify_categories → {result}")
            return result

        except json.JSONDecodeError as e:
            _log(f"JSON parse error (attempt {attempt+1}/{max_retries}): {e} | raw={raw[:200]!r}")
            if attempt == max_retries - 1:
                return []
            await asyncio.sleep(_backoff(attempt, base=3.0))

        except Exception as e:
            _log(
                f"error (attempt {attempt+1}/{max_retries}): "
                f"{type(e).__name__}: {e!r}"
            )
            if attempt == max_retries - 1:
                return []
            await asyncio.sleep(_backoff(attempt, base=5.0))

    return []


# ── Batch helper ──────────────────────────────────────────────────────────────

async def batch_identify_categories(
    image_paths: list[str],
    concurrency: int = 8,
    gemini_api_key: str = "",
    max_retries: int = 4,
) -> dict[str, list[str]]:
    """
    Identify categories for multiple images concurrently.

    Args:
        image_paths:    List of absolute image file paths.
        concurrency:    Maximum number of in-flight Gemini calls at once.
        gemini_api_key: API key (falls back to GEMINI_API_KEY env var).
        max_retries:    Retries per image on transient failures.

    Returns:
        Mapping  image_path → list[category_str].
        Failed images map to an empty list.
    """
    sem = asyncio.Semaphore(concurrency)
    results: dict[str, list[str]] = {}

    async def _process_one(path: str) -> None:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            _log(f"Cannot open image {path}: {e!r}")
            results[path] = []
            return
        try:
            cats = await identify_categories(img, sem, max_retries=max_retries)
        except Exception as e:
            _log(f"Unhandled error for {path}: {traceback.format_exc()}")
            cats = []
        results[path] = cats

    tasks = [_process_one(p) for p in image_paths]
    await asyncio.gather(*tasks, return_exceptions=True)
    return results


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="HopChain Step 1: Identify object categories in images via Gemini VL"
    )
    parser.add_argument(
        "image_paths", nargs="+", help="Path(s) to input image(s)"
    )
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="Max concurrent Gemini calls (default 8)",
    )
    parser.add_argument(
        "--gemini_api_key", type=str,
        default=os.environ.get("GEMINI_API_KEY", ""),
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )
    args = parser.parse_args()

    output = asyncio.run(
        batch_identify_categories(
            args.image_paths,
            concurrency=args.concurrency,
            gemini_api_key=args.gemini_api_key,
        )
    )
    for path, cats in output.items():
        print(f"{path}: {cats}")
