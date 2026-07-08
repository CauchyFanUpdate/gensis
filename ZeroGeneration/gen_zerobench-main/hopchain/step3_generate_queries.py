#!/usr/bin/env python3
"""
HopChain Step 3 — Generate Multi-Hop Visual Reasoning Queries

Given the original image and a combination of segmented instances (with crops),
call Gemini VL to design multi-hop questions that chain perception steps across
all instances and terminate in a numeric answer.

Output query dict schema:
    {
        "query":      str,          # the full question text
        "hop_count":  int,          # number of reasoning hops (4–6)
        "answer":     int | float,  # unambiguous numeric ground truth
        "reasoning":  str,          # step-by-step reasoning trace
    }
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
from typing import Optional, Union

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.gemini import call_gemini as _call_gemini

from PIL import Image


# ── Constants ─────────────────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-3.0-pro"

_SYSTEM_PROMPT = "You are an expert at designing multi-hop visual reasoning questions."

# ── Multi-hop prompt template ─────────────────────────────────────────────────

MULTIHOP_PROMPT = """\
You are given:
  - The ORIGINAL IMAGE (first image attachment)
  - {n_crops} CROPPED PATCHES of specific object instances (subsequent image attachments)

The cropped instances are:
{instance_descriptions}

Your task: Design {n_queries} independent multi-hop visual reasoning questions about the original image.

STRICT REQUIREMENTS for EACH question:
1. DEPENDENCY CHAIN: The question must involve ALL listed instances in a logical dependency chain
   (e.g., A → B → C → D). Each hop must use the result of the previous hop.
2. HOP STRUCTURE: Alternate between two levels within each question:
   - Level 1 (single-object perception): count, color, size, orientation, text on object, etc.
   - Level 2 (multi-object reasoning): compare, rank, sum, difference, ratio, spatial relation, etc.
3. HOP COUNT: Each question must have between 4 and 6 hops (reasoning steps).
4. NUMERIC TERMINAL ANSWER: The FINAL answer must be a specific, unambiguous NUMBER (integer or float).
   Examples of good terminals: "what is the total count?", "what is the difference?", "what is their sum?"
5. INSTANCE REFERENCES: Refer to each instance ONLY by its visible attributes + spatial position
   (e.g., "the red car on the left", "the person wearing a hat in the background").
   NEVER reference bounding boxes or crop indices.
6. INDEPENDENCE: The {n_queries} questions must be independent of each other (different reasoning chains).

OUTPUT FORMAT — return ONLY a JSON array with exactly {n_queries} objects:
[
  {{
    "query": "<full multi-hop question text — write it as a single question with all hops made explicit>",
    "hop_count": <integer 4-6>,
    "answer": <numeric value — integer or float, not a string>,
    "reasoning": "<step-by-step chain showing how each hop builds on the last, ending at the numeric answer>"
  }},
  ...
]

Do NOT include markdown fences, commentary, or any text outside the JSON array.
"""


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[hopchain/step3] {msg}", flush=True)


# ── Back-off ──────────────────────────────────────────────────────────────────

def _backoff(attempt: int, base: float = 5.0) -> float:
    wait = base * (2 ** attempt)
    jitter = wait * 0.3 * (2 * random.random() - 1)
    return max(2.0, wait + jitter)


# ── Image helpers ─────────────────────────────────────────────────────────────

def _pil_to_b64(img: Image.Image, max_side: int = 1568, fmt: str = "PNG") -> tuple[str, str]:
    """Encode PIL image as base64. Returns (b64_str, mime_type)."""
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    if fmt == "JPEG":
        img.save(buf, format="JPEG", quality=90)
        mime = "image/jpeg"
    else:
        img.save(buf, format="PNG", optimize=True)
        mime = "image/png"
    return base64.b64encode(buf.getvalue()).decode(), mime


def _image_content_part(img: Image.Image, max_side: int = 1568) -> dict:
    """Build an OpenAI-compatible image_url content part."""
    b64, mime = _pil_to_b64(img, max_side=max_side)
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime};base64,{b64}"},
    }


# ── Answer parsing ────────────────────────────────────────────────────────────

def _is_numeric(value) -> bool:
    """Return True if *value* is an int or float (not bool, not string)."""
    if isinstance(value, bool):
        return False
    return isinstance(value, (int, float))


def _parse_answer(raw) -> Optional[Union[int, float]]:
    """
    Coerce *raw* to a numeric type.

    Accepts: int, float, numeric string (e.g. "3", "3.5").
    Returns None if not convertible.
    """
    if isinstance(raw, bool):
        return None
    if isinstance(raw, (int, float)):
        return raw
    if isinstance(raw, str):
        raw = raw.strip()
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
    return None


# ── Core async function ───────────────────────────────────────────────────────

async def generate_multihop_queries(
    image: Image.Image,
    instances: list[dict],
    n_queries: int = 3,
    sem: Optional[asyncio.Semaphore] = None,
    max_retries: int = 4,
    **kwargs,
) -> list[dict]:
    """
    Ask Gemini VL to generate *n_queries* multi-hop reasoning questions about
    *image*, involving all *instances*.

    Args:
        image:     Original PIL image.
        instances: List of instance dicts from step2 (must contain 'crop',
                   'instance_id', 'category', 'bbox').
        n_queries: Number of independent questions to request per call.
        sem:       asyncio.Semaphore for rate-limiting.
        max_retries: Maximum retries on transient errors.

    Returns:
        List of query dicts (possibly fewer than n_queries if Gemini returns
        fewer valid numeric-answer items).  Empty list on total failure.
    """
    if sem is None:
        sem = asyncio.Semaphore(8)
    if not instances:
        _log("No instances provided — cannot generate queries")
        return []

    # Build instance description text for the prompt
    desc_lines: list[str] = []
    for i, inst in enumerate(instances):
        bbox = inst.get("bbox", [])
        x0, y0, x1, y1 = bbox if len(bbox) == 4 else (0, 0, 0, 0)
        cx = (x0 + x1) // 2
        cy = (y0 + y1) // 2
        img_w, img_h = image.size
        # Describe position in natural language
        h_pos = "left" if cx < img_w / 3 else ("center" if cx < 2 * img_w / 3 else "right")
        v_pos = "top" if cy < img_h / 3 else ("middle" if cy < 2 * img_h / 3 else "bottom")
        desc_lines.append(
            f"  Crop {i+1} — instance_id='{inst['instance_id']}', "
            f"category='{inst['category']}', "
            f"position='{v_pos}-{h_pos}' of the original image"
        )
    instance_descriptions = "\n".join(desc_lines)

    # Build image list: original image + one crop per instance
    all_images: list[Image.Image] = [image]
    for inst in instances:
        crop: Image.Image = inst["crop"]
        # Resize crops to reasonable size
        w, h = crop.size
        if max(w, h) > 512:
            scale = 512 / max(w, h)
            crop = crop.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
        all_images.append(crop)

    # Fill prompt template
    user_text = MULTIHOP_PROMPT.format(
        n_crops=len(instances),
        instance_descriptions=instance_descriptions,
        n_queries=n_queries,
    )

    for attempt in range(max_retries):
        try:
            async with sem:
                raw = await _call_gemini(
                    prompt=user_text,
                    images=all_images,
                    system=_SYSTEM_PROMPT,
                    temperature=0.7,
                    max_tokens=8192,
                    max_retries=1,
                )

            # Strip markdown fences
            raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.IGNORECASE)
            raw = re.sub(r"\s*```$", "", raw)

            # Extract the JSON array (may be inside prose)
            m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
            if m:
                raw = m.group(0)

            parsed = json.loads(raw)
            if not isinstance(parsed, list):
                raise ValueError(f"Expected JSON list, got {type(parsed).__name__}")

            # Validate and filter entries
            valid_queries: list[dict] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                query = item.get("query", "")
                hop_count = item.get("hop_count")
                raw_answer = item.get("answer")
                reasoning = item.get("reasoning", "")

                if not query:
                    continue

                answer = _parse_answer(raw_answer)
                if answer is None:
                    _log(
                        f"  discarding query (non-numeric answer={raw_answer!r}): "
                        f"{query[:60]!r}"
                    )
                    continue

                # Clamp hop_count to [4, 6]
                if not isinstance(hop_count, int):
                    try:
                        hop_count = int(hop_count)
                    except (TypeError, ValueError):
                        hop_count = 4
                hop_count = max(4, min(6, hop_count))

                valid_queries.append({
                    "query":     query,
                    "hop_count": hop_count,
                    "answer":    answer,
                    "reasoning": reasoning,
                })

            _log(
                f"generate_multihop_queries: {len(parsed)} raw → "
                f"{len(valid_queries)} valid"
            )
            return valid_queries

        except json.JSONDecodeError as e:
            _log(
                f"JSON parse error (attempt {attempt+1}/{max_retries}): "
                f"{e} | raw={raw[:300]!r}"
            )
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


# ── CLI (quick smoke-test) ────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="HopChain Step 3: Generate multi-hop queries via Gemini VL"
    )
    parser.add_argument("image", help="Path to input image")
    parser.add_argument(
        "--n_queries", type=int, default=3,
        help="Number of queries to generate (default 3)",
    )
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")

    # Build dummy instances (for smoke-test)
    w, h = img.size
    dummy_instances = [
        {
            "instance_id": "person_0",
            "category": "person",
            "bbox": [0, 0, w // 3, h // 2],
            "score": 0.9,
            "crop": img.crop((0, 0, w // 3, h // 2)),
            "mask": None,
        },
        {
            "instance_id": "car_0",
            "category": "car",
            "bbox": [w // 3, 0, 2 * w // 3, h // 2],
            "score": 0.85,
            "crop": img.crop((w // 3, 0, 2 * w // 3, h // 2)),
            "mask": None,
        },
    ]

    sem = asyncio.Semaphore(4)
    queries = asyncio.run(
        generate_multihop_queries(img, dummy_instances, n_queries=args.n_queries, sem=sem)
    )
    print(json.dumps(queries, indent=2, ensure_ascii=False))
