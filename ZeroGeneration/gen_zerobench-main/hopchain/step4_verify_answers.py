#!/usr/bin/env python3
"""
HopChain Step 4 — Verify Answers & Filter Difficulty

Two-stage quality gate:

  Stage A — Correctness verification
    Call Gemini (n_votes=2) + Qwen (n_votes=1) independently.
    Require ≥ 2 of 3 votes to match the expected answer (± 0.5 tolerance).

  Stage B — Difficulty filter
    Call weak Qwen model n_samples times.
    Discard sample if the weak model gets 100 % correct (too easy).

Only samples that pass BOTH stages are written to the output JSONL.

Output record schema (per verified sample):
    {
        "image_path":     str,
        "query":          str,
        "answer":         int | float,
        "hop_count":      int,
        "instances_used": [instance_id, ...],
        "verified":       True,
        "stage":          "hopchain"
    }
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import random
import re
import sys
import traceback
from collections import Counter
from pathlib import Path
from typing import Optional, Union

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.gemini import call_gemini as _call_gemini

import httpx as _httpx
from PIL import Image


# ── Constants ─────────────────────────────────────────────────────────────────

GEMINI_MODEL  = "gemini-3.0-pro"
QWEN_BASE_URL = "https://runway.devops.xiaohongshu.com/openai/qwen/v1"
QWEN_MODEL    = "qwen3-vl-plus"

_VERIFY_SYSTEM = (
    "/no_think\n"
    "You are a visual question answering assistant. "
    "Look at the image carefully and answer the given question. "
    "Your answer must be a SINGLE NUMBER (integer or decimal). "
    "Do not explain. Do not add units or words. Output the number only."
)


# ── Logging ───────────────────────────────────────────────────────────────────

def _log(msg: str) -> None:
    print(f"[hopchain/step4] {msg}", flush=True)


# ── Back-off ──────────────────────────────────────────────────────────────────

def _backoff(attempt: int, base: float = 5.0) -> float:
    wait = base * (2 ** attempt)
    jitter = wait * 0.3 * (2 * random.random() - 1)
    return max(2.0, wait + jitter)


# ── Image helper ──────────────────────────────────────────────────────────────

def _pil_to_b64(img: Image.Image, max_side: int = 1024) -> tuple[str, str]:
    """Encode PIL image as PNG base64. Returns (b64_str, mime)."""
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), "image/png"


# ── Numeric parsing ───────────────────────────────────────────────────────────

def _parse_number(text: str) -> Optional[Union[int, float]]:
    """
    Extract the first numeric value from *text*.

    Handles integers, decimals, and numbers embedded in short phrases.
    Returns None if no number is found.
    """
    text = text.strip()
    # Try direct cast first
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        pass
    # Extract first number from text
    m = re.search(r"-?\d+(?:\.\d+)?", text)
    if m:
        candidate = m.group(0)
        try:
            return int(candidate) if "." not in candidate else float(candidate)
        except ValueError:
            pass
    return None


def _answers_match(
    got: Optional[Union[int, float]],
    expected: Union[int, float],
    tolerance: float = 0.5,
) -> bool:
    """Return True if *got* is within *tolerance* of *expected*."""
    if got is None:
        return False
    try:
        return abs(float(got) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False


# ── Single VLM verification call ─────────────────────────────────────────────

async def _single_qwen_vote(
    image_b64: str,
    mime: str,
    query: str,
    qwen_api_key: str,
    sem: asyncio.Semaphore,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> Optional[Union[int, float]]:
    """
    Send one Qwen VL call via httpx and return the parsed numeric answer, or None on failure.
    """
    content = [
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{image_b64}"},
        },
        {"type": "text", "text": query},
    ]
    messages = [
        {"role": "system", "content": _VERIFY_SYSTEM},
        {"role": "user",   "content": content},
    ]
    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "max_tokens": 64,
        "temperature": temperature,
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
                _log(f"_single_qwen_vote: no 'choices' in response (attempt {attempt+1}/{max_retries}): {str(body)[:200]}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(_backoff(attempt, base=5.0))
                    continue
                return None

            raw = body["choices"][0]["message"]["content"].strip()
            return _parse_number(raw)
        except Exception as e:
            _log(
                f"_single_qwen_vote error (attempt {attempt+1}/{max_retries}): "
                f"{type(e).__name__}: {e!r}"
            )
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff(attempt, base=3.0))
    return None


async def _single_gemini_vote(
    image: Image.Image,
    query: str,
    sem: asyncio.Semaphore,
    temperature: float = 0.7,
    max_retries: int = 3,
) -> Optional[Union[int, float]]:
    """Send one Gemini VL call and return the parsed numeric answer."""
    for attempt in range(max_retries):
        try:
            async with sem:
                raw = await _call_gemini(
                    prompt=query,
                    images=[image],
                    system=_VERIFY_SYSTEM,
                    temperature=temperature,
                    max_tokens=1024,
                    max_retries=1,
                )
            return _parse_number(raw)
        except Exception as e:
            _log(f"_single_gemini_vote error (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e!r}")
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff(attempt, base=3.0))
    return None


# ── Public verify_query ───────────────────────────────────────────────────────

async def verify_query(
    image: Image.Image,
    query: str,
    expected_answer: Union[int, float],
    qwen_api_key: str,
    sem: asyncio.Semaphore,
    n_votes: int = 3,
    tolerance: float = 0.5,
) -> dict:
    """
    Ask *client* (model=*model*) *n_votes* times, then evaluate majority vote.

    Returns:
        {
            "pass":       bool,         # majority_answer == expected_answer (±tolerance)
            "votes":      list,         # parsed answer from each vote (None if failed)
            "majority":   int|float|None,
            "confidence": float,        # fraction of votes that matched expected
        }
    """
    b64, mime = _pil_to_b64(image)

    tasks = [
        _single_qwen_vote(b64, mime, query, qwen_api_key, sem, temperature=0.7)
        for _ in range(n_votes)
    ]
    raw_votes: list[Optional[Union[int, float]]] = await asyncio.gather(
        *tasks, return_exceptions=False
    )

    # Find majority among non-None votes
    valid_votes = [v for v in raw_votes if v is not None]

    majority: Optional[Union[int, float]] = None
    if valid_votes:
        # Round floats to 1 decimal for grouping, then pick most common
        rounded = [round(v, 1) for v in valid_votes]
        counter = Counter(rounded)
        majority_rounded, _ = counter.most_common(1)[0]
        # Map back to original value
        for v in valid_votes:
            if round(v, 1) == majority_rounded:
                majority = v
                break

    n_correct = sum(1 for v in valid_votes if _answers_match(v, expected_answer, tolerance))
    confidence = n_correct / n_votes  # denominator: total attempts, not just valid

    passed = _answers_match(majority, expected_answer, tolerance)

    return {
        "pass":       passed,
        "votes":      list(raw_votes),
        "majority":   majority,
        "confidence": confidence,
    }


# ── filter_difficulty ─────────────────────────────────────────────────────────

async def filter_difficulty(
    image: Image.Image,
    query: str,
    expected_answer: Union[int, float],
    qwen_api_key: str,
    sem: asyncio.Semaphore,
    n_samples: int = 4,
    easy_threshold: float = 1.0,
    tolerance: float = 0.5,
) -> bool:
    """
    Returns True if the query is NOT too easy (keep), False if too easy (discard).

    A query is considered too easy when the weak model (Qwen) answers it correctly
    in >= easy_threshold fraction of independent calls.
    """
    b64, mime = _pil_to_b64(image)

    tasks = [
        _single_qwen_vote(
            b64, mime, query, qwen_api_key, sem,
            temperature=0.0,   # greedy for difficulty check
        )
        for _ in range(n_samples)
    ]
    answers: list = await asyncio.gather(*tasks, return_exceptions=False)

    n_correct = sum(
        1 for a in answers if _answers_match(a, expected_answer, tolerance)
    )
    correct_rate = n_correct / n_samples

    too_easy = correct_rate >= easy_threshold
    _log(
        f"filter_difficulty: correct={n_correct}/{n_samples} "
        f"({correct_rate:.0%}) → {'DISCARD (easy)' if too_easy else 'KEEP'}"
    )
    return not too_easy  # True = keep


# ── run_verification_pipeline ─────────────────────────────────────────────────

async def run_verification_pipeline(
    image: Image.Image,
    queries: list[dict],
    qwen_api_key: str,
    sem: asyncio.Semaphore,
    gemini_sem: Optional[asyncio.Semaphore] = None,
    tolerance: float = 0.5,
) -> list[dict]:
    """
    Run the two-stage quality gate for all *queries*.

    Stage A: Gemini×2 + Qwen×1; require ≥ 2 correct → verified=True.
    Stage B: Qwen×4 at temperature 0; discard if 100 % correct.

    Returns the filtered list with added fields:
      verified, difficulty_ok, final_answer.
    """
    if gemini_sem is None:
        gemini_sem = sem

    async def _gate_one(q: dict) -> Optional[dict]:
        query          = q["query"]
        expected       = q["answer"]
        instance_label = f"hop_count={q.get('hop_count','?')} answer={expected}"

        # ── Stage A: correctness ──────────────────────────────────────────────
        try:
            # Gemini votes (2×) via native Gemini API
            gemini_votes = await asyncio.gather(*[
                _single_gemini_vote(image, query, gemini_sem, temperature=0.7)
                for _ in range(2)
            ])
            # Qwen vote (1×) via httpx
            qwen_result = await verify_query(
                image, query, expected,
                qwen_api_key=qwen_api_key,
                sem=sem,
                n_votes=1,
                tolerance=tolerance,
            )
        except Exception as e:
            _log(f"verify_query exception for [{instance_label}]: {e!r}")
            return None

        # Collect all 3 votes and check ≥2/3 correct
        all_votes = list(gemini_votes) + qwen_result["votes"]
        n_correct = sum(
            1 for v in all_votes if _answers_match(v, expected, tolerance)
        )
        verified = n_correct >= 2
        _log(
            f"Stage A [{instance_label}]: {n_correct}/3 correct → "
            f"{'PASS' if verified else 'FAIL'}"
        )

        if not verified:
            q_out = dict(q)
            q_out["verified"]      = False
            q_out["difficulty_ok"] = None
            q_out["final_answer"]  = expected
            return q_out   # include for record-keeping but not in final output

        # ── Stage B: difficulty filter ────────────────────────────────────────
        try:
            not_too_easy = await filter_difficulty(
                image, query, expected,
                qwen_api_key=qwen_api_key,
                sem=sem,
                n_samples=4,
                easy_threshold=1.0,
                tolerance=tolerance,
            )
        except Exception as e:
            _log(f"filter_difficulty exception for [{instance_label}]: {e!r}")
            not_too_easy = True   # default: keep on error

        q_out = dict(q)
        q_out["verified"]      = True
        q_out["difficulty_ok"] = not_too_easy
        q_out["final_answer"]  = expected
        return q_out

    tasks = [_gate_one(q) for q in queries]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    filtered: list[dict] = []
    for r in results:
        if isinstance(r, Exception):
            _log(f"Unhandled exception in _gate_one: {r!r}")
            continue
        if r is None:
            continue
        if r.get("verified") and r.get("difficulty_ok"):
            filtered.append(r)

    _log(
        f"run_verification_pipeline: {len(queries)} in → "
        f"{len(filtered)} kept (verified + not too easy)"
    )
    return filtered


# ── JSONL writer ──────────────────────────────────────────────────────────────

class _JsonlWriter:
    """Async-safe append-only JSONL writer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = asyncio.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    async def write(self, record: dict) -> None:
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ── End-to-end main entry point ───────────────────────────────────────────────

async def main(
    image_dir: str,
    categories_file: str,
    output_file: str,
    gemini_api_key: str = "",
    qwen_api_key: str = "",
    concurrency: int = 8,
    n_queries_per_combo: int = 3,
    combo_size: int = 4,
    max_combos: int = 10,
    sam3_device: str = "cuda",
    worker_id: int = 0,
    n_workers: int = 1,
) -> None:
    """
    End-to-end HopChain pipeline.

    Steps:
      1. Load pre-computed categories from *categories_file* (JSON: {image_path: [cats]})
         or run Step 1 on-the-fly if the file doesn't exist.
      2. For each image, run Step 2 (SAM3 segmentation).
      3. For each instance combination, run Step 3 (query generation).
      4. Run Step 4 (verification + difficulty filter).
      5. Write passing records to *output_file* as JSONL.

    Args:
        image_dir:          Directory containing input images (PNG/JPG).
        categories_file:    Path to JSON mapping image paths → category lists.
                            Created by Step 1 if it doesn't exist yet.
        output_file:        Path to output JSONL file.
        gemini_api_key:     Gemini API key (falls back to GEMINI_API_KEY env var).
        qwen_api_key:       Qwen API key (falls back to QWEN_API_KEY env var).
        concurrency:        Max concurrent VLM calls.
        n_queries_per_combo: Queries to request per instance combination.
        combo_size:         Instances per combination (Step 2).
        max_combos:         Max combinations per image.
        sam3_device:        Torch device for SAM3.
    """
    # Resolve API keys
    if not qwen_api_key:
        qwen_api_key = os.environ.get(
            "QWEN_API_KEY", os.environ.get("OPENAI_API_KEY", "")
        )
    # Gemini uses utils.gemini.call_gemini directly (httpx, trust_env=False)
    # Qwen uses httpx directly in _single_qwen_vote

    gemini_sem = asyncio.Semaphore(concurrency)
    qwen_sem   = asyncio.Semaphore(concurrency * 4)

    writer = _JsonlWriter(Path(output_file))

    # ── Collect image paths ───────────────────────────────────────────────────
    image_dir_path = Path(image_dir)
    image_extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    image_paths = sorted(
        p for p in image_dir_path.iterdir()
        if p.suffix.lower() in image_extensions
    )
    _log(f"Found {len(image_paths)} images in {image_dir}")

    if not image_paths:
        _log("No images found — exiting")
        return

    # Worker sharding: each worker handles every n_workers-th image
    if n_workers > 1:
        image_paths = image_paths[worker_id::n_workers]
        _log(f"Worker {worker_id}/{n_workers}: handling {len(image_paths)} images")

    # ── Step 1: Identify categories ───────────────────────────────────────────
    categories_path = Path(categories_file)
    if categories_path.exists():
        _log(f"Loading categories from {categories_path}")
        with open(categories_path, encoding="utf-8") as f:
            categories_map: dict[str, list[str]] = json.load(f)
    else:
        _log(f"Running Step 1 (identify_categories) on {len(image_paths)} images…")
        from hopchain.step1_identify_categories import batch_identify_categories  # type: ignore
        categories_map = await batch_identify_categories(
            [str(p) for p in image_paths],
            concurrency=concurrency,
            gemini_api_key=gemini_api_key,
        )
        categories_path.parent.mkdir(parents=True, exist_ok=True)
        with open(categories_path, "w", encoding="utf-8") as f:
            json.dump(categories_map, f, ensure_ascii=False, indent=2)
        _log(f"Categories saved to {categories_path}")

    # ── Step 2: Load SAM3 model ───────────────────────────────────────────────
    _log("Loading SAM3 model…")
    from hopchain.step2_segment_instances import load_sam3_model, segment_instances, select_instance_combinations  # type: ignore
    from hopchain.step3_generate_queries import generate_multihop_queries  # type: ignore
    sam_model, sam_processor = load_sam3_model(device=sam3_device)
    _log("SAM3 ready")

    # ── Per-image pipeline ────────────────────────────────────────────────────
    total_written = 0

    for img_path in image_paths:
        img_path_str = str(img_path)
        _log(f"\n{'='*60}")
        _log(f"Processing: {img_path.name}")

        try:
            pil_image = Image.open(img_path_str).convert("RGB")
        except Exception as e:
            _log(f"Cannot open {img_path_str}: {e!r}")
            continue

        categories = categories_map.get(img_path_str, [])
        if not categories:
            _log(f"No categories for {img_path.name} — skipping")
            continue

        # Step 2: Segment
        _log(f"Step 2: segmenting categories={categories}")
        try:
            instances = segment_instances(
                pil_image, categories, sam_model, sam_processor, min_score=0.4
            )
        except Exception as e:
            _log(f"Step 2 failed: {traceback.format_exc()}")
            continue

        if not instances:
            _log(f"No instances found for {img_path.name} — skipping")
            continue

        _log(f"Step 2: {len(instances)} instances found")

        # Combinations
        combos = select_instance_combinations(
            instances, combo_size=combo_size, max_combos=max_combos
        )
        if not combos:
            _log(f"No valid combinations — skipping")
            continue

        _log(f"Step 3: {len(combos)} combinations × {n_queries_per_combo} queries each")

        # Step 3 + 4: Generate and verify
        image_written = 0
        for combo_idx, combo in enumerate(combos):
            instance_ids = [inst["instance_id"] for inst in combo]
            _log(f"  Combo {combo_idx+1}/{len(combos)}: {instance_ids}")

            try:
                queries = await generate_multihop_queries(
                    pil_image, combo,
                    n_queries=n_queries_per_combo,
                    sem=gemini_sem,
                )
            except Exception as e:
                _log(f"  Step 3 failed: {e!r}")
                continue

            if not queries:
                _log(f"  No valid queries generated for combo {combo_idx+1}")
                continue

            _log(f"  Step 3: {len(queries)} queries generated")

            try:
                verified = await run_verification_pipeline(
                    pil_image, queries,
                    qwen_api_key=qwen_api_key,
                    sem=qwen_sem,
                    gemini_sem=gemini_sem,
                )
            except Exception as e:
                _log(f"  Step 4 failed: {e!r}")
                continue

            _log(f"  Step 4: {len(verified)} verified queries")

            # Write passing records
            for vq in verified:
                record = {
                    "image_path":     img_path_str,
                    "query":          vq["query"],
                    "answer":         vq["final_answer"],
                    "hop_count":      vq.get("hop_count", 4),
                    "instances_used": instance_ids,
                    "verified":       True,
                    "stage":          "hopchain",
                }
                await writer.write(record)
                image_written += 1
                total_written += 1

        _log(f"  Written {image_written} records for {img_path.name}")

    _log(f"\n{'='*60}")
    _log(f"Pipeline complete. Total records written: {total_written}")
    _log(f"Output JSONL: {output_file}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HopChain Step 4 / End-to-end pipeline: verify answers and write JSONL"
    )
    parser.add_argument(
        "--image_dir", required=True,
        help="Directory containing input images",
    )
    parser.add_argument(
        "--categories_file", required=True,
        help="JSON file mapping image paths to category lists (created by Step 1 if missing)",
    )
    parser.add_argument(
        "--output_file", required=True,
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--gemini_api_key", type=str,
        default=os.environ.get("GEMINI_API_KEY", ""),
    )
    parser.add_argument(
        "--qwen_api_key", type=str,
        default=os.environ.get("QWEN_API_KEY", ""),
    )
    parser.add_argument(
        "--concurrency", type=int, default=8,
        help="Max concurrent VLM calls (default 8)",
    )
    parser.add_argument(
        "--n_queries", type=int, default=3,
        help="Queries to generate per instance combination (default 3)",
    )
    parser.add_argument(
        "--combo_size", type=int, default=4,
        help="Instances per combination (default 4)",
    )
    parser.add_argument(
        "--max_combos", type=int, default=10,
        help="Max combinations per image (default 10)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="PyTorch device for SAM3 (default cuda)",
    )
    parser.add_argument(
        "--worker_id", type=int, default=0,
        help="Worker index for multi-GPU sharding (0-based, default 0)",
    )
    parser.add_argument(
        "--n_workers", type=int, default=1,
        help="Total number of parallel workers (default 1 = no sharding)",
    )
    args = parser.parse_args()

    asyncio.run(main(
        image_dir=args.image_dir,
        categories_file=args.categories_file,
        output_file=args.output_file,
        gemini_api_key=args.gemini_api_key,
        qwen_api_key=args.qwen_api_key,
        concurrency=args.concurrency,
        n_queries_per_combo=args.n_queries,
        combo_size=args.combo_size,
        max_combos=args.max_combos,
        sam3_device=args.device,
        worker_id=args.worker_id,
        n_workers=args.n_workers,
    ))
