#!/usr/bin/env python3
"""
Step 2: Generate Variant Images via GPT-image-2
================================================
Reads captions.jsonl produced by step1_generate_captions.py and calls
GPT-image-2 (text-to-image, /images/generations) to synthesize a SIMILAR
BUT DIFFERENT image for each generation_prompt.

For single-image samples: one image generated per sample.
For multi-image samples:  one image generated per sub-image entry, so the
  full set of replacement images matches the structure of the original question.

Output:
  output/variants/images/{sample_id}.png           (single-image samples)
  output/variants/images/{sample_id}_0.png ...     (multi-image samples)
  output/variants/variants.jsonl                   (full record per sample)

Usage:
    python3 step2_generate_images.py [--workers 8] [--max-samples N]

Resume-safe:
  - Samples already in variants.jsonl are skipped (done_ids).
  - Samples recorded in skipped.json (from step 1) are also skipped.
"""

import argparse
import asyncio
import base64
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

# ── Paths ──────────────────────────────────────────────────────────────────────

CAPTIONS_JSONL = Path("output/captions/captions.jsonl")
SKIPPED_JSON = Path("output/captions/skipped.json")  # produced by step 1
OUTPUT_DIR = Path("output/variants")

# ── API Config ─────────────────────────────────────────────────────────────────

GPT_API_BASE = "https://api.gpt.ge/v1"
GPT_API_KEY = "YOUR_GPT_IMAGE_API_KEY"
GPT_IMG_MODEL = "gpt-image-2-c"

GPT_MAX_CONCURRENT = 8
API_TIMEOUT = 300.0  # seconds
MAX_RETRIES = 4

IMAGE_SIZE = "1024x1024"
IMAGE_QUALITY = "high"

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("step2_images.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Semaphore ──────────────────────────────────────────────────────────────────

_gpt_sem: Optional[asyncio.Semaphore] = None


def _get_sem() -> asyncio.Semaphore:
    global _gpt_sem
    if _gpt_sem is None:
        _gpt_sem = asyncio.Semaphore(GPT_MAX_CONCURRENT)
    return _gpt_sem


# ── Retry helpers ──────────────────────────────────────────────────────────────

RETRYABLE = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
)


def _backoff(attempt: int, base: float = 5.0) -> float:
    import random

    wait = base * (2**attempt)
    return wait * (0.7 + 0.6 * random.random())


# ── GPT-image-2 generation ─────────────────────────────────────────────────────


async def generate_image(
    client: httpx.AsyncClient,
    prompt: str,
    log_prefix: str = "img",
) -> Optional[bytes]:
    """
    Call GPT-image-2 text-to-image generation.
    Returns raw PNG bytes or None on failure.
    """
    # Truncate prompt to 1000 chars (API limit)
    prompt_trunc = prompt[:1000]

    sem = _get_sem()
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.post(
                    f"{GPT_API_BASE}/images/generations",
                    json={
                        "model": GPT_IMG_MODEL,
                        "prompt": prompt_trunc,
                        "n": 1,
                        "size": IMAGE_SIZE,
                        "quality": IMAGE_QUALITY,
                        "output_format": "png",
                    },
                    headers={"Authorization": f"Bearer {GPT_API_KEY}"},
                )
                resp.raise_for_status()
                body = resp.json()

            items = body.get("data", [])
            if not items:
                log.warning(f"[{log_prefix}] Empty data in response — retrying")
                await asyncio.sleep(_backoff(attempt))
                continue

            item = items[0]

            if "b64_json" in item:
                return base64.b64decode(item["b64_json"])

            if "url" in item:
                # Fallback: download from URL
                try:
                    dl = await client.get(item["url"])
                    dl.raise_for_status()
                    return dl.content
                except Exception as e:
                    log.warning(f"[{log_prefix}] URL download failed: {e!r}")
                    return None

            log.warning(f"[{log_prefix}] Unexpected item keys: {list(item.keys())}")
            return None

        except RETRYABLE as e:
            wait = _backoff(attempt)
            log.warning(
                f"[{log_prefix}] Network error (attempt {attempt + 1}/{MAX_RETRIES}): "
                f"{type(e).__name__}: {e!r}  wait {wait:.1f}s"
            )
            await asyncio.sleep(wait)

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429:
                wait = _backoff(attempt, base=25.0)
                log.warning(
                    f"[{log_prefix}] HTTP 429, waiting {wait:.1f}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(wait)
            elif status in (400, 403):
                log.error(
                    f"[{log_prefix}] HTTP {status} (non-retryable): {e.response.text[:300]}"
                )
                return None
            else:
                wait = _backoff(attempt)
                log.warning(
                    f"[{log_prefix}] HTTP {status}, waiting {wait:.1f}s: "
                    f"{e.response.text[:200]}"
                )
                await asyncio.sleep(wait)

        except Exception as e:
            log.error(f"[{log_prefix}] Unexpected: {type(e).__name__}: {e!r}")
            return None

    log.error(f"[{log_prefix}] Image generation failed after {MAX_RETRIES} retries")
    return None


# ── JSONL helpers ──────────────────────────────────────────────────────────────


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()

    async def write(self, record: dict) -> None:
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_ids(jsonl_path: Path) -> set[str]:
    done: set[str] = set()
    if not jsonl_path.is_file():
        return done
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                sid = rec.get("sample_id", "")
                if sid:
                    done.add(sid)
            except Exception:
                pass
    return done


def load_skipped_ids(skipped_json: Path) -> set[str]:
    """Load sample_ids that were skipped in step 1 (non-editable classification)."""
    if not skipped_json.is_file():
        return set()
    try:
        data = json.loads(skipped_json.read_text(encoding="utf-8"))
        return set(data.keys())
    except Exception:
        return set()


# ── Per-sample worker ──────────────────────────────────────────────────────────


async def process_sample(
    caption_rec: dict,
    client: httpx.AsyncClient,
    images_dir: Path,
    writer: JsonlWriter,
) -> bool:
    """
    Process one caption record: generate image(s) → save → write JSONL record.
    Returns True on success.
    """
    sample_id = caption_rec["sample_id"]
    captions = caption_rec.get("captions", {})
    is_multi = caption_rec.get("is_multi_image", False)
    question = caption_rec.get("original_question", "")
    answer = caption_rec.get("original_answer", "")

    log.info(
        f"[{sample_id}] Generating {'multi' if is_multi else 'single'}-image variant"
    )

    generated_image_paths: list[str] = []
    generation_prompts: list[str] = []

    def _clean_and_enhance_prompt(
        base_prompt: str,
        question_reqs: list[dict],
        img_note: str = "",
    ) -> str:
        """
        Clean up the generation prompt and append scene-description reminders
        derived from question_requirements.

        Key design decision: we NEVER put question text or answer values into the
        GPT-image-2 prompt. Instead we extract the *visual detail requirements*
        from question_requirements and phrase them as scene-construction reminders.

        Examples of what gets appended:
          - "Ensure all price tags have clearly legible numbers."
          - "Every bottle must be individually distinguishable and countable."
          - "The small text on the warning label must be sharp and readable."
        """
        cleaned = base_prompt

        # Strip any legacy [QUESTION & ANSWER CONTEXT] block from old captions
        if "[QUESTION & ANSWER CONTEXT]" in cleaned:
            parts = cleaned.split("[QUESTION & ANSWER CONTEXT]", 1)
            after = parts[1]
            for marker in [
                "[CRITICAL SCENE REQUIREMENTS]",
                "[REQUIRED VISUAL ELEMENTS]",
                "[SCENE DESCRIPTION]",
                "\n\n",
            ]:
                if marker in after:
                    cleaned = parts[0] + after[after.index(marker) :]
                    break
            else:
                cleaned = parts[0] + after

        # Build scene-description reminders from question_requirements
        reminders: list[str] = []
        for req in question_reqs:
            attr = req.get("critical_attribute", "").strip()
            if attr:
                # critical_attribute is already a full sentence — use it directly
                reminders.append(f"- {attr}")
            else:
                element = req.get("element", "").strip()
                if element:
                    reminders.append(
                        f"- {element} must be present, unambiguous, and clearly rendered"
                    )

        suffix_parts: list[str] = []
        if img_note:
            suffix_parts.append(img_note)
        if reminders:
            suffix_parts.append(
                "VISUAL DETAIL REMINDERS (do NOT render these as text in the image):\n"
                + "\n".join(reminders[:10])  # cap at 10 to save prompt budget
            )

        if suffix_parts:
            cleaned = cleaned.rstrip() + "\n\n" + "\n".join(suffix_parts)

        return cleaned.strip()

    # Gather question_requirements for scene reminders
    all_question_reqs = captions.get("question_requirements", [])

    if is_multi:
        sub_images = captions.get("images", [])
        if not sub_images:
            log.error(f"[{sample_id}] No sub-images in caption — skipping")
            return False

        n_total = len(sub_images)
        for sub in sub_images:
            img_idx = sub.get("image_index", len(generated_image_paths))
            raw_prompt = sub.get("generation_prompt", "")
            if not raw_prompt:
                log.warning(
                    f"[{sample_id}] Sub-image {img_idx} missing generation_prompt — skipping"
                )
                continue

            # Filter requirements relevant to this sub-image
            sub_reqs = [
                r for r in all_question_reqs if r.get("image_index") == img_idx
            ] or all_question_reqs  # fallback to all if no index filtering

            img_note = (
                f"This is Image {img_idx + 1} of {n_total} in a multi-image question."
            )
            prompt = _clean_and_enhance_prompt(raw_prompt, sub_reqs, img_note)

            log_prefix = f"{sample_id}/{img_idx}"
            log.info(
                f"[{log_prefix}] Prompt ({len(prompt)} chars): {prompt[:100]!r}..."
            )

            img_bytes = await generate_image(client, prompt, log_prefix=log_prefix)
            if img_bytes is None:
                log.error(f"[{log_prefix}] Generation failed — skipping entire sample")
                return False

            img_path = images_dir / f"{sample_id}_{img_idx}.png"
            img_path.write_bytes(img_bytes)
            generated_image_paths.append(str(img_path))
            generation_prompts.append(prompt)
            log.info(
                f"[{log_prefix}] Saved {len(img_bytes) // 1024} KB → {img_path.name}"
            )

    else:
        raw_prompt = captions.get("generation_prompt", "")
        if not raw_prompt:
            log.error(f"[{sample_id}] Missing generation_prompt — skipping")
            return False

        prompt = _clean_and_enhance_prompt(raw_prompt, all_question_reqs)
        log.info(f"[{sample_id}] Prompt ({len(prompt)} chars): {prompt[:100]!r}...")

        img_bytes = await generate_image(client, prompt, log_prefix=sample_id)
        if img_bytes is None:
            log.error(f"[{sample_id}] Generation failed — skipping")
            return False

        img_path = images_dir / f"{sample_id}.png"
        img_path.write_bytes(img_bytes)
        generated_image_paths.append(str(img_path))
        generation_prompts.append(prompt)
        log.info(f"[{sample_id}] Saved {len(img_bytes) // 1024} KB → {img_path.name}")

    if not generated_image_paths:
        log.error(f"[{sample_id}] No images generated — skipping record")
        return False

    # Build output record
    # The original question is reused verbatim — no variant_question field.
    def _get_caption_field(field: str) -> object:
        """Extract a field from captions, handling single vs multi-image structure."""
        if is_multi:
            return [s.get(field, "") for s in captions.get("images", [])]
        return captions.get(field, "")

    record = {
        "sample_id": sample_id,
        "original_images": caption_rec.get("original_images", []),
        "question": caption_rec.get("original_question", ""),
        "original_answer": caption_rec.get("original_answer", ""),
        "is_multi_image": is_multi,
        # Structured caption from step 1
        "scene_caption": _get_caption_field("scene_caption"),
        "question_requirements": captions.get("question_requirements", []),
        "difficulty_features": (
            _get_caption_field("difficulty_sources")
            if not is_multi
            else [
                s.get("scene_caption", {}).get("difficulty_sources", "")
                for s in captions.get("images", [])
            ]
        ),
        # Generation
        "generation_prompts": generation_prompts,
        "variant_image_paths": generated_image_paths,
        "timestamp": datetime.now().isoformat(),
    }

    await writer.write(record)
    log.info(f"[{sample_id}] Done — {len(generated_image_paths)} image(s) saved")
    return True


# ── Main pipeline ──────────────────────────────────────────────────────────────


async def run(
    max_samples: Optional[int],
    workers: int,
    captions_path: Path,
    skipped_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "variants.jsonl"

    # Resume: skip already-generated samples
    done_ids = load_done_ids(out_path)
    log.info(f"Already processed: {len(done_ids)} samples — will skip these")

    # Skip samples that step 1 marked as non-editable
    skipped_ids = load_skipped_ids(skipped_path)
    log.info(
        f"Skipped by step 1 (non-editable): {len(skipped_ids)} samples — will skip these"
    )

    exclude_ids = done_ids | skipped_ids

    # Load captions
    if not captions_path.is_file():
        raise FileNotFoundError(
            f"Captions file not found: {captions_path}\n"
            "Run step1_generate_captions.py first."
        )

    with open(captions_path, encoding="utf-8") as f:
        all_captions = [json.loads(line) for line in f if line.strip()]

    log.info(f"Total caption records: {len(all_captions)}")

    pending = [
        rec for rec in all_captions if rec.get("sample_id", "") not in exclude_ids
    ]

    if max_samples is not None:
        pending = pending[:max_samples]

    log.info(f"Pending samples: {len(pending)} | workers: {workers}")

    if not pending:
        log.info("Nothing to do — all samples already processed.")
        return

    writer = JsonlWriter(out_path)

    # Override semaphore limit
    global _gpt_sem
    _gpt_sem = asyncio.Semaphore(workers)

    # GPT-image-2 is an external API → use proxy if available
    import os

    proxy_url = (
        os.environ.get("https_proxy")
        or os.environ.get("HTTPS_PROXY")
        or os.environ.get("http_proxy")
        or os.environ.get("HTTP_PROXY")
    )

    timeout = httpx.Timeout(API_TIMEOUT, read=API_TIMEOUT, pool=60)
    limits = httpx.Limits(
        max_connections=workers * 2, max_keepalive_connections=workers
    )

    success = failed = 0

    async with httpx.AsyncClient(
        timeout=timeout,
        limits=limits,
        proxy=proxy_url,
        trust_env=False,
    ) as client:
        tasks = [
            asyncio.create_task(process_sample(rec, client, images_dir, writer))
            for rec in pending
        ]
        for coro in asyncio.as_completed(tasks):
            try:
                ok = await coro
                if ok:
                    success += 1
                else:
                    failed += 1
            except Exception as e:
                log.error(f"Unhandled exception in worker: {e!r}")
                failed += 1

    total = success + failed
    log.info(f"Step 2 complete — success: {success}/{total}, failed: {failed}/{total}")
    log.info(f"Variant images in: {images_dir}")
    log.info(f"Variant records in: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: Generate variant images via GPT-image-2 from captions"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Max concurrent GPT-image-2 requests (default: 10)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing; default: all)",
    )
    parser.add_argument(
        "--captions",
        type=str,
        default="zerobench_output/captions/captions.jsonl",
        help="Path to captions.jsonl from step 1",
    )
    parser.add_argument(
        "--skipped",
        type=str,
        default="zerobench_output/captions/skipped.json",
        help="Path to skipped.json from step 1 (groups classified as non-editable)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="zerobench_output/variants",
        help="Output directory for images + variants.jsonl",
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            max_samples=args.max_samples,
            workers=args.workers,
            captions_path=Path(args.captions),
            skipped_path=Path(args.skipped),
            output_dir=Path(args.output_dir),
        )
    )


if __name__ == "__main__":
    main()
