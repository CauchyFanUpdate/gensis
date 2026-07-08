#!/usr/bin/env python3
"""
Step 2: Generate Variant Images via GPT-image-2 (BabyVision)
=============================================================
Adapted from zerobench_step2_generate_images.py for the BabyVision dataset.

Reads captions.jsonl produced by babyvision_step1_generate_captions.py and calls
GPT-image-2 (text-to-image, /images/generations) to synthesize a SIMILAR BUT
DIFFERENT image for each generation_prompt.

BabyVision tasks are always single-image, so one image is generated per record.

Output:
  <output_dir>/images/{sample_id}.png   — generated image
  <output_dir>/variants.jsonl           — full record per sample

Usage:
    python3 babyvision_step2_generate_images.py [--workers 8] [--max-samples N]

Resume-safe: samples already in variants.jsonl or in skipped.json are skipped.
"""

import argparse
import asyncio
import base64
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

# ── Paths ──────────────────────────────────────────────────────────────────────

CAPTIONS_JSONL = Path("babyvision_output/captions/captions.jsonl")
SKIPPED_JSON = Path("babyvision_output/captions/skipped.json")
OUTPUT_DIR = Path("babyvision_output/variants")

# ── API Config ─────────────────────────────────────────────────────────────────

GPT_API_BASE = "https://api.gpt.ge/v1"
GPT_API_KEY = "YOUR_GPT_IMAGE_API_KEY"
GPT_IMG_MODEL = "gpt-image-2-c"

GPT_MAX_CONCURRENT = 8
API_TIMEOUT = 300.0
MAX_RETRIES = 4

IMAGE_SIZE = "1024x1024"
IMAGE_QUALITY = "high"

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("babyvision_step2_images.log", encoding="utf-8"),
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
    """Call GPT-image-2 text-to-image. Returns raw PNG bytes or None."""
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
                sid = json.loads(line).get("sample_id", "")
                if sid:
                    done.add(sid)
            except Exception:
                pass
    return done


def load_skipped_ids(skipped_json: Path) -> set[str]:
    if not skipped_json.is_file():
        return set()
    try:
        return set(json.loads(skipped_json.read_text(encoding="utf-8")).keys())
    except Exception:
        return set()


# ── Prompt enhancement ─────────────────────────────────────────────────────────


def _clean_and_enhance_prompt(
    base_prompt: str,
    question_reqs: list[dict],
) -> str:
    """
    Clean the generation prompt and append scene-detail reminders derived
    from question_requirements. Never puts question or answer text in the prompt.
    """
    cleaned = base_prompt

    # Strip any legacy [QUESTION & ANSWER CONTEXT] blocks
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

    reminders: list[str] = []
    for req in question_reqs:
        attr = req.get("critical_attribute", "").strip()
        if attr:
            reminders.append(f"- {attr}")
        else:
            element = req.get("element", "").strip()
            if element:
                reminders.append(
                    f"- {element} must be present, unambiguous, and clearly rendered"
                )

    if reminders:
        cleaned = (
            cleaned.rstrip()
            + "\n\nVISUAL DETAIL REMINDERS (do NOT render these as text in the image):\n"
            + "\n".join(reminders[:10])
        )

    return cleaned.strip()


def _as_req_list(question_requirements) -> list[dict]:
    """Normalize question_requirements to a list of dicts regardless of source format."""
    if isinstance(question_requirements, list):
        return question_requirements
    if isinstance(question_requirements, dict):
        return [question_requirements]
    return []


# ── Per-sample worker ──────────────────────────────────────────────────────────


async def process_sample(
    caption_rec: dict,
    client: httpx.AsyncClient,
    images_dir: Path,
    writer: JsonlWriter,
) -> bool:
    """Generate one image from a caption record. Returns True on success."""
    sample_id = caption_rec["sample_id"]
    captions = caption_rec.get("captions", {})

    raw_prompt = captions.get("generation_prompt", "")
    if not raw_prompt:
        log.error(f"[{sample_id}] Missing generation_prompt — skipping")
        return False

    question_reqs = _as_req_list(captions.get("question_requirements"))
    prompt = _clean_and_enhance_prompt(raw_prompt, question_reqs)

    log.info(f"[{sample_id}] Prompt ({len(prompt)} chars): {prompt[:100]!r}...")

    img_bytes = await generate_image(client, prompt, log_prefix=sample_id)
    if img_bytes is None:
        log.error(f"[{sample_id}] Generation failed — skipping")
        return False

    img_path = images_dir / f"{sample_id}.png"
    img_path.write_bytes(img_bytes)
    log.info(f"[{sample_id}] Saved {len(img_bytes) // 1024} KB → {img_path.name}")

    record = {
        "sample_id": sample_id,
        "task_id": caption_rec.get("task_id"),
        "task_type": caption_rec.get("task_type", ""),
        "task_subtype": caption_rec.get("task_subtype", ""),
        "original_image": caption_rec.get("original_image", ""),
        "question": caption_rec.get("original_question", ""),
        "original_answer": caption_rec.get("original_answer", ""),
        "ans_type": caption_rec.get("ans_type", ""),
        "options": caption_rec.get("options", []),
        "scene_caption": captions.get("scene_caption", {}),
        "question_requirements": captions.get("question_requirements", {}),
        "difficulty_sources": captions.get("scene_caption", {}).get(
            "difficulty_sources", ""
        ),
        "generation_prompt": prompt,
        "variant_image_path": str(img_path),
        "timestamp": datetime.now().isoformat(),
    }

    await writer.write(record)
    log.info(f"[{sample_id}] Done ✓")
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

    done_ids = load_done_ids(out_path)
    log.info(f"Already processed: {len(done_ids)} samples")

    skipped_ids = load_skipped_ids(skipped_path)
    log.info(f"Skipped by step 1 (non-editable): {len(skipped_ids)} samples")

    exclude_ids = done_ids | skipped_ids

    if not captions_path.is_file():
        raise FileNotFoundError(
            f"Captions file not found: {captions_path}\n"
            "Run babyvision_step1_generate_captions.py first."
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

    global _gpt_sem
    _gpt_sem = asyncio.Semaphore(workers)

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
    log.info(
        f"Step 2 complete — success: {success}/{total}, failed: {failed}/{total}"
    )
    log.info(f"Variant images in: {images_dir}")
    log.info(f"Variant records in: {out_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 2: Generate variant images via GPT-image-2 (BabyVision)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Max concurrent GPT-image-2 requests (default: 8)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit number of samples (for testing; default: all)",
    )
    parser.add_argument(
        "--captions",
        type=str,
        default=str(CAPTIONS_JSONL),
        help=f"Path to captions.jsonl from step 1 (default: {CAPTIONS_JSONL})",
    )
    parser.add_argument(
        "--skipped",
        type=str,
        default=str(SKIPPED_JSON),
        help=f"Path to skipped.json from step 1 (default: {SKIPPED_JSON})",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(OUTPUT_DIR),
        help=f"Output directory (default: {OUTPUT_DIR})",
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
