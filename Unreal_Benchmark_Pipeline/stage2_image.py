#!/usr/bin/env python3
"""
stage2_image.py — Stage 2 of the zerogen pipeline (per-sample image only).

Stage 1 emits ONE record per SAMPLE (mc / oe split). Stage 2 generates ONE
image per sample at images/<sample_id>.png using GPT-Image-2 (gpt-image-2-c).

Output: stage2_output.jsonl (append-only, one record per sample). Resume by
skipping sample_ids whose ok record is already present AND whose PNG exists
on disk. Failed samples are retried on the next run.

Run:
  python stage2_image.py
  python stage2_image.py --limit 5
  python stage2_image.py --no-resume
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import time
import traceback
from pathlib import Path
from typing import Any

import httpx


# ── API constants (hardcoded by project convention) ───────────────────────
# Switch between the internal Rednote (Azure-style) endpoint and the
# external api.gpt.ge (standard OpenAI) endpoint by changing GPT_IMAGE_MODE.
GPT_IMAGE_MODE = "internal"  # "internal" | "external"

if GPT_IMAGE_MODE == "internal":
    # Azure-style: api-key header, /openai/...?api-version=..., bypass system proxy.
    GPT_IMAGE_ENDPOINT = "<INTERNAL_IMAGE_ENDPOINT>"
    GPT_IMAGE_API_KEY = os.environ.get("GPT_IMAGE_INT_KEY", "")
    GPT_IMAGE_API_VERSION = "2025-04-01-preview"
elif GPT_IMAGE_MODE == "external":
    # Standard OpenAI-compatible: Bearer auth, /v1/images/..., go through system proxy.
    GPT_IMAGE_ENDPOINT = "https://api.gpt.ge/v1"
    GPT_IMAGE_API_KEY = os.environ.get("GPT_IMAGE_EXT_KEY", "")
    GPT_IMAGE_API_VERSION = None
else:
    raise ValueError(f"unknown GPT_IMAGE_MODE: {GPT_IMAGE_MODE!r}")

GPT_IMAGE_MODEL = "gpt-image-2-c"
# GPT_IMAGE_SIZE = "1024x1024"
GPT_IMAGE_SIZE = "auto"
# GPT_IMAGE_QUALITY = "high"
GPT_IMAGE_QUALITY = "auto"

# ── Run config ────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
STAGE1_PATH = HERE / "stage1_output.jsonl"
OUTPUT_PATH = HERE / "stage2_output.jsonl"
IMAGES_DIR = HERE / "images"

IMAGE_CONCURRENCY = 64
DEFAULT_PARALLEL_SAMPLES = 256

MAX_RETRIES_IMAGE = 20
REQUEST_TIMEOUT_IMAGE = 600.0


# ── HTTP retryable error tuple ────────────────────────────────────────────

_RETRYABLE = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ProxyError,
)


# ── GPT-Image-2 text-to-image ─────────────────────────────────────────────

async def gen_image(
    client: httpx.AsyncClient,
    sem: asyncio.Semaphore,
    prompt: str,
    sample_id: str,
) -> bytes:
    """Returns raw PNG bytes. Raises on permanent failure after retries."""
    base = GPT_IMAGE_ENDPOINT.rstrip("/")
    if GPT_IMAGE_MODE == "internal":
        url = f"{base}/openai/images/generations?api-version={GPT_IMAGE_API_VERSION}"
        headers = {"api-key": GPT_IMAGE_API_KEY, "Content-Type": "application/json"}
    else:
        url = f"{base}/images/generations"
        headers = {"Authorization": f"Bearer {GPT_IMAGE_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": GPT_IMAGE_MODEL,
        "prompt": prompt,
        "n": 1,
        "size": GPT_IMAGE_SIZE,
        "quality": GPT_IMAGE_QUALITY,
    }

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES_IMAGE):
        try:
            async with sem:
                resp = await client.post(url, headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()

            if isinstance(body, dict) and "Code" in body and "Error" in body and "data" not in body:
                err_str = str(body.get("Error", ""))
                err_low = err_str.lower()
                if any(s in err_low for s in (
                    "engineoverloaded", "too many requests", "429",
                    "resource_exhausted", "quota", "nocapacity",
                    "high demand", "exceeds the maximum",
                )):
                    wait = min(30 * (attempt + 1), 180)
                    print(f"[stage2/{sample_id}] image proxy overload, wait {wait}s "
                          f"(attempt {attempt + 1}/{MAX_RETRIES_IMAGE}) | {err_str[:120]!r}", flush=True)
                    await asyncio.sleep(wait)
                    continue
                raise RuntimeError(f"image proxy error Code={body.get('Code')} | {err_str[:200]}")

            data = body.get("data") or []
            if not data:
                raise RuntimeError(f"image response missing data: {str(body)[:200]}")
            item = data[0]
            if "b64_json" in item:
                return base64.b64decode(item["b64_json"])
            if "url" in item:
                async with sem:
                    img_resp = await client.get(item["url"])
                    img_resp.raise_for_status()
                return img_resp.content
            raise RuntimeError(f"image response item has unexpected keys: {list(item.keys())}")

        except _RETRYABLE as e:
            last_err = e
            wait = min(2 ** attempt + attempt, 30)
            print(f"[stage2/{sample_id}] image network retry {attempt + 1}/{MAX_RETRIES_IMAGE}: "
                  f"{type(e).__name__}: {e!r} (wait {wait}s)", flush=True)
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body_head = (e.response.text or "")[:200]
            if status == 429:
                wait = min(25 * (attempt + 1), 180)
                print(f"[stage2/{sample_id}] image HTTP 429, wait {wait}s "
                      f"(attempt {attempt + 1}/{MAX_RETRIES_IMAGE})", flush=True)
                await asyncio.sleep(wait)
                last_err = e
            elif status in (400, 403):
                raise RuntimeError(f"image HTTP {status}: {body_head}")
            else:
                wait = min(5 * (attempt + 1), 60)
                print(f"[stage2/{sample_id}] image HTTP {status}, wait {wait}s | {body_head!r}",
                      flush=True)
                await asyncio.sleep(wait)
                last_err = e
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(f"[stage2/{sample_id}] image retry {attempt + 1}/{MAX_RETRIES_IMAGE}: "
                  f"{type(e).__name__}: {e!r} (wait {wait}s)", flush=True)
            await asyncio.sleep(wait)

    raise RuntimeError(f"image gen failed after {MAX_RETRIES_IMAGE} retries: {last_err!r}")


# ── JSONL writer + resume ─────────────────────────────────────────────────

class JsonlAppender:
    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()

    async def write(self, record: dict) -> None:
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_sample_ids(path: Path) -> set[str]:
    """Latest-wins. Done = status=='ok' AND image file exists on disk."""
    by_sid: dict[str, dict] = {}
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sample_id")
            if sid:
                by_sid[sid] = rec
    done: set[str] = set()
    base = path.parent
    for sid, rec in by_sid.items():
        if rec.get("status") != "ok":
            continue
        rel = rec.get("image_path")
        if not rel:
            continue
        if not (base / rel).exists():
            continue
        done.add(sid)
    return done


def load_stage1_samples(path: Path) -> list[dict]:
    """Latest-wins by sample_id; only ok records."""
    by_sid: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "ok":
                continue
            sid = rec.get("sample_id")
            if sid:
                by_sid[sid] = rec
    return list(by_sid.values())


# ── Per-sample orchestration ──────────────────────────────────────────────

async def process_one(
    sample_sem: asyncio.Semaphore,
    image_client: httpx.AsyncClient,
    image_sem: asyncio.Semaphore,
    writer: JsonlAppender,
    images_dir: Path,
    stage1_record: dict,
) -> tuple[str, bool]:
    sample_id = stage1_record["sample_id"]
    seed_id   = stage1_record["seed_id"]
    family_id = stage1_record["family_id"]
    qtype     = stage1_record["qtype"]
    image_prompt = stage1_record["stage1"]["image_prompt"]
    t0 = time.time()

    base_record: dict[str, Any] = {
        "sample_id": sample_id,
        "seed_id":   seed_id,
        "family_id": family_id,
        "qtype":     qtype,
        "status":    "failed",
        "image_path": None,
        "elapsed_s": None,
        "created_at": int(time.time()),
    }

    async with sample_sem:
        try:
            png_bytes = await gen_image(image_client, image_sem, image_prompt, sample_id)
        except Exception as e:
            base_record["error"] = f"{type(e).__name__}: {e!r}"
            base_record["elapsed_s"] = round(time.time() - t0, 2)
            await writer.write(base_record)
            print(
                f"[stage2/{sample_id}] FAILED in {base_record['elapsed_s']}s: "
                f"{type(e).__name__}: {str(e)[:160]}",
                flush=True,
            )
            return sample_id, False

    image_path = images_dir / f"{sample_id}.png"
    image_path.write_bytes(png_bytes)
    base_record["status"] = "ok"
    base_record["image_path"] = str(image_path.relative_to(HERE))
    base_record["elapsed_s"] = round(time.time() - t0, 2)
    await writer.write(base_record)
    print(
        f"[stage2/{sample_id}] ok in {base_record['elapsed_s']}s | "
        f"→ {base_record['image_path']}",
        flush=True,
    )
    return sample_id, True


# ── Main ──────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    stage1_path = Path(args.stage1)
    output_path = Path(args.output)
    images_dir = Path(args.images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    samples = load_stage1_samples(stage1_path)
    print(f"Loaded {len(samples)} ok stage1 samples from {stage1_path}.", flush=True)

    if args.limit > 0:
        samples = samples[: args.limit]
        print(f"--limit {args.limit} → processing first {len(samples)} samples", flush=True)

    if args.no_resume and output_path.exists():
        backup = output_path.with_suffix(output_path.suffix + ".bak")
        output_path.rename(backup)
        print(f"--no-resume: moved existing {output_path} → {backup}", flush=True)

    done = load_done_sample_ids(output_path)
    if done:
        print(f"resume: {len(done)} sample_ids already ok in {output_path} — will skip", flush=True)

    pending = [r for r in samples if r["sample_id"] not in done]
    if not pending:
        print("nothing to do — all samples already processed.", flush=True)
        return

    print(
        f"submitting {len(pending)} samples | parallel_samples={args.parallel_samples} | "
        f"image_concurrency={args.image_concurrency}",
        flush=True,
    )

    sample_sem = asyncio.Semaphore(args.parallel_samples)
    image_sem  = asyncio.Semaphore(args.image_concurrency)

    image_timeout = httpx.Timeout(REQUEST_TIMEOUT_IMAGE, read=REQUEST_TIMEOUT_IMAGE, pool=60)
    image_limits = httpx.Limits(
        max_connections=args.image_concurrency * 2,
        max_keepalive_connections=args.image_concurrency,
    )

    writer = JsonlAppender(output_path)

    t_start = time.time()
    # Internal Rednote endpoint must be reached directly (system proxy returns 503).
    # External api.gpt.ge endpoint must go through the system proxy.
    trust_env = (GPT_IMAGE_MODE == "external")
    async with httpx.AsyncClient(timeout=image_timeout, limits=image_limits, trust_env=trust_env) as image_client:
        tasks = [
            process_one(sample_sem, image_client, image_sem, writer, images_dir, rec)
            for rec in pending
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    n_ok = sum(1 for r in results if isinstance(r, tuple) and r[1])
    n_fail = len(results) - n_ok
    print(
        f"\nDone in {time.time() - t_start:.1f}s. "
        f"ok={n_ok} failed={n_fail} → {output_path} | images → {images_dir}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 2 (image only): GPT-Image-2 generation, ONE image per sample"
    )
    parser.add_argument("--stage1", default=str(STAGE1_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--images-dir", default=str(IMAGES_DIR))
    parser.add_argument("--parallel-samples", type=int, default=DEFAULT_PARALLEL_SAMPLES)
    parser.add_argument("--image-concurrency", type=int, default=IMAGE_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\ninterrupted — partial results are in the output JSONL; rerun to resume.",
              flush=True)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
