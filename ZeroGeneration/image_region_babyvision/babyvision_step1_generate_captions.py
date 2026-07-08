#!/usr/bin/env python3
"""
Step 1: Generate Captions via Qwen3.6-plus VL  (thinking mode enabled)
=======================================================================
Adapted from zerobench_step1_generate_captions.py for the BabyVision dataset.

Pipeline:
  For each BabyVision task (one image + one question):
    0. Skip if the task subtype is non-editable (Maze, Metro map, Connect the lines,
       Lines Observation, etc.).  Skipped tasks are recorded in skipped.json.
    1. Call Qwen3.6-plus VL with thinking=True:
       - Send the image together with the question and answer
       - Produce an exhaustive pixel-level scene description (caption)
       - Identify every visual element the question depends on
       - Write a GPT-image-2 generation prompt that:
           * preserves ALL difficulty features (occlusion, clutter, small text, etc.)
           * guarantees every question-critical element is explicitly required
           * keeps the scene as close to the original as possible
    2. Save the full result to output/captions/captions.jsonl

Output:
  <output_dir>/captions.jsonl   — successful caption records
  <output_dir>/skipped.json     — tasks skipped due to non-editable classification

Usage:
    python3 babyvision_step1_generate_captions.py [--workers 8] [--max-samples N]

Resume-safe: already-done AND already-skipped tasks are skipped at startup.
"""

import argparse
import asyncio
import base64
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/BabyVision")
DATASET_PARQUET = DATASET_DIR / "data/train-00000-of-00001.parquet"
IMAGES_DIR = DATASET_DIR / "images"

# ── API Config ─────────────────────────────────────────────────────────────────

QWEN_API_BASE = "https://runway.devops.xiaohongshu.com/openai/qwen/v1"
QWEN_API_KEY = "YOUR_QWEN_API_KEY"
QWEN_MODEL = "qwen3.6-plus"

QWEN_MAX_CONCURRENT = 8
API_TIMEOUT = 480.0
MAX_RETRIES = 4
MAX_TOKENS = 16000

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("babyvision_step1_captions.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── Semaphore ──────────────────────────────────────────────────────────────────

_qwen_sem: Optional[asyncio.Semaphore] = None


def _get_sem() -> asyncio.Semaphore:
    global _qwen_sem
    if _qwen_sem is None:
        _qwen_sem = asyncio.Semaphore(QWEN_MAX_CONCURRENT)
    return _qwen_sem


# ── Retry helpers ──────────────────────────────────────────────────────────────

RETRYABLE = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
)


def _backoff(attempt: int, base: float = 6.0) -> float:
    import random

    wait = base * (2**attempt)
    return wait * (0.7 + 0.6 * random.random())


# ── Non-editable subtype list ──────────────────────────────────────────────────
# These subtypes involve globally-coupled path or logic constraints that make
# surgical visual editing impossible.

NON_EDITABLE_SUBTYPES = {
    "Maze",
    "Metro map",
    "Connect the lines",
    "Lines Observation",
}

# ── Prompts ────────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are an expert in constructing visual reasoning benchmark datasets.

You will be given one image together with a VISUAL QUESTION and its answer.
Your job is to produce a highly detailed analysis and a precise image generation prompt,
so that a text-to-image model (GPT-image-2) can generate a NEW image that:

  1. PRESERVES the answerable difficulty for the question:
     - The original question (unchanged) can still be asked on the new image
     - The new image contains ALL visual information needed to answer the question
     - The difficulty is identical: same density, same occlusion patterns, same
       level of visual complexity, same need for careful observation

  2. IS AS SIMILAR AS POSSIBLE to the original:
     - Same scene type and domain
     - Same camera angle, framing, depth-of-field, lighting style
     - Same spatial layout and overall composition
     - Same number and arrangement of key elements
     - Only surface details may differ (brand names, numeric values, colors, labels)

CRITICAL — SKIP CLASSIFICATION:
Before doing anything else, you must decide whether the task is NON-EDITABLE.
A task is NON-EDITABLE if it falls into ANY of the following categories:

  A. Complex maze layouts
     Any cell change breaks path semantics (open/dead-end relationships are globally
     coupled). Local editing is impossible.

  B. Metro map / transit network diagrams
     Topology is globally coupled; changing any connection invalidates route queries.

  C. Line-tracing / connect-the-lines puzzles
     Every path endpoint and intersection is globally entangled; editing any segment
     breaks the solution.

  D. Sokoban, Sudoku, or other constraint puzzles
     Every cell participates in the global constraint network.

If the task matches ANY category above, output ONLY this JSON and stop:
  {"skip": true, "reason": "<one-sentence explanation of which category and why>"}

Otherwise, proceed with the full analysis below.

Think step by step. Use your full reasoning capacity to:
  - Describe the image at extreme detail (every object, position, text, relationship)
  - Identify exactly what the question needs visually to be answerable
  - Design a generation prompt that locks in all required elements and scene structure

STRICT OUTPUT RULES:
  - Output valid JSON only. Never repeat yourself inside string values.
  - If you are unsure of a detail, write a single best-guess sentence and move on.
  - Do NOT loop, retry, or write "let's assume..." more than once per field.
  - Keep each JSON string value concise and non-repetitive.
"""

_USER_TEMPLATE = """\
=== IMAGE TASK ===
Task type: {task_type} / {task_subtype}

=== QUESTION AND ANSWER ===
{question_block}

=== YOUR TASK ===
First, classify whether this image is NON-EDITABLE (see system prompt).
If non-editable: output only {{"skip": true, "reason": "..."}} and stop.

Otherwise, study the attached image with extreme care.
Produce a JSON object with ALL fields below.
Output raw JSON only — no markdown fences, no commentary outside the JSON.

{{
  "skip": false,

  "scene_caption": {{
    "overview": "2-3 sentence summary of the scene: what it is, where it is, what the overall composition looks like.",
    "camera": "Camera angle (e.g. straight-on, slight top-down, eye-level), distance (close-up / medium / wide), framing (portrait / landscape / square).",
    "spatial_layout": "Describe the positions of all major elements using grid references (top-left, center, bottom-right, etc.). Describe depth layers: foreground, midground, background.",
    "objects": [
      {{
        "name": "exact object name",
        "count": "exact count or 'N/A'",
        "position": "exact position in frame",
        "appearance": "color, shape, size relative to frame, orientation",
        "occlusion": "what it occludes or is occluded by, if anything",
        "readable_text": "any text or numbers on/near it, copied verbatim"
      }}
    ],
    "text_and_numbers": "List ALL text and numbers visible in the image. For each: exact content, font size impression (large/medium/small/tiny), position, color, background.",
    "occlusion_and_overlap": "Describe every occlusion relationship: which objects overlap which, partial visibility, what is hidden and how much.",
    "lighting_and_shadows": "Light source direction, shadow shapes and directions, any reflections, glare, or transparency effects.",
    "difficulty_sources": "List every feature that makes this image hard to reason about: dense packing, small text, partial visibility, similar-looking objects, perspective distortion, shadows obscuring key details, overlapping elements, fine-grained size differences, etc."
  }},

  "question_requirements": {{
    "element": "visual element name that the question targets",
    "why_needed": "what the question needs this element for",
    "location": "where it appears in the image",
    "critical_attribute": "exactly what must be legible/visible (e.g. 'the price number', 'the exact count', 'the shadow boundary')",
    "difficulty": "why this element is hard to read/find/count"
  }},

  "generation_prompt": "FULL generation prompt for GPT-image-2. Must be 800–1500 characters. Structure it with these sections in order:\n\n[CRITICAL SCENE REQUIREMENTS] (numbered list, strong imperative language):\nFor every question_requirements item, write a concrete scene instruction describing what MUST exist in the image. Describe elements as SCENE FACTS — e.g. 'A white price tag on the bottom shelf shows the number $3.49 in bold black font', 'Exactly 7 glass bottles are arranged in two rows: 4 in front, 3 behind'. Every question-critical detail (counts, text, numbers, spatial relationships, occlusions) must be woven into these scene descriptions. NEVER write the question text or answer text directly — embed the visual information that makes the question answerable. Use: 'MUST contain', 'clearly legible', 'explicitly visible'.\n\n[SCENE DESCRIPTION] (detailed prose):\nDescribe the full scene: scene type, camera angle and framing, spatial layout of all elements, density and arrangement, occlusion and overlap patterns, lighting direction, shadow shapes, depth layers, colors, textures, atmosphere. Match the original image's visual structure as closely as possible.\n\nIMPORTANT RULES:\n- NEVER include question text or answer text literally in the prompt\n- NEVER write 'the answer is X' or anything that reveals answer values\n- Describe the scene so all answer-critical information is naturally present as visual elements\n- Do NOT reference 'the original image'\n- Do NOT simplify the scene. Element count and density must match the original\n\nEnd with: 'Ultra-realistic photograph, shot on a DSLR camera, high resolution, no artistic filters, indistinguishable from a real photo.'"
}}
"""


def _build_question_block(question: str, answer: str, options: list[str]) -> str:
    if options:
        opts_text = "\n".join(f"  ({chr(65+i)}) {opt}" for i, opt in enumerate(options))
        return f"Question: {question}\nOptions:\n{opts_text}\nAnswer: {answer}"
    return f"Question: {question}\nAnswer: {answer}"


def _get_answer(row: pd.Series) -> str:
    if row["ansType"] == "choice":
        idx = int(row["choiceAns"])
        opts = list(row["options"])
        if 0 <= idx < len(opts):
            return f"({chr(65 + idx)}) {opts[idx]}"
        return str(idx)
    return str(row["blankAns"])


# ── Image loading ──────────────────────────────────────────────────────────────


def load_image_b64(img_path: str) -> Optional[tuple[str, str]]:
    abs_path = IMAGES_DIR / img_path
    if not abs_path.exists():
        log.warning(f"Image not found: {abs_path}")
        return None
    ext = abs_path.suffix.lower()
    mime = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}.get(
        ext, "image/jpeg"
    )
    return base64.b64encode(abs_path.read_bytes()).decode(), mime


# ── Qwen VL call (thinking mode) ───────────────────────────────────────────────


async def call_qwen_vl(
    client: httpx.AsyncClient,
    image_b64: tuple[str, str],
    user_text: str,
) -> Optional[dict]:
    b64, mime = image_b64
    content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
        {"type": "text", "text": user_text},
    ]

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": content},
    ]

    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 1.0,
        "enable_thinking": True,
    }

    sem = _get_sem()
    for attempt in range(MAX_RETRIES):
        try:
            async with sem:
                resp = await client.post(
                    f"{QWEN_API_BASE}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                )
                resp.raise_for_status()
                body = resp.json()

            if "choices" not in body or not body["choices"]:
                wait = _backoff(attempt)
                log.warning(f"No choices in response, retrying in {wait:.1f}s")
                await asyncio.sleep(wait)
                continue

            msg = body["choices"][0]["message"]
            return {
                "content": msg.get("content", "").strip(),
                "thinking": msg.get("reasoning_content", "").strip(),
            }

        except RETRYABLE as e:
            wait = _backoff(attempt)
            log.warning(
                f"Network error attempt {attempt + 1}/{MAX_RETRIES}: {type(e).__name__}: {e!r}  wait {wait:.1f}s"
            )
            await asyncio.sleep(wait)

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 429:
                wait = _backoff(attempt, base=20.0)
                log.warning(
                    f"HTTP 429, wait {wait:.1f}s (attempt {attempt + 1}/{MAX_RETRIES})"
                )
                await asyncio.sleep(wait)
            elif status == 400:
                log.error(f"HTTP 400: {e.response.text[:500]}")
                return None
            else:
                wait = _backoff(attempt)
                log.warning(f"HTTP {status}, wait {wait:.1f}s: {e.response.text[:200]}")
                await asyncio.sleep(wait)

        except Exception as e:
            log.error(f"Unexpected: {type(e).__name__}: {e!r}")
            return None

    log.error(f"Qwen VL failed after {MAX_RETRIES} retries")
    return None


# ── JSON extraction ────────────────────────────────────────────────────────────


def _try_parse(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", text)
        try:
            return json.loads(fixed)
        except Exception:
            return None


def _repair_truncated_json(text: str) -> Optional[dict]:
    """
    Attempt to recover a JSON object that was cut off mid-stream.
    Strategy: find the last complete top-level field value and close the object.
    """
    # Find where generation_prompt value ends (it's usually the last important field)
    # Try progressively shorter suffixes
    for end in range(len(text), max(len(text) - 500, 0), -1):
        chunk = text[:end].rstrip()
        # Try closing with varying numbers of braces/brackets
        for suffix in ["}", "]}", "}}", "]}}", "}}}", "\"}", "\"}}"]:
            candidate = chunk + suffix
            result = _try_parse(candidate)
            if result and isinstance(result, dict):
                return result
    return None


def extract_json(text: str) -> Optional[dict]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                candidate = text[start : i + 1]
                result = _try_parse(candidate)
                if result is not None:
                    return result
                start = None
    # No complete JSON found — try to repair truncated JSON
    if start is not None:
        return _repair_truncated_json(text[start:])
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
                done.add(json.loads(line)["sample_id"])
            except Exception:
                pass
    return done


# ── Skipped-samples JSON ───────────────────────────────────────────────────────


class SkippedRegistry:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._data: dict[str, dict] = self._load()

    def _load(self) -> dict[str, dict]:
        if not self.path.is_file():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def ids(self) -> set[str]:
        return set(self._data.keys())

    async def record(self, sample_id: str, info: dict) -> None:
        async with self._lock:
            self._data[sample_id] = info
            self.path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8"
            )


# ── Dataset loading ────────────────────────────────────────────────────────────


def load_tasks(parquet_path: Path) -> list[dict]:
    """Load BabyVision parquet and return list of task dicts."""
    df = pd.read_parquet(parquet_path)
    tasks = []
    for _, row in df.iterrows():
        task_id = int(row["taskId"])
        img_path = row["image"]["path"]
        question = str(row["question"])
        options = list(row["options"]) if len(row["options"]) > 0 else []
        answer = _get_answer(row)
        tasks.append(
            {
                "sample_id": f"babyvision_{task_id}",
                "task_id": task_id,
                "task_type": str(row["type"]),
                "task_subtype": str(row["subtype"]),
                "image_path": img_path,
                "question": question,
                "options": options,
                "answer": answer,
                "ans_type": str(row["ansType"]),
                "cot": str(row["coT"]) if pd.notna(row["coT"]) else "",
            }
        )
    return tasks


# ── Per-task worker ────────────────────────────────────────────────────────────


async def process_task(
    task: dict,
    client: httpx.AsyncClient,
    writer: JsonlWriter,
    skipped: SkippedRegistry,
) -> str:
    sample_id = task["sample_id"]
    subtype = task["task_subtype"]

    log.info(f"[{sample_id}] Processing ({task['task_type']} / {subtype})")

    # Pre-classify non-editable subtypes without calling the model
    if subtype in NON_EDITABLE_SUBTYPES:
        reason = f"Subtype '{subtype}' is non-editable (globally-coupled path/logic structure)"
        log.info(f"[{sample_id}] PRE-SKIP — {reason}")
        await skipped.record(
            sample_id,
            {
                "reason": reason,
                "image": task["image_path"],
                "question": task["question"],
                "timestamp": datetime.now().isoformat(),
            },
        )
        return "skipped"

    # Load image
    img_result = load_image_b64(task["image_path"])
    if img_result is None:
        log.error(f"[{sample_id}] Cannot load image {task['image_path']} — failing")
        return "failed"

    # Build user prompt
    question_block = _build_question_block(
        task["question"], task["answer"], task["options"]
    )
    user_text = _USER_TEMPLATE.format(
        task_type=task["task_type"],
        task_subtype=subtype,
        question_block=question_block,
    )

    # Call Qwen3 with thinking
    log.info(f"[{sample_id}] Calling Qwen3 (thinking mode)…")
    result = await call_qwen_vl(client, img_result, user_text)
    if result is None:
        log.error(f"[{sample_id}] API call failed — failing")
        return "failed"

    raw_content = result["content"]
    thinking = result["thinking"]
    log.info(
        f"[{sample_id}] Got response: content={len(raw_content)} chars, "
        f"thinking={len(thinking)} chars"
    )

    # Parse JSON
    parsed = extract_json(raw_content)
    if parsed is None:
        log.error(
            f"[{sample_id}] JSON parse failed. Raw content (first 600 chars):\n"
            f"{raw_content[:600]}"
        )
        debug_path = writer.path.parent / f"{sample_id}_raw_failed.txt"
        debug_path.write_text(raw_content, encoding="utf-8")
        return "failed"

    # Check skip signal from model
    if parsed.get("skip") is True:
        reason = parsed.get("reason", "")
        log.info(f"[{sample_id}] MODEL-SKIP — {reason}")
        await skipped.record(
            sample_id,
            {
                "reason": reason,
                "image": task["image_path"],
                "question": task["question"],
                "timestamp": datetime.now().isoformat(),
            },
        )
        return "skipped"

    # Validate generation_prompt
    if not parsed.get("generation_prompt"):
        log.error(f"[{sample_id}] Missing generation_prompt — failing")
        return "failed"

    log.info(
        f"[{sample_id}] generation_prompt length: {len(parsed['generation_prompt'])} chars"
    )

    record = {
        "sample_id": sample_id,
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "task_subtype": subtype,
        "original_image": task["image_path"],
        "original_question": task["question"],
        "original_answer": task["answer"],
        "ans_type": task["ans_type"],
        "options": task["options"],
        "captions": parsed,
        "thinking_summary": thinking[:1000] if thinking else "",
        "timestamp": datetime.now().isoformat(),
    }

    await writer.write(record)
    log.info(f"[{sample_id}] Done ✓")
    return "done"


# ── Main pipeline ──────────────────────────────────────────────────────────────


async def run(max_samples: Optional[int], workers: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "captions.jsonl"
    skipped_path = output_dir / "skipped.json"

    done_ids = load_done_ids(out_path)
    log.info(f"Already captioned: {len(done_ids)} task(s)")

    skipped = SkippedRegistry(skipped_path)
    skipped_ids = skipped.ids()
    log.info(f"Already skipped: {len(skipped_ids)} task(s)")

    already_handled = done_ids | skipped_ids

    if not DATASET_PARQUET.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PARQUET}")

    all_tasks = load_tasks(DATASET_PARQUET)
    log.info(f"Total tasks in dataset: {len(all_tasks)}")

    pending = [t for t in all_tasks if t["sample_id"] not in already_handled]

    if max_samples is not None:
        pending = pending[:max_samples]

    log.info(f"Pending tasks: {len(pending)} | workers: {workers}")

    if not pending:
        log.info("Nothing to do.")
        return

    writer = JsonlWriter(out_path)

    global _qwen_sem
    _qwen_sem = asyncio.Semaphore(workers)

    timeout = httpx.Timeout(API_TIMEOUT, read=API_TIMEOUT, pool=60)
    limits = httpx.Limits(
        max_connections=workers * 2, max_keepalive_connections=workers
    )

    success = skipped_count = failed = 0

    async with httpx.AsyncClient(
        timeout=timeout, limits=limits, trust_env=False
    ) as client:
        tasks = [
            asyncio.create_task(process_task(task, client, writer, skipped))
            for task in pending
        ]
        for coro in asyncio.as_completed(tasks):
            try:
                status = await coro
                if status == "done":
                    success += 1
                elif status == "skipped":
                    skipped_count += 1
                else:
                    failed += 1
            except Exception as e:
                log.error(f"Unhandled: {e!r}")
                failed += 1

    log.info(
        f"Step 1 done — success: {success}, skipped: {skipped_count}, failed: {failed}"
    )
    log.info(f"Captions → {out_path}")
    log.info(f"Skipped  → {skipped_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step 1: Generate captions + image prompts via Qwen3.6-plus VL (BabyVision)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Max concurrent Qwen requests (default: 8)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit tasks for testing (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="babyvision_output/captions",
        help="Output directory (default: babyvision_output/captions)",
    )
    args = parser.parse_args()

    asyncio.run(
        run(
            max_samples=args.max_samples,
            workers=args.workers,
            output_dir=Path(args.output_dir),
        )
    )


if __name__ == "__main__":
    main()
