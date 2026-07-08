#!/usr/bin/env python3
"""
Step 1: Generate Captions via Qwen3.6-plus VL  (thinking mode enabled)
=======================================================================
Pipeline:
  For each ZeroBench IMAGE GROUP (one or more images that share a question set):
    0. Skip if the image group is classified as non-editable (Sokoban, complex maze,
       tightly-coupled chart, Sudoku, etc.).  Skipped groups are recorded in
       skipped.json so Step 2 can also skip them.
    1. Call Qwen3.6-plus VL with thinking=True:
       - ALL questions that use this image group are sent together
       - Produce an exhaustive pixel-level scene description (caption)
       - Identify every visual element ANY of the questions depends on
       - Write a GPT-image-2 generation prompt that:
           * preserves ALL difficulty features (occlusion, clutter, small text, etc.)
           * guarantees every question-critical element is explicitly required
           * keeps the scene as close to the original as possible
    2. Save the full result to output/captions/captions.jsonl

For multi-image samples: all images sent together; one caption + prompt per image.

The ORIGINAL questions are reused verbatim on the new image. No variant_question.

Output:
  <output_dir>/captions.jsonl   — successful caption records
  <output_dir>/skipped.json     — groups skipped due to non-editable classification

Usage:
    python3 zerobench_step1_generate_captions.py [--workers 8] [--max-samples N]

Resume-safe: already-done AND already-skipped groups are skipped at startup.
"""

import argparse
import asyncio
import base64
import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx

# ── Paths ──────────────────────────────────────────────────────────────────────

DATASET_DIR = Path("/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/zerobench")
DATASET_JSONL = DATASET_DIR / "zerobench_converted.jsonl"
IMAGES_BASE = DATASET_DIR.parent  # "zerobench/images/X.png" relative to datasets/

OUTPUT_DIR = Path("output/captions")

# ── API Config ─────────────────────────────────────────────────────────────────

QWEN_API_BASE = "https://runway.devops.xiaohongshu.com/openai/qwen/v1"
QWEN_API_KEY = "YOUR_QWEN_API_KEY"
QWEN_MODEL = "qwen3.6-plus"

QWEN_MAX_CONCURRENT = 8
API_TIMEOUT = 480.0  # thinking mode needs more time
MAX_RETRIES = 4

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("step1_captions.log", encoding="utf-8"),
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


# ── Prompts ────────────────────────────────────────────────────────────────────
#
# Design (v5 — thinking mode, multi-question per image group):
#
# We leverage Qwen3's extended reasoning to produce:
#   (a) An exhaustive scene caption that describes every pixel-level detail
#   (b) A structured analysis of what EACH question visually depends on
#   (c) A GPT-image-2 generation prompt covering ALL questions at once
#
# Additionally, the model is asked to classify whether the image group is
# "non-editable" (Sokoban, complex maze, tightly-coupled chart, Sudoku, etc.)
# and output {"skip": true, "reason": "..."} in that case.
#
# Key constraints:
#   - NO simplification: element count, density, occlusion patterns are locked
#   - Surface variation only: brand/label text, numeric values, colors may change
#   - The original questions are asked verbatim on the new image
#   - The new image must be answerable (all required info present and legible)

_SYSTEM = """\
You are an expert in constructing visual reasoning benchmark datasets.

You will be given one or more images together with a SET OF VISUAL QUESTIONS and their answers.
Your job is to produce a highly detailed analysis and a precise image generation prompt,
so that a text-to-image model (GPT-image-2) can generate a NEW image that:

  1. PRESERVES the answerable difficulty for ALL questions:
     - EVERY original question (unchanged) can still be asked on the new image
     - The new image contains ALL visual information needed to answer every question
     - The difficulty is identical: same density, same occlusion patterns, same
       level of visual complexity, same need for careful observation

  2. IS AS SIMILAR AS POSSIBLE to the original:
     - Same scene type and domain
     - Same camera angle, framing, depth-of-field, lighting style
     - Same spatial layout and overall composition
     - Same number and arrangement of key elements
     - Only surface details may differ (brand names, numeric values, colors, labels)

CRITICAL — SKIP CLASSIFICATION:
Before doing anything else, you must decide whether the image group is NON-EDITABLE.
A group is NON-EDITABLE if it falls into ANY of the following categories:

  A. Sokoban or grid-logic puzzles
     Every cell participates in the global logic; changing any cell breaks correctness.
     Local/surgical editing is impossible.

  B. Complex maze layouts
     Any cell change breaks path semantics (open/dead-end relationships are globally
     coupled). Local editing is impossible.

  C. Highly-coupled diagrams / charts
     Examples: multi-series line charts, flowcharts, circuit diagrams — where every
     element has strict logical or visual dependencies; changing one element invalidates
     the meaning of others.

  D. Sudoku or similar constraint puzzles
     Changing any cell violates the global constraint network.

If the image group matches ANY category above, output ONLY this JSON and stop:
  {"skip": true, "reason": "<one-sentence explanation of which category and why>"}

Otherwise, proceed with the full analysis below.

Think step by step. Use your full reasoning capacity to:
  - Describe the image(s) at extreme detail (every object, position, text, relationship)
  - Identify exactly what EACH question needs visually to be answerable
  - Design a generation prompt that locks in all required elements and scene structure
"""

_USER_SINGLE_MULTI_Q = """\
=== IMAGE GROUP ===
This group has 1 image. All questions below apply to this single image.

=== QUESTIONS AND ANSWERS ===
{questions_block}

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
    "spatial_layout": "Describe the positions of all major elements using grid references (top-left, center, bottom-right, etc.). Describe depth layers: what is in the foreground, midground, background.",
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

  "question_requirements": [
    {{
      "question_index": 0,
      "element": "visual element name",
      "why_needed": "what this question needs this element for",
      "location": "where it appears in the image",
      "critical_attribute": "exactly what must be legible/visible (e.g. 'the price number', 'the exact count', 'the shadow boundary')",
      "difficulty": "why this element is hard to read/find/count"
    }}
  ],

  "generation_prompt": "FULL generation prompt for GPT-image-2. Must be 800–1500 characters. Structure it with these sections in order:\n\n[CRITICAL SCENE REQUIREMENTS] (numbered list, strong imperative language):\nFor EVERY question_requirements entry across ALL questions, write a concrete scene instruction describing what must exist in the image. Describe elements as SCENE FACTS — e.g. 'A white price tag on the bottom shelf shows the number $3.49 in bold black font', 'Exactly 7 glass bottles are arranged in two rows: 4 in front, 3 behind'. Every question-critical detail (counts, text, numbers, spatial relationships, occlusions) must be woven into these scene descriptions. NEVER write the question text or answer text directly — embed the visual information that makes questions answerable. Use: 'MUST contain', 'clearly legible', 'explicitly visible'.\n\n[SCENE DESCRIPTION] (detailed prose):\nDescribe the full scene: scene type, camera angle and framing, spatial layout of all elements, density and arrangement, occlusion and overlap patterns, lighting direction, shadow shapes, depth layers, colors, textures, atmosphere. Match the original image's visual structure as closely as possible.\n\nIMPORTANT RULES:\n- NEVER include question text or answer text literally in the prompt\n- NEVER write 'the answer is X' or anything that reveals answer values\n- Describe the scene so all answer-critical information is naturally present as visual elements\n\nEnd with: 'Ultra-realistic photograph, shot on a DSLR camera, high resolution, no artistic filters, indistinguishable from a real photo.'"
}}

IMPORTANT RULES:
- question_requirements must cover requirements from ALL {n_questions} questions
- generation_prompt [CRITICAL SCENE REQUIREMENTS] must address every item from question_requirements
- Do NOT simplify the scene. Element count and density must match the original
- Do NOT reference 'the original image' inside generation_prompt
"""

_USER_MULTI_IMG_MULTI_Q = """\
=== IMAGE GROUP ===
This group has {n_images} images (Image 1 through Image {n_images}).
All questions below apply to this image group.

=== QUESTIONS AND ANSWERS ===
{questions_block}

=== YOUR TASK ===
First, classify whether this image group is NON-EDITABLE (see system prompt).
If non-editable: output only {{"skip": true, "reason": "..."}} and stop.

Otherwise, study ALL {n_images} images carefully.
Produce a JSON object with ALL fields below.
Output raw JSON only — no markdown fences, no commentary outside the JSON.

{{
  "skip": false,

  "question_requirements": [
    {{
      "question_index": 0,
      "element": "visual element name",
      "image_index": 0,
      "why_needed": "what this question needs this element for",
      "location": "where it appears in that image",
      "critical_attribute": "what must be legible/visible",
      "difficulty": "why this element is hard to perceive"
    }}
  ],

  "images": [
    {{
      "image_index": 0,
      "scene_caption": {{
        "overview": "Summary of this image.",
        "camera": "Camera angle, distance, framing.",
        "spatial_layout": "Position of every major element, depth layers.",
        "objects": [
          {{
            "name": "object name",
            "count": "count",
            "position": "position",
            "appearance": "color, shape, size, orientation",
            "occlusion": "occlusion relationships",
            "readable_text": "any text or numbers verbatim"
          }}
        ],
        "text_and_numbers": "All text and numbers: content, size, position, color.",
        "occlusion_and_overlap": "All occlusion relationships in this image.",
        "lighting_and_shadows": "Light, shadows, reflections.",
        "difficulty_sources": "Every difficulty feature in this image."
      }},
      "generation_prompt": "FULL generation prompt for THIS image (800–1500 chars). Structure:\n\n[CRITICAL SCENE REQUIREMENTS] (numbered list): Every visual element this image MUST contain to support ALL questions that reference it — described as SCENE FACTS. Include counts, text, spatial relationships, occlusion patterns. NEVER write question or answer text directly.\n\n[SCENE DESCRIPTION]: Detailed scene brief matching the original's visual structure, camera angle, framing, layout, density, occlusions, lighting.\n\nIMPORTANT: NEVER include question text, answer text, or 'the answer is X'. Describe the scene so answer-critical information is naturally present.\n\nEnd: 'Ultra-realistic photograph, shot on a DSLR camera, high resolution, no artistic filters, indistinguishable from a real photograph.'"
    }}
  ],

  "difficulty_features_overall": "Difficulty features that arise from the combination of all images and the full question set."
}}

Fill one entry per image (indices 0 through {n_images_minus1}).
question_requirements entries cover ALL {n_questions} questions and may span multiple images.
Every generation_prompt must be fully self-contained — do NOT reference 'the original image'.
"""


def _build_questions_block(questions: list[str], answers: list[str]) -> str:
    """Format all questions + answers for injection into the prompt."""
    parts = []
    for i, (q, a) in enumerate(zip(questions, answers)):
        parts.append(f"[Question {i + 1}]\n{q}\n[Answer {i + 1}]: {a}")
    return "\n\n".join(parts)


# ── Image loading ──────────────────────────────────────────────────────────────


def load_image_b64(rel_path: str) -> Optional[tuple[str, str]]:
    abs_path = IMAGES_BASE / rel_path
    if not abs_path.exists():
        log.warning(f"Image not found: {abs_path}")
        return None
    ext = abs_path.suffix.lower()
    mime = "image/png" if ext == ".png" else "image/jpeg"
    return base64.b64encode(abs_path.read_bytes()).decode(), mime


# ── Qwen VL call (thinking mode) ───────────────────────────────────────────────


async def call_qwen_vl(
    client: httpx.AsyncClient,
    images_b64: list[tuple[str, str]],
    user_text: str,
) -> Optional[dict]:
    """
    Call Qwen3.6-plus VL with thinking=True.
    Returns {"content": "...", "thinking": "..."} or None on failure.
    """
    content: list[dict] = []
    for b64, mime in images_b64:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            }
        )
    content.append({"type": "text", "text": user_text})

    messages = [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": content},
    ]

    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "max_tokens": 8000,
        # thinking mode: temperature must be 1.0, enable_thinking=True
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


def extract_json(text: str) -> Optional[dict]:
    """Extract the first JSON object from text, stripping markdown fences."""
    text = re.sub(r"```(?:json)?\s*", "", text).strip("`").strip()
    # Find the outermost { ... }
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
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    # Try to fix common trailing-comma issues
                    fixed = re.sub(r",\s*([}\]])", r"\1", candidate)
                    try:
                        return json.loads(fixed)
                    except Exception:
                        pass
                start = None
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
    """
    Thread-safe registry that records groups the model decided to skip.
    Backed by a JSON file so state persists across runs.

    File format: a JSON object mapping sample_id → {reason, timestamp, images, questions}
    """

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


# ── Grouping: image-set → all questions ───────────────────────────────────────


def group_samples_by_images(
    all_samples: list[dict],
) -> list[dict]:
    """
    Group dataset samples by their canonical image set.

    Returns a list of group dicts:
      {
        "group_id":  str  (e.g. "zerobench_3"),
        "images":    list[str],  (sorted image paths, original order from first sample)
        "questions": list[str],
        "answers":   list[str],
        "raw_samples": list[dict],   # original JSONL rows
      }
    """
    # Use a stable key: tuple(sorted(images))
    key_to_group: dict[tuple, dict] = {}
    key_order: list[tuple] = []  # preserve first-seen order

    for sample in all_samples:
        imgs = sample.get("images", [])
        if not imgs:
            continue

        key = tuple(sorted(imgs))

        if key not in key_to_group:
            # Derive group_id from first image's base name
            first_stem = Path(imgs[0]).stem
            qid = first_stem.rsplit("_", 1)[0]
            group_id = f"zerobench_{qid}"

            key_to_group[key] = {
                "group_id": group_id,
                "images": list(imgs),  # preserve original order from first sample
                "questions": [],
                "answers": [],
                "raw_samples": [],
            }
            key_order.append(key)

        messages = sample.get("messages", [])
        question = (
            next((m["content"] for m in messages if m["role"] == "user"), "")
            .replace("<image>", "")
            .strip()
        )
        answer = next(
            (m["content"] for m in messages if m["role"] == "assistant"), ""
        ).strip()

        key_to_group[key]["questions"].append(question)
        key_to_group[key]["answers"].append(answer)
        key_to_group[key]["raw_samples"].append(sample)

    return [key_to_group[k] for k in key_order]


# ── Per-group worker ───────────────────────────────────────────────────────────


async def process_group(
    group: dict,
    client: httpx.AsyncClient,
    writer: JsonlWriter,
    skipped: SkippedRegistry,
) -> str:
    """
    Process one image group.

    Returns:
      "done"    — successfully captioned and written
      "skipped" — model classified as non-editable
      "failed"  — API/parse error
    """
    group_id = group["group_id"]
    images = group["images"]
    questions = group["questions"]
    answers = group["answers"]
    n_images = len(images)
    n_questions = len(questions)
    is_multi_img = n_images > 1

    log.info(
        f"[{group_id}] Processing ({n_images} image(s), {n_questions} question(s))"
    )

    # Load images
    loaded: list[tuple[str, str]] = []
    for img_rel in images:
        result = load_image_b64(img_rel)
        if result is None:
            log.error(f"[{group_id}] Cannot load image {img_rel} — failing")
            return "failed"
        loaded.append(result)

    # Build questions block
    questions_block = _build_questions_block(questions, answers)

    # Build user prompt
    if is_multi_img:
        user_text = _USER_MULTI_IMG_MULTI_Q.format(
            n_images=n_images,
            n_images_minus1=n_images - 1,
            n_questions=n_questions,
            questions_block=questions_block,
        )
    else:
        user_text = _USER_SINGLE_MULTI_Q.format(
            n_questions=n_questions,
            questions_block=questions_block,
        )

    # Call Qwen3 with thinking
    log.info(f"[{group_id}] Calling Qwen3 (thinking mode)…")
    result = await call_qwen_vl(client, loaded, user_text)
    if result is None:
        log.error(f"[{group_id}] API call failed — failing")
        return "failed"

    raw_content = result["content"]
    thinking = result["thinking"]
    log.info(
        f"[{group_id}] Got response: content={len(raw_content)} chars, "
        f"thinking={len(thinking)} chars"
    )

    # Parse JSON from content
    parsed = extract_json(raw_content)
    if parsed is None:
        log.error(
            f"[{group_id}] JSON parse failed. Raw content (first 600 chars):\n"
            f"{raw_content[:600]}"
        )
        debug_path = writer.path.parent / f"{group_id}_raw_failed.txt"
        debug_path.write_text(raw_content, encoding="utf-8")
        return "failed"

    # ── Check skip signal ──────────────────────────────────────────────────────
    if parsed.get("skip") is True:
        reason = parsed.get("reason", "")
        log.info(f"[{group_id}] SKIP — {reason}")
        await skipped.record(
            group_id,
            {
                "reason": reason,
                "images": images,
                "questions": questions,
                "timestamp": datetime.now().isoformat(),
            },
        )
        return "skipped"

    # ── Validate generation_prompt(s) ──────────────────────────────────────────
    if is_multi_img:
        imgs_parsed = parsed.get("images", [])
        if not imgs_parsed:
            log.error(f"[{group_id}] No 'images' array in response — failing")
            return "failed"
        bad = [i for i, s in enumerate(imgs_parsed) if not s.get("generation_prompt")]
        if bad:
            log.warning(f"[{group_id}] Sub-images {bad} missing generation_prompt")
        valid = [s for s in imgs_parsed if s.get("generation_prompt")]
        if not valid:
            log.error(f"[{group_id}] No valid generation_prompt — failing")
            return "failed"
    else:
        if not parsed.get("generation_prompt"):
            log.error(f"[{group_id}] Missing generation_prompt — failing")
            return "failed"

    # Log prompt length summary
    if is_multi_img:
        prompt_lens = [
            len(s.get("generation_prompt", "")) for s in parsed.get("images", [])
        ]
        log.info(f"[{group_id}] Prompt lengths per image: {prompt_lens}")
    else:
        log.info(
            f"[{group_id}] generation_prompt length: {len(parsed['generation_prompt'])} chars"
        )

    record = {
        "sample_id": group_id,
        "original_images": images,
        # All questions + answers for this image group
        "original_questions": questions,
        "original_answers": answers,
        # Legacy compat fields (first question only)
        "original_question": questions[0] if questions else "",
        "original_answer": answers[0] if answers else "",
        "n_questions": n_questions,
        "is_multi_image": is_multi_img,
        "captions": parsed,
        "thinking_summary": thinking[:1000] if thinking else "",
        "timestamp": datetime.now().isoformat(),
    }

    await writer.write(record)
    log.info(f"[{group_id}] Done ✓")
    return "done"


# ── Main pipeline ──────────────────────────────────────────────────────────────


async def run(max_samples: Optional[int], workers: int, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "captions.jsonl"
    skipped_path = output_dir / "skipped.json"

    # Resume: load already-done and already-skipped group ids
    done_ids = load_done_ids(out_path)
    log.info(f"Already captioned: {len(done_ids)} group(s)")

    skipped = SkippedRegistry(skipped_path)
    skipped_ids = skipped.ids()
    log.info(f"Already skipped: {len(skipped_ids)} group(s)")

    already_handled = done_ids | skipped_ids

    if not DATASET_JSONL.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_JSONL}")

    with open(DATASET_JSONL, encoding="utf-8") as f:
        all_samples = [json.loads(line) for line in f if line.strip()]

    log.info(f"Total samples (rows) in dataset: {len(all_samples)}")

    # Group by image set
    all_groups = group_samples_by_images(all_samples)
    log.info(f"Total image groups: {len(all_groups)}")

    # Filter pending groups
    pending = [g for g in all_groups if g["group_id"] not in already_handled]

    if max_samples is not None:
        pending = pending[:max_samples]

    log.info(f"Pending groups: {len(pending)} | workers: {workers}")

    if not pending:
        log.info("Nothing to do.")
        return

    writer = JsonlWriter(out_path)

    # Override semaphore
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
            asyncio.create_task(process_group(group, client, writer, skipped))
            for group in pending
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
        description="Step 1: Generate captions + image prompts via Qwen3.6-plus VL (thinking mode)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=100,
        help="Max concurrent Qwen requests (default: 100)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Limit image groups for testing (default: all)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="zerobench_output/captions",
        help="Output directory (default: zerobench_output/captions)",
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
