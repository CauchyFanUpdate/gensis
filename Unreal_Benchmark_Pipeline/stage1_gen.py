#!/usr/bin/env python3
"""
stage1_gen.py — Stage 1 of the zerogen pipeline (MC + OE dual-question variant).

For each seed in seed_bank.json, call Gemini 3.1 Pro (thinking mode ENABLED)
ONCE to produce TWO benchmark questions per seed:
  - mc:  a multiple-choice question with structured options
         (4 / 6 / 8 options; an option may contain ≤3 fill-in-the-blank parts)
  - oe:  an open-ended question whose answer is at most a few words

The two questions are EMITTED AS TWO INDEPENDENT SAMPLES — one record per
sample — keyed by `sample_id = "<seed_id>-mc"` / `"<seed_id>-oe"`. Both
records carry `seed_id`, `family_id`, `qtype` ("mc" | "oe") and a copy of
the shared `test_point_analysis` block.

Output: stage1_output.jsonl (append-only, one record per SAMPLE — i.e. two
lines per seed when ok). Resume by skipping seeds whose BOTH samples are
already ok in the file.

Run:
  python stage1_gen.py
  python stage1_gen.py --concurrency 16
  python stage1_gen.py --limit 5            # smoke test
  python stage1_gen.py --no-resume          # backs up existing jsonl to .bak
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx


# ── API constants (hardcoded by project convention) ───────────────────────
GEMINI_ENDPOINT = "<GEMINI_ENDPOINT>"
GEMINI_API_KEYS: list[str] = [
    # os.environ.get("GEMINI_API_KEY_1", ""),
    os.environ.get("GEMINI_API_KEY_2", ""),
]
GEMINI_MODEL = "gemini-3.1-pro-preview"

# ── Run config ────────────────────────────────────────────────────────────
HERE = Path(__file__).parent
INPUT_PATH = HERE / "seed_bank_merged.json"
OUTPUT_PATH = HERE / "stage1_output.jsonl"

PER_KEY_LIMIT = 32
DEFAULT_CONCURRENCY = PER_KEY_LIMIT * len(GEMINI_API_KEYS)
MAX_RETRIES = 6
REQUEST_TIMEOUT = 1200.0

TEMPERATURE = 0.5
TOP_P = 0.95
MAX_OUTPUT_TOKENS = 32768

GEMINI_SAFETY = [
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "OFF"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "OFF"},
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "OFF"},
]


# ── Per-family difficulty guidance ────────────────────────────────────────
FAMILY_GUIDANCE: dict[str, str] = {
    "COUNT": (
        "Counting under hard conditions: front-row occluding back-row, category "
        "filtering ('count only the red apples'), fine-grained discrimination "
        "between similar-looking items, multi-region aggregation, ordinal "
        "indexing. The answer is a specific integer."
    ),
    "SPAT": (
        "Spatial reasoning: precise relative position ('the 3rd from the left'), "
        "depth ordering, support relations, between/above/inside, viewpoint-"
        "dependent layout, projective geometry."
    ),
    "READ": (
        "Reading text in scene: small or partially-occluded text, rotated/skewed "
        "text, currency and units, distinguishing the relevant text from "
        "decorative or irrelevant text in the same scene."
    ),
    "ATTR": (
        "Attribute & state: precise color (under colored lighting), material "
        "identification (wear, coatings, transparency), surface texture, state "
        "(open/closed, full/empty, frosted/clear, on/off)."
    ),
    "COMP": (
        "Comparison & matching: identifying the odd-one-out, pairing items by a "
        "non-obvious feature, before/after diff, ranking by a hidden attribute, "
        "selecting the unique element among near-duplicates."
    ),
    "EXIS": (
        "Existence & verification: distinguishing real objects from depictions, "
        "reflections, or shadows; partial visibility; presence of small or "
        "peripheral objects; whether a constraint holds in the scene."
    ),
    "PHYS": (
        "Physics intuition: stability of a stack, support relations, what would "
        "tip/slide/fall, fluid level, gravity-sensitive arrangement, mechanical "
        "feasibility of a depicted configuration."
    ),
    "GEOM": (
        "Geometry in real scenes: parallel/perpendicular relations, angles, "
        "symmetry, congruence, estimating relative dimensions, counting "
        "geometric features (faces, edges, corners) in a real-world object."
    ),
    "DOM": (
        "Domain-specific reading: medical/technical/scientific scenes that "
        "require domain knowledge to interpret (gauges, dials, instrument "
        "scales, charts, lab equipment readings)."
    ),
    "TEMP": (
        "Temporal & sequential: order of events, timestamps, before/after "
        "states, elapsed-time inference from visible cues (clock hands, "
        "shadow length, candle wax, food state, etc.)."
    ),
    "DOCU": (
        "Document & table structure: cell lookup with header indirection, "
        "multi-criteria filtering across rows/columns, hierarchical headers, "
        "tables/forms with merged cells or footnotes."
    ),
    "MAPS": (
        "Maps & topological navigation: reachability, shortest path or "
        "valid route, identifying landmarks, orientation along a route, "
        "counting intersections or turns."
    ),
    "VIEW": (
        "Multi-view consistency: judging whether two views show the same "
        "object/scene, inferring hidden faces from visible ones, mirror/"
        "reflection consistency, viewpoint change effects."
    ),
    "OPTI": (
        "Transparent objects & optical effects: refraction shifts, mirror "
        "reflections doubling counts, glass distortion, glare hiding content, "
        "shadows giving evidence of unseen objects."
    ),
    "CONS": (
        "Semantic consistency & contradiction detection: noticing what does "
        "not belong, impossible juxtapositions, anachronisms, scale/material "
        "anomalies, items inconsistent with the apparent setting."
    ),
}


# ── System prompt for Gemini 3.1 Pro (MC + OE variant) ────────────────────

_FAMILY_BLOCK = "\n".join(f"  - {code}: {desc}" for code, desc in FAMILY_GUIDANCE.items())

SYSTEM_PROMPT = f"""You are a benchmark designer for vision-language model (VLM) evaluation. Your job is to take a short seed item and produce TWO hard, photorealistic visual-reasoning benchmark questions about the SAME high-level scene concept — one multiple-choice (MC) question and one open-ended (OE) question — each with its own elaborated scene description and image-generation prompt.

# Pipeline context

You are Stage 1 in a multi-stage pipeline:
  Stage 1 (you): seed → MC item + OE item, each with elaborated scene + image_prompt
  Stage 2: GPT-Image-2 generates ONE photorealistic image per question (so two images per seed)
  Stage 3: independent VLMs answer each question CLOSED-BOOK (gold answer hidden);
           MC is judged by exact-letter match, OE is judged by an LLM grader.

Every choice you make exists to make the final items HARD for a strong VLM, while remaining FAIR (the answer must be unambiguously visible in the resulting image, and a careful human can derive it).

# What "hard" means here

A hard item satisfies ALL of the following:
  - Multi-hop reasoning: ≥2 inferential steps from visible evidence to the answer.
  - Complex visual extraction: occlusion, fine-grained discrimination, lookalike confusers, optical effects, dense clutter, or precise spatial reasoning.
  - Unambiguous answer: a careful human looking at the image will derive the same answer with no other plausible answer.
  - Photorealistic feasibility: no impossible physics, no surreal lighting, no illustration style, no text overlays unless this is explicitly a text-reading item.
  - Lever-preserving: the seed's `lever` field describes the difficulty mechanism — both your MC and your OE items MUST exercise this same mechanism.

A trivial item (one-step lookup, dominant central object, large readable text giving the answer, answer derivable without the lever) is a failure of this stage.

# Inputs you receive

For each seed:
  - id           — opaque identifier; the prefix before '-' is the family code
  - sub_skill    — narrow skill being probed
  - domain       — scene category
  - scene        — short canonical scene description (the BACKBONE you elaborate)
  - question     — the seed's reference question (use as inspiration)
  - answer       — the seed's reference gold answer
  - lever        — WHY the seed is hard (MUST be preserved)

# Family guidance

The seed's id begins with a 3-4 letter family code. Use the matching family note to focus your difficulty design and avoid skill-drift:

{_FAMILY_BLOCK}

# Mandatory ordering inside each item

For BOTH the MC item and the OE item, you MUST design in this strict order:

  step 1: elaborated_scene  (concrete visual prose, all counts/colors/layout/occluders/optical effects)
  step 2: question          (clear, single-answer, exercises the lever)
  step 3: gold answer       (the correct answer to that question, derived from the scene)
  step 4: (MC only) options (only after the scene/question/answer are fixed — DO NOT design options first)
  step 5: image_prompt      (a single dense English paragraph for GPT-Image-2 that realises the scene)

Designing options before answer is forbidden. Designing image before scene is forbidden.

# MC item rules

- The MC question must exercise the seed's lever and stay on the seed's sub_skill.
- The number of options must be exactly 4, 6, or 8 (you choose; harder items deserve more distractors).
- Exactly ONE option is correct — the gold designed in step 3.
- Each option is a STRUCTURED object: `{{"label": "A", "parts": ["..."]}}` where `parts` is a list of 1-3 strings.
  * `parts` length = 1 for ordinary single-answer options.
  * `parts` length 2-3 turns the option into a fill-in-the-blank style: each `part` is the text for one blank; the option is "correct" iff EVERY `part` is correct. The MC question must phrase the prompt as "(blank 1) ... (blank 2) ..." style so the parts mapping is unambiguous.
- Use fill-in-the-blank options only when the question naturally has multiple coupled answers (e.g., "color and material of the rightmost flask: (___) and (___)"). Otherwise prefer single-part options.
- Distractors must be plausible (visually consistent with the image), but must be ruled out by careful inspection.
- Labels are uppercase letters starting at A. Never reorder.
- `correct_label` is the label of the correct option.
- The MC question stem must NOT itself reveal which option is correct, and must NOT include the gold answer text outside the correct option.

# OE item rules

- The OE question must exercise the seed's lever and stay on the seed's sub_skill.
- Phrase the question so the natural answer is extremely short — at most 4 words, ideally 1-3 (e.g., "How many ...?", "What color ...?", "Which side ...?", "What number ...?").
- The gold `answer` field is a SHORT string (≤ 30 chars, no full sentence, no trailing punctuation).
- The OE question may be a refinement of the seed question or the MC question, but it MUST be answerable WITHOUT seeing options (do not say "which of the above" etc.).
- The OE answer is what a strict grader will compare against the verifier model's free-form output.

# Image prompts (one per item)

The image_prompt is a TARGETED rendering brief, NOT a scene description. It feeds GPT-Image-2 and should contain ONLY what the answerer needs to see. There is no word-count limit — write as much or as little as you need, but every clause must earn its place.

- Necessity test: every clause you write must satisfy at least ONE of the following — otherwise DELETE it.
  (a) it depicts the visual evidence the question is asking about (the target object/state/count/text/relation),
  (b) it depicts a disambiguator that rules out the strongest confuser,
  (c) it is the minimum scene context required for the target to be physically/visually coherent (e.g., a counter for items to sit on).
- BAN list (do NOT mention unless directly load-bearing on the answer): ambient lighting mood, weather, time of day, decorative props, wall/floor colors, background scenery, camera lens type, depth of field, photographic-style adjectives, narrative details about people, brand names, weather aesthetics.
- Be specific where it matters: exact counts, colors, positions, states, sizes, occluders, and disambiguators of the question-relevant objects. Be silent everywhere else.
- Do NOT include the question or the answer text in the image prompt.
- Do NOT request any text overlay unless the item is a text-reading item.
- MC and OE prompts may share the same minimal scene backbone but may differ slightly to best serve their own question (e.g., framing nudged to make MC distractors clearly distinguishable). Both should look like the same domain.

Quality bar: a reader of your image_prompt should be able to predict your gold answer from the prompt alone, with no extraneous detail wasted on the renderer.

# Hard constraints (any violation = bad item)

- Photorealism only. The image_prompt may NOT use words like "illustration", "cartoon", "watercolor", "painting", "anime", "surreal", "dreamlike", "neon glow", "fantasy lighting", "impossible", "abstract".
- The image_prompt MUST NOT cause the answer to appear as readable text in the image (e.g., do NOT write "a sign reading '27'" for a counting question).
- Disambiguators MUST be present in the image_prompt.
- The lever MUST remain intact in BOTH items.
- Stay close to the seed domain and counts.
- Provide ≥2 entries in `reasoning_hops`.
- Output everything in English.

# Output

Return ONLY a single JSON object with EXACTLY these keys, in this order. No markdown fences, no leading text, no trailing text, no commentary:

{{
  "id": "<echo seed id>",
  "family_id": "<3-4 letter family code from id, e.g., COUNT, SPAT, OPTI>",
  "sub_skill": "<echo seed sub_skill>",
  "test_point_analysis": {{
    "lever_restated": "<one-sentence restatement of the seed lever>",
    "core_challenge": "<one sentence: why a strong VLM would struggle>",
    "reasoning_hops": ["hop 1: ...", "hop 2: ...", "hop 3 (optional): ..."],
    "confusers": ["<non-target item that visually resembles the target>", "..."],
    "disambiguators": ["<concrete visual cue that separates target from confuser>", "..."],
    "shared_notes": "<one sentence: any extra context shared by both MC and OE items>"
  }},
  "mc": {{
    "elaborated_scene": "<3-7 sentences of concrete visual prose>",
    "question": "<the MC question stem; if fill-in-the-blank options are used, mark blanks consistently>",
    "options": [
      {{"label": "A", "parts": ["..."]}},
      {{"label": "B", "parts": ["..."]}},
      "..."
    ],
    "correct_label": "<one of A/B/C/...>",
    "answer_derivation": "<short paragraph showing how the correct option follows from the scene>",
    "image_prompt": "<English paragraph for GPT-Image-2; ONLY load-bearing details for the question/answer/disambiguator; no ambient or decorative description; length is whatever the question needs, no padding>",
    "decision_notes": {{
      "n_options": 4,
      "uses_fill_in_blank": false,
      "rationale": "<one sentence on option-count + distractor design>"
    }}
  }},
  "oe": {{
    "elaborated_scene": "<3-7 sentences of concrete visual prose>",
    "question": "<the OE question; phrased to elicit a 1-4 word answer>",
    "answer": "<the gold short answer, ≤ 30 chars>",
    "answer_derivation": "<short paragraph showing how the answer follows from the scene>",
    "image_prompt": "<English paragraph for GPT-Image-2; ONLY load-bearing details for the question/answer/disambiguator; no ambient or decorative description; length is whatever the question needs, no padding>",
    "decision_notes": {{
      "answer_word_count": 2,
      "rationale": "<one sentence on phrasing for short-answer extractability>"
    }}
  }}
}}

Output the JSON object and nothing else."""


# ── Per-seed user prompt ──────────────────────────────────────────────────

def build_user_prompt(seed: dict) -> str:
    return (
        "Seed:\n"
        f"- id: {seed['id']}\n"
        f"- sub_skill: {seed['sub_skill']}\n"
        f"- domain: {seed['domain']}\n"
        f"- scene: {seed['scene']}\n"
        f"- question: {seed['question']}\n"
        f"- answer: {seed['answer']}\n"
        f"- lever: {seed['lever']}\n\n"
        "Now think carefully (use your reasoning) and produce the JSON object as specified. "
        "Remember: design elaborated_scene → question → gold answer FIRST, and only THEN design "
        "the MC options. Both items must preserve the seed's lever."
    )


# ── JSON extraction & validation ──────────────────────────────────────────

def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        return json.loads(text[start : end + 1])


_REQUIRED_TOP_KEYS = {"id", "family_id", "sub_skill", "test_point_analysis", "mc", "oe"}
_REQUIRED_TPA_KEYS = {
    "lever_restated", "core_challenge", "reasoning_hops",
    "confusers", "disambiguators", "shared_notes",
}
_REQUIRED_MC_KEYS = {
    "elaborated_scene", "question", "options",
    "correct_label", "answer_derivation", "image_prompt", "decision_notes",
}
_REQUIRED_OE_KEYS = {
    "elaborated_scene", "question", "answer",
    "answer_derivation", "image_prompt", "decision_notes",
}
_VALID_OPTION_COUNTS = {4, 6, 8}
_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def validate_output(seed_id: str, data: Any) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, f"top-level must be dict, got {type(data).__name__}"
    missing_top = _REQUIRED_TOP_KEYS - set(data.keys())
    if missing_top:
        return False, f"missing top-level keys: {sorted(missing_top)}"
    if data["id"] != seed_id:
        return False, f"id mismatch: expected {seed_id!r}, got {data['id']!r}"

    tpa = data.get("test_point_analysis")
    if not isinstance(tpa, dict):
        return False, "test_point_analysis must be a dict"
    missing_tpa = _REQUIRED_TPA_KEYS - set(tpa.keys())
    if missing_tpa:
        return False, f"missing test_point_analysis keys: {sorted(missing_tpa)}"
    if not isinstance(tpa["reasoning_hops"], list) or len(tpa["reasoning_hops"]) < 2:
        return False, "reasoning_hops must be a list with ≥2 entries"
    for k in ("confusers", "disambiguators"):
        if not isinstance(tpa[k], list):
            return False, f"test_point_analysis.{k} must be a list"

    mc = data.get("mc")
    if not isinstance(mc, dict):
        return False, "mc must be a dict"
    missing_mc = _REQUIRED_MC_KEYS - set(mc.keys())
    if missing_mc:
        return False, f"missing mc keys: {sorted(missing_mc)}"
    for f in ("elaborated_scene", "question", "answer_derivation", "image_prompt"):
        v = mc.get(f)
        if not isinstance(v, str) or not v.strip():
            return False, f"mc.{f} missing or empty"
    options = mc.get("options")
    if not isinstance(options, list) or len(options) not in _VALID_OPTION_COUNTS:
        return False, f"mc.options must be a list of 4/6/8 entries; got {len(options) if isinstance(options, list) else type(options).__name__}"
    expected_labels = list(_LABELS[: len(options)])
    seen_labels: list[str] = []
    for i, opt in enumerate(options):
        if not isinstance(opt, dict):
            return False, f"mc.options[{i}] must be a dict"
        label = opt.get("label")
        parts = opt.get("parts")
        if label != expected_labels[i]:
            return False, f"mc.options[{i}].label must be {expected_labels[i]!r}, got {label!r}"
        if not isinstance(parts, list) or not (1 <= len(parts) <= 3):
            return False, f"mc.options[{i}].parts must be a list of 1-3 strings"
        for j, p in enumerate(parts):
            if not isinstance(p, str) or not p.strip():
                return False, f"mc.options[{i}].parts[{j}] must be a non-empty string"
        seen_labels.append(label)
    correct_label = mc.get("correct_label")
    if correct_label not in seen_labels:
        return False, f"mc.correct_label {correct_label!r} not in option labels {seen_labels}"

    oe = data.get("oe")
    if not isinstance(oe, dict):
        return False, "oe must be a dict"
    missing_oe = _REQUIRED_OE_KEYS - set(oe.keys())
    if missing_oe:
        return False, f"missing oe keys: {sorted(missing_oe)}"
    for f in ("elaborated_scene", "question", "answer_derivation", "image_prompt"):
        v = oe.get(f)
        if not isinstance(v, str) or not v.strip():
            return False, f"oe.{f} missing or empty"
    a = oe.get("answer")
    if isinstance(a, (int, float)):
        a = str(a)
        oe["answer"] = a
    if not isinstance(a, str) or not a.strip():
        return False, "oe.answer missing or empty"
    if len(a) > 60:
        return False, f"oe.answer too long ({len(a)} chars); should be ≤ ~30"

    return True, ""


# ── Gemini call with retries ──────────────────────────────────────────────

_RETRYABLE = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.ProxyError,
)


class GeminiKeyPool:
    """Round-robin pool of Gemini API keys, each gated by its own semaphore."""

    def __init__(self, keys: list[str], per_key_limit: int):
        if not keys:
            raise ValueError("GeminiKeyPool requires at least one key")
        self._keys = list(keys)
        self._sems = [asyncio.Semaphore(per_key_limit) for _ in keys]
        self._idx = 0
        self._idx_lock = asyncio.Lock()

    @asynccontextmanager
    async def use_key(self):
        async with self._idx_lock:
            i = self._idx % len(self._keys)
            self._idx += 1
        async with self._sems[i]:
            yield self._keys[i]


async def call_gemini(
    client: httpx.AsyncClient,
    key_pool: GeminiKeyPool,
    seed: dict,
    system_prompt: str,
) -> dict:
    seed_id = seed["id"]
    user_text = build_user_prompt(seed)
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {
            "temperature": TEMPERATURE,
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
            "topP": TOP_P,
        },
        "safetySettings": GEMINI_SAFETY,
    }

    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            async with key_pool.use_key() as api_key:
                headers = {"api-key": api_key, "Content-Type": "application/json"}
                resp = await client.post(GEMINI_ENDPOINT, headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()

            if "Code" in body and "Error" in body and "candidates" not in body:
                err_str = str(body.get("Error", ""))
                low = err_str.lower()
                if any(s in low for s in ("429", "resource_exhausted", "quota", "exhausted")):
                    wait = min(25 * (attempt + 1), 180)
                    print(
                        f"[stage1/{seed_id}] proxy quota error, wait {wait}s "
                        f"(attempt {attempt + 1}/{MAX_RETRIES}) | {err_str[:160]!r}",
                        flush=True,
                    )
                    await asyncio.sleep(wait)
                    continue
                raise RuntimeError(f"Gemini proxy error Code={body.get('Code')} | {err_str[:300]}")

            candidates = body.get("candidates", [])
            if not candidates:
                raise RuntimeError(f"Gemini response has no candidates: {str(body)[:300]}")
            candidate = candidates[0]
            finish = candidate.get("finishReason", "")
            parts_out = candidate.get("content", {}).get("parts", [])
            text_parts = [p.get("text", "") for p in parts_out if "text" in p and not p.get("thought")]
            full_text = "".join(text_parts).strip()
            if not full_text:
                raise RuntimeError(f"empty visible-text response (finishReason={finish!r})")

            try:
                data = extract_json(full_text)
            except json.JSONDecodeError as e:
                raise RuntimeError(f"could not parse JSON: {e!r} | text head: {full_text[:300]!r}")

            ok, err = validate_output(seed_id, data)
            if not ok:
                raise RuntimeError(f"output schema invalid: {err}")
            return data

        except _RETRYABLE as e:
            last_err = e
            wait = min(2 ** attempt + attempt, 30)
            print(
                f"[stage1/{seed_id}] network retry {attempt + 1}/{MAX_RETRIES}: "
                f"{type(e).__name__}: {e!r} (wait {wait}s)",
                flush=True,
            )
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            body_head = (e.response.text or "")[:200]
            if status == 429:
                wait = min(20 * (attempt + 1), 120)
                print(
                    f"[stage1/{seed_id}] HTTP 429, wait {wait}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})",
                    flush=True,
                )
                await asyncio.sleep(wait)
                last_err = e
            elif status in (400, 403):
                raise RuntimeError(f"HTTP {status}: {body_head}")
            else:
                wait = min(5 * (attempt + 1), 60)
                print(
                    f"[stage1/{seed_id}] HTTP {status}, wait {wait}s | {body_head!r}",
                    flush=True,
                )
                await asyncio.sleep(wait)
                last_err = e
        except Exception as e:
            last_err = e
            wait = min(2 ** attempt, 30)
            print(
                f"[stage1/{seed_id}] retry {attempt + 1}/{MAX_RETRIES}: "
                f"{type(e).__name__}: {e!r} (wait {wait}s)",
                flush=True,
            )
            await asyncio.sleep(wait)

    raise RuntimeError(f"[{seed_id}] failed after {MAX_RETRIES} retries: {last_err!r}")


# ── JSONL writer + resume helpers ─────────────────────────────────────────

class JsonlAppender:
    def __init__(self, path: Path):
        self.path = path
        self._lock = asyncio.Lock()

    async def write(self, record: dict) -> None:
        async with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_done_seeds(path: Path) -> set[str]:
    """A seed is 'done' iff BOTH its samples (<seed_id>-mc, <seed_id>-oe) are ok."""
    done_samples: set[str] = set()
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
            if rec.get("status") != "ok":
                continue
            sample_id = rec.get("sample_id")
            if sample_id:
                done_samples.add(sample_id)
    done_seeds: set[str] = set()
    by_seed: dict[str, set[str]] = {}
    for sid in done_samples:
        if "-" not in sid:
            continue
        seed_id, qtype = sid.rsplit("-", 1)
        if qtype not in ("mc", "oe"):
            continue
        by_seed.setdefault(seed_id, set()).add(qtype)
    for seed_id, qtypes in by_seed.items():
        if {"mc", "oe"}.issubset(qtypes):
            done_seeds.add(seed_id)
    return done_seeds


# ── Per-seed orchestration ────────────────────────────────────────────────

def _build_sample_record(
    seed: dict,
    family_id: str,
    qtype: str,
    elapsed_s: float,
    *,
    status: str,
    stage1_payload: dict | None = None,
    error: str | None = None,
) -> dict:
    seed_id = seed["id"]
    rec: dict[str, Any] = {
        "sample_id": f"{seed_id}-{qtype}",
        "seed_id":   seed_id,
        "family_id": family_id,
        "qtype":     qtype,
        "status":    status,
        "seed":      seed,
        "elapsed_s": round(elapsed_s, 2),
        "created_at": int(time.time()),
    }
    if stage1_payload is not None:
        rec["stage1"] = stage1_payload
    if error is not None:
        rec["error"] = error
    return rec


async def process_one(
    client: httpx.AsyncClient,
    key_pool: GeminiKeyPool,
    writer: JsonlAppender,
    family_id: str,
    seed: dict,
    system_prompt: str,
) -> tuple[str, bool]:
    seed_id = seed["id"]
    t0 = time.time()
    try:
        data = await call_gemini(client, key_pool, seed, system_prompt)
    except Exception as e:
        elapsed = time.time() - t0
        err_str = f"{type(e).__name__}: {e!r}"
        # write one failed record per qtype so resume can retry the seed
        for qtype in ("mc", "oe"):
            await writer.write(_build_sample_record(
                seed, family_id, qtype, elapsed, status="failed", error=err_str,
            ))
        print(
            f"[stage1/{seed_id}] FAILED in {round(elapsed,2)}s: "
            f"{type(e).__name__}: {str(e)[:200]}",
            flush=True,
        )
        return seed_id, False

    elapsed = time.time() - t0
    shared_tpa = data["test_point_analysis"]

    mc_payload = {"test_point_analysis": shared_tpa, **data["mc"]}
    oe_payload = {"test_point_analysis": shared_tpa, **data["oe"]}

    await writer.write(_build_sample_record(
        seed, family_id, "mc", elapsed, status="ok", stage1_payload=mc_payload,
    ))
    await writer.write(_build_sample_record(
        seed, family_id, "oe", elapsed, status="ok", stage1_payload=oe_payload,
    ))

    n_options = len(data["mc"]["options"])
    correct = data["mc"]["correct_label"]
    oe_ans = data["oe"]["answer"]
    print(
        f"[stage1/{seed_id}] ok in {round(elapsed,2)}s | "
        f"mc={n_options}-opt correct={correct} | oe_ans={oe_ans!r} | "
        f"hops={len(shared_tpa['reasoning_hops'])} | wrote 2 samples",
        flush=True,
    )
    return seed_id, True


# ── Main ──────────────────────────────────────────────────────────────────

async def main_async(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output)

    with open(input_path, encoding="utf-8") as f:
        bank = json.load(f)
    families = bank["families"]

    flat: list[tuple[str, dict]] = []
    for fam in families:
        fid = fam["family_id"]
        for s in fam["seeds"]:
            flat.append((fid, s))
    total_seeds = len(flat)
    print(
        f"Loaded {len(families)} families, {total_seeds} seeds from {input_path}.",
        flush=True,
    )

    if args.limit > 0:
        flat = flat[: args.limit]
        print(f"--limit {args.limit} → processing first {len(flat)} seeds", flush=True)

    if args.no_resume and output_path.exists():
        backup = output_path.with_suffix(output_path.suffix + ".bak")
        output_path.rename(backup)
        print(f"--no-resume: moved existing {output_path} → {backup}", flush=True)

    done = load_done_seeds(output_path)
    if done:
        print(
            f"resume: {len(done)} seeds already have BOTH samples ok in {output_path} — will skip",
            flush=True,
        )

    pending = [(fid, s) for fid, s in flat if s["id"] not in done]
    if not pending:
        print("nothing to do — all seeds already processed.", flush=True)
        return
    n_keys = len(GEMINI_API_KEYS)
    per_key = max(1, args.concurrency // n_keys)
    total_concurrency = per_key * n_keys
    print(
        f"submitting {len(pending)} seeds | concurrency={total_concurrency} "
        f"({n_keys} keys × {per_key}/key) | model={GEMINI_MODEL} | thinking=on",
        flush=True,
    )

    key_pool = GeminiKeyPool(GEMINI_API_KEYS, per_key)
    timeout = httpx.Timeout(REQUEST_TIMEOUT, read=REQUEST_TIMEOUT, pool=60)
    limits = httpx.Limits(
        max_connections=total_concurrency * 2,
        max_keepalive_connections=total_concurrency,
    )
    writer = JsonlAppender(output_path)

    t_start = time.time()
    async with httpx.AsyncClient(timeout=timeout, limits=limits, trust_env=False) as client:
        tasks = [
            process_one(client, key_pool, writer, fid, s, SYSTEM_PROMPT)
            for fid, s in pending
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

    n_ok = 0
    n_fail = 0
    for r in results:
        if isinstance(r, tuple) and r[1]:
            n_ok += 1
        else:
            n_fail += 1
    print(
        f"\nDone in {time.time() - t_start:.1f}s. "
        f"ok={n_ok} failed={n_fail} → {output_path}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: Gemini 3.1 Pro seed → MC + OE benchmark items"
    )
    parser.add_argument("--input", default=str(INPUT_PATH))
    parser.add_argument("--output", default=str(OUTPUT_PATH))
    parser.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    parser.add_argument("--limit", type=int, default=0, help="if >0, process only first N seeds")
    parser.add_argument("--no-resume", action="store_true", help="back up existing JSONL and start fresh")
    args = parser.parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\ninterrupted — partial results are in the output JSONL; rerun to resume.", flush=True)
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
