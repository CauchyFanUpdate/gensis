"""
Shared utilities for MMMU edit pipeline (gen_pipeline_mmmu_pro).
Adapts edit_mmmu_pro infra to process standard MMMU (dev+val) data
from wanghan's selected_data/manifest.jsonl.
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Optional

# Override per-API concurrency BEFORE importing _v5_common
os.environ.setdefault("QWEN_MAX_CONCURRENT", "256")
os.environ.setdefault("GPT_IMAGE_MAX_CONCURRENT", "16")

# Import everything from _v5_common (parent of edit_mmmu)
_EDIT_MMMU_DIR = Path(__file__).resolve().parent.parent / "edit_mmmu"
sys.path.insert(0, str(_EDIT_MMMU_DIR))
from _v5_common import (  # noqa: E402, F401
    # API clients
    chat_gemini_vl, chat_qwen, gpt_image_edit,
    # Message builders
    qwen_text_messages, qwen_vl_messages,
    # Image utils
    prepare_image, extract_json,
    # Concurrency
    _get_gemini_sem, _get_qwen_sem, _get_gpt_img_sem,
    # Retry
    _backoff_with_jitter, RETRYABLE,
    # HTTP
    make_clients,
    # JSONL
    JsonlWriter, load_done_keys, jsonl_iter,
    # Logging
    log,
    # Image classification
    classify_image_type, IMAGE_TYPE_NOUNS,
    # Axis taxonomy
    UNIVERSAL_AXES, AXIS_TIER_DEFAULTS, NATURAL_CAPS,
    TEST_POINT_FROZEN_PRIORS, READABILITY_TEMPLATES,
)

# ── Paths ──────────────────────────────────────────────────────────────────

PIPELINE_DIR = Path(__file__).resolve().parent

# MMMU selected data (from wanghan's stage0 output, 505 high-score samples)
MANIFEST_JSONL = Path(
    "/mnt/tidal-alsh01/dataset/perceptionVLM/wanghan/edit_mmmu/selected_data/manifest.jsonl"
)

OUTPUT_DIR = PIPELINE_DIR / "output"

# ── API config ─────────────────────────────────────────────────────────────

QWEN_API_BASE = "https://runway.devops.xiaohongshu.com/openai/qwen/v1"
QWEN_API_KEY = "YOUR_QWEN_API_KEY"
QWEN_MODEL = "qwen3.6-plus"

GPT_IMAGE_ENDPOINT = os.environ.get("GPT_IMAGE_ENDPOINT", "https://runway.devops.rednote.life")
GPT_IMAGE_API_KEY = os.environ.get("GPT_IMAGE_API_KEY", "YOUR_GPT_IMAGE_API_KEY")

# ── Option utilities ────────────────────────────────────────────────────────

LETTERS = "ABCDEFGHIJ"


def parse_options(raw) -> list[str]:
    """Parse MMMU options which may be a list or a stringified Python list."""
    if isinstance(raw, list):
        return [str(o) for o in raw]
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, list):
                return [str(o) for o in parsed]
        except Exception:
            pass
        # Fallback: try JSON
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(o) for o in parsed]
        except Exception:
            pass
    return []


def format_options(options: list[str]) -> str:
    """Format options inline: A) x  B) y  ..."""
    return "  ".join(
        f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options) if i < len(LETTERS)
    )


def format_options_multiline(options: list[str]) -> str:
    """Format options one per line: A) x\\nB) y\\n..."""
    return "\n".join(
        f"{LETTERS[i]}) {opt}" for i, opt in enumerate(options) if i < len(LETTERS)
    )


def answer_text(letter: str, options: list[str]) -> str:
    """Map letter code to option text."""
    idx = ord(letter.upper()) - ord("A")
    if 0 <= idx < len(options):
        return options[idx]
    return letter


def fix_options_inline(
    actual_answer: str,
    options: list[str],
    original_answer_letter: str,
    rng: random.Random | None = None,
) -> tuple[list[str], str]:
    """
    Ensure actual_answer is in options. Returns (fixed_options, new_answer_letter).
    If already present: find its index. If not: replace a random distractor (never the
    original correct answer slot).
    """
    if rng is None:
        rng = random.Random()

    norm = lambda s: re.sub(r"[^\w]", "", s.lower().strip())
    actual_norm = norm(actual_answer)

    for i, opt in enumerate(options):
        if norm(opt) == actual_norm:
            return options, LETTERS[i]

    orig_idx = ord(original_answer_letter.upper()) - ord("A")
    replaceable = [i for i in range(len(options)) if i != orig_idx]
    if not replaceable:
        replaceable = list(range(len(options)))

    replace_idx = rng.choice(replaceable)
    new_options = list(options)
    new_options[replace_idx] = actual_answer

    rng.shuffle(new_options)
    new_idx = next(
        (i for i, o in enumerate(new_options) if norm(o) == actual_norm),
        0,
    )
    return new_options, LETTERS[new_idx]


def normalize_question(q: str) -> str:
    """Replace <image N> with <image>, collapse whitespace."""
    q = re.sub(r"<image\s*\d*\s*>", "<image>", q)
    q = re.sub(r"  +", " ", q)
    return q.strip()


# ── MMMU data loading ───────────────────────────────────────────────────────

def load_selected_samples(
    manifest_path: str | Path = MANIFEST_JSONL,
) -> list[dict]:
    """
    Load MMMU samples from wanghan's selected_data/manifest.jsonl.
    Returns enriched records ready for stage1_edit.
    """
    records = []
    for rec in jsonl_iter(str(manifest_path)):
        options = parse_options(rec.get("options") or rec.get("options_raw", []))
        if not options:
            log(f"  [{rec.get('id')}] no options, skip", "common")
            continue

        img_type = rec.get("img_type", "")
        records.append({
            "id": rec["id"],
            "question": rec["question"],
            "options": options,
            "options_raw": rec.get("options_raw", ""),
            "answer": rec["answer"],
            "question_type": rec.get("question_type", "multiple-choice"),
            "img_type": img_type,
            "subject": rec.get("subject", ""),
            "subfield": rec.get("subfield", ""),
            "split": rec.get("split", ""),
            "topic_difficulty": rec.get("topic_difficulty", ""),
            "images": rec.get("images", []),
            # Actual file paths (already resolved by wanghan's stage0)
            "source_image_paths": rec.get("selected_image_paths", []),
            "source_json_path": rec.get("selected_json_path", ""),
            "score": rec.get("score", 0),
            "image_type_group": classify_image_type(img_type),
        })
    return records


def get_image_bytes(record: dict) -> Optional[bytes]:
    """Read the first image file for a record."""
    paths = record.get("source_image_paths", [])
    if not paths:
        return None
    p = Path(paths[0])
    return p.read_bytes() if p.is_file() else None


def get_all_image_bytes(record: dict) -> list[bytes]:
    """Read all image files for a record (for multi-image MMMU samples)."""
    result = []
    for path_str in record.get("source_image_paths", []):
        p = Path(path_str)
        if p.is_file():
            result.append(p.read_bytes())
    return result


# ── Meta builder ────────────────────────────────────────────────────────────

def build_mmmu_meta(record: dict) -> dict:
    """Build metadata dict for output records."""
    return {
        "id": record["id"],
        "answer_letter": record["answer"],
        "options": list(record["options"]),
        "question_type": record.get("question_type", "multiple-choice"),
        "img_type": record.get("img_type", ""),
        "image_type_group": record.get("image_type_group", ""),
        "subject": record.get("subject", ""),
        "subfield": record.get("subfield", ""),
        "split": record.get("split", ""),
        "topic_difficulty": record.get("topic_difficulty", ""),
        "num_options": len(record.get("options", [])),
    }
