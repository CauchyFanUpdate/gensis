#!/usr/bin/env python3
"""
MMMU Pipeline — Stage 2 Only (emma_v3): Cosmetic Diversification from Original Images

Directly diversifies original MMMU images WITHOUT any Stage 1 answer-changing edit.
The original answer is preserved throughout. 1:5 ratio (1 original → 5 diversified images).

Per source sample:
  Call 1  [Qwen VL]      -> analysis JSON: scene_description + axis_table + frozen_predicate
                            (same prompt as stage1 but we only use the analysis fields)
  Selector [Qwen VL]     -> 5 diversification plans (axis valuations + slot assignments),
                            cached to disk for resume
  per plan (parallel):
    Call 2b [Qwen text]  -> diversification prose (150-260 words for GPT-Image-2)
    Call 3b [GPT-Image-2]-> 1 diversified image (from original image)
    Verifier [Qwen VL]   -> 4 yes/no checks (class / relation / readability / natural)
    CoT gen  [Qwen VL]   -> thinking-mode chain-of-thought anchored on original answer
    Verify   [Qwen VL]   -> confirm CoT+answer correct on generated image
    Rewrite  [Qwen text] -> rewrite distractor options for freshness

Output:
  output_emma_v3_stage2_only/stage2_train.jsonl
  output_emma_v3_stage2_only/stage2_rejected.jsonl
  output_emma_v3_stage2_only/stage2_images/{sample_key}.png
  output_emma_v3_stage2_only/stage2_selector_cache.jsonl
  output_emma_v3_stage2_only/stage2_call1_cache.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import random
import re
import time
import traceback
from pathlib import Path
from typing import Optional

import httpx

from _common import (
    AXIS_TIER_DEFAULTS,
    IMAGE_TYPE_NOUNS,
    NATURAL_CAPS,
    READABILITY_TEMPLATES,
    TEST_POINT_FROZEN_PRIORS,
    UNIVERSAL_AXES,
    answer_text,
    build_mmmu_meta,
    chat_qwen,
    classify_image_type,
    extract_json,
    fix_options_inline,
    format_options,
    format_options_multiline,
    get_all_image_bytes,
    gpt_image_edit,
    JsonlWriter,
    jsonl_iter,
    load_done_keys,
    load_selected_samples,
    log,
    make_clients,
    normalize_question,
    prepare_image,
    qwen_text_messages,
    qwen_vl_messages,
    _get_qwen_sem,
    _backoff_with_jitter,
    RETRYABLE,
    QWEN_API_BASE,
    QWEN_API_KEY,
    QWEN_MODEL,
    GPT_IMAGE_ENDPOINT,
    GPT_IMAGE_API_KEY,
    LETTERS,
    MANIFEST_JSONL,
)

OUTPUT_EMMA_V3_STAGE2_ONLY = Path(__file__).resolve().parent / "output_emma_v3_stage2_only"

# ── Constants ───────────────────────────────────────────────────────────────

OPS = ("remove", "add", "modify_property", "substitute_category", "context_change")
EFFECT_TYPES = ("delta", "literal_replace", "flip_boolean", "categorical_shift")
STRUCTURAL_PATTERNS = (
    "dual_value_disambiguation", "real_vs_depicted", "partial_visibility",
    "category_filtering", "lookalike_confusion", "spatial_ordering",
    "composition_state", "text_in_context", "table_lookup",
    "chart_value_reading", "diagram_label_reading",
    "medical_finding_identification", "music_notation_reading", "other",
)

# ── Image-type overrides for Call 1 ────────────────────────────────────────

_IMAGE_TYPE_ADDENDA: dict[str, str] = {
    "table": (
        "\n## IMAGE TYPE OVERRIDE: TABLE / SPREADSHEET\n\n"
        "ALL physical-world axes MUST be set to `frozen` with freeze_constraint = "
        '"not applicable to rendered table".\n'
        "Free styling axes: subject_color, subject_material, subject_specific_appearance.\n"
        'verifier_checks.natural_check: "Is this a legible, well-formatted data table?"\n'
    ),
    "chart": (
        "\n## IMAGE TYPE OVERRIDE: CHART / GRAPH\n\n"
        'Freeze ALL physical-world axes with freeze_constraint = "not applicable to rendered chart".\n'
        "Free styling axes: subject_color, subject_material, subject_specific_appearance.\n"
        'verifier_checks.natural_check: "Is this a legible, well-formatted chart with readable axes?"\n'
    ),
    "diagram": (
        "\n## IMAGE TYPE OVERRIDE: DIAGRAM / SCHEMATIC\n\n"
        'Freeze ALL physical-world axes with freeze_constraint = "not applicable to rendered diagram".\n'
        "Free styling axes: subject_color, subject_material, subject_specific_appearance.\n"
        'verifier_checks.natural_check: "Is this a legible, well-structured diagram with clear labels?"\n'
    ),
    "medical": (
        "\n## IMAGE TYPE OVERRIDE: MEDICAL IMAGE\n\n"
        "Freeze most axes. Only allow subject_specific_appearance and subject_color (free).\n"
        'Other axes frozen with "not applicable to clinical imaging".\n'
        'verifier_checks.natural_check: "Is this a plausible medical image with adequate quality?"\n'
    ),
    "sheet_music": (
        "\n## IMAGE TYPE OVERRIDE: SHEET MUSIC\n\n"
        "Freeze ALL physical-world axes. Free: subject_specific_appearance, subject_color.\n"
        'verifier_checks.natural_check: "Is this legible sheet music with correct staff structure?"\n'
    ),
}

# ── Selector type-specific guidance ─────────────────────────────────────────

_SELECTOR_TYPE_ADDENDA: dict[str, str] = {
    "table": (
        "\n## DIVERSIFICATION FOR TABLES\n\n"
        "Vary only visual styling (color scheme, fonts, borders, shading). "
        "Every data value and label must remain identical.\n"
    ),
    "chart": (
        "\n## DIVERSIFICATION FOR CHARTS\n\n"
        "Vary color palette, rendering style, grid style, legend position, background. "
        "Every data value and axis label must remain identical.\n"
    ),
    "diagram": (
        "\n## DIVERSIFICATION FOR DIAGRAMS\n\n"
        "Vary color scheme, line style, node/shape style, font, background. "
        "Every label and connection must remain identical.\n"
    ),
}

# ── Call 2b type-specific instructions ─────────────────────────────────────

_CALL2B_TYPE_ADDENDA: dict[str, str] = {
    "table": (
        "TABLE-SPECIFIC: Describe a completely new visual theme (color scheme, fonts, borders, "
        "cell padding). Be extremely concrete with hex colors and font names. "
        "Every cell value must be preserved EXACTLY. Do NOT describe the data content.\n\n"
    ),
    "chart": (
        "CHART-SPECIFIC: Describe a completely new visual theme (bar/line colors, background, "
        "grid style, legend position). Be concrete with hex colors. "
        "Every data value and axis label must be preserved EXACTLY.\n\n"
    ),
    "diagram": (
        "DIAGRAM-SPECIFIC: Describe a completely new visual theme (node colors, border style, "
        "line weight, font, background). Every label and connection must be preserved EXACTLY.\n\n"
    ),
}

_PROSE_BANNED_WORDS = (
    "subtle", "subtly", "slightly", "a touch of", "a hint of", "hints of",
    "gentle", "softly", "lightly", "mild", "mildly",
)

_PROSE_PUSH_WORDS = (
    "completely", "entirely", "wholly", "dramatically", "fully",
    "thoroughly", "transformed into",
)


# ── Call 1: Qwen VL analysis (axis_table + frozen_predicate) ───────────────

def _build_call1_params(
    question: str,
    answer: str,
    options: list[str],
    answer_letter: str = "",
    image_type_group: str = "natural_photo",
    n_images: int = 1,
) -> tuple[str, str]:
    caps = NATURAL_CAPS if image_type_group == "natural_photo" else {}
    axes_block = "\n".join(
        f"  - {ax}  [default: {AXIS_TIER_DEFAULTS[ax]}]"
        + (f"\n      natural cap: {caps[ax]}" if ax in caps else "")
        for ax in UNIVERSAL_AXES
    )
    readability_block = "\n".join(
        f"  - {tp}: {tmpl}" for tp, tmpl in READABILITY_TEMPLATES.items()
    )
    test_point_priors_block = "\n".join(
        f"  - {tp}: typically frozen → {list(axes)}"
        for tp, axes in TEST_POINT_FROZEN_PRIORS.items()
    )
    multi_img_note = (
        f"\n**Note: This sample has {n_images} images.** "
        "All images are shown to you for analysis.\n"
        if n_images > 1 else ""
    )

    system = (
        "You are an analytical assistant for an image diversification pipeline. "
        "Given an image (or images) and a visual multiple-choice question with its ground-truth answer, "
        "produce a structured analysis JSON for downstream diversification. "
        "We will use your analysis to apply cosmetic visual transformations to this image "
        "WHILE KEEPING THE ANSWER UNCHANGED.\n\n"
        + multi_img_note
        + "\n## Goal: answer-preserving visual diversification\n\n"
        "We want to generate 5 visually distinct versions of this image that all have the same answer. "
        "Your analysis identifies WHAT must stay frozen (the test point + answer-determining elements) "
        "and WHAT can be freely varied (background, lighting, color, surface, camera, etc.).\n\n"
        "## Core framework\n\n"
        "  - `frozen_predicate`: the logical structure that determines the answer. "
        "Anything in must_preserve must not change.\n"
        "  - `axis_table`: 19 visual axes, each classified as 'free' (vary aggressively) or 'frozen' (do not touch).\n"
        "  - `question_template` / `answer_template`: slot-templated question and answer.\n\n"
        "## When the image is un-diversifiable\n\n"
        "If the image has NO free axes (e.g., a pure symbolic diagram where every pixel is answer-relevant), "
        "set `is_diversifiable=false`. Otherwise set `is_diversifiable=true`.\n\n"
        "## axis_table — 19 universal visual axes:\n"
        f"{axes_block}\n\n"
        "## test_point → frozen-axes prior:\n"
        f"{test_point_priors_block}\n\n"
        "## readability check templates:\n"
        f"{readability_block}\n\n"
    )

    if image_type_group != "natural_photo":
        system += _IMAGE_TYPE_ADDENDA.get(image_type_group, "")

    system += "\n## Output\n\nReturn ONLY valid JSON, no markdown fences, no commentary."

    axis_schema_lines = ",\n".join(
        f'    {{"axis": "{ax}", "safety": "free|frozen|depends", '
        f'"aggressive_value": "<concrete ≥3-options string if free, else null>", '
        f'"freeze_constraint": "<what to preserve if frozen, else null>"}}'
        for ax in UNIVERSAL_AXES
    )

    opts_display = format_options(options)
    answer_display = f"{answer_letter} = {answer}" if answer_letter else answer

    user_text = (
        f"Original question: {question}\n"
        f"Options ({len(options)} choices): {opts_display}\n"
        f"Original answer: {answer_display}\n"
        f"Image type hint: {image_type_group}\n\n"
        "Analyze this image and produce JSON with EXACTLY these fields:\n\n"
        "{\n"
        f'  "image_type": "{image_type_group}",\n'
        '  "scene_description": "<1-2 sentence natural caption explicitly describing what the image shows>",\n'
        '  "test_point_analysis": {\n'
        '    "core_challenge": "<one sentence: why this question is non-trivial>",\n'
        f'    "structural_pattern": "<one of: {" | ".join(STRUCTURAL_PATTERNS)}>",\n'
        '    "must_preserve": ["<answer-relevant element that must not change>", "..."],\n'
        '    "edit_must_operate_within": "<one sentence: constraint any diversification must respect>"\n'
        "  },\n"
        '  "disambiguators_to_preserve": [\n'
        '    {"feature": "<concrete visual feature>", "scope": "<which objects>"}\n'
        "  ],\n"
        '  "diversification_assessment": {"is_diversifiable": true},\n'
        '  "frozen_predicate": {\n'
        '    "answer_function": "<predicate string>",\n'
        '    "answer_value": "<original answer as string>",\n'
        '    "must_preserve": ["<bullet 1>"],\n'
        '    "verifier_checks": {\n'
        '      "class_check": "<yes/no Q with slot placeholders>",\n'
        '      "relation_check": "<yes/no Q with slot placeholders>",\n'
        '      "readability_check": "<yes/no Q from readability templates>",\n'
        '      "natural_check": "Is this a plausible real photograph (not illustration / surreal)?"\n'
        "    },\n"
        '    "slots": {\n'
        '      "S": {\n'
        '        "role": "<short noun phrase>",\n'
        '        "shape_constraint": "<visual shape>",\n'
        '        "default_class_singular": "<e.g., wine bottle>",\n'
        '        "default_class_plural": "<e.g., wine bottles>",\n'
        '        "compatible_classes": [{"singular": "...", "plural": "..."}, ...],\n'
        '        "slot_replaceability": "free | shape_only | locked"\n'
        "      },\n"
        '      "N": {"value": "<answer value>", "frozen": true}\n'
        "    }\n"
        "  },\n"
        '  "question_template": "<question with slot placeholders, no answer value>",\n'
        '  "answer_template": "<usually <N> or literal>",\n'
        '  "axis_table": [\n'
        f"{axis_schema_lines}\n"
        "  ]\n"
        "}\n\n"
        "Constraints:\n"
        "  - axis_table MUST contain ALL 19 axes.\n"
        "  - For free axes, aggressive_value MUST list ≥3 concrete options separated by ' / '.\n"
        "  - If is_diversifiable=false, frozen_predicate/question_template/answer_template/axis_table can be empty.\n"
        "  - Output ONLY the JSON object. No markdown, no commentary."
    )
    return system, user_text


def _validate_call1_schema(call1: dict) -> tuple[bool, str]:
    div = call1.get("diversification_assessment", {})
    if not div.get("is_diversifiable", True):
        return True, ""

    fp = call1.get("frozen_predicate")
    if not isinstance(fp, dict) or not fp:
        return False, "missing/empty frozen_predicate"
    for f in ("answer_function", "answer_value", "verifier_checks", "slots"):
        if f not in fp:
            return False, f"frozen_predicate.{f} missing"
    vc = fp.get("verifier_checks", {})
    for f in ("class_check", "relation_check", "readability_check", "natural_check"):
        if not isinstance(vc.get(f), str) or not vc[f].strip():
            return False, f"verifier_checks.{f} missing or empty"
    slots = fp.get("slots", {})
    if not isinstance(slots, dict) or not slots:
        return False, "frozen_predicate.slots missing or empty"
    for sname, sdef in slots.items():
        if not isinstance(sdef, dict):
            return False, f"slot {sname!r} not a dict"
        if "compatible_classes" in sdef:
            ccs = sdef["compatible_classes"]
            if not isinstance(ccs, list):
                return False, f"slot {sname!r} compatible_classes not a list"
            if sdef.get("slot_replaceability") != "locked" and len(ccs) < 3:
                return False, f"slot {sname!r} needs ≥3 compatible_classes (got {len(ccs)})"
            for cc in ccs:
                if not isinstance(cc, dict) or "singular" not in cc or "plural" not in cc:
                    return False, f"slot {sname!r} compatible_classes entries must have singular+plural"
            sr = sdef.get("slot_replaceability", "")
            if sr not in ("free", "shape_only", "locked"):
                return False, f"slot {sname!r} slot_replaceability invalid: {sr!r}"

    qt = call1.get("question_template", "")
    if not isinstance(qt, str) or not qt.strip():
        return False, "question_template missing or empty"

    at = call1.get("answer_template", "")
    if not isinstance(at, str) or not at.strip():
        return False, "answer_template missing or empty"

    at_list = call1.get("axis_table", [])
    if not isinstance(at_list, list):
        return False, "axis_table not a list"
    seen_axes = set()
    for entry in at_list:
        if not isinstance(entry, dict):
            return False, "axis_table entry not a dict"
        ax = entry.get("axis")
        if ax not in UNIVERSAL_AXES:
            return False, f"axis_table unknown axis: {ax!r}"
        if ax in seen_axes:
            return False, f"axis_table duplicate axis: {ax!r}"
        seen_axes.add(ax)
        safety = entry.get("safety")
        if safety not in ("free", "frozen", "depends"):
            return False, f"axis {ax!r} safety invalid: {safety!r}"
        if safety in ("free", "depends"):
            av = entry.get("aggressive_value", "")
            if not isinstance(av, str) or not av.strip():
                return False, f"axis {ax!r} ({safety}) missing aggressive_value"
        if safety == "frozen":
            fc = entry.get("freeze_constraint", "")
            if not isinstance(fc, str) or not fc.strip():
                return False, f"axis {ax!r} (frozen) missing freeze_constraint"
    missing = set(UNIVERSAL_AXES) - seen_axes
    if missing:
        return False, f"axis_table missing axes: {sorted(missing)}"

    return True, ""


# ── Template instantiation ──────────────────────────────────────────────────

def _instantiate_template(template: str, slot_values: dict) -> str:
    if not template:
        return ""
    out = template
    for sname, sval in (slot_values or {}).items():
        if isinstance(sval, dict):
            singular = sval.get("singular", "")
            plural = sval.get("plural", singular)
            out = out.replace(f"<{sname}_plural>", plural)
            out = out.replace(f"<{sname}_singular>", singular)
            out = out.replace(f"<{sname}>", singular)
        else:
            out = out.replace(f"<{sname}>", str(sval))
    return out


# ── Disk caches (selector + call1) ─────────────────────────────────────────

class _DiskCache:
    """Generic disk-backed JSONL cache keyed by source_id."""

    def __init__(self, path: Path, key_field: str = "source_id"):
        self.path = path
        self.key_field = key_field
        self._mem: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            for line in open(self.path, encoding="utf-8"):
                rec = json.loads(line)
                k = rec.get(self.key_field)
                if k:
                    self._mem[k] = rec
        except Exception as e:
            log(f"cache load warning ({self.path.name}): {e!r}", "stage2")

    def get(self, key: str) -> Optional[dict]:
        return self._mem.get(key)

    async def put(self, key: str, data: dict) -> None:
        async with self._lock:
            rec = {self.key_field: key, **data, "created_at": int(time.time())}
            self._mem[key] = rec
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


class SelectorCache:
    """Disk-backed cache for selector plan lists. Key = source_id."""

    def __init__(self, path: Path):
        self.path = path
        self._mem: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            for line in open(self.path, encoding="utf-8"):
                rec = json.loads(line)
                k = rec.get("source_id")
                plans = rec.get("plans")
                if k and isinstance(plans, list):
                    self._mem[k] = plans
        except Exception as e:
            log(f"selector cache load warning: {e!r}", "stage2")

    def get(self, source_id: str) -> Optional[list[dict]]:
        return self._mem.get(source_id)

    async def put(self, source_id: str, plans: list[dict]) -> None:
        async with self._lock:
            self._mem[source_id] = plans
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "source_id": source_id,
                    "plans": plans,
                    "created_at": int(time.time()),
                }, ensure_ascii=False) + "\n")


# ── Selector prompt ──────────────────────────────────────────────────────────

def _slot_pool_brief(slots: dict) -> str:
    lines = []
    for sname, sdef in (slots or {}).items():
        if not isinstance(sdef, dict):
            continue
        if "compatible_classes" in sdef:
            ccs = sdef["compatible_classes"][:8]
            cc_str = ", ".join(c.get("singular", "?") for c in ccs if isinstance(c, dict))
            lines.append(
                f"    <{sname}>  role={sdef.get('role', '?')}  "
                f"replaceability={sdef.get('slot_replaceability', '?')}  "
                f"shape={sdef.get('shape_constraint', '')!r}\n"
                f"        compatible: [{cc_str}]"
            )
        elif "value" in sdef:
            lines.append(f"    <{sname}>  value={sdef.get('value', '?')!r}  frozen=true")
    return "\n".join(lines) if lines else "    (no slots)"


def _axis_table_brief(axis_table: list) -> str:
    lines = []
    for entry in axis_table or []:
        if not isinstance(entry, dict):
            continue
        ax = entry.get("axis", "?")
        sf = entry.get("safety", "?")
        if sf == "frozen":
            lines.append(f"  - {ax}  [FROZEN]  preserve: {entry.get('freeze_constraint', '')!r}")
        else:
            lines.append(f"  - {ax}  [{sf.upper()}]  options: {entry.get('aggressive_value', '')!r}")
    return "\n".join(lines) if lines else "  (empty)"


def _build_selector_prompt(call1: dict, n_prompts: int, image_type_group: str = "natural_photo") -> tuple[str, str]:
    type_noun = IMAGE_TYPE_NOUNS.get(image_type_group, "image")
    if image_type_group == "natural_photo":
        distribution_rule = (
            "    3. Stay strictly within the natural-photo distribution. "
            "No surreal effects, no illustrations, no extreme filters.\n"
        )
    else:
        distribution_rule = (
            f"    3. The result must remain a legible, well-structured {type_noun}. "
            f"Vary styling axes aggressively but keep structural integrity intact.\n"
        )

    system = (
        "You are an image augmentation strategy planner. You receive an ORIGINAL (unedited) image "
        "plus its per-image analysis (frozen_predicate + axis_table). Your job is to produce "
        f"{n_prompts} visually-distinct diversification plans that ALL preserve the original answer.\n\n"
        "  HARD constraints:\n"
        "    1. FROZEN axes must NOT be touched.\n"
        "    2. The frozen_predicate's verifier_checks must still hold on the diversified image.\n"
        + distribution_rule
        + "    4. For slots with replaceability=='locked', keep the default class.\n\n"
        "  OPTIMIZATION:\n"
        "    A. Be MAXIMALLY AGGRESSIVE on every free axis (push to the upper end of options).\n"
        "    B. INTER-PLAN diversity: different values for each axis across the N plans.\n"
        "    C. INTRA-PLAN coherence: each plan should describe a visually coherent scene.\n\n"
        "Return ONLY valid JSON, no markdown fences, no commentary."
    )

    if image_type_group in _SELECTOR_TYPE_ADDENDA:
        system += _SELECTOR_TYPE_ADDENDA[image_type_group]

    fp = call1.get("frozen_predicate", {})
    slots = fp.get("slots", {})
    test_point = call1.get("test_point_analysis", {})

    user_text = (
        f"## Image analysis (from Call 1)\n\n"
        f"scene_description: {call1.get('scene_description', '')}\n"
        f"test_point: structural_pattern={test_point.get('structural_pattern', '')}, "
        f"core_challenge={test_point.get('core_challenge', '')!r}\n\n"
        f"## frozen_predicate\n"
        f"  answer_function: {fp.get('answer_function', '')}\n"
        f"  answer_value:    {fp.get('answer_value', '')!r}\n"
        f"  must_preserve:   {fp.get('must_preserve', [])}\n"
        f"  verifier_checks: {fp.get('verifier_checks', {})}\n"
        f"  slots:\n{_slot_pool_brief(slots)}\n\n"
        f"## axis_table (FROZEN = off-limits; FREE/DEPENDS = your variables)\n"
        f"{_axis_table_brief(call1.get('axis_table', []))}\n\n"
        f"## Task: produce {n_prompts} maximally diverse plans, ALL preserving the original answer.\n\n"
        f"For each plan:\n"
        f"  - slot_assignments: singular + plural for each replaceable class slot.\n"
        f"  - axis_values: concrete value for every FREE axis.\n"
        f"  - diversity_signature: 3-5 short tags.\n"
        f"  - rationale: one sentence on why this plan is answer-safe.\n\n"
        f"Output JSON:\n"
        f"{{\n"
        f'  "plans": [\n'
        f"    {{\n"
        f'      "plan_id": 1,\n'
        f'      "slot_assignments": {{"S": {{"singular": "ceramic vase", "plural": "ceramic vases"}}}},\n'
        f'      "axis_values": {{"background_setting": "minimalist white studio", "...": "..."}},\n'
        f'      "diversity_signature": ["bg_studio", "lighting_cold"],\n'
        f'      "rationale": "<one sentence>"\n'
        f"    }}\n"
        f"  ]\n"
        f"}}\n\n"
        f"Generate exactly {n_prompts} plans. Output ONLY the JSON object."
    )
    return system, user_text


def _parse_selector_output(text: str, n_prompts: int) -> list[dict]:
    try:
        data = extract_json(text)
    except Exception:
        return []
    plans = data.get("plans") if isinstance(data, dict) else None
    if not isinstance(plans, list):
        return []
    out: list[dict] = []
    for p in plans:
        if not isinstance(p, dict):
            continue
        sa = p.get("slot_assignments", {}) or {}
        av = p.get("axis_values", {}) or {}
        if not isinstance(sa, dict) or not isinstance(av, dict):
            continue
        norm_sa: dict[str, dict] = {}
        for k, v in sa.items():
            if isinstance(v, dict) and "singular" in v and "plural" in v:
                norm_sa[k] = {"singular": str(v["singular"]), "plural": str(v["plural"])}
            elif isinstance(v, str):
                norm_sa[k] = {"singular": v, "plural": v}
        norm_av = {str(k): str(v) for k, v in av.items() if isinstance(v, (str, int, float))}
        sig = p.get("diversity_signature", []) or []
        if isinstance(sig, str):
            sig = [sig]
        out.append({
            "plan_id": int(p.get("plan_id", len(out) + 1)),
            "slot_assignments": norm_sa,
            "axis_values": norm_av,
            "diversity_signature": [str(x) for x in sig if x],
            "rationale": str(p.get("rationale", "")),
        })
    return out[:n_prompts] if len(out) > n_prompts else out


# ── Call 2b: diversification prose ─────────────────────────────────────────

def _build_call2b_prompt(call1: dict, plan: dict, image_type_group: str = "natural_photo") -> tuple[str, str]:
    fp = call1.get("frozen_predicate", {}) or {}
    must_preserve = fp.get("must_preserve", []) or []
    answer_function = fp.get("answer_function", "")
    type_noun = IMAGE_TYPE_NOUNS.get(image_type_group, "image")

    if image_type_group == "natural_photo":
        style_rule = (
            "NATURAL-PHOTO RULE:\n"
            "  - Result MUST look like a real photograph. NO illustration / cartoon / surreal.\n"
            "  - NO extreme filters. Outdoor weather: clear / overcast / golden hour only.\n\n"
        )
    else:
        style_rule = (
            f"STYLE RULE FOR {type_noun.upper()}:\n"
            f"  - Vary styling aggressively (colors, fonts, borders) but keep all content intact and legible.\n\n"
        )

    system = (
        "You write image editing instructions for an AI image editor (GPT-Image-2). "
        "Your output is a SINGLE FLOWING PARAGRAPH of natural English prose, 150-260 words.\n\n"
        "Format rules:\n"
        "  - NO bullet points, NO labels, NO section headers, NO line breaks within the paragraph.\n"
        "  - Every FREE axis must appear with at least one concrete adjective or noun phrase.\n"
        "  - Be vivid and specific (named materials, named colors, named objects).\n\n"
        "STRENGTH RULES:\n"
        f"  - Avoid softening words: {', '.join(_PROSE_BANNED_WORDS)}\n"
        f"  - Prefer amplification: {', '.join(_PROSE_PUSH_WORDS)}\n"
        "  - The result must look obviously different from the input — substantive change on every free axis.\n\n"
        + style_rule
        + _CALL2B_TYPE_ADDENDA.get(image_type_group, "")
        + "FROZEN PREDICATE:\n"
        "  - The must_preserve list appears at the end — paraphrase into the paragraph's close.\n"
        "  - Do NOT mention the question or answer.\n"
    )

    plan_lines = []
    for sname, sval in (plan.get("slot_assignments") or {}).items():
        if isinstance(sval, dict):
            plan_lines.append(f"  * slot <{sname}>: {sval.get('plural', sval.get('singular', '?'))}")
    for ax, val in (plan.get("axis_values") or {}).items():
        plan_lines.append(f"  * axis {ax}: {val}")
    plan_block = "\n".join(plan_lines) or "  (no axis instructions)"
    must_block = "\n".join(f"  - {x}" for x in must_preserve) or "  (none)"

    if image_type_group in ("table", "chart", "diagram"):
        open_instruction = f"  1. Open by describing the new visual theme for this {type_noun}."
    else:
        open_instruction = "  1. Open by describing the new background and surface with the subjects in place."

    user_text = (
        f"BASE SCENE (original image — repaint it with the new axis values):\n"
        f"  {call1.get('scene_description', '')}\n\n"
        f"AXIS VALUES + SLOT CLASSES (every entry must appear in your prose):\n"
        f"{plan_block}\n\n"
        f"FROZEN PREDICATE (must be preserved — answer logic = {answer_function!r}):\n"
        f"{must_block}\n\n"
        f"Write the single flowing paragraph. The paragraph should:\n"
        f"{open_instruction}\n"
        f"  2. Cover every FREE axis with a concrete adjective/phrase.\n"
        f"  3. Close with a positive paraphrase of the must_preserve constraints.\n"
        f"Output ONLY the paragraph, no other text."
    )
    return system, user_text


# ── Verifier ────────────────────────────────────────────────────────────────

def _build_verifier_prompt(verifier_checks: dict) -> tuple[str, str]:
    system = (
        "You are a strict visual verifier. Answer each question with a single word: yes or no. "
        "No explanation. Use only visual evidence. If a premise can't be evaluated, answer 'no'."
    )
    qs = verifier_checks
    user_text = (
        "Look at the image and answer each question with one word: yes or no.\n\n"
        f"Q1 (CLASS):       {qs.get('class_check', '')}\n"
        f"Q2 (RELATION):    {qs.get('relation_check', '')}\n"
        f"Q3 (READABILITY): {qs.get('readability_check', '')}\n"
        f"Q4 (NATURAL):     {qs.get('natural_check', '')}\n\n"
        "Output EXACTLY:\nA1: <yes|no>\nA2: <yes|no>\nA3: <yes|no>\nA4: <yes|no>\n"
    )
    return system, user_text


def _parse_verifier_output(text: str) -> dict:
    out = {"a1": "", "a2": "", "a3": "", "a4": "", "all_pass": False, "raw": text}
    for i in (1, 2, 3, 4):
        m = re.search(rf"A{i}\s*:\s*(yes|no)", text, re.IGNORECASE)
        if m:
            out[f"a{i}"] = m.group(1).strip().lower()
    out["all_pass"] = all(out[f"a{i}"] == "yes" for i in (1, 2, 3, 4))
    return out


# ── CoT generation (thinking mode) ─────────────────────────────────────────

SYSTEM_COT_GEN = (
    "You are an expert visual question answering assistant. "
    "Reason step by step about the image content to arrive at the correct answer. "
    "Be concise but thorough."
)

PROMPT_COT_GEN = (
    "Look at the image and answer the following question step by step.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\nCorrect answer: {answer}\n\n"
    "Reason carefully about the image to explain how you arrive at this answer."
)


async def _call_qwen_thinking(
    client: httpx.AsyncClient,
    messages: list[dict],
    max_tokens: int = 8000,
    max_retries: int = 20,
    log_prefix: str = "cot",
) -> Optional[str]:
    sem = _get_qwen_sem()
    for attempt in range(max_retries):
        try:
            async with sem:
                resp = await client.post(
                    f"{QWEN_API_BASE.rstrip('/')}/chat/completions",
                    json={
                        "model": QWEN_MODEL,
                        "messages": messages,
                        "max_tokens": max_tokens,
                        "temperature": 1.0,
                        "enable_thinking": True,
                    },
                    headers={"Authorization": f"Bearer {QWEN_API_KEY}"},
                )
                resp.raise_for_status()
                body = resp.json()
            if "choices" not in body or not body["choices"]:
                await asyncio.sleep(_backoff_with_jitter(10, attempt))
                continue
            msg = body["choices"][0]["message"]
            reasoning = (msg.get("reasoning_content") or "").strip()
            if reasoning:
                return reasoning
            content = (msg.get("content") or "").strip()
            return content or None
        except RETRYABLE as e:
            await asyncio.sleep(_backoff_with_jitter(2 ** attempt, 0, jitter_frac=0.5))
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                await asyncio.sleep(_backoff_with_jitter(15, attempt))
            elif e.response.status_code == 400:
                log(f"  [{log_prefix}] HTTP 400: {e.response.text[:300]}", "stage2")
                return None
            else:
                await asyncio.sleep(_backoff_with_jitter(5, attempt))
        except Exception as e:
            if attempt < max_retries - 1:
                await asyncio.sleep(_backoff_with_jitter(2, attempt))
            else:
                log(f"  [{log_prefix}] failed after {max_retries} retries: {e!r}", "stage2")
                return None
    return None


def _clean_cot(cot: str) -> str:
    cot = re.sub(r"\n*\*\*Final Answer[:\s].*$", "", cot, flags=re.DOTALL)
    cot = re.sub(r"\n*The correct (?:answer|option) is [A-Z]\.?\s*$", "", cot, flags=re.IGNORECASE)
    return cot.rstrip()


# ── Answer verification prompt ──────────────────────────────────────────────

SYSTEM_VERIFY_ANSWER = (
    "You are a rigorous visual question answering verifier. "
    "Determine whether the given answer is correct based on the image content. "
    "Be strict and accurate."
)

PROMPT_VERIFY_ANSWER = (
    "Look at the image and verify whether the following answer is correct.\n\n"
    "Question: {question}\n\nOptions:\n{options}\n\n"
    "Chain-of-thought:\n{cot}\n\nClaimed answer: {answer}\n\n"
    "Is the claimed answer correct? Respond in strict JSON:\n"
    '{{\n  "correct": true or false,\n  "reason": "brief explanation"\n}}'
)

# ── Option rewriting ─────────────────────────────────────────────────────────

SYSTEM_OPTION_REWRITE = (
    "You are an expert question designer. Generate new challenging multiple-choice distractors. "
    "Return ONLY a valid JSON object."
)

PROMPT_OPTION_REWRITE = (
    "Given this question, correct answer, and original options, "
    "generate {num_distractors} NEW distractor options.\n\n"
    "Question: {question}\nCorrect answer: {answer}\nOriginal options: {options}\n\n"
    "Requirements:\n"
    "1. Keep the correct answer EXACTLY as-is\n"
    "2. Generate exactly {num_distractors} NEW distractors different from originals\n"
    "3. Distractors must be plausible but clearly incorrect\n\n"
    'Return: {{\n  "distractors": ["d1", "d2", ...]\n}}'
)


# ── Stage 2 from-original pipeline ──────────────────────────────────────────

class Stage2FromOriginalPipeline:
    def __init__(
        self,
        output_dir: str | Path,
        manifest_path: str | Path = MANIFEST_JSONL,
        qwen_api_base: str = QWEN_API_BASE,
        qwen_api_key: str = QWEN_API_KEY,
        qwen_model: str = QWEN_MODEL,
        gpt_image_endpoint: str = GPT_IMAGE_ENDPOINT,
        gpt_image_api_key: str = GPT_IMAGE_API_KEY,
        max_concurrent: int = 64,
        max_retries: int = 20,
        api_timeout: float = 1200.0,
        resume: bool = True,
        limit: int = 0,
        n_prompts_per_source: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "stage2_images"
        self.train_jsonl = self.output_dir / "stage2_train.jsonl"
        self.rejected_jsonl = self.output_dir / "stage2_rejected.jsonl"
        self.selector_cache_path = self.output_dir / "stage2_selector_cache.jsonl"
        self.call1_cache_path = self.output_dir / "stage2_call1_cache.jsonl"
        self.manifest_path = manifest_path

        self.qwen_api_base = qwen_api_base
        self.qwen_api_key = qwen_api_key
        self.qwen_model = qwen_model
        self.gpt_image_endpoint = gpt_image_endpoint
        self.gpt_image_api_key = gpt_image_api_key

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.api_timeout = api_timeout
        self.resume = resume
        self.limit = limit
        self.n_prompts = n_prompts_per_source

        self._train_writer: Optional[JsonlWriter] = None
        self._rejected_writer: Optional[JsonlWriter] = None
        self._done_keys: set[str] = set()
        self._selector_cache: Optional[SelectorCache] = None
        self._call1_cache: Optional[_DiskCache] = None
        self._inner = None
        self._proxy = None

    async def _generate_cot(self, image_b64: str, question: str, options: list[str],
                             answer_str: str, sample_key: str) -> Optional[str]:
        prompt = PROMPT_COT_GEN.format(
            question=question, options=format_options_multiline(options), answer=answer_str,
        )
        messages = qwen_vl_messages(SYSTEM_COT_GEN, [(image_b64, "image/png")], prompt)
        return await _call_qwen_thinking(
            self._inner, messages, max_tokens=8000,
            max_retries=self.max_retries, log_prefix=f"cot/{sample_key}",
        )

    async def _verify_answer(self, image_b64: str, question: str, options: list[str],
                              cot: str, answer_str: str, sample_key: str) -> Optional[bool]:
        prompt = PROMPT_VERIFY_ANSWER.format(
            question=question, options=format_options_multiline(options),
            cot=cot, answer=answer_str,
        )
        messages = qwen_vl_messages(
            SYSTEM_VERIFY_ANSWER, [(image_b64, "image/png")], prompt, no_think=True
        )
        try:
            text = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                messages, max_tokens=512, temperature=0.1,
                max_retries=self.max_retries, log_prefix=f"verify_ans/{sample_key}",
            )
        except Exception as e:
            log(f"  [{sample_key}] verify_answer failed: {e!r}", "stage2")
            return None
        try:
            parsed = extract_json(text)
            return bool(parsed.get("correct", False)) if isinstance(parsed, dict) else None
        except Exception:
            return None

    async def _rewrite_options(self, question: str, correct_answer_text: str,
                                old_options: list[str], sample_key: str) -> Optional[tuple[list[str], str]]:
        num_distractors = max(1, len(old_options) - 1)
        prompt = PROMPT_OPTION_REWRITE.format(
            question=question, answer=correct_answer_text,
            options=json.dumps(old_options, ensure_ascii=False),
            num_distractors=num_distractors,
        )
        messages = qwen_text_messages(SYSTEM_OPTION_REWRITE, prompt, no_think=True)
        try:
            text = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                messages, max_tokens=2048, temperature=0.7,
                max_retries=self.max_retries, log_prefix=f"opts/{sample_key}",
            )
        except Exception as e:
            log(f"  [{sample_key}] rewrite_options failed: {e!r}", "stage2")
            return None
        try:
            parsed = extract_json(text)
            distractors = parsed.get("distractors", []) if isinstance(parsed, dict) else []
            if not isinstance(distractors, list) or len(distractors) < num_distractors:
                return None
            distractors = [str(d) for d in distractors[:num_distractors]]
            new_options = [correct_answer_text] + distractors
            random.shuffle(new_options)
            answer_idx = new_options.index(correct_answer_text)
            return new_options, LETTERS[answer_idx]
        except Exception:
            return None

    async def _process_one_plan_image(
        self,
        source_id: str,
        question: str,
        answer_letter: str,
        options: list[str],
        call1: dict,
        plan: dict,
        prompt_idx: int,
        stage2_bytes: bytes,
        image_type_group: str,
        mmmu_meta: dict,
    ) -> None:
        sample_key = f"{source_id}::p{prompt_idx}::s0"
        if sample_key in self._done_keys:
            return

        img_path = self.images_dir / f"{sample_key}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(stage2_bytes)
        stage2_b64 = base64.b64encode(stage2_bytes).decode()

        # Build slot values and instantiate templates
        fp = call1.get("frozen_predicate", {}) or {}
        slot_assignments = plan.get("slot_assignments", {}) or {}
        slot_values: dict = dict(slot_assignments)
        answer_txt = answer_text(answer_letter, options)
        for sname, sdef in (fp.get("slots") or {}).items():
            if isinstance(sdef, dict) and "value" in sdef and sname not in slot_values:
                slot_values[sname] = answer_txt or sdef.get("value", "")

        question_template = call1.get("question_template", "") or question
        answer_template = call1.get("answer_template", "") or answer_txt
        final_question = _instantiate_template(question_template, slot_values) or question
        final_answer_text = _instantiate_template(answer_template, slot_values) or answer_txt

        # Instantiate verifier_checks
        verifier_checks_raw = fp.get("verifier_checks") or {}
        verifier_checks_inst = {k: _instantiate_template(v, slot_values) for k, v in verifier_checks_raw.items()}
        if not verifier_checks_inst.get("natural_check"):
            if image_type_group == "natural_photo":
                verifier_checks_inst["natural_check"] = (
                    "Is this a plausible real photograph (not illustration / surreal / cartoon)?"
                )
            else:
                type_noun = IMAGE_TYPE_NOUNS.get(image_type_group, "image")
                verifier_checks_inst["natural_check"] = (
                    f"Is this a legible, well-structured {type_noun} (not corrupted or garbled)?"
                )

        # Verifier
        try:
            sys_p, usr_p = _build_verifier_prompt(verifier_checks_inst)
            ver_raw = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                qwen_vl_messages(sys_p, [(stage2_b64, "image/png")], usr_p, no_think=True),
                max_tokens=256, temperature=0.0,
                max_retries=self.max_retries, log_prefix=f"verify/{sample_key}",
            )
            verifier = _parse_verifier_output(ver_raw)
        except Exception as e:
            await self._rejected_writer.write({
                "sample_key": sample_key, "source_id": source_id,
                "reject_reason": "verifier_failed",
                "error": f"{type(e).__name__}: {e!r}", "created_at": int(time.time()),
            })
            return

        if not verifier["all_pass"]:
            await self._rejected_writer.write({
                "sample_key": sample_key, "source_id": source_id,
                "reject_reason": "verifier_check_failed",
                "verifier_answers": {k: verifier[k] for k in ("a1", "a2", "a3", "a4")},
                "created_at": int(time.time()),
            })
            return

        # CoT generation
        answer_display = f"{answer_letter}) {final_answer_text}" if answer_letter else final_answer_text
        cot_raw = await self._generate_cot(stage2_b64, final_question, options, answer_display, sample_key)
        if not cot_raw:
            await self._rejected_writer.write({
                "sample_key": sample_key, "source_id": source_id,
                "reject_reason": "cot_generation_failed", "created_at": int(time.time()),
            })
            return
        cot = _clean_cot(cot_raw)
        if not cot.strip():
            await self._rejected_writer.write({
                "sample_key": sample_key, "source_id": source_id,
                "reject_reason": "cot_empty_after_cleaning", "created_at": int(time.time()),
            })
            return

        # Verify answer
        verified = await self._verify_answer(stage2_b64, final_question, options, cot, answer_display, sample_key)
        if verified is False:
            await self._rejected_writer.write({
                "sample_key": sample_key, "source_id": source_id,
                "reject_reason": "answer_incorrect", "created_at": int(time.time()),
            })
            return

        # Rewrite options
        if len(options) >= 2:
            opt_result = await self._rewrite_options(final_question, final_answer_text, options, sample_key)
            new_options, new_letter = opt_result if opt_result else (list(options), answer_letter)
        else:
            new_options, new_letter = list(options), answer_letter

        # Build training record
        question_normalized = normalize_question(final_question)
        if "<image>" not in question_normalized:
            question_normalized = "<image>" + question_normalized
        options_str = format_options_multiline(new_options)
        user_content = f"{question_normalized}\n{options_str}" if new_options else question_normalized
        assistant_answer = new_letter if new_letter else final_answer_text
        assistant_content = f"<think>\n{cot}\n</think>\n<answer>\n{assistant_answer}\n</answer>"

        record = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [str(img_path)],
            "sample_key": sample_key,
            "source_stage": "stage2_from_original",
            "source_id": source_id,
            "original_question": question,
            "original_answer_letter": answer_letter,
            "original_answer_text": answer_txt,
            "final_question": final_question,
            "final_answer_text": final_answer_text,
            "new_answer_letter": new_letter,
            "new_options": new_options,
            "plan": plan,
            "verifier_answers": {k: verifier[k] for k in ("a1", "a2", "a3", "a4")},
            "image_type_group": image_type_group,
            "mmmu_meta": mmmu_meta,
            "prompt_idx": prompt_idx,
            "created_at": int(time.time()),
        }
        await self._train_writer.write(record)
        log(f"  [{sample_key}] done | answer={new_letter}", "stage2")

    async def _process_plan(
        self,
        source_id: str,
        question: str,
        answer_letter: str,
        options: list[str],
        call1: dict,
        plan: dict,
        gpt_bytes: bytes,
        gpt_mime: str,
        image_type_group: str,
        mmmu_meta: dict,
    ) -> None:
        prompt_idx = int(plan.get("plan_id", 0))
        sample_key = f"{source_id}::p{prompt_idx}::s0"
        if sample_key in self._done_keys:
            return

        # Call 2b: prose
        try:
            sys_p, usr_p = _build_call2b_prompt(call1, plan, image_type_group)
            edit_prose = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                qwen_text_messages(sys_p, usr_p, no_think=True),
                max_tokens=2048, temperature=0.6,
                max_retries=self.max_retries, log_prefix=f"call2b/{sample_key}",
            )
        except Exception as e:
            await self._rejected_writer.write({
                "sample_key": sample_key, "source_id": source_id,
                "reject_reason": "call2b_failed",
                "error": f"{type(e).__name__}: {e!r}", "created_at": int(time.time()),
            })
            return

        # Call 3b: GPT-Image-2
        try:
            stage2_imgs = await gpt_image_edit(
                self._proxy, self.gpt_image_endpoint, self.gpt_image_api_key,
                gpt_bytes, edit_prose, n=1, image_mime=gpt_mime,
                max_retries=self.max_retries, log_prefix=f"gpt3b/{sample_key}",
            )
        except Exception as e:
            await self._rejected_writer.write({
                "sample_key": sample_key, "source_id": source_id,
                "reject_reason": "call3b_gpt_image_failed",
                "error": f"{type(e).__name__}: {e!r}", "created_at": int(time.time()),
            })
            return

        if not stage2_imgs:
            await self._rejected_writer.write({
                "sample_key": sample_key, "source_id": source_id,
                "reject_reason": "call3b_no_image", "created_at": int(time.time()),
            })
            return

        await self._process_one_plan_image(
            source_id, question, answer_letter, options,
            call1, plan, prompt_idx, stage2_imgs[0],
            image_type_group, mmmu_meta,
        )

    async def _process_one(self, row: dict) -> None:
        source_id = row.get("id", "")
        question = row.get("question", "")
        answer_letter = str(row.get("answer", ""))
        options = row.get("options", [])
        answer_txt = answer_text(answer_letter, options)
        image_type_group = row.get("image_type_group") or classify_image_type(row.get("img_type", ""))
        mmmu_meta = build_mmmu_meta(row)

        async with self.semaphore:
            try:
                # Resume: if all 5 plans done, skip
                expected_all = [f"{source_id}::p{p}::s0" for p in range(1, self.n_prompts + 1)]
                if all(k in self._done_keys for k in expected_all):
                    log(f"  [{source_id}] resume: all {self.n_prompts} plans done, skip", "stage2")
                    return

                # Load images
                all_img_bytes = get_all_image_bytes(row)
                if not all_img_bytes:
                    log(f"  [{source_id}] no images, skip", "stage2")
                    return

                qwen_images = []
                for raw in all_img_bytes:
                    hq_bytes, hq_mime = prepare_image(raw, max_side=1568, fmt="PNG")
                    qwen_images.append((base64.b64encode(hq_bytes).decode(), hq_mime))

                gpt_bytes, gpt_mime = prepare_image(all_img_bytes[0], max_side=1024, fmt="JPEG")

                # Call 1: check cache first
                call1: Optional[dict] = None
                if self._call1_cache:
                    cached = self._call1_cache.get(source_id)
                    if cached:
                        call1 = cached.get("call1")

                if call1 is None:
                    log(f"  [{source_id}] Call 1 (qwen, type={image_type_group})...", "stage2")
                    sys_p, usr_p = _build_call1_params(
                        question, answer_txt, options,
                        answer_letter=answer_letter,
                        image_type_group=image_type_group,
                        n_images=len(qwen_images),
                    )
                    call1_raw = await chat_qwen(
                        self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                        qwen_vl_messages(sys_p, qwen_images, usr_p, no_think=False),
                        max_tokens=16384, temperature=0.3,
                        max_retries=self.max_retries, log_prefix=f"call1/{source_id}",
                    )
                    try:
                        call1 = extract_json(call1_raw)
                    except Exception as e:
                        log(f"  [{source_id}] call1 JSON parse error: {e!r}", "stage2")
                        await self._rejected_writer.write({
                            "sample_key": f"{source_id}::call1_json_error",
                            "source_id": source_id,
                            "reject_reason": "call1_json_parse_error",
                            "error": f"{type(e).__name__}: {e!r}",
                            "created_at": int(time.time()),
                        })
                        return

                    ok, err = _validate_call1_schema(call1)
                    if not ok:
                        log(f"  [{source_id}] call1 schema invalid: {err}", "stage2")
                        await self._rejected_writer.write({
                            "sample_key": f"{source_id}::call1_schema_invalid",
                            "source_id": source_id,
                            "reject_reason": "call1_schema_invalid",
                            "error": err, "created_at": int(time.time()),
                        })
                        return

                    if self._call1_cache:
                        await self._call1_cache.put(source_id, {"call1": call1})

                # Gate: is_diversifiable
                div = call1.get("diversification_assessment", {})
                if not div.get("is_diversifiable", True):
                    log(f"  [{source_id}] is_diversifiable=false, skip", "stage2")
                    return

                # Selector: try cache first
                plans = self._selector_cache.get(source_id) if self._selector_cache else None
                if plans is None:
                    log(f"  [{source_id}] selector ({self.n_prompts} plans)...", "stage2")
                    sys_p, usr_p = _build_selector_prompt(call1, self.n_prompts, image_type_group)
                    try:
                        sel_text = await chat_qwen(
                            self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                            qwen_vl_messages(sys_p, qwen_images[:1], usr_p),
                            max_tokens=8192, temperature=0.5,
                            max_retries=self.max_retries, log_prefix=f"selector/{source_id}",
                        )
                    except Exception as e:
                        log(f"  [{source_id}] selector failed: {e!r}", "stage2")
                        return
                    plans = _parse_selector_output(sel_text, self.n_prompts)
                    if plans and self._selector_cache:
                        await self._selector_cache.put(source_id, plans)

                if not plans:
                    log(f"  [{source_id}] selector returned no plans, skip", "stage2")
                    return

                for i, p in enumerate(plans, start=1):
                    p["plan_id"] = i

                log(f"  [{source_id}] {len(plans)} plans, processing...", "stage2")
                tasks = [
                    self._process_plan(
                        source_id, question, answer_letter, options,
                        call1, p, gpt_bytes, gpt_mime, image_type_group, mmmu_meta,
                    )
                    for p in plans
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                log(f"  [{source_id}] UNEXPECTED ERROR: {type(e).__name__}: {e!r}", "stage2")
                log(traceback.format_exc(), "stage2")

    async def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._train_writer = JsonlWriter(self.train_jsonl)
        self._rejected_writer = JsonlWriter(self.rejected_jsonl)
        self._selector_cache = SelectorCache(self.selector_cache_path)
        self._call1_cache = _DiskCache(self.call1_cache_path, key_field="source_id")
        log(f"selector cache: {len(self._selector_cache._mem)} entries", "stage2")
        log(f"call1 cache:    {len(self._call1_cache._mem)} entries", "stage2")

        if self.resume:
            self._done_keys = (
                load_done_keys(self.train_jsonl) | load_done_keys(self.rejected_jsonl)
            )
            log(f"resume: {len(self._done_keys)} sample_keys already done", "stage2")

        log(f"loading samples from {self.manifest_path}...", "stage2")
        rows = load_selected_samples(self.manifest_path)
        log(f"loaded {len(rows)} MMMU samples", "stage2")
        if self.limit > 0:
            rows = rows[:self.limit]
            log(f"limit to {self.limit}", "stage2")

        inner, proxy = make_clients(api_timeout=self.api_timeout)
        self._inner, self._proxy = inner, proxy
        try:
            await asyncio.gather(*[self._process_one(r) for r in rows], return_exceptions=True)
        finally:
            await inner.aclose()
            await proxy.aclose()

        total_train = sum(1 for _ in jsonl_iter(str(self.train_jsonl)))
        total_rejected = sum(1 for _ in jsonl_iter(str(self.rejected_jsonl)))
        log(f"stage2 done. train={total_train} | rejected={total_rejected}", "stage2")


# ── CLI ─────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="MMMU Stage 2 from original images (emma_v3): "
                    "answer-preserving diversification, 1:5 ratio"
    )
    parser.add_argument("--manifest", default=str(MANIFEST_JSONL))
    parser.add_argument("--output_dir", default=str(OUTPUT_EMMA_V3_STAGE2_ONLY))
    parser.add_argument("--qwen_api_base", default=QWEN_API_BASE)
    parser.add_argument("--qwen_api_key", default=QWEN_API_KEY)
    parser.add_argument("--qwen_model", default=QWEN_MODEL)
    parser.add_argument("--gpt_image_endpoint", default=GPT_IMAGE_ENDPOINT)
    parser.add_argument("--gpt_image_api_key", default=GPT_IMAGE_API_KEY)
    parser.add_argument("--max_concurrent", type=int, default=64)
    parser.add_argument("--n_prompts_per_source", type=int, default=5,
                        help="Diversification plans per original image (default 5 for 1:5 ratio)")
    parser.add_argument("--max_retries", type=int, default=20)
    parser.add_argument("--api_timeout", type=float, default=float(os.environ.get("API_TIMEOUT", "1200")))
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "0")))
    args = parser.parse_args()

    pipe = Stage2FromOriginalPipeline(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        qwen_api_base=args.qwen_api_base,
        qwen_api_key=args.qwen_api_key,
        qwen_model=args.qwen_model,
        gpt_image_endpoint=args.gpt_image_endpoint,
        gpt_image_api_key=args.gpt_image_api_key,
        max_concurrent=args.max_concurrent,
        n_prompts_per_source=args.n_prompts_per_source,
        max_retries=args.max_retries,
        api_timeout=args.api_timeout,
        resume=not args.no_resume,
        limit=args.limit,
    )
    await pipe.run()


if __name__ == "__main__":
    asyncio.run(main())
