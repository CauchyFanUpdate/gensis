#!/usr/bin/env python3
"""
MMMU Pipeline — Stage 1 (emma_v3): Surgical Answer-Changing Edit
1 original image → 5 generated images (1:5 ratio, n_edits_per_source=5)

Per source question:
  Call 1 [Qwen VL]    -> JSON: scene_description + test_point_analysis
                                + feasible_edits + axis_table + frozen_predicate
                                (with is_editable gate)
  pick 5 diverse edits
  per edit (parallel):
    Call 2a [Qwen text]   -> surgical edit prose (single paragraph for GPT-Image-2)
    Call 3a [GPT-Image-2] -> 1 edited image (from first/primary image only)
    Modifier [2x Qwen VL] -> dual cross-validation at temp 0.0 and 0.3
    Inline option fixup   -> ensure actual_new_answer appears in options

Output:
  output_emma_v3/stage1_samples.jsonl
  output_emma_v3/stage1_images/{sample_key}.png
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
    get_all_image_bytes,
    get_image_bytes,
    gpt_image_edit,
    JsonlWriter,
    jsonl_iter,
    load_done_keys,
    load_selected_samples,
    log,
    make_clients,
    prepare_image,
    qwen_text_messages,
    qwen_vl_messages,
    QWEN_API_BASE,
    QWEN_API_KEY,
    QWEN_MODEL,
    GPT_IMAGE_ENDPOINT,
    GPT_IMAGE_API_KEY,
    OUTPUT_DIR,
    LETTERS,
    MANIFEST_JSONL,
)

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

OUTPUT_EMMA_V3 = Path(__file__).resolve().parent / "output_emma_v3"

# ── Image-type-specific addenda for Call 1 ─────────────────────────────────

_IMAGE_TYPE_ADDENDA: dict[str, str] = {
    "table": (
        "\n## IMAGE TYPE OVERRIDE: TABLE / SPREADSHEET\n\n"
        "This image is a rendered table, spreadsheet, or structured data display — NOT a natural photograph.\n\n"
        "### Edit strategy for tables\n"
        "- The primary edit operation is `modify_property` on cell values: change a number, swap a text entry,\n"
        '  alter a percentage, or update a label. Params: {"object": "cell at row X, column Y",\n'
        '  "property": "value", "from": "old_value", "to": "new_value"}.\n'
        "- `add` means adding a row or column with new data. `remove` means deleting a row or column.\n"
        "- Do NOT use `substitute_category` or `context_change` (these assume physical objects).\n"
        "- `scene_description` should describe the table structure: number of rows/columns, headers, what data it shows.\n"
        "- `structural_pattern` will often be `table_lookup` or `dual_value_disambiguation`.\n\n"
        "### Axis table guidance for tables\n"
        "ALL physical-world axes MUST be set to `frozen` with freeze_constraint = "
        '"not applicable to rendered table".\n\n'
        "Only these styling axes should be `free`:\n"
        '  - subject_color, subject_material, subject_specific_appearance\n\n'
        "### Verifier checks for tables\n"
        'verifier_checks.natural_check should be: "Is this a legible, well-formatted data table (not corrupted or garbled)?"\n'
    ),
    "chart": (
        "\n## IMAGE TYPE OVERRIDE: CHART / GRAPH\n\n"
        "This image is a chart, graph, plot, or data visualization — NOT a natural photograph.\n\n"
        "### Edit strategy for charts\n"
        "- `modify_property` on data point values, axis labels, legend text, or title.\n"
        "- `add`/`remove` can mean adding/removing data series, bars, pie segments, or data points.\n"
        "- `structural_pattern` will often be `chart_value_reading` or `dual_value_disambiguation`.\n\n"
        "### Axis table guidance for charts\n"
        'Freeze ALL physical-world axes with freeze_constraint = "not applicable to rendered chart".\n\n'
        "Free styling axes: subject_color, subject_material, subject_specific_appearance.\n\n"
        "### Verifier checks for charts\n"
        'verifier_checks.natural_check should be: "Is this a legible, well-formatted chart with readable axes and data?"\n'
    ),
    "diagram": (
        "\n## IMAGE TYPE OVERRIDE: DIAGRAM / SCHEMATIC / BLUEPRINT\n\n"
        "This image is a diagram, flowchart, circuit, geometric figure, or technical drawing.\n\n"
        "### Edit strategy for diagrams\n"
        "- `modify_property` on labels, dimensions, node values, edge weights, or annotation text.\n"
        "- Be cautious: many diagrams have tightly-coupled elements. Set is_editable=false if changing\n"
        "  one label would cascade through the entire structure unpredictably.\n"
        "- `structural_pattern` will often be `diagram_label_reading` or `text_in_context`.\n\n"
        "### Axis table guidance for diagrams\n"
        'Freeze ALL physical-world axes with freeze_constraint = "not applicable to rendered diagram".\n\n'
        "Free styling axes: subject_color, subject_material, subject_specific_appearance.\n\n"
        "### Verifier checks for diagrams\n"
        'verifier_checks.natural_check should be: "Is this a legible, well-structured diagram with clear labels?"\n'
    ),
    "medical": (
        "\n## IMAGE TYPE OVERRIDE: MEDICAL IMAGE\n\n"
        "This image is a clinical or scientific medical image (X-ray, CT, MRI, microscopy, pathology).\n\n"
        "### Edit strategy for medical images\n"
        "- `modify_property` can change visible findings (lesion size, staining intensity, measurement values).\n"
        "- Be cautious: if verifying the edit requires deep specialist knowledge, prefer is_editable=false.\n"
        "- `structural_pattern` will often be `medical_finding_identification`.\n\n"
        "### Axis table guidance for medical images\n"
        "Freeze most axes. Only allow subject_specific_appearance and subject_color (free).\n"
        'Other axes frozen with "not applicable to clinical imaging".\n\n'
        "### Verifier checks for medical images\n"
        'verifier_checks.natural_check should be: "Is this a plausible medical image with adequate diagnostic quality?"\n'
    ),
    "sheet_music": (
        "\n## IMAGE TYPE OVERRIDE: SHEET MUSIC / MUSICAL NOTATION\n\n"
        "This image contains musical notation on a staff.\n\n"
        "### Edit strategy for sheet music\n"
        "- `modify_property` on note positions (pitch), accidentals, dynamic markings, clef type, or time signature.\n"
        "- `structural_pattern` will often be `music_notation_reading`.\n\n"
        "### Axis table guidance for sheet music\n"
        "Freeze ALL physical-world axes. Free styling axes: subject_specific_appearance, subject_color, subject_material.\n\n"
        "### Verifier checks for sheet music\n"
        'verifier_checks.natural_check should be: "Is this legible sheet music with correct staff line structure?"\n'
    ),
}


# ── Call 1: Qwen VL analysis JSON ──────────────────────────────────────────

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
        f"\n**Note: This sample has {n_images} images.** The question may reference multiple images. "
        "All images are shown to you. However, surgical edits (Stage 1) will only be applied to "
        "the FIRST/PRIMARY image — design all feasible_edits to operate on image 1 only.\n"
        if n_images > 1 else ""
    )

    system = (
        "You are an analytical assistant for an image-editing data generation pipeline. "
        "Given an image (or images) and a visual multiple-choice question with its ground-truth answer, "
        "produce a structured analysis JSON that downstream code uses to design controlled edits.\n\n"
        + multi_img_note
        + "\n## Dataset context: MMMU\n\n"
        "This dataset contains standard MMMU benchmark questions (dev + validation splits). "
        "All questions are multiple-choice with 2-6 options, covering STEM subjects including "
        "mathematics, physics, chemistry, biology, medicine, engineering, accounting, and more. "
        "Questions may reference tables, charts, diagrams, medical images, or natural photographs. "
        "Some samples have multiple images.\n\n"
        "## Pipeline context (two stages, both consume your analysis)\n\n"
        "Stage 1 — surgical answer-changing edit. The pipeline applies one edit that changes the "
        "answer in a controlled way. The edited image should look almost identical to the original "
        "except for the answer-relevant region. Your `feasible_edits` list drives this stage.\n\n"
        "Stage 2 — cosmetic diversification on top of Stage 1. The pipeline pushes ALL answer-irrelevant "
        "visual axes to aggressive new values while keeping the test point intact. "
        "Your `frozen_predicate`, `question_template`, `answer_template`, and `axis_table` drive this stage.\n\n"
        "## Core framework: 5 ops × 4 answer_effect_types\n\n"
        "Five edit operations:\n"
        "  - remove(obj): delete one specific object\n"
        "  - add(obj_like, location): add one same-category object at a feasible spot\n"
        "  - modify_property(obj, property, new_value): change color / size / shape / text content\n"
        "  - substitute_category(from, to): replace all objects of category X with category Y\n"
        "  - context_change(attribute, value): change background, lighting, viewpoint (Stage 2 only)\n\n"
        "Four answer_effect_types:\n"
        "  - delta: numeric ± (e.g., count 8 → 6)\n"
        "  - literal_replace: answer becomes the new literal value (e.g., color red → blue)\n"
        "  - flip_boolean: yes ↔ no\n"
        "  - categorical_shift: answer shifts among finite options\n\n"
        "**CRITICAL — every feasible_edit MUST change the answer.** "
        "If no edit can change the answer, set `is_editable=false`.\n\n"
        "## Test point analysis\n\n"
        "Identify the structural pattern:\n"
        "  - dual_value_disambiguation / real_vs_depicted / partial_visibility / category_filtering\n"
        "  - lookalike_confusion / spatial_ordering / composition_state / text_in_context\n"
        "  - table_lookup / chart_value_reading / diagram_label_reading\n"
        "  - medical_finding_identification / music_notation_reading / other\n\n"
        "## Feasibility self-check (all 4 must pass for each edit)\n"
        "  1. Physically realistic\n"
        "  2. Preserves test_point structure\n"
        "  3. Preserves disambiguators\n"
        "  4. Mechanically derivable new answer\n\n"
        "## When to set is_editable=false\n"
        "  - Puzzle grids / mazes where every cell determines the answer\n"
        "  - Tightly-coupled diagrams where any change cascades unpredictably\n"
        "  - Count > 15 objects (small ± not visibly clean)\n"
        "  - Deep expert knowledge required to verify the edit\n\n"
        "## Stage 2 schema — frozen_predicate + slots + axis_table\n\n"
        "### A. frozen_predicate\n"
        "  - answer_function, answer_value, must_preserve (1-3 bullets)\n"
        "  - verifier_checks: class_check, relation_check, readability_check, natural_check\n"
        "  - slots: dict with compatible_classes (≥5 unless locked), slot_replaceability\n\n"
        "### B. question_template — slot-templated, no answer value\n\n"
        "### C. answer_template — usually '<N>' or literal\n\n"
        "### D. axis_table — 19 universal visual axes:\n"
        f"{axes_block}\n\n"
        "### E. test_point → frozen-axes prior:\n"
        f"{test_point_priors_block}\n\n"
        "### F. readability check templates:\n"
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
        f"Image type hint: {image_type_group}\n"
        + (f"Number of images: {n_images} (edit only image 1)\n" if n_images > 1 else "")
        + "\nAnalyze this image and produce JSON with EXACTLY these fields:\n\n"
        "{\n"
        f'  "image_type": "{image_type_group}",\n'
        '  "scene_description": "<1-2 sentence natural caption explicitly describing the test_point structure>",\n'
        '  "question_analysis": {\n'
        '    "answer_determinant": "<what visual fact directly produces the answer>",\n'
        '    "answer_source_region": "<where in the image to look>",\n'
        '    "current_answer_echo": "<echo the source answer verbatim>"\n'
        "  },\n"
        '  "test_point_analysis": {\n'
        '    "core_challenge": "<one sentence: why this question is non-trivial>",\n'
        f'    "structural_pattern": "<one of: {" | ".join(STRUCTURAL_PATTERNS)}>",\n'
        '    "must_be_described_in_edit": ["<element 1>", "..."],\n'
        '    "edit_must_operate_within": "<one sentence constraint>"\n'
        "  },\n"
        '  "disambiguators_to_preserve": [\n'
        '    {"feature": "<concrete visual feature>", "scope": "<which objects>"}\n'
        "  ],\n"
        '  "feasibility_assessment": {"is_editable": true, "skip_reason": ""},\n'
        '  "feasible_edits": [\n'
        "    {\n"
        '      "edit_id": "e1",\n'
        f'      "op": "<one of: {" | ".join(OPS)}>",\n'
        '      "params": {},\n'
        '      "rationale_natural": "<plain English description>",\n'
        '      "answer_effect": {\n'
        f'        "type": "<one of: {" | ".join(EFFECT_TYPES)}>",\n'
        '        "new_answer": "<mechanically derived value>",\n'
        '        "expected_visual_delta": "<what should look different>"\n'
        "      },\n"
        '      "lock_list": ["<element that must not change>"]\n'
        "    }\n"
        "  ],\n"
        '  "frozen_predicate": {\n'
        '    "answer_function": "<predicate string>",\n'
        '    "answer_value": "<actual answer as string>",\n'
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
        '  "question_template": "<question with slot placeholders>",\n'
        '  "answer_template": "<usually <N> or literal>",\n'
        '  "axis_table": [\n'
        f"{axis_schema_lines}\n"
        "  ]\n"
        "}\n\n"
        "Constraints:\n"
        "  - feasible_edits: every entry MUST change the answer; aim for 5-7 across ≥2 effect_types\n"
        "    (we need at least 5 distinct edits for the 1:5 generation ratio).\n"
        "  - axis_table MUST contain ALL 19 axes.\n"
        "  - For free axes, aggressive_value MUST list ≥3 concrete options separated by ' / '.\n"
        "  - If is_editable=false, frozen_predicate/question_template/answer_template/axis_table can be empty.\n"
        "  - Output ONLY the JSON object. No markdown, no commentary."
    )
    return system, user_text


# ── Call 1 schema validator ──────────────────────────────────────────────────

def _validate_call1_schema(call1: dict) -> tuple[bool, str]:
    feas = call1.get("feasibility_assessment", {})
    if not feas.get("is_editable", False):
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


# ── Pick diverse feasible_edits ──────────────────────────────────────────────

def _pick_diverse_edits(
    feasible_edits: list[dict], n: int = 5, seed: int = 0
) -> list[dict]:
    if not feasible_edits:
        return []
    filtered = [
        e for e in feasible_edits
        if e.get("answer_effect", {}).get("type") in EFFECT_TYPES
    ]
    if not filtered:
        return []
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for e in filtered:
        et = e.get("answer_effect", {}).get("type", "?")
        buckets.setdefault(et, []).append(e)
    for vs in buckets.values():
        rng.shuffle(vs)
    picked: list[dict] = []
    bucket_keys = list(buckets.keys())
    rng.shuffle(bucket_keys)
    while len(picked) < n and any(buckets.values()):
        for k in bucket_keys:
            if len(picked) >= n:
                break
            if buckets[k]:
                picked.append(buckets[k].pop(0))
    return picked[:n]


# ── Call 2a: Qwen text — surgical edit prose ─────────────────────────────────

def _build_call2a_prompt(
    call1: dict, edit: dict, image_type_group: str = "natural_photo"
) -> tuple[str, str]:
    type_noun = IMAGE_TYPE_NOUNS.get(image_type_group, "image")
    system = (
        "You write image editing instructions for an AI image editor (GPT-Image-2). "
        "Your output is a SINGLE FLOWING PARAGRAPH of natural English prose, 150-250 words.\n\n"
        "Strict format rules:\n"
        "  - NO bullet points, NO labels (no 'SCENE:', 'EDIT:', 'KEEP:', etc.)\n"
        "  - NO section headers, NO line breaks within the paragraph\n"
        "  - Write as one continuous, naturally-flowing description\n"
        "  - Use subordinate clauses: 'while keeping...', 'without altering...', 'so that...'\n\n"
        "The paragraph must (in flowing order):\n"
        f"  1. Describe the {type_noun} layout, EXPLICITLY including the test-point structure\n"
        "  2. Specify the single edit operation with precise spatial/visual details\n"
        "  3. Weave in what must remain unchanged\n"
        "  4. Close with an integrity statement about the difficulty structure being preserved\n\n"
        "Stay concrete and visual. Do not invent details not in the inputs."
    )

    test_point = call1.get("test_point_analysis", {})
    disambs = call1.get("disambiguators_to_preserve", [])
    must_describe = test_point.get("must_be_described_in_edit", [])
    integrity = test_point.get("edit_must_operate_within", "")

    user_text = (
        "Compose the image editing instruction.\n\n"
        f"SCENE CONTEXT:\n{call1.get('scene_description', '')}\n\n"
        f"TEST POINT — must remain visible after the edit:\n"
        f"  Core challenge: {test_point.get('core_challenge', '')}\n"
        f"  Structural elements to preserve:\n"
        + "\n".join(f"    - {x}" for x in must_describe)
        + "\n\n"
        f"EDIT OPERATION:\n"
        f"  Type: {edit.get('op', '')}\n"
        f"  Specifics: {json.dumps(edit.get('params', {}), ensure_ascii=False)}\n"
        f"  Expected visual outcome: {edit.get('answer_effect', {}).get('expected_visual_delta', '')}\n\n"
        f"MUST NOT CHANGE:\n"
        + "\n".join(f"    - {x}" for x in edit.get("lock_list", []))
        + "\n"
        + "\n".join(f"    - {d.get('feature', '')}" for d in disambs)
        + "\n\n"
        f"INTEGRITY CONSTRAINT:\n  {integrity}\n\n"
        "Output ONLY the paragraph, no other text."
    )
    return system, user_text


# ── Modifier: dual Qwen VL cross-validation ──────────────────────────────────

def _build_modifier_prompt(
    edit: dict,
    new_question: str,
    options: list[str],
    image_type_group: str = "natural_photo",
) -> tuple[str, str]:
    type_noun = IMAGE_TYPE_NOUNS.get(image_type_group, "image")
    system = "You are reading the actual answer from an edited image."
    designed = edit.get("answer_effect", {}).get("new_answer", "")
    rationale = edit.get("rationale_natural", "")
    opts_line = f"Available options: {format_options(options)}\n" if options else ""

    user_text = (
        f"Background: the image shown was produced by editing an original {type_noun}.\n"
        f"The editing instruction asked to: {rationale}\n"
        f"If the edit succeeded perfectly, the answer would be: {designed}\n"
        "However, AI image editing may not be perfect — the actual outcome may differ slightly.\n\n"
        "Your task: look at THIS image and determine the ACTUAL answer to the following question.\n\n"
        f"Question: {new_question}\n"
        f"{opts_line}\n"
        "Output strictly in this format:\n"
        "ACTUAL_ANSWER: <the value you see, as a single short phrase or number>\n"
        "CONFIDENCE: high | medium | low\n"
        "REASONING: <one sentence describing what you see>\n"
    )
    return system, user_text


def _parse_modifier_output(text: str) -> dict:
    out: dict = {"actual_answer": "", "confidence": "low", "reasoning": "", "raw": text}

    def _grab(field: str) -> str:
        pat = rf"{field}\s*(.+?)(?=\n[A-Z][A-Z_]*:|\Z)"
        m = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    out["actual_answer"] = _grab("ACTUAL_ANSWER:").rstrip(".")
    m = re.search(r"CONFIDENCE:\s*(high|medium|low)", text, re.IGNORECASE)
    if m:
        out["confidence"] = m.group(1).strip().lower()
    out["reasoning"] = _grab("REASONING:")
    return out


def _normalize_answer(a: str) -> str:
    a = (a or "").strip().lower()
    a = re.sub(r"[^\w\s./-]", " ", a)
    a = re.sub(r"\s+", " ", a).strip()
    if re.fullmatch(r"\d+\.0+", a):
        a = a.split(".")[0]
    if a.endswith("s") and len(a) > 1 and not a.endswith("ss"):
        a_singular = a[:-1]
        if not a_singular.isdigit():
            a = a_singular
    return a


def _confidences_ok(c1: str, c2: str) -> bool:
    rank = {"high": 2, "medium": 1, "low": 0}
    return max(rank.get(c1, 0), rank.get(c2, 0)) >= 1


def _rephrase_new_question(orig_q: str, edit: dict) -> str:
    op = edit.get("op", "")
    params = edit.get("params", {})
    if op == "substitute_category":
        from_cat = params.get("from_category", "")
        to_cat = params.get("to_category", "")
        if from_cat and to_cat:
            pat = re.compile(re.escape(from_cat), re.IGNORECASE)
            return pat.sub(to_cat, orig_q)
    return orig_q


# ── Stage 1 Pipeline (emma_v3: 1:5 ratio) ────────────────────────────────────

class Stage1PipelineEmmaV3:
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
        n_edits_per_source: int = 5,
    ):
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "stage1_images"
        self.output_jsonl = self.output_dir / "stage1_samples.jsonl"
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
        self.n_edits = n_edits_per_source

        self._writer: Optional[JsonlWriter] = None
        self._done_keys: set[str] = set()
        self._inner = None
        self._proxy = None

    async def _process_edit(
        self,
        source_id: str,
        question_orig: str,
        answer_orig: str,
        answer_letter_orig: str,
        options: list[str],
        call1: dict,
        edit: dict,
        edit_idx: int,
        gpt_bytes: bytes,
        gpt_mime: str,
        mmmu_meta: dict,
        image_type_group: str = "natural_photo",
    ) -> None:
        sample_key = f"{source_id}::e{edit_idx}"
        if sample_key in self._done_keys:
            log(f"  [{sample_key}] resume: already done, skip", "stage1")
            return

        # Call 2a: Qwen text → edit prose
        try:
            sys_p, usr_p = _build_call2a_prompt(call1, edit, image_type_group)
            edit_prose = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                qwen_text_messages(sys_p, usr_p, no_think=True),
                max_tokens=2048, temperature=0.6,
                max_retries=self.max_retries,
                log_prefix=f"qwen2a/{sample_key}",
            )
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id, "stage": 1,
                "status": "failed", "reject_reason": "call2a_qwen_failed",
                "original_question": question_orig, "original_answer": answer_orig,
                "edit": edit, "call1_full": call1, "mmmu_meta": mmmu_meta,
                "error": f"{type(e).__name__}: {e!r}", "created_at": int(time.time()),
            })
            return

        new_question = _rephrase_new_question(question_orig, edit)

        # Call 3a: GPT-Image-2 edit
        try:
            edited_imgs = await gpt_image_edit(
                self._proxy, self.gpt_image_endpoint, self.gpt_image_api_key,
                gpt_bytes, edit_prose, n=1, image_mime=gpt_mime,
                size="1024x1024", quality="high",
                max_retries=self.max_retries,
                log_prefix=f"gpt3a/{sample_key}",
            )
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id, "stage": 1,
                "status": "failed", "reject_reason": "call3a_gpt_image_failed",
                "original_question": question_orig, "original_answer": answer_orig,
                "edit": edit, "edit_prose": edit_prose, "call1_full": call1,
                "mmmu_meta": mmmu_meta,
                "error": f"{type(e).__name__}: {e!r}", "created_at": int(time.time()),
            })
            return

        if not edited_imgs:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id, "stage": 1,
                "status": "failed", "reject_reason": "call3a_no_image",
                "original_question": question_orig, "original_answer": answer_orig,
                "edit": edit, "edit_prose": edit_prose, "call1_full": call1,
                "mmmu_meta": mmmu_meta, "created_at": int(time.time()),
            })
            return

        edited_bytes = edited_imgs[0]
        img_path = self.images_dir / f"{sample_key}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(edited_bytes)

        # Modifier: 2x Qwen VL at temp 0.0 and 0.3
        edited_b64 = base64.b64encode(edited_bytes).decode()
        mod_sys, mod_usr = _build_modifier_prompt(edit, new_question, options, image_type_group)

        async def _qwen_modify(temperature: float = 0.0) -> dict:
            text = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                qwen_vl_messages(mod_sys, [(edited_b64, "image/png")], mod_usr, no_think=True),
                max_tokens=512, temperature=temperature,
                max_retries=self.max_retries,
                log_prefix=f"mod_qwen_t{temperature}/{sample_key}",
            )
            return _parse_modifier_output(text)

        try:
            mod_qwen_1, mod_qwen_2 = await asyncio.gather(
                _qwen_modify(0.0), _qwen_modify(0.3)
            )
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id, "stage": 1,
                "status": "failed", "reject_reason": "modifier_failed",
                "original_question": question_orig, "original_answer": answer_orig,
                "edit": edit, "stage1_image_path": str(img_path),
                "edit_prose": edit_prose, "call1_full": call1, "mmmu_meta": mmmu_meta,
                "error": f"{type(e).__name__}: {e!r}", "created_at": int(time.time()),
            })
            return

        norm1 = _normalize_answer(mod_qwen_1["actual_answer"])
        norm2 = _normalize_answer(mod_qwen_2["actual_answer"])
        agree = norm1 == norm2 and norm1 != ""
        conf_ok = _confidences_ok(mod_qwen_1["confidence"], mod_qwen_2["confidence"])

        designed_new_answer = edit.get("answer_effect", {}).get("new_answer", "")

        if not agree or not conf_ok:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id, "stage": 1,
                "status": "rejected", "quality_pass": False,
                "reject_reason": "modifier_disagree" if not agree else "modifier_low_confidence",
                "original_question": question_orig, "original_answer": answer_orig,
                "edit": edit, "stage1_image_path": str(img_path),
                "edit_prose": edit_prose, "new_question": new_question,
                "designed_new_answer": designed_new_answer,
                "actual_new_answer": mod_qwen_1["actual_answer"],
                "modifier_qwen_1": mod_qwen_1, "modifier_qwen_2": mod_qwen_2,
                "call1_full": call1, "mmmu_meta": mmmu_meta,
                "image_type_group": image_type_group, "created_at": int(time.time()),
            })
            return

        actual_new_answer = mod_qwen_1["actual_answer"]

        # Inline option fixup
        rng = random.Random(hash(sample_key) & 0xFFFFFFFF)
        fixed_options, new_answer_letter = fix_options_inline(
            actual_new_answer, list(options), answer_letter_orig, rng
        )
        options_fixed = fixed_options != list(options)

        final_meta = dict(mmmu_meta)
        if options_fixed:
            final_meta["options"] = fixed_options
            final_meta["options_before_fix"] = list(options)
        final_meta["answer_letter"] = new_answer_letter

        record = {
            "sample_key": sample_key,
            "source_id": source_id,
            "stage": 1,
            "quality_pass": True,
            "original_question": question_orig,
            "original_answer": answer_orig,
            "edit": edit,
            "edit_prose": edit_prose,
            "new_question": new_question,
            "designed_new_answer": designed_new_answer,
            "actual_new_answer": actual_new_answer,
            "modifier_qwen_1": mod_qwen_1,
            "modifier_qwen_2": mod_qwen_2,
            "stage1_image_path": str(img_path),
            "call1_full": call1,
            "mmmu_meta": final_meta,
            "image_type_group": image_type_group,
            "options_fixed": options_fixed,
            "created_at": int(time.time()),
        }
        await self._writer.write(record)
        log(
            f"  [{sample_key}] stage1 done | actual={actual_new_answer} | "
            f"letter={new_answer_letter} | options_fixed={options_fixed}",
            "stage1",
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
                all_img_bytes = get_all_image_bytes(row)
                if not all_img_bytes:
                    log(f"  [{source_id}] no images, skip", "stage1")
                    return

                qwen_images = []
                for raw in all_img_bytes:
                    hq_bytes, hq_mime = prepare_image(raw, max_side=1568, fmt="PNG")
                    qwen_images.append((base64.b64encode(hq_bytes).decode(), hq_mime))

                gpt_bytes, gpt_mime = prepare_image(all_img_bytes[0], max_side=1024, fmt="JPEG")

                n_edits = self.n_edits

                if f"{source_id}::skipped" in self._done_keys:
                    log(f"  [{source_id}] resume: previously skipped", "stage1")
                    return
                if f"{source_id}::no_edits" in self._done_keys:
                    log(f"  [{source_id}] resume: previously no_edits", "stage1")
                    return
                if all(f"{source_id}::e{i+1}" in self._done_keys for i in range(n_edits)):
                    log(f"  [{source_id}] resume: all {n_edits} edits done", "stage1")
                    return

                log(
                    f"  [{source_id}] Call 1 (qwen, type={image_type_group}, "
                    f"opts={len(options)}, imgs={len(qwen_images)})...",
                    "stage1",
                )
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
                    max_retries=self.max_retries,
                    log_prefix=f"call1/{source_id}",
                )

                try:
                    call1 = extract_json(call1_raw)
                except Exception as e:
                    await self._writer.write({
                        "sample_key": f"{source_id}::call1_json_error",
                        "source_id": source_id, "stage": 1, "status": "failed",
                        "reject_reason": "call1_json_parse_error",
                        "original_question": question, "original_answer": answer_txt,
                        "error": f"{type(e).__name__}: {e!r}",
                        "raw_call1": call1_raw[:2000], "mmmu_meta": mmmu_meta,
                        "created_at": int(time.time()),
                    })
                    return

                ok, err = _validate_call1_schema(call1)
                if not ok:
                    await self._writer.write({
                        "sample_key": f"{source_id}::call1_schema_invalid",
                        "source_id": source_id, "stage": 1, "status": "failed",
                        "reject_reason": "call1_schema_invalid",
                        "original_question": question, "original_answer": answer_txt,
                        "error": err, "call1_full": call1, "mmmu_meta": mmmu_meta,
                        "created_at": int(time.time()),
                    })
                    log(f"  [{source_id}] schema invalid: {err}", "stage1")
                    return

                feas = call1.get("feasibility_assessment", {})
                if not feas.get("is_editable", False):
                    await self._writer.write({
                        "sample_key": f"{source_id}::skipped",
                        "source_id": source_id, "stage": 1, "status": "skipped",
                        "reject_reason": "is_editable_false",
                        "skip_reason": feas.get("skip_reason", ""),
                        "original_question": question, "original_answer": answer_txt,
                        "call1_full": call1, "mmmu_meta": mmmu_meta,
                        "created_at": int(time.time()),
                    })
                    log(f"  [{source_id}] is_editable=false, skip", "stage1")
                    return

                edits = _pick_diverse_edits(
                    call1.get("feasible_edits", []),
                    n=n_edits,
                    seed=hash(source_id) & 0xFFFF,
                )
                if not edits:
                    await self._writer.write({
                        "sample_key": f"{source_id}::no_edits",
                        "source_id": source_id, "stage": 1, "status": "skipped",
                        "reject_reason": "no_feasible_edits",
                        "original_question": question, "original_answer": answer_txt,
                        "call1_full": call1, "mmmu_meta": mmmu_meta,
                        "created_at": int(time.time()),
                    })
                    log(f"  [{source_id}] no feasible edits", "stage1")
                    return

                log(f"  [{source_id}] picked {len(edits)} edits (target 5), processing...", "stage1")
                tasks = [
                    self._process_edit(
                        source_id, question, answer_txt, answer_letter, options,
                        call1, e, idx + 1,
                        gpt_bytes, gpt_mime, mmmu_meta, image_type_group,
                    )
                    for idx, e in enumerate(edits)
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                log(f"  [{source_id}] UNEXPECTED ERROR: {type(e).__name__}: {e!r}", "stage1")
                log(traceback.format_exc(), "stage1")
                await self._writer.write({
                    "sample_key": f"{source_id}::unexpected_error",
                    "source_id": source_id, "stage": 1, "status": "failed",
                    "reject_reason": "unexpected_error",
                    "original_question": question, "original_answer": answer_txt,
                    "error": f"{type(e).__name__}: {e!r}", "mmmu_meta": mmmu_meta,
                    "created_at": int(time.time()),
                })

    async def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._writer = JsonlWriter(self.output_jsonl)

        if self.resume:
            self._done_keys = load_done_keys(self.output_jsonl)
            log(f"resume: {len(self._done_keys)} sample_keys already done", "stage1")

        log(f"loading samples from {self.manifest_path}...", "stage1")
        rows = load_selected_samples(self.manifest_path)
        log(f"loaded {len(rows)} MMMU samples", "stage1")
        if self.limit > 0:
            rows = rows[:self.limit]
            log(f"limit to {self.limit}", "stage1")

        inner, proxy = make_clients(api_timeout=self.api_timeout)
        self._inner, self._proxy = inner, proxy
        try:
            await asyncio.gather(*[self._process_one(r) for r in rows], return_exceptions=True)
        finally:
            await inner.aclose()
            await proxy.aclose()

        total = passed = rejected = failed = skipped = 0
        for rec in jsonl_iter(str(self.output_jsonl)):
            total += 1
            if rec.get("quality_pass"):
                passed += 1
            elif rec.get("status") == "rejected":
                rejected += 1
            elif rec.get("status") == "failed":
                failed += 1
            elif rec.get("status") == "skipped":
                skipped += 1
        log(
            f"stage1 done. total={total} | passed={passed} | rejected={rejected} "
            f"| failed={failed} | skipped={skipped}",
            "stage1",
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description="MMMU Stage 1 (emma_v3): 1:5 surgical edit — 1 original → 5 generated images"
    )
    parser.add_argument("--manifest", default=str(MANIFEST_JSONL),
                        help="Path to MMMU manifest.jsonl")
    parser.add_argument("--output_dir", default=str(OUTPUT_EMMA_V3))
    parser.add_argument("--qwen_api_base", default=QWEN_API_BASE)
    parser.add_argument("--qwen_api_key", default=QWEN_API_KEY)
    parser.add_argument("--qwen_model", default=QWEN_MODEL)
    parser.add_argument("--gpt_image_endpoint", default=GPT_IMAGE_ENDPOINT)
    parser.add_argument("--gpt_image_api_key", default=GPT_IMAGE_API_KEY)
    parser.add_argument("--max_concurrent", type=int, default=64)
    parser.add_argument("--n_edits_per_source", type=int, default=5,
                        help="Number of edits per source image (default 5 for 1:5 ratio)")
    parser.add_argument("--max_retries", type=int, default=20)
    parser.add_argument("--api_timeout", type=float, default=float(os.environ.get("API_TIMEOUT", "1200")))
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "0")))
    args = parser.parse_args()

    pipe = Stage1PipelineEmmaV3(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        qwen_api_base=args.qwen_api_base,
        qwen_api_key=args.qwen_api_key,
        qwen_model=args.qwen_model,
        gpt_image_endpoint=args.gpt_image_endpoint,
        gpt_image_api_key=args.gpt_image_api_key,
        max_concurrent=args.max_concurrent,
        n_edits_per_source=args.n_edits_per_source,
        max_retries=args.max_retries,
        api_timeout=args.api_timeout,
        resume=not args.no_resume,
        limit=args.limit,
    )
    await pipe.run()


if __name__ == "__main__":
    asyncio.run(main())
