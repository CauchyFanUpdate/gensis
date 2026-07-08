#!/usr/bin/env python3
"""
方案一 — Stage 1: 每题生成 5 个答案变化编辑变体（1:5 比例）

与 edit_mmmu/generate_samples_v5_stage1.py 的核心差异：
  - Call 1 要求输出 3-5 个 feasible_edits（1:5 比例，固定 n_target=5）
  - 每题固定生成 n_target = 5 个目标变体数量
  - 对每个 edit 执行 ceil(n_target / n_edits) 次 GPT-Image-2 调用（不同种子）
  - sample_key 格式：{id}::e{N}::s{K}（e=edit index, s=seed index）

需要后续步骤：Stage 1.5（MC 选项修正）→ Stage 3（CoT 生成 + 训练格式）
"""

import argparse
import asyncio
import base64
import json
import math
import os
import random
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Optional

# ── 引入 edit_mmmu 公共库 ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "edit_mmmu"))

from _v5_common import (
    AXIS_TIER_DEFAULTS,
    IMAGE_TYPE_NOUNS,
    NATURAL_CAPS,
    READABILITY_TEMPLATES,
    TEST_POINT_FROZEN_PRIORS,
    UNIVERSAL_AXES,
    chat_gemini_vl,
    chat_qwen,
    classify_image_type,
    extract_json,
    format_mmmu_options,
    get_mmmu_image_bytes_list,
    gpt_image_edit,
    JsonlWriter,
    jsonl_iter,
    load_done_keys,
    load_mmmu_manifest,
    log,
    make_clients,
    mmmu_answer_text,
    prepare_image,
    qwen_text_messages,
    qwen_vl_messages,
)


# ── Constants ──────────────────────────────────────────────────────────────────

OPS = ("remove", "add", "modify_property", "substitute_category", "context_change")
EFFECT_TYPES = ("delta", "literal_replace", "flip_boolean", "categorical_shift")

STRUCTURAL_PATTERNS = (
    "dual_value_disambiguation",
    "real_vs_depicted",
    "partial_visibility",
    "category_filtering",
    "lookalike_confusion",
    "spatial_ordering",
    "composition_state",
    "text_in_context",
    "table_lookup",
    "chart_value_reading",
    "diagram_label_reading",
    "medical_finding_identification",
    "music_notation_reading",
    "other",
)

# ── Image-type-specific Call 1 addenda (copied from edit_mmmu) ─────────────────

_IMAGE_TYPE_ADDENDA: dict[str, str] = {
    "table": (
        "\n## IMAGE TYPE OVERRIDE: TABLE / SPREADSHEET\n\n"
        "This image is a rendered table, spreadsheet, or structured data display — NOT a natural photograph.\n\n"
        "### Edit strategy for tables\n"
        "- The primary edit operation is `modify_property` on cell values: change a number, swap a text entry,\n"
        "  alter a percentage, or update a label.\n"
        "- `add` means adding a row or column with new data. `remove` means deleting a row or column.\n"
        "- Do NOT use `substitute_category` or `context_change`.\n"
        "- `structural_pattern` will often be `table_lookup` or `dual_value_disambiguation`.\n\n"
        "### Axis table guidance for tables\n"
        "ALL physical-world axes MUST be set to `frozen` with freeze_constraint = \"not applicable to rendered table\".\n\n"
        "Only these styling axes should be `free`:\n"
        "  - subject_color, subject_material, subject_specific_appearance\n\n"
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
        "Freeze ALL physical-world axes with freeze_constraint = \"not applicable to rendered chart\".\n\n"
        "Free styling axes: subject_color, subject_material, subject_specific_appearance\n\n"
        "### Verifier checks for charts\n"
        'verifier_checks.natural_check should be: "Is this a legible, well-formatted chart with readable axes and data?"\n'
    ),
    "diagram": (
        "\n## IMAGE TYPE OVERRIDE: DIAGRAM / SCHEMATIC / BLUEPRINT\n\n"
        "This image is a diagram, flowchart, circuit, or technical drawing — NOT a natural photograph.\n\n"
        "### Edit strategy for diagrams\n"
        "- `modify_property` on labels, dimensions, node values, edge weights, or annotation text.\n"
        "- `add`/`remove` can mean adding/removing nodes, arrows, or connections.\n"
        "- `structural_pattern` will often be `diagram_label_reading` or `text_in_context`.\n\n"
        "### Axis table guidance for diagrams\n"
        'Freeze ALL physical-world axes with freeze_constraint = "not applicable to rendered diagram".\n\n'
        "Free styling axes: subject_color, subject_material, subject_specific_appearance\n\n"
        "### Verifier checks for diagrams\n"
        'verifier_checks.natural_check should be: "Is this a legible, well-structured diagram with clear labels and connections?"\n'
    ),
    "medical": (
        "\n## IMAGE TYPE OVERRIDE: MEDICAL IMAGE\n\n"
        "This image is a clinical or scientific medical image — NOT a natural photograph.\n\n"
        "### Edit strategy for medical images\n"
        "- `modify_property` can change visible findings (lesion size, staining intensity, measurement values).\n"
        "- Be cautious: if verifying the edit requires deep specialist knowledge, prefer is_editable=false.\n"
        "- `structural_pattern` will often be `medical_finding_identification`.\n\n"
        "### Verifier checks for medical images\n"
        'verifier_checks.natural_check should be: "Is this a plausible medical image with adequate diagnostic quality?"\n'
    ),
    "sheet_music": (
        "\n## IMAGE TYPE OVERRIDE: SHEET MUSIC / MUSICAL NOTATION\n\n"
        "This image contains musical notation on a staff.\n\n"
        "### Edit strategy for sheet music\n"
        "- `modify_property` on note positions (pitch), accidentals, dynamic markings, clef type, or time signature.\n"
        "- `structural_pattern` will often be `music_notation_reading`.\n\n"
        "### Verifier checks for sheet music\n"
        'verifier_checks.natural_check should be: "Is this legible sheet music with correct staff line structure?"\n'
    ),
}


# ── Call 1: analysis JSON (requests 6-10 feasible_edits) ──────────────────────

def _build_call1_params(
    question: str,
    answer: str,
    options: list[str] | None = None,
    answer_letter: str = "",
    image_type_group: str = "natural_photo",
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

    system = (
        "You are an analytical assistant for an image-editing data generation pipeline. "
        "Given an image and a visual question with its ground-truth answer, your job is to "
        "produce a structured analysis JSON that downstream code uses to design controlled edits.\n\n"
        "## Pipeline context\n\n"
        "Stage 1 — surgical answer-changing edit. The pipeline applies one edit that changes the "
        "answer in a controlled way (e.g., remove one object, swap a color, change a text value). "
        "The edited image should look almost identical to the original except for the answer-relevant region.\n\n"
        "Stage 2 — cosmetic diversification on top of Stage 1. Your `frozen_predicate`, "
        "`question_template`, `answer_template`, and `axis_table` drive this stage.\n\n"
        "## Core framework: 5 ops × 4 answer_effect_types\n\n"
        "Five edit operations:\n"
        "  - remove(obj): delete one specific object\n"
        "  - add(obj_like, location): add one same-category object at a feasible spot\n"
        "  - modify_property(obj, property, new_value): change color / size / shape / orientation / text content\n"
        "  - substitute_category(from, to): replace all objects of category X with category Y\n"
        "  - context_change(attribute, value): change background, lighting, viewpoint\n\n"
        "Four answer_effect_types:\n"
        "  - delta: numeric ±\n"
        "  - literal_replace: answer becomes the new literal value\n"
        "  - flip_boolean: yes ↔ no\n"
        "  - categorical_shift: answer shifts among finite options\n\n"
        "**CRITICAL — every feasible_edit MUST change the answer.**\n\n"
        "## IMPORTANT FOR THIS PIPELINE: Aim for 3-5 feasible_edits\n\n"
        "This pipeline targets 5 variants per source image (1:5 ratio). Please aim to produce 3-5 diverse "
        "feasible_edits, spanning as many different answer_effect_types as possible. Only "
        "include edits that genuinely pass all four feasibility checks. If fewer than 3 truly "
        "valid edits exist, output as many as are valid (minimum 1 to pass the is_editable gate).\n\n"
        "## Test point analysis\n\n"
        "Identify which of these structural patterns applies:\n\n"
        "  - dual_value_disambiguation, real_vs_depicted, partial_visibility, category_filtering,\n"
        "    lookalike_confusion, spatial_ordering, composition_state, text_in_context,\n"
        "    table_lookup, chart_value_reading, diagram_label_reading, medical_finding_identification,\n"
        "    music_notation_reading, other\n\n"
        "## Feasibility self-check (all four must pass)\n\n"
        "  1. Physically realistic\n"
        "  2. Preserves test_point\n"
        "  3. Preserves disambiguators\n"
        "  4. Mechanically derivable answer\n\n"
        "## Stage 2 schema (NEW in v5.1) — frozen_predicate + slots + axis_table\n\n"
        "Stage 2 needs to know (a) what is FROZEN (must remain pixel-true so the question keeps "
        "its answer), and (b) what is FREE (can be aggressively varied to maximize visual diversity). "
        "You produce four interlocking pieces:\n\n"
        "### A. frozen_predicate — the answer-determining structure\n\n"
        "Decompose the question into a logical predicate over symbolic slots. Examples:\n"
        "  - 'How many bottles on the shelf?' → `count(<S>) == <N> ∧ on(<S>, <SURF>)` with N=5\n"
        "  - 'What color is the leftmost car?' → `color(leftmost(<S>)) == <V>` with V='red'\n"
        "  - 'Is there a clock on the wall?' → `exists(<S>) ∧ on(<S>, <SURF>)` with answer=true\n\n"
        "  - `answer_function`: documentation-only string showing the predicate in math-like notation.\n"
        "  - `answer_value`: the actual answer value (number / color / yes-no).\n"
        "  - `must_preserve`: 1-3 short bullet-style strings the editor must NOT change pixel-wise.\n"
        "  - `verifier_checks`: 4 yes/no questions Stage 2 Verifier will ask Qwen-VL on the edited image:\n"
        "      • `class_check`:       'Are the prominent subjects in this image <S_plural>?'\n"
        "      • `relation_check`:    'Are the <S_plural> on a <SURF>?'  (or appropriate spatial check)\n"
        "      • `readability_check`: see the test_point readability templates below — fill slot names in.\n"
        "      • `natural_check`:     'Is this a plausible real photograph (not illustration / surreal)?'\n"
        "    Use the EXACT slot placeholders `<S>`, `<S_plural>`, `<SURF>`, etc. — Stage 2 substitutes them.\n"
        "  - `slots`: dict from slot name (e.g., 'S', 'SURF', 'N') to slot definition:\n"
        "      • `role`: short noun phrase describing what this slot is\n"
        "      • `shape_constraint`: visual shape that must hold across all replacements (e.g., 'tall vessel with narrow neck')\n"
        "      • `default_class_singular` / `default_class_plural` (for class-like slots)\n"
        "      • `compatible_classes`: list of {singular, plural} pairs that are visually compatible. "
        "Provide ≥ 5 entries unless slot_replaceability=='locked'. The list MUST include the default class.\n"
        "      • `slot_replaceability`: one of 'free' / 'shape_only' / 'locked'\n"
        "          - free:        any compatible_class works (counting, spatial, attribute non-class questions)\n"
        "          - shape_only:  only shape-strict swaps (default for shape-sensitive questions)\n"
        "          - locked:      this slot's class is the answer; do NOT propose replacements (identity/brand questions)\n"
        "      • For numeric slots like `<N>`: use `{'value': '5', 'frozen': true}` (no compatible_classes).\n\n"
        "### B. question_template — answer-INDEPENDENT, slot-templated\n\n"
        "Rewrite the original question with slot placeholders. Examples:\n"
        "  - 'How many bottles are on the shelf?' → `'How many <S_plural> are on the <SURF>?'`\n"
        "  - 'What color is the leftmost car?' → `'What color is the leftmost <S>?'`\n"
        "  - 'Is there a clock on the wall?' → `'Is there a <S> on the <SURF>?'`\n"
        "Stage 2 substitutes slot values to get the final question text. Do NOT include the answer in the template.\n\n"
        "### C. answer_template — slot-templated answer string\n\n"
        "Usually just `<N>` (or `<V>` etc.) — Stage 2 substitutes the actual_new_answer (post-Stage-1 edit).\n"
        "For text_reading questions where the answer IS visible text, use the literal string (don't template).\n\n"
        "### D. axis_table — 19 universal visual axes (FIXED LIST, classify each)\n\n"
        "For EACH of the 19 axes below, output one entry in axis_table classifying it as:\n"
        "  - `free`: Stage 2 will aggressively vary this axis. Provide an `aggressive_value` field "
        "(a concrete instruction like 'each <S> a distinct vivid color: red / cobalt / amber / emerald / matte black'). "
        "The `aggressive_value` should describe ≥ 3 alternative concrete options separated by ' / ' so Stage 2 "
        "selector has material to pick 4 distinct valuations. ALL aggressive_value strings MUST stay in the natural-photo "
        "distribution (see natural caps below — no surreal effects, no illustrations, no extreme filters).\n"
        "  - `frozen`: Stage 2 must NOT change this axis. Provide a `freeze_constraint` field "
        "explaining what to preserve (e.g., 'exactly 5 distinct bottle silhouettes').\n"
        "  - `depends`: rare; only use if the axis only matters for some test_points and you genuinely cannot decide. "
        "Default to `frozen` when in doubt.\n\n"
        "The 19 axes (default safety in brackets — adjust per this image's test_point):\n"
        f"{axes_block}\n\n"
        "### E. test_point → frozen-axes prior (use as starting point; refine per image)\n\n"
        f"{test_point_priors_block}\n\n"
        "### F. readability check templates (pick the one matching test_point.structural_pattern, fill slots)\n\n"
        f"{readability_block}\n\n"
    )

    if image_type_group != "natural_photo":
        system += _IMAGE_TYPE_ADDENDA.get(image_type_group, "")

    system += "\n## Output\n\nReturn ONLY valid JSON, no markdown fences, no commentary."

    axis_schema_lines = ",\n".join(
        f'    {{"axis": "{ax}", "safety": "free|frozen|depends", '
        f'"aggressive_value": "<concrete ≥3-options string if free/depends, else null>", '
        f'"freeze_constraint": "<what to preserve if frozen, else null>"}}'
        for ax in UNIVERSAL_AXES
    )

    opts_block = ""
    if options:
        opts_display = format_mmmu_options(options)
        opts_block = f"Options: {opts_display}\n"
        answer_display = f"{answer_letter} = {answer}" if answer_letter else answer
    else:
        answer_display = answer

    user_text = (
        f"Original question: {question}\n"
        f"{opts_block}"
        f"Original answer: {answer_display}\n"
        f"Image type hint: {image_type_group}\n\n"
        "Analyze this image and produce JSON with EXACTLY these fields:\n\n"
        "{\n"
        f'  "image_type": "{image_type_group}",\n'
        '  "scene_description": "<1-2 sentence natural caption explicitly describing the test_point structure>",\n'
        '  "question_analysis": {\n'
        '    "answer_determinant": "<what visual fact produces the answer>",\n'
        '    "answer_source_region": "<where in the image to look>",\n'
        '    "current_answer_echo": "<echo source answer verbatim>"\n'
        "  },\n"
        '  "test_point_analysis": {\n'
        '    "core_challenge": "<why this question is non-trivial>",\n'
        f'    "structural_pattern": "<one of: {" | ".join(STRUCTURAL_PATTERNS)}>",\n'
        '    "must_be_described_in_edit": ["<structural element #1>", "..."],\n'
        '    "edit_must_operate_within": "<one-sentence constraint>"\n'
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
        '      "rationale_natural": "<plain English description of this change>",\n'
        '      "answer_effect": {\n'
        f'        "type": "<one of: {" | ".join(EFFECT_TYPES)}>",\n'
        '        "new_answer": "<mechanically derived value>",\n'
        '        "expected_visual_delta": "<what the editor should see different>"\n'
        "      },\n"
        '      "lock_list": ["<element that must not change>"]\n'
        "    }\n"
        "  ],\n"
        '  "frozen_predicate": {\n'
        '    "answer_function": "<predicate string, e.g., count(<S>) == <N> ∧ on(<S>, <SURF>)>",\n'
        '    "answer_value": "<the actual answer as a string>",\n'
        '    "must_preserve": ["<short bullet 1>", "<short bullet 2>", "..."],\n'
        '    "verifier_checks": {\n'
        '      "class_check":      "<yes/no Q with slot placeholders, e.g., \'Are the prominent subjects in this image <S_plural>?\'>",\n'
        '      "relation_check":   "<yes/no Q with slot placeholders, e.g., \'Are the <S_plural> on a <SURF>?\'>",\n'
        '      "readability_check": "<yes/no Q from readability templates above, with slots filled>",\n'
        '      "natural_check":    "Is this a plausible real photograph (not illustration / surreal)?"\n'
        "    },\n"
        '    "slots": {\n'
        '      "S": {\n'
        '        "role": "<short noun phrase>",\n'
        '        "shape_constraint": "<visual shape that must hold across replacements>",\n'
        '        "default_class_singular": "<e.g., \'wine bottle\'>",\n'
        '        "default_class_plural":   "<e.g., \'wine bottles\'>",\n'
        '        "compatible_classes": [\n'
        '          {"singular": "wine bottle", "plural": "wine bottles"},\n'
        '          {"singular": "ceramic vase", "plural": "ceramic vases"},\n'
        '          {"singular": "glass decanter", "plural": "glass decanters"},\n'
        '          {"singular": "narrow flower vase", "plural": "narrow flower vases"},\n'
        '          {"singular": "tall soy sauce bottle", "plural": "tall soy sauce bottles"}\n'
        "        ],\n"
        '        "slot_replaceability": "free | shape_only | locked"\n'
        "      }\n"
        "    }\n"
        "  },\n"
        '  "question_template": "<question with slot placeholders>",\n'
        '  "answer_template": "<usually \'<N>\' or literal>",\n'
        '  "axis_table": [\n'
        f"{axis_schema_lines}\n"
        "  ]\n"
        "}\n\n"
        "Constraints:\n"
        "  - feasible_edits: aim for 3-5 entries across ≥2 effect_types; every entry MUST change the answer\n"
        "  - axis_table MUST contain ALL 19 axes\n"
        "  - Output ONLY the JSON object"
    )
    return system, user_text


# ── Call 1 schema validator (same as edit_mmmu) ────────────────────────────────

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
                return False, f"slot {sname!r} needs ≥3 compatible_classes"
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


# ── Pick diverse edits (all available, up to n_target) ────────────────────────

def _pick_diverse_edits(
    feasible_edits: list[dict], n: int, seed: int = 0
) -> list[dict]:
    """Pick up to n edits, maximizing diversity over effect_type."""
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


# ── Call 2a: Qwen text — surgical edit prose ───────────────────────────────────

def _build_call2a_prompt(
    call1: dict, edit: dict, image_type_group: str = "natural_photo"
) -> tuple[str, str]:
    type_noun = IMAGE_TYPE_NOUNS.get(image_type_group, "image")
    system = (
        "You write image editing instructions for an AI image editor (GPT-Image-2). "
        "Your output is a SINGLE FLOWING PARAGRAPH of natural English prose, 150-250 words.\n\n"
        "Strict format rules:\n"
        "  - NO bullet points, NO labels, NO section headers, NO line breaks within the paragraph\n"
        "  - Write as one continuous, naturally-flowing description an art director would speak aloud\n"
        "  - Use subordinate clauses to express constraints ('while keeping...', 'without altering...')\n\n"
        "The paragraph must accomplish four things in flowing order:\n"
        f"  1. Describe the {type_noun} layout, EXPLICITLY including the test-point structure\n"
        "  2. Specify the single edit operation, with precise spatial/visual details\n"
        "  3. Specify what must remain unchanged, woven into the description\n"
        "  4. Close with a one-clause integrity statement about the difficulty structure being preserved\n\n"
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
        f"  Structural elements that must remain:\n"
        + "\n".join(f"    - {x}" for x in must_describe)
        + "\n\n"
        f"EDIT OPERATION:\n"
        f"  Type: {edit.get('op', '')}\n"
        f"  Specifics: {json.dumps(edit.get('params', {}), ensure_ascii=False)}\n"
        f"  Expected visual outcome: {edit.get('answer_effect', {}).get('expected_visual_delta', '')}\n\n"
        f"MUST NOT CHANGE (weave into description):\n"
        + "\n".join(f"    - {x}" for x in edit.get("lock_list", []))
        + "\n"
        + "\n".join(f"    - {d.get('feature', '')}" for d in disambs)
        + "\n\n"
        f"INTEGRITY CONSTRAINT:\n  {integrity}\n\n"
        "Now write the single flowing paragraph. Output ONLY the paragraph, no other text."
    )
    return system, user_text


# ── Modifier: dual-model cross-validation ─────────────────────────────────────

def _build_modifier_prompt(
    edit: dict,
    new_question: str,
    options: list[str] | None = None,
    image_type_group: str = "natural_photo",
) -> tuple[str, str]:
    type_noun = IMAGE_TYPE_NOUNS.get(image_type_group, "image")
    system = "You are reading the actual answer from an edited image."
    designed = edit.get("answer_effect", {}).get("new_answer", "")
    rationale = edit.get("rationale_natural", "")

    opts_line = ""
    if options:
        opts_line = f"Available options: {format_mmmu_options(options)}\n"

    user_text = (
        f"Background: the image was produced by editing an original {type_noun}.\n"
        f"The editing instruction asked to: {rationale}\n"
        f"If the edit succeeded perfectly, the answer would be: {designed}\n"
        "However, AI image editing may not be perfect — the actual outcome may differ.\n\n"
        "Your task: look at THIS image and determine the ACTUAL answer to the following question.\n\n"
        f"Question: {new_question}\n"
        f"{opts_line}\n"
        "Output strictly in this format (no markdown, no commentary):\n"
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


# ── Stage 1 pipeline ───────────────────────────────────────────────────────────

class Stage1Pipeline:
    def __init__(
        self,
        manifest: str,
        output_dir: str,
        gemini_endpoint: str,
        gemini_api_key: str,
        gemini_model: str,
        qwen_api_base: str,
        qwen_api_key: str,
        qwen_model: str,
        gpt_image_endpoint: str,
        gpt_image_api_key: str,
        max_concurrent: int = 64,
        max_retries: int = 20,
        api_timeout: float = 1200.0,
        resume: bool = True,
        limit: int = 0,
        n_min_edits: int = 5,
        n_max_edits: int = 5,
        use_gemini: bool = False,
    ):
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "stage1_images"
        self.output_jsonl = self.output_dir / "stage1_samples.jsonl"

        self.gemini_endpoint = gemini_endpoint
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model
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
        self.n_min_edits = n_min_edits
        self.n_max_edits = n_max_edits
        self.use_gemini = use_gemini

        self._writer: Optional[JsonlWriter] = None
        self._done_keys: set[str] = set()
        self._inner = None
        self._proxy = None

    async def _process_edit_seed(
        self,
        source_id: str,
        question_orig: str,
        answer_orig: str,
        call1: dict,
        edit: dict,
        edit_idx: int,
        seed_idx: int,
        gpt_bytes: bytes,
        gpt_mime: str,
        edit_prose: str,
        mmmu_meta: dict | None = None,
        image_type_group: str = "natural_photo",
    ) -> bool:
        """Process one (edit, seed) pair. Returns True if written as quality_pass."""
        sample_key = f"{source_id}::e{edit_idx}::s{seed_idx}"
        if sample_key in self._done_keys:
            log(f"  [{sample_key}] resume: already done, skip", "stage1")
            return False

        # Call 3a: GPT-Image-2
        try:
            edited_imgs = await gpt_image_edit(
                self._proxy,
                self.gpt_image_endpoint,
                self.gpt_image_api_key,
                gpt_bytes,
                edit_prose,
                n=1,
                image_mime=gpt_mime,
                max_retries=self.max_retries,
                log_prefix=f"gpt3a/{sample_key}",
            )
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key,
                "source_id": source_id,
                "stage": 1,
                "status": "failed",
                "reject_reason": "call3a_gpt_image_failed",
                "original_question": question_orig,
                "original_answer": answer_orig,
                "edit": edit,
                "edit_prose": edit_prose,
                "call1_full": call1,
                "error": f"{type(e).__name__}: {e!r}",
                "created_at": int(time.time()),
            })
            return False

        if not edited_imgs:
            await self._writer.write({
                "sample_key": sample_key,
                "source_id": source_id,
                "stage": 1,
                "status": "failed",
                "reject_reason": "call3a_no_image",
                "original_question": question_orig,
                "original_answer": answer_orig,
                "edit": edit,
                "edit_prose": edit_prose,
                "call1_full": call1,
                "created_at": int(time.time()),
            })
            return False

        edited_bytes = edited_imgs[0]
        img_path = self.images_dir / f"{sample_key}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(edited_bytes)

        # Modifier
        edited_b64 = base64.b64encode(edited_bytes).decode()
        options = mmmu_meta.get("options") if mmmu_meta else None
        new_question = _rephrase_new_question(question_orig, edit)
        mod_sys, mod_usr = _build_modifier_prompt(
            edit, new_question, options=options, image_type_group=image_type_group
        )

        async def _qwen_modify(temperature: float = 0.0):
            text = await chat_qwen(
                self._inner,
                self.qwen_api_base,
                self.qwen_api_key,
                self.qwen_model,
                qwen_vl_messages(mod_sys, [(edited_b64, "image/png")], mod_usr, no_think=True),
                max_tokens=1536,
                temperature=temperature,
                max_retries=self.max_retries,
                log_prefix=f"mod_qwen/{sample_key}",
            )
            return _parse_modifier_output(text)

        async def _gemini_modify():
            text = await chat_gemini_vl(
                self._inner,
                self.gemini_endpoint,
                self.gemini_api_key,
                mod_sys,
                [(edited_b64, "image/png")],
                mod_usr,
                max_tokens=1536,
                temperature=0.0,
                disable_thinking=True,
                max_retries=self.max_retries,
                log_prefix=f"mod_gemini/{sample_key}",
            )
            return _parse_modifier_output(text)

        try:
            if self.use_gemini:
                mod_a, mod_b = await asyncio.gather(_gemini_modify(), _qwen_modify())
            else:
                mod_a, mod_b = await asyncio.gather(
                    _qwen_modify(temperature=0.0),
                    _qwen_modify(temperature=0.3),
                )
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key,
                "source_id": source_id,
                "stage": 1,
                "status": "failed",
                "reject_reason": "modifier_failed",
                "original_question": question_orig,
                "original_answer": answer_orig,
                "edit": edit,
                "stage1_image_path": str(img_path),
                "edit_prose": edit_prose,
                "call1_full": call1,
                "error": f"{type(e).__name__}: {e!r}",
                "created_at": int(time.time()),
            })
            return False

        norm1 = _normalize_answer(mod_a["actual_answer"])
        norm2 = _normalize_answer(mod_b["actual_answer"])
        agree = norm1 == norm2 and norm1 != ""
        conf_ok = _confidences_ok(mod_a["confidence"], mod_b["confidence"])

        if not agree or not conf_ok:
            await self._writer.write({
                "sample_key": sample_key,
                "source_id": source_id,
                "stage": 1,
                "status": "rejected",
                "reject_reason": "modifier_disagree" if not agree else "modifier_low_confidence",
                "original_question": question_orig,
                "original_answer": answer_orig,
                "edit": edit,
                "stage1_image_path": str(img_path),
                "edit_prose": edit_prose,
                "new_question": new_question,
                "modifier_a": mod_a,
                "modifier_b": mod_b,
                "call1_full": call1,
                "created_at": int(time.time()),
            })
            return False

        actual_new_answer = mod_a["actual_answer"]

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
            "designed_new_answer": edit.get("answer_effect", {}).get("new_answer", ""),
            "actual_new_answer": actual_new_answer,
            "modifier_a": mod_a,
            "modifier_b": mod_b,
            "stage1_image_path": str(img_path),
            "call1_full": call1,
            "edit_idx": edit_idx,
            "seed_idx": seed_idx,
            "created_at": int(time.time()),
        }
        if mmmu_meta:
            record["mmmu_meta"] = mmmu_meta
            record["image_type_group"] = image_type_group
        await self._writer.write(record)
        log(f"  [{sample_key}] ✓ stage1 done | actual={actual_new_answer}", "stage1")
        return True

    async def _process_one(self, row: dict) -> None:
        source_id = row.get("id", "")
        question = row.get("question", "")
        answer_letter = str(row.get("answer", ""))
        options = row.get("options", [])
        if isinstance(options, str):
            from _v5_common import parse_mmmu_options
            options = parse_mmmu_options(options)
        answer_text = mmmu_answer_text(answer_letter, options)
        image_type_group = classify_image_type(row.get("img_type", ""))

        mmmu_meta = {
            "id": source_id,
            "answer_letter": answer_letter,
            "options": options,
            "question_type": row.get("question_type", ""),
            "img_type": row.get("img_type", ""),
            "image_type_group": image_type_group,
            "subject": row.get("subject", ""),
        }

        # Determine target count for this source
        rng = random.Random(hash(source_id) & 0xFFFFFFFF)
        n_target = rng.randint(self.n_min_edits, self.n_max_edits)

        async with self.semaphore:
            try:
                all_bytes = get_mmmu_image_bytes_list(row)
                if not all_bytes:
                    log(f"  [{source_id}] no image, skip", "stage1")
                    return

                gemini_images = []
                for img_bytes in all_bytes:
                    hq_bytes, hq_mime = prepare_image(img_bytes, max_side=1568, fmt="PNG")
                    gemini_images.append((base64.b64encode(hq_bytes).decode(), hq_mime))

                primary_bytes = all_bytes[0]
                gpt_bytes, gpt_mime = prepare_image(primary_bytes, max_side=1024, fmt="JPEG")

                # Resume: check if all expected samples done
                if f"{source_id}::skipped" in self._done_keys:
                    log(f"  [{source_id}] resume: previously skipped", "stage1")
                    return
                if f"{source_id}::no_edits" in self._done_keys:
                    log(f"  [{source_id}] resume: previously no_edits", "stage1")
                    return

                # Call 1 — analysis
                vl_backend = "gemini" if self.use_gemini else "qwen"
                log(f"  [{source_id}] Call 1 ({vl_backend}, type={image_type_group}, target={n_target})...", "stage1")
                sys_p, usr_p = _build_call1_params(
                    question, answer_text, options=options,
                    answer_letter=answer_letter, image_type_group=image_type_group,
                )
                if self.use_gemini:
                    call1_raw = await chat_gemini_vl(
                        self._inner, self.gemini_endpoint, self.gemini_api_key,
                        sys_p, gemini_images, usr_p,
                        max_tokens=32768, temperature=0.2, disable_thinking=False,
                        max_retries=self.max_retries, log_prefix=f"call1/{source_id}",
                    )
                else:
                    call1_raw = await chat_qwen(
                        self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                        qwen_vl_messages(sys_p, gemini_images, usr_p),
                        max_tokens=32768, temperature=0.2,
                        max_retries=self.max_retries, log_prefix=f"call1/{source_id}",
                    )
                try:
                    call1 = extract_json(call1_raw)
                except json.JSONDecodeError as e:
                    await self._writer.write({
                        "sample_key": f"{source_id}::call1_json_error",
                        "source_id": source_id, "stage": 1, "status": "failed",
                        "reject_reason": "call1_json_parse_error",
                        "original_question": question, "original_answer": answer_text,
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
                        "original_question": question, "original_answer": answer_text,
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
                        "original_question": question, "original_answer": answer_text,
                        "call1_full": call1, "mmmu_meta": mmmu_meta,
                        "created_at": int(time.time()),
                    })
                    log(f"  [{source_id}] is_editable=false, skip", "stage1")
                    return

                # Pick as many diverse edits as available (up to n_target)
                all_edits = _pick_diverse_edits(
                    call1.get("feasible_edits", []),
                    n=n_target,
                    seed=hash(source_id) & 0xFFFF,
                )
                if not all_edits:
                    await self._writer.write({
                        "sample_key": f"{source_id}::no_edits",
                        "source_id": source_id, "stage": 1, "status": "skipped",
                        "reject_reason": "no_feasible_edits",
                        "original_question": question, "original_answer": answer_text,
                        "call1_full": call1, "mmmu_meta": mmmu_meta,
                        "created_at": int(time.time()),
                    })
                    return

                n_edits_available = len(all_edits)
                seeds_per_edit = math.ceil(n_target / n_edits_available)

                log(
                    f"  [{source_id}] {n_edits_available} edits × {seeds_per_edit} seeds "
                    f"= up to {n_edits_available * seeds_per_edit} (target {n_target})",
                    "stage1",
                )

                # Generate edit prose for each edit (Call 2a), then fan out seeds
                tasks = []
                for edit_idx, edit in enumerate(all_edits, start=1):
                    # Call 2a per edit (not per seed — same prose, stochastic GPT-Image)
                    try:
                        sys_p, usr_p = _build_call2a_prompt(call1, edit, image_type_group=image_type_group)
                        edit_prose = await chat_qwen(
                            self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                            qwen_text_messages(sys_p, usr_p, no_think=True),
                            max_tokens=4096, temperature=0.6,
                            max_retries=self.max_retries, log_prefix=f"qwen2a/{source_id}::e{edit_idx}",
                        )
                    except Exception as e:
                        await self._writer.write({
                            "sample_key": f"{source_id}::e{edit_idx}::call2a_failed",
                            "source_id": source_id, "stage": 1, "status": "failed",
                            "reject_reason": "call2a_qwen_failed",
                            "original_question": question, "original_answer": answer_text,
                            "edit": edit, "call1_full": call1,
                            "error": f"{type(e).__name__}: {e!r}",
                            "created_at": int(time.time()),
                        })
                        continue

                    for seed_idx in range(seeds_per_edit):
                        tasks.append(self._process_edit_seed(
                            source_id, question, answer_text, call1, edit,
                            edit_idx, seed_idx, gpt_bytes, gpt_mime, edit_prose,
                            mmmu_meta=mmmu_meta, image_type_group=image_type_group,
                        ))

                # Run all (edit, seed) tasks in parallel
                await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                log(f"  [{source_id}] UNEXPECTED ERROR: {type(e).__name__}: {e!r}", "stage1")
                log(traceback.format_exc(), "stage1")
                await self._writer.write({
                    "sample_key": f"{source_id}::unexpected_error",
                    "source_id": source_id, "stage": 1, "status": "failed",
                    "reject_reason": "unexpected_error",
                    "original_question": question, "original_answer": answer_text,
                    "error": f"{type(e).__name__}: {e!r}",
                    "mmmu_meta": mmmu_meta, "created_at": int(time.time()),
                })

    async def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._writer = JsonlWriter(self.output_jsonl)

        if self.resume:
            self._done_keys = load_done_keys(self.output_jsonl)
            log(f"resume: {len(self._done_keys)} sample_keys already done", "stage1")

        log(f"loading manifest: {self.manifest}", "stage1")
        rows = load_mmmu_manifest(self.manifest)
        log(f"loaded {len(rows)} MMMU samples", "stage1")
        if self.limit > 0:
            rows = rows[: self.limit]
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
            f"stage1 done. total: {total} | passed: {passed} | rejected: {rejected} "
            f"| failed: {failed} | skipped: {skipped}",
            "stage1",
        )
        log(f"output jsonl: {self.output_jsonl}", "stage1")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="方案一 Stage 1: 5 answer-changing edits per source (1:5 ratio)")
    parser.add_argument("--manifest", required=True, help="Path to selected_data/manifest.jsonl")
    parser.add_argument("--output_dir", default="_archive/output_fangan1")
    # Gemini VL
    parser.add_argument("--gemini_endpoint", default=os.environ.get("GEMINI_VL_ENDPOINT", "https://runway.devops.rednote.life/openai/google/v1:generateContent"))
    parser.add_argument("--gemini_api_key", default=os.environ.get("GEMINI_VL_API_KEY", ""))
    parser.add_argument("--gemini_model", default=os.environ.get("GEMINI_VL_MODEL", "gemini-3.1-pro-preview"))
    # Qwen
    parser.add_argument("--qwen_api_base", default=os.environ.get("QWEN_API_BASE", "https://runway.devops.xiaohongshu.com/openai/qwen/v1"))
    parser.add_argument("--qwen_api_key", default=os.environ.get("OPENAI_API_KEY", "YOUR_QWEN_API_KEY"))
    parser.add_argument("--qwen_model", default=os.environ.get("QWEN_MODEL", "qwen3.6-plus"))
    # GPT-Image-2
    parser.add_argument("--gpt_image_endpoint", default=os.environ.get("GPT_IMAGE_ENDPOINT", "https://api.gpt.ge/v1"))
    parser.add_argument("--gpt_image_api_key", default=os.environ.get("GPT_IMAGE_API_KEY", "YOUR_GPT_IMAGE_API_KEY"))
    # Run
    parser.add_argument("--max_concurrent", type=int, default=int(os.environ.get("MAX_CONCURRENT", "12")))
    parser.add_argument("--n_min_edits", type=int, default=5, help="Min edits per source (default 5)")
    parser.add_argument("--n_max_edits", type=int, default=5, help="Max edits per source (default 5)")
    parser.add_argument("--max_retries", type=int, default=int(os.environ.get("MAX_RETRIES", "6")))
    parser.add_argument("--api_timeout", type=float, default=float(os.environ.get("API_TIMEOUT", "1200")))
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "0")))
    parser.add_argument("--use_gemini", action="store_true", help="Use Gemini VL for Call 1 and Modifier")
    args = parser.parse_args()

    if not args.gpt_image_api_key:
        parser.error("missing: --gpt_image_api_key / GPT_IMAGE_API_KEY")
    if not args.qwen_api_key:
        parser.error("missing: --qwen_api_key / OPENAI_API_KEY")

    pipe = Stage1Pipeline(
        manifest=args.manifest,
        output_dir=args.output_dir,
        gemini_endpoint=args.gemini_endpoint,
        gemini_api_key=args.gemini_api_key,
        gemini_model=args.gemini_model,
        qwen_api_base=args.qwen_api_base,
        qwen_api_key=args.qwen_api_key,
        qwen_model=args.qwen_model,
        gpt_image_endpoint=args.gpt_image_endpoint,
        gpt_image_api_key=args.gpt_image_api_key,
        max_concurrent=args.max_concurrent,
        n_min_edits=args.n_min_edits,
        n_max_edits=args.n_max_edits,
        max_retries=args.max_retries,
        api_timeout=args.api_timeout,
        resume=not args.no_resume,
        limit=args.limit,
        use_gemini=args.use_gemini,
    )
    asyncio.run(pipe.run())


if __name__ == "__main__":
    main()
