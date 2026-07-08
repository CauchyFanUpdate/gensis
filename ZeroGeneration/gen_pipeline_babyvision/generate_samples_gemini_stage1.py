#!/usr/bin/env python3
"""
ZeroBench Pipeline v5 — Stage 1: Surgical Answer-Changing Edit

Per source question:
  Call 1 [Gemini VL]    → JSON: scene_description prose + test_point_analysis
                                 + feasible_edits + diversification_options
                                 (含 is_editable gate)
  pick 2 feasible_edits (按 effect_type 多样性)
  ↓ per edit (parallel):
    Call 2a [Qwen text]   → surgical edit prose (single seed)
    Call 3a [gemini-3-pro-image-preview] → 1 张 stage1 image (与原图差异极小，只动答案区)
    Modifier [Gemini VL + Qwen3.6-plus VL]  → 双模型交叉验证 actual_new_answer
                                                一致 → pass；分歧 → reject

输出:
  _archive/output_v5/stage1_samples.jsonl
  _archive/output_v5/stage1_images/{sample_key}.png

每 source 题挑 2 edits × 1 seed → 2 张 stage1 候选图
"""

import argparse
import asyncio
import base64
import json
import random
import re
import time
import traceback
from pathlib import Path
from typing import Any, Optional

from _v5_common import (
    AXIS_TIER_DEFAULTS, NATURAL_CAPS, READABILITY_TEMPLATES,
    TEST_POINT_FROZEN_PRIORS, UNIVERSAL_AXES,
    chat_gemini_vl, chat_qwen,
    extract_json, get_image_bytes,
    JsonlWriter, jsonl_iter, load_done_keys, load_parquet, log,
    make_clients, prepare_image, qwen_text_messages, qwen_vl_messages,
)

import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from _gemini_image import (
    gemini_image_edit,
    GEMINI_IMAGE_ENDPOINT,
    GEMINI_IMAGE_API_KEY,
)


# ── Constants: enums ───────────────────────────────────────────────────────

OPS = ("remove", "add", "modify_property", "substitute_category", "context_change")

# v5: drop "no_change". Stage 1 edits MUST change the answer; answer-neutral
# operations (substitute_category with same count, lighting/scene shifts)
# belong in Stage 2's diversification_options where answer_neutral=true is the rule.
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
    "other",
)

# v5.1: DIV_CATEGORIES / DIV_TIER deleted — replaced by universal axis_table from
# _v5_common.UNIVERSAL_AXES + AXIS_TIER_DEFAULTS. Stage 2 selector samples
# axis valuations from Call 1's axis_table instead of picking from a fixed div pool.


# ── Call 1: Gemini VL — analysis JSON ──────────────────────────────────────

def _build_call1_params(question: str, answer: str) -> tuple[str, str]:
    """Build Call 1 (Gemini VL) system + user prompts. See module docstring for schema.

    v5.1: replaced `diversification_options` with `frozen_predicate(slots) +
    question_template + answer_template + axis_table(19)`. Stage 2 selector samples
    axis valuations from this taxonomy instead of picking from a fixed div pool.
    """
    axes_block = "\n".join(
        f"  - {ax}  [default: {AXIS_TIER_DEFAULTS[ax]}]"
        + (f"\n      natural cap: {NATURAL_CAPS[ax]}" if ax in NATURAL_CAPS else "")
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

        "## Pipeline context (two stages, both consume your analysis)\n\n"
        "Stage 1 — surgical answer-changing edit. The pipeline applies one edit that changes the "
        "answer in a controlled way (e.g., remove one object, swap a color, change a text value). "
        "The edited image should look almost identical to the original except for the answer-relevant "
        "region. Your `feasible_edits` list drives this stage.\n\n"
        "Stage 2 — cosmetic diversification on top of Stage 1. The pipeline pushes ALL answer-irrelevant "
        "visual axes (background, surface, color, lighting, camera, ...) to aggressive new values "
        "while keeping the test point intact, producing 4 visually distinct training samples per Stage 1 image. "
        "Your `frozen_predicate`, `question_template`, `answer_template`, and `axis_table` drive this stage.\n\n"

        "## Core framework: 5 ops × 5 answer_effect_types\n\n"
        "Five edit operations cover all reasonable image edits:\n"
        "  - remove(obj): delete one specific object\n"
        "  - add(obj_like, location): add one same-category object at a feasible spot\n"
        "  - modify_property(obj, property, new_value): change color / size / shape / orientation / text content\n"
        "  - substitute_category(from, to): replace all objects of category X with category Y, identical position/size\n"
        "  - context_change(attribute, value): change background, lighting, viewpoint (used in diversification, not feasible_edits)\n\n"
        "FOUR answer_effect_types describe how a Stage 1 edit changes the answer:\n"
        "  - delta: numeric ± (e.g., count 8 → 6)\n"
        "  - literal_replace: answer becomes the new literal value (e.g., color red → blue)\n"
        "  - flip_boolean: yes ↔ no (e.g., 'is there a clock?' goes from yes to no after removal)\n"
        "  - categorical_shift: answer shifts among finite options (e.g., 'which is taller' A → B after shrinking A)\n\n"
        "**CRITICAL — every feasible_edit MUST change the answer.** Identity-preserving operations "
        "(e.g., substitute_category that keeps count, lighting/viewpoint shifts, background swaps) "
        "do NOT belong in feasible_edits — those go in `diversification_options` (Stage 2). "
        "If the only edit you can think of for this image leaves the answer unchanged, "
        "set `is_editable=false` and do not list it in feasible_edits.\n\n"
        "This framework covers all STEM visual question types — counting, attribute, spatial, comparison, existence, text reading. Do NOT invent new ops or effect_types.\n\n"

        "## Test point analysis\n\n"
        "Most visual questions are non-trivial because of a specific structural difficulty (the 'test point'). "
        "Identify which of these 8 structural patterns applies (or 'other' if none fits):\n\n"
        "  - dual_value_disambiguation: image presents multiple candidate values; model must pick the correct one (e.g., a price tag showing both an original and a discounted price; the question asks for the original)\n"
        "  - real_vs_depicted: image mixes real objects with depictions of similar objects; model must judge what counts (e.g., a real cat and printed cartoon cats — count of 'cats' depends on whether to include depictions)\n"
        "  - partial_visibility: some answer-relevant objects are partially occluded; model must judge whether to include them\n"
        "  - category_filtering: multiple categories of similar objects coexist; model must filter by a specific feature (e.g., red bottles vs green bottles, count only the red)\n"
        "  - lookalike_confusion: visually similar non-target objects exist near the target (e.g., decorative sculptures next to real wine bottles)\n"
        "  - spatial_ordering: precise position/order matters (e.g., 'the second from the left', 'the tallest')\n"
        "  - composition_state: states like fill level, openness, completeness (e.g., glass water level, container open/closed)\n"
        "  - text_in_context: text plays a role and must be read correctly (e.g., price tag values, sign content)\n\n"
        "The test_point STRUCTURE (e.g., 'two prices coexist on one tag') must be preserved by any Stage 1 edit. "
        "Your `must_be_described_in_edit` list will be embedded into the Stage 1 edit prompt's scene description, "
        "so the image editor knows to keep that structure intact. Be specific: don't write 'the dual-price layout', "
        "write 'the price tag must show TWO prices: one original (smaller, possibly struck-through) and one discount (larger, highlighted)'.\n\n"

        "## Disambiguators\n\n"
        "Disambiguators are visual features that distinguish target objects from confusers (e.g., wine bottles have green caps, "
        "decorative ones have brass caps). These must NEVER be modified by any edit, otherwise the test point degenerates and "
        "the question becomes trivial. List every concrete disambiguator the model should rely on.\n\n"

        "## Feasibility self-check\n\n"
        "Before adding each feasible_edit to your list, mentally check ALL FOUR:\n"
        "  1. Physically realistic: does the edit make sense in the world? Don't add 30 bottles to a 12-slot shelf.\n"
        "  2. Preserves test_point: don't break the structural difficulty (e.g., if test_point is dual-price, can't remove one price).\n"
        "  3. Preserves disambiguators: don't change a feature that distinguishes target from confuser.\n"
        "  4. Mechanically derivable answer: the new answer follows from the operation by formula (count - K, color = new_color, etc.).\n\n"
        "Only output edits that pass all four checks. Aim for 3-5 feasible_edits across at least 2 different answer_effect_types. "
        "If no edit can pass, set `is_editable=false` and explain in `skip_reason`.\n\n"

        "## When to set is_editable=false\n\n"
        "Some images are inherently un-editable for our purposes:\n"
        "  - Sokoban-like puzzle grids where every cell determines the answer\n"
        "  - Complex maze layouts (any cell change ruins path-finding semantics)\n"
        "  - Charts/diagrams with many tightly-coupled elements\n"
        "  - Counting questions where the count is very large (>15) and a small ± wouldn't be visibly clean\n"
        "  - Questions whose answer depends on global structure (chess en passant validity, etc.) where surgical edits cascade unpredictably\n\n"
        "For these, output `is_editable=false` with a one-sentence `skip_reason`, and leave `feasible_edits` and `diversification_options` empty.\n\n"

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

        "## Output\n\n"
        "Return ONLY valid JSON, no markdown fences, no commentary. Schema in the user message."
    )

    axis_schema_lines = ",\n".join(
        f'    {{"axis": "{ax}", "safety": "free|frozen|depends", '
        f'"aggressive_value": "<concrete ≥3-options string if free/depends, else null>", '
        f'"freeze_constraint": "<what to preserve if frozen, else null>"}}'
        for ax in UNIVERSAL_AXES
    )

    user_text = (
        f"Original question: {question}\n"
        f"Original answer: {answer}\n\n"
        "Analyze this image and produce JSON with EXACTLY these fields and nothing else:\n\n"
        "{\n"
        '  "image_type": "natural_photo | diagram | chart | abstract_puzzle",\n'
        '\n'
        '  "scene_description": "<a 1-2 sentence natural caption of the scene that explicitly describes the test_point structure '
        '— this exact text will be reused as the SCENE part of the edit prompt sent to the image editor, so write it as natural prose, '
        'no labels, no bullet points>",\n'
        '\n'
        '  "question_analysis": {\n'
        '    "answer_determinant": "<what visual fact in the image directly produces the answer>",\n'
        '    "answer_source_region": "<where in the image to look — coarse region description, e.g., \'top shelf, left half\'>",\n'
        '    "current_answer_echo": "<echo the source answer verbatim for sanity check>"\n'
        '  },\n'
        '\n'
        '  "test_point_analysis": {\n'
        '    "core_challenge": "<one sentence: why this question is non-trivial; if it is trivial, say so>",\n'
        f'    "structural_pattern": "<one of: {" | ".join(STRUCTURAL_PATTERNS)}>",\n'
        '    "must_be_described_in_edit": [\n'
        '      "<concrete structural element #1 that must appear in every edit prompt scene description>",\n'
        '      "..."\n'
        '    ],\n'
        '    "edit_must_operate_within": "<one sentence constraint any edit must respect to keep the test_point intact>"\n'
        '  },\n'
        '\n'
        '  "disambiguators_to_preserve": [\n'
        '    {"feature": "<concrete visual feature that distinguishes target from confusers>", "scope": "<which objects this applies to>"}\n'
        '  ],\n'
        '\n'
        '  "feasibility_assessment": {\n'
        '    "is_editable": true,\n'
        '    "skip_reason": "<empty string if is_editable=true; one sentence if false>"\n'
        '  },\n'
        '\n'
        '  "feasible_edits": [\n'
        '    {\n'
        '      "edit_id": "e1",\n'
        f'      "op": "<one of: {" | ".join(OPS)}>",\n'
        '      "params": {\n'
        '        // op-specific. Examples:\n'
        '        // remove          → {"object": "the rightmost wine bottle on top shelf"}\n'
        '        // add             → {"object": "wine bottle", "location": "rightmost slot on top shelf, matching others"}\n'
        '        // modify_property → {"object": "the leftmost balloon", "property": "color", "from": "red", "to": "blue"}\n'
        '        // substitute_category → {"from_category": "wine bottles", "to_category": "mineral water bottles", "scope": "all on top shelf"}\n'
        '        // context_change  → {"attribute": "lighting", "from": "fluorescent", "to": "warm afternoon sun"}\n'
        '      },\n'
        '      "rationale_natural": "<one sentence describing this change in plain English; used by Stage 1 Modifier>",\n'
        '      "answer_effect": {\n'
        f'        "type": "<one of: {" | ".join(EFFECT_TYPES)}>",\n'
        '        "new_answer": "<mechanically derived value>",\n'
        '        "expected_visual_delta": "<one sentence: what the editor and modifier should see different in the result>"\n'
        '      },\n'
        '      "lock_list": ["<element that must not change>", "..."]\n'
        '    }\n'
        '  ],\n'
        '\n'
        '  "frozen_predicate": {\n'
        '    "answer_function": "<predicate string, e.g., count(<S>) == <N> ∧ on(<S>, <SURF>)>",\n'
        '    "answer_value": "<the actual answer as a string>",\n'
        '    "must_preserve": ["<short bullet 1>", "<short bullet 2>", "..."],\n'
        '    "verifier_checks": {\n'
        '      "class_check":      "<yes/no Q with slot placeholders, e.g., \'Are the prominent subjects in this image <S_plural>?\'>",\n'
        '      "relation_check":   "<yes/no Q with slot placeholders, e.g., \'Are the <S_plural> on a <SURF>?\'>",\n'
        '      "readability_check": "<yes/no Q from readability templates above, with slots filled>",\n'
        '      "natural_check":    "Is this a plausible real photograph (not illustration / surreal)?"\n'
        '    },\n'
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
        '        ],\n'
        '        "slot_replaceability": "free | shape_only | locked"\n'
        '      },\n'
        '      "SURF": {\n'
        '        "role": "horizontal supporting surface",\n'
        '        "shape_constraint": "flat horizontal surface able to support the <S>",\n'
        '        "default_class_singular": "wooden shelf",\n'
        '        "default_class_plural":   "wooden shelves",\n'
        '        "compatible_classes": [\n'
        '          {"singular": "shelf", "plural": "shelves"},\n'
        '          {"singular": "table", "plural": "tables"},\n'
        '          {"singular": "countertop", "plural": "countertops"},\n'
        '          {"singular": "marble plank", "plural": "marble planks"},\n'
        '          {"singular": "metal rack", "plural": "metal racks"}\n'
        '        ],\n'
        '        "slot_replaceability": "free"\n'
        '      },\n'
        '      "N": {"value": "<answer value as string>", "frozen": true}\n'
        '    }\n'
        '  },\n'
        '\n'
        '  "question_template": "<question with <S>/<S_plural>/<SURF>/etc placeholders, no answer value>",\n'
        '  "answer_template":   "<usually \'<N>\' or just the literal answer string for text_reading>",\n'
        '\n'
        '  "axis_table": [\n'
        f'{axis_schema_lines}\n'
        '  ]\n'
        "}\n\n"
        "Constraints reminder:\n"
        "  - feasible_edits: every entry MUST change the answer; aim for 3-5 entries across ≥2 of\n"
        "    {delta, literal_replace, flip_boolean, categorical_shift}\n"
        "  - frozen_predicate.slots:\n"
        "      • Slot keys MUST match the placeholders used in question_template / verifier_checks (case-sensitive,\n"
        "        without angle brackets in the key). Use 'S' for subject, 'SURF' for surface, 'N' for numeric answer, etc.\n"
        "      • For class-like slots: provide ≥ 5 compatible_classes unless slot_replaceability=='locked'.\n"
        "        Compatible classes MUST share the shape_constraint (don't list 'cup' if shape is 'tall narrow vessel').\n"
        "      • For numeric/boolean answer slots like <N>: only fields are 'value' (string) and 'frozen' (bool true).\n"
        "  - axis_table MUST contain ALL 19 axes in the order shown above. Each entry has 'safety' set to one of\n"
        "    free/frozen/depends; aggressive_value present iff free or depends; freeze_constraint present iff frozen.\n"
        "  - For free axes, aggressive_value MUST list ≥ 3 alternative concrete options separated by ' / ', so Stage 2\n"
        "    has material to pick 4 distinct valuations. Stay strictly within the natural-photo distribution\n"
        "    (no surreal weather, no illustrations, no extreme color filters).\n"
        "  - If is_editable=false, frozen_predicate / question_template / answer_template / axis_table can be empty objects;\n"
        "    feasible_edits must be empty.\n"
        "  - Output ONLY the JSON object. No markdown, no commentary."
    )
    return system, user_text


# ── v5.1: Call 1 schema validator ──────────────────────────────────────────

def _validate_call1_schema(call1: dict) -> tuple[bool, str]:
    """
    Validate v5.1 Call 1 output has all required fields with correct shape.
    Returns (is_valid, error_message). Empty error_message on success.

    Skipped sources (is_editable=false) are exempt from frozen_predicate / axis_table
    validation since those fields are allowed to be empty.
    """
    feas = call1.get("feasibility_assessment", {})
    if not feas.get("is_editable", False):
        return True, ""  # skipped sources don't need new schema

    # frozen_predicate
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
    # at least one class-like slot must have compatible_classes (allow numeric-only edge case)
    has_class_slot = False
    for sname, sdef in slots.items():
        if not isinstance(sdef, dict):
            return False, f"slot {sname!r} not a dict"
        if "compatible_classes" in sdef:
            has_class_slot = True
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
    if not has_class_slot:
        # numeric-only predicate (rare); allow but warn via field
        pass

    # question_template
    qt = call1.get("question_template", "")
    if not isinstance(qt, str) or not qt.strip():
        return False, "question_template missing or empty"

    # answer_template
    at = call1.get("answer_template", "")
    if not isinstance(at, str) or not at.strip():
        return False, "answer_template missing or empty"

    # axis_table — must be list with all 19 axes
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


# ── Pick 2 diverse feasible_edits by effect_type ───────────────────────────

def _pick_diverse_edits(feasible_edits: list[dict], n: int = 2, seed: int = 0) -> list[dict]:
    """
    Pick `n` edits, maximizing diversity over answer_effect.type.
    Strategy: bucket by effect_type, then round-robin pick one from each bucket.

    v5: filter out any edit with answer_effect.type == "no_change" (Stage 1 must change answer).
    """
    if not feasible_edits:
        return []
    # v5: enforce stage 1 = answer-changing only (safety net even if Call 1 violates)
    filtered = [e for e in feasible_edits
                if e.get("answer_effect", {}).get("type") in EFFECT_TYPES
                and e.get("answer_effect", {}).get("type") != "no_change"]
    if not filtered:
        return []
    rng = random.Random(seed)
    buckets: dict[str, list[dict]] = {}
    for e in filtered:
        et = e.get("answer_effect", {}).get("type", "?")
        buckets.setdefault(et, []).append(e)
    # shuffle within bucket for fairness
    for vs in buckets.values():
        rng.shuffle(vs)
    # round-robin across buckets
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


# ── Call 2a: Qwen text — surgical edit prose ───────────────────────────────

def _build_call2a_prompt(call1: dict, edit: dict) -> tuple[str, str]:
    """
    Build Qwen text prompt for Stage 1 surgical edit prose.
    Output is a single flowing paragraph for gemini-3-pro-image-preview.
    """
    system = (
        "You write image editing instructions for an AI image editor (gemini-3-pro-image-preview). "
        "Your output is a SINGLE FLOWING PARAGRAPH of natural English prose, 150-250 words.\n\n"
        "Strict format rules:\n"
        "  - NO bullet points, NO labels (no 'SCENE:', 'EDIT:', 'KEEP:', etc.)\n"
        "  - NO section headers, NO line breaks within the paragraph\n"
        "  - Write as one continuous, naturally-flowing description an art director would speak aloud\n"
        "  - Use subordinate clauses to express constraints ('while keeping...', 'without altering...', 'so that...')\n"
        "  - Naturally emphasize critical constraints with words like 'exactly', 'only', 'still'\n\n"
        "The paragraph must accomplish four things in flowing order:\n"
        "  1. Describe the scene vividly, EXPLICITLY including the test-point structure that creates the question's difficulty\n"
        "  2. Specify the single edit operation, with precise spatial/visual details\n"
        "  3. Specify what must remain unchanged, woven into the description (not listed)\n"
        "  4. Close with a one-clause integrity statement about the difficulty structure being preserved\n\n"
        "Stay concrete and visual. Do not invent details not in the inputs. Do not exaggerate."
    )

    test_point = call1.get("test_point_analysis", {})
    disambs = call1.get("disambiguators_to_preserve", [])
    must_describe = test_point.get("must_be_described_in_edit", [])
    integrity = test_point.get("edit_must_operate_within", "")

    user_text = (
        "Compose the image editing instruction.\n\n"
        f"SCENE CONTEXT (use this as the foundation of your description):\n{call1.get('scene_description', '')}\n\n"
        f"TEST POINT — the difficulty structure that MUST remain visible after the edit:\n"
        f"  Core challenge: {test_point.get('core_challenge', '')}\n"
        f"  Structural elements that must remain in the edited image:\n"
        + "\n".join(f"    - {x}" for x in must_describe)
        + "\n\n"
        f"EDIT OPERATION:\n"
        f"  Type: {edit.get('op', '')}\n"
        f"  Specifics: {json.dumps(edit.get('params', {}), ensure_ascii=False)}\n"
        f"  Expected visual outcome: {edit.get('answer_effect', {}).get('expected_visual_delta', '')}\n\n"
        f"MUST NOT CHANGE (weave these into the description, do not list):\n"
        + "\n".join(f"    - {x}" for x in edit.get('lock_list', []))
        + "\n"
        + "\n".join(f"    - {d.get('feature', '')}" for d in disambs)
        + "\n\n"
        f"INTEGRITY CONSTRAINT (close the paragraph with this idea, paraphrased):\n  {integrity}\n\n"
        "Now write the single flowing paragraph for gemini-3-pro-image-preview. Output ONLY the paragraph, no other text."
    )
    return system, user_text


# ── Modifier: dual-model cross-validation ──────────────────────────────────

def _build_modifier_prompt(edit: dict, new_question: str) -> tuple[str, str]:
    """Build Modifier prompt (Gemini VL or Qwen VL — same prompt body)."""
    system = (
        "You are reading the actual answer from an edited image."
    )
    designed = edit.get("answer_effect", {}).get("new_answer", "")
    rationale = edit.get("rationale_natural", "")
    user_text = (
        "Background: the image shown was produced by editing an original photograph.\n"
        f"The editing instruction asked to: {rationale}\n"
        f"If the edit succeeded perfectly, the answer would be: {designed}\n"
        "However, AI image editing may not be perfect — the actual outcome may differ slightly.\n\n"
        "Your task: look at THIS image (the edited result) and determine the ACTUAL answer "
        "to the following question, based on what you literally see in the image.\n\n"
        f"Question: {new_question}\n\n"
        "Output strictly in this format (no markdown, no commentary):\n"
        "ACTUAL_ANSWER: <the value you see, as a single short phrase or number>\n"
        "CONFIDENCE: high | medium | low\n"
        "REASONING: <one sentence describing what you see that supports your answer>\n"
    )
    return system, user_text


def _parse_modifier_output(text: str) -> dict:
    """Parse ACTUAL_ANSWER / CONFIDENCE / REASONING from modifier output."""
    out: dict = {"actual_answer": "", "confidence": "low", "reasoning": "", "raw": text}

    def _grab(field: str) -> str:
        # capture from `FIELD:` until next known FIELD_NAME: (containing _) or end
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
    """Loose normalization for cross-model agreement check."""
    a = (a or "").strip().lower()
    a = re.sub(r"[^\w\s./-]", " ", a)
    a = re.sub(r"\s+", " ", a).strip()
    # treat leading/trailing zeros, plurals lightly
    if re.fullmatch(r"\d+\.0+", a):
        a = a.split(".")[0]
    if a.endswith("s") and len(a) > 1 and not a.endswith("ss"):
        a_singular = a[:-1]
        # only treat as plural if contextual (heuristic)
        if not a_singular.isdigit():
            a = a_singular
    return a


def _confidences_ok(c1: str, c2: str) -> bool:
    """At least one of the two must be >= medium."""
    rank = {"high": 2, "medium": 1, "low": 0}
    return max(rank.get(c1, 0), rank.get(c2, 0)) >= 1


# ── Stage 1 main per-source flow ───────────────────────────────────────────

class Stage1Pipeline:
    def __init__(
        self,
        parquet: str,
        output_dir: str,
        gemini_endpoint: str,
        gemini_api_key: str,
        gemini_model: str,
        qwen_api_base: str,
        qwen_api_key: str,
        qwen_model: str,
        gemini_image_endpoint: str,
        gemini_image_api_key: str,
        max_concurrent: int = 64,
        max_retries: int = 20,         # v5.1: bumped 4→6 for larger schema parse-resilience
        api_timeout: float = 1200.0,  # v5.1: bumped 480→1200 (Call 1 thinking can run 60-90s)
        resume: bool = True,
        limit: int = 0,
        n_edits_per_source: int = 2,
    ):
        self.parquet = parquet
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "stage1_images"
        self.output_jsonl = self.output_dir / "stage1_samples.jsonl"

        self.gemini_endpoint = gemini_endpoint
        self.gemini_api_key = gemini_api_key
        self.gemini_model = gemini_model

        self.qwen_api_base = qwen_api_base
        self.qwen_api_key = qwen_api_key
        self.qwen_model = qwen_model

        self.gemini_image_endpoint = gemini_image_endpoint
        self.gemini_image_api_key = gemini_image_api_key

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.api_timeout = api_timeout
        self.resume = resume
        self.limit = limit
        self.n_edits = n_edits_per_source

        self._writer: Optional[JsonlWriter] = None
        self._done_keys: set[str] = set()
        self._inner: Optional[object] = None
        self._proxy: Optional[object] = None

    # — single edit subtask —
    async def _process_edit(
        self, source_id: str, question_orig: str, answer_orig: str,
        call1: dict, edit: dict, edit_idx: int,
        gemini_b64: str, gemini_mime: str,
        gpt_bytes: bytes, gpt_mime: str,
    ) -> None:
        sample_key = f"{source_id}::e{edit_idx}"
        if sample_key in self._done_keys:
            log(f"  [{sample_key}] resume: already done, skip", "stage1")
            return

        try:
            # Call 2a: Qwen text → edit prose
            sys_p, usr_p = _build_call2a_prompt(call1, edit)
            edit_prose = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                qwen_text_messages(sys_p, usr_p, no_think=True),
                max_tokens=4096, temperature=0.6,
                max_retries=self.max_retries, log_prefix=f"qwen2a/{sample_key}",
            )
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key,
                "source_id": source_id,
                "stage": 1,
                "status": "failed",
                "reject_reason": "call2a_qwen_failed",
                "original_question": question_orig,
                "original_answer": answer_orig,
                "edit": edit,
                "call1_full": call1,
                "error": f"{type(e).__name__}: {e!r}",
                "created_at": int(time.time()),
            })
            return

        # Compute new_question for Modifier (use rephrased text if substitute_category)
        new_question = _rephrase_new_question(question_orig, edit)

        # Call 3a: gemini-3-pro-image-preview single-seed edit
        try:
            edited_imgs = await gemini_image_edit(
                self._inner, self.gemini_image_endpoint, self.gemini_image_api_key,
                gpt_bytes, edit_prose, n=1, image_mime=gpt_mime,
                max_retries=self.max_retries, log_prefix=f"gemini3a/{sample_key}",
            )
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key,
                "source_id": source_id,
                "stage": 1,
                "status": "failed",
                "reject_reason": "call3a_gemini_image_failed",
                "original_question": question_orig,
                "original_answer": answer_orig,
                "edit": edit,
                "edit_prose": edit_prose,
                "call1_full": call1,
                "error": f"{type(e).__name__}: {e!r}",
                "created_at": int(time.time()),
            })
            return

        if not edited_imgs:
            await self._writer.write({
                "sample_key": sample_key,
                "source_id": source_id,
                "stage": 1,
                "status": "failed",
                "reject_reason": "call3a_gemini_no_image",
                "original_question": question_orig,
                "original_answer": answer_orig,
                "edit": edit,
                "edit_prose": edit_prose,
                "call1_full": call1,
                "created_at": int(time.time()),
            })
            return

        edited_bytes = edited_imgs[0]
        img_path = self.images_dir / f"{sample_key}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(edited_bytes)

        # Modifier: dual-model
        edited_b64 = base64.b64encode(edited_bytes).decode()
        mod_sys, mod_usr = _build_modifier_prompt(edit, new_question)

        async def _gemini_modify():
            # Modifier: thinking off (high frequency, narrow recognition task)
            text = await chat_gemini_vl(
                self._inner, self.gemini_endpoint, self.gemini_api_key,
                mod_sys, [(edited_b64, "image/png")], mod_usr,
                max_tokens=1536, temperature=0.0, disable_thinking=True,
                max_retries=self.max_retries, log_prefix=f"mod_gemini/{sample_key}",
            )
            return _parse_modifier_output(text)

        async def _qwen_modify():
            text = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                qwen_vl_messages(mod_sys, [(edited_b64, "image/png")], mod_usr, no_think=True),
                max_tokens=1536, temperature=0.0,
                max_retries=self.max_retries, log_prefix=f"mod_qwen/{sample_key}",
            )
            return _parse_modifier_output(text)

        try:
            gemini_mod, qwen_mod = await asyncio.gather(_gemini_modify(), _qwen_modify())
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
            return

        # Cross-validate
        norm1 = _normalize_answer(gemini_mod["actual_answer"])
        norm2 = _normalize_answer(qwen_mod["actual_answer"])
        agree = (norm1 == norm2 and norm1 != "")
        conf_ok = _confidences_ok(gemini_mod["confidence"], qwen_mod["confidence"])

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
                "designed_new_answer": edit.get("answer_effect", {}).get("new_answer", ""),
                "modifier_gemini": gemini_mod,
                "modifier_qwen": qwen_mod,
                "call1_full": call1,
                "created_at": int(time.time()),
            })
            return

        actual_new_answer = gemini_mod["actual_answer"]   # both agree, use either

        await self._writer.write({
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
            "modifier_gemini": gemini_mod,
            "modifier_qwen": qwen_mod,
            "stage1_image_path": str(img_path),
            "call1_full": call1,
            "created_at": int(time.time()),
        })
        log(f"  [{sample_key}] ✓ stage1 done | actual={actual_new_answer}", "stage1")

    # — single source —
    async def _process_one(self, row: dict) -> None:
        # Support BabyVision (taskId/question/blankAns+choiceAns) and ZeroBench (question_id/question_text/question_answer)
        source_id = str(row.get("question_id") or row.get("taskId", ""))
        question = row.get("question_text") or row.get("question", "")
        # BabyVision: blank questions use blankAns; choice questions use choiceAns (float → int → str)
        raw_ans = row.get("question_answer")
        if raw_ans is None:
            raw_ans = row.get("blankAns")
        if raw_ans is None or (isinstance(raw_ans, float) and __import__("math").isnan(raw_ans)):
            raw_choice = row.get("choiceAns")
            raw_ans = int(raw_choice) if raw_choice is not None and not (isinstance(raw_choice, float) and __import__("math").isnan(raw_choice)) else ""
        answer = str(raw_ans) if raw_ans != "" else ""

        async with self.semaphore:
            try:
                # Image prep
                raw_bytes = get_image_bytes(row)
                if not raw_bytes:
                    log(f"  [{source_id}] no image, skip", "stage1")
                    return
                hq_bytes, hq_mime = prepare_image(raw_bytes, max_side=1568, fmt="PNG")
                gpt_bytes, gpt_mime = prepare_image(raw_bytes, max_side=1024, fmt="JPEG")
                gemini_b64 = base64.b64encode(hq_bytes).decode()

                # Resume: skip source if it has been processed before in any of these ways:
                #   - Permanent skip markers (no point re-running Call 1 if already gated)
                #     · {source_id}::skipped         → is_editable=false (Call 1 said no)
                #     · {source_id}::no_edits        → Call 1 returned empty feasible_edits
                #   - All edit slots filled (success / reject / fail all count as "done")
                # Retryable failures (no permanent marker → Call 1 re-runs):
                #   · {source_id}::call1_json_error → JSON parse error, retry
                #   · {source_id}::unexpected_error → exception during processing, retry
                if f"{source_id}::skipped" in self._done_keys:
                    log(f"  [{source_id}] resume: previously skipped (is_editable=false)", "stage1")
                    return
                if f"{source_id}::no_edits" in self._done_keys:
                    log(f"  [{source_id}] resume: previously no_edits", "stage1")
                    return
                if all(f"{source_id}::e{i+1}" in self._done_keys for i in range(self.n_edits)):
                    log(f"  [{source_id}] resume: all edits done, skip", "stage1")
                    return

                # Call 1
                log(f"  [{source_id}] Call 1 (Gemini analysis)...", "stage1")
                sys_p, usr_p = _build_call1_params(question, answer)
                # Call 1 v5.1: quality-first — thinking ON + max_tokens 32k for axis_table+slots+CoT
                call1_raw = await chat_gemini_vl(
                    self._inner, self.gemini_endpoint, self.gemini_api_key,
                    sys_p, [(gemini_b64, hq_mime)], usr_p,
                    max_tokens=32768, temperature=0.2, disable_thinking=False,
                    max_retries=self.max_retries, log_prefix=f"call1/{source_id}",
                )
                try:
                    call1 = extract_json(call1_raw)
                except json.JSONDecodeError as e:
                    await self._writer.write({
                        "sample_key": f"{source_id}::call1_json_error",
                        "source_id": source_id,
                        "stage": 1,
                        "status": "failed",
                        "reject_reason": "call1_json_parse_error",
                        "original_question": question,
                        "original_answer": answer,
                        "error": f"{type(e).__name__}: {e!r}",
                        "raw_call1": call1_raw[:2000],
                        "created_at": int(time.time()),
                    })
                    return

                # v5.1: validate new schema (axis_table + frozen_predicate + slots + templates)
                ok, err = _validate_call1_schema(call1)
                if not ok:
                    await self._writer.write({
                        "sample_key": f"{source_id}::call1_schema_invalid",
                        "source_id": source_id,
                        "stage": 1,
                        "status": "failed",
                        "reject_reason": "call1_schema_invalid",
                        "original_question": question,
                        "original_answer": answer,
                        "error": err,
                        "call1_full": call1,
                        "created_at": int(time.time()),
                    })
                    log(f"  [{source_id}] schema invalid: {err}", "stage1")
                    return

                # Gate: is_editable
                feas = call1.get("feasibility_assessment", {})
                if not feas.get("is_editable", False):
                    await self._writer.write({
                        "sample_key": f"{source_id}::skipped",
                        "source_id": source_id,
                        "stage": 1,
                        "status": "skipped",
                        "reject_reason": "is_editable_false",
                        "skip_reason": feas.get("skip_reason", ""),
                        "original_question": question,
                        "original_answer": answer,
                        "call1_full": call1,
                        "created_at": int(time.time()),
                    })
                    log(f"  [{source_id}] is_editable=false, skip", "stage1")
                    return

                # Pick 2 diverse edits
                edits = _pick_diverse_edits(call1.get("feasible_edits", []), n=self.n_edits, seed=hash(source_id) & 0xFFFF)
                if not edits:
                    await self._writer.write({
                        "sample_key": f"{source_id}::no_edits",
                        "source_id": source_id,
                        "stage": 1,
                        "status": "skipped",
                        "reject_reason": "no_feasible_edits",
                        "original_question": question,
                        "original_answer": answer,
                        "call1_full": call1,
                        "created_at": int(time.time()),
                    })
                    return

                # Process each edit in parallel
                tasks = [
                    self._process_edit(
                        source_id, question, answer, call1, e, idx + 1,
                        gemini_b64, hq_mime, gpt_bytes, gpt_mime,
                    )
                    for idx, e in enumerate(edits)
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                log(f"  [{source_id}] UNEXPECTED ERROR: {type(e).__name__}: {e!r}", "stage1")
                log(traceback.format_exc(), "stage1")
                await self._writer.write({
                    "sample_key": f"{source_id}::unexpected_error",
                    "source_id": source_id,
                    "stage": 1,
                    "status": "failed",
                    "reject_reason": "unexpected_error",
                    "original_question": question,
                    "original_answer": answer,
                    "error": f"{type(e).__name__}: {e!r}",
                    "created_at": int(time.time()),
                })

    # — main —
    async def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._writer = JsonlWriter(self.output_jsonl)

        if self.resume:
            self._done_keys = load_done_keys(self.output_jsonl)
            log(f"resume: {len(self._done_keys)} sample_keys already done", "stage1")

        log(f"loading parquet: {self.parquet}", "stage1")
        rows = load_parquet(self.parquet)
        log(f"loaded {len(rows)} subquestions", "stage1")
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

        # Summary
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
        log(f"stage1 done. total records: {total} | passed: {passed} | rejected: {rejected} | failed: {failed} | skipped: {skipped}", "stage1")
        log(f"output jsonl: {self.output_jsonl}", "stage1")


# ── new_question rephraser ──────────────────────────────────────────────────

def _rephrase_new_question(orig_q: str, edit: dict) -> str:
    """
    Generate the new_question for Stage 1 based on the edit.
    For most ops, the question is unchanged (Modifier reads the actual answer from the new image).
    For substitute_category, swap the noun.
    """
    op = edit.get("op", "")
    params = edit.get("params", {})
    if op == "substitute_category":
        from_cat = params.get("from_category", "")
        to_cat = params.get("to_category", "")
        if from_cat and to_cat:
            # naive case-insensitive replace; works for plain noun phrases
            pat = re.compile(re.escape(from_cat), re.IGNORECASE)
            return pat.sub(to_cat, orig_q)
    return orig_q


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZeroBench v5 Stage 1: surgical edit + Modifier")
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output_dir", default="zerobench_pipeline/_archive/output_gemini")
    # Gemini VL
    parser.add_argument("--gemini_endpoint", default=os.environ.get("GEMINI_VL_ENDPOINT", "https://runway.devops.rednote.life/openai/google/v1:generateContent"))
    parser.add_argument("--gemini_api_key", default=os.environ.get("GEMINI_VL_API_KEY", ""))
    parser.add_argument("--gemini_model", default=os.environ.get("GEMINI_VL_MODEL", "gemini-3.1-pro-preview"))
    # Qwen
    parser.add_argument("--qwen_api_base", default=os.environ.get("QWEN_API_BASE", "https://runway.devops.xiaohongshu.com/openai/qwen/v1"))
    parser.add_argument("--qwen_api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--qwen_model", default=os.environ.get("QWEN_MODEL", "qwen3.6-plus"))
    # gemini-3-pro-image-preview
    parser.add_argument("--gemini_image_endpoint", default=os.environ.get("GEMINI_IMAGE_ENDPOINT", "https://runway.devops.rednote.life/openai/google/v1:generateContent"))
    parser.add_argument("--gemini_image_api_key", default=os.environ.get("GEMINI_IMAGE_API_KEY", "YOUR_GEMINI_IMAGE_API_KEY"))
    # Run
    parser.add_argument("--max_concurrent", type=int, default=int(os.environ.get("MAX_CONCURRENT", "64")))
    parser.add_argument("--n_edits_per_source", type=int, default=2)
    parser.add_argument("--max_retries", type=int, default=int(os.environ.get("MAX_RETRIES", "20")))
    parser.add_argument("--api_timeout", type=float, default=float(os.environ.get("API_TIMEOUT", "1200")))
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "0")))
    args = parser.parse_args()

    for required, name in [
        (args.gemini_api_key,  "--gemini_api_key  / GEMINI_VL_API_KEY"),
        (args.gemini_image_api_key, "--gemini_image_api_key / GEMINI_IMAGE_API_KEY"),
        (args.qwen_api_key,    "--qwen_api_key    / OPENAI_API_KEY"),
    ]:
        if not required:
            parser.error(f"missing: {name}")

    pipe = Stage1Pipeline(
        parquet=args.parquet,
        output_dir=args.output_dir,
        gemini_endpoint=args.gemini_endpoint,
        gemini_api_key=args.gemini_api_key,
        gemini_model=args.gemini_model,
        qwen_api_base=args.qwen_api_base,
        qwen_api_key=args.qwen_api_key,
        qwen_model=args.qwen_model,
        gemini_image_endpoint=args.gemini_image_endpoint,
        gemini_image_api_key=args.gemini_image_api_key,
        max_concurrent=args.max_concurrent,
        n_edits_per_source=args.n_edits_per_source,
        max_retries=args.max_retries,
        api_timeout=args.api_timeout,
        resume=not args.no_resume,
        limit=args.limit,
    )
    asyncio.run(pipe.run())


if __name__ == "__main__":
    import os
    main()
