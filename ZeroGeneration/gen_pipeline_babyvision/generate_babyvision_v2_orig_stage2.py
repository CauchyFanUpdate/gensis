#!/usr/bin/env python3
"""
BabyVision v2 — Original-Image Diversification (1:5)

Per source question (from parquet), operating on ORIGINAL images only:
  Call 1  [Gemini VL]   → frozen_predicate + axis_table analysis
                           is_editable=false → skip
  Selector [Qwen VL]    → 5 axis valuations (sees ORIGINAL image)
  ↓ per plan (parallel):
    Call 2b [Qwen text]  → diversification edit prose
    Call 3b [GPT-Image-2]→ 1 diversified image from ORIGINAL image
    Verifier [Qwen VL]   → 4 yes/no checks
    Call 4  [Gemini VL]  → CoT anchored on ORIGINAL answer

答案始终与原题保持一致（不经过 Stage 1 answer-changing 步骤）。

输出:
  <output_dir>/stage2_samples.jsonl
  <output_dir>/stage2_images/{sample_key}.png
  <output_dir>/stage2_selector_cache.jsonl

每 source 题 → 5 张多样化图（1:5 ratio）
"""

import argparse
import asyncio
import base64
import json
import os
import re
import time
import traceback
from pathlib import Path
from typing import Optional

from _v5_common import (
    AXIS_TIER_DEFAULTS, NATURAL_CAPS, READABILITY_TEMPLATES,
    TEST_POINT_FROZEN_PRIORS, UNIVERSAL_AXES,
    chat_gemini_vl, chat_qwen,
    extract_json, get_image_bytes, gpt_image_edit,
    JsonlWriter, jsonl_iter, load_done_keys, load_parquet, log,
    make_clients, prepare_image, qwen_text_messages, qwen_vl_messages,
)


# ── Call 1 (reused from stage 1): Gemini VL analysis ──────────────────────

OPS = ("remove", "add", "modify_property", "substitute_category", "context_change")
EFFECT_TYPES = ("delta", "literal_replace", "flip_boolean", "categorical_shift")
STRUCTURAL_PATTERNS = (
    "dual_value_disambiguation", "real_vs_depicted", "partial_visibility",
    "category_filtering", "lookalike_confusion", "spatial_ordering",
    "composition_state", "text_in_context", "other",
)


def _build_call1_params(question: str, answer: str) -> tuple[str, str]:
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

        "## Pipeline context\n\n"
        "This pipeline applies COSMETIC DIVERSIFICATION ONLY — the question and answer stay UNCHANGED. "
        "Your `frozen_predicate`, `question_template`, `answer_template`, and `axis_table` drive diversification. "
        "All answer-irrelevant visual axes (background, surface, color, lighting, camera, ...) will be "
        "pushed to aggressive new values while keeping the test point intact.\n\n"

        "## Core framework: 5 ops × 4 answer_effect_types\n\n"
        "Five edit operations cover all reasonable image edits:\n"
        "  - remove(obj), add(obj_like, location), modify_property(obj, property, new_value),\n"
        "    substitute_category(from, to), context_change(attribute, value)\n\n"

        "## Test point analysis\n\n"
        "Identify which of these 8 structural patterns applies:\n"
        "  - dual_value_disambiguation, real_vs_depicted, partial_visibility, category_filtering,\n"
        "    lookalike_confusion, spatial_ordering, composition_state, text_in_context, other\n\n"

        "## Stage 2 schema — frozen_predicate + slots + axis_table\n\n"
        "Stage 2 needs to know (a) what is FROZEN (must remain pixel-true) and (b) what is FREE.\n\n"

        "### A. frozen_predicate\n\n"
        "  - `answer_function`: predicate in math-like notation.\n"
        "  - `answer_value`: the actual answer value.\n"
        "  - `must_preserve`: 1-3 short bullets the editor must NOT change.\n"
        "  - `verifier_checks`: 4 yes/no questions for Qwen-VL on the edited image:\n"
        "      class_check / relation_check / readability_check / natural_check\n"
        "  - `slots`: dict of slot definitions with compatible_classes (≥5) and slot_replaceability.\n\n"

        "### B. question_template and answer_template\n\n"
        "  - question_template: question with slot placeholders (<S>, <S_plural>, <SURF>, etc.)\n"
        "  - answer_template: usually <N> or literal answer string.\n\n"

        "### C. axis_table — 19 universal visual axes\n\n"
        "For EACH of the 19 axes, classify as free/frozen/depends with aggressive_value or freeze_constraint.\n"
        f"{axes_block}\n\n"
        f"Test-point frozen-axes priors:\n{test_point_priors_block}\n\n"
        f"Readability check templates:\n{readability_block}\n\n"

        "## feasibility_assessment\n\n"
        "For this pipeline we only need `is_editable` to gate whether a clean frozen_predicate can be "
        "constructed. Set `is_editable=false` for complex puzzles, dense charts, or images where "
        "the test point cannot be cleanly described.\n\n"

        "Return ONLY valid JSON, no markdown fences, no commentary."
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
        "Produce JSON with EXACTLY these fields:\n\n"
        "{\n"
        '  "image_type": "natural_photo | diagram | chart | abstract_puzzle",\n'
        '  "scene_description": "<1-2 sentence natural caption>",\n'
        '  "test_point_analysis": {\n'
        '    "core_challenge": "<one sentence>",\n'
        f'    "structural_pattern": "<one of: {" | ".join(STRUCTURAL_PATTERNS)}>",\n'
        '    "must_be_described_in_edit": ["<element #1>"],\n'
        '    "edit_must_operate_within": "<one sentence constraint>"\n'
        '  },\n'
        '  "disambiguators_to_preserve": [\n'
        '    {"feature": "<visual feature>", "scope": "<which objects>"}\n'
        '  ],\n'
        '  "feasibility_assessment": {\n'
        '    "is_editable": true,\n'
        '    "skip_reason": ""\n'
        '  },\n'
        '  "feasible_edits": [],\n'
        '  "frozen_predicate": {\n'
        '    "answer_function": "<predicate string>",\n'
        '    "answer_value": "<actual answer>",\n'
        '    "must_preserve": ["<bullet 1>"],\n'
        '    "verifier_checks": {\n'
        '      "class_check":       "<yes/no Q with slot placeholders>",\n'
        '      "relation_check":    "<yes/no Q with slot placeholders>",\n'
        '      "readability_check": "<yes/no Q from readability templates>",\n'
        '      "natural_check":     "Is this a plausible real photograph (not illustration / surreal)?"\n'
        '    },\n'
        '    "slots": {\n'
        '      "S": {\n'
        '        "role": "<short noun phrase>",\n'
        '        "shape_constraint": "<visual shape>",\n'
        '        "default_class_singular": "<e.g., wine bottle>",\n'
        '        "default_class_plural":   "<e.g., wine bottles>",\n'
        '        "compatible_classes": [{"singular": "...", "plural": "..."}, ...],\n'
        '        "slot_replaceability": "free | shape_only | locked"\n'
        '      }\n'
        '    }\n'
        '  },\n'
        '  "question_template": "<question with placeholders>",\n'
        '  "answer_template":   "<usually <N> or literal string>",\n'
        '  "axis_table": [\n'
        f'{axis_schema_lines}\n'
        '  ]\n'
        "}\n\n"
        "Constraints:\n"
        "  - feasible_edits: leave empty [] (this pipeline does NOT change the answer).\n"
        "  - frozen_predicate.slots: ≥5 compatible_classes unless slot_replaceability=='locked'.\n"
        "  - axis_table MUST contain ALL 19 axes. Each free/depends entry needs aggressive_value (≥3 options).\n"
        "  - If is_editable=false, frozen_predicate/question_template/answer_template/axis_table can be empty.\n"
        "  - Output ONLY the JSON object."
    )
    return system, user_text


def _validate_call1_schema(call1: dict) -> tuple[bool, str]:
    """Validate Call 1 output has the required Stage 2 fields."""
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


# ── Selector helpers (identical to stage 2) ────────────────────────────────

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
            lines.append(f"    <{sname}>  numeric/value={sdef.get('value', '?')!r}  frozen={sdef.get('frozen', True)}")
    return "\n".join(lines) if lines else "    (no slots)"


def _axis_table_brief(axis_table: list) -> str:
    lines = []
    for entry in axis_table or []:
        if not isinstance(entry, dict):
            continue
        ax = entry.get("axis", "?")
        sf = entry.get("safety", "?")
        if sf == "frozen":
            lines.append(f"  - {ax}  [FROZEN]  must preserve: {entry.get('freeze_constraint', '')!r}")
        else:
            lines.append(
                f"  - {ax}  [{sf.upper()}]  aggressive options: {entry.get('aggressive_value', '')!r}"
            )
    return "\n".join(lines) if lines else "  (empty)"


def _build_selector_prompt(call1: dict, n_prompts: int) -> tuple[str, str]:
    system = (
        "You are an image augmentation strategy planner. You receive an original image "
        "plus a per-image analysis (frozen_predicate + axis_table). Your job is to "
        "produce N visually-distinct edit plans. Each plan picks one concrete value for "
        "every FREE axis, plus one slot-class for each replaceable slot, such that:\n\n"
        "  HARD constraints:\n"
        "    1. FROZEN axes must NOT be touched.\n"
        "    2. The frozen_predicate's verifier_checks must still hold true after editing.\n"
        "    3. Stay strictly within the natural-photo distribution. No surreal effects, "
        "no illustrations, no extreme color filters.\n"
        "    4. For slots with replaceability=='locked', do NOT change the slot class.\n\n"
        "  OPTIMIZATION:\n"
        "    A. Each plan should be MAXIMALLY AGGRESSIVE on every free axis.\n"
        "    B. INTER-PLAN diversity: across N plans, choose DIFFERENT concrete values for "
        "each axis so plans cover N distant points in axis space.\n"
        "    C. INTRA-PLAN coherence: a single plan should describe a coherent scene.\n\n"
        "Return ONLY valid JSON, no markdown fences, no commentary. Schema in the user message."
    )

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
        f"## axis_table (FROZEN axes are off-limits; FREE/DEPENDS axes are your variables)\n"
        f"{_axis_table_brief(call1.get('axis_table', []))}\n\n"
        f"## Task: produce {n_prompts} maximally diverse edit plans.\n\n"
        f"For each plan:\n"
        f"  - slot_assignments: pick a compatible_class for each replaceable class slot.\n"
        f"  - axis_values: pick a concrete value for every FREE axis.\n"
        f"  - diversity_signature: 3-5 short tags.\n"
        f"  - rationale: one short sentence.\n\n"
        f"Output JSON with EXACTLY this schema:\n"
        f"{{\n"
        f'  "plans": [\n'
        f'    {{\n'
        f'      "plan_id": 1,\n'
        f'      "slot_assignments": {{"S": {{"singular": "ceramic vase", "plural": "ceramic vases"}}}},\n'
        f'      "axis_values": {{"background_setting": "minimalist white studio", "...": "..."}},\n'
        f'      "diversity_signature": ["bg_studio", "lighting_cold"],\n'
        f'      "rationale": "<one short sentence>"\n'
        f'    }}\n'
        f'  ]\n'
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
        sig = [str(x) for x in sig if x]
        out.append({
            "plan_id":             int(p.get("plan_id", len(out) + 1)),
            "slot_assignments":    norm_sa,
            "axis_values":         norm_av,
            "diversity_signature": sig,
            "rationale":           str(p.get("rationale", "")),
        })
    return out[:n_prompts] if len(out) > n_prompts else out


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


# ── Selector disk cache ────────────────────────────────────────────────────

class SelectorCache:
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
                k = rec.get("source_key")
                plans = rec.get("plans")
                if k and isinstance(plans, list):
                    self._mem[k] = plans
        except Exception as e:
            log(f"selector cache load warning: {e!r}", "stage2_orig")

    def get(self, source_key: str) -> Optional[list[dict]]:
        return self._mem.get(source_key)

    async def put(self, source_key: str, plans: list[dict]) -> None:
        async with self._lock:
            self._mem[source_key] = plans
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "source_key": source_key,
                    "plans":      plans,
                    "created_at": int(time.time()),
                }, ensure_ascii=False) + "\n")


# ── Call 1 result disk cache ───────────────────────────────────────────────

class Call1Cache:
    def __init__(self, path: Path):
        self.path = path
        self._mem: dict[str, dict] = {}
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            for line in open(self.path, encoding="utf-8"):
                rec = json.loads(line)
                k = rec.get("source_id")
                call1 = rec.get("call1")
                if k and isinstance(call1, dict):
                    self._mem[k] = call1
        except Exception as e:
            log(f"call1 cache load warning: {e!r}", "stage2_orig")

    def get(self, source_id: str) -> Optional[dict]:
        return self._mem.get(source_id)

    async def put(self, source_id: str, call1: dict) -> None:
        async with self._lock:
            self._mem[source_id] = call1
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "source_id":  source_id,
                    "call1":      call1,
                    "created_at": int(time.time()),
                }, ensure_ascii=False) + "\n")


# ── Call 2b: Qwen text — diversification prose ────────────────────────────

_PROSE_BANNED_WORDS = (
    "subtle", "subtly", "slightly", "a touch of", "a hint of", "hints of",
    "gentle", "softly", "lightly", "mild", "mildly",
)
_PROSE_PUSH_WORDS = (
    "completely", "entirely", "wholly", "dramatically", "fully", "thoroughly", "transformed into",
)


def _build_call2b_prompt(call1: dict, plan: dict) -> tuple[str, str]:
    fp = call1.get("frozen_predicate", {}) or {}
    must_preserve = fp.get("must_preserve", []) or []
    answer_function = fp.get("answer_function", "")

    system = (
        "You write image editing instructions for an AI image editor (GPT-Image-2). "
        "Your output is a SINGLE FLOWING PARAGRAPH of natural English prose, 150-260 words.\n\n"
        "Format rules:\n"
        "  - NO bullet points, NO labels, NO section headers, NO line breaks within the paragraph.\n"
        "  - One continuous description an art director would speak aloud.\n"
        "  - Every FREE axis must be ADDRESSED with at least one concrete adjective or noun phrase.\n\n"
        "STRENGTH RULES:\n"
        f"  - Strictly avoid: {', '.join(_PROSE_BANNED_WORDS)}\n"
        f"  - Prefer amplification: {', '.join(_PROSE_PUSH_WORDS)}\n"
        "  - The result must look obviously different from the input image.\n\n"
        "NATURAL-PHOTO RULE:\n"
        "  - Must look like a real photograph. NO illustration / cartoon / surreal style.\n"
        "  - NO dramatic single-color stage lighting, no extreme color filters.\n"
        "  - NO surreal weather. Outdoor weather: clear / overcast / golden hour only.\n\n"
        "FROZEN PREDICATE:\n"
        "  - Keep the frozen_predicate intact. Paraphrase must_preserve constraints into the "
        "closing sentence using positive language ('the scene still shows ...').\n"
        "  - Do NOT mention the question or the answer in the prose.\n"
    )

    plan_lines = []
    for sname, sval in (plan.get("slot_assignments") or {}).items():
        if isinstance(sval, dict):
            plan_lines.append(f"  • slot <{sname}>: {sval.get('plural', sval.get('singular', '?'))}")
    for ax, val in (plan.get("axis_values") or {}).items():
        plan_lines.append(f"  • axis {ax}: {val}")
    plan_block = "\n".join(plan_lines) if plan_lines else "  (no axis instructions)"
    must_block = "\n".join(f"  - {x}" for x in must_preserve) if must_preserve else "  (none)"

    user_text = (
        f"BASE SCENE (original image — your prose instructs the editor to repaint with new axis values):\n"
        f"  {call1.get('scene_description', '')}\n\n"
        f"AXIS VALUES + SLOT CLASSES TO APPLY:\n"
        f"{plan_block}\n\n"
        f"FROZEN PREDICATE — MUST be preserved (answer logic = {answer_function!r}):\n"
        f"{must_block}\n\n"
        f"Write the single flowing paragraph. Open with new background_setting and surface, "
        f"cover every FREE axis concretely, close with must_preserve constraints paraphrased.\n"
        f"Output ONLY the paragraph, no other text."
    )
    return system, user_text


# ── Verifier ───────────────────────────────────────────────────────────────

def _build_verifier_prompt(verifier_checks_instantiated: dict) -> tuple[str, str]:
    system = (
        "You are a strict visual verifier. You will be shown an image and 4 short yes/no questions. "
        "Answer each one with a single word (yes or no). Do not explain. Do not hedge. "
        "Use only the visual evidence in the image. If a question's premise can't be evaluated, answer 'no'."
    )
    qs = verifier_checks_instantiated
    user_text = (
        "Look at the image and answer each question with one word: yes or no.\n\n"
        f"Q1 (CLASS):       {qs.get('class_check', '')}\n"
        f"Q2 (RELATION):    {qs.get('relation_check', '')}\n"
        f"Q3 (READABILITY): {qs.get('readability_check', '')}\n"
        f"Q4 (NATURAL):     {qs.get('natural_check', '')}\n\n"
        "Output EXACTLY this format (no extra text):\n"
        "A1: <yes|no>\n"
        "A2: <yes|no>\n"
        "A3: <yes|no>\n"
        "A4: <yes|no>\n"
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


# ── Call 4: Gemini VL CoT ──────────────────────────────────────────────────

def _build_call4_prompt(call1: dict, final_question: str, final_answer: str) -> tuple[str, str]:
    system = (
        "You are a careful visual reasoning assistant. You produce a brief, accurate chain of thought "
        "for a visual question, ending with the final answer.\n\n"
        "Style requirements:\n"
        "  - Natural, flowing English (not numbered steps, not bullet lists)\n"
        "  - Concise (3-6 short sentences total)\n"
        "  - Accurate — your reasoning must lead to the supplied final answer\n"
        "  - End with: **Final Answer: <answer>**\n"
    )
    disambs = call1.get("disambiguators_to_preserve", [])
    test_point = call1.get("test_point_analysis", {})
    soft_hints = ""
    if disambs:
        soft_hints += "Helpful hints (use only if natural to the reasoning):\n"
        for d in disambs[:3]:
            soft_hints += f"  - {d.get('feature', '')}\n"
    if test_point.get("core_challenge"):
        soft_hints += f"Note: {test_point['core_challenge']}\n"

    user_text = (
        f"Question: {final_question}\n\n"
        "Look at the image and reason briefly to the answer.\n\n"
        + soft_hints + "\n"
        f"End your response with: **Final Answer: {final_answer}**"
    )
    return system, user_text


# ── Main pipeline ──────────────────────────────────────────────────────────

class OrigStage2Pipeline:
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
        gpt_image_endpoint: str,
        gpt_image_api_key: str,
        max_concurrent: int = 64,
        max_retries: int = 20,
        api_timeout: float = 1200.0,
        resume: bool = True,
        limit: int = 0,
        n_prompts_per_source: int = 5,
    ):
        self.parquet = parquet
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "stage2_images"
        self.output_jsonl = self.output_dir / "stage2_samples.jsonl"
        self.selector_cache_path = self.output_dir / "stage2_selector_cache.jsonl"
        self.call1_cache_path = self.output_dir / "stage2_call1_cache.jsonl"

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
        self.n_prompts = n_prompts_per_source

        self._writer: Optional[JsonlWriter] = None
        self._done_keys: set[str] = set()
        self._selector_cache: Optional[SelectorCache] = None
        self._call1_cache: Optional[Call1Cache] = None
        self._inner = None
        self._proxy = None

    async def _process_one_plan_image(
        self,
        source_id: str, question: str, original_answer: str,
        call1: dict, plan: dict, prompt_idx: int,
        stage2_bytes: bytes, edit_prose: str,
    ) -> None:
        sample_key = f"{source_id}::p{prompt_idx}::s0"
        if sample_key in self._done_keys:
            return

        img_path = self.images_dir / f"{sample_key}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(stage2_bytes)
        stage2_b64 = base64.b64encode(stage2_bytes).decode()

        # Instantiate slots → final question / answer / verifier_checks
        slot_assignments = plan.get("slot_assignments", {}) or {}
        fp = call1.get("frozen_predicate", {}) or {}
        slot_values: dict = dict(slot_assignments)
        # Inject original answer into numeric/value slots
        for sname, sdef in (fp.get("slots") or {}).items():
            if isinstance(sdef, dict) and "value" in sdef and sname not in slot_values:
                slot_values[sname] = original_answer or sdef.get("value", "")

        question_template = call1.get("question_template", "") or question
        answer_template   = call1.get("answer_template", "") or original_answer
        final_question = _instantiate_template(question_template, slot_values) or question
        final_answer   = _instantiate_template(answer_template,   slot_values) or original_answer

        verifier_checks_raw = (fp.get("verifier_checks") or {})
        verifier_checks_inst = {
            k: _instantiate_template(v, slot_values) for k, v in verifier_checks_raw.items()
        }
        if not verifier_checks_inst.get("natural_check"):
            verifier_checks_inst["natural_check"] = (
                "Is this a plausible real photograph (not illustration / surreal / cartoon)?"
            )

        # Verifier
        try:
            sys_p, usr_p = _build_verifier_prompt(verifier_checks_inst)
            ver_raw = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                qwen_vl_messages(sys_p, [(stage2_b64, "image/png")], usr_p, no_think=True),
                max_tokens=1536, temperature=0.0,
                max_retries=self.max_retries, log_prefix=f"verify/{sample_key}",
            )
            verifier = _parse_verifier_output(ver_raw)
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id, "stage": 2,
                "status": "failed", "reject_reason": "verifier_failed",
                "stage2_image_path": str(img_path), "edit_prose": edit_prose, "plan": plan,
                "error": f"{type(e).__name__}: {e!r}", "created_at": int(time.time()),
            })
            return

        if not verifier["all_pass"]:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id, "stage": 2,
                "status": "rejected", "reject_reason": "verifier_check_failed",
                "stage2_image_path": str(img_path), "edit_prose": edit_prose, "plan": plan,
                "verifier_checks_instantiated": verifier_checks_inst,
                "verifier_answers": {k: verifier[k] for k in ("a1", "a2", "a3", "a4")},
                "verifier_raw": verifier["raw"][:1000],
                "created_at": int(time.time()),
            })
            return

        # Call 4: Gemini CoT anchored on original answer
        try:
            sys_p, usr_p = _build_call4_prompt(call1, final_question, final_answer)
            cot = await chat_gemini_vl(
                self._inner, self.gemini_endpoint, self.gemini_api_key,
                sys_p, [(stage2_b64, "image/png")], usr_p,
                max_tokens=4096, temperature=0.6, disable_thinking=True,
                max_retries=self.max_retries, log_prefix=f"call4/{sample_key}",
            )
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id, "stage": 2,
                "status": "failed", "reject_reason": "cot_failed",
                "stage2_image_path": str(img_path), "edit_prose": edit_prose, "plan": plan,
                "final_question": final_question, "final_answer": final_answer,
                "error": f"{type(e).__name__}: {e!r}", "created_at": int(time.time()),
            })
            return

        if "**Final Answer:" not in cot:
            cot = cot.rstrip() + f"\n\n**Final Answer: {final_answer}**"

        await self._writer.write({
            "sample_key": sample_key, "source_id": source_id, "stage": 2,
            "quality_pass": True,
            "original_question": question, "original_answer": original_answer,
            "final_question": final_question, "final_answer": final_answer,
            "cot": cot,
            "stage2_image_path": str(img_path),
            "edit_prose": edit_prose, "plan": plan,
            "verifier_checks_instantiated": verifier_checks_inst,
            "verifier_answers": {k: verifier[k] for k in ("a1", "a2", "a3", "a4")},
            "prompt_idx": prompt_idx,
            "created_at": int(time.time()),
        })
        log(f"  [{sample_key}] ✓ orig-stage2 done | answer={final_answer}", "stage2_orig")

    async def _process_plan(
        self,
        source_id: str, question: str, original_answer: str,
        call1: dict, plan: dict,
        gpt_bytes: bytes, gpt_mime: str,
    ) -> None:
        prompt_idx = int(plan.get("plan_id", 0))

        sample_key = f"{source_id}::p{prompt_idx}::s0"
        if sample_key in self._done_keys:
            log(f"  [{sample_key}] resume: already done, skip", "stage2_orig")
            return

        # Call 2b: prose
        try:
            sys_p, usr_p = _build_call2b_prompt(call1, plan)
            edit_prose = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                qwen_text_messages(sys_p, usr_p, no_think=True),
                max_tokens=4096, temperature=0.6,
                max_retries=self.max_retries, log_prefix=f"qwen2b/{source_id}::p{prompt_idx}",
            )
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id, "stage": 2,
                "status": "failed", "reject_reason": "call2b_failed",
                "plan": plan, "error": f"{type(e).__name__}: {e!r}",
                "created_at": int(time.time()),
            })
            return

        # Call 3b: GPT-Image-2 on ORIGINAL image
        try:
            imgs = await gpt_image_edit(
                self._proxy, self.gpt_image_endpoint, self.gpt_image_api_key,
                gpt_bytes, edit_prose, n=1, image_mime=gpt_mime,
                max_retries=self.max_retries, log_prefix=f"gpt3b/{source_id}::p{prompt_idx}",
            )
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id, "stage": 2,
                "status": "failed", "reject_reason": "call3b_gpt_image_failed",
                "edit_prose": edit_prose, "plan": plan,
                "error": f"{type(e).__name__}: {e!r}", "created_at": int(time.time()),
            })
            return

        if not imgs:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id, "stage": 2,
                "status": "failed", "reject_reason": "call3b_no_image",
                "edit_prose": edit_prose, "plan": plan, "created_at": int(time.time()),
            })
            return

        await self._process_one_plan_image(
            source_id, question, original_answer, call1, plan, prompt_idx, imgs[0], edit_prose,
        )

    async def _process_one(self, row: dict) -> None:
        source_id = str(row.get("question_id") or row.get("taskId", ""))
        question = row.get("question_text") or row.get("question", "")
        raw_ans = row.get("question_answer")
        if raw_ans is None:
            raw_ans = row.get("blankAns")
        if raw_ans is None or (isinstance(raw_ans, float) and __import__("math").isnan(raw_ans)):
            raw_choice = row.get("choiceAns")
            raw_ans = int(raw_choice) if raw_choice is not None and not (
                isinstance(raw_choice, float) and __import__("math").isnan(raw_choice)
            ) else ""
        original_answer = str(raw_ans) if raw_ans != "" else ""

        async with self.semaphore:
            try:
                # Resume: skip if all plans done or permanently skipped
                if f"{source_id}::skipped" in self._done_keys:
                    log(f"  [{source_id}] resume: previously skipped", "stage2_orig")
                    return
                expected_all = [f"{source_id}::p{p}::s0" for p in range(1, self.n_prompts + 1)]
                if all(k in self._done_keys for k in expected_all):
                    log(f"  [{source_id}] resume: all {self.n_prompts} plans done, skip", "stage2_orig")
                    return

                # Image prep
                raw_bytes = get_image_bytes(row)
                if not raw_bytes:
                    log(f"  [{source_id}] no image, skip", "stage2_orig")
                    return
                hq_bytes, hq_mime = prepare_image(raw_bytes, max_side=1568, fmt="PNG")
                gpt_bytes, gpt_mime = prepare_image(raw_bytes, max_side=1024, fmt="JPEG")
                gemini_b64 = base64.b64encode(hq_bytes).decode()

                # Call 1 (Gemini VL analysis) — try cache first
                call1 = self._call1_cache.get(source_id) if self._call1_cache else None
                if call1 is None:
                    log(f"  [{source_id}] Call 1 (Gemini analysis)...", "stage2_orig")
                    sys_p, usr_p = _build_call1_params(question, original_answer)
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
                            "source_id": source_id, "stage": 2, "status": "failed",
                            "reject_reason": "call1_json_parse_error",
                            "original_question": question, "original_answer": original_answer,
                            "error": f"{type(e).__name__}: {e!r}",
                            "raw_call1": call1_raw[:2000], "created_at": int(time.time()),
                        })
                        return
                    ok, err = _validate_call1_schema(call1)
                    if not ok:
                        await self._writer.write({
                            "sample_key": f"{source_id}::call1_schema_invalid",
                            "source_id": source_id, "stage": 2, "status": "failed",
                            "reject_reason": "call1_schema_invalid",
                            "original_question": question, "original_answer": original_answer,
                            "error": err, "call1_full": call1, "created_at": int(time.time()),
                        })
                        log(f"  [{source_id}] schema invalid: {err}", "stage2_orig")
                        return
                    # Cache valid call1
                    if self._call1_cache:
                        await self._call1_cache.put(source_id, call1)

                # Gate: is_editable
                feas = call1.get("feasibility_assessment", {})
                if not feas.get("is_editable", False):
                    await self._writer.write({
                        "sample_key": f"{source_id}::skipped",
                        "source_id": source_id, "stage": 2, "status": "skipped",
                        "reject_reason": "is_editable_false",
                        "skip_reason": feas.get("skip_reason", ""),
                        "original_question": question, "original_answer": original_answer,
                        "created_at": int(time.time()),
                    })
                    log(f"  [{source_id}] is_editable=false, skip", "stage2_orig")
                    return

                # Selector — try cache first
                plans = self._selector_cache.get(source_id) if self._selector_cache else None
                if plans is None:
                    log(f"  [{source_id}] Selector (Qwen VL)...", "stage2_orig")
                    sys_p, usr_p = _build_selector_prompt(call1, self.n_prompts)
                    try:
                        sel_text = await chat_qwen(
                            self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                            qwen_vl_messages(sys_p, [(gemini_b64, hq_mime)], usr_p),
                            max_tokens=8192, temperature=0.5,
                            max_retries=self.max_retries, log_prefix=f"selector/{source_id}",
                        )
                        plans = _parse_selector_output(sel_text, self.n_prompts)
                    except Exception as e:
                        log(f"  [{source_id}] selector failed: {e!r}", "stage2_orig")
                        plans = []
                    if plans and self._selector_cache:
                        await self._selector_cache.put(source_id, plans)

                if not plans:
                    log(f"  [{source_id}] selector returned no plans, skip", "stage2_orig")
                    return

                # Renumber plans 1..N
                for i, p in enumerate(plans, start=1):
                    p["plan_id"] = i

                # Process all plans in parallel
                tasks = [
                    self._process_plan(
                        source_id, question, original_answer, call1, p, gpt_bytes, gpt_mime,
                    )
                    for p in plans
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                log(f"  [{source_id}] UNEXPECTED ERROR: {type(e).__name__}: {e!r}", "stage2_orig")
                log(traceback.format_exc(), "stage2_orig")
                await self._writer.write({
                    "sample_key": f"{source_id}::unexpected_error",
                    "source_id": source_id, "stage": 2, "status": "failed",
                    "reject_reason": "unexpected_error",
                    "original_question": question, "original_answer": original_answer,
                    "error": f"{type(e).__name__}: {e!r}", "created_at": int(time.time()),
                })

    async def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._writer = JsonlWriter(self.output_jsonl)
        self._selector_cache = SelectorCache(self.selector_cache_path)
        self._call1_cache = Call1Cache(self.call1_cache_path)
        log(f"selector cache: {len(self._selector_cache._mem)} entries", "stage2_orig")
        log(f"call1 cache:    {len(self._call1_cache._mem)} entries", "stage2_orig")

        if self.resume:
            self._done_keys = load_done_keys(self.output_jsonl)
            log(f"resume: {len(self._done_keys)} sample_keys already done", "stage2_orig")

        log(f"loading parquet: {self.parquet}", "stage2_orig")
        rows = load_parquet(self.parquet)
        log(f"loaded {len(rows)} rows", "stage2_orig")
        if self.limit > 0:
            rows = rows[:self.limit]
            log(f"limit to {self.limit}", "stage2_orig")

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
            f"orig-stage2 done. total: {total} | passed: {passed} | "
            f"rejected: {rejected} | failed: {failed} | skipped: {skipped}",
            "stage2_orig",
        )
        log(f"output jsonl: {self.output_jsonl}", "stage2_orig")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="BabyVision v2 — Original-Image Diversification (Stage 2 on original, 1:5)"
    )
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--output_dir", default="output_babyvision_v2_orig")
    # Gemini VL
    parser.add_argument("--gemini_endpoint", default=os.environ.get("GEMINI_VL_ENDPOINT", "https://runway.devops.rednote.life/openai/google/v1:generateContent"))
    parser.add_argument("--gemini_api_key", default=os.environ.get("GEMINI_VL_API_KEY", ""))
    parser.add_argument("--gemini_model", default=os.environ.get("GEMINI_VL_MODEL", "gemini-3.1-pro-preview"))
    # Qwen
    parser.add_argument("--qwen_api_base", default=os.environ.get("QWEN_API_BASE", "https://runway.devops.xiaohongshu.com/openai/qwen/v1"))
    parser.add_argument("--qwen_api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--qwen_model", default=os.environ.get("QWEN_MODEL", "qwen3.6-plus"))
    # GPT-Image-2
    parser.add_argument("--gpt_image_endpoint", default=os.environ.get("GPT_IMAGE_ENDPOINT", "https://api.gpt.ge/v1"))
    parser.add_argument("--gpt_image_api_key", default=os.environ.get("GPT_IMAGE_API_KEY", ""))
    # Run
    parser.add_argument("--max_concurrent", type=int, default=int(os.environ.get("MAX_CONCURRENT", "64")))
    parser.add_argument("--n_prompts_per_source", type=int, default=5)
    parser.add_argument("--max_retries", type=int, default=int(os.environ.get("MAX_RETRIES", "20")))
    parser.add_argument("--api_timeout", type=float, default=float(os.environ.get("API_TIMEOUT", "1200")))
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "0")))
    args = parser.parse_args()

    for required, name in [
        (args.gemini_api_key,    "--gemini_api_key  / GEMINI_VL_API_KEY"),
        (args.gpt_image_api_key, "--gpt_image_api_key / GPT_IMAGE_API_KEY"),
        (args.qwen_api_key,      "--qwen_api_key    / OPENAI_API_KEY"),
    ]:
        if not required:
            parser.error(f"missing: {name}")

    pipe = OrigStage2Pipeline(
        parquet=args.parquet,
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
        n_prompts_per_source=args.n_prompts_per_source,
        max_retries=args.max_retries,
        api_timeout=args.api_timeout,
        resume=not args.no_resume,
        limit=args.limit,
    )
    asyncio.run(pipe.run())


if __name__ == "__main__":
    main()
