#!/usr/bin/env python3
"""
方案二 — Stage 2 直接多样性变体：对原始 MMMU 图片进行 5 个外观变体（1:5 比例）

与 edit_mmmu/generate_samples_v5_stage2.py 的核心差异：
  - 输入：直接读取 MMMU manifest（无需 Stage 1 中间产物）
  - 答案保持原始 MMMU 答案不变（无 Modifier 验证步骤）
  - Call 1 用于获取 frozen_predicate + axis_table（答案多样性信息）
  - Selector 生成 n_target = 5 个轴变体计划（1:5 比例，固定值）
  - 每个计划：Call 2b → GPT-Image-2 → Verifier → Call 4 (CoT)
  - sample_key 格式：{id}::p{N}::s0

无需 Stage 1.5（答案不变，原选项保持有效）
需要 Stage 3 lite（清洗 CoT + 格式化 → 训练 JSONL）
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
from typing import Optional

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


# ── Type-specific selector addenda (copied from edit_mmmu/generate_samples_v5_stage2.py) ──

_SELECTOR_TYPE_ADDENDA: dict[str, str] = {
    "table": (
        "\n## DIVERSIFICATION GUIDANCE FOR TABLES\n\n"
        "For table images, diversification means changing the VISUAL PRESENTATION while keeping\n"
        "all data values, row/column structure, and text content pixel-identical.\n\n"
        "Concrete dimensions to vary across plans (pick different values per plan):\n"
        "  - Color scheme: dark theme / light corporate / pastel / high-contrast / warm earth tones / cool blue-gray\n"
        "  - Font family: serif / sans-serif / monospace / rounded modern / condensed / handwritten casual\n"
        "  - Border style: thin hairline / thick double / no borders with alternating row shading / rounded-corner\n"
        "  - Row shading: zebra stripes / no shading / gradient rows / highlighted header only\n"
        "  - Background: plain white / textured paper / solid colored / gradient / subtle grid\n\n"
        "CRITICAL: Every cell value, every number, every label must remain EXACTLY the same.\n"
    ),
    "chart": (
        "\n## DIVERSIFICATION GUIDANCE FOR CHARTS/GRAPHS\n\n"
        "For chart images, diversification means changing the VISUAL STYLE while keeping\n"
        "all data values, axis ranges, labels, and legend text identical.\n\n"
        "Concrete dimensions to vary across plans:\n"
        "  - Color palette: warm / cool / monochrome / high-contrast / pastel / neon / earth tones\n"
        "  - Rendering style: flat 2D / 3D with shadows / gradient fills / hand-drawn sketch / minimalist\n"
        "  - Grid and axis style: minimal dotted / heavy solid / no grid / major only / dashed\n"
        "  - Background: white / light gray / dark mode / gradient / textured paper\n"
        "  - Legend position: top / bottom / right side / inline / none\n\n"
        "CRITICAL: Every data value, every axis label, every number must remain EXACTLY the same.\n"
    ),
    "diagram": (
        "\n## DIVERSIFICATION GUIDANCE FOR DIAGRAMS\n\n"
        "For diagram images, diversification means changing the VISUAL STYLE while keeping\n"
        "all labels, connections, node values, and structural layout identical.\n\n"
        "Concrete dimensions to vary:\n"
        "  - Color scheme: blue/gray professional / colorful rainbow / monochrome / earth tones / dark mode / pastel\n"
        "  - Line style: thin clean / thick bold / dashed / double / hand-drawn / dotted\n"
        "  - Node style: rounded rectangles / sharp rectangles / circles / hexagons / filled vs outlined\n"
        "  - Background: white / light grid / blueprint blue / dark / subtle texture\n\n"
        "CRITICAL: Every label, every connection, every node value must remain EXACTLY the same.\n"
    ),
    "medical": (
        "\n## DIVERSIFICATION GUIDANCE FOR MEDICAL IMAGES\n\n"
        "For medical images, diversification is limited to imaging parameters:\n"
        "  - Contrast/brightness: high / low / enhanced edges / normal clinical\n"
        "  - Color mapping: standard grayscale / inverted / false-color heat map / bone window / soft tissue window\n"
        "  - Annotation style: with measurement overlays / clean without annotations / different ruler styles\n\n"
        "CRITICAL: All diagnostic findings, anatomical structures, and measurements must remain identical.\n"
    ),
    "sheet_music": (
        "\n## DIVERSIFICATION GUIDANCE FOR SHEET MUSIC\n\n"
        "For sheet music, diversification means changing the engraving style:\n"
        "  - Ink color: black / dark blue / sepia brown / dark green\n"
        "  - Paper background: white / cream / aged parchment / light blue staff paper\n"
        "  - Staff line weight: thin hairline / standard / thick bold\n"
        "  - Spacing: compact dense layout / generous wide spacing\n\n"
        "CRITICAL: Every note position, accidental, dynamic marking must remain identical.\n"
    ),
}

_CALL2B_TYPE_ADDENDA: dict[str, str] = {
    "table": (
        "TABLE-SPECIFIC INSTRUCTIONS (very important):\n"
        "  - Describe a completely new visual theme for this table as if writing a CSS stylesheet in prose.\n"
        "  - Specify ALL of: background color, header row color, body row color/shading, border style and color,\n"
        "    font family, font size, text color, cell padding. Be extremely concrete.\n"
        "  - Every single cell value, number, and text label must be preserved EXACTLY.\n"
        "  - Do NOT describe the table content. Just describe the visual style.\n\n"
    ),
    "chart": (
        "CHART-SPECIFIC INSTRUCTIONS (very important):\n"
        "  - Describe a completely new visual theme as if restyling in a different plotting library.\n"
        "  - Specify ALL of: background color, bar/line/pie colors for each data series, grid line style,\n"
        "    axis label font and size, title font, legend style and position. Be extremely concrete.\n"
        "  - Every data value, axis label, and number must be preserved EXACTLY.\n"
        "  - Do NOT describe the data content. Just describe the visual style.\n\n"
    ),
    "diagram": (
        "DIAGRAM-SPECIFIC INSTRUCTIONS (very important):\n"
        "  - Describe a completely new visual theme as if applying a different drawing template.\n"
        "  - Specify: node fill color, node border style, connection line color and weight,\n"
        "    arrow head style, label font, background color. Be concrete.\n"
        "  - Every label, connection, and structural element must be preserved EXACTLY.\n\n"
    ),
    "medical": (
        "MEDICAL-SPECIFIC INSTRUCTIONS:\n"
        "  - Describe changes to imaging parameters (contrast, brightness, window/level, color mapping).\n"
        "  - Do not alter any diagnostic findings or anatomical structures.\n\n"
    ),
    "sheet_music": (
        "SHEET-MUSIC-SPECIFIC INSTRUCTIONS:\n"
        "  - Describe changes to engraving style: ink color, paper background, staff line weight, note spacing.\n"
        "  - Every note, rest, accidental, and dynamic marking must remain in the exact same position.\n\n"
    ),
}

_PROSE_BANNED_WORDS = ("subtle", "subtly", "slightly", "a touch of", "a hint of", "hints of", "gentle", "softly", "lightly", "mild", "mildly")
_PROSE_PUSH_WORDS = ("completely", "entirely", "wholly", "dramatically", "fully", "thoroughly", "transformed into")


# ── Call 1: get frozen_predicate + axis_table for diversification ──────────────
# (Same prompt as edit_mmmu/generate_samples_v5_stage1.py but we only NEED
# the diversification schema — feasible_edits will be ignored / not needed)

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
    readability_block = "\n".join(f"  - {tp}: {tmpl}" for tp, tmpl in READABILITY_TEMPLATES.items())
    test_point_priors_block = "\n".join(
        f"  - {tp}: typically frozen → {list(axes)}"
        for tp, axes in TEST_POINT_FROZEN_PRIORS.items()
    )

    system = (
        "You are an analytical assistant for an image editing data generation pipeline. "
        "Given an image and a visual question with its ground-truth answer, produce a structured "
        "analysis JSON that downstream code uses to design cosmetic diversification variants.\n\n"
        "## Pipeline context\n\n"
        "This pipeline (Stage 2 direct) applies COSMETIC diversification to original images. "
        "The goal is to produce 5 visually distinct variants of the same image (1:5 ratio) while keeping "
        "the answer UNCHANGED. Your frozen_predicate + axis_table drive this diversification.\n\n"
        "Unlike a standard Stage 1 pipeline, feasible_edits is NOT the main focus here. "
        "Please still fill in feasible_edits if they exist (for completeness), but the key output "
        "needed is: frozen_predicate, question_template, answer_template, axis_table.\n\n"
        "## Stage 2 schema — frozen_predicate + slots + axis_table\n\n"
        "You produce four interlocking pieces:\n"
        "  A. frozen_predicate — answer_function, answer_value, must_preserve, verifier_checks, slots\n"
        "  B. question_template — slot-templated question\n"
        "  C. answer_template — slot-templated answer string\n"
        "  D. axis_table — 19 universal visual axes classified as free/frozen/depends\n\n"
        "The 19 universal axes (with defaults):\n"
        f"{axes_block}\n\n"
        "Test-point → frozen-axes prior:\n"
        f"{test_point_priors_block}\n\n"
        "Readability check templates:\n"
        f"{readability_block}\n\n"
    )

    if image_type_group != "natural_photo":
        # _IMAGE_TYPE_ADDENDA defined later in this module (forward reference resolved at call time)
        import sys as _sys
        _mod = _sys.modules[__name__]
        system += getattr(_mod, "_IMAGE_TYPE_ADDENDA", {}).get(image_type_group, "")

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
        '  "scene_description": "<1-2 sentence caption>",\n'
        '  "question_analysis": {"answer_determinant": "...", "answer_source_region": "...", "current_answer_echo": "..."},\n'
        '  "test_point_analysis": {"core_challenge": "...", "structural_pattern": "...", "must_be_described_in_edit": [], "edit_must_operate_within": "..."},\n'
        '  "disambiguators_to_preserve": [{"feature": "...", "scope": "..."}],\n'
        '  "feasibility_assessment": {"is_editable": true, "skip_reason": ""},\n'
        '  "feasible_edits": [],\n'
        '  "frozen_predicate": {\n'
        '    "answer_function": "...",\n'
        '    "answer_value": "...",\n'
        '    "must_preserve": ["...", "..."],\n'
        '    "verifier_checks": {\n'
        '      "class_check": "...",\n'
        '      "relation_check": "...",\n'
        '      "readability_check": "...",\n'
        '      "natural_check": "Is this a plausible real photograph (not illustration / surreal)?"\n'
        "    },\n"
        '    "slots": {"S": {"role": "...", "shape_constraint": "...", "default_class_singular": "...", "default_class_plural": "...", "compatible_classes": [...], "slot_replaceability": "free"}}\n'
        "  },\n"
        '  "question_template": "<question with slot placeholders>",\n'
        '  "answer_template": "<usually \'<N>\' or literal>",\n'
        '  "axis_table": [\n'
        f"{axis_schema_lines}\n"
        "  ]\n"
        "}\n\n"
        "Constraints:\n"
        "  - axis_table MUST contain ALL 19 axes\n"
        "  - For free axes, aggressive_value MUST list ≥3 alternative concrete options separated by ' / '\n"
        "  - Output ONLY the JSON object"
    )
    return system, user_text


# Add image type addenda for Call 1 (copied from stage1)
_IMAGE_TYPE_ADDENDA: dict[str, str] = {
    "table": (
        "\n## IMAGE TYPE OVERRIDE: TABLE / SPREADSHEET\n\n"
        "This image is a rendered table — NOT a natural photograph.\n\n"
        "ALL physical-world axes MUST be `frozen` with freeze_constraint = \"not applicable to rendered table\".\n"
        "Only free styling axes: subject_color (color scheme), subject_material (border style), subject_specific_appearance (font/formatting).\n"
        'verifier_checks.natural_check: "Is this a legible, well-formatted data table (not corrupted or garbled)?"\n'
    ),
    "chart": (
        "\n## IMAGE TYPE OVERRIDE: CHART / GRAPH\n\n"
        "This image is a chart/graph — NOT a natural photograph.\n\n"
        "Freeze ALL physical-world axes with freeze_constraint = \"not applicable to rendered chart\".\n"
        "Free styling axes: subject_color (palette), subject_material (rendering style), subject_specific_appearance (grid/annotation).\n"
        'verifier_checks.natural_check: "Is this a legible, well-formatted chart with readable axes and data?"\n'
    ),
    "diagram": (
        "\n## IMAGE TYPE OVERRIDE: DIAGRAM / SCHEMATIC\n\n"
        "This image is a diagram — NOT a natural photograph.\n\n"
        'Freeze ALL physical-world axes with freeze_constraint = "not applicable to rendered diagram".\n'
        "Free styling axes: subject_color, subject_material, subject_specific_appearance.\n"
        'verifier_checks.natural_check: "Is this a legible, well-structured diagram with clear labels and connections?"\n'
    ),
    "medical": (
        "\n## IMAGE TYPE OVERRIDE: MEDICAL IMAGE\n\n"
        "This is a clinical/scientific medical image — NOT a natural photograph.\n"
        'verifier_checks.natural_check: "Is this a plausible medical image with adequate diagnostic quality?"\n'
    ),
    "sheet_music": (
        "\n## IMAGE TYPE OVERRIDE: SHEET MUSIC\n\n"
        "This image contains musical notation.\n"
        'verifier_checks.natural_check: "Is this legible sheet music with correct staff line structure?"\n'
    ),
}


# ── Call 1 schema validator (simplified — only need diversification schema) ────

def _validate_call1_schema(call1: dict) -> tuple[bool, str]:
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

    qt = call1.get("question_template", "")
    if not isinstance(qt, str) or not qt.strip():
        return False, "question_template missing or empty"

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


# ── Selector helpers (from edit_mmmu/generate_samples_v5_stage2.py) ─────────────

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
            lines.append(f"  - {ax}  [{sf.upper()}]  aggressive options: {entry.get('aggressive_value', '')!r}")
    return "\n".join(lines) if lines else "  (empty)"


def _build_selector_prompt(
    call1: dict, n_prompts: int, image_type_group: str = "natural_photo",
) -> tuple[str, str]:
    type_noun = IMAGE_TYPE_NOUNS.get(image_type_group, "image")
    if image_type_group == "natural_photo":
        distribution_constraint = (
            "    3. Stay strictly within the natural-photo distribution. No surreal effects, no illustrations,\n"
            "       no extreme color filters or weather (heavy fog/blizzard).\n"
        )
    else:
        distribution_constraint = (
            f"    3. The result must remain a plausible, legible {type_noun}. "
            f"Vary styling axes aggressively, but keep the structural integrity of the {type_noun} intact.\n"
        )

    system = (
        "You are an image augmentation strategy planner. You receive an original image "
        "plus a per-image analysis (frozen_predicate + axis_table). Your job is to "
        "produce N visually-distinct edit plans. Each plan picks one concrete value for "
        "every FREE axis, plus one slot-class for each replaceable slot, such that:\n\n"
        "  HARD constraints:\n"
        "    1. FROZEN axes must NOT be touched.\n"
        "    2. The frozen_predicate's verifier_checks must still hold true.\n"
        + distribution_constraint
        + "    4. For locked slots, do NOT change the slot class.\n\n"
        "  OPTIMIZATION:\n"
        "    A. Each plan should be MAXIMALLY AGGRESSIVE on every free axis.\n"
        "    B. INTER-PLAN diversity: across N plans, choose DIFFERENT concrete values for each axis.\n"
        "    C. INTRA-PLAN coherence: each plan should describe a coherent scene.\n\n"
        "Return ONLY valid JSON, no markdown fences, no commentary. Schema in the user message."
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
        f"## axis_table (FROZEN axes are off-limits; FREE/DEPENDS axes are your variables)\n"
        f"{_axis_table_brief(call1.get('axis_table', []))}\n\n"
        f"## Task: produce {n_prompts} maximally diverse edit plans.\n\n"
        f"Output JSON with EXACTLY this schema:\n"
        f'{{\n  "plans": [\n    {{\n'
        f'      "plan_id": 1,\n'
        f'      "slot_assignments": {{"S": {{"singular": "ceramic vase", "plural": "ceramic vases"}}}},\n'
        f'      "axis_values": {{"background_setting": "minimalist white studio", "...": "..."}},\n'
        f'      "diversity_signature": ["bg_studio", "lighting_cold"],\n'
        f'      "rationale": "<one short sentence>"\n'
        f"    }}\n  ]\n}}\n\n"
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
            "plan_id": int(p.get("plan_id", len(out) + 1)),
            "slot_assignments": norm_sa,
            "axis_values": norm_av,
            "diversity_signature": sig,
            "rationale": str(p.get("rationale", "")),
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


# ── Selector cache ─────────────────────────────────────────────────────────────

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
                f.write(json.dumps({"source_id": source_id, "plans": plans, "created_at": int(time.time())}, ensure_ascii=False) + "\n")


# ── Call 2b: Qwen text — diversification prose ────────────────────────────────

def _build_call2b_prompt(
    call1: dict,
    plan: dict,
    original_answer: str,
    image_type_group: str = "natural_photo",
) -> tuple[str, str]:
    fp = call1.get("frozen_predicate", {}) or {}
    must_preserve = fp.get("must_preserve", []) or []
    answer_function = fp.get("answer_function", "")
    type_noun = IMAGE_TYPE_NOUNS.get(image_type_group, "image")

    if image_type_group == "natural_photo":
        style_rule = (
            "NATURAL-PHOTO RULE (very important):\n"
            "  - The result MUST look like a real photograph from a normal camera in natural conditions.\n"
            "  - NO illustration / cartoon / cel-shaded / surreal style.\n"
            "  - NO dramatic single-color stage lighting, no extreme deep-blue/orange/purple color filters.\n"
            "  - NO surreal weather. Outdoor weather should be normal: clear / overcast / golden hour.\n\n"
        )
    else:
        style_rule = (
            f"STYLE RULE FOR {type_noun.upper()} (very important):\n"
            f"  - The result MUST remain a legible, well-structured {type_noun}.\n"
            f"  - You may dramatically change the visual styling (colors, fonts, line weights, backgrounds),\n"
            f"    but the structural content (data values, labels, layout) must be preserved exactly.\n\n"
        )

    system = (
        "You write image editing instructions for an AI image editor (GPT-Image-2). "
        "Your output is a SINGLE FLOWING PARAGRAPH of natural English prose, 150-260 words.\n\n"
        "Format rules:\n"
        "  - NO bullet points, NO labels, NO section headers, NO line breaks within the paragraph.\n"
        "  - One continuous description an art director would speak aloud.\n"
        "  - Every FREE axis below must be ADDRESSED with at least one concrete adjective or noun phrase.\n\n"
        "STRENGTH RULES (very important):\n"
        f"  - Strictly avoid these softening words: {', '.join(_PROSE_BANNED_WORDS)}\n"
        f"  - Prefer amplification: {', '.join(_PROSE_PUSH_WORDS)}\n"
        "  - The result image must look obviously different from the input image.\n\n"
        + style_rule
        + _CALL2B_TYPE_ADDENDA.get(image_type_group, "")
        + "FROZEN PREDICATE (the test point of the question):\n"
        "  - You MUST keep the frozen_predicate intact. The must_preserve list will appear in the user message.\n"
        "  - The answer to the original question must remain unchanged after the edit.\n"
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

    if image_type_group in ("table", "chart", "diagram", "medical", "sheet_music"):
        open_instruction = f"  1. Open by describing the new visual theme for this {type_noun} (color scheme, styling, fonts)."
    else:
        open_instruction = "  1. Open by describing the new background_setting and surface, with the slot subjects in place."

    user_text = (
        f"BASE SCENE (original image — your prose instructs the editor to apply a new visual style):\n"
        f"  {call1.get('scene_description', '')}\n\n"
        f"AXIS VALUES + SLOT CLASSES TO APPLY (every entry must appear in your prose):\n"
        f"{plan_block}\n\n"
        f"FROZEN PREDICATE — MUST be preserved (answer logic = {answer_function!r}):\n"
        f"  The original answer is: {original_answer!r} — this must remain correct after the edit.\n"
        f"{must_block}\n\n"
        f"Now write the single flowing paragraph for GPT-Image-2. The paragraph should:\n"
        f"{open_instruction}\n"
        f"  2. Cover every FREE axis in the list with a concrete adjective/phrase (no softeners).\n"
        f"  3. Close with a positive paraphrase of the must_preserve constraints.\n"
        f"Output ONLY the paragraph, no other text."
    )
    return system, user_text


# ── Verifier ───────────────────────────────────────────────────────────────────

def _build_verifier_prompt(verifier_checks_instantiated: dict) -> tuple[str, str]:
    system = (
        "You are a strict visual verifier. You will be shown an image and 4 short yes/no questions. "
        "Answer each one with a single word (yes or no). Do not explain. Do not hedge. Use only the visual "
        "evidence in the image. If a question's premise can't be evaluated from the image, answer 'no'."
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


# ── Call 4: CoT generation ─────────────────────────────────────────────────────

def _build_call4_prompt(
    call1: dict,
    final_question: str,
    final_answer: str,
    options: list[str] | None = None,
    answer_letter: str = "",
) -> tuple[str, str]:
    system = (
        "You are a careful visual reasoning assistant. You produce a brief, accurate chain of thought "
        "for a visual question, ending with the final answer.\n\n"
        "Style requirements:\n"
        "  - Write in natural, flowing English (not numbered steps, not bullet lists)\n"
        "  - Be concise (3-6 short sentences total)\n"
        "  - Be accurate — your reasoning must lead to the supplied final answer\n"
        "  - End with the line: **Final Answer: <answer>**\n"
    )
    disambs = call1.get("disambiguators_to_preserve", [])
    test_point = call1.get("test_point_analysis", {})
    soft_hints = ""
    if disambs:
        soft_hints += "Helpful hints (use only if natural):\n"
        for d in disambs[:3]:
            soft_hints += f"  - {d.get('feature', '')}\n"
    if test_point.get("core_challenge"):
        soft_hints += f"Note: {test_point['core_challenge']}\n"

    opts_block = ""
    if options:
        opts_block = f"Options: {format_mmmu_options(options)}\n"
    answer_display = f"{answer_letter} = {final_answer}" if answer_letter else final_answer

    user_text = (
        f"Question: {final_question}\n"
        f"{opts_block}\n"
        "Look at the image and reason briefly to the answer.\n\n" + soft_hints + "\n"
        f"End your response with: **Final Answer: {answer_display}**"
    )
    return system, user_text


# ── Stage 2 Direct Pipeline ────────────────────────────────────────────────────

class Stage2DirectPipeline:
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
        n_min_plans: int = 5,
        n_max_plans: int = 5,
        use_gemini: bool = False,
    ):
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "stage2_images"
        self.output_jsonl = self.output_dir / "stage2_samples.jsonl"
        self.selector_cache_path = self.output_dir / "stage2_selector_cache.jsonl"

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
        self.n_min_plans = n_min_plans
        self.n_max_plans = n_max_plans
        self.use_gemini = use_gemini

        self._writer: Optional[JsonlWriter] = None
        self._done_keys: set[str] = set()
        self._selector_cache: Optional[SelectorCache] = None
        self._inner = None
        self._proxy = None

    async def _process_plan(
        self,
        source_id: str,
        question_orig: str,
        answer_orig: str,
        call1: dict,
        plan: dict,
        gpt_bytes: bytes,
        gpt_mime: str,
        mmmu_meta: dict | None = None,
        image_type_group: str = "natural_photo",
    ) -> None:
        prompt_idx = int(plan.get("plan_id", 0))
        sample_key = f"{source_id}::p{prompt_idx}::s0"

        if sample_key in self._done_keys:
            return

        # Call 2b: prose
        try:
            sys_p, usr_p = _build_call2b_prompt(
                call1, plan, answer_orig, image_type_group=image_type_group
            )
            edit_prose = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                qwen_text_messages(sys_p, usr_p, no_think=True),
                max_tokens=4096, temperature=0.6,
                max_retries=self.max_retries, log_prefix=f"qwen2b/{sample_key}",
            )
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id,
                "stage": "2_direct", "status": "failed", "reject_reason": "call2b_failed",
                "plan": plan, "error": f"{type(e).__name__}: {e!r}",
                "created_at": int(time.time()),
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
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id,
                "stage": "2_direct", "status": "failed", "reject_reason": "call3b_gpt_image_failed",
                "edit_prose": edit_prose, "plan": plan, "error": f"{type(e).__name__}: {e!r}",
                "created_at": int(time.time()),
            })
            return

        if not stage2_imgs:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id,
                "stage": "2_direct", "status": "failed", "reject_reason": "call3b_no_image",
                "edit_prose": edit_prose, "plan": plan, "created_at": int(time.time()),
            })
            return

        stage2_bytes = stage2_imgs[0]
        img_path = self.images_dir / f"{sample_key}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(stage2_bytes)
        stage2_b64 = base64.b64encode(stage2_bytes).decode()

        # Instantiate slots → final question/answer/verifier_checks
        slot_assignments = plan.get("slot_assignments", {}) or {}
        fp = call1.get("frozen_predicate", {}) or {}
        slot_values: dict = dict(slot_assignments)
        for sname, sdef in (fp.get("slots") or {}).items():
            if isinstance(sdef, dict) and "value" in sdef and sname not in slot_values:
                slot_values[sname] = answer_orig

        question_template = call1.get("question_template", "") or question_orig
        answer_template = call1.get("answer_template", "") or answer_orig

        if image_type_group in ("table", "chart"):
            final_question = question_orig
            final_answer = answer_orig
        else:
            final_question = _instantiate_template(question_template, slot_values) or question_orig
            final_answer = _instantiate_template(answer_template, slot_values) or answer_orig

        verifier_checks_raw = fp.get("verifier_checks") or {}
        verifier_checks_inst = {k: _instantiate_template(v, slot_values) for k, v in verifier_checks_raw.items()}
        if not verifier_checks_inst.get("natural_check"):
            type_noun = IMAGE_TYPE_NOUNS.get(image_type_group, "image")
            if image_type_group == "natural_photo":
                verifier_checks_inst["natural_check"] = "Is this a plausible real photograph (not illustration / surreal / cartoon)?"
            else:
                verifier_checks_inst["natural_check"] = f"Is this a legible, well-structured {type_noun} (not corrupted or garbled)?"

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
                "sample_key": sample_key, "source_id": source_id,
                "stage": "2_direct", "status": "failed", "reject_reason": "verifier_failed",
                "stage2_image_path": str(img_path), "edit_prose": edit_prose, "plan": plan,
                "verifier_checks_instantiated": verifier_checks_inst,
                "error": f"{type(e).__name__}: {e!r}", "created_at": int(time.time()),
            })
            return

        if not verifier["all_pass"]:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id,
                "stage": "2_direct", "status": "rejected", "reject_reason": "verifier_check_failed",
                "stage2_image_path": str(img_path), "edit_prose": edit_prose, "plan": plan,
                "verifier_checks_instantiated": verifier_checks_inst,
                "verifier_answers": {k: verifier[k] for k in ("a1", "a2", "a3", "a4")},
                "verifier_raw": verifier["raw"][:1000], "created_at": int(time.time()),
            })
            return

        # Call 4: CoT
        options = mmmu_meta.get("options") if mmmu_meta else None
        answer_letter_out = mmmu_meta.get("answer_letter", "") if mmmu_meta else ""
        try:
            sys_p, usr_p = _build_call4_prompt(
                call1, final_question, final_answer, options=options, answer_letter=answer_letter_out,
            )
            cot = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                qwen_vl_messages(sys_p, [(stage2_b64, "image/png")], usr_p, no_think=True),
                max_tokens=4096, temperature=0.6,
                max_retries=self.max_retries, log_prefix=f"call4/{sample_key}",
            )
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key, "source_id": source_id,
                "stage": "2_direct", "status": "failed", "reject_reason": "cot_failed",
                "stage2_image_path": str(img_path), "edit_prose": edit_prose, "plan": plan,
                "final_question": final_question, "final_answer": final_answer,
                "error": f"{type(e).__name__}: {e!r}", "created_at": int(time.time()),
            })
            return

        if "**Final Answer:" not in cot:
            cot = cot.rstrip() + f"\n\n**Final Answer: {final_answer}**"

        record = {
            "sample_key": sample_key,
            "source_id": source_id,
            "stage": "2_direct",
            "quality_pass": True,
            "original_question": question_orig,
            "original_answer": answer_orig,
            "final_question": final_question,
            "final_answer": final_answer,
            "cot": cot,
            "stage2_image_path": str(img_path),
            "edit_prose": edit_prose,
            "plan": plan,
            "verifier_checks_instantiated": verifier_checks_inst,
            "verifier_answers": {k: verifier[k] for k in ("a1", "a2", "a3", "a4")},
            "prompt_idx": prompt_idx,
            "created_at": int(time.time()),
        }
        if mmmu_meta:
            record["mmmu_meta"] = mmmu_meta
            record["image_type_group"] = image_type_group
        await self._writer.write(record)
        log(f"  [{sample_key}] ✓ stage2_direct done | answer={final_answer}", "stage2")

    async def _select_plans_via_api(
        self,
        call1: dict,
        image_b64: str,
        n_prompts: int,
        log_prefix: str,
        image_type_group: str = "natural_photo",
    ) -> list[dict]:
        sys_p, usr_p = _build_selector_prompt(call1, n_prompts, image_type_group=image_type_group)
        try:
            text = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                qwen_vl_messages(sys_p, [(image_b64, "image/png")], usr_p),
                max_tokens=8192, temperature=0.5,
                max_retries=self.max_retries, log_prefix=log_prefix,
            )
        except Exception as e:
            log(f"  [{log_prefix}] selector API failed: {type(e).__name__}: {e!r}", "stage2")
            return []
        return _parse_selector_output(text, n_prompts)

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

        rng = random.Random(hash(source_id) & 0xFFFFFFFF)
        n_target = rng.randint(self.n_min_plans, self.n_max_plans)
        # table/chart: cap at 5 (fewer useful style axes; already matches 1:5 target)
        if image_type_group in ("table", "chart"):
            n_target = min(n_target, 5)

        async with self.semaphore:
            try:
                all_bytes = get_mmmu_image_bytes_list(row)
                if not all_bytes:
                    log(f"  [{source_id}] no image, skip", "stage2")
                    return

                gemini_images = []
                for img_bytes in all_bytes:
                    hq_bytes, hq_mime = prepare_image(img_bytes, max_side=1568, fmt="PNG")
                    gemini_images.append((base64.b64encode(hq_bytes).decode(), hq_mime))

                primary_bytes = all_bytes[0]
                gpt_bytes, gpt_mime = prepare_image(primary_bytes, max_side=1024, fmt="JPEG")
                primary_b64 = base64.b64encode(primary_bytes).decode()

                # Resume check
                if f"{source_id}::skipped" in self._done_keys:
                    log(f"  [{source_id}] resume: previously skipped", "stage2")
                    return

                # Call 1 — diversification analysis
                vl_backend = "gemini" if self.use_gemini else "qwen"
                log(f"  [{source_id}] Call 1 ({vl_backend}, type={image_type_group}, n_plans={n_target})...", "stage2")
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
                        "source_id": source_id, "stage": "2_direct", "status": "failed",
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
                        "source_id": source_id, "stage": "2_direct", "status": "failed",
                        "reject_reason": "call1_schema_invalid",
                        "original_question": question, "original_answer": answer_text,
                        "error": err, "call1_full": call1, "mmmu_meta": mmmu_meta,
                        "created_at": int(time.time()),
                    })
                    log(f"  [{source_id}] schema invalid: {err}", "stage2")
                    return

                # Selector: try cache first
                plans = self._selector_cache.get(source_id) if self._selector_cache else None
                if plans is None:
                    plans = await self._select_plans_via_api(
                        call1, primary_b64, n_prompts=n_target,
                        log_prefix=f"selector/{source_id}",
                        image_type_group=image_type_group,
                    )
                    if plans and self._selector_cache:
                        await self._selector_cache.put(source_id, plans)

                if not plans:
                    await self._writer.write({
                        "sample_key": f"{source_id}::skipped",
                        "source_id": source_id, "stage": "2_direct", "status": "skipped",
                        "reject_reason": "selector_no_plans",
                        "original_question": question, "original_answer": answer_text,
                        "call1_full": call1, "mmmu_meta": mmmu_meta,
                        "created_at": int(time.time()),
                    })
                    return

                # Renumber plans 1..N
                for i, p in enumerate(plans, start=1):
                    p["plan_id"] = i

                tasks = [
                    self._process_plan(
                        source_id, question, answer_text, call1, p,
                        gpt_bytes, gpt_mime,
                        mmmu_meta=mmmu_meta, image_type_group=image_type_group,
                    )
                    for p in plans
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                log(f"  [{source_id}] UNEXPECTED ERROR: {type(e).__name__}: {e!r}", "stage2")
                log(traceback.format_exc(), "stage2")
                await self._writer.write({
                    "sample_key": f"{source_id}::unexpected_error",
                    "source_id": source_id, "stage": "2_direct", "status": "failed",
                    "reject_reason": "unexpected_error",
                    "original_question": question, "original_answer": answer_text,
                    "error": f"{type(e).__name__}: {e!r}",
                    "mmmu_meta": mmmu_meta, "created_at": int(time.time()),
                })

    async def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._writer = JsonlWriter(self.output_jsonl)
        self._selector_cache = SelectorCache(self.selector_cache_path)
        log(f"selector cache: {len(self._selector_cache._mem)} entries loaded", "stage2")

        if self.resume:
            self._done_keys = load_done_keys(self.output_jsonl)
            log(f"resume: {len(self._done_keys)} sample_keys already done", "stage2")

        log(f"loading manifest: {self.manifest}", "stage2")
        rows = load_mmmu_manifest(self.manifest)
        log(f"loaded {len(rows)} MMMU samples", "stage2")
        if self.limit > 0:
            rows = rows[: self.limit]
            log(f"limit to {self.limit}", "stage2")

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
            f"stage2_direct done. total: {total} | passed: {passed} | rejected: {rejected} "
            f"| failed: {failed} | skipped: {skipped}",
            "stage2",
        )
        log(f"output jsonl: {self.output_jsonl}", "stage2")


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="方案二 Stage 2 Direct: 5 cosmetic variants of original MMMU images (1:5 ratio)")
    parser.add_argument("--manifest", required=True, help="Path to selected_data/manifest.jsonl")
    parser.add_argument("--output_dir", default="_archive/output_fangan2")
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
    parser.add_argument("--n_min_plans", type=int, default=5)
    parser.add_argument("--n_max_plans", type=int, default=5)
    parser.add_argument("--max_retries", type=int, default=int(os.environ.get("MAX_RETRIES", "6")))
    parser.add_argument("--api_timeout", type=float, default=float(os.environ.get("API_TIMEOUT", "1200")))
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--limit", type=int, default=int(os.environ.get("LIMIT", "0")))
    parser.add_argument("--use_gemini", action="store_true")
    args = parser.parse_args()

    if not args.gpt_image_api_key:
        parser.error("missing: --gpt_image_api_key / GPT_IMAGE_API_KEY")
    if not args.qwen_api_key:
        parser.error("missing: --qwen_api_key / OPENAI_API_KEY")

    pipe = Stage2DirectPipeline(
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
        n_min_plans=args.n_min_plans,
        n_max_plans=args.n_max_plans,
        max_retries=args.max_retries,
        api_timeout=args.api_timeout,
        resume=not args.no_resume,
        limit=args.limit,
        use_gemini=args.use_gemini,
    )
    asyncio.run(pipe.run())


if __name__ == "__main__":
    main()
