#!/usr/bin/env python3
"""
MMMU Stage 2: Cosmetic Diversification + CoT + Verification + Option Rewriting

Per quality-pass stage1 record:

  1. Selector [Qwen VL]      -> 2-4 diversification plans (cached)
  2. Per plan (parallel):
     a. Call 2b [Qwen text]  -> diversification prose
     b. Call 3b [gemini-3-pro-image-preview] -> 1 diversified image
     c. Verifier [Qwen VL]   -> 4 yes/no checks
     d. If pass:
        i.   CoT generation [Qwen VL, thinking mode]
        ii.  Clean CoT
        iii. Verify answer+CoT [Qwen VL]
        iv.  Rewrite options [Qwen text]
        v.   Build training-format output

Output:
  output/stage2_train.jsonl       (training records)
  output/stage2_images/           (diversified images)
  output/stage2_rejected.jsonl    (rejected records)
  output/stage2_selector_cache.jsonl
"""

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
from typing import Any, Optional

from _common import (
    # API clients
    chat_qwen,
    # Message builders
    qwen_text_messages,
    qwen_vl_messages,
    # Image utils
    prepare_image,
    extract_json,
    # Concurrency
    _get_qwen_sem,
    _get_gpt_img_sem,
    # Retry
    _backoff_with_jitter,
    RETRYABLE,
    # HTTP
    make_clients,
    # JSONL
    JsonlWriter,
    load_done_keys,
    jsonl_iter,
    # Logging
    log,
    # Image classification
    classify_image_type,
    IMAGE_TYPE_NOUNS,
    # Axis taxonomy
    UNIVERSAL_AXES,
    # Paths / config
    QWEN_API_BASE,
    QWEN_API_KEY,
    QWEN_MODEL,
    GEMINI_IMAGE_ENDPOINT,
    GEMINI_IMAGE_API_KEY,
    OUTPUT_DIR,
    # Option utilities
    LETTERS,
    format_options,
    format_options_multiline,
    answer_text,
    normalize_question,
)

import httpx

import sys as _sys
_sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
from _gemini_image import (
    gemini_image_edit,
    GEMINI_IMAGE_ENDPOINT,
    GEMINI_IMAGE_API_KEY,
)

# ── Concurrency ──────────────────────────────────────────────────────────────

MAX_CONCURRENT = int(os.environ.get("MAX_CONCURRENT", "64"))

# ── Type-specific diversification guidance ────────────────────────────────────

_SELECTOR_TYPE_ADDENDA: dict[str, str] = {
    "table": (
        "\n## DIVERSIFICATION GUIDANCE FOR TABLES\n\n"
        "For table images, diversification means changing the VISUAL PRESENTATION while keeping\n"
        "all data values, row/column structure, and text content pixel-identical.\n\n"
        "Concrete dimensions to vary across plans (pick different values per plan):\n"
        "  - Color scheme: dark theme (dark background + light text) / light corporate (white bg + blue headers) /\n"
        "    pastel (soft green/pink/lavender headers) / high-contrast (black-white with bold colored accents) /\n"
        "    warm earth tones (beige bg + brown borders) / cool blue-gray professional\n"
        "  - Font family: serif (Times, Georgia) / sans-serif (Arial, Helvetica) / monospace (Courier, Consolas) /\n"
        "    rounded modern (Nunito, Poppins) / condensed narrow / handwritten casual\n"
        "  - Font size and weight: compact dense text / generous large text with padding / bold headers + light body /\n"
        "    all-caps headers with small body text\n"
        "  - Border style: thin hairline borders / thick double borders / no borders with alternating row shading /\n"
        "    rounded-corner cells / dashed borders / heavy outer border only\n"
        "  - Row shading: zebra stripes (alternating gray) / no shading / gradient rows / highlighted header row only\n"
        "  - Cell padding: tight compact layout / spacious with generous padding / wide columns with narrow rows\n"
        "  - Background: plain white / textured paper / solid colored / gradient background / subtle grid pattern\n\n"
        "CRITICAL: Every cell value, every number, every label must remain EXACTLY the same.\n"
        "Only the visual styling changes -- think of it as applying a completely different CSS theme.\n"
    ),
    "chart": (
        "\n## DIVERSIFICATION GUIDANCE FOR CHARTS/GRAPHS\n\n"
        "For chart images, diversification means changing the VISUAL STYLE while keeping\n"
        "all data values, axis ranges, labels, and legend text identical.\n\n"
        "Concrete dimensions to vary across plans (pick different values per plan):\n"
        "  - Color palette: warm (reds/oranges/golds) / cool (blues/teals/purples) / monochrome grayscale /\n"
        "    high-contrast primary (red/blue/green/yellow) / pastel muted / neon vivid / earth tones\n"
        "  - Rendering style: flat 2D clean / 3D with perspective and shadows / gradient fills / hand-drawn sketch /\n"
        "    minimalist line-art / bold thick outlines\n"
        "  - Grid and axis style: minimal grid (light dotted lines) / heavy solid gridlines / no grid at all /\n"
        "    major gridlines only / dashed grid with tick marks\n"
        "  - Background: white / light gray / dark (dark mode) / subtle gradient / textured paper\n"
        "  - Font: sans-serif clean / serif formal / monospace technical / bold condensed / rounded modern\n"
        "  - Data label style: no labels (legend only) / inline data labels on each point / annotated with callouts /\n"
        "    percentage labels on pie segments\n"
        "  - Legend position and style: top / bottom / right side / inline / no legend (labeled directly)\n"
        "  - Bar/line style (if applicable): solid bars / hatched patterns / rounded bars / thin lines / thick lines /\n"
        "    markers on data points / smooth curves vs straight segments\n\n"
        "CRITICAL: Every data value, every axis label, every number must remain EXACTLY the same.\n"
        "Only the visual rendering changes -- think of it as switching the chart theme/template.\n"
    ),
    "diagram": (
        "\n## DIVERSIFICATION GUIDANCE FOR DIAGRAMS\n\n"
        "For diagram images, diversification means changing the VISUAL STYLE while keeping\n"
        "all labels, connections, node values, and structural layout identical.\n\n"
        "Concrete dimensions to vary across plans (pick different values per plan):\n"
        "  - Color scheme: blue/gray professional / colorful rainbow fills / monochrome black-white /\n"
        "    earth tones (brown/olive/tan) / dark mode (dark bg + bright nodes) / pastel soft\n"
        "  - Line style: thin clean lines / thick bold outlines / dashed lines / double lines /\n"
        "    hand-drawn sketch style / dotted connections\n"
        "  - Node/shape style: rounded rectangles / sharp rectangles / circles / hexagons /\n"
        "    filled vs outlined / drop shadow vs flat\n"
        "  - Font: sans-serif / serif / monospace / bold / italic annotations\n"
        "  - Arrow style: solid arrowheads / open arrowheads / no arrowheads (just lines) / thick arrows\n"
        "  - Background: white / light grid / blueprint blue / dark / subtle texture\n"
        "  - Annotation style: boxed labels / plain text / callout bubbles / underlined\n\n"
        "CRITICAL: Every label, every connection, every node value must remain EXACTLY the same.\n"
        "Only the visual styling changes.\n"
    ),
    "medical": (
        "\n## DIVERSIFICATION GUIDANCE FOR MEDICAL IMAGES\n\n"
        "For medical images, diversification is limited to imaging parameters:\n"
        "  - Contrast/brightness: high contrast / low contrast / enhanced edges / normal clinical\n"
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
        "  - Note head style: standard filled / open / diamond-shaped (for harmonics)\n"
        "  - Spacing: compact dense layout / generous wide spacing\n\n"
        "CRITICAL: Every note position, accidental, dynamic marking must remain identical.\n"
    ),
}

_CALL2B_TYPE_ADDENDA: dict[str, str] = {
    "table": (
        "TABLE-SPECIFIC INSTRUCTIONS (very important):\n"
        "  - Describe a completely new visual theme for this table as if you are writing a CSS stylesheet in prose.\n"
        "  - You MUST specify ALL of: background color, header row color, body row color/shading pattern,\n"
        "    border style and color, font family, font size, text color, cell padding.\n"
        "  - Be extremely concrete: say 'dark navy #1a1a2e background with gold #e6b800 header text' not just 'dark theme'.\n"
        "  - Say 'bold 14pt Helvetica headers with 11pt Georgia body text' not just 'different font'.\n"
        "  - The table must remain perfectly readable -- no color combinations that make text hard to read.\n"
        "  - Every single cell value, number, and text label must be preserved EXACTLY -- only the visual chrome changes.\n"
        "  - Do NOT describe the table content (row values, column names). Just describe the visual style.\n\n"
    ),
    "chart": (
        "CHART-SPECIFIC INSTRUCTIONS (very important):\n"
        "  - Describe a completely new visual theme for this chart as if you are restyling it in a different plotting library.\n"
        "  - You MUST specify ALL of: background color, bar/line/pie colors for each data series, grid line style,\n"
        "    axis label font and size, title font and size, legend style and position.\n"
        "  - Be extremely concrete: say 'bars in #e74c3c red, #3498db blue, #2ecc71 green' not just 'colorful bars'.\n"
        "  - Say 'x-axis labels in 10pt Helvetica rotated 45 degrees' not just 'rotated labels'.\n"
        "  - Specify rendering: 'flat 2D bars with 2px dark borders' or '3D bars with drop shadows and gradient fills'.\n"
        "  - Every data value, axis range, label, and number must be preserved EXACTLY -- only the visual rendering changes.\n"
        "  - Do NOT describe the data content. Just describe the visual style.\n\n"
    ),
    "diagram": (
        "DIAGRAM-SPECIFIC INSTRUCTIONS (very important):\n"
        "  - Describe a completely new visual theme for this diagram as if applying a different drawing template.\n"
        "  - You MUST specify: node fill color, node border style, connection line color and weight,\n"
        "    arrow head style, label font, background color.\n"
        "  - Be concrete: say 'nodes as rounded rectangles with #4a90d9 fill and 2px #2c3e50 borders' not just 'blue nodes'.\n"
        "  - Every label, connection, and structural element must be preserved EXACTLY -- only the visual styling changes.\n\n"
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


# ── Selector prompt ──────────────────────────────────────────────────────────


def _slot_pool_brief(slots: dict) -> str:
    lines = []
    for sname, sdef in (slots or {}).items():
        if not isinstance(sdef, dict):
            continue
        if "compatible_classes" in sdef:
            ccs = sdef["compatible_classes"][:8]
            cc_str = ", ".join(
                c.get("singular", "?") for c in ccs if isinstance(c, dict)
            )
            lines.append(
                f"    <{sname}>  role={sdef.get('role', '?')}  "
                f"replaceability={sdef.get('slot_replaceability', '?')}  "
                f"shape={sdef.get('shape_constraint', '')!r}\n"
                f"        compatible: [{cc_str}]"
            )
        elif "value" in sdef:
            lines.append(
                f"    <{sname}>  numeric/value={sdef.get('value', '?')!r}  frozen={sdef.get('frozen', True)}"
            )
    return "\n".join(lines) if lines else "    (no slots)"


def _axis_table_brief(axis_table: list) -> str:
    lines = []
    for entry in axis_table or []:
        if not isinstance(entry, dict):
            continue
        ax = entry.get("axis", "?")
        sf = entry.get("safety", "?")
        if sf == "frozen":
            lines.append(
                f"  - {ax}  [FROZEN]  must preserve: {entry.get('freeze_constraint', '')!r}"
            )
        else:
            lines.append(
                f"  - {ax}  [{sf.upper()}]  aggressive options: {entry.get('aggressive_value', '')!r}"
            )
    return "\n".join(lines) if lines else "  (empty)"


def _build_selector_prompt(
    call1: dict,
    n_prompts: int,
    image_type_group: str = "natural_photo",
) -> tuple[str, str]:
    type_noun = IMAGE_TYPE_NOUNS.get(image_type_group, "image")
    if image_type_group == "natural_photo":
        distribution_constraint = (
            "    3. Stay strictly within the natural-photo distribution. No surreal effects, no illustrations, "
            "       no extreme color filters or weather (heavy fog/blizzard).\n"
        )
    else:
        distribution_constraint = (
            f"    3. The result must remain a plausible, legible {type_noun}. "
            f"Vary styling axes (color scheme, line style, font, etc.) aggressively, but keep the "
            f"structural integrity of the {type_noun} intact.\n"
        )

    system = (
        "You are an image augmentation strategy planner. You receive an already-edited image "
        "(Stage 1 output) plus a per-image analysis (frozen_predicate + axis_table). Your job is to "
        "produce N visually-distinct edit plans for Stage 2. Each plan picks one concrete value for "
        "every FREE axis, plus one slot-class for each replaceable slot, such that:\n\n"
        "  HARD constraints:\n"
        "    1. FROZEN axes must NOT be touched. Don't propose values for them.\n"
        "    2. The frozen_predicate's verifier_checks (class/relation/readability/natural) must still "
        "       hold true on the imagined edited image.\n"
        + distribution_constraint
        + "    4. For slots with replaceability=='locked', do NOT change the slot class (keep default).\n\n"
        "  OPTIMIZATION:\n"
        "    A. Each plan should be MAXIMALLY AGGRESSIVE on every free axis (push to the upper end of the\n"
        "       provided aggressive_value options). The goal is for every plan to look very different from\n"
        "       the input image while keeping the test point intact.\n"
        "    B. INTER-PLAN diversity: across the N plans, choose DIFFERENT concrete values for each axis so\n"
        "       the plans cover N distant points in axis space. Use the slash-separated alternatives in each\n"
        "       axis's aggressive_value as the menu (or invent equally natural alternatives). For class slots,\n"
        "       pick a different compatible_class per plan when possible.\n"
        "    C. INTRA-PLAN coherence: a single plan should describe a coherent scene. E.g., 'minimalist white\n"
        "       studio' background is more coherent with 'cool fluorescent' lighting than with 'golden hour'.\n\n"
        "Each plan also produces a `diversity_signature` -- a short list of 3-5 tags (e.g., "
        "['bg_warehouse', 'surf_metal', 'lighting_cold']) used downstream for cross-plan diversity audit.\n\n"
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
        f"For each plan:\n"
        f"  - slot_assignments: pick a compatible_class for each replaceable class slot (keep locked slots at default).\n"
        f"    Use BOTH singular and plural forms.\n"
        f"  - axis_values: pick a concrete value for every FREE axis (key = axis name; value = single concrete\n"
        f"    instruction string drawn from or inspired by the axis's aggressive_value menu).\n"
        f"  - diversity_signature: 3-5 short tags summarizing this plan's character.\n"
        f"  - rationale: one short sentence on why this plan is answer-safe and visually impactful.\n\n"
        f"Output JSON with EXACTLY this schema:\n"
        f"{{\n"
        f'  "plans": [\n'
        f"    {{\n"
        f'      "plan_id": 1,\n'
        f'      "slot_assignments": {{\n'
        f'        "S": {{"singular": "ceramic vase", "plural": "ceramic vases"}},\n'
        f'        "SURF": {{"singular": "marble table", "plural": "marble tables"}}\n'
        f"      }},\n"
        f'      "axis_values": {{\n'
        f'        "background_setting": "minimalist white studio with seamless backdrop",\n'
        f'        "lighting_quality":   "cool 5500K daylight from the upper-left",\n'
        f'        "...":                "..."\n'
        f"      }},\n"
        f'      "diversity_signature": ["bg_studio", "lighting_cold", "surf_marble"],\n'
        f'      "rationale": "<one short sentence>"\n'
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
                norm_sa[k] = {
                    "singular": str(v["singular"]),
                    "plural": str(v["plural"]),
                }
            elif isinstance(v, str):
                norm_sa[k] = {"singular": v, "plural": v}
        norm_av = {
            str(k): str(v) for k, v in av.items() if isinstance(v, (str, int, float))
        }
        sig = p.get("diversity_signature", []) or []
        if isinstance(sig, str):
            sig = [sig]
        sig = [str(x) for x in sig if x]
        out.append(
            {
                "plan_id": int(p.get("plan_id", len(out) + 1)),
                "slot_assignments": norm_sa,
                "axis_values": norm_av,
                "diversity_signature": sig,
                "rationale": str(p.get("rationale", "")),
            }
        )
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


# ── Selector cache ────────────────────────────────────────────────────────────


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
                k = rec.get("stage1_key")
                plans = rec.get("plans")
                if k and isinstance(plans, list):
                    self._mem[k] = plans
        except Exception as e:
            log(f"selector cache load warning: {e!r}", "stage2")

    def get(self, stage1_key: str) -> Optional[list[dict]]:
        return self._mem.get(stage1_key)

    async def put(self, stage1_key: str, plans: list[dict]) -> None:
        async with self._lock:
            self._mem[stage1_key] = plans
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "stage1_key": stage1_key,
                            "plans": plans,
                            "created_at": int(time.time()),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )


# ── Call 2b: diversification prose ────────────────────────────────────────────

_PROSE_BANNED_WORDS = (
    "subtle", "subtly", "slightly", "a touch of", "a hint of", "hints of",
    "gentle", "softly", "lightly", "mild", "mildly",
)

_PROSE_PUSH_WORDS = (
    "completely", "entirely", "wholly", "dramatically", "fully",
    "thoroughly", "transformed into",
)


def _build_call2b_prompt(
    call1: dict,
    plan: dict,
    stage1_record: dict,
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
            "  - NO surreal weather (heavy fog, blizzard, sandstorm). Outdoor weather should be normal: clear / overcast / golden hour.\n\n"
        )
    else:
        style_rule = (
            f"STYLE RULE FOR {type_noun.upper()} (very important):\n"
            f"  - The result MUST remain a legible, well-structured {type_noun}.\n"
            f"  - You may dramatically change the visual styling (colors, fonts, line weights, backgrounds),\n"
            f"    but the structural content (data values, labels, layout) must be preserved exactly.\n"
            f"  - Vary the aesthetic aggressively -- the goal is maximal visual difference from the input\n"
            f"    while keeping all information intact and readable.\n\n"
        )

    system = (
        "You write image editing instructions for an AI image editor (gemini-3-pro-image-preview). "
        "Your output is a SINGLE FLOWING PARAGRAPH of natural English prose, 150-260 words.\n\n"
        "Format rules:\n"
        "  - NO bullet points, NO labels, NO section headers, NO line breaks within the paragraph.\n"
        "  - One continuous description an art director would speak aloud.\n"
        "  - Every FREE axis below must be ADDRESSED with at least one concrete adjective or noun phrase\n"
        "    (e.g., for 'lighting_quality' write 'lit by cool 5500K fluorescent panels overhead', not just\n"
        "    'naturally lit').\n"
        "  - Each new visual element should be vivid and specific (named materials, named colors, named objects,\n"
        "    explicit camera distance/angle).\n\n"
        "STRENGTH RULES (very important):\n"
        f"  - Strictly avoid these softening words: {', '.join(_PROSE_BANNED_WORDS)}\n"
        f"  - Prefer amplification: {', '.join(_PROSE_PUSH_WORDS)}\n"
        "  - The result image must look obviously different from the input image -- not 'the same scene with a\n"
        "    filter'. Substantive change on every free axis is the goal.\n\n"
        + style_rule
        + _CALL2B_TYPE_ADDENDA.get(image_type_group, "")
        + "FROZEN PREDICATE (the test point of the question):\n"
        "  - You MUST keep the frozen_predicate intact pixel-truth-wise. The must_preserve list will appear\n"
        "    at the end of the user message -- paraphrase the constraints into the closing of your paragraph\n"
        "    using positive language ('the scene still shows ...').\n"
        "  - Do NOT mention the question or the answer in the prose. The image editor doesn't need to know.\n"
    )

    plan_lines = []
    for sname, sval in (plan.get("slot_assignments") or {}).items():
        if isinstance(sval, dict):
            plan_lines.append(
                f"  * slot <{sname}>: {sval.get('plural', sval.get('singular', '?'))}"
            )
    for ax, val in (plan.get("axis_values") or {}).items():
        plan_lines.append(f"  * axis {ax}: {val}")
    plan_block = "\n".join(plan_lines) if plan_lines else "  (no axis instructions)"

    must_block = (
        "\n".join(f"  - {x}" for x in must_preserve) if must_preserve else "  (none)"
    )

    if image_type_group in ("table", "chart", "diagram", "medical", "sheet_music"):
        open_instruction = f"  1. Open by describing the new visual theme for this {type_noun} (color scheme, styling, fonts)."
    else:
        open_instruction = "  1. Open by describing the new background_setting and surface, with the slot subjects in place."

    user_text = (
        f"BASE SCENE (Stage 1 already edited image -- your prose REPLACES this with a new visual style;\n"
        f"the editor will look at the input image but you instruct it to repaint with the new axis values):\n"
        f"  {call1.get('scene_description', '')}\n\n"
        f"AXIS VALUES + SLOT CLASSES TO APPLY (every entry must appear in your prose):\n"
        f"{plan_block}\n\n"
        f"FROZEN PREDICATE -- MUST be preserved (answer logic = {answer_function!r}):\n"
        f"{must_block}\n\n"
        f"Now write the single flowing paragraph for gemini-3-pro-image-preview. The paragraph should:\n"
        f"{open_instruction}\n"
        f"  2. Cover every FREE axis in the list with a concrete adjective/phrase (no softeners).\n"
        f"  3. Close with a positive paraphrase of the must_preserve constraints (one sentence, woven in naturally).\n"
        f"Output ONLY the paragraph, no other text."
    )
    return system, user_text


# ── Verifier prompt ───────────────────────────────────────────────────────────


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


# ── CoT generation (thinking mode) ───────────────────────────────────────────

SYSTEM_COT_GEN = (
    "You are an expert visual question answering assistant. You will be shown an image "
    "and a question about the image. Reason step by step about the image content to arrive "
    "at the correct answer. Focus on what you actually observe in the image. Be concise but thorough."
)

PROMPT_COT_GEN = (
    "Look at the image and answer the following question step by step.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Correct answer: {answer}\n\n"
    "Reason carefully about the image to explain how you arrive at this answer. "
    "Focus on visual details from the image that support the answer."
)


async def call_qwen_thinking(
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
                wait = _backoff_with_jitter(10, attempt)
                log(f"  [{log_prefix}] no choices, wait {wait:.1f}s", "stage2")
                await asyncio.sleep(wait)
                continue

            msg = body["choices"][0]["message"]
            reasoning = (msg.get("reasoning_content") or "").strip()
            if reasoning:
                return reasoning
            content = (msg.get("content") or "").strip()
            if content:
                return content
            return None

        except RETRYABLE as e:
            wait = _backoff_with_jitter(2 ** attempt, 0, jitter_frac=0.5)
            log(f"  [{log_prefix}] retry {attempt+1}/{max_retries}: {type(e).__name__} (wait {wait:.1f}s)", "stage2")
            await asyncio.sleep(wait)
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                wait = _backoff_with_jitter(15, attempt)
                log(f"  [{log_prefix}] HTTP 429, wait {wait:.1f}s", "stage2")
                await asyncio.sleep(wait)
            elif e.response.status_code == 400:
                log(f"  [{log_prefix}] HTTP 400: {e.response.text[:300]}", "stage2")
                return None
            else:
                wait = _backoff_with_jitter(5, attempt)
                log(f"  [{log_prefix}] HTTP {e.response.status_code}, wait {wait:.1f}s", "stage2")
                await asyncio.sleep(wait)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = _backoff_with_jitter(2, attempt)
                log(f"  [{log_prefix}] error {type(e).__name__}: {e!r}, wait {wait:.1f}s", "stage2")
                await asyncio.sleep(wait)
            else:
                log(f"  [{log_prefix}] failed after {max_retries} retries: {e!r}", "stage2")
                return None
    return None


# ── CoT cleaning ──────────────────────────────────────────────────────────────


def clean_cot(cot: str) -> str:
    cot = re.sub(r"\n*\*\*Final Answer[:\s].*$", "", cot, flags=re.DOTALL)
    cot = re.sub(
        r"\n*(?:This|The answer|Therefore,?\s*the answer)\s+corresponds?\s+to\s+option\s+[A-Z]\.?\s*$",
        "",
        cot,
        flags=re.IGNORECASE,
    )
    cot = re.sub(
        r"\n*The correct (?:answer|option) is [A-Z]\.?\s*$",
        "",
        cot,
        flags=re.IGNORECASE,
    )
    return cot.rstrip()


# ── Answer verification prompt ────────────────────────────────────────────────

SYSTEM_VERIFY_ANSWER = (
    "You are a rigorous visual question answering verifier. You will be shown an image, "
    "a question, a chain-of-thought reasoning, and a claimed answer. Your job is to determine "
    "whether the answer is correct based on the image content. Be strict and accurate."
)

PROMPT_VERIFY_ANSWER = (
    "Look at the image and verify whether the following answer is correct.\n\n"
    "Question: {question}\n\n"
    "Options:\n{options}\n\n"
    "Chain-of-thought reasoning:\n{cot}\n\n"
    "Claimed answer: {answer}\n\n"
    "Based on the image, is the claimed answer correct? Respond in strict JSON format:\n"
    '{{\n  "correct": true or false,\n  "reason": "brief explanation"\n}}'
)


# ── Option rewriting prompt ──────────────────────────────────────────────────

SYSTEM_OPTION_REWRITE = (
    "You are an expert question designer. Your task is to generate new challenging "
    "multiple-choice distractors for a question. The distractors must be plausible but "
    "clearly incorrect. They should represent common misconceptions or close-but-wrong "
    "answers. Return ONLY a valid JSON object."
)

PROMPT_OPTION_REWRITE = (
    "Given the following question, correct answer, and original options, "
    "generate {num_distractors} NEW distractor options.\n\n"
    "Question: {question}\n"
    "Correct answer: {answer}\n"
    "Original options: {options}\n\n"
    "Requirements:\n"
    "1. Keep the correct answer EXACTLY as-is (do not modify it)\n"
    "2. Generate exactly {num_distractors} NEW distractor options that are DIFFERENT from the original distractors\n"
    "3. Distractors must be plausible, challenging, and similar in format/length to the correct answer\n"
    "4. Distractors must be clearly incorrect\n"
    "5. Do NOT include trivially wrong or absurd options\n\n"
    'Return a JSON object:\n{{\n  "distractors": ["distractor1", "distractor2", ...]\n}}'
)


# ── Stage 2 pipeline ─────────────────────────────────────────────────────────


class Stage2Pipeline:
    def __init__(
        self,
        stage1_jsonl: str,
        output_dir: str,
        max_concurrent: int = 64,
        max_retries: int = 20,
        api_timeout: float = 1200.0,
        resume: bool = True,
        limit: int = 0,
        n_prompts_default: int = 2,
        gemini_image_endpoint: str = GEMINI_IMAGE_ENDPOINT,
        gemini_image_api_key: str = GEMINI_IMAGE_API_KEY,
    ):
        self.stage1_jsonl = stage1_jsonl
        self.output_dir = Path(output_dir)
        self.images_dir = self.output_dir / "stage2_images"
        self.train_jsonl = self.output_dir / "stage2_train.jsonl"
        self.rejected_jsonl = self.output_dir / "stage2_rejected.jsonl"
        self.selector_cache_path = self.output_dir / "stage2_selector_cache.jsonl"

        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_retries = max_retries
        self.api_timeout = api_timeout
        self.resume = resume
        self.limit = limit
        self.n_prompts_default = n_prompts_default
        self.gemini_image_endpoint = gemini_image_endpoint
        self.gemini_image_api_key = gemini_image_api_key

        self._train_writer: Optional[JsonlWriter] = None
        self._rejected_writer: Optional[JsonlWriter] = None
        self._done_keys: set[str] = set()
        self._selector_cache: Optional[SelectorCache] = None
        self._inner: Optional[httpx.AsyncClient] = None
        self._proxy: Optional[httpx.AsyncClient] = None

    # ── Selector API call ─────────────────────────────────────────────────

    async def _select_plans_via_api(
        self,
        call1: dict,
        stage1_image_b64: str,
        n_prompts: int,
        log_prefix: str,
        image_type_group: str = "natural_photo",
    ) -> list[dict]:
        sys_p, usr_p = _build_selector_prompt(
            call1, n_prompts, image_type_group=image_type_group
        )
        try:
            text = await chat_qwen(
                self._inner,
                QWEN_API_BASE,
                QWEN_API_KEY,
                QWEN_MODEL,
                qwen_vl_messages(
                    sys_p, [(stage1_image_b64, "image/png")], usr_p
                ),  # thinking ON (no no_think)
                max_tokens=8192,
                temperature=0.5,
                max_retries=self.max_retries,
                log_prefix=log_prefix,
            )
        except Exception as e:
            log(
                f"  [{log_prefix}] selector API failed: {type(e).__name__}: {e!r}",
                "stage2",
            )
            return []
        return _parse_selector_output(text, n_prompts)

    # ── CoT generation ────────────────────────────────────────────────────

    async def _generate_cot(
        self,
        image_b64: str,
        question: str,
        options: list[str],
        answer_str: str,
        sample_key: str,
    ) -> Optional[str]:
        prompt = PROMPT_COT_GEN.format(
            question=question,
            options=format_options_multiline(options),
            answer=answer_str,
        )
        messages = qwen_vl_messages(
            SYSTEM_COT_GEN,
            [(image_b64, "image/png")],
            prompt,
        )
        cot = await call_qwen_thinking(
            self._inner,
            messages,
            max_tokens=8000,
            max_retries=self.max_retries,
            log_prefix=f"cot/{sample_key}",
        )
        return cot

    # ── Answer verification ───────────────────────────────────────────────

    async def _verify_answer(
        self,
        image_b64: str,
        question: str,
        options: list[str],
        cot: str,
        answer_str: str,
        sample_key: str,
    ) -> Optional[bool]:
        prompt = PROMPT_VERIFY_ANSWER.format(
            question=question,
            options=format_options_multiline(options),
            cot=cot,
            answer=answer_str,
        )
        messages = qwen_vl_messages(
            SYSTEM_VERIFY_ANSWER,
            [(image_b64, "image/png")],
            prompt,
            no_think=True,
        )
        try:
            text = await chat_qwen(
                self._inner,
                QWEN_API_BASE,
                QWEN_API_KEY,
                QWEN_MODEL,
                messages,
                max_tokens=512,
                temperature=0.1,
                max_retries=self.max_retries,
                log_prefix=f"verify_ans/{sample_key}",
            )
        except Exception as e:
            log(f"  [{sample_key}] verify_answer failed: {e!r}", "stage2")
            return None

        parsed = extract_json(text)
        if not isinstance(parsed, dict):
            return None
        return bool(parsed.get("correct", False))

    # ── Option rewriting ──────────────────────────────────────────────────

    async def _rewrite_options(
        self,
        question: str,
        correct_answer_text: str,
        old_options: list[str],
        sample_key: str,
    ) -> Optional[tuple[list[str], str]]:
        num_distractors = len(old_options) - 1
        if num_distractors < 1:
            num_distractors = 1

        prompt = PROMPT_OPTION_REWRITE.format(
            question=question,
            answer=correct_answer_text,
            options=json.dumps(old_options, ensure_ascii=False),
            num_distractors=num_distractors,
        )
        messages = qwen_text_messages(SYSTEM_OPTION_REWRITE, prompt, no_think=True)

        try:
            text = await chat_qwen(
                self._inner,
                QWEN_API_BASE,
                QWEN_API_KEY,
                QWEN_MODEL,
                messages,
                max_tokens=2048,
                temperature=0.7,
                max_retries=self.max_retries,
                log_prefix=f"opts/{sample_key}",
            )
        except Exception as e:
            log(f"  [{sample_key}] rewrite_options failed: {e!r}", "stage2")
            return None

        parsed = extract_json(text)
        if not isinstance(parsed, dict) or "distractors" not in parsed:
            log(f"  [{sample_key}] bad option rewrite response", "stage2")
            return None

        distractors = parsed["distractors"]
        if not isinstance(distractors, list) or len(distractors) < num_distractors:
            log(
                f"  [{sample_key}] not enough distractors: "
                f"{len(distractors) if isinstance(distractors, list) else 'N/A'}",
                "stage2",
            )
            return None

        distractors = [str(d) for d in distractors[:num_distractors]]
        new_options = [correct_answer_text] + distractors
        random.shuffle(new_options)
        answer_idx = new_options.index(correct_answer_text)
        new_letter = LETTERS[answer_idx]
        return new_options, new_letter

    # ── Process one plan image (verifier + CoT + verify + rewrite) ────────

    async def _process_one_plan_image(
        self,
        stage1_record: dict,
        plan: dict,
        prompt_idx: int,
        seed_idx: int,
        edit_prose: str,
        stage2_bytes: bytes,
        call1: dict,
        image_type_group: str = "natural_photo",
        options: list[str] | None = None,
        answer_letter: str = "",
    ) -> None:
        s1_key = stage1_record["sample_key"]
        sample_key = f"{s1_key}::p{prompt_idx}::s{seed_idx}"
        if sample_key in self._done_keys:
            return

        # Save image
        img_path = self.images_dir / f"{sample_key}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(stage2_bytes)
        stage2_b64 = base64.b64encode(stage2_bytes).decode()

        # Build slot values for template instantiation
        slot_assignments = plan.get("slot_assignments", {}) or {}
        actual_new_answer = stage1_record.get("actual_new_answer", "")
        slot_values: dict = dict(slot_assignments)
        fp = call1.get("frozen_predicate", {}) or {}
        for sname, sdef in (fp.get("slots") or {}).items():
            if isinstance(sdef, dict) and "value" in sdef and sname not in slot_values:
                slot_values[sname] = actual_new_answer or sdef.get("value", "")

        question_template = call1.get("question_template", "") or stage1_record.get(
            "new_question", ""
        )
        answer_template = call1.get("answer_template", "") or actual_new_answer

        if image_type_group in ("table", "chart"):
            final_question = stage1_record.get(
                "new_question", ""
            ) or _instantiate_template(question_template, slot_values)
            final_answer_text = actual_new_answer
        else:
            final_question = _instantiate_template(question_template, slot_values)
            final_answer_text = (
                _instantiate_template(answer_template, slot_values) or actual_new_answer
            )

        # Ensure options list is available
        if options is None:
            options = []
        num_options = len(options)

        # Instantiate verifier_checks
        verifier_checks_raw = fp.get("verifier_checks") or {}
        verifier_checks_inst = {
            k: _instantiate_template(v, slot_values)
            for k, v in verifier_checks_raw.items()
        }
        if not verifier_checks_inst.get("natural_check"):
            type_noun = IMAGE_TYPE_NOUNS.get(image_type_group, "image")
            if image_type_group == "natural_photo":
                verifier_checks_inst["natural_check"] = (
                    "Is this a plausible real photograph (not illustration / surreal / cartoon)?"
                )
            else:
                verifier_checks_inst["natural_check"] = (
                    f"Is this a legible, well-structured {type_noun} (not corrupted or garbled)?"
                )

        # ── Verifier (4 yes/no checks) ───────────────────────────────────
        try:
            sys_p, usr_p = _build_verifier_prompt(verifier_checks_inst)
            ver_raw = await chat_qwen(
                self._inner,
                QWEN_API_BASE,
                QWEN_API_KEY,
                QWEN_MODEL,
                qwen_vl_messages(
                    sys_p, [(stage2_b64, "image/png")], usr_p, no_think=True
                ),
                max_tokens=256,
                temperature=0.0,
                max_retries=self.max_retries,
                log_prefix=f"verify/{sample_key}",
            )
            verifier = _parse_verifier_output(ver_raw)
        except Exception as e:
            await self._rejected_writer.write(
                {
                    "sample_key": sample_key,
                    "derived_from_stage1": s1_key,
                    "reject_reason": "verifier_failed",
                    "error": f"{type(e).__name__}: {e!r}",
                    "created_at": int(time.time()),
                }
            )
            return

        if not verifier["all_pass"]:
            await self._rejected_writer.write(
                {
                    "sample_key": sample_key,
                    "derived_from_stage1": s1_key,
                    "reject_reason": "verifier_check_failed",
                    "verifier_answers": {
                        k: verifier[k] for k in ("a1", "a2", "a3", "a4")
                    },
                    "verifier_raw": verifier["raw"][:1000],
                    "created_at": int(time.time()),
                }
            )
            return

        # ── CoT generation (thinking mode) ───────────────────────────────
        if answer_letter and options:
            answer_display = answer_text(answer_letter, options)
        else:
            answer_display = final_answer_text

        cot_raw = await self._generate_cot(
            stage2_b64,
            final_question,
            options,
            f"{answer_letter}) {answer_display}" if answer_letter else answer_display,
            sample_key,
        )
        if not cot_raw:
            await self._rejected_writer.write(
                {
                    "sample_key": sample_key,
                    "derived_from_stage1": s1_key,
                    "reject_reason": "cot_generation_failed",
                    "created_at": int(time.time()),
                }
            )
            return

        cot = clean_cot(cot_raw)
        if not cot.strip():
            await self._rejected_writer.write(
                {
                    "sample_key": sample_key,
                    "derived_from_stage1": s1_key,
                    "reject_reason": "cot_empty_after_cleaning",
                    "created_at": int(time.time()),
                }
            )
            return

        # ── Verify answer+CoT correctness ────────────────────────────────
        verified = await self._verify_answer(
            stage2_b64,
            final_question,
            options,
            cot,
            f"{answer_letter}) {answer_display}" if answer_letter else answer_display,
            sample_key,
        )
        if verified is False:
            await self._rejected_writer.write(
                {
                    "sample_key": sample_key,
                    "derived_from_stage1": s1_key,
                    "reject_reason": "answer_incorrect",
                    "created_at": int(time.time()),
                }
            )
            return

        # ── Rewrite options ──────────────────────────────────────────────
        if num_options >= 2:
            opt_result = await self._rewrite_options(
                final_question,
                answer_display,
                options,
                sample_key,
            )
            if opt_result is not None:
                new_options, new_letter = opt_result
            else:
                new_options = list(options)
                new_letter = answer_letter
        else:
            new_options = list(options) if options else []
            new_letter = answer_letter

        # ── Build training-format output ─────────────────────────────────
        question_normalized = normalize_question(final_question)
        if "<image>" not in question_normalized:
            question_normalized = "<image>" + question_normalized

        if new_options:
            options_str = format_options_multiline(new_options)
            user_content = f"{question_normalized}\n{options_str}"
        else:
            user_content = question_normalized

        assistant_answer = new_letter if new_letter else answer_display
        assistant_content = f"<think>\n{cot}\n</think>\n<answer>\n{assistant_answer}\n</answer>"

        record = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content},
            ],
            "images": [str(img_path)],
            "sample_key": sample_key,
            "source_stage": "stage2",
            "derived_from_stage1": s1_key,
            "source_id": stage1_record.get("source_id", ""),
            "original_question": stage1_record.get("original_question", ""),
            "original_answer": stage1_record.get("original_answer", ""),
            "stage1_answer_letter": answer_letter,
            "new_answer_letter": new_letter,
            "new_options": new_options,
            "plan": plan,
            "verifier_answers": {k: verifier[k] for k in ("a1", "a2", "a3", "a4")},
            "prompt_idx": prompt_idx,
            "seed_idx": seed_idx,
            "created_at": int(time.time()),
        }
        mmmu_meta = stage1_record.get("mmmu_meta")
        if mmmu_meta:
            record["mmmu_meta"] = mmmu_meta
            record["image_type_group"] = image_type_group

        await self._train_writer.write(record)
        log(f"  [{sample_key}] done | answer={new_letter}", "stage2")

    # ── Process one plan (Call 2b + Call 3b + per-image processing) ────────

    async def _process_plan(
        self,
        stage1_record: dict,
        plan: dict,
        image_type_group: str = "natural_photo",
        options: list[str] | None = None,
        answer_letter: str = "",
    ) -> None:
        s1_key = stage1_record["sample_key"]
        prompt_idx = int(plan.get("plan_id", 0))
        call1 = stage1_record.get("call1_full", {})
        stage1_image_path = Path(stage1_record.get("stage1_image_path", ""))

        # Resume check
        expected_key = f"{s1_key}::p{prompt_idx}::s0"
        if expected_key in self._done_keys:
            log(f"  [{expected_key}] resume: already done, skip plan", "stage2")
            return

        if not stage1_image_path.is_file():
            log(f"  [{s1_key}::p{prompt_idx}] stage1 image not found, skip plan", "stage2")
            return

        try:
            raw_bytes = stage1_image_path.read_bytes()
        except Exception as e:
            log(f"  [{s1_key}::p{prompt_idx}] image read failed: {e!r}", "stage2")
            return
        gpt_bytes, gpt_mime = prepare_image(raw_bytes, max_side=1024, fmt="JPEG")

        # Call 2b: prose
        try:
            sys_p, usr_p = _build_call2b_prompt(
                call1, plan, stage1_record, image_type_group=image_type_group
            )
            edit_prose = await chat_qwen(
                self._inner,
                QWEN_API_BASE,
                QWEN_API_KEY,
                QWEN_MODEL,
                qwen_text_messages(sys_p, usr_p, no_think=True),
                max_tokens=2048,
                temperature=0.6,
                max_retries=self.max_retries,
                log_prefix=f"qwen2b/{s1_key}::p{prompt_idx}",
            )
        except Exception as e:
            await self._rejected_writer.write(
                {
                    "sample_key": f"{s1_key}::p{prompt_idx}::s0",
                    "derived_from_stage1": s1_key,
                    "reject_reason": "call2b_failed",
                    "plan": plan,
                    "error": f"{type(e).__name__}: {e!r}",
                    "created_at": int(time.time()),
                }
            )
            return

        # Call 3b: gemini-3-pro-image-preview (n=1)
        try:
            stage2_imgs = await gemini_image_edit(
                self._inner, self.gemini_image_endpoint, self.gemini_image_api_key,
                gpt_bytes,
                edit_prose,
                n=1,
                image_mime=gpt_mime,
                max_retries=self.max_retries,
                log_prefix=f"gemini3b/{s1_key}::p{prompt_idx}",
            )
        except Exception as e:
            await self._rejected_writer.write(
                {
                    "sample_key": f"{s1_key}::p{prompt_idx}::s0",
                    "derived_from_stage1": s1_key,
                    "reject_reason": "call3b_gemini_image_failed",
                    "edit_prose": edit_prose,
                    "plan": plan,
                    "error": f"{type(e).__name__}: {e!r}",
                    "created_at": int(time.time()),
                }
            )
            return

        if not stage2_imgs:
            await self._rejected_writer.write(
                {
                    "sample_key": f"{s1_key}::p{prompt_idx}::s0",
                    "derived_from_stage1": s1_key,
                    "reject_reason": "call3b_gemini_no_image",
                    "edit_prose": edit_prose,
                    "plan": plan,
                    "created_at": int(time.time()),
                }
            )
            return

        # Process each seed image
        tasks = [
            self._process_one_plan_image(
                stage1_record,
                plan,
                prompt_idx,
                seed_idx,
                edit_prose,
                img,
                call1,
                image_type_group=image_type_group,
                options=options,
                answer_letter=answer_letter,
            )
            for seed_idx, img in enumerate(stage2_imgs)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    # ── Process one stage1 record ─────────────────────────────────────────

    async def _process_one_stage1(self, stage1_record: dict) -> None:
        async with self.semaphore:
            try:
                if not stage1_record.get("quality_pass"):
                    return
                s1_key = stage1_record["sample_key"]

                # Extract metadata (uses mmmu_meta field)
                mmmu_meta = stage1_record.get("mmmu_meta", {}) or {}
                image_type_group = stage1_record.get(
                    "image_type_group",
                    mmmu_meta.get("image_type_group", "natural_photo"),
                )
                options = mmmu_meta.get("options")
                if isinstance(options, str):
                    try:
                        options = json.loads(options.replace("'", '"'))
                    except Exception:
                        options = []
                if not isinstance(options, list):
                    options = []
                answer_letter = mmmu_meta.get("answer_letter", "")

                # Table/chart: 1 plan; others: default (2)
                n_prompts = (
                    1 if image_type_group in ("table", "chart") else self.n_prompts_default
                )

                # Resume: if every prompt already in done keys, skip
                expected_all = [
                    f"{s1_key}::p{p}::s0"
                    for p in range(1, n_prompts + 1)
                ]
                if expected_all and all(k in self._done_keys for k in expected_all):
                    log(f"  [{s1_key}] resume: all {len(expected_all)} variants done, skip", "stage2")
                    return

                call1 = stage1_record.get("call1_full", {}) or {}
                if not call1.get("axis_table") or not call1.get("frozen_predicate"):
                    log(
                        f"  [{s1_key}] missing axis_table/frozen_predicate, skip",
                        "stage2",
                    )
                    return

                # Selector: try cache first
                plans = (
                    self._selector_cache.get(s1_key) if self._selector_cache else None
                )
                if plans is None:
                    stage1_image_path = Path(stage1_record.get("stage1_image_path", ""))
                    if not stage1_image_path.is_file():
                        log(f"  [{s1_key}] stage1 image not found for selector, skip", "stage2")
                        return
                    s1_b64 = base64.b64encode(stage1_image_path.read_bytes()).decode()
                    plans = await self._select_plans_via_api(
                        call1,
                        s1_b64,
                        n_prompts=n_prompts,
                        log_prefix=f"selector/{s1_key}",
                        image_type_group=image_type_group,
                    )
                    if plans and self._selector_cache:
                        await self._selector_cache.put(s1_key, plans)

                if not plans:
                    log(f"  [{s1_key}] selector returned no plans, skip", "stage2")
                    return

                # Renumber plans 1..N
                for i, p in enumerate(plans, start=1):
                    p["plan_id"] = i

                tasks = [
                    self._process_plan(
                        stage1_record,
                        p,
                        image_type_group=image_type_group,
                        options=options,
                        answer_letter=answer_letter,
                    )
                    for p in plans
                ]
                await asyncio.gather(*tasks, return_exceptions=True)

            except Exception as e:
                log(
                    f"  [{stage1_record.get('sample_key', '?')}] UNEXPECTED ERROR: "
                    f"{type(e).__name__}: {e!r}",
                    "stage2",
                )
                log(traceback.format_exc(), "stage2")

    # ── Main run ──────────────────────────────────────────────────────────

    async def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._train_writer = JsonlWriter(self.train_jsonl)
        self._rejected_writer = JsonlWriter(self.rejected_jsonl)
        self._selector_cache = SelectorCache(self.selector_cache_path)
        log(
            f"selector cache: {len(self._selector_cache._mem)} entries loaded",
            "stage2",
        )

        if self.resume:
            self._done_keys = (
                load_done_keys(self.train_jsonl)
                | load_done_keys(self.rejected_jsonl)
            )
            log(f"resume: {len(self._done_keys)} sample_keys already done", "stage2")

        log(f"loading stage1 jsonl: {self.stage1_jsonl}", "stage2")
        stage1_records = [
            r for r in jsonl_iter(self.stage1_jsonl) if r.get("quality_pass")
        ]
        log(f"loaded {len(stage1_records)} quality-pass stage1 records", "stage2")
        if self.limit > 0:
            stage1_records = stage1_records[: self.limit]
            log(f"limit to {self.limit}", "stage2")

        inner, proxy = make_clients(api_timeout=self.api_timeout)
        self._inner, self._proxy = inner, proxy
        try:
            await asyncio.gather(
                *[self._process_one_stage1(r) for r in stage1_records],
                return_exceptions=True,
            )
        finally:
            await inner.aclose()
            await proxy.aclose()

        # Summary
        total_train = total_rejected = 0
        for _ in jsonl_iter(str(self.train_jsonl)):
            total_train += 1
        for _ in jsonl_iter(str(self.rejected_jsonl)):
            total_rejected += 1
        log(
            f"stage2 done. train: {total_train} | rejected: {total_rejected}",
            "stage2",
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="MMMU Stage 2: diversification + CoT + verification + option rewriting"
    )
    parser.add_argument(
        "--stage1_jsonl",
        default=str(OUTPUT_DIR / "stage1_samples.jsonl"),
        help="Path to stage1_samples.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        default=str(OUTPUT_DIR),
        help="Output directory",
    )
    parser.add_argument(
        "--max_concurrent",
        type=int,
        default=MAX_CONCURRENT,
    )
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=2,
        help="Number of diversification plans per stage1 record (tables/charts always 1)",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=int(os.environ.get("MAX_RETRIES", "20")),
    )
    parser.add_argument(
        "--api_timeout",
        type=float,
        default=float(os.environ.get("API_TIMEOUT", "1200")),
    )
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=int(os.environ.get("LIMIT", "0")),
    )
    parser.add_argument("--gemini_image_endpoint", default=os.environ.get("GEMINI_IMAGE_ENDPOINT", GEMINI_IMAGE_ENDPOINT))
    parser.add_argument("--gemini_image_api_key", default=os.environ.get("GEMINI_IMAGE_API_KEY", GEMINI_IMAGE_API_KEY))
    args = parser.parse_args()

    pipe = Stage2Pipeline(
        stage1_jsonl=args.stage1_jsonl,
        output_dir=args.output_dir,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        api_timeout=args.api_timeout,
        resume=not args.no_resume,
        limit=args.limit,
        n_prompts_default=args.n_prompts,
        gemini_image_endpoint=args.gemini_image_endpoint,
        gemini_image_api_key=args.gemini_image_api_key,
    )
    asyncio.run(pipe.run())


if __name__ == "__main__":
    main()
