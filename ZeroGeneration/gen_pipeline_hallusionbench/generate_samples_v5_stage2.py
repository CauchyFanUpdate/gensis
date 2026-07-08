#!/usr/bin/env python3
"""
ZeroBench Pipeline v5.1 — Stage 2: Axis-Valuation Diversification (4 prompts × 1 seed)

Per stage1 sample (high-quality basis, with frozen_predicate + axis_table from Call 1):
  Selector [Qwen VL]      → 4 axis valuations (sees stage1 image; samples 4 visually-
                            distant points in axis space; instantiates slot values)
                            Result cached to disk for resume.
  ↓ per valuation (parallel):
    Call 2b [Qwen text]   → diversification edit prose (consumes axis_values + slot_assignments)
    Call 3b [GPT-Image-2] → 1 stage2 image (n=1; multi-seed disabled — diversity comes from selector)
    Verifier [Qwen VL]    → 4 yes/no checks: CLASS / RELATION / READABILITY / NATURAL
                            (NOT answer-correctness — we don't ask the model to solve hard ZB questions)
    Call 4 [Gemini VL]    → CoT, anchored on final_answer (from slot-instantiated answer_template)

输出:
  _archive/output_v5/stage2_samples.jsonl
  _archive/output_v5/stage2_images/{sample_key}.png
  _archive/output_v5/stage2_selector_cache.jsonl   (selector outputs cached by stage1_key)

Each stage1 → 4 prompts × 1 seed = 4 final candidates
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
from typing import Optional

from _v5_common import (
    UNIVERSAL_AXES,
    chat_gemini_vl, chat_qwen,
    extract_json, gpt_image_edit,
    JsonlWriter, jsonl_iter, load_done_keys, log,
    make_clients, qwen_text_messages, qwen_vl_messages,
)


# ── v5.1: Selector — Qwen-VL with stage1 image, outputs 4 axis valuations ──

def _slot_pool_brief(slots: dict) -> str:
    """One-line summary of each slot for selector prompt."""
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
    """One-line summary per axis for selector prompt."""
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


def _build_selector_prompt(
    call1: dict, n_prompts: int,
) -> tuple[str, str]:
    """
    Build Qwen-VL prompt to produce n_prompts axis valuations + slot assignments.
    Selector sees the stage1 image and the per-image axis_table + slots, outputs N
    valuations as far apart in axis space as possible.
    """
    system = (
        "You are an image augmentation strategy planner. You receive an already-edited image "
        "(Stage 1 output) plus a per-image analysis (frozen_predicate + axis_table). Your job is to "
        "produce N visually-distinct edit plans for Stage 2. Each plan picks one concrete value for "
        "every FREE axis, plus one slot-class for each replaceable slot, such that:\n\n"
        "  HARD constraints:\n"
        "    1. FROZEN axes must NOT be touched. Don't propose values for them.\n"
        "    2. The frozen_predicate's verifier_checks (class/relation/readability/natural) must still "
        "       hold true on the imagined edited image.\n"
        "    3. Stay strictly within the natural-photo distribution. No surreal effects, no illustrations, "
        "       no extreme color filters or weather (heavy fog/blizzard).\n"
        "    4. For slots with replaceability=='locked', do NOT change the slot class (keep default).\n\n"
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
        "Each plan also produces a `diversity_signature` — a short list of 3-5 tags (e.g., "
        "['bg_warehouse', 'surf_metal', 'lighting_cold']) used downstream for cross-plan diversity audit.\n\n"
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
        f"  - slot_assignments: pick a compatible_class for each replaceable class slot (keep locked slots at default).\n"
        f"    Use BOTH singular and plural forms.\n"
        f"  - axis_values: pick a concrete value for every FREE axis (key = axis name; value = single concrete\n"
        f"    instruction string drawn from or inspired by the axis's aggressive_value menu).\n"
        f"  - diversity_signature: 3-5 short tags summarizing this plan's character.\n"
        f"  - rationale: one short sentence on why this plan is answer-safe and visually impactful.\n\n"
        f"Output JSON with EXACTLY this schema:\n"
        f"{{\n"
        f'  "plans": [\n'
        f'    {{\n'
        f'      "plan_id": 1,\n'
        f'      "slot_assignments": {{\n'
        f'        "S": {{"singular": "ceramic vase", "plural": "ceramic vases"}},\n'
        f'        "SURF": {{"singular": "marble table", "plural": "marble tables"}}\n'
        f'      }},\n'
        f'      "axis_values": {{\n'
        f'        "background_setting": "minimalist white studio with seamless backdrop",\n'
        f'        "lighting_quality":   "cool 5500K daylight from the upper-left",\n'
        f'        "...":                "..."\n'
        f'      }},\n'
        f'      "diversity_signature": ["bg_studio", "lighting_cold", "surf_marble"],\n'
        f'      "rationale": "<one short sentence>"\n'
        f'    }}\n'
        f'  ]\n'
        f"}}\n\n"
        f"Generate exactly {n_prompts} plans. Output ONLY the JSON object."
    )
    return system, user_text


def _parse_selector_output(text: str, n_prompts: int) -> list[dict]:
    """
    Parse selector JSON output. Returns list of plans (dicts with slot_assignments, axis_values,
    diversity_signature, rationale). Returns [] on parse failure.
    Truncates to n_prompts if more produced; tolerates fewer (caller will warn).
    """
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
        # Normalize slot_assignments: ensure each value has singular+plural strings
        norm_sa: dict[str, dict] = {}
        for k, v in sa.items():
            if isinstance(v, dict) and "singular" in v and "plural" in v:
                norm_sa[k] = {"singular": str(v["singular"]), "plural": str(v["plural"])}
            elif isinstance(v, str):
                # Fallback: model gave just a string, use as both
                norm_sa[k] = {"singular": v, "plural": v}
        norm_av = {str(k): str(v) for k, v in av.items() if isinstance(v, (str, int, float))}
        sig = p.get("diversity_signature", []) or []
        if isinstance(sig, str):
            sig = [sig]
        sig = [str(x) for x in sig if x]
        out.append({
            "plan_id":            int(p.get("plan_id", len(out) + 1)),
            "slot_assignments":   norm_sa,
            "axis_values":        norm_av,
            "diversity_signature": sig,
            "rationale":          str(p.get("rationale", "")),
        })
    return out[:n_prompts] if len(out) > n_prompts else out


def _instantiate_template(template: str, slot_values: dict) -> str:
    """
    Substitute <S>, <S_plural>, <S_singular>, <SURF>, etc. in template with slot values.
    `slot_values` maps slot name (without angle brackets) → {singular, plural} dict
    or, for value slots, to a string.
    """
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


# ── v5.1: Selector cache (resume-friendly even with high temp) ─────────────

class SelectorCache:
    """Disk-backed cache for selector outputs. Key = stage1 sample_key."""

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
                f.write(json.dumps({
                    "stage1_key": stage1_key,
                    "plans":      plans,
                    "created_at": int(time.time()),
                }, ensure_ascii=False) + "\n")


# ── v5.1: Call 2b — Qwen text prose, consumes axis_values + slot_assignments ──

# Banned softening words; if model uses these the edit comes out weak.
_PROSE_BANNED_WORDS = (
    "subtle", "subtly", "slightly", "a touch of", "a hint of", "hints of",
    "gentle", "softly", "lightly", "mild", "mildly",
)

# Encouraged amplification words; encourage strong execution.
_PROSE_PUSH_WORDS = (
    "completely", "entirely", "wholly", "dramatically", "fully", "thoroughly", "transformed into",
)


def _build_call2b_prompt(
    call1: dict,
    plan: dict,            # {slot_assignments, axis_values, diversity_signature, rationale}
    stage1_record: dict,
) -> tuple[str, str]:
    """
    Build Qwen text prompt for Stage 2 axis-driven prose. Consumes a single Selector plan
    (slot_assignments + axis_values) and produces ONE flowing paragraph for GPT-Image-2.

    v5.1 changes vs v5:
      - Input is per-axis values + slot class names (not opaque div specs)
      - Output must mention every FREE axis at least once with a concrete adjective
      - Banned softening words; encouraged amplification words
      - Naturalness clause vs surreal/illustration
      - Frozen-predicate must_preserve copied verbatim into prose tail
    """
    fp = call1.get("frozen_predicate", {}) or {}
    must_preserve = fp.get("must_preserve", []) or []
    answer_function = fp.get("answer_function", "")

    system = (
        "You write image editing instructions for an AI image editor (GPT-Image-2). "
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
        "  - The result image must look obviously different from the input image — not 'the same scene with a\n"
        "    filter'. Substantive change on every free axis is the goal.\n\n"
        "NATURAL-PHOTO RULE (very important):\n"
        "  - The result MUST look like a real photograph from a normal camera in natural conditions.\n"
        "  - NO illustration / cartoon / cel-shaded / surreal style.\n"
        "  - NO dramatic single-color stage lighting, no extreme deep-blue/orange/purple color filters.\n"
        "  - NO surreal weather (heavy fog, blizzard, sandstorm). Outdoor weather should be normal: clear / overcast / golden hour.\n\n"
        "FROZEN PREDICATE (the test point of the question):\n"
        "  - You MUST keep the frozen_predicate intact pixel-truth-wise. The must_preserve list will appear\n"
        "    at the end of the user message — paraphrase the constraints into the closing of your paragraph\n"
        "    using positive language ('the scene still shows ...').\n"
        "  - Do NOT mention the question or the answer in the prose. The image editor doesn't need to know.\n"
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
        f"BASE SCENE (Stage 1 already edited image — your prose REPLACES this with a new visual style;\n"
        f"the editor will look at the input image but you instruct it to repaint with the new axis values):\n"
        f"  {call1.get('scene_description', '')}\n\n"
        f"AXIS VALUES + SLOT CLASSES TO APPLY (every entry must appear in your prose):\n"
        f"{plan_block}\n\n"
        f"FROZEN PREDICATE — MUST be preserved (answer logic = {answer_function!r}):\n"
        f"{must_block}\n\n"
        f"Now write the single flowing paragraph for GPT-Image-2. The paragraph should:\n"
        f"  1. Open by describing the new background_setting and surface, with the slot subjects in place.\n"
        f"  2. Cover every FREE axis in the list with a concrete adjective/phrase (no softeners).\n"
        f"  3. Close with a positive paraphrase of the must_preserve constraints (one sentence, woven in naturally).\n"
        f"Output ONLY the paragraph, no other text."
    )
    return system, user_text


# ── v5.1: Verifier — single Qwen-VL call, 4 sub-Q (test-point preservation) ──
#
# IMPORTANT design point: Verifier does NOT verify the answer's numeric correctness.
# ZeroBench is hard precisely because models can't reliably compute these answers — so
# asking "How many <S>? must == N" would be brittle (false rejects on good images, false
# accepts on bad ones). Instead, Verifier only checks whether the test point is still
# answerable (CLASS / RELATION / READABILITY / NATURAL — all yes/no recognition tasks
# that VL is reliable on).
#
# Trust assumption: Stage 1 Modifier already verified actual_new_answer post-edit. Stage 2
# axis_table locks subject_count as frozen. Call 2b prose explicitly states "exactly N <S>".
# We trust this construction; we don't ask VL to count again.

def _build_verifier_prompt(verifier_checks_instantiated: dict) -> tuple[str, str]:
    """
    Build Qwen-VL Verifier prompt. `verifier_checks_instantiated` is a dict with the 4
    yes/no questions already filled with concrete slot values.
    """
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
    """
    Parse Verifier output into {a1, a2, a3, a4, all_pass, raw}.
    """
    out = {"a1": "", "a2": "", "a3": "", "a4": "", "all_pass": False, "raw": text}
    for i in (1, 2, 3, 4):
        m = re.search(rf"A{i}\s*:\s*(yes|no)", text, re.IGNORECASE)
        if m:
            out[f"a{i}"] = m.group(1).strip().lower()
    out["all_pass"] = all(out[f"a{i}"] == "yes" for i in (1, 2, 3, 4))
    return out


# ── Call 4: Gemini VL CoT (fluent + accurate, no fixed format) ─────────────

def _build_call4_prompt(stage2_image_b64: str, call1: dict, final_question: str, final_answer: str) -> tuple[str, str]:
    system = (
        "You are a careful visual reasoning assistant. You produce a brief, accurate chain of thought "
        "for a visual question, ending with the final answer.\n\n"
        "Style requirements (important):\n"
        "  - Write in natural, flowing English (not numbered steps, not bullet lists)\n"
        "  - Be concise (3-6 short sentences total)\n"
        "  - Be accurate — your reasoning must lead to the supplied final answer\n"
        "  - You may, if helpful, briefly note distractors or features that distinguish the target. "
        "Do not force this; only mention if it makes the reasoning clearer.\n"
        "  - End with the line: **Final Answer: <answer>**\n"
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


# ── Stage 2 main pipeline ───────────────────────────────────────────────────

class Stage2Pipeline:
    def __init__(
        self,
        stage1_jsonl: str,
        output_dir: str,
        gemini_endpoint: str,
        gemini_api_key: str,
        gemini_model: str,
        qwen_api_base: str,
        qwen_api_key: str,
        qwen_model: str,
        gpt_image_endpoint: str,
        gpt_image_api_key: str,
        max_concurrent: int = 16,
        max_retries: int = 6,           # v5.1: bumped 4→6
        api_timeout: float = 1200.0,    # v5.1: bumped 480→1200
        resume: bool = True,
        limit: int = 0,
        n_prompts_per_stage1: int = 4,  # v5.1: bumped 2→4 (selector handles diversity)
        n_seeds_per_prompt: int = 1,    # v5.1: dropped 2→1 (multi-seed has low variance for image-edit)
    ):
        self.stage1_jsonl = stage1_jsonl
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
        self.n_prompts = n_prompts_per_stage1
        self.n_seeds = n_seeds_per_prompt   # v5.1 default 1; kept for backwards-compat

        self._writer: Optional[JsonlWriter] = None
        self._done_keys: set[str] = set()
        self._selector_cache: Optional[SelectorCache] = None
        self._inner = None
        self._proxy = None

    async def _process_one_plan_image(
        self, stage1_record: dict, plan: dict, prompt_idx: int, seed_idx: int,
        edit_prose: str, stage2_bytes: bytes, call1: dict,
        sub_b64s: list[tuple[str, str]] = None,
    ) -> None:
        if sub_b64s is None:
            sub_b64s = []
        """
        v5.1: process one (plan, image) tuple → Verifier → CoT → write success/reject.
        sample_key = "{s1_key}::p{prompt_idx}::s{seed_idx}" (s_idx kept for back-compat,
        usually 0 since n_seeds=1).
        """
        s1_key = stage1_record["sample_key"]
        sample_key = f"{s1_key}::p{prompt_idx}::s{seed_idx}"
        if sample_key in self._done_keys:
            return

        # Save image
        img_path = self.images_dir / f"{sample_key}.png"
        img_path.parent.mkdir(parents=True, exist_ok=True)
        img_path.write_bytes(stage2_bytes)
        stage2_b64 = base64.b64encode(stage2_bytes).decode()

        # Instantiate slots → final question/answer/verifier_checks
        slot_assignments = plan.get("slot_assignments", {}) or {}
        # Numeric slot N gets actual_new_answer from Stage 1 Modifier
        actual_new_answer = stage1_record.get("actual_new_answer", "")
        # Build slot_values dict for templating (combine class slots + N value)
        slot_values: dict = dict(slot_assignments)
        # If frozen_predicate has an N slot, inject actual_new_answer
        fp = call1.get("frozen_predicate", {}) or {}
        for sname, sdef in (fp.get("slots") or {}).items():
            if isinstance(sdef, dict) and "value" in sdef and sname not in slot_values:
                slot_values[sname] = actual_new_answer or sdef.get("value", "")

        question_template = call1.get("question_template", "") or stage1_record.get("new_question", "")
        answer_template   = call1.get("answer_template", "") or actual_new_answer
        final_question = _instantiate_template(question_template, slot_values)
        final_answer   = _instantiate_template(answer_template,   slot_values) or actual_new_answer

        # Instantiate verifier_checks
        verifier_checks_raw = (fp.get("verifier_checks") or {})
        verifier_checks_inst = {
            k: _instantiate_template(v, slot_values) for k, v in verifier_checks_raw.items()
        }
        # Always add a fixed natural_check (override LLM's version with our canonical phrasing)
        if not verifier_checks_inst.get("natural_check"):
            verifier_checks_inst["natural_check"] = (
                "Is this a plausible real photograph (not illustration / surreal / cartoon)?"
            )

        # Verifier (Qwen VL, 4 sub-Q in one call, no_think)
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
                "sample_key": sample_key,
                "derived_from_stage1": s1_key,
                "stage": 2,
                "status": "failed",
                "reject_reason": "verifier_failed",
                "stage2_image_path": str(img_path),
                "edit_prose": edit_prose,
                "plan": plan,
                "verifier_checks_instantiated": verifier_checks_inst,
                "error": f"{type(e).__name__}: {e!r}",
                "created_at": int(time.time()),
            })
            return

        if not verifier["all_pass"]:
            await self._writer.write({
                "sample_key": sample_key,
                "derived_from_stage1": s1_key,
                "stage": 2,
                "status": "rejected",
                "reject_reason": "verifier_check_failed",
                "stage2_image_path": str(img_path),
                "edit_prose": edit_prose,
                "plan": plan,
                "verifier_checks_instantiated": verifier_checks_inst,
                "verifier_answers": {k: verifier[k] for k in ("a1", "a2", "a3", "a4")},
                "verifier_raw": verifier["raw"][:1000],
                "created_at": int(time.time()),
            })
            return

        # Call 4 (Gemini CoT) — anchored on slot-instantiated final_answer
        # Pass [stage2_main] + sub_b64s so CoT sees answer choice images for MCQ questions
        cot_images = [(stage2_b64, "image/png")] + sub_b64s
        try:
            sys_p, usr_p = _build_call4_prompt(stage2_b64, call1, final_question, final_answer)
            cot = await chat_gemini_vl(
                self._inner, self.gemini_endpoint, self.gemini_api_key,
                sys_p, cot_images, usr_p,
                max_tokens=4096, temperature=0.6, disable_thinking=True,
                max_retries=self.max_retries, log_prefix=f"call4/{sample_key}",
            )
        except Exception as e:
            await self._writer.write({
                "sample_key": sample_key,
                "derived_from_stage1": s1_key,
                "stage": 2,
                "status": "failed",
                "reject_reason": "cot_failed",
                "stage2_image_path": str(img_path),
                "edit_prose": edit_prose,
                "plan": plan,
                "final_question": final_question,
                "final_answer": final_answer,
                "verifier_answers": {k: verifier[k] for k in ("a1", "a2", "a3", "a4")},
                "error": f"{type(e).__name__}: {e!r}",
                "created_at": int(time.time()),
            })
            return

        if "**Final Answer:" not in cot:
            cot = cot.rstrip() + f"\n\n**Final Answer: {final_answer}**"

        await self._writer.write({
            "sample_key": sample_key,
            "derived_from_stage1": s1_key,
            "source_id": stage1_record.get("source_id", ""),
            "stage": 2,
            "quality_pass": True,
            "original_question": stage1_record.get("original_question", ""),
            "original_answer": stage1_record.get("original_answer", ""),
            "stage1_question": stage1_record.get("new_question", ""),
            "stage1_answer": actual_new_answer,
            "final_question": final_question,
            "final_answer": final_answer,
            "cot": cot,
            "stage2_image_path": str(img_path),
            "sub_image_paths": stage1_record.get("sub_image_paths") or [],
            "edit_prose": edit_prose,
            "plan": plan,
            "verifier_checks_instantiated": verifier_checks_inst,
            "verifier_answers": {k: verifier[k] for k in ("a1", "a2", "a3", "a4")},
            "prompt_idx": prompt_idx,
            "seed_idx": seed_idx,
            "created_at": int(time.time()),
        })
        log(f"  [{sample_key}] ✓ stage2 done | answer={final_answer}", "stage2")

    async def _process_plan(self, stage1_record: dict, plan: dict, sub_b64s: list[tuple[str, str]] = None) -> None:
        """
        v5.1: process one Selector plan → Call 2b prose → GPT-Image-2 (n=n_seeds)
              → Verifier + Call 4 per seed.
        sub_b64s: choice sub-images from Stage 1, passed to CoT for multi-image questions.
        """
        if sub_b64s is None:
            sub_b64s = []
        s1_key = stage1_record["sample_key"]
        prompt_idx = int(plan.get("plan_id", 0))
        call1 = stage1_record.get("call1_full", {})
        stage1_image_path = Path(stage1_record.get("stage1_image_path", ""))

        # Resume: if all seeds already done, skip
        expected_seed_keys = [f"{s1_key}::p{prompt_idx}::s{i}" for i in range(self.n_seeds)]
        if all(k in self._done_keys for k in expected_seed_keys):
            log(f"  [{s1_key}::p{prompt_idx}] resume: all {self.n_seeds} seeds done, skip plan", "stage2")
            return

        if not stage1_image_path.is_file():
            log(f"  [{s1_key}::p{prompt_idx}] stage1 image not found, skip plan", "stage2")
            return

        from _v5_common import prepare_image
        try:
            raw_bytes = stage1_image_path.read_bytes()
        except Exception as e:
            log(f"  [{s1_key}::p{prompt_idx}] image read failed: {e!r}", "stage2")
            return
        gpt_bytes, gpt_mime = prepare_image(raw_bytes, max_side=1024, fmt="JPEG")

        # Call 2b: prose
        try:
            sys_p, usr_p = _build_call2b_prompt(call1, plan, stage1_record)
            edit_prose = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                qwen_text_messages(sys_p, usr_p, no_think=True),
                max_tokens=4096, temperature=0.6,
                max_retries=self.max_retries, log_prefix=f"qwen2b/{s1_key}::p{prompt_idx}",
            )
        except Exception as e:
            for seed_idx in range(self.n_seeds):
                await self._writer.write({
                    "sample_key": f"{s1_key}::p{prompt_idx}::s{seed_idx}",
                    "derived_from_stage1": s1_key,
                    "stage": 2,
                    "status": "failed",
                    "reject_reason": "call2b_failed",
                    "plan": plan,
                    "error": f"{type(e).__name__}: {e!r}",
                    "created_at": int(time.time()),
                })
            return

        # Call 3b: GPT-Image-2 (n=n_seeds; default 1 in v5.1)
        try:
            stage2_imgs = await gpt_image_edit(
                self._proxy, self.gpt_image_endpoint, self.gpt_image_api_key,
                gpt_bytes, edit_prose, n=self.n_seeds, image_mime=gpt_mime,
                max_retries=self.max_retries, log_prefix=f"gpt3b/{s1_key}::p{prompt_idx}",
            )
        except Exception as e:
            for seed_idx in range(self.n_seeds):
                await self._writer.write({
                    "sample_key": f"{s1_key}::p{prompt_idx}::s{seed_idx}",
                    "derived_from_stage1": s1_key,
                    "stage": 2,
                    "status": "failed",
                    "reject_reason": "call3b_gpt_image_failed",
                    "edit_prose": edit_prose,
                    "plan": plan,
                    "error": f"{type(e).__name__}: {e!r}",
                    "created_at": int(time.time()),
                })
            return

        if not stage2_imgs:
            for seed_idx in range(self.n_seeds):
                await self._writer.write({
                    "sample_key": f"{s1_key}::p{prompt_idx}::s{seed_idx}",
                    "derived_from_stage1": s1_key,
                    "stage": 2,
                    "status": "failed",
                    "reject_reason": "call3b_no_image",
                    "edit_prose": edit_prose,
                    "plan": plan,
                    "created_at": int(time.time()),
                })
            return

        # Verifier + Call 4 per seed (default 1 seed in v5.1)
        tasks = [
            self._process_one_plan_image(stage1_record, plan, prompt_idx, seed_idx, edit_prose, img, call1, sub_b64s)
            for seed_idx, img in enumerate(stage2_imgs)
        ]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _select_plans_via_api(
        self, call1: dict, stage1_image_b64: str, sub_b64s: list[tuple[str, str]],
        n_prompts: int, log_prefix: str,
    ) -> list[dict]:
        """
        Ask Qwen-VL to produce n_prompts axis valuations + slot assignments.
        Passes stage1 image + sub-images (choice options) so selector sees full question context.
        Higher temperature (0.5) for inter-plan diversity; resume safety via disk cache.
        """
        sys_p, usr_p = _build_selector_prompt(call1, n_prompts)
        all_imgs = [(stage1_image_b64, "image/png")] + sub_b64s
        try:
            text = await chat_qwen(
                self._inner, self.qwen_api_base, self.qwen_api_key, self.qwen_model,
                qwen_vl_messages(sys_p, all_imgs, usr_p),  # thinking ON (no /no_think)
                max_tokens=8192, temperature=0.5,
                max_retries=self.max_retries, log_prefix=log_prefix,
            )
        except Exception as e:
            log(f"  [{log_prefix}] selector API failed: {type(e).__name__}: {e!r}", "stage2")
            return []
        return _parse_selector_output(text, n_prompts)

    async def _process_one_stage1(self, stage1_record: dict) -> None:
        async with self.semaphore:
            try:
                if not stage1_record.get("quality_pass"):
                    return  # only operate on quality-pass stage1 records
                s1_key = stage1_record["sample_key"]

                # Resume: if every prompt × seed combination already in jsonl, skip whole stage1 source
                expected_all = [
                    f"{s1_key}::p{p}::s{s}"
                    for p in range(1, self.n_prompts + 1)
                    for s in range(self.n_seeds)
                ]
                if expected_all and all(k in self._done_keys for k in expected_all):
                    log(f"  [{s1_key}] resume: all {len(expected_all)} variants done, skip", "stage2")
                    return

                call1 = stage1_record.get("call1_full", {}) or {}
                # v5.1: schema requires axis_table + frozen_predicate from Stage 1
                if not call1.get("axis_table") or not call1.get("frozen_predicate"):
                    log(f"  [{s1_key}] missing axis_table/frozen_predicate (old-schema stage1?), skip", "stage2")
                    return

                # Load sub-images (answer choice images saved by Stage 1)
                sub_b64s: list[tuple[str, str]] = []
                for sp in (stage1_record.get("sub_image_paths") or []):
                    p = Path(sp)
                    if p.is_file():
                        sub_b64s.append((base64.b64encode(p.read_bytes()).decode(), "image/png"))

                # Selector: try cache first (resume-friendly)
                plans = self._selector_cache.get(s1_key) if self._selector_cache else None
                if plans is None:
                    # Read stage1 image as base64 for selector VL call
                    stage1_image_path = Path(stage1_record.get("stage1_image_path", ""))
                    if not stage1_image_path.is_file():
                        log(f"  [{s1_key}] stage1 image not found for selector, skip", "stage2")
                        return
                    s1_b64 = base64.b64encode(stage1_image_path.read_bytes()).decode()
                    plans = await self._select_plans_via_api(
                        call1, s1_b64, sub_b64s, n_prompts=self.n_prompts,
                        log_prefix=f"selector/{s1_key}",
                    )
                    if plans and self._selector_cache:
                        await self._selector_cache.put(s1_key, plans)

                if not plans:
                    log(f"  [{s1_key}] selector returned no plans, skip stage1 record", "stage2")
                    return
                # Renumber plans 1..N (in case the model dropped plan_ids or used 0-indexed)
                for i, p in enumerate(plans, start=1):
                    p["plan_id"] = i

                tasks = [self._process_plan(stage1_record, p, sub_b64s) for p in plans]
                await asyncio.gather(*tasks, return_exceptions=True)
            except Exception as e:
                log(f"  [{stage1_record.get('sample_key', '?')}] UNEXPECTED ERROR: {type(e).__name__}: {e!r}", "stage2")
                log(traceback.format_exc(), "stage2")

    async def run(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self._writer = JsonlWriter(self.output_jsonl)
        self._selector_cache = SelectorCache(self.selector_cache_path)
        log(f"selector cache: {len(self._selector_cache._mem)} entries loaded from {self.selector_cache_path}", "stage2")

        if self.resume:
            self._done_keys = load_done_keys(self.output_jsonl)
            log(f"resume: {len(self._done_keys)} sample_keys already done", "stage2")

        log(f"loading stage1 jsonl: {self.stage1_jsonl}", "stage2")
        stage1_records = [r for r in jsonl_iter(self.stage1_jsonl) if r.get("quality_pass")]
        log(f"loaded {len(stage1_records)} quality-pass stage1 records", "stage2")
        if self.limit > 0:
            stage1_records = stage1_records[: self.limit]
            log(f"limit to {self.limit}", "stage2")

        inner, proxy = make_clients(api_timeout=self.api_timeout)
        self._inner, self._proxy = inner, proxy
        try:
            await asyncio.gather(*[self._process_one_stage1(r) for r in stage1_records], return_exceptions=True)
        finally:
            await inner.aclose()
            await proxy.aclose()

        # Summary
        total = passed = rejected = failed = 0
        for rec in jsonl_iter(str(self.output_jsonl)):
            total += 1
            if rec.get("quality_pass"):
                passed += 1
            elif rec.get("status") == "rejected":
                rejected += 1
            elif rec.get("status") == "failed":
                failed += 1
        log(f"stage2 done. total: {total} | passed: {passed} | rejected: {rejected} | failed: {failed}", "stage2")
        log(f"output jsonl: {self.output_jsonl}", "stage2")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ZeroBench v5 Stage 2: cosmetic diversification + Recheck + CoT")
    parser.add_argument("--stage1_jsonl", required=True, help="Path to stage1_samples.jsonl")
    parser.add_argument("--output_dir", default="zerobench_pipeline/_archive/output_v5")
    # Gemini VL
    parser.add_argument("--gemini_endpoint", default=os.environ.get("GEMINI_VL_ENDPOINT", "https://runway.devops.rednote.life/openai/google/v1:generateContent"))
    parser.add_argument("--gemini_api_key", default=os.environ.get("GEMINI_VL_API_KEY", ""))
    parser.add_argument("--gemini_model", default=os.environ.get("GEMINI_VL_MODEL", "gemini-3.1-pro-preview"))
    # Qwen
    parser.add_argument("--qwen_api_base", default=os.environ.get("QWEN_API_BASE", "https://runway.devops.xiaohongshu.com/openai/qwen/v1"))
    parser.add_argument("--qwen_api_key", default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--qwen_model", default=os.environ.get("QWEN_MODEL", "qwen3.6-plus"))
    # GPT-Image-2
    parser.add_argument("--gpt_image_endpoint", default=os.environ.get("GPT_IMAGE_ENDPOINT", "https://runway.devops.rednote.life"))
    parser.add_argument("--gpt_image_api_key", default=os.environ.get("GPT_IMAGE_API_KEY", ""))
    # Run
    parser.add_argument("--max_concurrent", type=int, default=int(os.environ.get("MAX_CONCURRENT", "16")))
    parser.add_argument("--n_prompts_per_stage1", type=int, default=4)   # v5.1: 2→4
    parser.add_argument("--n_seeds_per_prompt",  type=int, default=1)    # v5.1: 2→1 (multi-seed has low variance)
    parser.add_argument("--max_retries", type=int, default=int(os.environ.get("MAX_RETRIES", "6")))
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

    pipe = Stage2Pipeline(
        stage1_jsonl=args.stage1_jsonl,
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
        n_prompts_per_stage1=args.n_prompts_per_stage1,
        n_seeds_per_prompt=args.n_seeds_per_prompt,
        max_retries=args.max_retries,
        api_timeout=args.api_timeout,
        resume=not args.no_resume,
        limit=args.limit,
    )
    asyncio.run(pipe.run())


if __name__ == "__main__":
    main()
