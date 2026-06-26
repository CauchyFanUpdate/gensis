# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

`zerogen` is an automated pipeline for synthesizing **hard visual-reasoning benchmark items** to evaluate VLMs (Vision-Language Models). It turns short, hand-written "seeds" (scene + question + answer + reasoning lever) into:

1. A fully fleshed-out scene description.
2. A multi-hop / complex-element-extraction question.
3. A prompt suitable for GPT-Image-2 image generation.
4. A generated image.
5. A verification verdict from independent VLMs.

The benchmark intentionally targets multi-hop reasoning, occlusion, fine-grained discrimination, optical effects, and semantic contradictions — i.e., items that strong VLMs should still find difficult.

## Pipeline (each stage is its own Python script, run manually by the user)

**Synthesis pipeline:**
`seed_bank.json` → `stage1_gen.py` (Gemini 3.1 Pro) → `stage1_output.jsonl`
→ `stage2_image.py` (GPT-Image-2 `gpt-image-2-c`) → `images/<sample_id>.png`
→ `stage3_verify.py` (Qwen 3.6 Plus + Gemini 3.1 Pro) → `stage3_output.jsonl`

**Release & evaluation pipeline:**
`stage3_output.jsonl` + `manual_review.jsonl` → `build_release.py` → `release/dataset.json` + `release/images/`
→ `eval_models.py` (multi-model evaluation) → `eval_results/<model>.jsonl` + `eval_results/summary.json`

Stages must be **independently runnable** — one stage's output is the next stage's input on disk; no in-process orchestration. The user invokes each script in sequence.

## Repository Layout

```
seed_bank.json              ← authoritative input: 29 families, 1215 seeds total (b1–b4 four batches)
seeds/                      ← individual seed files by family and batch (e.g., COUNT_b1.json, GRAV_b3.json)

Synthesis pipeline:
  stage1_gen.py             ← Stage 1: per-seed Gemini call → benchmark item
  stage1_output.jsonl       ← Stage 1 output (one record per seed; resume-aware)
  stage2_image.py           ← Stage 2: GPT-Image-2 image generation
  stage2_output.jsonl       ← Stage 2 output (image paths, status; resume-aware)
  stage3_verify.py          ← Stage 3: Qwen + Gemini closed-book verification
  stage3_output.jsonl       ← Stage 3 output (verification verdicts; resume-aware)
  manual_review.jsonl       ← Human curation: sample_id → valid (bool), corrections

Release & evaluation:
  build_release.py          ← Extract 200 valid samples; write release/dataset.json + images/
  release/dataset.json      ← 200 validated samples (MC+OE), options shuffled, labels randomized
  release/images/           ← 200 PNG files (copies from images/)
  eval_models.py            ← Multi-model evaluation on release/dataset.json
  eval_results/<model>.jsonl ← Per-model results (Qwen, Gemini, etc.)
  eval_results/summary.json ← Aggregated accuracy by model and family

Utility & docs:
  drop_both_correct.py      ← Remove 30 "easy" samples (both models correct)
  CLAUDE.md                 ← this file
```

Seeds are generated in four batches (b1–b4):
- **b1–b2**: Initial 200 seeds from original 15 families
- **b3**: 100 seeds for 10 lowest-performing families (targeted hardness increase)
- **b4**: 35 seeds for 7 concept-drift domains (Mechanics, Optics, Fluid, Bio, Astro, Thermo, EM)

`seed_bank.json` aggregates all batches. Each seed has flat fields: `{id, sub_skill, domain, scene, question, answer, lever}`.

## `seed_bank.json` Schema (English-only)

Top level: `{benchmark_name, version, language: "en", translated_from_zh_via, schema, families[]}`.

Each family: `{family_id, family_name, seeds[]}`.

Each seed: `{id, sub_skill, domain, scene, question, answer, lever}`.

- `id` — e.g. `COUNT-001`; the prefix before `-` is the family code (one of `COUNT, SPAT, READ, ATTR, COMP, EXIS, PHYS, GEOM, DOM, TEMP, DOCU, MAPS, VIEW, OPTI, CONS`)
- `sub_skill` — narrow skill (e.g., `occluded_count`, `attr_filter_count`)
- `domain` — scene category (warehouse, fruit stand, aquarium, …)
- `scene` — short canonical scene description (the BACKBONE that Stage 1 elaborates)
- `question` — the seed reference question
- `answer` — the seed reference gold answer
- `lever` — the *reasoning lever*: what makes the question hard. Stage 1 MUST preserve this mechanism.

The 29 families (as of b1–b4):
**Original 15**: `COUNT, SPAT, READ, ATTR, COMP, EXIS, PHYS, GEOM, DOM, TEMP, DOCU, MAPS, VIEW, OPTI, CONS`
**Concept-drift families (b2–b4)**: `ASTRO, BIO, EM, FLUID, GRAV, PHYSGEOM, PHYSMECH, PHYSOPT, THERMO, ODDX, OCRX, DIFF, PUZZ, MECH`

## Conventions for new scripts

- **Environment**: latest conda base. The current scripts depend only on `httpx` (no SDKs).
- **API keys**: hard-coded as constants at the top of each script (per the user's explicit instruction). Do not refactor to env vars or `.env` files unless asked. The keys/endpoints currently in use:
  - Gemini: `<GEMINI_ENDPOINT>`, key `<GEMINI_API_KEY_1>`, model `gemini-3.1-pro-preview`, auth via `api-key` header (Google native format, NOT OpenAI-compatible).
  - Qwen: `<QWEN_ENDPOINT>`, key `<QWEN_API_KEY>`, model `qwen3.6-plus`, OpenAI-compatible (`Bearer`).
  - GPT-Image-2: `<GENERIC_ENDPOINT>/v1`, key `<GPT_IMAGE_EXT_KEY>`, model `gpt-image-2-c`.
- **Gemini thinking mode**: ENABLED by default (do NOT pass `thinkingConfig`). Pass `thinkingConfig: {thinkingBudget: 0}` only if you want to disable it.
- **I/O contract between stages**: append-only JSONL, keyed by `seed_id`. Re-runs append; the script reads existing records and skips already-done `seed_id`s. Failed records are retried on next run.
- **Stage 1 output record shape**: `{seed_id, family_id, status: "ok"|"failed", seed: {<original seed>}, stage1: {<gemini json>} | error: <str>, elapsed_s, created_at}`. Downstream stages should read `record["stage1"]["image_prompt"]` for image generation, and `record["stage1"]["question"]` / `record["stage1"]["answer"]` for verification.
- **Prompt language**: **English everywhere** — all prompts and all model outputs are English.
- **Lever preservation**: Stage 1's prompt explicitly enforces lever preservation; downstream verifiers should re-check by comparing `stage1.test_point_analysis.lever_restated` against the visible image content.

## Release & Evaluation Pipeline

**build_release.py:**
- Reads `manual_review.jsonl` (latest-wins by `sample_id`) and filters `valid=True`
- Outputs `release/dataset.json` (200 samples, pretty-printed with options one per line)
- Copies selected images to `release/images/`
- Schema: `{sample_id, family, qtype, domain, question, image_path, [options, correct_label for MC] / [answer for OE]}`

**eval_models.py:**
- Multi-model evaluation on `release/dataset.json` (200 samples, 79 MC + 121 OE)
- Models: Qwen (VL + text judge), Gemini (VL only)
- MC: rule-based label parsing; OE: Qwen 3.6 Plus as judge (text-only)
- Output: `eval_results/<model>.jsonl` (one record per sample, append-only, resume-aware)
- Summary: `eval_results/summary.json` (accuracy per model, per qtype, per family)
- Temperature: all models at **0.0** (deterministic answers)

**Manual curation:**
- `manual_review.jsonl` is append-only, latest-wins by `sample_id`
- Fields: `sample_id, family_id, qtype, valid (bool), human_verified, notes, [corrections for gold/label]`
- Used by `build_release.py` to filter and apply human corrections

## Open design points (ask the user before deciding)

- Whether stage-3 verification should gate downstream inclusion (auto-reject vs. flag-only).
- Output directory layout for generated images and verification reports.
- Whether to support partial re-runs at the family or seed level.

## You must speak with user in Chinese
