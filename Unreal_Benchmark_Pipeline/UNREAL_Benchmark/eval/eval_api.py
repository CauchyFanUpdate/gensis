#!/usr/bin/env python3
"""
Evaluate UNREAL_Benchmark (600 samples) with closed-source VLMs.
Extract & judge pipeline aligned with VLMEvalKit.

Input  : ../Unreal_Benchmark.jsonl  (fields: id, qtype mc/oe, domain,
         subdomain, question, answer, image, options)
Images : ../images/<id>.png
Output : results/close/eval_<model_slug>.jsonl  (resume-aware by id)

Pipeline:
  1. VL inference — image + question (MathVision-style Hint embedded) → raw
  2. Extract & judge — MC: heuristic regex + GPT fallback → letter match
                      — OE: Qwen verifier one-step
  3. Stats per domain/subdomain

Resume: existing records in the output file with non-empty vl_response
        are kept; only the remaining ids are inferred.

Usage:
  python eval_api.py --model gemini-3.1-pro-preview
  python eval_api.py --model claude-opus-4-7
  python eval_api.py --model gpt-5.5 --phase 1
"""

import asyncio, base64, json, re, time, argparse, copy, string
from contextlib import asynccontextmanager
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
INPUT_PATH = BENCH_DIR / "Unreal_Benchmark.jsonl"
IMAGE_DIR = BENCH_DIR / "images"
RESULTS_DIR = HERE / "results" / "close"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Gemini: Google generateContent format
GEMINI_ENDPOINT = "<GEMINI_ENDPOINT>"
GEMINI_KEYS = [os.environ.get("GEMINI_API_KEY_1", ""), os.environ.get("GEMINI_API_KEY_2", "")]

# Qwen (VL inference + text-only judge/extract)
QWEN_ENDPOINT = "<QWEN_ENDPOINT>"
QWEN_KEYS = [os.environ.get("QWEN_API_KEY", ""), os.environ.get("QWEN_API_KEY_2", "")]

# GPT / Claude / Doubao / Kimi (OpenAI-compatible proxy)
GENERIC_ENDPOINT = "https://api.gpt.ge"
GENERIC_KEY = os.environ.get("GENERIC_API_KEY", "")

VL_CONCURRENCY = 128
QWEN_CONCURRENCY = 128
REQUEST_TIMEOUT = 120.0
REQUEST_TIMEOUT_VL = 600.0
MAX_RETRIES = 6

MAX_OUTPUT_TOKENS_GEMINI = 16384
MAX_OUTPUT_TOKENS_QWEN = 8192
MAX_OUTPUT_TOKENS_VERIFIER = 512

QWEN_MODEL = "qwen3.6-plus"

# id prefix → domain category (used for hint style)
PREFIX_TO_DOMAIN = {
    "MATH": "Mathematics",
    "PHYS": "Physics & Engineering",
    "REAS": "Reasoning & Logic",
    "PERC": "Perception",
    "NATS": "Natural Science",
}

# ── Verifier prompt (aligned with VLMEvalKit) ──────────────────────────────

VERIFIER_SYSTEM_EN = (
    "You are a strict grader for visual reasoning answers. "
    "Decide whether a candidate answer matches the gold answer. "
    "Compare ONLY the candidate's final answer — IGNORE any errors in reasoning steps. "
    "A candidate is CORRECT iff it conveys exactly the same information as the gold "
    "(synonyms, equivalent numerals, equivalent units, equivalent mathematical expressions are OK; "
    "missing parts, extra incorrect information, off-by-one numbers, or contradictions are NOT OK). "
    "If the candidate's answer is invalid (incomplete, repetitive, refusal to answer, or irrelevant), output false. "
    "Output STRICT JSON ONLY (no prose outside the JSON, no markdown fences):\n"
    '{"correct": <true|false>, "reasoning": "<one short sentence>"}'
)


# ── helpers ────────────────────────────────────────────────────────────────


def is_generic_model(model: str) -> bool:
    m = model.lower()
    return any(
        k in m
        for k in (
            "gpt-5",
            "claude-opus-4-7",
            "claude-sonnet-4-6",
            "doubao-seed-1-8",
            "kimi-k2.6",
        )
    )


def read_jsonl(path):
    recs = []
    if not path.exists():
        return recs
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    return recs


def load_image_b64(filename):
    p = IMAGE_DIR / filename
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return base64.b64encode(f.read()).decode()


def domain_of(record):
    """Resolve domain from record's 'domain' field, falling back to id prefix."""
    d = record.get("domain")
    if d:
        return d
    return PREFIX_TO_DOMAIN.get(record.get("id", "")[:4], "")


def extract_json_strict(text):
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.+?)\s*```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    start = text.find("{")
    if start < 0:
        raise json.JSONDecodeError("no {", text, 0)
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : i + 1])
    raise json.JSONDecodeError("unbalanced", text, 0)


class QwenKeyPool:
    def __init__(self, keys, limit):
        self._keys = keys
        self._sems = [asyncio.Semaphore(limit) for _ in keys]

    @asynccontextmanager
    async def use_key(self):
        while True:
            for i, sem in enumerate(self._sems):
                if not sem.locked():
                    async with sem:
                        yield self._keys[i]
                    return
            await asyncio.sleep(0.1)


# ── Question formatting with MathVision-style Hint ─────────────────────────


def _format_options(options):
    if not options:
        return ""
    if isinstance(options, dict):
        return "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))
    if isinstance(options, list):
        lines = []
        for opt in options:
            label = opt.get("label", "?")
            parts = opt.get("parts", [])
            lines.append(f"{label}. " + " / ".join(parts))
        return "\n".join(lines)
    return ""


def format_question_with_hint(record):
    """
    Hint style chosen by id prefix (= domain):
      PHYS  → short answer (4-word style)
      MATH  → "only the final answer"
      REAS / PERC / NATS → "concisely"
    MC items share two hint variants (PHYS uses 'ONLY single capital letter').
    """
    qtype = record.get("qtype", "oe")
    question = record.get("question", "")
    options = record.get("options")
    prefix = record.get("id", "")[:4]

    is_phys = prefix == "PHYS"
    is_math = prefix == "MATH"

    if qtype == "mc":
        opts_text = _format_options(options)
        if is_phys:
            hint = (
                "Hint: Please reason through the question and provide ONLY the single "
                "capital letter of the correct option at the end."
            )
        else:
            hint = (
                "Hint: Please reason through the question and provide only the letter "
                "of the correct option at the end."
            )
        return f"{hint}\nQuestion: {question}\n\nOptions:\n{opts_text}"

    if is_phys:
        hint = (
            "Hint: Please reason through the question and provide the final answer "
            "in at most 4 words at the end."
        )
    elif is_math:
        hint = (
            "Hint: Please reason through the question and provide only the final "
            "answer at the end."
        )
    else:
        hint = (
            "Hint: Please reason through the question and provide the final "
            "answer concisely at the end."
        )
    return f"{hint}\nQuestion: {question}"


# ── MC extraction: heuristic regex (ported from VLMEvalKit) ────────────────


def can_infer_option(answer, choices):
    if "Failed to obtain answer via API" in answer:
        return False
    reject_to_answer = [
        "Sorry, I can't help with images of people yet.",
        "I can't process this file.",
        "I'm sorry, but without the image provided",
        "Cannot determine the answer",
    ]
    for err in reject_to_answer:
        if err in answer:
            return "Z"

    def count_choice(splits, choices, prefix="", suffix=""):
        return sum(1 for c in choices if prefix + c + suffix in splits)

    answer_mod = copy.copy(answer)
    for c in ".()[],:;!*#{}":
        answer_mod = answer_mod.replace(c, " ")
    splits = [x.strip() for x in answer_mod.split()]
    count = count_choice(splits, choices)
    if count == 1:
        for ch in choices:
            if ch in splits:
                return ch
    elif count == 0 and count_choice(splits, {"Z", ""}) == 1:
        return "Z"
    return False


def can_infer_text(answer, choices):
    answer = answer.lower()
    if len(answer) > 2 * sum(len(str(v)) for v in choices.values()):
        return False
    assert isinstance(choices, dict)
    for k in choices:
        assert k in string.ascii_uppercase
        choices[k] = str(choices[k]).lower()
    cands = [k for k in choices if choices[k] in answer]
    if len(cands) == 1:
        return cands[0]
    return False


def can_infer(answer, choices):
    answer = str(answer)
    copt = can_infer_option(answer, choices)
    return copt if copt else can_infer_text(answer, choices)


def build_extract_prompt(question, options_text, prediction):
    return (
        "You are an AI assistant who will help me to match "
        "an answer with several options of a single-choice question. "
        "You are provided with a question, several options, and an answer, "
        "and you need to find which option is most similar to the answer. "
        "If the meaning of all options are significantly different from the answer, output Z. "
        "Your should output a single uppercase character in A, B, C, D (if they are valid options), and Z.\n"
        "Example 1:\n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: a cute teddy bear\nYour output: A\n"
        "Example 2:\n"
        "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
        "Answer: Spider\nYour output: Z\n"
        "Example 3:\n"
        f"Question: {question}\nOptions: {options_text}\n"
        f"Answer: {prediction}\nYour output: "
    )


def build_choices_dict(options):
    if isinstance(options, dict):
        return {k: str(v) for k, v in options.items()}
    if isinstance(options, list):
        return {opt.get("label", "?"): " / ".join(opt.get("parts", [])) for opt in options}
    return {}


# ── Phase 2 helpers ────────────────────────────────────────────────────────


async def call_qwen_text(client, key_pool, system_prompt, user_text, max_tokens=512):
    url = f"{QWEN_ENDPOINT.rstrip('/')}/chat/completions"
    payload = {
        "model": QWEN_MODEL,
        "messages": [
            {"role": "system", "content": "/no_think\n" + system_prompt},
            {"role": "user", "content": user_text},
        ],
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    for attempt in range(MAX_RETRIES):
        try:
            async with key_pool.use_key() as api_key:
                resp = await client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
                resp.raise_for_status()
            body = resp.json()
            if "choices" not in body or not body["choices"]:
                raise RuntimeError(f"no choices: {str(body)[:200]}")
            return body["choices"][0]["message"]["content"].strip()
        except httpx.HTTPStatusError as e:
            wait = min(5 * (attempt + 1), 30) if e.response.status_code == 429 else min(2 ** attempt, 3)
            await asyncio.sleep(wait)
        except Exception:
            await asyncio.sleep(min(2 ** attempt, 10))
    return ""


async def extract_answer_from_item(client, key_pool, record, response):
    question = record.get("question", "")
    options = record.get("options")
    choices = build_choices_dict(options)
    option_str = _format_options(options)
    ret = can_infer(response, choices)
    if ret:
        return ret
    prompt = build_extract_prompt(question, option_str, response)
    sysmsg = (
        "You are an AI assistant that matches an answer to multiple-choice options. "
        "Output ONLY a single uppercase letter (A-F) or Z."
    )
    ans = await call_qwen_text(client, key_pool, sysmsg, prompt, max_tokens=32)
    ret = can_infer(ans, choices)
    if ret:
        return ret
    import random
    return random.choice(list(choices.keys()) + ["Z"])


async def judge_oe_verifier(client, key_pool, question, gold, response):
    user_text = (
        f"Question:\n{question}\n\n"
        f"Gold answer:\n{gold}\n\n"
        f"Candidate response:\n{response}\n\n"
        "Output the JSON judgment."
    )
    raw = await call_qwen_text(
        client, key_pool, VERIFIER_SYSTEM_EN, user_text,
        max_tokens=MAX_OUTPUT_TOKENS_VERIFIER,
    )
    if not raw:
        return {"correct": None, "verifier_raw": raw, "judge_error": "empty verifier response"}
    try:
        data = extract_json_strict(raw)
    except Exception as e:
        return {"correct": None, "verifier_raw": raw, "judge_error": str(e)}
    cor = data.get("correct")
    if isinstance(cor, str):
        cor = cor.strip().lower() in ("true", "yes", "y", "1")
    return {
        "correct": bool(cor) if cor is not None else None,
        "verifier_raw": raw,
        "verifier_reasoning": data.get("reasoning"),
    }


# ── Phase 1: VL inference ──────────────────────────────────────────────────


def _gemini_text(body):
    cands = body.get("candidates", [])
    if not cands:
        return ""
    parts = cands[0].get("content", {}).get("parts", [])
    return "".join(
        p.get("text", "") for p in parts if "text" in p and not p.get("thought")
    ).strip()


async def vl_infer(client, sem, record, model, key_idx):
    question = format_question_with_hint(record)
    img_b64 = load_image_b64(record["image"])
    if not img_b64:
        return record["id"], None, "NO_IMAGE"
    ext = record["image"].rsplit(".", 1)[-1].lower()
    mime = "image/png" if ext == "png" else "image/jpeg"

    is_gemini = "gemini" in model.lower()
    is_generic = is_generic_model(model)
    max_tok = max(MAX_OUTPUT_TOKENS_GEMINI if is_gemini else MAX_OUTPUT_TOKENS_QWEN, 8192)

    if is_gemini:
        payload = {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": mime, "data": img_b64}},
                    {"text": question},
                ],
            }],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": max_tok,
                "topP": 0.95,
            },
        }
        headers = {
            "Content-Type": "application/json",
            "api-key": GEMINI_KEYS[key_idx % len(GEMINI_KEYS)],
        }
        endpoint = GEMINI_ENDPOINT
    elif is_generic:
        api_model = model
        if model.lower() == "doubao-seed-1-8":
            api_model = "doubao-seed-1-8-251228"
        payload = {
            "model": api_model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    {"type": "text", "text": question},
                ],
            }],
            "max_tokens": max_tok,
            "temperature": 0.0,
        }
        m_lower = model.lower()
        if "claude" in m_lower:
            payload["thinking"] = {"type": "enabled", "budget_tokens": 8000}
            payload["temperature"] = 1.0
            payload["max_tokens"] = max_tok
        elif "gpt" in m_lower:
            payload["reasoning_effort"] = "high"
            payload["temperature"] = 1.0
            payload["max_completion_tokens"] = max_tok
            payload.pop("max_tokens", None)
        elif any(k in m_lower for k in ("doubao", "glm", "kimi", "deepseek")):
            payload["enable_thinking"] = True
            payload["thinking"] = {"type": "enabled", "budget_tokens": 4096}
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GENERIC_KEY}",
        }
        endpoint = f"{GENERIC_ENDPOINT.rstrip('/')}/v1/chat/completions"
    else:
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{img_b64}"}},
                    {"type": "text", "text": question},
                ],
            }],
            "max_tokens": max_tok,
            "temperature": 0.0,
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {QWEN_KEYS[key_idx % len(QWEN_KEYS)]}",
        }
        endpoint = f"{QWEN_ENDPOINT.rstrip('/')}/chat/completions"

    async with sem:
        for attempt in range(MAX_RETRIES):
            try:
                resp = await client.post(
                    endpoint, headers=headers, json=payload, timeout=REQUEST_TIMEOUT_VL
                )
                if is_gemini:
                    if resp.status_code != 200:
                        raise RuntimeError(f"HTTP {resp.status_code}: {(resp.text or '')[:300]}")
                    return record["id"], _gemini_text(resp.json()), None
                if resp.status_code != 200:
                    raise RuntimeError(f"HTTP {resp.status_code}: {(resp.text or '')[:500]}")
                return record["id"], resp.json()["choices"][0]["message"]["content"], None
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    return record["id"], None, f"{type(e).__name__}: {str(e)[:200]}"
                await asyncio.sleep(min(2 ** attempt, 3))


# ── main ───────────────────────────────────────────────────────────────────


def _model_slug(model: str) -> str:
    return model.replace("/", "_").replace(".", "_").replace("-", "_")


async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], default=0)
    parser.add_argument("--model", default="gemini-3.1-pro-preview")
    parser.add_argument("--rejudge", action="store_true")
    parser.add_argument(
        "--retry-empty", action="store_true",
        help="re-run Phase 1 for items with empty response",
    )
    args = parser.parse_args()
    run_all = args.phase == 0
    RESULTS_PATH = RESULTS_DIR / f"eval_{_model_slug(args.model)}.jsonl"

    print(f"Model: {args.model}")
    print(f"Output: {RESULTS_PATH}")

    print("Loading benchmark...")
    all_recs = read_jsonl(INPUT_PATH)
    print(f"  {len(all_recs)} total")
    if args.limit > 0:
        all_recs = all_recs[: args.limit]
        print(f"  --limit {args.limit}")
    valid = [r for r in all_recs if (IMAGE_DIR / r.get("image", "")).exists()]
    print(f"  {len(valid)} with image")

    src_map = {r["id"]: r for r in all_recs}

    # ── phase 1: VL inference ──
    if run_all or args.phase == 1:
        existing = read_jsonl(RESULTS_PATH)
        done_ids = {r["id"] for r in existing if r.get("vl_response") or r.get("vl_error")}
        empty_ids = {
            r["id"] for r in existing
            if not r.get("vl_response") and not r.get("vl_error")
        }
        if args.retry_empty:
            print(f"  --retry-empty: {len(empty_ids)} empty responses will be re-inferred")
            done_ids -= empty_ids
        pending = [r for r in valid if r["id"] not in done_ids]
        print(f"\n[Phase 1] VL inference ({args.model}): {len(pending)} pending, {len(done_ids)} done")

        if pending:
            sem = asyncio.Semaphore(VL_CONCURRENCY)
            done_count = [0]
            lock = asyncio.Lock()

            async def infer(client, idx, rec):
                rid, response, error = await vl_infer(client, sem, rec, args.model, idx)
                result = {
                    "id": rid,
                    "vl_response": response,
                    "vl_error": error,
                    "gold": rec.get("answer", ""),
                    "qtype": rec.get("qtype", "oe"),
                    "domain": domain_of(rec),
                    "subdomain": rec.get("subdomain", ""),
                    "correct": None,
                    "extracted_letter": None,
                    "verifier_raw": None,
                    "verifier_reasoning": None,
                }
                async with lock:
                    done_count[0] += 1
                    if done_count[0] % 20 == 0:
                        print(f"    {done_count[0]}/{len(pending)}", flush=True)
                    with open(RESULTS_PATH, "a", encoding="utf-8") as f:
                        f.write(json.dumps(result, ensure_ascii=False) + "\n")

            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT_VL,
                trust_env=True if is_generic_model(args.model) else False,
                limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
            ) as client:
                for i in range(0, len(pending), 50):
                    batch = pending[i : i + 50]
                    await asyncio.gather(
                        *[infer(client, i + j, item) for j, item in enumerate(batch)]
                    )
            print(f"  Done: {len(pending)} processed")

    # Dedup (latest wins)
    results = read_jsonl(RESULTS_PATH)
    seen = {}
    for r in results:
        seen[r["id"]] = r
    results = list(seen.values())

    # ── phase 2: extract + judge ──
    if run_all or args.phase == 2:
        empty_count = 0
        for r in results:
            if r.get("correct") is not None and not args.rejudge:
                continue
            if not r.get("vl_response") and not r.get("vl_error"):
                r["extracted_letter"] = "Z"
                r["correct"] = False
                empty_count += 1

        mc_pending = [
            r for r in results
            if r.get("qtype") == "mc" and r.get("vl_response")
            and (args.rejudge or r.get("correct") is None)
        ]
        oe_pending = [
            r for r in results
            if r.get("qtype") == "oe" and r.get("vl_response")
            and (args.rejudge or r.get("correct") is None)
        ]
        print(
            f"\n[Phase 2] Extract & judge: {empty_count} empty, "
            f"{len(mc_pending)} MC + {len(oe_pending)} OE"
        )

        kp = QwenKeyPool(QWEN_KEYS, QWEN_CONCURRENCY // len(QWEN_KEYS))

        if mc_pending or oe_pending:
            async with httpx.AsyncClient(
                timeout=REQUEST_TIMEOUT,
                trust_env=False,
                limits=httpx.Limits(max_connections=200, max_keepalive_connections=100),
            ) as client:
                if mc_pending:
                    done_mc = [0]

                    async def mc_worker(idx, rec):
                        src = src_map.get(rec["id"], rec)
                        letter = await extract_answer_from_item(client, kp, src, rec["vl_response"])
                        rec["extracted_letter"] = letter
                        if letter and letter != "Z":
                            rec["correct"] = letter.upper() == rec["gold"].strip().upper()
                        else:
                            rec["correct"] = False
                        done_mc[0] += 1
                        if done_mc[0] % 20 == 0:
                            print(f"    MC {done_mc[0]}/{len(mc_pending)}", flush=True)

                    for i in range(0, len(mc_pending), 100):
                        batch = mc_pending[i : i + 100]
                        await asyncio.gather(
                            *[mc_worker(i + j, item) for j, item in enumerate(batch)]
                        )
                    print(f"  MC done: {len(mc_pending)}")

                if oe_pending:
                    done_oe = [0]

                    async def oe_worker(idx, rec):
                        src = src_map.get(rec["id"], rec)
                        result = await judge_oe_verifier(
                            client, kp,
                            src.get("question", ""),
                            rec.get("gold", ""),
                            rec["vl_response"],
                        )
                        rec["correct"] = result.get("correct")
                        rec["verifier_raw"] = result.get("verifier_raw")
                        rec["verifier_reasoning"] = result.get("verifier_reasoning")
                        if result.get("judge_error"):
                            rec["judge_error"] = result["judge_error"]
                        done_oe[0] += 1
                        if done_oe[0] % 20 == 0:
                            print(f"    OE {done_oe[0]}/{len(oe_pending)}", flush=True)

                    for i in range(0, len(oe_pending), 100):
                        batch = oe_pending[i : i + 100]
                        await asyncio.gather(
                            *[oe_worker(i + j, item) for j, item in enumerate(batch)]
                        )
                    print(f"  OE done: {len(oe_pending)}")

        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    results = read_jsonl(RESULTS_PATH)
    if not results:
        return

    # ── phase 3: stats ──
    if run_all or args.phase == 3:
        errors = sum(1 for r in results if r.get("vl_error"))
        total = len(results)
        valid_count = total - errors
        correct_count = sum(1 for r in results if r.get("correct") and not r.get("vl_error"))
        acc = correct_count / max(valid_count, 1) * 100
        print(f"\n{'=' * 60}")
        print(f"  Overall: {acc:.1f}% ({correct_count}/{valid_count})  errors={errors}")
        print(f"{'=' * 60}")

        for label, field in [("Domain", "domain"), ("Subdomain", "subdomain")]:
            print(f"\n  By {label}:")
            cats = sorted(
                set(r.get(field, "?") for r in results),
                key=lambda c: -sum(1 for r in results if r.get(field) == c),
            )
            for cat in cats:
                items = [r for r in results if r.get(field) == cat]
                cat_ok = sum(1 for r in items if r.get("correct") and not r.get("vl_error"))
                cat_err = sum(1 for r in items if r.get("vl_error"))
                cat_valid = len(items) - cat_err
                cat_acc = cat_ok / max(cat_valid, 1) * 100
                bar = "█" * int(cat_acc / 5)
                print(f"    {cat:35s} {cat_acc:5.1f}% ({cat_ok}/{cat_valid}) {bar}")

        print(f"\nResults: {RESULTS_PATH}")


if __name__ == "__main__":
    asyncio.run(async_main())
