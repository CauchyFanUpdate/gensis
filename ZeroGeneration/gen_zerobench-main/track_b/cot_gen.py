"""
CoT (Chain-of-Thought) generator for Track B programmatic puzzles.

Method C: programmatic state transitions → LLM translates to natural language CoT.

Supports: Sokoban, Sudoku (extensible to Chess, Maze, etc.)
"""

import asyncio
import re
from typing import Optional

import httpx

from track_b.generators.sokoban_gen import SokobanPuzzle, build_step_descriptions
from track_b.generators.sudoku_gen import SudokuPuzzle

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------
QWEN_BASE_URL   = "https://runway.devops.xiaohongshu.com/openai/qwen/v1"
QWEN_MODEL      = "qwen3.6-plus"
GEMINI_ENDPOINT = "https://runway.devops.rednote.life/openai/google/v1:generateContent"
GEMINI_MODEL    = "gemini-3.0-pro"
GEMINI_API_KEY  = "YOUR_GEMINI_API_KEY"

# Gemini safety settings (permissive)
_GEMINI_SAFETY = [
    {"category": c, "threshold": "BLOCK_NONE"}
    for c in (
        "HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH",
        "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT",
    )
]


async def _call_gemini_text(
    prompt: str,
    system: str = "",
    temperature: float = 0.3,
    max_tokens: int = 2048,
    max_retries: int = 4,
) -> str:
    """Call Gemini via runway native format (:generateContent). Returns text."""
    parts = []
    if system:
        parts.append({"text": system + "\n\n" + prompt})
    else:
        parts.append({"text": prompt})

    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "generationConfig": {
            "temperature": temperature,
            "maxOutputTokens": max_tokens,
            "topP": 0.95,
        },
        "safetySettings": _GEMINI_SAFETY,
    }
    headers = {"api-key": GEMINI_API_KEY, "Content-Type": "application/json"}

    async with httpx.AsyncClient(trust_env=False, timeout=120.0) as client:
        for attempt in range(max_retries):
            try:
                resp = await client.post(GEMINI_ENDPOINT, headers=headers, json=payload)
                resp.raise_for_status()
                body = resp.json()

                # Proxy error wrapper (HTTP 200 but body has Code/Error)
                if "Code" in body and "Error" in body and "candidates" not in body:
                    err = str(body.get("Error", ""))
                    if any(s in err for s in ("429", "RESOURCE_EXHAUSTED", "quota")):
                        await asyncio.sleep(min(30 * (attempt + 1), 120))
                        continue
                    raise RuntimeError(f"Gemini proxy error: {err[:200]}")

                candidates = body.get("candidates", [])
                if not candidates:
                    raise RuntimeError(f"Gemini returned no candidates: {str(body)[:200]}")
                content = candidates[0].get("content", {})
                parts_out = content.get("parts", [])
                if not parts_out:
                    # MAX_TOKENS with all tokens used for thinking — retry with more tokens
                    if attempt < max_retries - 1:
                        await asyncio.sleep(3)
                        continue
                    return ""
                return " ".join(
                    p.get("text", "") for p in parts_out
                    if "text" in p and not p.get("thought")
                ).strip()

            except (httpx.TimeoutException, httpx.ConnectError) as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                else:
                    raise
    return ""




# ---------------------------------------------------------------------------
# Sokoban CoT
# ---------------------------------------------------------------------------

def _sokoban_steps_to_structured(puzzle: SokobanPuzzle, moves: list) -> str:
    """Build structured move description (programmatic, always correct)."""
    steps = build_step_descriptions(puzzle, moves)
    lines = [
        f"The puzzle has {puzzle.n_boxes} box(es) and {len(puzzle.targets)} target(s).",
        f"Optimal solution: {len(moves)} moves.",
        "",
        "Move sequence:",
    ]
    for s in steps:
        if s["pushed_box"]:
            lines.append(
                f"  Step {s['step']}: Move {s['move']}. "
                f"Player moves from {s['player_from']} to {s['player_to']}, "
                f"pushing box from {s['box_from']} to {s['box_to']}."
            )
        else:
            lines.append(
                f"  Step {s['step']}: Move {s['move']}. "
                f"Player moves from {s['player_from']} to {s['player_to']} (no push)."
            )
    lines.append("")
    lines.append(f"Final answer: {', '.join(moves)}")
    return "\n".join(lines)


SOKOBAN_COT_PROMPT = """\
You are given a Sokoban puzzle and its complete step-by-step solution.
Your task is to write a natural-language chain-of-thought (CoT) reasoning trace
that a solver would follow when finding this solution.

The CoT must:
1. Start by briefly analyzing the puzzle: where the player, boxes, and targets are.
2. Explain the strategy: which box to move first and why.
3. Describe each move group in natural language (not just "move right").
4. End with the final answer in the exact format:
   <answer>
   move1, move2, move3, ...
   </answer>

Here is the structured solution to translate into a CoT:
{structured_steps}

Write the CoT now. Be clear, logical, and concise. Do not add moves that are not in the solution.
Wrap your reasoning in <think>...</think> and end with <answer>...</answer>.
"""


async def generate_sokoban_cot(
    puzzle: SokobanPuzzle,
    moves: list,
    **kwargs,
) -> dict:
    """Generate CoT for a Sokoban puzzle. Returns dict with think and answer."""
    structured = _sokoban_steps_to_structured(puzzle, moves)
    prompt = SOKOBAN_COT_PROMPT.format(structured_steps=structured)

    raw = await _call_gemini_text(prompt, temperature=0.3, max_tokens=2048)

    think = ""
    answer = ", ".join(moves)  # fallback

    think_m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_m:
        think = think_m.group(1).strip()

    ans_m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if ans_m:
        answer = ans_m.group(1).strip()

    return {
        "think": think or raw,
        "answer": answer,
        "raw_response": raw,
    }


# ---------------------------------------------------------------------------
# Sudoku CoT
# ---------------------------------------------------------------------------

def _sudoku_steps_to_structured(puzzle: SudokuPuzzle) -> str:
    """Build structured Sudoku solution description (programmatic)."""
    n_given = sum(1 for r in puzzle.clues for v in r if v != 0)
    lines = [
        f"Sudoku puzzle ({puzzle.difficulty} difficulty, {n_given} givens).",
        "Step-by-step solution:",
        "",
    ]
    for step in puzzle.solve_steps:
        lines.append(
            f"  Step {step['step']}: Place {step['value']} at "
            f"Row {step['row']}, Col {step['col']} — {step['reasoning']}"
        )
    lines.append("")
    lines.append("Final grid (row by row):")
    for i, row in enumerate(puzzle.solution):
        lines.append(f"  Row {i+1}: {' '.join(str(v) for v in row)}")
    return "\n".join(lines)


SUDOKU_COT_PROMPT = """\
You are given a Sudoku puzzle and its complete step-by-step solution.
Write a natural-language chain-of-thought (CoT) reasoning trace that explains
how to solve this puzzle logically.

The CoT must:
1. Briefly state the initial state (how many clues are given, any obvious constraints).
2. Walk through the key deduction steps, grouping related steps together.
3. Explain WHY each number is placed (which row/column/box constraint eliminates other candidates).
4. End with the complete solved grid in the exact format:
   <answer>
   Row 1: d1 d2 d3 d4 d5 d6 d7 d8 d9
   Row 2: ...
   ...
   Row 9: ...
   </answer>

Here is the structured solution to translate:
{structured_steps}

Write the CoT now. Wrap reasoning in <think>...</think> and end with <answer>...</answer>.
"""


async def generate_sudoku_cot(
    puzzle: SudokuPuzzle,
    **kwargs,
) -> dict:
    """Generate CoT for a Sudoku puzzle."""
    structured = _sudoku_steps_to_structured(puzzle)
    prompt = SUDOKU_COT_PROMPT.format(structured_steps=structured)

    raw = await _call_gemini_text(prompt, temperature=0.3, max_tokens=8192)

    think = ""
    answer_lines = [
        f"Row {i+1}: {' '.join(str(v) for v in row)}"
        for i, row in enumerate(puzzle.solution)
    ]
    answer = "\n".join(answer_lines)

    think_m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    if think_m:
        think = think_m.group(1).strip()

    ans_m = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if ans_m:
        answer = ans_m.group(1).strip()

    return {
        "think": think or raw,
        "answer": answer,
        "raw_response": raw,
    }


# ---------------------------------------------------------------------------
# Generic programmatic-CoT for new puzzle types
# ---------------------------------------------------------------------------

def _generic_steps(puzzle_type: str, puzzle) -> str:
    """Call build_step_descriptions on any puzzle that has it."""
    fn = getattr(puzzle, "build_step_descriptions", None)
    if fn:
        return fn()
    # Fallback: use the generator module's helper
    import importlib
    try:
        mod = importlib.import_module(f"track_b.generators.{puzzle_type}_gen")
        return mod.build_step_descriptions(puzzle)
    except Exception:
        return f"Solution: {puzzle.solution}"


_GENERIC_COT_PROMPT = """\
You are given a {puzzle_type} puzzle and its complete step-by-step solution.
Write a natural-language chain-of-thought (CoT) reasoning trace explaining how to solve it.

Requirements:
1. Start by describing the current puzzle state (positions, values, board layout).
2. Explain the reasoning strategy step by step.
3. Reference specific positions/values from the structured solution.
4. End with the exact answer in <answer>...</answer>.

Structured solution:
{structured_steps}

Write the CoT now. Wrap reasoning in <think>...</think> and end with <answer>...</answer>.
The content inside <answer> must be exactly: {answer}
"""


async def generate_generic_cot(
    puzzle_type: str,
    puzzle,
    **kwargs,
) -> dict:
    structured = _generic_steps(puzzle_type, puzzle)
    answer = str(puzzle.solution or "")
    prompt = _GENERIC_COT_PROMPT.format(
        puzzle_type=puzzle_type,
        structured_steps=structured,
        answer=answer,
    )

    raw = await _call_gemini_text(prompt, temperature=0.3, max_tokens=4096)

    think_m = re.search(r"<think>(.*?)</think>", raw, re.DOTALL)
    ans_m   = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)

    think   = think_m.group(1).strip() if think_m else raw
    cot_ans = ans_m.group(1).strip() if ans_m else answer

    return {"think": think or raw, "answer": cot_ans, "raw_response": raw}


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

_GENERIC_TYPES = {
    "chess", "maze", "connect4", "othello",
    "game2048", "nonogram", "pong", "graph",
}


async def generate_cot(puzzle_type: str, puzzle, moves=None, **kwargs) -> dict:
    """
    Unified CoT generator.

    Args:
        puzzle_type: 'sokoban', 'sudoku', or any track_b generic type
        puzzle: puzzle dataclass
        moves: required for sokoban (list of move strings)
    """
    if puzzle_type == "sokoban":
        assert moves is not None
        return await generate_sokoban_cot(puzzle, moves, **kwargs)
    elif puzzle_type == "sudoku":
        return await generate_sudoku_cot(puzzle, **kwargs)
    elif puzzle_type in _GENERIC_TYPES:
        return await generate_generic_cot(puzzle_type, puzzle, **kwargs)
    else:
        raise ValueError(f"Unknown puzzle_type: {puzzle_type}")
