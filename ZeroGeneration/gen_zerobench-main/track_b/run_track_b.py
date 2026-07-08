"""
Track B main pipeline: programmatic puzzle generation → Stage 1 (RL) + Stage 2 (SFT).

Stage 1 (RL):  standard render → (image, question, answer) — close to ZeroBench style
Stage 2 (SFT): GPT-Image-2 style diversification → (image, question, CoT, answer)

Supported puzzle types:
  sokoban, sudoku, chess, maze, connect4, othello, game2048, nonogram, pong, graph

Usage:
    python -m track_b.run_track_b --puzzle sokoban --n 50 --stage all
    python -m track_b.run_track_b --puzzle chess   --n 20 --stage rl
    python -m track_b.run_track_b --puzzle all     --n 40 --stage all
"""

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import time
from pathlib import Path

from openai import AsyncOpenAI
from PIL import Image

BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))

# ---------------------------------------------------------------------------
# Puzzle generators & renderers registry
# ---------------------------------------------------------------------------
from track_b.generators.sokoban_gen  import get_all_puzzles as _sokoban_puzzles,   solve_puzzle as _sokoban_solve
from track_b.generators.sudoku_gen   import generate_puzzles as _sudoku_puzzles,    puzzle_to_answer as _sudoku_solve
from track_b.generators.chess_gen    import generate_puzzles as _chess_puzzles,     solve_puzzle as _chess_solve
from track_b.generators.maze_gen     import generate_puzzles as _maze_puzzles,      solve_puzzle as _maze_solve
from track_b.generators.connect4_gen import generate_puzzles as _c4_puzzles,        solve_puzzle as _c4_solve
from track_b.generators.othello_gen  import generate_puzzles as _othello_puzzles,   solve_puzzle as _othello_solve
from track_b.generators.game2048_gen import generate_puzzles as _g2048_puzzles,     solve_puzzle as _g2048_solve
from track_b.generators.nonogram_gen import generate_puzzles as _nonogram_puzzles,  solve_puzzle as _nonogram_solve
from track_b.generators.pong_gen     import generate_puzzles as _pong_puzzles,      solve_puzzle as _pong_solve
from track_b.generators.graph_gen    import generate_puzzles as _graph_puzzles,     solve_puzzle as _graph_solve

from track_b.renderers.sokoban_render  import render_puzzle as _sokoban_render,  get_gpt_image2_style_prompt as _sokoban_style
from track_b.renderers.sudoku_render   import render_puzzle as _sudoku_render,   get_gpt_image2_style_prompt as _sudoku_style
from track_b.renderers.chess_render    import render_puzzle as _chess_render,    get_gpt_image2_style_prompt as _chess_style
from track_b.renderers.maze_render     import render_puzzle as _maze_render,     get_gpt_image2_style_prompt as _maze_style
from track_b.renderers.connect4_render import render_puzzle as _c4_render,       get_gpt_image2_style_prompt as _c4_style
from track_b.renderers.othello_render  import render_puzzle as _othello_render,  get_gpt_image2_style_prompt as _othello_style
from track_b.renderers.game2048_render import render_puzzle as _g2048_render,    get_gpt_image2_style_prompt as _g2048_style
from track_b.renderers.nonogram_render import render_puzzle as _nonogram_render, get_gpt_image2_style_prompt as _nonogram_style
from track_b.renderers.pong_render     import render_puzzle as _pong_render,     get_gpt_image2_style_prompt as _pong_style
from track_b.renderers.graph_render    import render_puzzle as _graph_render,    get_gpt_image2_style_prompt as _graph_style

from track_b.cot_gen import generate_cot

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GEMINI_ENDPOINT  = "https://runway.devops.rednote.life/openai/google/v1:generateContent"
GEMINI_MODEL     = "gemini-3.0-pro"
GEMINI_API_KEY   = "YOUR_GEMINI_API_KEY"
GPT_IMAGE2_URL     = "https://api.gpt.ge/v1"   # external endpoint, Bearer auth, uses system proxy
GPT_IMAGE2_API_KEY_DEFAULT = "YOUR_GPT_IMAGE_API_KEY"
GPT_IMAGE2_MODEL   = "gpt-image-2-c"
GPT_IMAGE2_SIZE    = "auto"
GPT_IMAGE2_QUALITY = "auto"

OUTPUT_DIR       = BASE_DIR / "outputs"
IMAGE_STAGE1_DIR = OUTPUT_DIR / "rl"
IMAGE_STAGE2_DIR = OUTPUT_DIR / "sft"

COT_CONCURRENCY = 8
IMG_CONCURRENCY = 2

SYSTEM_PROMPT = (
    "You are an expert visual puzzle solver. "
    "Output your reasoning inside <think>...</think> "
    "and your final answer inside <answer>...</answer>."
)

# ---------------------------------------------------------------------------
# Per-puzzle-type questions
# ---------------------------------------------------------------------------
PUZZLE_QUESTIONS = {
    "sokoban": (
        "This is a Sokoban puzzle. The player (@) must push all boxes ($) onto the target "
        "positions (.). Walls (#) cannot be moved through. Provide the optimal sequence of "
        "moves (up, down, left, right) to solve this puzzle."
    ),
    "sudoku": (
        "Solve the Sudoku puzzle shown in the image. Fill every empty cell with a digit from "
        "1 to 9 so that each row, each column, and each 3×3 box contains each digit exactly "
        "once. Provide the complete solved grid."
    ),
    "chess": (
        "The chess board shows a position where it is White's turn to move. "
        "There is a checkmate in one move. What is that move? "
        "Give your answer in standard algebraic notation (e.g. Qxe5#)."
    ),
    "maze": (
        "The maze contains several coloured players at different starting positions and one exit. "
        "Some players can reach the exit and some are trapped. "
        "What fraction of players can reach the exit? Give your answer as a simplified fraction or integer."
    ),
    "connect4": (
        "This is a Connect 4 board. The current player must find all columns where dropping "
        "a piece immediately creates four in a row (horizontally, vertically, or diagonally). "
        "List the winning column number(s) in ascending order."
    ),
    "othello": (
        "This is an Othello (Reversi) board. The current player should make the move that "
        "flips the most opponent pieces. How many pieces does that best move flip?"
    ),
    "game2048": (
        "This is a 2048 puzzle board. What is the optimal sequence of moves (Left, Right, Up, Down) "
        "to maximise the score over the next several moves? "
        "List the moves in order, separated by commas."
    ),
    "nonogram": (
        "This is a colour nonogram puzzle. The numbers on the rows and columns indicate the lengths "
        "of consecutive filled blocks. Solve the nonogram and identify what object or word it depicts. "
        "Give your final answer as a single word."
    ),
    "pong": (
        "This is an Atari Pong game state. You control the right paddle. "
        "What is the optimal sequence of actions to intercept the ball? "
        "Use digits: 0=stay, 2=move up, 5=move down. Give the sequence with no spaces (e.g. 22050)."
    ),
    "graph": (
        "The image shows a directed weighted graph. "
        "Follow the sequence of edge weights from the starting node and compute the product "
        "of the values of all visited nodes. Give your final answer as an integer."
    ),
}

# Style names for Stage 2 diversification per puzzle type
STYLE_NAMES = {
    "sokoban":  ["classic", "dark", "retro", "modern"],
    "sudoku":   ["dark", "warm", "minimal", "colorful"],
    "chess":    ["classic", "dark", "green", "marble"],
    "maze":     ["classic", "dark", "colorful"],
    "connect4": ["classic", "dark", "pastel"],
    "othello":  ["classic", "dark", "wood"],
    "game2048": ["classic", "dark", "pastel"],
    "nonogram": ["classic", "dark", "colorful"],
    "pong":     ["atari", "modern", "classic"],
    "graph":    ["classic", "dark", "blueprint"],
}

# ---------------------------------------------------------------------------
# Puzzle-type registry — maps type name to (generate_fn, solve_fn, render_fn, style_fn)
# ---------------------------------------------------------------------------
PUZZLE_REGISTRY = {
    "sokoban":  (None,             _sokoban_solve,  _sokoban_render,  _sokoban_style),
    "sudoku":   (_sudoku_puzzles,  _sudoku_solve,   _sudoku_render,   _sudoku_style),
    "chess":    (_chess_puzzles,   _chess_solve,    _chess_render,    _chess_style),
    "maze":     (_maze_puzzles,    _maze_solve,     _maze_render,     _maze_style),
    "connect4": (_c4_puzzles,      _c4_solve,       _c4_render,       _c4_style),
    "othello":  (_othello_puzzles, _othello_solve,  _othello_render,  _othello_style),
    "game2048": (_g2048_puzzles,   _g2048_solve,    _g2048_render,    _g2048_style),
    "nonogram": (_nonogram_puzzles,_nonogram_solve, _nonogram_render, _nonogram_style),
    "pong":     (_pong_puzzles,    _pong_solve,     _pong_render,     _pong_style),
    "graph":    (_graph_puzzles,   _graph_solve,    _graph_render,    _graph_style),
}

ALL_PUZZLE_TYPES = list(PUZZLE_REGISTRY.keys())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def save_image(img: Image.Image, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def make_rl_record(sample_key: str, image_path: str, question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"<image>{question}"},
            {"role": "assistant", "content": f"<answer>\n{answer}</answer>"},
        ],
        "images": [image_path],
        "sample_key": sample_key,
        "stage": "stage1_rl",
    }


def make_sft_record(sample_key: str, image_path: str, question: str, think: str, answer: str) -> dict:
    assistant_content = f"<think>\n{think}\n</think>\n<answer>\n{answer}</answer>"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"<image>{question}"},
            {"role": "assistant", "content": assistant_content},
        ],
        "images": [image_path],
        "sample_key": sample_key,
        "stage": "stage2_sft",
    }


import random as _random
import requests as _requests
import functools
import concurrent.futures

_gpt_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _gpt_image2_edit_sync(image_bytes: bytes, prompt: str, api_key: str, timeout: float = 300.0) -> dict:
    """Raw HTTP call via external api.gpt.ge endpoint; returns parsed JSON body."""
    url = f"{GPT_IMAGE2_URL}/images/edits"
    resp = _requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        files={"image": ("image.png", image_bytes, "image/png")},
        data={"model": GPT_IMAGE2_MODEL, "prompt": prompt, "n": "1",
              "size": GPT_IMAGE2_SIZE, "quality": GPT_IMAGE2_QUALITY},
        timeout=timeout,
        # No proxies override — let requests use system proxy (https_proxy env)
    )
    resp.raise_for_status()
    return resp.json()


async def diversify_image_gpt2(
    img: Image.Image,
    style_prompt: str,
    api_key: str,
    sem: asyncio.Semaphore,
    max_retries: int = 30,
) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    image_bytes = buf.getvalue()
    loop = asyncio.get_running_loop()
    fn = functools.partial(_gpt_image2_edit_sync, image_bytes, style_prompt, api_key)

    for attempt in range(max_retries):
        try:
            async with sem:
                body = await loop.run_in_executor(_gpt_executor, fn)

            # Proxy wrapper returns HTTP 200 with {Code, Error} on overload
            if isinstance(body, dict) and "Code" in body and "data" not in body:
                err_str = str(body.get("Error", ""))
                if any(s in err_str.lower() for s in (
                    "engineoverloaded", "too many requests", "429",
                    "nocapacity", "high demand",
                )):
                    # Cap wait at 60s to keep retrying aggressively
                    wait = min(60.0, 15.0 * (attempt + 1)) * (1 + 0.3 * (2 * _random.random() - 1))
                    print(f"[gpt_image2] overload, retry {attempt+1}/{max_retries} in {wait:.0f}s | {err_str[:100]!r}")
                    await asyncio.sleep(wait)
                    continue
                raise ValueError(f"GPT-Image-2 proxy error: {body}")

            items = body.get("data", []) if isinstance(body, dict) else []
            if not items:
                raise ValueError(f"GPT-Image-2 empty data: {body}")
            result_bytes = base64.b64decode(items[0]["b64_json"])
            return Image.open(io.BytesIO(result_bytes)).convert("RGB")

        except _requests.exceptions.Timeout:
            wait = min(60.0, 10.0 * (attempt + 1))
            print(f"[gpt_image2] timeout, retry {attempt+1}/{max_retries} in {wait:.0f}s")
            await asyncio.sleep(wait)
        except _requests.exceptions.HTTPError as e:
            status = e.response.status_code if e.response is not None else 0
            if status == 429:
                wait = min(60.0, 20.0 * (attempt + 1))
                print(f"[gpt_image2] HTTP 429, retry {attempt+1}/{max_retries} in {wait:.0f}s")
                await asyncio.sleep(wait)
            else:
                raise

    raise RuntimeError(f"GPT-Image-2 failed after {max_retries} retries")


def _write_jsonl(records: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Wrote {len(records)} records → {path}")


# ---------------------------------------------------------------------------
# Generic puzzle pipeline (used by all types except sokoban which has special handling)
# ---------------------------------------------------------------------------

async def run_puzzle_type(
    puzzle_type: str,
    n: int,
    stage: str,
    output_prefix: str,
    base_seed: int = 2025,
):
    gen_fn, solve_fn, render_fn, style_fn = PUZZLE_REGISTRY[puzzle_type]
    question = PUZZLE_QUESTIONS[puzzle_type]
    styles   = STYLE_NAMES[puzzle_type]

    # Generate puzzles
    if puzzle_type == "sokoban":
        all_puzzles = _sokoban_puzzles(shuffle=True)
        puzzles = []
        for p in all_puzzles:
            if len(puzzles) >= n:
                break
            sol = solve_fn(p)
            if sol is not None:
                p.solution = sol
                puzzles.append(p)
    else:
        import inspect as _inspect
        _sig = _inspect.signature(gen_fn)
        if "seed" in _sig.parameters:
            raw = gen_fn(n=n, seed=base_seed)
        elif "base_seed" in _sig.parameters:
            raw = gen_fn(n=n, base_seed=base_seed)
        else:
            raw = gen_fn(n=n)
        puzzles = []
        for p in raw:
            sol = solve_fn(p)
            if sol is not None:
                p.solution = sol
                puzzles.append(p)

    print(f"[{puzzle_type}] {len(puzzles)}/{n} puzzles ready.")

    gpt2_api_key  = GPT_IMAGE2_API_KEY_DEFAULT  # external api.gpt.ge key, not overridden by env
    cot_sem       = asyncio.Semaphore(COT_CONCURRENCY)
    img_sem       = asyncio.Semaphore(IMG_CONCURRENCY)

    rl_records, sft_records = [], []

    async def process_one(puzzle, idx: int):
        pid = getattr(puzzle, "puzzle_id", None) or getattr(puzzle, "level_id", idx)
        key = f"{puzzle_type}_{pid:03d}"
        answer = str(puzzle.solution)

        # Stage 1: standard render → RL
        img_stage1 = render_fn(puzzle)
        img_path1  = IMAGE_STAGE1_DIR / f"{puzzle_type}/{key}.png"
        save_image(img_stage1, img_path1)
        rl_records.append(make_rl_record(key, str(img_path1), question, answer))

        if stage in ("sft", "all"):
            # CoT generation
            async with cot_sem:
                cot_kwargs = {"model": GEMINI_MODEL}
                if puzzle_type == "sokoban":
                    moves = puzzle.solution if isinstance(puzzle.solution, list) else puzzle.solution.split(", ")
                    cot = await generate_cot("sokoban", puzzle, moves=moves, **cot_kwargs)
                else:
                    cot = await generate_cot(puzzle_type, puzzle, **cot_kwargs)

            # Stage 2: style diversification
            for si, sname in enumerate(styles):
                style_key = f"{key}_s{si}"
                try:
                    img_s2 = await diversify_image_gpt2(
                        img_stage1, style_fn(sname), gpt2_api_key, img_sem
                    )
                except Exception as e:
                    print(f"[{puzzle_type}] GPT-Image-2 failed for {style_key}: {e}. Using original.")
                    img_s2 = img_stage1

                img_path2 = IMAGE_STAGE2_DIR / f"{puzzle_type}/{style_key}.png"
                save_image(img_s2, img_path2)
                sft_records.append(make_sft_record(
                    style_key, str(img_path2), question, cot["think"], cot["answer"]
                ))

    tasks = [process_one(p, i) for i, p in enumerate(puzzles)]
    for i, task in enumerate(asyncio.as_completed(tasks)):
        await task
        if (i + 1) % 5 == 0:
            print(f"[{puzzle_type}] {i + 1}/{len(tasks)} done")

    _write_jsonl(rl_records,  OUTPUT_DIR / f"rl/{output_prefix}_{puzzle_type}_stage1.jsonl")
    _write_jsonl(sft_records, OUTPUT_DIR / f"sft/{output_prefix}_{puzzle_type}_stage2.jsonl")
    print(f"[{puzzle_type}] RL: {len(rl_records)}, SFT: {len(sft_records)} records written.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Track B: programmatic puzzle data generation")
    parser.add_argument(
        "--puzzle",
        choices=ALL_PUZZLE_TYPES + ["all"],
        default="all",
        help="Puzzle type to generate (default: all)",
    )
    parser.add_argument("--n",      type=int, default=50,  help="Number of puzzles per type")
    parser.add_argument("--stage",  choices=["rl", "sft", "all"], default="all")
    parser.add_argument("--prefix", default="zerobench",   help="Output file prefix")
    parser.add_argument("--seed",   type=int, default=2025, help="Base random seed")
    args = parser.parse_args()

    types_to_run = ALL_PUZZLE_TYPES if args.puzzle == "all" else [args.puzzle]

    async def _run():
        for ptype in types_to_run:
            await run_puzzle_type(ptype, args.n, args.stage, args.prefix, args.seed)

    asyncio.run(_run())


if __name__ == "__main__":
    main()
