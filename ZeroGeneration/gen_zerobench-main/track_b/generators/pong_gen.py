"""
Atari Pong move-sequence puzzle generator.

ZeroBench Q33: Given a Pong game state, produce the optimal right-paddle
action sequence (digits: 2=FIRE/UP, 5=DOWN, 0=NOOP) to intercept the ball.

Example solution: "222220000"

Simulation uses normalized coordinates (0.0–1.0).
Canvas reference: 160×210 Atari Pong (normalized to 0–1).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Constants (all in normalized 0–1 space)
# ---------------------------------------------------------------------------
CANVAS_W = 160
CANVAS_H = 210

PADDLE_HEIGHT = 0.15   # normalized paddle height
PADDLE_WIDTH  = 0.05   # normalized paddle width
BALL_RADIUS   = 0.02   # normalized ball radius
PADDLE_SPEED  = 0.05   # paddle movement per step
BALL_VX_MIN   = 0.04
BALL_VX_MAX   = 0.08

# Atari action IDs
NOOP = 0
UP   = 2
DOWN = 5

# Right-paddle x position (normalized)
RIGHT_PADDLE_X = 0.92
LEFT_PADDLE_X  = 0.08

# Wall boundaries
WALL_TOP    = 0.0
WALL_BOTTOM = 1.0
BALL_RIGHT_THRESHOLD = 0.95  # ball is "at right wall" if x > this


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------
@dataclass
class PongPuzzle:
    puzzle_id: int
    # Game state (normalized)
    ball_x: float
    ball_y: float
    ball_vx: float          # positive = moving right
    ball_vy: float
    right_paddle_y: float   # center y of right (controlled) paddle
    left_paddle_y: float    # center y of left (AI) paddle
    # Action sequence
    n_steps: int            # length of sequence (6–12)
    actions: list[int]      # per-step action: 0=NOOP, 2=UP, 5=DOWN
    solution: Optional[str] # "".join(str(a) for a in actions)
    difficulty: str         # "easy" / "medium" / "hard"


# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _simulate_ball(bx: float, by: float, vx: float, vy: float) -> tuple[float, float, float, float]:
    """Advance ball one step; bounce off top/bottom walls."""
    bx += vx
    by += vy
    if by - BALL_RADIUS <= WALL_TOP:
        by = WALL_TOP + BALL_RADIUS
        vy = abs(vy)
    elif by + BALL_RADIUS >= WALL_BOTTOM:
        by = WALL_BOTTOM - BALL_RADIUS
        vy = -abs(vy)
    return bx, by, vx, vy


def _move_paddle(paddle_y: float, action: int) -> float:
    """Move the right paddle according to the action."""
    if action == UP:
        paddle_y -= PADDLE_SPEED
    elif action == DOWN:
        paddle_y += PADDLE_SPEED
    # Clamp so paddle stays on screen
    half = PADDLE_HEIGHT / 2
    paddle_y = _clamp(paddle_y, half, 1.0 - half)
    return paddle_y


def _left_ai_move(left_y: float, ball_y: float) -> float:
    """Left AI paddle tracks the ball."""
    if left_y < ball_y - 0.02:
        left_y += PADDLE_SPEED
    elif left_y > ball_y + 0.02:
        left_y -= PADDLE_SPEED
    half = PADDLE_HEIGHT / 2
    left_y = _clamp(left_y, half, 1.0 - half)
    return left_y


def _optimal_action(paddle_y: float, target_y: float) -> int:
    """Choose the optimal action to move paddle_y toward target_y."""
    if paddle_y < target_y - 0.02:
        return DOWN
    elif paddle_y > target_y + 0.02:
        return UP
    return NOOP


def _project_ball_to_x(bx: float, by: float, vx: float, vy: float, target_x: float) -> float:
    """
    Project where the ball's y will be when its x reaches target_x.
    Accounts for wall bounces.
    """
    steps_left = 0
    if abs(vx) < 1e-9:
        return by
    steps_to_target = (target_x - bx) / vx
    if steps_to_target < 0:
        # Ball moving away; return current y
        return by

    cur_by, cur_vy = by, vy
    for _ in range(int(steps_to_target) + 2):
        if abs(vx) < 1e-9:
            break
        _, cur_by, _, cur_vy = _simulate_ball(bx, cur_by, vx, cur_vy)
        bx += vx
        if abs(bx - target_x) < abs(vx):
            break
    return _clamp(cur_by, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _make_puzzle(rng: random.Random, puzzle_id: int) -> PongPuzzle:
    """Generate a single PongPuzzle with a valid action sequence."""
    n_steps = rng.randint(6, 12)

    # Initial ball: left half, moving right
    bx = rng.uniform(0.10, 0.45)
    by = rng.uniform(0.15, 0.85)
    vx = rng.uniform(BALL_VX_MIN, BALL_VX_MAX)
    vy = rng.uniform(-0.06, 0.06)
    if abs(vy) < 0.005:
        vy = 0.02 * (1 if rng.random() > 0.5 else -1)

    # Paddle initial positions
    right_y = rng.uniform(0.3, 0.7)
    left_y  = rng.uniform(0.3, 0.7)

    init_bx, init_by, init_vx, init_vy = bx, by, vx, vy
    init_right_y = right_y
    init_left_y  = left_y

    actions: list[int] = []
    cur_bx, cur_by, cur_vx, cur_vy = bx, by, vx, vy
    cur_right_y = right_y
    cur_left_y  = left_y

    for _ in range(n_steps):
        # Project where ball will be at the right paddle's x when it arrives
        projected_y = _project_ball_to_x(cur_bx, cur_by, cur_vx, cur_vy, RIGHT_PADDLE_X)

        act = _optimal_action(cur_right_y, projected_y)
        actions.append(act)

        # Advance simulation
        cur_bx, cur_by, cur_vx, cur_vy = _simulate_ball(cur_bx, cur_by, cur_vx, cur_vy)
        cur_right_y = _move_paddle(cur_right_y, act)
        cur_left_y  = _left_ai_move(cur_left_y, cur_by)

    solution = "".join(str(a) for a in actions)

    # Difficulty by n_steps and vy magnitude
    if n_steps <= 7:
        difficulty = "easy"
    elif n_steps <= 10:
        difficulty = "medium"
    else:
        difficulty = "hard"

    return PongPuzzle(
        puzzle_id=puzzle_id,
        ball_x=init_bx,
        ball_y=init_by,
        ball_vx=init_vx,
        ball_vy=init_vy,
        right_paddle_y=init_right_y,
        left_paddle_y=init_left_y,
        n_steps=n_steps,
        actions=actions,
        solution=solution,
        difficulty=difficulty,
    )


def generate_puzzles(n: int = 40, seed: int = 42) -> list[PongPuzzle]:
    """Generate n PongPuzzle instances deterministically from seed."""
    rng = random.Random(seed)
    return [_make_puzzle(rng, puzzle_id=i + 1) for i in range(n)]


# ---------------------------------------------------------------------------
# Solve / CoT helpers
# ---------------------------------------------------------------------------

def solve_puzzle(puzzle: PongPuzzle) -> str:
    """Return the solution string (already stored in puzzle.actions)."""
    return "".join(str(a) for a in puzzle.actions)


def build_step_descriptions(puzzle: PongPuzzle) -> str:
    """
    Build a step-by-step Chain-of-Thought description for the puzzle.
    Returns a human-readable string.
    """
    action_names = {NOOP: "NOOP (stay)", UP: "UP (move up)", DOWN: "DOWN (move down)"}
    lines: list[str] = [
        "=== Pong Action Sequence — Chain of Thought ===",
        f"Initial state:",
        f"  Ball:         ({puzzle.ball_x:.3f}, {puzzle.ball_y:.3f})  "
        f"vel=({puzzle.ball_vx:.3f}, {puzzle.ball_vy:.3f})",
        f"  Right paddle: y={puzzle.right_paddle_y:.3f}",
        f"  Left paddle:  y={puzzle.left_paddle_y:.3f}",
        f"  Steps to compute: {puzzle.n_steps}",
        "",
    ]

    # Re-simulate to generate per-step descriptions
    bx, by, vx, vy = puzzle.ball_x, puzzle.ball_y, puzzle.ball_vx, puzzle.ball_vy
    right_y = puzzle.right_paddle_y
    left_y  = puzzle.left_paddle_y

    for step_i, act in enumerate(puzzle.actions):
        proj_y = _project_ball_to_x(bx, by, vx, vy, RIGHT_PADDLE_X)
        lines.append(
            f"Step {step_i + 1}: ball=({bx:.3f},{by:.3f}) vel=({vx:.3f},{vy:.3f}) "
            f"paddle_y={right_y:.3f}  projected_ball_y={proj_y:.3f}"
        )
        lines.append(f"         → action={act} ({action_names[act]})")

        bx, by, vx, vy = _simulate_ball(bx, by, vx, vy)
        right_y = _move_paddle(right_y, act)
        left_y  = _left_ai_move(left_y, by)

    lines.append("")
    lines.append(f"Solution: {''.join(str(a) for a in puzzle.actions)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    puzzles = generate_puzzles(n=5, seed=42)
    for p in puzzles:
        print(f"Puzzle {p.puzzle_id} [{p.difficulty}] n_steps={p.n_steps} "
              f"sol={p.solution}  ball=({p.ball_x:.2f},{p.ball_y:.2f})")
