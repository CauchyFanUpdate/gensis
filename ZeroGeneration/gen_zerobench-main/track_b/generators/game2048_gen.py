"""
2048 sliding-tile puzzle generator.

Generates 4x4 2048 board positions and computes a greedy-optimal move
sequence (5-10 moves) that maximises immediate score.

ZeroBench Q24 asks:
  "What is the sequence of moves to achieve the highest score?"
  e.g. answer: "Right, Right, Right, Up, Right, Right, Right, Right, Right, Up"
"""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class Game2048Puzzle:
    puzzle_id: int
    board: list[list[int]]       # 4x4, 0=empty, positive=tile value
    optimal_moves: list[str]     # e.g. ["Right", "Right", "Up", "Right"]
    score_gain: int              # total score gained from optimal_moves
    solution: Optional[str]      # comma-separated move string
    difficulty: str


# ---------------------------------------------------------------------------
# Core game logic
# ---------------------------------------------------------------------------

def _slide_left(row: list[int]) -> tuple[list[int], int]:
    """Slide and merge a single row to the left.

    Returns (new_row, score_delta).
    """
    # Remove zeros
    tiles = [v for v in row if v != 0]
    score = 0
    merged: list[int] = []
    skip = False
    for i in range(len(tiles)):
        if skip:
            skip = False
            continue
        if i + 1 < len(tiles) and tiles[i] == tiles[i + 1]:
            val = tiles[i] * 2
            merged.append(val)
            score += val
            skip = True
        else:
            merged.append(tiles[i])
    # Pad with zeros on the right
    while len(merged) < len(row):
        merged.append(0)
    return merged, score


def _move(board: list[list[int]], direction: str) -> tuple[list[list[int]], int, bool]:
    """Apply a move in the given direction.

    Returns (new_board, score_delta, moved).
    """
    rotations = {"Left": 0, "Up": 1, "Right": 2, "Down": 3}
    n_rot = rotations[direction]

    # Rotate board so that we always slide left
    b = [row[:] for row in board]
    for _ in range(n_rot):
        b = _rotate_cw(b)

    new_b: list[list[int]] = []
    total_score = 0
    for row in b:
        new_row, score = _slide_left(row)
        new_b.append(new_row)
        total_score += score

    # Rotate back
    for _ in range((4 - n_rot) % 4):
        new_b = _rotate_cw(new_b)

    moved = (new_b != board)
    return new_b, total_score, moved


def _rotate_cw(board: list[list[int]]) -> list[list[int]]:
    """Rotate a 4x4 board 90 degrees clockwise."""
    n = len(board)
    return [[board[n - 1 - j][i] for j in range(n)] for i in range(n)]


def _get_empty_cells(board: list[list[int]]) -> list[tuple[int, int]]:
    """Return list of (row, col) for empty cells."""
    return [(r, c) for r in range(4) for c in range(4) if board[r][c] == 0]


def _add_random_tile(board: list[list[int]], rng: random.Random) -> list[list[int]]:
    """Place a 2 (90%) or 4 (10%) tile in a random empty cell. Returns new board."""
    empty = _get_empty_cells(board)
    if not empty:
        return board
    b = [row[:] for row in board]
    r, c = rng.choice(empty)
    b[r][c] = 4 if rng.random() < 0.1 else 2
    return b


def _has_valid_move(board: list[list[int]]) -> bool:
    """Return True if at least one move is possible."""
    for direction in ("Left", "Right", "Up", "Down"):
        _, _, moved = _move(board, direction)
        if moved:
            return True
    return False


def _max_tile(board: list[list[int]]) -> int:
    return max(board[r][c] for r in range(4) for c in range(4))


def _corner_bonus(board: list[list[int]]) -> int:
    """Return value of max tile if it is in a corner, else 0."""
    corners = [(0, 0), (0, 3), (3, 0), (3, 3)]
    mx = _max_tile(board)
    for r, c in corners:
        if board[r][c] == mx:
            return mx
    return 0


def _score_board(board: list[list[int]], immediate_score: int) -> float:
    """Heuristic score for a board state."""
    empty = len(_get_empty_cells(board))
    corner = _corner_bonus(board)
    mx = _max_tile(board)
    # Weight: immediate merges > empty cells > corner bonus > max tile
    return immediate_score * 100 + empty * 10 + corner * 2 + mx


# ---------------------------------------------------------------------------
# Greedy optimal sequence
# ---------------------------------------------------------------------------

DIRECTIONS = ["Left", "Right", "Up", "Down"]


def _greedy_sequence(
    board: list[list[int]],
    rng: random.Random,
    min_moves: int = 5,
    max_moves: int = 10,
) -> tuple[list[str], int]:
    """Compute a greedy move sequence.

    At each step, pick the move that maximises the heuristic score.
    Tie-break by random shuffle among equal-scored moves.
    Returns (moves, total_score_gain).
    """
    moves: list[str] = []
    total_score = 0
    current = [row[:] for row in board]

    for _ in range(max_moves):
        if len(moves) >= min_moves and not _has_valid_move(current):
            break

        candidates: list[tuple[float, str, list[list[int]], int]] = []
        dirs = DIRECTIONS[:]
        rng.shuffle(dirs)
        for direction in dirs:
            new_b, score_delta, moved = _move(current, direction)
            if not moved:
                continue
            h = _score_board(new_b, score_delta)
            candidates.append((h, direction, new_b, score_delta))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_h = candidates[0][0]
        top = [c for c in candidates if c[0] == best_h]
        chosen = rng.choice(top)
        _, direction, new_b, score_delta = chosen

        moves.append(direction)
        total_score += score_delta
        # Spawn a new tile (simulate real game)
        new_b = _add_random_tile(new_b, rng)
        current = new_b

    return moves, total_score


# ---------------------------------------------------------------------------
# Starting board generation
# ---------------------------------------------------------------------------

def _generate_start_board(rng: random.Random) -> list[list[int]]:
    """Run a random 2048 game for 8-20 moves to get a realistic starting board."""
    board = [[0] * 4 for _ in range(4)]
    # Seed with two tiles
    board = _add_random_tile(board, rng)
    board = _add_random_tile(board, rng)

    n_moves = rng.randint(8, 20)
    for _ in range(n_moves):
        dirs = DIRECTIONS[:]
        rng.shuffle(dirs)
        for d in dirs:
            new_b, _, moved = _move(board, d)
            if moved:
                board = _add_random_tile(new_b, rng)
                break
        else:
            break  # no valid move
    return board


def _difficulty_from_board(board: list[list[int]]) -> str:
    mx = _max_tile(board)
    if mx <= 64:
        return "easy"
    if mx <= 256:
        return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_puzzles(n: int = 40, seed: int = 42) -> list[Game2048Puzzle]:
    """Generate n Game2048Puzzle instances."""
    rng = random.Random(seed)
    puzzles: list[Game2048Puzzle] = []
    attempts = 0

    while len(puzzles) < n and attempts < n * 20:
        attempts += 1
        board = _generate_start_board(rng)

        if not _has_valid_move(board):
            continue

        # Greedy-optimal sequence from this position
        local_rng = random.Random(rng.randint(0, 2**31))
        moves, score = _greedy_sequence(board, local_rng)

        if len(moves) < 5:
            continue  # need at least 5 moves

        if score == 0:
            continue  # no merges happened — boring

        solution_str = ", ".join(moves)
        difficulty = _difficulty_from_board(board)

        puzzles.append(Game2048Puzzle(
            puzzle_id=len(puzzles) + 1,
            board=[row[:] for row in board],
            optimal_moves=moves,
            score_gain=score,
            solution=solution_str,
            difficulty=difficulty,
        ))

    return puzzles


def solve_puzzle(puzzle: Game2048Puzzle) -> str:
    """Return the solution string (comma-separated moves)."""
    return puzzle.solution or ", ".join(puzzle.optimal_moves)


# ---------------------------------------------------------------------------
# CoT helper
# ---------------------------------------------------------------------------

def _board_to_str(board: list[list[int]]) -> str:
    lines = []
    for row in board:
        lines.append("  " + "  ".join(f"{v:4d}" if v else "   ." for v in row))
    return "\n".join(lines)


def build_step_descriptions(puzzle: Game2048Puzzle) -> str:
    """Build a chain-of-thought description of the greedy move sequence."""
    lines: list[str] = []
    lines.append("=== 2048 Move Sequence (Greedy Optimal) ===")
    lines.append(f"Starting board (score gain to achieve: {puzzle.score_gain}):")
    lines.append(_board_to_str(puzzle.board))
    lines.append("")

    board = [row[:] for row in puzzle.board]
    cumulative = 0
    rng = random.Random(0)  # deterministic for display

    for step, direction in enumerate(puzzle.optimal_moves, 1):
        new_b, score_delta, moved = _move(board, direction)
        cumulative += score_delta
        lines.append(f"Step {step}: Move {direction}")
        lines.append(f"  Score this move: +{score_delta}  (cumulative: {cumulative})")
        lines.append(f"  Empty cells after move: {len(_get_empty_cells(new_b))}")
        lines.append(_board_to_str(new_b))
        new_b = _add_random_tile(new_b, rng)
        board = new_b
        lines.append("")

    lines.append(f"Total score gain: {puzzle.score_gain}")
    lines.append(f"Optimal move sequence: {puzzle.solution}")
    return "\n".join(lines)
