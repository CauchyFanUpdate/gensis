"""
Colour Nonogram (picross) puzzle generator.

Generates binary nonogram puzzles derived from pixel-art templates
that depict simple recognisable objects (heart, star, house, …).

ZeroBench Q88:
  "Colour nonograms are logic puzzles played on a grid, where the goal
   is to fill squares with colours.  What does the solved nonogram depict?"
  Answer: e.g. "pen"
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class NonogramPuzzle:
    puzzle_id: int
    rows: int
    cols: int
    solution_grid: list[list[int]]   # 0=empty, 1=filled
    row_clues: list[list[int]]       # run-length of 1s per row
    col_clues: list[list[int]]       # run-length of 1s per column
    depicted_object: str             # e.g. "heart"
    solution: Optional[str]          # = depicted_object
    difficulty: str


# ---------------------------------------------------------------------------
# Pixel-art templates  (8×8 grids, 0=background, 1=filled)
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, list[list[int]]] = {
    "heart": [
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "star": [
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "house": [
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
    ],
    "tree": [
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
    ],
    "fish": [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "bird": [
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "cat": [
        [1, 1, 0, 0, 0, 0, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 0, 1, 1],
    ],
    "dog": [
        [1, 0, 0, 0, 0, 0, 0, 1],
        [1, 1, 0, 0, 0, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "flower": [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 0, 1, 1, 0, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
    ],
    "mushroom": [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "moon": [
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
    ],
    "sun": [
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 0, 1, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "key": [
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 1, 1, 0, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
    ],
    "cup": [
        [0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0, 1, 1, 0],
        [1, 0, 0, 0, 0, 1, 1, 0],
        [1, 0, 0, 0, 0, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
    ],
    "arrow": [
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "pen": [
        [0, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
    ],
    "boat": [
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "flag": [
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0],
    ],
    "crown": [
        [1, 0, 0, 1, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 0, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
    "hand": [
        [0, 1, 0, 1, 0, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1, 0],
        [0, 1, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0],
    ],
}

TEMPLATE_NAMES: list[str] = list(TEMPLATES.keys())  # 20 objects


# ---------------------------------------------------------------------------
# Clue computation
# ---------------------------------------------------------------------------

def _compute_clues(line: list[int]) -> list[int]:
    """Return run-length encoding of 1s in line.

    e.g. [0,1,1,0,1,0] -> [2, 1]
    Empty row/col returns [0].
    """
    runs: list[int] = []
    count = 0
    for v in line:
        if v == 1:
            count += 1
        else:
            if count > 0:
                runs.append(count)
                count = 0
    if count > 0:
        runs.append(count)
    return runs if runs else [0]


def _derive_clues(
    grid: list[list[int]],
) -> tuple[list[list[int]], list[list[int]]]:
    """Compute row_clues and col_clues from a solution grid."""
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    row_clues = [_compute_clues(grid[r]) for r in range(rows)]
    col_clues = [_compute_clues([grid[r][c] for r in range(rows)]) for c in range(cols)]
    return row_clues, col_clues


def _difficulty_from_grid(grid: list[list[int]]) -> str:
    rows = len(grid)
    cols = len(grid[0])
    filled = sum(grid[r][c] for r in range(rows) for c in range(cols))
    density = filled / (rows * cols)
    _, col_clues = _derive_clues(grid)
    avg_groups = sum(len(c) for c in col_clues) / max(len(col_clues), 1)
    if density < 0.35 or avg_groups < 1.5:
        return "easy"
    if density < 0.55 or avg_groups < 2.5:
        return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_puzzles(n: int = 30, seed: int = 42) -> list[NonogramPuzzle]:
    """Generate n NonogramPuzzle instances by cycling through templates."""
    rng = random.Random(seed)
    names = TEMPLATE_NAMES[:]
    rng.shuffle(names)

    puzzles: list[NonogramPuzzle] = []
    idx = 0
    for i in range(n):
        name = names[idx % len(names)]
        idx += 1
        grid = [row[:] for row in TEMPLATES[name]]
        row_clues, col_clues = _derive_clues(grid)
        difficulty = _difficulty_from_grid(grid)
        puzzles.append(NonogramPuzzle(
            puzzle_id=i + 1,
            rows=len(grid),
            cols=len(grid[0]),
            solution_grid=grid,
            row_clues=row_clues,
            col_clues=col_clues,
            depicted_object=name,
            solution=name,
            difficulty=difficulty,
        ))
    return puzzles


def solve_puzzle(puzzle: NonogramPuzzle) -> str:
    """Return the depicted object (the answer)."""
    return puzzle.depicted_object


# ---------------------------------------------------------------------------
# CoT helper
# ---------------------------------------------------------------------------

def _grid_to_str(grid: list[list[int]], show: bool = True) -> str:
    lines = []
    for row in grid:
        lines.append(" ".join("█" if v else "·" for v in row))
    return "\n".join(lines)


def _clues_str(clues: list[list[int]]) -> str:
    return "  ".join(",".join(str(v) for v in c) for c in clues)


def build_step_descriptions(puzzle: NonogramPuzzle) -> str:
    """Build a chain-of-thought description for solving the nonogram."""
    lines: list[str] = []
    lines.append(f"=== Nonogram: '{puzzle.depicted_object}' ({puzzle.rows}×{puzzle.cols}) ===")
    lines.append(f"Difficulty: {puzzle.difficulty}")
    lines.append("")
    lines.append("Row clues (left → right):")
    for r, clue in enumerate(puzzle.row_clues):
        lines.append(f"  Row {r+1}: {clue}")
    lines.append("")
    lines.append("Column clues (top → bottom):")
    for c, clue in enumerate(puzzle.col_clues):
        lines.append(f"  Col {c+1}: {clue}")
    lines.append("")
    lines.append("Solution grid (█=filled, ·=empty):")
    lines.append(_grid_to_str(puzzle.solution_grid))
    lines.append("")
    lines.append(f"The solved nonogram depicts: {puzzle.depicted_object}")
    return "\n".join(lines)
