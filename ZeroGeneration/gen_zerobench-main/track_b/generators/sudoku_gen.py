"""
Sudoku puzzle generator with unique-solution guarantee.

Generates standard 9x9 Sudoku puzzles. Each puzzle:
- Has exactly one solution (verified by solver)
- Difficulty is controlled by number of clues (25-35 givens)
- Full step-by-step solution available for CoT generation
"""

import random
import copy
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SudokuPuzzle:
    clues: list          # 9x9 grid, 0 = empty, 1-9 = given
    solution: list       # 9x9 grid, fully filled
    solve_steps: list    # list of (row, col, value, reasoning) for CoT
    difficulty: str      # 'easy' / 'medium' / 'hard'
    puzzle_id: int = 0


def _is_valid(grid: list, row: int, col: int, num: int) -> bool:
    """Check if num can be placed at grid[row][col]."""
    if num in grid[row]:
        return False
    if num in [grid[r][col] for r in range(9)]:
        return False
    br, bc = 3 * (row // 3), 3 * (col // 3)
    for r in range(br, br + 3):
        for c in range(bc, bc + 3):
            if grid[r][c] == num:
                return False
    return True


def _solve(grid: list, find_all: bool = False) -> list:
    """Backtracking solver. Returns list of solutions (up to 2 if find_all=False)."""
    solutions = []

    def _bt(g):
        if len(solutions) > 1:
            return
        # Find next empty cell
        for r in range(9):
            for c in range(9):
                if g[r][c] == 0:
                    for num in range(1, 10):
                        if _is_valid(g, r, c, num):
                            g[r][c] = num
                            _bt(g)
                            g[r][c] = 0
                    return
        solutions.append([row[:] for row in g])

    _bt([row[:] for row in grid])
    return solutions


def _generate_filled_grid(rng: random.Random) -> list:
    """Generate a complete valid 9x9 Sudoku grid."""
    grid = [[0] * 9 for _ in range(9)]

    def fill(pos=0):
        if pos == 81:
            return True
        r, c = divmod(pos, 9)
        nums = list(range(1, 10))
        rng.shuffle(nums)
        for num in nums:
            if _is_valid(grid, r, c, num):
                grid[r][c] = num
                if fill(pos + 1):
                    return True
                grid[r][c] = 0
        return False

    fill()
    return grid


def _remove_cells(filled: list, n_remove: int, rng: random.Random) -> Optional[list]:
    """Remove n_remove cells while keeping unique solution. Returns clue grid or None."""
    clues = [row[:] for row in filled]
    positions = list(range(81))
    rng.shuffle(positions)
    removed = 0
    for pos in positions:
        if removed >= n_remove:
            break
        r, c = divmod(pos, 9)
        backup = clues[r][c]
        clues[r][c] = 0
        sols = _solve(clues)
        if len(sols) == 1:
            removed += 1
        else:
            clues[r][c] = backup  # restore if not unique
    return clues if removed > 0 else None


def _get_solve_steps(clues: list, solution: list) -> list:
    """
    Record step-by-step solve order (cells filled in sequence).
    Uses a simple naked-single pass first, then backtracking order.
    Returns list of dicts with row, col, value, method.
    """
    steps = []
    grid = [row[:] for row in clues]

    def get_candidates(g, r, c):
        used = set(g[r])
        used |= {g[i][c] for i in range(9)}
        br, bc = 3 * (r // 3), 3 * (c // 3)
        for i in range(br, br + 3):
            for j in range(bc, bc + 3):
                used.add(g[i][j])
        return [n for n in range(1, 10) if n not in used]

    changed = True
    while changed:
        changed = False
        for r in range(9):
            for c in range(9):
                if grid[r][c] == 0:
                    cands = get_candidates(grid, r, c)
                    if len(cands) == 1:
                        grid[r][c] = cands[0]
                        steps.append({
                            "step": len(steps) + 1,
                            "row": r + 1,
                            "col": c + 1,
                            "value": cands[0],
                            "method": "naked_single",
                            "reasoning": f"Row {r+1}, Col {c+1}: only {cands[0]} fits (naked single)"
                        })
                        changed = True

    # Remaining cells (filled by trial)
    for r in range(9):
        for c in range(9):
            if grid[r][c] == 0:
                val = solution[r][c]
                grid[r][c] = val
                steps.append({
                    "step": len(steps) + 1,
                    "row": r + 1,
                    "col": c + 1,
                    "value": val,
                    "method": "trial",
                    "reasoning": f"Row {r+1}, Col {c+1}: place {val} (trial and error)"
                })
    return steps


DIFFICULTY_RANGES = {
    "easy":   (36, 46),  # 35-45 clues → 36-45 givens
    "medium": (27, 35),
    "hard":   (23, 27),
}


def generate_puzzle(
    difficulty: str = "medium",
    seed: Optional[int] = None,
    puzzle_id: int = 0,
) -> SudokuPuzzle:
    """Generate a Sudoku puzzle with unique solution."""
    rng = random.Random(seed)
    filled = _generate_filled_grid(rng)
    low, high = DIFFICULTY_RANGES[difficulty]
    n_givens = rng.randint(low, high)
    n_remove = 81 - n_givens
    clues = _remove_cells(filled, n_remove, rng)
    if clues is None:
        clues = [row[:] for row in filled]
    steps = _get_solve_steps(clues, filled)
    return SudokuPuzzle(
        clues=clues,
        solution=filled,
        solve_steps=steps,
        difficulty=difficulty,
        puzzle_id=puzzle_id,
    )


def generate_puzzles(
    n: int = 50,
    difficulty: str = "medium",
    base_seed: int = 42,
) -> list:
    return [
        generate_puzzle(difficulty=difficulty, seed=base_seed + i, puzzle_id=i + 1)
        for i in range(n)
    ]


def format_grid(grid: list) -> str:
    """Pretty-print a 9x9 grid for debugging."""
    lines = []
    for i, row in enumerate(grid):
        if i % 3 == 0 and i > 0:
            lines.append("------+-------+------")
        cells = []
        for j, v in enumerate(row):
            if j % 3 == 0 and j > 0:
                cells.append("|")
            cells.append(str(v) if v else ".")
        lines.append(" ".join(cells))
    return "\n".join(lines)


def puzzle_to_question(puzzle: SudokuPuzzle) -> str:
    """Format puzzle clues as question text."""
    rows = []
    for i, row in enumerate(puzzle.clues):
        row_str = " ".join(str(v) if v else "_" for v in row)
        rows.append(f"Row {i+1}: {row_str}")
    grid_text = "\n".join(rows)
    return (
        "Solve the following Sudoku puzzle. Fill in each empty cell (_) with a digit "
        "from 1 to 9 such that every row, column, and 3×3 box contains each digit "
        f"exactly once.\n\n{grid_text}\n\n"
        "What are all the missing values? Provide the complete solved grid."
    )


def puzzle_to_answer(puzzle: SudokuPuzzle) -> str:
    """Format solution as answer text."""
    rows = []
    for i, row in enumerate(puzzle.solution):
        row_str = " ".join(str(v) for v in row)
        rows.append(f"Row {i+1}: {row_str}")
    return "\n".join(rows)
