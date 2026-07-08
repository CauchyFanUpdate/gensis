import random
import copy
from collections import deque
from dataclasses import dataclass
from math import gcd
from typing import Optional


PLAYER_COLORS = [
    "#E74C3C",  # red
    "#3498DB",  # blue
    "#2ECC71",  # green
    "#F39C12",  # orange
    "#9B59B6",  # purple
]

PLAYER_SHAPES = ["circle", "square", "triangle"]

PLAYER_IDS = ["A", "B", "C", "D", "E"]


@dataclass
class MazePuzzle:
    puzzle_id: int
    grid: list          # list[list[int]]  0=open, 1=wall
    rows: int
    cols: int
    start: tuple        # (r, c) anchor used during generation
    end: tuple          # (r, c) exit cell
    players: list       # list of {"id", "pos", "color", "shape"}
    solution: Optional[str]
    question_type: str  # "winner" or "probability"
    difficulty: str


def _generate_maze(rows: int, cols: int, rng: random.Random) -> list:
    """Recursive-backtracking DFS on odd-dimension grid."""
    grid = [[1] * cols for _ in range(rows)]

    def carve(r, c):
        grid[r][c] = 0
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        rng.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 1:
                grid[r + dr // 2][c + dc // 2] = 0
                carve(nr, nc)

    carve(1, 1)
    return grid


def _open_cells(grid: list, rows: int, cols: int) -> list:
    return [(r, c) for r in range(rows) for c in range(cols) if grid[r][c] == 0]


def _bfs_reach_set(grid: list, rows: int, cols: int, src: tuple) -> set:
    visited = set()
    if grid[src[0]][src[1]] == 1:
        return visited
    queue = deque([src])
    visited.add(src)
    while queue:
        r, c = queue.popleft()
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in visited:
                visited.add((nr, nc))
                queue.append((nr, nc))
    return visited


def _bfs_reachable(grid: list, rows: int, cols: int, src: tuple, dst: tuple) -> bool:
    return dst in _bfs_reach_set(grid, rows, cols, src)


def _dead_end_neighbours(grid: list, rows: int, cols: int, cell: tuple) -> list:
    """Return open neighbours of cell that are corridor connectors (degree-2 nodes)."""
    r, c = cell
    neighbours = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            neighbours.append((nr, nc))
    return neighbours


def _try_isolate(grid: list, rows: int, cols: int, cell: tuple, end: tuple, rng: random.Random) -> Optional[tuple]:
    """
    Try to find a single wall-insertion point that disconnects `cell` from `end`.
    Returns the wall cell to block, or None if impossible without also trapping `end`.
    """
    neighbours = _dead_end_neighbours(grid, rows, cols, cell)
    rng.shuffle(neighbours)
    for nb in neighbours:
        test = copy.deepcopy(grid)
        test[nb[0]][nb[1]] = 1
        if not _bfs_reachable(test, rows, cols, cell, end) and _bfs_reachable(test, rows, cols, (1, 1), end):
            return nb
    return None


def _simplify_fraction(num: int, den: int) -> str:
    if den == 0 or num == 0:
        return "0"
    g = gcd(num, den)
    n, d = num // g, den // g
    if d == 1:
        return str(n)
    return f"{n}/{d}"


def _difficulty_for_size(rows: int, cols: int) -> str:
    area = rows * cols
    if area <= 11 * 11:
        return "easy"
    if area <= 13 * 13:
        return "medium"
    return "hard"


def _make_puzzle(puzzle_id: int, rng: random.Random) -> MazePuzzle:
    size_choices = [11, 13, 15, 17]
    rows = rng.choice(size_choices)
    cols = rng.choice(size_choices)

    grid = _generate_maze(rows, cols, rng)
    opens = _open_cells(grid, rows, cols)

    end = rng.choice(opens)
    while end == (1, 1):
        end = rng.choice(opens)

    n_players = rng.randint(3, 5)
    n_trapped = rng.randint(1, max(1, n_players - 1))

    candidate_positions = [p for p in opens if p != end]
    rng.shuffle(candidate_positions)

    working_grid = copy.deepcopy(grid)
    trapped_cells = set()
    free_positions = []

    for pos in candidate_positions:
        if len(trapped_cells) + len(free_positions) >= n_players:
            break
        if pos in trapped_cells or pos == end:
            continue

        if len(trapped_cells) < n_trapped:
            wall_nb = _try_isolate(working_grid, rows, cols, pos, end, rng)
            if wall_nb is not None:
                working_grid[wall_nb[0]][wall_nb[1]] = 1
                trapped_cells.add(pos)
                continue

        free_positions.append(pos)

    while len(free_positions) + len(trapped_cells) < n_players:
        remaining = [p for p in opens if p != end and p not in trapped_cells and p not in free_positions]
        if not remaining:
            break
        free_positions.append(remaining[0])

    all_positions = list(trapped_cells) + free_positions
    rng.shuffle(all_positions)
    all_positions = all_positions[:n_players]

    shapes = list(PLAYER_SHAPES)
    rng.shuffle(shapes)

    players = []
    for i, pos in enumerate(all_positions):
        players.append({
            "id": PLAYER_IDS[i],
            "pos": pos,
            "color": PLAYER_COLORS[i % len(PLAYER_COLORS)],
            "shape": shapes[i % len(shapes)],
        })

    winners = [p["id"] for p in players if _bfs_reachable(working_grid, rows, cols, p["pos"], end)]

    question_type = rng.choice(["winner", "probability"])

    if question_type == "winner":
        solution = ", ".join(sorted(winners)) if winners else "none"
    else:
        solution = _simplify_fraction(len(winners), len(players))

    difficulty = _difficulty_for_size(rows, cols)

    return MazePuzzle(
        puzzle_id=puzzle_id,
        grid=working_grid,
        rows=rows,
        cols=cols,
        start=(1, 1),
        end=end,
        players=players,
        solution=solution,
        question_type=question_type,
        difficulty=difficulty,
    )


def generate_puzzles(n: int = 30, seed: int = 42) -> list:
    rng = random.Random(seed)
    puzzles = []
    for i in range(n):
        puzzle = _make_puzzle(puzzle_id=i + 1, rng=rng)
        puzzles.append(puzzle)
    return puzzles


def solve_puzzle(puzzle: MazePuzzle) -> str:
    return puzzle.solution


def build_step_descriptions(puzzle: MazePuzzle) -> str:
    lines = []
    lines.append(f"Maze size: {puzzle.rows}x{puzzle.cols}")
    lines.append(f"Exit position: {puzzle.end}")
    lines.append(f"Question type: {puzzle.question_type}")
    lines.append("")

    for p in puzzle.players:
        can_reach = _bfs_reachable(puzzle.grid, puzzle.rows, puzzle.cols, p["pos"], puzzle.end)
        status = "can reach exit" if can_reach else "is trapped"
        lines.append(f"Player {p['id']} at {p['pos']} ({p['color']}, {p['shape']}): {status}")

    lines.append("")
    if puzzle.question_type == "winner":
        lines.append(f"Winners: {puzzle.solution}")
    else:
        winners = [p for p in puzzle.players
                   if _bfs_reachable(puzzle.grid, puzzle.rows, puzzle.cols, p["pos"], puzzle.end)]
        lines.append(f"Players reaching exit: {len(winners)} out of {len(puzzle.players)}")
        lines.append(f"Probability: {puzzle.solution}")

    return "\n".join(lines)
