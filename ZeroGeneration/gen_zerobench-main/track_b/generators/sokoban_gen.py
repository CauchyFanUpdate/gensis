"""
Sokoban level generator, solver, and data utilities.

Level format characters:
  '#' wall  ' ' floor  '@' player  '$' box  '.' target  '*' box-on-target  '+' player-on-target
"""

import random
import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Public-domain Microban levels (a subset of 50 levels by David Skinner)
# ---------------------------------------------------------------------------
MICROBAN_LEVELS = [
    # 1
    ["#####",
     "#   #",
     "#$  #",
     "###.##",
     "#    ##",
     "## ### #",
     "#@ #   #",
     "#   $  #",
     "######  #",
     "   #####"],
    # 2
    ["######",
     "#    #",
     "# #@ #",
     "#  $ #",
     "## . #",
     "#    #",
     "######"],
    # 3
    ["  #####",
     "  #   #",
     "  # # #",
     "###$. #",
     "#  $  #",
     "# @ . #",
     "#######"],
    # 4
    ["########",
     "#      #",
     "# .**@ #",
     "#      #",
     "## $####",
     " #  #",
     " ####"],
    # 5
    ["#####",
     "#.  ##",
     "#@$  #",
     "## $ #",
     " #.  #",
     " #####"],
    # 6
    ["######",
     "#    #",
     "# $@ #",
     "##.###",
     " # $  ##",
     " #.    #",
     " ########"],
    # 7
    ["  ###",
     "###.#####",
     "#   $   #",
     "# @ $ . #",
     "#########"],
    # 8
    ["#####",
     "#@$ #",
     "#   #",
     "# $ #",
     "#.#.#",
     "#   #",
     "#####"],
    # 9
    ["######",
     "#    #",
     "#@   #",
     "## # #",
     "#.$$. #",
     "#    ##",
     "######"],
    # 10
    ["########",
     "#  @   #",
     "#  $## #",
     "## #   #",
     "#.##$  #",
     "#.  $  #",
     "#   .  #",
     "########"],
    # 11
    ["  ####",
     "###  ####",
     "#   $   #",
     "#  $@$  #",
     "### . ###",
     "  #. .#",
     "  #####"],
    # 12
    ["########",
     "#  .   #",
     "#  $   #",
     "# .$@# #",
     "# $    #",
     "#  .   #",
     "########"],
    # 13
    ["#######",
     "#     #",
     "# .$. #",
     "# $@$ #",
     "# .$. #",
     "#     #",
     "#######"],
    # 14
    [" #####",
     " #   ##",
     "##$#  #",
     "#  $. #",
     "# @.  #",
     "#  #  #",
     "## ####"],
    # 15
    ["########",
     "#      #",
     "# $$$  #",
     "# .@.  #",
     "# . .  #",
     "#      #",
     "########"],
    # 16
    ["#####",
     "# @ #",
     "#   #",
     "#$#.#",
     "#   #",
     "#$#.#",
     "#   #",
     "#####"],
    # 17
    ["  ####",
     "###  #",
     "#    #",
     "#  # #",
     "# @$ #",
     "## $ #",
     " #.  #",
     " #. ##",
     " ####"],
    # 18
    ["#######",
     "#     #",
     "# @   #",
     "#$##$ #",
     "#  #  #",
     "#. #. #",
     "#######"],
    # 19
    ["  ####",
     "###  #",
     "#  . #",
     "# .$ #",
     "##$@ #",
     " #   #",
     " #####"],
    # 20
    ["######",
     "#    ##",
     "#  $  #",
     "# $.  #",
     "##.@  #",
     " #    #",
     " ######"],
]


@dataclass
class SokobanState:
    player: tuple          # (row, col)
    boxes: frozenset       # frozenset of (row, col)

    def __hash__(self):
        return hash((self.player, self.boxes))

    def __eq__(self, other):
        return self.player == other.player and self.boxes == other.boxes


@dataclass
class SokobanPuzzle:
    grid: list             # list of strings (the original level layout)
    walls: frozenset       # frozenset of (row, col)
    targets: frozenset     # frozenset of (row, col)
    init_state: SokobanState
    solution: Optional[list] = None   # list of direction strings after solving
    level_id: int = 0

    @property
    def n_boxes(self):
        return len(self.init_state.boxes)

    def is_solved(self, state: SokobanState) -> bool:
        return state.boxes == self.targets


MOVE_DIRS = {
    "up":    (-1,  0),
    "down":  ( 1,  0),
    "left":  ( 0, -1),
    "right": ( 0,  1),
}

MOVE_NAMES = ["up", "down", "left", "right"]


def parse_level(lines: list) -> SokobanPuzzle:
    walls, boxes, targets = set(), set(), set()
    player = None

    # Normalise rows to same width
    width = max(len(r) for r in lines)
    grid = [r.ljust(width) for r in lines]

    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            pos = (r, c)
            if ch == '#':
                walls.add(pos)
            elif ch == '@':
                player = pos
            elif ch == '+':   # player on target
                player = pos
                targets.add(pos)
            elif ch == '$':
                boxes.add(pos)
            elif ch == '*':   # box on target
                boxes.add(pos)
                targets.add(pos)
            elif ch == '.':
                targets.add(pos)

    if player is None:
        raise AssertionError("Level has no player")
    if len(boxes) != len(targets):
        raise AssertionError(f"boxes({len(boxes)}) != targets({len(targets)})")

    state = SokobanState(player=player, boxes=frozenset(boxes))
    return SokobanPuzzle(
        grid=grid,
        walls=frozenset(walls),
        targets=frozenset(targets),
        init_state=state,
    )


class SokobanSolver:
    """BFS solver. Returns list of move names (up/down/left/right) or None."""

    def __init__(self, puzzle: SokobanPuzzle, max_states: int = 500_000):
        self.puzzle = puzzle
        self.max_states = max_states

    def _successors(self, state: SokobanState):
        r, c = state.player
        walls = self.puzzle.walls
        for name in MOVE_NAMES:
            dr, dc = MOVE_DIRS[name]
            nr, nc = r + dr, c + dc
            if (nr, nc) in walls:
                continue
            if (nr, nc) in state.boxes:
                # Try to push box
                br, bc = nr + dr, nc + dc
                if (br, bc) in walls or (br, bc) in state.boxes:
                    continue
                new_boxes = (state.boxes - {(nr, nc)}) | {(br, bc)}
                yield name, SokobanState(player=(nr, nc), boxes=frozenset(new_boxes))
            else:
                yield name, SokobanState(player=(nr, nc), boxes=state.boxes)

    def solve(self) -> Optional[list]:
        start = self.puzzle.init_state
        if self.puzzle.is_solved(start):
            return []

        queue = deque([(start, [])])
        visited = {start}

        while queue:
            state, path = queue.popleft()
            if len(visited) > self.max_states:
                return None
            for move, next_state in self._successors(state):
                if next_state in visited:
                    continue
                new_path = path + [move]
                if self.puzzle.is_solved(next_state):
                    return new_path
                visited.add(next_state)
                queue.append((next_state, new_path))
        return None


def get_all_puzzles(shuffle: bool = False, seed: int = 42) -> list:
    """Return list of SokobanPuzzle objects parsed from MICROBAN_LEVELS."""
    puzzles = []
    for idx, level_lines in enumerate(MICROBAN_LEVELS):
        try:
            puzzle = parse_level(level_lines)
            puzzle.level_id = idx + 1
            puzzles.append(puzzle)
        except AssertionError:
            pass  # skip malformed levels
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(puzzles)
    return puzzles


def solve_puzzle(puzzle: SokobanPuzzle, max_states: int = 500_000) -> Optional[list]:
    solver = SokobanSolver(puzzle, max_states=max_states)
    return solver.solve()


def get_state_sequence(puzzle: SokobanPuzzle, moves: list) -> list:
    """Replay a solution and return the sequence of (state, move) pairs for CoT."""
    states = [puzzle.init_state]
    state = puzzle.init_state
    for move in moves:
        dr, dc = MOVE_DIRS[move]
        r, c = state.player
        nr, nc = r + dr, c + dc
        if (nr, nc) in state.boxes:
            br, bc = nr + dr, nc + dc
            new_boxes = (state.boxes - {(nr, nc)}) | {(br, bc)}
            state = SokobanState(player=(nr, nc), boxes=frozenset(new_boxes))
        else:
            state = SokobanState(player=(nr, nc), boxes=state.boxes)
        states.append(state)
    return states


def grid_to_str(puzzle: SokobanPuzzle, state: SokobanState) -> str:
    """Render current state as ASCII string for CoT description."""
    rows = list(puzzle.grid)
    # Build mutable grid
    grid = [list(row) for row in rows]
    # Clear dynamic cells
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch in '@+$*.':
                grid[r][c] = '.' if (r, c) in puzzle.targets else ' '
    # Place boxes
    for (r, c) in state.boxes:
        grid[r][c] = '*' if (r, c) in puzzle.targets else '$'
    # Place player
    pr, pc = state.player
    grid[pr][pc] = '+' if (pr, pc) in puzzle.targets else '@'
    return '\n'.join(''.join(row) for row in grid)


def build_step_descriptions(puzzle: SokobanPuzzle, moves: list) -> list:
    """Return list of dicts describing each move step (for CoT generation)."""
    states = get_state_sequence(puzzle, moves)
    steps = []
    for i, move in enumerate(moves):
        dr, dc = MOVE_DIRS[move]
        r, c = states[i].player
        nr, nc = r + dr, c + dc
        pushed = (nr, nc) in states[i].boxes
        step = {
            "step": i + 1,
            "move": move,
            "player_from": (r, c),
            "player_to": (nr, nc),
            "pushed_box": pushed,
            "box_from": (nr, nc) if pushed else None,
            "box_to": (nr + dr, nc + dc) if pushed else None,
            "is_final": (i == len(moves) - 1),
        }
        steps.append(step)
    return steps
