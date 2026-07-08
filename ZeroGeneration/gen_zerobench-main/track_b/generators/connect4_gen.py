from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Optional


ROWS = 6
COLS = 7


@dataclass
class Connect4Puzzle:
    puzzle_id: int
    board: list[list[int]]      # 6×7, 0=empty, 1=Red(current player), 2=Yellow(opponent)
    current_player: int         # 1=Red, 2=Yellow
    winning_columns: list[int]  # 1-indexed columns where current player wins immediately
    solution: Optional[str]     # e.g. "3" or "3, 6"
    difficulty: str


def _drop_piece(board: list[list[int]], col: int, player: int) -> Optional[list[list[int]]]:
    """Return new board with piece dropped in col, or None if column is full."""
    new_board = [row[:] for row in board]
    for row in range(ROWS - 1, -1, -1):
        if new_board[row][col] == 0:
            new_board[row][col] = player
            return new_board
    return None


def _check_winner(board: list[list[int]]) -> int:
    """Return 0 (no win), 1, or 2."""
    for r in range(ROWS):
        for c in range(COLS):
            p = board[r][c]
            if p == 0:
                continue
            # Horizontal
            if c + 3 < COLS and all(board[r][c + k] == p for k in range(4)):
                return p
            # Vertical
            if r + 3 < ROWS and all(board[r + k][c] == p for k in range(4)):
                return p
            # Diagonal down-right
            if r + 3 < ROWS and c + 3 < COLS and all(board[r + k][c + k] == p for k in range(4)):
                return p
            # Diagonal down-left
            if r + 3 < ROWS and c - 3 >= 0 and all(board[r + k][c - k] == p for k in range(4)):
                return p
    return 0


def _find_winning_moves(board: list[list[int]], player: int) -> list[int]:
    """Return 1-indexed columns where dropping player's piece wins immediately."""
    wins = []
    for col in range(COLS):
        new_board = _drop_piece(board, col, player)
        if new_board is not None and _check_winner(new_board) == player:
            wins.append(col + 1)
    return wins


def _simulate_random_game(rng: random.Random, n_moves: int) -> tuple[list[list[int]], int]:
    """Simulate a random game for n_moves total half-moves. Returns (board, next_player)."""
    board = [[0] * COLS for _ in range(ROWS)]
    player = 1
    for _ in range(n_moves):
        valid = [c for c in range(COLS) if board[0][c] == 0]
        if not valid:
            break
        col = rng.choice(valid)
        board = _drop_piece(board, col, player)
        if _check_winner(board) != 0:
            # Game over — return board as-is; caller will discard
            return board, 3 - player
        player = 3 - player
    return board, player


def _count_pieces(board: list[list[int]]) -> int:
    return sum(cell != 0 for row in board for cell in row)


def _difficulty_from_pieces(n: int) -> str:
    if n <= 14:
        return "easy"
    if n <= 24:
        return "medium"
    return "hard"


def generate_puzzles(n: int = 40, seed: int = 42) -> list[Connect4Puzzle]:
    rng = random.Random(seed)
    puzzles: list[Connect4Puzzle] = []
    attempts = 0
    puzzle_id = 1

    while len(puzzles) < n and attempts < n * 200:
        attempts += 1
        n_moves = rng.randint(10, 25)
        board, player = _simulate_random_game(rng, n_moves)

        if _check_winner(board) != 0:
            continue

        winning_cols = _find_winning_moves(board, player)
        if not winning_cols:
            continue

        n_pieces = _count_pieces(board)
        difficulty = _difficulty_from_pieces(n_pieces)

        solution = ", ".join(str(c) for c in sorted(winning_cols))

        puzzles.append(Connect4Puzzle(
            puzzle_id=puzzle_id,
            board=board,
            current_player=player,
            winning_columns=sorted(winning_cols),
            solution=solution,
            difficulty=difficulty,
        ))
        puzzle_id += 1

    return puzzles


def solve_puzzle(puzzle: Connect4Puzzle) -> str:
    """Return comma-separated 1-indexed winning columns (e.g. '3' or '3, 6')."""
    cols = _find_winning_moves(puzzle.board, puzzle.current_player)
    return ", ".join(str(c) for c in sorted(cols))


def build_step_descriptions(puzzle: Connect4Puzzle) -> str:
    player_name = "Red" if puzzle.current_player == 1 else "Yellow"
    opponent_name = "Yellow" if puzzle.current_player == 1 else "Red"
    n_pieces = _count_pieces(puzzle.board)
    lines = [
        f"Connect 4 puzzle ({puzzle.difficulty}, {n_pieces} pieces on board).",
        f"Current player: {player_name} (player {puzzle.current_player}).",
        f"Opponent: {opponent_name} (player {3 - puzzle.current_player}).",
        "",
        "Winning move analysis:",
    ]
    for col in range(1, COLS + 1):
        new_board = _drop_piece(puzzle.board, col - 1, puzzle.current_player)
        if new_board is None:
            lines.append(f"  Column {col}: FULL (cannot play).")
            continue
        winner = _check_winner(new_board)
        if winner == puzzle.current_player:
            lines.append(f"  Column {col}: WINNING MOVE — completes a four-in-a-row for {player_name}.")
        else:
            lines.append(f"  Column {col}: valid but does not win immediately.")
    lines.append("")
    if len(puzzle.winning_columns) == 1:
        lines.append(f"Answer: Column {puzzle.winning_columns[0]}")
    else:
        col_str = ", ".join(str(c) for c in puzzle.winning_columns)
        lines.append(f"Answer: Columns {col_str}")
    return "\n".join(lines)
