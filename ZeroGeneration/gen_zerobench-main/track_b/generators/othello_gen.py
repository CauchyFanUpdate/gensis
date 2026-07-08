from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional


SIZE = 8
DIRECTIONS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


@dataclass
class OthelloPuzzle:
    puzzle_id: int
    board: list[list[int]]      # 8×8, 0=empty, 1=Black, 2=White
    current_player: int         # 1=Black, 2=White
    best_move: tuple[int, int]  # (row, col) 0-indexed
    flipped_count: int          # pieces flipped by best_move
    solution: Optional[str]     # number of pieces flipped by best_move
    solution_format: str        # "flipped_count" or "move_coord"
    difficulty: str


def _get_flips(board: list[list[int]], row: int, col: int, player: int) -> list[tuple[int, int]]:
    """Return list of positions that would be flipped by placing player at (row, col)."""
    if board[row][col] != 0:
        return []
    opponent = 3 - player
    flips: list[tuple[int, int]] = []
    for dr, dc in DIRECTIONS:
        candidates: list[tuple[int, int]] = []
        r, c = row + dr, col + dc
        while 0 <= r < SIZE and 0 <= c < SIZE and board[r][c] == opponent:
            candidates.append((r, c))
            r += dr
            c += dc
        if candidates and 0 <= r < SIZE and 0 <= c < SIZE and board[r][c] == player:
            flips.extend(candidates)
    return flips


def _get_valid_moves(board: list[list[int]], player: int) -> list[tuple[int, int]]:
    """Return all valid (row, col) positions for player."""
    return [
        (r, c)
        for r in range(SIZE)
        for c in range(SIZE)
        if board[r][c] == 0 and _get_flips(board, r, c, player)
    ]


def _apply_move(board: list[list[int]], row: int, col: int, player: int) -> list[list[int]]:
    """Return new board after placing player at (row, col) and flipping pieces."""
    new_board = [r[:] for r in board]
    new_board[row][col] = player
    for (r, c) in _get_flips(board, row, col, player):
        new_board[r][c] = player
    return new_board


def _initial_board() -> list[list[int]]:
    board = [[0] * SIZE for _ in range(SIZE)]
    board[3][3] = 2
    board[3][4] = 1
    board[4][3] = 1
    board[4][4] = 2
    return board


def _find_best_move(board: list[list[int]], player: int) -> tuple[Optional[tuple[int, int]], int]:
    """Return (move, flipped_count) for the move that flips the most pieces."""
    best_move: Optional[tuple[int, int]] = None
    best_count = -1
    for r, c in _get_valid_moves(board, player):
        count = len(_get_flips(board, r, c, player))
        if count > best_count:
            best_count = count
            best_move = (r, c)
    return best_move, best_count


def _count_pieces(board: list[list[int]]) -> int:
    return sum(cell != 0 for row in board for cell in row)


def _difficulty_from_pieces(n: int) -> str:
    if n <= 20:
        return "easy"
    if n <= 36:
        return "medium"
    return "hard"


def _simulate_game(rng: random.Random, n_moves: int) -> tuple[list[list[int]], int]:
    """Simulate a random Othello game for up to n_moves half-moves."""
    board = _initial_board()
    player = 1
    for _ in range(n_moves):
        moves = _get_valid_moves(board, player)
        if not moves:
            player = 3 - player
            moves = _get_valid_moves(board, player)
            if not moves:
                break
        move = rng.choice(moves)
        board = _apply_move(board, move[0], move[1], player)
        player = 3 - player
    return board, player


def generate_puzzles(n: int = 40, seed: int = 42) -> list[OthelloPuzzle]:
    rng = random.Random(seed)
    puzzles: list[OthelloPuzzle] = []
    attempts = 0
    puzzle_id = 1

    while len(puzzles) < n and attempts < n * 200:
        attempts += 1
        n_moves = rng.randint(15, 40)
        board, player = _simulate_game(rng, n_moves)

        valid_moves = _get_valid_moves(board, player)
        if len(valid_moves) < 2:
            continue

        best_move, best_count = _find_best_move(board, player)
        if best_move is None or best_count < 1:
            continue

        n_pieces = _count_pieces(board)
        difficulty = _difficulty_from_pieces(n_pieces)

        puzzles.append(OthelloPuzzle(
            puzzle_id=puzzle_id,
            board=board,
            current_player=player,
            best_move=best_move,
            flipped_count=best_count,
            solution=str(best_count),
            solution_format="flipped_count",
            difficulty=difficulty,
        ))
        puzzle_id += 1

    return puzzles


def solve_puzzle(puzzle: OthelloPuzzle) -> str:
    """Return the number of pieces flipped by the best move as a string."""
    best_move, best_count = _find_best_move(puzzle.board, puzzle.current_player)
    return str(best_count) if best_move is not None else "0"


def build_step_descriptions(puzzle: OthelloPuzzle) -> str:
    player_name = "Black" if puzzle.current_player == 1 else "White"
    opponent_name = "White" if puzzle.current_player == 1 else "Black"
    n_pieces = _count_pieces(puzzle.board)
    black_count = sum(cell == 1 for row in puzzle.board for cell in row)
    white_count = sum(cell == 2 for row in puzzle.board for cell in row)
    valid_moves = _get_valid_moves(puzzle.board, puzzle.current_player)

    lines = [
        f"Othello puzzle ({puzzle.difficulty}, {n_pieces} pieces on board).",
        f"Current player: {player_name} (player {puzzle.current_player}).",
        f"Opponent: {opponent_name} (player {3 - puzzle.current_player}).",
        f"Score — Black: {black_count}, White: {white_count}.",
        "",
        f"Valid moves for {player_name} ({len(valid_moves)} total):",
    ]

    move_details = []
    for (r, c) in valid_moves:
        count = len(_get_flips(puzzle.board, r, c, puzzle.current_player))
        is_best = (r, c) == puzzle.best_move
        tag = " ← BEST MOVE" if is_best else ""
        move_details.append((count, r, c, tag))

    move_details.sort(key=lambda x: -x[0])
    for count, r, c, tag in move_details:
        lines.append(
            f"  Row {r + 1}, Col {c + 1}: flips {count} piece(s){tag}"
        )

    r, c = puzzle.best_move
    lines.append("")
    lines.append(
        f"Best move: Row {r + 1}, Col {c + 1} — flips {puzzle.flipped_count} piece(s)."
    )
    lines.append(f"Answer (flipped_count): {puzzle.flipped_count}")
    return "\n".join(lines)
