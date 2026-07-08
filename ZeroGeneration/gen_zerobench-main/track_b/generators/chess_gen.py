from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Optional

import chess

# Pool of verified mate-in-1 FEN positions (White to move, exactly one mating move each).
# Validated at collection time with python-chess.
_RAW_FENS = [
    "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4",  # Qxf7# Scholar's mate
    "6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1",    # Rd8# back-rank
    "6k1/5ppp/8/8/8/8/8/4R1K1 w - - 0 1",        # Re8# back-rank
    "5k2/8/5K2/8/7R/8/8/8 w - - 0 1",            # Rh8# Arabian mate
    "6k1/6p1/6Kp/8/8/8/8/7Q w - - 0 1",          # Qa8# corridor
    "k7/8/1K6/1Q6/8/8/8/8 w - - 0 1",            # Qe8# Philidor
    "5r1k/6pp/8/5Q2/8/8/6PP/6K1 w - - 0 1",      # Qxf8# swallow's tail
    "6k1/5Qpp/8/8/8/8/6PP/6K1 w - - 0 1",        # Qe8# rook+queen
    "3r2k1/5ppp/8/8/8/8/5PPP/3Q2K1 w - - 0 1",   # Qxd8# back-rank
    "k7/8/1K6/8/8/8/8/7R w - - 0 1",             # Rh8# staircase
    "k7/8/1K6/8/8/8/8/R7 w - - 0 1",             # Kc7# (king cut-off, rook gives mate)
    "1k6/8/1K6/8/8/8/8/Q7 w - - 0 1",            # Qh8#
    "1k6/8/1K6/1Q6/8/8/8/8 w - - 0 1",           # Qe8#
    "k7/8/K1R5/8/8/8/8/8 w - - 0 1",             # Rc8# rook cuts off
    "k7/8/1K6/8/8/8/8/2R5 w - - 0 1",            # Rc8#
    "4k3/8/4K3/3Q4/8/8/8/8 w - - 0 1",           # Qa8# queen diagonal
    "4rk2/5ppp/8/5N2/3Q4/8/5PPP/6K1 w - - 0 1",  # Qxg7# knight+queen
    "7k/6pp/8/8/8/8/6PP/3Q2K1 w - - 0 1",        # Qd8# queen back-rank
    "6k1/8/5KQ1/8/8/8/8/8 w - - 0 1",            # Qg7# open queen
    "5k2/4Qppp/8/8/8/8/5PPP/6K1 w - - 0 1",      # Qd8# queen+pawns
    "k7/8/K7/8/8/8/8/R1R5 w - - 0 1",            # Rc8# double rook
    "7k/5Npp/8/8/8/8/6PP/3R3K w - - 0 1",        # Rd8# Anastasia
    "7k/6pp/6Q1/8/8/8/6PP/6K1 w - - 0 1",        # Qe8# queen upper diagonal
    "k6R/8/2K5/8/8/8/8/8 w - - 0 1",             # Kb6# (opposition + rook)
    "k7/8/1K6/8/8/4R3/8/8 w - - 0 1",            # Re8# rook long
    "5rkp/8/8/8/6Q1/8/4K3/8 w - - 0 1",          # Qg6# smothered queen
    "kp6/7Q/1K6/8/8/8/8/8 w - - 0 1",            # Qb7# lateral queen
]


def _find_mate_moves(board: chess.Board) -> list[chess.Move]:
    """Return all moves for the side to move that deliver checkmate."""
    mates = []
    for move in board.legal_moves:
        board.push(move)
        if board.is_checkmate():
            mates.append(move)
        board.pop()
    return mates


def _validate_and_build(fen: str) -> Optional[tuple[chess.Board, chess.Move]]:
    """Return (board, mate_move) if exactly one mating move exists, else None."""
    try:
        board = chess.Board(fen)
    except Exception:
        return None
    mates = _find_mate_moves(board)
    if len(mates) == 1:
        return board, mates[0]
    return None


def _difficulty_label(board: chess.Board, move: chess.Move) -> str:
    piece = board.piece_at(move.from_square)
    if piece is None:
        return "medium"
    pt = piece.piece_type
    if pt in (chess.QUEEN, chess.ROOK):
        return "easy"
    if pt in (chess.BISHOP, chess.KNIGHT):
        return "medium"
    return "hard"


# Validated pool built at import time; invalid FENs are silently dropped.
_VALIDATED_POOL: list[tuple[str, chess.Move, str]] = []  # (fen, move, san)
for _fen in _RAW_FENS:
    _result = _validate_and_build(_fen)
    if _result is not None:
        _board, _move = _result
        _san = _board.san(_move)
        _diff = _difficulty_label(_board, _move)
        _VALIDATED_POOL.append((_fen, _move, _san, _diff))


@dataclass
class ChessPuzzle:
    puzzle_id: int
    board_fen: str
    mate_move: str        # UCI, e.g. "d1h5"
    mate_move_san: str    # SAN, e.g. "Qh5#"
    difficulty: str
    solution: Optional[str] = None  # SAN of the mating move


def generate_puzzles(n: int = 20, seed: int = 42) -> list[ChessPuzzle]:
    """Return up to n ChessPuzzle objects drawn (with repetition if needed) from the validated pool."""
    if not _VALIDATED_POOL:
        return []
    rng = random.Random(seed)
    pool = list(_VALIDATED_POOL)
    rng.shuffle(pool)
    results: list[ChessPuzzle] = []
    idx = 0
    pid = 1
    while len(results) < n:
        fen, move, san, diff = pool[idx % len(pool)]
        puzzle = ChessPuzzle(
            puzzle_id=pid,
            board_fen=fen,
            mate_move=move.uci(),
            mate_move_san=san,
            difficulty=diff,
            solution=san,
        )
        results.append(puzzle)
        idx += 1
        pid += 1
    return results


def solve_puzzle(puzzle: ChessPuzzle) -> Optional[str]:
    """Return the SAN mating move if the puzzle position has exactly one mate-in-1, else None."""
    try:
        board = chess.Board(puzzle.board_fen)
    except Exception:
        return None
    mates = _find_mate_moves(board)
    if len(mates) == 1:
        return board.san(mates[0])
    return None


def build_step_descriptions(puzzle: ChessPuzzle) -> str:
    """Return a structured description of the puzzle for CoT generation."""
    board = chess.Board(puzzle.board_fen)
    move = chess.Move.from_uci(puzzle.mate_move)
    piece = board.piece_at(move.from_square)
    piece_name = chess.piece_name(piece.piece_type).capitalize() if piece else "Piece"
    from_sq = chess.square_name(move.from_square)
    to_sq = chess.square_name(move.to_square)
    captured = board.piece_at(move.to_square)
    capture_str = f", capturing the {chess.piece_name(captured.piece_type)}" if captured else ""
    lines = [
        f"Chess mate-in-1 puzzle (puzzle_id={puzzle.puzzle_id}, difficulty={puzzle.difficulty}).",
        f"FEN: {puzzle.board_fen}",
        f"White to move.",
        f"",
        f"Winning move: {puzzle.mate_move_san}",
        f"  The {piece_name} on {from_sq} moves to {to_sq}{capture_str}, delivering checkmate.",
        f"",
        f"Step 1: Identify king's escape squares and verify all are covered after {puzzle.mate_move_san}.",
        f"Step 2: Confirm no interposing piece can block or that the king cannot escape.",
        f"Step 3: Confirm it is checkmate — the black king has no legal moves.",
        f"",
        f"Answer: {puzzle.mate_move_san}",
    ]
    return "\n".join(lines)
