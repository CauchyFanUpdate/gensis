"""
Classify ZeroBench main questions (100 total) into pipeline tracks.

Tracks:
  track_b_sokoban   — Sokoban push-box puzzles
  track_b_sudoku    — Sudoku / Killer Sudoku
  track_b_chess     — Chess (FEN, best-move)
  track_b_maze      — Maze navigation
  track_b_2048      — 2048 sliding-tile game
  track_b_connect4  — Connect 4
  track_b_othello   — Othello / Reversi
  track_b_nonogram  — Nonogram / picross
  track_b_graph     — Directed-graph traversal / cycle problems
  track_a           — Real-image counting / measurement / pricing
  hopchain          — Multi-hop real-image questions (auxiliary)

Usage:
    python classify/classify_questions.py \
        --data /mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/zerobench \
        --output classify/question_classification.jsonl
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Hardcoded classification based on manual analysis of all 100 questions.
# Keys are question_id (int).  Values are track labels.
# ---------------------------------------------------------------------------
TRACK_MAP = {
    # ---- Track B: programmatic structured puzzles ----
    3:  "track_b_sokoban",    # smiley-face maze → sequence of moves
    10: "track_b_sudoku",     # Killer Sudoku
    11: "track_b_chess",      # chess position → FEN after move 45
    16: "track_b_maze",       # coloured shapes in maze, find player that won
    24: "track_b_2048",       # 2048 board → optimal move sequence
    28: "track_b_connect4",   # Connect-4 board → columns to play
    33: "track_b_pong",       # Atari Pong → sequence of paddle moves
    54: "track_b_sudoku",     # classic Sudoku
    66: "track_b_chess",      # chess board → best move for white
    88: "track_b_nonogram",   # colour nonogram → solve to reveal picture
    93: "track_b_othello",    # Othello/Reversi → best move score
    # directed-graph traversal / cycle problems
    15: "track_b_graph",
    32: "track_b_graph",
    62: "track_b_graph",
    87: "track_b_graph",      # keyboard-key graph, find path length

    # ---- HopChain: multi-hop real-image (auxiliary, low-priority) ----
    # These require chaining multiple real-world lookups / inferences.
    74: "hopchain",           # crest → top scorer → year of loss → data lookup
    97: "hopchain",           # Catan board → longest road → scoring math

    # ---- Track A: real-image counting / measurement / pricing ----
    # All remaining questions fall here.
}

# Derive Track A as "everything not in TRACK_MAP"
_TRACK_B_AND_HOP = set(TRACK_MAP.keys())

def _default_track(qid: int) -> str:
    return TRACK_MAP.get(qid, "track_a")


def classify(data_dir: str, output_path: str):
    data_dir = Path(data_dir)
    parquet_path = data_dir / "data" / "zerobench-00000-of-00001.parquet"
    df = pd.read_parquet(parquet_path)

    records = []
    track_counts: dict[str, int] = {}

    for _, row in df.sort_values("question_id").iterrows():
        qid = int(row["question_id"])
        track = _default_track(qid)
        track_counts[track] = track_counts.get(track, 0) + 1

        images = list(row["question_images"])

        rec = {
            "question_id": qid,
            "track": track,
            "question_text": row["question_text"],
            "question_answer": str(row["question_answer"]),
            "images": images,
            "n_images": len(images),
        }
        records.append(rec)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"Wrote {len(records)} records → {output_path}")
    print("\nTrack distribution:")
    for track, count in sorted(track_counts.items()):
        print(f"  {track:<22} {count:3d}")


def load_classification(path: str) -> dict[int, dict]:
    """Load classification JSONL; return dict keyed by question_id."""
    result = {}
    with open(path) as f:
        for line in f:
            rec = json.loads(line.strip())
            result[rec["question_id"]] = rec
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="/mnt/tidal-alsh01/dataset/perceptionVLM/yangfan/datasets/zerobench",
    )
    parser.add_argument(
        "--output",
        default="classify/question_classification.jsonl",
    )
    args = parser.parse_args()
    classify(args.data, args.output)


if __name__ == "__main__":
    main()
