from __future__ import annotations

from typing import Optional

import chess
from PIL import Image, ImageDraw, ImageFont

from track_b.generators.chess_gen import ChessPuzzle

CELL_SIZE = 64
LABEL_MARGIN = 20  # pixels reserved for rank/file labels on each side

THEMES = {
    "classic": {
        "light_sq":   "#F0D9B5",
        "dark_sq":    "#B58863",
        "highlight":  "#F6F669",
        "label_bg":   "#8B6344",
        "label_fg":   "#F0D9B5",
        "border":     "#5A3A1A",
        "white_piece": "#FFFFFF",
        "black_piece": "#000000",
        "piece_outline": "#333333",
    },
    "dark": {
        "light_sq":   "#4D4D4D",
        "dark_sq":    "#1A1A1A",
        "highlight":  "#AAAA33",
        "label_bg":   "#0D0D0D",
        "label_fg":   "#AAAAAA",
        "border":     "#000000",
        "white_piece": "#E0E0E0",
        "black_piece": "#111111",
        "piece_outline": "#888888",
    },
    "green": {
        "light_sq":   "#FFFFDD",
        "dark_sq":    "#86A666",
        "highlight":  "#F6F669",
        "label_bg":   "#5A7A3A",
        "label_fg":   "#FFFFDD",
        "border":     "#3A5A1A",
        "white_piece": "#FFFFFF",
        "black_piece": "#1A1A1A",
        "piece_outline": "#444444",
    },
}

# Letter labels for pieces (ASCII — works with any font)
_PIECE_LABELS = {
    chess.KING:   "K",
    chess.QUEEN:  "Q",
    chess.ROOK:   "R",
    chess.BISHOP: "B",
    chess.KNIGHT: "N",
    chess.PAWN:   "P",
}

_FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_PATHS:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _sq_to_col_row(square: int, flip: bool = False) -> tuple[int, int]:
    """Return (col, row) in board display coords (row 0 = rank 8 at top)."""
    file = chess.square_file(square)
    rank = chess.square_rank(square)
    col = file
    row = 7 - rank  # rank 8 → row 0, rank 1 → row 7
    return col, row


def render_puzzle(
    puzzle: ChessPuzzle,
    theme_name: str = "classic",
    highlight_move: Optional[str] = None,
) -> Image.Image:
    """Render puzzle board to PIL Image. highlight_move is a UCI string like 'e2e4'."""
    theme = THEMES.get(theme_name, THEMES["classic"])
    cs = CELL_SIZE
    margin = LABEL_MARGIN
    board_px = 8 * cs
    img_w = board_px + 2 * margin
    img_h = board_px + 2 * margin

    img = Image.new("RGB", (img_w, img_h), color=theme["label_bg"])
    draw = ImageDraw.Draw(img)

    piece_font = _load_font(int(cs * 0.72))
    label_font = _load_font(int(margin * 0.7))

    # Highlighted squares from UCI move
    highlight_sqs: set[int] = set()
    if highlight_move and len(highlight_move) >= 4:
        try:
            mv = chess.Move.from_uci(highlight_move)
            highlight_sqs = {mv.from_square, mv.to_square}
        except Exception:
            pass

    board = chess.Board(puzzle.board_fen)

    # Draw squares and pieces
    for rank in range(8):
        for file in range(8):
            square = chess.square(file, rank)
            col = file
            row = 7 - rank  # rank 8 at top

            x = margin + col * cs
            y = margin + row * cs

            is_light = (file + rank) % 2 == 1
            sq_color = theme["light_sq"] if is_light else theme["dark_sq"]
            if square in highlight_sqs:
                sq_color = theme["highlight"]

            draw.rectangle([x, y, x + cs - 1, y + cs - 1], fill=sq_color)

            piece = board.piece_at(square)
            if piece is not None:
                label = _PIECE_LABELS[piece.piece_type]
                is_white = piece.color == chess.WHITE

                # Draw circle background for the piece
                pad = int(cs * 0.1)
                circle_fill = theme["white_piece"] if is_white else theme["black_piece"]
                circle_outline = theme["piece_outline"]
                draw.ellipse([x + pad, y + pad, x + cs - pad - 1, y + cs - pad - 1],
                             fill=circle_fill, outline=circle_outline, width=2)

                # Draw letter centered in the circle
                text_fill = theme["black_piece"] if is_white else theme["white_piece"]
                try:
                    bbox = piece_font.getbbox(label)
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                    tx = x + (cs - tw) // 2 - bbox[0]
                    ty = y + (cs - th) // 2 - bbox[1]
                except Exception:
                    tx, ty = x + cs // 4, y + cs // 4
                draw.text((tx, ty), label, fill=text_fill, font=piece_font)

    # File labels (a–h) at top and bottom
    files = "abcdefgh"
    for file in range(8):
        lbl = files[file]
        col = file
        cx = margin + col * cs + cs // 2
        for label_y in (margin // 2, img_h - margin // 2):
            try:
                bbox = label_font.getbbox(lbl)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
            except Exception:
                lw, lh = margin // 2, margin // 2
            draw.text((cx - lw // 2, label_y - lh // 2), lbl,
                      fill=theme["label_fg"], font=label_font)

    # Rank labels (1–8) at left and right
    for rank in range(8):
        lbl = str(rank + 1)
        row = 7 - rank
        cy = margin + row * cs + cs // 2
        for label_x in (margin // 2, img_w - margin // 2):
            try:
                bbox = label_font.getbbox(lbl)
                lw = bbox[2] - bbox[0]
                lh = bbox[3] - bbox[1]
            except Exception:
                lw, lh = margin // 2, margin // 2
            draw.text((label_x - lw // 2, cy - lh // 2), lbl,
                      fill=theme["label_fg"], font=label_font)

    # Outer border around the board grid
    draw.rectangle(
        [margin, margin, margin + board_px - 1, margin + board_px - 1],
        outline=theme["border"], width=2,
    )

    return img


def get_gpt_image2_style_prompt(theme_name: str) -> str:
    prompts = {
        "classic": (
            "Render this chess board in a classic wooden style. "
            "Use warm brown tones for the dark squares and cream for the light squares. "
            "Keep all pieces in their exact positions and make them clearly recognizable. "
            "Do not alter the piece placement in any way."
        ),
        "dark": (
            "Convert this chess board to a sleek dark theme. "
            "Use dark gray squares with light gray accents and neon-style piece colors. "
            "All pieces must remain in their exact same positions and be clearly visible."
        ),
        "green": (
            "Redesign this chess board with a classic green-and-cream color scheme, "
            "like a traditional tournament board. "
            "Pieces must stay in their exact positions with clear, readable symbols."
        ),
        "marble": (
            "Render this chess board as if carved from marble, with white and grey marble squares. "
            "Give pieces a three-dimensional stone-carved appearance. "
            "Keep every piece exactly in the same square as in the original."
        ),
    }
    return prompts.get(theme_name, prompts["classic"])
