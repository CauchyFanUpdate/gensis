from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from track_b.generators.othello_gen import OthelloPuzzle, _get_valid_moves, SIZE

CELL_SIZE = 60
PADDING = 16
HEADER_HEIGHT = 32    # for row/col coordinate labels
LABEL_WIDTH = 28      # left margin for row labels

THEMES = {
    "classic": {
        "board":        "#1A7A3C",
        "bg":           "#F5F5F5",
        "grid_line":    "#145E2E",
        "player1":      "#111111",   # Black piece
        "player2":      "#F8F8F8",   # White piece
        "piece_border": "#333333",
        "valid_dot":    "#90EE90",
        "best_ring":    "#FFD700",
        "label":        "#222222",
        "corner_dot":   "#145E2E",
    },
    "dark": {
        "board":        "#0D2E1A",
        "bg":           "#1A1A2E",
        "grid_line":    "#0A5C2F",
        "player1":      "#0A0A0A",
        "player2":      "#E8E8E8",
        "piece_border": "#00FF7F",
        "valid_dot":    "#00FF7F",
        "best_ring":    "#FF4500",
        "label":        "#CCCCCC",
        "corner_dot":   "#0A5C2F",
    },
    "wood": {
        "board":        "#8B6914",
        "bg":           "#F5DEB3",
        "grid_line":    "#5C4708",
        "player1":      "#1A1A1A",
        "player2":      "#FFFAF0",
        "piece_border": "#3B2507",
        "valid_dot":    "#D2B48C",
        "best_ring":    "#FF6347",
        "label":        "#3B2507",
        "corner_dot":   "#5C4708",
    },
}


def _get_font(size: int):
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def render_puzzle(
    puzzle: OthelloPuzzle,
    theme_name: str = "classic",
    show_valid: bool = True,
) -> Image.Image:
    theme = THEMES.get(theme_name, THEMES["classic"])
    cs = CELL_SIZE
    pad = PADDING

    board_pixel_w = SIZE * cs
    board_pixel_h = SIZE * cs

    img_w = LABEL_WIDTH + board_pixel_w + 2 * pad
    img_h = HEADER_HEIGHT + board_pixel_h + 2 * pad

    img = Image.new("RGB", (img_w, img_h), color=theme["bg"])
    draw = ImageDraw.Draw(img)

    label_font = _get_font(int(min(HEADER_HEIGHT, LABEL_WIDTH) * 0.55))

    board_x = pad + LABEL_WIDTH
    board_y = pad + HEADER_HEIGHT

    # Board background
    draw.rectangle(
        [board_x, board_y, board_x + board_pixel_w, board_y + board_pixel_h],
        fill=theme["board"],
    )

    # Column labels (A-H)
    for c in range(SIZE):
        cx = board_x + c * cs + cs // 2
        cy = pad + HEADER_HEIGHT // 2
        label = chr(ord("A") + c)
        try:
            bbox = label_font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 10, 14
        draw.text((cx - tw // 2, cy - th // 2), label, fill=theme["label"], font=label_font)

    # Row labels (1-8)
    for r in range(SIZE):
        cy = board_y + r * cs + cs // 2
        cx = pad + LABEL_WIDTH // 2
        label = str(r + 1)
        try:
            bbox = label_font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 10, 14
        draw.text((cx - tw // 2, cy - th // 2), label, fill=theme["label"], font=label_font)

    # Grid lines
    for i in range(SIZE + 1):
        # Vertical
        vx = board_x + i * cs
        draw.line([(vx, board_y), (vx, board_y + board_pixel_h)], fill=theme["grid_line"], width=1)
        # Horizontal
        hy = board_y + i * cs
        draw.line([(board_x, hy), (board_x + board_pixel_w, hy)], fill=theme["grid_line"], width=1)

    # Star points (standard Othello board corners of inner 4×4)
    star_positions = [(2, 2), (2, 6), (6, 2), (6, 6)]
    for (sr, sc) in star_positions:
        sx = board_x + sc * cs
        sy = board_y + sr * cs
        r_dot = 4
        draw.ellipse(
            [sx - r_dot, sy - r_dot, sx + r_dot, sy + r_dot],
            fill=theme["corner_dot"],
        )

    # Valid move positions
    valid_moves = set(_get_valid_moves(puzzle.board, puzzle.current_player)) if show_valid else set()

    # Piece margin and radii
    margin = int(cs * 0.08)
    piece_r = cs // 2 - margin
    valid_dot_r = int(cs * 0.13)
    best_ring_width = 3

    for r in range(SIZE):
        for c in range(SIZE):
            cx = board_x + c * cs + cs // 2
            cy = board_y + r * cs + cs // 2
            val = puzzle.board[r][c]

            if val != 0:
                fill = theme["player1"] if val == 1 else theme["player2"]
                draw.ellipse(
                    [cx - piece_r, cy - piece_r, cx + piece_r, cy + piece_r],
                    fill=fill,
                    outline=theme["piece_border"],
                    width=2,
                )
            elif show_valid and (r, c) in valid_moves:
                draw.ellipse(
                    [cx - valid_dot_r, cy - valid_dot_r, cx + valid_dot_r, cy + valid_dot_r],
                    fill=theme["valid_dot"],
                )

    # Highlight best move cell with a ring
    br, bc = puzzle.best_move
    best_cx = board_x + bc * cs + cs // 2
    best_cy = board_y + br * cs + cs // 2
    ring_r = piece_r - best_ring_width - 1
    if puzzle.board[br][bc] == 0:
        # Empty cell — draw a slightly larger highlighted dot
        draw.ellipse(
            [best_cx - ring_r - 4, best_cy - ring_r - 4,
             best_cx + ring_r + 4, best_cy + ring_r + 4],
            outline=theme["best_ring"],
            width=best_ring_width,
        )
    else:
        # Piece already there (shouldn't happen in a valid puzzle, but guard)
        draw.ellipse(
            [best_cx - piece_r, best_cy - piece_r,
             best_cx + piece_r, best_cy + piece_r],
            outline=theme["best_ring"],
            width=best_ring_width,
        )

    # Outer border
    draw.rectangle(
        [board_x, board_y, board_x + board_pixel_w, board_y + board_pixel_h],
        outline=theme["grid_line"],
        width=2,
    )

    return img


def get_gpt_image2_style_prompt(theme_name: str) -> str:
    prompts = {
        "classic": (
            "Render this Othello/Reversi board in a classic style with a vivid green board, "
            "crisp black and white circular pieces, and a clean coordinate grid. "
            "Preserve all piece positions exactly."
        ),
        "dark": (
            "Transform this Othello board into a dark neon style. Use a near-black green board "
            "with glowing neon-green grid lines. Black pieces should be almost invisible on the "
            "dark board; white pieces should glow. Keep all piece positions unchanged."
        ),
        "wood": (
            "Redesign this Othello board with a wood-grain aesthetic. Use a warm brown wooden "
            "board texture, dark and ivory circular pieces. Add subtle grain lines to the board "
            "surface. All piece positions must remain identical."
        ),
        "minimal": (
            "Simplify this Othello board to a clean, minimalist design. Use a light grey board, "
            "thin grid lines, and matte black and white circular pieces. "
            "All piece positions must stay exactly the same."
        ),
    }
    return prompts.get(theme_name, prompts["classic"])
