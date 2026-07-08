from __future__ import annotations

import math
from PIL import Image, ImageDraw, ImageFont

from track_b.generators.connect4_gen import Connect4Puzzle, _drop_piece, _check_winner, ROWS, COLS

CELL_SIZE = 60
HEADER_HEIGHT = 36   # space above board for column numbers
PADDING = 12

THEMES = {
    "classic": {
        "board":        "#1A4FBB",
        "bg":           "#F0F0F0",
        "empty":        "#DDEEFF",
        "player1":      "#E02020",   # Red
        "player2":      "#F0C020",   # Yellow
        "col_label":    "#222222",
        "border":       "#0D2E6B",
        "highlight":    "#FFFFFF",
        "star":         "#FFFFFF",
    },
    "dark": {
        "board":        "#111827",
        "bg":           "#1F2937",
        "empty":        "#0F172A",
        "player1":      "#EF4444",
        "player2":      "#FBBF24",
        "col_label":    "#E5E7EB",
        "border":       "#374151",
        "highlight":    "#F9FAFB",
        "star":         "#F9FAFB",
    },
    "pastel": {
        "board":        "#7C9CBF",
        "bg":           "#F3F4F6",
        "empty":        "#E8F0FA",
        "player1":      "#F87171",
        "player2":      "#FDE68A",
        "col_label":    "#374151",
        "border":       "#5A7FA8",
        "highlight":    "#FFFFFF",
        "star":         "#FFFFFF",
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


def _find_winning_cells(board: list[list[int]], player: int) -> set[tuple[int, int]]:
    """Return all (row, col) board positions that are part of a winning four after player drops."""
    winning_cells: set[tuple[int, int]] = set()
    for col in range(COLS):
        new_board = _drop_piece(board, col, player)
        if new_board is None:
            continue
        if _check_winner(new_board) != player:
            continue
        # Find landing row
        landing_row = -1
        for r in range(ROWS - 1, -1, -1):
            if board[r][col] == 0:
                landing_row = r
                break
        if landing_row == -1:
            continue
        # Enumerate four-in-a-row runs that include landing_row, col
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            run = []
            for k in range(-3, 4):
                r2 = landing_row + dr * k
                c2 = col + dc * k
                if 0 <= r2 < ROWS and 0 <= c2 < COLS and new_board[r2][c2] == player:
                    run.append((r2, c2))
                    if len(run) == 4:
                        winning_cells.update(run)
                        run = run[1:]
                else:
                    run = []
    return winning_cells


def render_puzzle(
    puzzle: Connect4Puzzle,
    theme_name: str = "classic",
    show_winning: bool = False,
) -> Image.Image:
    theme = THEMES.get(theme_name, THEMES["classic"])
    cs = CELL_SIZE
    pad = PADDING

    img_w = COLS * cs + 2 * pad
    img_h = ROWS * cs + HEADER_HEIGHT + 2 * pad

    img = Image.new("RGB", (img_w, img_h), color=theme["bg"])
    draw = ImageDraw.Draw(img)

    label_font = _get_font(int(cs * 0.40))
    col_font = _get_font(int(HEADER_HEIGHT * 0.55))

    board_x = pad
    board_y = pad + HEADER_HEIGHT

    # Board background
    draw.rectangle(
        [board_x, board_y, board_x + COLS * cs, board_y + ROWS * cs],
        fill=theme["board"],
    )

    # Winning cells for highlight
    winning_cells: set[tuple[int, int]] = set()
    if show_winning:
        winning_cells = _find_winning_cells(puzzle.board, puzzle.current_player)

    # Draw column numbers above board
    for col in range(COLS):
        cx = board_x + col * cs + cs // 2
        cy = pad + HEADER_HEIGHT // 2
        label = str(col + 1)
        try:
            bbox = col_font.getbbox(label)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = 12, 16
        draw.text((cx - tw // 2, cy - th // 2), label, fill=theme["col_label"], font=col_font)

    # Draw cells (circles)
    margin = int(cs * 0.10)
    for r in range(ROWS):
        for c in range(COLS):
            x = board_x + c * cs
            y = board_y + r * cs
            val = puzzle.board[r][c]

            if val == 0:
                fill = theme["empty"]
            elif val == 1:
                fill = theme["player1"]
            else:
                fill = theme["player2"]

            draw.ellipse(
                [x + margin, y + margin, x + cs - margin, y + cs - margin],
                fill=fill,
                outline=theme["border"],
                width=2,
            )

            if show_winning and (r, c) in winning_cells and val != 0:
                # Draw a small star-dot highlight in the centre
                cr = int(cs * 0.12)
                cx2 = x + cs // 2
                cy2 = y + cs // 2
                draw.ellipse(
                    [cx2 - cr, cy2 - cr, cx2 + cr, cy2 + cr],
                    fill=theme["star"],
                )

    # Outer board border
    draw.rectangle(
        [board_x, board_y, board_x + COLS * cs, board_y + ROWS * cs],
        outline=theme["border"],
        width=3,
    )

    # Vertical separators
    for c in range(1, COLS):
        lx = board_x + c * cs
        draw.line([(lx, board_y), (lx, board_y + ROWS * cs)], fill=theme["border"], width=1)

    # Horizontal separators
    for r in range(1, ROWS):
        ry = board_y + r * cs
        draw.line([(board_x, ry), (board_x + COLS * cs, ry)], fill=theme["border"], width=1)

    return img


def get_gpt_image2_style_prompt(theme_name: str) -> str:
    prompts = {
        "dark": (
            "Transform this Connect 4 board image into a dark, neon-lit style. "
            "The board background should be very dark (near-black), with glowing red and yellow "
            "circular pieces. Column numbers remain clearly visible. Do not alter piece positions."
        ),
        "pastel": (
            "Redraw this Connect 4 board with soft pastel colors. Use a light-blue board, "
            "pale-red and pale-yellow circles for pieces, and gentle shadows. "
            "Keep all pieces in their exact positions and all column numbers visible."
        ),
        "retro": (
            "Redesign this Connect 4 board in a retro pixel-art style. Use a limited color "
            "palette reminiscent of 8-bit games. Keep the board structure (6 rows × 7 columns), "
            "piece colors (red and yellow), and column labels unchanged."
        ),
        "classic": (
            "Render this Connect 4 board in a clean, classic style with a bright blue board, "
            "vivid red and yellow circular pieces, and bold column numbers above the board."
        ),
    }
    return prompts.get(theme_name, prompts["classic"])
