"""
Nonogram puzzle renderer: converts a NonogramPuzzle to a PIL Image.

Layout:
  - Column clues drawn above the grid (top-aligned, small font)
  - Row clues drawn to the left of the grid (right-aligned, small font)
  - Grid cells: 28 px each
  - Optionally show the solution (filled cells) or hide it (all blank)

Supports themes: 'classic', 'dark', 'colorful'.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont
from typing import Optional

from track_b.generators.nonogram_gen import NonogramPuzzle

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

CELL_SIZE = 28          # pixels per grid cell
CLUE_FONT_SIZE = 13     # font size for clue numbers
CLUE_PAD = 4            # inner padding within clue areas
OUTER_PAD = 12          # outer image padding
LINE_W_THIN = 1         # thin grid line width
LINE_W_THICK = 2        # border / block-separator width

# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

THEMES: dict[str, dict] = {
    "classic": {
        "bg":           "#FFFFFF",
        "clue_bg":      "#F0F0F0",
        "clue_fg":      "#222222",
        "cell_empty":   "#FFFFFF",
        "cell_filled":  "#222222",
        "grid_line":    "#AAAAAA",
        "border":       "#222222",
        "header_bg":    "#E0E0E0",
    },
    "dark": {
        "bg":           "#1E1E2E",
        "clue_bg":      "#2A2A3E",
        "clue_fg":      "#CDD6F4",
        "cell_empty":   "#313244",
        "cell_filled":  "#CBA6F7",
        "grid_line":    "#45475A",
        "border":       "#CDD6F4",
        "header_bg":    "#2A2A3E",
    },
    "colorful": {
        "bg":           "#FFF8F0",
        "clue_bg":      "#FFE5CC",
        "clue_fg":      "#6B2D2D",
        "cell_empty":   "#FFFFFF",
        "cell_filled":  "#E05C2A",
        "grid_line":    "#FFBE99",
        "border":       "#6B2D2D",
        "header_bg":    "#FFD6B0",
    },
}

# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------

def _get_font(size: int) -> ImageFont.FreeTypeFont:
    _candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    ]
    for path in _candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Text-measurement helpers
# ---------------------------------------------------------------------------

def _text_size(font: ImageFont.FreeTypeFont, text: str) -> tuple[int, int]:
    """Return (width, height) of text in the given font."""
    try:
        bb = font.getbbox(text)
        return bb[2] - bb[0], bb[3] - bb[1]
    except Exception:
        return len(text) * (CLUE_FONT_SIZE // 2 + 1), CLUE_FONT_SIZE


# ---------------------------------------------------------------------------
# Clue-area size computation
# ---------------------------------------------------------------------------

def _col_clue_area_height(puzzle: NonogramPuzzle, font: ImageFont.FreeTypeFont) -> int:
    """Pixel height needed above the grid for column clues."""
    max_groups = max(len(c) for c in puzzle.col_clues)
    _, line_h = _text_size(font, "0")
    return max_groups * (line_h + CLUE_PAD) + CLUE_PAD


def _row_clue_area_width(puzzle: NonogramPuzzle, font: ImageFont.FreeTypeFont) -> int:
    """Pixel width needed to the left of the grid for row clues."""
    max_w = 0
    for clue in puzzle.row_clues:
        text = " ".join(str(v) for v in clue)
        w, _ = _text_size(font, text)
        if w > max_w:
            max_w = w
    return max_w + CLUE_PAD * 2


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_puzzle(
    puzzle: NonogramPuzzle,
    show_solution: bool = False,
    theme_name: str = "classic",
    cell_size: int = CELL_SIZE,
) -> Image.Image:
    """Render a NonogramPuzzle to a PIL Image.

    Args:
        puzzle:        NonogramPuzzle instance
        show_solution: if True, draw filled cells; if False, all cells blank
        theme_name:    one of 'classic', 'dark', 'colorful'
        cell_size:     pixel size per grid cell
    """
    theme = THEMES.get(theme_name, THEMES["classic"])
    cs = cell_size
    font = _get_font(CLUE_FONT_SIZE)
    _, line_h = _text_size(font, "0")

    col_h = _col_clue_area_height(puzzle, font)
    row_w = _row_clue_area_width(puzzle, font)

    grid_w = puzzle.cols * cs
    grid_h = puzzle.rows * cs

    img_w = OUTER_PAD + row_w + grid_w + OUTER_PAD
    img_h = OUTER_PAD + col_h + grid_h + OUTER_PAD

    img = Image.new("RGB", (img_w, img_h), color=theme["bg"])
    draw = ImageDraw.Draw(img)

    # Offsets where the grid starts
    gx0 = OUTER_PAD + row_w   # left edge of grid
    gy0 = OUTER_PAD + col_h   # top edge of grid

    # --- Clue backgrounds ---
    # Top-left corner (above row clues / left of col clues)
    draw.rectangle(
        [OUTER_PAD, OUTER_PAD, gx0, gy0],
        fill=theme["header_bg"],
    )
    # Row clue strip
    draw.rectangle(
        [OUTER_PAD, gy0, gx0, gy0 + grid_h],
        fill=theme["clue_bg"],
    )
    # Col clue strip
    draw.rectangle(
        [gx0, OUTER_PAD, gx0 + grid_w, gy0],
        fill=theme["clue_bg"],
    )

    # --- Draw column clues ---
    for c, clue in enumerate(puzzle.col_clues):
        cx = gx0 + c * cs
        # Top-align: start from top of col clue area
        max_groups = max(len(cc) for cc in puzzle.col_clues)
        # Pad with empty entries at top so numbers are bottom-aligned
        padded = [None] * (max_groups - len(clue)) + clue
        for row_idx, val in enumerate(padded):
            if val is None:
                continue
            text = str(val)
            tw, th = _text_size(font, text)
            tx = cx + (cs - tw) // 2
            ty = OUTER_PAD + CLUE_PAD + row_idx * (line_h + CLUE_PAD)
            draw.text((tx, ty), text, fill=theme["clue_fg"], font=font)

    # --- Draw row clues ---
    for r, clue in enumerate(puzzle.row_clues):
        ry = gy0 + r * cs
        text = " ".join(str(v) for v in clue)
        tw, th = _text_size(font, text)
        # Right-align within the row clue area
        tx = gx0 - tw - CLUE_PAD
        ty = ry + (cs - th) // 2
        draw.text((tx, ty), text, fill=theme["clue_fg"], font=font)

    # --- Draw grid cells ---
    for r in range(puzzle.rows):
        for c in range(puzzle.cols):
            x = gx0 + c * cs
            y = gy0 + r * cs
            if show_solution and puzzle.solution_grid[r][c] == 1:
                fill = theme["cell_filled"]
            else:
                fill = theme["cell_empty"]
            draw.rectangle([x, y, x + cs - 1, y + cs - 1], fill=fill)

    # --- Grid lines (thin) ---
    for r in range(puzzle.rows + 1):
        y = gy0 + r * cs
        draw.line(
            [(gx0, y), (gx0 + grid_w, y)],
            fill=theme["grid_line"],
            width=LINE_W_THIN,
        )
    for c in range(puzzle.cols + 1):
        x = gx0 + c * cs
        draw.line(
            [(x, gy0), (x, gy0 + grid_h)],
            fill=theme["grid_line"],
            width=LINE_W_THIN,
        )

    # --- Border around grid (thick) ---
    draw.rectangle(
        [gx0, gy0, gx0 + grid_w, gy0 + grid_h],
        outline=theme["border"],
        width=LINE_W_THICK,
    )

    return img


def render_all_themes(
    puzzle: NonogramPuzzle,
    show_solution: bool = False,
) -> dict[str, Image.Image]:
    """Return {theme_name: PIL Image} for all themes."""
    return {
        name: render_puzzle(puzzle, show_solution=show_solution, theme_name=name)
        for name in THEMES
    }


def get_gpt_image2_style_prompt(theme_name: str) -> str:
    """Return a GPT-Image-2 style-transfer prompt for the given theme."""
    prompts: dict[str, str] = {
        "classic": (
            "Render this nonogram puzzle in a clean black-and-white style. "
            "The clue numbers should be in dark text on a light grey background. "
            "Filled cells are solid black; empty cells are white. "
            "Keep all clue numbers exactly as shown."
        ),
        "dark": (
            "Convert this nonogram puzzle to a dark mode theme with a deep navy "
            "background and soft lavender filled cells. Clue numbers should be "
            "bright and clearly readable. Keep all clue values identical."
        ),
        "colorful": (
            "Redesign this nonogram puzzle with warm, colourful tones: an orange-tinted "
            "clue area and vivid terracotta filled cells on a white grid. "
            "Maintain all clue numbers exactly as shown."
        ),
    }
    return prompts.get(theme_name, prompts["classic"])
