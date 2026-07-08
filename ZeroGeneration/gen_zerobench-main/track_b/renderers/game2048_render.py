"""
2048 puzzle renderer: converts a Game2048Puzzle to a PIL Image.

Supports three visual themes: 'classic', 'dark', 'pastel'.
"""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont
from typing import Optional

from track_b.generators.game2048_gen import Game2048Puzzle

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

CELL_SIZE = 96        # pixels per tile
GAP = 8               # gap between tiles
PADDING = 16          # outer padding
SCORE_BAR_H = 52      # height of score bar at top
CORNER_R = 6          # corner radius for tiles

# ---------------------------------------------------------------------------
# Tile colours (classic 2048 palette)
# ---------------------------------------------------------------------------

TILE_COLORS: dict[int, tuple[str, str]] = {
    # value: (bg_hex, fg_hex)
    0:    ("#cdc1b4", "#cdc1b4"),   # empty
    2:    ("#eee4da", "#776e65"),
    4:    ("#ede0c8", "#776e65"),
    8:    ("#f2b179", "#f9f6f2"),
    16:   ("#f59563", "#f9f6f2"),
    32:   ("#f67c5f", "#f9f6f2"),
    64:   ("#f65e3b", "#f9f6f2"),
    128:  ("#edcf72", "#f9f6f2"),
    256:  ("#edcc61", "#f9f6f2"),
    512:  ("#edc850", "#f9f6f2"),
    1024: ("#edc53f", "#f9f6f2"),
    2048: ("#edc22e", "#f9f6f2"),
}

_SUPER_TILE_BG = "#3c3a32"
_SUPER_TILE_FG = "#f9f6f2"

# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------

THEMES: dict[str, dict] = {
    "classic": {
        "bg":         "#faf8ef",   # page background
        "grid_bg":    "#bbada0",   # grid background / gap colour
        "score_bg":   "#bbada0",
        "score_fg":   "#f9f6f2",
        "title_fg":   "#776e65",
        "tile_colors": TILE_COLORS,
        "empty_bg":   "#cdc1b4",
    },
    "dark": {
        "bg":         "#1a1a2e",
        "grid_bg":    "#16213e",
        "score_bg":   "#0f3460",
        "score_fg":   "#e0e0e0",
        "title_fg":   "#e0e0e0",
        "tile_colors": {
            0:    ("#2c2c3e", "#2c2c3e"),
            2:    ("#4a4e69", "#c9c9ff"),
            4:    ("#5a3e6e", "#e0d7ff"),
            8:    ("#7b2d8b", "#f0e6ff"),
            16:   ("#9b1b30", "#ffe0e6"),
            32:   ("#b5294f", "#ffe0e6"),
            64:   ("#cf3669", "#ffe0e6"),
            128:  ("#1b6ca8", "#e0f4ff"),
            256:  ("#1a7abf", "#e0f4ff"),
            512:  ("#1987d1", "#e0f4ff"),
            1024: ("#1894e3", "#e0f4ff"),
            2048: ("#17a1f5", "#e0f4ff"),
        },
        "empty_bg":   "#2c2c3e",
    },
    "pastel": {
        "bg":         "#fef9ff",
        "grid_bg":    "#d5c8e0",
        "score_bg":   "#c3a8d1",
        "score_fg":   "#5a4170",
        "title_fg":   "#5a4170",
        "tile_colors": {
            0:    ("#ede5f5", "#ede5f5"),
            2:    ("#ffd6e7", "#7d3c5a"),
            4:    ("#ffc8dd", "#7d3c5a"),
            8:    ("#ffb3c6", "#7d3c5a"),
            16:   ("#ffa3b5", "#5a2040"),
            32:   ("#ff8fa3", "#5a2040"),
            64:   ("#ff7096", "#5a2040"),
            128:  ("#bde0fe", "#1a4a70"),
            256:  ("#a2d4fd", "#1a4a70"),
            512:  ("#90c8fc", "#1a4a70"),
            1024: ("#7cbcfb", "#1a4a70"),
            2048: ("#63aff9", "#1a4a70"),
        },
        "empty_bg":   "#ede5f5",
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
# Drawing helpers
# ---------------------------------------------------------------------------

def _draw_rounded_rect(
    draw: ImageDraw.Draw,
    xy: tuple[int, int, int, int],
    radius: int,
    fill: str,
) -> None:
    """Draw a rounded rectangle (falls back to plain rectangle on older Pillow)."""
    try:
        draw.rounded_rectangle(xy, radius=radius, fill=fill)
    except AttributeError:
        draw.rectangle(xy, fill=fill)


def _tile_colors(theme: dict, value: int) -> tuple[str, str]:
    tc = theme["tile_colors"]
    if value in tc:
        return tc[value]
    # Very large tiles
    return (_SUPER_TILE_BG, _SUPER_TILE_FG)


def _font_size_for_value(value: int, cs: int) -> int:
    """Scale font to fit larger numbers."""
    s = str(value)
    if len(s) <= 1:
        return int(cs * 0.45)
    if len(s) == 2:
        return int(cs * 0.42)
    if len(s) == 3:
        return int(cs * 0.36)
    return int(cs * 0.28)


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------

def render_puzzle(
    puzzle: Game2048Puzzle,
    theme_name: str = "classic",
    cell_size: int = CELL_SIZE,
) -> Image.Image:
    """Render a Game2048Puzzle to a PIL Image.

    Args:
        puzzle:     Game2048Puzzle instance
        theme_name: one of 'classic', 'dark', 'pastel'
        cell_size:  pixel size of each tile
    """
    theme = THEMES.get(theme_name, THEMES["classic"])
    cs = cell_size
    gap = max(4, cs // 12)
    pad = max(8, cs // 6)

    grid_px = 4 * cs + 5 * gap   # 4 tiles + 5 gaps (left, between×3, right)
    img_w = grid_px + 2 * pad
    score_h = SCORE_BAR_H
    img_h = score_h + grid_px + 2 * pad

    img = Image.new("RGB", (img_w, img_h), color=theme["bg"])
    draw = ImageDraw.Draw(img)

    # --- Score bar ---
    score_font = _get_font(max(16, cs // 5))
    title_font = _get_font(max(22, int(cs * 0.30)))

    draw.rectangle([0, 0, img_w, score_h], fill=theme["score_bg"])

    title_text = "2048"
    try:
        tb = title_font.getbbox(title_text)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    except Exception:
        tw, th = 60, 24
    draw.text(
        (pad, (score_h - th) // 2),
        title_text,
        fill=theme["score_fg"],
        font=title_font,
    )

    score_text = f"Score: {puzzle.score_gain}"
    try:
        sb = score_font.getbbox(score_text)
        sw = sb[2] - sb[0]
        sh = sb[3] - sb[1]
    except Exception:
        sw, sh = 100, 18
    draw.text(
        (img_w - pad - sw, (score_h - sh) // 2),
        score_text,
        fill=theme["score_fg"],
        font=score_font,
    )

    # --- Grid background ---
    grid_x0 = pad
    grid_y0 = score_h + pad
    _draw_rounded_rect(
        draw,
        (grid_x0, grid_y0, grid_x0 + grid_px, grid_y0 + grid_px),
        radius=CORNER_R * 2,
        fill=theme["grid_bg"],
    )

    # --- Tiles ---
    for row in range(4):
        for col in range(4):
            value = puzzle.board[row][col]
            x = grid_x0 + gap + col * (cs + gap)
            y = grid_y0 + gap + row * (cs + gap)

            bg, fg = _tile_colors(theme, value)
            _draw_rounded_rect(
                draw,
                (x, y, x + cs, y + cs),
                radius=CORNER_R,
                fill=bg,
            )

            if value != 0:
                fs = _font_size_for_value(value, cs)
                font = _get_font(fs)
                text = str(value)
                try:
                    bbox = font.getbbox(text)
                    tw2 = bbox[2] - bbox[0]
                    th2 = bbox[3] - bbox[1]
                    tx = x + (cs - tw2) // 2 - bbox[0]
                    ty = y + (cs - th2) // 2 - bbox[1]
                except Exception:
                    tx = x + cs // 4
                    ty = y + cs // 3
                draw.text((tx, ty), text, fill=fg, font=font)

    return img


def render_all_themes(puzzle: Game2048Puzzle) -> dict[str, Image.Image]:
    """Return {theme_name: PIL Image} for all themes."""
    return {name: render_puzzle(puzzle, theme_name=name) for name in THEMES}


def get_gpt_image2_style_prompt(theme_name: str) -> str:
    """Return a GPT-Image-2 style-transfer prompt for the given theme."""
    prompts: dict[str, str] = {
        "classic": (
            "Render this 2048 puzzle board in the classic beige-and-brown style, "
            "with warm tile colours (tan for low values, orange and red for mid values, "
            "gold for high values). Keep every tile number and position identical."
        ),
        "dark": (
            "Convert this 2048 puzzle to a dark, sleek theme with deep navy and purple "
            "backgrounds and bright neon tile colours. Keep every tile number and "
            "position exactly the same."
        ),
        "pastel": (
            "Redesign this 2048 puzzle board with soft pastel colours — pale pinks for "
            "low tiles and light blues for high tiles. Maintain all tile numbers and "
            "positions unchanged."
        ),
    }
    return prompts.get(theme_name, prompts["classic"])
