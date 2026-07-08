"""
Sokoban renderer: converts a SokobanPuzzle (and optionally a state) to a PIL Image.

Supports multiple visual themes to enable Stage-2 GPT-Image-2 diversification prompts.
"""

from PIL import Image, ImageDraw, ImageFont
from typing import Optional
import os

from track_b.generators.sokoban_gen import SokobanPuzzle, SokobanState, MOVE_DIRS

CELL_SIZE = 48  # pixels per cell

# Visual themes: (floor, wall, box, box_on_target, target, player, player_on_target, border)
THEMES = {
    "classic": {
        "floor":           "#F5DEB3",   # wheat
        "wall":            "#555555",   # dark grey
        "box":             "#CD853F",   # peru brown
        "box_on_target":   "#8B0000",   # dark red
        "target":          "#FFD700",   # gold dot
        "player":          "#4169E1",   # royal blue
        "player_on_target":"#00CED1",   # dark turquoise
        "border":          "#333333",
        "grid_line":       "#CCCCCC",
    },
    "dark": {
        "floor":           "#2C2C2C",
        "wall":            "#111111",
        "box":             "#8B6914",
        "box_on_target":   "#FF4500",
        "target":          "#00FF7F",
        "player":          "#00BFFF",
        "player_on_target":"#FF69B4",
        "border":          "#000000",
        "grid_line":       "#444444",
    },
    "pastel": {
        "floor":           "#FFFAF0",
        "wall":            "#8B7355",
        "box":             "#DEB887",
        "box_on_target":   "#FA8072",
        "target":          "#98FB98",
        "player":          "#87CEEB",
        "player_on_target":"#DDA0DD",
        "border":          "#7B68EE",
        "grid_line":       "#E0E0E0",
    },
}


def _draw_cell(draw: ImageDraw.Draw, x: int, y: int, ch: str, theme: dict, cs: int):
    """Draw one Sokoban cell."""
    margin = 3
    # Background
    if ch == '#':
        draw.rectangle([x, y, x + cs - 1, y + cs - 1], fill=theme["wall"])
    else:
        draw.rectangle([x, y, x + cs - 1, y + cs - 1], fill=theme["floor"])

    if ch in ('.', '+'):    # target dot
        r = cs // 6
        cx, cy = x + cs // 2, y + cs // 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=theme["target"])

    if ch in ('$', '*'):    # box
        color = theme["box_on_target"] if ch == '*' else theme["box"]
        draw.rectangle(
            [x + margin, y + margin, x + cs - margin - 1, y + cs - margin - 1],
            fill=color, outline=theme["border"], width=2,
        )

    if ch in ('@', '+'):    # player
        color = theme["player_on_target"] if ch == '+' else theme["player"]
        r = cs // 2 - margin - 2
        cx, cy = x + cs // 2, y + cs // 2
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=theme["border"], width=2)


def render_puzzle(
    puzzle: SokobanPuzzle,
    state: Optional[SokobanState] = None,
    theme_name: str = "classic",
    cell_size: int = CELL_SIZE,
    padding: int = 8,
) -> Image.Image:
    """
    Render a Sokoban puzzle to a PIL Image.

    Args:
        puzzle: the SokobanPuzzle object
        state: if given, render this state; else render initial state
        theme_name: one of 'classic', 'dark', 'pastel'
        cell_size: pixel size of each cell
        padding: outer padding in pixels
    """
    theme = THEMES.get(theme_name, THEMES["classic"])
    cs = cell_size
    if state is None:
        state = puzzle.init_state

    # Build display grid from state
    rows = len(puzzle.grid)
    cols = max(len(r) for r in puzzle.grid)
    grid = [[' '] * cols for _ in range(rows)]

    for r, row in enumerate(puzzle.grid):
        for c, ch in enumerate(row):
            if ch == '#':
                grid[r][c] = '#'
            elif (r, c) in puzzle.targets:
                grid[r][c] = '.'
            else:
                grid[r][c] = ' '

    for (r, c) in state.boxes:
        if grid[r][c] == '.':
            grid[r][c] = '*'
        else:
            grid[r][c] = '$'

    pr, pc = state.player
    if grid[pr][pc] == '.':
        grid[pr][pc] = '+'
    else:
        grid[pr][pc] = '@'

    img_w = cols * cs + 2 * padding
    img_h = rows * cs + 2 * padding
    img = Image.new("RGB", (img_w, img_h), color=theme["floor"])
    draw = ImageDraw.Draw(img)

    # Draw cells
    for r in range(rows):
        for c in range(cols):
            x = padding + c * cs
            y = padding + r * cs
            _draw_cell(draw, x, y, grid[r][c], theme, cs)

    return img


def render_all_themes(
    puzzle: SokobanPuzzle,
    state: Optional[SokobanState] = None,
) -> dict:
    """Return a dict of theme_name -> PIL Image for all themes."""
    return {
        name: render_puzzle(puzzle, state=state, theme_name=name)
        for name in THEMES
    }


def get_gpt_image2_style_prompt(theme_name: str) -> str:
    """Return a GPT-Image-2 edit prompt for style diversification."""
    prompts = {
        "dark": (
            "Transform this Sokoban puzzle image into a dark-themed version. "
            "Use dark background colors with bright neon accents. "
            "Keep all game elements (walls, boxes, targets, player) clearly visible "
            "and in their exact same positions. Do not change the puzzle layout."
        ),
        "pastel": (
            "Convert this Sokoban puzzle image to a soft pastel color scheme. "
            "Use gentle, light colors for all elements. "
            "Maintain the exact same grid layout and all game element positions."
        ),
        "retro": (
            "Render this Sokoban puzzle in a retro pixel-art style with a limited "
            "color palette. Keep all game elements (walls, boxes, targets, player) "
            "in their exact positions. Make it look like a classic 8-bit game."
        ),
        "modern": (
            "Redesign this Sokoban puzzle with a modern, flat UI design aesthetic. "
            "Use clean geometric shapes and contemporary colors. "
            "Preserve the exact grid layout and all element positions."
        ),
    }
    return prompts.get(theme_name, prompts["modern"])
