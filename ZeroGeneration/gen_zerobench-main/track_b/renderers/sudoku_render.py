"""
Sudoku renderer: converts a SudokuPuzzle to a PIL Image.
"""

from PIL import Image, ImageDraw, ImageFont
from typing import Optional
from track_b.generators.sudoku_gen import SudokuPuzzle

CELL_SIZE = 56
FONT_SIZE = 28

THEMES = {
    "classic": {
        "bg":           "#FFFFFF",
        "given_bg":     "#E8E8E8",
        "given_fg":     "#000000",
        "empty_fg":     "#4169E1",
        "thin_line":    "#AAAAAA",
        "thick_line":   "#000000",
        "border":       "#000000",
    },
    "dark": {
        "bg":           "#1E1E1E",
        "given_bg":     "#2D2D2D",
        "given_fg":     "#FFFFFF",
        "empty_fg":     "#00BFFF",
        "thin_line":    "#444444",
        "thick_line":   "#CCCCCC",
        "border":       "#FFFFFF",
    },
    "warm": {
        "bg":           "#FFF8E7",
        "given_bg":     "#FFE4B5",
        "given_fg":     "#8B4513",
        "empty_fg":     "#DC143C",
        "thin_line":    "#DEB887",
        "thick_line":   "#8B4513",
        "border":       "#8B4513",
    },
}


def _get_font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def render_puzzle(
    puzzle: SudokuPuzzle,
    show_solution: bool = False,
    theme_name: str = "classic",
    cell_size: int = CELL_SIZE,
    padding: int = 16,
) -> Image.Image:
    """
    Render a Sudoku puzzle to a PIL Image.

    Args:
        puzzle: SudokuPuzzle object
        show_solution: if True, fill in solved cells too
        theme_name: one of 'classic', 'dark', 'warm'
        cell_size: pixel size of each cell
        padding: outer padding in pixels
    """
    theme = THEMES.get(theme_name, THEMES["classic"])
    cs = cell_size
    grid_px = 9 * cs
    img_w = grid_px + 2 * padding
    img_h = grid_px + 2 * padding

    img = Image.new("RGB", (img_w, img_h), color=theme["bg"])
    draw = ImageDraw.Draw(img)
    font = _get_font(int(cs * 0.55))

    # Draw cell backgrounds and numbers
    display = puzzle.solution if show_solution else puzzle.clues
    for r in range(9):
        for c in range(9):
            x = padding + c * cs
            y = padding + r * cs
            val = display[r][c]
            is_given = puzzle.clues[r][c] != 0

            bg_color = theme["given_bg"] if is_given else theme["bg"]
            draw.rectangle([x, y, x + cs, y + cs], fill=bg_color)

            if val:
                fg = theme["given_fg"] if is_given else theme["empty_fg"]
                # Centre the digit
                try:
                    bbox = font.getbbox(str(val))
                    tw = bbox[2] - bbox[0]
                    th = bbox[3] - bbox[1]
                except Exception:
                    tw, th = cs // 2, cs // 2
                tx = x + (cs - tw) // 2
                ty = y + (cs - th) // 2
                draw.text((tx, ty), str(val), fill=fg, font=font)

    # Grid lines
    for i in range(10):
        thick = (i % 3 == 0)
        lw = 3 if thick else 1
        color = theme["thick_line"] if thick else theme["thin_line"]
        # Horizontal
        hy = padding + i * cs
        draw.line([(padding, hy), (padding + grid_px, hy)], fill=color, width=lw)
        # Vertical
        vx = padding + i * cs
        draw.line([(vx, padding), (vx, padding + grid_px)], fill=color, width=lw)

    # Outer border
    draw.rectangle(
        [padding, padding, padding + grid_px, padding + grid_px],
        outline=theme["border"], width=3,
    )

    return img


def render_all_themes(puzzle: SudokuPuzzle, show_solution: bool = False) -> dict:
    return {
        name: render_puzzle(puzzle, show_solution=show_solution, theme_name=name)
        for name in THEMES
    }


def get_gpt_image2_style_prompt(theme_name: str) -> str:
    prompts = {
        "dark": (
            "Convert this Sudoku puzzle image to a dark theme with high-contrast colors. "
            "Keep all numbers and grid lines clearly readable. "
            "Maintain the exact same numbers in the exact same positions."
        ),
        "warm": (
            "Redesign this Sudoku puzzle with warm, parchment-like colors. "
            "Keep all given numbers and grid structure identical. "
            "Make it look like a newspaper or magazine puzzle."
        ),
        "minimal": (
            "Simplify this Sudoku puzzle to a clean, minimalist design. "
            "Use thin grid lines and a monochrome color scheme. "
            "All numbers must remain in their exact same positions."
        ),
        "colorful": (
            "Redesign this Sudoku puzzle with a colorful, vibrant theme where "
            "each 3x3 box has a distinct background color. "
            "Keep all given numbers identical and clearly legible."
        ),
    }
    return prompts.get(theme_name, prompts["minimal"])
