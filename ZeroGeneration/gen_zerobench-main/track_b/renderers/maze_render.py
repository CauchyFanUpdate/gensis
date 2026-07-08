from PIL import Image, ImageDraw, ImageFont
from typing import Optional
import math

from track_b.generators.maze_gen import MazePuzzle

CELL_SIZE = 24

THEMES = {
    "classic": {
        "bg":       "#FFFFFF",
        "wall":     "#2C2C2C",
        "floor":    "#F8F8F8",
        "start":    "#27AE60",
        "end":      "#E74C3C",
        "text":     "#FFFFFF",
        "border":   "#1A1A1A",
    },
    "dark": {
        "bg":       "#1A1A2E",
        "wall":     "#0F0F1A",
        "floor":    "#16213E",
        "start":    "#00FF7F",
        "end":      "#FF4500",
        "text":     "#FFFFFF",
        "border":   "#000000",
    },
    "colorful": {
        "bg":       "#FFF8F0",
        "wall":     "#8B4513",
        "floor":    "#FFFACD",
        "start":    "#228B22",
        "end":      "#DC143C",
        "text":     "#FFFFFF",
        "border":   "#5C3317",
    },
}


def _draw_star(draw: ImageDraw.Draw, cx: int, cy: int, r_outer: int, color: str, outline: str):
    n_points = 5
    points = []
    r_inner = r_outer * 0.45
    for i in range(n_points * 2):
        angle = math.pi / n_points * i - math.pi / 2
        r = r_outer if i % 2 == 0 else r_inner
        points.append((cx + r * math.cos(angle), cy + r * math.sin(angle)))
    draw.polygon(points, fill=color, outline=outline)


def _draw_triangle(draw: ImageDraw.Draw, cx: int, cy: int, r: int, color: str, outline: str):
    pts = [
        (cx, cy - r),
        (cx - int(r * 0.866), cy + r // 2),
        (cx + int(r * 0.866), cy + r // 2),
    ]
    draw.polygon(pts, fill=color, outline=outline)


def _draw_player(draw: ImageDraw.Draw, cx: int, cy: int, shape: str, color: str, outline: str, cs: int):
    r = cs // 2 - 3
    if shape == "circle":
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=outline, width=2)
    elif shape == "square":
        draw.rectangle([cx - r, cy - r, cx + r, cy + r], fill=color, outline=outline, width=2)
    elif shape == "triangle":
        _draw_triangle(draw, cx, cy, r, color, outline)
    else:
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color, outline=outline, width=2)


def _get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def render_puzzle(puzzle: MazePuzzle, theme_name: str = "classic") -> Image.Image:
    theme = THEMES.get(theme_name, THEMES["classic"])
    cs = CELL_SIZE
    padding = 8

    img_w = puzzle.cols * cs + 2 * padding
    img_h = puzzle.rows * cs + 2 * padding
    img = Image.new("RGB", (img_w, img_h), color=theme["bg"])
    draw = ImageDraw.Draw(img)

    for r in range(puzzle.rows):
        for c in range(puzzle.cols):
            x = padding + c * cs
            y = padding + r * cs
            cell_val = puzzle.grid[r][c]
            fill = theme["wall"] if cell_val == 1 else theme["floor"]
            draw.rectangle([x, y, x + cs - 1, y + cs - 1], fill=fill)

    er, ec = puzzle.end
    ex = padding + ec * cs
    ey = padding + er * cs
    draw.rectangle([ex, ey, ex + cs - 1, ey + cs - 1], fill=theme["end"])
    ecx, ecy = ex + cs // 2, ey + cs // 2
    _draw_star(draw, ecx, ecy, cs // 2 - 2, theme["end"], theme["border"])

    font_small = _get_font(max(8, cs // 3))
    try:
        bbox = font_small.getbbox("E")
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    except Exception:
        tw, th = cs // 3, cs // 3
    draw.text((ecx - tw // 2, ecy - th // 2), "E", fill=theme["text"], font=font_small)

    for player in puzzle.players:
        pr, pc = player["pos"]
        px = padding + pc * cs
        py = padding + pr * cs
        pcx, pcy = px + cs // 2, py + cs // 2
        _draw_player(draw, pcx, pcy, player["shape"], player["color"], theme["border"], cs)

        pid = player["id"]
        try:
            bbox = font_small.getbbox(pid)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except Exception:
            tw, th = cs // 3, cs // 3
        draw.text((pcx - tw // 2, pcy - th // 2), pid, fill=theme["text"], font=font_small)

    return img


def get_gpt_image2_style_prompt(theme_name: str) -> str:
    prompts = {
        "classic": (
            "Render this maze puzzle in a clean classic style with white open paths and "
            "dark grey walls. Mark the exit with a red star labeled E. Each colored player "
            "shape must remain in its exact grid position. Keep all labels legible."
        ),
        "dark": (
            "Transform this maze puzzle into a dark sci-fi theme. Use a very dark navy "
            "background with near-black walls and slightly lighter dark-blue floor cells. "
            "Accent the exit with a neon orange-red star and player shapes with bright neon "
            "colors. Keep every element in its exact position."
        ),
        "colorful": (
            "Convert this maze puzzle to a warm colorful theme. Use cream-yellow open paths "
            "and rich brown walls. The exit star should be crimson red. Player shapes should "
            "use vibrant saturated colors. Maintain the exact maze layout and all positions."
        ),
    }
    return prompts.get(theme_name, prompts["classic"])
