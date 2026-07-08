"""
Atari Pong renderer: converts a PongPuzzle to a PIL Image.

Canvas: 320×420 px (2× Atari 160×210 resolution).
Supports 3 visual themes and optional ball trajectory overlay.
"""
from __future__ import annotations

import math
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from track_b.generators.pong_gen import (
    PongPuzzle,
    PADDLE_HEIGHT,
    PADDLE_WIDTH,
    BALL_RADIUS,
    RIGHT_PADDLE_X,
    LEFT_PADDLE_X,
    _simulate_ball,
    _move_paddle,
    _left_ai_move,
    NOOP, UP, DOWN,
)

# ---------------------------------------------------------------------------
# Render dimensions
# ---------------------------------------------------------------------------
RENDER_W = 320
RENDER_H = 420
SCALE_X  = RENDER_W   # multiply normalized x → pixel x
SCALE_Y  = RENDER_H   # multiply normalized y → pixel y

# Score area at top (pixels)
SCORE_AREA_H = 30

# Effective play area height
PLAY_AREA_H = RENDER_H - SCORE_AREA_H

# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------
THEMES: dict[str, dict] = {
    "atari": {
        "bg":              "#000000",
        "score_bg":        "#000000",
        "score_fg":        "#FFFFFF",
        "center_line":     "#808080",
        "left_paddle":     "#FFFFFF",
        "right_paddle":    "#FFFFFF",
        "ball":            "#FFFFFF",
        "trajectory":      "#404040",
        "border":          "#333333",
    },
    "modern": {
        "bg":              "#0A0A2E",
        "score_bg":        "#0A0A2E",
        "score_fg":        "#00FFFF",
        "center_line":     "#1A1A6E",
        "left_paddle":     "#FF6B6B",
        "right_paddle":    "#00FFFF",
        "ball":            "#FFD700",
        "trajectory":      "#1A3A3A",
        "border":          "#00FFFF",
    },
    "classic": {
        "bg":              "#1C1C1C",
        "score_bg":        "#111111",
        "score_fg":        "#DDDDDD",
        "center_line":     "#444444",
        "left_paddle":     "#CCCCCC",
        "right_paddle":    "#EEEEEE",
        "ball":            "#FFFFFF",
        "trajectory":      "#383838",
        "border":          "#555555",
    },
}


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def _to_px(nx: float, ny: float) -> tuple[int, int]:
    """Normalized (0–1) → pixel coordinates in the full RENDER_W × RENDER_H canvas."""
    px = int(nx * RENDER_W)
    py = SCORE_AREA_H + int(ny * PLAY_AREA_H)
    return px, py


def _paddle_rect(px_x: float, center_ny: float) -> tuple[int, int, int, int]:
    """
    Return (x0, y0, x1, y1) pixel rectangle for a paddle.
    px_x is normalized x-center of the paddle.
    """
    pw_px = int(PADDLE_WIDTH * RENDER_W)
    ph_px = int(PADDLE_HEIGHT * PLAY_AREA_H)
    cx, cy = _to_px(px_x, center_ny)
    x0 = cx - pw_px // 2
    x1 = cx + pw_px // 2
    y0 = cy - ph_px // 2
    y1 = cy + ph_px // 2
    return x0, y0, x1, y1


def _ball_rect(bx: float, by: float) -> tuple[int, int, int, int]:
    """Return (x0, y0, x1, y1) for the ball square."""
    r_px = max(2, int(BALL_RADIUS * RENDER_W))
    cx, cy = _to_px(bx, by)
    return cx - r_px, cy - r_px, cx + r_px, cy + r_px


# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------

def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Trajectory computation
# ---------------------------------------------------------------------------

def _compute_trajectory(
    bx: float, by: float, vx: float, vy: float,
    actions: list[int],
    right_y: float,
    left_y: float,
) -> list[tuple[float, float]]:
    """Return list of (bx, by) positions for each step (including initial)."""
    positions = [(bx, by)]
    for act in actions:
        bx, by, vx, vy = _simulate_ball(bx, by, vx, vy)
        right_y = _move_paddle(right_y, act)
        left_y  = _left_ai_move(left_y, by)
        positions.append((bx, by))
    return positions


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_puzzle(
    puzzle: PongPuzzle,
    theme_name: str = "atari",
    show_trajectory: bool = True,
) -> Image.Image:
    """
    Render a PongPuzzle to a 320×420 PIL Image.

    Args:
        puzzle: PongPuzzle to render
        theme_name: one of 'atari', 'modern', 'classic'
        show_trajectory: if True, draw faint dotted ball trajectory
    Returns:
        PIL.Image.Image (RGB)
    """
    theme = THEMES.get(theme_name, THEMES["atari"])

    img = Image.new("RGB", (RENDER_W, RENDER_H), color=theme["bg"])
    draw = ImageDraw.Draw(img)

    # ---- Score area background ----
    draw.rectangle([0, 0, RENDER_W - 1, SCORE_AREA_H - 1], fill=theme["score_bg"])

    # Fake scores (left AI = 0, right player = 0)
    font_score = _get_font(20)
    draw.text((RENDER_W // 4, 4), "0", fill=theme["score_fg"], font=font_score)
    draw.text((3 * RENDER_W // 4 - 10, 4), "0", fill=theme["score_fg"], font=font_score)

    # ---- Separator line between score and play area ----
    draw.line([(0, SCORE_AREA_H), (RENDER_W, SCORE_AREA_H)], fill=theme["border"], width=1)

    # ---- Dashed center line ----
    dash_len  = 10
    gap_len   = 8
    cx = RENDER_W // 2
    y  = SCORE_AREA_H
    drawing_dash = True
    while y < RENDER_H:
        end_y = min(y + (dash_len if drawing_dash else gap_len), RENDER_H)
        if drawing_dash:
            draw.line([(cx, y), (cx, end_y)], fill=theme["center_line"], width=2)
        y = end_y
        drawing_dash = not drawing_dash

    # ---- Ball trajectory (faint dotted) ----
    if show_trajectory:
        traj = _compute_trajectory(
            puzzle.ball_x, puzzle.ball_y, puzzle.ball_vx, puzzle.ball_vy,
            puzzle.actions, puzzle.right_paddle_y, puzzle.left_paddle_y,
        )
        for i, (tx, ty) in enumerate(traj):
            if i % 2 == 0:  # dotted: draw every other point
                px, py = _to_px(tx, ty)
                r = max(2, int(BALL_RADIUS * RENDER_W * 0.5))
                draw.ellipse(
                    [px - r, py - r, px + r, py + r],
                    fill=theme["trajectory"],
                )

    # ---- Left paddle (AI) ----
    lx0, ly0, lx1, ly1 = _paddle_rect(LEFT_PADDLE_X, puzzle.left_paddle_y)
    draw.rectangle([lx0, ly0, lx1, ly1], fill=theme["left_paddle"])

    # ---- Right paddle (controlled) ----
    rx0, ry0, rx1, ry1 = _paddle_rect(RIGHT_PADDLE_X, puzzle.right_paddle_y)
    draw.rectangle([rx0, ry0, rx1, ry1], fill=theme["right_paddle"])

    # ---- Ball ----
    bx0, by0, bx1, by1 = _ball_rect(puzzle.ball_x, puzzle.ball_y)
    draw.rectangle([bx0, by0, bx1, by1], fill=theme["ball"])

    # ---- Info overlay (puzzle id + difficulty) ----
    font_info = _get_font(11)
    info_text = f"#{puzzle.puzzle_id} | {puzzle.difficulty} | steps={puzzle.n_steps}"
    draw.text((4, RENDER_H - 16), info_text, fill=theme["score_fg"], font=font_info)

    return img


# ---------------------------------------------------------------------------
# Render all themes
# ---------------------------------------------------------------------------

def render_all_themes(puzzle: PongPuzzle, show_trajectory: bool = True) -> dict[str, Image.Image]:
    """Return dict of theme_name → PIL Image for all themes."""
    return {
        name: render_puzzle(puzzle, theme_name=name, show_trajectory=show_trajectory)
        for name in THEMES
    }


# ---------------------------------------------------------------------------
# GPT-Image-2 style prompts
# ---------------------------------------------------------------------------

def get_gpt_image2_style_prompt(theme_name: str) -> str:
    """Return a GPT-Image-2 edit prompt for visual style diversification."""
    prompts = {
        "atari": (
            "Transform this Pong game screenshot into the style of a classic Atari 2600 arcade game. "
            "Use a pure black background with sharp white rectangles for paddles and ball. "
            "Add a dashed white center line. Keep all game elements (paddles, ball) in their exact positions."
        ),
        "modern": (
            "Redesign this Pong game image in a modern neon cyberpunk aesthetic. "
            "Use a deep navy blue background with a cyan right paddle, red left paddle, and gold ball. "
            "Add glowing neon outlines around all elements. Keep exact positions of paddles and ball."
        ),
        "classic": (
            "Convert this Pong game image to a retro dark-gray arcade style. "
            "Use a dark charcoal background with light gray paddles and a white ball. "
            "Keep the exact layout of paddles and ball unchanged."
        ),
        "blueprint": (
            "Render this Pong game in a technical blueprint style with a blue background and white lines. "
            "Draw paddles and ball as clean white geometric shapes with thin outlines. "
            "Preserve the exact positions of all game elements."
        ),
    }
    return prompts.get(theme_name, prompts["atari"])


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from track_b.generators.pong_gen import generate_puzzles

    puzzles = generate_puzzles(n=3, seed=42)
    for p in puzzles:
        for tname in THEMES:
            img = render_puzzle(p, theme_name=tname, show_trajectory=True)
            out = f"/tmp/pong_{p.puzzle_id}_{tname}.png"
            img.save(out)
            print(f"Saved {out}")
