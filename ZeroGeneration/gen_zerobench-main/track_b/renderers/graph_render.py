"""
Directed graph renderer: converts a GraphPuzzle to a PIL Image.

Implementation note: does NOT use networkx drawing routines.
All layout and drawing is done manually with PIL.

Layout: circular node placement (evenly spaced on a circle).
Edges: straight lines with arrowheads; self-loops as small arcs.
Nodes: filled circles with label and value text.
"""
from __future__ import annotations

import math
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from track_b.generators.graph_gen import GraphPuzzle


# ---------------------------------------------------------------------------
# Canvas / layout constants
# ---------------------------------------------------------------------------
IMG_SIZE   = 600          # square canvas side in pixels
MARGIN     = 70           # margin around the circular layout
NODE_R     = 28           # node circle radius in pixels
ARROW_SIZE = 10           # arrowhead size in pixels
SELF_LOOP_R = 20          # radius of self-loop arc


# ---------------------------------------------------------------------------
# Themes
# ---------------------------------------------------------------------------
THEMES: dict[str, dict] = {
    "classic": {
        "bg":              "#FFFFFF",
        "node_fill":       "#4169E1",   # royal blue
        "node_outline":    "#000080",
        "node_text":       "#FFFFFF",
        "node_value_bg":   "#FFD700",
        "edge_color":      "#333333",
        "edge_text":       "#CC0000",
        "cycle_edge":      "#FF6600",
        "label_shadow":    "#CCCCCC",
        "title_color":     "#000000",
    },
    "dark": {
        "bg":              "#0D0D0D",
        "node_fill":       "#1A3A5C",
        "node_outline":    "#00BFFF",
        "node_text":       "#FFFFFF",
        "node_value_bg":   "#FF8C00",
        "edge_color":      "#888888",
        "edge_text":       "#00FF7F",
        "cycle_edge":      "#FF4500",
        "label_shadow":    "#222222",
        "title_color":     "#FFFFFF",
    },
    "blueprint": {
        "bg":              "#1B3A6B",
        "node_fill":       "#1B3A6B",
        "node_outline":    "#FFFFFF",
        "node_text":       "#FFFFFF",
        "node_value_bg":   "#4A90D9",
        "edge_color":      "#AADDFF",
        "edge_text":       "#FFD700",
        "cycle_edge":      "#FF6666",
        "label_shadow":    "#0A2040",
        "title_color":     "#FFFFFF",
    },
}


# ---------------------------------------------------------------------------
# Font helpers
# ---------------------------------------------------------------------------

def _get_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Layout: circular placement
# ---------------------------------------------------------------------------

def _circular_positions(
    nodes: list[str],
    cx: float,
    cy: float,
    radius: float,
) -> dict[str, tuple[float, float]]:
    """
    Place nodes evenly on a circle centered at (cx, cy) with the given radius.
    Returns dict: node_label → (px, py).
    """
    n = len(nodes)
    positions: dict[str, tuple[float, float]] = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n - math.pi / 2  # start at top
        px = cx + radius * math.cos(angle)
        py = cy + radius * math.sin(angle)
        positions[node] = (px, py)
    return positions


# ---------------------------------------------------------------------------
# Drawing primitives
# ---------------------------------------------------------------------------

def _draw_arrowhead(
    draw: ImageDraw.Draw,
    x1: float, y1: float,
    x2: float, y2: float,
    color: str,
    size: int = ARROW_SIZE,
    node_r: int = NODE_R,
):
    """
    Draw an arrowhead at the end (x2, y2) of the segment (x1,y1)→(x2,y2).
    The tip is placed at the edge of the destination node circle.
    """
    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return

    # Unit vector
    ux, uy = dx / length, dy / length

    # Tip of arrow: on the circle boundary
    tip_x = x2 - ux * node_r
    tip_y = y2 - uy * node_r

    # Base of arrowhead
    perp_x, perp_y = -uy, ux
    base_x = tip_x - ux * size
    base_y = tip_y - uy * size

    left_x  = base_x + perp_x * size * 0.5
    left_y  = base_y + perp_y * size * 0.5
    right_x = base_x - perp_x * size * 0.5
    right_y = base_y - perp_y * size * 0.5

    draw.polygon(
        [(tip_x, tip_y), (left_x, left_y), (right_x, right_y)],
        fill=color,
    )


def _draw_edge(
    draw: ImageDraw.Draw,
    x1: float, y1: float,
    x2: float, y2: float,
    weight: int,
    color: str,
    text_color: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    node_r: int = NODE_R,
    offset: float = 0.0,   # lateral offset for parallel edges
):
    """
    Draw a directed edge from (x1,y1) to (x2,y2) with weight label.
    offset != 0 shifts the edge slightly to handle reverse parallel edges.
    """
    if abs(x1 - x2) < 1 and abs(y1 - y2) < 1:
        # Self-loop: draw a small circle above the node
        loop_cx = x1
        loop_cy = y1 - node_r - SELF_LOOP_R
        r = SELF_LOOP_R
        draw.arc(
            [loop_cx - r, loop_cy - r, loop_cx + r, loop_cy + r],
            start=0, end=340, fill=color, width=2,
        )
        # Weight label
        draw.text((loop_cx + r, loop_cy - r // 2), str(weight),
                  fill=text_color, font=font)
        return

    dx = x2 - x1
    dy = y2 - y1
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return

    # Perpendicular unit vector for offset
    perp_x, perp_y = -dy / length, dx / length

    sx = x1 + perp_x * offset
    sy = y1 + perp_y * offset
    ex = x2 + perp_x * offset
    ey = y2 + perp_y * offset

    # Shorten line to node boundary
    ux, uy = (ex - sx) / length, (ey - sy) / length
    lx = sx + ux * node_r
    ly = sy + uy * node_r
    rx = ex - ux * node_r
    ry = ey - uy * node_r

    draw.line([(lx, ly), (rx, ry)], fill=color, width=2)

    # Arrowhead
    _draw_arrowhead(draw, sx, sy, ex, ey, color, node_r=node_r)

    # Weight label at midpoint
    mid_x = (lx + rx) / 2 + perp_x * 10
    mid_y = (ly + ry) / 2 + perp_y * 10
    label = str(weight)
    try:
        bbox = font.getbbox(label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw, th = 10, 12
    draw.text((mid_x - tw / 2, mid_y - th / 2), label, fill=text_color, font=font)


def _draw_node(
    draw: ImageDraw.Draw,
    px: float,
    py: float,
    label: str,
    value: int,
    theme: dict,
    font_label: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    font_value: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    node_r: int = NODE_R,
):
    """Draw a node circle with label and value."""
    # Main circle
    draw.ellipse(
        [px - node_r, py - node_r, px + node_r, py + node_r],
        fill=theme["node_fill"],
        outline=theme["node_outline"],
        width=3,
    )

    # Node label (e.g. "A") centred in the circle
    try:
        bbox_l = font_label.getbbox(label)
        lw = bbox_l[2] - bbox_l[0]
        lh = bbox_l[3] - bbox_l[1]
    except Exception:
        lw, lh = node_r, node_r
    draw.text(
        (px - lw / 2, py - lh / 2 - 2),
        label,
        fill=theme["node_text"],
        font=font_label,
    )

    # Node value: small badge below the label
    val_str = str(value)
    try:
        bbox_v = font_value.getbbox(val_str)
        vw = bbox_v[2] - bbox_v[0]
        vh = bbox_v[3] - bbox_v[1]
    except Exception:
        vw, vh = 8, 10

    badge_r = int(node_r * 0.45)
    badge_x = px + node_r * 0.55
    badge_y = py + node_r * 0.55
    draw.ellipse(
        [badge_x - badge_r, badge_y - badge_r, badge_x + badge_r, badge_y + badge_r],
        fill=theme["node_value_bg"],
        outline=theme["node_outline"],
        width=1,
    )
    draw.text(
        (badge_x - vw / 2, badge_y - vh / 2),
        val_str,
        fill=theme["node_outline"],
        font=font_value,
    )


# ---------------------------------------------------------------------------
# Main render function
# ---------------------------------------------------------------------------

def render_puzzle(
    puzzle: GraphPuzzle,
    theme_name: str = "classic",
    highlight_cycles: bool = False,
) -> Image.Image:
    """
    Render a GraphPuzzle to a 600×600 PIL Image.

    Args:
        puzzle: GraphPuzzle to render
        theme_name: one of 'classic', 'dark', 'blueprint'
        highlight_cycles: if True, draw cycle edges in a distinct color
    Returns:
        PIL.Image.Image (RGB)
    """
    theme = THEMES.get(theme_name, THEMES["classic"])

    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=theme["bg"])
    draw = ImageDraw.Draw(img)

    font_label = _get_font(18)
    font_value = _get_font(11)
    font_weight = _get_font(13)
    font_title  = _get_font(14)

    # ---- Layout ----
    cx = IMG_SIZE / 2
    cy = IMG_SIZE / 2
    radius = IMG_SIZE / 2 - MARGIN - NODE_R

    pos = _circular_positions(puzzle.nodes, cx, cy, radius)

    # ---- Identify cycle edges (for highlighting) ----
    cycle_edge_set: set[tuple[str, str]] = set()
    if highlight_cycles and puzzle.cycles:
        for cycle in puzzle.cycles:
            n = len(cycle)
            for i in range(n):
                fn = cycle[i]
                tn = cycle[(i + 1) % n]
                cycle_edge_set.add((fn, tn))

    # ---- Detect parallel (bidirectional) edges for offset ----
    # Two edges (A→B) and (B→A) need to be offset to avoid overlap.
    edge_pairs: set[frozenset] = set()
    parallel_pairs: set[frozenset] = set()
    for fn, tn, _ in puzzle.edges:
        pair = frozenset({fn, tn})
        if pair in edge_pairs:
            parallel_pairs.add(pair)
        edge_pairs.add(pair)

    # ---- Draw edges ----
    OFFSET = 10  # pixels

    for fn, tn, w in puzzle.edges:
        x1, y1 = pos[fn]
        x2, y2 = pos[tn]

        is_cycle_edge = (fn, tn) in cycle_edge_set
        color = theme["cycle_edge"] if is_cycle_edge else theme["edge_color"]

        pair = frozenset({fn, tn})
        if pair in parallel_pairs and fn < tn:
            off = OFFSET
        elif pair in parallel_pairs:
            off = -OFFSET
        else:
            off = 0.0

        _draw_edge(
            draw, x1, y1, x2, y2, w,
            color=color,
            text_color=theme["edge_text"],
            font=font_weight,
            offset=off,
        )

    # ---- Draw nodes (on top of edges) ----
    for node in puzzle.nodes:
        px, py = pos[node]
        _draw_node(
            draw, px, py,
            label=node,
            value=puzzle.node_values[node],
            theme=theme,
            font_label=font_label,
            font_value=font_value,
        )

    # ---- Title / info ----
    subtype_label = "Traversal" if puzzle.subtype == "traversal" else "Cycle"
    title = (
        f"Graph #{puzzle.puzzle_id} — {subtype_label} [{puzzle.difficulty}]"
    )
    draw.text((8, 8), title, fill=theme["title_color"], font=font_title)

    if puzzle.subtype == "traversal":
        seq_str = " → ".join(str(w) for w in puzzle.sequence)
        info = f"Start: {puzzle.start_node}  |  Sequence: {seq_str}"
        draw.text((8, IMG_SIZE - 22), info, fill=theme["title_color"], font=font_title)
    else:
        n_cy = len(puzzle.cycles)
        info = f"Cycles found: {n_cy}  |  Answer: {puzzle.cycle_answer}"
        draw.text((8, IMG_SIZE - 22), info, fill=theme["title_color"], font=font_title)

    # ---- Legend for highlighted cycles ----
    if highlight_cycles and cycle_edge_set:
        legend_y = 28
        draw.rectangle(
            [8, legend_y, 24, legend_y + 10],
            fill=theme["cycle_edge"],
        )
        draw.text(
            (28, legend_y - 1),
            "cycle edge",
            fill=theme["title_color"],
            font=_get_font(11),
        )

    return img


# ---------------------------------------------------------------------------
# All themes
# ---------------------------------------------------------------------------

def render_all_themes(
    puzzle: GraphPuzzle,
    highlight_cycles: bool = False,
) -> dict[str, Image.Image]:
    """Return dict of theme_name → PIL Image for all themes."""
    return {
        name: render_puzzle(puzzle, theme_name=name, highlight_cycles=highlight_cycles)
        for name in THEMES
    }


# ---------------------------------------------------------------------------
# GPT-Image-2 style prompts
# ---------------------------------------------------------------------------

def get_gpt_image2_style_prompt(theme_name: str) -> str:
    """Return a GPT-Image-2 edit prompt for visual style diversification."""
    prompts = {
        "classic": (
            "Transform this directed graph diagram into a clean, academic whiteboard style. "
            "Use white background, black edges with clear arrowheads, and blue filled node circles. "
            "Keep all node labels, edge weights, and graph topology exactly the same."
        ),
        "dark": (
            "Redesign this directed graph diagram in a dark-mode style with a near-black background. "
            "Use bright cyan outlines for nodes and light gray arrows for edges. "
            "Keep all labels, weights, and the exact graph structure unchanged."
        ),
        "blueprint": (
            "Convert this directed graph into a technical blueprint aesthetic. "
            "Use a deep blue background with white lines for edges and white circle nodes. "
            "All node labels and edge weights must remain in their exact positions."
        ),
        "neon": (
            "Render this directed graph with a neon cyberpunk look on a black background. "
            "Use glowing neon green edges with purple node circles and yellow weight labels. "
            "Maintain all node positions, labels, and edge connections exactly."
        ),
    }
    return prompts.get(theme_name, prompts["classic"])


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent.parent))
    from track_b.generators.graph_gen import generate_puzzles

    puzzles = generate_puzzles(n=4, seed=42)
    for p in puzzles:
        for tname in THEMES:
            img = render_puzzle(p, theme_name=tname, highlight_cycles=(p.subtype == "cycle"))
            out = f"/tmp/graph_{p.puzzle_id}_{p.subtype}_{tname}.png"
            img.save(out)
            print(f"Saved {out}")
