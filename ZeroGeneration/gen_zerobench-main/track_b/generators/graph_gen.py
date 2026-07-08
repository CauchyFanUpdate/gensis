"""
Directed graph puzzle generator.

Generates two puzzle subtypes inspired by ZeroBench graph questions:

  - "traversal": follow a sequence of edge weights from a start node;
                 answer = product of node_values along the visited path
                 (Q15-style)

  - "cycle":     find all distinct simple cycles in a directed graph;
                 answer = product of cycle lengths
                 (Q32-style)

Both subtypes guarantee a unique, deterministic answer.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class GraphPuzzle:
    puzzle_id: int
    subtype: str                        # "traversal" or "cycle"
    nodes: list[str]                    # node labels (e.g. ["A", "B", ...])
    node_values: dict[str, int]         # value at each node (1–9)
    edges: list[tuple[str, str, int]]   # (from_node, to_node, weight)

    # Traversal fields
    start_node: str
    sequence: list[int]                 # sequence of edge weights to follow
    traversal_answer: int               # product of node_values along path

    # Cycle fields
    cycles: list[list[str]]             # each cycle as ordered list of node labels
    cycle_answer: int                   # product of cycle lengths

    solution: Optional[str]
    difficulty: str                     # "easy" / "medium" / "hard"


# ---------------------------------------------------------------------------
# Utility: simple cycles via DFS (Johnson's-style for small graphs)
# ---------------------------------------------------------------------------

def _find_simple_cycles(adj: dict[str, list[str]]) -> list[list[str]]:
    """
    Find all simple cycles in a directed graph using a DFS approach.
    Returns each cycle as a list of nodes (start node NOT repeated at end).
    Suitable for small graphs (≤ 10 nodes).
    """
    nodes = sorted(adj.keys())
    cycles: list[list[str]] = []
    seen_sets: list[frozenset] = []  # dedup cycles by node-set

    def dfs(start: str, current: str, path: list[str], visited: set[str]):
        for nxt in adj.get(current, []):
            if nxt == start and len(path) >= 2:
                # Found a cycle
                cycle = path[:]
                cset = frozenset(cycle)
                # Canonical: rotate so smallest node is first to dedup
                min_idx = cycle.index(min(cycle))
                canonical = tuple(cycle[min_idx:] + cycle[:min_idx])
                if cset not in seen_sets:
                    seen_sets.append(cset)
                    cycles.append(list(canonical))
            elif nxt not in visited:
                visited.add(nxt)
                path.append(nxt)
                dfs(start, nxt, path, visited)
                path.pop()
                visited.remove(nxt)

    for node in nodes:
        dfs(node, node, [node], {node})

    return cycles


# ---------------------------------------------------------------------------
# Traversal puzzle generator
# ---------------------------------------------------------------------------

def _make_traversal_puzzle(rng: random.Random, puzzle_id: int) -> GraphPuzzle:
    """
    Generate a traversal-type GraphPuzzle.

    Strategy:
    1. Choose n_nodes in [5, 9] with labels A–I.
    2. Assign random values 1–9 to each node.
    3. Build a random path of length 3–6 from a start node.
    4. Add random extra edges (with distinct weights where possible).
    5. The sequence is the list of edge weights along the path.
    6. Answer = product of node_values along the path (including start).
    """
    n_nodes = rng.randint(5, 9)
    labels  = [chr(ord("A") + i) for i in range(n_nodes)]
    node_values = {lbl: rng.randint(1, 9) for lbl in labels}

    # Build a guaranteed path: choose path length 3–(n_nodes-1)
    path_len = rng.randint(3, min(6, n_nodes - 1))
    shuffled = labels[:]
    rng.shuffle(shuffled)
    path_nodes = shuffled[:path_len + 1]  # path_len edges → path_len+1 nodes

    # Assign unique weights to path edges
    # Use weights 1–9, ensuring each edge in path has a unique weight so
    # the traversal sequence is unambiguous (exactly one next node for each weight).
    available_weights = list(range(1, 10))
    rng.shuffle(available_weights)
    path_weights = available_weights[:path_len]

    edges: list[tuple[str, str, int]] = []
    used_edge_keys: set[tuple[str, int]] = set()  # (from_node, weight) for uniqueness

    for i in range(path_len):
        fn = path_nodes[i]
        tn = path_nodes[i + 1]
        w  = path_weights[i]
        edges.append((fn, tn, w))
        used_edge_keys.add((fn, w))

    # Add random extra edges (not conflicting with path-edge weights from same source)
    extra_pairs = [(a, b) for a in labels for b in labels if a != b]
    rng.shuffle(extra_pairs)
    extra_added = 0
    for fn, tn in extra_pairs:
        if extra_added >= n_nodes:
            break
        # Pick a weight not already used from fn
        used_from_fn = {w for (f, w) in used_edge_keys if f == fn}
        candidate_ws = [w for w in range(1, 10) if w not in used_from_fn]
        if not candidate_ws:
            continue
        # Check this edge isn't already in edges
        if any(e[0] == fn and e[1] == tn for e in edges):
            continue
        w = rng.choice(candidate_ws)
        edges.append((fn, tn, w))
        used_edge_keys.add((fn, w))
        extra_added += 1

    sequence = path_weights
    traversal_answer = 1
    for node in path_nodes:
        traversal_answer *= node_values[node]

    start_node = path_nodes[0]

    # Compute cycles for completeness (may be empty for traversal puzzles)
    adj = {lbl: [] for lbl in labels}
    for fn, tn, _ in edges:
        adj[fn].append(tn)
    cycles = _find_simple_cycles(adj)

    if path_len <= 3:
        difficulty = "easy"
    elif path_len <= 5:
        difficulty = "medium"
    else:
        difficulty = "hard"

    solution = str(traversal_answer)

    return GraphPuzzle(
        puzzle_id=puzzle_id,
        subtype="traversal",
        nodes=labels,
        node_values=node_values,
        edges=edges,
        start_node=start_node,
        sequence=sequence,
        traversal_answer=traversal_answer,
        cycles=cycles,
        cycle_answer=1,
        solution=solution,
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# Cycle puzzle generator
# ---------------------------------------------------------------------------

def _make_cycle_puzzle(rng: random.Random, puzzle_id: int) -> GraphPuzzle:
    """
    Generate a cycle-type GraphPuzzle.

    Strategy:
    1. Choose n_nodes in [4, 7].
    2. Build 2–4 explicit simple cycles (as lists of nodes).
    3. Add edges for each cycle, with random weights.
    4. Add a few extra random edges.
    5. Answer = product of cycle lengths.
    """
    n_nodes = rng.randint(4, 7)
    labels  = [chr(ord("A") + i) for i in range(n_nodes)]
    node_values = {lbl: rng.randint(1, 9) for lbl in labels}

    n_cycles = rng.randint(2, min(4, n_nodes - 1))

    # Build cycles by choosing subsets of labels
    target_cycles: list[list[str]] = []
    used_cycle_sets: list[frozenset] = []
    attempts = 0
    while len(target_cycles) < n_cycles and attempts < 200:
        attempts += 1
        clen = rng.randint(2, min(4, n_nodes))
        subset = rng.sample(labels, clen)
        fs = frozenset(subset)
        if fs not in used_cycle_sets:
            used_cycle_sets.append(fs)
            target_cycles.append(subset)

    edges_set: set[tuple[str, str, int]] = set()
    edge_key_set: set[tuple[str, str]] = set()  # (from, to) dedup

    for cycle in target_cycles:
        for i in range(len(cycle)):
            fn = cycle[i]
            tn = cycle[(i + 1) % len(cycle)]
            if (fn, tn) not in edge_key_set:
                w = rng.randint(1, 9)
                edges_set.add((fn, tn, w))
                edge_key_set.add((fn, tn))

    # Extra random edges
    all_pairs = [(a, b) for a in labels for b in labels if a != b]
    rng.shuffle(all_pairs)
    for fn, tn in all_pairs[:n_nodes]:
        if (fn, tn) not in edge_key_set:
            w = rng.randint(1, 9)
            edges_set.add((fn, tn, w))
            edge_key_set.add((fn, tn))

    edges = sorted(edges_set)

    # Find actual cycles in the built graph
    adj = {lbl: [] for lbl in labels}
    for fn, tn, _ in edges:
        adj[fn].append(tn)
    found_cycles = _find_simple_cycles(adj)

    if not found_cycles:
        # Fallback: force a small cycle A→B→A
        fn, tn = labels[0], labels[1]
        for pair in [(fn, tn), (tn, fn)]:
            if pair not in edge_key_set:
                w = rng.randint(1, 9)
                edges_set.add((pair[0], pair[1], w))
                edge_key_set.add(pair)
                adj[pair[0]].append(pair[1])
        edges = sorted(edges_set)
        found_cycles = _find_simple_cycles(adj)

    cycle_answer = 1
    for c in found_cycles:
        cycle_answer *= len(c)

    if n_nodes <= 4:
        difficulty = "easy"
    elif n_nodes <= 6:
        difficulty = "medium"
    else:
        difficulty = "hard"

    solution = str(cycle_answer)

    return GraphPuzzle(
        puzzle_id=puzzle_id,
        subtype="cycle",
        nodes=labels,
        node_values=node_values,
        edges=edges,
        start_node=labels[0],
        sequence=[],
        traversal_answer=1,
        cycles=found_cycles,
        cycle_answer=cycle_answer,
        solution=solution,
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_puzzles(n: int = 40, seed: int = 42) -> list[GraphPuzzle]:
    """
    Generate n GraphPuzzle instances (alternating traversal / cycle subtypes).
    Deterministic given seed.
    """
    rng = random.Random(seed)
    puzzles: list[GraphPuzzle] = []
    for i in range(n):
        pid = i + 1
        if i % 2 == 0:
            puzzles.append(_make_traversal_puzzle(rng, pid))
        else:
            puzzles.append(_make_cycle_puzzle(rng, pid))
    return puzzles


def solve_puzzle(puzzle: GraphPuzzle) -> str:
    """Return the solution string (integer answer)."""
    if puzzle.subtype == "traversal":
        return str(puzzle.traversal_answer)
    else:
        return str(puzzle.cycle_answer)


def build_step_descriptions(puzzle: GraphPuzzle) -> str:
    """
    Build a step-by-step Chain-of-Thought description.
    Returns a human-readable string.
    """
    lines: list[str] = [
        f"=== Graph Puzzle #{puzzle.puzzle_id} [{puzzle.subtype}] — Chain of Thought ===",
        f"Nodes: {', '.join(f'{n}={puzzle.node_values[n]}' for n in puzzle.nodes)}",
        f"Edges: {puzzle.edges}",
        "",
    ]

    if puzzle.subtype == "traversal":
        lines.append(f"Start node: {puzzle.start_node}")
        lines.append(f"Follow sequence: {puzzle.sequence}")
        lines.append("")
        # Trace path
        # Build adjacency keyed by (from_node, weight) → to_node
        adj_w: dict[tuple[str, int], str] = {}
        for fn, tn, w in puzzle.edges:
            adj_w[(fn, w)] = tn

        current = puzzle.start_node
        path = [current]
        product = puzzle.node_values[current]
        lines.append(f"  Start at {current} (value={puzzle.node_values[current]})  running_product={product}")
        for step, w in enumerate(puzzle.sequence):
            nxt = adj_w.get((current, w))
            if nxt is None:
                lines.append(f"  Step {step+1}: follow weight {w} from {current} → NO EDGE (error)")
                break
            product *= puzzle.node_values[nxt]
            lines.append(
                f"  Step {step+1}: follow edge {current}→{nxt} (weight={w}, "
                f"node_value={puzzle.node_values[nxt]})  running_product={product}"
            )
            path.append(nxt)
            current = nxt

        lines.append("")
        lines.append(f"Path visited: {' → '.join(path)}")
        lines.append(f"Answer (product of node values): {puzzle.traversal_answer}")

    else:  # cycle
        lines.append("Finding all simple cycles in the directed graph...")
        lines.append("")
        products: list[int] = []
        for ci, cycle in enumerate(puzzle.cycles):
            clen = len(cycle)
            products.append(clen)
            lines.append(f"  Cycle {ci+1}: {' → '.join(cycle)} → {cycle[0]}  length={clen}")
        lines.append("")
        lines.append(f"Cycle lengths: {[len(c) for c in puzzle.cycles]}")
        lines.append(f"Answer (product of cycle lengths): {puzzle.cycle_answer}")

    lines.append(f"\nSolution: {puzzle.solution}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    puzzles = generate_puzzles(n=6, seed=42)
    for p in puzzles:
        print(
            f"Puzzle {p.puzzle_id} [{p.subtype}] [{p.difficulty}] "
            f"nodes={p.nodes} sol={p.solution}"
        )
        print(f"  edges={p.edges}")
        if p.subtype == "traversal":
            print(f"  start={p.start_node} seq={p.sequence} ans={p.traversal_answer}")
        else:
            print(f"  cycles={p.cycles} ans={p.cycle_answer}")
        print()
