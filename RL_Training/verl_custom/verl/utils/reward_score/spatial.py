"""
Rule-based reward for spatial VLM tasks (crosspoint / mindcube / spar).

Scoring:
  Backward compatible: compute_score() = 0.1 * format_reward + 0.9 * result_reward

  For LLM judge integration:
    format_reward(solution_str) → float [0, 1]
      1.0 if both <think> and <answer> present
      0.5 if only <think> present
      0.0 otherwise

    result_reward(solution_str, ground_truth) → float [0, 1]
      Routes by answer type:
        MC               → exact match of letter [A-F]
        Movement instr.  → 5-dim MRA for move/rotate commands
        JSON bbox        → IoU between predicted and ground truth boxes
        Pure number      → single-value MRA
        Distance         → extract float and compute MRA
        Fallback (text)  → case-insensitive substring match
"""

import json
import re
import numpy as np

_CONFS = np.linspace(0.5, 0.95, int((0.95 - 0.5) / 0.05) + 2)

_MC_GT_RE    = re.compile(r"^([A-F])\.", re.IGNORECASE)
_PURE_NUM_RE = re.compile(r"^-?\d+(\.\d+)?$")
_FLOAT_RE    = re.compile(r"-?\d+\.?\d*")
_MOVE_KEY_RE = re.compile(r"^(move|rotate)_\w+$")
_THINK_RE    = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE   = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

_MOVE_DIMS = [
    ("move_right",   "move_left"),
    ("move_up",      "move_down"),
    ("move_forward", "move_backward"),
    ("rotate_right", "rotate_left"),
    ("rotate_up",    "rotate_down"),
]

_DIR_WORDS = {"left", "right", "up", "down", "forward", "backward"}

FORMAT_SCORE = 0.1


def _format_reward(text: str) -> float:
    has_think = bool(_THINK_RE.search(text))
    has_answer = bool(_ANSWER_RE.search(text))
    if has_think and has_answer:
        return 1.0
    if has_think:
        return 0.5
    return 0.0


def _mra(pred: float, gt: float) -> float:
    rel_err = abs(pred - gt) if gt == 0 else abs(pred - gt) / abs(gt)
    return float(np.mean(rel_err <= (1.0 - _CONFS)))


def _mra_dims(pred_vec, gt_vec):
    if not pred_vec or not gt_vec:
        return 0.0
    return float(np.mean([_mra(p, g) for p, g in zip(pred_vec, gt_vec)]))


def _extract_answer_tag(text: str) -> str | None:
    """Extract content from <answer>...</answer> tag (prefer last occurrence)"""
    m = _ANSWER_RE.findall(text)
    return m[-1].strip() if m else None


def _extract_float(text):
    m = _FLOAT_RE.search(text)
    return float(m.group()) if m else None


def _parse_move_instruction(text):
    result = {}
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        parts = token.split(":")
        if len(parts) != 2:
            return None
        key, val = parts[0].strip(), parts[1].strip()
        if not _MOVE_KEY_RE.match(key):
            return None
        try:
            result[key] = float(val)
        except ValueError:
            return None
    return result if result else None


def _move_to_vec(d):
    return [d.get(pos, 0.0) - d.get(neg, 0.0) for pos, neg in _MOVE_DIMS]


def _parse_bbox(text):
    try:
        obj = json.loads(text)
        box = obj.get("bbox_2d")
        if box and len(box) == 4:
            return [int(v) for v in box]
    except Exception:
        pass
    nums = re.findall(r"-?\d+", text)
    if len(nums) >= 4:
        return [int(n) for n in nums[:4]]
    return None


def _bbox_iou(b1, b2):
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _extract_choice(text):
    m = re.findall(r"<answer>\s*([A-F])\s*</answer>", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\\boxed\{([A-F])\}", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(r"\(([A-F])\)", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    m = re.findall(
        r"(?:answer\s+is\s*:?\s*|答案[是为：:]\s*|选择?[：:]\s*)([A-F])\b",
        text, re.IGNORECASE,
    )
    if m:
        return m[-1].upper()
    for line in reversed(text.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        m2 = re.match(r"^([A-F])[.\s）)]", line, re.IGNORECASE)
        if m2:
            return m2.group(1).upper()
        if re.fullmatch(r"[A-F]", line, re.IGNORECASE):
            return line.upper()
        break
    m = re.findall(r"\b([A-F])\.", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    return None


_DIR_NORMALIZE = {"leftward": "left", "rightward": "right", "upward": "up", "downward": "down"}

def _extract_dirs(text: str) -> set[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return {_DIR_NORMALIZE.get(w, w) for w in words if w in _DIR_WORDS or w in _DIR_NORMALIZE}


def _gt_type(gt):
    if _MC_GT_RE.match(gt):
        return "mc"
    if _parse_move_instruction(gt) is not None:
        return "move"
    if gt.lstrip().startswith("{"):
        return "bbox"
    if _PURE_NUM_RE.match(gt):
        return "number"
    if re.search(r"\d", gt) and "meter" in gt.lower():
        return "distance"
    if not re.search(r"\d", gt) and _extract_dirs(gt):
        return "direction"
    return "text"


def _result_reward(solution_str: str, ground_truth: str) -> float:
    gt = ground_truth.strip()
    resp_full = solution_str.strip()
    resp = _THINK_RE.sub("", resp_full).strip() or resp_full

    # Prefer content inside <answer> tag if present
    tagged = _extract_answer_tag(resp)
    resp = tagged if tagged is not None else resp

    kind = _gt_type(gt)

    if kind == "mc":
        gt_letter = _MC_GT_RE.match(gt).group(1).upper()
        return 1.0 if (_extract_choice(resp) == gt_letter) else 0.0

    if kind == "move":
        gt_dict = _parse_move_instruction(gt)
        pred_dict = _parse_move_instruction(resp)
        if pred_dict is None:
            pred_dict = _parse_move_instruction(resp.strip().split("\n")[-1])
        if pred_dict is None:
            return 0.0
        return _mra_dims(_move_to_vec(pred_dict), _move_to_vec(gt_dict))

    if kind == "bbox":
        gt_box = _parse_bbox(gt)
        pred_box = _parse_bbox(resp)
        if gt_box is None or pred_box is None:
            return 0.0
        return _bbox_iou(pred_box, gt_box)

    if kind == "number":
        pred_val = _extract_float(resp)
        if pred_val is None:
            return 0.0
        return _mra(pred_val, float(gt))

    if kind == "distance":
        gt_val = _extract_float(gt)
        pred_val = _extract_float(resp)
        if gt_val is None or pred_val is None:
            return 0.0
        return _mra(pred_val, gt_val)

    if kind == "direction":
        gt_dirs = _extract_dirs(gt)
        pred_dirs = _extract_dirs(resp)
        if not pred_dirs:
            return 0.0
        union = gt_dirs | pred_dirs
        return len(gt_dirs & pred_dirs) / len(union)

    gt_norm = gt.lower().rstrip(".")
    return 1.0 if gt_norm in resp.lower() else 0.0


def format_reward(solution_str: str) -> float:
    """Public API: return only format reward [0, 1]"""
    return _format_reward(solution_str)


def result_reward(solution_str: str, ground_truth: str) -> float:
    """Public API: return only result reward [0, 1]"""
    return _result_reward(solution_str, ground_truth)


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Backward compatible: returns fused score"""
    fmt = _format_reward(solution_str)
    res = _result_reward(solution_str, ground_truth)
    return FORMAT_SCORE * fmt + (1.0 - FORMAT_SCORE) * res
