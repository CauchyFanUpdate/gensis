"""
Rule-based reward for llava-format VLM tasks.

data_source → answer type:
  llava:multiple-choice  → single letter GT ("A") → letter extraction + exact match
  llava:number           → integer/float string    → numeric MRA
  llava:ocrtext          → free text               → case-insensitive exact / substring
  llava:bbox             → "[x1,y1,x2,y2]" (0-1)  → IoU

Score = 0.1 * format_reward + 0.9 * result_reward
format_reward:
  1.0  → both <think>...</think> AND <answer>...</answer> present
  0.5  → only <think>...</think> present
  0.0  → neither
"""

import json
import re

import numpy as np

FORMAT_SCORE = 0.1

_THINK_RE  = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.DOTALL | re.IGNORECASE)
_FLOAT_RE  = re.compile(r"-?\d+\.?\d*")
_CONFS     = np.linspace(0.5, 0.95, int((0.95 - 0.5) / 0.05) + 2)


# ── helpers ──────────────────────────────────────────────────────────────────

def _format_reward(text: str) -> float:
    has_think  = bool(_THINK_RE.search(text))
    has_answer = bool(_ANSWER_RE.search(text))
    if has_think and has_answer:
        return 1.0
    if has_think:
        return 0.5
    return 0.0


def _strip_think(text: str) -> str:
    stripped = _THINK_RE.sub("", text).strip()
    return stripped or text.strip()


def _mra(pred: float, gt: float) -> float:
    rel_err = abs(pred - gt) if gt == 0 else abs(pred - gt) / abs(gt)
    return float(np.mean(rel_err <= (1.0 - _CONFS)))


def _extract_float(text: str):
    m = _FLOAT_RE.search(text)
    return float(m.group()) if m else None


def _extract_choice(text: str):
    """Extract final answer letter from model response."""
    # <answer>A</answer>
    m = re.findall(r"<answer>\s*([A-F])\s*</answer>", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    # \boxed{A}
    m = re.findall(r"\\boxed\{([A-F])\}", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    # (A)
    m = re.findall(r"\(([A-F])\)", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    # "answer is A" / "答案是A"
    m = re.findall(
        r"(?:answer\s+is\s*:?\s*|答案[是为：:]\s*|选择?[：:]\s*)([A-F])\b",
        text, re.IGNORECASE,
    )
    if m:
        return m[-1].upper()
    # last non-empty line starting with a letter
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
    # fallback: last "A." style occurrence
    m = re.findall(r"\b([A-F])\.", text, re.IGNORECASE)
    if m:
        return m[-1].upper()
    return None


def _parse_bbox(text: str):
    """Parse [x1,y1,x2,y2] from text (normalized 0-1 coords)."""
    text = text.strip()
    # try JSON array
    try:
        obj = json.loads(text)
        if isinstance(obj, list) and len(obj) == 4:
            return [float(v) for v in obj]
    except Exception:
        pass
    # fallback: grab four floats
    nums = re.findall(r"-?\d+\.?\d*", text)
    if len(nums) >= 4:
        return [float(n) for n in nums[:4]]
    return None


def _bbox_iou(b1, b2) -> float:
    x1 = max(b1[0], b2[0]); y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2]); y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


# ── per-type result rewards ───────────────────────────────────────────────────

def _result_mc(resp: str, gt: str) -> float:
    gt_letter = gt.strip().upper()
    pred = _extract_choice(resp)
    return 1.0 if (pred is not None and pred == gt_letter) else 0.0


def _result_number(resp: str, gt: str) -> float:
    try:
        gt_val = float(gt.strip())
    except ValueError:
        return 0.0
    pred_val = _extract_float(resp)
    if pred_val is None:
        return 0.0
    return _mra(pred_val, gt_val)


def _result_ocrtext(resp: str, gt: str) -> float:
    gt_norm  = gt.strip().lower()
    resp_norm = resp.lower()
    # exact match first, then substring
    if gt_norm == resp_norm.strip():
        return 1.0
    return 1.0 if gt_norm in resp_norm else 0.0


def _result_bbox(resp: str, gt: str) -> float:
    gt_box   = _parse_bbox(gt)
    pred_box = _parse_bbox(resp)
    if gt_box is None or pred_box is None:
        return 0.0
    return _bbox_iou(pred_box, gt_box)


# ── public API ────────────────────────────────────────────────────────────────

_RESULT_FN = {
    "llava:multiple-choice": _result_mc,
    "llava:number":          _result_number,
    "llava:ocrtext":         _result_ocrtext,
    "llava:bbox":            _result_bbox,
}


def compute_score(solution_str: str, ground_truth: str,
                  data_source: str = "", extra_info=None) -> float:
    fmt  = _format_reward(solution_str)
    resp = _strip_think(solution_str)
    gt   = ground_truth.strip()

    result_fn = _RESULT_FN.get(data_source, _result_ocrtext)
    res = result_fn(resp, gt)

    return FORMAT_SCORE * fmt + (1.0 - FORMAT_SCORE) * res
