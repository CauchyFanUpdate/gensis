"""
Rule-based reward for zerobench data.

Scoring:
  Backward compatible: compute_score() = 0.1 * format_reward + 0.9 * result_reward

  For LLM judge integration:
    format_reward(solution_str) → float [0, 1]
      1.0 if both <think> and <answer> present
      0.5 if only <think> present
      0.0 otherwise

    result_reward(solution_str, ground_truth) → float [0, 1]
      Routes by answer type:
        numeric       → MRA (Mean Relative Accuracy)
        other (text)  → case-insensitive exact match (prefer <answer> tag content)
"""

import re

import numpy as np

FORMAT_SCORE = 0.1

_THINK_RE  = re.compile(r"<think>\s*(\S.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_ANSWER_RE = re.compile(r"<answer>\s*(\S.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_FLOAT_RE  = re.compile(r"-?\d+\.?\d*")
_PURE_NUM  = re.compile(r"^-?\d+(\.\d+)?$")
_CONFS     = np.linspace(0.5, 0.95, int((0.95 - 0.5) / 0.05) + 2)


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


def _extract_answer_tag(text: str) -> str | None:
    m = _ANSWER_RE.findall(text)
    return m[-1].strip() if m else None


def _mra(pred: float, gt: float) -> float:
    rel_err = abs(pred - gt) if gt == 0 else abs(pred - gt) / abs(gt)
    return float(np.mean(rel_err <= (1.0 - _CONFS)))


def _extract_float(text: str) -> float | None:
    m = _FLOAT_RE.search(text)
    return float(m.group()) if m else None


def _result_reward(resp: str, gt: str) -> float:
    # prefer content inside <answer> tag if present
    tagged = _extract_answer_tag(resp)
    answer_text = tagged if tagged is not None else resp

    if _PURE_NUM.match(gt):
        pred = _extract_float(answer_text)
        if pred is None:
            return 0.0
        return _mra(pred, float(gt))

    gt_l = gt.lower()
    ans_l = answer_text.lower()
    if gt_l == ans_l:
        return 1.0
    # partial credit: gt appears as a whole word in the answer
    # e.g. gt="left", answer="the left side" → 0.5
    if re.search(r"\b" + re.escape(gt_l) + r"\b", ans_l):
        return 0.5
    return 0.0


def format_reward(solution_str: str) -> float:
    """Public API: return only format reward [0, 1]"""
    return _format_reward(solution_str)


def result_reward(solution_str: str, ground_truth: str) -> float:
    """Public API: return only result reward [0, 1]"""
    resp = _strip_think(solution_str)
    gt = ground_truth.strip()
    return _result_reward(resp, gt)


def compute_score(solution_str: str, ground_truth: str) -> float:
    """Backward compatible: returns fused score"""
    fmt  = _format_reward(solution_str)
    resp = _strip_think(solution_str)
    gt   = ground_truth.strip()
    res  = _result_reward(resp, gt)
    return FORMAT_SCORE * fmt + (1.0 - FORMAT_SCORE) * res
