#!/usr/bin/env python3
"""
方案一 — Stage 1.5: MC 选项修正

对 stage1_samples.jsonl 中 quality_pass=True 的 multiple-choice 记录：
  - 若 actual_new_answer 已在原选项中 → 仅更新 answer_letter，原选项不变
  - 若 actual_new_answer 不在原选项中 → 随机顶掉一个干扰项（不顶掉原正确答案），重新打乱，重新分配字母
  - open-ended 记录 → 原样透传

支持方案一的 sample_key 格式：{id}::e{N}::s{K}

输出：_archive/output_fangan1/stage1p5_samples.jsonl
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

_LETTERS = "ABCDEFGHIJ"


def _letters_for(n: int) -> list[str]:
    return list(_LETTERS[:n])


def _norm(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"[^\w\s./-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if re.fullmatch(r"\d+\.0+", s):
        s = s.split(".")[0]
    return s


def _answer_in_options(answer: str, options: list[str]) -> int | None:
    norm_ans = _norm(answer)
    norm_opts = [_norm(o) for o in options]

    for i, no in enumerate(norm_opts):
        if norm_ans == no:
            return i

    if len(norm_ans) == 1 and norm_ans.isalpha():
        idx = ord(norm_ans) - ord("a")
        if 0 <= idx < len(options):
            return idx

    return None


def _fix_options(record: dict, rng: random.Random) -> dict:
    r = dict(record)
    meta = dict(r.get("mmmu_meta") or {})
    r["mmmu_meta"] = meta

    if meta.get("question_type") != "multiple-choice":
        return r

    options: list[str] = list(meta.get("options") or [])
    actual: str = r.get("actual_new_answer") or ""
    orig_answer_letter: str = meta.get("answer_letter") or ""

    if not options or not actual:
        return r

    matched_idx = _answer_in_options(actual, options)

    if matched_idx is not None:
        new_letter = _LETTERS[matched_idx] if matched_idx < len(_LETTERS) else "A"
        meta["answer_letter"] = new_letter
        r["options_fixed"] = False
        return r

    orig_letter_norm = orig_answer_letter.strip().upper()
    orig_idx: int | None = None
    if orig_letter_norm and orig_letter_norm in _LETTERS:
        orig_idx = _LETTERS.index(orig_letter_norm)
        if orig_idx >= len(options):
            orig_idx = None

    candidate_positions = [i for i in range(len(options)) if i != orig_idx]
    if not candidate_positions:
        options.append(actual)
        replace_idx = len(options) - 1
    else:
        replace_idx = rng.choice(candidate_positions)
        options[replace_idx] = actual

    rng.shuffle(options)

    new_matched_idx = _answer_in_options(actual, options)
    if new_matched_idx is None:
        for i, o in enumerate(options):
            if o == actual:
                new_matched_idx = i
                break
    if new_matched_idx is None:
        new_matched_idx = 0

    new_letter = _LETTERS[new_matched_idx] if new_matched_idx < len(_LETTERS) else "A"
    meta["options"] = options
    meta["answer_letter"] = new_letter
    r["options_fixed"] = True
    return r


def main():
    parser = argparse.ArgumentParser(
        description="方案一 Stage 1.5: fix options so actual_new_answer is always in the option list"
    )
    repo_root = Path(__file__).resolve().parent
    default_stage1 = repo_root / "_archive/output_fangan1/stage1_samples.jsonl"
    default_output = repo_root / "_archive/output_fangan1/stage1p5_samples.jsonl"

    parser.add_argument("--stage1_jsonl", default=str(default_stage1))
    parser.add_argument("--output_jsonl", default=str(default_output))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_resume", action="store_true")
    args = parser.parse_args()

    stage1_path = Path(args.stage1_jsonl)
    output_path = Path(args.output_jsonl)

    if not stage1_path.exists():
        print(f"[stage1.5] ERROR: stage1 jsonl not found: {stage1_path}", file=sys.stderr)
        sys.exit(1)

    if output_path.exists() and not args.no_resume:
        print(f"[stage1.5] 输出文件已存在，跳过（传 --no_resume 强制重新生成）：{output_path}")
        sys.exit(0)

    rng = random.Random(args.seed)

    all_records: list[dict] = []
    with stage1_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                all_records.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[stage1.5] WARNING: JSON parse error, skip line: {e}", file=sys.stderr)

    print(f"[stage1.5] 读取记录总数: {len(all_records)}")

    n_passed_mc = n_passed_open = n_not_passed = 0
    n_already_correct = n_fixed = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for record in all_records:
            if not record.get("quality_pass"):
                n_not_passed += 1
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            qt = (record.get("mmmu_meta") or {}).get("question_type", "")
            if qt == "multiple-choice":
                n_passed_mc += 1
            else:
                n_passed_open += 1

            fixed = _fix_options(record, rng)

            if fixed.get("options_fixed") is True:
                n_fixed += 1
            elif fixed.get("options_fixed") is False:
                n_already_correct += 1

            fout.write(json.dumps(fixed, ensure_ascii=False) + "\n")

    n_passed = n_passed_mc + n_passed_open
    print(f"[stage1.5] quality_pass 记录: {n_passed} （MC={n_passed_mc}, open-ended={n_passed_open}）")
    print(f"[stage1.5] 非 passed 记录（透传）: {n_not_passed}")
    print(f"[stage1.5] MC 中答案已在选项内（只更新字母）: {n_already_correct}")
    print(f"[stage1.5] MC 中替换了干扰项: {n_fixed}")
    print(f"[stage1.5] 输出: {output_path}")


if __name__ == "__main__":
    main()
