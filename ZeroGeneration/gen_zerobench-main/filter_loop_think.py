#!/usr/bin/env python3
"""
筛除 JSONL 中 <think> 内容陷入死循环的条目。
策略：先用启发式规则快速判断，不确定时调用 LLM API 裁决。
输出文件命名为 clear_<原文件名>，放在同目录下。

用法：
    python filter_loop_think.py <path_to.jsonl>
"""

import json
import re
import sys
import concurrent.futures
from collections import Counter
from pathlib import Path

import requests

# ── 配置 ──────────────────────────────────────────────────────────────────────
API_KEY         = "YOUR_API_KEY"
API_URL         = "https://runway.devops.xiaohongshu.com/openai/qwen/v1/chat/completions"
MODEL           = "qwen3-vl-plus"  # 与 API key 对应的模型
MAX_WORKERS     = 16             # 并发线程数（API 请求）
REPEAT_THRESHOLD = 5             # 同一句话重复 ≥ 此次数 → 直接判定为循环
MIN_THINK_LEN   = 100            # think 内容短于此长度则跳过检查
NGRAM_SIZE      = 10             # 滑窗 n-gram 长度
NGRAM_REPEAT    = 3              # n-gram 重复 ≥ 此次数 → 判定为循环


# ── 工具函数 ──────────────────────────────────────────────────────────────────

def extract_think(text: str) -> str | None:
    """提取 <think>...</think> 内的内容，容忍未闭合标签。"""
    m = re.search(r"<think>(.*?)(?:</think>|<think>|$)", text, re.DOTALL)
    return m.group(1).strip() if m else None


def heuristic_loop(think: str) -> bool | None:
    """
    启发式循环检测。
    返回 True  → 确定循环
    返回 False → 确定正常（当前逻辑暂未使用此分支）
    返回 None  → 不确定，需 API 裁决
    """
    # 规则 1：按句号/问号/叹号分句，统计重复
    sentences = [
        s.strip()
        for s in re.split(r"[.!?。！？]\s+", think)
        if len(s.strip()) > 20
    ]
    if sentences and max(Counter(sentences).values()) >= REPEAT_THRESHOLD:
        return True

    # 规则 2：滑窗 n-gram 重复
    words = think.split()
    if len(words) > NGRAM_SIZE * 2:
        ngrams = [
            " ".join(words[i : i + NGRAM_SIZE])
            for i in range(len(words) - NGRAM_SIZE + 1)
        ]
        if max(Counter(ngrams).values()) >= NGRAM_REPEAT:
            return True

    return None  # 交给 API 判断


def api_loop(think: str) -> bool:
    """调用 LLM API 判断 think 内容是否为无效循环，失败时默认保留（返回 False）。"""
    prompt = (
        "请判断下面的「思考过程」是否陷入了无效死循环"
        "（相同或极相似的句子/段落反复出现，推理没有实质进展）。\n"
        "只回答「是」或「否」，不要有任何解释。\n\n"
        f"思考内容：\n---\n{think[:4000]}\n---"
    )
    try:
        resp = requests.post(
            API_URL,
            headers={"api-key": API_KEY, "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 10,
            },
            timeout=30,
        )
        resp.raise_for_status()
        answer = resp.json()["choices"][0]["message"]["content"].strip()
        return "是" in answer
    except Exception as exc:
        print(f"    [API 失败，保留此条目] {exc}")
        return False


def check_entry(args: tuple) -> tuple:
    """处理单条 JSONL 行，返回 (序号, 原始行, 是否循环, 判断方式)。"""
    idx, total, line = args

    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return idx, line, False, "json_error"

    # 兼容两种格式：
    #   conversations[].from=="gpt"  -> value
    #   messages[].role=="assistant" -> content
    gpt_val = next(
        (c.get("value", "") for c in data.get("conversations", []) if c.get("from") == "gpt"),
        None,
    )
    if gpt_val is None:
        gpt_val = next(
            (c.get("content", "") for c in data.get("messages", []) if c.get("role") == "assistant"),
            None,
        )
    if not gpt_val:
        return idx, line, False, "no_gpt"

    think = extract_think(gpt_val)
    if not think or len(think) < MIN_THINK_LEN:
        return idx, line, False, "no_think"

    h = heuristic_loop(think)
    if h is not None:
        return idx, line, h, "heuristic"

    looping = api_loop(think)
    return idx, line, looping, "api"


# ── 主流程 ────────────────────────────────────────────────────────────────────

def filter_jsonl(input_path: str) -> None:
    src = Path(input_path).resolve()
    if not src.exists():
        print(f"文件不存在: {src}")
        sys.exit(1)

    dst = src.parent / f"clear_{src.name}"
    print(f"输入: {src}")
    print(f"输出: {dst}")
    print(f"模型: {MODEL}  并发: {MAX_WORKERS}\n")

    raw_lines = src.read_text("utf-8").splitlines()
    lines = [l.strip() for l in raw_lines if l.strip()]
    total = len(lines)
    tasks = [(i + 1, total, l) for i, l in enumerate(lines)]

    # 按原始顺序收集结果
    results: list[tuple[str, bool, str] | None] = [None] * total

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        future_to_pos = {pool.submit(check_entry, t): t[0] - 1 for t in tasks}
        for fut in concurrent.futures.as_completed(future_to_pos):
            pos = future_to_pos[fut]
            idx, line, loop, method = fut.result()
            results[pos] = (line, loop, method)
            status = "FILTERED" if loop else "kept   "
            print(f"[{idx:>5}/{total}] {status}  via={method}")

    kept = filtered = 0
    with dst.open("w", encoding="utf-8") as f:
        for line, loop, _ in results:  # type: ignore[misc]
            if not loop:
                f.write(line + "\n")
                kept += 1
            else:
                filtered += 1

    print(f"\n完成  总计={total}  保留={kept}  筛除={filtered}")
    print(f"已写入: {dst}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python filter_loop_think.py <path_to.jsonl>")
        sys.exit(1)
    filter_jsonl(sys.argv[1])
