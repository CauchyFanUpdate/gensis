"""
Spatial consistency reward manager.

Verifies cross-view correspondence claims in the model's <think> block using
a multimodal VLM judge. Two judge modes are supported:

  judge_mode="localize" (default)
      Ask the judge "where is this point in image2?" and reward based on the
      distance between the model's claimed coordinate and the judge's answer:
          reward_per_pair = exp(-dist / sigma)

  judge_mode="yn"
      Draw a red dot at the model's claimed positions on both images, then ask
      the judge a Yes/No question: "Are these the same 3D point?"
      Reward per pair: 1 if Yes, 0 if No.
      Final consistency score = yes_count / total_pairs.

Final reward formula (both modes)
--------------------
  consistency enabled (consistency_weight > 0):
      format_weight * fmt + (1 - format_weight - consistency_weight) * res
      + consistency_weight * consistency

  disabled (consistency_weight == 0):
      format_weight * fmt + (1 - format_weight) * res
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import math
import os
import random
import re
import urllib.request
from collections import defaultdict
from typing import Any

import torch
from PIL import Image

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

_LOGGER = logging.getLogger(__name__)

# ── Regex patterns ────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)

# "located at [x, y] in image N" — captures x, y, image-number
_IMG_SRC_RE = re.compile(
    r"located at \[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\] in image[\s_]?(\d+)",
    re.IGNORECASE,
)

# "corresponds to [x, y] in image N" — captures x, y, image-number
_IMG_DST_RE = re.compile(
    r"corresponds? to \[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\] in image[\s_]?(\d+)",
    re.IGNORECASE,
)

# {"point_2d": [x, y]} in judge output
_POINT2D_RE = re.compile(
    r'"point_2d"\s*:\s*\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
)

# Yes/No question variants for judge_mode="yn"
_YN_QUESTIONS = [
    "Is the red point in image1 aligned with the point shown in image2?",
    "Are the points marked in image1 and image2 representing the same 3D position?",
    "Do the highlighted points in image1 and image2 indicate the same real-world spot?",
    "Does the point marked in image1 correspond to the same location as the one in image2?",
    "Is the marked point in image1 the same physical position as the marked point in image2?",
]

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_think(text: str) -> str:
    m = _THINK_RE.search(text)
    return m.group(1) if m else ""


def _parse_pairs(
    think_text: str,
) -> list[tuple[int, float, float, int, float, float]]:
    """
    Return (src_img_idx, x1, y1, dst_img_idx, x2, y2) for each
    (src, dst) combination found in each 'Viewpoint Consistency Analysis' section.
    Image indices are 1-based. One section may yield multiple pairs when an
    object is tracked across more than two views.
    """
    sections = re.split(r"Viewpoint Consistency Analysis \d+[:.]?", think_text, flags=re.IGNORECASE)
    pairs: list[tuple[int, float, float, int, float, float]] = []
    for sec in sections:
        m_src = _IMG_SRC_RE.search(sec)
        if m_src is None:
            continue
        src_idx = int(m_src.group(3))
        x1, y1  = float(m_src.group(1)), float(m_src.group(2))
        for m_dst in _IMG_DST_RE.finditer(sec):
            dst_idx = int(m_dst.group(3))
            x2, y2  = float(m_dst.group(1)), float(m_dst.group(2))
            pairs.append((src_idx, x1, y1, dst_idx, x2, y2))
    return pairs


def _pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pair_reward(mx2: float, my2: float, px: float, py: float, sigma: float) -> float:
    """exp(-normalised_dist / sigma), coordinates in [0, 1000]."""
    dist = math.sqrt(((mx2 - px) / 1000) ** 2 + ((my2 - py) / 1000) ** 2)
    return math.exp(-dist / sigma)


def _draw_red_dot(img: Image.Image, x: float, y: float, radius: int = 8) -> Image.Image:
    """Return a copy of img with a red circle at pixel (x, y). Coordinates in [0,1000]."""
    from PIL import ImageDraw
    w, h = img.size
    px = int(x / 1000 * w)
    py = int(y / 1000 * h)
    out = img.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    draw.ellipse([px - radius, py - radius, px + radius, py + radius], fill=(255, 0, 0))
    return out


# ── Manager ───────────────────────────────────────────────────────────────────

@register("spatial_consistency")
class SpatialConsistencyRewardManager(AbstractRewardManager):
    """Rule-based reward augmented with cross-view consistency from a VLM judge."""

    def __init__(
        self,
        tokenizer,
        num_examine: int,
        compute_score=None,
        reward_fn_key: str = "data_source",
        format_weight: float = 0.1,
        consistency_weight: float = 0.0,
        consistency_judge_base_url: str | None = None,
        consistency_judge_api_key: str = "qwen",
        consistency_judge_model_name: str = "qwen",
        consistency_judge_timeout: float = 120.0,
        consistency_judge_max_concurrent: int = 16,
        consistency_judge_max_tokens: int = 64,
        consistency_judge_sigma: float = 0.1,
        consistency_judge_enable_sleep: bool = True,
        consistency_max_pairs_per_sample: int = 4,
        judge_mode: str = "localize",
        **kwargs,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key

        self._format_weight = float(format_weight)
        self._consistency_weight = float(consistency_weight)
        self._sigma = float(consistency_judge_sigma)
        self._max_pairs = int(consistency_max_pairs_per_sample)
        self._max_concurrent = int(consistency_judge_max_concurrent)
        self._timeout = float(consistency_judge_timeout)
        self._enable_sleep = bool(consistency_judge_enable_sleep)
        self._judge_model_name = consistency_judge_model_name
        self._judge_max_tokens = int(consistency_judge_max_tokens)

        self._judge_base_url = (
            consistency_judge_base_url
            or os.environ.get("CONSISTENCY_JUDGE_BASE_URL", "http://localhost:51002/v1")
        )
        self._judge_api_key = consistency_judge_api_key

        if judge_mode not in ("localize", "yn"):
            raise ValueError(f"judge_mode must be 'localize' or 'yn', got {judge_mode!r}")
        self._judge_mode = judge_mode

        # Lazy-initialised OpenAI clients
        self._client = None
        self._async_client = None

        _LOGGER.info(
            "SpatialConsistencyRewardManager: consistency_weight=%.2f format_weight=%.2f "
            "sigma=%.3f max_concurrent=%d judge_url=%s judge_mode=%s",
            self._consistency_weight, self._format_weight,
            self._sigma, self._max_concurrent, self._judge_base_url, self._judge_mode,
        )

    # ── Client setup ──────────────────────────────────────────────────────────

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        from openai import AsyncOpenAI, OpenAI
        self._client = OpenAI(
            api_key=self._judge_api_key,
            base_url=self._judge_base_url,
            timeout=self._timeout,
        )
        self._async_client = AsyncOpenAI(
            api_key=self._judge_api_key,
            base_url=self._judge_base_url,
            timeout=self._timeout,
        )

    def _server_root(self) -> str:
        """http://host:port (strip /v1 suffix)."""
        url = self._judge_base_url.rstrip("/")
        return url[:-3] if url.endswith("/v1") else url

    def _wake(self) -> None:
        try:
            urllib.request.urlopen(f"{self._server_root()}/wake_up", timeout=10)
        except Exception:
            pass

    def _sleep(self) -> None:
        try:
            urllib.request.urlopen(f"{self._server_root()}/sleep?level=1", timeout=10)
        except Exception:
            pass

    # ── Async judge call ──────────────────────────────────────────────────────

    async def _call_one(
        self,
        img1: Image.Image,
        img2: Image.Image,
        x1: float,
        y1: float,
        semaphore: asyncio.Semaphore,
    ) -> tuple[float, float] | None:
        img1_b64 = _pil_to_b64(img1)
        img2_b64 = _pil_to_b64(img2)
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}},
                {"type": "text", "text": (
                    "I am providing you with two images of the same scene from different viewpoints. "
                    f"In image2, where is the corresponding point for the coordinate "
                    f"[{int(x1)},{int(y1)}] in image1?"
                )},
            ],
        }]
        async with semaphore:
            try:
                resp = await self._async_client.chat.completions.create(
                    model=self._judge_model_name,
                    messages=messages,
                    max_tokens=self._judge_max_tokens,
                    temperature=0.0,
                )
                text = (resp.choices[0].message.content or "").strip()
                m = _POINT2D_RE.search(text)
                if m:
                    return float(m.group(1)), float(m.group(2))
            except Exception as exc:
                _LOGGER.warning("Consistency judge call failed: %s", exc)
        return None

    async def _call_one_yn(
        self,
        img1_marked: Image.Image,
        img2_marked: Image.Image,
        semaphore: asyncio.Semaphore,
    ) -> bool | None:
        """Ask a random Yes/No question; return True=Yes, False=No, None=failure."""
        img1_b64 = _pil_to_b64(img1_marked)
        img2_b64 = _pil_to_b64(img2_marked)
        question = random.choice(_YN_QUESTIONS)
        prefix = (
            "I am providing you with two images of the same scene from different viewpoints. "
            "A red point is marked on the first image and another point is marked on the second image.\n\n"
        )
        messages = [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img1_b64}"}},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img2_b64}"}},
                {"type": "text", "text": prefix + question + " Answer with Yes or No."},
            ],
        }]
        async with semaphore:
            try:
                resp = await self._async_client.chat.completions.create(
                    model=self._judge_model_name,
                    messages=messages,
                    max_tokens=self._judge_max_tokens,
                    temperature=0.0,
                )
                text = (resp.choices[0].message.content or "").strip()
                if _YES_RE.search(text):
                    return True
                if _NO_RE.search(text):
                    return False
            except Exception as exc:
                _LOGGER.warning("YN consistency judge call failed: %s", exc)
        return None

    async def _batch_async(
        self,
        requests: list[tuple[Image.Image, Image.Image, float, float]],
    ) -> list[tuple[float, float] | None] | list[bool | None]:
        if not requests:
            return []
        self._ensure_client()
        if self._enable_sleep:
            self._wake()
        semaphore = asyncio.Semaphore(self._max_concurrent)
        if self._judge_mode == "yn":
            # Draw red dots then ask Yes/No
            coros = []
            for img1, img2, x1, y1, x2, y2 in requests:  # type: ignore[misc]
                img1_marked = _draw_red_dot(img1, x1, y1)
                img2_marked = _draw_red_dot(img2, x2, y2)
                coros.append(self._call_one_yn(img1_marked, img2_marked, semaphore))
            results = await asyncio.gather(*coros)
        else:
            results = await asyncio.gather(*[
                self._call_one(img1, img2, x1, y1, semaphore)
                for img1, img2, x1, y1 in requests  # type: ignore[misc]
            ])
        if self._enable_sleep:
            self._sleep()
        return list(results)

    def _batch_judge(
        self,
        requests: list,
    ) -> list:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._batch_async(requests))
        finally:
            loop.close()

    # ── Main call ─────────────────────────────────────────────────────────────

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info: dict[str, list] = defaultdict(list)
        already_print: dict[str, int] = {}

        use_consistency = self._consistency_weight > 0.0

        # ── Pass 1: decode, rule score, parse pairs ──────────────────────────
        sample_meta: list[dict] = []
        # Flat judge-request list.
        # localize mode: (img1, img2, x1, y1)         — judge returns (px, py)
        # yn mode:       (img1, img2, x1, y1, x2, y2) — judge returns bool
        judge_requests: list = []
        # model_x2/y2 per request (localize mode only)
        model_claimed: list[tuple[float, float]] = []

        for i in range(len(data)):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_len = int(data_item.batch["attention_mask"][:prompt_length].sum())
            valid_prompt_ids = prompt_ids[-valid_prompt_len:] if valid_prompt_len > 0 else prompt_ids[:0]

            response_ids = data_item.batch["responses"]
            valid_resp_len = int(data_item.batch["attention_mask"][prompt_length:].sum())
            valid_response_ids = response_ids[:valid_resp_len]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})

            # Rule-based score — try to get fmt/result separately
            fmt_reward = res_reward = 0.0
            try:
                from verl.utils.reward_score import spatial as _spatial
                fmt_reward = _spatial.format_reward(response_str)
                res_reward = _spatial.result_reward(response_str, ground_truth)
            except Exception:
                score = self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                )
                fmt_reward = res_reward = float(score) if not isinstance(score, dict) else float(score.get("score", 0.0))

            # Parse think-block pairs
            pair_start = len(judge_requests)
            if use_consistency:
                think = _extract_think(response_str)
                pairs = _parse_pairs(think)
                if len(pairs) > self._max_pairs:
                    pairs = random.sample(pairs, self._max_pairs)

                images = data_item.non_tensor_batch.get("images")
                n_images = len(images) if images is not None else 0

                for src_idx, x1, y1, dst_idx, mx2, my2 in pairs:
                    # Convert 1-based image number to 0-based list index
                    i_src = src_idx - 1
                    i_dst = dst_idx - 1
                    if i_src == i_dst:
                        continue  # same image, no cross-view correspondence
                    if i_src < 0 or i_src >= n_images or i_dst < 0 or i_dst >= n_images:
                        continue
                    img_src = images[i_src] if isinstance(images[i_src], Image.Image) else None
                    img_dst = images[i_dst] if isinstance(images[i_dst], Image.Image) else None
                    if img_src is None or img_dst is None:
                        continue
                    if self._judge_mode == "yn":
                        judge_requests.append((img_src, img_dst, x1, y1, mx2, my2))
                    else:
                        judge_requests.append((img_src, img_dst, x1, y1))
                        model_claimed.append((mx2, my2))

            sample_meta.append({
                "fmt_reward": fmt_reward,
                "res_reward": res_reward,
                "valid_resp_len": valid_resp_len,
                "data_source": data_source,
                "prompt_str": prompt_str,
                "response_str": response_str,
                "ground_truth": ground_truth,
                "pair_start": pair_start,
                "pair_end": len(judge_requests),
            })

        # ── Batch judge call ─────────────────────────────────────────────────
        judge_results: list[tuple[float, float] | None] = []
        if use_consistency and judge_requests:
            _LOGGER.info("Calling consistency judge for %d pairs", len(judge_requests))
            judge_results = self._batch_judge(judge_requests)

        # ── Pass 2: compute final rewards ────────────────────────────────────
        for i, meta in enumerate(sample_meta):
            fmt = meta["fmt_reward"]
            res = meta["res_reward"]

            consistency = 0.0
            if use_consistency:
                pair_scores: list[float] = []
                for req_idx in range(meta["pair_start"], meta["pair_end"]):
                    result = judge_results[req_idx] if req_idx < len(judge_results) else None
                    if result is None:
                        continue
                    if self._judge_mode == "yn":
                        pair_scores.append(1.0 if result else 0.0)
                    else:
                        mx2, my2 = model_claimed[req_idx]
                        px, py = result
                        pair_scores.append(_pair_reward(mx2, my2, px, py, self._sigma))
                if pair_scores:
                    consistency = sum(pair_scores) / len(pair_scores)

            if use_consistency:
                result_w = 1.0 - self._format_weight - self._consistency_weight
                final = (
                    self._format_weight * fmt
                    + result_w * res
                    + self._consistency_weight * consistency
                )
            else:
                final = self._format_weight * fmt + (1.0 - self._format_weight) * res

            vrl = max(0, meta["valid_resp_len"] - 1)
            reward_tensor[i, vrl] = final

            reward_extra_info["format_reward"].append(fmt)
            reward_extra_info["result_reward"].append(res)
            reward_extra_info["consistency_reward"].append(consistency)
            reward_extra_info["reward"].append(final)

            ds = meta["data_source"]
            already_print.setdefault(ds, 0)
            if already_print[ds] < self.num_examine:
                already_print[ds] += 1
                print("[prompt]", meta["prompt_str"])
                print("[response]", meta["response_str"])
                print("[ground_truth]", meta["ground_truth"])
                print("[format_reward]", fmt)
                print("[result_reward]", res)
                print("[consistency_reward]", consistency)
                print("[reward]", final)

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
