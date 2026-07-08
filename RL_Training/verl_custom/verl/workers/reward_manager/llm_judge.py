"""LLM judge reward manager using vLLM backend for scoring model responses."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Any

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager

_LLM_JUDGE_REWARD_TYPES = {"llm_judge", "instruction_following_llm_judge"}
_ONLINE_JUDGE_BACKENDS = {"openai", "openai_api", "openai_server", "vllm_server", "server"}
_LOCAL_JUDGE_BACKENDS = {"vllm_engine", "vllm", "local_vllm", "engine"}
_LOGGER = logging.getLogger(__name__)


@register("llm_judge")
class LLMJudgeRewardManager(AbstractRewardManager):
    """Reward manager with vLLM-based LLM judge for scoring."""

    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        llm_judge_kwargs: dict | None = None,
        format_weight: float | None = None,
        rubric_weight: float | None = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = max_resp_len

        self.llm_judge_kwargs = llm_judge_kwargs or {}
        self._client = None
        self._async_client = None
        self._sampling_params: dict[str, Any] | None = None
        self._vllm_engine = None
        self._vllm_sampling_params = None
        self._vllm_tokenizer = None

        self._judge_model_name = self.llm_judge_kwargs.get("model_name", "qwen_vllm")
        self._judge_model_path = self.llm_judge_kwargs.get("model_path") or os.environ.get("VLLM_JUDGE_MODEL_PATH")

        raw_backend = self.llm_judge_kwargs.get("backend")
        if raw_backend is None and bool(self.llm_judge_kwargs.get("use_vllm_engine", False)):
            raw_backend = "vllm_engine"
        if raw_backend is None:
            raw_backend = "openai_server"
        self._judge_backend = str(raw_backend).strip().lower()
        if self._judge_backend in _ONLINE_JUDGE_BACKENDS:
            self._judge_backend = "openai_server"
        elif self._judge_backend in _LOCAL_JUDGE_BACKENDS:
            self._judge_backend = "vllm_engine"
        else:
            raise ValueError(
                f"Unsupported llm_judge_kwargs.backend={raw_backend!r}. "
                f"Supported: {sorted(_ONLINE_JUDGE_BACKENDS | _LOCAL_JUDGE_BACKENDS)}"
            )

        self._judge_api_key = self.llm_judge_kwargs.get("api_key") or os.environ.get("VLLM_API_KEY", "qwen")
        self._judge_base_url = (
            self.llm_judge_kwargs.get("base_url") or os.environ.get("VLLM_JUDGE_BASE_URL")
        )
        if self._judge_base_url is None:
            port = (
                self.llm_judge_kwargs.get("port")
                or os.environ.get("VLLM_JUDGE_PORT")
                or os.environ.get("VLLM_PORT")
                or "8000"
            )
            self._judge_base_url = f"http://localhost:{port}/v1"
        self._judge_timeout = float(self.llm_judge_kwargs.get("timeout", 120))

        self._judge_prompt_path = self.llm_judge_kwargs.get("prompt_path")
        self._judge_prompt_template = self.llm_judge_kwargs.get("prompt_template")

        # Load rubric-specific prompt if provided
        self._rubric_prompt_path = self.llm_judge_kwargs.get("rubric_prompt_path")
        self._rubric_prompt_template = None
        if self._rubric_prompt_path:
            if not os.path.exists(self._rubric_prompt_path):
                raise FileNotFoundError(f"rubric prompt not found: {self._rubric_prompt_path}")
            with open(self._rubric_prompt_path, "r", encoding="utf-8") as f:
                self._rubric_prompt_template = f.read()

        # rubric_mode: "general" uses rubric_prompt_template for all samples;
        # "sample" uses rubric_sample_prompt_path with per-sample {rubric} substitution.
        self._rubric_mode = str(self.llm_judge_kwargs.get("rubric_mode", "general")).lower()
        self._sample_rubric_prompt_template = None
        if self._rubric_mode == "sample":
            sample_path = self.llm_judge_kwargs.get("rubric_sample_prompt_path")
            if not sample_path:
                raise ValueError("rubric_mode=sample requires llm_judge_kwargs.rubric_sample_prompt_path")
            if not os.path.exists(sample_path):
                raise FileNotFoundError(f"rubric_sample_prompt not found: {sample_path}")
            with open(sample_path, "r", encoding="utf-8") as f:
                self._sample_rubric_prompt_template = f.read()

        # External rubric lookup: keyed by (first_image_path, question_text).
        # Training parquet has no stable id field; composite key is the only reliable match.
        self._rubric_lookup: dict[tuple[str, str], Any] = {}
        rubric_data_path = self.llm_judge_kwargs.get("rubric_data_path")
        if rubric_data_path:
            if not os.path.exists(rubric_data_path):
                raise FileNotFoundError(f"rubric_data_path not found: {rubric_data_path}")
            loaded, skipped = 0, 0
            with open(rubric_data_path, "r", encoding="utf-8") as _f:
                for _line in _f:
                    _line = _line.strip()
                    if not _line:
                        continue
                    try:
                        _rec = json.loads(_line)
                        _rubric = _rec.get("rubric")
                        if _rubric is None:
                            skipped += 1
                            continue
                        _imgs = _rec.get("images", [])
                        _img_path = _imgs[0].get("path", "") if _imgs else ""
                        _question = ""
                        for _turn in _rec.get("prompt", []):
                            if isinstance(_turn, dict) and _turn.get("role") == "user":
                                _c = _turn.get("content", "")
                                if isinstance(_c, str):
                                    _question = _c.strip()
                                break
                        _key = (_img_path, _question)
                        if _key not in self._rubric_lookup:
                            self._rubric_lookup[_key] = _rubric
                            loaded += 1
                        else:
                            skipped += 1
                    except Exception:
                        skipped += 1
            _LOGGER.info(
                "Loaded rubric_data_path=%s: %d entries, %d skipped/dup",
                rubric_data_path, loaded, skipped,
            )

        # Load default prompt for backward compatibility
        if self._judge_prompt_template is None:
            if self._judge_prompt_path is None:
                raise ValueError("llm_judge_kwargs.prompt_path or prompt_template must be provided")
            if not os.path.exists(self._judge_prompt_path):
                raise FileNotFoundError(f"LLM judge prompt not found: {self._judge_prompt_path}")
            with open(self._judge_prompt_path, "r", encoding="utf-8") as f:
                self._judge_prompt_template = f.read()

        self._judge_score_scale = self.llm_judge_kwargs.get("score_scale", "minus1_to_1")
        self._judge_enable_sleep = bool(self.llm_judge_kwargs.get("enable_sleep_mode", True))
        self._judge_sleep_level = int(self.llm_judge_kwargs.get("sleep_level", 1))

        if format_weight is None:
            raise ValueError("format_weight must be provided")
        try:
            self._format_weight = float(format_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid format_weight={format_weight!r}") from exc
        if not 0.0 <= self._format_weight <= 1.0:
            raise ValueError(f"format_weight must be in [0, 1], got {self._format_weight}")

        if rubric_weight is None:
            raise ValueError("rubric_weight must be provided")
        try:
            self._rubric_weight = float(rubric_weight)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid rubric_weight={rubric_weight!r}") from exc
        if not 0.0 <= self._rubric_weight <= 1.0:
            raise ValueError(f"rubric_weight must be in [0, 1], got {self._rubric_weight}")

        _LOGGER.info(
            "Initialized LLM judge backend=%s model=%s base_url=%s format_weight=%.2f rubric_weight=%.2f",
            self._judge_backend,
            self._judge_model_name,
            self._judge_base_url,
            self._format_weight,
            self._rubric_weight,
        )

    def _ensure_client(self) -> None:
        if self._judge_backend != "openai_server":
            return
        if self._client is not None:
            return
        try:
            from openai import AsyncOpenAI, OpenAI
        except Exception:
            try:
                from openai import OpenAI
            except Exception as exc:
                raise RuntimeError("openai python client is required") from exc
            AsyncOpenAI = None


        _LOGGER.info("Initializing OpenAI client for LLM judge")
        self._client = OpenAI(
            api_key=self._judge_api_key, base_url=self._judge_base_url, timeout=self._judge_timeout
        )
        if AsyncOpenAI is not None:
            self._async_client = AsyncOpenAI(
                api_key=self._judge_api_key, base_url=self._judge_base_url, timeout=self._judge_timeout
            )
        self._sampling_params = {
            "temperature": float(self.llm_judge_kwargs.get("temperature", 0.7)),
            "top_p": float(self.llm_judge_kwargs.get("top_p", 0.8)),
            "max_tokens": int(self.llm_judge_kwargs.get("max_tokens", 1024)),
            "presence_penalty": float(self.llm_judge_kwargs.get("presence_penalty", 1.5)),
            "frequency_penalty": float(self.llm_judge_kwargs.get("frequency_penalty", 1.0)),
        }

    def _ensure_vllm_engine(self) -> None:
        if self._judge_backend != "vllm_engine":
            return
        if self._vllm_engine is not None:
            return

        if not self._judge_model_path:
            raise ValueError("model_path must be set when backend=vllm_engine")

        try:
            from vllm import LLM, SamplingParams
        except Exception as exc:
            raise RuntimeError("vllm package is required when backend=vllm_engine") from exc

        tensor_parallel_size = int(
            self.llm_judge_kwargs.get("tensor_parallel_size", os.environ.get("VLLM_JUDGE_TENSOR_PARALLEL_SIZE", "1"))
        )
        dtype = self.llm_judge_kwargs.get("dtype", os.environ.get("VLLM_JUDGE_DTYPE", "bfloat16"))
        gpu_memory_utilization = float(
            self.llm_judge_kwargs.get("gpu_memory_utilization", os.environ.get("VLLM_JUDGE_GPU_MEMORY_UTILIZATION", "0.9"))
        )

        llm_kwargs = {
            "model": self._judge_model_path,
            "tensor_parallel_size": tensor_parallel_size,
            "dtype": dtype,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": bool(self.llm_judge_kwargs.get("trust_remote_code", False)),
        }
        if self._judge_enable_sleep:
            llm_kwargs["enable_sleep_mode"] = True

        _LOGGER.info("Initializing local vLLM engine kwargs=%s", llm_kwargs)
        self._vllm_engine = LLM(**llm_kwargs)
        self._vllm_tokenizer = self._vllm_engine.get_tokenizer()

        sampling_kwargs: dict[str, Any] = {
            "temperature": float(self.llm_judge_kwargs.get("temperature", 0.0)),
            "top_p": float(self.llm_judge_kwargs.get("top_p", 1.0)),
            "max_tokens": int(self.llm_judge_kwargs.get("max_tokens", 2048)),
        }
        self._vllm_sampling_params = SamplingParams(**sampling_kwargs)

    def _build_local_chat_prompt(self, prompt: str) -> str:
        if self._vllm_tokenizer is None:
            raise RuntimeError("vLLM tokenizer not initialized")
        messages = [{"role": "user", "content": prompt}]
        return self._vllm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _judge_batch_local(self, prompts: list[str]) -> tuple[list[float], list[str]]:
        self._ensure_vllm_engine()
        if not prompts:
            return [], []
        assert self._vllm_engine is not None

        chat_prompts = [self._build_local_chat_prompt(p) for p in prompts]
        _LOGGER.info("LLM judge generating for %d prompts", len(prompts))

        try:
            if self._judge_enable_sleep:
                self._vllm_engine.wake_up()
            outputs = self._vllm_engine.generate(chat_prompts, sampling_params=self._vllm_sampling_params, use_tqdm=False)
        finally:
            if self._judge_enable_sleep:
                self._vllm_engine.sleep(level=self._judge_sleep_level)

        scaled_scores = []
        texts = []
        for output in outputs:
            text = ""
            if output.outputs:
                text = (output.outputs[0].text or "").strip()
            raw = self._extract_score(text)
            scaled = self._scale_score(raw)
            scaled_scores.append(scaled)
            texts.append(text)

        return scaled_scores, texts

    async def _judge_one(self, prompt: str, semaphore: asyncio.Semaphore) -> tuple[float, str]:
        async with semaphore:
            request_messages = [{"role": "user", "content": prompt}]
            try:
                if self._async_client is not None:
                    response = await self._async_client.chat.completions.create(
                        model=self._judge_model_name,
                        messages=request_messages,
                        **self._sampling_params,
                    )
                else:
                    response = await asyncio.to_thread(
                        self._client.chat.completions.create,
                        model=self._judge_model_name,
                        messages=request_messages,
                        **self._sampling_params,
                    )
            except Exception as exc:
                _LOGGER.warning("LLM judge API error: %s", exc, exc_info=True)
                return 0.0, ""

        text = response.choices[0].message.content or ""
        raw_score = self._extract_score(text)
        scaled_score = self._scale_score(raw_score)
        return scaled_score, text

    async def _judge_batch_async(self, prompts: list[str]) -> tuple[list[float], list[str]]:
        self._ensure_client()
        _LOGGER.info("LLM judge generating for %d prompts via API", len(prompts))

        max_concurrent = int(self.llm_judge_kwargs.get("max_concurrent_requests", 8))
        semaphore = asyncio.Semaphore(max(1, max_concurrent))

        results = await asyncio.gather(*(self._judge_one(p, semaphore) for p in prompts))
        scaled_scores, texts = zip(*results) if results else ([], [])
        return list(scaled_scores), list(texts)

    def _judge_batch(self, prompts: list[str]) -> tuple[list[float], list[str]]:
        if self._judge_backend == "vllm_engine":
            return self._judge_batch_local(prompts)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._judge_batch_async(prompts))
        finally:
            loop.close()

    def _get_format_reward(self, solution_str: str, data_source: str) -> float:
        """Get format reward from the corresponding reward module."""
        try:
            if data_source in ["crosspoint", "mindcube", "spar"]:
                from verl.utils.reward_score import spatial
                return spatial.format_reward(solution_str)
            elif data_source in ["zerobench"]:
                from verl.utils.reward_score import zerobench
                return zerobench.format_reward(solution_str)
            else:
                return 0.0
        except Exception as e:
            _LOGGER.warning("Failed to get format reward for data_source=%s: %s", data_source, e)
            return 0.0

    def _should_use_llm_for_zerobench(self, ground_truth: str) -> bool:
        """Check if ground truth answer is simple (can use rules) or complex (needs LLM)."""
        try:
            gt = ground_truth.strip()

            # MC choice: "A.", "B.", etc.
            if re.match(r"^[A-F]\.", gt, re.IGNORECASE):
                return False

            # Pure number: "3.14", "-5", etc.
            if re.match(r"^-?\d+(\.\d+)?$", gt):
                return False

            # Simple text: short (< 50 chars), no special structure
            if len(gt) < 10 and not any(c in gt for c in ["<", ">", "{", "}"]):
                return False

            # Complex answer: use LLM judge
            return True
        except Exception as e:
            _LOGGER.warning("Error checking if zerobench needs LLM: %s", e)
            return True

    def _get_zerobench_result_reward_with_llm(self, solution_str: str, ground_truth: str, prompt_str: str) -> float:
        """For complex zerobench answers, use LLM to calculate result reward."""
        try:
            judge_prompt = self._judge_prompt_template
            judge_prompt = judge_prompt.replace("{question}", str(prompt_str))
            judge_prompt = judge_prompt.replace("{reference_answer}", str(ground_truth))
            judge_prompt = judge_prompt.replace("{model_answer}", str(solution_str))

            scaled_scores, _ = self._judge_batch([judge_prompt])
            if scaled_scores:
                return scaled_scores[0]
            return 0.0
        except Exception as e:
            _LOGGER.warning("Failed to get zerobench result reward with LLM: %s", e)
            return 0.0

    async def _get_zerobench_result_reward_with_llm_async(self, solution_str: str, ground_truth: str, prompt_str: str) -> float:
        """Async version for use inside a running event loop (avoids nested loop conflict)."""
        try:
            judge_prompt = self._judge_prompt_template
            judge_prompt = judge_prompt.replace("{question}", str(prompt_str))
            judge_prompt = judge_prompt.replace("{reference_answer}", str(ground_truth))
            judge_prompt = judge_prompt.replace("{model_answer}", str(solution_str))

            self._ensure_client()
            scaled_scores, _ = await self._judge_batch_async([judge_prompt])
            if scaled_scores:
                return scaled_scores[0]
            return 0.0
        except Exception as e:
            _LOGGER.warning("Failed to get zerobench result reward with LLM (async): %s", e)
            return 0.0

    def _get_result_reward(self, solution_str: str, ground_truth: str, data_source: str, prompt_str: str = "") -> float:
        """Get result reward from the corresponding reward module.

        For zerobench: uses rules for simple answers, LLM for complex ones.
        For spatial/crosspoint/etc: always uses rules.
        """
        try:
            if data_source in ["crosspoint", "mindcube", "spar"]:
                from verl.utils.reward_score import spatial
                return spatial.result_reward(solution_str, ground_truth)
            elif data_source in ["zerobench"]:
                # For zerobench, check if answer is simple or complex
                if not self._should_use_llm_for_zerobench(ground_truth):
                    # Simple answer: use rule-based
                    from verl.utils.reward_score import zerobench
                    return zerobench.result_reward(solution_str, ground_truth)
                else:
                    # Complex answer: use LLM judge
                    return self._get_zerobench_result_reward_with_llm(solution_str, ground_truth, prompt_str)
            else:
                return 0.0
        except Exception as e:
            _LOGGER.warning("Failed to get result reward for data_source=%s: %s", data_source, e)
            return 0.0

    def _format_rubric(self, rubric) -> str:
        """Convert a rubric dict (or JSON string) to human-readable text for the prompt."""
        if isinstance(rubric, str):
            try:
                rubric = json.loads(rubric)
            except Exception:
                return rubric
        if not isinstance(rubric, dict):
            return str(rubric)
        parts = []
        for key, criteria in rubric.items():
            label = key.replace("_", " ").title()
            parts.append(f"{label}:")
            for i, c in enumerate(criteria, 1):
                desc = c.get("description", "")
                weight = c.get("weight", "")
                parts.append(f"  {i}. {desc} (weight: {weight})")
        return "\n".join(parts)

    def _maybe_truncate_output(self, template_with_placeholder: str, output: str) -> str:
        """Truncate output tokens so that input_tokens + max_tokens <= judge_max_context_len."""
        max_tokens = int(self.llm_judge_kwargs.get("max_tokens", 1024))
        max_context_len = int(self.llm_judge_kwargs.get("judge_max_context_len", 12288))
        # Reserve extra tokens for chat template overhead (role tags etc. added by vLLM server)
        chat_overhead = int(self.llm_judge_kwargs.get("judge_chat_overhead", 64))
        budget = max_context_len - max_tokens - chat_overhead

        base = template_with_placeholder.replace("{output}", "")
        base_count = len(self.tokenizer.encode(base, add_special_tokens=False))
        output_budget = budget - base_count
        if output_budget <= 0:
            return ""

        output_ids = self.tokenizer.encode(output, add_special_tokens=False)
        if len(output_ids) <= output_budget:
            return output

        _LOGGER.warning(
            "Judge output truncated: %d → %d tokens (context_limit=%d, max_tokens=%d, base=%d)",
            len(output_ids), output_budget, max_context_len, max_tokens, base_count,
        )
        return self.tokenizer.decode(output_ids[:output_budget], skip_special_tokens=True)

    def _build_judge_prompt(self, conversation: str, output: str, label: str, sample_rubric=None) -> str:
        """Build judge prompt for rubric_reward.

        rubric_mode=general: uses rubric_prompt_template (or default prompt) for all samples.
        rubric_mode=sample:  uses sample_rubric_prompt_template with per-sample {rubric} fill.
        Falls back to general when sample_rubric is None or template is missing.
        """
        if self._rubric_mode == "sample" and sample_rubric is not None and self._sample_rubric_prompt_template:
            prompt = self._sample_rubric_prompt_template
            prompt = prompt.replace("{rubric}", self._format_rubric(sample_rubric))
        else:
            prompt = self._rubric_prompt_template or self._judge_prompt_template
        prompt = prompt.replace("{input}", str(conversation))
        prompt = prompt.replace("{label}", str(label))
        output = self._maybe_truncate_output(prompt, str(output))
        prompt = prompt.replace("{output}", output)
        return prompt

    def _extract_score(self, text: str) -> float | None:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                score = data.get("SCORE") or data.get("score") or data.get("average_score")
                if score is not None:
                    return float(score)
            except Exception:
                pass

        match = re.search(r"\"?(?:SCORE|average_score)\"?\s*:\s*\"?([0-9]+(?:\.[0-9]+)?)\"?", text, re.IGNORECASE)
        if match:
            return float(match.group(1))

        match = re.search(r"\b(10|[1-9])(?:\.\d+)?\b", text)
        if match:
            return float(match.group(1))

        return None

    def _extract_rubric_score(self, text: str, rubric: dict) -> float | None:
        """Parse criterion_scores from LLM output and compute weighted average using rubric weights."""
        all_criteria = (rubric.get("factual_criteria") or []) + (rubric.get("process_criteria") or [])
        weights = [c.get("weight", 0.0) for c in all_criteria]
        total_w = sum(weights)
        if not all_criteria or total_w <= 0:
            return None
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
                scores_list = data.get("criterion_scores", [])
                if isinstance(scores_list, list) and len(scores_list) == len(weights):
                    scores = [max(1.0, min(10.0, float(item.get("score", 1)))) for item in scores_list]
                    weighted_avg = sum(s * w for s, w in zip(scores, weights)) / total_w
                    return round(weighted_avg)
            except Exception:
                pass
        return None

    def _scale_score(self, raw_score: float | None) -> float:
        if raw_score is None:
            return 0.0
        raw_score = float(raw_score)

        # If score is already in [0, 1] range (both prompts return 0-1), use directly
        if 0.0 <= raw_score <= 1.0:
            return raw_score

        # If score is in [1, 10] range (backward compatibility), convert based on scale config
        raw_score = max(1.0, min(10.0, raw_score))
        scale = str(self._judge_score_scale).lower()
        if scale in {"raw", "none"}:
            return raw_score
        if scale in {"0-1", "0_to_1"}:
            return (raw_score - 1.0) / 9.0
        if scale in {"-1-1", "-1_to_1", "minus1_to_1"}:
            return (raw_score - 5.5) / 4.5
        return (raw_score - 1.0) / 9.0

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        batch_size = len(data)
        reward_extra_info["format_reward"] = [None] * batch_size
        reward_extra_info["result_reward"] = [None] * batch_size
        reward_extra_info["rubric_reward"] = [None] * batch_size
        reward_extra_info["rubric_judge_output"] = [None] * batch_size
        reward_extra_info["reward"] = [None] * batch_size

        llm_indices = []
        llm_prompts = []
        llm_meta = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:int(valid_response_length)]

            valid_prompt_len = int(valid_prompt_length)
            prompt_str = self.tokenizer.decode(prompt_ids[-valid_prompt_len:] if valid_prompt_len > 0 else [], skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            sample_rubric = data_item.non_tensor_batch.get("rubric", None)
            if sample_rubric is None and self._rubric_lookup:
                _img_paths = data_item.non_tensor_batch.get("image_paths")
                _img_path = _img_paths[0] if _img_paths is not None and len(_img_paths) > 0 else ""
                _question = data_item.non_tensor_batch.get("question_text") or ""
                sample_rubric = self._rubric_lookup.get((_img_path, _question))

            # Get format and result rewards separately
            fmt_reward = self._get_format_reward(response_str, data_source)
            res_reward = self._get_result_reward(response_str, ground_truth, data_source, prompt_str)

            reward_extra_info["format_reward"][i] = fmt_reward
            reward_extra_info["result_reward"][i] = res_reward

            need_judge = self._rubric_weight > 0.0

            if need_judge:
                judge_prompt = self._build_judge_prompt(
                    conversation=prompt_str,
                    output=response_str,
                    label=ground_truth,
                    sample_rubric=sample_rubric,
                )
                llm_indices.append(i)
                llm_prompts.append(judge_prompt)
                llm_meta.append({
                    "fmt_reward": fmt_reward,
                    "res_reward": res_reward,
                    "valid_response_length": int(valid_response_length),
                    "sample_rubric": sample_rubric,
                })
            else:
                # Pure rule-based: no judge needed
                # final = format_weight * fmt + (1 - format_weight) * res
                final_reward = self._format_weight * fmt_reward + (1.0 - self._format_weight) * res_reward
                reward_tensor[i, max(0, int(valid_response_length) - 1)] = final_reward
                reward_extra_info["rubric_reward"][i] = 0.0
                reward_extra_info["reward"][i] = final_reward

        if llm_prompts:
            _LOGGER.info("Scoring %d items with LLM judge", len(llm_prompts))
            scaled_scores, outputs = self._judge_batch(llm_prompts)
            for pos, (idx, scaled_score, meta) in enumerate(zip(llm_indices, scaled_scores, llm_meta)):
                output_text = outputs[pos]
                # For rubric_sample mode, recompute weighted score from criterion_scores.
                rubric_dict = meta.get("sample_rubric")
                if rubric_dict is not None:
                    computed = self._extract_rubric_score(output_text, rubric_dict)
                    if computed is not None:
                        scaled_score = self._scale_score(computed)

                fmt_reward = meta.get("fmt_reward") or 0.0
                res_reward = meta.get("res_reward") or 0.0
                rule_weight = 1.0 - self._rubric_weight
                rule_fused = rule_weight * res_reward + self._rubric_weight * scaled_score
                final_reward = self._format_weight * fmt_reward + (1.0 - self._format_weight) * rule_fused

                reward_tensor[idx, max(0, meta["valid_response_length"] - 1)] = final_reward
                reward_extra_info["rubric_reward"][idx] = scaled_score
                reward_extra_info["rubric_judge_output"][idx] = output_text
                reward_extra_info["reward"][idx] = final_reward

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor

    async def call_async(self, data: DataProto, return_dict: bool = False):
        """Async version of __call__ for use within a running asyncio event loop.

        Identical to __call__ but awaits _judge_batch_async instead of calling
        _judge_batch (which would create a nested event loop and fail).
        """
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        batch_size = len(data)
        reward_extra_info["format_reward"] = [None] * batch_size
        reward_extra_info["result_reward"] = [None] * batch_size
        reward_extra_info["rubric_reward"] = [None] * batch_size
        reward_extra_info["rubric_judge_output"] = [None] * batch_size
        reward_extra_info["reward"] = [None] * batch_size

        llm_indices = []
        llm_prompts = []
        llm_meta = []

        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:int(valid_response_length)]

            valid_prompt_len = int(valid_prompt_length)
            prompt_str = self.tokenizer.decode(
                prompt_ids[-valid_prompt_len:] if valid_prompt_len > 0 else [],
                skip_special_tokens=True,
            )
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch.get(self.reward_fn_key, "unknown")
            sample_rubric = data_item.non_tensor_batch.get("rubric", None)
            if sample_rubric is None and self._rubric_lookup:
                _img_paths = data_item.non_tensor_batch.get("image_paths")
                _img_path = _img_paths[0] if _img_paths is not None and len(_img_paths) > 0 else ""
                _question = data_item.non_tensor_batch.get("question_text") or ""
                sample_rubric = self._rubric_lookup.get((_img_path, _question))

            fmt_reward = self._get_format_reward(response_str, data_source)
            # For zerobench complex answers, use the async LLM path directly to
            # avoid _judge_batch creating a nested event loop (which conflicts with
            # the already-running loop in call_async). All other data sources use
            # sync rule-based scoring and are safe to call directly.
            if data_source == "zerobench" and self._should_use_llm_for_zerobench(ground_truth):
                res_reward = await self._get_zerobench_result_reward_with_llm_async(
                    response_str, ground_truth, prompt_str
                )
            else:
                res_reward = self._get_result_reward(response_str, ground_truth, data_source, prompt_str)

            reward_extra_info["format_reward"][i] = fmt_reward
            reward_extra_info["result_reward"][i] = res_reward

            if self._rubric_weight > 0.0:
                judge_prompt = self._build_judge_prompt(
                    conversation=prompt_str,
                    output=response_str,
                    label=ground_truth,
                    sample_rubric=sample_rubric,
                )
                llm_indices.append(i)
                llm_prompts.append(judge_prompt)
                llm_meta.append({
                    "fmt_reward": fmt_reward,
                    "res_reward": res_reward,
                    "valid_response_length": int(valid_response_length),
                    "sample_rubric": sample_rubric,
                })
            else:
                final_reward = self._format_weight * fmt_reward + (1.0 - self._format_weight) * res_reward
                reward_tensor[i, max(0, int(valid_response_length) - 1)] = final_reward
                reward_extra_info["rubric_reward"][i] = 0.0
                reward_extra_info["reward"][i] = final_reward

        if llm_prompts:
            _LOGGER.info("Scoring %d items with LLM judge (async)", len(llm_prompts))
            self._ensure_client()
            scaled_scores, outputs = await self._judge_batch_async(llm_prompts)
            for pos, (idx, scaled_score, meta) in enumerate(zip(llm_indices, scaled_scores, llm_meta)):
                output_text = outputs[pos]
                rubric_dict = meta.get("sample_rubric")
                if rubric_dict is not None:
                    computed = self._extract_rubric_score(output_text, rubric_dict)
                    if computed is not None:
                        scaled_score = self._scale_score(computed)

                fmt_reward = meta.get("fmt_reward") or 0.0
                res_reward = meta.get("res_reward") or 0.0
                rule_weight = 1.0 - self._rubric_weight
                rule_fused = rule_weight * res_reward + self._rubric_weight * scaled_score
                final_reward = self._format_weight * fmt_reward + (1.0 - self._format_weight) * rule_fused
                reward_tensor[idx, max(0, meta["valid_response_length"] - 1)] = final_reward
                reward_extra_info["rubric_reward"][idx] = scaled_score
                reward_extra_info["rubric_judge_output"][idx] = outputs[pos]
                reward_extra_info["reward"][idx] = final_reward

        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        return reward_tensor
