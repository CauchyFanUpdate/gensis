# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LLM Judge reward manager adapter for experimental reward loop."""

import logging

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.workers.reward_manager.llm_judge import LLMJudgeRewardManager

_LOGGER = logging.getLogger(__name__)


@register("llm_judge")
class LLMJudgeRewardManagerAdapter(RewardManagerBase):
    """Adapter for LLMJudgeRewardManager to work with experimental reward loop."""

    def __init__(self, config, tokenizer, compute_score, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        self.llm_judge_manager = LLMJudgeRewardManager(
            tokenizer=tokenizer,
            num_examine=kwargs.get("num_examine", 0),
            compute_score=compute_score,
            reward_fn_key=kwargs.get("reward_fn_key", "data_source"),
            max_resp_len=kwargs.get("max_resp_len", None),
            llm_judge_kwargs=kwargs.get("llm_judge_kwargs", {}),
            format_weight=kwargs.get("format_weight"),
            rubric_weight=kwargs.get("rubric_weight"),
        )

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"

        # Directly await the async judge path — no subprocess spawning.
        # The previous approach (spawning _judge_in_remote_worker via ray.remote
        # per sample) created one short-lived Ray worker process per sample per
        # training step. With PID 1 = sleep (no zombie reaping), those processes
        # accumulated as zombies and eventually exhausted the PID table.
        reward_dict = await self.llm_judge_manager.call_async(data, return_dict=True)

        reward_tensor = reward_dict["reward_tensor"]
        reward_extra_info = reward_dict["reward_extra_info"]

        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()

        reward_score = float(reward_tensor[0, max(0, int(valid_response_length) - 1)])

        return {
            "reward_score": reward_score,
            "reward_extra_info": reward_extra_info,
        }
