"""Spatial consistency reward manager adapter for experimental reward loop."""

import asyncio
import logging

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.workers.reward_manager.spatial_consistency import SpatialConsistencyRewardManager

_LOGGER = logging.getLogger(__name__)


@register("spatial_consistency")
class SpatialConsistencyRewardManagerAdapter(RewardManagerBase):
    """Adapter wrapping SpatialConsistencyRewardManager for the experimental reward loop."""

    def __init__(self, config, tokenizer, compute_score, **kwargs):
        super().__init__(config, tokenizer, compute_score)
        self._manager = SpatialConsistencyRewardManager(
            tokenizer=tokenizer,
            num_examine=kwargs.get("num_examine", 0),
            compute_score=compute_score,
            reward_fn_key=kwargs.get("reward_fn_key", "data_source"),
            format_weight=kwargs.get("format_weight", 0.1),
            consistency_weight=kwargs.get("consistency_weight", 0.0),
            consistency_judge_base_url=kwargs.get("consistency_judge_base_url"),
            consistency_judge_api_key=kwargs.get("consistency_judge_api_key", "qwen"),
            consistency_judge_model_name=kwargs.get("consistency_judge_model_name", "qwen"),
            consistency_judge_timeout=kwargs.get("consistency_judge_timeout", 120.0),
            consistency_judge_max_concurrent=kwargs.get("consistency_judge_max_concurrent", 16),
            consistency_judge_max_tokens=kwargs.get("consistency_judge_max_tokens", 64),
            consistency_judge_sigma=kwargs.get("consistency_judge_sigma", 0.1),
            consistency_judge_enable_sleep=kwargs.get("consistency_judge_enable_sleep", True),
            consistency_max_pairs_per_sample=kwargs.get("consistency_max_pairs_per_sample", 4),
        )

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"

        # SpatialConsistencyRewardManager.__call__ is synchronous and spawns its own
        # event loop internally for judge HTTP calls. Run it in a thread executor so
        # we don't block the outer event loop.
        loop = asyncio.get_event_loop()
        reward_dict = await loop.run_in_executor(
            None, lambda: self._manager(data, return_dict=True)
        )

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
