# CrossView-Qwen3VL monkey-patch functions for verl actor/ref FSDP forward.
#
# Mirrors the qwen3_vl.py pattern but adds:
#   - VGGT encoding per sample
#   - CrossViewConsistencyModule fusion before scatter
#   - Dummy pass for FSDP2 gradient sync when no images in batch

from typing import List, Optional

import torch

from verl.models.transformers.qwen3_vl import Qwen3VLCausalLMOutputForPPO


def crossview_base_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values=None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.Tensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    cache_position: Optional[torch.LongTensor] = None,
    vggt_features: Optional[List[torch.Tensor]] = None,
    vggt_camera_tokens: Optional[List[torch.Tensor]] = None,
    **kwargs,
):
    """Patched CrossViewQwen3VLModel.forward for verl.

    verl pre-computes position_ids via get_rope_index, so position_ids is
    always provided and the RoPE block inside the original forward is skipped.
    Adds a dummy visual pass when pixel_values is None to keep all parameters
    in the gradient graph under FSDP2.
    """
    if inputs_embeds is None:
        inputs_embeds = self.get_input_embeddings()(input_ids)

    image_mask = None
    deepstack_image_embeds = None

    if pixel_values is not None:
        image_embeds_tuple, deepstack_image_embeds = self.get_image_features(
            pixel_values, image_grid_thw
        )
        if vggt_features is not None and vggt_camera_tokens is not None:
            fused = self._apply_multiview(
                list(image_embeds_tuple),
                vggt_features,
                vggt_camera_tokens,
                image_grid_thw=image_grid_thw,
            )
        else:
            fused = torch.cat(image_embeds_tuple, dim=0)

        fused = fused.to(inputs_embeds.device, inputs_embeds.dtype)
        image_mask, _ = self.get_placeholder_mask(
            input_ids, inputs_embeds=inputs_embeds, image_features=fused
        )
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, fused)
    else:
        # Dummy pass: keep visual encoder and crossview module in gradient graph.
        # Without this, FSDP2 AllReduce will deadlock on unused parameters.
        cfg = self.config.vision_config
        patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size ** 2
        dummy_pv = torch.zeros(
            (16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        dummy_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        dummy_embeds_tuple, _ = self.get_image_features(dummy_pv, dummy_thw)
        inputs_embeds = inputs_embeds + 0.0 * torch.cat(dummy_embeds_tuple, dim=0).mean()
        for p in self.multiview_module.parameters():
            inputs_embeds = inputs_embeds + 0.0 * p.sum()

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None:
        visual_pos_masks = image_mask[..., 0]
        deepstack_visual_embeds = deepstack_image_embeds

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return self.language_model(
        input_ids=None,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        cache_position=cache_position,
        visual_pos_masks=visual_pos_masks,
        deepstack_visual_embeds=deepstack_visual_embeds,
        **kwargs,
    )


def crossview_forward_with_normal_backend(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    geometry_encoder_inputs: Optional[List[torch.Tensor]] = None,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    """Patched Qwen3VLForConditionalGenerationWithCrossViewAttn.forward for verl.

    Pops geometry_encoder_inputs from the call, runs VGGT encoding per sample,
    then delegates to the patched CrossViewQwen3VLModel.forward
    (crossview_base_forward) via self.model().
    """
    vggt_features = None
    vggt_camera_tokens = None
    if geometry_encoder_inputs is not None and self.model.vggt_encoder is not None:
        vggt_device = next(self.model.vggt_encoder.parameters()).device
        vggt_features = []
        vggt_camera_tokens = []
        for vggt_inp in geometry_encoder_inputs:
            if vggt_inp.device != vggt_device:
                vggt_inp = vggt_inp.to(vggt_device)
            feats, cams = self.model.vggt_encoder.encode(vggt_inp)
            vggt_features.append(feats)
            vggt_camera_tokens.append(cams)

    outputs = self.model(
        input_ids,
        vggt_features=vggt_features,
        vggt_camera_tokens=vggt_camera_tokens,
        **kwargs,
    )
    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    return Qwen3VLCausalLMOutputForPPO(
        logits=logits,
        past_key_values=outputs.past_key_values,
    )
