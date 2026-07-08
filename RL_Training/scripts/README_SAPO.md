# SAPO Training Scripts (Smoothed Adaptive Preference Optimization)

## Overview

SAPO (Smoothed Adaptive Preference Optimization) is a variant of PPO that uses soft tau-based smoothing instead of hard clipping. These scripts enable SAPO training with LLM-judge-based reward for Qwen3-VL and Qwen3.5 models.

**Key Paper**: [Smoothed Adaptive Preference Optimization for Policy Alignment](https://arxiv.org/pdf/2511.20347)

## Scripts

### 1. `run_sapo_qwen3vl_llm_judge.sh`
SAPO training for **Qwen3-VL-4B-Instruct** with LLM judge reward.
- Uses the full vision-language model pipeline
- Supports multi-modal (image + text) input
- Default: 8 GPU FSDP2 training

### 2. `run_sapo_qwen35_llm_judge.sh`
SAPO training for **Qwen3.5-4B** with LLM judge reward.
- Text-to-text model with image support
- More memory-efficient than Qwen3-VL
- Default: 8 GPU FSDP2 training

## Usage

### Basic Usage

```bash
# Qwen3-VL
bash run_sapo_qwen3vl_llm_judge.sh

# Qwen3.5
bash run_sapo_qwen35_llm_judge.sh
```

### With Custom Parameters

```bash
# Override model path, training data, and GPU count
MODEL_PATH=/path/to/model \
TRAIN_FILE=/path/to/train.parquet \
VAL_FILE=/path/to/val.parquet \
N_GPUS=8 \
bash run_sapo_qwen3vl_llm_judge.sh
```

### Custom SAPO Hyperparameters

```bash
# Adjust tau_pos and tau_neg for smoothing function
TAU_POS=1.0 \
TAU_NEG=1.05 \
bash run_sapo_qwen3vl_llm_judge.sh
```

## Key SAPO Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `loss_mode` | `sapo` | Policy loss function (sapo = smoothed, vanilla = clipped) |
| `tau_pos` | `1.0` | Positive tau for upside smoothing |
| `tau_neg` | `1.05` | Negative tau for downside smoothing (paper default for Qwen3-30B) |
| `adv_estimator` | `grpo` | Group-relative PPO for advantage estimation |

## Reward Configuration

Both scripts use **LLM-judge-based reward** with the following composition:

```
final_reward = 0.1 * format_reward 
             + 0.27 * result_reward 
             + 0.63 * judge_reward
```

**Parameters** (in `reward.reward_kwargs`):
- `format_weight`: 0.1 (fixed format score weight)
- `rubric_weight`: 0.7 (LLM judge weight in remaining 90%)
  - Result (rule-based): (1 - 0.7) × 0.9 = 27%
  - Judge: 0.7 × 0.9 = 63%

**LLM Judge Setup**:
- Judge model: Qwen3-32B (configurable via `JUDGE_MODEL_PATH`)
- Judge port: 51001 (Qwen3-VL) or 51002 (Qwen3.5)
- vLLM server launched automatically with sleep mode to save GPU

## Environment Variables

### Required
- None (all have sensible defaults)

### Optional
- `MODEL_PATH`: Path to actor model (default: Qwen3-VL-4B or Qwen3.5-4B)
- `TRAIN_FILE`: Training data path
- `VAL_FILE`: Validation data path
- `JUDGE_MODEL_PATH`: Judge model path (default: Qwen3-32B)
- `N_GPUS`: Number of GPUs per node (default: 8)
- `ROLLOUT_TP`: Tensor parallelism for rollout (default: 1)
- `VERL_ROOT`: Path to verl framework (auto-detected)
- `TAU_POS`: Positive tau (default: 1.0)
- `TAU_NEG`: Negative tau (default: 1.05)
- `START_JUDGE_SERVER`: Start judge server (1=yes, 0=no, default: 1)

## Monitoring

### WandB Logging
Both scripts log to W&B with the following metrics:
- **Training metrics**: loss, KL divergence, entropy
- **Reward components**: format_reward, result_reward, judge_reward, final_reward
- **Performance**: throughput, gradient norm, policy/value metrics

### Log Files
- Main training log: `$CKPTS_DIR/train.log`
- Judge server log: `/tmp/vllm_judge_${JUDGE_PORT}.log`

### Real-time Monitoring
```bash
# Watch training progress
tail -f $CKPTS_DIR/train.log

# Monitor judge server
tail -f /tmp/vllm_judge_51001.log
```

## Configuration

Both scripts use the GRPO base config from:
```
zerovision/scripts/config/grpo_qwen3vl_4b_llm_judge.yaml
```

To customize the base config, edit this file or override parameters on the command line.

## Data Format

Training data should be Parquet format with the following columns:
- `prompt`: List of chat messages
- `images`: List of PIL images (for VLM models)
- `reward_model.ground_truth`: Ground truth answer
- `data_source`: Data source identifier (e.g., "zerobench")
- `extra_info`: Additional metadata

See `workspace/grpo_datagen/prepare_parquet.py` for data preparation.

## Performance Tips

1. **GPU Memory**: If OOM, reduce `actor.ppo_micro_batch_size_per_gpu` from 2 to 1
2. **Judge Server**: Use `START_JUDGE_SERVER=0` if judge already running
3. **Tensor Parallelism**: Increase `ROLLOUT_TP` if vLLM OOMs during generation
4. **SAPO Tuning**: Adjust `TAU_NEG` up (e.g., 1.10) for softer clipping

## Troubleshooting

### Judge Server Fails to Start
```bash
# Check judge server log
cat /tmp/vllm_judge_51001.log

# Manually kill stale processes
pkill -9 -f "vllm serve"
pkill -9 -f "raylet"
```

### CUDA Out of Memory
```bash
# Reduce batch size
N_GPUS=8 bash run_sapo_qwen3vl_llm_judge.sh  # use all 8 GPUs

# Or reduce micro batch size in config
```

### Data Loading Issues
```bash
# Ensure parquet files exist and are readable
ls -lh $TRAIN_FILE $VAL_FILE

# Regenerate data if needed
cd workspace/grpo_datagen
python prepare_parquet.py --data_root dataset/ ...
```

## References

- **SAPO Paper**: [Smoothed Adaptive Preference Optimization for Policy Alignment](https://arxiv.org/pdf/2511.20347)
- **verl Framework**: [https://github.com/verl-project/verl](https://github.com/verl-project/verl)
- **GRPO**: Group-Relative Policy Optimization for dynamic advantage normalization
- **LLM Judge**: OpenAI-compatible API for model-based reward scoring
