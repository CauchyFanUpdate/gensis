# LLM Judge Integration for GRPO Training

本文档说明如何使用 LLM judge reward manager 进行 GRPO 训练。

## 架构概览

### 核心组件

1. **LLMJudgeRewardManager** (`verl/workers/reward_manager/llm_judge.py`)
   - 集成 vLLM 后端用于 LLM-based reward scoring
   - 支持远程 API（OpenAI-compatible）和本地 vLLM engine 两种后端
   - 与 rule-based reward 融合

2. **vLLM Judge Server**
   - 独立的 vLLM 服务，在指定端口监听
   - 支持 sleep/wake 机制，节省 GPU 显存
   - 通过 HTTP API 与训练进程通信

## 快速开始

### 1. 准备数据集

数据需要包含 `reward_type` 和 `reward_model.ground_truth` 字段：

```python
{
    "prompt": [{"role": "user", "content": "Question text"}],
    "images": [PIL.Image, ...],
    "reward_model": {
        "ground_truth": "Expected answer",
        "style": "rule"
    },
    "extra_info": {
        "reward_type": "llm_judge"  # 标记为需要 LLM judge 评分
    }
}
```

### 2. 启动 LLM Judge Server

```bash
bash scripts/grpo_qwen3vl/start_llm_judge_server.sh \
    /path/to/Qwen3-VL-4B-Instruct \
    51001 \
    1 \
    0.35
```

参数：
- `MODEL_PATH`：judge 模型路径
- `PORT`：监听端口（默认 51001）
- `TP`：张量并行大小
- `GPU_MEM`：GPU 显存利用率

### 3. 运行训练

```bash
# 方式 1：使用启动脚本（自动启动 judge server）
START_JUDGE_SERVER=1 bash scripts/grpo_qwen3vl/run_grpo_with_llm_judge.sh

# 方式 2：手动指定参数
bash scripts/grpo_qwen3vl/run_grpo_with_llm_judge.sh \
    MODEL_PATH=/path/to/actor \
    JUDGE_MODEL_PATH=/path/to/judge \
    JUDGE_PORT=51001 \
    N_GPUS=8 \
    TRAIN_FILE=train.parquet \
    VAL_FILE=val.parquet
```

## 配置详解

### Reward Manager 配置

```yaml
reward_model:
  reward_manager: llm_judge  # 使用 llm_judge manager
  reward_kwargs:
    # 分数融合权重
    format_weight: 0.1  # 格式奖励权重（0.1 = 10% format, 90% for rule+judge）
    rubric_weight: 0.7  # judge 权重（0.7 = 70% judge, 30% rule；1.0 = 纯judge，0.0 = 纯rule）
    
    # LLM Judge 后端配置
    llm_judge_kwargs:
      # 后端选择
      backend: openai_server  # 或 vllm_engine
      
      # OpenAI API 配置（backend=openai_server）
      base_url: "http://localhost:51001/v1"
      api_key: "qwen"
      model_name: "qwen"
      
      # 本地 vLLM 配置（backend=vllm_engine）
      # use_vllm_engine: true
      # model_path: /path/to/model
      # tensor_parallel_size: 1
      
      # Judge prompt
      prompt_path: "./llm_judge_prompt.txt"
      
      # 采样参数
      temperature: 0.7
      top_p: 0.8
      max_tokens: 1024
      presence_penalty: 1.5
      frequency_penalty: 1.0
      
      # 高级配置
      enable_sleep_mode: true  # 训练空闲时睡眠
      sleep_level: 1
      timeout: 120
      max_concurrent_requests: 8
```

## 混合评分模式

奖励融合公式：
```
final_reward = format_weight * format_reward 
             + (1 - format_weight) * ((1 - rubric_weight) * result_reward + rubric_weight * judge_score)
```

其中：
- `format_weight ∈ [0, 1]`：格式奖励的权重
- `rubric_weight ∈ [0, 1]`：judge 的权重（剩余部分则为 rule 的权重）

### 模式 1：纯 LLM Judge（推荐）
```yaml
format_weight: 0.1        # 10% format
rubric_weight: 1.0         # 100% judge, 0% rule
# 最终 = 0.1 * format + 0.9 * judge
```

### 模式 2：LLM Judge + Rule-based 混合（70-30）
```yaml
format_weight: 0.1        # 10% format
rubric_weight: 0.7         # 70% judge, 30% rule
# 最终 = 0.1 * format + 0.27 * rule + 0.63 * judge
```

### 模式 3：纯规则奖励（不调用 LLM judge）
```yaml
format_weight: 0.1        # 10% format
rubric_weight: 0.0         # 0% judge, 100% rule
# 最终 = 0.1 * format + 0.9 * result
# 此时不会调用 LLM judge，节省时间和成本
```

## Judge Prompt 定制

修改 `llm_judge_prompt.txt` 来改变评分逻辑：

```
You are an expert evaluator...

# Evaluation Criteria
1. Correctness: {your criteria}
2. ...

# Input
{input}

# Model Output
{output}

# Ground Truth
{label}

# Evaluation
Provide your evaluation in JSON with SCORE field (1-10):
{
  "SCORE": <integer 1-10>,
  "reasoning": "<explanation>"
}
```

关键占位符：
- `{input}`：原始问题/对话
- `{output}`：模型生成的答案
- `{label}`：标准答案

## 后端选择

### OpenAI Server（推荐）
- **优点**：通过 HTTP API，trainer 和 judge 独立，易于扩展
- **缺点**：有网络延迟
- **配置**：
  ```yaml
  backend: openai_server
  base_url: "http://localhost:51001/v1"
  ```

### vLLM Engine（本地）
- **优点**：同进程，无网络延迟，显存共享
- **缺点**：占用更多显存
- **配置**：
  ```yaml
  llm_judge_kwargs:
    use_vllm_engine: true
    model_path: /path/to/model
  ```

## Sleep/Wake 机制

启用 `enable_sleep_mode: true` 后，judge 模型在训练空闲时自动休眠，释放显存：

```yaml
enable_sleep_mode: true  # 启用
sleep_level: 1           # 休眠深度（1-3，数字越大显存释放越多）
```

## 常见问题

### Q1: 怎样调整权重比例？
修改 `format_weight` 和 `rubric_weight`：
```yaml
format_weight: 0.1         # 0-1，格式奖励的比例
rubric_weight: 0.7          # 0-1，在剩余 0.9 中 judge 的比例
```

如果 `rubric_weight=0.0`，自动跳过 LLM judge，纯规则评分。
如果 `rubric_weight=1.0`，只用 LLM judge，不使用规则奖励。

### Q2: Judge 评分范围不对？
修改 `score_scale` 参数：
```yaml
score_scale: "0_to_1"      # 0-1 范围
score_scale: "minus1_to_1" # -1 到 1 范围
score_scale: "raw"         # 原始 1-10
```

### Q3: 如何调试 judge 输出？
启用调试日志：
```bash
VERL_LOGGING_LEVEL=DEBUG bash run_grpo_with_llm_judge.sh
```
Judge 输出会保存到 checkpoint 目录。

### Q4: Judge 和 Actor 共享 GPU 怎么办？
使用本地 vLLM engine + sleep mode：
```yaml
llm_judge_kwargs:
  use_vllm_engine: true
  enable_sleep_mode: true
  gpu_memory_utilization: 0.35  # 降低显存占用
```

### Q5: Judge server 连接超时？
检查：
1. Judge server 是否运行：`curl http://localhost:51001/v1/models`
2. 防火墙是否开放端口
3. 增加 timeout：`timeout: 300`

## 性能优化

### 减少 Judge 调用
仅对部分样本进行 judge 评分：
```python
# 数据中标记 reward_type
if should_use_judge(sample):
    extra_info["reward_type"] = "llm_judge"
```

### 批量评分
自动批量处理多个样本：
```yaml
max_concurrent_requests: 8  # 并发请求数
```

### 降低 Judge 成本
使用更小的 judge 模型（如 Qwen2.5-7B 替代 Qwen3-VL-4B）。

## 监控和调试

训练日志中查看：
```
[reward_manager] Scoring X items with LLM judge
[llm_judge] judge batch complete total_s=XX avg_s=XX
```

Checkpoint 目录下的 debug 日志记录每个样本的：
- Judge prompt
- Judge 输出
- 解析的分数
- 最终融合的 reward

## 文件列表

```
scripts/grpo_qwen3vl/
├── llm_judge.py                    # Reward manager 实现
├── llm_judge_prompt.txt            # Judge prompt 模板
├── start_llm_judge_server.sh       # 启动 judge server
├── run_grpo_with_llm_judge.sh      # 完整训练脚本
├── grpo_llm_judge_config.yaml      # Hydra 配置示例
└── LLM_JUDGE_INTEGRATION.md        # 本文档
```

## 下一步

1. 准备你的数据集，标记 `reward_type = "llm_judge"`
2. 自定义 `llm_judge_prompt.txt` 匹配你的评分标准
3. 运行 `bash run_grpo_with_llm_judge.sh`
4. 监控训练日志，验证 judge 评分是否合理

有问题？查看 CLAUDE.md 中的 verl 文档。
