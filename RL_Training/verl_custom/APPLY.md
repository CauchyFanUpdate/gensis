# verl_custom — 自定义 RL 代码（叠加到上游 verl）

本目录**只包含 zerovision 对 verl 框架的自定义改动**，不含上游框架本体。
基线为 verl fork [`WangYipu2002/verl`](https://github.com/WangYipu2002/verl)（上游
[`verl-project/verl`](https://github.com/verl-project/verl)）在 merge-base
`e0668482` 之上的 4 个自定义 commit：

```
1a30563b  add llm judge reward manager and qwen3vl support
3ff15738  feat: add spatial consistency reward manager and zerobench reward
bf1189a6  feat: support CrossView-Qwen3VL in verl actor/ref FSDP
83c819bf  fix: patch AutoConfig loading for crossview-qwen3vl in HFModelConfig
```

共改动 24 个文件、+2281 行。以两种形式收录，任选其一使用：

- **`verl_custom.patch`** — 完整 `git diff`，可直接 `git apply` 到干净的上游 verl。
- **`verl/…`** — 改动后的文件按原路径就地摆放（本目录内的相对路径 = verl 仓库内的目标路径），便于直接阅读或整文件覆盖。

> 本目录内文件的路径**就是它在 verl 仓库中的落点**。例如
> `verl_custom/verl/workers/reward_manager/llm_judge.py` → 放到 verl 的
> `verl/workers/reward_manager/llm_judge.py`；`verl_custom/requirements.txt` → verl 根目录
> `requirements.txt`。

## 每个文件放到 verl 的哪里

下表 `类型` 列：**新增** = verl 里原本没有，直接拷贝即可落地；**改上游** = 需要在
上游已有文件里插入这几行（用 `git apply` 打 patch，或手动按下方"关键改动"补）。

### A. 新增文件（直接拷贝到对应路径，8 个）

| verl_custom 内路径（=verl 目标路径） | 行数 | 作用 |
| --- | --- | --- |
| `verl/workers/reward_manager/llm_judge.py` | 754 | LLM-as-judge reward manager；支持远程 OpenAI-兼容 API 与本地 vLLM engine 两种后端，与 rule-based reward 融合 |
| `verl/workers/reward_manager/spatial_consistency.py` | 411 | 空间一致性 reward manager |
| `verl/utils/reward_score/zerobench.py` | 102 | ZeroBench 答案判分（规则 + 复杂题回退 LLM judge） |
| `verl/utils/reward_score/spatial.py` | 253 | 空间任务判分 |
| `verl/utils/reward_score/llava.py` | 178 | LLaVA 风格判分（multiple-choice / number / ocrtext / bbox） |
| `verl/models/transformers/crossview_qwen3_vl.py` | 138 | CrossView-Qwen3-VL 模型（带 cross-view attention） |
| `verl/experimental/reward_loop/reward_manager/llm_judge.py` | 67 | 上面 reward manager 的 experimental reward_loop 适配版 |
| `verl/experimental/reward_loop/reward_manager/spatial_consistency.py` | 61 | 同上，spatial 适配版 |

### B. 需改上游已有文件（插入注册/接线，16 个）

| verl 目标文件 | 净增行 | 关键改动（要往里加什么） |
| --- | --- | --- |
| `verl/workers/reward_manager/__init__.py` | +4 | `from .llm_judge import LLMJudgeRewardManager` / `from .spatial_consistency import SpatialConsistencyRewardManager`，并加入 `__all__` —— **注册两个 reward manager** |
| `verl/utils/reward_score/__init__.py` | +12 | 在 `default_compute_score` 的分派里加 `data_source` 分支：`crosspoint/mindcube/spar/unknown→spatial`、`llava:*→llava`、`zerobench→zerobench` |
| `verl/experimental/reward_loop/reward_manager/__init__.py` | +4 | 注册 `LLMJudgeRewardManagerAdapter` / `SpatialConsistencyRewardManagerAdapter` |
| `verl/experimental/reward_loop/reward_loop.py` | +21 -5 | reward_loop 接入上述 adapter |
| `verl/models/mcore/registry.py` | +1 | 注册模型：`CROSSVIEW_QWEN3_VL = "Qwen3VLForConditionalGenerationWithCrossViewAttn"` |
| `verl/models/transformers/qwen3_vl.py` | +75 | Qwen3-VL forward / 适配改动 |
| `verl/models/transformers/monkey_patch.py` | +36 | 为 CrossView-Qwen3-VL 打 monkey patch（含 AutoConfig 加载修复） |
| `verl/trainer/ppo/ray_trainer.py` | +27 | 训练主循环接入自定义 reward manager |
| `verl/trainer/constants_ppo.py` | +18 -2 | 新增常量（reward_type 等） |
| `verl/utils/dataset/rl_dataset.py` | +53 | 数据集读入 `reward_type` / `reward_model.ground_truth` 等字段 |
| `verl/workers/config/model.py` | +22 -3 | 模型 config 支持 CrossView 参数 |
| `verl/workers/engine/fsdp/transformer_impl.py` | +8 | FSDP 引擎适配 |
| `verl/workers/engine_workers.py` | +1 | 加 `aggressive_empty_cache(force_sync=True)` |
| `verl/utils/model.py` | +6 | 模型加载/config 辅助 |
| `verl/utils/distributed.py` | +28 | 分布式辅助 |
| `requirements.txt` | +1 -1 | 依赖调整 |

## 应用方式

```bash
# 1) 取干净的上游 verl（fork 或上游均可）
git clone https://github.com/WangYipu2002/verl.git
cd verl
git checkout e0668482          # merge-base，确保 patch 干净套用

# 2) 套用自定义改动（推荐：一次性打全 24 个文件）
git apply /path/to/RL_Training/verl_custom/verl_custom.patch
#   —— 或者手动落地 ——
#   A 类 8 个新增文件：直接拷贝（路径一一对应）
# cp -r /path/to/RL_Training/verl_custom/verl/. ./verl/
#   B 类 16 个已有文件：按上表"关键改动"手动补，或只对这些文件 git apply

# 3) 安装
pip install -e .
```

> 若直接使用 fork 的最新分支（已含这 4 个 commit），则**无需**再套 patch，本目录仅作
> 「自定义部分」的独立留档与代码审阅。
