# 实验规范

## 1. 目的

统一 A 组和 B 组的模型、环境、配对实验、时间轴和结果字段，保证两组结果能够直接连接和复核。

公共配置：

```text
shared/project_config.yaml
shared/metadata_schema.json
```

学生可以使用本地路径覆盖配置，但不得私自修改模型、任务、初始状态、随机种子、open-loop steps、condition 名称、trace layer 和 mask control 规则。

## 2. 固定实验设置

```text
Model: OpenVLA-OFT
Checkpoint: moojink/openvla-7b-oft-finetuned-libero-object
Benchmark: LIBERO-Object
Task: pick up the alphabet soup and place it in the basket
Initial states: 0 / 1 / 2
Seed: 7
Conditions: baseline / target_mask / background_control
Open-loop steps: 8
```

任何修改必须通过独立 Issue 和共享配置 PR。

## 3. 配对实验

一个配对实验组必须共享：

```text
model
checkpoint
model_code_commit
suite
task_id
task_name
initial_state_index
random_seed
instruction
max_steps
num_open_loop_steps
```

只允许 `condition` 不同。

统一 ID：

```text
paired_group_id = <task_id>__<initial_state>__seed<seed>
episode_id = <paired_group_id>__<condition>
```

## 4. 时间轴

必须区分：

| 字段 | 含义 |
|---|---|
| `env_step` | 环境实际执行一步 |
| `policy_query_index` | 模型重新生成一次动作块 |
| `action_chunk_index` | 动作块内部的动作序号 |
| `trace_index` | 保存 hidden state 的策略查询序号 |

禁止使用视频帧编号代替 `policy_query_index`。

每条 trace 至少记录：

```text
episode_id
policy_query_index
env_step_start
env_step_end
condition
layer_name
feature_type
feature_shape
action_chunk_shape
artifact_path
```

## 5. A 组：表征追踪

第一阶段保存四个代表层，具体 layer id 必须写入配置。

优先保存：

- vision pooled feature；
- text pooled feature；
- joint pooled feature；
- action chunk；
- token count 和 feature shape。

基础指标：

```text
cosine drift
normalized L2 drift
layer-wise drift
time-wise drift
condition-paired drift
action-chunk distance
```

默认不保存全部 token。需要 full-token trace 时单独建立 Issue。

## 6. B 组：视觉干预

### Target mask

- 覆盖目标物体区域；
- 记录 mask 来源、面积和替换方式；
- 不修改 instruction、proprioception、initial state 和 seed。

### Background control

- 面积与 target mask 尽量一致；
- 不覆盖机器人、目标物体、目标容器和其他关键对象；
- 面积误差定义为：

```text
abs(area_background - area_target) / area_target
```

当前允许误差不超过 10%。

基础行为指标：

```text
action-chunk L2/cosine distance
per-dimension action distance
first divergence policy-query
trajectory divergence
success change
episode length change
failure stage
```

## 7. 失败阶段

除最终 `success` 外，至少记录：

```text
perception_or_approach
grasp
transport
place
timeout
unknown
```

失败阶段必须基于可复核规则或视频审查记录。

## 8. 跨组连接

跨组主键：

```text
paired_group_id
episode_id
condition
policy_query_index
```

联合分析前检查：

1. trace 和 action chunk 是否完整；
2. baseline、target、background 是否属于同一配对实验；
3. policy-query 数量是否一致；
4. 长度不一致时采用何种对齐方式；
5. 缺失 episode 和缺失 query 是否明确记录。

## 9. 实验产物

每次正式实验至少提交：

```text
config.yaml
command.txt
git_commit.txt
episode_manifest.jsonl
per_episode_results.csv
metrics.json
figures/
failure_cases/
result_summary.md
```

## 10. 基本要求

- 不允许不同 condition 使用不同 checkpoint；
- 不允许为获得更明显结果临时修改 instruction；
- 不删除失败 episode；
- 不只报告均值，必须保留逐 episode 结果；
- 不把 attention、embedding 或降维图直接表述为因果解释；
- metadata 不完整时不进入正式统计。
