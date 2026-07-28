# 第一周执行计划：可复现 Rollout、Trace 与 Paired Intervention

## 1. 本周目标

第一周只建立可靠实验基础，不训练新模型，不扩大任务集，不形成机制性结论。

```text
Group A：baseline rollout → hidden-state trace → step alignment
Group B：baseline/target/background → paired rollout → action divergence
```

## 2. 固定设置

使用 `shared/project_config.yaml`：

```text
Model: OpenVLA-OFT
Checkpoint: moojink/openvla-7b-oft-finetuned-libero-object
Benchmark: LIBERO-Object
Task: pick up alphabet soup and place it in basket
Initial states: 0 / 1 / 2
Conditions: baseline / target_mask / background_control
Seed: 7
Open-loop steps: 8
```

## 3. Group A：Representation Trace

### 唯一问题

> 能否在不改变模型输出的情况下，稳定保存并对齐四个代表层的 hidden state、policy query 和 action chunk？

### 任务

- 跑通 3 个 initial states 的 baseline rollout；
- 确认模型、环境和 checkpoint；
- 为四个代表层注册 hook；
- 保存 vision/text/joint pooled feature；
- 保存每次 policy query 的 action chunk；
- 建立 `policy_query_index ↔ env_step range ↔ trace` 映射；
- 输出第一张 layer-wise drift 图；
- 保存失败样本和完整日志。

### 最低交付

```text
groupA/reports/week1/
├── config.yaml
├── command.txt
├── git_commit.txt
├── episode_manifest.jsonl
├── trace_index.csv
├── per_episode_results.csv
├── metrics.json
├── figures/layer_wise_drift.png
├── failure_cases/
└── result_summary.md
```

### Gate A

- 3 个 initial states 均有 baseline episode；
- trace 数量与 policy-query 数量一致；
- action chunk 与 trace 使用同一 query index；
- hook 不改变 baseline action output；
- hidden state 无 NaN/Inf；
- 所有结果可追溯到 config、checkpoint 和 commit。

## 4. Group B：Behavior Intervention

### 唯一问题

> 能否在相同 initial state 和 seed 下，生成 target mask 与等面积 background control，并完成可比较的 paired rollouts？

### 任务

- 跑通 3 个 initial states 的 baseline rollout；
- 生成 target object mask；
- 生成等面积 background control mask；
- 验证 mask 不覆盖机器人、目标和关键对象；
- 对每个 initial state 执行 baseline/target/background 三个条件；
- 保存 action chunks、轨迹、success 和 failure stage；
- 输出第一张 action divergence 图；
- 保存 mask 可视化和失败样本。

### 最低交付

```text
groupB/reports/week1/
├── config.yaml
├── command.txt
├── git_commit.txt
├── episode_manifest.jsonl
├── mask_manifest.jsonl
├── paired_rollout_index.csv
├── per_episode_results.csv
├── metrics.json
├── figures/action_divergence.png
├── failure_cases/
└── result_summary.md
```

### Gate B

- 3 个 initial states × 3 个 conditions 均完成；
- paired episodes 的 task、state、seed、checkpoint 完全一致；
- target/background mask 面积差不超过 10%；
- mask 元数据和可视化完整；
- action chunk、success 和 episode length 无缺失；
- 所有结果可追溯到 config、checkpoint 和 commit。

## 5. 每日节点

| 日期 | Group A | Group B |
|---|---|---|
| Day 1 | 环境与 baseline rollout | 环境与 baseline rollout |
| Day 2 | hook 与 feature shape 检查 | target/background mask 生成 |
| Day 3 | 1 个 episode 完整 trace 对齐 | 1 个 initial state 完整 paired rollout |
| Day 4 | 3 个 states 正式 trace | 9 个正式 paired episodes |
| Day 5 | drift 图、失败样本、Gate 报告 | divergence 图、失败样本、Gate 报告 |

## 6. 跨组接口

两组只通过以下公共字段联调：

```text
paired_group_id
episode_id
task_id
initial_state_index
random_seed
condition
policy_query_index
env_step_start
env_step_end
action_chunk
```

第一周不要求联合统计，只要求字段可以连接。

## 7. 本周不做

- 不训练 OpenVLA；
- 不新增复杂解释器；
- 不运行 probing classifier；
- 不做跨任务泛化；
- 不把 layer drift 解释为因果机制；
- 不因为某个成功案例而扩展论文 claim。

## 8. 第一周决策

```text
P0：环境与 metadata
P1-A：trace 对齐
P1-B：paired intervention 对齐
```

两组均通过后才进入第二周联合分析。
