# 第一周计划：基础环境、表征追踪与配对干预

## 1. 本周目标

第一周完成实验基础设施，不训练新模型，不扩大任务范围，不形成机制性结论。

| 小组 | 本周目标 |
|---|---|
| A 组 | 跑通 baseline rollout，保存并对齐代表层 hidden state |
| B 组 | 完成 baseline、target mask、background control 配对 rollout |

固定设置见 `shared/project_config.yaml`。

## 2. A 组任务

### 任务

- 跑通 initial states 0、1、2 的 baseline rollout；
- 注册四个代表层 hook；
- 保存 vision、text、joint pooled feature；
- 保存每次策略查询生成的动作块；
- 建立 `policy_query_index ↔ env_step range ↔ trace` 对齐表；
- 验证 hook 不改变动作输出；
- 输出 layer-wise drift 图和失败样本。

### 交付物

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

### 验收标准

- 3 个 baseline episodes 均完成；
- trace 数量与策略查询数量一致；
- action chunk 与 trace 使用同一 query index；
- hidden state 无 NaN/Inf；
- hook 前后动作输出一致；
- 结果可追溯到配置、checkpoint 和 commit。

## 3. B 组任务

### 任务

- 跑通 initial states 0、1、2 的 baseline rollout；
- 生成目标物体 mask；
- 生成等面积 background control mask；
- 检查 background mask 不覆盖机器人、目标和关键对象；
- 对每个 initial state 执行 baseline、target、background 三种条件；
- 保存动作块、轨迹、success 和 failure stage；
- 输出 action divergence 图、mask 可视化和失败样本。

### 交付物

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

### 验收标准

- 3 个 initial states × 3 个 conditions 共 9 个 episodes 完成；
- 配对实验的 task、state、seed 和 checkpoint 完全一致；
- target/background mask 面积差不超过 10%；
- mask 元数据和可视化完整；
- action chunk、success 和 episode length 无缺失；
- 结果可追溯到配置、checkpoint 和 commit。

## 4. 每日安排

| 日期 | A 组 | B 组 |
|---|---|---|
| Day 1 | 环境和 baseline rollout | 环境和 baseline rollout |
| Day 2 | hook 与 feature shape 检查 | target/background mask 生成 |
| Day 3 | 完成 1 个 episode 的 trace 对齐 | 完成 1 个 initial state 的配对 rollout |
| Day 4 | 扩展到 3 个 initial states | 完成 9 个正式 episodes |
| Day 5 | 图表、失败样本和结果报告 | 图表、失败样本和结果报告 |

## 5. 跨组公共字段

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

第一周只检查字段能否连接，不进行跨组结论分析。

## 6. 本阶段范围

第一周不做：

- OpenVLA 训练或微调；
- probing classifier；
- 跨任务泛化；
- activation patching；
- 将 layer drift 直接解释为因果机制。

两组验收通过后进入第二周联合分析；未通过时先修复环境、数据和对齐问题。
