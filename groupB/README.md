# B 组：视觉干预与行为分析

## 1. 负责内容

B 组负责目标区域和背景区域的受控干预、配对 rollout 和行为结果分析。

主要问题：

> 在模型、任务、初始状态、随机种子和 instruction 不变时，目标区域干预是否比背景对照更稳定地改变动作和任务行为？

B 组不负责 hidden-state hook 的实现。

## 2. 当前任务

### 第一周

- 跑通 OpenVLA-OFT + LIBERO-Object；
- 生成 target mask 和 background control mask；
- 完成 3 个 initial states × 3 个 conditions 的配对 rollout；
- 保存动作块、轨迹、success 和 failure stage；
- 输出 action divergence 图、mask 可视化和失败样本。

### 第二周

- 计算 baseline-target 和 baseline-background 动作差异；
- 按平移、旋转和 gripper 分别报告动作变化；
- 定位首次动作分歧的策略查询；
- 对齐 A 组的 representation trace；
- 输出逐 episode 行为分析。

详细任务见：

- [`docs/WEEK1_PLAN.md`](../docs/WEEK1_PLAN.md)
- [`docs/WEEK2_PLAN.md`](../docs/WEEK2_PLAN.md)

## 3. 目录建议

```text
groupB/
├── src/
│   ├── masks/
│   ├── interventions/
│   ├── rollout/
│   └── metrics/
├── scripts/
├── configs/
├── tests/
└── reports/
    ├── week1/
    └── week2/
```

职责划分：

- `masks/`：target/background mask 生成和检查；
- `interventions/`：替换、遮挡和输入变换；
- `rollout/`：配对 episode 执行和保存；
- `metrics/`：动作、轨迹、success 和 failure-stage 分析；
- `scripts/`：参数解析和流程编排。

## 4. 输出格式

每个配对实验组至少输出：

```text
baseline episode
target_mask episode
background_control episode
mask_manifest.jsonl
paired_rollout_index.csv
per_episode_results.csv
```

`mask_manifest.jsonl` 至少包含：

```text
paired_group_id
episode_id
mask_type
mask_source
mask_area_ratio
replacement_mode
excluded_regions
artifact_path
```

`paired_rollout_index.csv` 至少包含：

```text
paired_group_id
episode_id
condition
task_id
initial_state_index
random_seed
success
episode_length
policy_query_count
```

## 5. 基本要求

- baseline、target、background 使用相同任务、state、seed 和 checkpoint；
- target/background mask 面积差不超过协议阈值；
- background control 不覆盖机器人、目标和关键对象；
- 干预只修改视觉输入，不修改 instruction 或 proprioception；
- 不提交视频和大规模观测；
- 所有实验产物通过 metadata 指向；
- 失败 episode 不得删除。

## 6. 验收标准

- baseline rollout 可复现；
- 3 个 states × 3 个 conditions 数据完整；
- mask control 满足面积和位置约束；
- 配对动作块可以按策略查询对齐；
- 配置、命令、代码版本和失败样本完整。

## 7. 分工

研究生 B 负责实验设置、mask 规则和 PR 审核；本科生负责 mask、配对 rollout、动作差异和失败样本整理。
