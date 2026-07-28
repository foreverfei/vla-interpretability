# Group B — Behavior Intervention

## 1. 研究问题

> 在保持任务、initial state、seed、instruction 与 checkpoint 不变时，目标区域干预是否比等面积背景控制更稳定地改变动作与任务行为？

Group B 负责干预协议与行为测量，不负责定义 hidden-state hook。

## 2. 第一周目标

- 跑通 OpenVLA-OFT + LIBERO-Object；
- 生成 Target Mask 与 Background Control Mask；
- 完成 3 个 initial states × 3 个 conditions 的 paired rollout；
- 保存 action chunks、trajectory、success 和 failure stage；
- 输出第一张 action divergence 图；
- 保存 mask 可视化和失败样本。

详细任务见 [`docs/WEEK1_PLAN.md`](../docs/WEEK1_PLAN.md)。

## 3. 第二周目标

- 计算 baseline-target 与 baseline-background 动作差异；
- 定位 first divergence policy-query；
- 比较 target-specific excess effect；
- 对齐 representation trace；
- 输出逐 episode 行为效应和失败阶段分析。

详细任务见 [`docs/WEEK2_PLAN.md`](../docs/WEEK2_PLAN.md)。

## 4. 目录结构

```text
groupB/
├── src/
│   ├── masks/
│   ├── interventions/
│   ├── rollout/
│   └── metrics/
├── scripts/
│   ├── generate_masks.py
│   ├── run_paired_rollouts.py
│   ├── validate_pairing.py
│   └── analyze_action_divergence.py
├── configs/
├── tests/
└── reports/
    ├── week1/
    └── week2/
```

代码职责：

- `masks/`：target/background mask 生成与验证；
- `interventions/`：替换、遮挡和输入变换；
- `rollout/`：paired episode 编排和保存；
- `metrics/`：action、trajectory、success 与 failure-stage 分析；
- `scripts/`：只做参数解析和流程编排。

## 5. 输出契约

每个 paired group 至少输出：

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

## 6. 代码约束

- baseline/target/background 必须共享任务、state、seed 和 checkpoint；
- target/background mask 面积差不得超过协议阈值；
- background control 不得覆盖机器人、目标和关键对象；
- 干预只改变视觉输入，不修改 instruction 或 proprioception；
- 不提交视频和大规模观测；
- 所有 artifact 必须通过 metadata 指向；
- 失败 episode 不得删除。

## 7. 验收 Gate

```text
P0：baseline rollout 可复现
P1：3 states × 3 conditions 完整
P2：mask control 约束通过
P3：paired action chunks 可按 query 对齐
```

## 8. 负责人

研究生 B 负责实验协议、mask 规则、组内审核与 PR review；本科生负责 mask、paired rollout、action divergence 和失败样本分析。
