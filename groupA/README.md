# A 组：表征追踪

## 1. 负责内容

A 组负责 OpenVLA-OFT 的内部表征采集、时间轴对齐和表征变化分析。

主要问题：

> 模型在不同层和不同策略查询中的表征变化，是否与视觉条件和动作输出变化一致？

A 组不负责制定 mask 和视觉干预协议。

## 2. 当前任务

### 第一周

- 跑通 OpenVLA-OFT + LIBERO-Object；
- 保存四个代表层的 hidden state；
- 保存 vision、text、joint pooled feature；
- 对齐 hidden state、策略查询、环境步和动作块；
- 输出 layer-wise drift 图和失败样本。

### 第二周

- 对齐 baseline、target、background 三种条件的 trace；
- 计算 layer × time representation drift；
- 对齐首次动作分歧；
- 输出逐 query 和逐 episode 结果。

详细任务见：

- [`docs/WEEK1_PLAN.md`](../docs/WEEK1_PLAN.md)
- [`docs/WEEK2_PLAN.md`](../docs/WEEK2_PLAN.md)

## 3. 目录建议

```text
groupA/
├── src/
│   ├── hooks/
│   ├── tracing/
│   ├── alignment/
│   └── metrics/
├── scripts/
├── configs/
├── tests/
└── reports/
    ├── week1/
    └── week2/
```

职责划分：

- `hooks/`：模型 hook 和 feature extraction；
- `tracing/`：trace 保存和索引；
- `alignment/`：策略查询、环境步和动作块对齐；
- `metrics/`：表征变化指标；
- `scripts/`：参数解析和流程编排。

## 4. 输出格式

每个 episode 至少输出：

```text
episode_metadata.json
trace_index.csv
action chunk 文件或外部存储引用
trace tensor 文件或外部存储引用
rollout_log.jsonl
```

`trace_index.csv` 至少包含：

```text
episode_id
policy_query_index
env_step_start
env_step_end
layer_name
feature_type
feature_shape
condition
artifact_path
```

## 5. 基本要求

- hook 不得改变模型 forward 输出；
- trace 保存与分析代码分开；
- 不使用视频帧编号代替策略查询序号；
- 不提交 hidden-state 大文件；
- 所有实验产物通过 metadata 指向；
- 降维图不能代替逐 episode 数值结果。

## 6. 验收标准

- baseline rollout 可复现；
- trace 数量与策略查询数量一致；
- action chunk 与 trace query 对齐；
- 三种 condition 的 trace 可以连接；
- 配置、命令、代码版本和失败样本完整。

## 7. 分工

研究生 A 负责模型接口、hook、实验设置和 PR 审核；本科生负责 rollout、trace 完整性、指标计算和失败样本整理。
