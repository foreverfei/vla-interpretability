# 第二周执行计划：Representation–Behavior Evidence Alignment

## 1. 本周定位

第二周不改变研究目标，也不扩大模型和任务范围。核心是将第一周两组独立产物连接起来，回答：

> 目标视觉干预引起的内部表征变化，是否与动作和任务行为变化在同一 paired episode 中稳定共现？

```text
Week 1：分别建立 trace 与 intervention pipeline
Week 2：统一对齐 → 分层度量 → control comparison → 证据 Gate
```

## 2. 前置条件

只有满足以下条件才启动第二周：

- Group A trace 与 policy-query 对齐通过；
- Group B paired intervention 对齐通过；
- 两组 episode metadata 可通过公共主键连接；
- baseline/target/background 均保留逐 episode 结果；
- 无法对齐的 episode 已明确列出。

## 3. Group A：Conditioned Representation Analysis

### 唯一问题

> 相同 policy-query 下，target intervention 引起的表示变化是否高于 background control，并呈现稳定的层级和时间结构？

### 任务

- 读取 baseline/target/background 的对齐 trace；
- 分别计算 vision/text/joint feature drift；
- 比较 target-mask drift 与 background-control drift；
- 生成 layer × time drift matrix；
- 标记 first action-divergence 前后的表示变化；
- 验证结果是否由 episode length 或 trace norm 驱动；
- 输出逐 episode 结果，不只报告均值。

### 最低交付

```text
groupA/reports/week2/
├── aligned_trace_index.csv
├── per_query_representation_drift.csv
├── per_episode_results.csv
├── metrics.json
├── figures/layer_time_drift.png
├── figures/target_vs_background.png
├── failure_cases/
└── result_summary.md
```

### Gate A2

- target/background 均能与 baseline 对齐；
- 至少 90% policy queries 有合法 trace；
- layer/time 指标定义固定；
- 结论在逐 episode 层面可复核；
- 降维图不作为唯一证据。

## 4. Group B：Paired Behavioral Effect Analysis

### 唯一问题

> Target Mask 是否比等面积 Background Control 更稳定地改变动作块、轨迹和任务结果？

### 任务

- 计算 baseline-target 与 baseline-background 的 action-chunk 距离；
- 定位 first divergence policy-query；
- 比较 trajectory divergence、success 和 failure stage；
- 分析 mask 面积误差与行为效应的关系；
- 对明显异常 episode 进行视频复核；
- 输出 target-specific excess effect：

```text
effect_target - effect_background
```

### 最低交付

```text
groupB/reports/week2/
├── paired_action_distance.csv
├── per_episode_results.csv
├── failure_stage_review.csv
├── metrics.json
├── figures/action_distance_over_time.png
├── figures/target_excess_effect.png
├── failure_cases/
└── result_summary.md
```

### Gate B2

- target/background control 均有完整 paired baseline；
- first divergence 能映射到 policy-query；
- target effect 与 background effect 分开报告；
- success episode 和 failure episode 均保留；
- 不以单个成功案例作为结论。

## 5. Integration：跨组联合分析

### 主键

```text
paired_group_id
episode_id
condition
policy_query_index
```

### 联合表

每行表示一个 paired policy query：

```text
representation_drift_vision
representation_drift_text
representation_drift_joint
action_chunk_distance
trajectory_divergence
before_or_after_first_divergence
condition
success
failure_stage
```

### 分析顺序

1. 验证两组数据覆盖率；
2. 分别报告 Group A 和 Group B 主结果；
3. 分析 representation drift 与 action divergence 的相关性；
4. 比较 target 与 background control；
5. 按 initial state 和 episode 分层；
6. 报告不一致样本：高表示变化但低行为变化，或反之；
7. 只在证据一致时提出解释性结论。

## 6. 推荐指标

### Representation

```text
cosine drift
normalized L2 drift
layer-wise relative drift
time-to-peak drift
```

### Action / Behavior

```text
action-chunk L2
per-dimension action distance
first divergence query
trajectory endpoint distance
success delta
episode length delta
```

### Cross-group

```text
Spearman correlation
paired target-vs-background difference
bootstrap confidence interval
per-episode consistency rate
```

## 7. 关键控制实验

- 相同面积、不同位置的 background mask；
- target mask 面积归一化；
- trace norm 与 drift 解耦；
- baseline 重复 rollout 检查随机波动；
- episode length 匹配或事件对齐；
- 不同 layer pooling 方式敏感性。

## 8. 本周不做

- 不训练新的解释模型；
- 不加入额外 VLA backbone；
- 不扩大到全部 LIBERO suite；
- 不将相关性直接称为因果机制；
- 不只展示 t-SNE/UMAP；
- 不隐藏 target intervention 无效的 episode。

## 9. 第二周 P2/P3 Gate

进入后续研究前必须满足：

1. target 与 background control 的 paired protocol 有效；
2. representation 和 action 能在 query 级对齐；
3. 至少一个指标显示 target-specific effect，而非普通遮挡效应；
4. 主要现象在多个 initial states 中重复出现；
5. 不一致和失败样本得到解释或明确保留；
6. 当前结果足以决定后续是扩大验证、调整指标或停止该假设。
