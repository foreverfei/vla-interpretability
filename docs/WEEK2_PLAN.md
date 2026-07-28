# 第二周计划：表征与行为联合分析

## 1. 本周目标

第二周连接第一周两组结果，分析目标视觉干预引起的表征变化是否与动作和任务行为变化在同一配对实验中共同出现。

启动条件：

- A 组 trace 与策略查询对齐完成；
- B 组配对干预数据完整；
- 两组 metadata 可以通过公共主键连接；
- 缺失 episode 和缺失 query 已明确记录。

## 2. A 组任务

### 任务

- 读取 baseline、target、background 三种条件的对齐 trace；
- 计算 vision、text、joint feature drift；
- 比较 target 与 background condition；
- 生成 layer × time drift matrix；
- 标记首次动作分歧前后的表示变化；
- 检查结果是否受 episode length、feature norm 或缺失 trace 影响；
- 保留逐 query 和逐 episode 结果。

### 交付物

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

### 验收标准

- 三种条件均可与 baseline 对齐；
- 至少 90% 的策略查询有合法 trace；
- layer 和 time 指标定义固定；
- 逐 episode 结果可复核；
- 降维图不作为唯一证据。

## 3. B 组任务

### 任务

- 计算 baseline-target 和 baseline-background 的动作块距离；
- 按平移、旋转和 gripper 分别报告动作差异；
- 定位首次动作分歧的策略查询；
- 分析首次分歧前、分歧时和分歧后的轨迹变化；
- 比较 trajectory、success、episode length 和 failure stage；
- 分析 mask 面积和位置对结果的影响；
- 复核异常 episode 和失败视频。

### 交付物

```text
groupB/reports/week2/
├── paired_action_distance.csv
├── per_episode_results.csv
├── failure_stage_review.csv
├── metrics.json
├── figures/action_distance_over_time.png
├── figures/target_vs_background.png
├── failure_cases/
└── result_summary.md
```

### 验收标准

- target 和 background 均有完整 baseline 配对；
- 首次动作分歧能够映射到策略查询；
- target 和 background 结果分开报告；
- 成功和失败 episode 均保留；
- 不以单个案例作为结论。

## 4. 跨组联合分析

连接主键：

```text
paired_group_id
episode_id
condition
policy_query_index
```

联合表每行表示一个配对策略查询，至少包含：

```text
representation_drift_vision
representation_drift_text
representation_drift_joint
action_chunk_distance
first_divergence_phase
condition
success
failure_stage
```

分析顺序：

1. 报告数据覆盖率和缺失情况；
2. 分别完成 A 组和 B 组主结果；
3. 比较 target 与 background condition；
4. 分析表征变化与动作差异的关系；
5. 按 initial state 和 episode 分层；
6. 报告不一致样本；
7. 在实验范围内给出结论。

## 5. 推荐指标

### 表征

```text
cosine drift
normalized L2 drift
layer-wise relative drift
time-to-peak drift
```

### 动作与行为

```text
action-chunk L2/cosine distance
per-dimension action distance
first divergence query
trajectory endpoint distance
success change
episode length change
```

### 联合分析

```text
paired target-background difference
Spearman correlation
bootstrap confidence interval
per-episode consistency rate
```

## 6. 控制实验

- 相同面积、不同位置的 background mask；
- baseline 重复 rollout，用于估计随机波动；
- mask 面积归一化；
- feature norm 与 drift 的解耦；
- episode length 匹配或事件对齐；
- 不同 pooling 方式的敏感性检查。

## 7. 本阶段范围

第二周不做：

- 训练新的解释模型；
- 增加新的 VLA backbone；
- 扩大到全部 LIBERO suite；
- 将相关性直接称为因果机制；
- 只展示 t-SNE/UMAP；
- 隐藏无效或未复现 episode。

## 8. 阶段验收

第二周完成后，由教师给出以下结论之一：

```text
通过：数据和现象支持进入下一阶段
补充后复验：协议或样本仍需完善
不通过：当前假设未得到支持
暂停：当前方向暂不继续投入
```

进入下一阶段前，应至少满足：

1. target/background 配对协议有效；
2. 表征和动作可以按策略查询连接；
3. 至少一个指标显示 target-specific effect；
4. 主要现象在多个 initial states 中重复；
5. 失败和不一致样本已保留并说明。
