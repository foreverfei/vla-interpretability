# Group A — Representation Trace

## 1. 研究问题

> VLA 在哪些层、哪些 policy-query 和哪些任务阶段产生与视觉条件及动作输出相关的表示变化？

Group A 负责表征追踪与分析，不负责定义行为干预协议。

## 2. 第一周目标

- 跑通 OpenVLA-OFT + LIBERO-Object；
- 保存 4 个代表层的 hidden state；
- 保存 vision/text/joint pooled feature；
- 保证 hidden state、policy query、env step 与 action chunk 对齐；
- 输出第一张 layer-wise drift 图；
- 形成完整 trace index 与失败样本记录。

详细任务见 [`docs/WEEK1_PLAN.md`](../docs/WEEK1_PLAN.md)。

## 3. 第二周目标

- 对齐 baseline/target/background traces；
- 计算 layer × time representation drift；
- 比较 target intervention 与 background control；
- 将表示变化与 first action divergence 对齐；
- 输出逐 episode 联合分析表。

详细任务见 [`docs/WEEK2_PLAN.md`](../docs/WEEK2_PLAN.md)。

## 4. 目录结构

```text
groupA/
├── src/
│   ├── hooks/
│   ├── tracing/
│   ├── alignment/
│   └── metrics/
├── scripts/
│   ├── run_baseline_rollout.py
│   ├── collect_trace.py
│   ├── validate_trace_alignment.py
│   └── analyze_representation_drift.py
├── configs/
├── tests/
└── reports/
    ├── week1/
    └── week2/
```

实际目录可逐步创建，但代码职责必须保持分离：

- `hooks/`：模型 hook 与 feature extraction；
- `tracing/`：trace writer、index 和 storage；
- `alignment/`：policy query / env step / action chunk 对齐；
- `metrics/`：drift 与跨条件比较；
- `scripts/`：只做参数解析和流程编排。

## 5. 输出契约

每个 episode 至少输出：

```text
episode_metadata.json
trace_index.csv
action_chunks.npy 或外部存储引用
trace tensors 或外部存储引用
rollout_log.jsonl
```

`trace_index.csv` 每行至少包含：

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

## 6. 代码约束

- hook 不得修改模型 forward 输出；
- trace storage 与分析代码解耦；
- 禁止用视频帧编号代替 policy-query index；
- 不提交 hidden-state 大文件；
- 所有 artifact 必须通过 metadata 指向；
- 降维可视化不能代替逐 episode 数值结果。

## 7. 验收 Gate

```text
P0：baseline rollout 可复现
P1：trace count == policy query count
P2：action chunk 与 trace query 完全对齐
P3：target/background paired trace 可连接
```

## 8. 负责人

研究生 A 负责模型、hook、协议审核与 PR review；本科生负责 rollout、trace integrity、指标和失败样本分析。
