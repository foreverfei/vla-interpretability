# 项目章程：VLA Interpretability

## 1. 稳定研究目标

本项目研究 Vision-Language-Action 模型在闭环操作任务中的可解释性，重点验证：

1. 模型内部表征在 rollout 过程中如何随视觉证据和动作阶段变化；
2. 对目标区域进行受控干预时，模型动作输出和任务行为是否发生可重复变化；
3. 表征变化与行为变化能否在同一 task、initial state、seed 和 policy-query 上严格对齐；
4. 哪些结论仅是相关性证据，哪些结论能够由受控干预支持。

研究目标不因仓库治理调整而改变。

## 2. 两条研究线

### Group A：Representation Trace

回答：

> VLA 在哪些层、哪些时间步形成与目标、指令和动作相关的表示变化？

主要工作：

- OpenVLA-OFT + LIBERO rollout；
- hidden-state hook；
- vision/text/joint pooled representations；
- policy-query 与 environment-step 对齐；
- layer-wise、time-wise drift；
- 表征变化与 action chunk 的关联分析。

### Group B：Behavior Intervention

回答：

> 改变目标视觉证据后，动作与任务行为是否发生超出背景控制的变化？

主要工作：

- Target Mask；
- 等面积 Background Control Mask；
- baseline/target/background paired rollout；
- action-chunk divergence；
- trajectory、success 和 failure-stage 分析；
- mask 面积、位置和替换方式控制。

## 3. 跨组证据链

```text
同一 task / initial state / seed
        ↓
baseline 与干预 paired episodes
        ↓
严格对齐 policy-query 与 action chunk
        ↓
Group A：表征变化
Group B：行为变化
        ↓
逐 episode 联合分析
        ↓
证据边界内的解释结论
```

任何跨组分析必须通过 `paired_group_id`、task、initial state、seed、condition 和 policy-query index 对齐。

## 4. Claim 边界

项目可以支持：

- 目标视觉干预与动作变化之间的受控关联；
- 某些层或时间步对干预更敏感；
- target intervention 相比 background control 产生更强行为影响；
- 表征变化与 action divergence 在 paired episodes 中共同出现。

项目不能仅凭当前实验声称：

- 模型真正“理解”目标；
- 某个 hidden state 是动作的唯一因果原因；
- 单个 attention map 等同于解释；
- 少量 LIBERO 任务可代表所有 VLA 模型与真实机器人场景。

## 5. 阶段 Gate

| Gate | 必须证明 | 未通过时 |
|---|---|---|
| P0 | 环境、模型、配置和 metadata 可复现 | 修复基础设施 |
| P1 | rollout step、policy query、trace、action chunk 严格对齐 | 不做表征统计 |
| P2 | target/background mask 可重复且满足控制约束 | 不做干预比较 |
| P3 | paired episodes 可逐条比较表示和行为变化 | 不做跨组结论 |
| P4 | 多 initial states / tasks 上复核主要现象 | 收缩论文 claim |

## 6. 实验对象固定项

第一阶段固定：

```text
Model: OpenVLA-OFT
Benchmark: LIBERO-Object
Task: pick up alphabet soup and place it in basket
Initial states: 0 / 1 / 2
Conditions: baseline / target_mask / background_control
Open-loop steps: 8
```

任何修改必须通过独立 Issue 和 shared protocol PR，不允许学生在组内分支私自改变。

## 7. 论文级最低产出

- 一个可复现的 paired VLA rollout protocol；
- 一套 policy-query 对齐的 representation trace；
- 一套 target/background controlled intervention protocol；
- 表征变化、动作变化与任务结果的联合证据；
- 对无效指标、失败样本和结论边界的系统分析。
