# 项目说明

## 1. 项目目标

本项目研究 Vision-Language-Action（VLA）模型在闭环操作任务中的内部表征和行为变化。

主要问题：

1. 模型在不同层、不同策略查询和不同任务阶段形成了哪些表征变化；
2. 改变目标视觉区域后，动作块、轨迹和任务结果是否发生稳定变化；
3. 表征变化与行为变化能否在同一配对实验中对齐；
4. 哪些结果属于相关性，哪些结果得到受控干预支持。

仓库治理和任务拆分不改变上述研究目标。

## 2. 研究范围

第一阶段固定：

```text
Model: OpenVLA-OFT
Benchmark: LIBERO-Object
Task: pick up the alphabet soup and place it in the basket
Initial states: 0 / 1 / 2
Conditions: baseline / target_mask / background_control
Open-loop steps: 8
```

本阶段不训练 OpenVLA，不扩大到其他任务集，不根据少量案例形成跨模型结论。

## 3. 小组分工

### A 组：表征追踪

负责：

- OpenVLA-OFT rollout；
- hidden-state hook；
- vision、text、joint pooled feature；
- 策略查询、环境步和动作块对齐；
- layer-wise 和 time-wise 表征变化分析。

### B 组：视觉干预与行为分析

负责：

- target mask；
- 等面积 background control mask；
- baseline、target、background 配对 rollout；
- 动作块、轨迹、成功率和失败阶段分析；
- mask 面积、位置和替换方式控制。

### 教师与集成

负责：

- 固定共享配置和 metadata；
- 处理跨组字段变化；
- 审核实验设置和结果；
- 组织跨组联合分析；
- 确定是否进入下一阶段。

## 4. 跨组数据关系

同一配对实验必须共享：

```text
model / checkpoint
task / instruction
initial_state_index
random_seed
num_open_loop_steps
```

只允许 `condition` 不同。

跨组连接使用：

```text
paired_group_id
episode_id
condition
policy_query_index
```

## 5. 阶段安排

| 阶段 | 主要工作 | 进入下一阶段的条件 |
|---|---|---|
| 基础环境 | 跑通模型、环境和 metadata | 配置和结果可复现 |
| 表征追踪 | 完成 hook 和时间轴对齐 | trace 与策略查询一致 |
| 视觉干预 | 完成 target/background 配对实验 | 控制条件满足协议 |
| 联合分析 | 对齐表征、动作和任务结果 | 数据覆盖率和缺失情况明确 |
| 扩展验证 | 增加任务、初始状态或模型 | 主要现象能够复核 |

## 6. 论文级预期产出

- 一套可复现的 VLA 配对 rollout 协议；
- 一套按策略查询对齐的表征追踪数据；
- 一套 target/background 视觉干预协议；
- 表征变化、动作变化和任务结果的联合分析；
- 对失败指标、未复现样本和结论范围的系统记录。

## 7. 结论范围

当前实验可以支持：

- 目标视觉干预与动作变化之间的受控关联；
- 某些层或时间步对干预更敏感；
- target condition 相比 background control 产生更强行为影响；
- 表征变化与动作变化在配对实验中共同出现。

当前实验不能直接支持：

- 模型真正“理解”目标物体；
- 某个 hidden state 是动作的唯一原因；
- 单个 attention map 或降维图等同于解释；
- 少量 LIBERO 任务可以代表全部 VLA 模型和真实机器人场景。
