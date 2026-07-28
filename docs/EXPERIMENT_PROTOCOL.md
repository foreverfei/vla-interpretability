# 统一实验协议

## 1. 目的

固定 Group A 与 Group B 共用的模型、环境、episode 对齐方式和结果字段，避免两组分别实现后无法联合分析。

## 2. Shared configuration

唯一公共配置：

```text
shared/project_config.yaml
```

学生可以创建本地覆盖配置，但不得提交绝对路径或私自修改以下字段：

- model checkpoint；
- LIBERO suite 与 task；
- initial state indices；
- random seed；
- open-loop steps；
- condition 名称；
- trace layer 规则；
- mask control 规则。

## 3. Episode pairing

一个 paired group 必须共享：

```text
model
checkpoint
suite
task_id
task_name
initial_state_index
random_seed
instruction
max_steps
num_open_loop_steps
```

只允许 condition 不同：

```text
baseline
target_mask
background_control
```

推荐 ID：

```text
paired_group_id = <task_id>__<initial_state>__seed<seed>
episode_id = <paired_group_id>__<condition>
```

## 4. 时间轴定义

必须区分：

- `env_step`：环境执行一步；
- `policy_query_index`：模型重新查询一次；
- `action_chunk_index`：一次 policy query 输出中的动作序号；
- `trace_index`：保存 hidden state 的 policy query 序号。

禁止用视频帧编号直接代替 policy-query index。

每个 trace row 至少包含：

```text
episode_id
policy_query_index
env_step_start
env_step_end
condition
layer_name
feature_type
feature_shape
action_chunk_shape
```

## 5. Group A trace protocol

第一阶段保存四个代表层：

```text
quarter points / 明确列出的 layer ids
```

优先保存：

- vision pooled feature；
- text pooled feature；
- joint pooled feature；
- action chunk；
- 必要的 token count 与 feature shape。

默认不保存全部 token，避免数据规模失控。若启用 full token，必须建立单独 Issue。

基础指标：

```text
layer-wise cosine drift
layer-wise L2 drift
time-wise drift
baseline-condition paired drift
action-chunk L2 / cosine distance
```

任何降维图仅用于可视化，不作为唯一统计证据。

## 6. Group B intervention protocol

Target Mask：

- 遮挡或替换 target object 区域；
- 记录 mask 来源、面积和 replacement mode；
- 不改变 instruction、proprioception、initial state 与 random seed。

Background Control：

- 与 target mask 面积尽量一致；
- 不覆盖机器人、目标物体、目标容器和其他关键对象；
- 记录面积误差：

```text
abs(area_background - area_target) / area_target
```

当前容差由配置固定为 10%。

基础行为指标：

```text
action-chunk L2 distance
action-chunk cosine distance
trajectory divergence
success change
episode length change
first divergence policy-query
failure stage
```

## 7. Success 与 failure stage

除最终 `success` 外，至少记录失败阶段：

```text
perception_or_approach
grasp
transport
place
timeout
unknown
```

失败阶段必须根据可复核规则或人工审查记录，不允许仅通过主观描述填写。

## 8. Cross-group join

跨组主键：

```text
paired_group_id
episode_id
policy_query_index
condition
```

联合分析要求：

1. Group A trace 完整；
2. Group B paired rollout 完整；
3. baseline/target/background 的 policy-query 能够按规则对齐；
4. 对长度不一致 episode 明确采用截断、事件对齐或不比较策略；
5. 任何缺失值必须显式报告。

## 9. Minimum report

每个正式实验必须产生：

```text
config.yaml
command.txt
git_commit.txt
episode_manifest.jsonl
per_episode_results.csv
metrics.json
figures/
failure_cases/
result_summary.md
```

## 10. 禁止事项

- 不允许不同 condition 使用不同 checkpoint；
- 不允许为获得更显著结果而修改 prompt；
- 不允许忽略失败 episode；
- 不允许只报告均值而不保留逐 episode 结果；
- 不允许将 attention/embedding 可视化直接表述为因果解释；
- 不允许在 metadata 不完整时进入正式统计。
