# 实验结果报告模板

## 1. 实验信息

```text
Experiment ID:
Issue:
Group:
Research question:
Author:
Git commit:
Model code commit:
Checkpoint:
Config:
Command:
Date:
```

## 2. 唯一问题

本实验只回答：

> 

不回答：

- 

## 3. 设置

### Model

```text
family:
checkpoint:
quantization:
open-loop steps:
```

### Environment

```text
benchmark:
task:
initial states:
random seeds:
max steps:
```

### Conditions

```text
baseline:
target intervention:
background control:
```

### Trace / Intervention

```text
trace layers:
pooling:
mask source:
mask area:
replacement mode:
```

## 4. 数据完整性

| 项目 | 预期 | 实际 | 状态 |
|---|---:|---:|---|
| episodes | | | |
| paired groups | | | |
| policy queries | | | |
| complete traces | | | |
| valid masks | | | |
| missing rows | 0 | | |

## 5. 主要结果

### Per-episode results

文件：

```text
per_episode_results.csv
```

### Aggregate metrics

| Metric | Baseline | Target | Background | Target - Background |
|---|---:|---:|---:|---:|
| success rate | | | | |
| episode length | | | | |
| action distance | | | | |
| representation drift | | | | |

## 6. 图表

- Figure 1：
- Figure 2：
- Figure 3：

每张图必须对应一个明确问题，不只展示美观的可视化。

## 7. 失败样本

| Episode | 现象 | 可能原因 | 是否影响结论 |
|---|---|---|---|
| | | | |

失败样本路径：

```text
failure_cases/
```

## 8. 判断

```text
PASS / FAIL / REPEAT / STOP
```

理由：

> 

## 9. 结论边界

当前结果支持：

- 

当前结果不支持：

- 

## 10. 下一步

只列出一个当前必要步骤：

- 
