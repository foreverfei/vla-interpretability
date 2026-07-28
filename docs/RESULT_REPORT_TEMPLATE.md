# 实验结果报告模板

## 1. 实验信息

```text
Experiment ID:
Issue:
Group:
Author:
Date:
Git commit:
Model code commit:
Checkpoint:
Config:
Command:
```

## 2. 研究问题

本实验回答：

> 

本实验不回答：

- 

## 3. 实验设置

### 模型

```text
family:
checkpoint:
quantization:
open-loop steps:
```

### 环境

```text
benchmark:
task:
initial states:
random seeds:
max steps:
```

### 实验条件

```text
baseline:
target intervention:
background control:
```

### 表征或干预设置

```text
trace layers:
pooling:
mask source:
mask area:
replacement mode:
```

## 4. 数据完整性

| 项目 | 预期 | 实际 | 说明 |
|---|---:|---:|---|
| episodes | | | |
| paired groups | | | |
| policy queries | | | |
| complete traces | | | |
| valid masks | | | |
| missing rows | 0 | | |

## 5. 结果

逐 episode 结果文件：

```text
per_episode_results.csv
```

主要指标：

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

每张图说明对应的问题、数据范围和主要观察，不只给出可视化。

## 7. 失败和异常样本

| Episode | 现象 | 已检查原因 | 是否影响结论 |
|---|---|---|---|
| | | | |

失败样本目录：

```text
failure_cases/
```

## 8. 结论

验收建议：

```text
通过 / 补充后复验 / 不通过 / 暂停
```

主要结论：

> 

## 9. 结论范围

当前结果支持：

- 

当前结果不支持：

- 

## 10. 下一步

只列当前最必要的一项：

- 
