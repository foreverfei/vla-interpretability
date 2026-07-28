# 代码与实验提交规范

## 1. 分支管理

- `main`：教师验收后的稳定版本；
- `group-a`、`group-b`：小组集成分支；
- 每项任务从小组分支创建短期任务分支；
- 禁止直接向 `main` 提交实验代码。

分支命名：

```text
group-a/week<N>-<task>
group-b/week<N>-<task>
integration/<task>
infra/<task>
docs/<task>
fix/<task>
```

## 2. 目录边界

- A 组默认只修改 `groupA/`；
- B 组默认只修改 `groupB/`；
- `shared/`、`docs/`、`.github/`、`tools/` 的修改需要单独说明；
- 共享字段变化必须同时更新 schema、文档和测试；
- 未沟通时不修改对方小组目录。

## 3. Issue

每项任务先建立 Issue，内容包括：

```text
任务目标
输入与依赖
任务清单
交付物
验收标准
分支名称
修改目录
本阶段范围
```

Issue 用于记录任务、进度和决策。正式结果必须进入仓库报告或 PR。

## 4. Commit

推荐格式：

```text
feat(groupA): add policy-query trace hook
feat(groupB): add target mask generator
fix(shared): align paired episode ids
test: validate metadata and query alignment
docs: update experiment report
```

避免使用：

```text
update
fix bug
continue experiment
final version
```

提交前运行：

```bash
python tools/validate_repository.py
pytest -q
```

## 5. Pull Request

每个 PR 只处理一个 Issue，正文至少包含：

```text
关联 Issue
任务目标
主要修改
运行命令
输入与配置
输出路径
主要结果
失败和已知问题
测试结果
结论范围
```

正式实验 PR 至少包含：

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

教师审核后使用 Squash and merge。跨组 PR 由两组负责人共同检查字段兼容性。

## 6. 实验记录

每次 rollout 至少记录：

```text
episode_id
paired_group_id
group
model
checkpoint
model_code_commit
suite
task_id
task_name
initial_state_index
random_seed
condition
instruction
success
episode_length
policy_query_count
config_path
config_hash
git_commit
```

表征追踪实验额外记录：

```text
trace_layers
trace_complete
trace_step_count
action_chunk_shape
token_pooling
```

视觉干预实验额外记录：

```text
mask_type
mask_area_ratio
mask_source
replacement_mode
paired_baseline_episode
```

## 7. 不提交的内容

- 模型权重和 checkpoint；
- 原始视频和大规模图像；
- hidden-state 大文件；
- 本地绝对路径、账号和密钥；
- 虚拟环境、缓存和 IDE 文件；
- 无配置、无代码版本、无法追溯的结果。

## 8. 结果表述

报告顺序：

```text
研究问题
实验设置
逐 episode 结果
汇总指标
对照和统计
失败样本
结论
结论范围
```

可接受：

> 在完全匹配 task、initial state 和 seed 的配对实验中，target-mask 条件的动作块距离高于 background control；当前样本量不足以支持跨任务结论。

不可接受：

> 实验证明模型理解了目标物体。

## 9. 阶段验收

统一使用：

```text
通过
补充后复验
不通过
暂停
```

协议或数据未通过验收时，先修复基础问题，不用增加实验数量或复杂模型代替。
