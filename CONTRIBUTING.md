# 协作与提交规范

## 1. 基本原则

- `main` 只保存教师验收后的稳定版本；
- `group-a`、`group-b` 是小组集成分支；
- 每项任务必须先建立 Issue，再创建短期任务分支；
- 一个 Issue、一个核心问题、一个 Pull Request；
- 禁止直接向 `main` 提交实验代码。

## 2. 分支命名

```text
group-a/week<N>-<task>
group-b/week<N>-<task>
infra/<task>
docs/<task>
fix/<task>
```

示例：

```text
group-a/week1-trace-hook
group-a/week2-layer-drift
group-b/week1-paired-mask
group-b/week2-action-divergence
infra/metadata-validator
```

## 3. 目录边界

- Group A 默认只修改 `groupA/`；
- Group B 默认只修改 `groupB/`；
- `shared/`、`docs/`、`.github/`、`tools/` 的修改必须单独说明兼容性影响；
- 禁止在未沟通情况下修改对方目录；
- 跨组接口变化必须同时更新 schema、文档和测试。

## 4. Issue 规则

Issue 必须包含：

```text
1. 唯一研究/工程问题
2. 输入与依赖
3. 具体任务
4. 最低交付物
5. 验收 Gate
6. 分支名称
7. 预计修改目录
8. 明确不做的内容
```

建议标题：

```text
[Group A][Week 1] Build aligned hidden-state trace
[Group B][Week 1] Build paired target/background interventions
[Teacher][P0] Lock shared protocol and metadata contract
[Integration][Week 2] Join trace and behavior evidence
```

Issue 用于记录任务与决策；实验结果必须进入仓库报告或 PR，不能只留在聊天记录中。

## 5. Commit 规则

每名学生每天至少一个可回滚 commit。推荐格式：

```text
feat(groupA): add policy-query trace hook
feat(groupB): add deterministic target mask generator
fix(shared): align episode ids across conditions
test: validate metadata and paired rollout keys
docs: update week2 results and blockers
```

禁止：

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

## 6. Pull Request 规则

PR 标题：

```text
[Group A][Week 1] Add aligned representation trace pipeline
[Group B][Week 1] Add paired intervention rollout pipeline
```

PR 正文必须包含：

```text
1. 关联 Issue
2. 本 PR 回答的唯一问题
3. 修改文件
4. 环境与运行命令
5. 输入数据与配置
6. 输出路径
7. 当前数字
8. 失败样本或已知问题
9. 自检与测试结果
10. 结论边界
```

最低可验收产物：

```text
config.yaml
command.txt
git_commit.txt
metrics.json
per_episode_results.csv
figures/
failure_cases/
result_summary.md
```

教师审核后使用 **Squash and merge**。跨组 PR 必须由两组负责人共同检查字段兼容性。

## 7. 实验记录

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

Trace 实验额外记录：

```text
trace_layers
trace_complete
trace_step_count
action_chunk_shape
token_pooling
```

Intervention 实验额外记录：

```text
mask_type
mask_area_ratio
mask_source
replacement_mode
paired_baseline_episode
```

## 8. 禁止提交

- 模型权重和 checkpoint；
- 原始视频和大规模图像；
- hidden-state 大文件；
- 本地绝对路径、账号、密钥；
- `__pycache__`、虚拟环境和 IDE 文件；
- 无配置、无 commit、无法追溯的实验结果。

## 9. 结果表述规范

按以下顺序写：

```text
问题
→ 设置
→ 结果
→ 数值证据
→ 判断
→ 失败样本
→ 结论边界
```

允许：

> 在完全匹配 task、initial state 和 seed 的 9 组 paired episodes 中，target-mask 条件的 action-chunk L2 距离高于 background control，但当前样本量不足以支持跨任务因果结论。

禁止：

> 实验证明模型理解了目标物体。

## 10. 教师决策

阶段验收只使用：

```text
PASS / FAIL / REPEAT / STOP
```

未达到当前阶段 Gate，不以增加实验数量或增加复杂模型代替协议修复。
