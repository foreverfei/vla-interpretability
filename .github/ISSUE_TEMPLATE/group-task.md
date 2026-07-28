---
name: Group research task
about: Define one reproducible research or engineering task
title: "[Group A/B][Week N] "
labels: ""
assignees: ""
---

## 唯一问题

> 本 Issue 只回答什么问题？

## 研究线

```text
Group A / Group B / Integration / Teacher Protocol
```

## 输入与依赖

- 配置：
- 模型/checkpoint：
- 数据/任务：
- 前置 Issue/PR：

## 具体任务

- [ ]
- [ ]
- [ ]

## 最低交付物

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

## 验收 Gate

- [ ] 输出可追溯到 config/checkpoint/commit
- [ ] 保留逐 episode 结果
- [ ] 失败样本完整
- [ ] `python tools/validate_repository.py` 通过
- [ ] `pytest -q` 通过
- [ ] 当前问题得到明确 PASS / FAIL / REPEAT / STOP

## 分支

```text
group-a/weekN-task 或 group-b/weekN-task
```

## 修改目录

```text
groupA/ 或 groupB/
```

## 明确不做

- 

## 每日记录

后续更新直接追加到本 Issue，格式见 `docs/STUDENT_WORKFLOW.md`。
