---
name: 实验或工程任务
about: 定义一项可执行、可验收的任务
title: "[Group A/B][Week N] "
labels: ""
assignees: ""
---

## 任务目标

> 本任务需要解决什么问题？

## 负责人和分组

```text
A 组 / B 组 / 跨组集成 / 共享配置
```

## 输入与依赖

- 配置：
- 模型/checkpoint：
- 数据/任务：
- 前置 Issue/PR：

## 任务清单

- [ ]
- [ ]
- [ ]

## 交付物

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

按任务实际情况删除不需要的文件，并补充专用输出。

## 验收标准

- [ ] 结果可追溯到配置、checkpoint 和 commit
- [ ] 保留逐 episode 结果
- [ ] 失败和异常样本完整
- [ ] `python tools/validate_repository.py` 通过
- [ ] `pytest -q` 通过
- [ ] 给出“通过 / 补充后复验 / 不通过 / 暂停”建议

## 分支

```text
group-a/weekN-task 或 group-b/weekN-task
```

## 修改目录

```text
groupA/ 或 groupB/
```

## 本阶段范围

- 本任务暂不处理：

## 进度记录

后续进度直接更新到本 Issue，格式见 `docs/STUDENT_WORKFLOW.md`。
