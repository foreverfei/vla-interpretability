# VLA Interpretability

面向 Vision-Language-Action 模型可解释性的协作实验仓库。

## 第一周双组并行任务

| 分支 | 小组 | 任务 |
|---|---|---|
| `group-a` | Representation Trace | rollout、hidden-state hook、表征变化分析 |
| `group-b` | Behavior Intervention | Target/Background Mask、paired rollout、动作变化分析 |
| `main` | 统一集成 | 共享配置、数据格式、通过 PR 合并两组成果 |

## 快速开始

```bash
git clone https://github.com/foreverfei/vla-interpretability.git
cd vla-interpretability
```

A 组：

```bash
git switch group-a
```

B 组：

```bash
git switch group-b
```

## 协作规则

1. 不直接向 `main` 提交实验代码。
2. A、B 两组只修改各自目录；共享字段修改提交独立 PR。
3. 运行结果、模型权重和大体积 trace 不提交 Git。
4. 每次实验必须记录 task、initial state、condition、checkpoint 和 commit。
5. 每周通过 Pull Request 合并到 `main`。

详细规则见 [CONTRIBUTING.md](CONTRIBUTING.md)。
