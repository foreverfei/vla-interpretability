# Group B — Behavior Intervention

## 本组目标

建立可复现的视觉干预实验链路：

```text
LIBERO rollout → Target / Background Mask → paired rollout → action divergence
```

## 从这里开始

1. 阅读 [`START_HERE.md`](START_HERE.md)。
2. 执行 `scripts/bootstrap_repos.sh` 拉取基础代码。
3. 执行 `python scripts/check_setup.py` 检查环境。
4. 按 `configs/week1.yaml` 跑通一条 baseline rollout。
5. 按 [`docs/PAIRED_ROLLOUT_FORMAT.md`](docs/PAIRED_ROLLOUT_FORMAT.md) 保存三种条件结果。

## 文件导航

| 文件 | 用途 |
|---|---|
| [`START_HERE.md`](START_HERE.md) | 第一周执行顺序和命令 |
| [`REFERENCES.md`](REFERENCES.md) | 论文、代码、checkpoint、数据链接 |
| [`configs/week1.yaml`](configs/week1.yaml) | 本组统一实验配置 |
| [`scripts/bootstrap_repos.sh`](scripts/bootstrap_repos.sh) | 拉取外部依赖仓库 |
| [`scripts/check_setup.py`](scripts/check_setup.py) | 检查 Python、GPU 和代码目录 |
| [`docs/PAIRED_ROLLOUT_FORMAT.md`](docs/PAIRED_ROLLOUT_FORMAT.md) | paired rollout 文件格式和对照要求 |

## 本周 P0

- 3 个相同 initial state；
- baseline / target_mask / background_control；
- 共 9 条 paired rollout；
- 3 组 mask 示例；
- 1 张 action divergence 图；
- 1 份 `reports/groupB_week1.md`。

## 负责人

研究生 B 负责环境、mask 规则、paired protocol 和组内审核；本科生 C 负责 mask 生成、paired rollout 与结果分析。
