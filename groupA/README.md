# Group A — Representation Trace

## 本组目标

建立 OpenVLA-OFT 的内部表征采集链路：

```text
LIBERO rollout → action chunk → hidden-state hook → trace 校验 → layer-wise 分析
```

## 从这里开始

1. 阅读 [`START_HERE.md`](START_HERE.md)。
2. 执行 `scripts/bootstrap_repos.sh` 拉取基础代码。
3. 执行 `python scripts/check_setup.py` 检查环境。
4. 按 `configs/week1.yaml` 跑通一条 baseline rollout。
5. 按 [`docs/TRACE_FORMAT.md`](docs/TRACE_FORMAT.md) 保存 hidden state。

## 文件导航

| 文件 | 用途 |
|---|---|
| [`START_HERE.md`](START_HERE.md) | 第一周执行顺序和命令 |
| [`REFERENCES.md`](REFERENCES.md) | 论文、代码、checkpoint、数据链接 |
| [`configs/week1.yaml`](configs/week1.yaml) | 本组统一实验配置 |
| [`scripts/bootstrap_repos.sh`](scripts/bootstrap_repos.sh) | 拉取外部依赖仓库 |
| [`scripts/check_setup.py`](scripts/check_setup.py) | 检查 Python、GPU 和代码目录 |
| [`docs/TRACE_FORMAT.md`](docs/TRACE_FORMAT.md) | trace 文件格式和字段要求 |

## 本周 P0

- 3 条 baseline rollout；
- 1 条完整 hidden-state trace；
- 4 个代表层；
- 1 张 layer-wise drift 图；
- 1 份 `reports/groupA_week1.md`。

## 负责人

研究生 A 负责模型、hook 和组内审核；本科生 A 负责环境与 rollout；本科生 B 负责 trace 检查与分析。
