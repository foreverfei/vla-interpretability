# VLA Interpretability

面向 **Vision-Language-Action（VLA）模型可解释性** 的协作实验仓库。

本项目保持两个互补研究方向：

1. **Representation Trace**：追踪视觉、语言、联合表征与动作输出随 rollout 的变化；
2. **Behavior Intervention**：通过目标区域与背景控制干预，验证视觉证据变化是否引起动作变化。

项目不以单张图的注意力可视化作为最终结论，而是建立：

```text
paired environment states
→ aligned VLA rollouts
→ representation trace
→ controlled visual intervention
→ action / success divergence
→ evidence-bounded interpretation
```

## 1. 当前研究问题

- 哪些层、哪些时间步的表示变化与动作变化稳定相关？
- 遮挡或替换目标视觉证据后，动作块、轨迹和任务成功率如何变化？
- 目标干预与等面积背景控制是否产生可区分的效应？
- 表征变化能否作为行为变化的可验证证据，而不是仅作为可视化描述？

## 2. 双组并行任务

| 分支/目录 | 小组 | 研究线 | 当前责任 |
|---|---|---|---|
| `group-a` / `groupA/` | Group A | Representation Trace | rollout、hidden-state hook、step 对齐、layer-wise drift |
| `group-b` / `groupB/` | Group B | Behavior Intervention | target/background mask、paired rollout、action divergence |
| `main` / `shared/` | 教师集成 | Shared Protocol | 配置、schema、实验 Gate、PR 审核与跨组结果对齐 |

两组使用相同的模型、任务、initial state、随机种子和 episode metadata。只有共享键完全一致的结果才能进行跨组分析。

## 3. 文档入口

| 文档 | 用途 |
|---|---|
| [项目章程](docs/PROJECT_CHARTER.md) | 固定研究目标、问题边界与阶段 Gate |
| [统一实验协议](docs/EXPERIMENT_PROTOCOL.md) | 模型、环境、paired rollout、trace 和 mask 约束 |
| [论文列表](docs/PAPER_LIST.md) | 已发表会议论文、近半年 arXiv、代码与阅读顺序 |
| [第一周计划](docs/WEEK1_PLAN.md) | 环境跑通、trace 与 paired intervention 基线 |
| [第二周计划](docs/WEEK2_PLAN.md) | 表征—行为对齐、分层分析与控制实验 |
| [学生协作手册](docs/STUDENT_WORKFLOW.md) | Issue、分支、每日记录、PR 与验收规则 |
| [结果报告模板](docs/RESULT_REPORT_TEMPLATE.md) | 每项实验的统一提交格式 |
| [贡献规范](CONTRIBUTING.md) | 代码边界、提交、审查与合并规则 |

## 4. 仓库结构

```text
vla-interpretability/
├── shared/
│   ├── project_config.yaml
│   └── metadata_schema.json
├── groupA/
│   └── README.md
├── groupB/
│   └── README.md
├── docs/
│   ├── PROJECT_CHARTER.md
│   ├── EXPERIMENT_PROTOCOL.md
│   ├── PAPER_LIST.md
│   ├── WEEK1_PLAN.md
│   ├── WEEK2_PLAN.md
│   ├── STUDENT_WORKFLOW.md
│   └── RESULT_REPORT_TEMPLATE.md
├── tools/
│   └── validate_repository.py
├── tests/
│   └── test_repository_contract.py
├── .github/
│   ├── ISSUE_TEMPLATE/group-task.md
│   ├── pull_request_template.md
│   └── workflows/ci.yml
├── pyproject.toml
└── CONTRIBUTING.md
```

运行产生的视频、观测、hidden state、模型权重和大体积 trace 统一写入 `artifacts/`、`outputs/` 或外部存储，不提交 Git。

## 5. 快速开始

```bash
git clone https://github.com/foreverfei/vla-interpretability.git
cd vla-interpretability
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
python tools/validate_repository.py
pytest -q
```

小组工作：

```bash
# Group A
git switch group-a
git pull --rebase

# Group B
git switch group-b
git pull --rebase
```

每个具体任务仍需从小组分支创建短期任务分支，例如：

```text
group-a/week1-trace-hook
group-b/week1-paired-mask
```

## 6. 核心协作规则

1. 不直接向 `main` 提交实验代码；
2. 每项任务先建立 GitHub Issue，再创建对应分支和 Pull Request；
3. A、B 两组只修改各自目录，共享协议修改单独提交 PR；
4. 每次实验必须记录 model/checkpoint、task、initial state、condition、seed、config 和 git commit；
5. 只提交可复现代码、配置、小体积统计和报告，不提交权重与大体积产物；
6. 每个 PR 只回答一个问题，必须给出运行命令、当前数字、失败样本和结论边界；
7. 教师审核后使用 **Squash and merge** 合并到 `main`。

## 7. 阶段 Gate

```text
P0：环境、配置和 metadata 可复现
P1：trace 与 rollout step 严格对齐
P2：target intervention 与 background control 成对有效
P3：表示变化与动作变化能够逐 episode 对齐
P4：跨任务/initial state 复核后再形成论文结论
```

未通过前一阶段 Gate，不进入后续复杂分析或方法训练。
