# VLA Interpretability

面向 Vision-Language-Action（VLA）模型可解释性的实验协作仓库。

## 1. 研究目标

本项目研究 VLA 模型在闭环操作任务中的内部表征和行为变化，重点回答：

1. 不同层和不同策略查询中的表征如何随任务执行变化；
2. 改变目标视觉区域后，动作块、轨迹和任务结果如何变化；
3. 表征变化与行为变化能否在同一配对实验中对齐；
4. 当前实验能够支持哪些结论，不能支持哪些结论。

第一阶段固定使用 OpenVLA-OFT 和 LIBERO-Object，不训练新模型，不扩大任务范围。

## 2. 小组分工

| 小组 | 目录 | 主要任务 |
|---|---|---|
| A 组：表征追踪 | `groupA/` | rollout、hidden-state hook、策略查询对齐、表征变化分析 |
| B 组：视觉干预与行为分析 | `groupB/` | target/background mask、配对 rollout、动作和任务结果分析 |
| 教师与集成 | `shared/`、`docs/` | 统一配置、数据字段、任务安排、结果验收和跨组分析 |

两组使用相同的模型、任务、初始状态、随机种子和 episode metadata。跨组分析必须通过统一主键连接。

## 3. 当前工作流程

```text
固定实验配置
→ 完成 baseline rollout
→ A 组保存并对齐内部表征
→ B 组完成目标和背景对照干预
→ 按策略查询对齐表征与动作
→ 汇总结果、失败样本和结论范围
```

## 4. 文档入口

| 文档 | 用途 |
|---|---|
| [项目说明](docs/PROJECT_CHARTER.md) | 研究目标、范围、分工和阶段安排 |
| [实验规范](docs/EXPERIMENT_PROTOCOL.md) | 模型、环境、配对实验、表征和干预字段 |
| [论文列表](docs/PAPER_LIST.md) | 相关论文、代码和阅读建议 |
| [第一周计划](docs/WEEK1_PLAN.md) | 基础环境、trace 和配对干预 |
| [第二周计划](docs/WEEK2_PLAN.md) | 表征与行为的联合分析 |
| [学生协作流程](docs/STUDENT_WORKFLOW.md) | Issue、分支、实验记录和 PR 流程 |
| [实验报告模板](docs/RESULT_REPORT_TEMPLATE.md) | 统一结果提交格式 |
| [文档表述规范](docs/STYLE_GUIDE.md) | 固定术语和文档结构 |
| [代码提交规范](CONTRIBUTING.md) | 分支、提交、测试和合并要求 |

## 5. 仓库结构

```text
vla-interpretability/
├── shared/                 # 共享配置、metadata schema 和公共代码
├── groupA/                 # A 组代码与报告
├── groupB/                 # B 组代码与报告
├── docs/                   # 项目说明、周计划、论文列表和模板
├── tools/                  # 仓库检查工具
├── tests/                  # 公共协议测试
├── .github/                # Issue、PR 模板和 CI
├── pyproject.toml
└── CONTRIBUTING.md
```

视频、模型权重、完整观测和 hidden-state 大文件写入 `artifacts/`、`outputs/` 或外部存储，不提交 Git。

## 6. 快速开始

```bash
git clone https://github.com/foreverfei/vla-interpretability.git
cd vla-interpretability
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python tools/validate_repository.py
pytest -q
```

小组成员先同步对应集成分支，再创建任务分支：

```bash
# A 组
git switch group-a
git merge main
git switch -c group-a/week1-trace-hook

# B 组
git switch group-b
git merge main
git switch -c group-b/week1-paired-mask
```

## 7. 协作要求

1. 每项任务先建立 Issue，再创建任务分支；
2. 每个 PR 只处理一个明确任务；
3. 实验必须保存配置、命令、代码版本、逐 episode 结果和失败样本；
4. A、B 两组默认只修改各自目录；
5. 共享字段变化必须同时更新 schema、文档和测试；
6. 教师验收后使用 Squash and merge 合并到 `main`。

## 8. 阶段验收

| 阶段 | 验收内容 |
|---|---|
| 基础环境 | 模型、环境、配置和 metadata 可复现 |
| 表征追踪 | trace、策略查询、环境步和动作块对齐 |
| 视觉干预 | target/background 配对实验满足控制条件 |
| 联合分析 | 表征和动作可以按 episode 和策略查询连接 |
| 扩展验证 | 主要现象在更多初始状态、任务或模型上复核 |

前一阶段未通过时，先修复协议和数据，不直接增加复杂分析。
