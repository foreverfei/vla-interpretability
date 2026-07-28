# 学生协作流程

## 1. 标准流程

每项任务按以下顺序执行：

```text
阅读周计划和相关论文
→ 在 Issue 中确认任务目标
→ 创建任务分支
→ 跑通最小样本
→ 提交可回滚 commit
→ 扩展正式实验
→ 完成实验报告
→ 创建 Pull Request
→ 教师验收
```

没有 Issue 时不新增实验方向。

## 2. 分支

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

任务完成后从任务分支创建 PR。小组集成分支只保留已审核工作。

## 3. Issue 更新

每个工作日结束前，在对应 Issue 中更新：

```markdown
## YYYY-MM-DD｜姓名

### 今日任务
- 

### 完成内容
- 

### 当前结果
- episodes：
- successful episodes：
- failed episodes：
- aligned policy queries：
- missing traces/masks：

### 产出路径
- 代码：
- 配置：
- 结果：
- 图表：
- 日志：

### 问题与阻塞
- 现象：
- 完整错误：
- 已尝试：
- 需要教师决定：

### 下一步
- 
```

不写“继续调试”“继续学习”“结果正常”等不可检查描述。

## 4. 问题上报

环境或代码问题至少提供：

```text
Issue number
Experiment ID
git commit
model checkpoint
model code commit
LIBERO version
config
command
full traceback
expected behavior
actual behavior
attempted fixes
relevant paths
```

实验假设未得到支持时，单独说明数据、对照和失败样本，不与环境问题混在一起。

## 5. 最小样本检查

批量运行前先完成：

```text
1 task
1 initial state
1 baseline episode
1 intervention pair
1 complete trace or mask output
```

检查：

- metadata 是否完整；
- 策略查询与环境步是否对齐；
- tensor shape、dtype 和数值范围是否正确；
- action chunk 是否完整；
- mask 可视化是否正确；
- 输出是否能由保存的命令重新生成。

## 6. 实验报告

每周报告按 `docs/RESULT_REPORT_TEMPLATE.md` 提交，至少包含：

```text
研究问题
实验设置
数据完整性
逐 episode 结果
汇总指标
失败样本
结论
结论范围
下一步
```

表述示例：

> 在 initial states 0、1、2 的配对实验中，target-mask 条件在 2/3 个状态上产生了更大的动作块距离；state 2 未复现，因此当前只报告初步趋势。

不要写：

> 模型明显依赖目标区域，证明其内部具有目标概念。

## 7. PR 前检查

```bash
python tools/validate_repository.py
pytest -q
```

同时确认：

- Issue 已更新；
- 配置、命令和 commit 已保存；
- 逐 episode 结果已保存；
- 失败样本未删除；
- 未提交模型权重和大文件；
- PR 未超出 Issue 任务范围。

## 8. 教师验收

教师依据以下材料验收：

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

验收结果统一为：

```text
通过
补充后复验
不通过
暂停
```
