# 学生协作手册

## 1. 开始任务

每项任务按以下顺序进行：

```text
阅读首页与周计划
→ 在 GitHub Issue 中确认唯一问题
→ 从小组分支创建任务分支
→ 完成最小样本
→ 提交可回滚 commit
→ 扩展正式实验
→ 创建 PR
→ 教师 Gate
```

禁止在没有 Issue 的情况下直接增加新实验方向。

## 2. 分支操作

Group A：

```bash
git switch group-a
git pull --rebase
git switch -c group-a/week1-trace-hook
```

Group B：

```bash
git switch group-b
git pull --rebase
git switch -c group-b/week1-paired-mask
```

任务完成后推送任务分支并创建 PR，不在小组集成分支长期堆积未审核代码。

## 3. 每日更新

在对应 Issue 中追加：

```markdown
## YYYY-MM-DD｜Group A/B｜姓名

### 今日问题
- 

### 实际完成
- 

### 当前数字
- episodes：
- successful episodes：
- failed episodes：
- aligned policy queries：
- missing traces/masks：

### 产出
- 代码：
- 配置：
- 数据：
- 图表：
- 日志：

### 当前阻塞
- 现象：
- 完整错误：
- 已尝试：
- 需要教师决定：

### 下一步
- 
```

禁止只写“继续调试”“继续学习”“结果正常”。

## 4. 问题上报

必须提供：

```text
Issue number
Exp ID
git commit
model checkpoint
model code commit
LIBERO version
data/config version
command
full traceback
expected behavior
actual behavior
attempted fixes
relevant paths
```

模型环境问题与实验假设问题需要分开报告。

## 5. 小样本优先

正式批量运行前必须通过：

```text
1 task
1 initial state
1 baseline episode
1 intervention pair
1 complete trace or mask output
```

检查：

- episode metadata；
- policy-query 对齐；
- tensor shape/dtype；
- action chunk；
- mask 可视化；
- 输出目录；
- 是否能从命令重新生成。

## 6. 周末个人报告

```markdown
# 第 N 周个人报告

## 本人负责的问题
## 输入与设置
## 完成内容
## 关键产出
## 主要数字
## 失败样本
## 当前问题
## 是否达到 Gate
## 下周建议
```

## 7. 结果写作

按以下顺序：

```text
问题
→ 实验设置
→ 逐 episode 结果
→ 汇总数字
→ 统计或对照
→ 失败样本
→ 判断
→ 结论边界
```

可接受：

> 在 initial states 0、1、2 的 paired episodes 中，target-mask 相比 background control 在 2/3 个状态上产生更大的 action-chunk 距离；state 2 未复现该趋势，因此当前结论限定为初步现象。

不可接受：

> 模型明显依赖目标区域，证明其内部具有目标概念。

## 8. PR 前检查

```bash
python tools/validate_repository.py
pytest -q
```

并确认：

- Issue 已更新；
- 配置与命令已提交；
- 大文件未进入 Git；
- 逐 episode 结果保留；
- 失败样本未删除；
- PR 只回答一个问题。

## 9. 教师验收

教师只依据仓库中以下材料验收：

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

最终状态：

```text
PASS / FAIL / REPEAT / STOP
```
