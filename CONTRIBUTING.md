# 协作规则

## 分支

- `main`：仅保存审核后的稳定版本。
- `group-a`：A 组表征追踪实验。
- `group-b`：B 组行为干预实验。

## 提交流程

```bash
git pull --rebase
git add <明确文件>
git commit -m "group-a: add hidden-state hook"
git push origin group-a
```

完成阶段任务后，从小组分支向 `main` 提交 Pull Request。

## 目录边界

- A 组只修改 `groupA/`。
- B 组只修改 `groupB/`。
- `shared/` 的修改必须单独说明兼容性影响。
- 禁止直接修改对方小组目录。

## 禁止提交

- 模型权重和 checkpoint；
- 原始视频和大规模图像；
- hidden-state 大文件；
- 本地绝对路径、账号、密钥；
- `__pycache__`、虚拟环境和 IDE 文件。

## Pull Request 最低内容

- 本次解决的问题；
- 运行命令；
- 主要输出路径；
- 已完成的测试；
- 尚未解决的问题。
