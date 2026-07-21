# Group A：开始执行

## 第一步：拉取本仓库

```bash
git clone https://github.com/foreverfei/vla-interpretability.git
cd vla-interpretability
git switch group-a
```

## 第二步：拉取基础代码

```bash
bash groupA/scripts/bootstrap_repos.sh
```

脚本会拉取：

```text
OpenVLA-OFT
LIBERO
VLA-Trace
mechanistic-steering-vlas
```

## 第三步：安装环境

先严格按照 OpenVLA-OFT 官方文档安装基础环境：

```text
groupA/third_party/openvla-oft/SETUP.md
groupA/third_party/openvla-oft/LIBERO.md
```

LIBERO 安装：

```bash
pip install -e groupA/third_party/LIBERO
pip install -r groupA/third_party/openvla-oft/experiments/robot/libero/libero_requirements.txt
```

## 第四步：检查环境

```bash
python groupA/scripts/check_setup.py
```

必须确认：

- Python 版本不低于 3.10；
- PyTorch 可导入；
- CUDA 可用；
- 四个基础仓库均存在；
- checkpoint 可以下载或已缓存。

## 第五步：跑通官方 baseline

在 `openvla-oft` 根目录运行：

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object \
  --num_trials_per_task 1 \
  --seed 7 \
  --center_crop True \
  --run_id_note groupA_smoke_test
```

注意：官方脚本默认遍历整个 suite。第一周需要由研究生 A 增加 `task_id` 和 `initial_state_indices` 过滤，避免一次运行全部任务。

## 第六步：加入 hidden-state hook

研究生 A 需要完成：

```text
定位 Transformer block
注册 forward hook
保存 4 个代表层
记录 policy query step
记录 action chunk index
```

保存格式见：

```text
groupA/docs/TRACE_FORMAT.md
```

## 第七步：第一周验收

```text
3 条 baseline rollout
1 条完整 hidden-state trace
4 个代表层
1 张 layer-wise drift 图
```

## 遇到问题时提交的信息

```text
完整运行命令
Python / CUDA / PyTorch 版本
GPU 型号
checkpoint 名称
Git commit
完整错误日志
```
