# Group B：开始执行

## 第一步：拉取本仓库

```bash
git clone https://github.com/foreverfei/vla-interpretability.git
cd vla-interpretability
git switch group-b
```

## 第二步：拉取基础代码

```bash
bash groupB/scripts/bootstrap_repos.sh
```

脚本会拉取：

```text
OpenVLA-OFT
LIBERO
VLA-Trace
```

## 第三步：安装环境

先严格按照 OpenVLA-OFT 官方文档安装：

```text
groupB/third_party/openvla-oft/SETUP.md
groupB/third_party/openvla-oft/LIBERO.md
```

LIBERO 安装：

```bash
pip install -e groupB/third_party/LIBERO
pip install -r groupB/third_party/openvla-oft/experiments/robot/libero/libero_requirements.txt
```

## 第四步：检查环境

```bash
python groupB/scripts/check_setup.py
```

## 第五步：跑通官方 baseline

在 `openvla-oft` 根目录运行：

```bash
python experiments/robot/libero/run_libero_eval.py \
  --pretrained_checkpoint moojink/openvla-7b-oft-finetuned-libero-object \
  --task_suite_name libero_object \
  --num_trials_per_task 1 \
  --seed 7 \
  --center_crop True \
  --run_id_note groupB_smoke_test
```

第一周需要由研究生 B 增加：

```text
task_id 过滤
initial_state_indices=[0,1,2]
condition 参数
mask hook
结果 JSONL 导出
```

## 第六步：先验证单帧 Mask

对一张 observation 生成：

```text
original.png
target_mask.png
target_masked.png
background_mask.png
background_masked.png
```

人工确认：

- Target Mask 覆盖 `alphabet_soup_1`；
- Background Mask 不覆盖目标物体；
- Background Mask 不覆盖机械臂；
- 两个 mask 面积差小于 10%。

## 第七步：运行 paired rollout

对每个 initial state 依次运行：

```text
baseline
target_mask
background_control
```

保存格式见：

```text
groupB/docs/PAIRED_ROLLOUT_FORMAT.md
```

## 第八步：第一周验收

```text
3 个 initial state
9 条 paired rollout
3 组 mask 示例
1 张 action divergence 图
```

## 遇到问题时提交的信息

```text
完整运行命令
Python / CUDA / PyTorch 版本
GPU 型号
checkpoint 名称
initial_state_index
condition
Git commit
完整错误日志
```
