# Group B：基础代码、模型与数据链接

## P0：第一周必须使用

### OpenVLA-OFT

- 代码：https://github.com/moojink/openvla-oft
- 安装文档：https://github.com/moojink/openvla-oft/blob/main/SETUP.md
- LIBERO 评估文档：https://github.com/moojink/openvla-oft/blob/main/LIBERO.md
- 主要用途：模型加载、action chunk、LIBERO rollout、success 与视频保存。

### LIBERO

- 代码：https://github.com/Lifelong-Robot-Learning/LIBERO
- 主要用途：仿真环境、initial state、instance segmentation、success condition。
- 第一周任务集：`libero_object`
- 第一周任务：`pick_up_the_alphabet_soup_and_place_it_in_the_basket`
- 目标实例：`alphabet_soup_1`
- 目标容器：`basket_1`

### OpenVLA-OFT LIBERO-Object Checkpoint

- 模型：https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-object
- 配置：`center_crop=True`、`num_open_loop_steps=8`、`use_proprio=True`。

### VLA-Trace

- 代码：https://github.com/VLA-Trace/VLA-Trace
- 论文：https://arxiv.org/abs/2605.30117
- 重点目录：
  - `vla_trace/behavior/patchmask.py`
  - `vla_trace/behavior/attention.py`
  - `vla_trace/evaluation/`
- 主要用途：PatchMask 条件、mask 替换方式、paired rollout 与行为指标参考。

## P1：链路跑通后再看

### LIBERO-CF / Vision Overrides Language

- 论文：https://arxiv.org/abs/2602.17659
- 用途：后续 instruction counterfactual 与视觉 shortcut 分析。

### Mechanistic Steering for VLAs

- 代码：https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas
- 用途：后续从图像级干预扩展到 activation-level intervention。

### SAFE

- 代码：https://github.com/vla-safe/SAFE
- 论文：https://arxiv.org/abs/2506.09937
- 用途：后续分析干预是否产生可预测的 failure signal。

## 数据说明

### 评估所需数据

LIBERO 仿真评估主要需要：

```text
LIBERO BDDL 文件
LIBERO init_states
OpenVLA-OFT checkpoint
```

### 可选训练数据

- RLDS 数据：https://huggingface.co/datasets/openvla/modified_libero_rlds
- 第一周不下载、不训练。

## 本组不提交到 Git 的文件

```text
模型权重
Hugging Face cache
rollout 视频
完整 observation 序列
segmentation arrays
大型 NPZ / PT 文件
```

只提交配置、脚本、mask 示例缩略图、结果 CSV 和图表。
