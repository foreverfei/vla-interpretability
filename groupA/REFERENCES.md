# Group A：基础代码、模型与数据链接

## P0：第一周必须使用

### OpenVLA-OFT

- 代码：https://github.com/moojink/openvla-oft
- 安装文档：https://github.com/moojink/openvla-oft/blob/main/SETUP.md
- LIBERO 评估文档：https://github.com/moojink/openvla-oft/blob/main/LIBERO.md
- 主要用途：模型加载、action chunk、LIBERO rollout、success 与视频保存。

### LIBERO

- 代码：https://github.com/Lifelong-Robot-Learning/LIBERO
- 主要用途：仿真环境、任务、initial state、success condition。
- 第一周任务集：`libero_object`
- 第一周任务：`pick_up_the_alphabet_soup_and_place_it_in_the_basket`

### OpenVLA-OFT LIBERO-Object Checkpoint

- 模型：https://huggingface.co/moojink/openvla-7b-oft-finetuned-libero-object
- 配置：`center_crop=True`、`num_open_loop_steps=8`、`use_proprio=True`。

### VLA-Trace

- 代码：https://github.com/VLA-Trace/VLA-Trace
- 论文：https://arxiv.org/abs/2605.30117
- 重点目录：
  - `vla_trace/representations/`
  - `vla_trace/knockout/`
  - `vla_trace/behavior/`
- 主要用途：representation bank、CKA、trace schema 和可视化接口参考。

### Mechanistic Steering for VLAs

- 代码：https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas
- 主要用途：OpenVLA 内部 feature 提取、forward hook 和 activation intervention 参考。

## P1：链路跑通后再看

### SAFE

- 代码：https://github.com/vla-safe/SAFE
- 论文：https://arxiv.org/abs/2506.09937
- 用途：hidden state 到 success/failure 的简单检测 baseline。

### TransformerLens

- 代码：https://github.com/TransformerLensOrg/TransformerLens
- 用途：学习 activation cache、hook 和 activation patching 的接口设计。
- 注意：不能直接用于 OpenVLA-OFT，只参考软件设计。

### Activation Patching 方法论文

- 论文：https://arxiv.org/abs/2309.16042
- 用途：理解 clean、corrupted、patched 输入和 restoration metric。

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
- 第一周不下载、不训练；只有后续需要微调时使用。

## 本组不提交到 Git 的文件

```text
模型权重
Hugging Face cache
rollout 视频
完整 observation 序列
hidden-state tensor
大型 NPZ / PT 文件
```

只提交配置、脚本、结果表和小体积示例。
