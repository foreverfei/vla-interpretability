# VLA-Trace π0.5 复现指南

## 环境准备

```bash
# 1. 系统依赖
apt-get install -y libegl1-mesa libegl1 libgl1-mesa-glx libosmesa6

# 2. Python 环境
conda create -n vlatrace python=3.11 -y
conda activate vlatrace
pip install -r requirements.txt

# 3. 环境变量 (~/.bashrc)
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH=$HOME/.libero
export HF_ENDPOINT=https://hf-mirror.com

# 4. 代码仓库
git clone https://github.com/xxx/VLA-Trace.git ~/autodl-tmp/VLA-Trace
git clone https://github.com/Physical-Intelligence/openpi.git ~/autodl-tmp/openpi
cd ~/VLA-Trace && pip install -e .

# 5. 模型权重（HuggingFace 下载）
huggingface-cli download pepijn223/pi05_libero_31_9_non_q \
  --local-dir ~/autodl-tmp/models/pi05_libero_finetuned
# 补丁：embed_tokens 从 PaliGemma 基座提取并裁剪 [257216,2048] → [257152,2048]
# 作为 ~/autodl-tmp/models/pi05_libero_finetuned/embed_tokens.safetensors

# 6. LIBERO-10 数据集
# 下载到 ~/data/libero_10/

# 7. LIBERO 配置文件 (~/.libero/config.yaml)
benchmark_root: .../libero/libero
bddl_files: .../libero/libero/bddl_files
init_files: .../libero/libero/init_files
datasets: ~/autodl-tmp/data/libero_10
assets: .../libero/libero/assets
```

## 自定义脚本

将以下 5 个脚本放到对应位置：

| 文件                           | 路径                                    | 用途             |
| ------------------------------ | --------------------------------------- | ---------------- |
| `scripts/pi05_adapter.py`      | `VLA-Trace/examples/pi05_adapter.py`    | 标准推理适配器   |
| `scripts/pi05_adapter_ko.py`   | `VLA-Trace/examples/pi05_adapter_ko.py` | Knockout 适配器  |
| `scripts/run_input_edit.py`    | `~/autodl-tmp/run_input_edit.py`        | S3 Input Editing |
| `scripts/run_attention_iou.py` | `~/autodl-tmp/run_attention_iou.py`     | S3 Attention IoU |
| `scripts/run_patchmask.py`     | `~/autodl-tmp/run_patchmask.py`         | S3 PatchMask     |

---

## Stage 1: 表征追踪 (CKA)

### 实验概述

对比 PaliGemma 基座 (C0) 和 π0.5 微调后 (C2) 的每层表征变化。

### 步骤

```bash
# 1.1 生成 LIBERO manifest（100 samples）
python ~/gen_manifest.py

# 1.2 提取 C0 bank（PaliGemma 基座）
vla-trace collect-repr --model pi0.5 \
  --model-path ~/autodl-tmp/models/pi05_libero_finetuned \
  --manifest ~/autodl-tmp/data/libero_10_cka_samples/manifest.jsonl \
  --adapter examples.pi05_repr_adapter:create_adapter \
  --bank-output ~/autodl-tmp/data/banks/C0_bank.npz

# C2 bank 同理，使用 finetuned checkpoint
vla-trace collect-repr --model pi0.5 \
  --model-path ~/autodl-tmp/models/pi05_libero_finetuned \
  --manifest ~/autodl-tmp/data/libero_10_cka_samples/manifest.jsonl \
  --adapter examples.pi05_repr_adapter:create_adapter \
  --bank-output ~/autodl-tmp/data/banks/C2_bank.npz

# 1.3 计算 cross-modal CKA
vla-trace cka --bank ~/autodl-tmp/data/banks/C0_bank.npz \
  --output ~/autodl-tmp/runs/pi05_libero10_C0_cka/cross_modal_cka_report.json
vla-trace cka --bank ~/autodl-tmp/data/banks/C2_bank.npz \
  --output ~/autodl-tmp/runs/pi05_libero10_C2_cka/cross_modal_cka_report.json

# 1.4 计算 checkpoint drift CKA
vla-trace cka --bank ~/autodl-tmp/data/banks/C0_bank.npz \
  --reference-bank ~/autodl-tmp/data/banks/C2_bank.npz \
  --output ~/autodl-tmp/runs/pi05_libero10_drift/checkpoint_drift_cka_report.json

# 1.5 画图
vla-trace plot-cka ~/autodl-tmp/runs/pi05_libero10_C0_cka/cross_modal_cka_report.json \
  --output ~/autodl-tmp/runs/pi05_libero10_C0_cka/cross_modal_cka.png
vla-trace plot-cka ~/autodl-tmp/runs/pi05_libero10_C2_cka/cross_modal_cka_report.json \
  --output ~/autodl-tmp/runs/pi05_libero10_C2_cka/cross_modal_cka.png
vla-trace plot-cka ~/autodl-tmp/runs/pi05_libero10_drift/checkpoint_drift_cka_report.json \
  --output ~/autodl-tmp/runs/pi05_libero10_drift/checkpoint_drift_cka.png
```

### 产出

| 文件                                                | 内容                    |
| --------------------------------------------------- | ----------------------- |
| `runs/pi05_libero10_C0_cka/cross_modal_cka.png`     | C0 跨模态 CKA 热力图    |
| `runs/pi05_libero10_C2_cka/cross_modal_cka.png`     | C2 跨模态 CKA 热力图    |
| `runs/pi05_libero10_drift/checkpoint_drift_cka.png` | C0→C2 逐层 drift 折线图 |

---

## Stage 2: 因果通路追踪 (Knockout)

### 实验概述

对 π0.5 的 18 层 Gemma Expert 做逐层滑动窗口注意力遮断。8 种模式 × 18 个 center_layer × 5 trials。

### 步骤

```bash
# 2.1 生成 sweep manifest
vla-trace knockout-sweep --model pi0.5 --dataset libero_10 \
  --window-size 5 --trials 5 \
  --output ~/autodl-tmp/runs/pi05_layerwise_ko/manifest.json

# 2.2 跑全部 147 个 jobs（使用 knockout 适配器）
for i in $(seq 0 146); do
  vla-trace eval-libero --model pi0.5 --dataset libero_10 \
    --model-path ~/autodl-tmp/models/pi05_libero_finetuned \
    --data-root ~/autodl-tmp/data/libero_10 \
    --adapter-factory examples.pi05_adapter_ko:create_policy \
    --knockout-manifest ~/autodl-tmp/runs/pi05_layerwise_ko/manifest.json \
    --job-index $i \
    --task-ids 0 --num-trials-per-task 5 \
    --output-dir runs/pi05_layerwise_ko/results
done

# 2.3 补全独立全层实验（不在 manifest 中的）
# Prefill
vla-trace eval-libero --model pi0.5 --dataset libero_10 \
  --model-path ~/autodl-tmp/models/pi05_libero_finetuned \
  --data-root ~/autodl-tmp/data/libero_10 \
  --adapter-factory examples.pi05_adapter_ko:create_policy \
  --mode no_vl --phase prefill --layers all \
  --task-ids 0 --num-trials-per-task 5 \
  --output-dir runs/pi05_layerwise_ko/results

# Generation (同样方式跑 no_image, no_text)

# 2.4 画图
vla-trace plot-knockout ~/autodl-tmp/runs/pi05_layerwise_ko/results \
  --output figures/s2_prefill.png
vla-trace plot-knockout ~/autodl-tmp/runs/pi05_layerwise_ko/results \
  --output figures/s2_generation.png
vla-trace plot-knockout-line ~/autodl-tmp/runs/pi05_layerwise_ko/results \
  --output-dir figures/s2_layerwise/
```

### 产出

| 文件                           | 内容                                         |
| ------------------------------ | -------------------------------------------- |
| `figures/s2_all_layers.png`    | 全层 knockout 柱状图                         |
| `figures/s2_layerwise_key.png` | 逐层折线图（no_image/no_text/prefill_no_vl） |
| `figures/s2_layerwise_all.png` | 全部 8 种模式逐层折线图                      |

---

## Stage 3: 行为探针

### 3.1 Attention IoU (注意力定位)

```bash
python ~/autodl-tmp/run_attention_iou.py
```

脚本内部流程：

1. `SegmentationRenderEnv` 跑 rollout，hook 层 8/17 抓取注意力
2. 3D 物体坐标 → 相机投影 → 生成 16×16 ground-truth mask
3. `vla-trace attention-export` 提取 action_to_image views
4. `vla-trace attention-metrics` 计算 IoU
5. `vla-trace attention-overlay` 生成定性叠加图

```bash
# 画定量 IoU 图
vla-trace plot-attention ~/autodl-tmp/runs/pi05_attention_iou_final/all_plot_data.csv \
  --output figures/attention_iou_line.png
```

### 产出

| 文件                                    | 内容                 |
| --------------------------------------- | -------------------- |
| `figures/attention_iou_line.png`        | IoU 定量折线图       |
| `figures/attention_overlay_step100.png` | 热力图+mask 叠加图   |
| `heatmap_videos/`                       | 热力图随时间变化视频 |

---

### 3.2 PatchMask (视觉遮挡)

```bash
python ~/autodl-tmp/run_patchmask.py
```

脚本内部流程：

1. `OffScreenRenderEnv` 跑 rollout
2. 3D 物体坐标投影 → 圆区域 mask
3. `apply_patch_mask` 黑色遮挡 → 发给模型推理
4. 生成 clean + masked 视频对比

```bash
# 画 PatchMask 柱状图
vla-trace plot-knockout ~/autodl-tmp/runs/patchmask_final \
  --setting patchmask \
  --output figures/patchmask_success.png
```

### 产出

| 文件                            | 内容                             |
| ------------------------------- | -------------------------------- |
| `figures/patchmask_success.png` | 4 种 variant 成功率柱状图        |
| `videos/`                       | 每个 trial clean/masked 对比视频 |

---

### 3.3 Input Editing (指令编辑)

```bash
python ~/autodl-tmp/run_input_edit.py
```

脚本内部流程：

1. 选 task 的 baseline 指令和被修改的 edited 指令
2. 各跑 10 trials，记录成功/失败
3. 生成对比视频

```bash
# 汇总
vla-trace input-edit \
  --results ~/autodl-tmp/runs/pi05_input_edit_video/input_edit_results.jsonl \
  --output ~/autodl-tmp/runs/pi05_input_edit_video/input_edit_summary.json
```

### 产出

| 文件                      | 内容                    |
| ------------------------- | ----------------------- |
| `input_edit_summary.json` | 成功率对比汇总          |
| `videos/`                 | baseline vs edited 视频 |

---
