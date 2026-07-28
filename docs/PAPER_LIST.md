# VLA Interpretability Paper List

> 更新时间：2026-07-28  
> 目标：为 Representation Trace、Behavior Intervention 和后续跨组实验提供统一的论文入口、代码入口与阅读优先级。

## 1. 使用说明

### 优先级

| 标记 | 含义 |
|---|---|
| **P0** | 与当前实验直接重叠，所有成员必须阅读 |
| **P1** | 与某一小组直接相关，应在对应实验前阅读 |
| **P2** | 架构、评测或方法论参考，按任务查阅 |

### 代码状态

| 标记 | 含义 |
|---|---|
| ✅ | 已核验公开代码 |
| 🟡 | 有项目页，但代码未公开或仍在整理 |
| — | 暂未核验到公开实现 |

学生阅读论文时，不只记录摘要。每篇论文至少回答：

```text
1. 它解释的是输入、内部表示、动作输出还是闭环行为？
2. 它使用相关性、输入干预、activation intervention 还是完整闭环实验？
3. 干预与行为结果在哪个时间尺度对齐？
4. 有哪些 control condition？
5. 它能够支持什么 claim，不能支持什么 claim？
6. 哪段代码可以直接复用？
7. 与本项目的差异是什么？
```

---

# 2. 当前课题在文献中的位置

当前项目计划建立：

```text
paired environment states
→ policy-query aligned hidden-state trace
→ target / background controlled intervention
→ action-chunk and closed-loop behavior divergence
→ evidence-bounded interpretation
```

该方向合理，但已经存在多项高度相关工作。以下文献决定了本项目必须达到的最低创新边界。

| 优先级 | 工作 | 已经覆盖 | 本项目不能重复停留在 | 建议保留的差异方向 |
|---|---|---|---|---|
| P0 | [VLA-Trace](https://arxiv.org/abs/2605.30117) | CKA 表征追踪、attention knockout、rollout 行为诊断 | layer drift + 行为曲线 | 同一 policy-query 上的 target-specific causal mediation 与 first-divergence 分析 |
| P0 | [Embodied Interpretability](https://arxiv.org/abs/2605.00321) | 基于视觉区域干预的 ISS、NMR 和泛化诊断 | 目标 mask 后动作变化 | 把视觉干预、hidden state 和闭环行为在时间上严格连接 |
| P0 | [Mechanistic Interpretability for Steering VLAs](https://proceedings.mlr.press/v305/haon25a.html) | 内部语义方向发现、activation steering、因果行为改变 | 只做 hook 或 feature visualization | 使用 activation patching 证明某层/某 query 是否介导目标证据到动作的传递 |
| P0 | [SAFE](https://proceedings.neurips.cc/paper_files/paper/2025/hash/392d0d05e2f514063e6ce6f8b370834c-Abstract-Conference.html) | 从 VLA 隐状态预测跨任务 failure | hidden state 与 success/failure 相关 | 区分 failure prediction 与目标视觉证据的机制归因 |
| P0 | [When Vision Overrides Language](https://arxiv.org/abs/2602.17659) | LIBERO-CF、视觉 shortcut、语言反事实失败 | 单一任务目标遮挡 | 将视觉依赖与语言依赖分开控制，并报告目标特异 excess effect |
| P1 | [DR.VLA](https://arxiv.org/abs/2603.19183) / [project](https://drvla.github.io/) | SAE 特征发现、记忆/通用特征区分、闭环 steering | pooled feature drift | 后续可扩展为 event/query-grounded feature intervention，但不是 Week 1 必需项 |
| P1 | [Event-SAE](https://arxiv.org/abs/2605.17204) | 以行为事件为单位解释 SAE feature，并做闭环干预 | 静态 SAE 可视化 | 使用任务阶段和 first-divergence event 对齐内部机制 |

## 当前更适合的论文问题

建议后续将论文问题收缩为：

> **Which layers and policy queries causally mediate target visual evidence into action-chunk changes and downstream task failure in VLA policies?**

对应的论文级实验应包含：

1. 输入级干预：baseline / target / matched background；
2. 表示级测量：同一 policy query、同一 layer 的 hidden-state change；
3. 表示级干预：clean-to-corrupted 或 corrupted-to-clean activation patching；
4. 输出级测量：即时 action-chunk recovery；
5. 闭环级测量：first divergence、trajectory divergence、failure stage 和 success；
6. 多任务与至少第二模型/第二 checkpoint 的复核。

---

# 3. 推荐阅读顺序

## 全体成员

1. [LIBERO](https://arxiv.org/abs/2306.03310)
2. [OpenVLA](https://proceedings.mlr.press/v270/kim25c.html)
3. [OpenVLA-OFT](https://arxiv.org/abs/2502.19645)
4. [VLA-Trace](https://arxiv.org/abs/2605.30117)
5. [Embodied Interpretability](https://arxiv.org/abs/2605.00321)
6. [Mechanistic Interpretability for Steering VLAs](https://proceedings.mlr.press/v305/haon25a.html)

## Group A：Representation Trace

1. VLA-Trace
2. Mechanistic Interpretability for Steering VLAs
3. SAFE
4. DR.VLA
5. Event-SAE
6. CKA
7. Activation Patching Best Practices

## Group B：Behavior Intervention

1. Embodied Interpretability
2. When Vision Overrides Language / LIBERO-CF
3. VLA-Trace
4. LIBERO-Occ
5. TrustVLA
6. SAFE

---

# 4. 已发表会议论文

## 4.1 VLA 模型与实验基座

| 优先级 | 年份 / 会议 | 论文 | 关键 contribution | 与本项目关系 | 代码 |
|---|---|---|---|---|---|
| P0 | NeurIPS 2023 Datasets & Benchmarks | [LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html) | 提供多任务机器人操作 benchmark、initial states、演示和 success condition | 当前 simulation、task pairing 和 failure-stage 的环境基础 | ✅ [Lifelong-Robot-Learning/LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO) |
| P0 | CoRL 2024 | [OpenVLA: An Open-Source Vision-Language-Action Model](https://proceedings.mlr.press/v270/kim25c.html) | 开源 7B VLA，将视觉语言预训练迁移到机器人动作预测 | 当前 OpenVLA-OFT 的模型基础；理解 token、层和 action decoding | ✅ [openvla/openvla](https://github.com/openvla/openvla) |
| P0 | RSS 2025 | [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://arxiv.org/abs/2502.19645) | 并行 decoding、连续动作、action chunk、L1 regression 的 OFT recipe | 当前 checkpoint、policy query 与 action chunk 定义的直接来源 | ✅ [moojink/openvla-oft](https://github.com/moojink/openvla-oft) |
| P1 | RSS 2024 | [Octo: An Open-Source Generalist Robot Policy](https://www.roboticsproceedings.org/rss20/p090.html) | 模块化 transformer diffusion policy，支持多机器人和灵活任务条件 | 后续验证解释协议能否迁移到非自回归 action head | ✅ [octo-models/octo](https://github.com/octo-models/octo) |
| P1 | CoRL 2025 | [π0.5: a Vision-Language-Action Model with Open-World Generalization](https://proceedings.mlr.press/v305/black25a.html) | 异构任务协同训练、knowledge insulation、长程开放世界泛化 | 多项最新解释工作使用的第二模型；适合后期跨架构验证 | ✅ [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi) |
| P2 | CoRL 2024 | [Robotic Control via Embodied Chain-of-Thought Reasoning](https://arxiv.org/abs/2407.08693) | 生成 task/plan/subtask/视觉 grounding 后再预测动作 | 显式 reasoning 可作为内部表征解释的外部对照，但不能视为 faithful explanation | ✅ [MichalZawalski/embodied-CoT](https://github.com/MichalZawalski/embodied-CoT) |
| P2 | ICLR 2026 | [HAMLET: Switch Your VLA into a History-Aware Policy](https://openreview.net/forum?id=KcJ9U0x6kO) | moment tokens + memory module，引入历史条件 | 当前 policy-query 时间对齐、历史状态和长程误差传播的重要参考 | ✅ [myungkyuKoo/HAMLET-Isaac-GR00T](https://github.com/myungkyuKoo/HAMLET-Isaac-GR00T) |
| P2 | CVPR 2026 | [Action-Sketcher: From Reasoning to Action via Visual Sketches](https://openaccess.thecvf.com/content/CVPR2026/html/Tan_Action-Sketcher_From_Reasoning_to_Action_via_Visual_Sketches_for_Robotic_CVPR_2026_paper.html) | 使用可编辑视觉 sketch 外显空间意图和分步计划 | 外显解释路线对照：可读中间变量不等同于内部机制解释 | 🟡 [project / code coming soon](https://action-sketcher.github.io/) |

## 4.2 VLA 可解释性、干预与安全诊断

| 优先级 | 年份 / 会议 | 论文 | 关键 contribution | 与本项目关系 | 代码 |
|---|---|---|---|---|---|
| P0 | CoRL 2025 | [Mechanistic Interpretability for Steering Vision-Language-Action Models](https://proceedings.mlr.press/v305/haon25a.html) | 识别 VLA 内部速度/方向等语义方向，并通过 activation steering 改变行为 | 当前 activation intervention、feature hook 和 causal claim 的关键基线 | ✅ [Physical-AI-Safety-Institute/mechanistic-steering-vlas](https://github.com/Physical-AI-Safety-Institute/mechanistic-steering-vlas) |
| P0 | NeurIPS 2025 | [SAFE: Multitask Failure Detection for Vision-Language-Action Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/392d0d05e2f514063e6ce6f8b370834c-Abstract-Conference.html) | 从 VLA latent feature 训练跨任务 failure detector，并用 conformal prediction 校准 | Group A 的 failure-related representation baseline；证明可预测不代表可解释 | ✅ [vla-safe/SAFE](https://github.com/vla-safe/SAFE) |
| P1 | NeurIPS 2025 Spotlight | [SafeVLA: Towards Safety Alignment of VLA Models](https://proceedings.neurips.cc/paper_files/paper/2025/hash/e185c7be603426028c32ae1003a59d78-Abstract-Conference.html) | CMDP、安全行为 elicitation、SafeRL 和 Safety-CHORES | failure-stage、安全行为和成功/安全解耦的评测参考 | ✅ [PKU-Alignment/SafeVLA](https://github.com/PKU-Alignment/SafeVLA) |
| P0 | ICML 2026 | [Embodied Interpretability: Linking Causal Understanding to Generalization in VLA Models](https://arxiv.org/abs/2605.00321) | 将视觉—动作 attribution 建模为干预估计，提出 ISS 和 NMR | 与 Group B 最直接重叠；必须作为 mask、control 和统计基线 | 🟡 [project](https://robot-future.github.io/vla-explain/)；代码待公开 |

## 4.3 通用机制可解释方法

| 优先级 | 年份 / 会议 | 论文 | 关键 contribution | 与本项目关系 | 代码 |
|---|---|---|---|---|---|
| P1 | ICML 2019 | [Similarity of Neural Network Representations Revisited](https://proceedings.mlr.press/v97/kornblith19a.html) | 提出 CKA，用于比较不同层和模型表示 | Group A layer-wise representation similarity 的基础 | ✅ [project / reference code](https://cka-similarity.github.io/) |
| P1 | ICML 2018 | [Interpretability Beyond Feature Attribution: TCAV](https://proceedings.mlr.press/v80/kim18d.html) | 用概念方向与方向导数量化高层概念敏感性 | 可用于目标、背景、阶段概念的 probe；必须配合干预验证 | ✅ [tensorflow/tcav](https://github.com/tensorflow/tcav) |
| P1 | NeurIPS 2022 | [Locating and Editing Factual Associations in GPT](https://papers.nips.cc/paper_files/paper/2022/hash/6f1d43d5a82a37e89b0665b33bf3a182-Abstract-Conference.html) | causal tracing 定位中间层机制，并用模型编辑验证 | activation patching / mediation recovery 的经典设计参考 | ✅ [kmeng01/rome](https://github.com/kmeng01/rome) |
| P1 | ICLR 2025 | [Scaling and Evaluating Sparse Autoencoders](https://proceedings.iclr.cc/paper_files/paper/2025/hash/42ef3308c230942d223c411adf182c88-Abstract-Conference.html) | k-sparse SAE、feature quality 和 downstream-effect 评价 | 后续 SAE 路线的训练和评价规范 | ✅ [openai/sparse_autoencoder](https://github.com/openai/sparse_autoencoder) |
| P2 | ICML 2025 | [AxBench: Steering LLMs? Even Simple Baselines Outperform Sparse Autoencoders](https://proceedings.mlr.press/v267/wu25a.html) | 系统比较 SAE、probe、steering 与简单 baseline | 防止学生默认 SAE 一定优于线性 probe 或差分方向 | 🟡 论文附代码入口；使用前核验版本 |

---

# 5. 近半年最贴切的 arXiv / 新近工作

时间窗口主要取 2026-01-28 至 2026-07-28。已正式接收 ICML/CVPR/ICLR 的工作放在会议论文部分，不重复计入。

| 优先级 | 日期 | 论文 | 关键 contribution | 与本项目关系 | 代码 |
|---|---|---|---|---|---|
| P0 | 2026-02 | [When Vision Overrides Language: Evaluating and Mitigating Counterfactual Failures in VLAs](https://arxiv.org/abs/2602.17659) | 构建 LIBERO-CF，揭示视觉 shortcut 压过语言指令，提出 CAG | 应增加视觉/语言 counterfactual control，避免把目标遮挡效应直接解释为目标理解 | ✅ [yuffish/LIBERO-CF](https://github.com/yuffish/LIBERO-CF) |
| P0 | 2026-03 | [Sparse Autoencoders Reveal Interpretable and Steerable Features in VLA Models](https://arxiv.org/abs/2603.19183) | SAE 分离通用运动特征与 episode memorization，并闭环 steering | 与 Group A 的内部表征分析高度相关；后期可做 feature-level intervention | 🟡 [project](https://drvla.github.io/)；审稿期代码入口可能变化 |
| P0 | 2026-05 | [Event-Grounded Sparse Autoencoders for VLA Policies](https://arxiv.org/abs/2605.17204) | 以抓取、移动、放置等行为事件对齐 SAE feature，并进行闭环验证 | 直接支持用任务事件和 first divergence 替代单纯时间平均 | ✅ [xc-j/Event-SAE](https://github.com/xc-j/Event-SAE) |
| P0 | 2026-05 | [VLA-Trace: Diagnosing VLA Models through Representation and Behavior Tracing](https://arxiv.org/abs/2605.30117) | representation dynamics、attention knockout、behavior tracing 的统一证据链 | 与当前两组结构高度重叠；必须复现其核心 baseline 并明确差异 | ✅ [VLA-Trace/VLA-Trace](https://github.com/VLA-Trace/VLA-Trace) |
| P1 | 2026-06 | [Revisiting Embodied Chain-of-Thought for Generalizable Robot Manipulation](https://arxiv.org/abs/2606.03784) | 发现显式 test-time CoT 不稳定，提出 reasoning-dropout 和 representation-shaping supervision | 说明“可读 reasoning”不一定是机制解释；内部表示监督可能更可靠 | 🟡 [project](https://taoshuaiz.github.io/ERVLA/)；代码待完整发布 |
| P1 | 2026-06 | [LIBERO-Occ: Evaluating and Improving VLAs under Scene-Induced Occlusion](https://arxiv.org/abs/2606.10862) | 构建遮挡 benchmark，并用 viewpoint imagination 补全视觉证据 | Group B 遮挡实验的重要 robustness 与 partial-observability 对照 | ✅ [litsh/Libero-Occ](https://github.com/litsh/Libero-Occ) |
| P1 | 2026-06 | [Steering Autoregressive VLA Policies via Action Token Intervention](https://arxiv.org/abs/2606.15021) | 直接干预 action-token 空间实现 inference-time trajectory steering | action-token intervention 与 hidden-state patching 的输出侧对照 | 🟡 [project](https://jasontchan.github.io/token-steering/) |
| P2 | 2026-06 | [ForesightSafety-VLA: A Unified Diagnostic Safety Benchmark for VLA Models](https://arxiv.org/abs/2606.27079) | 从视觉、语言和场景结构干预诊断过程安全，报告安全成本和风险暴露时间 | failure-stage、过程级风险和 unsafe success 的扩展指标参考 | — 暂未核验公开代码 |
| P2 | 2026-06 | [Training VLA Models with Dense Embodied Chain-of-Thought Supervision](https://arxiv.org/abs/2606.30552) | ZR-0 使用 dense ECoT 对齐跨 embodiment 表示，推理时可跳过 CoT | 表示塑造与可解释 reasoning 的邻近路线 | ✅ [RUCKBReasoning/ZR-0](https://github.com/RUCKBReasoning/ZR-0) |
| P1 | 2026-07 | [TrustVLA: Mechanism-Guided Inference-Time Defense Against VLA Backdoors](https://arxiv.org/abs/2607.12571) | 通过 per-token/per-layer evidence evolution、counterfactual masking 定位 causal footprint | 提供视觉区域定位、层级动态和反事实恢复的最新参考 | — 暂未核验公开代码 |

---

# 6. 方法实现与实验设计参考

## 6.1 Activation patching

- [Towards Best Practices of Activation Patching in Language Models](https://arxiv.org/abs/2309.16042)
- [How to Use and Interpret Activation Patching](https://arxiv.org/abs/2404.15255)
- [ROME / Causal Tracing](https://github.com/kmeng01/rome)
- [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens) — 仅参考 hook/cache 软件设计，不能直接套用到 OpenVLA-OFT。

本项目采用 patching 时至少报告：

```text
clean condition
corrupted condition
patched layer
patched policy query
patched token/feature scope
immediate action recovery
closed-loop recovery
negative controls
```

推荐中介恢复指标：

```text
MR(layer, query)
  = [d(action_mask, action_base) - d(action_patch, action_base)]
    / [d(action_mask, action_base) + epsilon]
```

`MR > 0` 表示 patching 恢复了部分 baseline action；必须使用 background patch、随机层和错误时间步作为负对照。

## 6.2 表征相似度

- CKA：跨层、跨 condition、跨 checkpoint 的整体表示关系；
- cosine / L2：paired sample 的局部漂移；
- linear probe：只证明信息可读出，不证明信息被策略使用；
- activation patching / knockout：用于验证该表示是否介导输出。

## 6.3 输入干预

目标 mask 至少与以下 control 比较：

```text
matched-area background
non-target object
container/destination region
robot region
random patch
blur / mean-fill / inpainting replacement
```

不能只使用 target mask 与单一 background mask，就声称解释具有因果特异性。

## 6.4 闭环时间对齐

baseline 与干预 episode 在首次 action divergence 后会进入不同环境状态。因此结果必须分成：

```text
A. first-divergence 之前：同源 observation 下的即时 policy effect
B. first-divergence query：行为分叉点
C. divergence 之后：由闭环状态差异累积产生的 downstream effect
```

后续轨迹差异不能全部归因于最初视觉干预的直接模型机制。

---

# 7. 学生论文笔记模板

每篇论文在组内报告时使用以下模板：

```markdown
# Paper title

## 基本信息
- Venue / status:
- Paper:
- Code:
- Model / benchmark:

## 研究问题

## 干预对象
- Input / token / hidden state / action / environment:

## 对齐单位
- Image / timestep / policy query / episode / task:

## 主要指标

## 最强证据

## 关键限制

## 可复用代码入口

## 与本课题关系
- 可直接复现：
- 必须作为 baseline：
- 本课题需要超越：
```

---

# 8. 维护规则

1. 优先加入正式会议论文和最近六个月直接相关 arXiv；
2. 新论文必须填写 contribution、与本课题关系和 code status；
3. 代码链接必须指向官方仓库或作者项目页；
4. 未核验代码统一标记 `—` 或 `🟡`，禁止猜测仓库地址；
5. 若论文被正式会议接收，应从 arXiv 区移动到会议区；
6. 每两周由教师或指定学生检查一次近期 VLA interpretability、steering、failure diagnosis 和 counterfactual benchmark 工作。
