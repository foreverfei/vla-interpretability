# Group B：Paired Rollout 文件格式

## Paired 实验单位

一个 pair 由相同 task 和相同 initial state 下的三条 rollout 组成：

```text
baseline
target_mask
background_control
```

## Episode ID

```text
groupB_libero_object_t0_init0_baseline
groupB_libero_object_t0_init0_target_mask
groupB_libero_object_t0_init0_background_control
```

## 目录结构

```text
groupB/artifacts/pairs/task0_init0/
├── pair_metadata.json
├── baseline/
│   ├── metadata.json
│   ├── actions.npz
│   └── rollout.mp4
├── target_mask/
│   ├── metadata.json
│   ├── actions.npz
│   └── rollout.mp4
├── background_control/
│   ├── metadata.json
│   ├── actions.npz
│   └── rollout.mp4
└── mask_preview/
    ├── original.png
    ├── target_mask.png
    ├── target_masked.png
    ├── background_mask.png
    └── background_masked.png
```

## pair_metadata.json 必须字段

```json
{
  "pair_id": "libero_object_t0_init0",
  "task_id": 0,
  "task_name": "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
  "initial_state_index": 0,
  "random_seed": 7,
  "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-object",
  "target_instance": "alphabet_soup_1",
  "target_mask_area": 1320,
  "background_mask_area": 1288,
  "area_relative_error": 0.024,
  "conditions": [
    "baseline",
    "target_mask",
    "background_control"
  ],
  "git_commit": "..."
}
```

## 单条 metadata.json

```json
{
  "episode_id": "groupB_libero_object_t0_init0_target_mask",
  "pair_id": "libero_object_t0_init0",
  "condition": "target_mask",
  "success": false,
  "episode_length": 140,
  "num_policy_queries": 18,
  "initial_state_index": 0,
  "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-object"
}
```

## 动作数据

`actions.npz` 至少包含：

```text
raw_action_chunks
executed_actions
policy_query_steps
```

建议 shape：

```text
raw_action_chunks: [num_queries, chunk_length, action_dim]
executed_actions: [episode_length, action_dim]
policy_query_steps: [num_queries]
```

## 配对要求

三种条件必须保持一致：

```text
task_id
initial_state_index
instruction
checkpoint
center_crop
num_open_loop_steps
max_steps
```

仅允许改变图像干预条件。

## Mask 要求

- Target Mask 覆盖目标实例；
- Background Mask 不覆盖目标、容器、机器人和夹爪；
- 两个 mask 面积相对误差小于 10%；
- replacement mode 在三组 paired samples 中保持固定；
- 每次 policy query 使用相同 mask 规则。

## 无效 pair

以下任一情况出现时，该 pair 无效：

- 缺少任意一种条件；
- initial state 不一致；
- instruction 或 checkpoint 不一致；
- mask 面积误差超过 10%；
- background mask 覆盖关键物体；
- action 数组无法与 policy query step 对齐；
- 缺失 pair metadata。
