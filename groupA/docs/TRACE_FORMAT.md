# Group A：Trace 文件格式

## Episode 目录

```text
groupA/artifacts/{episode_id}/
├── metadata.json
├── steps.jsonl
├── actions/
│   └── action_chunks.npz
├── hidden_states/
│   ├── policy_query_000000.npz
│   └── ...
├── observations/
└── rollout.mp4
```

## Episode ID

```text
groupA_libero_object_t0_init0_baseline
```

必须使用 `initial_state_index`，不要仅使用随机 seed 表示初始场景。

## metadata.json 必须字段

```json
{
  "episode_id": "groupA_libero_object_t0_init0_baseline",
  "group": "groupA",
  "model": "openvla_oft",
  "checkpoint": "moojink/openvla-7b-oft-finetuned-libero-object",
  "suite": "libero_object",
  "task_id": 0,
  "task_name": "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
  "initial_state_index": 0,
  "random_seed": 7,
  "condition": "baseline",
  "success": true,
  "episode_length": 120,
  "num_policy_queries": 15,
  "num_open_loop_steps": 8,
  "git_commit": "..."
}
```

## steps.jsonl

每个环境 step 一行：

```json
{
  "env_step": 10,
  "policy_query_index": 0,
  "action_chunk_offset": 0,
  "action_index": 0,
  "done": false
}
```

## hidden-state NPZ

每次 policy query 保存一个文件：

```text
policy_query_000000.npz
```

第一周至少包含：

```text
vision_pooled/layer_08
vision_pooled/layer_16
vision_pooled/layer_24
vision_pooled/layer_31
text_pooled/layer_08
...
joint_pooled/layer_31
```

每个数组建议 shape：

```text
[hidden_dim]
```

第一周不默认保存所有 token 的完整 tensor。

## 对齐要求

必须能够从任意 hidden-state 文件追溯到：

```text
episode_id
policy_query_index
env_step
action chunk
initial_state_index
```

## 完整性检查

判定无效的情况：

- metadata 缺失；
- policy query 数量与 hidden-state 文件数量不一致；
- action chunk 无法映射到 env step；
- layer key 不完整；
- tensor 含 NaN 或 Inf；
- episode 无 checkpoint 或 commit 信息。
