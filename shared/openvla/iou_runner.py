#!/usr/bin/env python
"""Stage 3 IoU data-collection runner.

Collects attention_maps.npz + step_masks.npz + step_objects.json +
metadata.json + obs/ during a single LIBERO episode.

Usage::

    python iou_runner.py \
        --model-path /path/to/ckpt \
        --data-root /path/to/LIBERO \
        --adapter-factory my_policy:create_policy \
        --model OpenVLA --dataset libero_10 \
        --unnorm-key libero_10_no_noops \
        --task-id 0 --episode-id 0 --max-steps 300 \
        --output-dir runs/stage3_iou/task0_ep0

    vla-trace attention-metrics \
        --attention runs/stage3_iou/task0_ep0/attention_maps.npz \
        --masks     runs/stage3_iou/task0_ep0/step_masks.npz \
        --metadata  runs/stage3_iou/task0_ep0/metadata.json \
        --objects   runs/stage3_iou/task0_ep0/step_objects.json \
        --output    runs/stage3_iou/task0_ep0/attention_metrics.csv \
        --summary   runs/stage3_iou/task0_ep0/attention_summary.json \
        --plot-summary runs/stage3_iou/task0_ep0/attention_plot.csv

    vla-trace plot-attention runs/stage3_iou/task0_ep0/attention_plot.csv \
        --metric iou_top10_gt --output figures/iou_task0_ep0.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

from iou_collectors import (
    AttentionCollector,
    StepMaskCollector,
    _unwrap_vla,
    compute_token_layout,
)


def main() -> int:
    args = _parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "obs").mkdir(parents=True, exist_ok=True)

    _seed_everything(args.seed)

    # ── 1. LIBERO ──────────────────────────────────────────────────

    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import SegmentationRenderEnv

    _ensure_libero_config(args)
    bd = benchmark.get_benchmark_dict()
    suite = bd[args.dataset]()
    task = suite.get_task(args.task_id)
    states = suite.get_task_init_states(args.task_id)

    env = SegmentationRenderEnv(
        bddl_file_name=os.path.join(
            get_libero_path("bddl_files"),
            task.problem_folder,
            task.bddl_file,
        ),
        camera_heights=256,
        camera_widths=256,
        camera_segmentations="instance",
    )
    env.seed(args.seed)

    # ── 2. policy  (baseline mode — SDPA, no knockout) ───────────

    from vla_trace.evaluation.adapters import (
        LiberoStep, PolicyBuildRequest, build_policy,
    )
    from vla_trace.evaluation.libero import (
        get_libero_dummy_action, get_libero_image,
    )

    # Baseline only: SDPA attention (works correctly).  knockout_config=None
    # means the adapter loads via openvla_utils.get_vla() with default SDPA.
    # HF's LlamaSdpaAttention supports output_attentions=True by recomputing
    # the attention matrix after the SDPA call — no eager required.
    policy_wrapper = build_policy(
        PolicyBuildRequest(
            model=args.model,
            dataset=args.dataset,
            model_path=args.model_path,
            data_root=args.data_root,
            output_dir=str(out),
            device=args.device,
            seed=args.seed,
            center_crop=args.center_crop,
            unnorm_key=args.unnorm_key,
            knockout_config=None,
        ),
        adapter_factory=args.adapter_factory,
    )

    # Unwrap PolicyRuntimeAdapter → real policy
    inner_policy = getattr(policy_wrapper, "policy", policy_wrapper)

    # ── 3. token layout ────────────────────────────────────────────

    vla = _unwrap_vla(inner_policy)

    processor = inner_policy.processor
    prompt = f"In: What action should the robot take to {task.language.lower()}?\nOut:"
    prompt_token_count = int(
        processor.tokenizer(prompt, return_tensors="pt").input_ids.shape[-1]
    )

    has_proprio = getattr(inner_policy, "proprio_projector", None) is not None
    has_diff = (
        getattr(inner_policy, "noisy_action_projector", None) is not None
        and getattr(inner_policy, "action_head", None) is not None
        and hasattr(inner_policy.action_head, "noise_scheduler")
    )

    layout = compute_token_layout(
        vla, prompt_token_count,
        has_proprio=has_proprio, has_diffusion=has_diff,
    )
    print(
        f"[iou_runner] token layout: visual={layout['total_visual']} "
        f"text={prompt_token_count} action=56 "
        f"agentview={layout['agentview_patches']} "
        f"total={layout['total_tokens']}"
    )

    # ── 4. collectors  (SDPA handles output_attentions natively) ────

    attn_collector = AttentionCollector(layout)
    attn_collector.attach(vla)
    mask_collector = StepMaskCollector()

    # ── 5. rollout ─────────────────────────────────────────────────

    policy_wrapper.reset()
    env.reset()
    obs = env.set_init_state(states[args.episode_id])
    desc = task.language
    wait = get_libero_dummy_action(args.model)
    success, steps_taken = False, 0

    for _ in range(args.num_steps_wait):
        obs, _r, done, _i = env.step(wait)
        if done:
            success = True
            break

    if not success:
        for sid in range(args.max_steps):
            mask_collector.collect(sid, env, obs)
            _save_obs(out / "obs", sid, obs)

            attn_collector.set_step(sid)
            step = LiberoStep(
                raw_observation=obs,
                image=get_libero_image(obs, 224),
                wrist_image=None,
                state=None,
                task_description=desc,
                task_id=args.task_id,
                episode_id=args.episode_id,
                step_id=sid,
            )
            action = policy_wrapper.predict_action(step)
            obs, _r, done, _i = env.step(action.astype(float).tolist())
            steps_taken = sid + 1

            if sid % 50 == 0:
                print(f"  step {sid}/{args.max_steps}")
            if done:
                success = True
                break

    if hasattr(env, "close"):
        env.close()
    attn_collector.detach()

    # ── 6. save artifacts ──────────────────────────────────────────

    print(f"\n[iou_runner] success={success}  steps={steps_taken}")

    attn_path = attn_collector.save(str(out / "attention_maps.npz"))
    print(f"[iou_runner] → {attn_path}")

    mask_collector.save_masks(str(out / "step_masks.npz"))
    print(f"[iou_runner] → {out / 'step_masks.npz'}")

    mask_collector.save_objects(str(out / "step_objects.json"))
    print(f"[iou_runner] → {out / 'step_objects.json'}")

    _write_json(out / "metadata.json", {
        "trace_id": f"{args.model}_{args.dataset}_task{args.task_id}_ep{args.episode_id}",
        "model": args.model,
        "dataset": args.dataset,
        "task_id": args.task_id,
        "episode_id": args.episode_id,
        "task_description": desc,
        "success": success,
        "total_steps": steps_taken,
    })
    print(f"[iou_runner] → {out / 'metadata.json'}")

    # ── 7. video (optional) ────────────────────────────────────────

    if args.video:
        import imageio
        attn_maps = np.load(str(attn_path))
        video_path = str(out / "attention_video.mp4")
        writer = imageio.get_writer(
            video_path, fps=args.video_fps, format="FFMPEG", mode="I",
        )
        frame_count = 0
        for step_key in sorted(attn_maps.keys()):
            step_id = int(step_key.split("_")[1])
            obs_img_path = out / "obs" / f"obs_step_{step_id:03d}.npy"
            if not obs_img_path.exists():
                continue
            img = np.load(str(obs_img_path))
            frame = _render_attention_frame(
                img, attn_maps[step_key], grid_size=16, top_percent=10.0, alpha=0.55,
            )
            writer.append_data(frame)
            frame_count += 1
        writer.close()
        print(f"[iou_runner] video: {frame_count} frames → {video_path}")

    # ── 8. next steps hint ─────────────────────────────────────────

    print(f"\n{'=' * 60}")
    print("Next: compute metrics")
    print(f"{'=' * 60}")
    print(
        f"vla-trace attention-metrics \\\n"
        f"  --attention {out}/attention_maps.npz \\\n"
        f"  --masks     {out}/step_masks.npz \\\n"
        f"  --metadata  {out}/metadata.json \\\n"
        f"  --objects   {out}/step_objects.json \\\n"
        f"  --output    {out}/attention_metrics.csv \\\n"
        f"  --summary   {out}/attention_summary.json \\\n"
        f"  --plot-summary {out}/attention_plot.csv"
    )
    print(
        f"\nvla-trace plot-attention {out}/attention_plot.csv \\\n"
        f"  --metric iou_top10_gt --output figures/iou_task{args.task_id}_ep{args.episode_id}.png"
    )
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def _save_obs(obs_dir: Path, step_id: int, obs) -> None:
    key = (
        "agentview_image" if "agentview_image" in obs
        else next((k for k in obs if "image" in k.lower()), None)
    )
    if key:
        img = np.asarray(obs[key])[::-1, ::-1]
        np.save(obs_dir / f"obs_step_{step_id:03d}.npy", img)


def _render_attention_frame(image, attn_vec, *,
                             grid_size=16, top_percent=10.0, alpha=0.55):
    """Return an RGB uint8 frame with attention heatmap overlaid.

    Parameters
    ----------
    image : np.ndarray [H, W, 3] uint8
        Background observation image.
    attn_vec : np.ndarray [256]
        Action→agentview attention vector.
    grid_size : int
        Patch grid side length (16 → ViT-L/14 on 224×224).
    top_percent : float
        Only highlight the top X% brightest attention patches.
    alpha : float
        Heatmap opacity.
    """
    heatmap = attn_vec.reshape(grid_size, grid_size).astype(np.float64)

    # Upsample to image resolution
    h, w = image.shape[:2]
    rows = np.linspace(0, grid_size - 1, h).round().astype(int)
    cols = np.linspace(0, grid_size - 1, w).round().astype(int)
    up = heatmap[np.ix_(rows, cols)]

    # Top-percent mask
    thresh = np.nanpercentile(heatmap, 100.0 - top_percent)
    a_mask = (up >= thresh).astype(np.float64) * alpha

    # Normalize to [0, 1] for colormap
    vmax = up.max()
    if vmax > 1e-12:
        up_norm = up / vmax
    else:
        up_norm = up

    # "hot" colormap: black → red → yellow → white
    r = np.clip(up_norm * 3.0, 0, 1)
    g = np.clip(up_norm * 3.0 - 1.0, 0, 1)
    b = np.clip(up_norm * 3.0 - 2.0, 0, 1)
    heatmap_rgb = np.stack([r, g, b], axis=-1)

    # Blend
    img_f = image.astype(np.float64) / 255.0
    a = a_mask[..., None]
    blended = img_f * (1 - a) + heatmap_rgb * a
    return np.clip(blended * 255, 0, 255).astype(np.uint8)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _ensure_libero_config(args) -> None:
    import importlib.util
    import yaml

    dr = Path(args.data_root).expanduser()
    bm = dr if (dr / "bddl_files").is_dir() else None
    if bm is None:
        spec = importlib.util.find_spec("libero")
        if spec and spec.origin:
            bm = Path(spec.origin).resolve().parent
    if bm is None:
        return
    cd = Path(args.output_dir) / ".libero_config"
    cd.mkdir(parents=True, exist_ok=True)
    os.environ["LIBERO_CONFIG_PATH"] = str(cd)
    with (cd / "config.yaml").open("w") as f:
        yaml.safe_dump(
            {
                "benchmark_root": str(bm),
                "bddl_files": str(bm / "bddl_files"),
                "init_states": str(bm / "init_files"),
                "datasets": str(dr),
                "assets": str(bm / "assets"),
            },
            f,
        )


def _parse_args():
    p = argparse.ArgumentParser(description="Stage 3 IoU data-collection runner")
    p.add_argument("--model-path", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--adapter-factory", required=True)
    p.add_argument("--model", default="openvla")
    p.add_argument("--dataset", default="libero_10")
    p.add_argument("--unnorm-key")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--task-id", type=int, default=0)
    p.add_argument("--episode-id", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--num-steps-wait", type=int, default=10)
    p.add_argument("--center-crop", action="store_true")
    p.add_argument("--video", action="store_true", help="Generate attention heatmap video after collection")
    p.add_argument("--video-fps", type=int, default=10, help="Video frame rate (default 10)")
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


if __name__ == "__main__":
    raise SystemExit(main())
