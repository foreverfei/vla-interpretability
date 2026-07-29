"""Stage 3 IoU data collectors — attention hooks + mask capture."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from vla_trace.behavior.patchmask import _ensure_segmentation_ids


# ═══════════════════════════════════════════════════════════════════════════
# helpers
# ═══════════════════════════════════════════════════════════════════════════

def _get_transformer_layers(model):
    lm = getattr(model, "language_model", None)
    if lm is None:
        raise AttributeError("Model does not expose `language_model`")
    for attr in ("model", "transformer"):
        candidate = getattr(lm, attr, None)
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
    raise AttributeError("Cannot locate transformer layers")


def _unwrap_vla(policy) -> Any:
    """Extract the raw VLA model from any adapter wrapper.

    Handles:
      - Direct:        obj.vla, obj._model, obj.model
      - Nested:        obj.policy.vla   (PolicyRuntimeAdapter → OpenVLAOFTPolicy)
      - Nested²:       obj.policy.policy.vla
      - Fallback:      scan attributes for language_model
    """
    for attr in ("vla", "_model", "model", "_vla"):
        obj = getattr(policy, attr, None)
        if obj is not None and hasattr(obj, "language_model"):
            return obj

    inner = getattr(policy, "policy", None)
    if inner is not None:
        for attr in ("vla", "_model", "model", "_vla"):
            obj = getattr(inner, attr, None)
            if obj is not None and hasattr(obj, "language_model"):
                return obj
        inner2 = getattr(inner, "policy", None)
        if inner2 is not None:
            for attr in ("vla", "_model", "model", "_vla"):
                obj = getattr(inner2, attr, None)
                if obj is not None and hasattr(obj, "language_model"):
                    return obj
        if hasattr(inner, "language_model"):
            return inner

    for name in sorted(dir(policy)):
        if name.startswith("_"):
            continue
        try:
            obj = getattr(policy, name)
        except Exception:
            continue
        if hasattr(obj, "language_model"):
            return obj

    raise RuntimeError("Cannot locate VLA model. Add its attribute name to _unwrap_vla().")


def _ensure_eager(model) -> None:
    impl = None
    if hasattr(model, "config"):
        impl = getattr(model.config, "_attn_implementation", None) or \
               getattr(model.config, "attn_implementation", None)
    if impl is None:
        lm_cfg = getattr(getattr(model, "language_model", None), "config", None)
        if lm_cfg:
            impl = getattr(lm_cfg, "_attn_implementation", None) or \
                   getattr(lm_cfg, "attn_implementation", None)
    if impl and str(impl).lower() != "eager":
        print(f"[iou] WARNING: attention impl is '{impl}', not eager. "
              f"Attention weights will be None. Load with attn_implementation='eager'.")
    elif impl and str(impl).lower() == "eager":
        print(f"[iou] attention impl is eager — OK")


# ═══════════════════════════════════════════════════════════════════════════
# token layout  (mirrors modeling_prismatic.py _build_multimodal_attention)
# ═══════════════════════════════════════════════════════════════════════════

def compute_token_layout(
    vla, prompt_token_count: int, *, has_proprio: bool, has_diffusion: bool,
) -> dict:
    """Return start/end indices for each token group.

    Sequence:  [BOS] [visual patches] [text] [action pads] [STOP]
    """
    patches_per_image = vla.vision_backbone.get_num_patches()
    num_images = vla.vision_backbone.get_num_images_in_input()

    agentview_patches = patches_per_image
    total_image_patches = patches_per_image * num_images

    total_visual = total_image_patches
    if has_proprio:
        total_visual += 1
    if has_diffusion:
        total_visual += 1

    prefix_len = 1
    visual_start = prefix_len
    visual_end = prefix_len + total_visual
    text_end = visual_end + prompt_token_count
    action_start = text_end
    action_end = action_start + 56

    agentview_key_start = visual_start
    agentview_key_end = visual_start + agentview_patches

    return {
        "visual_start": visual_start,
        "visual_end": visual_end,
        "action_start": action_start,
        "action_end": action_end,
        "agentview_key_start": agentview_key_start,
        "agentview_key_end": agentview_key_end,
        "total_image_patches": total_image_patches,
        "total_visual": total_visual,
        "agentview_patches": agentview_patches,
        "total_tokens": action_end + 1,
    }


# ═══════════════════════════════════════════════════════════════════════════
# AttentionCollector
# ═══════════════════════════════════════════════════════════════════════════

class AttentionCollector:
    """Hooks each self_attn.forward, extracts action→agentview attention,
    averages across heads / layers / action tokens, and stores a 1-D
    [256] vector per step for vla-trace attention-metrics.
    """

    def __init__(self, layout: dict):
        self._layout = layout
        self._handles: list[tuple] = []
        self._enabled = False
        self._step = 0
        self._accum: dict[int, tuple[np.ndarray, int]] = {}

    # ── public ──────────────────────────────────────────────────────

    def attach(self, model) -> None:
        if self._enabled:
            return
        layers = _get_transformer_layers(model)
        for layer_idx, decoder_layer in enumerate(layers):
            attn = decoder_layer.self_attn
            orig = attn.forward

            def _make_hook(_orig, _lidx):
                def hooked(*args, **kwargs):
                    kwargs = dict(kwargs)
                    kwargs["output_attentions"] = True
                    result = _orig(*args, **kwargs)
                    if isinstance(result, tuple) and len(result) >= 2 and result[1] is not None:
                        self._capture(_lidx, result[1])
                    return result
                return hooked

            attn.forward = _make_hook(orig, layer_idx)
            self._handles.append((attn, orig))
        self._enabled = True

    def detach(self) -> None:
        for attn, orig in self._handles:
            attn.forward = orig
        self._handles.clear()
        self._enabled = False

    def set_step(self, step_id: int) -> None:
        self._step = step_id

    def flush(self) -> dict[int, np.ndarray]:
        result = {}
        for step, (acc_sum, acc_count) in sorted(self._accum.items()):
            result[step] = acc_sum / float(acc_count)
        self._accum.clear()
        return result

    def save(self, path: str | Path) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        maps = self.flush()
        keys = {f"step_{s:03d}": arr for s, arr in maps.items()}
        np.savez_compressed(str(out), **keys)
        return out

    # ── internal ────────────────────────────────────────────────────

    def _capture(self, layer_idx: int, attn_weights: torch.Tensor) -> None:
        L = self._layout
        sub = attn_weights[
            :, :,
            L["action_start"]:L["action_end"],
            L["agentview_key_start"]:L["agentview_key_end"],
        ]  # → [batch, heads, 56, 256]

        vec = sub.float().mean(dim=(0, 1, 2)).detach().cpu().numpy().astype(np.float32)

        step = self._step
        if step not in self._accum:
            self._accum[step] = (vec, 1)
        else:
            prev_sum, prev_count = self._accum[step]
            self._accum[step] = (prev_sum + vec, prev_count + 1)


# ═══════════════════════════════════════════════════════════════════════════
# StepMaskCollector
# ═══════════════════════════════════════════════════════════════════════════

_ROBOT_KW = ("panda", "robot", "gripper", "ur5", "iiwa", "sawyer", "jaco",
             "rethink", "mount")


class StepMaskCollector:
    """Per-step instance segmentation mask collection."""

    def __init__(self, grid_size: int = 16):
        self.grid_size = grid_size
        self._masks: dict[str, np.ndarray] = {}
        self._objects: dict[int, list[dict]] = {}

    def collect(self, step_id: int, env, obs,
                categories: dict[str, str] | None = None) -> None:
        categories = categories or {}
        seg_keys = _detect_seg_keys(obs)
        for view_name, seg_key in seg_keys:
            raw = np.asarray(obs[seg_key])
            _ensure_segmentation_ids(env)
            try:
                instances = env.get_segmentation_instances(np.array(raw, copy=True))
            except Exception as e:
                print(f"[iou] WARNING: get_segmentation_instances failed "
                      f"for {seg_key}: {e}")
                continue
            for inst_name, inst_mask in instances.items():
                mask_bool = np.asarray(inst_mask).astype(bool)
                if mask_bool.ndim == 3 and mask_bool.shape[-1] == 1:
                    mask_bool = mask_bool[:, :, 0]
                # Flip to match model's view: get_libero_image does raw[::-1, ::-1]
                mask_flipped = mask_bool[::-1, ::-1]
                mask_grid = _resize_mask(mask_flipped, self.grid_size)
                suffix = f"_{view_name}" if view_name != "agentview" else ""
                key = f"step_{step_id:03d}_{inst_name}{suffix}"
                self._masks[key] = mask_grid
                cat = categories.get(inst_name, _guess(inst_name))
                self._objects.setdefault(step_id, []).append({
                    "mask_key_suffix": f"{inst_name}{suffix}",
                    "instance_name": inst_name,
                    "category": cat,
                })

    def save_masks(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **self._masks)

    def save_objects(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = [{"step": s, "objects": objs}
                   for s, objs in sorted(self._objects.items())]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# internal helpers
# ═══════════════════════════════════════════════════════════════════════════

def _detect_seg_keys(obs) -> list[tuple[str, str]]:
    """Return [(view_name, obs_key)] for segmentation arrays only.

    LIBERO SegmentationRenderEnv produces keys like
    ``agentview_segmentation_instance``.  We require ``"seg"`` in the key
    name to avoid matching the RGB images (which also contain ``agentview``
    but have shape ``[H, W, 3]`` and are uint8).
    """
    pairs = []
    agent_key = None
    wrist_key = None
    for key, val in obs.items():
        if not isinstance(val, np.ndarray):
            continue
        if val.ndim not in (2, 3):
            continue
        if val.ndim == 3 and val.shape[-1] != 1:
            continue
        if not np.issubdtype(val.dtype, np.integer):
            continue
        lo = key.lower()
        if "seg" not in lo:
            continue
        if "agentview" in lo:
            agent_key = key
        elif "eye_in_hand" in lo or ("hand" in lo and "gripper" not in lo):
            wrist_key = key
    if agent_key:
        pairs.append(("agentview", agent_key))
    if wrist_key:
        pairs.append(("wrist", wrist_key))
    return pairs


def _resize_mask(mask: np.ndarray, grid_size: int) -> np.ndarray:
    h, w = mask.shape
    if (h, w) == (grid_size, grid_size):
        return mask
    rows = np.linspace(0, h - 1, grid_size).round().astype(int)
    cols = np.linspace(0, w - 1, grid_size).round().astype(int)
    return mask[np.ix_(rows, cols)]


def _guess(name: str) -> str:
    lo = name.lower()
    if any(k in lo for k in _ROBOT_KW):
        return "gripper" if "gripper" in lo else "robot"
    return "object"
