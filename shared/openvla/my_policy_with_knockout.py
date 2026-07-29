"""Stage 2 policy adapter for OpenVLA-OFT checkpoints with knockout support.

Baseline path is unchanged from the official evaluation script.  Knockout
is activated lazily the first time a non-baseline ``knockout_config`` is
received: the model is reloaded with eager attention and attention-layer
hooks are installed to merge per‑layer knockout masks.

Usage
-----
.. code-block:: bash

   # Baseline
   vla-trace eval-libero --model openvla --dataset libero_spatial \\
       --model-path /path/to/ckpt \\
       --adapter-factory examples.adapters.openvla_oft_policy_adapter:create_policy

   # Knockout
   vla-trace eval-libero --model openvla --dataset libero_spatial \\
       --model-path /path/to/ckpt \\
       --adapter-factory examples.adapters.openvla_oft_policy_adapter:create_policy \\
       --knockout-phase generation --knockout-mode no_image --knockout-layers 15
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action


# ═════════════════════════════════════════════════════════════════════════
# public factory
# ═════════════════════════════════════════════════════════════════════════
def create_policy(request: Any) -> "OpenVLAOFTPolicy":
    return OpenVLAOFTPolicy(request)


# ═════════════════════════════════════════════════════════════════════════
# knockout infrastructure (added on top of working adapter)
# ═════════════════════════════════════════════════════════════════════════

def _get_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """Return the LLM decoder layers (Llama / Mistral / Phi layout)."""
    lm = model.language_model
    for attr in ("model", "transformer"):
        candidate = getattr(lm, attr, None)
        if candidate is not None and hasattr(candidate, "layers"):
            return candidate.layers
    raise AttributeError("Cannot locate transformer layers on language model.")


class _KnockoutState:
    """Thread-safe(ish) per‑layer knockout mask storage."""

    def __init__(self, num_layers: int) -> None:
        self._masks: dict[int, torch.Tensor] = {}
        self._num_layers = num_layers

    def set_mask(self, layer_mask_map: dict[int, np.ndarray], device: torch.device) -> None:
        self._masks.clear()
        for lidx, arr in layer_mask_map.items():
            if 0 <= lidx < self._num_layers:
                self._masks[lidx] = torch.as_tensor(arr, device=device, dtype=torch.float32)

    def clear(self) -> None:
        self._masks.clear()

    def get(self, layer_idx: int) -> torch.Tensor | None:
        return self._masks.get(layer_idx)


def _install_knockout_hooks(model: nn.Module, state: _KnockoutState) -> None:
    """Replace every ``self_attn.forward`` with a wrapper that merges
    knockout additive masks into the 4‑D causal attention mask.

    Uses ``*args, **kwargs`` passthrough for robustness across HF versions.
    """
    layers = _get_transformer_layers(model)

    for layer_idx, decoder_layer in enumerate(layers):
        attn = decoder_layer.self_attn
        orig = attn.forward

        _debug_logged: set[int] = set()

        def _make_patched(_orig, _lidx: int):
            def patched(*args: Any, **kwargs: Any) -> Any:
                knock = state.get(_lidx)
                if knock is not None:
                    am = kwargs.get("attention_mask")

                    if am is not None and am.ndim == 4:
                        # eager path: 4-D mask already present, add knockout
                        k = knock[: am.shape[-2], : am.shape[-1]]
                        kwargs = dict(kwargs)
                        kwargs["attention_mask"] = am + k.unsqueeze(0).unsqueeze(0)

                    elif am is None:
                        # SDPA path: no explicit mask (is_causal handled
                        # internally).  Build a full causal + knockout mask.
                        hidden_states = kwargs["hidden_states"]
                        seq_len = hidden_states.shape[1]
                        device = hidden_states.device
                        dtype = hidden_states.dtype
                        causal = torch.triu(
                            torch.full((seq_len, seq_len), float("-inf"),
                                       device=device, dtype=dtype),
                            diagonal=1,
                        )
                        k = knock[:seq_len, :seq_len].to(device=device, dtype=dtype)
                        combined = causal + k
                        kwargs = dict(kwargs)
                        kwargs["attention_mask"] = combined.unsqueeze(0).unsqueeze(0)

                    # log once per layer
                    if _lidx not in _debug_logged:
                        _debug_logged.add(_lidx)
                        _am = kwargs.get("attention_mask")
                        print(f"[KO-DEBUG] layer={_lidx} "
                              f"am_is_None={_am is None} "
                              f"am_ndim={_am.ndim if _am is not None else 'N/A'} "
                              f"am_dtype={_am.dtype if _am is not None else 'N/A'} "
                              f"knock_min={knock.min().item():.2f}", flush=True)

                    # Force math kernel — flash kernel silently ignores
                    # custom 4-D attn_mask.
                    if kwargs.get("attention_mask") is not None:
                        with torch.backends.cuda.sdp_kernel(
                            enable_flash=False,
                            enable_math=True,
                            enable_mem_efficient=False,
                        ):
                            return _orig(*args, **kwargs)

                return _orig(*args, **kwargs)

            return patched

        attn.forward = _make_patched(orig, layer_idx)


# ═════════════════════════════════════════════════════════════════════════
# policy class
# ═════════════════════════════════════════════════════════════════════════
class OpenVLAOFTPolicy:

    # ── init / config ───────────────────────────────────────────────
    def __init__(self, request: Any) -> None:
        if not request.model_path:
            raise ValueError("OpenVLA-OFT adapter requires --model-path")
        self.request = request
        self.model_family = getattr(request, "model", "openvla").lower()
        self.knockout_config: dict[str, Any] | None = request.knockout_config
        self.cfg = self._build_cfg(request)

        # model components (populated by load)
        self.vla = None
        self.processor = None
        self.action_head = None
        self.proprio_projector = None
        self.noisy_action_projector = None
        self._get_vla_action = None
        self._resize_image_for_policy = None

        # knockout state
        self._knockout_state: _KnockoutState | None = None
        self._pending_knockout_spec: dict[str, Any] | None = None
        self._num_layers: int = 0
        self._hooks_installed: bool = False

        self._action_queue: list[np.ndarray] = []
        self.load()

    @staticmethod
    def _build_cfg(request: Any) -> SimpleNamespace:
        extra = dict(getattr(request, "extra_config", {}) or {})
        return SimpleNamespace(
            pretrained_checkpoint=request.model_path,
            use_l1_regression=bool(extra.get("use_l1_regression", True)),
            use_diffusion=bool(extra.get("use_diffusion", False)),
            use_film=bool(extra.get("use_film", False)),
            num_images_in_input=int(extra.get("num_images_in_input", 2)),
            use_proprio=bool(extra.get("use_proprio", True)),
            load_in_8bit=bool(extra.get("load_in_8bit", False)),
            load_in_4bit=bool(extra.get("load_in_4bit", False)),
            center_crop=bool(getattr(request, "center_crop", False) or extra.get("center_crop", True)),
            num_open_loop_steps=int(extra.get("num_open_loop_steps", 8)),
            unnorm_key=getattr(request, "unnorm_key", None) or f"{request.dataset}_no_noops",
            lora_rank=int(extra.get("lora_rank", 32)),
            num_diffusion_steps_train=int(extra.get("num_diffusion_steps_train", 50)),
            num_diffusion_steps_inference=int(extra.get("num_diffusion_steps_inference", 50)),
        )

    # ── load ───────────────────────────────────────────────────────
    def load(self) -> None:
        try:
            from experiments.robot import openvla_utils
            from prismatic.vla.constants import PROPRIO_DIM
        except ImportError as exc:
            raise RuntimeError(
                "OpenVLA-OFT adapter requires the OpenVLA-OFT repository "
                "and dependencies on PYTHONPATH."
            ) from exc

        # Knockout needs SDPA (not flash) for custom 4-D mask injection.
        needs_knockout = (
            self.knockout_config is not None
            and self.knockout_config.get("mode") != "baseline"
        )

        self.processor = openvla_utils.get_processor(self.cfg)
        self._get_vla_action = openvla_utils.get_vla_action
        self._resize_image_for_policy = getattr(openvla_utils, "resize_image_for_policy", None)

        if needs_knockout:
            self.vla = self._load_vla_for_knockout()
        else:
            self.vla = openvla_utils.get_vla(self.cfg)

        if self.cfg.use_l1_regression or self.cfg.use_diffusion:
            self.action_head = openvla_utils.get_action_head(self.cfg, llm_dim=self.vla.llm_dim)

        if self.cfg.use_proprio:
            self.proprio_projector = openvla_utils.get_proprio_projector(
                self.cfg, llm_dim=self.vla.llm_dim, proprio_dim=PROPRIO_DIM,
            )

        if self.cfg.use_diffusion:
            get_noisy = getattr(openvla_utils, "get_noisy_action_projector", None)
            if get_noisy is None:
                raise RuntimeError("This OpenVLA-OFT checkout does not expose get_noisy_action_projector()")
            self.noisy_action_projector = get_noisy(self.cfg, llm_dim=self.vla.llm_dim)

        # knockout bookkeeping
        self._num_layers = len(_get_transformer_layers(self.vla))
        self._pending_knockout_spec = None
        self._hooks_installed = False

        if needs_knockout:
            self._knockout_state = _KnockoutState(self._num_layers)
            _install_knockout_hooks(self.vla, self._knockout_state)
            self._hooks_installed = True
            self.configure_knockout(self.knockout_config)
            print("[OpenVLAOFTPolicy] loaded with SDPA attention + knockout hooks active.")
        elif self.knockout_config:
            self.configure_knockout(self.knockout_config)

    def _load_vla_for_knockout(self) -> Any:
        """Load the VLA with SDPA attention for knockout mask injection.

        SDPA (``scaled_dot_product_attention``, torch >= 2.0) accepts a
        custom 4-D additive mask via the ``attn_mask`` parameter, so
        knockout ``-inf`` additions work identically to eager attention.
        Flash-attention does NOT support custom 4-D masks and is avoided.
        """
        import json as _json
        import torch as _torch
        from huggingface_hub import HfApi, hf_hub_download
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

        ckpt = self.cfg.pretrained_checkpoint

        try:
            HfApi().model_info(ckpt)
        except Exception:
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
            # Mirror get_vla() setup — required for local checkpoints
            from experiments.robot.openvla_utils import update_auto_map, check_model_logic_mismatch
            update_auto_map(ckpt)
            check_model_logic_mismatch(ckpt)

        vla = AutoModelForVision2Seq.from_pretrained(
            ckpt,
            torch_dtype=_torch.bfloat16,
            load_in_8bit=self.cfg.load_in_8bit,
            load_in_4bit=self.cfg.load_in_4bit,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            attn_implementation="eager",
        )

        if self.cfg.use_film:
            vla = _apply_film_official(vla, self.cfg)

        vla.vision_backbone.set_num_images_in_input(self.cfg.num_images_in_input)
        vla.eval()

        if not self.cfg.load_in_8bit and not self.cfg.load_in_4bit:
            vla = vla.to(_torch.device("cuda") if _torch.cuda.is_available() else _torch.device("cpu"))

        try:
            HfApi().model_info(ckpt)
            stats_path = hf_hub_download(repo_id=ckpt, filename="dataset_statistics.json")
        except Exception:
            stats_path = os.path.join(ckpt, "dataset_statistics.json")
        if os.path.isfile(stats_path):
            with open(stats_path, "r") as _f:
                vla.norm_stats = _json.load(_f)

        return vla


    # ── knockout control ────────────────────────────────────────────
    def configure_knockout(self, config: dict[str, Any] | None) -> None:
        """Called by the eval runner before every episode."""
        self.knockout_config = config

        active = config is not None and config.get("mode") != "baseline"

        if not active:
            if self._knockout_state is not None:
                self._knockout_state.clear()
            self._pending_knockout_spec = None
            return

        ko_layers = self._resolve_knockout_layers(config)
        if not ko_layers:
            if self._knockout_state is not None:
                self._knockout_state.clear()
            self._pending_knockout_spec = None
            return

        self._pending_knockout_spec = config
        if self._knockout_state is not None:
            self._knockout_state.clear()  # rebuilt per inference step


    

    # ── mask construction ───────────────────────────────────────────
    def _apply_knockout_masks(self, prompt_token_count: int) -> None:
        if self._knockout_state is None or self.vla is None:
            return

        config = self._pending_knockout_spec
        if config is None:
            return

        # --- token partitions matching model's actual layout ---
        # _build_multimodal_attention inserts projected_patch_embeddings
        # after BOS (index 0).  projected_patch_embeddings includes:
        #   image_patches | proprio_token (if any) | diffusion_token (if any)
        num_image_patches = (
            self.vla.vision_backbone.get_num_patches()
            * self.vla.vision_backbone.get_num_images_in_input()
        )
        num_patches = num_image_patches
        use_diff = (
            self.noisy_action_projector is not None
            and self.action_head is not None
            and hasattr(self.action_head, "noise_scheduler")
        )
        if self.proprio_projector is not None:
            num_patches += 1       # proprio sits at the end of projected_patch_embeddings
        if use_diff:
            num_patches += 1       # diffusion timestep embedding

        from vla_trace.knockout.builders import make_openvla_partitions
        from vla_trace.knockout.specs import KnockoutSpec

        actual_text_tokens = prompt_token_count 

        # FIX 2: predict_action appends 56 action placeholder tokens (ACTION_DIM * NUM_ACTIONS_CHUNK)
        #         followed by 1 STOP token (STOP_INDEX=2) → total = 57
        actual_action_tokens = 7 * 8 + 1  # 56 + STOP = 57

        partitions = make_openvla_partitions(
            visual_tokens=num_patches,
            text_tokens=actual_text_tokens,
            action_tokens=actual_action_tokens,
            prefix_tokens=1,
        )

        phase = str(config.get("knockout_phase", config.get("phase", "generation")))
        mode = str(config.get("mode", "baseline"))
        text_scope = str(config.get("text_knockout_scope", config.get("text_scope", "semantic_instruction")))
        direction = config.get("direction")
        ko_layers = self._resolve_knockout_layers(config)

        # combined phase modes
        phase_policy = config.get("phase_policy", {})
        if phase_policy:
            pre = phase_policy.get("prefill", {})
            gen = phase_policy.get("generation", {})
            if pre.get("mode") and gen.get("mode"):
                mode = f"openvla_prefill_{pre['mode']}__generation_{gen['mode']}"
                phase = "both"

        spec = KnockoutSpec(
            family="openvla",
            phase=phase,
            mode=mode,
            layers=tuple(ko_layers),
            text_scope=text_scope,
            direction=direction,
            additive_block_value=float("-inf"),
        )

        from vla_trace.knockout.builders import build_additive_mask

        additive = build_additive_mask(spec, partitions)

        # FIX 3: proprio + diffusion tokens are appended AFTER the real image
        #         patches inside projected_patch_embeddings.  The builders
        #         treat the entire block as "visual" and no_image / no_vl
        #         masks block action→visual, which incorrectly severs the
        #         model's access to proprioception.  We zero out the extra
        #         positions so action can still attend to proprio/diffusion.
        extra_start = 1 + num_image_patches   # prefix(1) + image_patches
        visual_stop = 1 + num_patches         # prefix(1) + image_patches + extra
        if extra_start < visual_stop:
            for i in range(additive.values.shape[0]):
                additive.values[i, :, extra_start:visual_stop] = 0.0

        layer_masks: dict[int, np.ndarray] = {}
        for i, lidx in enumerate(additive.layers):
            layer_masks[int(lidx)] = np.asarray(additive.values[i], dtype=np.float32)
        device = next(self.vla.parameters()).device
        self._knockout_state.set_mask(layer_masks, device)

    def _resolve_knockout_layers(self, config: dict[str, Any]) -> list[int]:
        raw = config.get("knockout_layers", config.get("layers"))
        if raw is None:
            return []
        if isinstance(raw, str):
            if raw.strip().lower() == "all":
                return list(range(self._num_layers))
            return [int(p) for p in raw.replace(";", ",").split(",") if p.strip()]
        if isinstance(raw, (list, tuple)):
            return [int(v) for v in raw]
        if isinstance(raw, int):
            return [int(raw)]
        return []

    # ── predict (post-processing from official eval) ────────────────
    def reset(self) -> None:
        self._action_queue.clear()

    def predict_action(self, step: Any) -> np.ndarray:
        if self._action_queue:
            action = self._action_queue.pop(0)
            action = normalize_gripper_action(action, binarize=True)
            if self.model_family == "openvla":
                action = invert_gripper_action(action)
            return action

        if self._get_vla_action is None:
            self.load()

        observation = self._build_oft_observation(step)

        # compute text token count & install knockout masks (no‑op for baseline)
        prompt = f"In: What action should the robot take to {step.task_description.lower()}?\nOut:"

        prompt_token_count = int(self.processor.tokenizer(prompt, return_tensors="pt").input_ids.shape[-1])

        print(f"[VERIFY] prompt='{prompt.strip()}'  tokens={prompt_token_count}", flush=True)
        self._apply_knockout_masks(prompt_token_count)

        try:
            actions = self._get_vla_action(
                self.cfg,
                self.vla,
                self.processor,
                observation,
                step.task_description,
                action_head=self.action_head,
                proprio_projector=self.proprio_projector,
                noisy_action_projector=self.noisy_action_projector,
                use_film=self.cfg.use_film,
            )
        finally:
            if self._knockout_state is not None:
                self._knockout_state.clear()

        self._action_queue = [np.asarray(action, dtype=np.float32) for action in actions]
        if not self._action_queue:
            raise RuntimeError("OpenVLA-OFT policy returned an empty action chunk")

        action = self._action_queue.pop(0)
        action = normalize_gripper_action(action, binarize=True)
        if self.model_family == "openvla":
            action = invert_gripper_action(action)
        return action

    # ── observation helpers (unchanged) ─────────────────────────────
    def _build_oft_observation(self, step: Any) -> dict[str, Any]:
        raw = dict(step.raw_observation)
        full_image = self._raw_libero_image(raw, "agentview_image")
        if full_image is None:
            full_image = np.asarray(step.image, dtype=np.uint8)
        observation: dict[str, Any] = {
            "full_image": self._resize_openvla_image(full_image),
            "task_description": step.task_description,
        }
        if self.cfg.num_images_in_input > 1:
            if step.wrist_image is not None:
                wrist_image = np.asarray(step.wrist_image, dtype=np.uint8)
            else:
                wrist_image = self._raw_libero_image(raw, "robot0_eye_in_hand_image")
            if wrist_image is not None:
                observation["wrist_image"] = self._resize_openvla_image(wrist_image)
        if self.cfg.use_proprio:
            if step.state is not None:
                observation["state"] = np.asarray(step.state, dtype=np.float32)
            else:
                from vla_trace.evaluation.libero import get_libero_state
                observation["state"] = get_libero_state(raw)
        return observation

    @staticmethod
    def _raw_libero_image(raw: dict[str, Any], key: str) -> np.ndarray | None:
        if key not in raw:
            return None
        return np.asarray(raw[key], dtype=np.uint8)[::-1, ::-1]

    def _resize_openvla_image(self, image: np.ndarray) -> np.ndarray:
        if self._resize_image_for_policy is not None:
            return np.asarray(self._resize_image_for_policy(image, 224), dtype=np.uint8)
        return np.asarray(image, dtype=np.uint8)


# ═════════════════════════════════════════════════════════════════════════
# FiLM helper (mirrors openvla_utils._apply_film_to_vla)
# ═════════════════════════════════════════════════════════════════════════
def _apply_film_official(vla: nn.Module, cfg: SimpleNamespace) -> nn.Module:
    from peft import LoraConfig, get_peft_model
    from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone

    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)
    new_bb = FiLMedPrismaticVisionBackbone(
        vision_backbone=vla.vision_backbone, llm_dim=vla.llm_dim,
    )
    vla.model.vision_backbone = new_bb

    # Only search local directories; Hub checkpoints are not supported here
    # (FiLM is not used by standard openvla-oft fine-tuned models).
    ckpt = cfg.pretrained_checkpoint
    if os.path.isdir(ckpt):
        import os as _os
        matches = [
            _os.path.join(ckpt, f) for f in _os.listdir(ckpt)
            if "vision_backbone" in f and "checkpoint" in f
        ]
        if len(matches) == 1:
            state_dict = torch.load(matches[0], weights_only=True)
            vla.model.vision_backbone.load_state_dict(state_dict)
    vla = vla.model
    vla.vision_backbone = vla.vision_backbone.to(torch.bfloat16)
    return vla
