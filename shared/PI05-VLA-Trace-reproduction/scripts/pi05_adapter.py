import sys, os, numpy as np, torch, json
from types import SimpleNamespace
from PIL import Image
import safetensors.torch as sf
import sentencepiece as spm
sys.path.insert(0, os.path.expanduser("~/autodl-tmp/openpi/src"))
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

class Pi05Policy:
    def __init__(self, request):
        self.device = torch.device("cuda")
        with open(os.path.join(request.model_path, "train_config.json")) as f:
            tc = json.load(f)["policy"]
        cfg = SimpleNamespace(pi05=True, paligemma_variant=tc.get("paligemma_variant","gemma_2b"),
            action_expert_variant=tc.get("action_expert_variant","gemma_300m"),
            action_dim=tc.get("max_action_dim",32), action_horizon=tc.get("n_action_steps",50),
            dtype=tc.get("dtype","bfloat16"), pytorch_compile_mode=None)
        self.model = PI0Pytorch(cfg).to(self.device); self.model.eval()
        ckpt = sf.load_file(os.path.join(request.model_path, "model.safetensors"))
        ep = os.path.join(request.model_path, "embed_tokens.safetensors")
        if os.path.exists(ep):
            for ek, ev in sf.load_file(ep).items(): ckpt[ek] = ev
        st = self.model.state_dict(); matched = {}
        for k, v in ckpt.items():
            nk = k[6:] if k.startswith("model.") else k
            if nk in st and st[nk].shape == v.shape: matched[nk] = v
        tgt = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
        if tgt not in matched:
            for ck, cv in ckpt.items():
                if "lm_head" in ck and cv.shape == st[tgt].shape: matched[tgt] = cv; break
        self.model.load_state_dict(matched, strict=False)
        tp = request.tokenizer_path or os.path.join(request.model_path, "tokenizer.model")
        self.sp = spm.SentencePieceProcessor(model_file=tp) if os.path.exists(tp) else None
        self.action_dim = 7; self.num_inference_steps = tc.get("num_inference_steps",10)
        self.image_size = tc.get("image_resolution",[224,224])

        # Load normalization stats
        pre = sf.load_file(os.path.join(request.model_path, "policy_preprocessor_step_2_normalizer_processor.safetensors"))
        post = sf.load_file(os.path.join(request.model_path, "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
        self.state_mean = pre["observation.state.mean"].numpy()
        self.state_std = pre["observation.state.std"].numpy()
        self.action_mean = post["action.mean"].numpy()
        self.action_std = post["action.std"].numpy()
        print(f"Pi05Policy loaded on {self.device}, state_norm: MEAN_STD, action_unnorm: MEAN_STD")

    def reset(self): pass
    def configure_knockout(self, config): pass

    def predict_action(self, step):
        from PIL import Image as PILImage
        h, w = self.image_size
        def pp(img):
            return np.array(PILImage.fromarray(np.asarray(img)).resize((w,h),PILImage.LANCZOS), dtype=np.float32)/255.0
        b = pp(step.image)
        wr = pp(step.wrist_image) if step.wrist_image is not None else np.zeros_like(b)
        s_raw = np.asarray(step.state, dtype=np.float32) if step.state is not None else np.zeros(8, dtype=np.float32)
        # Normalize state
        s = (s_raw - self.state_mean) / (self.state_std + 1e-8)
        tok = self.sp.encode(step.task_description, out_type=int)[:200] if self.sp else [0]
        obs = SimpleNamespace(
            state=torch.from_numpy(s).unsqueeze(0).to(torch.float32).to(self.device),
            images={"base_0_rgb":torch.from_numpy(b).permute(2,0,1).unsqueeze(0).to(self.device),
                    "left_wrist_0_rgb":torch.from_numpy(wr).permute(2,0,1).unsqueeze(0).to(self.device),
                    "right_wrist_0_rgb":torch.zeros(1,3,h,w,device=self.device)},
            image_masks={"base_0_rgb":torch.ones(1,dtype=torch.bool,device=self.device),
                         "left_wrist_0_rgb":torch.ones(1,dtype=torch.bool,device=self.device),
                         "right_wrist_0_rgb":torch.zeros(1,dtype=torch.bool,device=self.device)},
            tokenized_prompt=torch.from_numpy(np.array(tok,dtype=np.int64)).unsqueeze(0).to(self.device),
            tokenized_prompt_mask=torch.ones(1,len(tok),dtype=torch.bool,device=self.device),
            token_ar_mask=None, token_loss_mask=None,
        )
        with torch.no_grad():
            a_norm = self.model.sample_actions(self.device, obs, num_steps=self.num_inference_steps)
        a_norm = a_norm[0,0,:self.action_dim].cpu().numpy().astype(np.float32)
        # Unnormalize action
        a_raw = a_norm * self.action_std + self.action_mean
        return a_raw

def create_policy(request): return Pi05Policy(request)
