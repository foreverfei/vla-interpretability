"""Paper-style Attention IoU — 3D projection masks (no EGL seg required)."""
import os, sys, json, numpy as np
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["LIBERO_CONFIG_PATH"] = os.path.expanduser("~/.libero")
sys.path.insert(0, os.path.expanduser("~/autodl-tmp/openpi/src"))
sys.path.insert(0, os.path.expanduser("~/autodl-tmp/VLA-Trace"))

import torch, imageio, cv2
from vla_trace.evaluation.adapters import PolicyBuildRequest
from examples.pi05_adapter import create_policy
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from vla_trace.evaluation.libero import get_libero_image_pi0, get_libero_state, get_libero_dummy_action
from vla_trace.behavior.attention_views import extract_attention_views
from vla_trace.behavior.attention import attention_to_grid

MODEL_PATH = os.path.expanduser("~/autodl-tmp/models/pi05_libero_finetuned")
OUTPUT_DIR = os.path.expanduser("~/autodl-tmp/runs/pi05_attention_iou_final")
TASK_IDS = [0, 1]; NUM_TRIALS = 5; MAX_STEPS = 400; CAPTURE_EVERY = 20
HOOK_LAYERS = [8, 17]; DEVICE = torch.device("cuda")
IMG_H, IMG_W = 256, 256; GRID_SIZE = 16
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "videos"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "heatmap_videos"), exist_ok=True)

request = PolicyBuildRequest(
    model="pi05", dataset="libero_10", model_path=MODEL_PATH,
    data_root=os.path.expanduser("~/autodl-tmp/data/libero_10"),
    output_dir=OUTPUT_DIR, device="cuda", seed=0,
    tokenizer_path=os.path.join(MODEL_PATH, "tokenizer.model"),
)
policy = create_policy(request); model = policy.model
print("Policy loaded")

# Hooks
captured = {}; orig_fns = {}
def register():
    layers = model.paligemma_with_expert.gemma_expert.model.layers
    for idx in HOOK_LAYERS:
        orig_fns[idx] = layers[idx].self_attn.forward
        def mw(i, o):
            def w(*a, **kw):
                kw = dict(kw); kw["output_attentions"] = True
                r = o(*a, **kw)
                if isinstance(r, tuple) and len(r) >= 2 and r[1] is not None:
                    captured[i] = r[1].detach().cpu().float().numpy()
                return r
            return w
        layers[idx].self_attn.forward = mw(idx, orig_fns[idx])
def restore():
    layers = model.paligemma_with_expert.gemma_expert.model.layers
    for idx, o in orig_fns.items(): layers[idx].self_attn.forward = o
register()

def predict_step(img, wrist, s_arr, instr):
    captured.clear()
    step = type('Step', (), dict(image=img, wrist_image=wrist, state=s_arr, task_description=instr))()
    return policy.predict_action(step)

# Camera projection for mask generation
def build_camera_projection(env):
    """Get camera intrinsics/extrinsics for agentview."""
    cid = env.env.sim.model.camera_name2id("agentview")
    cam_mat = env.env.sim.model.cam_mat0[cid].copy().reshape(3, 3)
    cam_pos = env.env.sim.model.cam_pos0[cid].copy()
    fovy = float(env.env.sim.model.cam_fovy[cid]) * np.pi / 180.0
    f = (IMG_H / 2.0) / np.tan(fovy / 2.0)
    return cam_mat, cam_pos, f

def project_to_grid(world_pos, cam_mat, cam_pos, f):
    """Project 3D world position to 16x16 attention grid cell."""
    rel = np.array(world_pos) - cam_pos
    cam = cam_mat @ rel  # [x, y, z] in camera frame
    if cam[2] <= 0.01: return None  # behind camera
    u = f * cam[0] / cam[2] + IMG_W / 2.0
    v = f * (-cam[1]) / cam[2] + IMG_H / 2.0
    gu = int(np.clip(u / IMG_W * GRID_SIZE, 0, GRID_SIZE - 1))
    gv = int(np.clip(v / IMG_H * GRID_SIZE, 0, GRID_SIZE - 1))
    return gv, gu  # row, col

def make_object_mask(world_pos, cam_mat, cam_pos, f, radius=2):
    """Create [GRID_SIZE, GRID_SIZE] bool mask around projected position."""
    mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
    cell = project_to_grid(world_pos, cam_mat, cam_pos, f)
    if cell is None: return mask
    r, c = cell
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            nr, nc = r + dr, c + dc
            if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                mask[nr, nc] = True
    return mask

# Heatmap
def hm_overlay(img, grid, a=0.45):
    h, w = img.shape[:2]
    up = cv2.resize(grid.astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
    norm = (up - up.min()) / max(up.max() - up.min(), 1e-8)
    hm = cv2.cvtColor(cv2.applyColorMap((norm*255).astype(np.uint8), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return (img.astype(np.float32)*(1-a) + hm.astype(np.float32)*a).astype(np.uint8)
def hstack2(a, b):
    h_ = max(a.shape[0], b.shape[0])
    return np.hstack([np.pad(a,((0,h_-a.shape[0]),(0,0),(0,0))), np.pad(b,((0,h_-b.shape[0]),(0,0),(0,0)))])

bd = benchmark.get_benchmark_dict(); ts = bd["libero_10"]()
da = get_libero_dummy_action("pi05"); summaries = []

for tid in TASK_IDS:
    task = ts.get_task(tid); inits = ts.get_task_init_states(tid)
    bf = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    instr = task.language
    print("\nTask %d: %s" % (tid, instr))
    for trial in range(NUM_TRIALS):
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=IMG_H, camera_widths=IMG_W)
        env.seed(trial); obs = env.reset(); obs = env.set_init_state(inits[trial % len(inits)])
        policy.reset()
        for _ in range(10): obs, _, _, _ = env.step(da)
        cam_mat, cam_pos, focal = build_camera_projection(env)
        # Get object names (non-robot objects with _pos in obs)
        obj_names = sorted(set(k.replace("_pos", "") for k in obs if k.endswith("_pos") and not k.startswith("robot")))
        td = os.path.join(OUTPUT_DIR, "task%02d_trial%02d" % (tid, trial))
        for d in ["attention_maps", "obs", "step_masks"]: os.makedirs(os.path.join(td, d), exist_ok=True)
        rf, hf, success, steps = [], [], False, 0
        all_attn = {}; all_masks = {}

        for sid in range(MAX_STEPS):
            img_pi0, wrist_pi0 = get_libero_image_pi0(obs, 224)
            state = get_libero_state(obs)
            action = predict_step(img_pi0, wrist_pi0, state, instr)
            obs, _, done, _ = env.step(action.tolist())
            steps = sid + 1; rf.append(img_pi0.copy())

            if captured and sid % CAPTURE_EVERY == 0:
                sn = "step_%03d" % sid
                raw = None
                for li in HOOK_LAYERS:
                    if li in captured: raw = captured[li]; break
                if raw is not None:
                    views = extract_attention_views(raw[None,...], visual_span=slice(0,256), text_span=slice(768,781), action_span=slice(0,50), normalize_rows=True)
                    grid = attention_to_grid(views["action_to_image"])
                    all_attn[sn] = grid.astype(np.float32)
                np.save(os.path.join(td, "obs", "obs_%s.npy" % sn), img_pi0)
                # Generate masks from object positions
                for oname in obj_names:
                    pos_key = oname + "_pos"
                    if pos_key in obs:
                        mask_grid = make_object_mask(obs[pos_key], cam_mat, cam_pos, focal, radius=2)
                        if mask_grid.any():
                            all_masks["%s_%s" % (sn, oname)] = mask_grid.astype(np.uint8)
                if all_attn:
                    hm = hm_overlay(img_pi0, list(all_attn.values())[-1])
                    hf.append(hstack2(img_pi0, hm))
            if done: success = True; break

        if all_attn:
            np.savez_compressed(os.path.join(td, "attention_maps", "attention_maps.npz"), **all_attn)
        if all_masks:
            np.savez_compressed(os.path.join(td, "step_masks", "step_masks.npz"), **all_masks)
        status = "success" if success else "failure"
        meta = {"trace_id":"pi05_libero10_attn_iou_task%02d_trial%02d"%(tid,trial),"model":"pi05","dataset":"libero_10","task_id":tid,"task_description":instr,"trial":trial,"success":success,"total_steps":steps,"captured_steps":len(all_attn),"mask_objects":len(all_masks),"hook_layers":HOOK_LAYERS}
        with open(os.path.join(td,"metadata.json"),"w") as f: json.dump(meta,f,indent=2)
        summaries.append({"task_id":tid,"trial":trial,"success":success,"steps":steps})
        sr = sum(1 for s in summaries if s["success"])/len(summaries)
        print("  [t=%d.%d] %s st=%d attn=%d masks=%d sr=%.2f"%(tid,trial,status,steps,len(all_attn),len(all_masks),sr),flush=True)
        vp = os.path.join(OUTPUT_DIR,"videos","task%02d_trial%02d_%s.mp4"%(tid,trial,status))
        try:
            with imageio.get_writer(vp,fps=20,format="FFMPEG",mode="I") as w:
                for f in rf: w.append_data(f)
        except: pass
        if hf:
            hp = os.path.join(OUTPUT_DIR,"heatmap_videos","task%02d_trial%02d_%s_heatmap.mp4"%(tid,trial,status))
            try:
                with imageio.get_writer(hp,fps=10,format="FFMPEG",mode="I") as w:
                    for f in hf: w.append_data(f)
            except: pass
        env.close()

restore()
print("\n=== DONE ===")
for t in TASK_IDS:
    ts_ = [s for s in summaries if s["task_id"]==t]; sc = sum(1 for s in ts_ if s["success"])
    print("  Task %d: %d/%d = %d%%"%(t,sc,len(ts_),int(sc/len(ts_)*100)))
sc_all = sum(1 for s in summaries if s["success"])
print("  Overall: %d/%d = %d%%"%(sc_all,len(summaries),int(sc_all/len(summaries)*100)))
with open(os.path.join(OUTPUT_DIR,"summary.json"),"w") as f: json.dump({"task_ids":TASK_IDS,"num_trials":NUM_TRIALS,"results":summaries},f,indent=2)
print("Output:",OUTPUT_DIR)
