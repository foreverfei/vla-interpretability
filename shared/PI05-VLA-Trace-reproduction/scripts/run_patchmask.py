"""PatchMask — verified pipeline: project → circle → flip → resize."""
import os, sys, json, numpy as np, imageio
os.environ["MUJOCO_GL"] = "egl"; os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["LIBERO_CONFIG_PATH"] = os.path.expanduser("~/.libero")
sys.path.insert(0, os.path.expanduser("~/autodl-tmp/openpi/src"))
sys.path.insert(0, os.path.expanduser("~/autodl-tmp/VLA-Trace"))

import torch
from vla_trace.evaluation.adapters import PolicyBuildRequest
from examples.pi05_adapter import create_policy
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from vla_trace.evaluation.libero import get_libero_image_pi0, get_libero_state, get_libero_dummy_action
from PIL import Image as PILImage

MODEL_PATH = os.path.expanduser("~/autodl-tmp/models/pi05_libero_finetuned")
OUTPUT_DIR = os.path.expanduser("~/autodl-tmp/runs/patchmask_final")
TASK_ID = 0; NUM_TRIALS = 5; MAX_STEPS = 400
IMG_H, IMG_W = 256, 256
os.makedirs(os.path.join(OUTPUT_DIR, "videos"), exist_ok=True)

request = PolicyBuildRequest(model="pi05", dataset="libero_10", model_path=MODEL_PATH,
    data_root=os.path.expanduser("~/autodl-tmp/data/libero_10"), output_dir=OUTPUT_DIR,
    device="cuda", seed=0, tokenizer_path=os.path.join(MODEL_PATH, "tokenizer.model"))
policy = create_policy(request)

def project(env, name, obs):
    d = env.env.sim.data; cam_id = 2; fov = 45.0
    cmat = d.cam_xmat[cam_id].reshape(3,3); cpos = d.cam_xpos[cam_id]
    fl = IMG_W/2/np.tan(np.radians(fov)/2)
    k = name + "_pos" if not name.startswith("robot") else name
    pos = obs.get(k)
    if pos is None: return None
    cam = cmat.T @ (np.array(pos)-cpos); z = -cam[2]
    if z <= 0.01: return None
    u = fl*cam[0]/z + IMG_W/2; v = fl*cam[1]/z + IMG_H/2
    return (int(u),int(v)) if 0 <= u < IMG_W and 0 <= v < IMG_H else None

# Fixed-radius lookup by object name (from geom sizes)
OBJ_RADII = {
    'alphabet_soup_1': 18, 'tomato_sauce_1': 18, 'cream_cheese_1': 16,
    'ketchup_1': 18, 'orange_juice_1': 20, 'milk_1': 20,
    'butter_1': 16, 'basket_1': 32, 'robot0_eef_pos': 10,
    'chocolate_pudding_1': 16, 'alphabet_soup_2': 18, 'tomato_sauce_2': 18,
}

def build_mask_224(env, obs, variant):
    """Single pass: build 256 mask, flip, resize with PIL → return 224 mask."""
    m256 = np.zeros((IMG_H, IMG_W), dtype=bool)
    if variant == "baseline": return np.zeros((224, 224), dtype=bool)
    
    all_objs = [k.replace("_pos","") for k in obs if k.endswith("_pos") and not k.startswith("robot") and "_to_" not in k]
    targets = list(getattr(env, "obj_of_interest", []) or [])
    if not targets: targets = all_objs
    
    names = targets if variant != "mask_gripper" else ["robot0_eef_pos"]
    for n in names:
        p = project(env, n, obs)
        if p is None: continue
        R = OBJ_RADII.get(n, 18)
        x0, x1 = max(0, p[0]-R), min(IMG_W-1, p[0]+R)
        y0, y1 = max(0, p[1]-R), min(IMG_H-1, p[1]+R)
        m256[y0:y1, x0:x1] = True
    
    if variant == "mask_background" and m256.any(): m256 = ~m256
    if not m256.any(): return np.zeros((224, 224), dtype=bool)
    
    # Flip + PIL resize (verified in debug)
    flipped = (m256[::-1, ::-1] * 255).astype(np.uint8)
    pil = PILImage.fromarray(flipped, mode='L')
    pil = pil.resize((224, 224), PILImage.BILINEAR)
    return np.array(pil) > 127

bd = benchmark.get_benchmark_dict(); ts = bd["libero_10"](); task = ts.get_task(TASK_ID)
inits = ts.get_task_init_states(TASK_ID)
bf = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
instr = task.language; da = get_libero_dummy_action("pi05")
variants = ["baseline", "mask_target", "mask_gripper", "mask_background"]
results = []

for variant in variants:
    for trial in range(NUM_TRIALS):
        env = OffScreenRenderEnv(bddl_file_name=bf, camera_heights=IMG_H, camera_widths=IMG_W)
        env.seed(trial); obs = env.reset(); obs = env.set_init_state(inits[trial % len(inits)])
        policy.reset()
        for _ in range(10): obs, _, _, _ = env.step(da)
        success, steps = False, 0
        clean_frames, masked_frames = [], []

        for sid in range(MAX_STEPS):
            img_pi0, wrist_pi0 = get_libero_image_pi0(obs, 224)
            state = get_libero_state(obs)

            clean_frames.append(img_pi0.copy())

            # Build mask and apply
            mask_224 = build_mask_224(env, obs, variant)
            masked = img_pi0.copy()
            if mask_224.any(): masked[mask_224] = 0
            masked_frames.append(masked)

            img_input = masked if (variant != "baseline" and mask_224.any()) else img_pi0
            # Debug first frame
            if sid == 0 and trial == 0:
                print(f"[DEBUG] {variant}: mask_224.any()={mask_224.any()} sum={mask_224.sum()} obj_of_interest={getattr(env,'obj_of_interest',[])}")

            step = type('Step', (), dict(image=img_input, wrist_image=wrist_pi0, state=state, task_description=instr))()
            action = policy.predict_action(step)
            obs, _, done, _ = env.step(action.tolist())
            steps = sid + 1
            if done: success = True; break

        status = "success" if success else "failure"
        results.append({"variant":variant,"trial":trial,"success":success,"steps":steps})
        vd_ = os.path.join(OUTPUT_DIR, "videos")
        cp = os.path.join(vd_, f"{variant}_trial{trial}_clean_{status}.mp4")
        mp = os.path.join(vd_, f"{variant}_trial{trial}_masked_{status}.mp4")
        try: imageio.mimsave(cp, clean_frames, fps=20); imageio.mimsave(mp, masked_frames, fps=20)
        except: pass
        print(f"  {variant} trial {trial}: {'OK' if success else 'FAIL'} st={steps}", flush=True)
        env.close()
    vr = [r for r in results if r["variant"]==variant]
    print(f"{variant}: {sum(1 for r in vr if r['success'])}/{len(vr)}", flush=True)

print("\n=== RESULT ===")
for v in variants:
    vr = [r for r in results if r["variant"]==v]
    print(f"  {v}: {sum(1 for r in vr if r['success'])}/{len(vr)} = {int(sum(1 for r in vr if r['success'])/len(vr)*100)}%")
with open(os.path.join(OUTPUT_DIR,"results.json"),"w") as f: json.dump(results,f,indent=2)
