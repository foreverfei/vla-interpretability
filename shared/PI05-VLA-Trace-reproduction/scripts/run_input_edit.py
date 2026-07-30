"""Input Editing — reuse Pi05Policy directly, only change instruction."""
import os, sys, json, numpy as np
from pathlib import Path

os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["LIBERO_CONFIG_PATH"] = os.path.expanduser("~/.libero")

sys.path.insert(0, os.path.expanduser("~/autodl-tmp/openpi/src"))
sys.path.insert(0, os.path.expanduser("~/autodl-tmp/VLA-Trace"))

from vla_trace.evaluation.adapters import PolicyBuildRequest
from examples.pi05_adapter import create_policy
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from vla_trace.evaluation.libero import get_libero_image_pi0, get_libero_state, get_libero_dummy_action

MODEL_PATH = os.path.expanduser("~/autodl-tmp/models/pi05_libero_finetuned")
TASK_ID = 0
NUM_TRIALS = 10
OUTPUT_DIR = os.path.expanduser("~/autodl-tmp/runs/pi05_input_edit_video")

# Build policy EXACTLY like eval-libero does
request = PolicyBuildRequest(
    model="pi05", dataset="libero_10",
    model_path=MODEL_PATH,
    data_root=os.path.expanduser("~/autodl-tmp/data/libero_10"),
    output_dir=OUTPUT_DIR,
    device="cuda", seed=0,
    tokenizer_path=os.path.join(MODEL_PATH, "tokenizer.model"),
)
policy = create_policy(request)

# LIBERO setup
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_10"]()
task = task_suite.get_task(TASK_ID)
initial_states = task_suite.get_task_init_states(TASK_ID)
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
actual = task.language
print(f"Task {TASK_ID}: {actual}")

BASE = actual
EDITED = actual.replace("alphabet soup", "cream cheese").replace("tomato sauce", "chocolate pudding")
print(f"Baseline: {BASE}")
print(f"Edited:   {EDITED}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
vd = os.path.join(OUTPUT_DIR, "videos")
os.makedirs(vd, exist_ok=True)
da = get_libero_dummy_action("pi05")
results = []

for trial in range(NUM_TRIALS):
    for label, instr in [("baseline", BASE), ("edited", EDITED)]:
        env = OffScreenRenderEnv(bddl_file_name=task_bddl_file, camera_heights=256, camera_widths=256)
        env.seed(trial)
        obs = env.reset()
        obs = env.set_init_state(initial_states[trial])
        policy.reset()
        for _ in range(10):
            obs, _, _, _ = env.step(da)

        images, success, steps = [], False, 0
        for step_id in range(520):
            img_pi0, wrist_pi0 = get_libero_image_pi0(obs, 224)
            state = get_libero_state(obs)
            # Monkey-patch task_description on the fly
            step = type('Step', (), dict(
                image=img_pi0, wrist_image=wrist_pi0, state=state,
                task_description=instr,
            ))()
            action = policy.predict_action(step)
            images.append(img_pi0.copy())
            obs, _, done, _ = env.step(action.tolist())
            steps = step_id + 1
            if done:
                success = True
                break

        status = "success" if success else "failure"
        vp = os.path.join(vd, f"trial{trial:02d}_{label}_{status}.mp4")
        try:
            import imageio
            with imageio.get_writer(vp, fps=20, format="FFMPEG", mode="I") as w:
                for f in images:
                    w.append_data(f)
            print(f"[{label}] trial={trial} success={success} steps={steps} ✓")
        except Exception as e:
            print(f"[{label}] trial={trial} success={success} steps={steps} ✗ {e}")

        results.append(dict(
            edit_id="edit_ingredients", edit_type="instruction_replace",
            task_id=TASK_ID, trial=trial, condition=label,
            instruction=instr, success=success, total_steps=steps,
        ))
        env.close()

rp = os.path.join(OUTPUT_DIR, "input_edit_results.jsonl")
with open(rp, "w") as f:
    for r in results:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

bl = sum(1 for r in results if r["condition"] == "baseline" and r["success"])
ed = sum(1 for r in results if r["condition"] == "edited" and r["success"])
print(f"\nBaseline: {bl}/{NUM_TRIALS}  Edited: {ed}/{NUM_TRIALS}")
