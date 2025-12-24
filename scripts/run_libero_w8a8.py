#!/usr/bin/env python3
"""Run LIBERO evaluation with W8A8 model from HuggingFace.

Usage:
    python scripts/run_libero_w8a8.py --task-suite libero_spatial --num-trials 20
"""

import math
import os
import sys

# Allow numpy arrays in torch.load (needed for LIBERO init states)
try:
    from torch.serialization import add_safe_globals
    import numpy as _np
    add_safe_globals([_np.core.multiarray._reconstruct, _np.ndarray])
except Exception:
    pass

import torch
_libero_original_torch_load = torch.load
def _libero_torch_load_wrapper(args, kwargs):
    kwargs = dict(kwargs)
    kwargs.setdefault("weights_only", False)
    return _libero_original_torch_load(*args, **kwargs)
torch.load = lambda *a, **k: _libero_torch_load_wrapper(a, k)

# Setup environment
os.environ["LD_LIBRARY_PATH"] = "/usr/local/cuda-12.8/lib64:" + os.environ.get("LD_LIBRARY_PATH", "")
os.environ["OPENPI_DISABLE_TORCH_COMPILE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHDYNAMO_DISABLE"] = "1"

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "libero"))

import argparse
import logging
import pathlib

logging.basicConfig(level=logging.INFO)

# Constants from original libero evaluation
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256
NUM_STEPS_WAIT = 10
RESIZE_SIZE = 224
REPLAN_STEPS = 5


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    import numpy as np
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def main():
    parser = argparse.ArgumentParser(description="Run LIBERO with W8A8 model")
    parser.add_argument("--hf-repo", type=str, default="fatdove/pi05-libero-w8a8", help="HuggingFace repo ID")
    parser.add_argument("--task-suite", type=str, default="libero_spatial", help="LIBERO task suite")
    parser.add_argument("--num-trials", type=int, default=20, help="Number of trials per task")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    args = parser.parse_args()

    print("=" * 70)
    print("LIBERO Evaluation with W8A8 Model")
    print("=" * 70)
    print(f"HuggingFace repo: {args.hf_repo}")
    print(f"Task suite: {args.task_suite}")
    print(f"Num trials: {args.num_trials}")
    print(f"Seed: {args.seed}")
    print()

    # Step 1: Load W8A8 model from HuggingFace
    print("[Step 1] Loading W8A8 model from HuggingFace...")
    from openpi.models_pytorch.bitblas_w8a8_layers import load_w8a8_policy

    policy = load_w8a8_policy(
        args.hf_repo,
        policy_config_name="pi05_libero",
        enable_tuning=False,
    )
    print(f"Model loaded! W8A8 layers: {policy._w8a8_layer_count}")

    # Step 2: Run LIBERO evaluation
    print(f"\n[Step 2] Running LIBERO {args.task_suite} evaluation...")

    # Import LIBERO evaluation code
    import collections
    from libero.libero import benchmark
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import numpy as np
    from openpi_client import image_tools

    # Set random seed
    np.random.seed(args.seed)

    # Get task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite]()
    num_tasks = task_suite.n_tasks

    print(f"Task suite: {args.task_suite}, num_tasks: {num_tasks}")

    # Set max steps based on task suite
    if args.task_suite == "libero_spatial":
        max_steps = 220
    elif args.task_suite == "libero_object":
        max_steps = 280
    elif args.task_suite == "libero_goal":
        max_steps = 300
    elif args.task_suite == "libero_10":
        max_steps = 520
    elif args.task_suite == "libero_90":
        max_steps = 400
    else:
        max_steps = 600

    # Results
    results = {}
    total_success = 0
    total_trials = 0

    for task_id in range(num_tasks):
        task = task_suite.get_task(task_id)
        task_name = task.name
        task_description = task.language
        # Construct full path to BDDL file
        task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file

        # Get default initial states
        initial_states = task_suite.get_task_init_states(task_id)

        print(f"\n--- Task {task_id + 1}/{num_tasks}: {task_name} ---")
        print(f"Description: {task_description}")

        # Create environment
        env_args = {
            "bddl_file_name": task_bddl_file,
            "camera_heights": LIBERO_ENV_RESOLUTION,
            "camera_widths": LIBERO_ENV_RESOLUTION,
        }
        env = OffScreenRenderEnv(**env_args)
        env.seed(args.seed)

        task_successes = 0

        for trial in range(args.num_trials):
            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states - this returns the first observation
            obs = env.set_init_state(initial_states[trial % len(initial_states)])

            t = 0
            done = False

            while t < max_steps + NUM_STEPS_WAIT:
                # Wait for objects to stabilize
                if t < NUM_STEPS_WAIT:
                    obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                    t += 1
                    continue

                # Preprocess images (rotate 180 degrees to match training)
                img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(img, RESIZE_SIZE, RESIZE_SIZE)
                )
                wrist_img = image_tools.convert_to_uint8(
                    image_tools.resize_with_pad(wrist_img, RESIZE_SIZE, RESIZE_SIZE)
                )

                if not action_plan:
                    # Prepare observation dict
                    policy_obs = {
                        "observation/image": img,
                        "observation/wrist_image": wrist_img,
                        "observation/state": np.concatenate([
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        ]),
                        "prompt": str(task_description),
                    }

                    # Get action from policy
                    result = policy.infer(policy_obs)
                    action_chunk = result["actions"]
                    action_plan.extend(action_chunk[:REPLAN_STEPS])

                action = action_plan.popleft()

                # Execute action
                obs, reward, done, info = env.step(action.tolist())
                if done:
                    task_successes += 1
                    total_success += 1
                    break
                t += 1

            trial_result = "SUCCESS" if done else "FAIL"
            print(f"  Trial {trial + 1}/{args.num_trials}: {trial_result} (steps: {t})")

        env.close()

        success_rate = task_successes / args.num_trials
        results[task_name] = {
            "successes": task_successes,
            "trials": args.num_trials,
            "success_rate": success_rate,
        }
        total_trials += args.num_trials

        print(f"Task {task_name}: {task_successes}/{args.num_trials} = {success_rate:.1%}")

    # Print summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"Task Suite: {args.task_suite}")
    print(f"Model: {args.hf_repo}")
    print()

    for task_name, result in results.items():
        print(f"  {task_name}: {result['successes']}/{result['trials']} = {result['success_rate']:.1%}")

    overall_rate = total_success / total_trials if total_trials > 0 else 0
    print()
    print(f"OVERALL: {total_success}/{total_trials} = {overall_rate:.1%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
