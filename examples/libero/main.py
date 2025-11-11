import collections
import csv
import dataclasses
import json
import logging
import math
import os
import pathlib
import time
from pathlib import Path

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

try:
    from torch.serialization import add_safe_globals

    import numpy as _np

    add_safe_globals([_np.core.multiarray._reconstruct, _np.ndarray])  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - compatibility shim
    pass

import torch

_libero_original_torch_load = torch.load


def _libero_torch_load_wrapper(args, kwargs):
    kwargs = dict(kwargs)
    kwargs.setdefault("weights_only", False)
    return _libero_original_torch_load(*args, **kwargs)


torch.load = lambda *a, **k: _libero_torch_load_wrapper(a, k)  # noqa: E731

_DATASETS_ROOT = Path(__file__).resolve().parent / "dataset" / "datasets"
_DATASETS_ROOT.mkdir(parents=True, exist_ok=True)

try:  # Optional: only present when DuQuant is active
    from openpi.models_pytorch.duquant_preprocess import _DUQUANT_PROFILER
except Exception:  # pragma: no cover - duquant disabled or JAX policy
    _DUQUANT_PROFILER = None


class _PolicyCallProfiler:
    """Collect policy inference latency stats regardless of DuQuant usage."""

    def __init__(self) -> None:
        flag = os.environ.get("OPENPI_POLICY_PROFILE", os.environ.get("OPENPI_DUQUANT_PROFILE", "0"))
        self.enabled = flag not in ("0", "false", "False")
        self._stats = {"policy_infer": {"time_ms": 0.0, "count": 0, "elements": 0.0, "bytes": 0.0}}

    def record(self, duration_ms: float, *, elements: int = 0, byte_size: int = 0) -> None:
        if not self.enabled:
            return
        entry = self._stats["policy_infer"]
        entry["time_ms"] += float(duration_ms)
        entry["count"] += 1
        entry["elements"] += float(elements)
        entry["bytes"] += float(byte_size)

    def report(self, context: str, *, reset: bool = False) -> None:
        if not self.enabled:
            return
        entry = self._stats.get("policy_infer")
        if not entry or entry["count"] == 0:
            return
        total_ms = entry["time_ms"]
        calls = int(entry["count"])
        avg_ms = total_ms / calls if calls else 0.0
        elems = int(entry["elements"])
        total_bytes = entry["bytes"]
        gbps = (total_bytes / 1e9) / (total_ms / 1000.0) if total_ms > 0 else 0.0
        print("=" * 100)
        print(f"[POLICY][PROFILE] inference summary [{context}]")
        header = (
            f"{'Label':<28} {'Calls':>8} {'Total ms':>12} {'Avg ms':>10} "
            f"{'Elems':>14} {'GB/s':>10}"
        )
        print(header)
        print("-" * len(header))
        print(
            f"{'policy_infer':<28} {calls:>8d} {total_ms:12.2f} {avg_ms:10.3f} {elems:14d} {gbps:10.2f}"
        )
        print("=" * 100)
        if reset:
            self._stats = {
                "policy_infer": {"time_ms": 0.0, "count": 0, "elements": 0.0, "bytes": 0.0}
            }


_POLICY_PROFILER = _PolicyCallProfiler()

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _duquant_report_if_enabled(context: str) -> None:
    profiler = _DUQUANT_PROFILER
    if profiler is not None and getattr(profiler, "enabled", False):
        profiler.report(reset=True, header_suffix=f"[{context}]")


def _policy_report_if_enabled(context: str, *, reset: bool = False) -> None:
    if _POLICY_PROFILER.enabled:
        _POLICY_PROFILER.report(context, reset=reset)


def _maybe_register_linear_shape_logging(model) -> None:
    flag = os.environ.get("OPENPI_PRINT_LINEAR_SHAPES", "0")
    if flag in ("0", "false", "False"):
        return
    try:
        import torch
    except ImportError:  # pragma: no cover - torch missing
        logging.warning("OPENPI_PRINT_LINEAR_SHAPES set but torch is unavailable.")
        return

    target_prefixes = (
        "paligemma_with_expert.paligemma.model.language_model.",
        "paligemma_with_expert.gemma_expert.model.",
    )

    handles = []

    def _should_track(name: str) -> bool:
        return any(name.startswith(prefix) for prefix in target_prefixes)

    def _make_hook(module_name: str):
        def _hook(mod: torch.nn.Module, inputs, _output):
            if not inputs:
                print(f"[LINEAR-MM] {module_name}: missing input tensor")
                return
            x = inputs[0]
            x_shape = tuple(x.shape) if hasattr(x, "shape") else "unknown"
            weight = getattr(mod, "weight", None)
            w_shape = tuple(weight.shape) if isinstance(weight, torch.Tensor) else "unknown"
            x_shape_str = list(x_shape) if isinstance(x_shape, tuple) else x_shape
            w_shape_str = list(w_shape) if isinstance(w_shape, tuple) else w_shape
            print(f"[LINEAR-MM] {module_name}: x{x_shape_str} @ W{w_shape_str}")

        return _hook

    linear_cls = torch.nn.Linear
    for name, module in model.named_modules():
        if isinstance(module, linear_cls) and _should_track(name):
            handles.append(module.register_forward_hook(_make_hook(name)))

    if handles:
        print(f"[LINEAR-MM] logging enabled on {len(handles)} Linear layers (LLM/DiT scopes).")
        setattr(model, "_linear_shape_logging_handles", handles)


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Headless mode parameters
    #################################################################################################################
    headless: bool = False  # Run without WebSocket/UI (local direct inference)
    policy_config: str = "pi05_libero"  # Policy config name (for headless mode)
    policy_dir: str = ""  # Policy checkpoint directory (for headless mode)

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_out_path: str = "results/libero"  # Path to save evaluation results

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    custom_order = os.environ.get("LIBERO_TASK_ORDER", "").strip()
    if custom_order:
        try:
            raw_indices = [int(token.strip()) for token in custom_order.split(",") if token.strip() != ""]
        except ValueError:
            logging.warning("Invalid LIBERO_TASK_ORDER value '%s'; expected comma-separated integers.", custom_order)
        else:
            seen: set[int] = set()
            ordered_indices: list[int] = []
            for idx in raw_indices:
                if 0 <= idx < num_tasks_in_suite and idx not in seen:
                    ordered_indices.append(idx)
                    seen.add(idx)
            for idx in range(num_tasks_in_suite):
                if idx not in seen:
                    ordered_indices.append(idx)
            if len(ordered_indices) == num_tasks_in_suite:
                task_suite.tasks = [task_suite.tasks[i] for i in ordered_indices]
                logging.info("Applied custom LIBERO task order: %s", ordered_indices)
            else:
                logging.warning(
                    "LIBERO_TASK_ORDER did not cover all tasks (got %s); using default ordering.",
                    custom_order,
                )

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.results_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    # Initialize policy client (WebSocket or Local)
    if args.headless:
        logging.info("Running in HEADLESS mode - no WebSocket connection")
        # Import locally to avoid dependency issues when not using headless mode
        from openpi_client import local_policy as _local_policy
        from openpi.policies import policy_config as _policy_config
        from openpi.training import config as _config
        import torch

        # Set deterministic mode for reproducibility (optional, controlled by env var)
        # Deterministic mode is slower (~5-15%) but ensures fully reproducible results
        # Set OPENPI_DETERMINISTIC=1 to enable, default is disabled for maximum speed
        enable_deterministic = os.environ.get("OPENPI_DETERMINISTIC", "0") == "1"
        if enable_deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.benchmark = False
            logging.info("Deterministic mode ENABLED (slower but fully reproducible)")
            logging.info("Remember to set: export CUBLAS_WORKSPACE_CONFIG=:4096:8")
        else:
            torch.backends.cudnn.benchmark = True
            logging.info("Deterministic mode DISABLED (faster, ~5-15% speedup)")

        # Load policy directly
        if not args.policy_dir:
            # Use default checkpoint from environment variable or error
            args.policy_dir = os.environ.get("CKPT", "")
            if not args.policy_dir:
                raise ValueError("--policy-dir or CKPT environment variable must be set in headless mode")

        logging.info(f"Loading policy: config={args.policy_config}, dir={args.policy_dir}")
        policy_obj = _policy_config.create_trained_policy(
            _config.get_config(args.policy_config),
            args.policy_dir,
            default_prompt=None,
        )
        if getattr(policy_obj, "_is_pytorch_model", False):
            _maybe_register_linear_shape_logging(policy_obj._model)

        client = _local_policy.LocalPolicy(policy_obj)
    else:
        logging.info("Running in WebSocket mode")
        client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    all_infer_ms: list[float] = []
    all_results = []  # Store results for CSV/JSON export

    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            episode_infer_ms: list[float] = []

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action and record inference cost
                        call_start = time.perf_counter()
                        infer_result = client.infer(element)
                        elapsed_ms = (time.perf_counter() - call_start) * 1000.0
                        action_chunk = infer_result["actions"]
                        infer_ms = infer_result.get("policy_timing", {}).get("infer_ms")
                        recorded_ms = float(infer_ms) if infer_ms is not None else elapsed_ms
                        episode_infer_ms.append(recorded_ms)
                        try:
                            import torch  # type: ignore
                        except ImportError:  # pragma: no cover - torch missing
                            torch = None  # type: ignore

                        if torch is not None and torch.is_tensor(action_chunk):
                            elems = int(action_chunk.numel())
                            byte_size = int(action_chunk.element_size() * action_chunk.numel())
                        else:
                            action_chunk_np = np.asarray(action_chunk)
                            elems = int(action_chunk_np.size)
                            byte_size = int(action_chunk_np.nbytes)
                        _POLICY_PROFILER.record(
                            recorded_ms,
                            elements=elems,
                            byte_size=byte_size,
                        )
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Store episode results
            episode_result = {
                "task_id": task_id,
                "task_name": task_description,
                "episode_idx": episode_idx,
                "success": bool(done),
                "steps": t,
                "max_steps": max_steps + args.num_steps_wait,
            }
            all_results.append(episode_result)

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            if episode_infer_ms:
                total_ms = sum(episode_infer_ms)
                avg_ms = total_ms / len(episode_infer_ms)
                throughput = len(episode_infer_ms) / (total_ms / 1000.0)
                logging.info(
                    f"[PERF] Episode {total_episodes}: {len(episode_infer_ms)} policy calls, "
                    f"avg {avg_ms:.1f} ms, throughput {throughput:.2f} calls/s"
                )
                all_infer_ms.extend(episode_infer_ms)
            _duquant_report_if_enabled(f"episode {total_episodes}")
            _policy_report_if_enabled(f"episode {total_episodes}", reset=True)

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    _policy_report_if_enabled("overall", reset=False)
    if all_infer_ms:
        total_ms = sum(all_infer_ms)
        total_calls = len(all_infer_ms)
        avg_ms = total_ms / total_calls
        throughput = total_calls / (total_ms / 1000.0)
        logging.info(
            f"[PERF] Overall policy throughput: {total_calls} calls, avg {avg_ms:.1f} ms, "
            f"throughput {throughput:.2f} calls/s"
        )

    # Save results to CSV and JSON
    timestamp = pathlib.Path(args.results_out_path).stem
    csv_path = pathlib.Path(args.results_out_path) / f"{args.task_suite_name}_results.csv"
    json_path = pathlib.Path(args.results_out_path) / f"{args.task_suite_name}_results.json"

    # Save CSV
    with open(csv_path, "w", newline="") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
    logging.info(f"Results saved to: {csv_path}")

    # Save JSON with summary statistics
    summary = {
        "task_suite": args.task_suite_name,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0,
        "seed": args.seed,
        "headless": args.headless,
        "results": all_results,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    logging.info(f"Results saved to: {json_path}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
