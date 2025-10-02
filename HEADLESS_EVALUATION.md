# Headless Evaluation Mode - Implementation Summary

## Overview

This document describes the headless evaluation mode implemented for LIBERO benchmarking, which completely bypasses WebSocket communication to avoid timeout errors (error 1011) during long inference runs with quantization.

## Problem Statement

When running LIBERO evaluation with DuQuant quantization, inference can be slow, causing WebSocket keepalive ping timeouts (error 1011). This results in:
- All tasks failing with 1011 errors
- Success rate of 0% despite correct implementation
- Need to restart server/client repeatedly

## Solution Architecture

### 1. Local Policy Wrapper (`LocalPolicy`)

**File:** `packages/openpi-client/src/openpi_client/local_policy.py`

A new policy wrapper that implements the `BasePolicy` interface but directly calls the policy object instead of using WebSocket:

```python
class LocalPolicy(BasePolicy):
    def infer(self, obs: Dict) -> Dict:
        return self._policy.infer(obs)  # Direct call, no network
```

### 2. Modified Evaluation Script

**File:** `examples/libero/main.py`

**New parameters:**
- `--args.headless`: Enable headless mode (no WebSocket)
- `--args.policy-config`: Policy configuration name (e.g., "pi05_libero")
- `--args.policy-dir`: Path to checkpoint directory
- `--args.results-out-path`: Where to save evaluation results

**Key changes:**

1. **Conditional policy loading:**
   ```python
   if args.headless:
       # Load policy directly in same process
       policy_obj = create_trained_policy(config, checkpoint_dir)
       client = LocalPolicy(policy_obj)
   else:
       # Use WebSocket client (original behavior)
       client = WebsocketClientPolicy(host, port)
   ```

2. **Deterministic evaluation:**
   ```python
   torch.use_deterministic_algorithms(True, warn_only=True)
   torch.backends.cudnn.benchmark = False
   ```

3. **Policy warmup:**
   - Run dummy inference before evaluation to:
     - Trigger JIT/torch.compile compilation
     - Allocate GPU memory
     - Initialize quantization tables

4. **Results logging:**
   - Every episode result stored in memory
   - CSV export: `task_id, task_name, episode_idx, success, steps, max_steps`
   - JSON export: Full results + summary statistics

### 3. Helper Script

**File:** `examples/libero/run_headless_eval.sh`

Bash script that:
- Checks environment variables (`CKPT`)
- Sets up Python path
- Runs evaluation with common DuQuant parameters
- Provides clear status messages

## Usage

### Basic Usage

```bash
source examples/libero/.venv-libero/bin/activate
export PYTHONPATH=$PWD/src:$PWD/third_party/libero
export CKPT=/path/to/checkpoint

python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 20
```

### With DuQuant Quantization

```bash
OPENPI_DUQUANT_DEBUG=1 \
OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model." \
OPENPI_DUQUANT_WBITS_DEFAULT=4 \
OPENPI_DUQUANT_ABITS=8 \
python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 20
```

### Using Helper Script

```bash
export CKPT=/path/to/checkpoint
export TASK_SUITE=libero_spatial
export NUM_TRIALS=20
./examples/libero/run_headless_eval.sh
```

## Output Files

### CSV Results
**Location:** `results/libero/{task_suite}_results.csv`

Format:
```csv
task_id,task_name,episode_idx,success,steps,max_steps
0,put the black bowl on the plate,0,True,45,230
0,put the black bowl on the plate,1,False,230,230
...
```

### JSON Results
**Location:** `results/libero/{task_suite}_results.json`

Format:
```json
{
  "task_suite": "libero_spatial",
  "total_episodes": 200,
  "total_successes": 195,
  "success_rate": 0.975,
  "seed": 42,
  "headless": true,
  "results": [...]
}
```

### Videos
**Location:** `data/libero/videos/*.mp4`

One video per episode showing the robot's perspective.

## Key Benefits

✅ **No WebSocket errors:** Eliminates 1011 keepalive timeout issues

✅ **Deterministic:** Fixed seeds ensure reproducible results

✅ **Faster:** No network serialization/deserialization overhead

✅ **Quantization-friendly:** Works seamlessly with slow DuQuant inference

✅ **Results tracking:** Automatic CSV/JSON export for analysis

✅ **Backward compatible:** Original WebSocket mode still works when `--headless` not specified

## Implementation Details

### Evaluation Loop

The evaluation loop remains unchanged - it still calls `client.infer(obs)`. The only difference is what happens inside:

**WebSocket mode:**
```
client.infer(obs)
  → msgpack.pack(obs)
  → websocket.send(data)
  → websocket.recv()
  → msgpack.unpack(response)
```

**Headless mode:**
```
client.infer(obs)
  → policy.infer(obs)
  → model.sample_actions(obs)
  → return actions
```

### Environment Setup

- Uses same LIBERO environment as WebSocket mode
- Offscreen rendering (EGL/OSMesa)
- No changes to physics simulation or action execution

### Reproducibility

To ensure deterministic results:
1. Set `torch.use_deterministic_algorithms(True)`
2. Disable cuDNN benchmark mode
3. Use fixed random seeds for numpy and torch
4. Seed LIBERO environment with `base_seed + episode_idx`

## Verification Checklist

- [x] No WebSocket connection created in headless mode
- [x] No "waiting for server" or "1011" errors
- [x] Results saved to CSV with all episode data
- [x] Results saved to JSON with summary statistics
- [x] Videos still generated (offscreen rendering)
- [x] DuQuant quantization works correctly
- [x] Success rates match expectations
- [x] Backward compatible (WebSocket mode still works)

## Testing

To verify the implementation works:

```bash
# Set checkpoint path
export CKPT=/path/to/your/checkpoint

# Run short evaluation (2 episodes per task)
export NUM_TRIALS=2
export TASK_SUITE=libero_spatial
./examples/libero/run_headless_eval.sh

# Check outputs
ls results/libero/
ls data/libero/videos/

# Verify no WebSocket errors in logs
# Should see "Running in HEADLESS mode - no WebSocket connection"
```

## Future Enhancements

Possible improvements:
- [ ] Add progress bar with ETA for long evaluations
- [ ] Support video recording toggle (for faster evaluation)
- [ ] Add real-time plotting of success rates
- [ ] Support multi-GPU evaluation (parallel episodes)
- [ ] Add checkpoint resuming for interrupted runs

## Technical Notes

### Why Not Use Multiprocessing?

We considered using separate processes for policy and environment, but this would still require some form of IPC (shared memory, pipes, etc.) and wouldn't solve the fundamental issue of slow inference blocking the evaluation loop.

### Why Not Increase WebSocket Timeout?

WebSocket keepalive timeouts are designed to detect network failures. Increasing them arbitrarily is not a clean solution and can mask real connectivity issues.

### Policy Memory Management

The policy object is loaded once at startup and kept in memory for all episodes. This ensures:
- Consistent inference speed across episodes
- No model reloading overhead
- Stable GPU memory usage

## References

- Original evaluation script: `examples/libero/main.py`
- WebSocket policy client: `packages/openpi-client/src/openpi_client/websocket_client_policy.py`
- Base policy interface: `packages/openpi-client/src/openpi_client/base_policy.py`
- Policy implementation: `src/openpi/policies/policy.py`