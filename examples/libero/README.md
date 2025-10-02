# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv

source ~/VLM_REPO/openpi/examples/libero/.venv/bin/activate
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py




rm -rf examples/libero/.venv-libero
uv python install 3.9
uv venv --python 3.9 examples/libero/.venv-libero
module load python/3.9
python3.9 -m venv examples/libero/.venv-libero
source examples/libero/.venv-libero/bin/activate

# 2) 装 pip，并升级
python -m ensurepip --upgrade
python -m pip install --upgrade pip

# 3) 仅按 README 安装 LIBERO 依赖（注意 cu113 源）
pip install -r examples/libero/requirements.txt \
            -r third_party/libero/requirements.txt \
            --extra-index-url https://download.pytorch.org/whl/cu113

# 4) 装 openpi-client 与必需包
pip install -e packages/openpi-client
pip install sentencepiece    # tokenizer 用

# 5) 可选：执行 robosuite 宏生成（warning 不影响运行）
python $(python -c "import robosuite, pathlib; \
    print(pathlib.Path(robosuite.__file__).parent / 'scripts' / 'setup_macros.py')")

# 6) 启动客户端前设好 PYTHONPATH
export PYTHONPATH=$PWD/src:$PWD/third_party/libero
export CKPT=/global/homes/y/yunta/repo/openpi/ckpts/pi05_libero_torch
python examples/libero/main.py \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 20 \
  --args.video-out-path data/libero/videos \
  --args.seed 42

export CKPT=/path/to/your/checkpoint

cd ~/VLM_REPO/openpi
source examples/libero/.venv/bin/activate

export PYTHONPATH=$PWD/src:$PWD/third_party/libero
# export CUBLAS_WORKSPACE_CONFIG=:4096:8

# 完整体 DuQuant 默认配置
export OPENPI_DUQUANT_DEBUG=1
export OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."
export OPENPI_DUQUANT_WBITS_DEFAULT=4
export OPENPI_DUQUANT_ABITS=8
export OPENPI_DUQUANT_BLOCK=16           # 默认 16，比你的 64 快
export OPENPI_DUQUANT_PERMUTE=1          # 默认开启，比你的 0 好
export OPENPI_DUQUANT_ROW_ROT=restore    # 默认 restore，比你的 0 好
export OPENPI_DUQUANT_ACT_PCT=99.9       # 默认 99.9，比你的 98.0 好
export OPENPI_DUQUANT_CALIB_STEPS=32     # 默认 32，比你的 64 快
export OPENPI_DUQUANT_LS=0.15            # lambda smooth 默认值

# Pack 目录命名：b16_p1_rrestore_a999
export OPENPI_DUQUANT_PACKDIR="duquant_packed_b16_p1_rrestore_a999"

python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 20 \
  --args.seed 42

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```


## Headless Evaluation Mode (No WebSocket)

To avoid WebSocket timeout issues (error 1011), you can run evaluation in **headless mode** where the policy runs directly in the same process without any network communication.

### Prerequisites

Headless mode requires both LIBERO and OpenPI dependencies. Choose one setup method:

**Method A: Quick setup (if you have existing .venv-libero)**
```bash
source examples/libero/.venv-libero/bin/activate
pip install -e .  # Install OpenPI from project root
```

**Method B: Complete new environment (recommended)**
```bash
bash examples/libero/setup_headless_env.sh
source examples/libero/.venv-headless/bin/activate
```

See [ENVIRONMENT_SETUP.md](ENVIRONMENT_SETUP.md) for detailed instructions.

### Option 1: Using the helper script (recommended)

```bash
export CKPT=/path/to/your/checkpoint

# Use DuQuant defaults (block=16, permute=1, row-rot=restore)
export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch
DUQUANT_PRESET=default bash examples/libero/run_headless_eval.sh

# OR use custom config (block=64, permute=0, row-rot=0, act_pct=98.0)
DUQUANT_PRESET=custom bash examples/libero/run_headless_eval.sh
# OR simply (custom is the default):
bash examples/libero/run_headless_eval.sh
```

### Option 2: Direct Python command

```bash
source examples/libero/.venv-libero/bin/activate
export PYTHONPATH=$PWD/src:$PWD/third_party/libero
export CKPT=/path/to/your/checkpoint  # e.g., ~/VLM_REPO/openpi/ckpts/pi05_libero_torch

# A) Use DuQuant internal defaults (block=16, permute on, row-rot restore)
OPENPI_DUQUANT_DEBUG=1 \
OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model." \
python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 20 \
  --args.seed 42

# B) Use custom explicit parameters (block=64, permute=0, row-rot=0)
OPENPI_DUQUANT_DEBUG=1 \
OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model." \
OPENPI_DUQUANT_WBITS_DEFAULT=4 \
export CUBLAS_WORKSPACE_CONFIG=:4096:8\
OPENPI_DUQUANT_ABITS=8 \
OPENPI_DUQUANT_ACT_PCT=98.0 \
OPENPI_DUQUANT_BLOCK=64 \
OPENPI_DUQUANT_PERMUTE=0 \
OPENPI_DUQUANT_ROW_ROT=0 \
OPENPI_DUQUANT_CALIB_STEPS=64 \
python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 20 \
  --args.seed 42
```
export CUBLAS_WORKSPACE_CONFIG=:4096:8\
export PYTHONWARNINGS="ignore::UserWarning"
OPENPI_DUQUANT_DEBUG=1 \
OPENPI_DUQUANT_PACKDIR="duquant_packed" \
OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model." \
python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name libero_spatial \
  --args.num-trials-per-task 20 \
  --args.seed 42

**DuQuant Configuration Comparison:**

| Config | BLOCK | PERMUTE | ROW_ROT | ACT_PCT | CALIB_STEPS | Notes |
|--------|-------|---------|---------|---------|-------------|-------|
| **Default** | 16 (internal) | 1 (internal) | restore (internal) | - | - | Uses DuQuant's internal defaults |
| **Custom** | 64 | 0 | 0 | 98.0 | 64 | Explicit parameters for faster inference |

Choose "default" if you want to match the original DuQuant paper settings, or "custom" for potentially faster inference.

**Key benefits of headless mode:**
- ✅ No WebSocket connection or timeout errors
- ✅ Direct local inference: `policy.predict()` → `env.step()`
- ✅ Deterministic evaluation with fixed seeds
- ✅ Results automatically saved to CSV and JSON
- ✅ Works seamlessly with DuQuant quantization

**Output:**
- Videos: `data/libero/videos/*.mp4`
- Results: `results/libero/{task_suite}_results.csv` and `.json`

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8/97 | 98.2 | 98.0 | 92.4 | 96.85

## DuQuant PTQ (optional)

- Dry-run which Linear layers would be quantized (no replacement):
  - `export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch`
  - `
  OPENPI_DUQUANT_DEBUG=1
  OPENPI_DUQUANT_DRYRUN=1 OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model."
    uv run --active scripts/serve_policy.py --env LIBERO \
      policy:checkpoint --policy.config=pi05_libero --policy.dir="$CKPT"`

- Run PTQ simulation (default W4A8, block=16, permute on, input x->xPR_in, row-rot restore):
  - `CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch`
  CKPT=/global/homes/y/yunta/repo/openpi/ckpts/pi05_libero_torch
  export OPENPI_DISABLE_TORCH_COMPILE=0
  OPENPI_DUQUANT_DEBUG=1
  OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model." \
    uv run --active scripts/serve_policy.py --env LIBERO \
      policy:checkpoint --policy.config=pi05_libero --policy.dir="$CKPT"`

OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model." \ python scripts/serve_policy.py --env LIBERO \ policy:checkpoint --policy.config=pi05_libero --policy.dir="$CKPT"

- Change bits/settings:
  - `OPENPI_DUQUANT_WBITS_DEFAULT=4 OPENPI_DUQUANT_ABITS=8 OPENPI_DUQUANT_BLOCK=16 OPENPI_DUQUANT_PERMUTE=1 OPENPI_DUQUANT_LS=0.15 \
      uv run --active scripts/serve_policy.py --env LIBERO \
        policy:checkpoint --policy.config=pi05_libero --policy.dir="$CKPT"`
  - Per-layer bits: `OPENPI_DUQUANT_WBITS="paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj:6"`
  - Include/exclude by regex: `OPENPI_DUQUANT_INCLUDE` / `OPENPI_DUQUANT_EXCLUDE`
  - Whitelist exact names: `OPENPI_DUQUANT_LAYERS="a,b,c"`
  - Input transform (R vs R.T): default uses R; set `OPENPI_DUQUANT_INPUT_TRANSPOSE=1` to use R.T
  - Row rotation mode: `OPENPI_DUQUANT_ROW_ROT={0,restore,propagate}` (default restore); row block size `OPENPI_DUQUANT_BLOCK_OUT`



export OPENPI_DUQUANT_PACKDIR="duquant_packed_p64_pct99"   # 新的打包目录
export CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch
export OPENPI_DISABLE_TORCH_COMPILE=1
OPENPI_DUQUANT_DEBUG=1
OPENPI_DUQUANT_SCOPE="paligemma_with_expert.gemma_expert.model." \
OPENPI_DUQUANT_WBITS_DEFAULT=4 OPENPI_DUQUANT_ABITS=8 \
OPENPI_DUQUANT_ACT_PCT=98.0 OPENPI_DUQUANT_BLOCK=64 OPENPI_DUQUANT_PERMUTE=0 \
OPENPI_DUQUANT_ROW_ROT=0 OPENPI_DUQUANT_CALIB_STEPS=64 \
uv run --active scripts/serve_policy.py --env LIBERO \
  policy:checkpoint --policy.config=pi05_libero --policy.dir="$CKPT"

- Evaluate variants and generate a Markdown report:
  - `CKPT=~/VLM_REPO/openpi/ckpts/pi05_libero_torch uv run --active bash examples/libero/eval_libero.sh libero_10 5 duquant_eval.md`

- List all Linear layers (filtered by scope):
  - `uv run --active scripts/list_linears.py pi05_libero $CKPT --scope paligemma_with_expert.gemma_expert.model.`
