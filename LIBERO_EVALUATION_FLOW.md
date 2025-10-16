# LIBERO 评估流程详解

本文档详细说明 OpenPI 在 LIBERO 任务上的完整评估流程，从脚本启动到模型推理的每一个步骤。

## 1. 整体流程概览

```
bash run_llm_dit_mlp_w4a8.sh
    ↓
examples/libero/main.py (入口)
    ↓
eval_libero() 函数
    ↓
创建 Policy 对象 (包含模型 + transforms)
    ↓
LIBERO 环境循环 (每个任务 × 每个episode)
    ↓
Policy.infer() 推理
    ↓
环境执行动作
```

## 2. 详细调用链

### 阶段 1: 脚本启动 (run_llm_dit_mlp_w4a8.sh)

**文件**: `examples/libero/run_llm_dit_mlp_w4a8.sh`

**关键步骤**:
1. 设置环境变量 (PYTHONPATH, CUDA配置等)
2. 设置 DuQuant 配置 (OPENPI_DUQUANT_*)
3. 启动 Python 脚本

```bash
time python examples/libero/main.py \
  --args.headless \
  --args.policy-config pi05_libero \
  --args.policy-dir "$CKPT" \
  --args.task-suite-name "$TASK_SUITE" \
  --args.num-trials-per-task 20 \
  --args.seed "$SEED"
```

---

### 阶段 2: 主程序入口 (main.py)

**文件**: `examples/libero/main.py`

**函数**: `eval_libero(args: Args)`

**行号**: 171-438

#### 2.1 初始化阶段 (行 173-196)

```python
# 设置随机种子
np.random.seed(args.seed)

# 加载 LIBERO benchmark
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[args.task_suite_name]()  # 例如: libero_spatial

# 根据任务类型设置最大步数
if args.task_suite_name == "libero_spatial":
    max_steps = 220
elif args.task_suite_name == "libero_10":
    max_steps = 520
# ...
```

#### 2.2 创建 Policy 对象 (行 198-235)

**关键代码**:
```python
# 导入本地 policy 模块
from openpi_client import local_policy as _local_policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

# 加载 policy
policy_obj = _policy_config.create_trained_policy(
    _config.get_config(args.policy_config),  # 加载 pi05_libero 配置
    args.policy_dir,                          # checkpoint 目录
    default_prompt=None,
)

# 包装成 LocalPolicy 客户端
client = _local_policy.LocalPolicy(policy_obj)
```

**调用链**:
```
_policy_config.create_trained_policy()
    → src/openpi/policies/policy_config.py:16
```

---

### 阶段 3: 创建 Policy (policy_config.py)

**文件**: `src/openpi/policies/policy_config.py`

**函数**: `create_trained_policy()`

**行号**: 16-102

#### 3.1 检测模型类型 (行 48-50)

```python
# 检查是否是 PyTorch 模型
weight_path = os.path.join(checkpoint_dir, "model.safetensors")
is_pytorch = os.path.exists(weight_path)
```

#### 3.2 加载 PyTorch 模型 (行 53-65)

```python
if is_pytorch:
    # 调用 ModelConfig.load_pytorch()
    model = train_config.model.load_pytorch(train_config, weight_path)

    # 转换部分参数到 bfloat16
    model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")

    # 🔥 关键步骤：启用 DuQuant 量化 🔥
    from openpi.models_pytorch.duquant_layers import enable_duquant_if_configured
    enable_duquant_if_configured(model)
```

**调用链**:
```
train_config.model.load_pytorch()
    → src/openpi/models/model.py:285

enable_duquant_if_configured()
    → src/openpi/models_pytorch/duquant_layers.py:422
```

#### 3.3 创建 Policy 对象 (行 85-102)

```python
return _policy.Policy(
    model,
    transforms=[
        # 输入 transforms: 图像预处理、标准化等
        transforms.InjectDefaultPrompt(default_prompt),
        *data_config.data_transforms.inputs,
        transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.model_transforms.inputs,
    ],
    output_transforms=[
        # 输出 transforms: 反标准化等
        *data_config.model_transforms.outputs,
        transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
        *data_config.data_transforms.outputs,
    ],
    sample_kwargs=sample_kwargs,
    is_pytorch=is_pytorch,
    pytorch_device=pytorch_device,
)
```

---

### 阶段 4: 加载 PyTorch 模型 (model.py)

**文件**: `src/openpi/models/model.py`

**函数**: `load_pytorch()`

**行号**: 285-289

```python
def load_pytorch(self, train_config, weight_path: str):
    logger.info(f"train_config: {train_config}")

    # 创建 PI0Pytorch 模型实例
    model = pi0_pytorch.PI0Pytorch(config=train_config.model)

    # 从 safetensors 文件加载权重
    safetensors.torch.load_model(model, weight_path)

    return model
```

**PI0Pytorch 模型结构** (在 `src/openpi/models_pytorch/pi0_pytorch.py` 中定义):

```python
class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.paligemma_with_expert = PaliGemmaWithExpert(config)
        # paligemma_with_expert 包含:
        # - paligemma.vision_tower (SigLIP 视觉编码器)
        # - paligemma.language_model (Gemma LLM)
        # - gemma_expert (DiT transformer)
```

---

### 阶段 5: 启用 DuQuant 量化 (duquant_layers.py)

**文件**: `src/openpi/models_pytorch/duquant_layers.py`

**函数**: `enable_duquant_if_configured(model)`

**行号**: 422-467

#### 5.1 检查环境变量 (行 430-434)

```python
env = os.environ
keys = [k for k in env.keys() if k.startswith("OPENPI_DUQUANT_")]
activate = any(k not in ("OPENPI_DUQUANT_PACKDIR",) for k in keys)
if not activate:
    return  # 没有设置 DuQuant 配置，直接返回
```

#### 5.2 读取配置 (行 437-444)

```python
scope = env.get("OPENPI_DUQUANT_SCOPE", "policy.dit.")
inc = env.get("OPENPI_DUQUANT_INCLUDE", r".*(q_proj|k_proj|v_proj|o_proj|...).*")
exc = env.get("OPENPI_DUQUANT_EXCLUDE", r"(?:^|\.)(norm|ln|...)(?:\.|$)")
per_layer_wbits = _parse_per_layer_wbits(env.get("OPENPI_DUQUANT_WBITS"))
dry_run = env.get("OPENPI_DUQUANT_DRYRUN", "0") not in ("0", "false", "False")

# 创建 DuQuant 配置对象
cfg = DuQuantConfig()
```

**DuQuantConfig 默认值** (行 28-62):
```python
@dataclasses.dataclass
class DuQuantConfig:
    weight_bits: int = int(os.environ.get("OPENPI_DUQUANT_WBITS_DEFAULT", "4"))
    act_bits: int = int(os.environ.get("OPENPI_DUQUANT_ABITS", "0"))  # 0=禁用激活量化
    block_size: int = int(os.environ.get("OPENPI_DUQUANT_BLOCK", "128"))
    block_out_size: int | None = None  # 输出通道的 block size (默认同 block_size)
    enable_permute: bool = os.environ.get("OPENPI_DUQUANT_PERMUTE", "0") not in ("0", "false", "False")
    row_rot_mode: str = os.environ.get("OPENPI_DUQUANT_ROW_ROT", "disabled")  # disabled/restore/propagate
    act_percentile: float = float(os.environ.get("OPENPI_DUQUANT_ACT_PCT", "99.9"))
    calib_steps: int = int(os.environ.get("OPENPI_DUQUANT_CALIB_STEPS", "32"))
    lambda_smooth: float = float(os.environ.get("OPENPI_DUQUANT_LS", "0.5"))
```

#### 5.3 选择目标层 (行 447-454)

```python
# 使用正则表达式匹配层名
targets = select_targets(
    model,
    include_regex=inc,   # 匹配需要量化的层
    exclude_regex=exc,   # 排除不需要量化的层
    scope_prefix=scope,  # 只在指定 scope 内搜索
    whitelist=whitelist_list,
)
```

**select_targets() 函数** (行 351-377):
```python
def select_targets(model, *, include_regex, exclude_regex, scope_prefix, ...):
    inc = re.compile(include_regex)
    exc = re.compile(exclude_regex)
    results = []

    # 遍历模型所有模块
    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue  # 只处理 Linear 层

        # 检查是否在 scope 内
        if scope_prefix is not None and not name.startswith(scope_prefix):
            continue

        # 检查是否匹配 INCLUDE 正则
        if not inc.search(name):
            continue

        # 检查是否匹配 EXCLUDE 正则
        if exc.search(name):
            continue

        results.append((name, mod))

    return results
```

**你的配置会匹配**:
- ✅ `language_model.layers.*.self_attn.q_proj` (LLM attention)
- ✅ `language_model.layers.*.mlp.gate_proj` (LLM MLP)
- ✅ `gemma_expert.model.layers.*.mlp.gate_proj` (DiT MLP)
- ❌ `gemma_expert.model.layers.*.self_attn.q_proj` (DiT attention - 被 EXCLUDE 排除)

#### 5.4 替换为 DuQuantLinear (行 456-467)

```python
if targets:
    print(f"[DUQUANT] Matched Linear layers: {len(targets)}")

    # 调用 wrap_duquant 替换层
    wrap_duquant(
        model,
        layer_names=[name for name, _ in targets],
        cfg=cfg,
        per_layer_wbits=per_layer_wbits,
        dry_run=dry_run,
    )
```

**wrap_duquant() 函数** (行 380-419):
```python
def wrap_duquant(model, layer_names, cfg, per_layer_wbits, dry_run):
    replaced = 0
    for name in layer_names:
        # 获取父模块和属性名
        parent, attr = _get_parent_module_and_attr(model, name)
        mod = getattr(parent, attr)  # 原始的 nn.Linear

        if dry_run:
            print(f"[DUQUANT][DRYRUN] {name}: Linear({mod.in_features}->{mod.out_features}) ...")
            continue

        # 创建 DuQuantLinear 包装器
        dq = DuQuantLinear(mod, name=name, cfg=cfg, weight_bits=wbits)

        # 替换原始层
        setattr(parent, attr, dq)

        print(f"[DUQUANT][REPLACED] {name}: Linear(...) -> DuQuantLinear W{wbits} A{cfg.act_bits}")
        replaced += 1

    print(f"[DUQUANT] Total layers replaced: {replaced}")
```

---

### 阶段 6: DuQuantLinear 初始化

**文件**: `src/openpi/models_pytorch/duquant_layers.py`

**类**: `DuQuantLinear`

**行号**: 66-188

#### 6.1 构造函数 (行 66-121)

```python
class DuQuantLinear(nn.Module):
    def __init__(
        self,
        orig_linear: nn.Linear,
        *,
        name: str,
        cfg: DuQuantConfig,
        weight_bits: int,
    ):
        super().__init__()
        self.name = name
        self.cfg = cfg
        self.weight_bits = weight_bits

        # 保存原始 Linear 层的参数
        self.in_features = orig_linear.in_features
        self.out_features = orig_linear.out_features
        self.bias = orig_linear.bias

        # 🔥 关键：尝试从 pack 文件加载预计算的变换矩阵
        packdir = os.environ.get("OPENPI_DUQUANT_PACKDIR")
        pack_path = Path(packdir) / f"{name}.npz" if packdir else None

        if pack_path and pack_path.exists():
            # 从磁盘加载
            self.pack = PackResult(**dict(np.load(pack_path, allow_pickle=True)))
            print(f"[DUQUANT][LOADED] {name}: pack from {pack_path}")
        else:
            # 第一次运行：在线计算并保存
            self.pack = duquant_pack_single_layer(
                orig_linear.weight.detach(),
                name=name,
                cfg=cfg,
                weight_bits=weight_bits,
            )
            if pack_path:
                np.savez(pack_path, **dataclasses.asdict(self.pack))
                print(f"[DUQUANT][PACKED] {name}: saved to {pack_path}")
```

#### 6.2 Pack 文件内容 (duquant_preprocess.py)

**PackResult 数据结构** (行 140-149):
```python
@dataclass
class PackResult:
    # 输入侧变换 (列变换)
    R_in_blocks: Dict[int, np.ndarray]   # block_index -> 旋转矩阵 R_in (BxB)
    perm: np.ndarray                      # 排列索引 (通道重排)

    # 输出侧变换 (行变换)
    R_out_blocks: Dict[int, np.ndarray]  # block_index -> 旋转矩阵 R_out (BxB)

    # 量化参数
    weight_scale: np.ndarray              # 每个输出通道的量化 scale
    meta: Dict[str, Any]                  # 元数据 (block_size, lambda_smooth 等)
```

**duquant_pack_single_layer() 函数** (行 673-836):
```python
def duquant_pack_single_layer(weight_tensor, *, name, cfg, weight_bits):
    """
    对单个 Linear 层执行 DuQuant 预处理，计算旋转矩阵、排列和量化 scale

    步骤:
    1. 输入通道分块 (block_size)
    2. 每个 block 计算旋转矩阵 R_in (使用 SVD)
    3. 计算 zigzag 排列 (基于权重能量)
    4. 计算输出旋转矩阵 R_out (可选)
    5. 计算量化 scale (每个输出通道)
    """
    W = weight_tensor.cpu().numpy().astype(np.float32)
    out_features, in_features = W.shape
    block_size = cfg.block_size

    # 步骤 1: 输入通道分块，计算旋转矩阵
    R_in_blocks = {}
    n_blocks = (in_features + block_size - 1) // block_size
    for b in range(n_blocks):
        start = b * block_size
        end = min((b + 1) * block_size, in_features)
        W_block = W[:, start:end]

        # 计算该 block 的旋转矩阵 (使用 SVD: W_block = U @ S @ Vt)
        R = compute_block_rotation(W_block)  # 返回 Vt 的前几行
        R_in_blocks[b] = R

    # 步骤 2: 应用旋转
    W_rotated = apply_rotation_to_weight(W, R_in_blocks, block_size)

    # 步骤 3: 计算 zigzag 排列
    if cfg.enable_permute:
        perm = compute_zigzag_permutation(W_rotated, block_size, cfg.lambda_smooth)
        W_permuted = W_rotated[:, perm]
    else:
        perm = None
        W_permuted = W_rotated

    # 步骤 4: 输出旋转 (可选)
    R_out_blocks = None
    if cfg.row_rot_mode != "disabled":
        R_out_blocks = compute_output_rotation(W_permuted, cfg.block_out_size)

    # 步骤 5: 计算量化 scale
    weight_scale = compute_weight_scale(W_permuted, weight_bits)

    return PackResult(
        R_in_blocks=R_in_blocks,
        perm=perm,
        R_out_blocks=R_out_blocks,
        weight_scale=weight_scale,
        meta={"block_size": block_size, "lambda_smooth": cfg.lambda_smooth},
    )
```

#### 6.3 预计算块对角矩阵 (行 142-188)

```python
# 批量旋转优化：将 128 个小矩阵乘法合并成 1 个大矩阵乘法
self._use_batched_rotation = os.environ.get("OPENPI_DUQUANT_BATCH_ROT", "1") not in ("0", "false", "False")
self.register_buffer("_R_in_all", None)
self.register_buffer("_R_out_all", None)

if self._use_batched_rotation:
    self._precompute_block_diagonal_matrices()
```

**_precompute_block_diagonal_matrices() 函数** (行 142-188):
```python
def _precompute_block_diagonal_matrices(self):
    """
    预计算块对角矩阵，加速前向传播

    原始方法: 对每个 block 分别做矩阵乘法
        for b in range(n_blocks):
            x_block = x[:, b*B:(b+1)*B]
            x_rot = x_block @ R_in[b]

    优化方法: 构造一个大的块对角矩阵，一次矩阵乘法完成
        R_all = block_diag(R_in[0], R_in[1], ..., R_in[n-1])
        x_rot = x @ R_all

    性能提升: 256 次小 matmul -> 2 次大 matmul (10-20x 加速)
    """
    if self._R_in_block_indices:
        # 构建输入旋转的块对角矩阵
        R_list = []
        for b in range(n_blocks):
            R = getattr(self, f"_R_in_{b}")
            R_list.append(R)

        # 使用 torch.block_diag 构造块对角矩阵
        R_all = torch.block_diag(*R_list)
        self._R_in_all = R_all

    # 同样处理输出旋转
    if self._R_out_block_indices:
        R_out_list = [getattr(self, f"_R_out_{b}") for b in range(n_out_blocks)]
        R_out_all = torch.block_diag(*R_out_list)
        self._R_out_all = R_out_all
```

---

### 阶段 7: LIBERO 环境循环 (main.py)

**文件**: `examples/libero/main.py`

**行号**: 245-398

#### 7.1 外层循环：遍历所有任务 (行 245-397)

```python
for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
    # 获取任务描述
    task = task_suite.get_task(task_id)
    task_description = task.language  # 例如: "put the red mug on the plate"

    # 获取初始状态
    initial_states = task_suite.get_task_init_states(task_id)

    # 创建环境
    env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
```

#### 7.2 内层循环：每个任务的多次试验 (行 257-393)

```python
    for episode_idx in range(args.num_trials_per_task):  # 默认 20 次
        # 重置环境
        env.reset()
        action_plan = collections.deque()  # 存储动作序列

        # 设置初始状态
        obs = env.set_init_state(initial_states[episode_idx])

        t = 0
        replay_images = []
        episode_infer_ms = []
```

#### 7.3 时间步循环 (行 273-354)

```python
        while t < max_steps + args.num_steps_wait:
            # 前 10 步等待物体稳定 (物理仿真需要时间)
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            # 获取图像观测
            img = obs["agentview_image"][::-1, ::-1]  # 旋转 180 度
            wrist_img = obs["robot0_eye_in_hand_image"][::-1, ::-1]

            # 预处理图像
            img = image_tools.resize_with_pad(img, 224, 224)
            wrist_img = image_tools.resize_with_pad(wrist_img, 224, 224)
```

#### 7.4 模型推理 (行 296-340)

```python
            if not action_plan:
                # 动作队列为空，需要重新规划

                # 准备输入数据
                element = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": np.concatenate((
                        obs["robot0_eef_pos"],        # 末端执行器位置 (3D)
                        _quat2axisangle(obs["robot0_eef_quat"]),  # 旋转 (轴角, 3D)
                        obs["robot0_gripper_qpos"],   # 夹爪开合 (1D)
                    )),
                    "prompt": task_description,  # 任务描述文本
                }

                # 🔥 调用模型推理 🔥
                call_start = time.perf_counter()
                infer_result = client.infer(element)
                elapsed_ms = (time.perf_counter() - call_start) * 1000.0

                # 获取动作序列 (例如: shape [15, 7])
                action_chunk = infer_result["actions"]

                # 只使用前 5 步 (replan_steps)
                action_plan.extend(action_chunk[:args.replan_steps])

            # 从队列中取出一个动作
            action = action_plan.popleft()

            # 执行动作
            obs, reward, done, info = env.step(action.tolist())

            if done:  # 任务成功
                task_successes += 1
                break
            t += 1
```

---

### 阶段 8: Policy 推理 (local_policy.py + policy.py)

**文件**: `src/openpi_client/local_policy.py`

**函数**: `LocalPolicy.infer()`

```python
class LocalPolicy:
    def __init__(self, policy_obj):
        self._policy = policy_obj

    def infer(self, element):
        """
        执行推理

        Args:
            element: 包含 observation 和 prompt 的字典

        Returns:
            {
                "actions": numpy array of shape [action_horizon, action_dim],
                "policy_timing": {"infer_ms": ...}
            }
        """
        # 调用 Policy 对象的 sample_actions
        actions = self._policy.sample_actions(element)
        return {"actions": actions}
```

**文件**: `src/openpi/policies/policy.py`

**函数**: `Policy.sample_actions()`

```python
class Policy:
    def __init__(self, model, transforms, output_transforms, sample_kwargs, ...):
        self.model = model
        self.transforms = transforms
        self.output_transforms = output_transforms
        self._is_pytorch_model = is_pytorch

    def sample_actions(self, data):
        """
        完整的推理流程

        步骤:
        1. 应用输入 transforms
        2. 调用模型前向传播
        3. 应用输出 transforms
        """
        # 步骤 1: 输入预处理
        for transform in self.transforms:
            data = transform(data)

        # 数据转换为模型输入格式
        batch = self._prepare_batch(data)

        # 步骤 2: 模型前向传播
        if self._is_pytorch_model:
            with torch.no_grad():
                output = self.model(batch)  # 🔥 调用 PI0Pytorch.forward()
        else:
            output = self.model(batch)

        # 步骤 3: 输出后处理
        actions = output["actions"]
        for transform in self.output_transforms:
            actions = transform(actions)

        return actions
```

---

### 阶段 9: 模型前向传播 (pi0_pytorch.py)

**文件**: `src/openpi/models_pytorch/pi0_pytorch.py`

**类**: `PI0Pytorch`

**函数**: `forward()`

```python
class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 主模型：PaliGemma + Expert DiT
        self.paligemma_with_expert = PaliGemmaWithExpert(config)

    def forward(self, batch):
        """
        前向传播

        输入:
            batch = {
                "observation": {
                    "image": [B, 224, 224, 3],
                    "wrist_image": [B, 224, 224, 3],
                    "state": [B, 7],  # eef_pos(3) + eef_rot(3) + gripper(1)
                },
                "prompt": [B, max_token_len],  # tokenized text
            }

        输出:
            {
                "actions": [B, action_horizon, action_dim],
            }
        """
        # 调用 PaliGemmaWithExpert
        return self.paligemma_with_expert(batch)
```

**PaliGemmaWithExpert 结构**:
```python
class PaliGemmaWithExpert(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 视觉编码器 (SigLIP)
        self.paligemma = PaliGemmaForConditionalGeneration(...)

        # DiT expert (动作预测)
        self.gemma_expert = GemmaExpert(config)

    def forward(self, batch):
        # 1. 编码图像
        vision_features = self.paligemma.vision_tower(batch["observation"]["image"])

        # 2. LLM 处理文本 + 视觉特征
        text_embeddings = self.paligemma.language_model(
            input_ids=batch["prompt"],
            vision_features=vision_features,
        )

        # 3. DiT 预测动作
        # 🔥 这里会经过 DuQuantLinear 层 🔥
        actions = self.gemma_expert(
            text_embeddings=text_embeddings,
            state=batch["observation"]["state"],
        )

        return {"actions": actions}
```

---

### 阶段 10: DuQuantLinear 前向传播

**文件**: `src/openpi/models_pytorch/duquant_layers.py`

**函数**: `DuQuantLinear.forward()`

**行号**: 280-340

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    DuQuant 前向传播

    步骤:
    1. 应用输入变换 (旋转 + 排列)
    2. 激活量化 (如果启用)
    3. 权重量化
    4. 矩阵乘法
    5. 应用输出变换 (旋转恢复)
    """
    # 步骤 1: 输入变换 (使用预计算的块对角矩阵)
    if self._use_batched_rotation and self._R_in_all is not None:
        original_shape = x.shape
        x_t = x.reshape(-1, self.in_features)

        # 应用排列
        if self._perm_cache is not None:
            x_t = x_t.index_select(dim=-1, index=self._perm_cache)

        # 应用批量输入旋转 (单次大 matmul)
        x_t = x_t @ self._R_in_all  # 🚀 10-20x 加速
        x_t = x_t.reshape(*original_shape)
    else:
        # 回退：逐块旋转 (慢)
        from .duquant_preprocess import apply_input_transform_optimized
        x_t = apply_input_transform_optimized(x, self.pack, ...)

    # 步骤 2: 激活量化 (fake quantization)
    if self.cfg.act_bits > 0:
        s_a = self._get_act_scale(x_t)  # 获取 activation scale
        x_t = fake_quantize_sym(x_t, s_a, self.cfg.act_bits, label="activation_forward")

    # 步骤 3: 权重量化 + 矩阵乘法
    if self._weight_quantized_cached:
        # 使用预量化的权重 (第二次前向传播开始使用)
        y_lin = torch.nn.functional.linear(x_t, self._W_t_quantized, None)
    elif self.weight_bits > 0:
        # 第一次前向传播：在线量化
        y_lin = torch.nn.functional.linear(
            x_t,
            fake_quantize_sym(
                self._W_t,
                self._w_scales[:, None],
                self.weight_bits,
                label="weight_fallback",
            ),
            None
        )
    else:
        y_lin = torch.nn.functional.linear(x_t, self._W_t, None)

    # 步骤 4: 输出旋转恢复
    if self.cfg.row_rot_mode == "restore" and self.pack.R_out_blocks is not None:
        from .duquant_preprocess import apply_output_restore_optimized
        y_lin = apply_output_restore_optimized(
            y_lin, self.pack, self._get_R_out_cache(), self._block_out_size
        )
        # 恢复后再加 bias
        if self.bias is not None:
            y_lin = y_lin + self.bias
    else:
        # 传播模式或禁用：bias 在当前基下
        if self.bias is not None:
            y_lin = y_lin + self.bias

    return y_lin
```

**fake_quantize_sym() 函数** (duquant_preprocess.py:120-137):
```python
def fake_quantize_sym(x, scale, bits, *, label=None):
    """
    对称量化 (symmetric quantization)

    公式:
        x_quant = clamp(round(x / scale), -max_q-1, max_q) * scale

    其中 max_q = 2^(bits-1) - 1
    例如: 4-bit -> max_q = 7, 范围 [-8, 7]
          8-bit -> max_q = 127, 范围 [-128, 127]
    """
    if bits <= 0:
        return x

    max_q = 2**(bits-1) - 1  # qmax(bits)
    x_scaled = x / scale      # 归一化
    x_clamped = torch.clamp(torch.round(x_scaled), -max_q - 1, max_q)  # 量化 + 截断
    return x_clamped * scale  # 反量化 (fake quantization)
```

---

## 3. 关键数据流

### 3.1 图像数据流

```
原始图像 (256x256x3)
    ↓
旋转 180° (LIBERO 预处理)
    ↓
Resize with padding (224x224x3)
    ↓
Normalize (transforms.Normalize)
    ↓
SigLIP Vision Encoder
    ↓
Vision features [B, num_patches, hidden_dim]
```

### 3.2 文本数据流

```
Task description (string)
    ↓
Tokenize (Gemma tokenizer)
    ↓
Token IDs [B, max_token_len]
    ↓
Gemma LLM (language_model)
    ↓
Text embeddings [B, seq_len, hidden_dim]
```

### 3.3 状态数据流

```
Robot state (7D)
    ├─ end_effector_pos (3D)
    ├─ end_effector_rot_axisangle (3D)
    └─ gripper_position (1D)
    ↓
Normalize (transforms.Normalize)
    ↓
Concatenate with embeddings
    ↓
DiT Transformer
```

### 3.4 动作输出流

```
DiT output [B, action_horizon, action_dim]
    ↓
Unnormalize (transforms.Unnormalize)
    ↓
动作序列 [15, 7]
    ├─ end_effector_delta_pos (3D)
    ├─ end_effector_delta_rot (3D)
    └─ gripper_command (1D)
```

---

## 4. DuQuant 量化流程详解

### 4.1 离线 Packing 阶段 (第一次运行)

```
原始权重 W [out_features, in_features]
    ↓
【步骤 1】输入通道分块 + 计算旋转矩阵
    for each block:
        W_block = W[:, b*block_size:(b+1)*block_size]
        U, S, Vt = SVD(W_block.T @ W_block)
        R_in[b] = Vt[:block_size, :]
    ↓
【步骤 2】应用旋转
    W_rotated = W @ block_diag(R_in[0], R_in[1], ...)
    ↓
【步骤 3】计算 zigzag 排列
    energy = sum(W_rotated^2, axis=0)  # 每个输入通道的能量
    perm = zigzag_permute(energy, block_size, lambda_smooth)
    W_permuted = W_rotated[:, perm]
    ↓
【步骤 4】输出旋转 (可选)
    for each output block:
        compute R_out[b] using similar SVD
    ↓
【步骤 5】计算量化 scale
    weight_scale = max(abs(W_permuted), axis=1) / qmax(weight_bits)
    ↓
保存到 pack 文件 (.npz)
    - R_in_blocks
    - perm
    - R_out_blocks
    - weight_scale
    - meta
```

### 4.2 在线推理阶段 (后续运行)

```
输入激活 x [batch, in_features]
    ↓
【步骤 1】应用输入变换
    x_perm = x[:, perm]  # 排列
    x_rot = x_perm @ R_in_all  # 旋转 (使用预计算的块对角矩阵)
    ↓
【步骤 2】激活量化 (A8)
    scale_a = percentile(abs(x_rot), 99.9) / 127
    x_quant = fake_quantize_sym(x_rot, scale_a, 8)
    ↓
【步骤 3】权重量化 (W4) + 矩阵乘法
    W_quant = fake_quantize_sym(W_transformed, weight_scale, 4)
    y = x_quant @ W_quant.T
    ↓
【步骤 4】输出旋转恢复 (如果启用)
    y_restored = y @ R_out_all.T
    ↓
输出激活 [batch, out_features]
```

---

## 5. 性能优化技巧

### 5.1 批量旋转优化 (Batched Rotation)

**问题**: 原始实现需要对每个 block 单独做矩阵乘法
```python
# 慢: 128 次小 matmul
for b in range(128):
    x_rot_b = x[:, b*16:(b+1)*16] @ R_in[b]  # [B, 16] @ [16, 16]
```

**解决方案**: 预计算块对角矩阵
```python
# 快: 1 次大 matmul
R_in_all = torch.block_diag(R_in[0], R_in[1], ..., R_in[127])  # [2048, 2048]
x_rot = x @ R_in_all  # [B, 2048] @ [2048, 2048]
```

**加速比**: 10-20x (减少 GPU kernel 启动开销)

### 5.2 权重预量化缓存

**问题**: 每次前向传播都量化权重很慢
```python
# 慢: 每次都量化
y = F.linear(x, fake_quantize_sym(W, scale, 4))
```

**解决方案**: 第一次前向传播后缓存量化权重
```python
# 快: 只量化一次
if not self._weight_quantized_cached:
    self._W_t_quantized = fake_quantize_sym(self._W_t, self._w_scales, 4)
    self._weight_quantized_cached = True

y = F.linear(x, self._W_t_quantized)
```

### 5.3 激活量化 calibration

**问题**: 激活的 scale 需要根据数据分布确定

**解决方案**: 使用 calibration 阶段收集统计信息
```python
# 前 32 步收集激活统计
if self.calibrator is not None and not self.calibrator.is_full():
    self.calibrator.observe(x)
    if self.calibrator.is_full():
        scale = compute_scale_from_calibration()
```

### 5.4 CUDA 内存优化

**问题**: 块对角矩阵增加内存占用 (~2GB for 126 layers × 16MB)

**解决方案**: 启用 PyTorch expandable segments
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 6. 调试技巧

### 6.1 打印层信息

```bash
# 打印所有被量化的层
export OPENPI_DUQUANT_DEBUG=1

# Dry-run 模式：只打印不替换
export OPENPI_DUQUANT_DRYRUN=1
```

### 6.2 性能分析

```bash
# 启用 DuQuant profiling
export OPENPI_DUQUANT_PROFILE=1

# 启用 Policy 推理 profiling
export OPENPI_POLICY_PROFILE=1
```

输出示例:
```
[DUQUANT][PROFILE] fake quantization summary
Label                    Calls    Total ms    Avg ms    Elems       GB/s
activation_forward       1234     123.45      0.100     12345678    10.23
weight_quantize          126      45.67       0.362     9876543     8.91
```

### 6.3 打印 Linear 层形状

```bash
export OPENPI_PRINT_LINEAR_SHAPES=1
```

输出示例:
```
[LINEAR-MM] language_model.layers.0.self_attn.q_proj: x[1, 1024] @ W[2048, 1024]
[LINEAR-MM] gemma_expert.model.layers.0.mlp.gate_proj: x[1, 1024] @ W[4096, 1024]
```

---

## 7. 常见问题

### Q1: 为什么第一次运行很慢？
**A**: 第一次需要计算旋转矩阵并保存 pack 文件。后续运行会直接加载 pack 文件，速度快很多。

### Q2: Pack 文件保存在哪里？
**A**: `$OPENPI_DUQUANT_PACKDIR/<layer_name>.npz`

例如: `duquant_packed_llm_dit_mlp_w4a8/paligemma_with_expert.paligemma.model.language_model.layers.0.mlp.gate_proj.npz`

### Q3: 如何验证量化是否生效？
**A**:
1. 检查日志中的 `[DUQUANT][REPLACED]` 信息
2. 检查 pack 目录中的文件数量
3. 启用 `OPENPI_DUQUANT_DEBUG=1` 查看详细日志

### Q4: 量化后准确率下降怎么办？
**A**:
1. 增加 `OPENPI_DUQUANT_CALIB_STEPS` (更多 calibration 步数)
2. 调整 `OPENPI_DUQUANT_ACT_PCT` (更保守的激活量化)
3. 增大 `OPENPI_DUQUANT_BLOCK` (更大的 block size)
4. 调整 `OPENPI_DUQUANT_LS` (lambda smooth 参数)

### Q5: OOM 错误怎么解决？
**A**:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## 8. 总结

整个流程可以概括为：

1. **脚本启动** → 设置环境变量
2. **主程序** → 加载模型和配置
3. **模型加载** → 从 safetensors 恢复权重
4. **DuQuant 初始化** → 替换 Linear 为 DuQuantLinear
5. **Pack 加载/计算** → 加载或计算旋转矩阵
6. **LIBERO 循环** → 遍历任务和 episodes
7. **推理** → Policy.sample_actions()
8. **前向传播** → PI0Pytorch.forward()
9. **DuQuant 前向** → DuQuantLinear.forward()
10. **动作执行** → 环境执行动作并评估

关键优化：
- ✅ 预计算旋转矩阵 (packing)
- ✅ 批量旋转 (block diagonal)
- ✅ 权重预量化缓存
- ✅ 激活 calibration
- ✅ CUDA 内存优化

通过这些优化，DuQuant 可以在几乎不降低准确率的前提下，实现 2-4x 的内存压缩和接近原始速度的推理性能。
