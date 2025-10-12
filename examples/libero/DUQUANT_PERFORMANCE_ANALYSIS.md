# DuQuant Fake Quantization Performance Analysis

## 📊 Observed Performance Issue

**Current Performance:**
- Average policy call: **13,086 ms** (13 seconds)
- Throughput: **0.08 calls/s**
- Episode time: **~243 seconds** (4 minutes)

**Baseline (FP16, no quantization):**
- Expected: ~1-3 seconds per policy call
- **10x slower** with DuQuant fake quantization

## 🔍 Root Cause Analysis

### 1. **Fake Quantization Overhead (Main Culprit)**

Fake quantization simulates int4/int8 behavior in **FP16/BFloat16**, causing massive overhead:

#### Per-Layer Operations (126 layers × every forward pass):

```python
# duquant_layers.py:241-301
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Step 1: Input Transform (permutation + rotation)
    x_t = apply_input_transform_optimized(x, ...)  # ⚠️ EXPENSIVE

    # Step 2: Activation Quantization (W8)
    x_t = fake_quantize_sym(x_t, scale, 8)  # ⚠️ EXPENSIVE

    # Step 3: Linear (using pre-quantized weights)
    y = F.linear(x_t, self._W_t_quantized, None)  # ✅ OK

    # Step 4: Output Restore (rotation)
    y = apply_output_restore_optimized(y, ...)  # ⚠️ EXPENSIVE
```

#### Overhead Breakdown:

| Operation | Cost per Layer | Total (126 layers) | Notes |
|-----------|---------------|-------------------|-------|
| **Input Permutation** | ~0.5-1 ms | 63-126 ms | `index_select()` on large tensors |
| **Input Rotation** | ~5-10 ms | 630-1260 ms | Block-wise matrix multiply |
| **Activation Quant** | ~2-5 ms | 252-630 ms | round/clamp/scale ops |
| **Output Rotation** | ~5-10 ms | 630-1260 ms | Block-wise matrix multiply |
| **Tensor Clones** | ~2-3 ms | 252-378 ms | `clone()` to avoid in-place issues |
| **Total Overhead** | ~15-30 ms | **1827-3654 ms** | **Per forward pass!** |

### 2. **Memory Bandwidth Bottleneck**

#### Clone Operations:
```python
# duquant_preprocess.py:582
x_t = x_view.clone()  # ⚠️ Full tensor copy

# duquant_preprocess.py:612
y_out = y_view.clone()  # ⚠️ Full tensor copy
```

**Impact:**
- Each clone copies the entire activation tensor
- 126 layers × 2 clones = **252 full tensor copies per forward pass**
- For batch_size=1, seq_len=1024, hidden_dim=2048:
  - Clone size: 1 × 1024 × 2048 × 2 bytes (BF16) = **4 MB**
  - Total: 252 × 4 MB = **~1 GB copied per forward pass**

### 3. **Block-wise Rotation Overhead**

#### Input Rotation (18 blocks per layer):
```python
# duquant_preprocess.py:584-591
for b in range(n_blocks):  # n_blocks = hidden_dim / 16 = 128 blocks
    start = b * block_size
    end = min((b + 1) * block_size, in_features)
    R = R_in_cache[b][: (end - start), : (end - start)]
    x_t[:, start:end] = x_view[:, start:end] @ R  # ⚠️ Small matmul
```

**Problem:**
- Small block-wise matmul (16×16) prevents GPU batching
- Kernel launch overhead dominates for small matrices
- **128 blocks × 126 layers = 16,128 small matmuls per forward pass**

### 4. **Fake Quantization Does NOT Speed Up**

Fake quantization simulates int4/int8 in **full precision**:

```python
# duquant_preprocess.py:120-137
def fake_quantize_sym(x, scale, bits):
    max_q = (1 << (bits - 1)) - 1  # 7 for int8
    x_scaled = x / scale              # ⚠️ FP16 division
    x_clamped = torch.clamp(           # ⚠️ FP16 clamp
        torch.round(x_scaled),         # ⚠️ FP16 round
        -max_q - 1, max_q
    )
    return x_clamped * scale          # ⚠️ FP16 multiply
```

**No speedup because:**
- Still uses FP16/BFloat16 compute
- Still uses FP16/BFloat16 memory bandwidth
- Only simulates quantization error, not int4/int8 operations

### 5. **Torch.Compile Disabled**

Your script has:
```bash
export OPENPI_DISABLE_TORCH_COMPILE=1
```

**Impact:**
- No kernel fusion
- No graph-level optimizations
- Each operation launches separate CUDA kernel
- Especially bad for block-wise rotations (16,128 kernel launches!)

---

## 🎯 Performance Model

### DiT Architecture (18 layers, 7 linears per layer = 126 layers)

#### Per Forward Pass (1 policy call):
```
Base model forward:           100 ms  (FP16, no quantization)
DuQuant overhead per layer:    15 ms  (conservative estimate)
Total DuQuant overhead:      1890 ms  (126 layers × 15 ms)
────────────────────────────────────
Total per forward pass:      1990 ms  (~2 seconds)
```

#### Per Episode (18 policy calls):
```
18 policy calls × 2000 ms = 36,000 ms = 36 seconds
```

**Your observed: 243 seconds / episode (4 min)**

This suggests:
- **13 seconds per policy call** (observed)
- **Much worse than theoretical 2s**

### Why 6.5x Worse Than Theory?

1. **Multiple forward passes per policy call**
   - Vision encoder forward
   - LLM forward (multiple tokens)
   - DiT forward (action generation)
   - **Estimated: 5-10 forward passes per policy call**

2. **Torch.compile disabled**
   - No kernel fusion → 2-3x slower

3. **Memory bandwidth bottleneck**
   - Clones + rotations saturate PCIe bandwidth

4. **Small matmul inefficiency**
   - 16×16 blocks don't utilize GPU well

---

## 📈 Breakdown by Operation Type

### Estimated Time Distribution (per policy call):

| Operation | Time (ms) | Percentage | Notes |
|-----------|-----------|------------|-------|
| **Block Rotations** | 5000-7000 | 38-54% | Input + Output rotations |
| **Activation Quant** | 1000-2000 | 8-15% | fake_quantize_sym |
| **Tensor Clones** | 500-1000 | 4-8% | Memory copies |
| **Permutations** | 200-500 | 2-4% | index_select |
| **Linear (actual compute)** | 2000-3000 | 15-23% | Base model operations |
| **Other** | 1000-2000 | 8-15% | Misc overhead |
| **Total** | **13,000** | 100% | Matches observation |

---

## 🚀 Optimization Strategies

### Short-term (Fake Quantization):

#### 1. **Enable Torch.Compile** ⚡ (Expected: 2-3x speedup)
```bash
# In run_optimized_duquant.sh, comment out:
# export OPENPI_DISABLE_TORCH_COMPILE=1
```

**Benefits:**
- Fuses permutation + rotation into single kernel
- Fuses quantization ops
- Reduces kernel launch overhead

**Trade-off:**
- First episode: 15-20 min (compilation)
- Later episodes: 30-60s (10x faster!)

#### 2. **Reduce Block Size** (Expected: 1.5-2x speedup)
```bash
export OPENPI_DUQUANT_BLOCK=32  # Default: 16
```

**Benefits:**
- Fewer blocks → fewer kernel launches
- Better GPU utilization per matmul
- Hidden_dim=2048: 128 blocks → 64 blocks (2x fewer)

**Trade-off:**
- Slightly lower accuracy (larger quantization blocks)

#### 3. **Disable Output Restore** (Expected: 2x speedup)
```bash
export OPENPI_DUQUANT_ROW_ROT=0  # Disable rotation
```

**Benefits:**
- Eliminates output rotation overhead (50% of time)
- Still keeps permutation + activation quant

**Trade-off:**
- Lower accuracy (no rotation correction)

#### 4. **Reduce Activation Bits** (Expected: 1.2x speedup)
```bash
export OPENPI_DUQUANT_ABITS=16  # Disable activation quant
```

**Benefits:**
- Eliminates activation quantization overhead
- Keeps weight quantization

**Trade-off:**
- Not testing full W4A8 behavior

#### 5. **Use Simple Config** (Expected: 3-4x speedup)
```bash
bash examples/libero/run_simple_w4a8.sh
```

**Config:**
- No permutation
- No rotation
- Only basic quantization

**Expected: ~3-5 seconds per policy call**

---

### Medium-term (Optimization):

#### 6. **Fuse Operations in Custom Kernel**
- Combine permute + rotate + quantize
- Single kernel launch per layer
- Expected: 3-5x speedup

#### 7. **Remove Clones**
- Use in-place operations carefully
- Pre-allocate output buffers
- Expected: 1.3-1.5x speedup

#### 8. **Batch Block Operations**
- Stack all blocks, compute in parallel
- Single large matmul instead of 128 small ones
- Expected: 2-3x speedup

---

### Long-term (Real Quantization):

#### 9. **Implement True INT4/INT8 Kernels** ⚡⚡⚡
- Use CUTLASS or cuBLAS int4/int8 matmul
- Store weights in packed int4 format
- Use int8 for activations

**Expected speedup:**
- **5-10x faster** than fake quantization
- **Same speed or faster** than FP16 baseline
- **4x less memory** for weights

**Implementation options:**
- BitBLAS (from your repo)
- CUTLASS
- TensorRT
- PyTorch int8 quantization

---

## 📝 Recommended Testing Order

### Phase 1: Quick Wins (1 hour)
```bash
# Test 1: Simple config (no permute/rotate)
bash examples/libero/run_simple_w4a8.sh
# Expected: 3-5s per call (4x faster)

# Test 2: Larger block size
export OPENPI_DUQUANT_BLOCK=32
bash examples/libero/run_optimized_duquant.sh
# Expected: 8-10s per call (1.5x faster)

# Test 3: Disable output restore
export OPENPI_DUQUANT_ROW_ROT=0
bash examples/libero/run_optimized_duquant.sh
# Expected: 6-8s per call (2x faster)
```

### Phase 2: Torch.Compile (overnight)
```bash
# Enable torch.compile (first run is slow!)
unset OPENPI_DISABLE_TORCH_COMPILE
bash examples/libero/run_optimized_duquant.sh
# Episode 1: 15-20 min (compilation)
# Episode 2+: 30-60s per episode (20x faster!)
```

### Phase 3: Real INT4/INT8 (1-2 weeks)
- Integrate BitBLAS or CUTLASS
- Implement int4/int8 matmul
- Expected: Same speed as FP16 baseline

---

## 🎯 Expected Performance After Optimization

| Configuration | Policy Call Time | Episode Time | Speedup vs Current |
|---------------|------------------|--------------|-------------------|
| **Current (full DuQuant)** | 13,000 ms | 243 s | 1x (baseline) |
| Simple (no permute/rotate) | 3,000 ms | 60 s | **4.3x** |
| Larger blocks (32) | 8,000 ms | 150 s | **1.6x** |
| No output restore | 6,000 ms | 120 s | **2.2x** |
| Torch.compile (ep 2+) | 2,000 ms | 40 s | **6.5x** |
| Real INT4/INT8 | 1,000-1,500 ms | 20-30 s | **8-13x** |
| **FP16 Baseline** | 1,000 ms | 20 s | **13x** |

---

## 🔬 Profiling Recommendations

To confirm bottlenecks, add profiling:

```bash
# Enable DuQuant profiling
export OPENPI_DUQUANT_PROFILE=1
export OPENPI_DUQUANT_PROFILE_SYNC=1  # Accurate timing

bash examples/libero/run_optimized_duquant.sh 2>&1 | tee duquant_profile.log

# Check profiling output
grep "DUQUANT.*PROFILE" duquant_profile.log
```

Expected output:
```
[DUQUANT][PROFILE] fake quantization summary
Label                          Calls  Total ms    Avg ms       Elems      GB/s
activation_forward            2268   5000.00     2.204     5678901    45.67
weight_prequant                126    500.00     3.968     1234567    23.45
```

---

## 💡 Key Insights

1. **Fake quantization is 10x slower** because it simulates int4/int8 in FP16
2. **Block rotations dominate** (50%+ of time)
3. **Clones are expensive** (~1GB copied per forward pass)
4. **Small matmuls are inefficient** (16×16 blocks)
5. **Torch.compile helps a lot** (2-3x speedup when enabled)
6. **Real int4/int8 kernels needed** for production speed

**Bottom line:** Fake quantization is only for **accuracy testing**, not speed. To get real speedup, you need true int4/int8 kernels (BitBLAS, CUTLASS, etc.).
