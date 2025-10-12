# DuQuant W4A8 Fake Quantization Testing Guide

This guide helps you test DuQuant W4A8 fake quantization on PI0.5 LIBERO model with different configurations.

## 📋 Overview

All scripts implement **fake quantization** (simulated int4/int8 without real kernel acceleration):
- **W4**: 4-bit weight quantization (simulated)
- **A8**: 8-bit activation quantization (simulated)
- **Complete DuQuant**: Includes permutation + rotation + quantization + output restore

## 🎯 Available Test Scripts

### 1. DiT QKVO Only (`run_dit_qkvo_w4a8.sh`)
**Target**: DiT attention layers only (q_proj, k_proj, v_proj, o_proj)
**Expected layers**: 72-144 (depending on architecture)
**Memory saving**: ~15-20%

```bash
bash examples/libero/run_dit_qkvo_w4a8.sh
```

**Use case**: Test DuQuant impact on attention layers specifically

---

### 2. DiT All Layers (`run_optimized_duquant.sh`)
**Target**: DiT all layers (attention + MLP)
**Expected layers**: ~126
**Memory saving**: ~20-30%

```bash
bash examples/libero/run_optimized_duquant.sh
```

**Use case**: Standard DiT quantization (existing script)

---

### 3. LLM Only (`run_llm_w4a8.sh`)
**Target**: Gemma LLM all layers (attention + MLP)
**Expected layers**: ~126
**Memory saving**: ~20-30%

```bash
bash examples/libero/run_llm_w4a8.sh
```

**Use case**: Test LLM quantization without affecting DiT

---

### 4. Full LLM+DiT (`run_full_llm_dit_w4a8.sh`) ⚠️
**Target**: ALL linear layers in both LLM and DiT
**Expected layers**: ~252
**Memory saving**: ~40-50%

```bash
bash examples/libero/run_full_llm_dit_w4a8.sh
```

**Use case**: Maximum quantization for worst-case testing

⚠️ **WARNING**: Most aggressive quantization, may cause significant accuracy degradation!

---

## 🔍 Verification Script

Before running full evaluation, verify layer counts:

```bash
bash examples/libero/verify_duquant_layers.sh
```

This runs in dry-run mode and shows:
- Actual number of layers matched by each configuration
- Layer names that will be quantized
- No actual quantization performed

---

## 📊 Expected Layer Counts

| Configuration | Target | Expected Layers | Notes |
|---------------|--------|-----------------|-------|
| DiT QKVO | DiT attention only | 72-144 | Depends on dual attention |
| DiT All | DiT all | ~126 | 18 layers × 7 linears |
| LLM Only | Gemma LLM all | ~126 | 18 layers × 7 linears |
| Full LLM+DiT | Both LLM+DiT | ~252 | Maximum compression |

**Layer breakdown per transformer block**:
- Attention: 4 linears (q_proj, k_proj, v_proj, o_proj)
- MLP: 3 linears (gate_proj, up_proj, down_proj)
- Total per block: 7 linears

---

## 🧪 Testing Procedure

### Step 1: Verify Layer Counts
```bash
bash examples/libero/verify_duquant_layers.sh
```

Check output for `[DUQUANT] Dry-run total layers listed: XXX`

### Step 2: Run Baseline (No Quantization)
```bash
# Disable all DuQuant
unset OPENPI_DUQUANT_SCOPE
bash examples/libero/run_optimized_duquant.sh
```

Record baseline accuracy for comparison.

### Step 3: Test Individual Configurations

```bash
# Test 1: DiT QKVO only
bash examples/libero/run_dit_qkvo_w4a8.sh

# Test 2: DiT all layers
bash examples/libero/run_optimized_duquant.sh

# Test 3: LLM only
bash examples/libero/run_llm_w4a8.sh

# Test 4: Full LLM+DiT
bash examples/libero/run_full_llm_dit_w4a8.sh
```

### Step 4: Compare Results

Check `results/libero/` for accuracy metrics:
- Success rate per task
- Average episode length
- Memory usage (if logged)

---

## 📈 Performance Profiling

All scripts enable profiling by default:
```bash
export OPENPI_DUQUANT_PROFILE=1
```

After evaluation, check stdout for:
```
[DUQUANT][PROFILE] fake quantization summary
Label                          Calls  Total ms    Avg ms       Elems      GB/s
weight_prequant                 252    1234.56     4.900    12345678    45.67
activation_forward             5040   2345.67     0.465     6789012    23.45
```

This shows overhead of fake quantization operations.

---

## 🔧 Configuration Variables

All scripts support these environment variables:

### DuQuant Config
```bash
OPENPI_DUQUANT_WBITS_DEFAULT=4    # Weight bits (2/4/8/16)
OPENPI_DUQUANT_ABITS=8            # Activation bits (4/8/16)
OPENPI_DUQUANT_BLOCK=16           # Block size for quantization
OPENPI_DUQUANT_PERMUTE=1          # Enable permutation (0/1)
OPENPI_DUQUANT_ROW_ROT=restore    # Rotation mode (0/restore/propagate)
OPENPI_DUQUANT_LS=0.15            # Lambda smoothing
OPENPI_DUQUANT_ACT_PCT=99.9       # Activation percentile
OPENPI_DUQUANT_CALIB_STEPS=32     # Calibration steps
```

### Evaluation Config
```bash
TASK_SUITE=libero_spatial         # Task suite to test
NUM_TRIALS=20                     # Trials per task
SEED=42                           # Random seed
```

### Optimization Config
```bash
OPENPI_DISABLE_TORCH_COMPILE=1    # Disable torch.compile (faster startup)
TORCH_COMPILE_DISABLE=1
TORCHDYNAMO_DISABLE=1
```

---

## 📝 Expected Output

Each script will print:
1. **Configuration summary** with all DuQuant settings
2. **Target layers** and expected count
3. **Actual replaced layers**: `[DUQUANT] Total layers replaced: XXX`
4. **Evaluation progress** with timing per episode
5. **Profiling summary** (if enabled)
6. **Final results** in `results/libero/`

Example:
```
========================================
LIBERO Headless Evaluation
DuQuant: DiT QKVO Only (W4A8)
========================================
...
[DUQUANT] SCOPE filter: 'paligemma_with_expert.gemma_expert.model.'
[DUQUANT] Matched Linear layers: 72
[DUQUANT][REPLACED] ...q_proj: Linear(2048->2048) -> DuQuantLinear W4 A8
...
[DUQUANT] Total layers replaced: 72
...
Evaluation complete!
```

---

## ⚠️ Important Notes

1. **Fake Quantization**: These scripts implement fake quantization only
   - Simulates quantization error without speed improvement
   - Used to test accuracy impact before real int4/int8 kernels
   - Memory savings are from packing, not from int4/int8 storage

2. **First Run is Slow**:
   - Computes and caches rotation matrices
   - Saved to `OPENPI_DUQUANT_PACKDIR`
   - Subsequent runs reuse cached data

3. **Memory Usage**:
   - Fake quant uses same memory as FP16 during forward pass
   - Packed weights reduce checkpoint size
   - Real memory savings require int4/int8 kernels

4. **Torch Compile**:
   - Disabled by default for faster startup
   - Enable for better throughput: `unset OPENPI_DISABLE_TORCH_COMPILE`
   - First episode: ~15-20 min (compilation)
   - Later episodes: ~30-60s (cached)

---

## 🐛 Troubleshooting

### "No layers matched"
Check scope prefix matches your model:
```bash
# Debug: List all linear layers
OPENPI_DUQUANT_DRYRUN=1 OPENPI_DUQUANT_SCOPE="" bash verify_duquant_layers.sh
```

### "Pack directory permission denied"
Change pack directory:
```bash
export OPENPI_DUQUANT_PACKDIR="$HOME/duquant_cache"
```

### "CUDA out of memory"
Reduce batch size or disable torch.compile:
```bash
export OPENPI_DISABLE_TORCH_COMPILE=1
```

---

## 📚 Next Steps

After testing fake quantization:

1. **Analyze accuracy degradation** by comparing with baseline
2. **Identify acceptable quantization strategies** (e.g., DiT-only vs Full)
3. **Implement real int4/int8 kernels** if accuracy is acceptable
4. **Measure actual speedup** with real kernels

For questions, check:
- DuQuant paper: [arxiv.org/abs/2410.09837](https://arxiv.org/abs/2410.09837)
- Source code: `src/openpi/models_pytorch/duquant_*.py`
