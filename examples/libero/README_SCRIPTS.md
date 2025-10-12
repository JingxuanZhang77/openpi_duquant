# LIBERO DuQuant Testing Scripts Overview

## 📁 Script Summary

| Script | Target | Expected Layers | Use Case |
|--------|--------|-----------------|----------|
| [`run_simple_w4a8.sh`](run_simple_w4a8.sh) | DiT (no permute/rotate) | ~126 | Fast baseline |
| [`run_optimized_duquant.sh`](run_optimized_duquant.sh) | DiT (full DuQuant) | ~126 | Standard DiT |
| [`run_llm_w4a8.sh`](run_llm_w4a8.sh) | LLM (full DuQuant) | ~126 | LLM only |
| [`run_dit_qkvo_w4a8.sh`](run_dit_qkvo_w4a8.sh) ⭐ | DiT QKVO only | 72-144 | Attention only |
| [`run_full_llm_dit_w4a8.sh`](run_full_llm_dit_w4a8.sh) ⭐ | LLM+DiT full | ~252 | Maximum quant |
| [`verify_duquant_layers.sh`](verify_duquant_layers.sh) 🔍 | Dry-run all configs | N/A | Layer verification |

⭐ = New scripts created
🔍 = Verification utility

## 🚀 Quick Start

### 1. Verify Your Setup
```bash
# Check how many layers will be quantized
bash examples/libero/verify_duquant_layers.sh
```

### 2. Test Individual Configurations
```bash
# Smallest: DiT attention only
bash examples/libero/run_dit_qkvo_w4a8.sh

# Medium: DiT all layers
bash examples/libero/run_optimized_duquant.sh

# Medium: LLM all layers
bash examples/libero/run_llm_w4a8.sh

# Maximum: LLM+DiT all layers
bash examples/libero/run_full_llm_dit_w4a8.sh
```

## 📊 Configuration Comparison

### Feature Matrix

|  | Simple | Optimized | LLM | QKVO⭐ | Full⭐ |
|---|--------|-----------|-----|--------|--------|
| **Target** | DiT | DiT | LLM | DiT QKVO | LLM+DiT |
| **Layers** | ~126 | ~126 | ~126 | 72-144 | ~252 |
| **Permutation** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Rotation** | ❌ | ✅ | ✅ | ✅ | ✅ |
| **Lambda Smooth** | N/A | 0.15 | 0.15 | 0.15 | 0.15 |
| **Memory Save** | ~20% | ~20% | ~20% | ~15% | ~40% |
| **Speed** | Fast | Medium | Medium | Medium | Slow |
| **Accuracy** | Lower | Higher | Higher | Higher | Lowest |

### Scope Filters

| Script | OPENPI_DUQUANT_SCOPE | Layer Pattern |
|--------|----------------------|---------------|
| Simple/Optimized | `gemma_expert.model.` | DiT all |
| LLM | `language_model.` | LLM all |
| QKVO⭐ | `gemma_expert.model.` | DiT q/k/v/o only |
| Full⭐ | `paligemma_with_expert.` | LLM+DiT all |

## 🎯 Recommended Testing Order

1. **Baseline (No Quantization)**
   ```bash
   unset OPENPI_DUQUANT_SCOPE
   # Run any script - it will skip quantization
   ```

2. **Start Small: QKVO Only**
   ```bash
   bash examples/libero/run_dit_qkvo_w4a8.sh
   ```
   - Tests attention layers specifically
   - Lower risk of accuracy loss

3. **Medium: Single Component**
   ```bash
   # Option A: DiT only
   bash examples/libero/run_optimized_duquant.sh

   # Option B: LLM only
   bash examples/libero/run_llm_w4a8.sh
   ```
   - Isolate quantization impact
   - Compare which component is more sensitive

4. **Aggressive: Full Quantization**
   ```bash
   bash examples/libero/run_full_llm_dit_w4a8.sh
   ```
   - Maximum compression
   - Worst-case accuracy test

## 📈 Expected Performance

### Speed (without torch.compile)
- Simple: ~1-2 min/episode
- Optimized/LLM/QKVO: ~2-3 min/episode
- Full: ~3-5 min/episode

### Speed (with torch.compile)
First episode: 15-30 min (compilation)
Later episodes: 0.5-2 min (cached)

To enable torch.compile:
```bash
# In any script, comment out these lines:
# export OPENPI_DISABLE_TORCH_COMPILE=1
# export TORCH_COMPILE_DISABLE=1
# export TORCHDYNAMO_DISABLE=1
```

## 🔍 Debugging

### Check Layer Counts
```bash
# Dry-run shows matched layers without quantizing
bash examples/libero/verify_duquant_layers.sh
```

### View Layer Names
```bash
# Add DEBUG env var to any script
export OPENPI_DUQUANT_DEBUG=1
bash examples/libero/run_dit_qkvo_w4a8.sh 2>&1 | grep DUQUANT
```

### Test Equivalence (No Quantization)
```bash
# Disable quantization but keep transforms
export OPENPI_DUQUANT_WBITS_DEFAULT=32
export OPENPI_DUQUANT_ABITS=32
bash examples/libero/run_optimized_duquant.sh
```
Should match FP16 baseline exactly (within numerical precision).

## 📝 Log Interpretation

### Successful Quantization
```
[DUQUANT] SCOPE filter: 'paligemma_with_expert.gemma_expert.model.'
[DUQUANT] Matched Linear layers: 72
[DUQUANT][REPLACED] paligemma_with_expert.gemma_expert.model.layers.0.self_attn.q_proj: Linear(2048->2048) -> DuQuantLinear W4 A8 perm=True block_in=16 block_out=16 row_rot=restore
...
[DUQUANT] Total layers replaced: 72
```

### Profile Output
```
[DUQUANT][PROFILE] fake quantization summary (cuda_sync=on)
Label                          Calls  Total ms    Avg ms       Elems      GB/s
weight_prequant                   72    123.45     1.715    1234567    45.67
activation_forward              1440    234.56     0.163     567890    23.45
```

## 🐛 Common Issues

### Issue: "No layers matched"
**Solution**: Check scope with dry-run
```bash
bash examples/libero/verify_duquant_layers.sh
```

### Issue: "CUDA out of memory"
**Solution**: Disable torch.compile or reduce batch size
```bash
export OPENPI_DISABLE_TORCH_COMPILE=1
```

### Issue: "Permission denied" on pack directory
**Solution**: Change pack directory
```bash
export OPENPI_DUQUANT_PACKDIR="$HOME/duquant_cache"
```

### Issue: Scripts run but no quantization happens
**Solution**: Check env vars are exported
```bash
echo $OPENPI_DUQUANT_SCOPE
echo $OPENPI_DUQUANT_WBITS_DEFAULT
```

## 📚 Related Files

- **Implementation**: [`src/openpi/models_pytorch/duquant_*.py`](../../src/openpi/models_pytorch/)
- **Testing Guide**: [`DUQUANT_TESTING.md`](DUQUANT_TESTING.md)
- **Equivalence Test**: [`scripts/duquant_equiv_test.py`](../../scripts/duquant_equiv_test.py)

## 🔗 References

- DuQuant Paper: https://arxiv.org/abs/2410.09837
- OpenPI Documentation: [Main README](../../README.md)
