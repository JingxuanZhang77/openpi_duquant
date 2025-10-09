# BitBLAS Integration Status

## Current Status: ✅ WORKING (PyTorch Fallback)

BitBLAS has been successfully compiled and installed, but we are currently using an **optimized PyTorch fallback implementation** for W4A8 quantization instead of native BitBLAS kernels.

## Benchmark Results (2-layer DiT model, 128 seq, batch 1)

| Backend | Time per call | Speedup vs FP16 | Error vs FP16 |
|---------|---------------|-----------------|---------------|
| **FP16 Baseline** | 0.67 ms | 1.0x (baseline) | - |
| **Fake Quant (DuQuant)** | 120.94 ms | **0.0055x (180x slower)** ❌ | MAE=4.74e-2 |
| **BitBLAS (PyTorch fallback)** | 137.08 ms | **0.0049x (204x slower)** ❌ | MAE=4.74e-2 |

### Key Observations

1. **Quantization is working correctly** - numerical accuracy is good (MAE < 5%)
2. **Performance is slow** - PyTorch fallback is 200x slower than FP16
3. **Native BitBLAS kernels needed** - to achieve expected 3-5x speedup

## Why is the Current Implementation Slow?

The PyTorch fallback implementation in `bitblas_fallback.py` does:

```python
# 1. Unpack INT4 weights to INT8 (memory bound)
q_w = _unpack_int4(packed_w, out_aligned, in_aligned).to(torch.float32)

# 2. Quantize activations to INT8
x_q = torch.clamp(torch.round(flat_x * inv_scale), -128, 127)
dequant_x = x_q * s_a

# 3. Full precision matmul (FP32, not using Tensor Cores)
acc = torch.matmul(dequant_x, q_w.t())

# 4. Scale output
acc = acc * s_w
```

**Problems:**
- No Tensor Core usage (INT4 or INT8 Tensor Cores)
- FP32 matmul instead of INT8 accumulation
- Python overhead from unpacking and quantization ops
- No kernel fusion

## What Would True BitBLAS Integration Provide?

Native BitBLAS W4A8 kernels would use:

1. **INT4 Tensor Cores** - 4x faster than FP16 Tensor Cores
2. **INT8 activations with INT4 weights** - optimal for Ampere/Ada GPUs
3. **Kernel fusion** - single CUDA kernel for unpack + matmul + scale
4. **Optimized memory layout** - better cache utilization

**Expected speedup: 3-5x faster than FP16** (vs current 200x slower)

## Requirements for Native BitBLAS Integration

To use native BitBLAS INT4 kernels, we need to:

### 1. Weight Format Conversion

**Current DuQuant packing:**
```python
# Pack two INT4 values into one uint8 byte
packed = low | (high << 4)  # [out, in//2] uint8
```

**BitBLAS expected format:**
- Use `bitblas.Linear` with `W_dtype='int4'`
- BitBLAS expects weights in its own packed format
- Need to convert DuQuant packed weights → BitBLAS format

### 2. Scaling Factor Integration

**Current approach:**
```python
# Separate weight and activation scales
y = (x / s_a) @ (W / s_w).T
```

**BitBLAS approach:**
- BitBLAS `Linear` has `with_scaling=True` and `group_size` parameters
- Need to integrate DuQuant's per-channel scales with BitBLAS scaling

### 3. Code Changes Required

**File: `src/openpi/models_pytorch/quant_backends/bitblas_utils.py`**

```python
def ensure_bitblas_linear_kernel():
    """Need to implement proper BitBLAS kernel initialization."""
    from bitblas import Linear, MatmulConfig

    @functools.lru_cache(maxsize=128)
    def get_bitblas_linear(in_features, out_features, has_bias):
        """Create BitBLAS Linear module for given shape."""
        return Linear(
            in_features=in_features,
            out_features=out_features,
            bias=has_bias,
            A_dtype='float16',      # Input activations
            W_dtype='int4',          # 4-bit weights
            accum_dtype='int32',     # INT32 accumulation
            out_dtype='float16',     # Output dtype
            group_size=-1,           # Per-channel quantization
            with_scaling=True,       # Enable per-channel scaling
            enable_tuning=False,     # Disable auto-tuning for faster init
        )

    def linear_w4a8(x, packed_w, s_w, s_a, *, bias=None):
        # TODO: Convert DuQuant packed_w to BitBLAS format
        # TODO: Set BitBLAS scaling factors from s_w and s_a
        # TODO: Call BitBLAS Linear.forward()
        pass
```

### 4. Testing Strategy

1. **Unit test**: Convert small weight tensor, verify BitBLAS can load it
2. **Numerical test**: Compare BitBLAS output vs PyTorch fallback
3. **Performance test**: Verify 3-5x speedup vs FP16 baseline
4. **Integration test**: Run full LIBERO evaluation with BitBLAS

## Current Workaround: Use Fake Quantization

For now, if you want to test DuQuant W4A8 quantization without the performance penalty, you can:

1. **Use fake quantization** to test numerical accuracy:
   ```bash
   export OPENPI_DUQUANT_BACKEND=fake
   ./examples/libero/run_llm_w4a8.sh
   ```

2. **Accept 180x slowdown** for now with BitBLAS backend:
   ```bash
   export OPENPI_DUQUANT_BACKEND=bitblas
   ./examples/libero/run_llm_w4a8.sh
   ```

## Next Steps

To complete native BitBLAS integration:

1. ✅ **Compile BitBLAS** - DONE (TVM + TileLang compiled)
2. ✅ **Test BitBLAS import** - DONE (imports successfully)
3. ❌ **Weight format conversion** - TODO
4. ❌ **Scaling integration** - TODO
5. ❌ **Kernel function implementation** - TODO
6. ❌ **Performance testing** - TODO

## Alternative: Wait for BitBLAS to Support DuQuant Format

Another option is to contribute to BitBLAS to add native support for DuQuant's packing format and rotation matrices. This would benefit the entire community.

## Files Modified

- `src/openpi/models_pytorch/quant_backends/bitblas_utils.py` - BitBLAS kernel utilities
- `src/openpi/models_pytorch/quant_backends/bitblas_fallback.py` - PyTorch fallback implementation
- `src/openpi/models_pytorch/quant_backends/bitblas_linear.py` - QLinearW4A8BitBLAS module
- `third_party/BitBLAS/` - BitBLAS library (compiled from source)

## Conclusion

✅ **DuQuant W4A8 quantization is working correctly with good numerical accuracy**
❌ **Performance is 200x slower than FP16 due to PyTorch fallback**
🔧 **Native BitBLAS kernel integration needed for 3-5x speedup**

For production use, recommend:
- Use **FP16 baseline** for best performance
- Use **fake quantization** to test accuracy without slowdown
- Wait for native BitBLAS integration before deploying W4A8

---

**Last updated:** 2025-10-09
**Status:** Compilation complete, integration incomplete
