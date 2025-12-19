#!/usr/bin/env python3
"""Test script to verify BitBLAS quantization without running full LIBERO."""

import os
import sys

# Setup paths
sys.path.insert(0, 'src')
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-12.8/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')

print("=" * 80)
print("BitBLAS W4FP16 Quantization Test")
print("=" * 80)
print()

# Set environment variables for BitBLAS
os.environ['OPENPI_BITBLAS_ENABLE'] = '1'
os.environ['OPENPI_BITBLAS_WBITS'] = '4'
os.environ['OPENPI_BITBLAS_GROUP_SIZE'] = '128'
os.environ['OPENPI_BITBLAS_ENABLE_TUNING'] = '0'
os.environ['OPENPI_BITBLAS_OPT_M'] = '1,16,32'
os.environ['OPENPI_BITBLAS_DUQUANT_PACKDIR'] = 'duquant_packed_full_llm_dit_mlp_w4a8_atm'
os.environ['OPENPI_BITBLAS_INCLUDE'] = r'(.*language_model.*(gate_proj|up_proj|down_proj).*|.*gemma_expert.*(gate_proj|up_proj|down_proj).*)'
os.environ['OPENPI_BITBLAS_EXCLUDE'] = r'(?:^|\.)(norm|ln|layernorm|emb|embed|vision_tower|multi_modal_projector|lm_head)(?:\.|$)'
os.environ['OPENPI_BITBLAS_DEBUG'] = '1'

# Disable DuQuant
for key in ['OPENPI_DUQUANT_WBITS_DEFAULT', 'OPENPI_DUQUANT_ABITS']:
    os.environ.pop(key, None)

import torch

print("1. Testing BitBLAS module import...")
try:
    from openpi.models_pytorch.bitblas_layers import (
        BitBLASQuantLinear,
        enable_bitblas_if_configured,
    )
    from openpi.models_pytorch.duquant_to_bitblas_converter import (
        DuQuantToBitBLASConverter,
    )
    print("   ✓ All modules imported successfully")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

print()
print("2. Creating test Linear layer...")
try:
    test_linear = torch.nn.Linear(256, 256, bias=True)
    test_linear.weight.data.normal_(0, 0.02)
    test_linear.bias.data.zero_()
    print(f"   ✓ Created Linear layer: {test_linear.weight.shape}")
    print(f"   Memory: {test_linear.weight.nelement() * test_linear.weight.element_size() / 1024:.2f} KB")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print()
print("3. Creating BitBLAS quantized layer...")
try:
    bitblas_layer = BitBLASQuantLinear(
        in_features=256,
        out_features=256,
        name='test_layer',
        bits=4,
        group_size=128,
        bias=True,
        enable_tuning=False,
    )
    print(f"   ✓ BitBLAS layer created")
    print(f"   qweight shape: {bitblas_layer.qweight.shape}")
    print(f"   scales shape: {bitblas_layer.scales.shape}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("4. Converting weights to INT4...")
try:
    # Create dummy DuQuant pack for testing
    import numpy as np
    from openpi.models_pytorch.duquant_preprocess import PackResult

    dummy_pack = PackResult(
        perm=None,
        R_in_blocks={},
        R_out_blocks={},
        weight_scale=np.ones(256, dtype=np.float32) * 0.01,  # Small scales for testing
        meta={'block_size': 16, 'block_out_size': 16, 'row_rot_mode': '0'},
    )

    bitblas_layer.load_from_linear(test_linear, duquant_pack=dummy_pack)

    print(f"   ✓ Weights converted to INT4")
    print(f"   qweight shape: {bitblas_layer.qweight.shape}")
    print(f"   qweight dtype: {bitblas_layer.qweight.dtype}")
    print(f"   qweight memory: {bitblas_layer.qweight.nelement() * bitblas_layer.qweight.element_size() / 1024:.2f} KB")
    print(f"   scales shape: {bitblas_layer.scales.shape}")

    # Check memory savings
    orig_mem = test_linear.weight.nelement() * test_linear.weight.element_size()
    quant_mem = bitblas_layer.qweight.nelement() * bitblas_layer.qweight.element_size()
    quant_mem += bitblas_layer.scales.nelement() * bitblas_layer.scales.element_size()

    print(f"   Memory: {orig_mem / 1024:.2f} KB -> {quant_mem / 1024:.2f} KB")
    print(f"   Savings: {(1 - quant_mem / orig_mem) * 100:.1f}%")

except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("5. Testing forward pass...")
try:
    test_input = torch.randn(2, 256).half().cuda()
    bitblas_layer = bitblas_layer.cuda()

    with torch.no_grad():
        output = bitblas_layer(test_input)

    print(f"   ✓ Forward pass successful")
    print(f"   Input shape: {test_input.shape}")
    print(f"   Output shape: {output.shape}")
    print(f"   Output range: [{output.min():.6f}, {output.max():.6f}]")

except Exception as e:
    print(f"   ✗ Failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("✅ All tests passed! BitBLAS INT4 quantization is working.")
print("=" * 80)
print()
print("You can now run the full evaluation:")
print("  bash examples/libero/run_quantvla_w4fp16.sh")
