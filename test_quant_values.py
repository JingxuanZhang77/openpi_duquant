"""Test what values are being quantized."""

import torch
import numpy as np

# Simulate quantization
W_original = torch.tensor([[1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 0.0, 1.5]], dtype=torch.float32)
scale = torch.tensor([[0.25]], dtype=torch.float32)

print(f"Original W: {W_original}")
print(f"Scale: {scale}")

# Quantize to signed INT4
signed_q = torch.clamp(torch.round(W_original / scale), -8, 7).to(torch.int8)
print(f"Signed quantized: {signed_q}")

# Convert to unsigned
unsigned_q = signed_q + 8
print(f"Unsigned quantized: {unsigned_q}")

# Dequantize from unsigned
W_dequant = (unsigned_q.to(torch.int16) - 8).half() * scale.half()
print(f"Dequantized: {W_dequant}")

# Compare
error = (W_dequant - W_original.half()).abs()
print(f"Error: {error}")
print(f"Max error: {error.max()}")
