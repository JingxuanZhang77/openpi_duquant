#!/usr/bin/env python3
import torch

from openpi.models_pytorch.duquant_preprocess import pack_weight, transform_weight_for_forward, apply_input_transform, apply_output_restore


def main():
    torch.manual_seed(0)
    B, C, O = 4, 96, 128
    x = torch.randn(B, C, dtype=torch.float64)
    W = torch.randn(O, C, dtype=torch.float64)

    pack = pack_weight(W, block_size=16, block_out_size=16, enable_permute=True)

    # No quantization: bits>=16 disables quantization inside transform_weight_for_forward
    W_hat, _ = transform_weight_for_forward(W, pack, weight_bits=32, apply_row_rot=True)

    y0 = x @ W.t()
    x1 = apply_input_transform(x, pack, use_transpose=False)  # x P R_in
    y1 = x1 @ W_hat.t()  # = x W^T R_out^T
    y2 = apply_output_restore(y1, pack)  # right-multiply R_out

    err = (y0 - y2).abs().max().item()
    print(f"max|y0 - y2| = {err:.3e}")
    assert err < 1e-6, "Equivalence failed"


if __name__ == "__main__":
    main()
