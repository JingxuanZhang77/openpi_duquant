"""Test INT4 packing/unpacking correctness."""

import numpy as np
import torch
import sys

sys.path.insert(0, "/home/jz97/VLM_REPO/openpi/src")

from openpi.models_pytorch.duquant_to_bitblas_converter import unpack_int4_from_int8

def test_packing():
    """Test that pack/unpack is lossless."""

    # Create test UNSIGNED INT4 values (range: 0 to 15)
    # BitBLAS expects unsigned INT4 after the signed->unsigned conversion
    test_values_signed = np.array([
        [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8],
    ], dtype=np.int8)

    # Convert to unsigned [0, 15]
    test_values = test_values_signed + 8

    print(f"Original signed INT4 values:")
    print(test_values_signed)
    print(f"\nUnsigned INT4 values (what BitBLAS expects):")
    print(test_values)

    # Pack manually using the converter's logic
    out_features, in_features = test_values.shape
    packed_np = np.zeros((out_features, in_features // 2), dtype=np.int8)

    for i in range(in_features // 2):
        # Get two consecutive int4 values
        val1 = test_values[:, 2*i] & 0x0F      # Low 4 bits
        val2 = test_values[:, 2*i + 1] & 0x0F  # Low 4 bits

        # Pack: val2 in high 4 bits, val1 in low 4 bits
        packed_np[:, i] = (val2 << 4) | val1

    print(f"\nPacked bytes (hex):")
    for row in packed_np:
        print(" ".join(f"{b & 0xFF:02x}" for b in row))

    # Unpack using the converter's function
    packed_torch = torch.from_numpy(packed_np).cuda()
    unpacked_torch = unpack_int4_from_int8(packed_torch, out_features, in_features)
    unpacked_np = unpacked_torch.cpu().numpy()

    print(f"\nUnpacked INT4 values:")
    print(unpacked_np)

    # Check if unpacking is correct
    matches = (unpacked_np == test_values).all()

    print(f"\nExpected unsigned values:")
    print(test_values)
    print(f"\nUnpacked unsigned values:")
    print(unpacked_np)

    if matches:
        print(f"\n✅ Packing/unpacking is correct!")

        # Also verify we can convert back to signed
        unpacked_signed = unpacked_np - 8
        matches_signed = (unpacked_signed == test_values_signed).all()
        print(f"\nUnpacked as signed:")
        print(unpacked_signed)

        if matches_signed:
            print(f"✅ Signed conversion is also correct!")
        else:
            print(f"❌ Signed conversion FAILED!")

        return True
    else:
        print(f"\n❌ Packing/unpacking FAILED!")
        print(f"Differences:")
        diff = unpacked_np - test_values
        print(diff)

        # Find where they differ
        for i in range(out_features):
            for j in range(in_features):
                if unpacked_np[i, j] != test_values[i, j]:
                    orig_byte = test_values[i, j]
                    unpacked_byte = unpacked_np[i, j]
                    print(f"  Row {i}, Col {j}: {orig_byte} != {unpacked_byte}")

        return False

if __name__ == "__main__":
    success = test_packing()
    sys.exit(0 if success else 1)
