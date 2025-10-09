import torch, importlib
import bitblas

# 1) 打印 bitblas 里有哪些“dequant/matmul”可用
print("bitblas version:", getattr(bitblas, "__version__", "unknown"))
candidates = []
if hasattr(bitblas, "gpu"):
    for name in dir(bitblas.gpu):
        mod = getattr(bitblas.gpu, name)
        if hasattr(mod, "matmul_dequantize"):
            candidates.append(("gpu."+name, mod.matmul_dequantize))
print("candidates:", [n for n,_ in candidates])

assert candidates, "No matmul_dequantize candidate found in bitblas.gpu.*"

matmul_deq = candidates[0][1]  # 取一个

# 2) 造一组 W4A8 测试数据（和你 QLinear 的打包一致：两 4bit 拼 1 字节）
B, K, N = 32, 1024, 2048   # x:(B,K), W:(N,K)
torch.manual_seed(0)
x = torch.randn(B, K, device="cuda", dtype=torch.float16).abs()  # 正数便于直观
w = torch.randn(N, K, device="cuda", dtype=torch.float16)

# 权重 per-row scale & 4bit 量化
s_w = w.abs().amax(dim=1).clamp_(1e-6) / 7.0  # 对应 [-8,7]
q_w = torch.clamp(torch.round(w / s_w[:,None]), -8, 7).to(torch.int8)

# nibble 打包（K 保证偶数）
def pack_nibbles_int4(q):
    qm = torch.remainder(q.to(torch.int16), 16).to(torch.uint8)
    lo = qm[:,0::2]; hi = qm[:,1::2]
    return (lo | (hi<<4)).contiguous()

packed_w = pack_nibbles_int4(q_w)  # [N, K/2] uint8

# 激活 per-col scale（百分位法可换成你的 calibrator）
s_a = x.abs().quantile(0.999, dim=0).clamp_(1e-6) / 127.0  # A8 -> /127

# 3) 调用 bitblas 的 dequant GEMM
y = matmul_deq(x, packed_w, s_w.to(torch.float16), s_a.to(torch.float16), bias=None)
print("y:", y.shape, y.dtype, y.is_cuda)

# 4) 与 FP16 参考对比（反量化等价：x/sa @ (qw*sw)^T）
x_q = torch.round(x / s_a).clamp(-127,127)
w_dq = (q_w.to(torch.float16) * s_w[:,None])
y_ref = (x_q.to(torch.float16) @ w_dq.T)
print("L2:", (y - y_ref).norm().item())
