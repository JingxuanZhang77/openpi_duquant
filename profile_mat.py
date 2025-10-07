import torch
import time
# Matrix dimensions
def bandwidth_bound():
    M = 1
    K = 4096
    N_values = [4096 * i for i in range(1, 9)]  # N from 4096 to 8*4096
    print(f"{'N(1,4k)*(4k,N)':>12} | {'Time (ms)':>16} | {'FLOPs (TFLOPs)':>22} | {'Bandwidth (TB/s)':>24} | {'AI':>28}")
    print("-" * 120)
    
    for N in N_values:
        A = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    
        # Warmup
        for _ in range(10):
            _ = torch.matmul(A, B)
    
        torch.cuda.synchronize()
        start = time.time()
    
        for _ in range(100):
            _ = torch.matmul(A, B)
    
        torch.cuda.synchronize()
        end = time.time()
    
        avg_time = (end - start) / 100  # seconds
        flops = 2 * M * K * N / avg_time / 1e12  # TFLOPs
        bandwidth = 2 * (M * K + K * N + M * N) / avg_time / 1e12  # TB/s
        AI = 2 * M * K * N / (2 * (M * K + K * N + M * N))
        print(f"{N:12d} | {avg_time * 1000:16.2f} | {flops:22.2f} | {bandwidth:24.2f} | {AI:28.2f}")


def FFN_bound():
    M = 8192
    K = 4096
    N_values = [64 * i for i in range(1, 9)]  # N from 4096 to 8*4096
    print(f"{'N(N,4k)*(4k,8k)':>12} | {'Time (ms)':>16} | {'FLOPs (TFLOPs)':>22} | {'Bandwidth (TB/s)':>24} | {'AI':>28}")
    print("-" * 120)
    
    for N in N_values:
        A = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(K, M, device="cuda", dtype=torch.bfloat16)
    
        # Warmup
        for _ in range(10):
            _ = torch.matmul(A, B)
    
        torch.cuda.synchronize()
        start = time.time()
    
        for _ in range(100):
            _ = torch.matmul(A, B)
    
        torch.cuda.synchronize()
        end = time.time()
    
        avg_time = (end - start) / 100  # seconds
        flops = 2 * M * K * N / avg_time / 1e12  # TFLOPs
        bandwidth = 2 * (M * K + K * N + M * N) / avg_time / 1e12  # TB/s
        AI = 2 * M * K * N / (2 * (M * K + K * N + M * N))
        print(f"{N:12d} | {avg_time * 1000:16.2f} | {flops:22.2f} | {bandwidth:24.2f} | {AI:28.2f}")

def square_both():
    K_values = [1024 * i for i in range(1, 9)]  # N from 4096 to 8*4096
    print(f"{'N(N,N)*(N,N)':>12} | {'Time (ms)':>16} | {'FLOPs (TFLOPs)':>22} | {'Bandwidth (TB/s)':>24} | {'AI':>28}")
    print("-" * 120)
    
    for K in K_values:
        A = torch.randn(K, K, device="cuda", dtype=torch.bfloat16)
        B = torch.randn(K, K, device="cuda", dtype=torch.bfloat16)
    
        # Warmup
        for _ in range(10):
            _ = torch.matmul(A, B)
    
        torch.cuda.synchronize()
        start = time.time()
    
        for _ in range(100):
            _ = torch.matmul(A, B)
    
        torch.cuda.synchronize()
        end = time.time()
    
        avg_time = (end - start) / 100  # seconds
        flops = 2 * K * K * K / avg_time / 1e12  # TFLOPs: A100-312TFLOPS
        bandwidth = 2 * (K * K + K * K + K * K) / avg_time / 1e12  # TB/s: A100-1.9TB/s
        AI = 2 * K * K * K / (2 * (K * K + K * K + K * K))
        print(f"{K:12d} | {avg_time * 1000:16.2f} | {flops:22.2f} | {bandwidth:24.2f} | {AI:28.2f}")
        
print("-" * 120)
print("Test on A100: Peak FLOPS: 312TFLOPS  /  Peak Bandwidth: 1.9TB/s  / AI: 316")
print("-" * 120)
bandwidth_bound()
print("-" * 120)
square_both()
print("-" * 120)
FFN_bound()
print("-" * 120)