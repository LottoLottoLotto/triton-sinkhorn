# triton-sinkhorn

**triton-sinkhorn** is a high-performance, CUDA-fused implementation of the Sinkhorn-Knopp algorithm. It serves as a drop-in replacement for standard PyTorch layers to generate doubly-stochastic matrices (soft permutations) in deep learning models.

Written in **OpenAI Triton**, this layer fuses the iterative normalization steps into a single CUDA kernel launch. This approach significantly reduces CPU overhead, eliminates memory fragmentation, and reduces the memory complexity of the backward pass from $\mathcal{O}(L \times \text{Iters})$ to $\mathcal{O}(1)$.

---

## 🚀 Performance Benchmarks

Benchmarks run on a consumer NVIDIA GPU (WSL2) comparing a naive PyTorch implementation (20 iterations) vs. FusedMHC.

| Batch Size | Lanes | PyTorch (ms) | FusedMHC (ms) | Speedup | Memory Saved |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **32** | 4 | 1.676 | **0.263** | ⚡ **6.4x** | 📉 **94.1%** |
| **128** | 4 | 1.784 | **0.266** | ⚡ **6.7x** | 📉 **94.0%** |
| **32** | 8 | 1.744 | **0.264** | ⚡ **6.6x** | 📉 **93.7%** |
| **128** | 8 | 1.958 | **0.275** | ⚡ **7.1x** | 📉 **93.7%** |

### Key Benefits
* **⚡ 7x Faster Layer Latency:** Reduces layer execution time from ~1.8ms to ~0.27ms, eliminating CPU-bound kernel launch overhead.
* **💾 94% Memory Reduction:** Uses a Riemannian gradient derivation for the backward pass, avoiding the need to save intermediate tensors for every iteration of the loop.
* **🛡️ Numerical Stability:** Operates internally in **Log-Space** (LogSumExp) using Float32 accumulation. This prevents underflow/overflow issues common in standard `exp()` implementations while supporting Float16/BFloat16 inputs.

---

## 📦 Requirements

* **Python:** 3.8+
* **PyTorch:** 2.0+ (with CUDA support)
* **Triton:** (Included with PyTorch 2.0+ on Linux/WSL)
* **OS:** Linux or WSL2 (Windows Subsystem for Linux) is required for Triton compilation.

---

## 🛠️ Usage

### 1. Basic Implementation

Use `FusedMHC` as a standard PyTorch module. It expects a square input matrix of shape `(Batch, Lanes, Lanes)`.

```python
import torch
from mhc.layer import FusedMHC

# 1. Initialize Layer (Define number of Sinkhorn iterations)
# 20 iterations is recommended for convergence to doubly-stochastic
layer = FusedMHC(mhc_iters=20).to('cuda')

# 2. Create Input (Logits)
# Shape: (Batch Size, Lanes, Lanes)
x = torch.randn(32, 4, 4, device='cuda', dtype=torch.float16, requires_grad=True)

# 3. Forward Pass
# Returns a doubly-stochastic matrix (Rows and Cols sum to 1.0)
out = layer(x)

print(out[0]) 
# tensor([[0.25, 0.25, 0.25, 0.25], ...])


### 2. Integration: Lane Mixing
Example wrapper for learning mixing weights in a Transformer-like architecture:

```Python

import torch.nn as nn

class FusedSinkhornLaneMixer(nn.Module):
    def __init__(self, num_lanes, iters=20):
        super().__init__()
        self.lanes = num_lanes
        
        # Learnable mixing weights (initialized in Float32 for stability)
        self.mixing_logits = nn.Parameter(torch.randn(num_lanes, num_lanes) * 0.02)
        
        # The Fused Kernel
        self.sinkhorn = FusedMHC(mhc_iters=iters)

    def forward(self, x_lanes):
        # x_lanes shape: (Batch, Seq, Lanes, Dim)
        original_dtype = x_lanes.dtype 
        
        # 1. Normalize weights (Compute P)
        # Unsqueeze to add batch dim: (1, Lanes, Lanes)
        logits_batched = self.mixing_logits.unsqueeze(0).float()
        P = self.sinkhorn(logits_batched).squeeze(0)
        
        # 2. Mix the lanes (Apply P)
        # "bsid,oi->bsod" -> Apply mixing matrix P to the lane dimension
        out = torch.einsum('bsid,oi->bsod', x_lanes, P.to(x_lanes.device))
        
        # 3. Cast back to original precision (e.g., float16)
        return out.to(original_dtype)```

### 📊 Benchmark Script
To reproduce the performance results, save this code as benchmark.py.

Note: The reference implementation below correctly normalizes across rows (dim 2) and columns (dim 1) for batched inputs.

```Python

import torch
import torch.nn as nn
import pandas as pd
from mhc.layer import FusedMHC

class SlowSinkhorn(nn.Module):
    def __init__(self, iters=20, eps=1e-6):
        super().__init__()
        self.iters = iters
        self.eps = eps

    def forward(self, x):
        # Naive implementation (exp -> normalize)
        P = torch.exp(x.float())
        for _ in range(self.iters):
            # Row normalize: sum over columns
            P = P / (P.sum(dim=2, keepdim=True) + self.eps)
            # Col normalize: sum over rows
            P = P / (P.sum(dim=1, keepdim=True) + self.eps)
        return P

def benchmark_layer(name, layer, x, iters=100):
    # Warmup
    for _ in range(10):
        loss = layer(x).sum(); loss.backward(); x.grad = None
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    torch.cuda.reset_peak_memory_stats()
    mem_start = torch.cuda.memory_allocated()

    start_event.record()
    for _ in range(iters):
        out = layer(x)
        loss = out.sum()
        loss.backward()
        x.grad = None
    end_event.record()
    torch.cuda.synchronize()
    
    return start_event.elapsed_time(end_event) / iters, (torch.cuda.max_memory_allocated() - mem_start) / 1024**2

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Benchmarking on {device}...")
    
    configs = [{"B": 32, "L": 4}, {"B": 128, "L": 4}, {"B": 32, "L": 8}, {"B": 128, "L": 8}]
    results = []

    for cfg in configs:
        B, L, I = cfg["B"], cfg["L"], 20
        x = torch.randn(B, L, L, device=device, requires_grad=True)
        
        t_slow, mem_slow = benchmark_layer("Slow", SlowSinkhorn(I).to(device), x)
        t_fast, mem_fast = benchmark_layer("Fast", FusedMHC(I).to(device), x)
        
        results.append({
            "Batch": B, "Lanes": L,
            "Slow (ms)": f"{t_slow:.3f}", "Fast (ms)": f"{t_fast:.3f}",
            "Speedup": f"{t_slow/t_fast:.1f}x" if t_fast > 0 else "N/A",
            "Mem Saved": f"{100*(1-(mem_fast/mem_slow)):.1f}%" if mem_slow > 0 else "N/A"
        })

    print(pd.DataFrame(results).to_string(index=False))```

### 📂 File Structure
kernels.py: Contains the raw Triton kernels (_mhc_sinkhorn_fwd_kernel and _mhc_sinkhorn_bwd_kernel).

layer.py: The PyTorch autograd.Function wrapper and nn.Module interface. Handles contiguous memory enforcement and type casting.

benchmark.py: Script to reproduce the performance results.

##License
MIT License. Free to use in personal and commercial projects.
