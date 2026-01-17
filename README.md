# triton-sinkhorn

**triton-sinkhorn** is a CUDA-fused implementation of the Sinkhorn-Knopp normalization loop designed to produce doubly-stochastic matrices (soft permutations) for deep learning workloads. 

It exposes a PyTorch `nn.Module` (`FusedMHC`) backed by [OpenAI Triton](https://github.com/openai/triton) forward/backward kernels that run the iterative row/column log-normalization in **one compiled kernel per pass**.

The implementation performs the algorithm in log-space (LogSumExp-style) with `float32` accumulation for stability, while accepting lower-precision inputs (FP16/BF16) and casting the result back to the input dtype.

## 🚀 Performance Benchmarks

The following benchmarks compare `FusedMHC` against a standard native PyTorch implementation. The fused kernel significantly reduces memory overhead by avoiding the materialization of intermediate tensors for every iteration of the Sinkhorn loop.

| Batch Size | Lanes | PyTorch (ms) | FusedMHC (ms) | Speedup | Memory Saved |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 32 | 4 | 2.038 | **0.279** | **7.3x** | **21.7%** |
| 128 | 4 | 1.823 | **0.284** | **6.4x** | **16.1%** |
| 32 | 8 | 1.730 | **0.301** | **5.7x** | **11.8%** |
| 128 | 8 | 2.193 | **0.282** | **7.8x** | **10.1%** |


## ✨ Key Benefits

* **Fused Forward**: Row log-softmax and column log-softmax operations are unrolled for `ITERS` iterations completely inside a single Triton kernel.
* **Exact Backward**: Backpropagation is implemented in Triton by replaying the loop in reverse using saved per-iteration log-space states.
* **Numerical Stability**: The forward pass clamps logits with `max(w, -1e5)` and operates entirely in log-space before exponentiating at the very end to return the transport matrix $P$.
* **Memory Efficiency**: Massive reduction in VRAM usage during training, as intermediate states for the normalization steps are not stored in global memory.

## 📦 Requirements

* Python 3.8+
* PyTorch 2.0+ with CUDA support
* [Triton](https://github.com/openai/triton) (Standard in most modern PyTorch nightly/CUDA builds)
* Linux or WSL2 (Recommended for Triton compilation)

## 🛠️ Usage

### Basic Usage

`FusedMHC` expects the last two dimensions to be square (`n_lanes` x `n_lanes`). It internally flattens any leading dimensions into a batch of matrices.

```
import torch
from mhc.layer import FusedMHC

# Initialize layer (runs 20 iterations of Sinkhorn by default)
layer = FusedMHC(mhc_iters=20).to("cuda")

# Input: [Batch, Lanes, Lanes]
x = torch.randn(32, 4, 4, device="cuda", dtype=torch.float16, requires_grad=True)

# Forward pass
out = layer(x)

# Backward pass
loss = out.sum()
loss.backward()
```
### Implementation Note: 
Autocast is disabled internally. The module runs the fused op in float32 for precision, then casts the output back to the original dtype of the input.

### Integration Example: Lane Mixing
This pattern is useful for "Lane Mixing" in Mixture-of-Experts or multi-lane architectures where you need a soft permutation matrix to mix information between lanes.

Python
```
import torch
import torch.nn as nn
from mhc.layer import FusedMHC

class FusedSinkhornLaneMixer(nn.Module):
    def __init__(self, num_lanes, iters=20):
        super().__init__()
        self.lanes = num_lanes
        # Initialize small random logits
        self.mixing_logits = nn.Parameter(torch.randn(num_lanes, num_lanes) * 0.02)
        self.sinkhorn = FusedMHC(mhc_iters=iters)

    def forward(self, x_lanes):
        original_dtype = x_lanes.dtype
        
        # 1. Compute Soft Permutation Matrix P
        logits_batched = self.mixing_logits.unsqueeze(0).float()
        P = self.sinkhorn(logits_batched).squeeze(0)
        
        # 2. Apply mixing (einsum is usually faster in higher precision)
        # x_lanes shape: [Batch, Seq, Lanes, Dim]
        out = torch.einsum("bsid,oi->bsod", x_lanes, P.to(x_lanes.device))
        
        return out.to(original_dtype)
```

## File Structure
mhc/kernels.py: Contains the raw Triton kernels _mhc_sinkhorn_fwd_kernel and _mhc_sinkhorn_bwd_kernel.

mhc/layer.py: The PyTorch autograd.Function wrapper (MHCSinkhornFunction) and the high-level module (FusedMHC) that handles contiguity and dtype management.

benchmark.py: A script to verify speedups and memory savings against a naive PyTorch implementation.

## Benchmarking
You can reproduce the performance numbers by running the provided benchmark script.

Save the code below as benchmark.py.

Run python benchmark.py.
```
Python

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
        P = torch.exp(x.float())
        for _ in range(self.iters):
            P = P / (P.sum(dim=2, keepdim=True) + self.eps)
            P = P / (P.sum(dim=1, keepdim=True) + self.eps)
        return P

def benchmark_layer(layer, x, iters=100):
    # Warmup
    for _ in range(10):
        loss = layer(x).sum()
        loss.backward()
        x.grad = None
        
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    torch.cuda.reset_peak_memory_stats()
    mem_start = torch.cuda.memory_allocated()
    
    start.record()
    for _ in range(iters):
        out = layer(x)
        out.sum().backward()
        x.grad = None
    end.record()
    
    torch.cuda.synchronize()
    t_ms = start.elapsed_time(end) / iters
    mem_mb = (torch.cuda.max_memory_allocated() - mem_start) / 1024**2
    return t_ms, mem_mb

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    configs = [{"B": 32, "L": 4}, {"B": 128, "L": 4}, {"B": 32, "L": 8}, {"B": 128, "L": 8}]
    
    results = []
    print(f"Benchmarking on {torch.cuda.get_device_name(0)}...")
    
    for cfg in configs:
        B, L, I = cfg["B"], cfg["L"], 20
        x = torch.randn(B, L, L, device=device, requires_grad=True)
        
        t_slow, mem_slow = benchmark_layer(SlowSinkhorn(I).to(device), x)
        t_fast, mem_fast = benchmark_layer(FusedMHC(I).to(device), x)
        
        results.append({
            "Batch": B, "Lanes": L,
            "Slow (ms)": f"{t_slow:.3f}",
            "Fast (ms)": f"{t_fast:.3f}",
            "Speedup": f"{t_slow/t_fast:.1f}x" if t_fast > 0 else "N/A",
            "Mem Saved": f"{100*(1-(mem_fast/mem_slow)):.1f}%" if mem_slow > 0 else "N/A"
        })
        
    print(pd.DataFrame(results).to_string(index=False))
```
## License
MIT License.
