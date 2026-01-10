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
