# triton-sinkhorn

**triton-sinkhorn** is a CUDA-fused implementation of the Sinkhorn-Knopp normalization loop designed to produce doubly-stochastic matrices (soft permutations) for deep learning workloads. 

It exposes a PyTorch `nn.Module` (`FusedMHC`) backed by [OpenAI Triton](https://github.com/openai/triton) forward/backward kernels that run the iterative row/column log-normalization in **one compiled kernel per pass**.

The implementation performs the algorithm in log-space (LogSumExp-style) with `float32` accumulation for stability, while accepting lower-precision inputs (FP16/BF16) and casting the result back to the input dtype.

## 🚀 Performance Benchmarks

The following benchmarks compare `FusedMHC` against a standard native PyTorch implementation. The fused kernel significantly reduces memory overhead by avoiding the materialization of intermediate tensors for every iteration of the Sinkhorn loop.

| Batch Size | Lanes | PyTorch (ms) | FusedMHC (ms) | Speedup | Memory Saved |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 32 | 4 | 1.676 | **0.263** | **6.4x** | **94.1%** |
| 128 | 4 | 1.784 | **0.266** | **6.7x** | **94.0%** |
| 32 | 8 | 1.744 | **0.264** | **6.6x** | **93.7%** |
| 128 | 8 | 1.958 | **0.275** | **7.1x** | **93.7%** |

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

```python
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
