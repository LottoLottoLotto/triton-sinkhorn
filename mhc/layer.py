import torch
import torch.nn as nn
import triton
from .kernels import _mhc_sinkhorn_fwd_kernel, _mhc_sinkhorn_bwd_kernel

class MHCSinkhornFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, W, num_iters=5):
        # CRITICAL: Contiguous memory required for flattened pointer math
        W = W.contiguous()
        
        B, n, _ = W.shape
        M = torch.empty_like(W)
        
        # Power of 2 block size
        BLOCK_SIZE = triton.next_power_of_2(n * n)
        
        _mhc_sinkhorn_fwd_kernel[(B,)](
            W, M,
            W.stride(0),
            N_LANES=n, 
            ITERS=num_iters,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        ctx.save_for_backward(M)
        ctx.n_lanes = n
        return M
    
    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.contiguous()
        M, = ctx.saved_tensors
        grad_W = torch.empty_like(grad_output)
        
        B, n = M.shape[0], ctx.n_lanes
        BLOCK_SIZE = triton.next_power_of_2(n * n)
        
        _mhc_sinkhorn_bwd_kernel[(B,)](
            grad_output, M, grad_W,
            M.stride(0), 
            N_LANES=n,
            BLOCK_SIZE=BLOCK_SIZE
        )
        
        return grad_W, None

class FusedMHC(nn.Module):
    def __init__(self, mhc_iters=5):
        super().__init__()
        self.mhc_iters = mhc_iters

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        input_dtype = x.dtype
        input_shape = x.shape
        n_lanes = input_shape[-1]
        
        # 1. Reshape and Force Float32
        # Ensure contiguous layout BEFORE viewing as flat
        x_flat = x.contiguous().view(-1, n_lanes, n_lanes).float()
        
        # 2. Apply Fused Kernel
        out_flat = MHCSinkhornFunction.apply(x_flat, self.mhc_iters)
        
        # 3. Restore Shape and Dtype
        return out_flat.view(input_shape).to(input_dtype)

def mhc_warmup(n_lanes=4, batch_size=32, device='cuda'):
    """
    Compiles Triton kernels before the training loop begins.
    """
    if not torch.cuda.is_available():
        return
        
    print(f"🔥 Warming up MHC kernels for {n_lanes} lanes...")
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        # Fix: Input must require grad for backward() to trigger
        dummy = torch.randn(batch_size, n_lanes, n_lanes, device=device, requires_grad=True)
        layer = FusedMHC(mhc_iters=5).to(device)
        out = layer(dummy)
        out.sum().backward()
        
    torch.cuda.current_stream().wait_stream(stream)
    print("✅ MHC Warmup Complete.")
