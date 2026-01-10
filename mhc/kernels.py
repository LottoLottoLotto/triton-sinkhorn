import torch
import triton
import triton.language as tl

# ============================================================================
# FORWARD KERNEL: Fused Register-Based Sinkhorn
# ============================================================================
@triton.jit
def _mhc_sinkhorn_fwd_kernel(
    W_ptr, M_ptr,
    stride_batch,
    N_LANES: tl.constexpr, 
    ITERS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    batch_idx = tl.program_id(0)
    
    # 1. Setup Offsets
    offsets = tl.arange(0, BLOCK_SIZE)
    # Global mask for valid matrix elements
    mask = offsets < (N_LANES * N_LANES)
    
    # 2. Load Data with Cache Hint
    w_base = W_ptr + batch_idx * stride_batch
    w_flat = tl.load(w_base + offsets, mask=mask, other=-float('inf'), 
                     eviction_policy='evict_last').to(tl.float32)
    
    # 3. Stability Clamp (Guardrail)
    log_w = tl.maximum(w_flat, -1e5)
    
    # 4. Fused Sinkhorn Loop (In Registers)
    for _ in tl.static_range(ITERS):
        # --- Row Norm ---
        for i in tl.static_range(N_LANES):
            row_start = i * N_LANES
            # EXPLICIT SAFETY: Ensure we never touch padded threads
            row_mask = (offsets >= row_start) & (offsets < row_start + N_LANES) & mask
            
            row_data = tl.where(row_mask, log_w, -float('inf'))
            row_max = tl.max(row_data, axis=0)
            
            # Safe LogSumExp
            row_lse = row_max + tl.log(tl.sum(tl.exp(row_data - row_max), axis=0))
            
            # Update Row
            log_w = tl.where(row_mask, log_w - row_lse, log_w)
        
        # --- Col Norm ---
        for j in tl.static_range(N_LANES):
            # EXPLICIT SAFETY: Ensure we never touch padded threads
            col_mask = ((offsets % N_LANES) == j) & mask
            
            col_data = tl.where(col_mask, log_w, -float('inf'))
            col_max = tl.max(col_data, axis=0)
            
            # Safe LogSumExp
            col_lse = col_max + tl.log(tl.sum(tl.exp(col_data - col_max), axis=0))
            
            # Update Col
            log_w = tl.where(col_mask, log_w - col_lse, log_w)
            
    # 5. Store Result
    m_flat = tl.exp(log_w)
    m_base = M_ptr + batch_idx * stride_batch
    tl.store(m_base + offsets, m_flat, mask=mask)

# ============================================================================
# BACKWARD KERNEL: Projected Riemannian Gradient
# ============================================================================
@triton.jit
def _mhc_sinkhorn_bwd_kernel(
    grad_M_ptr, M_ptr, grad_W_ptr,
    stride_batch, 
    N_LANES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    """
    Computes the gradient via projection onto the tangent space of the 
    Birkhoff polytope.
    """
    batch_idx = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < (N_LANES * N_LANES)
    
    offset_ptr = batch_idx * stride_batch
    
    # Load Matrices
    m = tl.load(M_ptr + offset_ptr + offsets, mask=mask, eviction_policy='evict_last').to(tl.float32)
    grad_m = tl.load(grad_M_ptr + offset_ptr + offsets, mask=mask, eviction_policy='evict_last').to(tl.float32)
    
    # 1. Global Mean
    grad_sum = tl.sum(tl.where(mask, grad_m, 0.0), axis=0)
    global_mean = grad_sum / (N_LANES * N_LANES)

    # 2. Row Means
    row_means_map = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in tl.static_range(N_LANES):
        row_start = i * N_LANES
        row_mask = (offsets >= row_start) & (offsets < row_start + N_LANES)
        row_vals = tl.where(row_mask, grad_m, 0.0)
        row_mean = tl.sum(row_vals, axis=0) / N_LANES
        row_means_map = tl.where(row_mask, row_mean, row_means_map)

    # 3. Col Means
    col_means_map = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for j in tl.static_range(N_LANES):
        col_mask = ((offsets % N_LANES) == j) & mask
        col_vals = tl.where(col_mask, grad_m, 0.0)
        col_mean = tl.sum(col_vals, axis=0) / N_LANES
        col_means_map = tl.where(col_mask, col_mean, col_means_map)

    # 4. Apply Projection
    grad_w = m * (grad_m - row_means_map - col_means_map + global_mean)
    
    tl.store(grad_W_ptr + offset_ptr + offsets, grad_w, mask=mask)
