import time

import numpy as np
import torch


def verify_fiber_math():
    print("🧪 Verifying Fiber Memory Mathematical Core (V3 - Row Vector Fix)...")
    
    # 1. Setup Synthetic Tensors
    batch, pos, d_model = 1, 8, 768
    resid_old = torch.randn(batch, pos, d_model)
    
    # 2. Create a "Neutralizing" Transport Matrix R
    # For row vectors: v' = v @ R
    # R = [[ cos, sin],
    #      [-sin, cos]]
    theta = 0.5 # radians
    R_np = np.eye(d_model)
    R_np[0, 0] = np.cos(theta)
    R_np[0, 1] = np.sin(theta)
    R_np[1, 0] = -np.sin(theta)
    R_np[1, 1] = np.cos(theta)
    
    R_torch = torch.from_numpy(R_np).float()
    
    # 3. Apply Transformation
    print(f"  Input Shape: {resid_old.shape}")
    print(f"  Transport Matrix Shape: {R_torch.shape}")
    
    resid_new = resid_old @ R_torch
    
    # 4. Math Check (Row Vector Rotation)
    # x' = x*cos - y*sin (if using standard counter-clockwise rotation)
    # Actually, let's just do the dot product manually to be absolutely sure what @ is doing
    # resid_new[0,0,0] = resid_old[0,0,0]*R[0,0] + resid_old[0,0,1]*R[1,0] + ...
    x = resid_old[0, 0, 0]
    y = resid_old[0, 0, 1]
    r00 = R_torch[0, 0]
    r10 = R_torch[1, 0]
    
    expected_x = x * r00 + y * r10
    actual_x = resid_new[0, 0, 0]
    
    diff = torch.abs(actual_x - expected_x).item()
    print(f"  Math Check (Dot Product Error): {diff:.8e}")
    
    # float32 epsilon is around 1e-7
    assert diff < 1e-6, f"Mathematical dot product failed with diff: {diff:.8e}"
    
    # 5. Performance Benchmarking
    iterations = 1000
    start_bench = time.time()
    for _ in range(iterations):
        _ = resid_old @ R_torch
    end_bench = time.time()
    
    avg_latency = (end_bench - start_bench) / iterations
    print(f"  Average Hook Latency: {avg_latency*1000000:.2f} microseconds")

    print("\n✅ Fiber Memory Core Math Verified Successfully.")

if __name__ == "__main__":
    verify_fiber_math()
