"""
测试内在维度 (ID) 计算的数值稳定性
"""
import numpy as np
import torch

def robust_calc_id(acts):
    res = {} 
    try:
        from sklearn.neighbors import NearestNeighbors
        # 模拟 acts
        if len(acts) > 2000: acts = acts[np.random.choice(len(acts), 2000, False)]
        
        # 归一化 (关键点: std 可能为 0)
        std = acts.std(0)
        if np.any(std == 0):
            print("Warning: Feature std is 0")
        subs = (acts - acts.mean(0))/(std + 1e-8)
        
        nn_ = NearestNeighbors(n_neighbors=11).fit(subs)
        d, _ = nn_.kneighbors(subs)
        
        # d[:, 0] is self-distance (0)
        # d[:, 1] is 1st NN
        
        # Edge case: distinct points but very close
        d = np.maximum(d[:, 1:], 1e-10)
        
        ratio = d[:, -1:] / d[:, :-1]
        
        # Edge case: ratio = 1 (if points are equidistant or identical)
        # log(1) = 0 -> sum = 0 -> 10/0 = inf
        log_ratio = np.log(np.maximum(ratio, 1.0001))
        
        est = 10 / np.sum(log_ratio, axis=1)
        return float(np.mean(est))
    except Exception as e:
        print(f"Error: {e}")
        return float('nan')

def test():
    print("Test 1: Normal uniform data")
    data = np.random.rand(1000, 64)
    print(f"ID: {robust_calc_id(data)}")
    
    print("\nTest 2: All zeros (degenerate)")
    data = np.zeros((1000, 64))
    print(f"ID: {robust_calc_id(data)}")
    
    print("\nTest 3: Repeated points (clusters)")
    # 5 clusters of identical points
    centers = np.random.rand(5, 64)
    data = np.repeat(centers, 200, axis=0) # 1000 points
    print(f"ID: {robust_calc_id(data)}")
    
    print("\nTest 4: High dimensional manifold (sphere)")
    data = np.random.randn(1000, 64)
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    print(f"ID: {robust_calc_id(data)}")

if __name__ == "__main__":
    test()
