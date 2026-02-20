"""
FiberNet 离线几何审计工具 (v1.1 - Debug)
功能：加载 Checkpoint，执行数值稳定的 ID 估计、TDA 持久同调与频谱周期检测。
"""
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

# 引用原始模型定义 (确保与训练脚本一致)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fibernet_wikitext_scaling import ScaledFiberNet, WikiDataset, DATA_DIR

def audit_checkpoint(epoch):
    ckpt_path = os.path.join(DATA_DIR, f"fiber_20m_ep{epoch}.pth")
    if not os.path.exists(ckpt_path):
        print(f"[!] 找不到 Checkpoint: {ckpt_path}"); return
    
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScaledFiberNet().to(DEV)
    model.load_state_dict(torch.load(ckpt_path, map_location=DEV))
    model.eval()
    print(f"[*] 审计 Epoch {epoch} 流形特性...")
    
    # 加载验证数据
    data_path = os.path.join(DATA_DIR, "wiki_v4_3.npy")
    ds = WikiDataset(data_path)
    ld = DataLoader(ds, 64, shuffle=False)
    
    # 提取激活 (Logic Stream)
    A = []
    with torch.no_grad():
        for i, (x, _) in enumerate(ld):
            if i >= 40: break
            _, _, memory = model(x.to(DEV))
            # [Batch, Seq, Dim] -> Take last token
            A.append(memory[:, -1, :].cpu().numpy())
    acts = np.concatenate(A)
    
    # 核心分析
    print(f"  [Stats] Mean: {acts.mean():.6f}, Std: {acts.std():.6f}")
    
    # 距离分布检查
    from sklearn.neighbors import NearestNeighbors
    if len(acts) > 2000: acts = acts[np.random.choice(len(acts), 2000, False)]
    
    # Normalization check
    std = acts.std(0)
    print(f"  [Feat] Zero std features: {np.sum(std==0)} / {len(std)}")
    acts = (acts - acts.mean(0))/(std+1e-8) 
    
    nn_ = NearestNeighbors(n_neighbors=11).fit(acts)
    d, _ = nn_.kneighbors(acts)
    d1 = d[:, 1]; dk = d[:, -1]
    print(f"  [Dist] d1_mean: {d1.mean():.6f}, dk_mean: {dk.mean():.6f}")
    
    # Ratio check
    ratio = dk / np.maximum(d1, 1e-10)
    print(f"  [Ratio] Mean: {ratio.mean():.6f}, Max: {ratio.max():.6f}, Min: {ratio.min():.6f}")
    
    # ID calculation
    log_ratio = np.log(np.maximum(ratio, 1.0001))
    id_val = 10 / np.mean(log_ratio)
    print(f"  >> [ID] 内在维度 (Audit): {id_val:.4f}")
    
    # 简单的 TDA (可选)
    try:
        from ripser import ripser
        print("  [TDA] Running persistent homology...")
        r = ripser(acts[np.random.choice(len(acts), 1000, False)], maxdim=1, thresh=1.5)['dgms']
        b1_cnt = len([p for p in r[1] if (p[1]-p[0]) > 0.1]) if len(r) > 1 else 0
        print(f"  >> [TDA] Beta-1 (环面显著性): {b1_cnt}")
    except Exception as e:
        print(f"  [TDA] Skipped: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        audit_checkpoint(int(sys.argv[1]))
    else:
        audit_checkpoint(6)
