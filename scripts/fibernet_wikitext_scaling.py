"""
FiberNet Phase 3 - WikiText 20M Scaling Experiment (v3 - Optimized)
核心目标：寻找内在维度 (ID) 下降转折点 + TDA β1 涌现
更新：
1. 1250 steps/epoch (约 1 Full Pass)
2. 每 Epoch 进行几何分析与 Checkpoint 保存
3. 实时 JSON 写入
"""
import os
import time
import json
import glob
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata")

class WikiDataset(Dataset):
    def __init__(self, data_path, seq_len=128):
        self.seq_len = seq_len
        self.data = torch.from_numpy(np.load(data_path).astype(np.int64))
        print(f"[Data] 加载 Wiki 数据: {len(self.data)} tokens")
    def __len__(self): return len(self.data) - self.seq_len - 1
    def __getitem__(self, i): return self.data[i:i+self.seq_len], self.data[i+1:i+self.seq_len+1]

class FiberLayer(nn.Module):
    def __init__(self, d_logic, d_memory, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.logic_attn = nn.MultiheadAttention(d_logic, nhead, batch_first=True, dropout=dropout)
        self.logic_norm1 = nn.LayerNorm(d_logic)
        self.logic_ffn = nn.Sequential(nn.Linear(d_logic, d_ff//2), nn.GELU(), nn.Linear(d_ff//2, d_logic))
        self.logic_norm2 = nn.LayerNorm(d_logic)
        self.nhead = nhead; self.d_memory = d_memory; self.head_dim = d_memory // nhead
        self.W_Q = nn.Linear(d_logic, d_memory); self.W_K = nn.Linear(d_logic, d_memory); self.W_V = nn.Linear(d_memory, d_memory); self.W_O = nn.Linear(d_memory, d_memory)
        self.mem_norm1 = nn.LayerNorm(d_memory)
        self.mem_ffn = nn.Sequential(nn.Linear(d_memory, d_ff), nn.GELU(), nn.Linear(d_ff, d_memory))
        self.mem_norm2 = nn.LayerNorm(d_memory)

    def forward(self, x_logic, x_memory, mask):
        res = x_logic
        x_logic, _ = self.logic_attn(x_logic, x_logic, x_logic, attn_mask=mask, is_causal=False)
        x_logic = self.logic_norm1(res + x_logic)
        x_logic = self.logic_norm2(x_logic + self.logic_ffn(x_logic))
        B, S, _ = x_logic.shape
        Q = self.W_Q(x_logic).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        K = self.W_K(x_logic).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        V = self.W_V(x_memory).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask, is_causal=False)
        out = out.transpose(1, 2).reshape(B, S, self.d_memory)
        out = self.W_O(out)
        x_memory = self.mem_norm1(x_memory + out)
        x_memory = self.mem_norm2(x_memory + self.mem_ffn(x_memory))
        return x_logic, x_memory

class ScaledFiberNet(nn.Module):
    def __init__(self, vocab_size=256, d_model=384, n_layers=12, nhead=12, d_ff=1536, dropout=0.1, max_len=512):
        super().__init__()
        self.d_logic = d_model // 2; self.d_memory = d_model
        self.pos_embed = nn.Embedding(max_len, self.d_logic)
        self.tok_embed = nn.Embedding(vocab_size, self.d_memory)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([FiberLayer(self.d_logic, self.d_memory, nhead, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(self.d_memory); self.head = nn.Linear(self.d_memory, vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, S = x.shape; pos = torch.arange(S, device=x.device).unsqueeze(0)
        logic = self.drop(self.pos_embed(pos).expand(B, -1, -1)); memory = self.drop(self.tok_embed(x))
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        for layer in self.layers: logic, memory = layer(logic, memory, mask)
        return self.head(self.norm(memory)), logic, memory

class EmergenceMonitor:
    @staticmethod
    def calc_id(acts):
        res = {} 
        try:
            from sklearn.neighbors import NearestNeighbors
            subs = acts[np.random.choice(len(acts), min(2000, len(acts)), False)]
            # Robust normalization
            std = subs.std(0)
            subs = (subs - subs.mean(0))/(std + 1e-8)
            
            nn_ = NearestNeighbors(n_neighbors=11).fit(subs)
            d, _ = nn_.kneighbors(subs)
            d = np.maximum(d[:, 1:], 1e-10)
            ratio = d[:, -1:] / d[:, :-1]
            
            # Robust log
            log_ratio = np.log(np.maximum(ratio, 1.0001))
            est = 10 / np.sum(log_ratio, axis=1)
            return float(np.mean(est))
        except Exception as e:
            print(f"Error calculating ID: {e}")
            return float('nan')

def find_latest_ckpt():
    files = glob.glob(os.path.join(DATA_DIR, "fiber_20m_ep*.pth"))
    if not files: return 0, None
    latest = max(files, key=os.path.getmtime)
    ep = int(re.search(r"ep(\d+)", latest).group(1))
    return ep, latest

def train():
    print("Start Phase 3 (WikiText-2 @ 20M) - Optimized...")
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = os.path.join(DATA_DIR, "wiki_v4_3.npy")
    ds = WikiDataset(data_path); ld = DataLoader(ds, 64, shuffle=True, drop_last=True)
    
    model = ScaledFiberNet().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    start_ep, ckpt = find_latest_ckpt()
    results = []
    res_path = os.path.join(DATA_DIR, "phase3_wiki_results.json")
    if os.path.exists(res_path):
        try: results = json.load(open(res_path))
        except: pass
    
    if ckpt:
        print(f"[Resume] Loading checkpoint from Epoch {start_ep}: {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=DEV))
    else:
        print("[Init] Starting fresh training")
        
    print(f"[Model] FiberNet 20M Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
    
    for ep in range(start_ep + 1, 51): # 50 Epochs
        model.train(); L, T = 0, 0; t0 = time.time()
        for i, (x, y) in enumerate(ld):
            if i > 1250: break # 1250 steps approx 1 pass
            x, y = x.to(DEV), y.to(DEV); opt.zero_grad()
            logits, _, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, 256), y.view(-1))
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            L += loss.item()*y.numel(); T += y.numel()
            if (i+1)%250==0: print(f"  Batch {i+1} | loss: {L/T:.4f}")
        
        avg_loss = L/max(T,1)
        
        # 每 Epoch 都进行分析与保存
        print(f"[*] 分析几何演化 (Epoch {ep})...")
        model.eval(); A = []
        with torch.no_grad():
            for x, _ in list(ld)[:20]: A.append(model(x.to(DEV))[2][:,-1,:].cpu().numpy())
        acts = np.concatenate(A)
        id_val = EmergenceMonitor.calc_id(acts)
        print(f"  >> Intrinsic Dim: {id_val:.2f}")
        
        # Save Checkpoint every epoch (overwrite previous to save space, or keep all?)
        # Keep all for now since step size is small
        torch.save(model.state_dict(), os.path.join(DATA_DIR, f"fiber_20m_ep{ep}.pth"))
        
        # 实时保存结果
        results.append({'ep': ep, 'loss': avg_loss, 'id': id_val})
        json.dump(results, open(res_path, "w"))
            
    print("\n[+] 实验完成。数据已保存。")

if __name__ == "__main__": train()
