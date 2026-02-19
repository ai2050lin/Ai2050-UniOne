"""
FiberNet v4.1 - 几何涌现验证完整版
功能：训练 5 epoch -> 保存模型 -> 提取激活 -> TDA/几何分析
"""
import os, sys, time, math, json, urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata")
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

def download_shakespeare():
    os.makedirs(DATA_DIR, exist_ok=True)
    fp = os.path.join(DATA_DIR, "shakespeare.txt")
    if not os.path.exists(fp):
        urllib.request.urlretrieve(SHAKESPEARE_URL, fp)
    with open(fp, 'r', encoding='utf-8') as f: return f.read()

class CharDataset(Dataset):
    def __init__(self, text, seq_len=128):
        self.seq_len = seq_len
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.c2i = {c: i for i, c in enumerate(chars)}
        self.data = torch.tensor([self.c2i[c] for c in text], dtype=torch.long)
    def __len__(self): return max(0, len(self.data) - self.seq_len - 1)
    def __getitem__(self, i): return self.data[i:i+self.seq_len], self.data[i+1:i+self.seq_len+1]

# ====================== Models (Manual Mask) ======================
class FiberLayer(nn.Module):
    def __init__(self, d_logic, d_memory, nhead, d_ff, dropout=0.1):
        super().__init__()
        self.logic_attn = nn.MultiheadAttention(d_logic, max(1, d_logic//32), batch_first=True, dropout=dropout)
        self.logic_norm1 = nn.LayerNorm(d_logic)
        self.logic_ffn = nn.Sequential(nn.Linear(d_logic,d_ff//2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff//2,d_logic), nn.Dropout(dropout))
        self.logic_norm2 = nn.LayerNorm(d_logic)
        
        self.nhead = nhead; self.d_memory = d_memory; self.head_dim = d_memory // nhead
        self.W_Q = nn.Linear(d_logic, d_memory)
        self.W_K = nn.Linear(d_logic, d_memory)
        self.W_V = nn.Linear(d_memory, d_memory)
        self.W_O = nn.Linear(d_memory, d_memory)
        self.mem_norm1 = nn.LayerNorm(d_memory)
        self.mem_ffn = nn.Sequential(nn.Linear(d_memory,d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff,d_memory), nn.Dropout(dropout))
        self.mem_norm2 = nn.LayerNorm(d_memory)

    def forward(self, x_logic, x_memory, causal_mask):
        res = x_logic
        x_logic, _ = self.logic_attn(x_logic, x_logic, x_logic, attn_mask=causal_mask, is_causal=False)
        x_logic = self.logic_norm1(res + x_logic)
        res = x_logic
        x_logic = self.logic_norm2(res + self.logic_ffn(x_logic))

        B, S, _ = x_logic.shape
        Q = self.W_Q(x_logic).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        K = self.W_K(x_logic).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        V = self.W_V(x_memory).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        
        out = F.scaled_dot_product_attention(Q, K, V, attn_mask=causal_mask, is_causal=False)
        out = out.transpose(1, 2).reshape(B, S, self.d_memory)
        out = self.W_O(out)
        
        x_memory = self.mem_norm2(self.mem_norm1(x_memory + out) + self.mem_ffn(self.mem_norm1(x_memory + out)))
        return x_logic, x_memory

class ScaledFiberNet(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=8, nhead=8, d_ff=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.d_logic = d_model//2; self.d_memory = d_model
        self.pos_embed = nn.Embedding(max_len, self.d_logic)
        self.tok_embed = nn.Embedding(vocab_size, self.d_memory)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([FiberLayer(self.d_logic, self.d_memory, nhead, d_ff, dropout) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(self.d_memory)
        self.head = nn.Linear(self.d_memory, vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        logic = self.drop(self.pos_embed(pos).expand(B, -1, -1))
        memory = self.drop(self.tok_embed(x))
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        for layer in self.layers:
            logic, memory = layer(logic, memory, mask)
        return self.head(self.norm(memory)), logic, memory

class ScaledTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=8, nhead=8, d_ff=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout, batch_first=True, activation='gelu')
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        B, S = x.shape
        mask = nn.Transformer.generate_square_subsequent_mask(S, device=x.device)
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        h = self.drop(self.tok_embed(x) + self.pos_embed(pos))
        for layer in self.layers:
            h = layer(h, src_mask=mask, is_causal=False)
        return self.head(self.norm(h)), None, h

# ====================== Utils ======================
class EmergenceDetector:
    @staticmethod
    def analyze(acts):
        res = {'b1': 0, 'b1_sig': 0, 'dim': 0.0, 'periodic': False}
        # TDA
        try: 
            from ripser import ripser
            if len(acts)>1500: subs = acts[np.random.choice(len(acts), 1500, False)]
            else: subs = acts
            subs = (subs - subs.mean(0))/(subs.std(0)+1e-8)
            r = ripser(subs, maxdim=1, thresh=2.0)['dgms']
            if len(r)>1:
                h1 = r[1]
                res['b1'] = len(h1)
                res['b1_sig'] = len([p for p in h1 if (p[1]-p[0])>0.1])
        except: pass
        
        # Intrinsic Dim
        try:
            from sklearn.neighbors import NearestNeighbors
            if len(acts)>2000: subs = acts[np.random.choice(len(acts), 2000, False)]
            else: subs = acts
            subs = (subs - subs.mean(0))/(subs.std(0)+1e-8)
            nn_ = NearestNeighbors(n_neighbors=11).fit(subs)
            d, _ = nn_.kneighbors(subs)
            d = np.maximum(d[:,1:], 1e-10)
            est = 10 / np.sum(np.log(d[:,-1:] / d[:,:-1]), axis=1)
            res['dim'] = float(np.mean(est))
        except: pass

        # Spectral
        try:
            sp = np.abs(np.fft.fft(subs, axis=0)).mean(1); sp[0]=0
            peaks = np.where(sp > (sp.mean()+2*sp.std()))[0]
            res['periodic'] = len(peaks)>3
        except: pass
        return res

@torch.no_grad()
def get_acts(model, ld, dev):
    model.eval()
    A = []
    for i, (x, _) in enumerate(ld):
        if i>=50: break
        x=x.to(dev)
        _, _, me = model(x)
        if me is not None: A.append(me[:,-1,:].cpu().numpy())
    return np.concatenate(A) if A else None

def train_ep(model, ld, opt, dev, ep, name):
    model.train()
    L, T = 0, 0
    t0 = time.time()
    n = len(ld)
    for i, (x, y) in enumerate(ld):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        loss = F.cross_entropy(model(x)[0].view(-1, model.head.out_features), y.view(-1))
        if not torch.isnan(loss):
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            L += loss.item()*y.numel(); T += y.numel()
        if (i+1)%100==0 or (i+1)==n: 
            el = time.time()-t0
            print(f"\r[{name}] Ep {ep} | {i+1}/{n} | loss={L/T:.4f} | {el:.0f}s", end='', flush=True)
    print()
    return L/max(T,1)

@torch.no_grad()
def eval_ep(model, ld, dev):
    model.eval()
    L, T = 0, 0
    for x, y in ld:
        x, y = x.to(dev), y.to(dev)
        loss = F.cross_entropy(model(x)[0].view(-1, model.head.out_features), y.view(-1))
        if not torch.isnan(loss): L+=loss.item()*y.numel(); T+=y.numel()
    return L/max(T,1)

def main():
    print("Start v4.1 (Full Experiment)...")
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    txt = download_shakespeare()
    sp = int(len(txt)*0.9)
    ds_t = CharDataset(txt[:sp]); ds_v = CharDataset(txt[sp:])
    ld_t = DataLoader(ds_t, 64, shuffle=True, drop_last=True)
    ld_v = DataLoader(ds_v, 64, drop_last=True)
    VS = ds_t.vocab_size
    
    fnet = ScaledFiberNet(VS).to(DEV)
    tnet = ScaledTransformer(VS).to(DEV)
    
    results = {}
    
    for n, m in [("FiberNet", fnet), ("Transformer", tnet)]:
        print(f"\nTraining {n}...")
        opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
        hist = []
        for ep in range(1, 6): # 5 epochs
            tr = train_ep(m, ld_t, opt, DEV, ep, n)
            vl = eval_ep(m, ld_v, DEV)
            print(f"  Result: tr={tr:.4f} vl={vl:.4f}")
            hist.append({'tr': tr, 'vl': vl})
        
        # Save
        torch.save(m.state_dict(), os.path.join(DATA_DIR, f"{n}_v4.pth"))
        
        # Analyze
        print(f"Analyzing {n}...")
        acts = get_acts(m, ld_v, DEV)
        emg = EmergenceDetector.analyze(acts)
        print(f"  Emergence: {emg}")
        results[n] = {'history': hist, 'emergence': emg}

    json.dump(results, open(os.path.join(DATA_DIR, "v4_1_report.json"), "w"))
    print("\nDone. Report saved.")

if __name__ == "__main__": main()
