"""
FiberNet vs Transformer - Shakespeare 几何涌现验证实验 v3
修复: 因果掩码用 is_causal=True 避免手动掩码, loss 计算验证
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


# ====================== 数据集 ======================
def download_shakespeare():
    os.makedirs(DATA_DIR, exist_ok=True)
    fp = os.path.join(DATA_DIR, "shakespeare.txt")
    if not os.path.exists(fp):
        print("[*] 正在下载 Shakespeare...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, fp)
    with open(fp, 'r', encoding='utf-8') as f:
        return f.read()


class CharDataset(Dataset):
    def __init__(self, text, seq_len=128):
        self.seq_len = seq_len
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.c2i = {c: i for i, c in enumerate(chars)}
        self.data = torch.tensor([self.c2i[c] for c in text], dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len - 1)

    def __getitem__(self, idx):
        return self.data[idx:idx+self.seq_len], self.data[idx+1:idx+self.seq_len+1]


# ====================== FiberNet ======================
class FiberLayer(nn.Module):
    """Logic/Memory 双流纤维丛层 - 使用 F.scaled_dot_product_attention 避免掩码问题"""
    def __init__(self, d_logic, d_memory, nhead, d_ff, dropout=0.1):
        super().__init__()
        nh_logic = max(1, d_logic // 32)  # 保证 head_dim 合理
        self.logic_attn = nn.MultiheadAttention(d_logic, nh_logic, batch_first=True, dropout=dropout)
        self.logic_norm1 = nn.LayerNorm(d_logic)
        self.logic_ffn = nn.Sequential(
            nn.Linear(d_logic, d_ff // 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff // 2, d_logic), nn.Dropout(dropout))
        self.logic_norm2 = nn.LayerNorm(d_logic)

        self.nhead = nhead
        self.d_memory = d_memory
        self.head_dim = d_memory // nhead
        self.W_Q = nn.Linear(d_logic, d_memory)
        self.W_K = nn.Linear(d_logic, d_memory)
        self.W_V = nn.Linear(d_memory, d_memory)
        self.W_O = nn.Linear(d_memory, d_memory)
        self.mem_norm1 = nn.LayerNorm(d_memory)
        self.mem_ffn = nn.Sequential(
            nn.Linear(d_memory, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_memory), nn.Dropout(dropout))
        self.mem_norm2 = nn.LayerNorm(d_memory)

    def forward(self, x_logic, x_memory):
        # Logic 自注意力 (is_causal 自动处理因果掩码)
        res = x_logic
        x_logic, _ = self.logic_attn(x_logic, x_logic, x_logic, is_causal=True)
        x_logic = self.logic_norm1(res + x_logic)
        res = x_logic
        x_logic = self.logic_norm2(res + self.logic_ffn(x_logic))

        # Memory 传输: Logic 驱动 Q/K, Memory 提供 V
        B, S, _ = x_logic.shape
        Q = self.W_Q(x_logic).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        K = self.W_K(x_logic).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        V = self.W_V(x_memory).view(B, S, self.nhead, self.head_dim).transpose(1, 2)

        # SDPA 自动处理因果掩码, 数值稳定
        out = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
        out = out.transpose(1, 2).reshape(B, S, self.d_memory)
        out = self.W_O(out)

        res = x_memory
        x_memory = self.mem_norm1(res + out)
        res = x_memory
        x_memory = self.mem_norm2(res + self.mem_ffn(x_memory))

        return x_logic, x_memory


class ScaledFiberNet(nn.Module):
    def __init__(self, vocab_size, d_model=256, n_layers=8, nhead=8, d_ff=1024,
                 dropout=0.1, max_len=512):
        super().__init__()
        self.d_logic = d_model // 2
        self.d_memory = d_model
        self.pos_embed = nn.Embedding(max_len, self.d_logic)
        self.tok_embed = nn.Embedding(vocab_size, self.d_memory)
        self.drop = nn.Dropout(dropout)
        self.layers = nn.ModuleList([
            FiberLayer(self.d_logic, self.d_memory, nhead, d_ff, dropout)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(self.d_memory)
        self.head = nn.Linear(self.d_memory, vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        logic = self.drop(self.pos_embed(pos).expand(B, -1, -1))
        memory = self.drop(self.tok_embed(x))
        for layer in self.layers:
            logic, memory = layer(logic, memory)
        return self.head(self.norm(memory)), logic, memory


# ====================== Transformer 基线 ======================
class ScaledTransformer(nn.Module):
    """等参数量 Transformer 基线 - 手动循环 Layer 避免 Encoder 版本冲突"""
    def __init__(self, vocab_size, d_model=256, n_layers=8, nhead=8, d_ff=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, d_ff, dropout,
                                      batch_first=True, activation='gelu')
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.tok_embed.weight
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, x):
        B, S = x.shape
        pos = torch.arange(S, device=x.device).unsqueeze(0)
        h = self.drop(self.tok_embed(x) + self.pos_embed(pos))
        
        # 手动循环，强制使用 is_causal=True (PyTorch 2.x 优化路径)
        for layer in self.layers:
            h = layer(h, src_mask=None, is_causal=True)
            
        return self.head(self.norm(h)), None, h


# ====================== 涌现检测 ======================
class EmergenceDetector:
    @staticmethod
    def persistent_homology(acts, n=1500):
        try:
            from ripser import ripser
        except ImportError:
            print("[!] ripser 未安装"); return None
        if len(acts) > n:
            acts = acts[np.random.choice(len(acts), n, replace=False)]
        acts = (acts - acts.mean(0)) / (acts.std(0) + 1e-8)
        print(f"    [TDA] 计算中 (n={len(acts)}, d={acts.shape[1]})...")
        r = ripser(acts, maxdim=1, thresh=2.0)
        dg = r['dgms']
        h0 = dg[0]; h0f = h0[np.isfinite(h0[:,1])]
        h1 = dg[1] if len(dg) > 1 else np.array([]).reshape(0,2)
        sig = h1[(h1[:,1]-h1[:,0]) > 0.1] if len(h1) > 0 else np.array([])
        return {'b0': len(h0f), 'b1': len(h1), 'b1_sig': len(sig),
                'b1_pers': float(np.mean(h1[:,1]-h1[:,0])) if len(h1)>0 else 0.0}

    @staticmethod
    def intrinsic_dim(acts, k=10, n=2000):
        from sklearn.neighbors import NearestNeighbors
        if len(acts) > n:
            acts = acts[np.random.choice(len(acts), n, replace=False)]
        acts = (acts - acts.mean(0)) / (acts.std(0) + 1e-8)
        nn = NearestNeighbors(n_neighbors=k+1).fit(acts)
        d, _ = nn.kneighbors(acts)
        d = np.maximum(d[:,1:], 1e-10)
        est = (k-1) / np.sum(np.log(d[:,-1:] / d[:,:-1]), axis=1)
        return {'mean': float(np.mean(est)), 'std': float(np.std(est)),
                'ambient': acts.shape[1]}

    @staticmethod
    def spectral(acts, n=2000):
        if len(acts) > n:
            acts = acts[np.random.choice(len(acts), n, replace=False)]
        sp = np.abs(np.fft.fft(acts, axis=0)).mean(1)
        sp[0] = 0
        th = sp.mean() + 2*sp.std()
        peaks = np.where(sp > th)[0]
        return {'peaks': len(peaks), 'periodic': len(peaks) > 3}


# ====================== 训练 ======================
def train_one_epoch(model, loader, opt, dev, name, ep, total):
    model.train()
    tot_loss = 0.0
    tot_tok = 0
    n = len(loader)
    t0 = time.time()
    for i, (x, y) in enumerate(loader):
        x, y = x.to(dev), y.to(dev)
        opt.zero_grad()
        logits = model(x)[0]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        if torch.isnan(loss):
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot_loss += loss.item() * y.numel()
        tot_tok += y.numel()
        if (i+1) % 200 == 0 or (i+1) == n:
            elapsed = time.time() - t0
            eta = elapsed / (i+1) * (n - i - 1)
            avg = tot_loss / tot_tok
            print(f"\r  [{name}] Ep {ep:02d}/{total} | {i+1}/{n} ({100*(i+1)/n:.0f}%) "
                  f"| loss={avg:.4f} | {elapsed:.0f}s elapsed, ~{eta:.0f}s left",
                  end='', flush=True)
    print()
    return tot_loss / max(tot_tok, 1)


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    tot_loss = 0.0; tot_tok = 0
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        logits = model(x)[0]
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        if not torch.isnan(loss):
            tot_loss += loss.item() * y.numel()
            tot_tok += y.numel()
    return tot_loss / max(tot_tok, 1)


@torch.no_grad()
def get_acts(model, loader, dev, max_b=50):
    model.eval()
    logic_all, mem_all = [], []
    for i, (x, _) in enumerate(loader):
        if i >= max_b: break
        x = x.to(dev)
        _, lo, me = model(x)
        if me is not None: mem_all.append(me[:,-1,:].cpu().numpy())
        if lo is not None: logic_all.append(lo[:,-1,:].cpu().numpy())
    return (np.concatenate(logic_all) if logic_all else None,
            np.concatenate(mem_all) if mem_all else None)


# ====================== 主实验 ======================
def main():
    print("="*70)
    print("  FiberNet 几何涌现实验 v3 (Shakespeare Char-Level)")
    print("="*70)

    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CFG = dict(seq=128, bs=64, ep=30, lr=3e-4, dm=256, nl=8, nh=8, ff=1024)
    print(f"\n[Config] device={DEV}")
    for k, v in CFG.items():
        print(f"  {k}={v}", end='')
    print()

    # 数据
    text = download_shakespeare()
    print(f"[Data] {len(text):,} chars")
    sp = int(len(text) * 0.9)
    train_ds = CharDataset(text[:sp], CFG['seq'])
    val_ds = CharDataset(text[sp:], CFG['seq'])
    VS = train_ds.vocab_size
    train_ld = DataLoader(train_ds, CFG['bs'], shuffle=True, num_workers=0, drop_last=True)
    val_ld = DataLoader(val_ds, CFG['bs'], shuffle=False, num_workers=0, drop_last=True)
    print(f"[Data] vocab={VS}, train={len(train_ds):,}, val={len(val_ds):,}, "
          f"batches/ep={len(train_ld)}")

    # 模型
    fnet = ScaledFiberNet(VS, CFG['dm'], CFG['nl'], CFG['nh'], CFG['ff']).to(DEV)
    tnet = ScaledTransformer(VS, CFG['dm'], CFG['nl'], CFG['nh'], CFG['ff']).to(DEV)
    fp = sum(p.numel() for p in fnet.parameters())
    tp = sum(p.numel() for p in tnet.parameters())
    print(f"[Model] FiberNet: {fp:,} params")
    print(f"[Model] Transformer: {tp:,} params")
    print(f"[Model] ratio: {fp/tp:.2f}x")

    # 冒烟测试
    print("\n[*] 冒烟测试...")
    with torch.no_grad():
        tx = torch.randint(0, VS, (2, CFG['seq'])).to(DEV)
        of = fnet(tx)[0]; ot = tnet(tx)[0]
        print(f"  FiberNet: mean={of.mean():.4f} nan={torch.isnan(of).any()}")
        print(f"  Transformer: mean={ot.mean():.4f} nan={torch.isnan(ot).any()}")
        if torch.isnan(of).any():
            print("  [!!!] FiberNet 前向传播就有 NaN，需要调查！")
            return

    # 训练
    results = {}
    for name, model in [("FiberNet", fnet), ("Transformer", tnet)]:
        print(f"\n{'='*60}")
        print(f"  训练 {name} (共 {CFG['ep']} epochs)")
        print(f"{'='*60}")
        opt = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=0.01)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, CFG['ep'])
        hist = {'train': [], 'val': []}
        t_all = time.time()

        for ep in range(1, CFG['ep']+1):
            t0 = time.time()
            tr = train_one_epoch(model, train_ld, opt, DEV, name, ep, CFG['ep'])
            vl = evaluate(model, val_ld, DEV)
            sch.step()
            hist['train'].append(tr)
            hist['val'].append(vl)
            dt = time.time() - t0
            ela = time.time() - t_all
            eta = ela / ep * (CFG['ep'] - ep)
            bpc = vl / math.log(2)
            print(f"  >>> [{name}] Ep {ep:02d} | train={tr:.4f} val={vl:.4f} "
                  f"BPC={bpc:.3f} | {dt:.0f}s | ETA {eta/60:.0f}min")

        results[name] = hist

    # 涌现检测
    print(f"\n{'='*60}")
    print("  几何涌现检测")
    print(f"{'='*60}")
    det = EmergenceDetector()
    emg = {}
    for name, model in [("FiberNet", fnet), ("Transformer", tnet)]:
        print(f"\n--- {name} ---")
        lo, me = get_acts(model, val_ld, DEV)
        acts = me
        if acts is None:
            print("  无激活数据"); continue
        print(f"  shape: {acts.shape}")
        tda = det.persistent_homology(acts)
        dim = det.intrinsic_dim(acts)
        spc = det.spectral(acts)
        emg[name] = {'tda': tda, 'dim': dim, 'spec': spc}
        if tda:
            print(f"  [TDA] b0={tda['b0']} b1={tda['b1']} sig={tda['b1_sig']} pers={tda['b1_pers']:.4f}")
        print(f"  [DIM] {dim['mean']:.1f} +/- {dim['std']:.1f} (ambient={dim['ambient']})")
        print(f"  [FFT] peaks={spc['peaks']} periodic={spc['periodic']}")

    # 总结
    fb_vl = results['FiberNet']['val'][-1]
    tr_vl = results['Transformer']['val'][-1]
    print(f"\n{'='*70}")
    print(f"  总结")
    print(f"{'='*70}")
    print(f"  FiberNet    val={fb_vl:.4f} BPC={fb_vl/math.log(2):.3f}")
    print(f"  Transformer val={tr_vl:.4f} BPC={tr_vl/math.log(2):.3f}")
    print(f"  gap={fb_vl-tr_vl:+.4f} {'FiberNet wins' if fb_vl<tr_vl else 'Transformer wins'}")

    if emg.get('FiberNet') and emg.get('Transformer'):
        fd = emg['FiberNet']; td = emg['Transformer']
        fb1 = fd['tda']['b1_sig'] if fd['tda'] else 0
        tb1 = td['tda']['b1_sig'] if td['tda'] else 0
        fid = fd['dim']['mean']; tid = td['dim']['mean']
        print(f"\n  指标              FiberNet    Transformer  判定")
        print(f"  {'-'*55}")
        print(f"  b1_significant    {fb1:<12}{tb1:<13}{'EMERGED' if fb1>tb1 else 'NOT'}")
        print(f"  intrinsic_dim     {fid:<12.1f}{tid:<13.1f}{'COMPRESSED' if fid<tid else 'NOT'}")
        fp2 = fd['spec']['periodic']; tp2 = td['spec']['periodic']
        print(f"  periodic          {str(fp2):<12}{str(tp2):<13}{'DETECTED' if fp2 and not tp2 else 'PENDING'}")

    # 保存
    os.makedirs(DATA_DIR, exist_ok=True)
    rpt = {
        'config': CFG, 'params': {'fiber': fp, 'trans': tp},
        'training': {'fiber_val': fb_vl, 'trans_val': tr_vl},
        'emergence': {n: {k: v for k, v in d.items() if v is not None}
                      for n, d in emg.items()}
    }
    rp = os.path.join(DATA_DIR, "emergence_report.json")
    json.dump(rpt, open(rp, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f"\n[+] 报告: {rp}")
    print("="*70)


if __name__ == "__main__":
    main()
