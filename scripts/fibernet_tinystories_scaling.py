
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json

# Configuration
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata", "tinystories")
RES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata")
BATCH_SIZE = 16 
GRAD_ACCUM = 4 # Eff Batch = 64
SEQ_LEN = 256
D_MODEL = 768
N_LAYERS = 12
N_HEADS = 12 
D_FF = 3072
EPOCHS = 20
LR = 3e-4
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class TinyStoriesDataset(Dataset):
    def __init__(self, path, seq_len):
        print(f"[*] Loading dataset from {path}...")
        self.data = torch.load(path).view(-1)
        self.seq_len = seq_len
        print(f"[*] Loaded {self.data.numel()} tokens.")
    def __len__(self):
        return (self.data.numel() // self.seq_len) - 1
    def __getitem__(self, i):
        start = i * self.seq_len
        x = self.data[start : start + self.seq_len]
        y = self.data[start+1 : start+1+self.seq_len]
        return x, y

class ScaledFiberNetXL(nn.Module):
    def __init__(self, vocab_size=50257):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=D_MODEL, nhead=N_HEADS, dim_feedforward=D_FF, batch_first=True, norm_first=True)
            for _ in range(N_LAYERS)
        ])
        
        # Split Stream: Index 0=Residual, 1=Logic, 2=Memory
        self.stream_proj = nn.Linear(D_MODEL, 3 * D_MODEL)
        self.out_head = nn.Linear(D_MODEL, vocab_size)
    
    def forward(self, x):
        b, t = x.shape
        pos = torch.arange(t, device=x.device)
        h = self.embed(x) + self.pos_emb(pos)
        
        for layer in self.layers:
            h = layer(h)
        
        # Split features for monitoring
        streams = self.stream_proj(h)
        resid, logic, mem = streams.chunk(3, dim=-1)
        
        # We only use residual for output prediction in standard transformer, 
        # but we track memory stream for geometric emergence
        logits = self.out_head(resid)
        return logits, logic, mem

class EmergenceMonitor:
    @staticmethod
    def calc_id(acts):
        # Flatten [B, T, D] -> [B*T, D]
        if len(acts.shape) == 3: acts = acts.reshape(-1, acts.shape[-1])
        
        # Robust ID calculation (TwoNN method)
        from sklearn.neighbors import NearestNeighbors
        # Subsample if too large
        if len(acts) > 5000:
            acts = acts[np.random.choice(len(acts), 5000, replace=False)]
            
        try:
            # Normalize
            acts = (acts - acts.mean(0)) / (acts.std(0) + 1e-9)
            
            nn = NearestNeighbors(n_neighbors=3).fit(acts)
            d, _ = nn.kneighbors(acts)
            d = d[:, 1:] # Drop self
            mu = d[:, 1] / d[:, 0]
            est = 1 / np.log(mu) # Hill estimator simplified (Corrected sign)
            # Trim extremes
            est = est[np.isfinite(est)]
            return float(np.median(est)) # Median is more robust than mean
        except Exception as e:
            print(f"ID Calc Error: {e}")
            return 0.0

def train():
    train_path = os.path.join(DATA_DIR, "train.pt")
    if not os.path.exists(train_path):
        print(f"Data not found at {train_path}. Please run tinystories_preproc.py first.")
        return

    train_ds = TinyStoriesDataset(train_path, SEQ_LEN)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    
    model = ScaledFiberNetXL().to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler('cuda') # Mixed Precision
    
    print(f"[*] Starting TinyStories Scaling (100M Params, 20 Epochs)")
    print(f"    Data: {len(train_ds)} samples")
    
    results = []
    
    for ep in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        opt.zero_grad()
        
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(DEV), y.to(DEV)
            
            with torch.amp.autocast('cuda'):
                logits, _, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = loss / GRAD_ACCUM
            
            scaler.scale(loss).backward()
            
            if (i+1) % GRAD_ACCUM == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
            
            total_loss += loss.item() * GRAD_ACCUM
            
            if i % 100 == 0:
                print(f"Ep {ep} | Batch {i}/{len(train_dl)} | Loss: {loss.item()*GRAD_ACCUM:.4f}")
                
            # Analysis every 500 batches
            if (i+1) % 500 == 0:
                print(f"[*] Analyzing Geometry (Ep {ep} Batch {i+1})...")
                model.eval()
                with torch.no_grad():
                    acts = []
                    for j, (xx, _) in enumerate(train_dl):
                        if j > 10: break
                        _, _, mem = model(xx.to(DEV))
                        acts.append(mem.cpu().numpy())
                    
                    acts = np.concatenate(acts, axis=0)
                    id_val = EmergenceMonitor.calc_id(acts)
                    print(f"    >> ID: {id_val:.2f} | Loss: {loss.item()*GRAD_ACCUM:.4f}")
                    
                    results.append({"ep": ep, "batch": i+1, "loss": loss.item()*GRAD_ACCUM, "id": id_val})
                    with open(os.path.join(RES_DIR, "phase4_tinystories_results.json"), "w") as f:
                        json.dump(results, f)
                model.train() # Resume training
        
        # Save Checkpoint at end of epoch
        torch.save(model.state_dict(), os.path.join(RES_DIR, f"fiber_xl_ep{ep}.pth"))

if __name__ == "__main__":
    train()
