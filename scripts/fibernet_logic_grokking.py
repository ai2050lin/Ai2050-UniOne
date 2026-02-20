
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
from sklearn.neighbors import NearestNeighbors

# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata", "logic_mix")
RES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tempdata")
os.makedirs(RES_DIR, exist_ok=True)

# Medium Sized Model for Grokking (Overparameterized relative to task)
D_MODEL = 256
N_LAYERS = 6
N_HEADS = 8
D_FF = 1024
SEQ_LEN = 128
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 100 # Long training for Grokking
GRAD_ACCUM = 1
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tokenizer ---
class LogicTokenizer:
    def __init__(self):
        # Build vocab (Must match generation script)
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+->=?.(): "
        self.stoi = {c:i+1 for i,c in enumerate(chars)}
        self.itos = {i+1:c for i,c in enumerate(chars)}
        self.vocab_size = len(chars) + 2 # +1 for padding/unk
        self.pad_token = 0
        
    def encode(self, text):
        return [self.stoi.get(c, 0) for c in text]
    
    def decode(self, ids):
        return "".join([self.itos.get(i, '') for i in ids])

# --- Dataset ---
class LogicDataset(Dataset):
    def __init__(self, path, seq_len):
        self.data = torch.load(path)
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.data) - self.seq_len
    
    def __getitem__(self, idx):
        # Sliding window
        chunk = self.data[idx:idx+self.seq_len+1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y

# --- Model ---
class FiberNetLogic(nn.Module):
    def __init__(self, vocab_size):
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
        
        logits = self.out_head(resid)
        return logits, logic, mem

# --- Geometry Monitor ---
class EmergenceMonitor:
    @staticmethod
    def calc_id(acts):
        # Hill Estimator for Intrinsic Dimension
        # acts: [N, D]
        # Remove try-except to debug, force n_jobs=1
        nn = NearestNeighbors(n_neighbors=3, n_jobs=1).fit(acts)
        d, _ = nn.kneighbors(acts)
        d = d[:, 1:] # Drop self
        mu = d[:, 1] / d[:, 0]
        est = 1 / np.log(mu) # Hill estimator
        # Trim extremes
        est = est[np.isfinite(est)]
        return float(np.median(est)) # Median is more robust than mean

# --- Training Loop ---
def train():
    tokenizer = LogicTokenizer()
    print(f"[*] Loading LogicMix from {DATA_DIR}...")
    
    train_ds = LogicDataset(os.path.join(DATA_DIR, "train.pt"), SEQ_LEN)
    val_ds = LogicDataset(os.path.join(DATA_DIR, "val.pt"), SEQ_LEN)
    
    # Subsample Val for speed
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    
    model = FiberNetLogic(tokenizer.vocab_size).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1.0) # High weight decay for Grokking
    scaler = torch.amp.GradScaler('cuda')
    
    print(f"[*] Starting Logic Grokking (Epochs: {EPOCHS}, Model: D{D_MODEL}L{N_LAYERS})")
    print(f"    Vocab: {tokenizer.vocab_size}")
    
    results = []
    
    for ep in range(1, EPOCHS+1):
        model.train()
        total_loss = 0
        
        # Training
        for i, (x, y) in enumerate(train_dl):
            x, y = x.to(DEV), y.to(DEV)
            
            with torch.amp.autocast('cuda'):
                logits, _, _ = model(x)
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            
            total_loss += loss.item()
            
            if i % 100 == 0:
                print(f"Ep {ep} | Batch {i} | Loss: {loss.item():.4f}", end='\r', flush=True)

            # Frequent Analysis (Debug frequency: 200)
            if (i+1) % 200 == 0:
                # Validation & Geometry
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    # 1. Calc Val Loss
                    for j, (vx, vy) in enumerate(val_dl):
                        if j > 50: break # Limit val
                        vx, vy = vx.to(DEV), vy.to(DEV)
                        vlogits, _, vmem = model(vx)
                        vloss = F.cross_entropy(vlogits.reshape(-1, vlogits.size(-1)), vy.reshape(-1))
                        val_loss += vloss.item()
                    
                    avg_val_loss = val_loss / 51
                    
                    # 2. Calc Geometry
                    acts = []
                    for j, (vx, _) in enumerate(val_dl):
                        if j > 5: break
                        _, _, mem = model(vx.to(DEV))
                        acts.append(mem.cpu().numpy())
                    acts = np.concatenate(acts, axis=0)
                    acts_sub = acts.reshape(-1, D_MODEL)
                    if acts_sub.shape[0] > 5000:
                        indices = np.random.choice(acts_sub.shape[0], 5000, replace=False)
                        acts_sub = acts_sub[indices]
                    
                    id_val = EmergenceMonitor.calc_id(acts_sub)
                    
                    print(f"\n[*] Ep {ep} Batch {i+1} | Train: {loss.item():.4f} | Val: {avg_val_loss:.4f} | ID: {id_val:.2f}", flush=True)
                    
                    # 3. Log
                    results.append({
                        "ep": ep,
                        "batch": i+1,
                        "train_loss": loss.item(),
                        "val_loss": avg_val_loss,
                        "id": id_val
                    })
                    with open(os.path.join(RES_DIR, "phase5_logic_results.json"), "w") as f:
                        json.dump(results, f)
                model.train() # Resume training

        avg_train_loss = total_loss / len(train_dl)
        
        # Validation & Geometry
        model.eval()
        val_loss = 0
        with torch.no_grad():
            # 1. Calc Val Loss
            for j, (vx, vy) in enumerate(val_dl):
                if j > 50: break # Limit val
                vx, vy = vx.to(DEV), vy.to(DEV)
                vlogits, _, vmem = model(vx)
                vloss = F.cross_entropy(vlogits.reshape(-1, vlogits.size(-1)), vy.reshape(-1))
                val_loss += vloss.item()
            
            avg_val_loss = val_loss / 51
            
            # 2. Calc Geometry
            acts = []
            for j, (vx, _) in enumerate(val_dl):
                if j > 5: break
                _, _, mem = model(vx.to(DEV))
                acts.append(mem.cpu().numpy())
            acts = np.concatenate(acts, axis=0) # [B*5, T, D]
            # Subsample points for ID calc speed
            acts_sub = acts.reshape(-1, D_MODEL)
            if acts_sub.shape[0] > 5000:
                indices = np.random.choice(acts_sub.shape[0], 5000, replace=False)
                acts_sub = acts_sub[indices]
            
            id_val = EmergenceMonitor.calc_id(acts_sub)
            
            print(f"\n[*] Ep {ep} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | ID: {id_val:.2f}")
            
            # 3. Log
            results.append({
                "ep": ep,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "id": id_val
            })
            with open(os.path.join(RES_DIR, "phase5_logic_results.json"), "w") as f:
                json.dump(results, f)
                
        # Simple checkpoint
        if ep % 10 == 0:
            torch.save(model.state_dict(), os.path.join(RES_DIR, f"fiber_logic_ep{ep}.pth"))

if __name__ == "__main__":
    train()
