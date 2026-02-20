import os, sys, json, torch, numpy as np
from torch.utils.data import DataLoader

# Setup
DATA_DIR = r"d:\develop\TransformerLens-main\tempdata"
sys.path.append(r"d:\develop\TransformerLens-main\scripts")
from fibernet_wikitext_scaling import ScaledFiberNet, WikiDataset

def robust_calc_id(acts):
    from sklearn.neighbors import NearestNeighbors
    if len(acts) > 2000: acts = acts[np.random.choice(len(acts), 2000, False)]
    acts = (acts - acts.mean(0))/(acts.std(0)+1e-8)
    nn_ = NearestNeighbors(n_neighbors=11).fit(acts)
    d, _ = nn_.kneighbors(acts)
    d = np.maximum(d[:, 1:], 1e-10)
    ratio = d[:, -1:] / d[:, :-1]
    log_ratio = np.log(np.maximum(ratio, 1.0001))
    id_val = 10 / np.mean(np.sum(log_ratio, axis=1))
    return float(id_val)

def get_id(ep):
    ckpt = os.path.join(DATA_DIR, f"fiber_20m_ep{ep}.pth")
    if not os.path.exists(ckpt): return None
    
    DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ScaledFiberNet().to(DEV)
    model.load_state_dict(torch.load(ckpt, map_location=DEV))
    model.eval()
    
    ds = WikiDataset(os.path.join(DATA_DIR, "wiki_v4_3.npy"))
    ld = DataLoader(ds, 64, shuffle=False)
    
    A = []
    with torch.no_grad():
        for i, (x, _) in enumerate(ld):
            if i >= 20: break
            # Index 2 = Memory
            A.append(model(x.to(DEV))[2][:,-1,:].cpu().numpy())
    acts = np.concatenate(A)
    
    # Calculate ID
    from sklearn.neighbors import NearestNeighbors
    if len(acts) > 2000: acts = acts[np.random.choice(len(acts), 2000, False)]
    acts = (acts - acts.mean(0))/(acts.std(0)+1e-8)
    nn_ = NearestNeighbors(n_neighbors=11).fit(acts)
    d, _ = nn_.kneighbors(acts)
    d = np.maximum(d[:, 1:], 1e-10)
    ratio = d[:, -1:] / d[:, :-1]
    log_ratio = np.log(np.maximum(ratio, 1.0001))
    id_val = 10 / np.mean(np.sum(log_ratio, axis=1))
    return float(id_val)

def main():
    res_path = os.path.join(DATA_DIR, "phase3_wiki_results.json")
    data = json.load(open(res_path))
    
    updates = {}
    for ep in range(7, 12): # 7..11
        print(f"Auditing Ep {ep}...")
        try:
            val = get_id(ep)
            if val:
                print(f"  ID: {val:.4f}")
                updates[ep] = val
        except Exception as e:
            print(f"  Error: {e}")
            
    # Apply updates
    for d in data:
        if d['ep'] in updates:
            d['id'] = updates[d['ep']]
            
    json.dump(data, open(res_path, 'w'))
    print("Updated JSON.")

if __name__ == "__main__":
    main()
