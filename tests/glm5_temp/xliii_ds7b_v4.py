# -*- coding: utf-8 -*-
"""Phase XLIII DeepSeek7B v4 - 极简版, 减少内存使用, 逐层释放"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys, gc, time, json
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

LOG_FILE = Path("tests/glm5_temp/xliii_ds7b_v4_log.txt")
OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

class Logger:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w', encoding='utf-8', buffering=1)
    def log(self, msg):
        self.f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        self.f.flush()
    def close(self): self.f.close()

L = Logger(LOG_FILE)
L.log("=== Phase XLIII DeepSeek7B v4 (minimal memory) ===")

MP = r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"

STIMULI = {
    "fruit_family": ["apple","banana","pear","orange","grape","mango","strawberry","watermelon","cherry","peach","lemon","lime"],
    "animal_family": ["cat","dog","rabbit","horse","lion","eagle","elephant","dolphin","tiger","bear","fox","wolf"],
    "vehicle_family": ["car","bus","train","plane","boat","bicycle","truck","helicopter","motorcycle","ship","subway","taxi"],
    "furniture_family": ["chair","table","desk","sofa","bed","cabinet","shelf","bench","stool","dresser","couch","armchair"],
    "tool_family": ["hammer","wrench","screwdriver","saw","drill","pliers","knife","scissors","shovel","rake","axe","chisel"],
    "color_attrs": ["red","green","yellow","orange","brown","white","blue","black","pink","purple","gray","gold"],
    "taste_attrs": ["sweet","sour","bitter","salty","crisp","soft","spicy","fresh","tart","savory","rich","mild"],
    "size_attrs": ["big","small","tall","short","long","wide","thin","thick","heavy","light","huge","tiny"],
    "fruit_color_combos": ["red apple","green apple","yellow banana","orange orange","green pear","red grape","yellow mango","red cherry","pink peach","yellow lemon"],
    "fruit_taste_combos": ["sweet apple","sour apple","sweet banana","sour orange","sweet pear","bitter grape","sweet mango","sweet cherry","tart lemon","sweet peach"],
    "animal_color_combos": ["brown cat","white dog","brown rabbit","black horse","golden eagle","white cat","black dog","orange tiger","brown bear","red fox"],
}

TEMPLATES = ["The {word} is", "A {word} can be", "This {word} has", "I saw a {word}", "The {word} was", "My {word} is", "That {word} looks"]

t_total = time.time()

# 1. Load model
L.log("[1] Loading model...")
tok = AutoTokenizer.from_pretrained(MP, trust_remote_code=True, local_files_only=True, use_fast=False)
if tok.pad_token is None: tok.pad_token = tok.eos_token
mdl = AutoModelForCausalLM.from_pretrained(
    MP, dtype=torch.bfloat16, trust_remote_code=True,
    local_files_only=True, low_cpu_mem_usage=True,
    attn_implementation="eager", device_map="cpu"
)
mdl = mdl.to("cuda")
mdl.eval()
device = next(mdl.parameters()).device
n_layers = mdl.config.num_hidden_layers
d_model = mdl.config.hidden_size
d_mlp = None
layers = mdl.model.layers
for name, param in layers[0].mlp.named_parameters():
    if 'weight' in name and param.shape[0] > d_model:
        d_mlp = param.shape[0]; break
L.log(f"  OK: {n_layers}L, d={d_model}, d_mlp={d_mlp}")

# 2. Collect hidden states
L.log("[2] Collecting hidden states...")
single_words = []
for key in ["fruit_family","animal_family","vehicle_family","furniture_family","tool_family","color_attrs","taste_attrs","size_attrs"]:
    single_words.extend(STIMULI[key])

single_hs = {}
for wi, word in enumerate(single_words):
    word_mean = None
    for template in TEMPLATES:
        prompt = template.replace("{word}", word)
        inp = tok(prompt, return_tensors="pt", padding=False, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True, use_cache=False)
        hs = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
        if word_mean is None: word_mean = hs
        else: word_mean += hs
        del out, inp
    single_hs[word] = word_mean / len(TEMPLATES)
    if (wi+1) % 20 == 0: L.log(f"  [{wi+1}/{len(single_words)}]")

combo_keys = ["fruit_color_combos", "fruit_taste_combos", "animal_color_combos"]
combos = []
for key in combo_keys: combos.extend(STIMULI[key])

combo_hs = {}
for ci, combo in enumerate(combos):
    combo_mean = None
    for template in TEMPLATES:
        prompt = template.replace("{word}", combo)
        inp = tok(prompt, return_tensors="pt", padding=False, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True, use_cache=False)
        hs = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
        if combo_mean is None: combo_mean = hs
        else: combo_mean += hs
        del out, inp
    combo_hs[combo] = combo_mean / len(TEMPLATES)

L.log(f"  Data: {len(single_hs)} singles, {len(combo_hs)} combos")

fruit_words = STIMULI["fruit_family"]
fruit_mean = torch.stack([single_hs[w] for w in fruit_words]).mean(0)
apple_hs = single_hs.get("apple")

# 3. Release model first, then do pure CPU analysis from model files
L.log("[3] Releasing model (will reload per-layer for weight analysis)...")
del mdl, tok
gc.collect()
torch.cuda.empty_cache()
L.log("  Model released")

# 4. Per-layer analysis: reload model briefly for each layer, then release
L.log("[4] Per-layer analysis (reload model per 5 layers)...")

p257_results = {"per_layer": []}
p258_results = {"per_layer": []}
p259_results = {"per_layer": []}

# Process in batches of 5 layers to balance memory
BATCH_SIZE = 5
for batch_start in range(0, n_layers, BATCH_SIZE):
    batch_end = min(batch_start + BATCH_SIZE, n_layers)
    L.log(f"  Batch L{batch_start}-L{batch_end-1}...")
    
    # Reload model
    tok2 = AutoTokenizer.from_pretrained(MP, trust_remote_code=True, local_files_only=True, use_fast=False)
    mdl2 = AutoModelForCausalLM.from_pretrained(
        MP, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, low_cpu_mem_usage=True,
        attn_implementation="eager", device_map="cpu"
    )
    mdl2 = mdl2.to("cuda")
    mdl2.eval()
    layers2 = mdl2.model.layers
    
    for li in range(batch_start, batch_end):
        t_layer = time.time()
        mlp = layers2[li].mlp
        # Extract weights as float32 on CPU
        W_gate = mlp.gate_proj.weight.detach().float().cpu().T  # [d_model, d_mlp]
        W_in = mlp.up_proj.weight.detach().float().cpu().T
        W_out = mlp.down_proj.weight.detach().float().cpu().T   # [d_mlp, d_model]
        
        # === P257: Simplified SVD (skip full SVD, use variance estimation) ===
        # Use Frobenius norm ratio as proxy for variance concentration
        W_out_norm = torch.linalg.norm(W_out, 'fro')
        n_rows = W_out.shape[0]
        # Sample random rows to estimate variance concentration
        rng = torch.Generator().manual_seed(42)
        n_sample = min(500, n_rows)
        sample_idx = torch.randperm(n_rows, generator=rng)[:n_sample]
        W_out_sample = W_out[sample_idx]
        row_norms = torch.linalg.norm(W_out_sample, dim=1)
        top50_norm = torch.topk(row_norms, min(50, n_sample)).values.sum()
        cumvar_50_proxy = float((top50_norm / row_norms.sum())**2) if row_norms.sum() > 0 else 0
        
        # Fruit direction alignment
        fruit_dir = fruit_mean[li+1] - fruit_mean[li]
        fruit_norm = torch.linalg.norm(fruit_dir)
        fruit_Wout_max = 0
        if fruit_norm > 1e-8:
            fruit_dir_n = fruit_dir / fruit_norm
            cos_with_Wout = F.cosine_similarity(W_out, fruit_dir_n.unsqueeze(0), dim=1)
            fruit_Wout_max = float(cos_with_Wout.max())
        
        # G-term alignment
        G = apple_hs[li+1] - fruit_mean[li+1] if apple_hs is not None else torch.zeros(d_model)
        G_norm = torch.linalg.norm(G)
        G_Wout_max = 0
        if G_norm > 1e-8:
            cos_G_Wout = F.cosine_similarity(W_out, (G/G_norm).unsqueeze(0), dim=1)
            G_Wout_max = float(cos_G_Wout.max())
        
        # Gate-In overlap (sampled)
        n_g = min(500, W_gate.shape[1])
        cos_gate_in = F.cosine_similarity(W_gate[:,:n_g].T.unsqueeze(1), W_in[:,:n_g].T.unsqueeze(0), dim=2)
        gate_in_mean = float(cos_gate_in.mean())
        
        p257_results["per_layer"].append({
            "layer": li,
            "cumvar_50_proxy": round(cumvar_50_proxy, 4),
            "fruit_dir_Wout_cos_max": round(fruit_Wout_max, 4),
            "ffn_G_cos_max": round(G_Wout_max, 4),
            "gate_in_overlap": round(gate_in_mean, 4),
        })
        
        # === P258: FFN causal chain ===
        cos_G_ffn = 0
        ffn_norm_val = 0
        top10_info = {"indices": [], "values": []}
        if apple_hs is not None:
            h = apple_hs[li].unsqueeze(0)
            gate = F.gelu(h @ W_gate)
            post_in = h @ W_in
            act = gate * post_in
            ffn_out = act @ W_out
            
            ffn_norm_val = float(torch.linalg.norm(ffn_out[0]))
            if G_norm > 1e-8 and ffn_norm_val > 1e-8:
                cos_G_ffn = float(F.cosine_similarity(G.unsqueeze(0), ffn_out, dim=1)[0])
            
            top_neurons = torch.topk(act[0].abs(), 10)
            top10_info = {
                "indices": top_neurons.indices.tolist(),
                "values": [round(float(v), 4) for v in top_neurons.values],
            }
            del gate, post_in, act, ffn_out
        
        p258_results["per_layer"].append({
            "layer": li,
            "ffn_out_norm": round(ffn_norm_val, 2),
            "G_norm": round(float(G_norm), 2),
            "cos_G_ffn": round(cos_G_ffn, 4),
            "top10": top10_info,
        })
        
        # === P259: Apple-specific neurons (simplified - only top5) ===
        if apple_hs is not None:
            h = apple_hs[li].unsqueeze(0)
            gate_a = F.gelu(h @ W_gate)
            post_a = h @ W_in
            apple_act = (gate_a * post_a)[0]
            del gate_a, post_a
            
            # Compute fruit mean activation iteratively to save memory
            fruit_act_sum = torch.zeros(d_mlp)
            for fw in fruit_words:
                if fw in single_hs:
                    fh = single_hs[fw][li].unsqueeze(0)
                    fg = F.gelu(fh @ W_gate)
                    fp = fh @ W_in
                    fruit_act_sum += (fg * fp)[0]
                    del fg, fp
            fruit_act_mean = fruit_act_sum / len(fruit_words)
            del fruit_act_sum
            
            apple_excess = apple_act - fruit_act_mean
            del apple_act, fruit_act_mean
            top_excess = torch.topk(apple_excess.abs(), 5)
            del apple_excess
            
            p259_results["per_layer"].append({
                "layer": li,
                "top5_neurons": top_excess.indices.tolist(),
                "top5_values": [round(float(v), 4) for v in top_excess.values],
            })
            del top_excess
        
        del W_gate, W_in, W_out
        elapsed = time.time() - t_layer
        L.log(f"    L{li}: fruit_cos={fruit_Wout_max:.4f} G_cos={G_Wout_max:.4f} G_ffn={cos_G_ffn:.4f} ({elapsed:.1f}s)")
    
    # Release model after batch
    del mdl2, tok2, layers2
    gc.collect()
    torch.cuda.empty_cache()
    L.log(f"  Batch L{batch_start}-L{batch_end-1} done, model released")

L.log(f"  P257: {len(p257_results['per_layer'])} layers")
L.log(f"  P258: {len(p258_results['per_layer'])} layers") 
L.log(f"  P259: {len(p259_results['per_layer'])} layers")

# 5. Save results
L.log("[5] Saving results...")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = OUT_DIR / f"phase_xliii_p257_259_deepseek7b_{ts}.json"

all_results = {
    "model": "deepseek7b",
    "n_layers": n_layers, "d_model": d_model, "d_mlp": d_mlp,
    "p257": p257_results, "p258": p258_results, "p259": p259_results,
    "elapsed_seconds": round(time.time() - t_total, 1),
}

def sanitize(obj):
    if isinstance(obj, dict): return {k: sanitize(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [sanitize(v) for v in obj]
    elif isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)): return float(obj)
    elif isinstance(obj, np.ndarray): return obj.tolist()
    elif isinstance(obj, torch.Tensor): return obj.tolist()
    elif isinstance(obj, tuple): return list(obj)
    return obj

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(sanitize(all_results), f, ensure_ascii=False, indent=2)

L.log(f"  Saved: {out_path}")
L.log(f"  Total: {round(time.time()-t_total, 1)}s")
L.log("ALL DONE!")
L.close()
