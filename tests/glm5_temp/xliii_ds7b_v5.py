# -*- coding: utf-8 -*-
"""Phase XLIII DeepSeek7B v5 - 不释放模型, 逐层分析, 写文件日志"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys, gc, time, json
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

LOG_FILE = Path("tests/glm5_temp/xliii_ds7b_v5_log.txt")
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
L.log("=== Phase XLIII DeepSeek7B v5 (keep model, per-layer) ===")

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

# 1. Kill ollama to free VRAM
L.log("[0] Killing Ollama...")
os.system("taskkill /F /IM ollama.exe 2>nul")
time.sleep(2)

# 2. Load model
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

# 3. Collect hidden states
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

# 4. Per-layer analysis (model stays on GPU, extract weights per layer)
L.log("[3] Per-layer analysis (P257+P258+P259)...")

p257_results = {"per_layer": []}
p258_results = {"per_layer": []}
p259_results = {"per_layer": []}

for li in range(n_layers):
    t_layer = time.time()
    mlp = layers[li].mlp
    
    # Extract weights for this layer only, move to CPU for analysis
    W_gate = mlp.gate_proj.weight.detach().float().cpu().T   # [d_model, d_mlp]
    W_in = mlp.up_proj.weight.detach().float().cpu().T       # [d_model, d_mlp]
    W_out = mlp.down_proj.weight.detach().float().cpu().T    # [d_mlp, d_model]
    
    # === P257: Variance proxy + direction alignment ===
    # Use row-norm concentration as proxy for SVD variance concentration
    row_norms = torch.linalg.norm(W_out, dim=1)
    total_norm_sq = (row_norms**2).sum()
    top50_sq = torch.topk(row_norms**2, min(50, len(row_norms))).values.sum()
    cumvar_50_proxy = float(top50_sq / total_norm_sq) if total_norm_sq > 0 else 0
    top200_sq = torch.topk(row_norms**2, min(200, len(row_norms))).values.sum()
    cumvar_200_proxy = float(top200_sq / total_norm_sq) if total_norm_sq > 0 else 0
    top500_sq = torch.topk(row_norms**2, min(500, len(row_norms))).values.sum()
    cumvar_500_proxy = float(top500_sq / total_norm_sq) if total_norm_sq > 0 else 0
    
    # Fruit direction alignment
    fruit_dir = fruit_mean[li+1] - fruit_mean[li]
    fruit_norm = torch.linalg.norm(fruit_dir)
    fruit_Wout_max = 0
    fruit_Wout_top10 = 0
    if fruit_norm > 1e-8:
        fruit_dir_n = fruit_dir / fruit_norm
        cos_with_Wout = F.cosine_similarity(W_out, fruit_dir_n.unsqueeze(0), dim=1)
        fruit_Wout_max = float(cos_with_Wout.max())
        fruit_Wout_top10 = float(torch.topk(cos_with_Wout, 10).values.mean())
    
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
    del cos_gate_in
    
    p257_results["per_layer"].append({
        "layer": li,
        "cumvar_50_proxy": round(cumvar_50_proxy, 4),
        "cumvar_200_proxy": round(cumvar_200_proxy, 4),
        "cumvar_500_proxy": round(cumvar_500_proxy, 4),
        "fruit_dir_Wout_cos_max": round(fruit_Wout_max, 4),
        "fruit_dir_Wout_top10_mean": round(fruit_Wout_top10, 4),
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
    
    # === P259: Apple-specific neurons ===
    if apple_hs is not None:
        h = apple_hs[li].unsqueeze(0)
        gate_a = F.gelu(h @ W_gate)
        post_a = h @ W_in
        apple_act = (gate_a * post_a)[0]
        del gate_a, post_a
        
        # Compute fruit mean activation one by one
        fruit_act_sum = torch.zeros(d_mlp)
        n_fruit = 0
        for fw in fruit_words:
            if fw in single_hs:
                fh = single_hs[fw][li].unsqueeze(0)
                fg = F.gelu(fh @ W_gate)
                fp = fh @ W_in
                fruit_act_sum += (fg * fp)[0]
                del fg, fp
                n_fruit += 1
        fruit_act_mean = fruit_act_sum / max(n_fruit, 1)
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
    gc.collect()
    
    elapsed = time.time() - t_layer
    L.log(f"  L{li}: cumvar50={cumvar_50_proxy:.4f} fruit_cos={fruit_Wout_max:.4f} G_ffn={cos_G_ffn:.4f} ({elapsed:.1f}s)")

L.log(f"  P257: {len(p257_results['per_layer'])} layers")
L.log(f"  P258: {len(p258_results['per_layer'])} layers")
L.log(f"  P259: {len(p259_results['per_layer'])} layers")

# 5. Release model
L.log("[4] Releasing model...")
del mdl, tok, layers, single_hs, combo_hs, fruit_mean
gc.collect()
torch.cuda.empty_cache()
L.log("  Released")

# 6. Save results
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
