# -*- coding: utf-8 -*-
"""Phase XLIII DeepSeek7B - 全部输出写入文件, 避免stdout超时"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys, gc, time, json
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# 全部输出到文件
LOG_FILE = Path("tests/glm5_temp/xliii_ds7b_log.txt")
OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

class Logger:
    def __init__(self, path):
        self.path = path
        self.f = open(path, 'w', encoding='utf-8', buffering=1)
    def log(self, msg):
        self.f.write(f"{time.strftime('%H:%M:%S')} {msg}\n")
        self.f.flush()
    def close(self):
        self.f.close()

L = Logger(LOG_FILE)
L.log("=== Phase XLIII DeepSeek7B ===")

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

TEMPLATES = [
    "The {word} is", "A {word} can be", "This {word} has",
    "I saw a {word}", "The {word} was", "My {word} is", "That {word} looks"
]

def extract_ffn_weights(mdl):
    weights = {}
    layers = mdl.model.layers
    for li in range(len(layers)):
        mlp = layers[li].mlp
        wd = {}
        if hasattr(mlp, 'gate_proj'):
            wd['W_gate'] = mlp.gate_proj.weight.detach().float().cpu().T
            wd['W_in'] = mlp.up_proj.weight.detach().float().cpu().T
            wd['W_out'] = mlp.down_proj.weight.detach().float().cpu().T
        weights[li] = wd
    return weights

# === Main ===
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
L.log(f"  OK: {n_layers}L, d={d_model}, d_mlp={d_mlp}, GPU={torch.cuda.memory_allocated(0)/1e9:.2f}GB")

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
        hs = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])  # [n_layers+1, d_model]
        if word_mean is None:
            word_mean = hs
        else:
            word_mean += hs
        del out, inp
    single_hs[word] = word_mean / len(TEMPLATES)
    if (wi+1) % 20 == 0:
        L.log(f"  [{wi+1}/{len(single_words)}] {word}")

combo_keys = ["fruit_color_combos", "fruit_taste_combos", "animal_color_combos"]
combos = []
for key in combo_keys:
    combos.extend(STIMULI[key])

combo_hs = {}
for ci, combo in enumerate(combos):
    combo_mean = None
    for template in TEMPLATES:
        prompt = template.replace("{word}", combo)
        inp = tok(prompt, return_tensors="pt", padding=False, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True, use_cache=False)
        hs = torch.stack([h[0, -1].float().cpu() for h in out.hidden_states])
        if combo_mean is None:
            combo_mean = hs
        else:
            combo_mean += hs
        del out, inp
    combo_hs[combo] = combo_mean / len(TEMPLATES)
    if (ci+1) % 10 == 0:
        L.log(f"  [{ci+1}/{len(combos)}] {combo}")

L.log(f"  Data collected: {len(single_hs)} singles, {len(combo_hs)} combos")

# 3. Extract FFN weights
L.log("[3] FFN weights extraction...")
ffn_weights = extract_ffn_weights(mdl)
L.log(f"  OK: {len(ffn_weights)} layers")

# 4. Save intermediate data (release model before analysis)
L.log("[4] Saving data & releasing model...")
data_path = OUT_DIR / f"xliii_ds7b_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
torch.save({"single_hs": single_hs, "combo_hs": combo_hs, "ffn_weights": ffn_weights,
            "n_layers": n_layers, "d_model": d_model, "d_mlp": d_mlp}, data_path)
L.log(f"  Saved: {data_path}")

del mdl, tok
gc.collect()
torch.cuda.empty_cache()
L.log("  Model released")

# 5. P257: FFN weight structure analysis
L.log("[5] P257: FFN weight structure analysis...")
p257 = {"per_layer": []}
fruit_words = STIMULI["fruit_family"]
fruit_mean = torch.stack([single_hs[w] for w in fruit_words]).mean(0)  # [n_layers+1, d_model]

for li in range(n_layers):
    wd = ffn_weights[li]
    if 'W_gate' not in wd:
        continue
    W_gate, W_in, W_out = wd['W_gate'], wd['W_in'], wd['W_out']
    
    # SVD of W_out
    try:
        U, S, Vt = torch.linalg.svd(W_out, full_matrices=False)
        cumvar = (S**2).cumsum(0) / (S**2).sum()
        svd_50 = int((cumvar < 0.5).sum()) + 1
        svd_90 = int((cumvar < 0.9).sum()) + 1
    except:
        svd_50, svd_90 = -1, -1
    
    # Fruit direction alignment with W_out columns
    fruit_dir = fruit_mean[li+1] - fruit_mean[li]  # FFN contribution direction
    fruit_norm = torch.linalg.norm(fruit_dir)
    if fruit_norm > 1e-8:
        fruit_dir_n = fruit_dir / fruit_norm
        cos_with_Wout = F.cosine_similarity(W_out, fruit_dir_n.unsqueeze(0), dim=1)
        fruit_Wout_max = float(cos_with_Wout.max())
        fruit_Wout_top10 = float(torch.topk(cos_with_Wout, 10).values.mean())
    else:
        fruit_Wout_max, fruit_Wout_top10 = 0, 0
    
    # G-term alignment
    apple_hs = single_hs.get("apple")
    if apple_hs is not None:
        G = apple_hs[li+1] - fruit_mean[li+1]
        G_norm = torch.linalg.norm(G)
        if G_norm > 1e-8:
            cos_G_Wout = F.cosine_similarity(W_out, (G/G_norm).unsqueeze(0), dim=1)
            G_Wout_max = float(cos_G_Wout.max())
        else:
            G_Wout_max = 0
    else:
        G_Wout_max = 0
    
    # Gate-In overlap
    cos_gate_in = F.cosine_similarity(W_gate.unsqueeze(1), W_in.unsqueeze(0), dim=2)
    gate_in_mean = float(cos_gate_in.mean())
    
    p257["per_layer"].append({
        "layer": li,
        "svd_50_components": svd_50,
        "svd_90_components": svd_90,
        "fruit_dir_Wout_cos_max": round(fruit_Wout_max, 4),
        "fruit_dir_Wout_top10_mean": round(fruit_Wout_top10, 4),
        "ffn_G_cos_max": round(G_Wout_max, 4),
        "gate_in_overlap": round(gate_in_mean, 4),
    })
    if li % 5 == 0:
        L.log(f"  L{li}: svd50={svd_50}, fruit_cos={fruit_Wout_max:.4f}, G_cos={G_Wout_max:.4f}")

L.log(f"  P257 done: {len(p257['per_layer'])} layers")

# 6. P258: FFN causal chain
L.log("[6] P258: FFN causal chain...")
p258 = {"per_layer": []}
for li in range(n_layers):
    wd = ffn_weights[li]
    if 'W_gate' not in wd:
        continue
    W_gate, W_in, W_out = wd['W_gate'], wd['W_in'], wd['W_out']
    
    apple_hs = single_hs.get("apple")
    fruit_mean_layer = fruit_mean[li]  # FFN input
    
    if apple_hs is None:
        continue
    
    # FFN forward (pure weight computation)
    h = apple_hs[li].unsqueeze(0)  # [1, d_model]
    gate = F.gelu(h @ W_gate)
    post_in = h @ W_in
    act = gate * post_in
    ffn_out = act @ W_out  # [1, d_model]
    
    # G-term = apple specific after removing fruit backbone
    G = apple_hs[li+1] - fruit_mean[li+1]
    G_norm = torch.linalg.norm(G)
    ffn_norm = torch.linalg.norm(ffn_out[0])
    
    if G_norm > 1e-8 and ffn_norm > 1e-8:
        cos_G_ffn = float(F.cosine_similarity(G.unsqueeze(0), ffn_out, dim=1)[0])
    else:
        cos_G_ffn = 0
    
    # Top contributing neurons
    neuron_contribution = (act * W_out.T.unsqueeze(0)).sum(-1)  # simplified
    top_neurons = torch.topk(act[0].abs(), 10)
    
    p258["per_layer"].append({
        "layer": li,
        "ffn_out_norm": round(float(ffn_norm), 2),
        "G_norm": round(float(G_norm), 2),
        "cos_G_ffn": round(cos_G_ffn, 4),
        "top10_neuron_indices": top_neurons.indices.tolist(),
        "top10_neuron_acts": [round(float(v), 4) for v in top_neurons.values],
    })
    if li % 5 == 0:
        L.log(f"  L{li}: G_norm={float(G_norm):.2f}, cos_G_ffn={cos_G_ffn:.4f}")

L.log(f"  P258 done: {len(p258['per_layer'])} layers")

# 7. P259: Apple-fruit weight-level source
L.log("[7] P259: Apple-fruit weight-level source...")
p259 = {"per_layer": []}
apple_specific_neurons_all = []

for li in range(n_layers):
    wd = ffn_weights[li]
    if 'W_gate' not in wd:
        continue
    W_gate, W_in, W_out = wd['W_gate'], wd['W_in'], wd['W_out']
    
    apple_hs = single_hs.get("apple")
    if apple_hs is None:
        continue
    
    # Compute apple FFN activation
    h = apple_hs[li].unsqueeze(0)
    gate = F.gelu(h @ W_gate)
    post_in = h @ W_in
    act = gate * post_in  # [1, d_mlp]
    
    # Compute fruit mean FFN activation
    fruit_acts = []
    for fw in fruit_words:
        if fw in single_hs:
            fh = single_hs[fw][li].unsqueeze(0)
            fg = F.gelu(fh @ W_gate)
            fp = fh @ W_in
            fruit_acts.append((fg * fp)[0])
    
    if len(fruit_acts) < 3:
        continue
    
    fruit_act_mean = torch.stack(fruit_acts).mean(0)  # [d_mlp]
    apple_excess = act[0] - fruit_act_mean  # apple-specific activation
    
    # Top apple-specific neurons
    top_excess = torch.topk(apple_excess.abs(), 20)
    apple_specific_neurons_all.extend([(li, int(idx), float(apple_excess[idx])) for idx in top_excess.indices])
    
    # W_out rows for top neurons
    top5_Wout = W_out[top_excess.indices[:5]]  # [5, d_model]
    cos_among_top5 = F.cosine_similarity(top5_Wout.unsqueeze(1), top5_Wout.unsqueeze(0), dim=2)
    mean_cos_top5 = float(cos_among_top5[cos_among_top5 < 0.99].mean()) if (cos_among_top5 < 0.99).any() else 0
    
    p259["per_layer"].append({
        "layer": li,
        "top5_apple_excess_neurons": top_excess.indices[:5].tolist(),
        "top5_excess_values": [round(float(v), 4) for v in top_excess.values[:5]],
        "top5_Wout_mean_cos": round(mean_cos_top5, 4),
    })

L.log(f"  P259 done: {len(p259['per_layer'])} layers, {len(apple_specific_neurons_all)} apple-specific neurons total")

# 8. Save results
L.log("[8] Saving results...")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = OUT_DIR / f"phase_xliii_p257_259_deepseek7b_{ts}.json"

all_results = {
    "model": "deepseek7b",
    "n_layers": n_layers,
    "d_model": d_model,
    "d_mlp": d_mlp,
    "p257_ffn_weight_analysis": p257,
    "p258_ffn_causal_chain": p258,
    "p259_weight_level_source": p259,
    "apple_specific_neurons_count": len(apple_specific_neurons_all),
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

L.log(f"  Results saved: {out_path}")
L.log(f"  Total time: {round(time.time()-t_total, 1)}s")
L.log("ALL DONE!")
L.close()
