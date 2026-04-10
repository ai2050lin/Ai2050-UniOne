# -*- coding: utf-8 -*-
"""Phase XLIII DeepSeek7B 分步测试 - Step 1: 数据收集+权重提取"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, os, sys, gc, time, json
from datetime import datetime
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
import functools, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
print = functools.partial(print, flush=True)

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

print("=" * 50)
print("DeepSeek7B Step1: 数据收集+权重提取")
print("=" * 50)

# 1. 加载模型
print("[1] 加载模型...")
t0 = time.time()
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
# d_mlp
d_mlp = None
layers = mdl.model.layers
mlp0 = layers[0].mlp
for name, param in mlp0.named_parameters():
    if 'weight' in name and param.shape[0] > d_model:
        d_mlp = param.shape[0]; break
print(f"  OK! {n_layers}L, d={d_model}, d_mlp={d_mlp}, GPU={torch.cuda.memory_allocated(0)/1e9:.2f}GB, {time.time()-t0:.0f}s")

# 2. 收集hidden states
print("[2] 收集hidden states...")
t1 = time.time()
single_hs = {}  # word -> {template -> [layer tensors]}
combo_hs = {}

# 单词
single_words = []
for key in ["fruit_family","animal_family","vehicle_family","furniture_family","tool_family","color_attrs","taste_attrs","size_attrs"]:
    single_words.extend(STIMULI[key])

print(f"  单词数: {len(single_words)}, 模板数: {len(TEMPLATES)}")

for wi, word in enumerate(single_words):
    word_hs = {}
    for template in TEMPLATES:
        prompt = template.replace("{word}", word)
        inp = tok(prompt, return_tensors="pt", padding=False, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True, use_cache=False)
        hs = [h[0, -1].detach().float().cpu() for h in out.hidden_states]
        word_hs[template] = hs
        del out, inp
    single_hs[word] = word_hs
    if (wi+1) % 20 == 0:
        print(f"    [{wi+1}/{len(single_words)}] {word}")

# 组合词
combo_keys = ["fruit_color_combos", "fruit_taste_combos", "animal_color_combos"]
combos = []
for key in combo_keys:
    combos.extend(STIMULI[key])

print(f"  组合词数: {len(combos)}")

for ci, combo in enumerate(combos):
    combo_h = {}
    for template in TEMPLATES:
        prompt = template.replace("{word}", combo)
        inp = tok(prompt, return_tensors="pt", padding=False, truncation=True, max_length=128).to(device)
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True, use_cache=False)
        hs = [h[0, -1].detach().float().cpu() for h in out.hidden_states]
        combo_h[template] = hs
        del out, inp
    combo_hs[combo] = combo_h
    if (ci+1) % 10 == 0:
        print(f"    [{ci+1}/{len(combos)}] {combo}")

print(f"  数据收集完成: {time.time()-t1:.0f}s")

# 3. 提取FFN权重
print("[3] FFN权重提取...")
t2 = time.time()
ffn_weights = {}
for li in range(n_layers):
    mlp = layers[li].mlp
    wd = {}
    if hasattr(mlp, 'gate_proj'):
        wd['W_gate'] = mlp.gate_proj.weight.detach().float().cpu().T  # [d_model, d_mlp]
        wd['W_in'] = mlp.up_proj.weight.detach().float().cpu().T      # [d_model, d_mlp]
        wd['W_out'] = mlp.down_proj.weight.detach().float().cpu().T   # [d_mlp, d_model]
    if hasattr(mlp, 'gate_proj') and hasattr(mlp.gate_proj, 'bias') and mlp.gate_proj.bias is not None:
        wd['b_gate'] = mlp.gate_proj.bias.detach().float().cpu()
    if hasattr(mlp, 'up_proj') and hasattr(mlp.up_proj, 'bias') and mlp.up_proj.bias is not None:
        wd['b_in'] = mlp.up_proj.bias.detach().float().cpu()
    if hasattr(mlp, 'down_proj') and hasattr(mlp.down_proj, 'bias') and mlp.down_proj.bias is not None:
        wd['b_out'] = mlp.down_proj.bias.detach().float().cpu()
    ffn_weights[li] = wd
print(f"  权重提取完成: {time.time()-t2:.0f}s")

# 4. 保存中间数据
print("[4] 保存中间数据...")
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = OUT_DIR / f"xliii_ds7b_data_{ts}.pt"
torch.save({
    "single_hs": single_hs,
    "combo_hs": combo_hs,
    "ffn_weights": ffn_weights,
    "n_layers": n_layers,
    "d_model": d_model,
    "d_mlp": d_mlp,
}, save_path)
print(f"  保存: {save_path}")

# 5. 释放模型
print("[5] 释放模型...")
del mdl, tok
gc.collect()
torch.cuda.empty_cache()
print(f"  总用时: {time.time()-t0:.0f}s")
print("Step1完成!")
