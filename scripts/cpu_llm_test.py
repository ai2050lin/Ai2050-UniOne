"""CPU-only LLM test"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import json
import time

print("=" * 50)
print("Real LLM Validation (CPU)")
print("=" * 50)

results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "device": "cpu"
}

print("\n[1] Loading GPT-2 (CPU)...")
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"  OK: {model.config.n_layer}L, {model.config.n_embd}D, {n_params:,} params")

results["model"] = {"n_layers": model.config.n_layer, "n_params": n_params}

# 生成
print("\n[2] Generation...")
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors='pt')
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
text = tokenizer.decode(out[0])
print(f"  '{prompt}' -> '{text}'")
results["generation"] = text

# 激活提取
print("\n[3] Activations...")
activations = {}

def hook_fn(name):
    def hook(module, input, output):
        activations[name] = output[0].detach()
    return hook

hooks = []
for i in [3, 6, 9]:
    h = model.transformer.h[i].register_forward_hook(hook_fn(f"L{i}"))
    hooks.append(h)

with torch.no_grad():
    _ = model(**inputs)

for h in hooks:
    h.remove()

print(f"  Extracted: {list(activations.keys())}")
results["activations"] = {k: list(v.shape) for k, v in activations.items()}

# 曲率
print("\n[4] Curvature...")
import numpy as np
from sklearn.decomposition import PCA

curvs = {}
for name, act in activations.items():
    flat = act.reshape(-1, act.size(-1)).numpy()
    pca = PCA(n_components=min(10, flat.shape[1]-1))
    pca.fit(flat)
    curv = 1 - np.sum(pca.explained_variance_ratio_[:3])
    curvs[name] = float(curv)
    print(f"  {name}: {curv:.3f}")

results["curvatures"] = curvs

# 保存
os.makedirs("tempdata", exist_ok=True)
with open("tempdata/real_llm_validation.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: tempdata/real_llm_validation.json")
print(f"\nAvg curvature: {np.mean(list(curvs.values())):.3f}")
print("\nOK!")
