"""最小化 LLM 测试 - 直接使用 transformers"""
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import json
import time

print("=" * 50)
print("真实 LLM 几何验证 (transformers)")
print("=" * 50)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

results = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "device": device
}

# 使用 transformers 直接加载
print("\n[1] Loading GPT-2...")
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model.eval()

n_params = sum(p.numel() for p in model.parameters())
print(f"  Loaded: {model.config.n_layer} layers, {model.config.n_embd} dim, {n_params:,} params")

results["model"] = {
    "n_layers": model.config.n_layer,
    "d_model": model.config.n_embd,
    "n_params": n_params
}

# 基础生成
print("\n[2] Base generation...")
prompts = [
    "The capital of France is",
    "2 + 2 =",
]

base_outputs = []
for p in prompts:
    inputs = tokenizer(p, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=5, do_sample=False)
    text = tokenizer.decode(out[0])
    base_outputs.append({"prompt": p, "output": text})
    print(f"  '{p}' -> '{text}'")

results["base_generation"] = base_outputs

# 提取激活
print("\n[3] Extracting activations...")
activations = {}

def get_activation_hook(name):
    def hook(module, input, output):
        activations[name] = output[0].detach().cpu()
    return hook

# 注册钩子
hooks = []
target_layers = [3, 6, 9]
for i in target_layers:
    h = model.transformer.h[i].register_forward_hook(get_activation_hook(f"layer_{i}"))
    hooks.append(h)

# 运行
test_input = tokenizer("Paris is the capital of", return_tensors='pt').to(device)
with torch.no_grad():
    _ = model(**test_input)

# 移除钩子
for h in hooks:
    h.remove()

print(f"  Extracted {len(activations)} layers")
for name, act in activations.items():
    print(f"    {name}: {list(act.shape)}")

results["activations"] = {k: list(v.shape) for k, v in activations.items()}

# 曲率计算
print("\n[4] Computing curvature...")
import numpy as np
from sklearn.decomposition import PCA

curvatures = {}
for name, act in activations.items():
    flat = act.reshape(-1, act.size(-1)).numpy()
    
    pca = PCA(n_components=min(10, flat.shape[1]-1))
    pca.fit(flat)
    
    # 曲率 = 1 - 前3主成分解释方差
    curv = 1 - np.sum(pca.explained_variance_ratio_[:3])
    curvatures[name] = float(curv)
    print(f"  {name}: curvature={curv:.3f}")

results["curvatures"] = curvatures

# 干预测试
print("\n[5] Intervention test...")

# 简单的激活噪声干预
intervention_results = []

def noise_hook(scale):
    def hook(module, input, output):
        noise = torch.randn_like(output[0]) * scale
        return (output[0] + noise,)
    return hook

for layer in [6, 9]:
    # 注册干预钩子
    h = model.transformer.h[layer].register_forward_hook(noise_hook(0.1))
    
    int_outputs = []
    for p in ["The capital of Germany is", "5 + 3 ="]:
        inputs = tokenizer(p, return_tensors='pt').to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=3, do_sample=False)
        text = tokenizer.decode(out[0])
        int_outputs.append(text)
    
    h.remove()
    
    intervention_results.append({
        "layer": layer,
        "outputs": int_outputs
    })
    print(f"  Layer {layer}: intervention done")

results["interventions"] = intervention_results

# 保存
import os
os.makedirs("tempdata", exist_ok=True)
with open("tempdata/real_llm_validation.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved: tempdata/real_llm_validation.json")

# 总结
print("\n" + "=" * 50)
print("Summary")
print("=" * 50)
print(f"Model: GPT-2 ({n_params:,} params)")
print(f"Activations: {len(activations)} layers")
print(f"Avg curvature: {np.mean(list(curvatures.values())):.3f}")

print("\n✓ Real LLM validation complete!")
