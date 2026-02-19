"""
GPT-2 Medium 模型验证测试
=========================

测试更大的 GPT-2 模型 (355M 参数)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
import json
import time

print("=" * 60)
print("GPT-2 Medium 模型验证测试")
print("=" * 60)

device = "cpu"
print(f"设备: CPU")

results = {
    "device": device,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

# ============================================================================
# 加载 GPT-2 Medium
# ============================================================================
print("\n[1] 加载 GPT-2 Medium (355M)...")

from transformers import GPT2LMHeadModel, GPT2Tokenizer

try:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-medium')
    model = GPT2LMHeadModel.from_pretrained('gpt2-medium')
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    n_layers = model.config.n_layer
    d_model = model.config.n_embd
    
    results["model"] = {
        "name": "gpt2-medium",
        "n_params": n_params,
        "n_layers": n_layers,
        "d_model": d_model
    }
    
    print(f"  ✓ 加载成功")
    print(f"    参数量: {n_params:,}")
    print(f"    层数: {n_layers}")
    print(f"    维度: {d_model}")
    
except Exception as e:
    print(f"  ✗ 加载失败: {e}")
    results["error"] = str(e)
    exit(1)

# ============================================================================
# 测试生成
# ============================================================================
print("\n[2] 测试生成...")

test_prompts = [
    "The capital of France is",
    "Artificial intelligence will",
]

generation_results = []

for prompt in test_prompts:
    inputs = tokenizer(prompt, return_tensors='pt')
    
    with torch.no_grad():
        outputs = model(**inputs)
        next_token = outputs.logits[0, -1].argmax(dim=-1)
        next_word = tokenizer.decode(next_token)
    
    print(f"  '{prompt}' -> '{next_word}'")
    generation_results.append({"prompt": prompt, "next_token": next_word})

results["generation"] = generation_results

# ============================================================================
# 提取激活
# ============================================================================
print("\n[3] 提取中间层激活...")

activations = {}

def get_hook(name):
    def hook(module, input, output):
        activations[name] = output[0].detach()
    return hook

hooks = []
layer_indices = [0, 6, 12, 18, 23]  # GPT-2 Medium 有 24 层

for i in layer_indices:
    h = model.transformer.h[i].register_forward_hook(get_hook(f'layer_{i}'))
    hooks.append(h)

prompt = "The capital of Germany is"
inputs = tokenizer(prompt, return_tensors='pt')

with torch.no_grad():
    _ = model(**inputs)

for h in hooks:
    h.remove()

print(f"  ✓ 提取了 {len(activations)} 层激活")
results["activations"] = {k: list(v.shape) for k, v in activations.items()}

# ============================================================================
# 曲率计算
# ============================================================================
print("\n[4] 计算激活流形曲率...")

from sklearn.decomposition import PCA

curvatures = {}

for name, act in activations.items():
    flat = act.reshape(-1, act.size(-1)).numpy()
    n_samples, n_features = flat.shape
    n_comp = min(3, n_samples - 1, n_features - 1)
    
    if n_comp < 1:
        continue
    
    pca = PCA(n_components=n_comp)
    pca.fit(flat)
    
    curv = 1 - np.sum(pca.explained_variance_ratio_[:n_comp])
    curvatures[name] = float(curv)
    print(f"  {name}: 曲率={curv:.4f}")

results["curvatures"] = curvatures

if curvatures:
    avg_curv = np.mean(list(curvatures.values()))
    print(f"\n  平均曲率: {avg_curv:.4f}")

# ============================================================================
# 干预测试
# ============================================================================
print("\n[5] 几何干预测试...")

def intervention_hook(module, input, output):
    noise = torch.randn_like(output[0]) * 0.1
    return (output[0] + noise,)

# 原始
prompt = "The capital of Japan is"
inputs = tokenizer(prompt, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)
    orig_token = outputs.logits[0, -1].argmax(dim=-1)
    orig_word = tokenizer.decode(orig_token)

# 干预
h = model.transformer.h[12].register_forward_hook(intervention_hook)
with torch.no_grad():
    outputs_int = model(**inputs)
    int_token = outputs_int.logits[0, -1].argmax(dim=-1)
    int_word = tokenizer.decode(int_token)
h.remove()

changed = orig_word != int_word
print(f"  原始: '{orig_word}' -> 干预: '{int_word}'")
print(f"  改变: {changed}")

results["intervention"] = {
    "original": orig_word,
    "intervened": int_word,
    "changed": changed
}

# ============================================================================
# 保存结果
# ============================================================================
os.makedirs("tempdata", exist_ok=True)

with open("tempdata/gpt2_medium_validation.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n报告保存: tempdata/gpt2_medium_validation.json")

# ============================================================================
# 与 GPT-2 Small 对比
# ============================================================================
print("\n" + "=" * 60)
print("与 GPT-2 Small 对比")
print("=" * 60)

print(f"\nGPT-2 Small:  124M 参数, 12 层, 768 维, 曲率 ~0.014")
print(f"GPT-2 Medium: {n_params/1e6:.0f}M 参数, {n_layers} 层, {d_model} 维, 曲率 ~{avg_curv:.4f}")

print("\n✓ GPT-2 Medium 验证完成!")
