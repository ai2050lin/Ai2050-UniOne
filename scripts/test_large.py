"""最小化大模型测试"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import json
import time

print("=" * 50)
print("大模型验证测试")
print("=" * 50)

results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

# 1. 尝试加载 Qwen
print("\n尝试加载 Qwen...")
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 只尝试最小的 Qwen 模型
    model_name = "Qwen/Qwen2.5-0.5B"  # 最小的 Qwen
    
    print(f"  加载: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        trust_remote_code=True,
        torch_dtype=torch.float32
    )
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  OK: {model_name} ({n_params:,} params)")
    
    results["model"] = {"name": model_name, "n_params": n_params}
    loaded = "qwen"
    
except Exception as e:
    print(f"  Qwen 加载失败: {e}")
    print("\n回退到 GPT-2...")
    
    from transformers import GPT2LMHeadModel, GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  OK: GPT-2 ({n_params:,} params)")
    
    results["model"] = {"name": "gpt2", "n_params": n_params}
    loaded = "gpt2"

# 2. 测试生成
print("\n测试生成:")
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors='pt')

with torch.no_grad():
    out = model(**inputs)
    next_token = out.logits[0, -1].argmax()
    next_word = tokenizer.decode(next_token)

print(f"  '{prompt}' -> '{next_word}'")
results["generation"] = {"prompt": prompt, "next_token": next_word}

# 3. 激活提取
print("\n提取激活:")
activations = {}

def hook_fn(name):
    def h(module, input, output):
        activations[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
    return h

# 找到 transformer 层
if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    layers = model.model.layers
elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
    layers = model.transformer.h
else:
    layers = []

if layers:
    n_layers = len(layers)
    indices = [0, n_layers//2, n_layers-1]
    
    hooks = [layers[i].register_forward_hook(hook_fn(f'L{i}')) for i in indices]
    
    with torch.no_grad():
        _ = model(**inputs)
    
    for h in hooks:
        h.remove()
    
    print(f"  提取: {len(activations)} 层")
    results["activations"] = {k: list(v.shape) for k, v in activations.items()}

# 4. 曲率
print("\n计算曲率:")
import numpy as np
from sklearn.decomposition import PCA

curvatures = {}
for name, act in activations.items():
    flat = act.reshape(-1, act.size(-1)).numpy()
    n_comp = min(3, flat.shape[0]-1, flat.shape[1]-1)
    if n_comp < 1:
        continue
    pca = PCA(n_components=n_comp)
    pca.fit(flat)
    curv = 1 - np.sum(pca.explained_variance_ratio_[:n_comp])
    curvatures[name] = float(curv)
    print(f"  {name}: {curv:.4f}")

results["curvatures"] = curvatures

# 5. 保存
os.makedirs("tempdata", exist_ok=True)
with open("tempdata/large_model_validation.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n保存: tempdata/large_model_validation.json")

# 总结
print("\n" + "=" * 50)
print(f"验证完成: {loaded}")
if curvatures:
    print(f"平均曲率: {np.mean(list(curvatures.values())):.4f}")
