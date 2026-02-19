"""Qwen 测试 - 保存结果到文件"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import json
import time

result = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

# 加载 Qwen
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    print("Loading Qwen 0.5B...")
    
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)
    model.eval()
    
    n_params = sum(p.numel() for p in model.parameters())
    result["model"] = {"name": "Qwen2.5-0.5B", "n_params": n_params}
    print(f"Loaded: {n_params:,} params")
    
    # 测试生成
    prompt = "The capital of France is"
    inputs = tokenizer(prompt, return_tensors='pt')
    
    with torch.no_grad():
        out = model(**inputs)
        next_token = out.logits[0, -1].argmax()
        next_word = tokenizer.decode(next_token)
    
    print(f"Generated: '{prompt}' -> '{next_word}'")
    result["generation"] = {"prompt": prompt, "next_token": next_word}
    
    # 激活提取
    activations = {}
    def hook_fn(name):
        def h(module, input, output):
            activations[name] = output[0].detach() if isinstance(output, tuple) else output.detach()
        return h
    
    layers = model.model.layers
    n_layers = len(layers)
    
    hooks = [layers[i].register_forward_hook(hook_fn(f'L{i}')) for i in [0, n_layers//2, n_layers-1]]
    
    with torch.no_grad():
        _ = model(**inputs)
    
    for h in hooks:
        h.remove()
    
    print(f"Extracted: {len(activations)} layers")
    result["activations"] = {k: list(v.shape) for k, v in activations.items()}
    
    # 曲率
    import numpy as np
    from sklearn.decomposition import PCA
    
    curvatures = {}
    for name, act in activations.items():
        # 转换为 float32 避免类型错误
        act_float = act.float()
        flat = act_float.reshape(-1, act_float.size(-1)).numpy()
        n_comp = min(3, flat.shape[0]-1, flat.shape[1]-1)
        if n_comp < 1:
            continue
        pca = PCA(n_components=n_comp)
        pca.fit(flat)
        curv = 1 - np.sum(pca.explained_variance_ratio_[:n_comp])
        curvatures[name] = float(curv)
    
    result["curvatures"] = curvatures
    print(f"Curvatures: {curvatures}")
    
except Exception as e:
    import traceback
    result["error"] = str(e)
    traceback.print_exc()

# 保存
os.makedirs("tempdata", exist_ok=True)
with open("tempdata/qwen_validation.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\nSaved to: tempdata/qwen_validation.json")
