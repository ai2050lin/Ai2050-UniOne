"""
真实模型几何干预测试 - HookedTransformer 版
============================================

使用 TransformerLens 的 HookedTransformer 进行几何干预
"""

import torch
import numpy as np
import json
import time

print("=" * 60)
print("真实模型几何干预测试 (HookedTransformer)")
print("=" * 60)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"设备: {device}")

results = {}

# ============================================================================
# 1. 加载模型
# ============================================================================

print("\n[1] 加载 HookedTransformer...")

try:
    from transformer_lens import HookedTransformer
    
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    print(f"  模型加载成功: {model.cfg.n_layers} 层, {model.cfg.d_model} 维")
    
    results["model_loaded"] = True
    results["n_layers"] = model.cfg.n_layers
    results["d_model"] = model.cfg.d_model
    
except Exception as e:
    print(f"  模型加载失败: {e}")
    results["model_loaded"] = False
    results["error"] = str(e)
    
    # 保存并退出
    import os
    os.makedirs("tempdata", exist_ok=True)
    with open("tempdata/geo_intervention_report.json", "w") as f:
        json.dump(results, f, indent=2)
    exit(1)

# ============================================================================
# 2. 提取激活
# ============================================================================

print("\n[2] 提取层激活...")

test_prompts = [
    "The capital of France is",
    "2 + 2 equals",
]

activations = {}

def get_hook_fn(layer_name):
    def hook_fn(activation, hook):
        activations[layer_name] = activation.detach().clone()
        return activation
    return hook_fn

# 注册钩子并运行
hooks = []
layer_indices = [3, 6, 9, 11]  # GPT-2 有 12 层

for idx in layer_indices:
    hooks.append(model.blocks[idx].hook_resid_post.add_hook(get_hook_fn(f"layer_{idx}")))

# 运行模型
for prompt in test_prompts:
    tokens = model.to_tokens(prompt)
    with torch.no_grad():
        _ = model(tokens)

print(f"  提取了 {len(activations)} 层的激活")

for name, act in activations.items():
    print(f"    {name}: shape={list(act.shape)}")

results["activations"] = {k: list(v.shape) for k, v in activations.items()}

# ============================================================================
# 3. 计算曲率
# ============================================================================

print("\n[3] 计算激活曲率...")

curvature_results = {}

for layer_name, act in activations.items():
    # 展平激活
    flat = act.reshape(-1, act.size(-1)).cpu().numpy()
    
    # 计算局部曲率 (使用近邻方差)
    from sklearn.neighbors import NearestNeighbors
    
    k = min(5, len(flat) - 1)
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(flat)
    distances, indices = nbrs.kneighbors(flat)
    
    # 曲率 ≈ 近邻距离的方差
    curvatures = []
    for i in range(len(flat)):
        neighbor_dists = distances[i, 1:]  # 排除自身
        curvatures.append(np.var(neighbor_dists))
    
    mean_curv = np.mean(curvatures)
    curvature_results[layer_name] = float(mean_curv)
    print(f"  {layer_name}: 曲率={mean_curv:.4f}")

results["curvatures"] = curvature_results

# ============================================================================
# 4. 几何干预
# ============================================================================

print("\n[4] 执行几何干预...")

def heat_kernel_smoothing(activation, temperature=0.1):
    """热核平滑"""
    flat = activation.reshape(-1, activation.size(-1))
    
    # 计算距离矩阵
    dist = torch.cdist(flat, flat)
    
    # 热核
    kernel = torch.exp(-dist ** 2 / (4 * temperature))
    kernel = kernel / kernel.sum(dim=1, keepdim=True)
    
    # 平滑
    smoothed = kernel @ flat
    
    return smoothed.reshape(activation.shape)

# 在中间层应用干预
target_layer = 6
intervention_strength = 0.3

print(f"\n  目标层: {target_layer}, 强度: {intervention_strength}")

# 原始生成
prompt = "The capital of France is"
tokens = model.to_tokens(prompt)

print(f"\n  原始生成:")
with torch.no_grad():
    original_logits = model(tokens)
    original_pred = model.to_string(original_logits[0, -1].argmax(dim=-1))
    print(f"    下一个token: '{original_pred}'")

# 干预后生成
def intervention_hook(activation, hook):
    smoothed = heat_kernel_smoothing(activation, temperature=0.1)
    return (1 - intervention_strength) * activation + intervention_strength * smoothed

print(f"\n  干预生成:")
with model.hooks([(f"blocks.{target_layer}.hook_resid_post", intervention_hook)]):
    with torch.no_grad():
        intervened_logits = model(tokens)
        intervened_pred = model.to_string(intervened_logits[0, -1].argmax(dim=-1))
        print(f"    下一个token: '{intervened_pred}'")

# 计算干预效果
logit_diff = (intervened_logits - original_logits).abs().mean().item()
print(f"\n  Logit 差异: {logit_diff:.4f}")

results["intervention"] = {
    "target_layer": target_layer,
    "strength": intervention_strength,
    "original_prediction": original_pred,
    "intervened_prediction": intervened_pred,
    "logit_difference": logit_diff
}

# ============================================================================
# 5. 测地线引导
# ============================================================================

print("\n[5] 测地线引导测试...")

# 计算主方向 (PCA)
from sklearn.decomposition import PCA

if "layer_6" in activations:
    act = activations["layer_6"].cpu().numpy()
    flat = act.reshape(-1, act.shape[-1])
    
    pca = PCA(n_components=3)
    pca.fit(flat)
    
    print(f"  主成分解释方差比: {pca.explained_variance_ratio_}")
    
    # 沿第一主成分引导
    direction = torch.tensor(pca.components_[0], device=device, dtype=torch.float32)
    
    def geodesic_hook(activation, hook):
        # 沿方向微调
        proj = (activation @ direction) / (direction @ direction) * direction
        return activation + 0.05 * proj
    
    print(f"\n  测地线引导生成:")
    with model.hooks([(f"blocks.{target_layer}.hook_resid_post", geodesic_hook)]):
        with torch.no_grad():
            steered_logits = model(tokens)
            steered_pred = model.to_string(steered_logits[0, -1].argmax(dim=-1))
            print(f"    下一个token: '{steered_pred}'")
    
    results["geodesic_steering"] = {
        "explained_variance": list(pca.explained_variance_ratio_),
        "prediction": steered_pred
    }

# ============================================================================
# 保存结果
# ============================================================================

import os
os.makedirs("tempdata", exist_ok=True)

with open("tempdata/geo_intervention_report.json", "w") as f:
    json.dump({
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        **results
    }, f, indent=2)

print(f"\n报告保存到: tempdata/geo_intervention_report.json")

# ============================================================================
# 总结
# ============================================================================

print("\n" + "=" * 60)
print("几何干预测试总结")
print("=" * 60)

print(f"\n模型: GPT-2 Small ({model.cfg.n_layers} 层)")
print(f"提取激活: {len(activations)} 层")
print(f"曲率计算: {len(curvature_results)} 层")
print(f"\n干预效果:")
print(f"  原始预测: '{original_pred}'")
print(f"  干预后预测: '{intervened_pred}'")
print(f"  Logit 差异: {logit_diff:.4f}")

if logit_diff > 0.01:
    print("\n结论: 几何干预有效，成功改变了模型输出")
else:
    print("\n结论: 干预效果微弱，需要调整参数")
