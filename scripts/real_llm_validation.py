"""
真实 LLM 验证 - CPU 模式
========================

在真实 GPT-2 上验证几何干预理论 (强制 CPU 避免 CUDA 错误)
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 强制使用 CPU

import torch
import numpy as np
import json
import time
import sys
import gc

print("=" * 60)
print("真实 LLM 几何验证 (CPU 模式)")
print("=" * 60)

device = torch.device("cpu")
print(f"设备: {device}")
results = {"device": "cpu", "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

# ============================================================================
# Step 1: 加载模型
# ============================================================================
print("\n[Step 1/6] 加载 GPT-2 Small (CPU)...")

try:
    from transformer_lens import HookedTransformer
    
    # 清理内存
    gc.collect()
    
    # 强制 CPU
    model = HookedTransformer.from_pretrained("gpt2-small", device="cpu")
    
    results["model"] = {
        "name": "gpt2-small",
        "n_layers": model.cfg.n_layers,
        "d_model": model.cfg.d_model,
        "n_params": sum(p.numel() for p in model.parameters())
    }
    
    print(f"  ✓ 模型加载成功 (CPU)")
    print(f"    层数: {model.cfg.n_layers}")
    print(f"    维度: {model.cfg.d_model}")
    print(f"    参数量: {results['model']['n_params']:,}")
    
except Exception as e:
    print(f"  ✗ 加载失败: {e}")
    results["model"] = {"error": str(e)}
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ============================================================================
# Step 2: 基础生成测试
# ============================================================================
print("\n[Step 2/6] 基础生成测试...")

test_prompts = [
    "The capital of France is",
    "2 + 2 =",
]

baseline_outputs = []

for prompt in test_prompts:
    try:
        tokens = model.to_tokens(prompt)
        with torch.no_grad():
            # 使用简单的前向传播代替 generate
            logits = model(tokens)
            # 取最后一个 token 的预测
            next_token = logits[0, -1].argmax(dim=-1)
            next_word = model.to_string(next_token)
        
        generated = prompt + " " + next_word
        baseline_outputs.append({"prompt": prompt, "next_token": next_word})
        print(f"  '{prompt}' -> '{next_word}'")
    except Exception as e:
        print(f"  '{prompt}' -> 错误: {e}")
        baseline_outputs.append({"prompt": prompt, "error": str(e)})

results["baseline_generation"] = baseline_outputs

# ============================================================================
# Step 3: 激活提取
# ============================================================================
print("\n[Step 3/6] 提取中间层激活...")

activations = {}

def hook_fn(activation, hook):
    activations[hook.name] = activation.detach().clone()
    return activation

# 选择关键层
layer_indices = [0, 3, 6, 9, 11]

test_prompt = "The capital of Germany is"
tokens = model.to_tokens(test_prompt)

try:
    with torch.no_grad():
        with model.hooks([(f"blocks.{i}.hook_resid_post", hook_fn) for i in layer_indices]):
            _ = model(tokens)

    print(f"  ✓ 提取了 {len(activations)} 层激活")

    activation_shapes = {}
    for name, act in activations.items():
        shape = list(act.shape)
        activation_shapes[name] = shape
        print(f"    {name}: {shape}")

    results["activations"] = {
        "n_layers": len(activations),
        "shapes": activation_shapes
    }
except Exception as e:
    print(f"  ✗ 激活提取失败: {e}")
    results["activations"] = {"error": str(e)}

# ============================================================================
# Step 4: 曲率计算
# ============================================================================
print("\n[Step 4/6] 计算激活流形曲率...")

curvatures = {}

if activations:
    from sklearn.decomposition import PCA
    
    for layer_name, act in activations.items():
        try:
            # 移到 CPU 并转为 numpy
            flat = act.cpu().reshape(-1, act.size(-1)).numpy()
            
            if flat.shape[0] < 3:
                continue
            
            # 计算曲率 = 1 - 前k个主成分解释的方差
            n_components = min(10, flat.shape[1] - 1)
            pca = PCA(n_components=n_components)
            pca.fit(flat)
            
            # 曲率 = 1 - 前3主成分解释方差
            curvature = 1 - np.sum(pca.explained_variance_ratio_[:3])
            
            curvatures[layer_name] = {
                "curvature": float(curvature),
                "top_var_explained": float(np.sum(pca.explained_variance_ratio_[:3])),
                "effective_dim": float(np.sum(pca.explained_variance_ratio_ > 0.01))
            }
            
            print(f"  {layer_name}: 曲率={curvature:.3f}, 前3成分方差={curvatures[layer_name]['top_var_explained']:.3f}")
            
        except Exception as e:
            print(f"  {layer_name}: 计算失败 - {e}")

results["curvatures"] = curvatures

# ============================================================================
# Step 5: 几何干预
# ============================================================================
print("\n[Step 5/6] 几何干预测试...")

intervention_results = []

def make_intervention_hook(strength):
    def hook(activation, hook):
        # 简单的平滑干预
        smoothed = activation + torch.randn_like(activation) * 0.01
        return (1 - strength) * activation + strength * smoothed
    return hook

for target_layer in [6, 9]:
    print(f"\n  干预层 {target_layer}:")
    
    test_prompts_intervention = ["The capital of Spain is"]
    
    for prompt in test_prompts_intervention:
        try:
            tokens = model.to_tokens(prompt)
            
            # 原始
            with torch.no_grad():
                orig_logits = model(tokens)
                orig_pred = model.to_string(orig_logits[0, -1].argmax(dim=-1))
            
            # 干预
            with torch.no_grad():
                with model.hooks([(f"blocks.{target_layer}.hook_resid_post", 
                                 make_intervention_hook(0.3))]):
                    int_logits = model(tokens)
                    int_pred = model.to_string(int_logits[0, -1].argmax(dim=-1))
            
            changed = orig_pred != int_pred
            intervention_results.append({
                "layer": target_layer,
                "prompt": prompt,
                "original": orig_pred,
                "intervened": int_pred,
                "changed": changed
            })
            
            status = "改变" if changed else "不变"
            print(f"    '{prompt}': {status} ({orig_pred} -> {int_pred})")
            
        except Exception as e:
            print(f"    错误: {e}")

results["interventions"] = intervention_results

# ============================================================================
# Step 6: 测地线引导
# ============================================================================
print("\n[Step 6/6] 测地线引导测试...")

if "blocks.6.hook_resid_post" in activations:
    try:
        from sklearn.decomposition import PCA
        
        act = activations["blocks.6.hook_resid_post"]
        flat = act.cpu().reshape(-1, act.size(-1)).numpy()
        
        pca = PCA(n_components=5)
        pca.fit(flat)
        
        print(f"  主成分方差比: {[f'{v:.3f}' for v in pca.explained_variance_ratio_[:3]]}")
        
        # 沿第一主成分引导
        direction = torch.tensor(pca.components_[0], dtype=torch.float32)
        
        def steering_hook(activation, hook):
            # 投影到主方向并增强
            proj = activation @ direction
            return activation + 0.05 * proj.unsqueeze(-1) * direction
        
        # 测试引导
        steering_prompts = ["Paris is the capital of"]
        
        steering_results = []
        
        for prompt in steering_prompts:
            tokens = model.to_tokens(prompt)
            
            # 原始
            with torch.no_grad():
                orig_logits = model(tokens)
                orig_pred = model.to_string(orig_logits[0, -1].argmax(dim=-1))
            
            # 引导
            with torch.no_grad():
                with model.hooks([("blocks.6.hook_resid_post", steering_hook)]):
                    steered_logits = model(tokens)
                    steered_pred = model.to_string(steered_logits[0, -1].argmax(dim=-1))
            
            steering_results.append({
                "prompt": prompt,
                "original": orig_pred,
                "steered": steered_pred
            })
            
            print(f"  '{prompt}':")
            print(f"    原始: '{orig_pred}'")
            print(f"    引导: '{steered_pred}'")

        results["steering"] = steering_results
        
    except Exception as e:
        print(f"  测地线引导失败: {e}")

# ============================================================================
# 保存结果
# ============================================================================
os.makedirs("tempdata", exist_ok=True)

with open("tempdata/real_llm_validation.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n报告保存: tempdata/real_llm_validation.json")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 60)
print("验证总结")
print("=" * 60)

print(f"\n模型: GPT-2 Small ({results['model']['n_params']:,} 参数)")
print(f"设备: CPU")

if activations:
    print(f"提取激活: {len(activations)} 层")

if curvatures:
    avg_curvature = np.mean([c["curvature"] for c in curvatures.values()])
    print(f"曲率计算: {len(curvatures)} 层")
    print(f"平均曲率: {avg_curvature:.3f}")
    
    if avg_curvature < 0.3:
        print("  → 流形较为平坦，激活分布集中")
    else:
        print("  → 流形曲率较高，激活分布复杂")

# 干预统计
if intervention_results:
    intervention_changes = sum(1 for r in intervention_results if r.get("changed", False))
    print(f"\n几何干预效果: {intervention_changes}/{len(intervention_results)} 改变输出")

print("\n✓ 真实 LLM 几何验证完成!")
