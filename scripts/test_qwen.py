"""
Qwen 模型验证测试
=================

测试 Qwen3-4B 或 Qwen2.5-3B 模型
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = ''  # 允许使用 GPU
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
import json
import time

print("=" * 60)
print("Qwen 模型验证测试")
print("=" * 60)

# 检查是否有 GPU
if torch.cuda.is_available():
    device = "cuda"
    print(f"设备: CUDA GPU")
    # 清理 GPU 缓存
    torch.cuda.empty_cache()
else:
    device = "cpu"
    print(f"设备: CPU")

results = {
    "device": device,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
}

# ============================================================================
# 尝试加载 Qwen 模型
# ============================================================================
print("\n[1] 加载 Qwen 模型...")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    # 尝试不同的 Qwen 模型
    model_options = [
        "Qwen/Qwen2.5-3B",        # 3B 参数，较小
        "Qwen/Qwen2.5-1.5B",      # 1.5B 参数，更小
        "Qwen/Qwen2-1.5B",        # Qwen2 1.5B
        "Qwen/Qwen2-7B",          # 7B 参数
    ]
    
    model = None
    tokenizer = None
    loaded_model_name = None
    
    for model_name in model_options:
        try:
            print(f"  尝试加载: {model_name}")
            
            # 加载 tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, 
                trust_remote_code=True
            )
            
            # 加载模型
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32 if device == "cpu" else torch.float16,
                device_map=device if device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            if device == "cpu":
                model = model.to("cpu")
            
            model.eval()
            loaded_model_name = model_name
            print(f"  ✓ 成功加载: {model_name}")
            break
            
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            continue
    
    if model is None:
        print("  所有 Qwen 模型加载失败，尝试 GPT-2 作为备选...")
        from transformers import GPT2LMHeadModel, GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')
        model.eval()
        loaded_model_name = "gpt2"
        print("  ✓ 加载 GPT-2 作为备选")
    
    # 获取模型信息
    n_params = sum(p.numel() for p in model.parameters())
    
    # 尝试获取层数和维度
    if hasattr(model.config, 'num_hidden_layers'):
        n_layers = model.config.num_hidden_layers
        d_model = model.config.hidden_size
    elif hasattr(model.config, 'n_layer'):
        n_layers = model.config.n_layer
        d_model = model.config.n_embd
    else:
        n_layers = "unknown"
        d_model = "unknown"
    
    results["model"] = {
        "name": loaded_model_name,
        "n_params": n_params,
        "n_layers": n_layers,
        "d_model": d_model
    }
    
    print(f"\n  模型信息:")
    print(f"    名称: {loaded_model_name}")
    print(f"    参数量: {n_params:,}")
    print(f"    层数: {n_layers}")
    print(f"    维度: {d_model}")
    
except Exception as e:
    print(f"  ✗ 加载失败: {e}")
    import traceback
    traceback.print_exc()
    results["error"] = str(e)
    exit(1)

# ============================================================================
# 测试生成
# ============================================================================
print("\n[2] 测试生成...")

test_prompts = [
    "The capital of France is",
    "2 + 2 =",
    "人工智能的未来",
]

generation_results = []

for prompt in test_prompts:
    try:
        inputs = tokenizer(prompt, return_tensors='pt')
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # 简单前向传播
            outputs = model(**inputs)
            next_token = outputs.logits[0, -1].argmax(dim=-1)
            next_word = tokenizer.decode(next_token)
        
        print(f"  '{prompt}' -> '{next_word}'")
        generation_results.append({
            "prompt": prompt,
            "next_token": next_word
        })
        
    except Exception as e:
        print(f"  '{prompt}' -> 错误: {e}")

results["generation"] = generation_results

# ============================================================================
# 提取激活
# ============================================================================
print("\n[3] 提取中间层激活...")

activations = {}

def get_hook(name):
    def hook(module, input, output):
        if isinstance(output, tuple):
            activations[name] = output[0].detach().cpu()
        else:
            activations[name] = output.detach().cpu()
    return hook

# 注册钩子
hooks = []

# 尝试找到 transformer 层
if hasattr(model, 'model') and hasattr(model.model, 'layers'):
    # Qwen 架构
    layer_attr = model.model.layers
    layer_prefix = "model.layers"
elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
    # GPT-2 架构
    layer_attr = model.transformer.h
    layer_prefix = "transformer.h"
else:
    print("  无法找到 transformer 层，跳过激活提取")
    layer_attr = None

if layer_attr is not None:
    n_layers_total = len(layer_attr)
    layer_indices = [0, n_layers_total // 4, n_layers_total // 2, 
                     3 * n_layers_total // 4, n_layers_total - 1]
    
    print(f"  总层数: {n_layers_total}, 提取层: {layer_indices}")
    
    for i in layer_indices:
        h = layer_attr[i].register_forward_hook(get_hook(f'layer_{i}'))
        hooks.append(h)
    
    # 运行模型
    prompt = "The capital of Germany is"
    inputs = tokenizer(prompt, return_tensors='pt')
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        _ = model(**inputs)
    
    # 移除钩子
    for h in hooks:
        h.remove()
    
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

# ============================================================================
# 曲率计算
# ============================================================================
print("\n[4] 计算激活流形曲率...")

if activations:
    from sklearn.decomposition import PCA
    
    curvatures = {}
    
    for name, act in activations.items():
        try:
            flat = act.reshape(-1, act.size(-1)).numpy()
            n_samples, n_features = flat.shape
            
            # 确保 n_components 合法
            n_comp = min(3, n_samples - 1, n_features - 1)
            if n_comp < 1:
                print(f"  {name}: 样本不足，跳过")
                continue
            
            pca = PCA(n_components=n_comp)
            pca.fit(flat)
            
            curv = 1 - np.sum(pca.explained_variance_ratio_[:n_comp])
            curvatures[name] = float(curv)
            print(f"  {name}: 曲率={curv:.4f}")
            
        except Exception as e:
            print(f"  {name}: 计算失败 - {e}")
    
    results["curvatures"] = curvatures
    
    if curvatures:
        avg_curv = np.mean(list(curvatures.values()))
        print(f"\n  平均曲率: {avg_curv:.4f}")

# ============================================================================
# 干预测试
# ============================================================================
print("\n[5] 几何干预测试...")

if layer_attr is not None and len(layer_attr) > n_layers_total // 2:
    intervention_layer = n_layers_total // 2
    
    def intervention_hook(module, input, output):
        if isinstance(output, tuple):
            noise = torch.randn_like(output[0]) * 0.1
            return (output[0] + noise,) + output[1:]
        else:
            noise = torch.randn_like(output) * 0.1
            return output + noise
    
    prompt = "The capital of Japan is"
    inputs = tokenizer(prompt, return_tensors='pt')
    if device == "cuda":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 原始
    with torch.no_grad():
        outputs = model(**inputs)
        orig_token = outputs.logits[0, -1].argmax(dim=-1)
        orig_word = tokenizer.decode(orig_token)
    
    # 干预
    h = layer_attr[intervention_layer].register_forward_hook(intervention_hook)
    with torch.no_grad():
        outputs_int = model(**inputs)
        int_token = outputs_int.logits[0, -1].argmax(dim=-1)
        int_word = tokenizer.decode(int_token)
    h.remove()
    
    changed = orig_word != int_word
    print(f"  原始: '{orig_word}' -> 干预: '{int_word}'")
    print(f"  改变: {changed}")
    
    results["intervention"] = {
        "layer": intervention_layer,
        "original": orig_word,
        "intervened": int_word,
        "changed": changed
    }

# ============================================================================
# 保存结果
# ============================================================================
os.makedirs("tempdata", exist_ok=True)

output_file = "tempdata/qwen_validation.json"
with open(output_file, "w") as f:
    # 转换 numpy 类型
    def convert(obj):
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        return obj
    
    json.dump(convert(results), f, indent=2)

print(f"\n报告保存: {output_file}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 60)
print("验证总结")
print("=" * 60)

print(f"\n模型: {loaded_model_name}")
print(f"参数量: {n_params:,}")

if activations:
    print(f"提取激活: {len(activations)} 层")

if curvatures:
    avg = np.mean(list(curvatures.values()))
    print(f"平均曲率: {avg:.4f}")
    
    if avg < 0.1:
        print("  → 流形非常平坦，激活分布高度集中")
    elif avg < 0.3:
        print("  → 流形较为平坦")
    else:
        print("  → 流形曲率较高")

print("\n✓ Qwen 模型验证完成!")
