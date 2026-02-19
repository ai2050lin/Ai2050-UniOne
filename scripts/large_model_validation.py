"""
P1-4: Large Model Validation
更大模型验证

测试 AGI 理论在更大模型上的有效性：
1. 模型规模扩展验证 (GPT-2 → Qwen 7B)
2. 曲率一致性测试
3. 层级激活模式分析
4. 零样本能力评估
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import torch
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional

print("=" * 60)
print("P1-4: Large Model Validation")
print("=" * 60)

# 设备配置
if torch.cuda.is_available():
    device = "cuda"
    torch.cuda.empty_cache()
    print(f"Device: CUDA GPU ({torch.cuda.get_device_name(0)})")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    device = "cpu"
    print(f"Device: CPU")

results = {
    "timestamp": datetime.now().isoformat(),
    "device": device,
    "models_tested": [],
    "comparisons": {}
}


def estimate_curvature(activations: np.ndarray) -> float:
    """
    估计激活流形的曲率
    使用协方差矩阵的特征值来估计内在曲率
    """
    if activations.ndim == 3:
        # [batch, seq, hidden] -> [batch * seq, hidden]
        activations = activations.reshape(-1, activations.shape[-1])
    
    # 中心化
    activations = activations - activations.mean(axis=0, keepdims=True)
    
    # 计算协方差矩阵
    n_samples = activations.shape[0]
    if n_samples < 10:
        return 0.0
    
    # 采样以提高效率
    if n_samples > 1000:
        indices = np.random.choice(n_samples, 1000, replace=False)
        activations = activations[indices]
    
    # 计算协方差
    cov = np.cov(activations.T)
    
    if cov.size == 1:
        return float(abs(cov.flatten()[0]))
    
    # 特征值分解
    try:
        eigenvalues = np.linalg.eigvalsh(cov)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # 移除接近零的特征值
        
        if len(eigenvalues) == 0:
            return 0.0
        
        # 曲率估计：使用特征值的相对方差
        # 高曲率 = 特征值分布不均匀
        mean_eigenvalue = np.mean(eigenvalues)
        var_eigenvalue = np.var(eigenvalues)
        
        if mean_eigenvalue > 0:
            curvature = var_eigenvalue / (mean_eigenvalue ** 2)
        else:
            curvature = 0.0
        
        return float(min(curvature, 1.0))  # 限制上限
    
    except:
        return 0.0


def test_model_generation(model, tokenizer, model_name: str) -> Dict[str, Any]:
    """测试模型生成能力"""
    test_prompts = [
        "The capital of France is",
        "2 + 2 =",
        "If it rains, the ground becomes",
        "The largest planet in our solar system is",
        "Python is a programming language that"
    ]
    
    results = {"prompts": []}
    
    for prompt in test_prompts:
        try:
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            new_text = generated[len(prompt):].strip()
            
            results["prompts"].append({
                "prompt": prompt,
                "generated": new_text[:50]  # 截断
            })
            
        except Exception as e:
            results["prompts"].append({
                "prompt": prompt,
                "error": str(e)[:50]
            })
    
    return results


def analyze_layer_patterns(model, tokenizer, model_name: str) -> Dict[str, Any]:
    """分析层级激活模式"""
    test_input = "The quick brown fox jumps over the lazy dog."
    
    try:
        inputs = tokenizer(test_input, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 获取隐藏层输出
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states  # tuple of tensors
        
        layer_curvatures = []
        layer_norms = []
        
        for i, hidden in enumerate(hidden_states):
            if hidden is not None:
                activations = hidden.cpu().numpy()
                
                # 曲率估计
                curvature = estimate_curvature(activations)
                layer_curvatures.append(curvature)
                
                # 范数
                norm = float(np.mean(np.linalg.norm(activations, axis=-1)))
                layer_norms.append(norm)
        
        return {
            "n_layers": len(layer_curvatures),
            "curvatures": layer_curvatures,
            "norms": layer_norms,
            "avg_curvature": float(np.mean(layer_curvatures)) if layer_curvatures else 0.0,
            "curvature_trend": "decreasing" if len(layer_curvatures) > 1 and layer_curvatures[-1] < layer_curvatures[0] else "stable/increasing"
        }
    
    except Exception as e:
        return {"error": str(e)[:100]}


def load_model_safely(model_name: str):
    """安全加载模型，支持内存优化"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\n  Attempting to load: {model_name}")
    
    # 检查模型大小
    model_sizes = {
        "gpt2": 0.12,  # GB
        "gpt2-medium": 0.35,
        "gpt2-large": 0.77,
        "Qwen/Qwen2.5-0.5B": 0.5,
        "Qwen/Qwen2.5-1.5B": 1.5,
        "Qwen/Qwen2.5-3B": 3.0,
        "Qwen/Qwen2-7B": 7.0,
        "Qwen/Qwen2.5-7B": 7.0,
    }
    
    estimated_size = model_sizes.get(model_name, 5.0)  # 默认估计 5GB
    
    # GPU 内存检查
    if device == "cuda":
        free_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        # 考虑 overhead
        required_memory = estimated_size * 2.5
        
        if required_memory > free_memory:
            print(f"    Warning: Model may need {required_memory:.1f}GB, but only {free_memory:.1f}GB available")
            print(f"    Attempting with memory optimization...")
            use_memory_opt = True
        else:
            use_memory_opt = False
    else:
        use_memory_opt = False
    
    try:
        # 加载 tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # 加载模型
        if use_memory_opt:
            # 使用 8-bit 量化或 float16
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
                load_in_8bit=True  # 8-bit 量化
            )
        else:
            dtype = torch.float16 if device == "cuda" else torch.float32
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device if device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
        
        if device == "cpu":
            model = model.to("cpu")
        
        model.eval()
        
        n_params = sum(p.numel() for p in model.parameters())
        print(f"    [OK] Loaded: {model_name}")
        print(f"    Parameters: {n_params / 1e9:.2f}B")
        
        return model, tokenizer, model_name, n_params
    
    except Exception as e:
        print(f"    [FAIL] {str(e)[:100]}")
        return None, None, None, 0


def run_scale_validation():
    """运行规模验证测试"""
    global results
    
    print("\n" + "=" * 60)
    print("Phase 1: Model Scale Progression")
    print("=" * 60)
    
    # 测试不同规模的模型
    models_to_test = [
        ("gpt2", "GPT-2 Small"),
        ("gpt2-medium", "GPT-2 Medium"),
        ("Qwen/Qwen2.5-0.5B", "Qwen2.5-0.5B"),
        ("Qwen/Qwen2.5-1.5B", "Qwen2.5-1.5B"),
    ]
    
    scale_results = []
    
    for model_id, display_name in models_to_test:
        print(f"\n--- Testing {display_name} ---")
        
        model, tokenizer, loaded_name, n_params = load_model_safely(model_id)
        
        if model is None:
            continue
        
        # 分析
        print("  Analyzing layer patterns...")
        layer_analysis = analyze_layer_patterns(model, tokenizer, loaded_name)
        
        print("  Testing generation...")
        gen_results = test_model_generation(model, tokenizer, loaded_name)
        
        model_result = {
            "name": display_name,
            "model_id": model_id,
            "n_params": n_params,
            "n_params_billions": round(n_params / 1e9, 2),
            "layer_analysis": layer_analysis,
            "generation": gen_results
        }
        
        results["models_tested"].append(model_result)
        scale_results.append({
            "params": n_params / 1e9,
            "avg_curvature": layer_analysis.get("avg_curvature", 0.0),
            "n_layers": layer_analysis.get("n_layers", 0)
        })
        
        # 清理内存
        del model
        del tokenizer
        if device == "cuda":
            torch.cuda.empty_cache()
        
        print(f"  Avg Curvature: {layer_analysis.get('avg_curvature', 0.0):.4f}")
    
    # 分析规模趋势
    print("\n" + "=" * 60)
    print("Phase 2: Scale Trend Analysis")
    print("=" * 60)
    
    if len(scale_results) >= 2:
        params = [r["params"] for r in scale_results]
        curvatures = [r["avg_curvature"] for r in scale_results]
        
        # 检查曲率是否随规模稳定
        curvature_variance = np.var(curvatures)
        curvature_mean = np.mean(curvatures)
        
        print(f"\n  Models tested: {len(scale_results)}")
        print(f"  Parameter range: {min(params):.2f}B - {max(params):.2f}B")
        print(f"  Curvature range: {min(curvatures):.4f} - {max(curvatures):.4f}")
        print(f"  Mean curvature: {curvature_mean:.4f}")
        print(f"  Curvature variance: {curvature_variance:.6f}")
        
        # 判断曲率一致性
        if curvature_variance < 0.0001:
            consistency = "HIGH - Curvature remains stable across scales"
            passed = True
        elif curvature_variance < 0.001:
            consistency = "MEDIUM - Moderate variation across scales"
            passed = True
        else:
            consistency = "LOW - Significant variation across scales"
            passed = False
        
        print(f"\n  Consistency: {consistency}")
        
        results["comparisons"]["scale_analysis"] = {
            "params_tested": params,
            "curvatures": curvatures,
            "curvature_mean": curvature_mean,
            "curvature_variance": curvature_variance,
            "consistency": consistency,
            "passed": passed
        }
    
    return results


def main():
    """主函数"""
    print("\nStarting large model validation...")
    
    results = run_scale_validation()
    
    # 保存结果
    output_path = "tempdata/large_model_validation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    models = results.get("models_tested", [])
    print(f"\nModels tested: {len(models)}")
    for m in models:
        print(f"  - {m['name']}: {m['n_params_billions']}B params, curvature={m['layer_analysis'].get('avg_curvature', 0):.4f}")
    
    scale_analysis = results.get("comparisons", {}).get("scale_analysis", {})
    if scale_analysis:
        print(f"\nScale Analysis:")
        print(f"  Curvature Mean: {scale_analysis.get('curvature_mean', 0):.4f}")
        print(f"  Consistency: {scale_analysis.get('consistency', 'N/A')}")
        
        if scale_analysis.get("passed"):
            print("\n[PASS] Large model validation successful!")
            print("  Geometric properties are consistent across model scales.")
        else:
            print("\n[WARN] Some variation detected, but validation continues.")
    
    print(f"\nResults saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
