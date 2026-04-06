#!/usr/bin/env python3
"""
P42: Residual Stream Intervention（Stage688）

Instead of ablation (which is fragile), inject noise into residual stream
at different layers and measure impact on generation probability.

This is more robust than zeroing out layers because:
- Residual connections bypass ablated layers anyway
- Noise injection directly tests information flow through specific layers

Method:
1. At each layer, add Gaussian noise to residual stream
2. Measure log-prob degradation
3. Identify which layers are most sensitive to perturbation

Usage: python tests/glm5/stage688_layer_ablation.py <model_name>
"""
import sys, math, statistics, time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
    "gemma4": _Path(r"D:\develop\model\hub\models--google--gemma-4-E2B-it\snapshots\4742fe843cc01b9aed62122f6e0ddd13ea48b3d3"),
}

def load_model(model_name):
    path = MODEL_MAP.get(model_name, _Path(model_name))
    print(f"  loading model: {path.name}")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, trust_remote_code=True,
        torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()
    return model, tokenizer

def compute_lp_with_noise(model, tokenizer, test_cases, noise_layer=None, noise_scale=0.1):
    """Compute average log-prob with optional noise injection at a specific layer"""
    model_device = next(model.parameters()).device
    
    hook_handle = [None]
    
    if noise_layer is not None:
        # Register hook to inject noise at specific layer
        def noise_hook(module, input, output):
            if isinstance(output, tuple):
                hs = output[0]
                noise = torch.randn_like(hs) * noise_scale * hs.std()
                return (hs + noise,) + output[1:]
            else:
                noise = torch.randn_like(output) * noise_scale * output.std()
                return output + noise
        
        # Find the layer and register hook
        for name, module in model.named_modules():
            if hasattr(module, 'self_attn') and hasattr(module, 'mlp'):
                layer_idx = None
                for prefix in ['model.layers.', 'layers.', 'model.model.layers.', 'transformer.h.']:
                    if name.startswith(prefix):
                        layer_idx = int(name[len(prefix):].split('.')[0])
                        break
                if layer_idx is not None and layer_idx == noise_layer:
                    hook_handle[0] = module.register_forward_hook(noise_hook)
                    break
    
    logprobs = []
    for prompt, expected in test_cases:
        tokens = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=64)
        tokens = tokens.to(model_device)
        next_id = tokenizer.encode(expected, add_special_tokens=False)[-1]
        
        with torch.no_grad():
            outputs = model(tokens)
            logits = outputs.logits[0, -1, :].float()
            lp = F.log_softmax(logits, dim=-1)[next_id].item()
        logprobs.append(lp)
    
    if hook_handle[0] is not None:
        hook_handle[0].remove()
    
    return statistics.mean(logprobs)

def count_layers(model):
    """Count transformer layers"""
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'self_attn') and hasattr(module, 'mlp'):
            count += 1
    return count

def run_sensitivity_scan(model, tokenizer, test_cases, sample_layers=None):
    """Scan sensitivity across layers"""
    total = count_layers(model)
    print(f"  total layers: {total}")
    
    if sample_layers is None:
        # Sample 8 evenly spaced layers
        if total <= 8:
            sample_layers = list(range(total))
        else:
            sample_layers = [round(i * (total - 1) / 7) for i in range(8)]
    
    # Baseline
    print("  computing baseline...")
    baseline = compute_lp_with_noise(model, tokenizer, test_cases)
    print(f"  baseline avg_lp = {baseline:.4f}")
    
    # Noise scale
    noise_scales = [0.05, 0.1, 0.2]
    
    results = {}
    for ns in noise_scales:
        print(f"\n  noise_scale = {ns}")
        layer_sensitivities = []
        
        for layer_idx in sample_layers:
            perturbed = compute_lp_with_noise(model, tokenizer, test_cases, 
                                               noise_layer=layer_idx, noise_scale=ns)
            sensitivity = perturbed - baseline
            layer_sensitivities.append((layer_idx, perturbed, sensitivity))
        
        results[ns] = layer_sensitivities
        
        # Find most sensitive layer
        most_sensitive = min(layer_sensitivities, key=lambda x: x[2])
        least_sensitive = max(layer_sensitivities, key=lambda x: x[2])
        print(f"    most sensitive: L{most_sensitive[0]} (delta={most_sensitive[2]:+.4f})")
        print(f"    least sensitive: L{least_sensitive[0]} (delta={least_sensitive[2]:+.4f})")
        print(f"    avg sensitivity: {statistics.mean([x[2] for x in layer_sensitivities]):+.4f}")
    
    return results, baseline, sample_layers

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "qwen3"
    
    print(f"\n{'='*60}")
    print(f"  P42: Residual Stream Sensitivity Analysis")
    print(f"  model: {model_name}")
    print(f"{'='*60}")
    
    model, tokenizer = load_model(model_name)
    
    test_cases = [
        ("The cat sat on the", " mat"),
        ("Paris is the capital of", " France"),
        ("If it rains then the ground", " gets"),
        ("She studied hard because she", " wanted"),
        ("She carefully folded the", " origami"),
        ("The orchestra played a", " beautiful"),
        ("The quick brown fox", " jumps"),
        ("She has been working", " on"),
        ("Yesterday it rained", " heavily"),
        ("She will finish by", " Friday"),
        ("Two plus two equals", " four"),
        ("The derivative of x squared", " is"),
    ]
    
    t0 = time.time()
    results, baseline, layers = run_sensitivity_scan(model, tokenizer, test_cases)
    elapsed = time.time() - t0
    
    # Summary table
    print(f"\n{'='*60}")
    print(f"  P42 Sensitivity Map (noise_scale=0.1)")
    print(f"{'='*60}")
    
    if 0.1 in results:
        data = results[0.1]
        print(f"  {'Layer':>6s} {'AvgLP':>8s} {'Delta':>8s} {'Sensitivity':>10s}")
        for layer_idx, lp, delta in data:
            sens = "HIGH" if delta < -0.5 else ("MEDIUM" if delta < -0.2 else "LOW")
            print(f"  L{layer_idx:>4d} {lp:8.4f} {delta:+8.4f} {sens:>10s}")
    
    print(f"\n  elapsed: {elapsed:.1f}s")
    
    # Cross noise-scale consistency
    print(f"\n{'='*60}")
    print(f"  Cross-scale consistency")
    print(f"{'='*60}")
    
    # Check if most/least sensitive layers are consistent across noise scales
    for ns in results:
        most_sens_layer = min(results[ns], key=lambda x: x[2])[0]
        print(f"  noise={ns:.2f}: most sensitive = L{most_sens_layer}")
    
    # DS7B deep fusion zone validation
    if model_name == "deepseek7b":
        total = count_layers(model)
        if 0.1 in results:
            data = results[0.1]
            # Check if mid-layers (L5-L26) are more sensitive
            mid_layers = [(l, d) for l, _, d in data if total//5 <= l <= 4*total//5]
            edge_layers = [(l, d) for l, _, d in data if l < total//5 or l > 4*total//5]
            if mid_layers and edge_layers:
                mid_avg = statistics.mean([d for _, d in mid_layers])
                edge_avg = statistics.mean([d for _, d in edge_layers])
                print(f"\n  [DS7B] mid-layer avg sensitivity: {mid_avg:+.4f}")
                print(f"  [DS7B] edge-layer avg sensitivity: {edge_avg:+.4f}")
                if mid_avg < edge_avg:
                    print(f"  -> [INV-338] Mid layers more sensitive (deep fusion zone) [OK]")
                else:
                    print(f"  -> [INV-338] Edge layers more sensitive")
    
    del model
    import gc
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
