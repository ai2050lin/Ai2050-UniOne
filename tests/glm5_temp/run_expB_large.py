"""
Phase 36 ExpB: 训练 vs 未训练模型对照
对GLM4/DS7B使用CPU进行未训练模型测试
"""
import sys
sys.path.insert(0, 'tests/glm5')

import numpy as np
import torch
import json
import time
from collections import defaultdict
from sklearn.decomposition import TruncatedSVD

from model_utils import (load_model, get_layers, get_model_info, release_model, MODEL_CONFIGS)
from ccml_phase36_routing_vs_geometry import (collect_all_layer_hs, inject_and_collect,
                                               CONTEXT_TEMPLATES, compute_proj_ratio)

CONCEPTS = ["apple", "dog"]
SOURCE_LAYERS_DICT = {"qwen3": [0, 12, 24], "glm4": [0, 20], "deepseek7b": [0, 12]}
N_TRIALS = 5
EPS = 0.01
N_SVD = 200

def get_W_U_np(model):
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head.weight.detach().cpu().float().numpy()
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'output_layer'):
        return model.transformer.output_layer.weight.detach().cpu().float().numpy()
    else:
        return model.get_output_embeddings().weight.detach().cpu().float().numpy()

def get_subspace_basis(W_U, n_components=200):
    svd = TruncatedSVD(n_components=n_components)
    svd.fit(W_U)
    return svd.components_, svd.singular_values_

def run_test_on_model(model, tokenizer, device, model_name, basis_wu, d_model, n_layers, 
                      source_layers, tag="", use_cpu=False):
    """在给定模型上运行投影比测试"""
    results = {}
    target_layers = list(range(0, n_layers, 4)) + [n_layers - 1]
    target_layers = sorted(set(target_layers))
    
    for concept in CONCEPTS:
        results[concept] = {}
        t0 = time.time()
        baseline_hs, _ = collect_all_layer_hs(
            model, tokenizer, device, concept, CONTEXT_TEMPLATES[0], n_layers)
        if baseline_hs is None:
            print(f"  {tag} {concept}: baseline失败", flush=True)
            continue
        print(f"  {tag} {concept}: baseline collected ({time.time()-t0:.1f}s)", flush=True)
        
        for src_l in source_layers:
            if src_l not in baseline_hs:
                continue
            
            h_scale = np.linalg.norm(baseline_hs[src_l])
            actual_eps = EPS * max(h_scale, 0.01)
            
            np.random.seed(42)
            proj_ratios = defaultdict(list)
            inject_ratios = []
            
            for trial in range(N_TRIALS):
                rand_dir = np.random.randn(d_model).astype(np.float32)
                rand_dir /= np.linalg.norm(rand_dir)
                inject_ratios.append(compute_proj_ratio(rand_dir, basis_wu))
                
                perturbed_hs, _, _ = inject_and_collect(
                    model, tokenizer, device, concept, CONTEXT_TEMPLATES[0],
                    src_l, rand_dir, actual_eps, n_layers)
                
                if perturbed_hs is None:
                    continue
                
                for tgt_l in target_layers:
                    if tgt_l in perturbed_hs and tgt_l in baseline_hs:
                        delta = perturbed_hs[tgt_l] - baseline_hs[tgt_l]
                        if np.linalg.norm(delta) > 1e-10:
                            ratio = compute_proj_ratio(delta, basis_wu)
                            proj_ratios[tgt_l].append(float(ratio))
                
                print(f"  {tag} {concept} L{src_l} trial {trial+1}/{N_TRIALS}", flush=True)
            
            results[concept][str(src_l)] = {
                "inject_ratio": float(np.mean(inject_ratios)),
                "target_ratios": {
                    str(tl): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                    for tl, v in proj_ratios.items() if len(v) > 0
                }
            }
            
            final_key = str(n_layers - 1)
            final_r = results[concept][str(src_l)]["target_ratios"].get(final_key, {}).get("mean", 0)
            inject_r = results[concept][str(src_l)]["inject_ratio"]
            print(f"  {tag} {concept} L{src_l}: inject={inject_r:.4f} → L{n_layers-1}={final_r:.4f} "
                  f"(Δ={final_r - inject_r:+.4f})", flush=True)
    
    return results

def main():
    model_name = sys.argv[1] if len(sys.argv) > 1 else "glm4"
    print(f"Phase 36 ExpB: {model_name}", flush=True)
    
    source_layers = SOURCE_LAYERS_DICT.get(model_name, [0, 12])
    
    # ===== 训练模型 =====
    print("\n=== 训练模型 ===", flush=True)
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    n_layers = info.n_layers
    d_model = info.d_model
    
    W_U = get_W_U_np(model).astype(np.float32)
    print(f"  W_U shape: {W_U.shape}", flush=True)
    
    basis_wu, S_wu = get_subspace_basis(W_U, N_SVD)
    total_energy = np.sum(S_wu[:N_SVD]**2) / np.sum(np.linalg.svd(W_U, compute_uv=False)**2)
    print(f"  W_U subspace: {N_SVD} components, energy fraction≈{total_energy:.3f}", flush=True)
    
    trained_results = run_test_on_model(model, tokenizer, device, model_name, basis_wu, d_model, 
                                         n_layers, source_layers, tag="训练")
    
    trained_W_U = W_U.copy()
    
    # 释放训练模型
    print("\n释放训练模型...", flush=True)
    release_model(model)
    import gc; gc.collect(); torch.cuda.empty_cache()
    print(f"GPU mem after release: {torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
    
    # ===== 未训练模型 (使用CPU!) =====
    print("\n=== 未训练模型 (CPU) ===", flush=True)
    from transformers import AutoModelForCausalLM, AutoConfig
    
    model_path = MODEL_CONFIGS[model_name]["path"]
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    print("  创建随机初始化模型 (CPU)...", flush=True)
    untrained_model = AutoModelForCausalLM.from_config(config, torch_dtype=torch.float32, 
                                                         trust_remote_code=True)
    # 保持模型在CPU上!
    untrained_model.eval()
    cpu_device = torch.device('cpu')
    
    # 复制训练好的lm_head
    if hasattr(untrained_model, 'lm_head') and untrained_model.lm_head is not None:
        with torch.no_grad():
            untrained_model.lm_head.weight.copy_(
                torch.tensor(trained_W_U, dtype=untrained_model.lm_head.weight.dtype))
    
    print(f"  模型在CPU上, 开始测试 (会较慢)...", flush=True)
    
    untrained_results = run_test_on_model(untrained_model, tokenizer, cpu_device, model_name, 
                                           basis_wu, d_model, n_layers, source_layers, tag="未训练")
    
    # ===== 判别分析 =====
    print(f"\n{'='*60}", flush=True)
    print(f"判别分析: routing是学来的还是架构固有的?", flush=True)
    print(f"{'='*60}", flush=True)
    
    for concept in CONCEPTS:
        for src_l_str in [str(sl) for sl in source_layers]:
            tr = trained_results.get(concept, {}).get(src_l_str, {})
            ut = untrained_results.get(concept, {}).get(src_l_str, {})
            
            if not tr or not ut:
                continue
            
            tr_final = tr["target_ratios"].get(str(n_layers-1), {}).get("mean", 0)
            ut_final = ut["target_ratios"].get(str(n_layers-1), {}).get("mean", 0)
            tr_inject = tr["inject_ratio"]
            ut_inject = ut["inject_ratio"]
            
            tr_delta = tr_final - tr_inject
            ut_delta = ut_final - ut_inject
            
            supports_H1 = tr_delta > 0 and abs(tr_delta) > abs(ut_delta) * 1.5
            supports_H0 = abs(tr_delta - ut_delta) < abs(tr_delta) * 0.3 and abs(tr_delta) < 0.01
            
            if supports_H1:
                verdict = "★H1(路由是训练学来的)"
            elif supports_H0:
                verdict = "H0(架构固有/几何效应)"
            else:
                verdict = "不确定"
            
            print(f"  {concept} L{src_l_str}:", flush=True)
            print(f"    训练: inject={tr_inject:.4f} → final={tr_final:.4f} (Δ={tr_delta:+.4f})", flush=True)
            print(f"    未训练: inject={ut_inject:.4f} → final={ut_final:.4f} (Δ={ut_delta:+.4f})", flush=True)
            print(f"    → {verdict}", flush=True)
    
    # 保存结果
    output = {
        "trained": trained_results,
        "untrained": untrained_results,
        "model": model_name
    }
    
    out_path = f"tests/glm5_temp/ccml_phase36_expB_{model_name}_results.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n结果已保存: {out_path}", flush=True)
    
    del untrained_model
    gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
