"""
CCXXIV(372): 信息压缩策略的层间演化 — 破解"为什么DS7B压缩更激进"
=====================================================================

★★★★★ 核心问题:
  CCXXIII确认: DS7B的暗物质维度坍缩不是8bit量化导致
  新问题: 为什么DS7B的W_U矩阵更"低秩"(Top50 energy=0.232 vs 0.092-0.137)?
  深层问题: 什么决定了模型的"信息压缩策略"?

★★★★★ 关键假设:
  H1: 压缩策略由训练数据分布决定 (更均匀的数据→更分散的谱)
  H2: 压缩策略由模型深度/宽度比决定 (深层/窄模型→更集中的谱)
  H3: 压缩策略是"信息瓶颈"的结果 (更小的d_model/更大的vocab→更集中)

★★★★★ 实验设计:
  Exp1: 三模型各层的残差流PCA分析 — 层间信息压缩的动态过程
  Exp2: W_U奇异值谱与d_model/vocab比例的关系
  Exp3: 信息瓶颈理论验证: 互信息I(X;Y)与暗物质维度的关系

★★★★★ 核心目标:
  如果能建立 "模型架构 → 信息压缩策略 → 暗物质维度" 的因果链,
  就找到了语言数学结构的一个关键机制

用法:
  python ccxxiv_compression_strategy.py --model qwen3 --exp 1
  python ccxxiv_compression_strategy.py --model qwen3 --exp 2
  python ccxxiv_compression_strategy.py --exp all
"""

import argparse, os, sys, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
from sklearn.decomposition import PCA
from scipy.linalg import svd

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model, get_W_U

TEMP = Path("tests/glm5_temp")

# 50概念定义 (复用)
CONCEPT_CATEGORIES_50 = {
    "animal": ["dog", "cat", "bird", "fish", "snake", "horse", "cow", "pig", "sheep", "monkey"],
    "emotion": ["happy", "sad", "angry", "scared", "calm", "excited", "proud", "ashamed", "surprised", "disgusted"],
    "profession": ["doctor", "teacher", "chef", "artist", "lawyer", "farmer", "engineer", "soldier", "scientist", "writer"],
    "nature": ["mountain", "ocean", "forest", "desert", "river", "lake", "island", "valley", "cave", "volcano"],
    "object": ["car", "house", "book", "phone", "chair", "table", "door", "window", "knife", "lamp"],
}


# ================================================================
# Exp1: 各层残差流PCA — 层间信息压缩动态
# ================================================================

def run_exp1(model_name):
    """分析各层残差流的信息压缩动态"""
    print(f'\n{"="*60}')
    print(f'Exp1: Layer-wise Residual Stream PCA ({model_name})')
    print(f'{"="*60}')
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    
    # 收集各层的概念表示
    layers_list = get_layers(model)
    embed_layer = model.get_input_embeddings()
    
    # 采样层 (每2层一个)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 15)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    sample_layers = sorted(set(sample_layers))
    print(f'  Testing layers: {sample_layers}')
    
    # 收集概念向量
    all_concept_vectors = {}  # {layer_idx: {concept_name: vector}}
    
    for domain_name, words in CONCEPT_CATEGORIES_50.items():
        for word in words:
            prompt = f"The word is {word}"
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            
            with torch.no_grad():
                inputs_embeds = embed_layer(input_ids)
                
                captured = {}
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0].detach().float().cpu().numpy()
                        else:
                            captured[key] = output.detach().float().cpu().numpy()
                    return hook
                
                hooks = []
                for li in sample_layers:
                    if li < len(layers_list):
                        hooks.append(layers_list[li].register_forward_hook(make_hook(f"L{li}")))
                
                _ = model(inputs_embeds=inputs_embeds)
                
                for h in hooks:
                    h.remove()
                
                for li in sample_layers:
                    key = f"L{li}"
                    if key in captured:
                        vec = captured[key][0, -1, :]
                        concept_name = f"{domain_name}_{word}"
                        if li not in all_concept_vectors:
                            all_concept_vectors[li] = {}
                        all_concept_vectors[li][concept_name] = vec
    
    # 释放模型
    release_model(model)
    
    # 分析各层的PCA
    layer_pca = {}
    
    for li in sample_layers:
        if li not in all_concept_vectors:
            continue
        
        cv = all_concept_vectors[li]
        if len(cv) < 5:
            continue
        
        X = np.array(list(cv.values()))
        concept_names = list(cv.keys())
        
        # PCA
        pca = PCA()
        pca.fit(X)
        
        # 有效维度
        cum_var = np.cumsum(pca.explained_variance_ratio_)
        eff_dim_90 = int(np.argmax(cum_var >= 0.90) + 1)
        eff_dim_95 = int(np.argmax(cum_var >= 0.95) + 1)
        
        # Top k 方差占比
        top1 = pca.explained_variance_ratio_[0]
        top3 = np.sum(pca.explained_variance_ratio_[:3])
        top5 = np.sum(pca.explained_variance_ratio_[:5])
        top10 = np.sum(pca.explained_variance_ratio_[:10])
        
        # 奇异值谱的幂律指数
        s = pca.singular_values_
        log_s = np.log10(s[s > 0])
        log_k = np.log10(np.arange(1, len(log_s) + 1))
        n_fit = min(len(log_s) // 2, 25)  # 用前半部分拟合
        if n_fit > 2:
            coeffs = np.polyfit(log_k[:n_fit], log_s[:n_fit], 1)
            alpha = -coeffs[0]
        else:
            alpha = 0
        
        layer_pca[li] = {
            "n_concepts": len(cv),
            "eff_dim_90": eff_dim_90,
            "eff_dim_95": eff_dim_95,
            "top1_var": float(top1),
            "top3_var": float(top3),
            "top5_var": float(top5),
            "top10_var": float(top10),
            "power_law_alpha": float(alpha),
            "singular_values_top20": s[:20].tolist(),
        }
        
        print(f'  L{li:2d}: eff_dim_95={eff_dim_95}, top1={top1:.3f}, '
              f'top3={top3:.3f}, top10={top10:.3f}, α={alpha:.3f}')
    
    # 分析压缩策略的层间演化
    print(f'\n{"="*60}')
    print(f'  Compression Strategy Evolution:')
    print(f'{"="*60}')
    
    # 检测"压缩转折点": top1_var开始急剧上升的层
    layers_sorted = sorted(layer_pca.keys())
    top1_vals = [layer_pca[li]["top1_var"] for li in layers_sorted]
    
    # 计算top1的增长率
    growth_rates = []
    for i in range(1, len(top1_vals)):
        if top1_vals[i-1] > 0:
            growth_rates.append(top1_vals[i] / top1_vals[i-1])
        else:
            growth_rates.append(0)
    
    # 找到最大增长率的位置
    if growth_rates:
        max_growth_idx = np.argmax(growth_rates)
        compression_start_layer = layers_sorted[max_growth_idx]
        print(f'  Compression start layer: L{compression_start_layer} '
              f'(growth rate = {growth_rates[max_growth_idx]:.3f})')
    
    # 保存结果
    result = {
        "model": model_name,
        "n_layers": n_layers,
        "d_model": model_info.d_model,
        "vocab_size": model_info.vocab_size,
        "layer_pca": {str(k): v for k, v in layer_pca.items()},
    }
    
    outpath = TEMP / f"ccxxiv_{model_name}_layer_pca.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'\n  Saved to {outpath}')
    
    return result


# ================================================================
# Exp2: W_U谱与模型架构参数的关系
# ================================================================

def run_exp2():
    """分析W_U奇异值谱与d_model/vocab比例的关系"""
    print(f'\n{"="*60}')
    print(f'Exp2: W_U Spectrum vs Architecture Parameters')
    print(f'{"="*60}')
    
    all_results = {}
    
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        path = TEMP / f"ccxxiii_{model_name}_wu_svd.json"
        if not path.exists():
            print(f'  {model_name}: W_U SVD data not found, skipping')
            continue
        
        with open(path, "r", encoding="utf-8") as f:
            wu_data = json.load(f)
        
        d_model = wu_data["d_model"]
        vocab_size = wu_data["vocab_size"]
        
        # 架构参数
        model, tokenizer, device = load_model(model_name)
        model_info = get_model_info(model, model_name)
        
        params = {
            "d_model": d_model,
            "vocab_size": vocab_size,
            "n_layers": model_info.n_layers,
            "intermediate_size": model_info.intermediate_size,
            "d_model/vocab": d_model / vocab_size,
            "vocab/d_model": vocab_size / d_model,
            "W_U_rank_95": wu_data["eff_ranks"]["rank_950"],
            "W_U_rank_99": wu_data["eff_ranks"]["rank_990"],
            "W_U_top50_energy": wu_data.get("singular_values_top200", [0]*200),
            "power_law_alpha": wu_data["power_law_alpha"],
            "condition_number": wu_data["condition_number"],
            "n_unique_values": wu_data["n_unique_values"],
        }
        
        # 计算信息瓶颈指标
        # IB理论: I(X;Y) = I(X;Z) - I(X;Z|Y) ≈ rank(W_U) × log(SNR)
        # SNR ≈ s[0]^2 / s[-1]^2
        s = wu_data.get("singular_values_top200", [])
        if len(s) > 10:
            s = np.array(s)
            snr = s[0]**2 / max(s[-1]**2, 1e-10)
            info_capacity = wu_data["eff_ranks"]["rank_950"] * np.log2(max(snr, 1.01))
        else:
            snr = 0
            info_capacity = 0
        
        params["SNR"] = float(snr)
        params["info_capacity_bits"] = float(info_capacity)
        params["rank95_per_dmodel"] = wu_data["eff_ranks"]["rank_950"] / d_model
        
        all_results[model_name] = params
        
        print(f'\n  {model_name}:')
        print(f'    d_model={d_model}, vocab={vocab_size}, layers={model_info.n_layers}')
        print(f'    vocab/d_model = {vocab_size/d_model:.1f}')
        print(f'    W_U rank95 = {params["W_U_rank_95"]} ({params["rank95_per_dmodel"]:.3f} per d_model)')
        print(f'    SNR = {snr:.2e}')
        print(f'    Info capacity = {info_capacity:.1f} bits')
        print(f'    Power-law α = {params["power_law_alpha"]:.3f}')
        
        release_model(model)
    
    # 跨模型比较
    print(f'\n{"="*60}')
    print(f'  Cross-Model Comparison:')
    print(f'{"="*60}')
    
    if len(all_results) >= 2:
        # 检验H3: vocab/d_model比例越大→谱越集中
        for name, p in all_results.items():
            compression_level = p.get("power_law_alpha", 0)
            print(f'  {name}: vocab/d_model={p["vocab/d_model"]:.1f}, '
                  f'α={compression_level:.3f}, rank95/d={p["rank95_per_dmodel"]:.3f}')
    
    # 保存
    outpath = TEMP / "ccxxiv_architecture_comparison.json"
    with open(outpath, "w", encoding="utf-8") as f:
        # 移除大数组
        save_results = {}
        for name, p in all_results.items():
            save_results[name] = {k: v for k, v in p.items() 
                                  if not isinstance(v, list)}
        json.dump(save_results, f, indent=2, ensure_ascii=False)
    print(f'\n  Saved to {outpath}')
    
    return all_results


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3",
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="1",
                       choices=["1", "2", "all"])
    args = parser.parse_args()
    
    if args.exp == "1":
        run_exp1(args.model)
    elif args.exp == "2":
        run_exp2()
    elif args.exp == "all":
        for name in ["qwen3", "glm4", "deepseek7b"]:
            run_exp1(name)
        run_exp2()


if __name__ == "__main__":
    main()
