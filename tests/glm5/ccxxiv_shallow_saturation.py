"""
CCXXIV(324): 浅层饱和因果分析
======================================================================
CCXX发现: Qwen3/GLM4 L0 dim≤8(不随n_rel增长), 但DS7B L0遵循单纯形!
为什么非蒸馏模型L0饱和? 浅层使用了什么固定特征?

假设:
  1. 位置编码/注意力模式占据大量维度 → 挤压语义空间
  2. 嵌入层将token映射到低秩空间 → d_embed < d_model
  3. 浅层layer norm权重有方向偏好

实验设计:
  1. 分析L0残差的SVD: 有多少维度被"固定特征"占据?
  2. 比较不同词汇在L0的残差: 语义差异 vs 非语义差异(位置/注意力)
  3. 检查位置编码的维度占用
  4. 比较embedding矩阵的有效秩 vs d_model
  5. LayerNorm权重分析: 哪些维度被放大/抑制?

用法:
  python ccxxiv_shallow_saturation.py --model qwen3
  python ccxxiv_shallow_saturation.py --model glm4
  python ccxxiv_shallow_saturation.py --model deepseek7b
"""
import argparse, os, sys, json, gc
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy import stats

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxxiv_shallow_saturation_log.txt"

# 多种语义关系的词汇
RELATION_WORDS = {
    "habitat_land": ["dog", "cat", "lion", "tiger", "horse", "cow", "fox", "deer"],
    "habitat_ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "crab", "seal", "squid"],
    "habitat_sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "falcon", "swallow"],
    "food_meat": ["beef", "pork", "chicken", "lamb", "venison", "bacon", "ham", "steak"],
    "food_fruit": ["apple", "banana", "orange", "grape", "mango", "peach", "pear", "cherry"],
    "food_grain": ["rice", "wheat", "corn", "oats", "barley", "rye", "millet", "quinoa"],
    "material_metal": ["iron", "steel", "copper", "gold", "silver", "bronze", "tin", "lead"],
    "material_wood": ["oak", "pine", "cedar", "maple", "birch", "elm", "ash", "walnut"],
    "material_stone": ["granite", "marble", "slate", "limestone", "sandstone", "basalt", "quartz", "obsidian"],
    "color_warm": ["red", "orange", "yellow", "pink", "crimson", "scarlet", "amber", "gold"],
    "color_cool": ["blue", "green", "purple", "teal", "indigo", "violet", "cyan", "turquoise"],
    "size_big": ["mountain", "ocean", "continent", "planet", "galaxy", "elephant", "whale", "building"],
    "size_small": ["ant", "grain", "atom", "pixel", "drop", "seed", "flea", "mote"],
}

TEMPLATE = "The {}"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    log(f"\n{'='*70}\nCCXXIV(324): 浅层饱和因果分析 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {}
    
    # ===== Step 1: Embedding矩阵的有效秩 =====
    log("\n--- Step 1: Embedding矩阵的有效秩 ---")
    
    embed_weight = model.get_input_embeddings().weight.detach().float().cpu().numpy()
    vocab_size = embed_weight.shape[0]
    
    # SVD of embedding
    # 太大, 只取前500个奇异值
    k_svd = min(500, min(embed_weight.shape) - 1)
    from scipy.sparse.linalg import svds
    U_emb, S_emb, Vt_emb = svds(embed_weight, k=k_svd)
    S_emb = np.sort(S_emb)[::-1]
    
    # 有效秩(F>10阈值)
    total_var = np.sum(S_emb ** 2)
    eff_dim_F10 = np.sum(S_emb > 10)
    eff_dim_1pct = np.sum((S_emb ** 2) / total_var > 0.01)
    eff_dim_01pct = np.sum((S_emb ** 2) / total_var > 0.001)
    
    # Shannon熵定义的有效秩
    p = S_emb ** 2 / total_var
    entropy = -np.sum(p * np.log(p + 1e-20))
    eff_dim_entropy = np.exp(entropy)
    
    results["embedding_analysis"] = {
        "vocab_size": vocab_size,
        "d_model": d_model,
        "top10_sv": [round(float(s), 2) for s in S_emb[:10]],
        "eff_dim_F10": int(eff_dim_F10),
        "eff_dim_1pct": int(eff_dim_1pct),
        "eff_dim_01pct": int(eff_dim_01pct),
        "eff_dim_entropy": round(float(eff_dim_entropy), 2),
    }
    
    log(f"  vocab={vocab_size}, d_model={d_model}")
    log(f"  top10 SV: {[round(float(s), 2) for s in S_emb[:10]]}")
    log(f"  eff_dim(F>10)={eff_dim_F10}, eff_dim(1%)={eff_dim_1pct}, eff_dim(0.1%)={eff_dim_01pct}")
    log(f"  eff_dim(entropy)={eff_dim_entropy:.2f}")
    
    # ===== Step 2: L0残差的SVD — "固定特征"维度 =====
    log("\n--- Step 2: L0残差的SVD ---")
    
    # 收集大量不同词汇在L0的残差
    test_layers_list = [0, 1, n_layers // 2, n_layers - 1]
    layer_resids = {li: [] for li in test_layers_list}
    layer_labels = {li: [] for li in test_layers_list}
    
    for rel_name, words in RELATION_WORDS.items():
        for word in words:
            prompt = TEMPLATE.format(word)
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            seq_len = toks.input_ids.shape[1]
            last_pos = seq_len - 1
            
            captured = {}
            def mk_hook(k):
                def hook(m, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    captured[k] = o[0, last_pos, :].detach().float().cpu().numpy()
                return hook
            
            hooks = [layers[li].register_forward_hook(mk_hook(f"L{li}")) for li in test_layers_list]
            with torch.no_grad():
                _ = model(**toks)
            for h in hooks:
                h.remove()
            
            for li in test_layers_list:
                if f"L{li}" in captured:
                    layer_resids[li].append(captured[f"L{li}"])
                    layer_labels[li].append(rel_name)
    
    log(f"  收集了 {len(layer_resids[0])} 个词汇残差, {len(set(layer_labels[0]))} 种关系")
    
    # 每层的SVD分析
    for li in test_layers_list:
        X = np.array(layer_resids[li])
        if X.shape[0] < 10:
            continue
        
        mean = X.mean(axis=0)
        X_centered = X - mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        total_var = np.sum(S ** 2)
        explained_ratio = (S ** 2) / total_var
        
        # 有效维度
        eff_dim_F10 = np.sum(S > 10)
        eff_dim_1pct = np.sum(explained_ratio > 0.01)
        
        # Shannon熵有效维度
        p = explained_ratio
        entropy = -np.sum(p * np.log(p + 1e-20))
        eff_dim_entropy = np.exp(entropy)
        
        # cumsum
        cum_var = np.cumsum(explained_ratio)
        
        results[f"layer_svd_L{li}"] = {
            "layer": li,
            "n_samples": X.shape[0],
            "top10_sv": [round(float(s), 2) for s in S[:10]],
            "top10_explained": [round(float(r), 4) for r in explained_ratio[:10]],
            "eff_dim_F10": int(eff_dim_F10),
            "eff_dim_1pct": int(eff_dim_1pct),
            "eff_dim_entropy": round(float(eff_dim_entropy), 2),
            "cum_var_5": round(float(cum_var[4]), 4) if len(cum_var) >= 5 else None,
            "cum_var_10": round(float(cum_var[9]), 4) if len(cum_var) >= 10 else None,
            "cum_var_20": round(float(cum_var[19]), 4) if len(cum_var) >= 20 else None,
            "cum_var_50": round(float(cum_var[49]), 4) if len(cum_var) >= 50 else None,
        }
        
        log(f"  L{li}: eff_dim(F>10)={eff_dim_F10}, eff_dim(1%)={eff_dim_1pct}, eff_dim(entropy)={eff_dim_entropy:.2f}")
        log(f"  L{li}: cum_var[5]={cum_var[4]:.4f}, [10]={cum_var[9]:.4f}, [20]={cum_var[19]:.4f}, [50]={cum_var[49]:.4f}")
    
    # ===== Step 3: 位置编码分析 =====
    log("\n--- Step 3: 位置编码分析 ---")
    
    # 用同一个词在不同位置, 看位置编码占据多少维度
    test_word = "elephant"
    pos_resids_L0 = []
    
    # 构造不同长度的前缀
    prefixes = [
        "",
        "The ",
        "The big ",
        "The big red ",
        "The big red old ",
        "The big red old fat ",
        "The big red old fat nice ",
    ]
    
    for prefix in prefixes:
        prompt = prefix + test_word
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = toks.input_ids.shape[1]
        last_pos = seq_len - 1
        
        captured = {}
        def mk_hook(k):
            def hook(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured[k] = o[0, last_pos, :].detach().float().cpu().numpy()
            return hook
        
        hooks = [layers[0].register_forward_hook(mk_hook("L0"))]
        with torch.no_grad():
            _ = model(**toks)
        for h in hooks:
            h.remove()
        
        if "L0" in captured:
            pos_resids_L0.append({"prefix": prefix, "seq_len": seq_len, "resid": captured["L0"]})
    
    # 分析位置差异
    if len(pos_resids_L0) >= 3:
        pos_vecs = np.array([r["resid"] for r in pos_resids_L0])
        pos_mean = pos_vecs.mean(axis=0)
        pos_centered = pos_vecs - pos_mean
        
        # 两两余弦
        pos_cos = []
        for i in range(len(pos_vecs)):
            for j in range(i+1, len(pos_vecs)):
                v1 = pos_vecs[i] - pos_mean
                v2 = pos_vecs[j] - pos_mean
                n1 = np.linalg.norm(v1)
                n2 = np.linalg.norm(v2)
                if n1 > 1e-10 and n2 > 1e-10:
                    pos_cos.append(float(np.dot(v1, v2) / (n1 * n2)))
        
        results["position_encoding"] = {
            "n_positions": len(pos_resids_L0),
            "position_cos_range": [round(min(pos_cos), 4), round(max(pos_cos), 4)] if pos_cos else [],
            "position_cos_mean": round(float(np.mean(pos_cos)), 4) if pos_cos else 0,
            "seq_lengths": [r["seq_len"] for r in pos_resids_L0],
        }
        
        log(f"  位置编码: {len(pos_resids_L0)} 个位置, cos范围=[{min(pos_cos):.4f}, {max(pos_cos):.4f}]")
    
    # ===== Step 4: LayerNorm权重分析 =====
    log("\n--- Step 4: LayerNorm权重分析 ---")
    
    for li in [0, 1, n_layers // 2, n_layers - 1]:
        layer = layers[li]
        
        # 输入LayerNorm
        for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                if hasattr(ln, "weight"):
                    ln_w = ln.weight.detach().float().cpu().numpy()
                    
                    # 哪些维度被放大/抑制?
                    high_dims = np.where(ln_w > 2.0)[0]
                    low_dims = np.where(ln_w < 0.5)[0]
                    
                    # 有效维度(基于权重分布)
                    sorted_w = np.sort(ln_w)[::-1]
                    total_w = np.sum(ln_w)
                    cum_w = np.cumsum(sorted_w) / total_w
                    eff_dim_90 = np.searchsorted(cum_w, 0.9) + 1
                    
                    results[f"layernorm_L{li}"] = {
                        "layer": li,
                        "type": "input_layernorm",
                        "mean": round(float(ln_w.mean()), 4),
                        "std": round(float(ln_w.std()), 4),
                        "min": round(float(ln_w.min()), 4),
                        "max": round(float(ln_w.max()), 4),
                        "n_high": len(high_dims),
                        "n_low": len(low_dims),
                        "eff_dim_90pct": int(eff_dim_90),
                    }
                    
                    log(f"  L{li} input_ln: mean={ln_w.mean():.4f}, std={ln_w.std():.4f}, "
                        f"high(>2)={len(high_dims)}, low(<0.5)={len(low_dims)}, eff_dim_90={eff_dim_90}")
                break
        
        # 后注意力LayerNorm
        for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                if hasattr(ln, "weight"):
                    ln_w = ln.weight.detach().float().cpu().numpy()
                    
                    high_dims = np.where(ln_w > 2.0)[0]
                    low_dims = np.where(ln_w < 0.5)[0]
                    
                    sorted_w = np.sort(ln_w)[::-1]
                    total_w = np.sum(ln_w)
                    cum_w = np.cumsum(sorted_w) / total_w
                    eff_dim_90 = np.searchsorted(cum_w, 0.9) + 1
                    
                    results[f"post_attn_layernorm_L{li}"] = {
                        "layer": li,
                        "type": "post_attn_layernorm",
                        "mean": round(float(ln_w.mean()), 4),
                        "std": round(float(ln_w.std()), 4),
                        "min": round(float(ln_w.min()), 4),
                        "max": round(float(ln_w.max()), 4),
                        "n_high": len(high_dims),
                        "n_low": len(low_dims),
                        "eff_dim_90pct": int(eff_dim_90),
                    }
                    
                    log(f"  L{li} post_attn_ln: mean={ln_w.mean():.4f}, std={ln_w.std():.4f}, "
                        f"high(>2)={len(high_dims)}, low(<0.5)={len(low_dims)}, eff_dim_90={eff_dim_90}")
                break
    
    # ===== Step 5: 语义 vs 非语义方差分解 =====
    log("\n--- Step 5: 语义 vs 非语义方差分解 ---")
    
    # 对于L0, 将方差分解为:
    # - 语义方差(同类词之间的差异)
    # - 组间方差(不同关系之间的差异)
    # F-test: 组间/组内方差比 → 语义信息是否显著
    
    for li in [0, 1, n_layers // 2, n_layers - 1]:
        if len(layer_resids[li]) < 10:
            continue
        
        X = np.array(layer_resids[li])
        labels = layer_labels[li]
        mean_all = X.mean(axis=0)
        
        # 组间方差(不同关系之间的差异)
        group_means = {}
        group_counts = {}
        for i, lab in enumerate(labels):
            if lab not in group_means:
                group_means[lab] = np.zeros(d_model)
                group_counts[lab] = 0
            group_means[lab] += X[i]
            group_counts[lab] += 1
        
        for lab in group_means:
            group_means[lab] /= group_counts[lab]
        
        between_var = np.sum([group_counts[lab] * np.linalg.norm(group_means[lab] - mean_all)**2 
                             for lab in group_means])
        
        # 组内方差(同类词之间的差异)
        within_var = np.sum([np.linalg.norm(X[i] - group_means[labels[i]])**2 
                            for i in range(len(labels))])
        
        # F比 = 组间/组内 (归一化)
        n_groups = len(group_means)
        n_total = len(labels)
        F_ratio = (between_var / (n_groups - 1)) / (within_var / (n_total - n_groups)) if within_var > 0 else 0
        
        # 语义方差占比
        total_var = between_var + within_var
        semantic_ratio = between_var / total_var if total_var > 0 else 0
        
        results[f"variance_decomposition_L{li}"] = {
            "layer": li,
            "n_groups": n_groups,
            "n_total": n_total,
            "between_var": round(float(between_var), 2),
            "within_var": round(float(within_var), 2),
            "F_ratio": round(float(F_ratio), 2),
            "semantic_ratio": round(float(semantic_ratio), 4),
        }
        
        log(f"  L{li}: semantic_ratio={semantic_ratio:.4f}, F_ratio={F_ratio:.2f}, "
            f"between={between_var:.1f}, within={within_var:.1f}")
    
    # ===== Step 6: L0到L1的变化 — "解冻"分析 =====
    log("\n--- Step 6: L0→L1变化分析 ---")
    
    # L0残差 vs L1残差的差异
    if len(layer_resids[0]) >= 10 and len(layer_resids[1]) >= 10:
        X0 = np.array(layer_resids[0])
        X1 = np.array(layer_resids[1])
        
        # L0的SVD
        mean0 = X0.mean(axis=0)
        X0c = X0 - mean0
        U0, S0, Vt0 = np.linalg.svd(X0c, full_matrices=False)
        
        # L1的SVD
        mean1 = X1.mean(axis=0)
        X1c = X1 - mean1
        U1, S1, Vt1 = np.linalg.svd(X1c, full_matrices=False)
        
        # PC对齐
        pc_align = []
        for k in range(min(10, Vt0.shape[0], Vt1.shape[0])):
            cos_val = float(np.abs(np.dot(Vt0[k], Vt1[k])))
            pc_align.append(round(cos_val, 4))
        
        # 子空间重叠(CCA或Grassmann距离)
        # 简单方法: 前5个PC的子空间投影
        k_sub = 5
        P0 = Vt0[:k_sub].T @ Vt0[:k_sub]  # L0前5个PC的投影矩阵
        P1 = Vt1[:k_sub].T @ Vt1[:k_sub]  # L1前5个PC的投影矩阵
        
        # 子空间重叠 = ||P0 @ P1||_F / sqrt(k)
        overlap = float(np.linalg.norm(P0 @ P1, 'fro')) / np.sqrt(k_sub)
        
        results["L0_to_L1"] = {
            "pc_alignment": pc_align,
            "subspace_overlap_k5": round(overlap, 4),
            "L0_eff_dim_F10": int(np.sum(S0 > 10)),
            "L1_eff_dim_F10": int(np.sum(S1 > 10)),
            "L0_eff_dim_entropy": round(float(np.exp(float(-np.sum((S0**2/np.sum(S0**2))*np.log(S0**2/np.sum(S0**2)+1e-20))))), 2),
            "L1_eff_dim_entropy": round(float(np.exp(float(-np.sum((S1**2/np.sum(S1**2))*np.log(S1**2/np.sum(S1**2)+1e-20))))), 2),
        }
        
        log(f"  L0→L1 PC对齐: {pc_align}")
        log(f"  L0→L1 子空间重叠(K=5): {overlap:.4f}")
        log(f"  L0 eff_dim(F>10)={np.sum(S0>10)}, L1 eff_dim(F>10)={np.sum(S1>10)}")
    
    # 保存结果
    out_path = TEMP / f"ccxxiv_shallow_saturation_{model_name}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"model": model_name, "d_model": d_model, "n_layers": n_layers, "results": results}, f, ensure_ascii=False, indent=2)
    log(f"\n结果保存到: {out_path}")
    
    release_model(model)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    if args.model == "qwen3":
        with open(LOG, "w", encoding="utf-8") as f:
            f.write("")
    
    run(args.model)
