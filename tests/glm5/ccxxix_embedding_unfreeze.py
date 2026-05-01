"""
CCXXIX(329): Embedding→L0→L1信息流追踪 - 语义信息如何"解冻"
======================================================================
CCXXIV发现: L0仅17-22%语义方差, L1立即"解冻"(eff_dim翻倍)。
关键问题: 语义信息如何从低秩Embedding通过注意力层"解冻"?

实验设计:
  1. 语义token的embedding→L0→L1→L2→Lmid的残差变化
  2. 位置编码贡献: 同词不同位置的残差差异
  3. 注意力头贡献: L0各注意力头对语义方差增加的贡献
  4. MLP贡献: L0的MLP是否扩展语义空间
  5. LayerNorm作用: 移除LayerNorm后语义方差变化
  6. 信息流分解: Δ(resid) = f(attention) + g(MLP) + bias

用法:
  python ccxxix_embedding_unfreeze.py --model qwen3
  python ccxxix_embedding_unfreeze.py --model glm4
  python ccxxix_embedding_unfreeze.py --model deepseek7b
"""
import argparse, os, sys, json, gc
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model, get_layer_weights

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxxix_embedding_unfreeze_log.txt"

# Habitat词汇(语义分类)
WORDS_BY_HABITAT = {
    "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit", "fox", "deer"],
    "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle", "crab", "seal", "squid", "lobster"],
    "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow", "falcon", "pigeon", "robin"],
}

# 更多词汇
EXTRA_WORDS = [
    "table", "chair", "car", "bus", "apple", "banana", "iron", "steel",
    "red", "blue", "mountain", "river", "sun", "moon", "book", "pen",
    "house", "tree", "water", "fire", "earth", "wind", "stone", "gold",
]

TEMPLATE = "The {}"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def compute_semantic_variance_ratio(residuals, labels):
    """计算语义方差比: 语义组间方差 / 总方差"""
    X = np.array(residuals)
    if X.shape[0] < 3:
        return 0.0, 0.0, 0
    
    # 总方差
    total_var = np.var(X, axis=0).sum()
    if total_var < 1e-10:
        return 0.0, 0.0, X.shape[1]
    
    # 组间方差
    grand_mean = X.mean(axis=0)
    between_var = 0
    unique_labels = set(labels)
    for lab in unique_labels:
        mask = [i for i, l in enumerate(labels) if l == lab]
        if len(mask) < 1:
            continue
        group_mean = X[mask].mean(axis=0)
        between_var += len(mask) * np.sum((group_mean - grand_mean) ** 2)
    
    # 有效维度(基于奇异值)
    X_centered = X - X.mean(axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    total_energy = np.sum(S**2)
    if total_energy < 1e-10:
        eff_dim = 0
    else:
        p = (S**2 / total_energy)
        eff_dim = 1.0 / np.sum(p**2)
    
    ratio = between_var / total_var if total_var > 0 else 0
    return float(ratio), float(eff_dim), X.shape[1]


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    log(f"\n{'='*70}\nCCXXIX(329): Embedding→L0→L1信息流追踪 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {}
    
    # 准备词汇
    all_words = []
    word_labels = []
    for hab, words in WORDS_BY_HABITAT.items():
        all_words.extend(words)
        word_labels.extend([hab] * len(words))
    all_words.extend(EXTRA_WORDS)
    word_labels.extend(["other"] * len(EXTRA_WORDS))
    
    # ===== Step 1: 逐层残差收集 =====
    log("\n--- Step 1: 逐层残差收集 ---")
    
    # 收集Embedding层和各Transformer层的残差
    target_layers = [0, 1, 2, 3, n_layers // 4, n_layers // 2, n_layers - 1]
    target_layers = sorted(set([l for l in target_layers if l < n_layers]))
    
    # Embedding残差
    embed_layer = model.get_input_embeddings()
    
    layer_resids = {li: [] for li in target_layers}
    embed_resids = []
    token_embeds = {}  # 缓存token embedding
    
    for w in all_words:
        prompt = TEMPLATE.format(w)
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        seq_len = toks.input_ids.shape[1]
        last_pos = seq_len - 1
        
        # 获取token embedding
        with torch.no_grad():
            embed_out = embed_layer(toks.input_ids)
        embed_resid = embed_out[0, last_pos, :].detach().float().cpu().numpy()
        embed_resids.append(embed_resid)
        
        # 收集各层残差
        captured = {}
        def mk_hook(k):
            def hook(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                captured[k] = o[0, last_pos, :].detach().float().cpu().numpy()
            return hook
        
        hooks = [layers[li].register_forward_hook(mk_hook(f"L{li}")) for li in target_layers]
        with torch.no_grad():
            _ = model(**toks)
        for h in hooks:
            h.remove()
        
        for li in target_layers:
            if f"L{li}" in captured:
                layer_resids[li].append(captured[f"L{li}"])
    
    log(f"  收集了 {len(embed_resids)} 个词汇的残差")
    
    # ===== Step 2: 逐层语义方差比 =====
    log("\n--- Step 2: 逐层语义方差比 ---")
    
    # 只用habitat词汇计算语义方差
    hab_mask = [i for i, l in enumerate(word_labels) if l in ["land", "ocean", "sky"]]
    hab_labels = [word_labels[i] for i in hab_mask]
    
    # Embedding
    embed_hab = [embed_resids[i] for i in hab_mask]
    embed_sem_ratio, embed_eff_dim, _ = compute_semantic_variance_ratio(embed_hab, hab_labels)
    
    layer_sem_info = {"embedding": {
        "semantic_ratio": round(embed_sem_ratio, 4),
        "eff_dim": round(embed_eff_dim, 2),
    }}
    log(f"  Embedding: sem_ratio={embed_sem_ratio:.4f}, eff_dim={embed_eff_dim:.2f}")
    
    for li in target_layers:
        hab_resids = [layer_resids[li][i] for i in hab_mask if i < len(layer_resids[li])]
        if len(hab_resids) < 3:
            continue
        sem_ratio, eff_dim, _ = compute_semantic_variance_ratio(hab_resids, hab_labels)
        layer_sem_info[f"L{li}"] = {
            "semantic_ratio": round(sem_ratio, 4),
            "eff_dim": round(eff_dim, 2),
        }
        log(f"  L{li}: sem_ratio={sem_ratio:.4f}, eff_dim={eff_dim:.2f}")
    
    results["layer_semantic_variance"] = layer_sem_info
    
    # ===== Step 3: L0→L1的信息流分解 =====
    log("\n--- Step 3: L0→L1的信息流分解 ---")
    
    # 分析: L0残差 = input_embed + attn_output + residual
    # 关键: attention层和MLP层分别贡献了多少语义方差?
    
    # 方法: 在L0和L1分别收集:
    # - 正常残差
    # - 移除注意力输出后的残差 (将注意力输出置零)
    # - 移除MLP输出后的残差 (将MLP输出置零)
    
    # 使用hook修改输出
    for target_layer in [0, 1]:
        # 3a: 正常残差(已收集)
        normal_resids = np.array(layer_resids[target_layer])
        
        # 3b: 移除注意力输出 - 在self_attn后置零
        attn_zero_resids = []
        mlp_zero_resids = []
        
        for w in all_words[:15]:  # 用子集
            prompt = TEMPLATE.format(w)
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            seq_len = toks.input_ids.shape[1]
            last_pos = seq_len - 1
            
            # 收集正常L0和L1输出
            captured_attn_zero = {}
            captured_mlp_zero = {}
            
            # 方案: 通过修改层内attention输出为0
            # 需要在self_attn的输出上hook, 将其置零
            attn_zeroed = [False]
            def attn_zero_hook(m, inp, out):
                if attn_zeroed[0]:
                    return
                attn_zeroed[0] = True
                o = out[0] if isinstance(out, tuple) else out
                o_new = torch.zeros_like(o)
                if isinstance(out, tuple):
                    return (o_new,) + out[1:]
                return o_new
            
            # L0的attention输出置零, 收集L0最终输出
            hook_attn = layers[target_layer].self_attn.register_forward_hook(attn_zero_hook)
            
            captured_l = {}
            def mk_hook_l(k):
                def hook(m, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    captured_l[k] = o[0, last_pos, :].detach().float().cpu().numpy()
                return hook
            
            hook_read = layers[target_layer].register_forward_hook(mk_hook_l(f"L{target_layer}"))
            
            with torch.no_grad():
                _ = model(**toks)
            
            hook_attn.remove()
            hook_read.remove()
            
            if f"L{target_layer}" in captured_l:
                attn_zero_resids.append(captured_l[f"L{target_layer}"])
            
            # L0的MLP输出置零
            mlp_zeroed = [False]
            def mlp_zero_hook(m, inp, out):
                if mlp_zeroed[0]:
                    return
                mlp_zeroed[0] = True
                o = out[0] if isinstance(out, tuple) else out
                o_new = torch.zeros_like(o)
                if isinstance(out, tuple):
                    return (o_new,) + out[1:]
                return o_new
            
            captured_l2 = {}
            def mk_hook_l2(k):
                def hook(m, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    captured_l2[k] = o[0, last_pos, :].detach().float().cpu().numpy()
                return hook
            
            hook_mlp = layers[target_layer].mlp.register_forward_hook(mlp_zero_hook)
            hook_read2 = layers[target_layer].register_forward_hook(mk_hook_l2(f"L{target_layer}"))
            
            with torch.no_grad():
                _ = model(**tokenizer(prompt, return_tensors="pt").to(device))
            
            hook_mlp.remove()
            hook_read2.remove()
            
            if f"L{target_layer}" in captured_l2:
                mlp_zero_resids.append(captured_l2[f"L{target_layer}"])
        
        # 计算移除后语义方差
        sub_labels = word_labels[:15]
        sub_hab_mask = [i for i, l in enumerate(sub_labels) if l in ["land", "ocean", "sky"]]
        sub_hab_labels = [sub_labels[i] for i in sub_hab_mask]
        
        if len(attn_zero_resids) >= 3 and len(sub_hab_mask) >= 3:
            attn_zero_hab = [attn_zero_resids[i] for i in sub_hab_mask if i < len(attn_zero_resids)]
            attn_zero_sem, attn_zero_eff, _ = compute_semantic_variance_ratio(attn_zero_hab, sub_hab_labels)
        else:
            attn_zero_sem = 0
            attn_zero_eff = 0
        
        if len(mlp_zero_resids) >= 3 and len(sub_hab_mask) >= 3:
            mlp_zero_hab = [mlp_zero_resids[i] for i in sub_hab_mask if i < len(mlp_zero_resids)]
            mlp_zero_sem, mlp_zero_eff, _ = compute_semantic_variance_ratio(mlp_zero_hab, sub_hab_labels)
        else:
            mlp_zero_sem = 0
            mlp_zero_eff = 0
        
        results[f"component_ablation_L{target_layer}"] = {
            "layer": target_layer,
            "n_test_words": 15,
            "attn_zeroed_sem_ratio": round(attn_zero_sem, 4),
            "attn_zeroed_eff_dim": round(attn_zero_eff, 2),
            "mlp_zeroed_sem_ratio": round(mlp_zero_sem, 4),
            "mlp_zeroed_eff_dim": round(mlp_zero_eff, 2),
        }
        
        log(f"  L{target_layer}: attn_zeroed→sem={attn_zero_sem:.4f}, eff={attn_zero_eff:.2f}; "
            f"mlp_zeroed→sem={mlp_zero_sem:.4f}, eff={mlp_zero_eff:.2f}")
    
    # ===== Step 4: 位置编码贡献 =====
    log("\n--- Step 4: 位置编码贡献 ---")
    
    # 同词不同位置: "The dog" vs "dog is" vs "is the dog"
    position_templates = [
        ("pos0", "{}"),          # 词在位置0
        ("pos1", "The {}"),      # 词在位置1
        ("pos2", "In the {}"),   # 词在位置2
        ("pos3", "I saw the {} today"),  # 词在位置3
    ]
    
    test_words = ["dog", "cat", "lion", "whale", "shark", "eagle", "hawk", "fox", "salmon", "owl"]
    test_labels_pos = ["land", "land", "land", "ocean", "ocean", "sky", "sky", "land", "ocean", "sky"]
    
    pos_resids = {pos_name: {li: [] for li in [0, 1]} for pos_name, _ in position_templates}
    
    for pos_name, tmpl in position_templates:
        for w in test_words:
            prompt = tmpl.format(w)
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            seq_len = toks.input_ids.shape[1]
            last_pos = seq_len - 1
            
            captured = {}
            def mk_hook(k):
                def hook(m, inp, out):
                    o = out[0] if isinstance(out, tuple) else out
                    captured[k] = o[0, last_pos, :].detach().float().cpu().numpy()
                return hook
            
            hooks = [layers[li].register_forward_hook(mk_hook(f"L{li}")) for li in [0, 1]]
            with torch.no_grad():
                _ = model(**toks)
            for h in hooks:
                h.remove()
            
            for li in [0, 1]:
                if f"L{li}" in captured:
                    pos_resids[pos_name][li].append(captured[f"L{li}"])
    
    # 分析: 同词不同位置, 残差变化多少是位置编码导致的?
    for li in [0, 1]:
        pos_sem_info = {}
        for pos_name, _ in position_templates:
            resids = pos_resids[pos_name][li]
            if len(resids) >= 3:
                sem_ratio, eff_dim, _ = compute_semantic_variance_ratio(resids, test_labels_pos)
                pos_sem_info[pos_name] = {
                    "semantic_ratio": round(sem_ratio, 4),
                    "eff_dim": round(eff_dim, 2),
                }
        
        # 跨位置的平均语义方差
        all_pos_resids = []
        all_pos_labels = []
        for pos_name, _ in position_templates:
            all_pos_resids.extend(pos_resids[pos_name][li])
            all_pos_labels.extend(test_labels_pos)
        
        if len(all_pos_resids) >= 3:
            cross_pos_sem, cross_pos_eff, _ = compute_semantic_variance_ratio(all_pos_resids, all_pos_labels)
        else:
            cross_pos_sem = 0
            cross_pos_eff = 0
        
        results[f"position_encoding_L{li}"] = {
            "layer": li,
            "by_position": pos_sem_info,
            "cross_position_sem_ratio": round(cross_pos_sem, 4),
            "cross_position_eff_dim": round(cross_pos_eff, 2),
        }
        
        log(f"  L{li}: by_position = {pos_sem_info}")
        log(f"        cross_pos sem={cross_pos_sem:.4f}, eff={cross_pos_eff:.2f}")
    
    # ===== Step 5: 注意力头贡献分析 =====
    log("\n--- Step 5: 注意力头贡献分析 ---")
    
    # L0和L1各注意力头对语义方向的贡献
    for target_layer in [0, 1]:
        lw = get_layer_weights(layers[target_layer], d_model, info.mlp_type)
        
        # W_O: [d_model, d_model] (或 [d_model, d_kv*n_heads])
        # 分解为各头: W_O_h = W_O[:, h*d_head:(h+1)*d_head]
        W_O = lw.W_o  # [d_model, d_model]
        
        # 确定头数和头维度
        n_heads = info.d_model  # 默认
        d_head = 1
        if hasattr(layers[target_layer].self_attn, 'num_heads'):
            n_heads = layers[target_layer].self_attn.num_heads
            d_head = d_model // n_heads
        elif hasattr(layers[target_layer].self_attn, 'n_heads'):
            n_heads = layers[target_layer].self_attn.n_heads
            d_head = d_model // n_heads
        
        log(f"  L{target_layer}: n_heads={n_heads}, d_head={d_head}")
        
        # 收集该层正常残差(已有)
        normal_resids_arr = np.array(layer_resids[target_layer])
        hab_indices = [i for i, l in enumerate(word_labels) if l in ["land", "ocean", "sky"] and i < len(normal_resids_arr)]
        
        if len(hab_indices) >= 3:
            hab_resids_arr = normal_resids_arr[hab_indices]
            hab_labs = [word_labels[i] for i in hab_indices]
            
            # 语义方向(3个habitat的均值方向)
            grand_mean = hab_resids_arr.mean(axis=0)
            sem_dirs = {}
            for hab in ["land", "ocean", "sky"]:
                mask = [i for i, l in enumerate(hab_labs) if l == hab]
                if len(mask) >= 2:
                    group_mean = hab_resids_arr[mask].mean(axis=0)
                    direction = group_mean - grand_mean
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        sem_dirs[hab] = direction / norm
            
            # 各注意力头的W_O投影到语义方向
            head_sem_alignment = {}
            for h_idx in range(min(n_heads, 32)):  # 最多32头
                W_O_h = W_O[:, h_idx*d_head:(h_idx+1)*d_head]  # [d_model, d_head]
                
                # 头的输出空间: W_O_h的列空间
                # 头对语义方向的贡献: ||W_O_h @ W_O_h^T @ sem_dir|| / ||sem_dir||
                h_align = {}
                for hab, sem_dir in sem_dirs.items():
                    proj = W_O_h @ (W_O_h.T @ sem_dir)
                    proj_norm = np.linalg.norm(proj)
                    sem_norm = np.linalg.norm(sem_dir)
                    ratio = proj_norm / sem_norm if sem_norm > 1e-10 else 0
                    h_align[hab] = round(float(ratio), 4)
                
                head_sem_alignment[f"head_{h_idx}"] = h_align
            
            # 找出对语义方向贡献最大的头
            head_scores = {}
            for h_idx in range(min(n_heads, 32)):
                score = np.mean([head_sem_alignment[f"head_{h_idx}"][hab] 
                               for hab in sem_dirs.keys() 
                               if hab in head_sem_alignment[f"head_{h_idx}"]])
                head_scores[h_idx] = round(float(score), 4)
            
            top_heads = sorted(head_scores.items(), key=lambda x: x[1], reverse=True)[:5]
            
            results[f"attn_head_contribution_L{target_layer}"] = {
                "layer": target_layer,
                "n_heads": n_heads,
                "d_head": d_head,
                "top5_heads": [(f"head_{h}", s) for h, s in top_heads],
                "head_scores_mean": round(float(np.mean(list(head_scores.values()))), 4),
                "head_scores_max": round(float(max(head_scores.values())), 4),
            }
            
            log(f"  L{target_layer}: top5 heads = {top_heads}")
            log(f"         mean_score={np.mean(list(head_scores.values())):.4f}, max={max(head_scores.values()):.4f}")
    
    # ===== Step 6: Embedding→L0的残差分解 =====
    log("\n--- Step 6: Embedding→L0残差分解 ---")
    
    # Embedding残差 vs L0残差 vs L1残差 的对齐
    for li in [0, 1]:
        if li >= len(target_layers) or li not in [0, 1]:
            continue
        
        embed_arr = np.array(embed_resids[:len(layer_resids[li])])
        layer_arr = np.array(layer_resids[li])
        
        if embed_arr.shape[0] < 3 or layer_arr.shape[0] < 3:
            continue
        
        # PCA on embed
        embed_mean = embed_arr.mean(axis=0)
        embed_centered = embed_arr - embed_mean
        U_e, S_e, Vt_e = np.linalg.svd(embed_centered, full_matrices=False)
        
        # PCA on layer
        layer_mean = layer_arr.mean(axis=0)
        layer_centered = layer_arr - layer_mean
        U_l, S_l, Vt_l = np.linalg.svd(layer_centered, full_matrices=False)
        
        # 子空间重叠
        n_pc = min(10, Vt_e.shape[0], Vt_l.shape[0])
        sub_embed = Vt_e[:n_pc]  # [n_pc, d_model]
        sub_layer = Vt_l[:n_pc]
        
        # 子空间重叠 = ||sub_embed @ sub_layer^T||_F / n_pc
        overlap_matrix = sub_embed @ sub_layer.T
        overlap = np.linalg.norm(overlap_matrix, 'fro') / n_pc
        
        # PC对齐
        pc_alignment = {}
        for i in range(min(5, n_pc)):
            cos_val = float(np.dot(sub_embed[i], sub_layer[i]))
            pc_alignment[f"PC{i}"] = round(cos_val, 4)
        
        results[f"embed_to_L{li}_alignment"] = {
            "layer": li,
            "subspace_overlap_K10": round(float(overlap), 4),
            "pc_alignment": pc_alignment,
        }
        
        log(f"  Embed→L{li}: overlap={overlap:.4f}, PC对齐={pc_alignment}")
    
    # 保存结果
    out_path = TEMP / f"ccxxix_embedding_unfreeze_{model_name}.json"
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
