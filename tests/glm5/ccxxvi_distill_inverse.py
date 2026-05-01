"""
CCXXVI(326): 蒸馏展开逆操作
======================================================================
CCXXII发现: DS7B L26→L27展开(3→5维), 但展开=维度重建+信息丢失。
本实验: 能否从L27恢复L26的信息?

核心问题:
  1. L27的PC与L26的PC的线性映射关系?
  2. 能否用L27的残差重建L26的残差?
  3. 展开维度(4-5)是否包含L26的信息?
  4. 逆变换的质量如何?

实验设计:
  1. 收集DS7B L26和L27的残差
  2. 计算L26→L27和L27→L26的线性映射
  3. 重建质量: R², 余弦相似度
  4. perturb测试: L26的语义方向在L27是否保持?
  5. 非蒸馏模型的层间映射对比

用法:
  python ccxxvi_distill_inverse.py --model deepseek7b
  python ccxxvi_distill_inverse.py --model qwen3  # 对比
  python ccxxvi_distill_inverse.py --model glm4   # 对比
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
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxxvi_distill_inverse_log.txt"

WORDS_BY_HABITAT = {
    "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "sheep", "rabbit", "fox", "deer"],
    "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "turtle", "crab", "seal", "squid", "lobster"],
    "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "swallow", "falcon", "pigeon", "robin"],
}

# 更多词汇用于拟合线性映射
EXTRA_WORDS = [
    "table", "chair", "desk", "lamp", "sofa", "bed", "door", "window",
    "car", "bus", "train", "plane", "ship", "boat", "bike", "truck",
    "apple", "banana", "grape", "orange", "mango", "peach", "pear", "cherry",
    "iron", "steel", "copper", "gold", "silver", "wood", "stone", "glass",
    "red", "blue", "green", "yellow", "purple", "orange", "pink", "brown",
    "mountain", "river", "lake", "forest", "desert", "island", "valley", "cliff",
    "sun", "moon", "star", "cloud", "rain", "snow", "wind", "storm",
]

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
    
    log(f"\n{'='*70}\nCCXXVI(326): 蒸馏展开逆操作 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {}
    
    # 选择层对
    if model_name == "deepseek7b":
        # 关键: L26(展开前) vs L27(展开后)
        layer_pairs = [(25, 26), (26, 27)]
        pair_labels = ["L25vsL26", "L26vsL27(展开)"]
    elif model_name == "qwen3":
        # 对比: 最后两层
        layer_pairs = [(33, 34), (34, 35)]
        pair_labels = ["L33vsL34", "L34vsL35"]
    elif model_name == "glm4":
        layer_pairs = [(37, 38), (38, 39)]
        pair_labels = ["L37vsL38", "L38vsL39"]
    
    # 收集大量词汇的残差
    all_words = []
    for hab, words in WORDS_BY_HABITAT.items():
        all_words.extend(words)
    all_words.extend(EXTRA_WORDS)
    all_words = list(set(all_words))  # 去重
    
    log(f"\n  收集 {len(all_words)} 个词汇的残差...")
    
    # 采样: 太多词汇太慢, 取50个
    if len(all_words) > 50:
        np.random.seed(42)
        all_words = list(np.random.choice(all_words, 50, replace=False))
    
    all_test_layers = sorted(set([li for pair in layer_pairs for li in pair]))
    
    layer_resids = {li: [] for li in all_test_layers}
    
    for word in all_words:
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
        
        hooks = [layers[li].register_forward_hook(mk_hook(f"L{li}")) for li in all_test_layers]
        with torch.no_grad():
            _ = model(**toks)
        for h in hooks:
            h.remove()
        
        for li in all_test_layers:
            if f"L{li}" in captured:
                layer_resids[li].append(captured[f"L{li}"])
    
    log(f"  收集完成, 每层 {len(layer_resids[all_test_layers[0]])} 个残差")
    
    # ===== Part 1: 层间线性映射 =====
    log("\n--- Part 1: 层间线性映射 ---")
    
    for (li1, li2), pair_label in zip(layer_pairs, pair_labels):
        X1 = np.array(layer_resids[li1])  # [N, d_model]
        X2 = np.array(layer_resids[li2])  # [N, d_model]
        
        if X1.shape[0] < 10:
            continue
        
        N = X1.shape[0]
        
        # 1a: 正向映射 X1 → X2
        # X2 ≈ X1 @ W + b (线性回归)
        # 添加偏置项
        X1_aug = np.hstack([X1, np.ones((N, 1))])  # [N, d+1]
        
        # 最小二乘: W = (X1_aug^T X1_aug)^{-1} X1_aug^T X2
        # 使用SVD避免奇异矩阵
        try:
            W_forward, res_forward, rank_forward, sv_forward = np.linalg.lstsq(X1_aug, X2, rcond=None)
            
            # 预测
            X2_pred = X1_aug @ W_forward
            
            # R² (每个维度)
            ss_res = np.sum((X2 - X2_pred) ** 2, axis=0)
            ss_tot = np.sum((X2 - X2.mean(axis=0)) ** 2, axis=0)
            r2_per_dim = 1 - ss_res / (ss_tot + 1e-10)
            mean_r2 = float(np.mean(r2_per_dim))
            
            # 余弦相似度(每个样本)
            cos_per_sample = []
            for k in range(N):
                delta = X2[k] - X2_pred[k]
                cos_val = float(np.dot(X2[k], X2_pred[k]) / 
                               (np.linalg.norm(X2[k]) * np.linalg.norm(X2_pred[k]) + 1e-10))
                cos_per_sample.append(cos_val)
            mean_cos = float(np.mean(cos_per_sample))
            
            # 残差范数比
            resid_norms = np.linalg.norm(X2 - X2_pred, axis=1)
            orig_norms = np.linalg.norm(X2, axis=1)
            mean_resid_ratio = float(np.mean(resid_norms / (orig_norms + 1e-10)))
            
            results[f"forward_{pair_label}"] = {
                "from_layer": li1,
                "to_layer": li2,
                "mean_R2": round(mean_r2, 4),
                "mean_cos": round(mean_cos, 4),
                "mean_resid_ratio": round(mean_resid_ratio, 4),
            }
            
            log(f"  正向 {pair_label}: R²={mean_r2:.4f}, cos={mean_cos:.4f}, resid_ratio={mean_resid_ratio:.4f}")
        except Exception as e:
            log(f"  正向 {pair_label} 失败: {e}")
        
        # 1b: 逆向映射 X2 → X1
        X2_aug = np.hstack([X2, np.ones((N, 1))])
        try:
            W_inverse, _, _, _ = np.linalg.lstsq(X2_aug, X1, rcond=None)
            
            X1_pred = X2_aug @ W_inverse
            
            ss_res = np.sum((X1 - X1_pred) ** 2, axis=0)
            ss_tot = np.sum((X1 - X1.mean(axis=0)) ** 2, axis=0)
            r2_per_dim = 1 - ss_res / (ss_tot + 1e-10)
            mean_r2_inv = float(np.mean(r2_per_dim))
            
            cos_per_sample_inv = []
            for k in range(N):
                cos_val = float(np.dot(X1[k], X1_pred[k]) / 
                               (np.linalg.norm(X1[k]) * np.linalg.norm(X1_pred[k]) + 1e-10))
                cos_per_sample_inv.append(cos_val)
            mean_cos_inv = float(np.mean(cos_per_sample_inv))
            
            resid_norms_inv = np.linalg.norm(X1 - X1_pred, axis=1)
            orig_norms_inv = np.linalg.norm(X1, axis=1)
            mean_resid_ratio_inv = float(np.mean(resid_norms_inv / (orig_norms_inv + 1e-10)))
            
            results[f"inverse_{pair_label}"] = {
                "from_layer": li2,
                "to_layer": li1,
                "mean_R2": round(mean_r2_inv, 4),
                "mean_cos": round(mean_cos_inv, 4),
                "mean_resid_ratio": round(mean_resid_ratio_inv, 4),
            }
            
            log(f"  逆向 {pair_label}: R²={mean_r2_inv:.4f}, cos={mean_cos_inv:.4f}, resid_ratio={mean_resid_ratio_inv:.4f}")
        except Exception as e:
            log(f"  逆向 {pair_label} 失败: {e}")
    
    # ===== Part 2: PC空间的层间映射 =====
    log("\n--- Part 2: PC空间的层间映射 ---")
    
    for (li1, li2), pair_label in zip(layer_pairs, pair_labels):
        X1 = np.array(layer_resids[li1])
        X2 = np.array(layer_resids[li2])
        
        if X1.shape[0] < 10:
            continue
        
        # 各层的PCA
        mean1 = X1.mean(axis=0)
        mean2 = X2.mean(axis=0)
        X1c = X1 - mean1
        X2c = X2 - mean2
        
        U1, S1, Vt1 = np.linalg.svd(X1c, full_matrices=False)
        U2, S2, Vt2 = np.linalg.svd(X2c, full_matrices=False)
        
        # PC对齐
        K = min(10, Vt1.shape[0], Vt2.shape[0])
        pc_align = []
        for k in range(K):
            cos_val = float(np.abs(np.dot(Vt1[k], Vt2[k])))
            pc_align.append(round(cos_val, 4))
        
        # 子空间重叠(前5个PC)
        k_sub = 5
        P1 = Vt1[:k_sub].T @ Vt1[:k_sub]
        P2 = Vt2[:k_sub].T @ Vt2[:k_sub]
        overlap = float(np.trace(P1 @ P2)) / k_sub  # 0到1
        
        # 奇异值分布相关性
        min_len = min(len(S1), len(S2), 20)
        s1_norm = S1[:min_len] / np.sum(S1[:min_len])
        s2_norm = S2[:min_len] / np.sum(S2[:min_len])
        sv_corr = float(np.corrcoef(s1_norm, s2_norm)[0, 1]) if np.std(s1_norm) > 1e-10 else 0
        
        results[f"pc_mapping_{pair_label}"] = {
            "from_layer": li1,
            "to_layer": li2,
            "pc_alignment": pc_align,
            "subspace_overlap_k5": round(overlap, 4),
            "sv_correlation": round(sv_corr, 4),
            "S1_top10": [round(float(s), 2) for s in S1[:10]],
            "S2_top10": [round(float(s), 2) for s in S2[:10]],
        }
        
        log(f"  PC对齐 {pair_label}: {pc_align}")
        log(f"  子空间重叠 {pair_label}: {overlap:.4f}, sv_corr={sv_corr:.4f}")
    
    # ===== Part 3: 语义方向保持性 =====
    log("\n--- Part 3: 语义方向保持性 ---")
    
    # 对habitat方向, 检查li1→li2的保持
    for (li1, li2), pair_label in zip(layer_pairs, pair_labels):
        # 收集habitat残差
        hab_resids1 = {"land": [], "ocean": [], "sky": []}
        hab_resids2 = {"land": [], "ocean": [], "sky": []}
        
        for hab, words in WORDS_BY_HABITAT.items():
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
                
                hooks = [layers[li1].register_forward_hook(mk_hook(f"L{li1}")),
                        layers[li2].register_forward_hook(mk_hook(f"L{li2}"))]
                with torch.no_grad():
                    _ = model(**toks)
                for h in hooks:
                    h.remove()
                
                if f"L{li1}" in captured:
                    hab_resids1[hab].append(captured[f"L{li1}"])
                if f"L{li2}" in captured:
                    hab_resids2[hab].append(captured[f"L{li2}"])
        
        # 计算各层的habitat方向
        X1_all = np.array([v for hab in ["land", "ocean", "sky"] for v in hab_resids1[hab]])
        X2_all = np.array([v for hab in ["land", "ocean", "sky"] for v in hab_resids2[hab]])
        
        if len(X1_all) < 6 or len(X2_all) < 6:
            continue
        
        mean1 = X1_all.mean(axis=0)
        mean2 = X2_all.mean(axis=0)
        
        hab_dirs1 = {}
        hab_dirs2 = {}
        for hab in ["land", "ocean", "sky"]:
            if len(hab_resids1[hab]) >= 3:
                d1 = np.mean(hab_resids1[hab], axis=0) - mean1
                n1 = np.linalg.norm(d1)
                if n1 > 1e-10:
                    hab_dirs1[hab] = d1 / n1
            
            if len(hab_resids2[hab]) >= 3:
                d2 = np.mean(hab_resids2[hab], axis=0) - mean2
                n2 = np.linalg.norm(d2)
                if n2 > 1e-10:
                    hab_dirs2[hab] = d2 / n2
        
        # 方向保持: li1的habitat方向在li2是否保持?
        dir_preservation = {}
        for hab in ["land", "ocean", "sky"]:
            if hab in hab_dirs1 and hab in hab_dirs2:
                cos_val = float(np.dot(hab_dirs1[hab], hab_dirs2[hab]))
                dir_preservation[hab] = round(cos_val, 4)
        
        # habitat分类准确率
        # 用li1的habitat方向分类li2的残差
        if len(hab_dirs1) == 3:
            dir_matrix = np.array([hab_dirs1[h] for h in ["land", "ocean", "sky"]])  # [3, d]
            
            correct1 = 0
            correct2 = 0
            total = 0
            for ci, hab in enumerate(["land", "ocean", "sky"]):
                for vec in hab_resids2[hab]:
                    # 用L1的方向分类L2的残差
                    scores = dir_matrix @ (vec - mean1)
                    pred = np.argmax(scores)
                    if pred == ci:
                        correct1 += 1
                    
                    # 用L2自身方向分类
                    scores2 = np.array([hab_dirs2.get(h, np.zeros(d_model)) for h in ["land", "ocean", "sky"]]) @ (vec - mean2)
                    pred2 = np.argmax(scores2)
                    if pred2 == ci:
                        correct2 += 1
                    total += 1
            
            cross_acc = correct1 / total if total > 0 else 0
            self_acc = correct2 / total if total > 0 else 0
        
        results[f"semantic_preservation_{pair_label}"] = {
            "from_layer": li1,
            "to_layer": li2,
            "direction_cosines": dir_preservation,
            "cross_layer_classification": round(cross_acc, 4),
            "self_classification": round(self_acc, 4),
        }
        
        log(f"  语义保持 {pair_label}: dir_cos={dir_preservation}")
        log(f"  跨层分类: {cross_acc:.4f}, 自身分类: {self_acc:.4f}")
    
    # ===== Part 4: 展开维度的信息量 =====
    log("\n--- Part 4: 展开维度的信息量 (DS7B特有) ---")
    
    if model_name == "deepseek7b":
        # L26 vs L27: 展开维度(4-5)包含什么信息?
        X26 = np.array(layer_resids[26])
        X27 = np.array(layer_resids[27])
        
        if X26.shape[0] >= 10:
            mean26 = X26.mean(axis=0)
            mean27 = X27.mean(axis=0)
            X26c = X26 - mean26
            X27c = X27 - mean27
            
            U26, S26, Vt26 = np.linalg.svd(X26c, full_matrices=False)
            U27, S27, Vt27 = np.linalg.svd(X27c, full_matrices=False)
            
            # L27的前3维(核心) vs 第4-5维(展开) 对L26信息的编码
            # 核心维度能解释多少L26的方差?
            # 展开4-5维能解释多少L26的方差?
            
            # 方法: 将X26c投影到L27的各PC方向上
            for dim_range, dim_name in [((0, 3), "core_1to3"), ((3, 5), "expand_4to5"), ((0, 5), "all_1to5")]:
                start, end = dim_range
                # L27的PC子空间
                P_sub = Vt27[start:end].T @ Vt27[start:end]  # 投影矩阵
                
                # X26在L27子空间中的投影
                X26_proj = (P_sub @ X26c.T).T  # [N, d]
                
                # 投影保持了X26的多少方差?
                var_orig = np.sum(X26c ** 2)
                var_proj = np.sum(X26_proj ** 2)
                var_ratio = var_proj / var_orig if var_orig > 0 else 0
                
                # 重建质量: 用L27子空间重建X26
                # 先将X26投影到L27子空间, 然后用线性映射回去
                X26_in_L27 = Vt27[start:end] @ X26c.T  # [K, N] - X26在L27子空间中的系数
                X26_reconstructed = Vt27[start:end].T @ X26_in_L27  # [d, N]
                
                # 余弦相似度
                cos_vals = []
                for k in range(X26c.shape[0]):
                    cos_val = float(np.dot(X26c[k], X26_reconstructed[:, k]) / 
                                   (np.linalg.norm(X26c[k]) * np.linalg.norm(X26_reconstructed[:, k]) + 1e-10))
                    cos_vals.append(cos_val)
                
                results[f"expand_info_L26vsL27_{dim_name}"] = {
                    "dim_range": dim_name,
                    "var_ratio": round(float(var_ratio), 4),
                    "mean_recon_cos": round(float(np.mean(cos_vals)), 4),
                }
                
                log(f"  L27 {dim_name}: var_ratio={var_ratio:.4f}, recon_cos={np.mean(cos_vals):.4f}")
    
    # 保存结果
    out_path = TEMP / f"ccxxvi_distill_inverse_{model_name}.json"
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
