"""
CCXXIII(371): 信息压缩理论验证 — W_U有效秩与暗物质维度
====================================================

★★★★★ 核心问题:
  MEMO P0/P2: 验证信息压缩理论
  假设: W_U维度(~400有效秩) → 暗物质压缩 → 正交性
  
★★★★★ 关键发现:
  - DS7B浅层暗物质维度极低 (eff_rank≈1.7 at L12 for 50 concepts)
  - Qwen3/GLM4暗物质维度正常增长 (eff_rank≈9-42)
  - 问题: 是8bit量化还是W_U容量限制导致?

★★★★★ 实验设计:
  Exp1: 三模型W_U矩阵的SVD分析 — 有效秩、奇异值谱
  Exp2: W_U有效秩与暗物质维度的定量关系
  Exp3: DS7B权重量化精度分析 — 间接验证8bit效应

★★★★★ 核心假设:
  如果W_U有效秩决定了暗物质维度的上限:
  - rank(W_U) 越高 → 暗物质维度上限越高
  - 8bit量化降低W_U有效秩 → 暗物质维度坍缩

用法:
  python ccxxiii_wu_compression_theory.py --exp 1
  python ccxxiii_wu_compression_theory.py --exp 2
  python ccxxiii_wu_compression_theory.py --exp 3
  python ccxxiii_wu_compression_theory.py --exp all
"""

import argparse, os, sys, json, gc, time, warnings
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import torch
from scipy.linalg import svd
from sklearn.decomposition import PCA

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model, get_W_U

TEMP = Path("tests/glm5_temp")


# ================================================================
# Exp1: W_U矩阵的SVD分析
# ================================================================

def run_exp1(model_name):
    """分析W_U矩阵的SVD谱和有效秩"""
    print(f"\n{'='*70}")
    print(f"Exp1: W_U SVD分析 ({model_name})")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    
    # 获取W_U
    W_U = get_W_U(model)  # [vocab_size, d_model]
    vocab_size, d_model = W_U.shape
    print(f"  W_U shape: {vocab_size} × {d_model}")
    
    # SVD分析
    print(f"  Computing SVD...")
    U, s, Vt = svd(W_U, full_matrices=False)
    
    # 奇异值谱分析
    total_energy = np.sum(s**2)
    cumulative_energy = np.cumsum(s**2) / total_energy
    
    # 不同阈值下的有效秩
    eff_ranks = {}
    for threshold in [0.5, 0.8, 0.9, 0.95, 0.99, 0.999]:
        rank = int(np.argmax(cumulative_energy >= threshold) + 1)
        eff_ranks[f"rank_{int(threshold*1000)}"] = rank
    
    # 奇异值分布统计
    s_norm = s / s[0]  # 归一化
    s_stats = {
        "s_max": float(s[0]),
        "s_min": float(s[-1]),
        "s_mean": float(np.mean(s)),
        "s_median": float(np.median(s)),
        "s_std": float(np.std(s)),
        "condition_number": float(s[0] / max(s[-1], 1e-30)),
        "s_top10": s[:10].tolist(),
        "s_top100_ratio": float(np.sum(s[:100]**2) / total_energy),
        "s_top200_ratio": float(np.sum(s[:200]**2) / total_energy),
        "s_top400_ratio": float(np.sum(s[:400]**2) / total_energy),
        "s_top800_ratio": float(np.sum(s[:800]**2) / total_energy),
    }
    
    # 检测数值精度 (8bit量化会导致权重离散化)
    # 计算权重的唯一值数量
    W_U_flat = W_U.flatten()
    n_unique = len(np.unique(W_U_flat))
    precision_ratio = n_unique / len(W_U_flat)
    
    # 8bit权重通常有256^1种不同值, FP16有2^16种
    # 但实际上8bit量化后的唯一值数远少于FP16
    print(f"\n  W_U SVD Results:")
    print(f"  Full rank: {min(vocab_size, d_model)}")
    for key, val in eff_ranks.items():
        print(f"  {key}: {val}")
    print(f"  Condition number: {s_stats['condition_number']:.2e}")
    print(f"  s_top100_ratio: {s_stats['s_top100_ratio']:.4f}")
    print(f"  s_top400_ratio: {s_stats['s_top400_ratio']:.4f}")
    print(f"  Unique values in W_U: {n_unique} / {len(W_U_flat)} ({precision_ratio:.6e})")
    
    # 深入分析: 奇异值谱的形状
    # 如果是幂律分布 s_k ∝ k^(-α), 则 α 反映信息压缩程度
    log_s = np.log10(s[s > 0])
    log_k = np.log10(np.arange(1, len(log_s) + 1))
    
    # 线性拟合 log(s) vs log(k)
    if len(log_s) > 10:
        # 只用前50%的奇异值拟合幂律
        n_fit = len(log_s) // 2
        coeffs = np.polyfit(log_k[:n_fit], log_s[:n_fit], 1)
        alpha = -coeffs[0]  # 幂律指数 (负号因为衰减)
        r2_fit = 1 - np.sum((log_s[:n_fit] - np.polyval(coeffs, log_k[:n_fit]))**2) / \
                    np.sum((log_s[:n_fit] - np.mean(log_s[:n_fit]))**2)
        print(f"  Power-law exponent α = {alpha:.3f} (R²={r2_fit:.4f})")
    else:
        alpha = 0
        r2_fit = 0
    
    # 分析W_U行空间与embedding空间的关系
    W_E = model.get_input_embeddings().weight.detach().cpu().float().numpy()
    
    # W_U和W_E的子空间重叠
    # 取W_U的前rank_95个左奇异向量
    rank_95 = eff_ranks["rank_950"]
    U_top = U[:, :rank_95]  # [vocab, rank_95]
    
    # W_E的PCA
    pca_E = PCA(n_components=min(rank_95, W_E.shape[1]))
    pca_E.fit(W_E)
    V_E = pca_E.components_.T  # [d_model, rank_95]
    
    # 子空间重叠 = ||U_top^T @ V_E||_F / sqrt(rank_95)
    # 这需要转换到相同的空间... 不对，U_top是vocab空间, V_E是d_model空间
    # 需要用W_U的右奇异向量Vt
    Vt_top = Vt[:rank_95, :]  # [rank_95, d_model]
    
    # 子空间重叠: Tr(Vt_top @ V_E @ V_E^T @ Vt_top^T) / rank_95
    overlap_matrix = Vt_top @ V_E @ V_E.T @ Vt_top.T
    subspace_overlap = np.trace(overlap_matrix) / rank_95
    
    # 也计算Grassmann距离
    # P_U和P_E投影矩阵之间的主角度
    from scipy.linalg import subspace_angles
    try:
        angles = subspace_angles(Vt_top.T, V_E)
        mean_angle = float(np.mean(angles))
        max_angle = float(np.max(angles))
    except:
        mean_angle = 0
        max_angle = 0
    
    print(f"\n  W_U-W_E Subspace Analysis:")
    print(f"  rank_95(W_U): {rank_95}")
    print(f"  rank_95(W_E): {np.argmax(np.cumsum(pca_E.explained_variance_ratio_) >= 0.95) + 1}")
    print(f"  Subspace overlap: {subspace_overlap:.4f}")
    print(f"  Mean subspace angle: {np.degrees(mean_angle):.2f}°")
    print(f"  Max subspace angle: {np.degrees(max_angle):.2f}°")
    
    results = {
        "model": model_name,
        "d_model": d_model,
        "vocab_size": vocab_size,
        "full_rank": min(vocab_size, d_model),
        "eff_ranks": eff_ranks,
        "s_stats": s_stats,
        "power_law_alpha": float(alpha),
        "power_law_r2": float(r2_fit),
        "n_unique_values": int(n_unique),
        "precision_ratio": float(precision_ratio),
        "subspace_overlap": float(subspace_overlap),
        "mean_subspace_angle_deg": float(np.degrees(mean_angle)),
        "singular_values_sample": s[:500].tolist(),  # 保存前500个奇异值
        "cumulative_energy_sample": cumulative_energy[:500].tolist(),
    }
    
    outpath = TEMP / f"ccxxiii_{model_name}_wu_svd.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {outpath}")
    
    release_model(model)
    return results


# ================================================================
# Exp2: W_U有效秩与暗物质维度的定量关系
# ================================================================

def run_exp2():
    """跨模型比较W_U有效秩与暗物质维度的关系"""
    print(f"\n{'='*70}")
    print(f"Exp2: W_U有效秩与暗物质维度的定量关系")
    print(f"{'='*70}")
    
    all_wu_results = {}
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        path = TEMP / f"ccxxiii_{model_name}_wu_svd.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                all_wu_results[model_name] = json.load(f)
        else:
            print(f"  Running Exp1 for {model_name} first...")
            all_wu_results[model_name] = run_exp1(model_name)
    
    # 加载CCXX Exp3的暗物质维度数据
    ccxx_data = {}
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        path = TEMP / f"ccxx_exp3_{model_name}_results.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                ccxx_data[model_name] = json.load(f)
    
    # 核心比较表
    print(f"\n{'='*70}")
    print("  W_U Rank vs Dark Matter Dimension Comparison")
    print(f"{'='*70}")
    
    comparison = {}
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        wu = all_wu_results[model_name]
        ccxx = ccxx_data.get(model_name, {})
        
        # 取L24(深层)的暗物质维度
        scaling_L24 = ccxx.get("scaling_L24", [])
        dm_dim_50 = 0
        dm_dim_per_concept = 0
        for entry in scaling_L24:
            if entry["n_concepts"] == 50:
                dm_dim_50 = entry["eff_rank"]
                dm_dim_per_concept = dm_dim_50 / 50
                break
        
        comp = {
            "d_model": wu["d_model"],
            "vocab_size": wu["vocab_size"],
            "W_U_rank_50": wu["eff_ranks"].get("rank_500", 0),
            "W_U_rank_80": wu["eff_ranks"].get("rank_800", 0),
            "W_U_rank_90": wu["eff_ranks"].get("rank_900", 0),
            "W_U_rank_95": wu["eff_ranks"].get("rank_950", 0),
            "W_U_rank_99": wu["eff_ranks"].get("rank_990", 0),
            "W_U_condition_number": wu["s_stats"]["condition_number"],
            "W_U_power_law_alpha": wu["power_law_alpha"],
            "n_unique_values": wu["n_unique_values"],
            "precision_ratio": wu["precision_ratio"],
            "subspace_overlap": wu["subspace_overlap"],
            "DM_dim_L24_50concepts": dm_dim_50,
            "DM_dim_per_concept": dm_dim_per_concept,
        }
        comparison[model_name] = comp
        
        print(f"\n  {model_name}:")
        print(f"    d_model: {comp['d_model']}, vocab: {comp['vocab_size']}")
        print(f"    W_U rank(50%): {comp['W_U_rank_50']}")
        print(f"    W_U rank(90%): {comp['W_U_rank_90']}")
        print(f"    W_U rank(95%): {comp['W_U_rank_95']}")
        print(f"    W_U rank(99%): {comp['W_U_rank_99']}")
        print(f"    Condition number: {comp['W_U_condition_number']:.2e}")
        print(f"    Power-law α: {comp['W_U_power_law_alpha']:.3f}")
        print(f"    Unique values: {comp['n_unique_values']} ({comp['precision_ratio']:.2e})")
        print(f"    DM dim (L24, 50 concepts): {comp['DM_dim_L24_50concepts']:.2f} ({comp['DM_dim_per_concept']:.3f}/concept)")
    
    # 关键分析: DM维度 / W_U_rank_95 的比值
    print(f"\n{'='*70}")
    print("  Critical Ratio: DM_dim / W_U_rank_95")
    print(f"{'='*70}")
    
    for model_name, comp in comparison.items():
        if comp["W_U_rank_95"] > 0:
            ratio = comp["DM_dim_L24_50concepts"] / comp["W_U_rank_95"]
            print(f"  {model_name}: {comp['DM_dim_L24_50concepts']:.2f} / {comp['W_U_rank_95']} = {ratio:.4f}")
    
    # 量化效应验证
    print(f"\n{'='*70}")
    print("  Quantization Effect Analysis")
    print(f"{'='*70}")
    
    ds7b = comparison.get("deepseek7b", {})
    qwen3 = comparison.get("qwen3", {})
    glm4 = comparison.get("glm4", {})
    
    # 唯一值数量比较 (8bit量化会导致唯一值极少)
    print(f"  Unique values in W_U:")
    for name, comp in comparison.items():
        print(f"    {name}: {comp['n_unique_values']} ({comp['precision_ratio']:.2e} ratio)")
    
    # 判断: DS7B是否被8bit量化
    ds7b_unique = ds7b.get("n_unique_values", 0)
    qwen3_unique = qwen3.get("n_unique_values", 0)
    glm4_unique = glm4.get("n_unique_values", 0)
    
    avg_fp_unique = (qwen3_unique + glm4_unique) / 2 if (qwen3_unique + glm4_unique) > 0 else 1
    ds7b_unique_ratio = ds7b_unique / avg_fp_unique
    
    print(f"\n  DS7B unique values / avg(FP models) = {ds7b_unique_ratio:.4f}")
    
    if ds7b_unique_ratio < 0.01:
        print("  ★★★★★ DS7B W_U is HEAVILY QUANTIZED! (unique values << FP models)")
        print("  This strongly suggests 8bit quantization limits dark matter expression.")
    elif ds7b_unique_ratio < 0.1:
        print("  ★★★★ DS7B W_U shows significant quantization effects.")
    else:
        print("  ★★ DS7B W_U quantization level is similar to FP models.")
        print("  The low DM dimension may be due to model architecture, not quantization.")
    
    # 保存结果
    outpath = TEMP / "ccxxiii_wu_dm_comparison.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {outpath}")
    
    return comparison


# ================================================================
# Exp3: DS7B权重量化精度深度分析
# ================================================================

def run_exp3():
    """深度分析DS7B权重矩阵的数值精度, 判断是否被8bit量化"""
    print(f"\n{'='*70}")
    print(f"Exp3: DS7B权重量化精度深度分析")
    print(f"{'='*70}")
    
    results = {}
    
    for model_name in ["qwen3", "glm4", "deepseek7b"]:
        print(f"\n  Loading {model_name}...")
        model, tokenizer, device = load_model(model_name)
        model_info = get_model_info(model, model_name)
        
        # 分析多个权重矩阵的数值精度
        weight_analysis = {}
        
        # 1. W_U (lm_head)
        W_U = get_W_U(model)
        wu_unique = len(np.unique(W_U.flatten()))
        wu_total = W_U.size
        wu_dtype = str(model.lm_head.weight.dtype)
        
        # 2. W_E (embedding)
        W_E = model.get_input_embeddings().weight.detach().cpu().float().numpy()
        we_unique = len(np.unique(W_E.flatten()))
        we_total = W_E.size
        we_dtype = str(model.get_input_embeddings().weight.dtype)
        
        # 3. Layer 0 权重
        layers = get_layers(model)
        layer0 = layers[0]
        
        # Q projection
        W_q = layer0.self_attn.q_proj.weight.detach().cpu().float().numpy()
        wq_unique = len(np.unique(W_q.flatten()))
        wq_total = W_q.size
        wq_dtype = str(layer0.self_attn.q_proj.weight.dtype)
        
        # MLP up projection
        if hasattr(layer0.mlp, 'up_proj'):
            W_up = layer0.mlp.up_proj.weight.detach().cpu().float().numpy()
        elif hasattr(layer0.mlp, 'gate_up_proj'):
            W_gu = layer0.mlp.gate_up_proj.weight.detach().cpu().float().numpy()
            W_up = W_gu  # merged
        else:
            W_up = None
        
        if W_up is not None:
            wup_unique = len(np.unique(W_up.flatten()))
            wup_total = W_up.size
            wup_dtype = str(layer0.mlp.up_proj.weight.dtype if hasattr(layer0.mlp, 'up_proj') else 
                          layer0.mlp.gate_up_proj.weight.dtype)
        
        # 4. 检查weight的直方图特征
        # 8bit量化后, 权重会有明显的离散化模式
        W_U_flat = W_U.flatten()
        
        # 计算权重值之间的最小间隔
        W_U_sorted = np.sort(W_U_flat)
        if len(W_U_sorted) > 1000:
            diffs = np.diff(W_U_sorted[:10000])
            min_diff = np.min(np.abs(diffs[diffs > 0]))
            median_diff = np.median(np.abs(diffs[diffs > 0]))
        else:
            min_diff = 0
            median_diff = 0
        
        # 值域范围
        value_range = float(np.max(W_U_flat) - np.min(W_U_flat))
        
        # 量化检测: 如果8bit, 唯一值数约为256或256×n
        # FP16: 唯一值数 >> 256
        quantization_suspect = wu_unique < 10000  # 粗略阈值
        
        weight_analysis = {
            "W_U": {
                "unique_values": int(wu_unique),
                "total_values": int(wu_total),
                "ratio": float(wu_unique / wu_total),
                "dtype": wu_dtype,
                "min_diff": float(min_diff),
                "median_diff": float(median_diff),
                "value_range": value_range,
            },
            "W_E": {
                "unique_values": int(we_unique),
                "total_values": int(we_total),
                "ratio": float(we_unique / we_total),
                "dtype": we_dtype,
            },
            "W_q_L0": {
                "unique_values": int(wq_unique),
                "total_values": int(wq_total),
                "ratio": float(wq_unique / wq_total),
                "dtype": wq_dtype,
            },
            "W_up_L0": {
                "unique_values": int(wup_unique) if W_up is not None else 0,
                "total_values": int(wup_total) if W_up is not None else 0,
                "ratio": float(wup_unique / wup_total) if W_up is not None else 0,
                "dtype": wup_dtype if W_up is not None else "N/A",
            },
            "quantization_suspect": quantization_suspect,
        }
        
        results[model_name] = weight_analysis
        
        print(f"\n  {model_name} Weight Analysis:")
        print(f"    W_U: {wu_unique} unique / {wu_total} total ({wu_unique/wu_total:.2e}), dtype={wu_dtype}")
        print(f"    W_E: {we_unique} unique / {we_total} total ({we_unique/we_total:.2e}), dtype={we_dtype}")
        print(f"    W_q: {wq_unique} unique / {wq_total} total ({wq_unique/wq_total:.2e}), dtype={wq_dtype}")
        print(f"    W_up: {wup_unique} unique / {wup_total} total ({wup_unique/wup_total:.2e}), dtype={wup_dtype}")
        print(f"    W_U min_diff: {min_diff:.6e}, median_diff: {median_diff:.6e}")
        print(f"    W_U value range: {value_range:.4f}")
        print(f"    Quantization suspect: {quantization_suspect}")
        
        release_model(model)
    
    # 跨模型比较
    print(f"\n{'='*70}")
    print("  Cross-model Quantization Assessment")
    print(f"{'='*70}")
    
    for name, wa in results.items():
        wu_ratio = wa["W_U"]["ratio"]
        we_ratio = wa["W_E"]["ratio"]
        wq_ratio = wa["W_q_L0"]["ratio"]
        
        # FP16模型: ratio通常 > 0.5
        # 8bit量化: ratio通常 < 0.01
        quant_level = "FP16/FP32" if wu_ratio > 0.1 else "4-8bit quantized" if wu_ratio < 0.01 else "Mixed/Low precision"
        
        print(f"  {name}: W_U unique ratio={wu_ratio:.2e} → {quant_level}")
    
    # 保存
    outpath = TEMP / "ccxxiii_quantization_analysis.json"
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {outpath}")
    
    return results


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, default="1",
                       choices=["1", "2", "3", "all"])
    args = parser.parse_args()
    
    if args.exp == "1":
        for model_name in ["qwen3", "glm4", "deepseek7b"]:
            run_exp1(model_name)
    elif args.exp == "2":
        run_exp2()
    elif args.exp == "3":
        run_exp3()
    elif args.exp == "all":
        for model_name in ["qwen3", "glm4", "deepseek7b"]:
            run_exp1(model_name)
        run_exp2()
        run_exp3()


if __name__ == "__main__":
    main()
