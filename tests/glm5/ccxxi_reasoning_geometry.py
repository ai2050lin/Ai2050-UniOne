"""
CCXXI(321): 推理几何的精确刻画 — "大"方向 vs SVD主轴 + PC对齐分析
======================================================================
CCXIX发现"大"方向是语义空间的固有方向(cos=0.97-0.99), 但:
1. "大"方向是否等于某个SVD主轴(PC)?
2. "大"方向与3维语义流形(CCXV)的3个主轴有什么关系?
3. 推理方向在SVD空间中的投影结构?

设计:
  - 收集大量"A is bigger than B"的残差
  - PCA/SVD分析推理残差空间
  - 计算推理方向与W_U的SVD主轴的对齐度
  - 计算"大"方向与语义关系SVD主轴的cos
  - 测试: perturb "大"方向对应的PC, 看是否影响比较推理

用法:
  python ccxxi_reasoning_geometry.py --model qwen3
  python ccxxi_reasoning_geometry.py --model glm4
  python ccxxi_reasoning_geometry.py --model deepseek7b
"""
import argparse, os, sys, json, gc
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
import numpy as np
import torch
from scipy import stats
from scipy.sparse.linalg import svds

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import load_model, get_layers, get_model_info, release_model, get_W_U

TEMP = Path("tests/glm5_temp")
LOG = TEMP / "ccxxi_reasoning_geometry_log.txt"

# 比较词对
SIZE_PAIRS = [
    ("elephant", "mouse"), ("whale", "fish"), ("horse", "cat"),
    ("lion", "rabbit"), ("eagle", "sparrow"), ("bear", "fox"),
    ("cow", "chicken"), ("shark", "crab"), ("tiger", "rat"),
    ("mountain", "hill"), ("tree", "bush"), ("house", "shed"),
    ("bus", "car"), ("ship", "boat"), ("plane", "kite"),
]

WEIGHT_PAIRS = [
    ("iron", "feather"), ("rock", "leaf"), ("steel", "paper"),
    ("gold", "cotton"), ("lead", "silk"), ("stone", "grass"),
    ("concrete", "foam"), ("brick", "straw"), ("copper", "wool"),
    ("marble", "petal"), ("silver", "dust"), ("bronze", "thimble"),
]

SPEED_PAIRS = [
    ("cheetah", "turtle"), ("falcon", "snail"), ("horse", "slug"),
    ("rocket", "cart"), ("jet", "boat"), ("leopard", "worm"),
    ("eagle", "ant"), ("tiger", "sloth"), ("deer", "beetle"),
    ("train", "wheelbarrow"), ("car", "bicycle"), ("plane", "canoe"),
]

# 语义关系词(用于SVD主轴对比)
HABITAT_WORDS = {
    "land": ["dog", "cat", "lion", "tiger", "horse", "cow", "fox", "deer", "bear", "rabbit"],
    "ocean": ["whale", "shark", "dolphin", "octopus", "salmon", "seal", "crab", "squid"],
    "sky": ["eagle", "hawk", "owl", "parrot", "crow", "sparrow", "falcon", "swallow"],
}

TEMPLATE_COMPARE = "The {} is bigger than the {}"
TEMPLATE_HABITAT = "The {} lives in the"


def log(msg):
    print(msg, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def collect_resid(model, tokenizer, device, layers, prompt, test_layers):
    """收集各层残差"""
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    seq_len = toks.input_ids.shape[1]
    last_pos = seq_len - 1
    
    captured = {}
    def mk_hook(k):
        def hook(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            captured[k] = o[0, last_pos, :].detach().float().cpu().numpy()
        return hook
    
    hooks = [layers[li].register_forward_hook(mk_hook(f"L{li}")) for li in test_layers]
    with torch.no_grad():
        _ = model(**toks)
    for h in hooks:
        h.remove()
    return captured


def run(model_name):
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    layers = get_layers(model)
    n_layers, d_model = info.n_layers, info.d_model
    
    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]
    test_layers = sorted(set(test_layers))
    
    log(f"\n{'='*70}\nCCXXI(321): 推理几何的精确刻画 - {model_name}")
    log(f"  d_model={d_model}, n_layers={n_layers}")
    log(f"{'='*70}")
    
    results = {}
    W_U = get_W_U(model)  # [vocab_size, d_model]
    
    # ===== Step 1: 收集比较推理残差 =====
    log("\n--- Step 1: 收集比较推理残差 ---")
    
    # 收集所有比较对的残差
    compare_resids = {li: [] for li in test_layers}
    compare_dirs = {li: [] for li in test_layers}  # vec(A>B) 方向
    
    for dim_name, pairs in [("size", SIZE_PAIRS), ("weight", WEIGHT_PAIRS), ("speed", SPEED_PAIRS)]:
        for big, small in pairs:
            prompt = TEMPLATE_COMPARE.format(big, small)
            try:
                resid = collect_resid(model, tokenizer, device, layers, prompt, test_layers)
                for li in test_layers:
                    compare_resids[li].append(resid[f"L{li}"])
            except Exception as e:
                log(f"  跳过 {big}>{small}: {e}")
    
    # 也收集habitat残差(用于SVD对比)
    habitat_resids = {li: {"land": [], "ocean": [], "sky": []} for li in test_layers}
    for hab, words in HABITAT_WORDS.items():
        for word in words:
            prompt = TEMPLATE_HABITAT.format(word)
            try:
                resid = collect_resid(model, tokenizer, device, layers, prompt, test_layers)
                for li in test_layers:
                    habitat_resids[li][hab].append(resid[f"L{li}"])
            except Exception as e:
                pass
    
    log(f"  收集了 {len(compare_resids[test_layers[0]])} 个比较残差")
    
    # ===== Step 2: 比较残差空间的PCA =====
    log("\n--- Step 2: 比较残差空间的PCA ---")
    
    for li in test_layers:
        X = np.array(compare_resids[li])
        n_samples = X.shape[0]
        if n_samples < 3:
            continue
        
        # PCA via SVD
        mean = X.mean(axis=0)
        X_centered = X - mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 主成分方差解释比
        total_var = np.sum(S ** 2)
        explained_ratio = (S ** 2) / total_var
        
        log(f"  L{li}: top5 PC方差解释比 = {[round(r, 4) for r in explained_ratio[:5]]}")
        log(f"  L{li}: top5 奇异值 = {[round(s, 2) for s in S[:5]]}")
        
        results[f"pca_L{li}"] = {
            "layer": li,
            "n_samples": n_samples,
            "top5_singular_values": [round(float(s), 2) for s in S[:5]],
            "top5_explained_ratio": [round(float(r), 4) for r in explained_ratio[:5]],
            "top10_explained_ratio": [round(float(r), 4) for r in explained_ratio[:10]],
            "cumulative_5": round(float(np.cumsum(explained_ratio[:5])[-1]), 4),
        }
    
    # ===== Step 3: "大"方向 vs SVD主轴对齐 =====
    log("\n--- Step 3: 大方向 vs SVD主轴对齐 ---")
    
    # 计算"大"方向: 所有size比较的均值方向
    for li in test_layers:
        X = np.array(compare_resids[li])
        if X.shape[0] < 3:
            continue
        
        # PCA
        mean = X.mean(axis=0)
        X_centered = X - mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 计算比较方向(所有比较的均值-基线)
        # 用不同方式定义"大"方向
        # 方法1: 所有size比较的均值
        size_only = []
        for i, (big, small) in enumerate(SIZE_PAIRS):
            if i < len(compare_resids[li]):
                size_only.append(compare_resids[li][i])
        
        if len(size_only) < 3:
            continue
        
        big_dir_mean = np.mean(size_only, axis=0) - mean
        big_dir_norm = np.linalg.norm(big_dir_mean)
        if big_dir_norm < 1e-10:
            continue
        big_dir = big_dir_mean / big_dir_norm
        
        # 与各PC的cos
        pc_alignments = []
        for k in range(min(10, Vt.shape[0])):
            pc_dir = Vt[k]
            cos_val = float(np.abs(np.dot(big_dir, pc_dir)))
            pc_alignments.append(round(cos_val, 4))
        
        best_pc = np.argmax(pc_alignments)
        best_cos = pc_alignments[best_pc]
        
        log(f"  L{li}: 大方向 vs PC对齐 = {pc_alignments[:5]}")
        log(f"  L{li}: 最佳对齐 PC{best_pc} (cos={best_cos:.4f})")
        
        results[f"big_dir_vs_pc_L{li}"] = {
            "layer": li,
            "pc_alignments": pc_alignments,
            "best_pc": int(best_pc),
            "best_cos": round(best_cos, 4),
        }
    
    # ===== Step 4: "大"方向 vs W_U的SVD主轴 =====
    log("\n--- Step 4: 大方向 vs W_U的SVD主轴 ---")
    
    # W_U的SVD
    W_U_T = W_U.T.astype(np.float32)
    k_svd = min(50, min(W_U_T.shape) - 2)
    U_wu, S_wu, Vt_wu = svds(W_U_T, k=k_svd)
    U_wu = np.asarray(U_wu, dtype=np.float64)
    
    for li in test_layers:
        X = np.array(compare_resids[li])
        if X.shape[0] < 3:
            continue
        
        mean = X.mean(axis=0)
        size_only = [compare_resids[li][i] for i in range(min(len(SIZE_PAIRS), len(compare_resids[li])))]
        if len(size_only) < 3:
            continue
        
        big_dir_mean = np.mean(size_only, axis=0) - mean
        big_dir_norm = np.linalg.norm(big_dir_mean)
        if big_dir_norm < 1e-10:
            continue
        big_dir = big_dir_mean / big_dir_norm
        
        # 与W_U的SVD主轴对齐
        wu_alignments = []
        for k in range(min(10, U_wu.shape[1])):
            wu_dir = U_wu[:, k]
            wu_dir_norm = np.linalg.norm(wu_dir)
            if wu_dir_norm > 1e-10:
                wu_dir = wu_dir / wu_dir_norm
                cos_val = float(np.abs(np.dot(big_dir, wu_dir)))
                wu_alignments.append(round(cos_val, 4))
        
        # "大"方向在W_U行空间中的投影
        proj = U_wu @ (U_wu.T @ big_dir)
        proj_ratio = float(np.linalg.norm(proj) ** 2 / np.linalg.norm(big_dir) ** 2)
        
        log(f"  L{li}: 大方向 vs W_U PC对齐 = {wu_alignments[:5]}")
        log(f"  L{li}: 大方向在W_U行空间投影比 = {proj_ratio:.4f}")
        
        results[f"big_dir_vs_wu_L{li}"] = {
            "layer": li,
            "wu_alignments": wu_alignments,
            "wu_projection_ratio": round(proj_ratio, 4),
        }
    
    # ===== Step 5: "大"方向 vs 语义关系SVD主轴 =====
    log("\n--- Step 5: 大方向 vs 语义关系SVD主轴 ---")
    
    for li in test_layers:
        # 计算habitat SVD主轴
        hab_vecs = []
        for hab in ["land", "ocean", "sky"]:
            hab_vecs.extend(habitat_resids[li][hab])
        
        if len(hab_vecs) < 6:
            continue
        
        X_hab = np.array(hab_vecs)
        hab_mean = X_hab.mean(axis=0)
        X_hab_centered = X_hab - hab_mean
        
        U_hab, S_hab, Vt_hab = np.linalg.svd(X_hab_centered, full_matrices=False)
        
        # "大"方向
        size_only = [compare_resids[li][i] for i in range(min(len(SIZE_PAIRS), len(compare_resids[li])))]
        if len(size_only) < 3:
            continue
        
        compare_mean = np.mean(compare_resids[li], axis=0)
        big_dir_mean = np.mean(size_only, axis=0) - compare_mean
        big_dir_norm = np.linalg.norm(big_dir_mean)
        if big_dir_norm < 1e-10:
            continue
        big_dir = big_dir_mean / big_dir_norm
        
        # 与habitat PC的对齐
        hab_alignments = []
        for k in range(min(5, Vt_hab.shape[0])):
            hab_dir = Vt_hab[k]
            cos_val = float(np.abs(np.dot(big_dir, hab_dir)))
            hab_alignments.append(round(cos_val, 4))
        
        log(f"  L{li}: 大方向 vs habitat PC对齐 = {hab_alignments}")
        
        results[f"big_dir_vs_hab_L{li}"] = {
            "layer": li,
            "hab_alignments": hab_alignments,
        }
    
    # ===== Step 6: PC perturb测试 =====
    log("\n--- Step 6: PC perturb测试 ---")
    
    # 在最后层, perturb最佳对齐的PC方向, 看是否影响比较输出
    li_last = test_layers[-1]
    X = np.array(compare_resids[li_last])
    if X.shape[0] >= 3:
        mean = X.mean(axis=0)
        X_centered = X - mean
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 比较token
        bigger_tokens = ["bigger", "larger", "greater", "huge", "enormous"]
        smaller_tokens = ["smaller", "tiny", "little", "miniature", "minute"]
        
        # 找到bigger/smaller的logit token IDs
        bigger_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in bigger_tokens if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
        smaller_ids = [tokenizer.encode(t, add_special_tokens=False)[0] for t in smaller_tokens if len(tokenizer.encode(t, add_special_tokens=False)) > 0]
        
        # 测试perturb PC1方向
        test_prompt = "The elephant is bigger than the mouse. The whale is"
        
        # Baseline
        toks = tokenizer(test_prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            base_logits = model(**toks).logits[0, -1, :].detach().float().cpu().numpy()
        
        base_bigger = np.mean([base_logits[tid] for tid in bigger_ids if tid < len(base_logits)])
        base_smaller = np.mean([base_logits[tid] for tid in smaller_ids if tid < len(base_logits)]) if smaller_ids else 0
        
        # Perturb along PC1 at last layer
        pc1_dir = Vt[0]
        alpha = 5.0
        
        captured_perturb = {}
        def perturb_hook(m, inp, out):
            o = out[0] if isinstance(out, tuple) else out
            o_perturbed = o.clone()
            o_perturbed[0, -1, :] += alpha * torch.tensor(pc1_dir, dtype=o.dtype, device=device)
            if isinstance(out, tuple):
                return (o_perturbed,) + out[1:]
            return o_perturbed
        
        hook = layers[li_last].register_forward_hook(perturb_hook)
        with torch.no_grad():
            perturb_logits = model(**toks).logits[0, -1, :].detach().float().cpu().numpy()
        hook.remove()
        
        perturb_bigger = np.mean([perturb_logits[tid] for tid in bigger_ids if tid < len(perturb_logits)])
        perturb_smaller = np.mean([perturb_logits[tid] for tid in smaller_ids if tid < len(perturb_logits)])
        
        log(f"  PC1 perturb (alpha={alpha}):")
        log(f"    bigger logits: {base_bigger:.3f} → {perturb_bigger:.3f} (shift={perturb_bigger-base_bigger:.3f})")
        log(f"    smaller logits: {base_smaller:.3f} → {perturb_smaller:.3f} (shift={perturb_smaller-base_smaller:.3f})")
        
        results["pc_perturb"] = {
            "alpha": alpha,
            "base_bigger": round(float(base_bigger), 3),
            "perturb_bigger": round(float(perturb_bigger), 3),
            "bigger_shift": round(float(perturb_bigger - base_bigger), 3),
            "base_smaller": round(float(base_smaller), 3),
            "perturb_smaller": round(float(perturb_smaller), 3),
            "smaller_shift": round(float(perturb_smaller - base_smaller), 3),
        }
    
    # 保存结果
    out_path = TEMP / f"ccxxi_reasoning_geometry_{model_name}.json"
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
