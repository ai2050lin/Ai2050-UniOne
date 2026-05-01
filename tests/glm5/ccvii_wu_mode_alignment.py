"""
CCVII(307): 语义方向 vs W_U SVD模式对齐度
============================================
前置发现: DS7B L27的语义方向被W_U放大7x(vs随机), 
但投影比仅1.78x。假设: 语义方向对齐W_U大奇异值模式。

验证:
  1. 计算W_U的SVD
  2. 计算语义方向在W_U各SVD模式上的投影分布
  3. 对比随机方向的分布
  4. 计算"对齐度": 语义方向投影到大奇异值模式的能量占比

用法:
  python ccvii_wu_mode_alignment.py --model qwen3
  python ccvii_wu_mode_alignment.py --model glm4
  python ccvii_wu_mode_alignment.py --model deepseek7b
"""
import argparse, os, sys, time, gc, json
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from collections import defaultdict
from scipy import stats
from scipy.sparse.linalg import svds

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
)

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "ccvii_wu_mode_alignment_log.txt"

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "lion", "bear", "fish", "snake", "whale",
                "rabbit", "deer", "fox", "wolf", "tiger"],
    "food":    ["apple", "rice", "bread", "cheese", "pizza", "banana", "mango", "pasta", "salad", "steak",
                "soup", "cake", "cookie", "grape", "lemon"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "screw", "pliers", "chisel", "level", "ruler",
                "shovel", "axe", "clamp", "welder", "plane"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "valley", "canyon", "island", "meadow", "glacier",
                "volcano", "waterfall", "swamp", "tundra", "prairie"],
}

TEMPLATE = "The {} is"

CAT_PAIRS = [
    ("animals", "tools"),
    ("food", "nature"),
    ("animals", "food"),
    ("tools", "nature"),
]


def log_f(msg="", end="\n"):
    print(msg, end=end, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + end)


def compute_wu_svd(W_U, d_model, n_modes=500):
    """计算W_U的SVD"""
    W_U_T = W_U.T.astype(np.float32)  # [d_model, vocab]
    k = min(n_modes, min(W_U_T.shape) - 2)
    U, s, Vt = svds(W_U_T, k=k)
    # Sort by decreasing singular value
    sort_idx = np.argsort(-s)
    U = np.asarray(U[:, sort_idx], dtype=np.float64)  # [d_model, k]
    s = s[sort_idx]
    return U, s


def collect_force_lines(model, tokenizer, device, model_info, sample_layers):
    """收集语义力线方向"""
    layers = get_layers(model)
    categories = list(CONCEPTS.keys())
    
    all_words = []
    word_cats = []
    for cat in categories:
        words = CONCEPTS[cat][:15]
        all_words.extend(words)
        word_cats.extend([cat] * len(words))
    
    residual_stream = {}
    for wi, word in enumerate(all_words):
        prompt = TEMPLATE.format(word)
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        
        captured = {}
        hooks = []
        def make_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    captured[key] = output[0].detach().float().cpu()
                else:
                    captured[key] = output.detach().float().cpu()
            return hook
        
        for li in sample_layers:
            hooks.append(layers[li].register_forward_hook(make_hook(f"L{li}")))
        
        with torch.no_grad():
            try:
                _ = model(**toks)
            except:
                pass
        
        for h in hooks:
            h.remove()
        
        for li in sample_layers:
            if f"L{li}" in captured:
                vec = captured[f"L{li}"][0, -1, :].numpy()
                if li not in residual_stream:
                    residual_stream[li] = []
                residual_stream[li].append(vec)
    
    for li in residual_stream:
        residual_stream[li] = np.array(residual_stream[li])
    
    # Compute force lines
    cat_indices = defaultdict(list)
    for wi, cat in enumerate(word_cats):
        cat_indices[cat].append(wi)
    
    force_lines = {}
    for li in sample_layers:
        if li not in residual_stream:
            continue
        h = residual_stream[li]
        centroids = {c: h[cat_indices[c]].mean(axis=0) for c in categories}
        
        for cat1, cat2 in CAT_PAIRS:
            direction = centroids[cat2] - centroids[cat1]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction_norm = direction / norm
            else:
                direction_norm = direction
            pair_name = f"{cat1}->{cat2}"
            force_lines[(li, pair_name)] = direction_norm
    
    return force_lines


def analyze_mode_alignment(force_lines, U_wu, s_wu, d_model, sample_layers):
    """分析语义方向与W_U SVD模式的对齐度"""
    log_f("\n  === Mode Alignment Analysis ===")
    
    n_modes = U_wu.shape[1]
    
    results = {}
    
    for li in sample_layers:
        li_results = {}
        
        # Semantic directions
        sem_mode_distributions = []
        for (layer_idx, pair_name), direction in force_lines.items():
            if layer_idx != li:
                continue
            
            # Project onto W_U SVD modes
            proj = U_wu.T @ direction  # [n_modes]
            proj_energy = proj**2  # Energy per mode
            total_energy = np.sum(proj_energy)
            
            if total_energy < 1e-10:
                continue
            
            # Normalized energy distribution
            energy_dist = proj_energy / total_energy
            
            # Cumulative energy from top modes
            sorted_energy = np.sort(energy_dist)[::-1]
            cum_energy = np.cumsum(sorted_energy)
            
            # Key metrics
            top10_energy = float(np.sum(sorted_energy[:10]))
            top50_energy = float(np.sum(sorted_energy[:50]))
            top100_energy = float(np.sum(sorted_energy[:100]))
            
            # Weighted mode index (average mode weighted by energy)
            weighted_idx = float(np.sum(np.arange(1, n_modes+1) * energy_dist))
            
            # Energy in top-10% modes
            top10pct = int(0.1 * n_modes)
            top10pct_energy = float(np.sum(sorted_energy[:top10pct]))
            
            li_results[pair_name] = {
                "top10_energy": top10_energy,
                "top50_energy": top50_energy,
                "top100_energy": top100_energy,
                "top10pct_energy": top10pct_energy,
                "weighted_mode_idx": weighted_idx,
                "total_proj_energy": float(total_energy),
                "energy_dist_top20": sorted_energy[:20].tolist(),
            }
            
            sem_mode_distributions.append(energy_dist)
            
            log_f(f"  L{li} {pair_name}: top10={top10_energy:.4f}, top50={top50_energy:.4f}, "
                  f"top100={top100_energy:.4f}, weighted_idx={weighted_idx:.1f}")
        
        # Random directions
        np.random.seed(42)
        rnd_mode_distributions = []
        rnd_results = []
        for ri in range(20):
            rand_dir = np.random.randn(d_model)
            rand_dir = rand_dir / np.linalg.norm(rand_dir)
            
            proj = U_wu.T @ rand_dir
            proj_energy = proj**2
            total_energy = np.sum(proj_energy)
            
            if total_energy < 1e-10:
                continue
            
            energy_dist = proj_energy / total_energy
            rnd_mode_distributions.append(energy_dist)
            
            sorted_energy = np.sort(energy_dist)[::-1]
            top10_energy = float(np.sum(sorted_energy[:10]))
            top50_energy = float(np.sum(sorted_energy[:50]))
            top100_energy = float(np.sum(sorted_energy[:100]))
            weighted_idx = float(np.sum(np.arange(1, n_modes+1) * energy_dist))
            
            rnd_results.append({
                "top10_energy": top10_energy,
                "top50_energy": top50_energy,
                "top100_energy": top100_energy,
                "weighted_mode_idx": weighted_idx,
            })
        
        # Compare SEM vs RND
        if sem_mode_distributions and rnd_mode_distributions:
            avg_sem = {
                "top10": np.mean([r["top10_energy"] for r in li_results.values()]),
                "top50": np.mean([r["top50_energy"] for r in li_results.values()]),
                "top100": np.mean([r["top100_energy"] for r in li_results.values()]),
                "weighted_idx": np.mean([r["weighted_mode_idx"] for r in li_results.values()]),
            }
            avg_rnd = {
                "top10": np.mean([r["top10_energy"] for r in rnd_results]),
                "top50": np.mean([r["top50_energy"] for r in rnd_results]),
                "top100": np.mean([r["top100_energy"] for r in rnd_results]),
                "weighted_idx": np.mean([r["weighted_mode_idx"] for r in rnd_results]),
            }
            
            li_results["_summary"] = {
                "avg_semantic": avg_sem,
                "avg_random": avg_rnd,
                "top10_ratio": float(avg_sem["top10"] / max(avg_rnd["top10"], 1e-10)),
                "top50_ratio": float(avg_sem["top50"] / max(avg_rnd["top50"], 1e-10)),
                "weighted_idx_ratio": float(avg_sem["weighted_idx"] / max(avg_rnd["weighted_idx"], 1e-10)),
            }
            
            log_f(f"  L{li} SUMMARY: SEM top10={avg_sem['top10']:.4f} vs RND={avg_rnd['top10']:.4f} "
                  f"(ratio={avg_sem['top10']/max(avg_rnd['top10'],1e-10):.2f}), "
                  f"SEM weighted_idx={avg_sem['weighted_idx']:.1f} vs RND={avg_rnd['weighted_idx']:.1f}")
        
        results[li] = li_results
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    t0 = time.time()
    
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"CCVII(307) W_U Mode Alignment - {model_name}\n\n")
    
    log_f(f"=== CCVII(307) W_U Mode Alignment - {model_name} ===")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    log_f(f"Model: {model_name}, L={model_info.n_layers}, d={model_info.d_model}")
    
    nl = model_info.n_layers
    sample_layers = sorted(set([nl//4, nl//2, 3*nl//4, nl-1]))
    log_f(f"Sample layers: {sample_layers}")
    
    # Step 1: Compute W_U SVD
    log_f("\n=== Step 1: W_U SVD ===")
    W_U = get_W_U(model)
    U_wu, s_wu = compute_wu_svd(W_U, model_info.d_model)
    log_f(f"  W_U SVD: {U_wu.shape}, top-5 sv: {s_wu[:5].round(2).tolist()}")
    
    # Step 2: Collect force lines
    log_f("\n=== Step 2: Collect Force Lines ===")
    force_lines = collect_force_lines(model, tokenizer, device, model_info, sample_layers)
    log_f(f"  Collected {len(force_lines)} force lines")
    
    # Step 3: Analyze mode alignment
    results = analyze_mode_alignment(force_lines, U_wu, s_wu, model_info.d_model, sample_layers)
    
    # Release model
    release_model(model)
    gc.collect()
    
    # Save
    save_data = {
        "model": model_name,
        "d_model": model_info.d_model,
        "wu_n_modes": U_wu.shape[1],
        "wu_top5_sv": s_wu[:5].tolist(),
        "results": {},
    }
    
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(x) for x in obj]
        return obj
    
    save_data["results"] = convert(results)
    
    json_path = TEMP_DIR / f"ccvii_wu_mode_alignment_{model_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log_f(f"\nResults saved to {json_path}")
    
    elapsed = time.time() - t0
    log_f(f"Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
