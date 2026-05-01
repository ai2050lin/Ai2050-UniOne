"""
CCVI(306): 语义力线→W_U→Logit空间 映射分析
===============================================
前置发现:
  - d_model空间中, 类别质心差(cat_animals_tools等)是最有效的语义方向
  - LDA方向在深层可达100%top1改变
  - 需要理解: 语义力线如何被W_U映射到logit空间?

实验设计:
  Step 1: 计算各类别对在d_model空间的语义力线方向
  Step 2: 计算这些方向在W_U行空间中的投影比
  Step 3: 计算语义力线→logit的增益(recoding_gain)
  Step 4: 对比语义方向 vs 随机方向在W_U中的投影
  Step 5: 追踪语义力线在各层的演变

用法:
  python ccvi_semantic_to_logit.py --model qwen3
  python ccvi_semantic_to_logit.py --model glm4
  python ccvi_semantic_to_logit.py --model deepseek7b
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
    get_layer_weights, MODEL_CONFIGS,
)

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "ccvi_semantic_to_logit_log.txt"

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "lion", "bear", "fish", "snake", "whale",
                "rabbit", "deer", "fox", "wolf", "tiger", "monkey", "elephant", "dolphin", "parrot", "duck"],
    "food":    ["apple", "rice", "bread", "cheese", "pizza", "banana", "mango", "pasta", "salad", "steak",
                "soup", "cake", "cookie", "grape", "lemon", "peach", "corn", "bean", "pepper", "onion"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "screw", "pliers", "chisel", "level", "ruler",
                "shovel", "axe", "clamp", "welder", "plane", "anvil", "lathe", "forge", "drill", "mallet"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "valley", "canyon", "island", "meadow", "glacier",
                "volcano", "waterfall", "swamp", "tundra", "prairie", "cliff", "reef", "lagoon", "cave", "ridge"],
}

TEMPLATE = "The {} is"
N_CONCEPTS_PER_CAT = 15

# Category pairs for semantic force lines
CAT_PAIRS = [
    ("animals", "tools"),
    ("food", "nature"),
    ("animals", "food"),
    ("tools", "nature"),
    ("animals", "nature"),
    ("food", "tools"),
]


def log_f(msg="", end="\n"):
    print(msg, end=end, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + end)


def collect_residual_and_logits(model, tokenizer, device, model_info, sample_layers):
    """收集各层residual stream + 最终logits"""
    layers = get_layers(model)
    n_layers = model_info.n_layers
    categories = list(CONCEPTS.keys())
    
    all_words = []
    word_cats = []
    for cat in categories:
        words = CONCEPTS[cat][:N_CONCEPTS_PER_CAT]
        all_words.extend(words)
        word_cats.extend([cat] * len(words))
    
    n_words = len(all_words)
    log_f(f"  Collecting for {n_words} words")
    
    residual_stream = {}
    all_logits = {}
    
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
            out = model(**toks)
            logits = out.logits[0, -1].float().cpu().numpy()
            all_logits[word] = logits
        
        for h in hooks:
            h.remove()
        
        for li in sample_layers:
            key = f"L{li}"
            if key in captured:
                vec = captured[key][0, -1, :].numpy()
                if li not in residual_stream:
                    residual_stream[li] = []
                residual_stream[li].append(vec)
        
        if (wi + 1) % 20 == 0:
            log_f(f"    {wi+1}/{n_words} done")
    
    for li in residual_stream:
        residual_stream[li] = np.array(residual_stream[li])
    
    return residual_stream, all_logits, all_words, word_cats


def compute_semantic_force_lines(residual_stream, word_cats, categories, sample_layers):
    """计算各类别对的语义力线方向"""
    log_f("\n  === Step 1: Semantic Force Lines ===")
    
    cat_indices = defaultdict(list)
    for wi, cat in enumerate(word_cats):
        cat_indices[cat].append(wi)
    
    force_lines = {}  # (li, pair_name) -> direction
    
    for li in sample_layers:
        if li not in residual_stream:
            continue
        h = residual_stream[li]
        
        # Category centroids
        centroids = {c: h[cat_indices[c]].mean(axis=0) for c in categories}
        
        for cat1, cat2 in CAT_PAIRS:
            # Force line direction: cat1 -> cat2
            direction = centroids[cat2] - centroids[cat1]
            norm = np.linalg.norm(direction)
            if norm > 1e-10:
                direction_normalized = direction / norm
            else:
                direction_normalized = direction
            
            pair_name = f"{cat1}->{cat2}"
            force_lines[(li, pair_name)] = {
                "direction": direction,
                "direction_normalized": direction_normalized,
                "norm": float(norm),
                "centroid1_norm": float(np.linalg.norm(centroids[cat1])),
                "centroid2_norm": float(np.linalg.norm(centroids[cat2])),
            }
            
            log_f(f"  L{li} {pair_name}: ||dir||={norm:.2f}, "
                  f"||c1||={np.linalg.norm(centroids[cat1]):.1f}, "
                  f"||c2||={np.linalg.norm(centroids[cat2]):.1f}")
    
    return force_lines


def analyze_wu_projection(force_lines, W_U, d_model, sample_layers):
    """分析语义力线在W_U行空间中的投影"""
    log_f("\n  === Step 2: W_U Projection of Force Lines ===")
    
    vocab_size = W_U.shape[0]
    log_f(f"  W_U shape: {W_U.shape}")
    
    # Compute W_U SVD basis (top-500 components)
    W_U_T = W_U.T.astype(np.float32)  # [d_model, vocab_size]
    k = min(500, min(W_U_T.shape) - 2)
    U_wut, s_wut, _ = svds(W_U_T, k=k)
    U_wut = np.asarray(U_wut, dtype=np.float64)  # [d_model, k]
    sort_idx = np.argsort(-s_wut)
    U_wut = U_wut[:, sort_idx]
    s_wut = s_wut[sort_idx]
    
    # W_U effective rank
    total_energy = np.sum(s_wut**2)
    cum_energy = np.cumsum(s_wut**2) / total_energy
    r90 = np.searchsorted(cum_energy, 0.90) + 1
    r95 = np.searchsorted(cum_energy, 0.95) + 1
    r99 = np.searchsorted(cum_energy, 0.99) + 1
    log_f(f"  W_U effective rank: r90={r90}, r95={r95}, r99={r99}")
    log_f(f"  W_U top-5 sv: {s_wut[:5].round(2).tolist()}")
    
    results = {}
    
    for li in sample_layers:
        results[li] = {}
        
        # Semantic force lines
        sem_projections = []
        for (layer_idx, pair_name), fl in force_lines.items():
            if layer_idx != li:
                continue
            direction = fl["direction_normalized"]
            norm = fl["norm"]
            
            # Projection onto W_U row space
            proj_coeffs = U_wut.T @ direction  # [k]
            proj_energy = np.sum(proj_coeffs**2)
            recoding_ratio = min(proj_energy / max(np.linalg.norm(direction)**2, 1e-10), 1.0)
            
            # Top-10 component energies
            top10_energy = float(np.sum(np.sort(proj_coeffs**2)[-10:])) if len(proj_coeffs) >= 10 else float(np.sum(proj_coeffs**2))
            
            # Distribution across W_U SVD modes
            # Which W_U modes does this force line project onto?
            mode_weights = proj_coeffs**2
            top5_modes = np.argsort(-mode_weights)[:5]
            
            results[li][pair_name] = {
                "recoding_ratio": float(recoding_ratio),
                "proj_energy": float(proj_energy),
                "top10_energy": float(top10_energy),
                "force_line_norm": float(norm),
                "top5_wu_modes": top5_modes.tolist(),
                "top5_mode_weights": [float(mode_weights[m]) for m in top5_modes],
            }
            
            sem_projections.append(recoding_ratio)
            
            log_f(f"  L{li} {pair_name}: recoding_ratio={recoding_ratio:.4f}, "
                  f"proj_energy={proj_energy:.4f}, top5_modes={top5_modes.tolist()}")
        
        # Random directions for comparison
        np.random.seed(42)
        rnd_projections = []
        for ri in range(10):
            rand_dir = np.random.randn(d_model)
            rand_dir = rand_dir / np.linalg.norm(rand_dir)
            proj_coeffs = U_wut.T @ rand_dir
            proj_energy = np.sum(proj_coeffs**2)
            recoding_ratio = min(proj_energy / max(np.linalg.norm(rand_dir)**2, 1e-10), 1.0)
            rnd_projections.append(recoding_ratio)
        
        avg_sem = np.mean(sem_projections) if sem_projections else 0
        avg_rnd = np.mean(rnd_projections)
        
        results[li]["_summary"] = {
            "avg_semantic_ratio": float(avg_sem),
            "avg_random_ratio": float(avg_rnd),
            "semantic_random_ratio": float(avg_sem / max(avg_rnd, 1e-10)),
        }
        
        log_f(f"  L{li} summary: SEM avg={avg_sem:.4f}, RND avg={avg_rnd:.4f}, "
              f"SEM/RND={avg_sem/max(avg_rnd, 1e-10):.2f}")
    
    return results, U_wut, s_wut


def logit_space_analysis(force_lines, residual_stream, W_U, word_cats, categories, sample_layers):
    """分析语义力线→logit空间的效果"""
    log_f("\n  === Step 3: Logit Space Analysis ===")
    
    cat_indices = defaultdict(list)
    for wi, cat in enumerate(word_cats):
        cat_indices[cat].append(wi)
    
    results = {}
    
    for li in sample_layers:
        if li not in residual_stream:
            continue
        h = residual_stream[li]
        
        centroids = {c: h[cat_indices[c]].mean(axis=0) for c in categories}
        
        results[li] = {}
        
        for cat1, cat2 in CAT_PAIRS:
            pair_name = f"{cat1}->{cat2}"
            direction = centroids[cat2] - centroids[cat1]
            dir_norm = np.linalg.norm(direction)
            
            if dir_norm < 1e-10:
                continue
            direction_norm = direction / dir_norm
            
            # Project direction through W_U
            # If we move along this direction in d_model space by delta,
            # the logit change = W_U @ delta
            logit_shift = W_U @ direction  # [vocab_size]
            logit_shift_norm = np.linalg.norm(logit_shift)
            
            # Which tokens are most affected by this semantic force line?
            top_tokens = np.argsort(-np.abs(logit_shift))[:10]
            
            # Per-category token analysis: which category's tokens are affected?
            # Get category-representative tokens
            cat_token_effects = {}
            for cat in categories:
                cat_words = CONCEPTS[cat][:10]
                # For each word, check if the logit shift at its token position is positive or negative
                effects = []
                for word in cat_words:
                    # This is approximate - we're looking at the logit direction
                    pass  # Will analyze through actual logit differences
                cat_token_effects[cat] = 0  # placeholder
            
            # Variance of logit_shift (how focused is the effect?)
            logit_shift_var = float(np.var(logit_shift))
            logit_shift_max = float(np.max(np.abs(logit_shift)))
            logit_shift_mean = float(np.mean(np.abs(logit_shift)))
            focus_ratio = logit_shift_max / max(logit_shift_mean, 1e-10)
            
            results[li][pair_name] = {
                "dir_norm": float(dir_norm),
                "logit_shift_norm": float(logit_shift_norm),
                "logit_shift_var": float(logit_shift_var),
                "focus_ratio": float(focus_ratio),
                "logit_shift_per_dir_norm": float(logit_shift_norm / dir_norm),
            }
            
            log_f(f"  L{li} {pair_name}: ||dir||={dir_norm:.1f}, "
                  f"||W_U*dir||={logit_shift_norm:.2f}, "
                  f"||W_U*dir||/||dir||={logit_shift_norm/dir_norm:.4f}, "
                  f"focus={focus_ratio:.1f}")
        
        # Random direction comparison
        np.random.seed(42)
        rnd_shift_norms = []
        rnd_shift_per_dir = []
        for ri in range(10):
            rand_dir = np.random.randn(h.shape[1])
            rand_dir = rand_dir / np.linalg.norm(rand_dir)
            rand_logit_shift = W_U @ rand_dir
            rnd_shift_norms.append(np.linalg.norm(rand_logit_shift))
            rnd_shift_per_dir.append(np.linalg.norm(rand_logit_shift))
        
        avg_rnd_shift = np.mean(rnd_shift_per_dir)
        
        sem_shifts = [results[li][f"{c1}->{c2}"]["logit_shift_per_dir_norm"] 
                      for c1, c2 in CAT_PAIRS if f"{c1}->{c2}" in results[li]]
        avg_sem_shift = np.mean(sem_shifts) if sem_shifts else 0
        
        results[li]["_summary"] = {
            "avg_semantic_logit_shift": float(avg_sem_shift),
            "avg_random_logit_shift": float(avg_rnd_shift),
            "semantic_random_ratio": float(avg_sem_shift / max(avg_rnd_shift, 1e-10)),
        }
        
        log_f(f"  L{li} logit summary: SEM avg={avg_sem_shift:.4f}, "
              f"RND avg={avg_rnd_shift:.4f}, SEM/RND={avg_sem_shift/max(avg_rnd_shift,1e-10):.2f}")
    
    return results


def trace_force_line_evolution(force_lines, sample_layers):
    """追踪语义力线在各层的演变"""
    log_f("\n  === Step 4: Force Line Evolution ===")
    
    for cat1, cat2 in CAT_PAIRS:
        pair_name = f"{cat1}->{cat2}"
        norms = []
        for li in sample_layers:
            if (li, pair_name) in force_lines:
                norms.append((li, force_lines[(li, pair_name)]["norm"]))
        
        if norms:
            log_f(f"  {pair_name} norm evolution:")
            for li, n in norms:
                log_f(f"    L{li}: {n:.2f}")
            
            # Compute growth rate
            if len(norms) >= 2:
                first_li, first_n = norms[0]
                last_li, last_n = norms[-1]
                if first_n > 0 and last_li > first_li:
                    growth = (last_n / first_n) ** (1.0 / (last_li - first_li))
                    log_f(f"    Growth rate: {growth:.4f} per layer")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    t0 = time.time()
    
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"CCVI(306) Semantic→Logit - {model_name}\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    log_f(f"=== CCVI(306) Semantic→Logit - {model_name} ===")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    log_f(f"Model: {model_name}, L={model_info.n_layers}, d={model_info.d_model}")
    
    nl = model_info.n_layers
    sample_layers = sorted(set([0, nl//6, nl//4, nl//3, nl//2, 2*nl//3, 3*nl//4, 5*nl//6, nl-1]))
    log_f(f"Sample layers: {sample_layers}")
    
    # Step 1: Collect residual stream + logits
    log_f("\n=== Step 1: Collect Data ===")
    residual_stream, all_logits, all_words, word_cats = collect_residual_and_logits(
        model, tokenizer, device, model_info, sample_layers
    )
    
    categories = list(CONCEPTS.keys())
    
    # Step 2: Compute semantic force lines
    force_lines = compute_semantic_force_lines(residual_stream, word_cats, categories, sample_layers)
    
    # Step 3: Analyze W_U projection
    W_U = get_W_U(model)
    wu_results, U_wut, s_wut = analyze_wu_projection(force_lines, W_U, model_info.d_model, sample_layers)
    
    # Step 4: Logit space analysis
    logit_results = logit_space_analysis(force_lines, residual_stream, W_U, word_cats, categories, sample_layers)
    
    # Step 5: Force line evolution
    trace_force_line_evolution(force_lines, sample_layers)
    
    # Release model
    release_model(model)
    gc.collect()
    
    # Save results
    save_data = {
        "model": model_name,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
        "sample_layers": sample_layers,
        "wu_effective_rank": {
            "r90": int(np.searchsorted(np.cumsum(s_wut**2)/np.sum(s_wut**2), 0.90) + 1),
            "r95": int(np.searchsorted(np.cumsum(s_wut**2)/np.sum(s_wut**2), 0.95) + 1),
        },
        "wu_projection": {},
        "logit_analysis": {},
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
    
    # Save W_U projection results
    for li in sample_layers:
        li_data = {}
        for pair_name, data in wu_results.get(li, {}).items():
            li_data[pair_name] = data
        save_data["wu_projection"][str(li)] = li_data
    
    # Save logit results
    for li in sample_layers:
        li_data = {}
        for pair_name, data in logit_results.get(li, {}).items():
            li_data[pair_name] = data
        save_data["logit_analysis"][str(li)] = li_data
    
    save_data = convert(save_data)
    
    json_path = TEMP_DIR / f"ccvi_semantic_to_logit_{model_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log_f(f"\nResults saved to {json_path}")
    
    elapsed = time.time() - t0
    log_f(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
