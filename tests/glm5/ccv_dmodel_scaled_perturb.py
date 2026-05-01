"""
CCV(305): d_model空间norm-relative语义方向Perturb
===================================================
CCIV发现: d_model swap有效(50-67%), 但加性perturb无效(eps太小)。
现在用norm-relative eps, 在d_model空间做语义方向vs随机方向perturb。

关键改进:
  eps = alpha * ||h||, alpha = 0.1, 0.2, 0.5
  这样perturbation相对于residual stream的大小是固定的

实验:
  1. 语义方向: ANOVA top维度方向, LDA方向, PC1方向
  2. 随机方向: 同维度随机方向
  3. 测量: top1改变率, ||Δlogits||, logit_cos

用法:
  python ccv_dmodel_scaled_perturb.py --model qwen3
  python ccv_dmodel_scaled_perturb.py --model glm4
  python ccv_dmodel_scaled_perturb.py --model deepseek7b
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

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, MODEL_CONFIGS,
)

TEMP_DIR = Path("tests/glm5_temp")
LOG_FILE = TEMP_DIR / "ccv_dmodel_scaled_perturb_log.txt"

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

# Test words (disjoint from training)
TEST_WORDS = {
    "animals": ["tiger", "eagle", "whale", "fox", "deer", "zebra", "owl", "seal", "crow", "hawk"],
    "food":    ["mango", "steak", "soup", "grape", "peach", "plum", "corn", "bean", "lamb", "pear"],
    "tools":   ["pliers", "chisel", "shovel", "clamp", "anvil", "vice", "file", "wedge", "lever", "pulley"],
    "nature":  ["glacier", "canyon", "meadow", "swamp", "cliff", "tundra", "dune", "gorge", "brook", "fjord"],
}

ALPHA_LIST = [0.05, 0.1, 0.2, 0.5]  # eps = alpha * ||h||
N_RANDOM = 5


def log_f(msg="", end="\n"):
    print(msg, end=end, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(msg + end)


def collect_and_analyze(model, tokenizer, device, model_info, sample_layers):
    """收集residual stream + ANOVA + PCA, 返回方向集"""
    layers = get_layers(model)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    categories = list(CONCEPTS.keys())
    
    # 收集所有词的residual stream
    all_words = []
    word_cats = []
    for cat in categories:
        words = CONCEPTS[cat][:N_CONCEPTS_PER_CAT]
        all_words.extend(words)
        word_cats.extend([cat] * len(words))
    
    n_words = len(all_words)
    log_f(f"  Collecting for {n_words} words across {len(sample_layers)} layers")
    
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
            key = f"L{li}"
            if key in captured:
                vec = captured[key][0, -1, :].numpy()
                if li not in residual_stream:
                    residual_stream[li] = []
                residual_stream[li].append(vec)
    
    for li in residual_stream:
        residual_stream[li] = np.array(residual_stream[li])
    
    # ANOVA
    cat_indices = defaultdict(list)
    for wi, cat in enumerate(word_cats):
        cat_indices[cat].append(wi)
    
    directions_per_layer = {}
    norms_per_layer = {}
    
    for li in sample_layers:
        h = residual_stream[li]
        avg_norm = float(np.linalg.norm(h, axis=1).mean())
        norms_per_layer[li] = avg_norm
        
        # ANOVA per dimension
        groups = [h[cat_indices[c]] for c in categories]
        
        top_dims = []
        for dim in range(d_model):
            dim_vals = [g[:, dim] for g in groups]
            F, p = stats.f_oneway(*dim_vals)
            if p < 0.001:
                ss_between = sum(len(g) * (g[:, dim].mean() - h[:, dim].mean())**2 for g in groups)
                ss_total = np.sum((h[:, dim] - h[:, dim].mean())**2)
                eta_sq = ss_between / max(ss_total, 1e-10)
                per_cat_means = {c: float(g[:, dim].mean()) for c, g in zip(categories, groups)}
                top_dims.append((float(F), dim, float(eta_sq), per_cat_means))
        
        top_dims.sort(reverse=True)
        
        # Build directions
        directions = {}
        
        # (a) ANOVA top-1
        if len(top_dims) > 0:
            dim1 = top_dims[0][1]
            dir_anova1 = np.zeros(d_model)
            dir_anova1[dim1] = 1.0
            directions["anova_top1"] = dir_anova1
        
        # (b) ANOVA top-5 weighted
        if len(top_dims) >= 5:
            dir_anova5 = np.zeros(d_model)
            for F, dim, eta_sq, means in top_dims[:5]:
                dir_anova5[dim] = F
            dir_anova5 = dir_anova5 / max(np.linalg.norm(dir_anova5), 1e-10)
            directions["anova_top5"] = dir_anova5
        
        # (c) ANOVA top-20 weighted
        if len(top_dims) >= 20:
            dir_anova20 = np.zeros(d_model)
            for F, dim, eta_sq, means in top_dims[:20]:
                dir_anova20[dim] = F
            dir_anova20 = dir_anova20 / max(np.linalg.norm(dir_anova20), 1e-10)
            directions["anova_top20"] = dir_anova20
        
        # (d) LDA
        n_lda = min(50, len(top_dims))
        if n_lda >= 4:
            lda_dims = [d for _, d, _, _ in top_dims[:n_lda]]
            h_sub = h[:, lda_dims]
            overall_mean = h_sub.mean(axis=0)
            
            S_w = np.zeros((n_lda, n_lda))
            S_b = np.zeros((n_lda, n_lda))
            for cat in categories:
                idx = cat_indices[cat]
                class_mean = h_sub[idx].mean(axis=0)
                S_b += len(idx) * np.outer(class_mean - overall_mean, class_mean - overall_mean)
                for x in h_sub[idx]:
                    diff = (x - class_mean).reshape(-1, 1)
                    S_w += diff @ diff.T
            
            try:
                S_w_reg = S_w + 1e-6 * np.eye(n_lda)
                S_w_inv = np.linalg.pinv(S_w_reg)
                eigvals, eigvecs = np.linalg.eig(S_w_inv @ S_b)
                sort_idx = np.argsort(-np.real(eigvals))
                lda_dir_sub = np.real(eigvecs[:, sort_idx[0]])
                
                dir_lda = np.zeros(d_model)
                for i, dim in enumerate(lda_dims):
                    dir_lda[dim] = lda_dir_sub[i]
                dir_lda = dir_lda / max(np.linalg.norm(dir_lda), 1e-10)
                directions["lda"] = dir_lda
            except:
                pass
        
        # (e) PC1
        h_centered = h - h.mean(axis=0, keepdims=True)
        U, s, Vt = np.linalg.svd(h_centered, full_matrices=False)
        directions["pc1"] = Vt[0] / max(np.linalg.norm(Vt[0]), 1e-10)
        
        # (f) Category centroid difference direction (animals - tools)
        cat_centroids = {c: h[cat_indices[c]].mean(axis=0) for c in categories}
        dir_cat = cat_centroids["animals"] - cat_centroids["tools"]
        dir_cat = dir_cat / max(np.linalg.norm(dir_cat), 1e-10)
        directions["cat_animals_tools"] = dir_cat
        
        # Another category pair
        dir_cat2 = cat_centroids["food"] - cat_centroids["nature"]
        dir_cat2 = dir_cat2 / max(np.linalg.norm(dir_cat2), 1e-10)
        directions["cat_food_nature"] = dir_cat2
        
        # (g) Random directions
        np.random.seed(42)
        for ri in range(N_RANDOM):
            rand_dir = np.random.randn(d_model)
            rand_dir = rand_dir / np.linalg.norm(rand_dir)
            directions[f"random_{ri}"] = rand_dir
        
        directions_per_layer[li] = directions
        log_f(f"  L{li}: ||h||={avg_norm:.1f}, {len(directions)} directions, "
              f"top ANOVA F={top_dims[0][0]:.1f} (dim{top_dims[0][1]})" if top_dims else "")
    
    return directions_per_layer, norms_per_layer


def run_perturb(model, tokenizer, device, model_info, directions_per_layer, norms_per_layer, sample_layers):
    """Norm-relative perturb实验"""
    log_f("\n  === Step 2: Norm-Relative Perturb ===")
    
    layers = get_layers(model)
    d_model = model_info.d_model
    categories = list(TEST_WORDS.keys())
    
    results = {}
    
    for li in sample_layers:
        if li not in directions_per_layer:
            continue
        
        directions = directions_per_layer[li]
        avg_norm = norms_per_layer[li]
        
        log_f(f"\n  --- L{li} (||h||={avg_norm:.1f}) ---")
        
        results[li] = {}
        
        for dir_name, direction in directions.items():
            results[li][dir_name] = {}
            
            for alpha in ALPHA_LIST:
                eps = alpha * avg_norm  # Scale relative to norm
                
                top1_changes = 0
                total_tests = 0
                delta_norms = []
                logit_coss = []
                
                for cat, words in TEST_WORDS.items():
                    for word in words:
                        prompt = TEMPLATE.format(word)
                        toks = tokenizer(prompt, return_tensors="pt").to(device)
                        
                        # Baseline
                        with torch.no_grad():
                            base_out = model(**toks)
                            base_logits = base_out.logits[0, -1].float().cpu().numpy()
                            base_top1 = int(np.argmax(base_logits))
                        
                        # Hook perturb
                        def make_perturb_hook(direction_np, eps_val):
                            def hook(module, input, output):
                                if isinstance(output, tuple):
                                    out = output[0].clone()
                                else:
                                    out = output.clone()
                                dir_tensor = torch.tensor(direction_np, dtype=out.dtype, device=out.device)
                                out[0, -1, :] += eps_val * dir_tensor
                                if isinstance(output, tuple):
                                    return (out,) + output[1:]
                                return out
                            return hook
                        
                        hook = layers[li].register_forward_hook(make_perturb_hook(direction, eps))
                        
                        with torch.no_grad():
                            pert_out = model(**toks)
                            pert_logits = pert_out.logits[0, -1].float().cpu().numpy()
                            pert_top1 = int(np.argmax(pert_logits))
                        
                        hook.remove()
                        
                        delta_logits = pert_logits - base_logits
                        delta_norm = float(np.linalg.norm(delta_logits))
                        
                        base_norm = float(np.linalg.norm(base_logits))
                        pert_norm = float(np.linalg.norm(pert_logits))
                        if base_norm > 0 and pert_norm > 0:
                            cos = float(np.dot(pert_logits, base_logits) / (pert_norm * base_norm))
                        else:
                            cos = 1.0
                        
                        if pert_top1 != base_top1:
                            top1_changes += 1
                        total_tests += 1
                        delta_norms.append(delta_norm)
                        logit_coss.append(cos)
                
                avg_delta = float(np.mean(delta_norms))
                avg_cos = float(np.mean(logit_coss))
                top1_rate = top1_changes / max(total_tests, 1)
                
                results[li][dir_name][str(alpha)] = {
                    "top1_rate": float(top1_rate),
                    "avg_delta_norm": avg_delta,
                    "avg_logit_cos": avg_cos,
                    "eps": float(eps),
                    "n_tests": total_tests,
                }
                
                is_semantic = not dir_name.startswith("random_")
                label = "SEM" if is_semantic else "RND"
                log_f(f"    {label} {dir_name} alpha={alpha} eps={eps:.1f}: "
                      f"top1={top1_rate:.3f} ||dL||={avg_delta:.1f} cos={avg_cos:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "glm4", "deepseek7b"])
    args = parser.parse_args()
    
    model_name = args.model
    t0 = time.time()
    
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write(f"CCV(305) d_model Norm-Relative Perturb - {model_name}\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    log_f(f"=== CCV(305) d_model Norm-Relative Perturb - {model_name} ===")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    log_f(f"Model: {model_name}, L={model_info.n_layers}, d={model_info.d_model}")
    
    nl = model_info.n_layers
    # Focus on mid-to-late layers where swap was effective
    sample_layers = sorted(set([nl//4, nl//3, nl//2, 2*nl//3, 3*nl//4, nl-1]))
    log_f(f"Sample layers: {sample_layers}")
    
    # Step 1: Collect and analyze
    log_f("\n=== Step 1: Collect + Analyze ===")
    directions_per_layer, norms_per_layer = collect_and_analyze(
        model, tokenizer, device, model_info, sample_layers
    )
    
    # Step 2: Norm-relative perturb
    perturb_results = run_perturb(
        model, tokenizer, device, model_info,
        directions_per_layer, norms_per_layer, sample_layers
    )
    
    # Save
    save_data = {
        "model": model_name,
        "n_layers": model_info.n_layers,
        "d_model": model_info.d_model,
        "n_inter": model_info.intermediate_size,
        "sample_layers": sample_layers,
        "norms": {str(k): v for k, v in norms_per_layer.items()},
        "perturb": {},
    }
    
    for li, dirs in perturb_results.items():
        save_data["perturb"][str(li)] = dirs
    
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
    
    save_data = convert(save_data)
    
    json_path = TEMP_DIR / f"ccv_dmodel_scaled_perturb_{model_name}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    log_f(f"\nResults saved to {json_path}")
    
    # Summary
    log_f("\n=== Summary ===")
    for li in sample_layers:
        if str(li) not in save_data["perturb"]:
            continue
        log_f(f"\nL{li} (||h||={norms_per_layer[li]:.1f}):")
        for alpha in ALPHA_LIST:
            sem_deltas = []
            rnd_deltas = []
            sem_top1s = []
            rnd_top1s = []
            sem_coss = []
            rnd_coss = []
            
            for dir_name, eps_data in save_data["perturb"][str(li)].items():
                a_str = str(alpha)
                if a_str not in eps_data:
                    continue
                d = eps_data[a_str]
                if dir_name.startswith("random_"):
                    rnd_deltas.append(d["avg_delta_norm"])
                    rnd_top1s.append(d["top1_rate"])
                    rnd_coss.append(d["avg_logit_cos"])
                else:
                    sem_deltas.append(d["avg_delta_norm"])
                    sem_top1s.append(d["top1_rate"])
                    sem_coss.append(d["avg_logit_cos"])
            
            if sem_deltas and rnd_deltas:
                log_f(f"  alpha={alpha}: SEM ||dL||={np.mean(sem_deltas):.1f} vs "
                      f"RND ||dL||={np.mean(rnd_deltas):.1f}, "
                      f"SEM top1={np.mean(sem_top1s):.3f} vs RND top1={np.mean(rnd_top1s):.3f}, "
                      f"SEM cos={np.mean(sem_coss):.4f} vs RND cos={np.mean(rnd_coss):.4f}")
    
    release_model(model)
    gc.collect()
    
    elapsed = time.time() - t0
    log_f(f"\nTotal time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
