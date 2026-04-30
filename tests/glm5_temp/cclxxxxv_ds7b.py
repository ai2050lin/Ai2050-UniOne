"""CCLXXXXV DS7B only runner"""
import sys, os
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

LOG = r"d:\Ai2050\TransformerLens-Project\tests\glm5_temp\cclxxxxv_log.txt"

def log(msg):
    with open(LOG, 'a', encoding='utf-8') as f:
        import time
        f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        f.flush()

log("=== CCLXXXXV DS7B-only run started ===")

import json, time, gc, traceback
from pathlib import Path
import numpy as np
from collections import defaultdict
from itertools import combinations

import torch
sys.path.insert(0, r"d:\Ai2050\TransformerLens-Project")
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
    get_layer_weights, MODEL_CONFIGS,
)

CATEGORIES_13 = {
    "animal": ["dog", "cat", "horse", "cow", "pig", "sheep", "goat", "donkey", "lion", "tiger", "bear", "wolf", "fox", "deer", "rabbit", "elephant", "giraffe", "zebra", "monkey", "camel"],
    "bird": ["eagle", "hawk", "owl", "crow", "swan", "goose", "duck", "penguin", "parrot", "robin", "sparrow", "pigeon", "seagull", "falcon", "vulture", "crane", "stork", "heron", "peacock", "flamingo"],
    "fish": ["shark", "whale", "dolphin", "salmon", "trout", "tuna", "cod", "bass", "carp", "catfish", "perch", "pike", "eel", "herring", "sardine", "anchovy", "flounder", "sole", "mackerel", "swordfish"],
    "insect": ["ant", "bee", "spider", "butterfly", "mosquito", "fly", "wasp", "beetle", "cockroach", "grasshopper", "cricket", "dragonfly", "ladybug", "moth", "flea", "tick", "mantis", "caterpillar", "worm", "snail"],
    "plant": ["tree", "flower", "grass", "bush", "shrub", "vine", "fern", "moss", "algae", "weed", "oak", "pine", "maple", "birch", "willow", "cactus", "bamboo", "palm", "rose", "lily"],
    "fruit": ["apple", "orange", "banana", "grape", "pear", "peach", "cherry", "plum", "mango", "lemon", "lime", "melon", "berry", "strawberry", "blueberry", "raspberry", "fig", "date", "coconut", "pineapple"],
    "vegetable": ["carrot", "potato", "tomato", "onion", "garlic", "cabbage", "lettuce", "spinach", "celery", "pea", "bean", "corn", "mushroom", "pepper", "cucumber", "pumpkin", "squash", "radish", "turnip", "broccoli"],
    "body_part": ["hand", "foot", "head", "heart", "brain", "eye", "ear", "nose", "mouth", "tooth", "neck", "shoulder", "arm", "finger", "knee", "chest", "back", "hip", "ankle", "wrist"],
    "tool": ["hammer", "knife", "scissors", "saw", "drill", "wrench", "screwdriver", "plier", "axe", "chisel", "ruler", "file", "clamp", "level", "shovel", "rake", "hoe", "trowel", "spade", "mallet"],
    "vehicle": ["car", "bus", "truck", "train", "bicycle", "motorcycle", "airplane", "helicopter", "boat", "ship", "submarine", "rocket", "tractor", "van", "taxi", "ambulance", "sled", "canoe", "wagon", "cart"],
    "clothing": ["shirt", "dress", "hat", "coat", "shoe", "belt", "scarf", "glove", "jacket", "sweater", "vest", "skirt", "pants", "jeans", "sock", "boot", "sandal", "tie", "uniform", "cape"],
    "weapon": ["sword", "spear", "bow", "arrow", "shield", "axe_w", "dagger", "mace", "pike_w", "lance", "crossbow", "catapult", "pistol", "rifle", "cannon", "grenade", "dynamite", "knife_w", "club", "whip"],
    "furniture": ["chair", "table", "desk", "bed", "sofa", "couch", "shelf", "cabinet", "drawer", "wardrobe", "dresser", "bench", "stool", "armchair", "bookcase", "mirror", "lamp", "rug", "curtain", "pillow"],
}

SUPERCLASS_MAP = {
    "animal": "animate", "bird": "animate", "fish": "animate", "insect": "animate",
    "plant": "plant", "fruit": "plant", "vegetable": "plant",
    "body_part": "body",
    "tool": "artifact", "vehicle": "artifact", "clothing": "artifact",
    "weapon": "artifact", "furniture": "artifact",
}

FRACTURE_LAYERS = {"qwen3": 6, "glm4": 2, "deepseek7b": 7}


def compute_stats(vecs):
    vecs = np.array(vecs)
    mean = vecs.mean(axis=0)
    var_per_dim = vecs.var(axis=0)
    total_var = var_per_dim.mean()
    norms = np.linalg.norm(vecs, axis=1)
    avg_norm = norms.mean()
    d = vecs.shape[1]
    sample_dims = np.linspace(0, d-1, min(100, d), dtype=int)
    kurtosis_vals, skew_vals = [], []
    for dim in sample_dims:
        vals = vecs[:, dim]
        s = np.std(vals)
        if s > 1e-10:
            m = np.mean(vals)
            skew_vals.append(np.mean(((vals - m) / s) ** 3))
            kurtosis_vals.append(np.mean(((vals - m) / s) ** 4) - 3)
    return {
        'mean_norm': avg_norm, 'total_var': total_var,
        'var_per_dim_mean': var_per_dim.mean(), 'var_per_dim_max': var_per_dim.max(),
        'var_per_dim_min': var_per_dim.min(),
        'avg_kurtosis': np.mean(kurtosis_vals) if kurtosis_vals else 0,
        'avg_skewness': np.mean(skew_vals) if skew_vals else 0,
    }


def run_ds7b():
    model_name = "deepseek7b"
    log(f"=== Starting {model_name} ===")
    try:
        model, tokenizer, device = load_model(model_name)
        info = get_model_info(model, model_name)
        log(f"Model: {info.model_class}, {info.n_layers} layers, d_model={info.d_model}")
        
        frac_layer = FRACTURE_LAYERS[model_name]
        layers_list = get_layers(model)
        d_model = info.d_model
        n_layers = info.n_layers
        
        # Sample layers
        sample_layers = set()
        for l in range(max(0, frac_layer - 3), min(n_layers, frac_layer + 4)):
            sample_layers.add(l)
        for l in range(0, n_layers, 5):
            sample_layers.add(l)
        sample_layers.add(n_layers - 1)
        sample_layers = sorted(sample_layers)
        log(f"  Sample layers: {sample_layers}")
        
        # Collect residual streams
        all_residuals = {l: defaultdict(list) for l in sample_layers}
        residual_cache = {}
        hook_handles = []
        
        def make_hook(layer_idx):
            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    residual_cache[layer_idx] = output[0].detach()
                else:
                    residual_cache[layer_idx] = output.detach()
            return hook_fn
        
        for l in sample_layers:
            h = layers_list[l].register_forward_hook(make_hook(l))
            hook_handles.append(h)
        
        cat_names = sorted(CATEGORIES_13.keys())
        for cat in cat_names:
            words = CATEGORIES_13[cat]
            for word in words:
                inputs = tokenizer(word, return_tensors="pt", padding=False, truncation=True)
                input_ids = inputs["input_ids"].to(device)
                with torch.no_grad():
                    residual_cache.clear()
                    try:
                        model(input_ids)
                    except:
                        continue
                for l in sample_layers:
                    if l in residual_cache:
                        rs = residual_cache[l][0, -1, :].cpu().float().numpy()
                        all_residuals[l][cat].append(rs)
        
        for h in hook_handles:
            h.remove()
        
        total_tokens = sum(len(v) for v in all_residuals.get(frac_layer, {}).values())
        log(f"  Total tokens at fracture layer: {total_tokens}")
        
        # Exp1: Global statistics
        log(f"\n{'='*60}")
        log(f"Exp1: Per-Layer Residual Stream Statistics ({model_name})")
        log(f"{'='*60}")
        
        layer_stats = {}
        log(f"{'Layer':>6} {'Norm':>8} {'TotalVar':>10} {'VarMax':>10} {'VarMean':>10} {'Kurtosis':>10} {'Skewness':>10} {'Marker':>20}")
        for l in sample_layers:
            all_vecs = []
            for cat in all_residuals[l]:
                all_vecs.extend(all_residuals[l][cat])
            if len(all_vecs) < 2:
                continue
            stats = compute_stats(all_vecs)
            layer_stats[l] = stats
            marker = ""
            if l == frac_layer: marker = "*** FRACTURE ***"
            elif l == frac_layer - 1: marker = "*** Pre-fracture ***"
            elif l == frac_layer + 1: marker = "*** Post-fracture ***"
            log(f"L{l:>4} {stats['mean_norm']:>8.3f} {stats['total_var']:>10.4f} {stats['var_per_dim_max']:>10.4f} {stats['var_per_dim_mean']:>10.5f} {stats['avg_kurtosis']:>10.3f} {stats['avg_skewness']:>10.3f} {marker}")
        
        # Statistic jumps
        log(f"\n--- Statistic jumps ---")
        sorted_layers = sorted(layer_stats.keys())
        for i in range(1, len(sorted_layers)):
            l_prev, l_curr = sorted_layers[i-1], sorted_layers[i]
            s_prev, s_curr = layer_stats[l_prev], layer_stats[l_curr]
            norm_ratio = s_curr['mean_norm'] / max(s_prev['mean_norm'], 1e-10)
            var_ratio = s_curr['total_var'] / max(s_prev['total_var'], 1e-10)
            marker = "*** FRACTURE ***" if l_curr == frac_layer else ""
            log(f"L{l_prev}->L{l_curr}: norm_ratio={norm_ratio:.3f}, var_ratio={var_ratio:.3f} {marker}")
        
        # Exp2: Geometric structure
        log(f"\n{'='*60}")
        log(f"Exp2: Geometric Structure ({model_name})")
        log(f"{'='*60}")
        
        key_layers = [l for l in [max(0, frac_layer-2), frac_layer-1, frac_layer, frac_layer+1, min(sample_layers[-1], frac_layer+2)] if l in all_residuals]
        for l in key_layers:
            marker = "FRACTURE" if l == frac_layer else f"L{l}"
            log(f"\n--- Layer {marker} ---")
            
            all_vecs = []
            for cat in cat_names:
                all_vecs.extend(all_residuals[l][cat])
            all_vecs = np.array(all_vecs)
            all_vecs_centered = all_vecs - all_vecs.mean(axis=0, keepdims=True)
            
            U, S, Vt = np.linalg.svd(all_vecs_centered, full_matrices=False)
            total_var_svd = (S ** 2).sum()
            cumvar = np.cumsum(S ** 2) / total_var_svd
            
            log(f"  PCA: Top-1={S[0]**2/total_var_svd*100:.1f}%, Top-3={cumvar[2]*100:.1f}%, Top-5={cumvar[4]*100:.1f}%, Top-10={cumvar[9]*100:.1f}%")
            log(f"  Eff dim(95%)={np.searchsorted(cumvar, 0.95)+1}, Eff dim(99%)={np.searchsorted(cumvar, 0.99)+1}")
            
            # PC0 category scores
            pc0 = Vt[0]
            pc0_scores = {}
            for cat in cat_names:
                vecs = np.array(all_residuals[l][cat])
                scores = vecs @ pc0
                pc0_scores[cat] = scores.mean()
            sorted_cats = sorted(pc0_scores.items(), key=lambda x: x[1], reverse=True)
            log(f"  PC0: {', '.join(f'{c}={s:+.3f}' for c, s in sorted_cats[:5])} ... {', '.join(f'{c}={s:+.3f}' for c, s in sorted_cats[-3:])}")
            
            # Inter-category distance
            cat_means = {cat: np.mean(all_residuals[l][cat], axis=0) for cat in cat_names}
            dists = [np.linalg.norm(cat_means[c1] - cat_means[c2]) for c1, c2 in combinations(cat_names, 2)]
            log(f"  Inter-cat dist: mean={np.mean(dists):.4f}, std={np.std(dists):.4f}")
        
        # Exp3: Per-dimension change
        log(f"\n{'='*60}")
        log(f"Exp3: Per-Dimension Change ({model_name})")
        log(f"{'='*60}")
        
        if frac_layer - 1 in all_residuals and frac_layer in all_residuals:
            pre_var = np.zeros(d_model)
            frac_var = np.zeros(d_model)
            for dim in range(d_model):
                pre_vals, frac_vals = [], []
                for cat in cat_names:
                    pre_vals.extend([v[dim] for v in all_residuals[frac_layer-1][cat]])
                    frac_vals.extend([v[dim] for v in all_residuals[frac_layer][cat]])
                pre_var[dim] = np.var(pre_vals)
                frac_var[dim] = np.var(frac_vals)
            
            var_ratio = frac_var / np.maximum(pre_var, 1e-10)
            top_increase = np.argsort(var_ratio)[-10:][::-1]
            
            log(f"  Top-10 dims with LARGEST variance increase:")
            for dim in top_increase:
                log(f"    Dim {dim:>5}: pre_var={pre_var[dim]:.6f}, frac_var={frac_var[dim]:.6f}, ratio={var_ratio[dim]:.2f}")
            
            # Per-superclass shift
            log(f"\n  Per-superclass dimension shift:")
            for sup in ["animate", "plant", "body", "artifact"]:
                sup_cats = [c for c in cat_names if SUPERCLASS_MAP[c] == sup]
                pre_vecs = [v for cat in sup_cats for v in all_residuals[frac_layer-1][cat]]
                frac_vecs = [v for cat in sup_cats for v in all_residuals[frac_layer][cat]]
                shift = np.mean(frac_vecs, axis=0) - np.mean(pre_vecs, axis=0)
                log(f"    {sup:>10}: shift_norm={np.linalg.norm(shift):.4f}")
        
        # Exp4: LayerNorm analysis
        log(f"\n{'='*60}")
        log(f"Exp4: LayerNorm Analysis ({model_name})")
        log(f"{'='*60}")
        
        key_layers_ln = sorted(set([max(0, frac_layer-2), frac_layer-1, frac_layer, frac_layer+1]))
        
        ln_cache = {}
        hook_handles = []
        
        def make_ln_hook(layer_idx, is_post_attn=False):
            def hook_fn(module, input, output):
                key = f"L{layer_idx}_{'post_attn' if is_post_attn else 'input'}"
                if isinstance(input, tuple) and len(input) > 0:
                    ln_cache[f"{key}_input"] = input[0].detach().clone()
                if isinstance(output, tuple):
                    ln_cache[f"{key}_output"] = output[0].detach().clone()
                else:
                    ln_cache[f"{key}_output"] = output.detach().clone()
            return hook_fn
        
        for l in key_layers_ln:
            layer = layers_list[l]
            for ln_name in ["input_layernorm", "ln_1", "layernorm"]:
                if hasattr(layer, ln_name):
                    ln = getattr(layer, ln_name)
                    h = ln.register_forward_hook(make_ln_hook(l, False))
                    hook_handles.append(h)
                    break
            for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
                if hasattr(layer, ln_name):
                    ln = getattr(layer, ln_name)
                    h = ln.register_forward_hook(make_ln_hook(l, True))
                    hook_handles.append(h)
                    break
        
        ln_data = defaultdict(lambda: defaultdict(list))
        for cat in cat_names:
            for word in CATEGORIES_13[cat]:
                inputs = tokenizer(word, return_tensors="pt", padding=False, truncation=True)
                input_ids = inputs["input_ids"].to(device)
                with torch.no_grad():
                    ln_cache.clear()
                    try:
                        model(input_ids)
                    except:
                        continue
                for key in list(ln_cache.keys()):
                    if key.endswith('_input') or key.endswith('_output'):
                        vec = ln_cache[key][0, -1, :].cpu().float().numpy()
                        ln_data[key][cat].append(vec)
        
        for h in hook_handles:
            h.remove()
        
        # Analyze LayerNorm
        for l in key_layers_ln:
            marker = "FRACTURE" if l == frac_layer else f"L{l}"
            for ln_type in ["input", "post_attn"]:
                input_key = f"L{l}_{ln_type}_input"
                output_key = f"L{l}_{ln_type}_output"
                if input_key not in ln_data or output_key not in ln_data:
                    continue
                
                log(f"\n  {marker} {ln_type}_layernorm:")
                
                for label, key in [("Before", input_key), ("After", output_key)]:
                    all_vecs = [v for cat in cat_names for v in ln_data[key][cat]]
                    norms = np.linalg.norm(all_vecs, axis=1)
                    log(f"    {label} LN: norm_mean={norms.mean():.4f}, norm_std={norms.std():.4f}")
                
                # LN cos change
                in_means = {cat: np.mean(ln_data[input_key][cat], axis=0) for cat in cat_names if cat in ln_data[input_key]}
                out_means = {cat: np.mean(ln_data[output_key][cat], axis=0) for cat in cat_names if cat in ln_data[output_key]}
                
                in_coses = [np.dot(in_means[c1], in_means[c2]) / (np.linalg.norm(in_means[c1]) * np.linalg.norm(in_means[c2]) + 1e-10) for c1, c2 in combinations(sorted(in_means.keys()), 2)]
                out_coses = [np.dot(out_means[c1], out_means[c2]) / (np.linalg.norm(out_means[c1]) * np.linalg.norm(out_means[c2]) + 1e-10) for c1, c2 in combinations(sorted(out_means.keys()), 2)]
                
                log(f"    avg_cos: before={np.mean(in_coses):.4f}, after={np.mean(out_coses):.4f}, change={np.mean(out_coses)-np.mean(in_coses):+.4f}")
        
        release_model(model)
        
        result_dir = Path(rf"d:\Ai2050\TransformerLens-Project\results\causal_fiber\{model_name}_cclxxxxv")
        result_dir.mkdir(parents=True, exist_ok=True)
        save_stats = {str(l): {k: float(v) for k, v in stats.items()} for l, stats in layer_stats.items()}
        with open(result_dir / "layer_stats.json", 'w') as f:
            json.dump(save_stats, f, indent=2)
        
        log(f"=== Finished {model_name} ===\n")
    except Exception as e:
        log(f"ERROR: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    run_ds7b()
