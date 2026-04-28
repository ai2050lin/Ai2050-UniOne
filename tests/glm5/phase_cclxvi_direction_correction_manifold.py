"""
Phase CCLXVI: 方向纠正流形(Directional Correction Manifold)验证
================================================
核心假说: 残差流空间中存在低维"语义流形"(约5维=PCA5),
  网络的逐层传播把任何偏移都投影回这个流形.

推理链:
  INV-34: 随机扰动被放大7~1000x  -> 扰动在"流形外"方向被放大
  INV-35: 语义差分方向在深层大幅旋转 -> 差分被"投影"到流形方向
  INV-31: 干预后自然回归target方向 -> 回归到流形上的正确位置
  INV-8:  PCA5探针0.98-1.00 -> 类别信息集中在~5维子空间

3个实验:
  Exp1: 流形禁闭增长(逐层PCA方差解释比)
        -> 如果top-k分量方差解释比随深度增加, 证明残差流逐层收敛到低维流形
  Exp2: 扰动分解(沿流形 vs 垂直流形)
        -> 如果沿流形分量被保留/放大, 垂直分量被抑制, 证明投影纠正
  Exp3: 有效维度变化(参与比维度估计)
        -> 如果有效维度随深度降低, 证明残差流被压缩到低维流形

用法:
  python phase_cclxvi_direction_correction_manifold.py --model qwen3 --exp 1
  python phase_cclxvi_direction_correction_manifold.py --model qwen3 --exp all
"""
import argparse, os, sys, json, time, gc
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
from pathlib import Path
from datetime import datetime
import numpy as np
import torch
from collections import defaultdict

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from tests.glm5.model_utils import (
    load_model, get_layers, get_model_info, release_model, get_W_U,
)

OUTPUT_DIR = Path("results/causal_fiber")

CONCEPTS = {
    "animals": ["dog", "cat", "horse", "eagle", "shark", "snake",
                "lion", "bear", "whale", "dolphin", "rabbit", "deer"],
    "food":    ["apple", "rice", "bread", "cheese", "salmon", "mango",
                "grape", "banana", "pasta", "pizza", "cookie", "steak"],
    "tools":   ["hammer", "knife", "saw", "drill", "wrench", "chisel",
                "pliers", "ruler", "level", "clamp", "file", "shovel"],
    "nature":  ["mountain", "river", "ocean", "forest", "desert", "volcano",
                "canyon", "glacier", "meadow", "island", "valley", "cliff"],
}

TEMPLATES = [
    "The {} is",
    "I saw a {} today",
    "This {} was",
]


def json_serialize(obj):
    if isinstance(obj, dict):
        return {str(k): json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [json_serialize(x) for x in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, bool):
        return obj
    elif obj is None:
        return None
    return obj


def proper_cos(v1, v2):
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-10 or n2 < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


def collect_residuals_for_categories(model, tokenizer, device, model_info, n_per_cat=8):
    """Collect residual streams for multiple concepts across categories.
    Returns: {layer: {category: [residual_vectors]}}
    """
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers_list = get_layers(model)
    
    cat_residuals = defaultdict(lambda: defaultdict(list))
    
    for cat, concepts in CONCEPTS.items():
        for concept in concepts[:n_per_cat]:
            for template in TEMPLATES:
                prompt = template.format(concept)
                input_ids = tokenizer(prompt, return_tensors="pt").to(device).input_ids
                last_pos = input_ids.shape[1] - 1
                
                captured = {}
                
                def make_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            captured[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                        else:
                            captured[key] = output[0, last_pos].detach().float().cpu().numpy()
                    return hook
                
                hooks = []
                # Embedding hook
                embed = model.get_input_embeddings()
                # We need L0 = embedding output. Use model.model.embed_tokens
                # Collect from each transformer layer
                for li in range(n_layers):
                    hooks.append(layers_list[li].register_forward_hook(make_hook(f"L{li}")))
                
                with torch.no_grad():
                    try:
                        _ = model(input_ids)
                    except Exception as e:
                        print(f"  Forward failed: {e}")
                
                for h in hooks:
                    h.remove()
                
                # L0 = embedding; L_i = output of layer i
                # Get embedding separately
                with torch.no_grad():
                    embed_out = model.get_input_embeddings()(input_ids)
                    cat_residuals[0][cat].append(embed_out[0, last_pos].detach().float().cpu().numpy())
                
                for li in range(n_layers):
                    if f"L{li}" in captured:
                        cat_residuals[li + 1][cat].append(captured[f"L{li}"])
    
    return dict(cat_residuals)


def compute_category_pca(cat_residuals, layer, n_components=20):
    """Compute PCA on category-labeled residuals at a given layer.
    Returns: (pca_directions, variance_ratios, mean_vector)
    """
    all_vecs = []
    for cat in cat_residuals.get(layer, {}):
        for v in cat_residuals[layer][cat]:
            all_vecs.append(v)
    
    if len(all_vecs) < n_components:
        return None, None, None
    
    X = np.array(all_vecs, dtype=np.float32)
    mean = X.mean(axis=0)
    X_centered = X - mean
    
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    pca_dirs = Vt[:n_components]
    var_ratios = (S ** 2) / (S ** 2).sum()
    
    return pca_dirs, var_ratios, mean


# ============================================================
# Exp1: 流形禁闭增长 (Category Manifold Confinement)
# ============================================================
def exp1_manifold_confinement(model_name):
    """Does top-k PCA variance explained increase with depth?
    
    If Direction Correction Manifold is correct:
      - Category-relevant variance (top-k PCA) should concentrate with depth
      - var_top5 and var_top10 should INCREASE with layer index
    """
    print(f"\n{'='*70}")
    print(f"Exp1: Category Manifold Confinement")
    print(f"  Model: {model_name}")
    print(f"  Key test: Does top-k PCA variance explained INCREASE with depth?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    
    print("\n  Collecting category residuals (4 cats x 8 concepts x 3 templates = 96 samples)...")
    cat_residuals = collect_residuals_for_categories(model, tokenizer, device, model_info, n_per_cat=8)
    
    print(f"  Collected layers: {sorted(cat_residuals.keys())}")
    
    # Compute PCA at each layer
    layer_data = []
    max_components = 20
    
    for l in sorted(cat_residuals.keys()):
        pca_dirs, var_ratios, mean_vec = compute_category_pca(cat_residuals, l, max_components)
        
        if pca_dirs is None:
            continue
        
        cum_var = np.cumsum(var_ratios)
        
        # Effective dimensionality (participation ratio)
        p = var_ratios / var_ratios.sum()
        participation_ratio = 1.0 / sum(p ** 2) if sum(p ** 2) > 0 else 0
        
        # Dimension to explain 90%, 95%
        dim_90 = int(np.searchsorted(cum_var, 0.90)) + 1
        dim_95 = int(np.searchsorted(cum_var, 0.95)) + 1
        
        layer_data.append({
            "layer": l,
            "var_top5": float(cum_var[4]) if len(cum_var) >= 5 else float(cum_var[-1]),
            "var_top10": float(cum_var[9]) if len(cum_var) >= 10 else float(cum_var[-1]),
            "var_top20": float(cum_var[19]) if len(cum_var) >= 20 else float(cum_var[-1]),
            "participation_ratio": float(participation_ratio),
            "dim_90": dim_90,
            "dim_95": dim_95,
            "singular_values": [float(s) for s in var_ratios[:20]],
        })
    
    # Print table
    print(f"\n  {'Layer':>6} | {'Top5':>8} | {'Top10':>8} | {'Top20':>8} | {'PR':>6} | {'dim90':>5} | {'dim95':>5}")
    print(f"  {'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*6}-+-{'-'*5}-+-{'-'*5}")
    for ld in layer_data:
        print(f"  {ld['layer']:>6} | {ld['var_top5']:>8.4f} | {ld['var_top10']:>8.4f} | {ld['var_top20']:>8.4f} | {ld['participation_ratio']:>6.2f} | {ld['dim_90']:>5} | {ld['dim_95']:>5}")
    
    # Compute slopes
    var5 = [ld["var_top5"] for ld in layer_data]
    var10 = [ld["var_top10"] for ld in layer_data]
    pr_vals = [ld["participation_ratio"] for ld in layer_data]
    dim90 = [ld["dim_90"] for ld in layer_data]
    
    x = np.arange(len(var5))
    slope5 = np.polyfit(x, var5, 1)[0]
    slope10 = np.polyfit(x, var10, 1)[0]
    slope_pr = np.polyfit(x, pr_vals, 1)[0]
    slope_dim90 = np.polyfit(x, dim90, 1)[0]
    
    print(f"\n  Slopes (vs layer depth):")
    print(f"    var_top5:  {slope5:+.6f}")
    print(f"    var_top10: {slope10:+.6f}")
    print(f"    PR:        {slope_pr:+.4f}")
    print(f"    dim_90:    {slope_dim90:+.4f}")
    
    if slope5 > 0 and slope10 > 0:
        conclusion = "SUPPORT"
        print(f"\n  >>> SUPPORT: Category manifold confinement INCREASES with depth")
        print(f"  >>> Top5 var slope={slope5:+.6f}, Top10 var slope={slope10:+.6f}")
    elif slope5 < 0 and slope10 < 0:
        conclusion = "AGAINST"
        print(f"\n  >>> AGAINST: Category manifold confinement DECREASES with depth")
    else:
        conclusion = "MIXED"
        print(f"\n  >>> MIXED: Conflicting trends")
    
    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxvi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp1_manifold_confinement.json"
    
    summary = {
        "experiment": "exp1_manifold_confinement",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Category manifold confinement increases with depth (var_topk increases)",
        "n_concepts_per_cat": 8,
        "n_templates": len(TEMPLATES),
        "total_samples": 8 * 4 * len(TEMPLATES),
        "layer_data": layer_data,
        "slopes": {
            "var_top5": float(slope5),
            "var_top10": float(slope10),
            "participation_ratio": float(slope_pr),
            "dim_90": float(slope_dim90),
        },
        "conclusion": conclusion,
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# Exp2: 扰动分解 (Perturbation Decomposition: Along vs Orthogonal to Manifold)
# ============================================================
def exp2_perturbation_decomposition(model_name):
    """Decompose perturbation amplification into manifold-parallel and orthogonal components.
    
    Method:
      1. Compute category PCA at L0 to define the "manifold"
      2. Generate perturbations: along-manifold (project onto top-5 PCA) and orthogonal
      3. Inject at L0, measure propagation at all layers
      4. Compare amplification of along vs orthogonal components
    
    Key prediction:
      If Direction Correction Manifold is correct:
        Along-manifold perturbation is preserved/amplified MORE than orthogonal
        => along_gain / orth_gain > 1 at deep layers
    """
    print(f"\n{'='*70}")
    print(f"Exp2: Perturbation Decomposition (Along vs Orthogonal to Manifold)")
    print(f"  Model: {model_name}")
    print(f"  Key test: Is along-manifold perturbation amplified more than orthogonal?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers_list = get_layers(model)
    
    # Step 1: Compute category PCA at L0
    print("\n  Step 1: Computing category PCA at L0...")
    cat_residuals = collect_residuals_for_categories(model, tokenizer, device, model_info, n_per_cat=6)
    
    pca0_dirs, pca0_vars, pca0_mean = compute_category_pca(cat_residuals, 0, n_components=20)
    
    if pca0_dirs is None:
        print("  ERROR: Could not compute PCA at L0")
        release_model(model)
        return None
    
    k_manifold = 5  # Use top-5 as the "category manifold"
    
    # Build projection matrices
    proj_along = np.zeros((d_model, d_model), dtype=np.float32)
    for i in range(k_manifold):
        proj_along += np.outer(pca0_dirs[i], pca0_dirs[i])
    proj_orth = np.eye(d_model, dtype=np.float32) - proj_along
    
    print(f"  Manifold dimension: {k_manifold}")
    print(f"  L0 top-5 variance explained: {sum(pca0_vars[:5]):.4f}")
    
    # Step 2: Generate perturbation directions
    print("\n  Step 2: Perturbation experiment...")
    
    rng = np.random.RandomState(42)
    eps = 0.05  # perturbation magnitude as fraction of h0 norm
    
    ref_texts = [
        "The dog is running",
        "The apple is sweet",
        "The hammer is heavy",
        "The mountain is tall",
    ]
    
    n_trials = 15  # per direction type per text
    
    # Collect results
    along_gains = defaultdict(list)  # layer -> [gain values]
    orth_gains = defaultdict(list)
    rand_gains = defaultdict(list)
    
    for text_idx, ref_text in enumerate(ref_texts):
        print(f"\n  [{text_idx+1}/{len(ref_texts)}] {ref_text}")
        
        input_ids = tokenizer(ref_text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1
        
        # Get baseline residual stream
        baseline = {}
        hooks_bl = []
        for li in range(n_layers):
            def make_bl_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        baseline[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                    else:
                        baseline[key] = output[0, last_pos].detach().float().cpu().numpy()
                return hook
            hooks_bl.append(layers_list[li].register_forward_hook(make_bl_hook(f"L{li}")))
        
        with torch.no_grad():
            _ = model(input_ids)
        
        for h in hooks_bl:
            h.remove()
        
        # Get L0 embedding
        with torch.no_grad():
            h0 = model.get_input_embeddings()(input_ids)[0, last_pos].detach().float().cpu().numpy()
        
        h0_norm = np.linalg.norm(h0)
        
        for trial in range(n_trials):
            # Generate base random direction
            rand_vec = rng.randn(d_model).astype(np.float32)
            rand_vec = rand_vec / np.linalg.norm(rand_vec)
            
            # Along-manifold direction
            d_along = proj_along @ rand_vec
            d_along_norm = np.linalg.norm(d_along)
            if d_along_norm < 1e-8:
                continue
            d_along = d_along / d_along_norm
            
            # Orthogonal direction
            d_orth = proj_orth @ rand_vec
            d_orth_norm = np.linalg.norm(d_orth)
            if d_orth_norm < 1e-8:
                continue
            d_orth = d_orth / d_orth_norm
            
            # Random direction (already normalized)
            d_rand = rand_vec
            
            # Test each direction type
            for d, dir_type in [(d_along, "along"), (d_orth, "orth"), (d_rand, "rand")]:
                delta_vec = eps * h0_norm * d
                model_dtype = next(model.parameters()).dtype
                delta_tensor = torch.tensor(delta_vec, dtype=model_dtype, device=device)
                
                # Inject perturbation at embedding layer and capture outputs
                perturbed = {}
                
                def make_inject_hook():
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            out = output[0].clone()
                            out[0, last_pos] = out[0, last_pos] + delta_tensor
                            return (out,) + output[1:]
                        else:
                            out = output.clone()
                            out[0, last_pos] = out[0, last_pos] + delta_tensor
                            return out
                    return hook
                
                def make_capture_hook(key):
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            perturbed[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                        else:
                            perturbed[key] = output[0, last_pos].detach().float().cpu().numpy()
                    return hook
                
                # Register hooks
                hooks = []
                # Inject at embedding (model.model.embed_tokens)
                embed_layer = model.get_input_embeddings()
                # We need to hook the embed_tokens output
                # For HuggingFace models, the embedding is inside model.model
                # Let's hook each transformer layer instead, and inject at L0 output
                
                # Actually, let's use the same approach as CCLXV: hook the first layer
                # to add perturbation, then capture all layers
                
                for li in range(n_layers):
                    if li == 0:
                        hooks.append(layers_list[li].register_forward_hook(make_inject_hook()))
                    else:
                        hooks.append(layers_list[li].register_forward_hook(make_capture_hook(f"L{li}")))
                
                # Also need to capture L0 output (with injection)
                # The inject hook on L0 modifies the output, we also need to capture it
                # Use a combined hook
                perturbed["L0"] = None
                
                def make_inject_and_capture_hook():
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            out = output[0].clone()
                            out[0, last_pos] = out[0, last_pos] + delta_tensor
                            perturbed["L0"] = out[0, last_pos].detach().float().cpu().numpy()
                            return (out,) + output[1:]
                        else:
                            out = output.clone()
                            out[0, last_pos] = out[0, last_pos] + delta_tensor
                            perturbed["L0"] = out[0, last_pos].detach().float().cpu().numpy()
                            return out
                    return hook
                
                # Remove old hooks and re-register
                for h in hooks:
                    h.remove()
                
                hooks = []
                hooks.append(layers_list[0].register_forward_hook(make_inject_and_capture_hook()))
                for li in range(1, n_layers):
                    hooks.append(layers_list[li].register_forward_hook(make_capture_hook(f"L{li}")))
                
                with torch.no_grad():
                    try:
                        _ = model(input_ids)
                    except Exception as e:
                        print(f"    Forward failed: {e}")
                
                for h in hooks:
                    h.remove()
                
                # Compute gains
                delta_l0 = perturbed.get("L0") - baseline.get("L0")
                delta_l0_norm = np.linalg.norm(delta_l0) if delta_l0 is not None else 0
                
                if delta_l0_norm < 1e-10:
                    continue
                
                for li in range(1, n_layers):
                    key = f"L{li}"
                    if key in perturbed and key in baseline:
                        delta_li = perturbed[key] - baseline[key]
                        delta_li_norm = np.linalg.norm(delta_li)
                        gain = delta_li_norm / delta_l0_norm
                        
                        if dir_type == "along":
                            along_gains[li].append(gain)
                        elif dir_type == "orth":
                            orth_gains[li].append(gain)
                        else:
                            rand_gains[li].append(gain)
            
            if (trial + 1) % 5 == 0:
                print(f"    Trial {trial+1}/{n_trials}")
    
    # Aggregate and print
    print(f"\n  Perturbation gain (||delta_L|| / ||delta_L0||) by direction type:")
    print(f"  {'Layer':>6} | {'Along':>10} | {'Orth':>10} | {'Random':>10} | {'Along/Orth':>12}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")
    
    layer_summary = []
    sample_layers = list(range(1, n_layers, max(1, n_layers // 12)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    for li in sorted(set(list(along_gains.keys()) + list(orth_gains.keys()))):
        along_avg = np.mean(along_gains[li]) if along_gains[li] else 0
        orth_avg = np.mean(orth_gains[li]) if orth_gains[li] else 0
        rand_avg = np.mean(rand_gains[li]) if rand_gains[li] else 0
        ao_ratio = along_avg / orth_avg if orth_avg > 1e-10 else float('inf')
        
        layer_summary.append({
            "layer": li,
            "along_gain": float(along_avg),
            "orth_gain": float(orth_avg),
            "rand_gain": float(rand_avg),
            "along_vs_orth": float(ao_ratio),
        })
        
        if li in sample_layers or li == n_layers - 1:
            print(f"  {li:>6} | {along_avg:>10.4f} | {orth_avg:>10.4f} | {rand_avg:>10.4f} | {ao_ratio:>12.4f}")
    
    # Check deep layer behavior
    deep_threshold = int(n_layers * 0.6)
    deep_data = [ls for ls in layer_summary if ls["layer"] >= deep_threshold]
    shallow_data = [ls for ls in layer_summary if ls["layer"] < deep_threshold]
    
    if deep_data and shallow_data:
        deep_ao = np.mean([ls["along_vs_orth"] for ls in deep_data])
        shallow_ao = np.mean([ls["along_vs_orth"] for ls in shallow_data])
        print(f"\n  Shallow layer along/orth ratio: {shallow_ao:.4f}")
        print(f"  Deep layer along/orth ratio: {deep_ao:.4f}")
        print(f"  Deep/Shallow ratio: {deep_ao/shallow_ao:.4f}" if shallow_ao > 0 else "")
        
        if deep_ao > shallow_ao:
            conclusion = "SUPPORT"
            print(f"\n  >>> SUPPORT: Along-manifold perturbation is selectively amplified in deep layers")
        elif deep_ao < shallow_ao:
            conclusion = "AGAINST"
            print(f"\n  >>> AGAINST: No selective amplification of manifold-parallel directions")
        else:
            conclusion = "NEUTRAL"
    else:
        deep_ao = 0
        shallow_ao = 0
        conclusion = "INSUFFICIENT_DATA"
    
    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxvi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp2_perturbation_decomposition.json"
    
    summary = {
        "experiment": "exp2_perturbation_decomposition",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Along-manifold perturbation amplified more than orthogonal in deep layers",
        "eps": eps,
        "k_manifold": k_manifold,
        "n_trials_per_text": n_trials,
        "n_ref_texts": len(ref_texts),
        "layer_summary": layer_summary,
        "deep_along_vs_orth": float(deep_ao) if deep_data else None,
        "shallow_along_vs_orth": float(shallow_ao) if shallow_data else None,
        "conclusion": conclusion,
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# Exp3: Per-Layer Perturbation Decomposition (at each source layer)
# ============================================================
def exp3_per_layer_decomposition(model_name):
    """Instead of injecting at L0, inject at EACH layer and decompose.
    
    This tests whether the "correction" happens at specific layers or uniformly.
    If Direction Correction Manifold is correct:
      - At shallow layers: along and orth gains should be similar (no selective correction yet)
      - At deep layers: along should be preserved more than orth (correction kicks in)
    
    Method: For each source layer l:
      1. Get clean residual h_l
      2. Compute local PCA at layer l from category residuals
      3. Add along-manifold and orthogonal perturbations
      4. Measure at layer l+1
    
    This is the most direct test of "direction correction at each layer".
    """
    print(f"\n{'='*70}")
    print(f"Exp3: Per-Layer Perturbation Decomposition")
    print(f"  Model: {model_name}")
    print(f"  Key test: Does selective amplification emerge at specific layers?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers_list = get_layers(model)
    
    # Collect category residuals for local PCA
    print("\n  Collecting category residuals for local PCA...")
    cat_residuals = collect_residuals_for_categories(model, tokenizer, device, model_info, n_per_cat=6)
    
    # Compute local PCA at each layer
    print("  Computing local PCA at each layer...")
    local_pca = {}
    for l in sorted(cat_residuals.keys()):
        pca_dirs, var_ratios, mean_vec = compute_category_pca(cat_residuals, l, n_components=10)
        if pca_dirs is not None:
            local_pca[l] = {"dirs": pca_dirs, "vars": var_ratios, "mean": mean_vec}
    
    # Per-layer perturbation experiment
    print("\n  Running per-layer perturbation decomposition...")
    
    rng = np.random.RandomState(42)
    eps = 0.05
    n_trials = 10
    k_manifold = 5
    
    ref_text = "The dog is running in the park"
    input_ids = tokenizer(ref_text, return_tensors="pt").to(device).input_ids
    last_pos = input_ids.shape[1] - 1
    
    # Get clean residual stream at all layers
    clean_rs = {}
    hooks_bl = []
    for li in range(n_layers):
        def make_bl_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    clean_rs[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                else:
                    clean_rs[key] = output[0, last_pos].detach().float().cpu().numpy()
            return hook
        hooks_bl.append(layers_list[li].register_forward_hook(make_bl_hook(f"L{li}")))
    
    with torch.no_grad():
        _ = model(input_ids)
    
    for h in hooks_bl:
        h.remove()
    
    layer_results = []
    
    # Sample layers (not every layer - too expensive)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    for src_layer in sample_layers:
        print(f"\n  Source layer L{src_layer}...")
        
        # Get local PCA for this layer
        if src_layer not in local_pca:
            print(f"    No PCA for layer {src_layer}, skipping")
            continue
        
        pca_dirs = local_pca[src_layer]["dirs"]
        
        # Build projection matrices
        proj_along = np.zeros((d_model, d_model), dtype=np.float32)
        for i in range(min(k_manifold, len(pca_dirs))):
            proj_along += np.outer(pca_dirs[i], pca_dirs[i])
        proj_orth = np.eye(d_model, dtype=np.float32) - proj_along
        
        # Get clean residual at this layer
        h_clean = clean_rs.get(f"L{src_layer}")
        if h_clean is None:
            continue
        
        h_clean_norm = np.linalg.norm(h_clean)
        
        along_ratios = []
        orth_ratios = []
        
        for trial in range(n_trials):
            # Generate directions
            rand_vec = rng.randn(d_model).astype(np.float32)
            rand_vec = rand_vec / np.linalg.norm(rand_vec)
            
            d_along = proj_along @ rand_vec
            d_along_norm = np.linalg.norm(d_along)
            if d_along_norm < 1e-8:
                continue
            d_along = d_along / d_along_norm
            
            d_orth = proj_orth @ rand_vec
            d_orth_norm = np.linalg.norm(d_orth)
            if d_orth_norm < 1e-8:
                continue
            d_orth = d_orth / d_orth_norm
            
            # For each direction, inject at src_layer and measure at src_layer+1
            for d, dir_type in [(d_along, "along"), (d_orth, "orth")]:
                delta_vec = eps * h_clean_norm * d
                model_dtype = next(model.parameters()).dtype
                delta_tensor = torch.tensor(delta_vec, dtype=model_dtype, device=device)
                
                perturbed_next = {}
                
                def make_inject_and_capture():
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            out = output[0].clone()
                            out[0, last_pos] = out[0, last_pos] + delta_tensor
                            perturbed_next["L_src"] = out[0, last_pos].detach().float().cpu().numpy()
                            return (out,) + output[1:]
                        else:
                            out = output.clone()
                            out[0, last_pos] = out[0, last_pos] + delta_tensor
                            perturbed_next["L_src"] = out[0, last_pos].detach().float().cpu().numpy()
                            return out
                    return hook
                
                tgt_layer = src_layer + 1
                if tgt_layer >= n_layers:
                    continue
                
                def make_capture_hook():
                    def hook(module, input, output):
                        if isinstance(output, tuple):
                            perturbed_next["L_tgt"] = output[0][0, last_pos].detach().float().cpu().numpy()
                        else:
                            perturbed_next["L_tgt"] = output[0, last_pos].detach().float().cpu().numpy()
                    return hook
                
                hooks = []
                hooks.append(layers_list[src_layer].register_forward_hook(make_inject_and_capture()))
                hooks.append(layers_list[tgt_layer].register_forward_hook(make_capture_hook()))
                
                with torch.no_grad():
                    try:
                        _ = model(input_ids)
                    except Exception as e:
                        print(f"    Forward failed: {e}")
                
                for h in hooks:
                    h.remove()
                
                # Compute single-step gain
                delta_src = perturbed_next.get("L_src")
                delta_tgt = perturbed_next.get("L_tgt")
                h_clean_tgt = clean_rs.get(f"L{tgt_layer}")
                
                if delta_src is not None and delta_tgt is not None and h_clean_tgt is not None:
                    # delta at src (injected)
                    delta_in = delta_src - h_clean
                    # delta at tgt (measured)
                    delta_out = delta_tgt - h_clean_tgt
                    
                    in_norm = np.linalg.norm(delta_in)
                    out_norm = np.linalg.norm(delta_out)
                    
                    if in_norm > 1e-10:
                        ratio = out_norm / in_norm
                        if dir_type == "along":
                            along_ratios.append(ratio)
                        else:
                            orth_ratios.append(ratio)
        
        avg_along = np.mean(along_ratios) if along_ratios else 0
        avg_orth = np.mean(orth_ratios) if orth_ratios else 0
        ao_ratio = avg_along / avg_orth if avg_orth > 1e-10 else float('inf')
        
        layer_results.append({
            "src_layer": src_layer,
            "along_single_step_ratio": float(avg_along),
            "orth_single_step_ratio": float(avg_orth),
            "along_vs_orth": float(ao_ratio),
            "n_trials_along": len(along_ratios),
            "n_trials_orth": len(orth_ratios),
        })
        
        print(f"    Along: {avg_along:.4f}, Orth: {avg_orth:.4f}, Along/Orth: {ao_ratio:.4f}")
    
    # Print summary
    print(f"\n  Per-layer decomposition summary:")
    print(f"  {'Layer':>6} | {'Along':>10} | {'Orth':>10} | {'A/O':>10}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for lr in layer_results:
        print(f"  {lr['src_layer']:>6} | {lr['along_single_step_ratio']:>10.4f} | {lr['orth_single_step_ratio']:>10.4f} | {lr['along_vs_orth']:>10.4f}")
    
    # Check trend
    shallow = [lr for lr in layer_results if lr["src_layer"] < n_layers * 0.4]
    deep = [lr for lr in layer_results if lr["src_layer"] >= n_layers * 0.6]
    
    if shallow and deep:
        shallow_ao = np.mean([lr["along_vs_orth"] for lr in shallow])
        deep_ao = np.mean([lr["along_vs_orth"] for lr in deep])
        print(f"\n  Shallow avg A/O: {shallow_ao:.4f}")
        print(f"  Deep avg A/O: {deep_ao:.4f}")
        
        if deep_ao > shallow_ao + 0.05:
            conclusion = "SUPPORT"
            print(f"  >>> SUPPORT: Selective amplification increases with depth")
        elif deep_ao < shallow_ao - 0.05:
            conclusion = "AGAINST"
            print(f"  >>> AGAINST: No increase in selective amplification")
        else:
            conclusion = "NEUTRAL"
            print(f"  >>> NEUTRAL: No clear trend")
    else:
        shallow_ao = 0
        deep_ao = 0
        conclusion = "INSUFFICIENT"
    
    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxvi"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp3_per_layer_decomposition.json"
    
    summary = {
        "experiment": "exp3_per_layer_decomposition",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Selective along-manifold amplification increases with depth",
        "eps": eps,
        "k_manifold": k_manifold,
        "n_trials": n_trials,
        "layer_results": layer_results,
        "shallow_ao": float(shallow_ao) if shallow else None,
        "deep_ao": float(deep_ao) if deep else None,
        "conclusion": conclusion,
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, required=True,
                        choices=["1", "2", "3", "all"])
    args = parser.parse_args()
    
    if args.exp in ["1", "all"]:
        exp1_manifold_confinement(args.model)
    
    if args.exp in ["2", "all"]:
        exp2_perturbation_decomposition(args.model)
    
    if args.exp in ["3", "all"]:
        exp3_per_layer_decomposition(args.model)
