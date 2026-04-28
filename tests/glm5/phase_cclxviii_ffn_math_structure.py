"""
Phase CCLXVIII: FFN纠正力的数学结构 (FFN Correction Force Mathematical Structure)
================================================
核心问题: FFN的"纠正力"的数学结构是什么?

背景:
  INV-41: FFN贡献70-80%的扰动放大
  INV-42: FFN旋转概念方向(cosF<0中层)
  INV-43: FFN驱动干预纠正(ffn_corr>0中深层)
  
  关键问题: FFN的纠正力是线性的吗? 是否可以用W_down矩阵的投影来解释?
  
验证:
  Exp1: FFN输出在W_U行空间中的投影比
        -> FFN是否在"词汇空间"中纠正? FFN输出多大比例指向vocab方向?
  Exp2: FFN线性近似的有效性
        -> FFN(h) ≈ W_down @ Diag(σ(W_gate @ h)) @ W_up @ h 的哪个分量贡献纠正?
        -> 用Jacobian近似FFN, 分解纠正力的来源
  Exp3: W_down矩阵与W_U行空间的对齐度
        -> W_down的行(输出方向)是否天然对齐vocab空间?

用法:
  python phase_cclxviii_ffn_math_structure.py --model qwen3 --exp 1
  python phase_cclxviii_ffn_math_structure.py --model qwen3 --exp all
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
    get_layer_weights, LayerWeights,
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


# ============================================================
# Exp1: FFN输出在W_U行空间中的投影比
# ============================================================
def exp1_ffn_wu_projection(model_name):
    """Measure how much FFN output lies in W_U row space.
    
    Key idea:
      - FFN(h) = W_down @ act(W_gate @ h) ⊙ (W_up @ h)
      - If FFN's "correction" is to push residual toward vocab space,
        then FFN output should increasingly project onto W_U row space
      - We compare: projection ratio of FFN output vs random baseline
    
    Method:
      1. Compute W_U SVD to get row space basis U_wu [d_model, k]
      2. Run concept sentences, capture FFN output at each layer
      3. Measure: ||proj_WU(ffn_out)||^2 / ||ffn_out||^2
      4. Compare with same ratio for residual stream and Attn output
    """
    print(f"\n{'='*70}")
    print(f"Exp1: FFN Output Projection onto W_U Row Space")
    print(f"  Model: {model_name}")
    print(f"  Key test: Does FFN output project strongly onto vocabulary space?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers_list = get_layers(model)
    
    # Compute W_U SVD basis
    W_U = get_W_U(model)  # [vocab, d_model]
    from scipy.sparse.linalg import svds
    k_svd = min(200, min(W_U.shape) - 2)
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_svd)
    U_wu = U_wu.astype(np.float64)  # [d_model, k]
    print(f"  W_U SVD: shape={W_U.shape}, k={k_svd}, top-5 singular values: {S_wu[-5:]}")
    
    def project_ratio(vec, U_basis):
        """Fraction of vec's energy in U_basis's column space"""
        vnorm = np.linalg.norm(vec)
        if vnorm < 1e-10:
            return 0.0, 0.0
        proj = U_basis @ (U_basis.T @ vec)
        proj_norm = np.linalg.norm(proj)
        return float(proj_norm ** 2 / vnorm ** 2), float(proj_norm)
    
    # Collect concept sentences
    all_sentences = []
    all_labels = []
    for cat, words in CONCEPTS.items():
        for word in words:
            for tmpl in TEMPLATES:
                all_sentences.append(tmpl.format(word))
                all_labels.append(cat)
    
    # Sample: 4 words per category x 2 templates = 8 per category, 32 total
    rng = np.random.RandomState(42)
    sample_indices = []
    for cat in CONCEPTS:
        cat_indices = [i for i, l in enumerate(all_labels) if l == cat]
        sample_indices.extend(rng.choice(cat_indices, min(8, len(cat_indices)), replace=False))
    
    print(f"  Total sentences: {len(sample_indices)}")
    
    # Aggregate results
    ffn_proj_ratios = defaultdict(list)    # layer -> [ratio]
    attn_proj_ratios = defaultdict(list)
    resid_proj_ratios = defaultdict(list)
    ffn_proj_norms = defaultdict(list)     # layer -> [proj_norm]
    ffn_norms = defaultdict(list)
    attn_norms = defaultdict(list)
    
    for sent_idx in sample_indices:
        sent = all_sentences[sent_idx]
        input_ids = tokenizer(sent, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1
        
        ffn_out = {}
        attn_out = {}
        h_in = {}
        h_out = {}
        
        hooks = []
        for li in range(n_layers):
            layer = layers_list[li]
            
            def make_pre(key):
                def hook(module, args):
                    h_in[key] = args[0][0, last_pos].detach().float().cpu().numpy() if isinstance(args, tuple) else args[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            def make_attn(key):
                def hook(module, input, output):
                    attn_out[key] = output[0][0, last_pos].detach().float().cpu().numpy() if isinstance(output, tuple) else output[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            def make_ffn(key):
                def hook(module, input, output):
                    ffn_out[key] = output[0][0, last_pos].detach().float().cpu().numpy() if isinstance(output, tuple) else output[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            def make_post(key):
                def hook(module, input, output):
                    h_out[key] = output[0][0, last_pos].detach().float().cpu().numpy() if isinstance(output, tuple) else output[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            hooks.append(layer.register_forward_pre_hook(make_pre(f"L{li}")))
            hooks.append(layer.self_attn.register_forward_hook(make_attn(f"L{li}")))
            if hasattr(layer, 'mlp'):
                hooks.append(layer.mlp.register_forward_hook(make_ffn(f"L{li}")))
            hooks.append(layer.register_forward_hook(make_post(f"L{li}")))
        
        with torch.no_grad():
            _ = model(input_ids)
        
        for h in hooks:
            h.remove()
        
        # Compute projections
        for li in range(n_layers):
            key = f"L{li}"
            
            if key in ffn_out:
                ffn_vec = ffn_out[key]
                r, pn = project_ratio(ffn_vec, U_wu)
                ffn_proj_ratios[li].append(r)
                ffn_proj_norms[li].append(pn)
                ffn_norms[li].append(np.linalg.norm(ffn_vec))
            
            if key in attn_out:
                attn_vec = attn_out[key]
                r, _ = project_ratio(attn_vec, U_wu)
                attn_proj_ratios[li].append(r)
                attn_norms[li].append(np.linalg.norm(attn_vec))
            
            if key in h_out:
                h_vec = h_out[key]
                r, _ = project_ratio(h_vec, U_wu)
                resid_proj_ratios[li].append(r)
    
    # Print results
    print(f"\n  FFN/Attn/Resid projection onto W_U row space:")
    print(f"  {'Layer':>6} | {'FFN_proj%':>9} | {'Attn_proj%':>10} | {'Resid_proj%':>11} | {'||FFN||':>8} | {'||Attn||':>8}")
    print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*10}-+-{'-'*11}-+-{'-'*8}-+-{'-'*8}")
    
    layer_summary = []
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    for li in range(n_layers):
        avg_fpr = np.mean(ffn_proj_ratios[li]) if ffn_proj_ratios[li] else 0
        avg_apr = np.mean(attn_proj_ratios[li]) if attn_proj_ratios[li] else 0
        avg_rpr = np.mean(resid_proj_ratios[li]) if resid_proj_ratios[li] else 0
        avg_fn = np.mean(ffn_norms[li]) if ffn_norms[li] else 0
        avg_an = np.mean(attn_norms[li]) if attn_norms[li] else 0
        
        layer_summary.append({
            "layer": li,
            "ffn_proj_ratio": float(avg_fpr),
            "attn_proj_ratio": float(avg_apr),
            "resid_proj_ratio": float(avg_rpr),
            "ffn_norm": float(avg_fn),
            "attn_norm": float(avg_an),
        })
        
        if li in sample_layers or li == n_layers - 1:
            print(f"  {li:>6} | {avg_fpr:>9.4f} | {avg_apr:>10.4f} | {avg_rpr:>11.4f} | {avg_fn:>8.2f} | {avg_an:>8.2f}")
    
    # Analysis: does FFN output project more onto W_U than Attn?
    shallow_fpr = np.mean([ls["ffn_proj_ratio"] for ls in layer_summary if ls["layer"] < n_layers * 0.3])
    mid_fpr = np.mean([ls["ffn_proj_ratio"] for ls in layer_summary if n_layers * 0.3 <= ls["layer"] < n_layers * 0.7])
    deep_fpr = np.mean([ls["ffn_proj_ratio"] for ls in layer_summary if ls["layer"] >= n_layers * 0.7])
    
    shallow_apr = np.mean([ls["attn_proj_ratio"] for ls in layer_summary if ls["layer"] < n_layers * 0.3])
    mid_apr = np.mean([ls["attn_proj_ratio"] for ls in layer_summary if n_layers * 0.3 <= ls["layer"] < n_layers * 0.7])
    deep_apr = np.mean([ls["attn_proj_ratio"] for ls in layer_summary if ls["layer"] >= n_layers * 0.7])
    
    print(f"\n  FFN projection ratio by depth:")
    print(f"    Shallow: {shallow_fpr:.4f}")
    print(f"    Mid:     {mid_fpr:.4f}")
    print(f"    Deep:    {deep_fpr:.4f}")
    print(f"  Attn projection ratio by depth:")
    print(f"    Shallow: {shallow_apr:.4f}")
    print(f"    Mid:     {mid_apr:.4f}")
    print(f"    Deep:    {deep_apr:.4f}")
    
    if deep_fpr > shallow_fpr + 0.05:
        print(f"  >>> FFN output increasingly aligns with vocab space in deep layers")
    elif deep_fpr < shallow_fpr - 0.05:
        print(f"  >>> FFN output moves AWAY from vocab space in deep layers")
    else:
        print(f"  >>> FFN output vocab alignment roughly constant across layers")
    
    if mid_fpr > mid_apr + 0.05:
        print(f"  >>> FFN projects more onto vocab space than Attn (mid layers)")
    
    # Random baseline: random vectors in R^d_model
    rng_test = np.random.RandomState(123)
    random_ratios = []
    for _ in range(1000):
        rv = rng_test.randn(d_model).astype(np.float64)
        rv = rv / np.linalg.norm(rv)
        r, _ = project_ratio(rv, U_wu)
        random_ratios.append(r)
    random_baseline = np.mean(random_ratios)
    print(f"  Random baseline projection ratio: {random_baseline:.4f}")
    print(f"  FFN / random ratio (mid): {mid_fpr / max(random_baseline, 1e-6):.2f}x")
    
    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxviii"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp1_ffn_wu_projection.json"
    
    summary = {
        "experiment": "exp1_ffn_wu_projection",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "FFN output increasingly projects onto W_U row space",
        "k_svd": k_svd,
        "random_baseline": float(random_baseline),
        "layer_summary": layer_summary,
        "shallow_ffn_proj": float(shallow_fpr),
        "mid_ffn_proj": float(mid_fpr),
        "deep_ffn_proj": float(deep_fpr),
        "shallow_attn_proj": float(shallow_apr),
        "mid_attn_proj": float(mid_apr),
        "deep_attn_proj": float(deep_apr),
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# Exp2: FFN线性近似与Jacobian分解
# ============================================================
def exp2_ffn_jacobian_decomposition(model_name):
    """Decompose FFN's 'correction force' using Jacobian analysis.
    
    Key idea:
      FFN(h) = W_down @ (σ(W_gate @ h̃) ⊙ (W_up @ h̃))   where h̃ = LN(h)
      
      The Jacobian of FFN at point h is:
      J_FFN(h) = W_down @ Diag(σ'(W_gate @ h̃) ⊙ (W_up @ h̃) + σ(W_gate @ h̃)) @ [W_up @ J_LN(h)]
                 ... (simplified, but too complex analytically)
      
      Instead, we use a numerical approach:
      1. Compute FFN's Jacobian-vector product: J_FFN(h) @ v ≈ (FFN(h + εv) - FFN(h)) / ε
      2. Compare this with the "linear approximation" FFN_linear = W_down @ W_up @ h̃
      3. Measure: how much of FFN's response is linear vs nonlinear?
      4. For concept differences: does the linear part explain the correction?
    
    More practically:
      - FFN(h) = f_out, the actual output
      - FFN_linear(h) = W_down @ W_up @ LN(h) (skip the activation)
      - Residual = FFN(h) - FFN_linear(h) (the nonlinear part)
      - Which part aligns with the target direction?
    """
    print(f"\n{'='*70}")
    print(f"Exp2: FFN Jacobian Decomposition (Linear vs Nonlinear)")
    print(f"  Model: {model_name}")
    print(f"  Key test: Is FFN's correction force linear or nonlinear?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)
    
    # Get W_U for target direction
    W_U = get_W_U(model)
    from scipy.sparse.linalg import svds
    k_svd = min(200, min(W_U.shape) - 2)
    U_wu, _, _ = svds(W_U.T.astype(np.float32), k=k_svd)
    U_wu = U_wu.astype(np.float64)
    
    template = "The {} is"
    
    # Concept pairs
    pairs = [
        ("dog", "cat"),     # within animals
        ("dog", "apple"),   # across categories
        ("hammer", "knife"), # within tools
        ("mountain", "river"), # within nature
    ]
    
    # Results
    pair_results = []
    
    for wordA, wordB in pairs:
        print(f"\n  Pair: {wordA} -> {wordB}")
        
        textA = template.format(wordA)
        textB = template.format(wordB)
        
        # Get target direction: B - A in residual space at final layer
        # We need to run both and get the full residual stream
        
        def run_full(text):
            input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
            last_pos = input_ids.shape[1] - 1
            
            h_in = {}
            ffn_out = {}
            ln_out = {}  # post-attention layernorm output (FFN input)
            
            hooks = []
            for li in range(n_layers):
                layer = layers_list[li]
                
                def make_pre(key):
                    def hook(module, args):
                        h_in[key] = args[0][0, last_pos].detach().float().cpu().numpy() if isinstance(args, tuple) else args[0, last_pos].detach().float().cpu().numpy()
                    return hook
                
                def make_ffn(key):
                    def hook(module, input, output):
                        ffn_out[key] = output[0][0, last_pos].detach().float().cpu().numpy() if isinstance(output, tuple) else output[0, last_pos].detach().float().cpu().numpy()
                    return hook
                
                # Capture FFN input (post-attention layernorm output)
                def make_ffn_pre(key):
                    def hook(module, args):
                        if isinstance(args, tuple):
                            ln_out[key] = args[0][0, last_pos].detach().float().cpu().numpy()
                        else:
                            ln_out[key] = args[0, last_pos].detach().float().cpu().numpy()
                    return hook
                
                hooks.append(layer.register_forward_pre_hook(make_pre(f"L{li}")))
                # Hook on MLP to capture its input (post-LN)
                if hasattr(layer, 'mlp'):
                    hooks.append(layer.mlp.register_forward_pre_hook(make_ffn_pre(f"L{li}")))
                    hooks.append(layer.mlp.register_forward_hook(make_ffn(f"L{li}")))
            
            with torch.no_grad():
                _ = model(input_ids)
            
            for h in hooks:
                h.remove()
            
            return h_in, ffn_out, ln_out
        
        hA, fA, lnA = run_full(textA)
        hB, fB, lnB = run_full(textB)
        
        # Target direction: residual difference B - A
        # We want to know: at each layer, does FFN output point toward B or A?
        
        layer_data = []
        for li in range(n_layers):
            key = f"L{li}"
            if key not in fA or key not in fB:
                continue
            if key not in lnA or key not in lnB:
                continue
            
            ffn_A = fA[key]
            ffn_B = fB[key]
            ln_A = lnA[key]  # FFN input for A
            ln_B = lnB[key]  # FFN input for B
            
            # Concept difference in FFN output
            delta_ffn = ffn_B - ffn_A  # FFN's contribution to concept signal
            
            # Concept difference in FFN input (post-LN residual)
            delta_ln = ln_B - ln_A
            
            # Linear approximation: W_down @ W_up @ delta_ln
            lw = get_layer_weights(layers_list[li], d_model, mlp_type)
            W_up = lw.W_up    # [inter, d_model]
            W_down = lw.W_down  # [d_model, inter]
            W_gate = lw.W_gate  # [inter, d_model]
            
            if W_up is None or W_down is None or W_gate is None:
                continue
            
            # Linear part: W_down @ W_up @ delta_ln (no activation)
            linear_part = W_down @ (W_up @ delta_ln)
            
            # Nonlinear part: actual FFN output difference minus linear
            nonlinear_part = delta_ffn - linear_part
            
            # Metrics
            delta_ffn_norm = np.linalg.norm(delta_ffn)
            linear_norm = np.linalg.norm(linear_part)
            nonlinear_norm = np.linalg.norm(nonlinear_part)
            
            # How much of the linear/nonlinear part aligns with delta_ffn?
            if delta_ffn_norm > 1e-10:
                cos_linear = proper_cos(linear_part, delta_ffn)
                cos_nonlinear = proper_cos(nonlinear_part, delta_ffn)
            else:
                cos_linear = 0
                cos_nonlinear = 0
            
            # How much of each part projects onto W_U?
            def proj_ratio(vec):
                vn = np.linalg.norm(vec)
                if vn < 1e-10:
                    return 0.0
                proj = U_wu @ (U_wu.T @ vec)
                return float(np.linalg.norm(proj) ** 2 / vn ** 2)
            
            linear_wu = proj_ratio(linear_part)
            nonlinear_wu = proj_ratio(nonlinear_part)
            total_wu = proj_ratio(delta_ffn)
            
            layer_data.append({
                "layer": li,
                "delta_ffn_norm": float(delta_ffn_norm),
                "linear_norm": float(linear_norm),
                "nonlinear_norm": float(nonlinear_norm),
                "linear_fraction": float(linear_norm / max(delta_ffn_norm, 1e-10)),
                "cos_linear_total": float(cos_linear),
                "cos_nonlinear_total": float(cos_nonlinear),
                "linear_wu_proj": float(linear_wu),
                "nonlinear_wu_proj": float(nonlinear_wu),
                "total_wu_proj": float(total_wu),
            })
        
        pair_results.append({
            "pair": f"{wordA}->{wordB}",
            "layer_data": layer_data,
        })
    
    # Print summary
    print(f"\n  FFN Concept Difference: Linear vs Nonlinear Decomposition:")
    for pr in pair_results:
        print(f"\n  Pair: {pr['pair']}")
        print(f"  {'Layer':>6} | {'||dFFN||':>9} | {'Lin%':>6} | {'cos_lin':>8} | {'cos_nl':>8} | {'lin_WU%':>8} | {'nl_WU%':>8}")
        print(f"  {'-'*6}-+-{'-'*9}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
        
        sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)
        
        for ld in pr['layer_data']:
            if ld['layer'] in sample_layers or ld['layer'] == n_layers - 1:
                print(f"  {ld['layer']:>6} | {ld['delta_ffn_norm']:>9.3f} | {ld['linear_fraction']:>6.3f} | {ld['cos_linear_total']:>+8.4f} | {ld['cos_nonlinear_total']:>+8.4f} | {ld['linear_wu_proj']:>8.4f} | {ld['nonlinear_wu_proj']:>8.4f}")
    
    # Aggregate across pairs
    all_linear_frac = defaultdict(list)
    all_cos_linear = defaultdict(list)
    all_cos_nonlinear = defaultdict(list)
    all_lin_wu = defaultdict(list)
    all_nl_wu = defaultdict(list)
    
    for pr in pair_results:
        for ld in pr['layer_data']:
            li = ld['layer']
            all_linear_frac[li].append(ld['linear_fraction'])
            all_cos_linear[li].append(ld['cos_linear_total'])
            all_cos_nonlinear[li].append(ld['cos_nonlinear_total'])
            all_lin_wu[li].append(ld['linear_wu_proj'])
            all_nl_wu[li].append(ld['nonlinear_wu_proj'])
    
    print(f"\n  Aggregate (mean across pairs):")
    print(f"  {'Layer':>6} | {'Lin%':>6} | {'cos_lin':>8} | {'cos_nl':>8} | {'lin_WU%':>8} | {'nl_WU%':>8}")
    print(f"  {'-'*6}-+-{'-'*6}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*8}")
    
    agg_summary = []
    for li in range(n_layers):
        if li not in all_linear_frac:
            continue
        avg_lf = np.mean(all_linear_frac[li])
        avg_cl = np.mean(all_cos_linear[li])
        avg_cnl = np.mean(all_cos_nonlinear[li])
        avg_lwu = np.mean(all_lin_wu[li])
        avg_nlwu = np.mean(all_nl_wu[li])
        
        agg_summary.append({
            "layer": li,
            "linear_fraction": float(avg_lf),
            "cos_linear": float(avg_cl),
            "cos_nonlinear": float(avg_cnl),
            "linear_wu_proj": float(avg_lwu),
            "nonlinear_wu_proj": float(avg_nlwu),
        })
        
        if li in sample_layers or li == n_layers - 1:
            print(f"  {li:>6} | {avg_lf:>6.3f} | {avg_cl:>+8.4f} | {avg_cnl:>+8.4f} | {avg_lwu:>8.4f} | {avg_nlwu:>8.4f}")
    
    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxviii"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp2_ffn_jacobian.json"
    
    summary = {
        "experiment": "exp2_ffn_jacobian_decomposition",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "FFN's correction is primarily nonlinear",
        "pair_results": pair_results,
        "aggregate_summary": agg_summary,
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# Exp3: W_down矩阵与W_U行空间的对齐度
# ============================================================
def exp3_wdown_wu_alignment(model_name):
    """Measure how much W_down's output space aligns with W_U row space.
    
    Key idea:
      FFN(h) = W_down @ act(intermediate)
      W_down maps from intermediate space to d_model
      If W_down's column space (output space) aligns with W_U row space,
      then FFN naturally produces outputs in vocabulary space
    
    Method:
      1. For each layer, compute W_down's column space (via SVD)
      2. Compute overlap with W_U row space
      3. Compare with random matrix baseline
    
    Measures:
      - Subspace overlap: ||P_WU @ P_Wdown||_F / sqrt(k)
      - Average projection: mean of W_down columns projected onto W_U
    """
    print(f"\n{'='*70}")
    print(f"Exp3: W_down Column Space Alignment with W_U Row Space")
    print(f"  Model: {model_name}")
    print(f"  Key test: Is W_down's output space naturally aligned with vocab space?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type
    layers_list = get_layers(model)
    
    # W_U SVD basis
    W_U = get_W_U(model)
    from scipy.sparse.linalg import svds
    k_wu = min(200, min(W_U.shape) - 2)
    U_wu, S_wu, _ = svds(W_U.T.astype(np.float32), k=k_wu)
    U_wu = U_wu.astype(np.float64)  # [d_model, k_wu]
    
    # Projector onto W_U row space
    P_wu = U_wu @ U_wu.T  # [d_model, d_model]
    
    layer_summary = []
    
    print(f"\n  {'Layer':>6} | {'Wdown_proj%':>11} | {'Wup_proj%':>10} | {'Wgate_proj%':>11} | {'Random%':>8} | {'Wd/Wu_corr':>11}")
    print(f"  {'-'*6}-+-{'-'*11}-+-{'-'*10}-+-{'-'*11}-+-{'-'*8}-+-{'-'*11}")
    
    for li in range(n_layers):
        lw = get_layer_weights(layers_list[li], d_model, mlp_type)
        W_down = lw.W_down   # [d_model, intermediate]
        W_up = lw.W_up       # [intermediate, d_model] (if split)
        W_gate = lw.W_gate   # [intermediate, d_model] (if split)
        
        if W_down is None:
            continue
        
        # W_down column space: each column is a d_model vector
        # Measure: average fraction of each column's energy in W_U row space
        n_cols = min(W_down.shape[1], 500)  # sample columns to save time
        rng_sample = np.random.RandomState(42)
        col_indices = rng_sample.choice(W_down.shape[1], n_cols, replace=False)
        
        wdown_proj_ratios = []
        for ci in col_indices:
            col = W_down[:, ci].astype(np.float64)
            col_norm = np.linalg.norm(col)
            if col_norm < 1e-10:
                continue
            proj = P_wu @ col
            wdown_proj_ratios.append(np.linalg.norm(proj) ** 2 / col_norm ** 2)
        
        avg_wdown_proj = np.mean(wdown_proj_ratios) if wdown_proj_ratios else 0
        
        # Same for W_up rows (each row is a d_model vector)
        wup_proj_ratios = []
        if W_up is not None:
            n_rows = min(W_up.shape[0], 500)
            row_indices = rng_sample.choice(W_up.shape[0], n_rows, replace=False)
            for ri in row_indices:
                row = W_up[ri, :].astype(np.float64)
                row_norm = np.linalg.norm(row)
                if row_norm < 1e-10:
                    continue
                proj = P_wu @ row
                wup_proj_ratios.append(np.linalg.norm(proj) ** 2 / row_norm ** 2)
        
        avg_wup_proj = np.mean(wup_proj_ratios) if wup_proj_ratios else 0
        
        # Same for W_gate rows
        wgate_proj_ratios = []
        if W_gate is not None:
            n_rows = min(W_gate.shape[0], 500)
            row_indices = rng_sample.choice(W_gate.shape[0], n_rows, replace=False)
            for ri in row_indices:
                row = W_gate[ri, :].astype(np.float64)
                row_norm = np.linalg.norm(row)
                if row_norm < 1e-10:
                    continue
                proj = P_wu @ row
                wgate_proj_ratios.append(np.linalg.norm(proj) ** 2 / row_norm ** 2)
        
        avg_wgate_proj = np.mean(wgate_proj_ratios) if wgate_proj_ratios else 0
        
        # Random baseline: random matrix [d_model, intermediate]
        random_proj_ratios = []
        for _ in range(100):
            rv = rng_sample.randn(d_model).astype(np.float64)
            rv = rv / np.linalg.norm(rv)
            proj = P_wu @ rv
            random_proj_ratios.append(np.linalg.norm(proj) ** 2)
        avg_random_proj = np.mean(random_proj_ratios)
        
        # Correlation between W_down and W_up
        # W_down @ W_up should be related to the "effective" FFN linear map
        if W_up is not None:
            # Subsample for speed
            n_sample = min(200, W_down.shape[1])
            Wd_sub = W_down[:, :n_sample].astype(np.float64)
            Wu_sub = W_up[:n_sample, :].astype(np.float64)
            effective_linear = Wd_sub @ Wu_sub  # [d_model, d_model]
            # This is the linear part of FFN
            # How much of effective_linear's row space aligns with W_U?
            # Use Frobenius norm of projection
            eff_proj = np.linalg.norm(P_wu @ effective_linear, 'fro') ** 2
            eff_total = np.linalg.norm(effective_linear, 'fro') ** 2
            wd_wu_corr = eff_proj / max(eff_total, 1e-10)
        else:
            wd_wu_corr = 0
        
        layer_summary.append({
            "layer": li,
            "wdown_proj_ratio": float(avg_wdown_proj),
            "wup_proj_ratio": float(avg_wup_proj),
            "wgate_proj_ratio": float(avg_wgate_proj),
            "random_baseline": float(avg_random_proj),
            "wdown_vs_random": float(avg_wdown_proj / max(avg_random_proj, 1e-10)),
            "effective_linear_wu_proj": float(wd_wu_corr),
        })
        
        sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
        if n_layers - 1 not in sample_layers:
            sample_layers.append(n_layers - 1)
        
        if li in sample_layers or li == n_layers - 1:
            print(f"  {li:>6} | {avg_wdown_proj:>11.4f} | {avg_wup_proj:>10.4f} | {avg_wgate_proj:>11.4f} | {avg_random_proj:>8.4f} | {wd_wu_corr:>11.4f}")
    
    # Analysis
    shallow_wd = np.mean([ls["wdown_proj_ratio"] for ls in layer_summary if ls["layer"] < n_layers * 0.3])
    mid_wd = np.mean([ls["wdown_proj_ratio"] for ls in layer_summary if n_layers * 0.3 <= ls["layer"] < n_layers * 0.7])
    deep_wd = np.mean([ls["wdown_proj_ratio"] for ls in layer_summary if ls["layer"] >= n_layers * 0.7])
    random_bl = np.mean([ls["random_baseline"] for ls in layer_summary])
    
    print(f"\n  W_down projection onto W_U by depth:")
    print(f"    Shallow: {shallow_wd:.4f} ({shallow_wd/random_bl:.1f}x random)")
    print(f"    Mid:     {mid_wd:.4f} ({mid_wd/random_bl:.1f}x random)")
    print(f"    Deep:    {deep_wd:.4f} ({deep_wd/random_bl:.1f}x random)")
    print(f"    Random baseline: {random_bl:.4f}")
    
    # Effective linear map alignment
    shallow_eff = np.mean([ls["effective_linear_wu_proj"] for ls in layer_summary if ls["layer"] < n_layers * 0.3])
    mid_eff = np.mean([ls["effective_linear_wu_proj"] for ls in layer_summary if n_layers * 0.3 <= ls["layer"] < n_layers * 0.7])
    deep_eff = np.mean([ls["effective_linear_wu_proj"] for ls in layer_summary if ls["layer"] >= n_layers * 0.7])
    
    print(f"\n  Effective linear (W_down @ W_up) alignment with W_U:")
    print(f"    Shallow: {shallow_eff:.4f}")
    print(f"    Mid:     {mid_eff:.4f}")
    print(f"    Deep:    {deep_eff:.4f}")
    
    if deep_wd > random_bl * 2:
        print(f"  >>> W_down columns are significantly aligned with W_U row space")
    else:
        print(f"  >>> W_down columns are NOT significantly aligned with W_U row space")
    
    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxviii"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp3_wdown_wu_alignment.json"
    
    summary = {
        "experiment": "exp3_wdown_wu_alignment",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "W_down output space aligns with W_U row space",
        "k_wu_svd": k_wu,
        "random_baseline": float(random_bl),
        "layer_summary": layer_summary,
        "shallow_wdown_proj": float(shallow_wd),
        "mid_wdown_proj": float(mid_wd),
        "deep_wdown_proj": float(deep_wd),
        "shallow_eff_proj": float(shallow_eff),
        "mid_eff_proj": float(mid_eff),
        "deep_eff_proj": float(deep_eff),
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
        exp1_ffn_wu_projection(args.model)
    
    if args.exp in ["2", "all"]:
        exp2_ffn_jacobian_decomposition(args.model)
    
    if args.exp in ["3", "all"]:
        exp3_wdown_wu_alignment(args.model)
