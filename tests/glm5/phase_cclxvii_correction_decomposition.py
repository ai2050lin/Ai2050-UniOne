"""
Phase CCLXVII: 纠正机制分解 (Correction Mechanism Decomposition)
================================================
核心问题: INV-31(干预后自然回归target)的机制是什么?

背景:
  REF-13: 不是吸引子(扰动被放大7~1000x)
  REF-14: 不是流形投影(各向同性放大, A/O=1.01)
  那么"纠正"从何而来?

关键假说: 残差连接结构中 Attn 和 FFN 扮演不同角色
  h_{l+1} = h_l + Attn_l(h_l) + FFN_l(h_l')
  - Attn: 信息检索, 倾向于纠正(输出指向正确方向)
  - FFN: 特征变换, 倾向于放大(各向同性放大任何输入)

验证:
  Exp1: 随机扰动传播的Attn/FFN分解
        -> Attn和FFN各贡献多少扰动放大?
  Exp2: 概念差异传播的Attn/FFN分解
        -> Attn和FFN各贡献多少概念信号?
  Exp3: 中层干预的Attn/FFN纠正分解
        -> 干预后, Attn和FFN哪个驱动"回归target"?

用法:
  python phase_cclxvii_correction_decomposition.py --model qwen3 --exp 1
  python phase_cclxvii_correction_decomposition.py --model qwen3 --exp all
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
# Exp1: 随机扰动传播的Attn/FFN分解
# ============================================================
def exp1_random_perturbation_decomposition(model_name):
    """Decompose random perturbation amplification into Attn and FFN contributions.
    
    Method:
      1. Run clean forward, capture: h_l (residual), a_l (attn out), f_l (ffn out)
      2. Run perturbed forward (add random noise at embedding), capture same
      3. At each layer:
         - delta_h = h_pert - h_clean (total perturbation at output)
         - delta_a = a_pert - a_clean (attn's response to perturbation)
         - delta_f = f_pert - f_clean (ffn's response to perturbation)
         - Verify: delta_h_{l+1} ~ delta_h_l + delta_a_l + delta_f_l
    
    Key measures:
      - ||delta_a|| vs ||delta_f|| (which amplifies more?)
      - cos(delta_a, delta_h_l) (does Attn preserve perturbation direction?)
      - cos(delta_f, delta_h_l) (does FFN preserve perturbation direction?)
    """
    print(f"\n{'='*70}")
    print(f"Exp1: Random Perturbation Attn/FFN Decomposition")
    print(f"  Model: {model_name}")
    print(f"  Key test: Does Attn or FFN contribute more to perturbation amplification?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers_list = get_layers(model)
    
    rng = np.random.RandomState(42)
    eps = 0.05  # perturbation magnitude as fraction of h0 norm
    
    ref_texts = [
        "The dog is running",
        "The apple is sweet",
        "The hammer is heavy",
        "The mountain is tall",
    ]
    
    n_trials = 12
    
    # Aggregate results across trials
    attn_norms = defaultdict(list)  # layer -> [||delta_a||]
    ffn_norms = defaultdict(list)   # layer -> [||delta_f||]
    total_norms = defaultdict(list) # layer -> [||delta_h||]
    attn_cos_with_input = defaultdict(list)  # cos(delta_a, delta_h_l)
    ffn_cos_with_input = defaultdict(list)   # cos(delta_f, delta_h_l')
    attn_frac = defaultdict(list)  # ||delta_a|| / (||delta_a|| + ||delta_f||)
    
    for text_idx, ref_text in enumerate(ref_texts):
        print(f"\n  [{text_idx+1}/{len(ref_texts)}] {ref_text}")
        
        input_ids = tokenizer(ref_text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1
        
        # === Run clean forward, capture all components ===
        clean_h = {}   # residual at input of each layer
        clean_a = {}   # attn output at each layer
        clean_f = {}   # ffn output at each layer
        clean_h_post = {}  # residual at output of each layer
        
        # Hook to capture residual input (pre-hook on each layer)
        def make_pre_hook(key):
            def hook(module, args):
                if isinstance(args, tuple):
                    h = args[0]
                else:
                    h = args
                clean_h[key] = h[0, last_pos].detach().float().cpu().numpy()
            return hook
        
        # Hook to capture attn output
        def make_attn_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    clean_a[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                else:
                    clean_a[key] = output[0, last_pos].detach().float().cpu().numpy()
            return hook
        
        # Hook to capture ffn output
        def make_ffn_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    clean_f[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                else:
                    clean_f[key] = output[0, last_pos].detach().float().cpu().numpy()
            return hook
        
        # Hook to capture layer output (post-hook)
        def make_post_hook(key):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    clean_h_post[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                else:
                    clean_h_post[key] = output[0, last_pos].detach().float().cpu().numpy()
            return hook
        
        hooks = []
        for li in range(n_layers):
            layer = layers_list[li]
            hooks.append(layer.register_forward_pre_hook(make_pre_hook(f"L{li}")))
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(f"L{li}")))
            if hasattr(layer, 'mlp'):
                hooks.append(layer.mlp.register_forward_hook(make_ffn_hook(f"L{li}")))
            hooks.append(layer.register_forward_hook(make_post_hook(f"L{li}")))
        
        with torch.no_grad():
            _ = model(input_ids)
        
        for h in hooks:
            h.remove()
        
        # Get embedding
        with torch.no_grad():
            embed_out = model.get_input_embeddings()(input_ids)
            h0 = embed_out[0, last_pos].detach().float().cpu().numpy()
        
        # === Run perturbed forward ===
        for trial in range(n_trials):
            random_dir = rng.randn(d_model).astype(np.float32)
            random_dir = random_dir / np.linalg.norm(random_dir)
            h0_norm = np.linalg.norm(h0)
            delta_vec = eps * h0_norm * random_dir
            
            pert_h = {}
            pert_a = {}
            pert_f = {}
            pert_h_post = {}
            
            # Pre-hooks to capture residual input
            def make_pert_pre_hook(key):
                def hook(module, args):
                    if isinstance(args, tuple):
                        h = args[0]
                    else:
                        h = args
                    pert_h[key] = h[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            def make_pert_attn_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        pert_a[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                    else:
                        pert_a[key] = output[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            def make_pert_ffn_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        pert_f[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                    else:
                        pert_f[key] = output[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            def make_pert_post_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        pert_h_post[key] = output[0][0, last_pos].detach().float().cpu().numpy()
                    else:
                        pert_h_post[key] = output[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            # Add perturbation at embedding via hook on first layer's pre-hook
            model_dtype = next(model.parameters()).dtype
            delta_tensor = torch.tensor(delta_vec, dtype=model_dtype, device=device)
            
            first_pre_hook_done = [False]
            
            def inject_perturbation(module, args):
                if not first_pre_hook_done[0]:
                    if isinstance(args, tuple):
                        h = args[0].clone()
                        h[0, last_pos] += delta_tensor
                        first_pre_hook_done[0] = True
                        return (h,) + args[1:]
                    else:
                        h = args.clone()
                        h[0, last_pos] += delta_tensor
                        first_pre_hook_done[0] = True
                        return h
                return args
            
            hooks = []
            hooks.append(layers_list[0].register_forward_pre_hook(inject_perturbation))
            for li in range(n_layers):
                layer = layers_list[li]
                if li > 0:
                    hooks.append(layer.register_forward_pre_hook(make_pert_pre_hook(f"L{li}")))
                else:
                    # For L0, we already have the pre-hook for injection
                    # Also capture the input
                    def capture_l0_input(module, args):
                        if isinstance(args, tuple):
                            pert_h["L0"] = args[0][0, last_pos].detach().float().cpu().numpy()
                        else:
                            pert_h["L0"] = args[0, last_pos].detach().float().cpu().numpy()
                    hooks.append(layers_list[0].register_forward_pre_hook(capture_l0_input))
                hooks.append(layer.self_attn.register_forward_hook(make_pert_attn_hook(f"L{li}")))
                if hasattr(layer, 'mlp'):
                    hooks.append(layer.mlp.register_forward_hook(make_pert_ffn_hook(f"L{li}")))
                hooks.append(layer.register_forward_hook(make_pert_post_hook(f"L{li}")))
            
            with torch.no_grad():
                try:
                    _ = model(input_ids)
                except Exception as e:
                    print(f"    Forward failed: {e}")
            
            for h in hooks:
                h.remove()
            
            # Compute deltas
            for li in range(n_layers):
                key = f"L{li}"
                if key in pert_a and key in clean_a and key in pert_f and key in clean_f:
                    da = pert_a[key] - clean_a[key]
                    df = pert_f[key] - clean_f[key]
                    da_norm = np.linalg.norm(da)
                    df_norm = np.linalg.norm(df)
                    
                    # Get the input perturbation at this layer
                    if key in pert_h and key in clean_h:
                        dh_input = pert_h[key] - clean_h[key]
                    else:
                        dh_input = None
                    
                    attn_norms[li].append(da_norm)
                    ffn_norms[li].append(df_norm)
                    
                    if da_norm + df_norm > 1e-10:
                        attn_frac[li].append(da_norm / (da_norm + df_norm))
                    
                    if dh_input is not None and np.linalg.norm(dh_input) > 1e-10:
                        attn_cos_with_input[li].append(proper_cos(da, dh_input))
                        ffn_cos_with_input[li].append(proper_cos(df, dh_input))
        
        print(f"    Done")
    
    # Print results
    print(f"\n  Attn/FFN perturbation decomposition:")
    print(f"  {'Layer':>6} | {'||dA||':>10} | {'||dF||':>10} | {'A/(A+F)':>8} | {'cos(dA,dh)':>11} | {'cos(dF,dh)':>11}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*10}-+-{'-'*8}-+-{'-'*11}-+-{'-'*11}")
    
    layer_summary = []
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    for li in range(n_layers):
        avg_an = np.mean(attn_norms[li]) if attn_norms[li] else 0
        avg_fn = np.mean(ffn_norms[li]) if ffn_norms[li] else 0
        avg_af = np.mean(attn_frac[li]) if attn_frac[li] else 0
        avg_ac = np.mean(attn_cos_with_input[li]) if attn_cos_with_input[li] else 0
        avg_fc = np.mean(ffn_cos_with_input[li]) if ffn_cos_with_input[li] else 0
        
        layer_summary.append({
            "layer": li,
            "attn_norm": float(avg_an),
            "ffn_norm": float(avg_fn),
            "attn_frac": float(avg_af),
            "attn_cos_input": float(avg_ac),
            "ffn_cos_input": float(avg_fc),
        })
        
        if li in sample_layers or li == n_layers - 1:
            print(f"  {li:>6} | {avg_an:>10.4f} | {avg_fn:>10.4f} | {avg_af:>8.4f} | {avg_ac:>11.4f} | {avg_fc:>11.4f}")
    
    # Analyze: which component amplifies more?
    shallow_af = np.mean([ls["attn_frac"] for ls in layer_summary if ls["layer"] < n_layers * 0.3])
    deep_af = np.mean([ls["attn_frac"] for ls in layer_summary if ls["layer"] > n_layers * 0.7])
    
    print(f"\n  Shallow attn fraction: {shallow_af:.4f}")
    print(f"  Deep attn fraction: {deep_af:.4f}")
    
    if deep_af < shallow_af - 0.05:
        print(f"  >>> FFN dominates perturbation amplification in deep layers")
    elif deep_af > shallow_af + 0.05:
        print(f"  >>> Attn dominates perturbation amplification in deep layers")
    else:
        print(f"  >>> Attn and FFN contribute roughly equally")
    
    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxvii"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp1_random_decomposition.json"
    
    summary = {
        "experiment": "exp1_random_decomposition",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "FFN contributes more to perturbation amplification than Attn",
        "eps": eps,
        "n_trials_per_text": n_trials,
        "layer_summary": layer_summary,
        "shallow_attn_frac": float(shallow_af),
        "deep_attn_frac": float(deep_af),
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# Exp2: 概念差异传播的Attn/FFN分解
# ============================================================
def exp2_concept_difference_decomposition(model_name):
    """Decompose concept difference propagation into Attn and FFN.
    
    Method:
      1. Run concept A (e.g., "dog"), capture: h_l, a_l, f_l
      2. Run concept B (e.g., "cat"), capture: h_l, a_l, f_l
      3. At each layer:
         - Delta_h = h_B - h_A (total concept difference)
         - Delta_a = a_B - a_A (attn's concept signal)
         - Delta_f = f_B - f_A (ffn's concept signal)
    
    Key measures:
      - ||Delta_a|| vs ||Delta_f|| (which carries more concept signal?)
      - cos(Delta_a, Delta_h) (does Attn concept signal align with total?)
      - cos(Delta_f, Delta_h) (does FFN concept signal align with total?)
    """
    print(f"\n{'='*70}")
    print(f"Exp2: Concept Difference Attn/FFN Decomposition")
    print(f"  Model: {model_name}")
    print(f"  Key test: Does Attn or FFN carry more concept signal?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers_list = get_layers(model)
    
    # Concept pairs: same category (within) and different categories (across)
    template = "The {} is"
    
    within_pairs = [
        ("dog", "cat"),    # animals
        ("apple", "rice"), # food
        ("hammer", "knife"), # tools
        ("mountain", "river"), # nature
    ]
    
    across_pairs = [
        ("dog", "apple"),   # animal vs food
        ("hammer", "mountain"), # tool vs nature
        ("cat", "river"),   # animal vs nature
        ("bread", "saw"),   # food vs tool
    ]
    
    def run_and_capture(text):
        """Run model and capture h, attn_out, ffn_out at all layers"""
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1
        
        h_in = {}   # residual at layer input
        a_out = {}  # attn output
        f_out = {}  # ffn output
        h_out = {}  # residual at layer output
        
        hooks = []
        for li in range(n_layers):
            layer = layers_list[li]
            
            def make_pre(key):
                def hook(module, args):
                    h_in[key] = args[0][0, last_pos].detach().float().cpu().numpy() if isinstance(args, tuple) else args[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            def make_attn(key):
                def hook(module, input, output):
                    a_out[key] = output[0][0, last_pos].detach().float().cpu().numpy() if isinstance(output, tuple) else output[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            def make_ffn(key):
                def hook(module, input, output):
                    f_out[key] = output[0][0, last_pos].detach().float().cpu().numpy() if isinstance(output, tuple) else output[0, last_pos].detach().float().cpu().numpy()
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
        
        # Also get embedding
        with torch.no_grad():
            embed = model.get_input_embeddings()(input_ids)
            h_in["L0_embed"] = embed[0, last_pos].detach().float().cpu().numpy()
        
        return h_in, a_out, f_out, h_out
    
    # Collect results
    within_results = defaultdict(lambda: defaultdict(list))
    across_results = defaultdict(lambda: defaultdict(list))
    
    print("\n  Within-category pairs:")
    for wordA, wordB in within_pairs:
        print(f"    {wordA} vs {wordB}")
        textA = template.format(wordA)
        textB = template.format(wordB)
        
        hA, aA, fA, hoA = run_and_capture(textA)
        hB, aB, fB, hoB = run_and_capture(textB)
        
        for li in range(n_layers):
            key = f"L{li}"
            if key in aA and key in aB and key in fA and key in fB:
                da = aB[key] - aA[key]
                df = fB[key] - fA[key]
                dh = hB.get(key, hoB.get(f"L{li-1}", None))
                if dh is None:
                    dh = hoB.get(f"L{li}", np.zeros(d_model)) - hoA.get(f"L{li}", np.zeros(d_model))
                else:
                    dh = dh - hA.get(key, np.zeros(d_model))
                
                da_norm = np.linalg.norm(da)
                df_norm = np.linalg.norm(df)
                
                within_results[li]["attn_norm"].append(da_norm)
                within_results[li]["ffn_norm"].append(df_norm)
                if da_norm + df_norm > 1e-10:
                    within_results[li]["attn_frac"].append(da_norm / (da_norm + df_norm))
                if np.linalg.norm(dh) > 1e-10:
                    within_results[li]["attn_cos"].append(proper_cos(da, dh))
                    within_results[li]["ffn_cos"].append(proper_cos(df, dh))
    
    print("\n  Across-category pairs:")
    for wordA, wordB in across_pairs:
        print(f"    {wordA} vs {wordB}")
        textA = template.format(wordA)
        textB = template.format(wordB)
        
        hA, aA, fA, hoA = run_and_capture(textA)
        hB, aB, fB, hoB = run_and_capture(textB)
        
        for li in range(n_layers):
            key = f"L{li}"
            if key in aA and key in aB and key in fA and key in fB:
                da = aB[key] - aA[key]
                df = fB[key] - fA[key]
                dh = hB.get(key, np.zeros(d_model)) - hA.get(key, np.zeros(d_model))
                
                da_norm = np.linalg.norm(da)
                df_norm = np.linalg.norm(df)
                
                across_results[li]["attn_norm"].append(da_norm)
                across_results[li]["ffn_norm"].append(df_norm)
                if da_norm + df_norm > 1e-10:
                    across_results[li]["attn_frac"].append(da_norm / (da_norm + df_norm))
                if np.linalg.norm(dh) > 1e-10:
                    across_results[li]["attn_cos"].append(proper_cos(da, dh))
                    across_results[li]["ffn_cos"].append(proper_cos(df, dh))
    
    # Print results
    print(f"\n  Concept difference Attn/FFN decomposition:")
    print(f"  {'Layer':>6} | {'W_A/(A+F)':>10} | {'W_cosA':>8} | {'W_cosF':>8} | {'X_A/(A+F)':>10} | {'X_cosA':>8} | {'X_cosF':>8}")
    print(f"  {'-'*6}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}-+-{'-'*8}-+-{'-'*8}")
    
    layer_summary = []
    sample_layers = list(range(0, n_layers, max(1, n_layers // 12)))
    if n_layers - 1 not in sample_layers:
        sample_layers.append(n_layers - 1)
    
    for li in range(n_layers):
        w_af = np.mean(within_results[li]["attn_frac"]) if within_results[li]["attn_frac"] else 0
        w_ac = np.mean(within_results[li]["attn_cos"]) if within_results[li]["attn_cos"] else 0
        w_fc = np.mean(within_results[li]["ffn_cos"]) if within_results[li]["ffn_cos"] else 0
        x_af = np.mean(across_results[li]["attn_frac"]) if across_results[li]["attn_frac"] else 0
        x_ac = np.mean(across_results[li]["attn_cos"]) if across_results[li]["attn_cos"] else 0
        x_fc = np.mean(across_results[li]["ffn_cos"]) if across_results[li]["ffn_cos"] else 0
        
        layer_summary.append({
            "layer": li,
            "within_attn_frac": float(w_af),
            "within_attn_cos": float(w_ac),
            "within_ffn_cos": float(w_fc),
            "across_attn_frac": float(x_af),
            "across_attn_cos": float(x_ac),
            "across_ffn_cos": float(x_fc),
        })
        
        if li in sample_layers or li == n_layers - 1:
            print(f"  {li:>6} | {w_af:>10.4f} | {w_ac:>8.4f} | {w_fc:>8.4f} | {x_af:>10.4f} | {x_ac:>8.4f} | {x_fc:>8.4f}")
    
    # Analysis
    w_shallow_af = np.mean([ls["within_attn_frac"] for ls in layer_summary if ls["layer"] < n_layers * 0.3])
    w_deep_af = np.mean([ls["within_attn_frac"] for ls in layer_summary if ls["layer"] > n_layers * 0.7])
    
    print(f"\n  Within-category:")
    print(f"    Shallow attn fraction: {w_shallow_af:.4f}")
    print(f"    Deep attn fraction: {w_deep_af:.4f}")
    
    if w_deep_af > w_shallow_af + 0.05:
        print(f"    >>> Attn carries MORE concept signal in deep layers")
    elif w_deep_af < w_shallow_af - 0.05:
        print(f"    >>> FFN carries MORE concept signal in deep layers")
    else:
        print(f"    >>> Roughly equal contribution")
    
    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxvii"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp2_concept_decomposition.json"
    
    summary = {
        "experiment": "exp2_concept_decomposition",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Attn carries more concept signal than FFN (especially in shallow/mid layers)",
        "layer_summary": layer_summary,
        "within_shallow_attn_frac": float(w_shallow_af),
        "within_deep_attn_frac": float(w_deep_af),
    }
    
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(json_serialize(summary), f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {out_path}")
    
    release_model(model)
    return summary


# ============================================================
# Exp3: 中层干预的Attn/FFN纠正分解
# ============================================================
def exp3_intervention_correction_decomposition(model_name):
    """Decompose 'correction after intervention' into Attn and FFN contributions.
    
    Method:
      1. Run concept A ("The dog is"), capture all residuals h_A[l]
      2. Run concept B ("The cat is"), capture all residuals h_B[l]
      3. For intervention at layer l:
         a. Replace A's residual at layer l with B's (using pre-hook)
         b. Run forward, capture Attn and FFN outputs at layers l, l+1, l+2
         c. Compare with clean B's outputs
    
    Key measure: After intervention, which component (Attn or FFN) at each 
    subsequent layer drives the residual toward B's direction?
    """
    print(f"\n{'='*70}")
    print(f"Exp3: Intervention Correction Attn/FFN Decomposition")
    print(f"  Model: {model_name}")
    print(f"  Key test: After intervention, does Attn or FFN drive correction?")
    print(f"{'='*70}")
    
    model, tokenizer, device = load_model(model_name)
    model_info = get_model_info(model, model_name)
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    layers_list = get_layers(model)
    
    template = "The {} is"
    
    # Concept pairs for intervention
    pairs = [
        ("dog", "cat"),     # within animals
        ("dog", "apple"),   # across categories
    ]
    
    def run_and_capture_full(text):
        """Run model and capture full residual stream + attn/ffn"""
        input_ids = tokenizer(text, return_tensors="pt").to(device).input_ids
        last_pos = input_ids.shape[1] - 1
        
        h_in = {}
        a_out = {}
        f_out = {}
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
                    a_out[key] = output[0][0, last_pos].detach().float().cpu().numpy() if isinstance(output, tuple) else output[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            def make_ffn(key):
                def hook(module, input, output):
                    f_out[key] = output[0][0, last_pos].detach().float().cpu().numpy() if isinstance(output, tuple) else output[0, last_pos].detach().float().cpu().numpy()
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
        
        return h_in, a_out, f_out, h_out, input_ids, last_pos
    
    # Intervention layers to test
    intervention_layers = list(range(0, n_layers, max(1, n_layers // 6)))
    if n_layers - 1 not in intervention_layers:
        intervention_layers.append(n_layers - 1)
    
    all_results = {}
    
    for wordA, wordB in pairs:
        pair_key = f"{wordA}->{wordB}"
        print(f"\n  Pair: {pair_key}")
        
        textA = template.format(wordA)
        textB = template.format(wordB)
        
        # Run clean A and B
        print(f"    Running clean A and B...")
        hA, aA, fA, hoA, idsA, last_pos = run_and_capture_full(textA)
        hB, aB, fB, hoB, idsB, _ = run_and_capture_full(textB)
        
        pair_results = []
        
        for int_layer in intervention_layers:
            print(f"    Intervention at L{int_layer}...")
            
            # We need h_B at the input of layer int_layer
            # h_B at input of layer l = h_out of layer l-1 (or embedding if l=0)
            if int_layer == 0:
                # Replace embedding
                with torch.no_grad():
                    embed_B = model.get_input_embeddings()(idsB)
                    h_B_at_l = embed_B[0, last_pos].detach().float().cpu().numpy()
            else:
                h_B_at_l = hoB.get(f"L{int_layer-1}", None)
                if h_B_at_l is None:
                    # Use h_in of layer int_layer from B's run
                    h_B_at_l = hB.get(f"L{int_layer}", None)
            
            if h_B_at_l is None:
                print(f"      Skip: no B residual at layer {int_layer}")
                continue
            
            # Run A with intervention: replace input at layer int_layer with B's residual
            model_dtype = next(model.parameters()).dtype
            h_B_tensor = torch.tensor(h_B_at_l, dtype=model_dtype, device=device)
            
            int_a = {}
            int_f = {}
            int_h_out = {}
            intervention_done = [False]
            
            def make_intervention_pre(l):
                def hook(module, args):
                    if l == int_layer and not intervention_done[0]:
                        if isinstance(args, tuple):
                            h = args[0].clone()
                            h[0, last_pos] = h_B_tensor
                            intervention_done[0] = True
                            return (h,) + args[1:]
                        else:
                            h = args.clone()
                            h[0, last_pos] = h_B_tensor
                            intervention_done[0] = True
                            return h
                    return args
                return hook
            
            def make_int_attn(key):
                def hook(module, input, output):
                    int_a[key] = output[0][0, last_pos].detach().float().cpu().numpy() if isinstance(output, tuple) else output[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            def make_int_ffn(key):
                def hook(module, input, output):
                    int_f[key] = output[0][0, last_pos].detach().float().cpu().numpy() if isinstance(output, tuple) else output[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            def make_int_post(key):
                def hook(module, input, output):
                    int_h_out[key] = output[0][0, last_pos].detach().float().cpu().numpy() if isinstance(output, tuple) else output[0, last_pos].detach().float().cpu().numpy()
                return hook
            
            hooks = []
            for li in range(n_layers):
                layer = layers_list[li]
                hooks.append(layer.register_forward_pre_hook(make_intervention_pre(li)))
                hooks.append(layer.self_attn.register_forward_hook(make_int_attn(f"L{li}")))
                if hasattr(layer, 'mlp'):
                    hooks.append(layer.mlp.register_forward_hook(make_int_ffn(f"L{li}")))
                hooks.append(layer.register_forward_hook(make_int_post(f"L{li}")))
            
            with torch.no_grad():
                try:
                    _ = model(idsA)
                except Exception as e:
                    print(f"      Forward failed: {e}")
            
            for h in hooks:
                h.remove()
            
            # Analyze: at each layer AFTER intervention, how do Attn and FFN compare?
            int_data = {"int_layer": int_layer}
            
            for li in range(int_layer, min(int_layer + 10, n_layers)):
                key = f"L{li}"
                if key in int_a and key in aB and key in int_f and key in fB:
                    # "Correction" = how much the intervened output moves toward B's output
                    # After intervention, the residual should approach B's
                    
                    # Intervened Attn vs clean B's Attn
                    int_a_vec = int_a[key]
                    clean_a_B = aB[key]
                    clean_a_A = aA.get(key, np.zeros_like(clean_a_B))
                    
                    int_f_vec = int_f[key]
                    clean_f_B = fB[key]
                    clean_f_A = fA.get(key, np.zeros_like(clean_f_B))
                    
                    # How close is the intervened output to B's output?
                    cos_int_a_B = proper_cos(int_a_vec, clean_a_B)
                    cos_int_a_A = proper_cos(int_a_vec, clean_a_A)
                    cos_int_f_B = proper_cos(int_f_vec, clean_f_B)
                    cos_int_f_A = proper_cos(int_f_vec, clean_f_A)
                    
                    int_data[f"L{li}"] = {
                        "attn_cos_B": float(cos_int_a_B),
                        "attn_cos_A": float(cos_int_a_A),
                        "attn_correction": float(cos_int_a_B - cos_int_a_A),  # positive = correcting toward B
                        "ffn_cos_B": float(cos_int_f_B),
                        "ffn_cos_A": float(cos_int_f_A),
                        "ffn_correction": float(cos_int_f_B - cos_int_f_A),
                    }
            
            pair_results.append(int_data)
        
        all_results[pair_key] = pair_results
    
    # Print results
    print(f"\n  Intervention correction decomposition:")
    for pair_key, pair_results in all_results.items():
        print(f"\n  Pair: {pair_key}")
        for int_data in pair_results:
            int_layer = int_data["int_layer"]
            print(f"    Intervention at L{int_layer}:")
            for li in range(int_layer, min(int_layer + 6, n_layers)):
                key = f"L{li}"
                if key in int_data:
                    d = int_data[key]
                    print(f"      L{li}: attn_corr={d['attn_correction']:+.4f}, ffn_corr={d['ffn_correction']:+.4f}")
    
    # Save
    out_dir = OUTPUT_DIR / f"{model_name}_cclxvii"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "exp3_intervention_decomposition.json"
    
    summary = {
        "experiment": "exp3_intervention_decomposition",
        "model": model_name,
        "timestamp": datetime.now().isoformat(),
        "hypothesis": "Attn drives correction toward target after intervention",
        "results": all_results,
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
        exp1_random_perturbation_decomposition(args.model)
    
    if args.exp in ["2", "all"]:
        exp2_concept_difference_decomposition(args.model)
    
    if args.exp in ["3", "all"]:
        exp3_intervention_correction_decomposition(args.model)
