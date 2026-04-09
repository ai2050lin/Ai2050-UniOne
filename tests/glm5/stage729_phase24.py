#!/usr/bin/env python3
"""
Stage 729: Phase XXIV — 高维切空间+传播操控+参与比维度+流形平滑度
================================================================================
Phase XXIII发现: 10维切空间只捕获66~88%, 中间层膨胀到31~55%。
Phase XXIV深入分析:

  P161: K维扫描 — 切空间维度从1到200, 找"饱和K"
  P162: 传播操控 — tangent/random操控后通过剩余层传播, 测真实KL
  P163: 参与比(Participation Ratio) — 估计流形有效维度(考虑高阶矩)
  P164: 流形平滑度 — 曲率沿层连续性, 检测"尖点"
  P165: 流形层级结构 — 不同层区域的维度和曲率分析

测试规模: 100文本×全层, 3模型

用法: python stage729_phase24.py --model qwen3
"""

import sys, time, gc, json, os, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path as _Path
from datetime import datetime
from collections import defaultdict

class Logger:
    def __init__(self, log_dir, name):
        os.makedirs(log_dir, exist_ok=True)
        self.f = open(os.path.join(log_dir, f"{name}.log"), "w", encoding="utf-8")
    def __call__(self, msg):
        try: print(msg)
        except UnicodeEncodeError:
            safe = msg.encode("gbk", errors="replace").decode("gbk")
            print(safe)
        self.f.write(msg + "\n")
        self.f.flush()
    def close(self):
        self.f.close()

log = None

def build_texts():
    T = []
    gen_en = [
        "The cat sat on the mat.", "A beautiful sunset over the ocean.", "The stock market crashed today.",
        "Climate change is a global challenge.", "The restaurant serves excellent Italian cuisine.",
        "The Renaissance was a period of great cultural achievement.", "The election results surprised everyone last night.",
        "The concert featured a symphony orchestra performance.", "She walked through the garden with a smile.",
        "The ancient castle stood on top of the hill.", "Breakfast is the most important meal of the day.",
        "The children played happily in the park.", "A sudden storm interrupted the picnic.",
        "The library has thousands of rare books.", "He traveled across three continents last year.",
        "The museum exhibit attracted record visitors.", "The teacher explained the concept clearly.",
        "Spring flowers bloom in every color.", "The old man told stories by the fireplace.",
        "Music brings people together across cultures.",
    ]
    for t in gen_en: T.append((t, "gen_en"))
    math_sci = [
        "Mathematical proof by induction.", "Einstein's theory of relativity changed physics.",
        "The quadratic formula solves ax^2+bx+c=0.", "Quantum entanglement connects distant particles.",
        "The Navier-Stokes equations describe fluid dynamics.", "Fermat's Last Theorem was proven by Andrew Wiles.",
        "The Riemann hypothesis remains unproven.", "Thermodynamics governs energy conservation.",
        "The periodic table organizes chemical elements.", "Bayesian inference updates probability distributions.",
        "The Fourier transform converts time to frequency domain.", "Topology studies properties preserved under deformation.",
        "Godel's incompleteness theorem limits formal systems.", "The standard model describes fundamental particles.",
        "Catalan numbers count binary trees.", "Group theory classifies symmetries.",
        "Information theory defines entropy as H=-sum(p*log(p)).", "The Boltzmann distribution describes statistical mechanics.",
        "Lambda calculus provides a foundation for computation.", "The Goldbach conjecture proposes every even number is the sum of two primes.",
    ]
    for t in math_sci: T.append((t, "math_sci"))
    poetry = [
        "Roses are red, violets are blue.", "The road not taken diverged in a yellow wood.",
        "Shall I compare thee to a summer's day?", "Two roads diverged in a wood, and I took the one less traveled.",
        "The fog comes on little cat feet.", "Hope is the thing with feathers that perches in the soul.",
        "In Xanadu did Kubla Khan a stately pleasure-dome decree.", "Do not go gentle into that good night.",
        "The Waste Land is T.S. Eliot's masterpiece.", "Emily Dickinson wrote about death and immortality.",
        "Haiku: an old silent pond / a frog jumps into the pond / splash! silence again.",
        "The Raven by Edgar Allan Poe features nevermore.", "i carry your heart with me by e.e. cummings.",
        "Daffodils by William Wordsworth celebrates nature.", "The Love Song of J. Alfred Prufrock is by T.S. Eliot.",
        "Ode to a Nightingale by John Keats explores beauty.", "Robert Frost won four Pulitzer Prizes for poetry.",
        "Free verse poetry breaks traditional meter.", "Sonnet 18 is Shakespeare's most famous poem.",
        "The Lake Isle of Innisfree by Yeats yearns for escape.", "Langston Hughes captured the Harlem Renaissance.",
    ]
    for t in poetry: T.append((t, "poetry"))
    code = [
        "def fibonacci(n): return n if n<2 else fibonacci(n-1)+fibonacci(n-2)",
        "for i in range(len(arr)): print(arr[i])",
        "class Node: def __init__(self, val): self.val=val; self.next=None",
        "x = [i**2 for i in range(10)]", "import numpy as np; A=np.random.randn(3,3)",
        "def quicksort(arr): if len(arr)<=1: return arr; pivot=arr[0]",
        "SELECT * FROM users WHERE age > 18 ORDER BY name;",
        "git commit -m 'fix: resolve null pointer exception'",
        "docker run -d -p 8080:80 nginx", "npm install express --save",
        "const sum = arr.reduce((a,b) => a+b, 0);",
        "public static void main(String[] args) { System.out.println('Hello'); }",
        "func handler(w http.ResponseWriter, r *http.Request) { fmt.Fprintf(w, 'OK') }",
        "print(f'The answer is {42}')", "x = torch.randn(2, 3, requires_grad=True)",
        "model.fit(X_train, y_train, epochs=100, batch_size=32)",
        "var xhr = new XMLHttpRequest(); xhr.open('GET', '/api/data');",
        "CREATE TABLE IF NOT EXISTS orders (id INT PRIMARY KEY, total DECIMAL);",
        "python -m pytest tests/ -v --cov=src", "curl -X POST -H 'Content-Type: application/json' localhost:5000",
        "async function fetchData() { const res = await fetch(url); return res.json(); }",
    ]
    for t in code: T.append((t, "code"))
    chinese = [
        "\u4eba\u5de5\u667a\u80fd\u6b63\u5728\u6539\u53d8\u4e16\u754c\u3002","\u4e2d\u56fd\u7684\u7ecf\u6d4e\u53d1\u5c55\u53d6\u5f97\u4e86\u663e\u8457\u6210\u5c31\u3002",
        "\u6559\u80b2\u662f\u56fd\u4e4b\u5927\u8ba1\uff0c\u515a\u4e4b\u5927\u8ba1\u3002","\u79d1\u6280\u81ea\u7acb\u81ea\u5f3a\u662f\u56fd\u5bb6\u53d1\u5c55\u7684\u6218\u7565\u652f\u6491\u3002",
        "\u6587\u5316\u81ea\u4fe1\u662f\u4e00\u4e2a\u6c11\u65cf\u6700\u57fa\u672c\u7684\u529b\u91cf\u3002","\u7eff\u6c34\u9752\u5c71\u5c31\u662f\u91d1\u5c71\u94f6\u5c71\u3002",
        "\u4eba\u6c11\u5bf9\u7f8e\u597d\u751f\u6d3b\u7684\u5411\u5f80\u5c31\u662f\u6211\u4eec\u7684\u594b\u6597\u76ee\u6807\u3002","\u521b\u65b0\u662f\u5f15\u9886\u53d1\u5c55\u7684\u7b2c\u4e00\u52a8\u529b\u3002",
        "\u6570\u5b57\u7ecf\u6d4e\u6210\u4e3a\u656c\u7684\u589e\u957f\u5f15\u64ce\u3002","\u4e61\u6751\u632f\u5174\u6218\u7565\u5168\u9762\u63a8\u8fdb\u3002",
        "\u9ad8\u8d28\u91cf\u53d1\u5c55\u662f\u65f6\u4ee3\u7684\u8981\u6c42\u3002","\u5bf9\u5916\u5f00\u653e\u7684\u57fa\u672c\u56fd\u7b56\u4e0d\u4f1a\u6539\u53d8\u3002",
        "\u4e2d\u56fd\u5f0f\u73b0\u4ee3\u5316\u9053\u8def\u8d8a\u8d70\u8d8a\u5bbd\u5e7f\u3002","\u78b3\u4e2d\u548c\u76ee\u6807\u63a8\u52a8\u80fd\u6e90\u8f6c\u578b\u3002",
        "\u4e2d\u533b\u836f\u4f20\u627f\u521b\u65b0\u53d1\u5c55\u3002","\u4f20\u7edf\u6587\u5316\u7684\u521b\u9020\u6027\u8f6c\u5316\u548c\u521b\u65b0\u6027\u53d1\u5c55\u3002",
        "\u56fd\u5bb6\u6cbb\u7406\u4f53\u7cfb\u548c\u6cbb\u7406\u80fd\u529b\u73b0\u4ee3\u5316\u3002","\u6784\u5efa\u4eba\u7c7b\u547d\u8fd0\u5171\u540c\u4f53\u3002",
        "\u4e00\u5e26\u4e00\u8def\u5021\u8bae\u4fc3\u8fdb\u5171\u540c\u53d1\u5c55\u3002","\u592a\u7a7a\u63a2\u7d22\u53d6\u5f97\u91cd\u5927\u7a81\u7834\u3002",
    ]
    for t in chinese: T.append((t, "chinese"))
    return T

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
}

def load_model(mname):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = MODEL_MAP[mname]
    log(f"  Loading {mname} from {p.name}...")
    model = AutoModelForCausalLM.from_pretrained(
        str(p), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    cfg = model.config
    n_layers = getattr(cfg, 'num_hidden_layers', None) or getattr(cfg, 'n_layers', None)
    d_model = getattr(cfg, 'hidden_size', None) or getattr(cfg, 'd_model', 2048)
    log(f"  {mname}: n_layers={n_layers}, d_model={d_model}, device={model.device}")
    return model, tokenizer, n_layers, d_model

def get_unembed(model):
    if hasattr(model, 'lm_head'):
        um = model.lm_head
    elif hasattr(model, 'get_output_embeddings'):
        um = model.get_output_embeddings()
    else:
        return None, None
    w = um.weight.detach().to(torch.float32).cpu()
    b = um.bias.detach().to(torch.float32).cpu() if (hasattr(um, 'bias') and um.bias is not None) else None
    return w, b

def kl_div(p, q):
    p = p + 1e-10
    q = q + 1e-10
    return (p * (p.log() - q.log())).sum().item()

def cos_sim(a, b):
    a, b = a.float(), b.float()
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()

def get_model_layers(model):
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model.model.layers, model.model.norm
    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return model.transformer.h, model.transformer.ln_f
    return None, None

def clean_forward(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    logits = outputs.logits[0, -1, :].float().cpu()
    h_all = {}
    for l in range(len(outputs.hidden_states)):
        h_all[l] = outputs.hidden_states[l][:, -1, :].float().cpu().squeeze(0)
    return logits, h_all

def precompute_all_hidden(model, tokenizer, texts, n_layers):
    data = {}
    for i, (text, cat) in enumerate(texts):
        logits, h_all = clean_forward(model, tokenizer, text)
        data[i] = {"text": text, "cat": cat, "logits": logits, "h": h_all}
        if (i + 1) % 20 == 0:
            log(f"  Precomputed {i+1}/{len(texts)} texts")
    return data


# =====================================================================
# P161: K-dimensional Tangent Space Scan
# =====================================================================
def p161_k_dim_scan(data, n_layers, d_model):
    """
    P161: Scan tangent space dimension K from 1 to min(200, d_model).
    Find the "saturation K" where adding more dimensions gives diminishing returns.
    """
    log("\n" + "="*80)
    log("P161: K-Dimensional Tangent Space Scan")
    log("="*80)
    
    K_values = [1, 2, 3, 5, 10, 15, 20, 30, 50, 75, 100, 150, 200]
    # Filter K_values that exceed d_model
    K_values = [k for k in K_values if k <= d_model]
    
    # Test at specific layers
    test_layers = list(range(1, n_layers - 1, max(1, n_layers // 6)))
    
    n_texts = min(50, len(data))
    
    # Store results: capture[layer][K] = avg capture
    capture_by_layer = defaultdict(dict)
    
    for l in test_layers:
        for K in K_values:
            captures = []
            for idx in range(n_texts):
                h = data[idx]["h"]
                # Build local tangent space from window around layer l
                window = range(max(0, l - 3), min(n_layers, l + 4))
                H_local = torch.stack([h[ll] for ll in window], dim=0)
                H_centered = H_local - H_local.mean(dim=0, keepdim=True)
                
                try:
                    _, S, Vt = torch.linalg.svd(H_centered, full_matrices=False)
                    K_use = min(K, Vt.shape[0], Vt.shape[1])
                    tangent_basis = Vt[:K_use]
                    
                    h_l = h[l]
                    h_proj = torch.zeros_like(h_l)
                    for k_idx in range(K_use):
                        v = tangent_basis[k_idx]
                        v_norm = v.norm()
                        if v_norm > 1e-8:
                            v = v / v_norm
                            h_proj += torch.dot(h_l, v) * v
                    
                    h_norm_sq = h_l.norm().item() ** 2
                    proj_norm_sq = h_proj.norm().item() ** 2
                    cap = proj_norm_sq / h_norm_sq if h_norm_sq > 1e-8 else 0
                    captures.append(cap)
                except:
                    pass
            
            if captures:
                capture_by_layer[l][K] = np.mean(captures)
    
    log(f"\nP161 Results: Tangent Space Capture vs K")
    log("-" * 70)
    header = f"{'Layer':>6}"
    for K in K_values:
        header += f" | K={K:>3d}"
    log(header)
    log("-" * 70)
    
    for l in test_layers:
        row = f"L{l:>3d}   "
        for K in K_values:
            val = capture_by_layer[l].get(K, 0)
            row += f" | {val:>6.3f}"
        log(row)
    
    # Find saturation K (where capture first exceeds 90%)
    log(f"\nP161 Saturation Analysis (capture >= 90%):")
    for l in test_layers:
        for K in K_values:
            cap = capture_by_layer[l].get(K, 0)
            if cap >= 0.90:
                log(f"  L{l}: saturates at K={K} (capture={cap:.3f})")
                break
        else:
            max_cap = max(capture_by_layer[l].values()) if capture_by_layer[l] else 0
            log(f"  L{l}: does NOT saturate at K={K_values[-1]} (max={max_cap:.3f})")
    
    return {"capture": {str(l): v for l, v in capture_by_layer.items()},
            "K_values": K_values, "test_layers": test_layers}


# =====================================================================
# P162: Propagated Manipulation (with forward hooks)
# =====================================================================
def p162_propagated_manipulation(model, tokenizer, data, uw, ub, n_layers, d_model):
    """
    P162: Manipulate h at layer l, then propagate through remaining layers using forward hooks.
    Compare tangent vs random direction after propagation.
    """
    log("\n" + "="*80)
    log("P162: Propagated Manipulation (forward hook)")
    log("="*80)
    
    layers, norm = get_model_layers(model)
    if layers is None:
        log("  ERROR: Cannot get model layers")
        return {}
    
    n_texts = min(20, len(data))
    mid_layer = n_layers // 2
    test_layers = [0, mid_layer // 2, mid_layer, min(3 * mid_layer // 2, n_layers - 2)]
    scale = 1.0
    
    tangent_kls_prop = defaultdict(list)
    random_kls_prop = defaultdict(list)
    
    for idx in range(n_texts):
        text = data[idx]["text"]
        logits_orig = data[idx]["logits"]
        probs_orig = F.softmax(logits_orig, dim=-1)
        
        for layer in test_layers:
            if layer >= n_layers - 1:
                continue
            
            h = data[idx]["h"]
            h_l = h[layer]
            
            # Get tangent direction
            tangent = h[layer + 1] - h[layer]
            t_norm = tangent.norm()
            if t_norm < 1e-8:
                continue
            tangent_dir = tangent / t_norm
            
            # Random direction
            random_dir = torch.randn(d_model)
            random_dir = random_dir / random_dir.norm()
            
            # Manipulation magnitude = ||delta|| * scale
            mag = t_norm.item() * scale
            
            for dir_name, dir_vec in [("tangent", tangent_dir), ("random", random_dir)]:
                h_modified = h_l + mag * dir_vec
                
                # Use forward hook to inject modified h at layer
                patch_applied = [False]
                
                def hook_fn(module, input, output, h_mod=h_modified):
                    if patch_applied[0]:
                        return output
                    h_out = output[0] if isinstance(output, tuple) else output
                    h_out_mod = h_out.clone()
                    h_out_mod[0, -1, :] = h_mod.float().to(h_out.device)
                    patch_applied[0] = True
                    if isinstance(output, tuple):
                        return (h_out_mod,) + output[1:]
                    return h_out_mod
                
                handle = layers[layer].register_forward_hook(hook_fn)
                
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).to(model.device)
                with torch.no_grad():
                    outputs = model(**inputs)
                
                handle.remove()
                
                logits_mod = outputs.logits[0, -1, :].float().cpu()
                probs_mod = F.softmax(logits_mod, dim=-1)
                kl = kl_div(probs_orig, probs_mod)
                
                if dir_name == "tangent":
                    tangent_kls_prop[layer].append(kl)
                else:
                    random_kls_prop[layer].append(kl)
    
    log(f"\nP162 Results: Propagated Manipulation (scale={scale})")
    log("-" * 70)
    log(f"{'Layer':>6} | {'Tangent KL':>11} | {'Random KL':>10} | {'T/R ratio':>10}")
    log("-" * 70)
    
    for layer in test_layers:
        if layer in tangent_kls_prop and tangent_kls_prop[layer]:
            kt = np.mean(tangent_kls_prop[layer])
            kr = np.mean(random_kls_prop[layer]) if random_kls_prop[layer] else 0
            ratio = kt / kr if kr > 1e-8 else 0
            log(f"L{layer:>3d}   | {kt:>11.4f} | {kr:>10.4f} | {ratio:>10.4f}")
    
    log(f"\nP162 Key Findings:")
    # Compare with Phase XXIII P159 (direct readout)
    log(f"  Propagated manipulation: direction DOES matter after propagation?")
    
    return {"tangent_kls": {str(k): float(np.mean(v)) for k, v in tangent_kls_prop.items()},
            "random_kls": {str(k): float(np.mean(v)) for k, v in random_kls_prop.items()}}


# =====================================================================
# P163: Participation Ratio (Effective Dimensionality)
# =====================================================================
def p163_participation_ratio(data, n_layers, d_model):
    """
    P163: Compute Participation Ratio (PR) as effective dimensionality.
    PR = (sum(lambda_i))^2 / sum(lambda_i^2) where lambda_i are eigenvalues.
    This considers the actual distribution of energy, not just PCA count.
    """
    log("\n" + "="*80)
    log("P163: Participation Ratio (Effective Dimensionality)")
    log("="*80)
    
    pr_values = defaultdict(list)
    pca90_dim = defaultdict(list)
    
    n_texts = min(50, len(data))
    
    for idx in range(n_texts):
        h = data[idx]["h"]
        
        for l in range(n_layers):
            # Compute covariance of h across positions (using last token only)
            # Instead, use singular values of the h vector itself decomposed
            # Better: use multiple texts to build a covariance matrix
            pass
    
    # Alternative: Use eigenvalue spectrum of h_l across texts
    # Build covariance: C = (1/N) sum_i h_l_i @ h_l_i^T
    # But this is d x d which is too large. Use Hutchinson estimator instead.
    
    # Simpler: Use PCA explained variance from multiple texts
    log("  Computing PR from text covariance...")
    
    for l in range(n_layers):
        # Collect h_l vectors from all texts
        H = torch.stack([data[idx]["h"][l] for idx in range(n_texts)], dim=0)  # N x d
        H_centered = H - H.mean(dim=0, keepdim=True)
        
        # Use randomized SVD for efficiency (power iteration)
        # Compute top-100 singular values
        try:
            # Use torch.svd_lowrank for efficiency
            Q, S, V = torch.svd_lowrank(H_centered, q=min(100, d_model, n_texts))
            
            # Participation Ratio from singular values
            S_np = S.numpy()
            S_sq = S_np ** 2
            PR = (np.sum(S_sq)) ** 2 / np.sum(S_sq ** 2) if np.sum(S_sq ** 2) > 0 else 0
            pr_values[l].append(PR)
            
            # PCA90: how many components to explain 90% variance
            cumvar = np.cumsum(S_sq) / np.sum(S_sq)
            n90 = np.searchsorted(cumvar, 0.90) + 1
            pca90_dim[l].append(n90)
        except Exception as e:
            log(f"  Warning: SVD failed at L{l}: {e}")
    
    log(f"\nP163 Results: Participation Ratio and PCA90")
    log("-" * 60)
    log(f"{'Layer':>6} | {'PR (eff dim)':>13} | {'PCA90 dim':>10}")
    log("-" * 60)
    
    for l in sorted(pr_values.keys()):
        pr_val = np.mean(pr_values[l])
        pca90 = np.mean(pca90_dim[l]) if l in pca90_dim else 0
        log(f"L{l:>3d}   | {pr_val:>13.1f} | {pca90:>10.1f}")
    
    log(f"\nP163 Key Findings:")
    all_pr = [v[0] for v in pr_values.values()]
    all_pca90 = [v[0] for v in pca90_dim.values()]
    log(f"  INV-493: Mean PR = {np.mean(all_pr):.1f} (effective dimensionality)")
    log(f"  INV-494: Mean PCA90 = {np.mean(all_pca90):.1f} dimensions for 90% variance")
    log(f"  PR/d_model ratio = {np.mean(all_pr)/d_model:.4f}")
    
    # Layer profile
    n_third = n_layers // 3
    early_pr = [pr_values[l][0] for l in range(n_third) if l in pr_values]
    mid_pr = [pr_values[l][0] for l in range(n_third, 2*n_third) if l in pr_values]
    late_pr = [pr_values[l][0] for l in range(2*n_third, n_layers) if l in pr_values]
    
    log(f"  Early PR: {np.mean(early_pr):.1f}, Mid PR: {np.mean(mid_pr):.1f}, Late PR: {np.mean(late_pr):.1f}")
    
    return {
        "PR": {str(k): float(v[0]) for k, v in pr_values.items()},
        "PCA90": {str(k): float(v[0]) for k, v in pca90_dim.items()},
        "INV-493": float(np.mean(all_pr)),
        "INV-494": float(np.mean(all_pca90)),
    }


# =====================================================================
# P164: Manifold Smoothness (Curvature Continuity)
# =====================================================================
def p164_manifold_smoothness(data, n_layers, d_model):
    """
    P164: Measure how smoothly curvature changes across layers.
    Detect "sharp points" where curvature jumps discontinuously.
    """
    log("\n" + "="*80)
    log("P164: Manifold Smoothness Analysis")
    log("="*80)
    
    n_texts = min(100, len(data))
    
    # Compute per-layer metrics
    step_angles = defaultdict(list)
    delta_norms = defaultdict(list)
    h_norms = defaultdict(list)
    
    for idx in range(n_texts):
        h = data[idx]["h"]
        for l in range(n_layers - 1):
            delta = h[l + 1] - h[l]
            delta_norms[l].append(delta.norm().item())
            h_norms[l].append(h[l].norm().item())
            
            if l > 0:
                delta_prev = h[l] - h[l - 1]
                dn = delta.norm().item()
                dpn = delta_prev.norm().item()
                if dn > 1e-8 and dpn > 1e-8:
                    cos_ang = cos_sim(delta_prev, delta)
                    angle = math.degrees(math.acos(max(-1, min(1, cos_ang))))
                    step_angles[l].append(angle)
    
    # Smoothness: |K_{l+1} - K_l| (curvature difference between adjacent layers)
    curvature_diffs = {}
    for l in range(1, n_layers - 2):
        if l in step_angles and (l + 1) in step_angles:
            k_l = np.mean(step_angles[l])
            k_l1 = np.mean(step_angles[l + 1])
            curvature_diffs[l] = abs(k_l1 - k_l)
    
    log(f"\nP164 Results: Curvature Smoothness")
    log("-" * 60)
    log(f"{'Layer':>6} | {'Step Angle':>11} | {'Delta Norm':>11} | {'Curv Diff':>10}")
    log("-" * 60)
    
    for l in range(1, n_layers - 2):
        sa = np.mean(step_angles[l]) if l in step_angles else 0
        dn = np.mean(delta_norms[l]) if l in delta_norms else 0
        cd = curvature_diffs.get(l, 0)
        marker = " ***" if cd > 15 else ""
        log(f"L{l:>3d}   | {sa:>10.2f} deg | {dn:>11.2f} | {cd:>10.2f}{marker}")
    
    # Find sharp points (curvature jump > 15 degrees)
    sharp_points = [l for l, cd in curvature_diffs.items() if cd > 15]
    log(f"\nP164 Key Findings:")
    log(f"  INV-495: Sharp curvature points (diff > 15 deg): {sharp_points}")
    log(f"  Mean curvature diff: {np.mean(list(curvature_diffs.values())):.2f}")
    log(f"  Max curvature diff: {np.max(list(curvature_diffs.values())):.2f} at L{max(curvature_diffs, key=curvature_diffs.get)}")
    
    # Norm smoothness
    norm_diffs = {}
    for l in range(1, n_layers):
        if l in h_norms and (l-1) in h_norms:
            n1 = np.mean(h_norms[l-1])
            n2 = np.mean(h_norms[l])
            if n1 > 1e-8:
                norm_diffs[l] = abs(math.log(n2 / n1))  # log-ratio
    
    log(f"  INV-496: Mean log-norm ratio change: {np.mean(list(norm_diffs.values())):.4f}")
    
    return {
        "curvature_diffs": {str(k): float(v) for k, v in curvature_diffs.items()},
        "sharp_points": sharp_points,
        "INV-495": sharp_points,
        "INV-496": float(np.mean(list(norm_diffs.values()))),
    }


# =====================================================================
# P165: Manifold Hierarchy (Region Analysis)
# =====================================================================
def p165_manifold_hierarchy(data, n_layers, d_model):
    """
    P165: Divide layers into regions and analyze each region's properties.
    Region 1: L0 to ~L_N/3 (feature extraction)
    Region 2: ~L_N/3 to ~2L_N/3 (feature integration)
    Region 3: ~2L_N/3 to L_N (decision/output)
    """
    log("\n" + "="*80)
    log("P165: Manifold Hierarchy (Region Analysis)")
    log("="*80)
    
    n_texts = min(100, len(data))
    
    regions = {
        "early": (0, n_layers // 3),
        "mid": (n_layers // 3, 2 * n_layers // 3),
        "late": (2 * n_layers // 3, n_layers),
    }
    
    region_metrics = {}
    
    for region_name, (start, end) in regions.items():
        angles = []
        norm_ratios = []
        cos_consecutive = []
        local_dims = []
        
        for idx in range(n_texts):
            h = data[idx]["h"]
            for l in range(max(1, start), min(end, n_layers - 1)):
                delta = h[l + 1] - h[l]
                delta_prev = h[l] - h[l - 1]
                dn = delta.norm().item()
                dpn = delta_prev.norm().item()
                
                if dn > 1e-8 and dpn > 1e-8:
                    cos_ang = cos_sim(delta_prev, delta)
                    angle = math.degrees(math.acos(max(-1, min(1, cos_ang))))
                    angles.append(angle)
                
                # Norm growth rate
                h_norm = h[l].norm().item()
                if h_norm > 1e-8:
                    norm_ratios.append(dn / h_norm if h_norm > 0 else 0)
                
                # Consecutive h cosine
                cos_h = cos_sim(h[l], h[l + 1])
                cos_consecutive.append(cos_h)
        
        # Effective dimensionality per region (simplified)
        H_region = []
        for idx in range(min(30, n_texts)):
            for l in range(start, min(end, n_layers)):
                H_region.append(data[idx]["h"][l])
        
        if H_region:
            H_stack = torch.stack(H_region, dim=0)
            try:
                _, S, _ = torch.svd_lowrank(H_stack - H_stack.mean(0), q=min(50, d_model))
                S_sq = S.numpy() ** 2
                eff_dim = (np.sum(S_sq))**2 / np.sum(S_sq**2) if np.sum(S_sq**2) > 0 else 0
                local_dims.append(eff_dim)
            except:
                pass
        
        region_metrics[region_name] = {
            "mean_angle": np.mean(angles) if angles else 0,
            "std_angle": np.std(angles) if angles else 0,
            "mean_norm_ratio": np.mean(norm_ratios) if norm_ratios else 0,
            "mean_cos_consec": np.mean(cos_consecutive) if cos_consecutive else 0,
            "eff_dim": local_dims[0] if local_dims else 0,
            "n_layers": end - start,
        }
    
    log(f"\nP165 Results: Manifold Region Properties")
    log("-" * 70)
    log(f"{'Region':>8} | {'Layers':>6} | {'Avg Angle':>10} | {'Angle Std':>10} | {'Norm Rat':>8} | {'cos(h,h+)':>10} | {'Eff Dim':>8}")
    log("-" * 70)
    
    for name in ["early", "mid", "late"]:
        m = region_metrics[name]
        log(f"{name:>8s} | {m['n_layers']:>6d} | {m['mean_angle']:>9.2f} deg | {m['std_angle']:>10.2f} | {m['mean_norm_ratio']:>8.4f} | {m['mean_cos_consec']:>10.4f} | {m['eff_dim']:>8.1f}")
    
    log(f"\nP165 Key Findings:")
    # Compare regions
    early_dim = region_metrics["early"]["eff_dim"]
    mid_dim = region_metrics["mid"]["eff_dim"]
    late_dim = region_metrics["late"]["eff_dim"]
    
    log(f"  INV-497: Effective dimensions — Early: {early_dim:.0f}, Mid: {mid_dim:.0f}, Late: {late_dim:.0f}")
    
    early_angle = region_metrics["early"]["mean_angle"]
    mid_angle = region_metrics["mid"]["mean_angle"]
    late_angle = region_metrics["late"]["mean_angle"]
    
    log(f"  INV-498: Curvature — Early: {early_angle:.1f} deg, Mid: {mid_angle:.1f} deg, Late: {late_angle:.1f} deg")
    
    # Most curved region
    max_region = max(["early", "mid", "late"], key=lambda x: region_metrics[x]["mean_angle"])
    log(f"  Most curved region: {max_region} ({region_metrics[max_region]['mean_angle']:.1f} deg)")
    
    # Most variable region
    max_var = max(["early", "mid", "late"], key=lambda x: region_metrics[x]["std_angle"])
    log(f"  Most variable region: {max_var} (std={region_metrics[max_var]['std_angle']:.1f} deg)")
    
    return {"regions": region_metrics,
            "INV-497": {"early": float(early_dim), "mid": float(mid_dim), "late": float(late_dim)},
            "INV-498": {"early": float(early_angle), "mid": float(mid_angle), "late": float(late_angle)}}


# =====================================================================
# Main
# =====================================================================
def main():
    global log
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()
    
    mname = args.model
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    out_dir = _Path(f"tests/glm5_temp/stage729_phase24_{mname}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log = Logger(str(out_dir), "results")
    log(f"Phase XXIV: High-Dim Tangent + Propagated Manip + Participation Ratio — {mname}")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    t0 = time.time()
    
    model, tokenizer, n_layers, d_model = load_model(mname)
    uw, ub = get_unembed(model)
    
    texts = build_texts()
    log(f"\nPrecomputing hidden states for {len(texts)} texts, {n_layers} layers...")
    data = precompute_all_hidden(model, tokenizer, texts, n_layers)
    log(f"Precomputation done in {time.time()-t0:.1f}s")
    
    results = {}
    
    # P161: K-dim scan
    t1 = time.time()
    results["P161"] = p161_k_dim_scan(data, n_layers, d_model)
    log(f"P161 done in {time.time()-t1:.1f}s")
    
    # P163: Participation Ratio
    t3 = time.time()
    results["P163"] = p163_participation_ratio(data, n_layers, d_model)
    log(f"P163 done in {time.time()-t3:.1f}s")
    
    # P164: Manifold smoothness
    t4 = time.time()
    results["P164"] = p164_manifold_smoothness(data, n_layers, d_model)
    log(f"P164 done in {time.time()-t4:.1f}s")
    
    # P165: Manifold hierarchy
    t5 = time.time()
    results["P165"] = p165_manifold_hierarchy(data, n_layers, d_model)
    log(f"P165 done in {time.time()-t5:.1f}s")
    
    # P162: Propagated manipulation (most expensive, last)
    t2 = time.time()
    results["P162"] = p162_propagated_manipulation(model, tokenizer, data, uw, ub, n_layers, d_model)
    log(f"P162 done in {time.time()-t2:.1f}s")
    
    total_time = time.time() - t0
    results["meta"] = {"model": mname, "n_layers": n_layers, "d_model": d_model,
                       "n_texts": len(texts), "total_time_min": round(total_time/60, 1)}
    
    def convert(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    results = convert(results)
    
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    log(f"\nTotal time: {total_time:.1f}s ({total_time/60:.1f} min)")
    log(f"Results saved to {out_dir / 'results.json'}")
    
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    log.close()
    return str(out_dir)

if __name__ == "__main__":
    main()
