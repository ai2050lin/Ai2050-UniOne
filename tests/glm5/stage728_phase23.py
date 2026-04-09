#!/usr/bin/env python3
"""
Stage 728: Phase XXIII — 语言流形理论+残差流曲率+跨模型流形对比
================================================================================
Phase XXII发现: RMSNorm不是纯缩放, 三模型最后层机制完全不同, 修复ratio=4~509x。
Phase XXIII深入分析"语言流形"的几何结构:

理论框架: LCS(语言计算结构)假设 — 语言在高维空间中形成特定流形M
  P156: 残差流曲率测量 — 沿路径测局部曲率, 量化"弯曲程度"
  P157: 校正项方向分析 — delta_h的方向与h的夹角, 正交/平行分解
  P158: 跨模型流形同构性 — 不同模型的h流是否可对齐? Procrustes分析
  P159: 流形上操控 vs 欧几里得操控 — 沿切方向操控 vs 随机方向操控
  P160: 流形切空间投影 — 投影到局部切空间后操控的效果对比

测试规模: 100文本×全层, 3模型, 大数据量确保统计可靠性

用法: python stage728_phase23.py --model qwen3
      python stage728_phase23.py --model deepseek7b
      python stage728_phase23.py --model glm4
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
        "人工智能正在改变世界。","中国的经济发展取得了显著成就。",
        "教育是国之大计，党之大计。","科技自立自强是国家发展的战略支撑。",
        "文化自信是一个民族最基本的力量。","绿水青山就是金山银山。",
        "人民对美好生活的向往就是我们的奋斗目标。","创新是引领发展的第一动力。",
        "数字经济成为新的增长引擎。","乡村振兴战略全面推进。",
        "高质量发展是时代的要求。","对外开放的基本国策不会改变。",
        "中国式现代化道路越走越宽广。","碳中和目标推动能源转型。",
        "中医药传承创新发展。","传统文化的创造性转化和创新性发展。",
        "国家治理体系和治理能力现代化。","构建人类命运共同体。",
        "一带一路倡议促进共同发展。","太空探索取得重大突破。",
    ]
    for t in chinese: T.append((t, "chinese"))
    philosophy = [
        "I think, therefore I am.", "The unexamined life is not worth living.",
        "To be or not to be, that is the question.", "Knowledge is power.",
        "The only true wisdom is in knowing you know nothing.", "Existence precedes essence.",
        "Man is condemned to be free.", "The categorical imperative demands universal maxims.",
        "All perception is theory-laden.", "Truth is correspondence between thought and reality.",
        "Utilitarianism maximizes happiness.", "The social contract governs political legitimacy.",
        "Phenomenology studies structures of experience.", "Hermeneutics is the art of interpretation.",
        "Nihilism denies inherent meaning.", "Pragmatism judges truth by practical consequences.",
        "Rationalism prioritizes reason as source of knowledge.", "Empiricism derives knowledge from experience.",
        "The mind-body problem asks how mental relates to physical.", "Free will debates determinism versus agency.",
    ]
    for t in philosophy: T.append((t, "philosophy"))
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
    """Precompute hidden states for all texts and all layers."""
    data = {}
    for i, (text, cat) in enumerate(texts):
        logits, h_all = clean_forward(model, tokenizer, text)
        data[i] = {"text": text, "cat": cat, "logits": logits, "h": h_all}
        if (i + 1) % 20 == 0:
            log(f"  Precomputed {i+1}/{len(texts)} texts")
    return data


# =====================================================================
# P156: Residual Flow Curvature Measurement
# =====================================================================
def p156_curvature_measurement(data, n_layers, d_model):
    """
    P156: Measure local curvature of the residual flow path.
    
    Curvature definition: For a path h_0, h_1, ..., h_N:
    - Secant curvature at layer l: K_l = ||delta_l - delta_{l-1}|| / ||delta_{l-1}||
      where delta_l = h_{l+1} - h_l (the residual step)
    - Menger curvature (3-point): K_l = 2*area(h_{l-1}, h_l, h_{l+1}) / (||h_l-h_{l-1}|| * ||h_{l+1}-h_l|| * ||h_{l+1}-h_{l-1}||)
    - Geodesic vs Euclidean distance ratio: R_l = ||h_{l+2}-h_l|| / (||h_{l+1}-h_l|| + ||h_{l+2}-h_{l+1}||)
    """
    log("\n" + "="*80)
    log("P156: Residual Flow Curvature Measurement")
    log("="*80)
    
    # Curvature metrics per layer
    secant_curv = defaultdict(list)  # secant curvature
    menger_curv = defaultdict(list)   # menger curvature  
    path_ratio = defaultdict(list)    # geodesic/euclidean ratio
    step_angle = defaultdict(list)    # angle between consecutive steps
    
    for idx in range(min(100, len(data))):
        h = data[idx]["h"]
        
        for l in range(1, n_layers - 1):
            h_prev = h[l - 1]
            h_curr = h[l]
            h_next = h[l + 1]
            
            delta_prev = h_curr - h_prev  # step l-1 -> l
            delta_curr = h_next - h_curr  # step l -> l+1
            
            # 1. Secant curvature: how much does direction change?
            norm_prev = delta_prev.norm().item()
            norm_curr = delta_curr.norm().item()
            if norm_prev > 1e-8 and norm_curr > 1e-8:
                secant = torch.norm(delta_curr - delta_prev).item() / norm_prev
                secant_curv[l].append(secant)
                
                # 2. Angle between consecutive steps
                cos_ang = cos_sim(delta_prev, delta_curr)
                angle = math.acos(max(-1, min(1, cos_ang)))
                step_angle[l].append(math.degrees(angle))
            
            # 3. Menger curvature (area of triangle / product of sides)
            side_a = (h_curr - h_prev).norm().item()
            side_b = (h_next - h_curr).norm().item()
            side_c = (h_next - h_prev).norm().item()
            
            if side_a > 1e-8 and side_b > 1e-8 and side_c > 1e-8:
                # Heron's formula for area
                s = (side_a + side_b + side_c) / 2
                area_sq = s * (s - side_a) * (s - side_b) * (s - side_c)
                area = math.sqrt(max(0, area_sq))
                menger = 2 * area / (side_a * side_b * side_c)
                menger_curv[l].append(menger)
                
                # 4. Path ratio (geodesic shortcut vs path length)
                ratio = side_c / (side_a + side_b) if (side_a + side_b) > 1e-8 else 0
                path_ratio[l].append(ratio)
    
    # Aggregate results
    log("\nP156 Results: Residual Flow Curvature")
    log("-" * 70)
    log(f"{'Layer':>6} | {'Secant K':>10} | {'Menger K':>10} | {'Path R':>10} | {'Step Angle':>11} | {'N texts':>7}")
    log("-" * 70)
    
    for l in sorted(secant_curv.keys()):
        sc = np.mean(secant_curv[l])
        mc = np.mean(menger_curv[l]) if l in menger_curv else 0
        pr = np.mean(path_ratio[l]) if l in path_ratio else 0
        sa = np.mean(step_angle[l]) if l in step_angle else 0
        n = len(secant_curv[l])
        log(f"L{l:>3d}   | {sc:>10.4f} | {mc:>10.6f} | {pr:>10.4f} | {sa:>10.2f} deg | {n:>7d}")
    
    # Key findings
    log("\nP156 Key Findings:")
    all_secant = [v for vals in secant_curv.values() for v in vals]
    all_menger = [v for vals in menger_curv.values() for v in vals]
    all_angle = [v for vals in step_angle.values() for v in vals]
    all_ratio = [v for vals in path_ratio.values() for v in vals]
    
    log(f"  INV-482: Mean secant curvature = {np.mean(all_secant):.4f} (range: {np.min(all_secant):.4f} ~ {np.max(all_secant):.4f})")
    log(f"  INV-483: Mean Menger curvature = {np.mean(all_menger):.6f} (range: {np.min(all_menger):.6f} ~ {np.max(all_menger):.6f})")
    log(f"  INV-484: Mean step angle = {np.mean(all_angle):.2f} deg (range: {np.min(all_angle):.2f} ~ {np.max(all_angle):.2f} deg)")
    log(f"  INV-485: Mean path ratio = {np.mean(all_ratio):.4f} (1.0=straight line, <1.0=curved)")
    
    # Layer profile: early vs mid vs late
    early_l = [l for l in range(1, n_layers // 3) if l in secant_curv]
    mid_l = [l for l in range(n_layers // 3, 2 * n_layers // 3) if l in secant_curv]
    late_l = [l for l in range(2 * n_layers // 3, n_layers - 1) if l in secant_curv]
    
    for name, layers in [("Early", early_l), ("Mid", mid_l), ("Late", late_l)]:
        if layers:
            sc_vals = [v for l in layers for v in secant_curv[l]]
            ang_vals = [v for l in layers for v in step_angle.get(l, [])]
            pr_vals = [v for l in layers for v in path_ratio.get(l, [])]
            log(f"  {name} layers: secant={np.mean(sc_vals):.4f}, angle={np.mean(ang_vals):.2f}deg, ratio={np.mean(pr_vals):.4f}")
    
    return {
        "secant_curvature": {str(k): float(np.mean(v)) for k, v in secant_curv.items()},
        "menger_curvature": {str(k): float(np.mean(v)) for k, v in menger_curv.items()},
        "path_ratio": {str(k): float(np.mean(v)) for k, v in path_ratio.items()},
        "step_angle": {str(k): float(np.mean(v)) for k, v in step_angle.items()},
        "INV-482": float(np.mean(all_secant)),
        "INV-483": float(np.mean(all_menger)),
        "INV-484": float(np.mean(all_angle)),
        "INV-485": float(np.mean(all_ratio)),
    }


# =====================================================================
# P157: Residual Correction Direction Analysis
# =====================================================================
def p157_correction_direction(data, n_layers, d_model, uw, ub):
    """
    P157: Analyze the direction of the residual correction delta_h = h_{l+1} - h_l
    relative to h_l.
    
    Decompose delta_h into:
    - Parallel component: proj_h(delta_h) = (delta_h . h_l / ||h_l||^2) * h_l
    - Orthogonal component: delta_h - proj_h(delta_h)
    
    Measure: ||parallel|| / ||delta_h|| (fraction parallel to h_l)
    Also: cos(delta_h, h_l) and cos(delta_h, logits_final)
    """
    log("\n" + "="*80)
    log("P157: Residual Correction Direction Analysis")
    log("="*80)
    
    frac_parallel = defaultdict(list)
    cos_delta_h = defaultdict(list)
    cos_delta_logits = defaultdict(list)
    delta_norm_ratio = defaultdict(list)  # ||delta|| / ||h||
    
    for idx in range(min(100, len(data))):
        h = data[idx]["h"]
        logits = data[idx]["logits"]
        
        for l in range(n_layers - 1):
            h_curr = h[l]
            h_next = h[l + 1]
            delta = h_next - h_curr
            
            h_norm = h_curr.norm().item()
            d_norm = delta.norm().item()
            
            if h_norm > 1e-8 and d_norm > 1e-8:
                # Fraction parallel
                dot = torch.dot(delta, h_curr).item()
                proj_len = abs(dot) / h_norm
                frac_p = proj_len / d_norm
                frac_parallel[l].append(frac_p)
                
                # cos(delta_h, h_l)
                cos_dh = dot / (d_norm * h_norm)
                cos_delta_h[l].append(cos_dh)
                
                # cos(delta_h @ W^T, logits)
                if uw is not None:
                    delta_logits = delta @ uw.T
                    if ub is not None:
                        delta_logits = delta_logits + ub
                    cos_dl = cos_sim(delta_logits, logits)
                    cos_delta_logits[l].append(cos_dl)
                
                # Norm ratio
                delta_norm_ratio[l].append(d_norm / h_norm if h_norm > 1e-8 else 0)
    
    log("\nP157 Results: Correction Direction Decomposition")
    log("-" * 70)
    log(f"{'Layer':>6} | {'Frac Para':>10} | {'cos(dh,h)':>10} | {'cos(dh,L)':>10} | {'||d||/||h||':>12}")
    log("-" * 70)
    
    for l in sorted(frac_parallel.keys()):
        fp = np.mean(frac_parallel[l])
        cdh = np.mean(cos_delta_h[l])
        cdl = np.mean(cos_delta_logits[l]) if l in cos_delta_logits else 0
        nr = np.mean(delta_norm_ratio[l])
        log(f"L{l:>3d}   | {fp:>10.4f} | {cdh:>10.4f} | {cdl:>10.4f} | {nr:>12.4f}")
    
    log("\nP157 Key Findings:")
    all_frac = [v for vals in frac_parallel.values() for v in vals]
    all_cdh = [v for vals in cos_delta_h.values() for v in vals]
    all_cdl = [v for vals in cos_delta_logits.values() for v in vals]
    
    log(f"  INV-486: Mean parallel fraction = {np.mean(all_frac):.4f}")
    log(f"  INV-487: Mean cos(delta_h, h_l) = {np.mean(all_cdh):.4f}")
    log(f"  INV-488: Mean cos(delta_h@W^T, logits) = {np.mean(all_cdl):.4f}")
    
    # Check: is delta mostly orthogonal to h?
    n_orth = sum(1 for v in all_frac if v < 0.3)  # mostly orthogonal
    n_para = sum(1 for v in all_frac if v > 0.7)   # mostly parallel
    log(f"  Distribution: {n_orth}/{len(all_frac)} orthogonal(frac<0.3), {n_para}/{len(all_frac)} parallel(frac>0.7)")
    
    return {
        "frac_parallel": {str(k): float(np.mean(v)) for k, v in frac_parallel.items()},
        "cos_delta_h": {str(k): float(np.mean(v)) for k, v in cos_delta_h.items()},
        "cos_delta_logits": {str(k): float(np.mean(v)) for k, v in cos_delta_logits.items()},
        "norm_ratio": {str(k): float(np.mean(v)) for k, v in delta_norm_ratio.items()},
        "INV-486": float(np.mean(all_frac)),
        "INV-487": float(np.mean(all_cdh)),
        "INV-488": float(np.mean(all_cdl)),
    }


# =====================================================================
# P158: Cross-Model Manifold Isomorphism
# =====================================================================
def p158_cross_model_isomorphism(data, n_layers, d_model):
    """
    P158: Test if different models' residual flows are isomorphic.
    
    Method: For each text, compare h_l across models using:
    1. CKA (Centered Kernel Alignment) - a standard representation similarity measure
    2. Procrustes distance: min||A*h1 - h2||_F over orthogonal A
    3. Simplified: just measure cos similarity of the delta_h direction (normalized)
    
    Since models have different d_model, we use the delta_h direction
    (which is in the same "semantic space" if the manifold is isomorphic).
    """
    log("\n" + "="*80)
    log("P158: Cross-Model Manifold Isomorphism Analysis")
    log("="*80)
    
    # For each text, compute the residual flow "direction profile"
    # delta direction at each layer (normalized)
    log("  Computing residual flow direction profiles...")
    
    # This experiment requires multiple models' data - we'll compute
    # intra-model consistency instead (since we only have one model per run)
    # But we CAN do: does the direction profile of delta_h correlate with text category?
    
    category_profiles = defaultdict(lambda: defaultdict(list))
    
    for idx in range(min(100, len(data))):
        h = data[idx]["h"]
        cat = data[idx]["cat"]
        
        for l in range(n_layers - 1):
            delta = h[l + 1] - h[l]
            norm = delta.norm().item()
            if norm > 1e-8:
                # Normalize delta direction
                delta_dir = delta / norm
                # Measure cos of delta direction with category centroid
                category_profiles[cat][l].append(delta_dir)
    
    # Compute inter-category divergence
    categories = sorted(category_profiles.keys())
    log(f"\n  Categories: {categories}")
    
    # For each layer, compute centroid of each category's delta direction
    cat_centroids = {}
    for cat in categories:
        cat_centroids[cat] = {}
        for l in category_profiles[cat]:
            deltas = torch.stack(category_profiles[cat][l])
            centroid = deltas.mean(dim=0)
            centroid_norm = centroid.norm()
            if centroid_norm > 1e-8:
                cat_centroids[cat][l] = centroid / centroid_norm
    
    # Measure inter-category similarity at each layer
    log(f"\nP158 Results: Category-Specific Flow Direction Analysis")
    log("-" * 70)
    
    # Pick representative layers
    rep_layers = sorted(set.intersection(*[set(cat_centroids[c].keys()) for c in categories]))
    sample_layers = rep_layers[::max(1, len(rep_layers)//5)]  # ~5 layers
    
    inter_cat_cos = defaultdict(list)
    
    for l in sample_layers:
        cos_matrix = {}
        for i, c1 in enumerate(categories):
            for c2 in categories[i+1:]:
                cos_val = cos_sim(cat_centroids[c1][l], cat_centroids[c2][l])
                cos_matrix[f"{c1[:4]}-{c2[:4]}"] = cos_val
                inter_cat_cos[l].append(cos_val)
        
        avg_cos = np.mean(list(cos_matrix.values()))
        log(f"  L{l}: avg inter-cat cos = {avg_cos:.4f}", )
    
    log(f"\nP158 Key Findings:")
    all_inter = [v for vals in inter_cat_cos.values() for v in vals]
    log(f"  INV-489: Mean inter-category cos = {np.mean(all_inter):.4f} (range: {np.min(all_inter):.4f} ~ {np.max(all_inter):.4f})")
    
    if np.mean(all_inter) > 0.9:
        log("  → Delta directions are category-INDEPENDENT (flow structure is universal)")
    elif np.mean(all_inter) > 0.5:
        log("  → Delta directions are partially category-dependent")
    else:
        log("  → Delta directions are category-DEPENDENT (flow adapts to content)")
    
    # Also measure intra-category consistency
    intra_cat_cos = defaultdict(list)
    for cat in categories:
        for l in sample_layers:
            if l in cat_centroids[cat]:
                deltas = torch.stack(category_profiles[cat][l])
                n = min(20, len(deltas))
                sample_idx = torch.randperm(len(deltas))[:n]
                pairs = 0
                total_cos = 0
                for i in range(min(n, 5)):
                    for j in range(i+1, min(n, 5)):
                        total_cos += cos_sim(deltas[sample_idx[i]], deltas[sample_idx[j]])
                        pairs += 1
                if pairs > 0:
                    intra_cat_cos[cat].append(total_cos / pairs)
    
    log(f"\n  Intra-category consistency (avg cos within category):")
    for cat in categories:
        if intra_cat_cos[cat]:
            log(f"    {cat}: {np.mean(intra_cat_cos[cat]):.4f}")
    
    return {
        "inter_cat_cos": {str(k): float(np.mean(v)) for k, v in inter_cat_cos.items()},
        "INV-489": float(np.mean(all_inter)),
        "intra_cat_cos": {k: float(np.mean(v)) for k, v in intra_cat_cos.items()},
    }


# =====================================================================
# P159: Manifold vs Euclidean Manipulation
# =====================================================================
def p159_manifold_vs_euclidean(model, tokenizer, data, uw, ub, n_layers, d_model):
    """
    P159: Compare manipulation along manifold tangent vs random direction.
    
    For each text at layer N//2:
    - (A) Tangent manipulation: perturb h along delta_{N//2} direction (the "natural" flow direction)
    - (B) Random manipulation: perturb h along random direction
    - (C) Orthogonal manipulation: perturb h along direction orthogonal to delta
    - Measure: KL after propagating through remaining layers
    
    Hypothesis: tangent direction perturbation causes less KL (more "natural")
    """
    log("\n" + "="*80)
    log("P159: Manifold vs Euclidean Manipulation")
    log("="*80)
    
    # Pick a middle layer for manipulation
    mid_layer = n_layers // 2
    last_layer = n_layers - 1
    target_layer = n_layers - 2  # second to last
    log(f"  Manipulation at L{mid_layer}, measuring effect at output")
    
    n_texts = min(30, len(data))
    scales = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0]
    
    tangent_kls = defaultdict(list)
    random_kls = defaultdict(list)
    orthogonal_kls = defaultdict(list)
    
    layers_to_test = [0, mid_layer // 2, mid_layer, min(3 * mid_layer // 2, n_layers - 2)]
    
    for idx in range(n_texts):
        text = data[idx]["text"]
        logits_orig = data[idx]["logits"]
        h = data[idx]["h"]
        probs_orig = F.softmax(logits_orig, dim=-1)
        
        for layer in layers_to_test:
            h_l = h[layer]
            
            # Get tangent direction (delta_h at this layer)
            if layer < n_layers - 1:
                tangent = h[layer + 1] - h[layer]
            else:
                tangent = h[layer] - h[layer - 1]
            
            t_norm = tangent.norm()
            if t_norm < 1e-8:
                continue
            tangent_dir = tangent / t_norm
            
            # Random direction
            random_dir = torch.randn_like(h_l)
            random_dir = random_dir / random_dir.norm()
            
            # Orthogonal to tangent
            proj = torch.dot(random_dir, tangent_dir) * tangent_dir
            orth_dir = random_dir - proj
            orth_norm = orth_dir.norm()
            if orth_norm > 1e-8:
                orth_dir = orth_dir / orth_norm
            else:
                # If random happened to be parallel, try another
                random_dir2 = torch.randn_like(h_l)
                proj2 = torch.dot(random_dir2, tangent_dir) * tangent_dir
                orth_dir = random_dir2 - proj2
                orth_norm = orth_dir.norm()
                if orth_norm > 1e-8:
                    orth_dir = orth_dir / orth_norm
                else:
                    continue
            
            for scale in scales:
                # Manipulate and measure direct KL (not propagated)
                h_tangent = h_l + scale * t_norm * tangent_dir
                h_random = h_l + scale * t_norm * random_dir
                h_orth = h_l + scale * t_norm * orth_dir
                
                if uw is not None:
                    logits_t = h_tangent @ uw.T
                    logits_r = h_random @ uw.T
                    logits_o = h_orth @ uw.T
                    if ub is not None:
                        logits_t = logits_t + ub
                        logits_r = logits_r + ub
                        logits_o = logits_o + ub
                    
                    probs_t = F.softmax(logits_t, dim=-1)
                    probs_r = F.softmax(logits_r, dim=-1)
                    probs_o = F.softmax(logits_o, dim=-1)
                    
                    kl_t = kl_div(probs_orig, probs_t)
                    kl_r = kl_div(probs_orig, probs_r)
                    kl_o = kl_div(probs_orig, probs_o)
                    
                    tangent_kls[(layer, scale)].append(kl_t)
                    random_kls[(layer, scale)].append(kl_r)
                    orthogonal_kls[(layer, scale)].append(kl_o)
    
    log(f"\nP159 Results: Tangent vs Random vs Orthogonal (direct KL)")
    log("-" * 80)
    log(f"{'Layer':>6} | {'Scale':>6} | {'Tangent KL':>10} | {'Random KL':>10} | {'Orth KL':>10} | {'T/R ratio':>10}")
    log("-" * 80)
    
    for layer in layers_to_test:
        for scale in scales:
            key = (layer, scale)
            if key in tangent_kls and tangent_kls[key]:
                kt = np.mean(tangent_kls[key])
                kr = np.mean(random_kls[key])
                ko = np.mean(orthogonal_kls[key])
                ratio = kt / kr if kr > 1e-8 else 0
                log(f"L{layer:>3d}   | {scale:>6.2f} | {kt:>10.4f} | {kr:>10.4f} | {ko:>10.4f} | {ratio:>10.4f}")
    
    log(f"\nP159 Key Findings:")
    # Compare at scale=1.0, mid_layer
    mid_key = (mid_layer, 1.0)
    if mid_key in tangent_kls and tangent_kls[mid_key]:
        kt = np.mean(tangent_kls[mid_key])
        kr = np.mean(random_kls[mid_key])
        ko = np.mean(orthogonal_kls[mid_key])
        log(f"  INV-490: At mid layer, scale=1.0:")
        log(f"    Tangent KL = {kt:.4f}, Random KL = {kr:.4f}, Orth KL = {ko:.4f}")
        log(f"    Tangent/Random ratio = {kt/kr:.4f}" if kr > 1e-8 else "    Random KL ~ 0")
    
    return {
        "tangent_kls": {f"{k[0]}_{k[1]}": float(np.mean(v)) for k, v in tangent_kls.items()},
        "random_kls": {f"{k[0]}_{k[1]}": float(np.mean(v)) for k, v in random_kls.items()},
        "orthogonal_kls": {f"{k[0]}_{k[1]}": float(np.mean(v)) for k, v in orthogonal_kls.items()},
    }


# =====================================================================
# P160: Tangent Space Projection
# =====================================================================
def p160_tangent_space_projection(data, n_layers, d_model, uw, ub):
    """
    P160: Project h_l onto the local tangent space of the manifold,
    then measure how much information is preserved in the tangent space.
    
    The tangent space at layer l is spanned by the "natural" directions:
    delta_{l-1} (direction from l-1 to l) and its top-K PCA directions.
    
    Measure: what fraction of ||h_l||^2 is captured by the tangent space?
    Also: what fraction of logit information is in the tangent space?
    """
    log("\n" + "="*80)
    log("P160: Tangent Space Projection Analysis")
    log("="*80)
    
    tangent_capture = defaultdict(list)  # fraction of norm in tangent space
    logit_capture = defaultdict(list)    # fraction of logit info in tangent space
    
    K_tangent = 10  # number of tangent directions to use
    
    for idx in range(min(50, len(data))):
        h = data[idx]["h"]
        logits = data[idx]["logits"]
        probs_orig = F.softmax(logits, dim=-1)
        
        for l in range(1, n_layers - 1):
            h_l = h[l]
            
            # Build tangent basis from neighboring delta directions
            # Use delta_{l-1} and delta_l as two tangent vectors
            delta_prev = h[l] - h[l - 1]
            delta_next = h[l + 1] - h[l]
            
            # Stack as tangent basis (could also add PCA of nearby h's)
            tangent_basis = torch.stack([delta_prev, delta_next], dim=0)  # 2 x d
            
            # Also add random perturbation directions to expand basis
            if d_model >= K_tangent:
                # Use PCA-like approach: take top singular vectors of local h's
                # Collect h values in a window around l
                window = range(max(0, l - 3), min(n_layers, l + 4))
                H_local = torch.stack([h[ll] for ll in window], dim=0)  # 7 x d
                # Center
                H_centered = H_local - H_local.mean(dim=0, keepdim=True)
                # SVD to get top-K directions
                try:
                    _, S, Vt = torch.linalg.svd(H_centered, full_matrices=False)
                    # Top-K right singular vectors span the local tangent space
                    tangent_basis = Vt[:K_tangent]  # K x d
                except:
                    continue
            
            # Project h_l onto tangent space
            # h_proj = sum_i (h_l . v_i) * v_i  for each tangent vector v_i
            h_proj = torch.zeros_like(h_l)
            for k in range(min(K_tangent, tangent_basis.shape[0])):
                v = tangent_basis[k]
                v_norm = v.norm()
                if v_norm > 1e-8:
                    v = v / v_norm
                    h_proj += torch.dot(h_l, v) * v
            
            # Fraction of norm captured
            h_norm_sq = h_l.norm().item() ** 2
            proj_norm_sq = h_proj.norm().item() ** 2
            capture = proj_norm_sq / h_norm_sq if h_norm_sq > 1e-8 else 0
            tangent_capture[l].append(capture)
            
            # Fraction of logit information captured
            if uw is not None:
                logits_full = h_l @ uw.T + (ub if ub is not None else 0)
                logits_proj = h_proj @ uw.T + (ub if ub is not None else 0)
                
                probs_full = F.softmax(logits_full, dim=-1)
                probs_proj = F.softmax(logits_proj, dim=-1)
                
                kl = kl_div(probs_orig, probs_proj)
                logit_capture[l].append(kl)
    
    log(f"\nP160 Results: Tangent Space Capture (K={K_tangent} directions)")
    log("-" * 70)
    log(f"{'Layer':>6} | {'Norm Capture':>13} | {'Logit KL':>10} | {'N texts':>7}")
    log("-" * 70)
    
    for l in sorted(tangent_capture.keys()):
        nc = np.mean(tangent_capture[l])
        lk = np.mean(logit_capture[l]) if l in logit_capture else 0
        n = len(tangent_capture[l])
        log(f"L{l:>3d}   | {nc:>13.4f} | {lk:>10.4f} | {n:>7d}")
    
    log(f"\nP160 Key Findings:")
    all_capture = [v for vals in tangent_capture.values() for v in vals]
    all_logit_kl = [v for vals in logit_capture.values() for v in vals]
    
    log(f"  INV-491: Mean tangent space norm capture = {np.mean(all_capture):.4f}")
    log(f"  INV-492: Mean tangent space logit KL = {np.mean(all_logit_kl):.4f}")
    
    if np.mean(all_capture) > 0.95:
        log("  → Local tangent space captures >95% of norm: manifold is LOW-dimensional")
    elif np.mean(all_capture) > 0.8:
        log("  → Local tangent space captures 80-95% of norm: moderate dimensionality")
    else:
        log("  → Local tangent space captures <80% of norm: manifold is HIGH-dimensional or non-smooth")
    
    return {
        "tangent_capture": {str(k): float(np.mean(v)) for k, v in tangent_capture.items()},
        "logit_capture": {str(k): float(np.mean(v)) for k, v in logit_capture.items()},
        "INV-491": float(np.mean(all_capture)),
        "INV-492": float(np.mean(all_logit_kl)),
    }


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
    out_dir = _Path(f"tests/glm5_temp/stage728_phase23_{mname}_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    log = Logger(str(out_dir), "results")
    log(f"Phase XXIII: Language Manifold Theory — {mname}")
    log(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    t0 = time.time()
    
    # Load model
    model, tokenizer, n_layers, d_model = load_model(mname)
    uw, ub = get_unembed(model)
    
    # Build texts and precompute hidden states
    texts = build_texts()
    log(f"\nPrecomputing hidden states for {len(texts)} texts, {n_layers} layers...")
    data = precompute_all_hidden(model, tokenizer, texts, n_layers)
    log(f"Precomputation done in {time.time()-t0:.1f}s")
    
    results = {}
    
    # P156: Curvature
    t1 = time.time()
    results["P156"] = p156_curvature_measurement(data, n_layers, d_model)
    log(f"P156 done in {time.time()-t1:.1f}s")
    
    # P157: Correction direction
    t2 = time.time()
    results["P157"] = p157_correction_direction(data, n_layers, d_model, uw, ub)
    log(f"P157 done in {time.time()-t2:.1f}s")
    
    # P158: Cross-model isomorphism (intra-model category analysis)
    t3 = time.time()
    results["P158"] = p158_cross_model_isomorphism(data, n_layers, d_model)
    log(f"P158 done in {time.time()-t3:.1f}s")
    
    # P159: Manifold vs Euclidean
    t4 = time.time()
    results["P159"] = p159_manifold_vs_euclidean(model, tokenizer, data, uw, ub, n_layers, d_model)
    log(f"P159 done in {time.time()-t4:.1f}s")
    
    # P160: Tangent space projection
    t5 = time.time()
    results["P160"] = p160_tangent_space_projection(data, n_layers, d_model, uw, ub)
    log(f"P160 done in {time.time()-t5:.1f}s")
    
    # Save results
    total_time = time.time() - t0
    results["meta"] = {"model": mname, "n_layers": n_layers, "d_model": d_model,
                       "n_texts": len(texts), "total_time_min": round(total_time/60, 1)}
    
    # Convert numpy to float for JSON
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
    
    # Free GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    
    log.close()
    return str(out_dir)

if __name__ == "__main__":
    main()
