"""
Phase LXXXVI-P423/424/425/426: 多层累积放大精确理论与V_lang完备性
================================================================================

阶段I核心任务 - 链式Jacobian传播 + V_lang完备性定理 + GLM4信号来源 + 语言空间几何

P423: 多层累积放大精确公式——链式Jacobian传播
  - P419发现: Jacobian仅解释1-27%实际增益
  - 核心问题: J_gain^L的修正因子是什么? 为什么1.7^36>>170?
  - 方法:
    1. 逐层计算Jacobian: J_l = d(output_l)/d(input_l)
    2. 链式传播: J_total = J_L * J_{L-1} * ... * J_1
    3. 但J_total不是简单乘积! 因为每层的Jacobian依赖于x_l(输入相关)
    4. 测量: 逐层注入信号, 计算每层的实际增益增量
    5. 比较: 链式J_gain预测 vs 实际增益

P424: V_lang完备性定理严格证明
  - P422发现: Participation ratio=功能维度数, V_lang/d_model=3-5%
  - 核心问题: V_lang的上界是什么? 与d_model的关系?
  - 方法:
    1. 从W_lm SVD谱推导: V_lang = PR(W_lm) = (tr(S^2))^2 / tr(S^4)
    2. 证明: PR ≤ rank(W_lm) ≤ min(vocab, d_model)
    3. 关键: PR ≈ 功能维度数 = V_lang_eff
    4. 推导: V_lang ≤ C * d_model, C由SVD谱决定
    5. 验证: 三模型的PR vs 功能维度数 vs d_model

P425: GLM4信号放大的真正来源
  - P419发现: GLM4 MLP Jacobian=0.13x, 但actual_gain=99.7
  - 核心问题: GLM4的信号放大来自哪个组件?
  - 方法:
    1. 逐层追踪: 在每层注入信号, 测量下游各层的输出变化
    2. 分离Attention/MLP/LayerNorm/残差各自的贡献
    3. 特别关注: LayerNorm的增益(方差归一化可能放大信号)
    4. 测量: 每个组件的Jacobian增益

P426: 语言空间几何——V_lang作为hidden_dim子空间
  - 核心问题: V_lang子空间的几何结构是什么?
  - 方法:
    1. 用功能维度方向构造V_lang子空间
    2. 分析子空间的曲率、体积、边界
    3. 计算V_lang子空间与随机子空间的重叠度
    4. 分析信号在V_lang子空间中的传播特征

实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免GPU OOM)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIM_PAIRS = {
    "style": [("formal", "informal")],
    "logic": [("true", "false")],
    "grammar": [("active", "passive")],
    "sentiment": [("happy", "sad")],
    "tense": [("was", "is")],
    "certainty": [("definitely", "maybe")],
    "quantity": [("many", "few")],
}

MODEL_CONFIGS = {
    "qwen3": {
        "path": r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c",
        "trust_remote_code": True, "use_fast": False,
    },
    "glm4": {
        "path": r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf",
        "trust_remote_code": True, "use_fast": False,
    },
    "deepseek7b": {
        "path": r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60",
        "trust_remote_code": True, "use_fast": False,
    },
}

PROMPTS = [
    "The apple is",
    "In the future, people will",
    "The scientist explained that",
    "When the rain stopped,",
    "She looked at the sky and",
    "The old man walked slowly",
]


def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = MODEL_CONFIGS[model_name]
    print(f"Loading {model_name}...")
    mdl = AutoModelForCausalLM.from_pretrained(
        cfg["path"], dtype=torch.bfloat16, trust_remote_code=cfg["trust_remote_code"],
        local_files_only=True, low_cpu_mem_usage=True, attn_implementation="eager", device_map="cpu",
    )
    if torch.cuda.is_available():
        mdl = mdl.to("cuda")
    mdl.eval()
    device = mdl.device
    tok = AutoTokenizer.from_pretrained(
        cfg["path"], trust_remote_code=cfg["trust_remote_code"],
        local_files_only=True, use_fast=cfg["use_fast"],
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return mdl, tok, device


def get_layers(model):
    if hasattr(model.model, "layers"):
        return list(model.model.layers)
    elif hasattr(model.model, "encoder"):
        return list(model.model.encoder.layers)
    raise ValueError("Cannot find layers")


def get_w_lm(model, tokenizer, word):
    tok_ids = tokenizer.encode(word, add_special_tokens=False)
    tok_id = tok_ids[0]
    lm_head = model.lm_head
    w = lm_head.weight[tok_id].detach().cpu().float()
    w_norm = w / w.norm()
    return w_norm.numpy(), w.numpy()


def get_dimension_direction(model, tokenizer, word_pos, word_neg):
    w_pos, _ = get_w_lm(model, tokenizer, word_pos)
    w_neg, _ = get_w_lm(model, tokenizer, word_neg)
    diff = w_pos - w_neg
    norm = np.linalg.norm(diff)
    if norm < 1e-8:
        return w_pos, 0.0
    return diff / norm, norm


# ========== P423: 多层累积放大精确公式 ==========

def run_p423(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P423: Layer-by-layer Signal Propagation - {model_name}")
    print(f"{'='*60}")
    
    layers = get_layers(model)
    n_layers = len(layers)
    embed = model.get_input_embeddings()
    
    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    
    # 测试4个维度
    test_dims = {
        "style": ("formal", "informal"),
        "logic": ("true", "false"),
        "sentiment": ("happy", "sad"),
        "tense": ("was", "is"),
    }
    
    beta = 8.0
    results = {"model": model_name, "exp": "p423"}
    
    # 先获取基线residual stream
    # 用hook收集每层的residual
    baseline_resids = {}
    
    def make_resid_hook(layer_idx):
        def hook_fn(module, input, output):
            # input[0]是layer的输入 = residual stream
            baseline_resids[layer_idx] = input[0].detach().cpu().float()
        return hook_fn
    
    hooks = []
    for l in range(n_layers):
        h = layers[l].register_forward_hook(make_resid_hook(l))
        hooks.append(h)
    
    with torch.no_grad():
        baseline_output = model(input_ids)
        baseline_logits = baseline_output.logits[0, -1].detach().cpu().float()
    
    for h in hooks:
        h.remove()
    
    print(f"  Collected baseline residuals from {len(baseline_resids)} layers")
    
    # 逐层注入信号, 测量每层后的增益
    for dim_name, (w1, w2) in test_dims.items():
        direction, _ = get_dimension_direction(model, tokenizer, w1, w2)
        direction_t = torch.tensor(direction * beta, dtype=torch.float32)
        
        print(f"\n  Dim: {dim_name} ({w1}/{w2})")
        dim_results = {}
        
        # 在每层注入, 测量下游logit变化
        cumulative_gains = []
        
        # 选择关键层测试 (避免太慢)
        test_layers = list(range(0, n_layers, max(1, n_layers // 10)))
        if n_layers - 1 not in test_layers:
            test_layers.append(n_layers - 1)
        
        for l in test_layers:
            # 在层l的输入处注入信号
            def inj_hook(module, input, output, d=direction_t, li=l):
                modified = output.clone()
                # 在最后一个token位置注入到residual
                # 但这里hook在layer输出上, 需要在residual stream中注入
                # 简化: 在embed输出注入, 但只计算到层l之后的影响
                return modified
            
            # 更精确的方法: 用embed hook注入, 但逐层测量
            # 我们直接在embed层注入, 然后用hook收集各层的residual
            intervened_resids = {}
            
            def make_intervened_hook(layer_idx):
                def hook_fn(module, input, output):
                    intervened_resids[layer_idx] = input[0].detach().cpu().float()
                return hook_fn
            
            hooks2 = []
            for ll in range(n_layers):
                h = layers[ll].register_forward_hook(make_intervened_hook(ll))
                hooks2.append(h)
            
            # 在embed注入
            def embed_inj_hook(module, input, output, d=direction_t):
                modified = output.clone()
                modified[0, -1, :] += d.to(output.device)
                return modified
            
            h_embed = embed.register_forward_hook(embed_inj_hook)
            
            with torch.no_grad():
                intervened_output = model(input_ids)
                intervened_logits = intervened_output.logits[0, -1].detach().cpu().float()
            
            h_embed.remove()
            for h in hooks2:
                h.remove()
            
            # 计算logit变化
            delta_logit = (intervened_logits - baseline_logits).norm().item()
            gain = delta_logit / beta
            
            # 计算每层的residual变化
            layer_deltas = {}
            for ll in range(n_layers):
                if ll in baseline_resids and ll in intervened_resids:
                    dh = (intervened_resids[ll][0, -1] - baseline_resids[ll][0, -1])
                    dh_norm = dh.norm().item()
                    layer_deltas[ll] = round(dh_norm, 4)
            
            cumulative_gains.append({
                "inject_at": "embed",
                "delta_logit": round(delta_logit, 4),
                "gain": round(gain, 4),
                "layer_deltas": layer_deltas,
            })
        
        # 分析增益增长模式
        # 关键: 信号在residual stream中的增长模式
        if cumulative_gains:
            # 从layer_deltas分析逐层增长
            all_layer_norms = {}
            for cg in cumulative_gains:
                for ll, dn in cg["layer_deltas"].items():
                    if ll not in all_layer_norms:
                        all_layer_norms[ll] = []
                    all_layer_norms[ll].append(dn)
            
            # 平均各层的dh_norm
            avg_layer_norms = {}
            for ll, norms in sorted(all_layer_norms.items()):
                avg_layer_norms[ll] = round(np.mean(norms), 4)
            
            # 计算逐层增长率
            growth_rates = {}
            sorted_layers = sorted(avg_layer_norms.keys())
            for i in range(1, len(sorted_layers)):
                l_prev = sorted_layers[i-1]
                l_curr = sorted_layers[i]
                if avg_layer_norms[l_prev] > 1e-10:
                    rate = avg_layer_norms[l_curr] / avg_layer_norms[l_prev]
                    growth_rates[f"L{l_prev}->L{l_curr}"] = round(rate, 4)
            
            dim_results["avg_layer_norms"] = avg_layer_norms
            dim_results["growth_rates"] = growth_rates
            dim_results["final_gain"] = cumulative_gains[-1]["gain"] if cumulative_gains else 0
            
            # 打印关键层
            print(f"    Layer norms: ", end="")
            for ll in sorted_layers[::max(1, len(sorted_layers)//5)]:
                print(f"L{ll}={avg_layer_norms[ll]:.1f}", end=" ")
            print()
            print(f"    Final gain: {dim_results['final_gain']:.2f}")
        
        results[dim_name] = dim_results
    
    # 额外实验: 逐层注入, 测量单层增益
    print(f"\n  === Per-layer injection experiment ===")
    
    # 对logic维度, 在每层注入信号
    dim_name = "logic"
    w1, w2 = test_dims[dim_name]
    direction, _ = get_dimension_direction(model, tokenizer, w1, w2)
    direction_t = torch.tensor(direction * beta, dtype=torch.float32)
    
    per_layer_gains = {}
    # 选择每4层测试一次 (节省时间)
    sample_layers = list(range(0, n_layers, max(1, n_layers // 8)))
    
    for l in sample_layers:
        # 在层l的输入处注入
        def make_inj_at_layer(layer_idx):
            def hook_fn(module, input, output, d=direction_t):
                modified = output.clone()
                # 在residual stream中注入: output = layer_output, 
                # 但我们想在层输入处注入, 需要用pre-hook
                return modified
            return hook_fn
        
        # 简化: 在层l的residual_pre注入
        # 用register_forward_pre_hook
        def pre_inj_hook(module, args, d=direction_t):
            # args是tuple, 第一个是input tensor
            if isinstance(args, tuple) and len(args) > 0:
                modified = args[0].clone()
                modified[0, -1, :] += d.to(args[0].device)
                return (modified,) + args[1:]
            return args
        
        h = layers[l].register_forward_pre_hook(pre_inj_hook)
        
        with torch.no_grad():
            output = model(input_ids)
            logits = output.logits[0, -1].detach().cpu().float()
        
        h.remove()
        
        delta = (logits - baseline_logits).norm().item()
        gain = delta / beta
        
        per_layer_gains[str(l)] = round(gain, 4)
        print(f"    Inject at L{l}: gain={gain:.2f}")
    
    results["per_layer_injection_gains"] = per_layer_gains
    
    # 分析: 增益随注入层的变化
    gains_list = [(int(k), v) for k, v in per_layer_gains.items()]
    gains_list.sort()
    
    if len(gains_list) > 1:
        # 拟合: gain(l) = A * exp(alpha * l) 或 gain(l) = A * l^beta
        ls = np.array([g[0] for g in gains_list])
        gs = np.array([g[1] for g in gains_list])
        
        # 对数拟合
        gs_log = np.log(gs + 1e-10)
        if len(ls) > 2:
            # 线性拟合: log(gain) = a + b*l
            coeffs = np.polyfit(ls, gs_log, 1)
            b, a = coeffs
            exponential_rate = b
            
            results["exponential_fit"] = {
                "rate_per_layer": round(float(b), 4),
                "intercept": round(float(a), 4),
                "doubling_layers": round(float(np.log(2) / b), 2) if b > 0 else float('inf'),
            }
            print(f"\n  Exponential fit: gain = exp({a:.2f} + {b:.4f}*l)")
            print(f"  Doubling every {np.log(2)/b:.1f} layers" if b > 0 else "  No growth")
    
    # 总结
    print(f"\n  === P423 Summary for {model_name} ===")
    for dim_name, dim_data in results.items():
        if dim_name in ["per_layer_injection_gains", "exponential_fit"]:
            continue
        if isinstance(dim_data, dict) and "final_gain" in dim_data:
            print(f"  {dim_name}: final_gain={dim_data['final_gain']:.2f}")
    if "exponential_fit" in results:
        ef = results["exponential_fit"]
        print(f"  Exponential rate: {ef['rate_per_layer']:.4f}/layer")
        print(f"  Doubling: every {ef['doubling_layers']:.1f} layers")
    
    ts = time.strftime("%Y%m%d_%H%M")
    out_file = OUT_DIR / f"phase_lxxxvi_p423_{model_name}_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved to {out_file}")
    return results


# ========== P424: V_lang完备性定理严格证明 ==========

def run_p424(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P424: V_lang Completeness Theorem - {model_name}")
    print(f"{'='*60}")
    
    W_U = model.lm_head.weight.detach().cpu().float().numpy()
    vocab_size, d_model = W_U.shape
    print(f"  W_lm: {vocab_size} x {d_model}")
    
    results = {"model": model_name, "exp": "p424", "vocab_size": vocab_size, "d_model": d_model}
    
    # 1. SVD (用随机SVD避免内存问题)
    print("  Computing SVD...")
    from sklearn.utils.extmath import randomized_svd
    n_components = min(d_model, 500)
    if vocab_size * d_model > 500_000_000:
        U, S, Vt = randomized_svd(W_U, n_components=n_components, random_state=42)
    else:
        U, S, Vt = np.linalg.svd(W_U, full_matrices=False)
    
    # 2. Participation ratio (精确计算)
    S_sq = S**2
    PR = (S_sq.sum())**2 / (S_sq**2).sum()
    results["participation_ratio"] = round(float(PR), 2)
    print(f"  Participation ratio: {PR:.2f}")
    
    # 3. PR与功能维度数的关系
    # 收集功能维度
    embed = model.get_input_embeddings()
    beta = 8.0
    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    
    with torch.no_grad():
        baseline_logits = model(input_ids).logits[0, -1].detach().cpu().float()
    
    # 扩展维度列表
    all_dim_pairs = {
        "style": ("formal", "informal"), "logic": ("true", "false"),
        "grammar": ("active", "passive"), "sentiment": ("happy", "sad"),
        "tense": ("was", "is"), "certainty": ("definitely", "maybe"),
        "quantity": ("many", "few"), "complexity": ("complex", "simple"),
        "formality": ("professional", "casual"), "size": ("large", "small"),
        "strength": ("strong", "weak"), "beauty": ("beautiful", "ugly"),
        "truth": ("truth", "lie"), "freedom": ("free", "bound"),
        "wisdom": ("wise", "foolish"), "courage": ("brave", "cowardly"),
        "peace": ("peace", "war"), "love": ("love", "hate"),
        "depth": ("deep", "shallow"), "width": ("wide", "narrow"),
        "height": ("tall", "short"), "sharpness": ("sharp", "dull"),
        "health": ("healthy", "sick"), "order": ("ordered", "chaotic"),
        "wealth": ("rich", "poor"), "power": ("powerful", "helpless"),
        "safety": ("safe", "dangerous"), "honesty": ("honest", "deceitful"),
        "humor": ("funny", "serious"), "novelty": ("novel", "familiar"),
        "anger": ("angry", "calm"), "fear": ("afraid", "confident"),
        "joy": ("joyful", "sorrowful"), "trust_dim": ("trust", "distrust"),
        "surprise": ("surprising", "expected"), "color": ("colorful", "colorless"),
        "motion": ("moving", "still"), "growth": ("growing", "shrinking"),
        "change": ("changing", "stable"), "balance": ("balanced", "unbalanced"),
        "logic_dim": ("logical", "illogical"), "reason": ("reasonable", "unreasonable"),
        "fact": ("factual", "fictional"), "reality": ("real", "imaginary"),
        "possibility": ("possible", "impossible"), "precision": ("precise", "approximate"),
        "consistency": ("consistent", "inconsistent"), "simplicity": ("simple", "complex2"),
        "flexibility": ("flexible", "rigid"), "efficiency": ("efficient", "wasteful"),
        "creativity": ("creative", "conventional"), "intelligence": ("intelligent", "foolish2"),
        "knowledge": ("knowledgeable", "ignorant"), "art": ("artistic", "technical"),
        "science": ("scientific", "intuitive"), "consciousness": ("conscious", "unconscious"),
        "memory_dim": ("memorable", "forgettable"), "imagination": ("imaginative", "literal"),
        "hope": ("hopeful", "hopeless"), "faith": ("faithful", "faithless"),
        "curiosity": ("curious", "indifferent"), "justice": ("just", "unjust"),
        "fairness": ("fair", "unfair"), "empathy": ("empathetic", "apathetic"),
        "kindness": ("kind", "cruel"), "generosity": ("generous", "stingy"),
    }
    
    functional_dirs = []
    functional_dlogits = []
    
    for dim_name, (w1, w2) in all_dim_pairs.items():
        direction, norm = get_dimension_direction(model, tokenizer, w1, w2)
        if norm < 1e-8:
            continue
        
        direction_t = torch.tensor(direction * beta, dtype=torch.float32)
        
        def inj_hook(module, input, output, d=direction_t):
            modified = output.clone()
            modified[0, -1, :] += d.to(output.device)
            return modified
        
        h = embed.register_forward_hook(inj_hook)
        with torch.no_grad():
            intervened_logits = model(input_ids).logits[0, -1].detach().cpu().float()
        h.remove()
        
        delta_logit = intervened_logits - baseline_logits
        t1 = tokenizer.encode(w1, add_special_tokens=False)
        t2 = tokenizer.encode(w2, add_special_tokens=False)
        target_dlogit = 0
        if t1 and t2:
            target_dlogit = abs(delta_logit[t1[0]].item() - delta_logit[t2[0]].item())
        
        if target_dlogit > 1.0:
            functional_dirs.append(direction)
            functional_dlogits.append(target_dlogit)
    
    n_functional = len(functional_dirs)
    results["n_functional"] = n_functional
    results["PR_vs_functional"] = round(float(PR / n_functional), 4) if n_functional > 0 else 0
    print(f"  Functional dimensions: {n_functional}")
    print(f"  PR / n_functional = {PR/n_functional:.3f}" if n_functional > 0 else "  No functional dims")
    
    # 4. V_lang上界推导
    # 定理: V_lang_eff = PR(W_lm) ≤ rank(W_lm) ≤ min(vocab, d_model)
    # 推论: V_lang_eff ≤ d_model (因为rank ≤ d_model)
    # 更精确: V_lang_eff ≈ n_functional ≈ PR
    
    rank_approx = int(np.searchsorted(np.cumsum(S) / S.sum(), 0.99)) + 1
    results["rank_99"] = rank_approx
    results["V_lang_upper_bound"] = d_model
    results["V_lang_eff"] = n_functional
    
    # 5. V_lang/d_model的比例分析
    ratio = n_functional / d_model
    results["V_ratio"] = round(ratio, 6)
    
    # 6. 验证完备性: 功能维度能覆盖多少W_lm的行空间?
    if len(functional_dirs) > 1:
        dirs_matrix = np.array(functional_dirs)
        n_dirs, d = dirs_matrix.shape
        
        # 计算V_lang子空间与W_lm行空间的对齐度
        # W_lm的行空间由Vt的前rank行张成
        Vt_top = Vt[:min(rank_approx, Vt.shape[0]), :]  # [rank, d_model]
        
        # 功能方向在W_lm行空间中的投影比例
        proj_quality = []
        for i in range(min(n_dirs, 100)):
            v = dirs_matrix[i]
            # 投影到Vt_top张成的子空间
            proj_coeffs = Vt_top @ v  # [rank]
            proj_v = Vt_top.T @ proj_coeffs  # [d_model]
            proj_ratio = np.linalg.norm(proj_v) / (np.linalg.norm(v) + 1e-10)
            proj_quality.append(proj_ratio)
        
        mean_proj_quality = np.mean(proj_quality)
        results["mean_proj_quality"] = round(float(mean_proj_quality), 6)
        print(f"  Mean projection quality: {mean_proj_quality:.4f}")
        
        # V_lang子空间能重建多少W_lm方差?
        # 用功能方向作为基, 投影W_lm的行
        # 简化: 用top-SVD方向 vs 功能方向
        # 如果功能方向覆盖了top SVD方向, 则V_lang是"有效的"
        
        # SVD top方向 vs 功能方向的对齐
        alignment_scores = []
        for i in range(min(20, Vt_top.shape[0])):
            svd_dir = Vt_top[i] / (np.linalg.norm(Vt_top[i]) + 1e-10)
            max_align = 0
            for j in range(min(n_dirs, 100)):
                func_dir = dirs_matrix[j]
                func_dir_norm = func_dir / (np.linalg.norm(func_dir) + 1e-10)
                align = abs(np.dot(svd_dir, func_dir_norm))
                max_align = max(max_align, align)
            alignment_scores.append(max_align)
        
        mean_alignment = np.mean(alignment_scores)
        results["svd_func_alignment"] = round(float(mean_alignment), 4)
        print(f"  SVD-functional alignment: {mean_alignment:.4f}")
    
    # 7. 完备性定理陈述
    # 定理: V_lang_eff = PR(W_lm) ≈ n_functional, V_lang_eff ≤ d_model
    # 推论: 要使V_lang完备(重建误差<eps), 需要V_lang >= d_model * (1 - eps^2)
    # 证明: 由随机投影理论, n个方向在d维空间中的重建误差 = sqrt(1 - n/d)
    
    print(f"\n  === P424 Summary for {model_name} ===")
    print(f"  PR = {PR:.2f}, n_functional = {n_functional}, PR/n_func = {PR/n_functional:.3f}" if n_functional > 0 else "")
    print(f"  V_lang/d_model = {ratio:.4f}")
    print(f"  V_lang upper bound = d_model = {d_model}")
    if "mean_proj_quality" in results:
        print(f"  Projection quality = {results['mean_proj_quality']:.4f}")
    
    ts = time.strftime("%Y%m%d_%H%M")
    out_file = OUT_DIR / f"phase_lxxxvi_p424_{model_name}_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved to {out_file}")
    return results


# ========== P425: GLM4信号放大的真正来源 ==========

def run_p425(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P425: Signal Amplification Source - {model_name}")
    print(f"{'='*60}")
    
    layers = get_layers(model)
    n_layers = len(layers)
    embed = model.get_input_embeddings()
    
    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    
    beta = 8.0
    results = {"model": model_name, "exp": "p425"}
    
    # 测试维度
    w1, w2 = "true", "false"
    direction, _ = get_dimension_direction(model, tokenizer, w1, w2)
    direction_t = torch.tensor(direction * beta, dtype=torch.float32)
    
    # 基线
    with torch.no_grad():
        baseline_output = model(input_ids)
        baseline_logits = baseline_output.logits[0, -1].detach().cpu().float()
    
    # 收集每层各组件的输出
    # 标准Transformer层: x -> LayerNorm -> Attention -> residual -> LayerNorm -> MLP -> residual
    
    # 选择5个关键层
    test_layers = [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    
    component_analysis = {}
    
    for l in test_layers:
        layer = layers[l]
        print(f"\n  Analyzing layer {l}...")
        
        # 1. 收集该层的residual输入
        layer_input = {}
        def input_hook(module, args, li=l):
            if isinstance(args, tuple) and len(args) > 0:
                layer_input[li] = args[0].detach().cpu().float()
            return args
        
        h_in = layer.register_forward_pre_hook(input_hook)
        
        # 2. 收集Attention输出
        attn_output = {}
        def attn_hook(module, input, output, li=l):
            # output可能是tuple, 第一个是attn output
            if isinstance(output, tuple):
                attn_output[li] = output[0].detach().cpu().float()
            else:
                attn_output[li] = output.detach().cpu().float()
            return output
        
        # 找到self_attn
        if hasattr(layer, "self_attn"):
            h_attn = layer.self_attn.register_forward_hook(attn_hook)
        elif hasattr(layer, "attention"):
            h_attn = layer.attention.register_forward_hook(attn_hook)
        else:
            h_attn = None
        
        # 3. 收集MLP输出
        mlp_output = {}
        def mlp_hook(module, input, output, li=l):
            if isinstance(output, tuple):
                mlp_output[li] = output[0].detach().cpu().float()
            else:
                mlp_output[li] = output.detach().cpu().float()
            return output
        
        if hasattr(layer, "mlp"):
            h_mlp = layer.mlp.register_forward_hook(mlp_hook)
        else:
            h_mlp = None
        
        # 基线forward
        with torch.no_grad():
            model(input_ids)
        
        # 移除hooks
        h_in.remove()
        if h_attn: h_attn.remove()
        if h_mlp: h_mlp.remove()
        
        # 4. 现在注入信号, 重复收集
        layer_input_inj = {}
        attn_output_inj = {}
        mlp_output_inj = {}
        
        def input_hook_inj(module, args, li=l):
            if isinstance(args, tuple) and len(args) > 0:
                layer_input_inj[li] = args[0].detach().cpu().float()
            return args
        
        def attn_hook_inj(module, input, output, li=l):
            if isinstance(output, tuple):
                attn_output_inj[li] = output[0].detach().cpu().float()
            else:
                attn_output_inj[li] = output.detach().cpu().float()
            return output
        
        def mlp_hook_inj(module, input, output, li=l):
            if isinstance(output, tuple):
                mlp_output_inj[li] = output[0].detach().cpu().float()
            else:
                mlp_output_inj[li] = output.detach().cpu().float()
            return output
        
        h_in2 = layer.register_forward_pre_hook(input_hook_inj)
        if hasattr(layer, "self_attn"):
            h_attn2 = layer.self_attn.register_forward_hook(attn_hook_inj)
        elif hasattr(layer, "attention"):
            h_attn2 = layer.attention.register_forward_hook(attn_hook_inj)
        else:
            h_attn2 = None
        if hasattr(layer, "mlp"):
            h_mlp2 = layer.mlp.register_forward_hook(mlp_hook_inj)
        else:
            h_mlp2 = None
        
        # 在embed注入
        def embed_inj(module, input, output, d=direction_t):
            modified = output.clone()
            modified[0, -1, :] += d.to(output.device)
            return modified
        
        h_embed = embed.register_forward_hook(embed_inj)
        
        with torch.no_grad():
            model(input_ids)
        
        h_embed.remove()
        h_in2.remove()
        if h_attn2: h_attn2.remove()
        if h_mlp2: h_mlp2.remove()
        
        # 5. 计算各组件的信号增益
        layer_result = {}
        
        # 输入变化
        if l in layer_input and l in layer_input_inj:
            dh_input = (layer_input_inj[l][0, -1] - layer_input[l][0, -1])
            dh_input_norm = dh_input.norm().item()
            layer_result["dh_input_norm"] = round(dh_input_norm, 4)
        
        # Attention输出变化
        if l in attn_output and l in attn_output_inj:
            dh_attn = (attn_output_inj[l][0, -1] - attn_output[l][0, -1])
            dh_attn_norm = dh_attn.norm().item()
            layer_result["dh_attn_norm"] = round(dh_attn_norm, 4)
            
            # Attention增益
            if dh_input_norm > 1e-10:
                attn_gain = dh_attn_norm / dh_input_norm
                layer_result["attn_gain"] = round(attn_gain, 4)
        
        # MLP输出变化
        if l in mlp_output and l in mlp_output_inj:
            dh_mlp = (mlp_output_inj[l][0, -1] - mlp_output[l][0, -1])
            dh_mlp_norm = dh_mlp.norm().item()
            layer_result["dh_mlp_norm"] = round(dh_mlp_norm, 4)
            
            # MLP增益
            if dh_input_norm > 1e-10:
                mlp_gain = dh_mlp_norm / dh_input_norm
                layer_result["mlp_gain"] = round(mlp_gain, 4)
        
        # LayerNorm增益 (通过输入/输出比例推断)
        # 如果有input_layernorm
        if hasattr(layer, "input_layernorm"):
            # LayerNorm: output = (x - mean) / std * gamma + beta
            # 增益来自1/std: 小std -> 大增益
            if l in layer_input:
                x = layer_input[l][0, -1]
                std = x.std().item()
                layer_result["input_std"] = round(std, 4)
                layer_result["layernorm_gain_factor"] = round(1.0 / (std + 1e-10), 4)
        
        print(f"    dh_input={layer_result.get('dh_input_norm', 'N/A')}, "
              f"dh_attn={layer_result.get('dh_attn_norm', 'N/A')}, "
              f"dh_mlp={layer_result.get('dh_mlp_norm', 'N/A')}, "
              f"attn_gain={layer_result.get('attn_gain', 'N/A')}, "
              f"mlp_gain={layer_result.get('mlp_gain', 'N/A')}")
        
        component_analysis[str(l)] = layer_result
    
    results["component_analysis"] = component_analysis
    
    # 总结
    print(f"\n  === P425 Summary for {model_name} ===")
    for l, data in component_analysis.items():
        print(f"  L{l}: attn_gain={data.get('attn_gain', 'N/A')}, mlp_gain={data.get('mlp_gain', 'N/A')}, "
              f"LN_factor={data.get('layernorm_gain_factor', 'N/A')}")
    
    ts = time.strftime("%Y%m%d_%H%M")
    out_file = OUT_DIR / f"phase_lxxxvi_p425_{model_name}_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved to {out_file}")
    return results


# ========== P426: 语言空间几何 ==========

def run_p426(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P426: Language Space Geometry - {model_name}")
    print(f"{'='*60}")
    
    W_U = model.lm_head.weight.detach().cpu().float().numpy()
    vocab_size, d_model = W_U.shape
    print(f"  W_lm: {vocab_size} x {d_model}")
    
    results = {"model": model_name, "exp": "p426", "vocab_size": vocab_size, "d_model": d_model}
    
    # 1. 收集功能维度方向
    embed = model.get_input_embeddings()
    beta = 8.0
    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    
    with torch.no_grad():
        baseline_logits = model(input_ids).logits[0, -1].detach().cpu().float()
    
    # 用简洁的维度列表
    dim_list = {
        "style": ("formal", "informal"), "logic": ("true", "false"),
        "grammar": ("active", "passive"), "sentiment": ("happy", "sad"),
        "tense": ("was", "is"), "certainty": ("definitely", "maybe"),
        "quantity": ("many", "few"), "complexity": ("complex", "simple"),
        "size": ("large", "small"), "strength": ("strong", "weak"),
        "beauty": ("beautiful", "ugly"), "truth": ("truth", "lie"),
        "freedom": ("free", "bound"), "wisdom": ("wise", "foolish"),
        "peace": ("peace", "war"), "love": ("love", "hate"),
        "depth": ("deep", "shallow"), "width": ("wide", "narrow"),
        "health": ("healthy", "sick"), "order": ("ordered", "chaotic"),
        "wealth": ("rich", "poor"), "power": ("powerful", "helpless"),
        "safety": ("safe", "dangerous"), "honesty": ("honest", "deceitful"),
        "humor": ("funny", "serious"), "novelty": ("novel", "familiar"),
        "anger": ("angry", "calm"), "fear": ("afraid", "confident"),
        "joy": ("joyful", "sorrowful"), "trust_dim": ("trust", "distrust"),
        "color": ("colorful", "colorless"), "motion": ("moving", "still"),
        "growth": ("growing", "shrinking"), "change": ("changing", "stable"),
        "balance": ("balanced", "unbalanced"), "reason": ("reasonable", "unreasonable"),
        "fact": ("factual", "fictional"), "reality": ("real", "imaginary"),
        "possibility": ("possible", "impossible"), "precision": ("precise", "approximate"),
        "consistency": ("consistent", "inconsistent"), "flexibility": ("flexible", "rigid"),
        "efficiency": ("efficient", "wasteful"), "creativity": ("creative", "conventional"),
        "intelligence": ("intelligent", "foolish2"), "knowledge": ("knowledgeable", "ignorant"),
        "consciousness": ("conscious", "unconscious"), "hope": ("hopeful", "hopeless"),
        "curiosity": ("curious", "indifferent"), "justice": ("just", "unjust"),
        "empathy": ("empathetic", "apathetic"), "kindness": ("kind", "cruel"),
    }
    
    functional_dirs = []
    functional_names = []
    
    for dim_name, (w1, w2) in dim_list.items():
        direction, norm = get_dimension_direction(model, tokenizer, w1, w2)
        if norm < 1e-8:
            continue
        
        direction_t = torch.tensor(direction * beta, dtype=torch.float32)
        
        def inj_hook(module, input, output, d=direction_t):
            modified = output.clone()
            modified[0, -1, :] += d.to(output.device)
            return modified
        
        h = embed.register_forward_hook(inj_hook)
        with torch.no_grad():
            intervened_logits = model(input_ids).logits[0, -1].detach().cpu().float()
        h.remove()
        
        delta_logit = intervened_logits - baseline_logits
        t1 = tokenizer.encode(w1, add_special_tokens=False)
        t2 = tokenizer.encode(w2, add_special_tokens=False)
        target_dlogit = 0
        if t1 and t2:
            target_dlogit = abs(delta_logit[t1[0]].item() - delta_logit[t2[0]].item())
        
        if target_dlogit > 1.0:
            functional_dirs.append(direction)
            functional_names.append(dim_name)
    
    n_func = len(functional_dirs)
    print(f"  Functional dimensions: {n_func}")
    results["n_functional"] = n_func
    
    if n_func < 2:
        print("  Too few functional dimensions for geometry analysis")
        return results
    
    # 2. V_lang子空间分析
    dirs_matrix = np.array(functional_dirs)  # [n_func, d_model]
    
    # SVD of functional directions
    U, S, Vt = np.linalg.svd(dirs_matrix, full_matrices=False)
    
    # 3. 子空间维度分析
    total_energy = S.sum()
    cumsum = np.cumsum(S) / total_energy
    
    # 有效维度 (participation ratio of directions)
    S_sq = S**2
    PR_dirs = (S_sq.sum())**2 / (S_sq**2).sum()
    results["PR_of_directions"] = round(float(PR_dirs), 2)
    
    # 4. 曲率分析: 功能方向之间的角度分布
    norms = np.linalg.norm(dirs_matrix, axis=1, keepdims=True)
    normalized = dirs_matrix / (norms + 1e-10)
    cos_matrix = normalized @ normalized.T
    np.fill_diagonal(cos_matrix, 0)
    
    # 角度分布
    angles = np.arccos(np.clip(cos_matrix, -1, 1)) * 180 / np.pi
    abs_angles = np.abs(angles)
    
    results["angle_stats"] = {
        "mean": round(float(np.mean(abs_angles)), 2),
        "std": round(float(np.std(abs_angles)), 2),
        "median": round(float(np.median(abs_angles)), 2),
        "min": round(float(np.min(abs_angles)), 2),
        "max": round(float(np.max(abs_angles)), 2),
        "percentile_10": round(float(np.percentile(abs_angles, 10)), 2),
        "percentile_90": round(float(np.percentile(abs_angles, 90)), 2),
    }
    print(f"  Angle stats: mean={results['angle_stats']['mean']:.1f} deg, "
          f"std={results['angle_stats']['std']:.1f}, "
          f"range=[{results['angle_stats']['min']:.1f}, {results['angle_stats']['max']:.1f}]")
    
    # 5. V_lang子空间与随机子空间的重叠度
    # 生成同维度的随机子空间
    np.random.seed(42)
    n_subspace = min(n_func, PR_dirs)
    random_dirs = np.random.randn(int(n_subspace), d_model)
    random_norms = np.linalg.norm(random_dirs, axis=1, keepdims=True)
    random_normalized = random_dirs / (random_norms + 1e-10)
    
    # 计算两个子空间的重叠度: ||P_func * P_rand||_F / sqrt(min(n1, n2))
    # P = U * U^T (投影矩阵)
    U_func = normalized[:, :min(n_func, d_model)]  # 不对, 应该用SVD
    # 用Vt的前n_subspace行作为V_lang子空间的基
    V_func = Vt[:int(n_subspace), :]  # [n_subspace, d_model]
    V_rand, _ = np.linalg.qr(random_dirs.T)
    V_rand = V_rand[:, :int(n_subspace)].T  # [n_subspace, d_model]
    
    # 子空间重叠 = ||V_func @ V_rand^T||_F^2 / n_subspace
    overlap_matrix = V_func @ V_rand.T  # [n_subspace, n_subspace]
    overlap = np.linalg.norm(overlap_matrix, 'fro')**2 / n_subspace
    
    results["subspace_overlap_with_random"] = round(float(overlap), 6)
    print(f"  Subspace overlap with random: {overlap:.4f} (1.0=random, n_subspace/d_model={n_subspace/d_model:.4f})")
    
    # 6. 信号传播特征: 不同方向在层间的放大差异
    # 计算每个功能维度的"方向质量": Δlogit向量的聚焦度
    direction_qualities = {}
    
    for i, dim_name in enumerate(functional_names[:20]):  # 只测前20个
        direction = functional_dirs[i]
        direction_t = torch.tensor(direction * beta, dtype=torch.float32)
        
        def inj_hook2(module, input, output, d=direction_t):
            modified = output.clone()
            modified[0, -1, :] += d.to(output.device)
            return modified
        
        h = embed.register_forward_hook(inj_hook2)
        with torch.no_grad():
            intervened_logits = model(input_ids).logits[0, -1].detach().cpu().float()
        h.remove()
        
        delta_logit = intervened_logits - baseline_logits
        
        # 聚焦度: Δlogit的稀疏度
        dl_abs = delta_logit.abs()
        dl_norm = dl_abs.norm().item()
        dl_max = dl_abs.max().item()
        sparsity = (dl_max / (dl_norm + 1e-10))**2  # 1/n_eff
        
        # 前10个受影响最大的词
        top10_idx = dl_abs.argsort(descending=True)[:10]
        top10_vals = dl_abs[top10_idx].tolist()
        
        direction_qualities[dim_name] = {
            "delta_norm": round(dl_norm, 4),
            "delta_max": round(dl_max, 4),
            "sparsity": round(float(sparsity), 6),
            "n_effective": round(1.0 / (sparsity + 1e-10), 1),
        }
    
    results["direction_qualities"] = direction_qualities
    
    # 7. 语言空间的拓扑结构
    # 分析功能维度之间的"邻近关系"
    # 用cos_matrix做层次聚类
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    
    # 距离 = 1 - |cos|
    dist_matrix = 1 - np.abs(cos_matrix)
    np.fill_diagonal(dist_matrix, 0)
    
    # 转换为压缩距离矩阵
    condensed = squareform(dist_matrix)
    
    # 层次聚类
    Z = linkage(condensed, method='ward')
    
    # 分成5类
    n_clusters = min(5, n_func)
    clusters = fcluster(Z, n_clusters, criterion='maxclust')
    
    cluster_members = {}
    for i, c in enumerate(clusters):
        c_str = str(c)
        if c_str not in cluster_members:
            cluster_members[c_str] = []
        cluster_members[c_str].append(functional_names[i])
    
    results["clusters"] = cluster_members
    print(f"\n  Language space clusters:")
    for c, members in cluster_members.items():
        print(f"    Cluster {c}: {members[:5]}{'...' if len(members) > 5 else ''}")
    
    # 总结
    print(f"\n  === P426 Summary for {model_name} ===")
    print(f"  Functional dims: {n_func}")
    print(f"  PR of directions: {PR_dirs:.2f}")
    print(f"  Angle mean: {results['angle_stats']['mean']:.1f} deg")
    print(f"  Subspace overlap with random: {overlap:.4f}")
    print(f"  Clusters: {len(cluster_members)}")
    
    ts = time.strftime("%Y%m%d_%H%M")
    out_file = OUT_DIR / f"phase_lxxxvi_p426_{model_name}_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved to {out_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase LXXXVI: Cumulative Amplification & V_lang Completeness")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, required=True, choices=["p423", "p424", "p425", "p426"])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    
    if args.exp == "p423":
        run_p423(model, tokenizer, device, args.model)
    elif args.exp == "p424":
        run_p424(model, tokenizer, device, args.model)
    elif args.exp == "p425":
        run_p425(model, tokenizer, device, args.model)
    elif args.exp == "p426":
        run_p426(model, tokenizer, device, args.model)
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
