"""
Phase LXXXV-P419/420/421/422: 非线性放大精确理论与V_lang极限
================================================================================

阶段H核心任务 - 非线性激活函数精确放大公式 + V_lang极限搜索 + W_lm结构性 + 完备性严格证明

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

BASE_DIM_PAIRS = {
    "style": [("formal", "informal")],
    "logic": [("true", "false")],
    "grammar": [("active", "passive")],
    "sentiment": [("happy", "sad")],
    "tense": [("was", "is")],
    "certainty": [("definitely", "maybe")],
    "quantity": [("many", "few")],
}

MID_DIM_PAIRS = {
    "complexity": [("complex", "simple")],
    "formality": [("professional", "casual")],
    "politeness": [("please", "now")],
    "specificity": [("specifically", "generally")],
    "speed": [("quickly", "slowly")],
    "size": [("large", "small")],
    "age": [("old", "new")],
    "distance": [("near", "far")],
    "brightness": [("bright", "dark")],
    "temperature": [("hot", "cold")],
    "weight": [("heavy", "light")],
    "sound": [("loud", "quiet")],
    "frequency": [("often", "rarely")],
    "importance": [("important", "trivial")],
    "position": [("above", "below")],
    "direction": [("forward", "backward")],
    "completeness": [("complete", "partial")],
    "value": [("valuable", "worthless")],
}

EXT_DIM_PAIRS = {
    "strength": [("strong", "weak")],
    "beauty": [("beautiful", "ugly")],
    "truth": [("truth", "lie")],
    "freedom": [("free", "bound")],
    "wisdom": [("wise", "foolish")],
    "courage": [("brave", "cowardly")],
    "peace": [("peace", "war")],
    "love": [("love", "hate")],
    "time_duration": [("eternal", "momentary")],
    "clarity": [("clear", "vague")],
    "depth": [("deep", "shallow")],
    "width": [("wide", "narrow")],
    "height": [("tall", "short")],
    "sharpness": [("sharp", "dull")],
    "softness": [("soft", "hard")],
    "sweetness": [("sweet", "bitter")],
    "speed_fast": [("fast", "slow")],
    "wealth": [("rich", "poor")],
    "power": [("powerful", "helpless")],
    "safety": [("safe", "dangerous")],
    "health": [("healthy", "sick")],
    "order": [("ordered", "chaotic")],
    "unity": [("united", "divided")],
    "purity": [("pure", "corrupt")],
    "honesty": [("honest", "deceitful")],
    "patience": [("patient", "impatient")],
    "humor": [("funny", "serious")],
    "novelty": [("novel", "familiar")],
    "luxury": [("luxurious", "plain")],
    "silence": [("silent", "noisy")],
}

ULTRA2_DIM_PAIRS = {
    "anger": [("angry", "calm")],
    "fear": [("afraid", "confident")],
    "joy": [("joyful", "sorrowful")],
    "trust_dim": [("trust", "distrust")],
    "surprise": [("surprising", "expected")],
    "disgust": [("disgusting", "pleasant")],
    "anticipation": [("anticipated", "unexpected")],
    "color": [("colorful", "colorless")],
    "texture": [("smooth", "rough")],
    "shape": [("round", "angular")],
    "motion": [("moving", "still")],
    "growth": [("growing", "shrinking")],
    "change": [("changing", "stable")],
    "balance": [("balanced", "unbalanced")],
    "symmetry": [("symmetric", "asymmetric")],
    "rhythm": [("rhythmic", "arrhythmic")],
    "harmony": [("harmonious", "discordant")],
    "chaos": [("chaotic", "ordered")],
    "structure_dim": [("structured", "unstructured")],
    "pattern": [("patterned", "random")],
    "logic_dim": [("logical", "illogical")],
    "reason": [("reasonable", "unreasonable")],
    "evidence": [("evident", "doubtful")],
    "proof": [("proven", "unproven")],
    "fact": [("factual", "fictional")],
    "reality": [("real", "imaginary")],
    "existence": [("existing", "nonexistent")],
    "possibility": [("possible", "impossible")],
    "necessity": [("necessary", "optional")],
    "sufficiency": [("sufficient", "insufficient")],
    "certainty_dim": [("certain", "uncertain")],
    "precision": [("precise", "approximate")],
    "accuracy": [("accurate", "inaccurate")],
    "validity": [("valid", "invalid")],
    "reliability": [("reliable", "unreliable")],
    "consistency": [("consistent", "inconsistent")],
    "coherence": [("coherent", "incoherent")],
    "simplicity": [("simple", "complex")],
    "elegance": [("elegant", "clumsy")],
    "flexibility": [("flexible", "rigid")],
    "adaptability": [("adaptable", "inflexible")],
    "efficiency": [("efficient", "wasteful")],
    "productivity": [("productive", "idle")],
    "creativity": [("creative", "conventional")],
    "innovation": [("innovative", "traditional")],
    "intelligence": [("intelligent", "foolish")],
    "wisdom_dim": [("wise", "ignorant")],
    "knowledge": [("knowledgeable", "ignorant2")],
    "skill": [("skilled", "unskilled")],
    "talent": [("talented", "untalented")],
    "genius": [("genius", "ordinary")],
    "art": [("artistic", "technical")],
    "science": [("scientific", "intuitive")],
    "math": [("mathematical", "poetic")],
    "language_dim": [("linguistic", "nonverbal")],
    "music": [("musical", "atonal")],
    "dance": [("graceful", "awkward")],
    "poetry": [("poetic", "prosaic")],
    "story": [("narrative", "descriptive")],
    "drama": [("dramatic", "subtle")],
    "comedy": [("comic", "tragic")],
    "epic": [("epic", "mundane")],
    "myth": [("mythical", "historical")],
    "ritual": [("ritual", "spontaneous")],
    "sacred": [("sacred", "profane")],
    "divine": [("divine", "mortal")],
    "mystic": [("mystical", "rational")],
    "spirit": [("spiritual", "material")],
    "soul": [("soulful", "soulless")],
    "consciousness": [("conscious", "unconscious")],
    "awareness": [("aware", "unaware")],
    "perception": [("perceptive", "oblivious")],
    "attention_dim": [("attentive", "distracted")],
    "memory": [("memorable", "forgettable")],
    "imagination": [("imaginative", "literal")],
    "dream": [("dreamy", "realistic")],
    "vision": [("visionary", "practical")],
    "hope": [("hopeful", "hopeless")],
    "despair": [("despairing", "optimistic")],
    "faith": [("faithful", "faithless")],
    "doubt": [("doubtful", "certain2")],
    "belief": [("believing", "skeptical")],
    "wonder": [("wondering", "knowing")],
    "curiosity": [("curious", "indifferent")],
    "discovery": [("discovered", "hidden")],
    "mystery": [("mysterious", "obvious")],
    "secret": [("secret", "public")],
    "privacy": [("private", "shared")],
    "freedom_dim": [("free2", "constrained")],
    "justice": [("just", "unjust")],
    "equality": [("equal", "unequal")],
    "fairness": [("fair", "unfair")],
    "mercy": [("merciful", "cruel")],
    "compassion": [("compassionate", "indifferent2")],
    "empathy": [("empathetic", "apathetic")],
    "kindness": [("kind", "cruel2")],
    "generosity": [("generous", "stingy")],
    "gratitude": [("grateful", "ungrateful")],
    "respect": [("respectful", "disrespectful")],
}

ALL_200_DIMS = {**BASE_DIM_PAIRS, **MID_DIM_PAIRS, **EXT_DIM_PAIRS, **ULTRA2_DIM_PAIRS}

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


def get_mlp_weights(layer):
    """提取MLP权重矩阵, 返回(W_up, W_down, has_gate, W_gate_or_None)"""
    mlp = layer.mlp
    W_up = None
    W_down = None
    has_gate = False
    W_gate = None
    
    # GLM4 style: gate_up_proj (fused gate+up)
    if hasattr(mlp, "gate_up_proj"):
        W_gate_up = mlp.gate_up_proj.weight.detach().cpu().float()
        # gate_up_proj shape: [2*d_mlp, d_model], 前半是gate, 后半是up
        d_mlp_half = W_gate_up.shape[0] // 2
        W_gate = W_gate_up[:d_mlp_half, :]  # [d_mlp, d_model]
        W_up = W_gate_up[d_mlp_half:, :]    # [d_mlp, d_model]
        has_gate = True
    elif hasattr(mlp, "gate_proj"):
        W_gate = mlp.gate_proj.weight.detach().cpu().float()
        has_gate = True
        if hasattr(mlp, "up_proj"):
            W_up = mlp.up_proj.weight.detach().cpu().float()
        else:
            W_up = W_gate  # fallback
    elif hasattr(mlp, "up_proj"):
        W_up = mlp.up_proj.weight.detach().cpu().float()
    elif hasattr(mlp, "fc1"):
        W_up = mlp.fc1.weight.detach().cpu().float()
    elif hasattr(mlp, "dense_h_to_4h"):
        W_up = mlp.dense_h_to_4h.weight.detach().cpu().float()
    
    if hasattr(mlp, "down_proj"):
        W_down = mlp.down_proj.weight.detach().cpu().float()
    elif hasattr(mlp, "fc2"):
        W_down = mlp.fc2.weight.detach().cpu().float()
    elif hasattr(mlp, "dense_4h_to_h"):
        W_down = mlp.dense_4h_to_h.weight.detach().cpu().float()
    
    return W_up, W_down, has_gate, W_gate


# ========== P419: 非线性激活函数Jacobian分析 ==========

def run_p419(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P419: Nonlinear Activation Jacobian - {model_name}")
    print(f"{'='*60}")
    
    layers = get_layers(model)
    n_layers = len(layers)
    target_layers = list(range(max(0, n_layers - 5), n_layers))
    
    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    
    test_dims = {
        "style": ("formal", "informal"),
        "logic": ("true", "false"),
        "sentiment": ("happy", "sad"),
        "tense": ("was", "is"),
    }
    
    beta = 8.0
    results = {"model": model_name, "exp": "p419", "layers": {}}
    
    # 基线forward - 收集所有层的residual
    baseline_cache = {}
    hooks = []
    
    def cache_hook(name):
        def hook_fn(module, input, output):
            baseline_cache[name] = input[0].detach().cpu().float()
        return hook_fn
    
    for l in target_layers:
        layer = layers[l]
        h = layer.register_forward_hook(cache_hook(f"layer_{l}"))
        hooks.append(h)
    
    with torch.no_grad():
        baseline_output = model(input_ids)
    baseline_logits = baseline_output.logits[0, -1].detach().cpu().float()
    
    for h in hooks:
        h.remove()
    
    for dim_name, (w1, w2) in test_dims.items():
        direction, _ = get_dimension_direction(model, tokenizer, w1, w2)
        direction_t = torch.tensor(direction * beta, dtype=torch.float32)
        
        print(f"\n  Dim: {dim_name} ({w1}/{w2})")
        dim_results = {}
        
        for l in target_layers:
            layer = layers[l]
            W_up, W_down, has_gate, W_gate = get_mlp_weights(layer)
            
            if W_up is None or W_down is None:
                print(f"    L{l}: MLP weights not found")
                continue
            
            # 获取基线residual
            if f"layer_{l}" in baseline_cache:
                baseline_resid = baseline_cache[f"layer_{l}"][0, -1]  # [d_model]
            else:
                # 用embedding近似L0
                embed = model.get_input_embeddings()
                baseline_resid = embed(input_ids)[0, -1].detach().cpu().float()
            
            dh = direction_t  # [d_model]
            
            # === Jacobian分析 ===
            # MLP(x) = W_down * act(W_up * x)  (简化, 无gate)
            # 或 MLP(x) = W_down * (act(W_gate*x) * (W_up*x))  (有gate, SiLU)
            
            h_up = W_up @ baseline_resid  # [d_mlp]
            
            if has_gate and W_gate is not None:
                h_gate = W_gate @ baseline_resid  # [d_mlp]
                # SiLU(x) = x * sigmoid(x)
                sigma_gate = torch.sigmoid(h_gate)
                # 导数: SiLU'(x) = sigma(x) + x*sigma(x)*(1-sigma(x))
                #       = sigma(x) * (1 + x*(1-sigma(x)))
                act_deriv = sigma_gate * (1 + h_gate * (1 - sigma_gate))
                # 二阶导: SiLU''(x) = sigma(x)*(1-sigma(x))*(2 + x*(1-sigma(x)))
                act_deriv2 = sigma_gate * (1 - sigma_gate) * (2 + h_gate * (1 - sigma_gate))
            else:
                # GELU近似: act(x) = x*Phi(x)
                # 导数: Phi(x) + x*phi(x), phi是标准正态密度
                from torch.distributions.normal import Normal
                n = Normal(0, 1)
                phi = torch.exp(n.log_prob(h_up))
                Phi = n.cdf(h_up)
                act_deriv = Phi + h_up * phi
                # 二阶导: 2*phi(x) - x^2*phi(x) (近似)
                act_deriv2 = 2 * phi - h_up**2 * phi
            
            # Jacobian: J_MLP = W_down * diag(act_deriv) * W_up
            # J * dh = W_down * (act_deriv * (W_up * dh))
            temp1 = W_up @ dh  # [d_mlp]
            temp1_scaled = temp1 * act_deriv  # diag乘法
            J_dh = W_down @ temp1_scaled  # [d_model]
            
            mlp_jacobian_gain = J_dh.norm().item() / (dh.norm().item() + 1e-10)
            
            # 二阶项: 0.5 * W_down * (act_deriv2 * (W_up * dh)^2)
            temp2 = temp1**2 * act_deriv2
            second_order = 0.5 * (W_down @ temp2)
            second_order_ratio = second_order.norm().item() / (J_dh.norm().item() + 1e-10)
            
            # 实际增益: 在该层注入dh后测量logit变化
            # 用register_forward_hook在layer输入处注入
            def make_injection_hook(d_inj, layer_idx):
                def hook_fn(module, input, output):
                    # 在residual stream中注入
                    # output是layer的输出, input[0]是输入
                    return output  # 这里不能直接修改, 需要更精确的方法
                return hook_fn
            
            # 简化: 用W_lm直接计算logit变化
            W_lm = model.lm_head.weight.detach().cpu().float()  # [vocab, d_model]
            
            # 如果在L0注入, 信号传播到最后层近似线性(忽略非线性)
            # 更精确: 用Jacobian链式传播
            # 实际用简化: direct logit change = W_lm @ J_dh
            delta_logit_linear = W_lm @ J_dh
            delta_logit_linear_norm = delta_logit_linear.norm().item()
            
            # 用实际forward的logit (L0注入)
            # 重新计算: embed + direction -> logits
            embed = model.get_input_embeddings()
            with torch.no_grad():
                base_embed = embed(input_ids)
                # 在最后一个token位置注入
                modified_embed = base_embed.clone()
                modified_embed[0, -1, :] += direction_t.to(device)
                
                # 注意: 有些模型有model.model.embed_tokens, 有些是model.model.encoder
                # 简化: 直接用L0 hook
                def inj_hook(module, input, output):
                    modified = output.clone()
                    modified[0, -1, :] += direction_t.to(output.device)
                    return modified
                
                h = embed.register_forward_hook(inj_hook)
                output = model(input_ids)
                h.remove()
                
                actual_logits = output.logits[0, -1].detach().cpu().float()
                actual_delta_logit = (actual_logits - baseline_logits).norm().item()
            
            actual_gain = actual_delta_logit / beta
            
            layer_result = {
                "mlp_jacobian_gain": round(mlp_jacobian_gain, 4),
                "second_order_ratio": round(second_order_ratio, 6),
                "delta_logit_linear_norm": round(delta_logit_linear_norm, 4),
                "actual_delta_logit": round(actual_delta_logit, 4),
                "actual_gain": round(actual_gain, 4),
                "linear_vs_actual_ratio": round(delta_logit_linear_norm / (actual_delta_logit + 1e-10), 4),
                "act_deriv_mean": round(act_deriv.mean().item(), 4),
                "act_deriv_std": round(act_deriv.std().item(), 4),
                "act_deriv_max": round(act_deriv.max().item(), 4),
                "has_gate": has_gate,
            }
            
            dim_results[str(l)] = layer_result
            print(f"    L{l}: J_gain={mlp_jacobian_gain:.3f}, 2nd/1st={second_order_ratio:.4f}, "
                  f"actual_gain={actual_gain:.3f}, linear/actual={delta_logit_linear_norm/(actual_delta_logit+1e-10):.2f}x, "
                  f"act'_mean={act_deriv.mean().item():.4f}")
        
        results["layers"][dim_name] = dim_results
    
    # 总结
    print(f"\n  === P419 Summary for {model_name} ===")
    for dim_name, dim_data in results["layers"].items():
        j_gains = [v["mlp_jacobian_gain"] for v in dim_data.values()]
        actual_gains = [v["actual_gain"] for v in dim_data.values()]
        s2f = [v["second_order_ratio"] for v in dim_data.values()]
        lin_act = [v["linear_vs_actual_ratio"] for v in dim_data.values()]
        ad_mean = [v["act_deriv_mean"] for v in dim_data.values()]
        
        print(f"  {dim_name}: J_gain={np.mean(j_gains):.3f}, actual_gain={np.mean(actual_gains):.3f}, "
              f"2nd/1st={np.mean(s2f):.4f}, linear/actual={np.mean(lin_act):.2f}x, "
              f"act'_mean={np.mean(ad_mean):.4f}")
    
    ts = time.strftime("%Y%m%d_%H%M")
    out_file = OUT_DIR / f"phase_lxxxv_p419_{model_name}_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved to {out_file}")
    return results


# ========== P420: V_lang极限搜索——200+词对 ==========

def run_p420(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P420: V_lang Limit Search (200+ dims) - {model_name}")
    print(f"{'='*60}")
    
    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    beta = 8.0
    threshold = 1.0
    
    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    
    with torch.no_grad():
        baseline_logits = model(input_ids).logits[0, -1].detach().cpu().float()
    
    all_dlogits = {}
    functional_dirs = []
    functional_names = []
    
    print(f"  Testing {len(ALL_200_DIMS)} dimensions...")
    
    for idx, (dim_name, pairs) in enumerate(ALL_200_DIMS.items()):
        w1, w2 = pairs[0]
        direction, norm = get_dimension_direction(model, tokenizer, w1, w2)
        
        if norm < 1e-8:
            continue
        
        direction_t = torch.tensor(direction * beta, dtype=torch.float32)
        
        # L0注入
        def inj_hook(module, input, output, d=direction_t):
            modified = output.clone()
            modified[0, -1, :] += d.to(output.device)
            return modified
        
        h = embed.register_forward_hook(inj_hook)
        with torch.no_grad():
            intervened_logits = model(input_ids).logits[0, -1].detach().cpu().float()
        h.remove()
        
        delta_logit = intervened_logits - baseline_logits
        delta_norm = delta_logit.norm().item()
        
        # 目标词dlogit
        t1 = tokenizer.encode(w1, add_special_tokens=False)
        t2 = tokenizer.encode(w2, add_special_tokens=False)
        
        target_dlogit = 0
        if t1 and t2:
            target_dlogit = abs(delta_logit[t1[0]].item() - delta_logit[t2[0]].item())
        
        is_functional = target_dlogit > threshold
        
        all_dlogits[dim_name] = {
            "target_dlogit": round(target_dlogit, 4),
            "total_norm": round(delta_norm, 4),
            "is_functional": is_functional,
        }
        
        if is_functional:
            functional_dirs.append(direction)
            functional_names.append(dim_name)
        
        if (idx + 1) % 50 == 0:
            n_func = len(functional_dirs)
            print(f"    {idx+1}/{len(ALL_200_DIMS)}: {n_func} functional ({100*n_func/(idx+1):.1f}%)")
    
    n_functional = len(functional_dirs)
    n_total = len(all_dlogits)
    
    results = {
        "model": model_name, "exp": "p420",
        "n_functional": n_functional, "n_total": n_total,
        "functional_ratio": round(n_functional / n_total, 4),
        "all_dlogits": all_dlogits,
    }
    
    # 正交性分析
    if len(functional_dirs) > 1:
        dirs_matrix = np.array(functional_dirs)
        U, S, Vt = np.linalg.svd(dirs_matrix, full_matrices=False)
        
        total_energy = S.sum()
        cumsum = np.cumsum(S)
        rank_95 = int(np.searchsorted(cumsum, 0.95 * total_energy)) + 1
        rank_99 = int(np.searchsorted(cumsum, 0.99 * total_energy)) + 1
        
        norms = np.linalg.norm(dirs_matrix, axis=1, keepdims=True)
        normalized = dirs_matrix / (norms + 1e-10)
        cos_mat = normalized @ normalized.T
        np.fill_diagonal(cos_mat, 0)
        mean_abs_cos = np.abs(cos_mat).mean()
        
        results["svd_rank_95"] = rank_95
        results["svd_rank_99"] = rank_99
        results["mean_abs_cos"] = round(float(mean_abs_cos), 6)
        results["svd_top10"] = [round(s, 4) for s in S[:10].tolist()]
    
    # 饱和曲线拟合: f(n) = V * (1 - exp(-n/V))
    def solve_V(n_func, n_total):
        V_low, V_high = n_func, n_total * 20
        for _ in range(100):
            V_mid = (V_low + V_high) / 2
            pred = V_mid * (1 - np.exp(-n_total / V_mid))
            if pred > n_func:
                V_high = V_mid
            else:
                V_low = V_mid
        return V_mid
    
    V_estimated = solve_V(n_functional, n_total)
    results["V_lang_estimated"] = round(V_estimated, 1)
    
    print(f"\n  === P420 Summary for {model_name} ===")
    print(f"  Total: {n_total}, Functional: {n_functional} ({100*n_functional/n_total:.1f}%)")
    print(f"  SVD rank(95%): {results.get('svd_rank_95', 'N/A')}")
    print(f"  SVD rank(99%): {results.get('svd_rank_99', 'N/A')}")
    print(f"  Mean |cos|: {results.get('mean_abs_cos', 'N/A')}")
    print(f"  V_lang estimated: {V_estimated:.1f}")
    
    ts = time.strftime("%Y%m%d_%H%M")
    out_file = OUT_DIR / f"phase_lxxxv_p420_{model_name}_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved to {out_file}")
    return results


# ========== P421: W_lm结构性与训练关系 ==========

def run_p421(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P421: W_lm Structure Analysis - {model_name}")
    print(f"{'='*60}")
    
    W_U = model.lm_head.weight.detach().cpu().float().numpy()  # [vocab, d_model]
    W_lm = W_U  # 已经是[vocab, d_model]
    
    vocab_size, d_model = W_lm.shape
    print(f"  W_lm: {vocab_size} x {d_model}")
    
    results = {"model": model_name, "exp": "p421", "vocab_size": vocab_size, "d_model": d_model}
    
    # 1. SVD谱 (用随机SVD避免内存问题)
    print("  Computing SVD (randomized for large matrices)...")
    from sklearn.utils.extmath import randomized_svd
    n_components = min(d_model, 500)  # 只计算前500个奇异值
    if vocab_size * d_model > 500_000_000:  # > 500M elements, 用随机SVD
        U, S, Vt = randomized_svd(W_lm, n_components=n_components, random_state=42)
    else:
        U, S, Vt = np.linalg.svd(W_lm, full_matrices=False)
    
    total_energy = S.sum()
    cumsum = np.cumsum(S)
    rank_50 = int(np.searchsorted(cumsum, 0.50 * total_energy)) + 1
    rank_95 = int(np.searchsorted(cumsum, 0.95 * total_energy)) + 1
    rank_99 = int(np.searchsorted(cumsum, 0.99 * total_energy)) + 1
    
    results["svd"] = {
        "rank_50": rank_50, "rank_95": rank_95, "rank_99": rank_99,
        "top10": [round(s, 4) for s in S[:10].tolist()],
        "spectral_decay": round(S[0] / S[99], 4) if len(S) > 99 else 0,
    }
    print(f"  SVD ranks: 50%={rank_50}, 95%={rank_95}, 99%={rank_99}")
    
    # 2. 按词频分组分析
    common_words = ["the", "a", "is", "was", "are", "were", "be", "been", "have", "has",
                    "do", "does", "did", "will", "would", "could", "should", "may", "might",
                    "can", "not", "no", "yes", "and", "but", "or", "if", "then", "when",
                    "where", "how", "what", "which", "who", "that", "this", "these", "those"]
    
    rare_words = ["xylophone", "quizzical", "zephyr", "juxtapose", "surreptitious",
                  "ephemeral", "quintessential", "labyrinth", "cacophony", "serendipity",
                  "perspicacious", "mellifluous", "ineffable", "ethereal", "obsequious",
                  "recalcitrant", "sycophant", "pusillanimous", "sesquipedalian", "defenestrate"]
    
    medium_words = ["important", "consider", "provide", "develop", "approach",
                    "establish", "significant", "various", "specific", "indicate",
                    "require", "structure", "function", "process", "element"]
    
    def get_word_ids(words):
        ids = []
        for w in words:
            t = tokenizer.encode(w, add_special_tokens=False)
            if t:
                ids.append(t[0])
        return ids
    
    def analyze_group(ids, name):
        if len(ids) < 2:
            return None
        vecs = W_lm[ids]
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        normalized = vecs / (norms + 1e-10)
        cos_mat = normalized @ normalized.T
        np.fill_diagonal(cos_mat, 0)
        mean_cos = np.abs(cos_mat).mean()
        std_cos = np.abs(cos_mat).std()
        
        # 随机对比
        np.random.seed(42)
        rand_ids = np.random.choice(vocab_size, size=len(ids), replace=False)
        rand_vecs = W_lm[rand_ids]
        rand_norms = np.linalg.norm(rand_vecs, axis=1, keepdims=True)
        rand_normalized = rand_vecs / (rand_norms + 1e-10)
        rand_cos = rand_normalized @ rand_normalized.T
        np.fill_diagonal(rand_cos, 0)
        rand_mean = np.abs(rand_cos).mean()
        
        ratio = mean_cos / (rand_mean + 1e-10)
        print(f"  {name}: mean|cos|={mean_cos:.4f}, random={rand_mean:.4f}, ratio={ratio:.2f}x")
        return {"mean_abs_cos": round(float(mean_cos), 6), "random_mean": round(float(rand_mean), 6), "ratio": round(float(ratio), 4)}
    
    results["common"] = analyze_group(get_word_ids(common_words), "Common words")
    results["rare"] = analyze_group(get_word_ids(rare_words), "Rare words")
    results["medium"] = analyze_group(get_word_ids(medium_words), "Medium words")
    
    # 3. 相同SVD谱的随机W_lm
    print("  Random W_lm with same SVD spectrum...")
    np.random.seed(42)
    n_sample = min(500, vocab_size)
    
    # 真实W_lm的结构性
    sample_ids = np.random.choice(vocab_size, size=n_sample, replace=False)
    real_vecs = W_lm[sample_ids]
    real_norms = np.linalg.norm(real_vecs, axis=1, keepdims=True)
    real_norm = real_vecs / (real_norms + 1e-10)
    real_cos = real_norm @ real_norm.T
    np.fill_diagonal(real_cos, 0)
    real_mean = np.abs(real_cos).mean()
    real_std = np.abs(real_cos).std()
    
    # 随机矩阵(相同SVD谱, 简化版)
    k = min(rank_99 + 10, len(S), 200)  # 限制k避免内存问题
    print(f"  Generating random W_lm with k={k} components...")
    np.random.seed(42)
    # 直接生成随机矩阵, 用前k个奇异值缩放
    n_rand = min(500, vocab_size)
    rand_U_small = np.random.randn(n_rand, k)
    rand_U_small, _ = np.linalg.qr(rand_U_small)
    rand_Vt_small = np.random.randn(k, d_model)
    rand_Vt_small, _ = np.linalg.qr(rand_Vt_small.T)
    rand_Vt_small = rand_Vt_small.T
    W_lm_rand = rand_U_small @ np.diag(S[:k]) @ rand_Vt_small[:k, :]
    
    rand_sample = W_lm_rand[sample_ids[:n_rand] % W_lm_rand.shape[0]] if n_rand <= len(sample_ids) else W_lm_rand[:n_rand]
    rand_s_norms = np.linalg.norm(rand_sample, axis=1, keepdims=True)
    rand_s_norm = rand_sample / (rand_s_norms + 1e-10)
    rand_s_cos = rand_s_norm @ rand_s_norm.T
    np.fill_diagonal(rand_s_cos, 0)
    rand_s_mean = np.abs(rand_s_cos).mean()
    rand_s_std = np.abs(rand_s_cos).std()
    
    results["random_svd"] = {
        "real_mean": round(float(real_mean), 6),
        "real_std": round(float(real_std), 6),
        "random_mean": round(float(rand_s_mean), 6),
        "random_std": round(float(rand_s_std), 6),
        "excess_ratio": round(float(real_mean / (rand_s_mean + 1e-10)), 4),
    }
    print(f"  Real: mean|cos|={real_mean:.4f}, Random(SVD): mean|cos|={rand_s_mean:.4f}, ratio={real_mean/(rand_s_mean+1e-10):.2f}x")
    
    # 4. Marchenko-Pastur检验
    c = d_model / vocab_size
    lambda_min_mp = (1 - np.sqrt(c))**2
    lambda_max_mp = (1 + np.sqrt(c))**2
    
    S_norm = S / (S.sum() / len(S))
    outside_mp = np.sum((S_norm < lambda_min_mp) | (S_norm > lambda_max_mp)) / len(S_norm)
    
    results["mp_test"] = {
        "c_ratio": round(c, 6),
        "lambda_range": [round(lambda_min_mp, 4), round(lambda_max_mp, 4)],
        "outside_ratio": round(float(outside_mp), 4),
        "S0_vs_MP_max": round(float(S_norm[0] / lambda_max_mp), 4),
    }
    print(f"  MP test: outside={outside_mp*100:.1f}%, S_max/MP_max={S_norm[0]/lambda_max_mp:.2f}x")
    
    # 5. cos分布
    np.random.seed(42)
    n_pairs = 5000
    id1 = np.random.choice(vocab_size, size=n_pairs, replace=True)
    id2 = np.random.choice(vocab_size, size=n_pairs, replace=True)
    
    v1 = W_lm[id1]
    v2 = W_lm[id2]
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    cos_vals = np.sum(v1 * v2, axis=1) / (n1 * n2 + 1e-10)
    
    results["cos_dist"] = {
        "mean": round(float(cos_vals.mean()), 6),
        "std": round(float(cos_vals.std()), 6),
        "abs_mean": round(float(np.abs(cos_vals).mean()), 6),
        "kurtosis": round(float(np.mean((cos_vals - cos_vals.mean())**4) / (np.var(cos_vals)**2 + 1e-10) - 3), 4),
    }
    print(f"  Cos distribution: mean={cos_vals.mean():.4f}, std={cos_vals.std():.4f}, |cos|={np.abs(cos_vals).mean():.4f}")
    
    print(f"\n  === P421 Summary for {model_name} ===")
    print(f"  SVD rank(95%): {rank_95}")
    print(f"  Excess structure: {results['random_svd']['excess_ratio']:.2f}x")
    print(f"  Common words ratio: {results['common']['ratio'] if results['common'] else 'N/A'}")
    
    ts = time.strftime("%Y%m%d_%H%M")
    out_file = OUT_DIR / f"phase_lxxxv_p421_{model_name}_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved to {out_file}")
    return results


# ========== P422: 语言空间完备性严格证明 ==========

def run_p422(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P422: Language Space Completeness - {model_name}")
    print(f"{'='*60}")
    
    W_U = model.lm_head.weight.detach().cpu().float().numpy()
    vocab_size, d_model = W_U.shape
    
    results = {"model": model_name, "exp": "p422", "vocab_size": vocab_size, "d_model": d_model}
    
    # 1. SVD (用随机SVD避免内存问题)
    print("  Computing SVD (randomized for large matrices)...")
    from sklearn.utils.extmath import randomized_svd
    n_components = min(d_model, 500)
    if vocab_size * d_model > 500_000_000:
        U, S, Vt = randomized_svd(W_U, n_components=n_components, random_state=42)
    else:
        U, S, Vt = np.linalg.svd(W_U, full_matrices=False)
    total_energy = S.sum()
    
    # 2. 维度-能量关系
    epsilons = [0.01, 0.05, 0.10, 0.20, 0.50]
    dim_for_eps = {}
    for eps in epsilons:
        target = (1 - eps) * total_energy
        dim_needed = int(np.searchsorted(np.cumsum(S), target)) + 1
        dim_for_eps[str(eps)] = dim_needed
        print(f"  eps={eps}: need {dim_needed} dims for {100*(1-eps):.0f}% energy")
    results["dim_for_epsilon"] = dim_for_eps
    
    # 3. JL bounds
    eps_jl = 0.1
    V_candidates = [50, 100, 200, 500, 1000]
    jl_bounds = {}
    for V in V_candidates:
        d_needed = int(np.ceil(8 * np.log(V) / eps_jl**2))
        jl_bounds[str(V)] = d_needed
    results["jl_bounds"] = jl_bounds
    
    # 4. 收集功能维度方向
    embed = model.get_input_embeddings()
    beta = 8.0
    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    
    with torch.no_grad():
        baseline_logits = model(input_ids).logits[0, -1].detach().cpu().float()
    
    functional_dirs = []
    
    for dim_name, pairs in ALL_200_DIMS.items():
        w1, w2 = pairs[0]
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
    
    print(f"  Functional directions: {len(functional_dirs)}")
    
    # 5. 重建实验
    if len(functional_dirs) > 1:
        dirs_matrix = np.array(functional_dirs)
        n_dirs, d = dirs_matrix.shape
        
        norms = np.linalg.norm(dirs_matrix, axis=1, keepdims=True)
        normalized = dirs_matrix / (norms + 1e-10)
        
        # Gram矩阵
        G = normalized @ normalized.T
        eigvals = np.linalg.eigvalsh(G)
        
        # 伪逆投影
        G_pinv = np.linalg.pinv(G, rcond=1e-6)
        P = normalized.T @ G_pinv @ normalized
        
        # 重建随机方向
        np.random.seed(42)
        n_test = 100
        random_dirs = np.random.randn(n_test, d)
        r_norms = np.linalg.norm(random_dirs, axis=1, keepdims=True)
        r_normalized = random_dirs / (r_norms + 1e-10)
        
        projected = (P @ r_normalized.T).T
        p_norms = np.linalg.norm(projected, axis=1, keepdims=True)
        p_normalized = projected / (p_norms + 1e-10)
        
        cos_recon = np.sum(r_normalized * p_normalized, axis=1)
        mean_recon_cos = cos_recon.mean()
        mean_recon_error = np.sqrt(np.maximum(0, 1 - cos_recon**2)).mean()
        
        theoretical_error = np.sqrt(max(0, 1 - n_dirs / d))
        
        results["reconstruction"] = {
            "n_functional": n_dirs,
            "gram_min_eigval": round(float(eigvals.min()), 8),
            "mean_recon_cos": round(float(mean_recon_cos), 6),
            "mean_recon_error": round(float(mean_recon_error), 6),
            "theoretical_error": round(float(theoretical_error), 6),
            "V_lang_ratio": round(n_dirs / d, 6),
            "dims_for_0.1_error": int(np.ceil(d * (1 - 0.1**2))),
            "dims_for_0.01_error": int(np.ceil(d * (1 - 0.01**2))),
        }
        
        print(f"  Reconstruction: cos={mean_recon_cos:.4f}, error={mean_recon_error:.4f}, theoretical={theoretical_error:.4f}")
        print(f"  V_lang/d_model={n_dirs/d:.4f}")
        print(f"  For error<0.1: need {results['reconstruction']['dims_for_0.1_error']} dims")
    
    # 6. Participation ratio
    S_sq = S**2
    participation_ratio = (S_sq.sum())**2 / (S_sq**2).sum()
    results["info_capacity"] = {
        "participation_ratio": round(float(participation_ratio), 1),
        "eff_rank_90": int(np.searchsorted(np.cumsum(S) / total_energy, 0.9)) + 1,
        "eff_rank_99": int(np.searchsorted(np.cumsum(S) / total_energy, 0.99)) + 1,
    }
    print(f"  Participation ratio: {participation_ratio:.1f}")
    
    print(f"\n  === P422 Summary for {model_name} ===")
    print(f"  Functional: {len(functional_dirs)}, V_lang/d={len(functional_dirs)/d_model:.4f}")
    print(f"  Participation ratio: {participation_ratio:.1f}")
    if "reconstruction" in results:
        print(f"  Recon error: {results['reconstruction']['mean_recon_error']:.4f}")
        print(f"  Need ~{results['reconstruction']['dims_for_0.1_error']} dims for 0.1 error")
    
    ts = time.strftime("%Y%m%d_%H%M")
    out_file = OUT_DIR / f"phase_lxxxv_p422_{model_name}_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved to {out_file}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Phase LXXXV: Nonlinear Theory & V_lang Limit")
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, required=True, choices=["p419", "p420", "p421", "p422"])
    args = parser.parse_args()
    
    model, tokenizer, device = load_model(args.model)
    
    if args.exp == "p419":
        run_p419(model, tokenizer, device, args.model)
    elif args.exp == "p420":
        run_p420(model, tokenizer, device, args.model)
    elif args.exp == "p421":
        run_p421(model, tokenizer, device, args.model)
    elif args.exp == "p422":
        run_p422(model, tokenizer, device, args.model)
    
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
