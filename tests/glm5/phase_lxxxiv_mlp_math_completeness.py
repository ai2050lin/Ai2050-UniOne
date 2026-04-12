"""
Phase LXXXIV-P415/416/417/418: 暗能量数学机制与语言空间完备性
================================================================================

阶段G核心任务 - MLP暗能量的精确数学机制 + V_lang完备性证明:

P415: MLP权重SVD分析——为什么MLP放大信号5-10倍于Attention?
  - P411发现: 最后层MLP贡献5-10倍于Attention
  - 核心问题: MLP权重矩阵有什么特殊结构导致这种放大?
  - 方法: 
    1. 提取最后3层的MLP和Attention权重矩阵
    2. 对权重矩阵做SVD, 比较奇异值分布
    3. 计算信号在MLP和Attention中的放大因子: ||Wx||/||x||
    4. 分析dh在MLP权重空间中的投影

P416: V_lang上界搜索——系统搜索100+词对
  - P412发现: 55维中44-55个是功能维度
  - 核心问题: V_lang的饱和点在哪里?
  - 方法:
    1. 扩展到100+词对(增加更多语义领域)
    2. 计算功能维度的累计占比
    3. 用信息论方法估计V_lang的上界
    4. 检验功能维度之间的正交性是否饱和

P417: 信号聚焦机制——不同维度的信号增益分析
  - P410/P414发现: eff_L1差异200倍, Δlogit分布是高斯的
  - 核心问题: 为什么某些维度的信号增益远大于其他?
  - 方法:
    1. 对每个维度计算"信号增益"G = Δlogit_target / β
    2. 分析G与||W_diff||, cos(dh, W_diff)的关系
    3. 构建信号增益的理论公式
    4. 比较三模型的增益分布

P418: 语言空间完备性定理——V_lang维度上限的数学证明
  - 核心问题: V_lang的理论上界是什么?
  - 方法:
    1. 用随机投影理论分析: V个向量在d维空间中两两正交的条件
    2. Johnson-Lindenstrauss引理: m个向量ε-正交需要d=O(log m/ε²)
    3. 实际测量: 100+功能维度的正交性
    4. 推导V_lang ≤ C * hidden_dim (C是什么?)

实验模型: qwen3 -> glm4 -> deepseek7b (串行, 避免GPU OOM)
"""

import argparse
import json
import time
import traceback
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

EXTRA_DIM_PAIRS = {
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

EXTENDED_DIM_PAIRS = {
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

# P416: 进一步扩展到100+词对 - 新增45个维度
ULTRA_DIM_PAIRS = {
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
    "completeness_dim": [("complete", "incomplete")],
    "correctness": [("correct", "incorrect")],
    "perfection": [("perfect", "imperfect")],
    "excellence": [("excellent", "mediocre")],
    "superiority": [("superior", "inferior")],
    "mastery": [("masterful", "amateurish")],
    "expertise": [("expert", "novice")],
}

ALL_100_DIMS = {**DIM_PAIRS, **EXTRA_DIM_PAIRS, **EXTENDED_DIM_PAIRS, **ULTRA_DIM_PAIRS}

PROMPTS = [
    "The apple is",
    "In the future, people will",
    "The scientist explained that",
    "When the rain stopped,",
    "She looked at the sky and",
    "The old man walked slowly",
]

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


def get_layers(model):
    if hasattr(model.model, "layers"):
        return list(model.model.layers)
    elif hasattr(model.model, "encoder"):
        return list(model.model.encoder.layers)
    raise ValueError("Cannot find layers")


# ========== P415: MLP权重SVD分析 ==========

def run_p415(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P415: MLP weight SVD analysis - {model_name}")
    print(f"{'='*60}")

    layers = get_layers(model)
    n_layers = len(layers)
    
    # 分析最后5层
    analyze_layers = list(range(max(0, n_layers - 5), n_layers))
    print(f"  Analyzing layers: {analyze_layers}")

    results_per_layer = {}
    
    for l in analyze_layers:
        layer = layers[l]
        layer_result = {"layer": l}
        
        # ★ 提取MLP权重 ★
        # 标准Transformer: mlp = up_proj + gate_proj + down_proj
        # 或者: mlp = fc1 + fc2 (2层MLP)
        # 或者: mlp = dense_h_to_4h + dense_4h_to_h
        
        mlp_weights = {}
        attn_weights = {}
        
        # MLP权重
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            # Qwen/GLM style: gate_proj, up_proj, down_proj
            if hasattr(mlp, "gate_proj"):
                mlp_weights["gate_proj"] = mlp.gate_proj.weight.detach().cpu().float().numpy()
            if hasattr(mlp, "up_proj"):
                mlp_weights["up_proj"] = mlp.up_proj.weight.detach().cpu().float().numpy()
            if hasattr(mlp, "down_proj"):
                mlp_weights["down_proj"] = mlp.down_proj.weight.detach().cpu().float().numpy()
            # Old style: fc1, fc2
            if hasattr(mlp, "fc1"):
                mlp_weights["fc1"] = mlp.fc1.weight.detach().cpu().float().numpy()
            if hasattr(mlp, "fc2"):
                mlp_weights["fc2"] = mlp.fc2.weight.detach().cpu().float().numpy()
            # GLM style: dense_h_to_4h, dense_4h_to_h
            if hasattr(mlp, "dense_h_to_4h"):
                mlp_weights["dense_h_to_4h"] = mlp.dense_h_to_4h.weight.detach().cpu().float().numpy()
            if hasattr(mlp, "dense_4h_to_h"):
                mlp_weights["dense_4h_to_h"] = mlp.dense_4h_to_h.weight.detach().cpu().float().numpy()
        
        # Attention权重
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            if hasattr(attn, "q_proj"):
                attn_weights["q_proj"] = attn.q_proj.weight.detach().cpu().float().numpy()
                attn_weights["k_proj"] = attn.k_proj.weight.detach().cpu().float().numpy()
                attn_weights["v_proj"] = attn.v_proj.weight.detach().cpu().float().numpy()
                attn_weights["o_proj"] = attn.o_proj.weight.detach().cpu().float().numpy()
            if hasattr(attn, "query"):
                attn_weights["query"] = attn.query.weight.detach().cpu().float().numpy()
                attn_weights["key"] = attn.key.weight.detach().cpu().float().numpy()
                attn_weights["value"] = attn.value.weight.detach().cpu().float().numpy()
                attn_weights["dense"] = attn.dense.weight.detach().cpu().float().numpy()
        elif hasattr(layer, "attention"):
            attn = layer.attention
            if hasattr(attn, "q_proj"):
                for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                    if hasattr(attn, name):
                        attn_weights[name] = getattr(attn, name).weight.detach().cpu().float().numpy()

        # ★ SVD分析MLP权重 ★
        print(f"\n  Layer {l}:")
        mlp_svd_info = {}
        for name, W in mlp_weights.items():
            # 计算Frobenius范数
            frob_norm = float(np.linalg.norm(W, 'fro'))
            shape = list(W.shape)
            
            # 信号放大因子: 对于随机x, E[||Wx||/||x||] ≈ ||W||_F / sqrt(d_in)
            spectral_norm = float(np.linalg.norm(W, 2)) if min(shape) <= 512 else float(np.sqrt(frob_norm**2 / min(shape)))
            
            # 对小矩阵做SVD
            if min(shape) <= 512:
                s = np.linalg.svd(W, compute_uv=False)
                top_sv = float(s[0])
                condition_number = float(s[0] / max(s[-1], 1e-10))
                sv_ratio_10 = float(s[0] / max(s[min(9, len(s)-1)], 1e-10))
            else:
                # 用随机投影估计
                n_proj = 100
                random_x = np.random.randn(shape[1], n_proj) / np.sqrt(shape[1])
                y = W @ random_x
                gain = float(np.mean(np.linalg.norm(y, axis=0) / np.linalg.norm(random_x, axis=0)))
                top_sv = gain
                condition_number = 0
                sv_ratio_10 = 0
            
            mlp_svd_info[name] = {
                "shape": shape,
                "frob_norm": frob_norm,
                "spectral_norm": top_sv,
                "gain_factor": frob_norm / np.sqrt(max(shape[1], 1)),
                "condition_number": condition_number,
            }
            print(f"    MLP {name}: shape={shape}, ||W||_F={frob_norm:.1f}, gain={frob_norm/np.sqrt(max(shape[1],1)):.3f}")
        
        # ★ SVD分析Attention权重 ★
        attn_svd_info = {}
        for name, W in attn_weights.items():
            frob_norm = float(np.linalg.norm(W, 'fro'))
            shape = list(W.shape)
            attn_svd_info[name] = {
                "shape": shape,
                "frob_norm": frob_norm,
                "gain_factor": frob_norm / np.sqrt(max(shape[1], 1)),
            }
            print(f"    Attn {name}: shape={shape}, ||W||_F={frob_norm:.1f}, gain={frob_norm/np.sqrt(max(shape[1],1)):.3f}")
        
        # ★ 关键: MLP down_proj的放大因子 vs Attention o_proj ★
        # MLP的信号路径: x -> up_proj(W1) -> activation -> down_proj(W2)
        # 有效增益 ≈ gain(W1) * gain(W2) * activation_factor
        # Attention的信号路径: x -> q,k,v -> attention -> o_proj
        # 有效增益 ≈ gain(o_proj) * attention_factor
        
        mlp_down_name = None
        for n in ["down_proj", "fc2", "dense_4h_to_h"]:
            if n in mlp_svd_info:
                mlp_down_name = n
                break
        
        attn_out_name = None
        for n in ["o_proj", "dense"]:
            if n in attn_svd_info:
                attn_out_name = n
                break
        
        mlp_up_name = None
        for n in ["up_proj", "gate_proj", "fc1", "dense_h_to_4h"]:
            if n in mlp_svd_info:
                mlp_up_name = n
                break
        
        if mlp_down_name and attn_out_name:
            mlp_gain = mlp_svd_info[mlp_down_name]["gain_factor"]
            attn_gain = attn_svd_info[attn_out_name]["gain_factor"]
            ratio = mlp_gain / max(attn_gain, 1e-10)
            
            # MLP总增益: up_proj * down_proj (近似)
            if mlp_up_name:
                mlp_total_gain = mlp_svd_info[mlp_up_name]["gain_factor"] * mlp_svd_info[mlp_down_name]["gain_factor"]
            else:
                mlp_total_gain = mlp_gain
            
            layer_result["mlp_gain"] = mlp_gain
            layer_result["attn_gain"] = attn_gain
            layer_result["mlp_attn_gain_ratio"] = ratio
            layer_result["mlp_total_gain"] = mlp_total_gain
            
            print(f"    ★ MLP/Attn gain ratio = {ratio:.2f}x, MLP total gain = {mlp_total_gain:.2f}")
        
        layer_result["mlp_svd"] = mlp_svd_info
        layer_result["attn_svd"] = attn_svd_info
        results_per_layer[str(l)] = layer_result

    # 总结
    print(f"\n  === P415 Summary ===")
    for l in analyze_layers:
        lr = results_per_layer[str(l)]
        ratio = lr.get("mlp_attn_gain_ratio", 0)
        mlp_total = lr.get("mlp_total_gain", 0)
        print(f"  L{l}: MLP/Attn gain ratio = {ratio:.2f}x, MLP total gain = {mlp_total:.2f}")

    results = {
        "n_layers_total": n_layers,
        "analyze_layers": analyze_layers,
        "results_per_layer": results_per_layer,
    }
    return results


# ========== P416: V_lang上界搜索——100+词对 ==========

def run_p416(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P416: V_lang upper bound search - 100+ dimensions - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    beta = 8.0

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()

    results_per_dim = {}
    functional_directions = {}  # 存储功能维度的方向向量用于正交性分析
    n_functional = 0
    n_structural = 0
    n_marginal = 0
    n_error = 0

    for dim_name, pairs in ALL_100_DIMS.items():
        pos, neg = pairs[0]
        try:
            direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
            if norm < 1e-8:
                n_error += 1
                continue

            pos_id = tokenizer.encode(pos, add_special_tokens=False)[0]
            neg_id = tokenizer.encode(neg, add_special_tokens=False)[0]

            w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
            inputs_embeds_int = inputs_embeds_base.clone()
            inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

            with torch.no_grad():
                logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()

            delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
            delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
            dlogit = delta_pos - delta_neg

            is_functional = abs(dlogit) > 1.0
            is_structural = abs(dlogit) < 0.5

            if is_functional:
                n_functional += 1
                functional_directions[dim_name] = direction
            elif is_structural:
                n_structural += 1
            else:
                n_marginal += 1

            results_per_dim[dim_name] = {
                "pos": pos, "neg": neg,
                "dlogit": dlogit,
                "is_functional": is_functional,
                "is_structural_only": is_structural,
            }

        except Exception as e:
            n_error += 1

    print(f"\n  === V_lang Upper Bound Summary ===")
    print(f"  Total tested: {len(results_per_dim)}")
    print(f"  Functional: {n_functional} ({100*n_functional/max(len(results_per_dim),1):.1f}%)")
    print(f"  Structural: {n_structural}")
    print(f"  Marginal: {n_marginal}")
    print(f"  Errors: {n_error}")

    # ★ 正交性分析: 功能维度之间 ★
    func_names = list(functional_directions.keys())
    n_func = len(func_names)
    if n_func >= 2:
        cos_matrix = np.zeros((n_func, n_func))
        for i in range(n_func):
            for j in range(i+1, n_func):
                cos_val = float(np.dot(functional_directions[func_names[i]], functional_directions[func_names[j]]))
                cos_matrix[i, j] = cos_val
                cos_matrix[j, i] = cos_val
        
        # 统计
        upper_tri = cos_matrix[np.triu_indices(n_func, k=1)]
        mean_cos = float(np.mean(np.abs(upper_tri)))
        max_cos = float(np.max(np.abs(upper_tri)))
        pct_above_03 = float(np.mean(np.abs(upper_tri) > 0.3) * 100)
        pct_above_05 = float(np.mean(np.abs(upper_tri) > 0.5) * 100)
        
        print(f"\n  === Orthogonality among {n_func} functional dims ===")
        print(f"  Mean |cos| = {mean_cos:.4f}")
        print(f"  Max |cos| = {max_cos:.4f}")
        print(f"  % pairs with |cos|>0.3 = {pct_above_03:.1f}%")
        print(f"  % pairs with |cos|>0.5 = {pct_above_05:.1f}%")
        
        # SVD分析: 功能维度方向矩阵的秩
        func_matrix = np.array([functional_directions[n] for n in func_names])  # [n_func, hidden_dim]
        if n_func > 0:
            s = np.linalg.svd(func_matrix, compute_uv=False)
            eff_rank_95 = int(np.sum(np.cumsum(s**2) / np.sum(s**2) < 0.95)) + 1
            eff_rank_99 = int(np.sum(np.cumsum(s**2) / np.sum(s**2) < 0.99)) + 1
            print(f"  SVD effective rank (95% var): {eff_rank_95}")
            print(f"  SVD effective rank (99% var): {eff_rank_99}")
            print(f"  Top 20 singular values: {[f'{v:.2f}' for v in s[:20]]}")

    # ★ Johnson-Lindenstrauss分析 ★
    # JL引理: m个点在d维空间中ε-保持距离, 需要 d = O(log(m)/ε²)
    # 对于我们的情况: m = n_func个功能维度, d = hidden_dim
    # ε = sqrt(8 * log(m) / d) (JL bound)
    hidden_dim = functional_directions[func_names[0]].shape[0] if n_func > 0 else 0
    if n_func > 1 and hidden_dim > 0:
        epsilon_jl = np.sqrt(8 * np.log(n_func) / hidden_dim)
        # 实际测量的ε
        actual_epsilon = np.sqrt(np.mean((1 - np.abs(upper_tri))**2))  # 理想正交时cos=0
        print(f"\n  === Johnson-Lindenstrauss Analysis ===")
        print(f"  n_func = {n_func}, hidden_dim = {hidden_dim}")
        print(f"  JL bound ε = sqrt(8*log({n_func})/{hidden_dim}) = {epsilon_jl:.4f}")
        print(f"  Actual ε (from cos distribution) = {actual_epsilon:.4f}")
        print(f"  JL is {'tight' if actual_epsilon < epsilon_jl else 'loose'}")

    results = {
        "n_total": len(results_per_dim),
        "n_functional": n_functional,
        "n_structural": n_structural,
        "n_marginal": n_marginal,
        "n_error": n_error,
        "mean_cos_functional": mean_cos if n_func >= 2 else 0,
        "max_cos_functional": max_cos if n_func >= 2 else 0,
        "eff_rank_95": eff_rank_95 if n_func > 0 else 0,
        "eff_rank_99": eff_rank_99 if n_func > 0 else 0,
        "results_per_dim": results_per_dim,
    }
    return results


# ========== P417: 信号增益分析 ==========

def run_p417(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P417: Signal gain analysis - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()
    vocab_size, hidden_dim = W_lm.shape

    layers = get_layers(model)
    n_layers = len(layers)

    # 对55个维度分析信号增益
    all_dims = {**DIM_PAIRS, **EXTRA_DIM_PAIRS, **EXTENDED_DIM_PAIRS}
    beta = 8.0

    dim_info = {}
    for name, pairs in all_dims.items():
        pos, neg = pairs[0]
        try:
            direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
            if norm < 1e-8:
                continue
            w_pos_raw = get_w_lm(model, tokenizer, pos)[1]
            w_neg_raw = get_w_lm(model, tokenizer, neg)[1]
            w_diff = w_pos_raw - w_neg_raw
            w_diff_norm = float(np.linalg.norm(w_diff))
            pos_id = tokenizer.encode(pos, add_special_tokens=False)[0]
            neg_id = tokenizer.encode(neg, add_special_tokens=False)[0]
            dim_info[name] = {
                "direction": direction, "w_diff": w_diff, "w_diff_norm": w_diff_norm,
                "pos_id": pos_id, "neg_id": neg_id,
                "w_pos_raw": w_pos_raw, "w_neg_raw": w_neg_raw,
            }
        except:
            continue

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线
    captured_base = {}
    def make_hook(storage, key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().float()
            else:
                storage[key] = output.detach().float()
        return hook

    handle = layers[-1].register_forward_hook(make_hook(captured_base, "last"))
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()
    handle.remove()
    h_base = captured_base["last"][0, -1, :].cpu().numpy()

    results_per_dim = {}
    for dim_name in dim_info:
        direction = dim_info[dim_name]["direction"]
        w_diff = dim_info[dim_name]["w_diff"]
        w_diff_norm = dim_info[dim_name]["w_diff_norm"]
        pos_id = dim_info[dim_name]["pos_id"]
        neg_id = dim_info[dim_name]["neg_id"]

        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        captured_int = {}
        handle = layers[-1].register_forward_hook(make_hook(captured_int, "last"))
        with torch.no_grad():
            logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()
        handle.remove()
        h_int = captured_int["last"][0, -1, :].cpu().numpy()

        dh = h_int - h_base
        dh_norm = float(np.linalg.norm(dh))

        # ★ 信号增益 G = Δlogit_target / β ★
        delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
        delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
        dlogit = delta_pos - delta_neg
        G = dlogit / beta

        # cos(dh, W_diff)
        cos_dh_wdiff = float(np.dot(dh, w_diff) / max(dh_norm * w_diff_norm, 1e-10)) if dh_norm > 1e-8 else 0

        # 理论增益: G_theory = ||W_diff|| * cos(dh, W_diff) * dh_norm / beta
        # 简化: G = W_diff · dh / beta
        G_from_dh = float(np.dot(w_diff, dh)) / beta

        # MLP vs Attention贡献的增益
        # 我们用P411的结果: mlp_ratio ≈ 0.8-0.9 for DS7B
        # 但这里我们直接测量

        # W_lm行向量统计
        delta_logits = W_lm @ dh
        total_energy_L1 = float(np.sum(np.abs(delta_logits)))
        eff_L1 = abs(dlogit) / max(total_energy_L1, 1e-10)

        results_per_dim[dim_name] = {
            "dlogit": dlogit,
            "G": G,
            "G_from_dh": G_from_dh,
            "dh_norm": dh_norm,
            "cos_dh_wdiff": cos_dh_wdiff,
            "w_diff_norm": w_diff_norm,
            "eff_L1": eff_L1,
            "total_energy_L1": total_energy_L1,
        }

    # ★ 分析增益G的分布 ★
    G_values = [r["G"] for r in results_per_dim.values()]
    abs_G = [abs(g) for g in G_values]
    dh_norms = [r["dh_norm"] for r in results_per_dim.values()]
    cos_values = [abs(r["cos_dh_wdiff"]) for r in results_per_dim.values()]
    w_diff_norms = [r["w_diff_norm"] for r in results_per_dim.values()]
    eff_L1s = [r["eff_L1"] for r in results_per_dim.values()]

    print(f"\n  === Signal Gain Distribution ===")
    print(f"  |G|: mean={np.mean(abs_G):.3f}, std={np.std(abs_G):.3f}, max={np.max(abs_G):.3f}")
    print(f"  dh_norm: mean={np.mean(dh_norms):.1f}, std={np.std(dh_norms):.1f}")
    print(f"  |cos(dh,W_diff)|: mean={np.mean(cos_values):.4f}, std={np.std(cos_values):.4f}")
    print(f"  ||W_diff||: mean={np.mean(w_diff_norms):.3f}, std={np.std(w_diff_norms):.3f}")
    print(f"  eff_L1: mean={np.mean(eff_L1s):.2e}, std={np.std(eff_L1s):.2e}")

    # ★ 相关性分析: G与||W_diff||, cos, dh_norm的关系 ★
    if len(abs_G) > 5:
        # G ≈ ||W_diff|| * cos(dh, W_diff) * dh_norm / beta ?
        # 实际上 G = Δlogit/β = (W_diff · dh)/β
        # 所以 G = ||W_diff|| * dh_norm * cos(dh, W_diff) / beta
        predicted_G = np.array(w_diff_norms) * np.array(dh_norms) * np.array(cos_values) / beta
        actual_G = np.array(abs_G)
        correlation = float(np.corrcoef(predicted_G, actual_G)[0, 1])
        mean_ratio = float(np.mean(actual_G / np.maximum(predicted_G, 1e-10)))
        
        print(f"\n  === Gain Prediction ===")
        print(f"  G = ||W_diff|| * ||dh|| * |cos| / β")
        print(f"  Correlation(predicted, actual) = {correlation:.4f}")
        print(f"  Mean ratio(actual/predicted) = {mean_ratio:.4f}")

        # 更简单的模型: G ≈ ||W_diff|| * dh_norm / beta (忽略cos)
        predicted_G_simple = np.array(w_diff_norms) * np.array(dh_norms) / beta
        correlation_simple = float(np.corrcoef(predicted_G_simple, actual_G)[0, 1])
        print(f"  G ≈ ||W_diff|| * ||dh|| / β: correlation = {correlation_simple:.4f}")

        # G与||W_diff||的相关性
        corr_wdiff = float(np.corrcoef(np.array(w_diff_norms), actual_G)[0, 1])
        print(f"  G vs ||W_diff||: correlation = {corr_wdiff:.4f}")
        
        # G与dh_norm的相关性
        corr_dh = float(np.corrcoef(np.array(dh_norms), actual_G)[0, 1])
        print(f"  G vs ||dh||: correlation = {corr_dh:.4f}")
        
        # G与cos的相关性
        corr_cos = float(np.corrcoef(np.array(cos_values), actual_G)[0, 1])
        print(f"  G vs |cos|: correlation = {corr_cos:.4f}")

    # Top 10 by |G|
    sorted_dims = sorted(results_per_dim.items(), key=lambda x: abs(x[1]["G"]), reverse=True)
    print(f"\n  Top 10 by |G|:")
    for name, r in sorted_dims[:10]:
        print(f"    {name}: |G|={abs(r['G']):.3f}, ||W_diff||={r['w_diff_norm']:.3f}, cos={abs(r['cos_dh_wdiff']):.4f}")

    results = {
        "n_dims": len(results_per_dim),
        "mean_abs_G": float(np.mean(abs_G)),
        "std_abs_G": float(np.std(abs_G)),
        "correlation_wdiff": corr_wdiff if len(abs_G) > 5 else 0,
        "correlation_dh": corr_dh if len(abs_G) > 5 else 0,
        "correlation_cos": corr_cos if len(abs_G) > 5 else 0,
        "results_per_dim": results_per_dim,
    }
    return results


# ========== P418: 语言空间完备性定理 ==========

def run_p418(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P418: Language space completeness theorem - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()
    vocab_size, hidden_dim = W_lm.shape
    
    config = model.config
    hidden_size = getattr(config, "hidden_size", 0)
    num_heads = getattr(config, "num_attention_heads", 0)
    intermediate_size = getattr(config, "intermediate_size", 0)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    
    print(f"  Model: {model_name}")
    print(f"  hidden_dim = {hidden_dim} (from W_lm)")
    print(f"  hidden_size = {hidden_size} (from config)")
    print(f"  num_heads = {num_heads}, num_kv_heads = {num_kv_heads}")
    print(f"  intermediate_size = {intermediate_size}")
    print(f"  vocab_size = {vocab_size}")

    # ★ 1. W_lm的行空间结构 ★
    print(f"\n  === 1. W_lm Row Space Structure ===")
    # 采样1000行做完整SVD
    np.random.seed(42)
    n_sample = min(2000, vocab_size)
    idx = np.random.choice(vocab_size, n_sample, replace=False)
    W_sub = W_lm[idx]
    s = np.linalg.svd(W_sub, compute_uv=False)
    
    # 有效秩
    cumvar = np.cumsum(s**2) / np.sum(s**2)
    rank_50 = int(np.searchsorted(cumvar, 0.50) + 1)
    rank_90 = int(np.searchsorted(cumvar, 0.90) + 1)
    rank_95 = int(np.searchsorted(cumvar, 0.95) + 1)
    rank_99 = int(np.searchsorted(cumvar, 0.99) + 1)
    
    print(f"  Rank covering 50% variance: {rank_50}")
    print(f"  Rank covering 90% variance: {rank_90}")
    print(f"  Rank covering 95% variance: {rank_95}")
    print(f"  Rank covering 99% variance: {rank_99}")
    print(f"  Top 20 singular values: {[f'{v:.2f}' for v in s[:20]]}")
    print(f"  Power law fit: s_k ~ k^(-α)")
    # 拟合power law
    k_vals = np.arange(1, min(200, len(s)) + 1)
    log_k = np.log(k_vals)
    log_s = np.log(s[:len(k_vals)] + 1e-10)
    # 线性回归
    A = np.vstack([log_k, np.ones(len(log_k))]).T
    result = np.linalg.lstsq(A, log_s, rcond=None)
    alpha_pl = -result[0][0]
    print(f"  Power law exponent α = {alpha_pl:.3f}")

    # ★ 2. W_lm行向量的统计独立性 ★
    print(f"\n  === 2. Statistical Independence of Row Vectors ===")
    # 如果行向量是统计独立的(在hidden_dim维空间中), 
    # 那么两两cos的分布应该接近N(0, 1/d)
    n_pairs = 10000
    idx1 = np.random.choice(vocab_size, n_pairs, replace=True)
    idx2 = np.random.choice(vocab_size, n_pairs, replace=True)
    # 去掉自对
    mask = idx1 != idx2
    idx1, idx2 = idx1[mask], idx2[mask]
    
    w1 = W_lm[idx1]
    w2 = W_lm[idx2]
    norms1 = np.linalg.norm(w1, axis=1, keepdims=True)
    norms2 = np.linalg.norm(w2, axis=1, keepdims=True)
    cos_vals = np.sum(w1 * w2, axis=1) / (norms1.squeeze() * norms2.squeeze() + 1e-10)
    
    mean_cos = float(np.mean(cos_vals))
    std_cos = float(np.std(cos_vals))
    expected_std = float(1.0 / np.sqrt(hidden_dim))
    
    print(f"  Mean cos(random pairs) = {mean_cos:.6f} (expected ≈ 0)")
    print(f"  Std cos(random pairs) = {std_cos:.6f} (expected ≈ 1/sqrt(d) = {expected_std:.6f})")
    print(f"  Ratio actual/expected std = {std_cos/expected_std:.4f}")

    # ★ 3. V_lang上界的理论推导 ★
    print(f"\n  === 3. V_lang Upper Bound Derivation ===")
    # 定理: 如果m个单位向量在d维空间中两两|cos| < ε,
    #        则m ≤ d * (1 + 2/ε²) (Welch bound的变体)
    # 
    # 更精确: Johnson-Lindenstrauss引理
    # 对于m个点和ε-近似, 需要 d ≥ 8*log(m)/ε²
    # 反过来: 给定d, m的上界为 m ≤ exp(d*ε²/8)
    #
    # 对于我们的情况:
    # - d = hidden_dim
    # - ε = 实际测量的mean|cos| among functional dims
    # - 我们要找: 最多能有多少个近似正交的功能维度?

    # 从P416的结果获取(或估计)
    # 假设ε = 0.1 (90%正交)
    epsilon_vals = [0.05, 0.1, 0.15, 0.2, 0.3]
    for eps in epsilon_vals:
        # JL bound: d ≥ 8*log(m)/ε² → m ≤ exp(d*ε²/8)
        m_max_jl = int(np.exp(hidden_dim * eps**2 / 8))
        # Welch bound: m ≤ d*(1/ε² + 1)
        m_max_welch = int(hidden_dim * (1.0/eps**2 + 1))
        print(f"  ε={eps:.2f}: JL bound m ≤ {m_max_jl:,}, Welch bound m ≤ {m_max_welch:,}")

    # ★ 4. 实际V_lang维度与理论界的比较 ★
    print(f"\n  === 4. V_lang vs Theoretical Bounds ===")
    # 从P412/P416我们知道: GLM4有55个功能维度(100%)
    # 假设实际ε≈0.1 (功能维度之间平均cos≈0.1)
    actual_func_dims = {  # 从P412结果
        "qwen3": 44,
        "glm4": 55,
        "deepseek7b": 37,
    }
    actual_n = actual_func_dims.get(model_name, 44)
    
    # 用信息论方法估计上界
    # 每个功能维度贡献的信息量 ≈ log2(1 + SNR)
    # SNR = (Δlogit/σ_Δlogit)²
    # 总信息量 ≤ hidden_dim * log2(1 + ||W_lm||²_F / (hidden_dim * vocab_size))
    
    W_lm_F_sq = float(np.sum(W_lm**2))
    snr_total = W_lm_F_sq / (hidden_dim * vocab_size)
    info_capacity = hidden_dim * np.log2(1 + snr_total)
    
    print(f"  ||W_lm||_F^2 = {W_lm_F_sq:.1f}")
    print(f"  SNR_total = {snr_total:.6f}")
    print(f"  Information capacity = {info_capacity:.1f} bits")
    print(f"  Actual V_lang dimensions = {actual_n}")
    print(f"  V_lang/hidden_dim = {actual_n/hidden_dim:.4f}")

    # ★ 5. V_lang完备性条件 ★
    print(f"\n  === 5. Completeness Condition ===")
    # 如果V_lang是"完备的", 则任何语义方向都可以被功能维度张成
    # 检验: 用已知的n个功能维度, 能否重建一个随机方向?
    
    # 从P416获取功能维度方向(或用已知维度)
    func_dirs = {}
    for name, pairs in {**DIM_PAIRS, **EXTRA_DIM_PAIRS}.items():
        try:
            d, norm = get_dimension_direction(model, tokenizer, pairs[0][0], pairs[0][1])
            if norm > 1e-8:
                func_dirs[name] = d
        except:
            pass

    # 构建功能维度矩阵
    func_names = list(func_dirs.keys())
    func_matrix = np.array([func_dirs[n] for n in func_names])  # [n_func, hidden_dim]
    n_func = len(func_names)
    
    if n_func > 0:
        # 对随机方向做重建
        np.random.seed(42)
        n_random = 50
        random_dirs = np.random.randn(n_random, hidden_dim)
        random_dirs = random_dirs / np.linalg.norm(random_dirs, axis=1, keepdims=True)
        
        # 用功能维度的线性组合重建
        # func_matrix.T @ c ≈ random_dir → c = (func_matrix @ func_matrix.T)^{-1} @ func_matrix @ random_dir
        G = func_matrix @ func_matrix.T  # [n_func, n_func]
        try:
            G_inv = np.linalg.inv(G + 1e-6 * np.eye(n_func))
        except:
            G_inv = np.linalg.pinv(G)
        
        reconstruction_errors = []
        for i in range(n_random):
            coeffs = G_inv @ func_matrix @ random_dirs[i]
            reconstructed = func_matrix.T @ coeffs
            error = float(np.linalg.norm(reconstructed - random_dirs[i]) / np.linalg.norm(random_dirs[i]))
            reconstruction_errors.append(error)
        
        mean_error = float(np.mean(reconstruction_errors))
        print(f"  Functional dims: {n_func}")
        print(f"  Mean reconstruction error (random dirs): {mean_error:.4f}")
        print(f"  Completeness ratio: {1-mean_error:.4f}")
        print(f"  V_lang is {'complete' if mean_error < 0.1 else 'incomplete'} for random directions")

    results = {
        "hidden_dim": hidden_dim,
        "vocab_size": vocab_size,
        "W_lm_rank_95": rank_95,
        "W_lm_rank_99": rank_99,
        "power_law_alpha": alpha_pl,
        "mean_cos_random": mean_cos,
        "std_cos_random": std_cos,
        "expected_std_cos": expected_std,
        "info_capacity_bits": float(info_capacity),
        "actual_func_dims": actual_n,
        "n_func_tested": n_func,
        "mean_reconstruction_error": mean_error if n_func > 0 else 1.0,
    }
    return results


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["p415", "p416", "p417", "p418", "all"])
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())

    results = {}
    exps = ["p415", "p416", "p417", "p418"] if args.exp == "all" else [args.exp]

    for exp in exps:
        try:
            if exp == "p415":
                results["p415"] = run_p415(model, tokenizer, device, args.model)
            elif exp == "p416":
                results["p416"] = run_p416(model, tokenizer, device, args.model)
            elif exp == "p417":
                results["p417"] = run_p417(model, tokenizer, device, args.model)
            elif exp == "p418":
                results["p418"] = run_p418(model, tokenizer, device, args.model)
        except Exception as e:
            print(f"Error in {exp}: {e}")
            traceback.print_exc()
            results[exp] = {"error": str(e)}

    out_file = OUT_DIR / f"phase_lxxxiv_p415_418_{args.model}_{timestamp}.json"

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results = convert(results)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
