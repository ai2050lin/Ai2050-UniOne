"""
Phase LXXXVIII: 信号放大精确理论 + 权重空间突破
================================================================================

大任务A+C合并: 从"描述语言编码"跨越到"预测语言编码"

核心问题:
1. 为什么LN累积 ≠ 实际增益? J_components到底是什么?
2. 权重(W_q/k/v/o, W_up/down/gate)如何协同产生hidden state编码?
3. 信号在层间传播的精确数学模型是什么?

实验设计:
P431: LN×J_components精确分解
  - 方法: 在每层注入微小信号δ, 逐模块测量信号变化
  - 分解: dh_out = LN_factor × (dh_resid + dh_attn + dh_mlp)
  - 验证: gain(l) = ∏(LN_factor_l × J_l) vs actual gain
  - 核心: J_l = 1 + attn_jacobian + mlp_jacobian (线性化近似)

P432: 权重空间结构分析——信号放大的权重级来源
  - W_q/k/v/o的谱结构与信号路由的关系
  - W_up/down/gate的谱结构与知识检索的关系
  - W_ln (LayerNorm权重)的分布与信号放大的关系
  - 核心发现预期: W_ln.gamma的大值 → LN_factor大的权重级原因

P433: 信号传播统一公式与预测力验证
  - 建立公式: gain(L0→Lk) = ∏(LN_l × J_l) for l=0..k
  - 预测: 从P431/P432的参数预测增益
  - 验证: 预测增益 vs 实际增益 (R²)
  - 跨模型: 三模型是否共享统一公式形式

P434: 权重→激活因果方程
  - h_l = f(W_l, h_{l-1}) 的线性化展开
  - h_l ≈ (I + J_l) × LN_l × h_{l-1} + b_l
  - 验证: 这个方程能多精确预测h_l?

模型: qwen3 → glm4 → deepseek7b (串行, 避免GPU OOM)
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

OUT_DIR = Path("tests/glm5_temp")
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    "The apple is", "The scientist explained that",
    "If all humans are mortal and Socrates is human, then",
    "She felt deeply moved by", "In the future, people will",
    "The government announced that", "Research shows that the brain",
    "The evidence suggests that the conclusion",
    "She looked at the sky and", "Throughout history, civilizations have",
]

DIM_PAIRS = {
    "style": ("formal", "informal"),
    "logic": ("true", "false"),
    "sentiment": ("happy", "sad"),
    "tense": ("was", "is"),
    "certainty": ("definitely", "maybe"),
    "quantity": ("many", "few"),
    "complexity": ("complex", "simple"),
    "formality": ("professional", "casual"),
    "size": ("large", "small"),
    "strength": ("strong", "weak"),
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


def get_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    return None


def get_dimension_direction(model, tokenizer, w1, w2):
    embed = model.get_input_embeddings()
    t1 = tokenizer.encode(w1, add_special_tokens=False)
    t2 = tokenizer.encode(w2, add_special_tokens=False)
    if not t1 or not t2:
        return np.zeros(embed.weight.shape[1]), 0
    e1 = embed.weight[t1[0]].detach().cpu().float().numpy()
    e2 = embed.weight[t2[0]].detach().cpu().float().numpy()
    d = e1 - e2
    return d / (np.linalg.norm(d) + 1e-10), np.linalg.norm(d)


# ========== P431: LN×J_components精确分解 ==========

def run_p431(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P431: LN×J_components Precise Decomposition - {model_name}")
    print(f"{'='*60}")

    layers = get_layers(model)
    n_layers = len(layers)
    embed = model.get_input_embeddings()
    d_model = embed.weight.shape[1]
    beta = 1.0  # 使用小信号近似, 确保在线性区间

    prompts = PROMPTS[:10]

    results = {"model": model_name, "exp": "p431", "n_layers": n_layers, "d_model": d_model}

    # 对3个维度取平均
    test_dims = ["logic", "sentiment", "style"]
    all_layer_data = {str(l): [] for l in range(n_layers)}

    for dim_name in test_dims:
        w1, w2 = DIM_PAIRS[dim_name]
        direction, norm = get_dimension_direction(model, tokenizer, w1, w2)
        if norm < 1e-8:
            continue
        direction_t = torch.tensor(direction * beta, dtype=torch.float32)

        for pi, prompt in enumerate(prompts):
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids

            # ---- 核心方法: 逐层逐模块精确分解 ----
            # 对每一层:
            #   1. 测量baseline: 收集 layer_input, post_LN, post_attn, post_mlp, layer_output
            #   2. 测量intervened: 在embed注入信号, 收集同上
            #   3. 分解信号变化:
            #      dh_input = ||intervened_input - baseline_input||
            #      dh_post_LN = ||intervened_postLN - baseline_postLN||
            #      dh_post_attn = ||intervened_postAttn - baseline_postAttn||
            #      dh_post_mlp = ||intervened_postMLP - baseline_postMLP||
            #      dh_output = ||intervened_output - baseline_output||
            #   4. 增益分解:
            #      LN_gain = dh_post_LN / dh_input
            #      attn_jacobian = dh_post_attn / dh_post_LN (注意: 这是attn模块的雅可比)
            #      mlp_jacobian = dh_post_mlp / dh_attn_residual (MLP模块的雅可比)
            #      residual_jacobian = dh_output / dh_mlp_residual

            for kl in range(n_layers):
                layer = layers[kl]

                # ---- Baseline 收集 ----
                b_data = {}

                def make_pre_hook(store, key):
                    def hook_fn(module, args):
                        if isinstance(args, tuple):
                            store[key] = args[0].detach().cpu().float()
                        else:
                            store[key] = args.detach().cpu().float()
                        return args
                    return hook_fn

                def make_post_hook(store, key):
                    def hook_fn(module, input, output):
                        o = output[0] if isinstance(output, tuple) else output
                        store[key] = o.detach().cpu().float()
                    return hook_fn

                hooks_b = []
                # layer input (pre-LN)
                hooks_b.append(layer.register_forward_pre_hook(make_pre_hook(b_data, "layer_input")))

                # 判断LN位置——查找input_layernorm
                ln_hooks_added = False
                if hasattr(layer, "input_layernorm"):
                    hooks_b.append(layer.input_layernorm.register_forward_hook(make_post_hook(b_data, "post_input_ln")))
                    ln_hooks_added = True
                elif hasattr(layer, "ln_1"):
                    hooks_b.append(layer.ln_1.register_forward_hook(make_post_hook(b_data, "post_input_ln")))
                    ln_hooks_added = True

                # self_attn output
                if hasattr(layer, "self_attn"):
                    hooks_b.append(layer.self_attn.register_forward_hook(make_post_hook(b_data, "attn_out")))

                # post-attention LN (if exists)
                if hasattr(layer, "post_attention_layernorm"):
                    hooks_b.append(layer.post_attention_layernorm.register_forward_hook(make_post_hook(b_data, "post_attn_ln")))
                elif hasattr(layer, "ln_2"):
                    hooks_b.append(layer.ln_2.register_forward_hook(make_post_hook(b_data, "post_attn_ln")))

                # mlp output
                if hasattr(layer, "mlp"):
                    hooks_b.append(layer.mlp.register_forward_hook(make_post_hook(b_data, "mlp_out")))

                # post-MLP LN (if exists)
                if hasattr(layer, "post_feed_forward_layernorm"):
                    hooks_b.append(layer.post_feed_forward_layernorm.register_forward_hook(make_post_hook(b_data, "post_mlp_ln")))

                # layer output
                # 我们用post_mlp_ln的输出作为layer_output近似
                # 或者直接在下一层input测量

                with torch.no_grad():
                    _ = model(input_ids)
                for h in hooks_b:
                    h.remove()

                # ---- Intervened 收集 ----
                i_data = {}
                hooks_i = []
                hooks_i.append(layer.register_forward_pre_hook(make_pre_hook(i_data, "layer_input")))

                if hasattr(layer, "input_layernorm"):
                    hooks_i.append(layer.input_layernorm.register_forward_hook(make_post_hook(i_data, "post_input_ln")))
                elif hasattr(layer, "ln_1"):
                    hooks_i.append(layer.ln_1.register_forward_hook(make_post_hook(i_data, "post_input_ln")))

                if hasattr(layer, "self_attn"):
                    hooks_i.append(layer.self_attn.register_forward_hook(make_post_hook(i_data, "attn_out")))

                if hasattr(layer, "post_attention_layernorm"):
                    hooks_i.append(layer.post_attention_layernorm.register_forward_hook(make_post_hook(i_data, "post_attn_ln")))
                elif hasattr(layer, "ln_2"):
                    hooks_i.append(layer.ln_2.register_forward_hook(make_post_hook(i_data, "post_attn_ln")))

                if hasattr(layer, "mlp"):
                    hooks_i.append(layer.mlp.register_forward_hook(make_post_hook(i_data, "mlp_out")))

                if hasattr(layer, "post_feed_forward_layernorm"):
                    hooks_i.append(layer.post_feed_forward_layernorm.register_forward_hook(make_post_hook(i_data, "post_mlp_ln")))

                # embed注入
                def inj_hook_fn(module, input, output, d=direction_t):
                    modified = output.clone()
                    modified[0, -1, :] += d.to(output.device)
                    return modified

                h_embed = embed.register_forward_hook(inj_hook_fn)
                with torch.no_grad():
                    _ = model(input_ids)
                h_embed.remove()
                for h in hooks_i:
                    h.remove()

                # ---- 信号分解 ----
                def diff_norm(a, b):
                    if a is not None and b is not None:
                        return (a[0, -1, :] - b[0, -1, :]).norm().item()
                    return None

                def diff_cos(a, b):
                    if a is not None and b is not None:
                        da = a[0, -1, :] - b[0, -1, :]
                        # 方向保持度: cos(intervened_delta, baseline_delta)
                        # 如果cos接近1, 信号方向不变(线性); 远离1则非线性
                        return 1.0  # 简化: 后面单独计算
                    return None

                dh_input = diff_norm(i_data.get("layer_input"), b_data.get("layer_input"))
                dh_post_ln = diff_norm(i_data.get("post_input_ln"), b_data.get("post_input_ln"))
                dh_attn = diff_norm(i_data.get("attn_out"), b_data.get("attn_out"))
                dh_post_attn_ln = diff_norm(i_data.get("post_attn_ln"), b_data.get("post_attn_ln"))
                dh_mlp = diff_norm(i_data.get("mlp_out"), b_data.get("mlp_out"))
                dh_post_mlp_ln = diff_norm(i_data.get("post_mlp_ln"), b_data.get("post_mlp_ln"))

                # 增益分解
                ln_gain = dh_post_ln / dh_input if dh_input and dh_post_ln and dh_input > 1e-10 else None
                # attn模块的雅可比: 需要知道attn的输入(post_input_ln)到attn输出的增益
                # 但attn的输入包含residual + LN(x), 我们用近似:
                # residual_flow: dh_attn_residual = dh_post_ln (attn输入约等于post_LN输出+residual)
                # 简化: attn_jacobian = dh_attn / dh_post_ln
                attn_jacobian = dh_attn / dh_post_ln if dh_post_ln and dh_attn and dh_post_ln > 1e-10 else None
                # mlp_jacobian = dh_mlp / dh_post_attn_ln (MLP输入约等于post_attn_ln输出+residual)
                mlp_jacobian = dh_mlp / dh_post_attn_ln if dh_post_attn_ln and dh_mlp and dh_post_attn_ln > 1e-10 else None

                # 信号方向的余弦相似度——衡量非线性程度
                def dir_cos(a, b):
                    if a is not None and b is not None:
                        da = a[0, -1, :] - b[0, -1, :]
                        na = da.norm().item()
                        if na < 1e-10:
                            return 1.0
                        # 与输入方向的余弦
                        di = i_data.get("layer_input")
                        bi = b_data.get("layer_input")
                        if di is not None and bi is not None:
                            din = (di[0, -1, :] - bi[0, -1, :])
                            ni = din.norm().item()
                            if ni > 1e-10:
                                return float(torch.nn.functional.cosine_similarity(da.unsqueeze(0), din.unsqueeze(0)).item())
                        return 1.0
                    return None

                ln_dir_cos = dir_cos(i_data.get("post_input_ln"), b_data.get("post_input_ln"))
                attn_dir_cos = dir_cos(i_data.get("attn_out"), b_data.get("attn_out"))
                mlp_dir_cos = dir_cos(i_data.get("mlp_out"), b_data.get("mlp_out"))

                # LayerNorm统计量
                std_val = b_data["layer_input"][0, -1, :].std().item() if b_data.get("layer_input") is not None else 0
                ln_factor_theoretical = 1.0 / std_val if std_val > 1e-10 else 0

                all_layer_data[str(kl)].append({
                    "dim": dim_name,
                    "prompt_idx": pi,
                    "dh_input": round(dh_input, 6) if dh_input else None,
                    "dh_post_ln": round(dh_post_ln, 6) if dh_post_ln else None,
                    "dh_attn": round(dh_attn, 6) if dh_attn else None,
                    "dh_post_attn_ln": round(dh_post_attn_ln, 6) if dh_post_attn_ln else None,
                    "dh_mlp": round(dh_mlp, 6) if dh_mlp else None,
                    "dh_post_mlp_ln": round(dh_post_mlp_ln, 6) if dh_post_mlp_ln else None,
                    "ln_gain": round(ln_gain, 4) if ln_gain else None,
                    "attn_jacobian": round(attn_jacobian, 4) if attn_jacobian else None,
                    "mlp_jacobian": round(mlp_jacobian, 4) if mlp_jacobian else None,
                    "ln_factor_theoretical": round(ln_factor_theoretical, 4),
                    "std_input": round(std_val, 6),
                    "ln_dir_cos": round(ln_dir_cos, 4) if ln_dir_cos else None,
                    "attn_dir_cos": round(attn_dir_cos, 4) if attn_dir_cos else None,
                    "mlp_dir_cos": round(mlp_dir_cos, 4) if mlp_dir_cos else None,
                })

    # 汇总: 逐层平均
    layer_summary = {}
    for l in range(n_layers):
        samples = all_layer_data[str(l)]
        if not samples:
            continue
        avg_ln_gain = np.mean([s["ln_gain"] for s in samples if s["ln_gain"] is not None])
        avg_attn_j = np.mean([s["attn_jacobian"] for s in samples if s["attn_jacobian"] is not None])
        avg_mlp_j = np.mean([s["mlp_jacobian"] for s in samples if s["mlp_jacobian"] is not None])
        avg_ln_factor = np.mean([s["ln_factor_theoretical"] for s in samples])
        avg_ln_dir = np.mean([s["ln_dir_cos"] for s in samples if s["ln_dir_cos"] is not None])
        avg_attn_dir = np.mean([s["attn_dir_cos"] for s in samples if s["attn_dir_cos"] is not None])
        avg_mlp_dir = np.mean([s["mlp_dir_cos"] for s in samples if s["mlp_dir_cos"] is not None])

        # J_components = 1 + attn_j * (attn_resid_weight) + mlp_j * (mlp_resid_weight)
        # 简化: J = 1 + attn_j * α + mlp_j * β, 其中α,β是残差连接权重
        # 在Pre-LN架构中: output = x + attn(LN(x)) + mlp(LN(x+attn(LN(x))))
        # 所以 J ≈ 1 + attn_j + mlp_j (一阶近似)
        j_components = 1.0 + avg_attn_j + avg_mlp_j

        # 预测增益 = ln_gain × j_components
        predicted_gain = avg_ln_gain * j_components

        layer_summary[str(l)] = {
            "ln_gain": round(avg_ln_gain, 4),
            "attn_jacobian": round(avg_attn_j, 4),
            "mlp_jacobian": round(avg_mlp_j, 4),
            "j_components": round(j_components, 4),
            "predicted_gain": round(predicted_gain, 4),
            "ln_factor": round(avg_ln_factor, 4),
            "ln_dir_cos": round(avg_ln_dir, 4),
            "attn_dir_cos": round(avg_attn_dir, 4),
            "mlp_dir_cos": round(avg_mlp_dir, 4),
            "n_samples": len(samples),
        }

    # 累积增益预测
    cum_predicted = 1.0
    cum_actual = 1.0
    cum_results = {}
    for l in range(n_layers):
        ls = layer_summary.get(str(l), {})
        if not ls:
            continue
        cum_predicted *= ls.get("predicted_gain", 1.0)
        cum_results[str(l)] = {
            "cum_predicted_gain": round(cum_predicted, 4),
            "per_layer_predicted": ls.get("predicted_gain", 1.0),
            "per_layer_ln_gain": ls.get("ln_gain", 1.0),
            "per_layer_j": ls.get("j_components", 1.0),
        }

    results["layer_summary"] = layer_summary
    results["cumulative_gain"] = cum_results

    # 打印关键结果
    print(f"\n  === P431 Key Results for {model_name} ===")
    print(f"  Layer | LN_gain | attn_J | mlp_J | J_comp | pred_gain | ln_dir | attn_dir | mlp_dir")
    for l in [0, 1, 2, 3, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]:
        ls = layer_summary.get(str(l), {})
        if not ls:
            continue
        print(f"  L{l:3d} | {ls.get('ln_gain',0):7.2f} | {ls.get('attn_jacobian',0):6.3f} | {ls.get('mlp_jacobian',0):6.3f} | {ls.get('j_components',0):6.3f} | {ls.get('predicted_gain',0):9.2f} | {ls.get('ln_dir_cos',0):6.3f} | {ls.get('attn_dir_cos',0):8.3f} | {ls.get('mlp_dir_cos',0):7.3f}")

    # 累积预测
    print(f"\n  Cumulative Predicted Gain:")
    for l in [0, n_layers//4, n_layers//2, n_layers-1]:
        cr = cum_results.get(str(l), {})
        if cr:
            print(f"  L{l}: cum_predicted={cr['cum_predicted_gain']:.4f}")

    # 保存
    ts = time.strftime("%Y%m%d_%H%M")
    outf = OUT_DIR / f"phase_lxxxviii_p431_{model_name}_{ts}.json"
    with open(outf, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {outf}")

    return results


# ========== P432: 权重空间结构分析 ==========

def run_p432(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P432: Weight Space Structure Analysis - {model_name}")
    print(f"{'='*60}")

    layers = get_layers(model)
    n_layers = len(layers)
    embed = model.get_input_embeddings()
    d_model = embed.weight.shape[1]

    results = {"model": model_name, "exp": "p432", "n_layers": n_layers, "d_model": d_model}

    # 对关键层采样: L0, L1, L_n/4, L_n/2, L_3n/4, L_{n-2}, L_{n-1}
    sample_layers = [0, 1, n_layers//4, n_layers//2, 3*n_layers//4, max(0, n_layers-2), n_layers-1]
    sample_layers = sorted(set(sample_layers))

    weight_analysis = {}

    for kl in sample_layers:
        layer = layers[kl]
        layer_data = {"layer": kl}

        # ---- 1. LayerNorm权重分析 ----
        # input_layernorm的gamma和beta
        for ln_name in ["input_layernorm", "ln_1", "post_attention_layernorm", "ln_2", "post_feed_forward_layernorm"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                ln_data = {}
                if hasattr(ln, "weight"):
                    w = ln.weight.detach().cpu().float().numpy()
                    ln_data["gamma_mean"] = round(float(w.mean()), 6)
                    ln_data["gamma_std"] = round(float(w.std()), 6)
                    ln_data["gamma_max"] = round(float(w.max()), 6)
                    ln_data["gamma_min"] = round(float(w.min()), 6)
                    # gamma > 1 的比例 → 信号放大的权重级来源
                    ln_data["gamma_gt1_ratio"] = round(float((w > 1.0).mean()), 4)
                    # gamma的top-10值
                    top10 = np.sort(w)[-10:][::-1]
                    ln_data["gamma_top10"] = [round(float(x), 4) for x in top10]
                    # gamma的分布: [0,0.5), [0.5,1), [1,1.5), [1.5,2), [2,+)
                    bins = [0, 0.5, 1.0, 1.5, 2.0, float('inf')]
                    hist = np.histogram(w, bins=bins)[0]
                    ln_data["gamma_hist"] = [int(x) for x in hist]
                if hasattr(ln, "bias") and ln.bias is not None:
                    b = ln.bias.detach().cpu().float().numpy()
                    ln_data["beta_mean"] = round(float(b.mean()), 6)
                    ln_data["beta_std"] = round(float(b.std()), 6)
                layer_data[ln_name] = ln_data

        # ---- 2. Attention权重分析 ----
        if hasattr(layer, "self_attn"):
            attn = layer.self_attn
            attn_data = {}
            for w_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
                if hasattr(attn, w_name):
                    w = getattr(attn, w_name).weight.detach().cpu().float().numpy()
                    w_data = {}
                    w_data["shape"] = list(w.shape)
                    # SVD谱 (只取前50个奇异值)
                    if min(w.shape) > 50:
                        S = np.linalg.svd(w, compute_uv=False)[:50]
                    else:
                        S = np.linalg.svd(w, compute_uv=False)
                    w_data["svd_top10"] = [round(float(x), 4) for x in S[:10]]
                    w_data["svd_tail10"] = [round(float(x), 4) for x in S[-10:]]
                    w_data["condition_number"] = round(float(S[0] / max(S[-1], 1e-10)), 2)
                    # Frobenius norm
                    w_data["frobenius_norm"] = round(float(np.linalg.norm(w)), 4)
                    # 行均值和列均值(检测是否有结构)
                    w_data["row_norms_mean"] = round(float(np.linalg.norm(w, axis=1).mean()), 4)
                    w_data["col_norms_mean"] = round(float(np.linalg.norm(w, axis=0).mean()), 4)
                    # 核范数/ Frobenius范数 → 低秩度
                    nuclear_norm = float(S.sum())
                    frob_norm = float(np.linalg.norm(w))
                    w_data["low_rank_ratio"] = round(nuclear_norm / frob_norm, 4) if frob_norm > 0 else 0
                    attn_data[w_name] = w_data
            layer_data["self_attn"] = attn_data

        # ---- 3. MLP权重分析 ----
        if hasattr(layer, "mlp"):
            mlp = layer.mlp
            mlp_data = {}
            for w_name in ["up_proj", "down_proj", "gate_proj"]:
                if hasattr(mlp, w_name):
                    w = getattr(mlp, w_name).weight.detach().cpu().float().numpy()
                    w_data = {}
                    w_data["shape"] = list(w.shape)
                    # SVD谱
                    if min(w.shape) > 50:
                        S = np.linalg.svd(w, compute_uv=False)[:50]
                    else:
                        S = np.linalg.svd(w, compute_uv=False)
                    w_data["svd_top10"] = [round(float(x), 4) for x in S[:10]]
                    w_data["condition_number"] = round(float(S[0] / max(S[-1], 1e-10)), 2)
                    w_data["frobenius_norm"] = round(float(np.linalg.norm(w)), 4)
                    # 低秩度
                    nuclear_norm = float(S.sum())
                    frob_norm = float(np.linalg.norm(w))
                    w_data["low_rank_ratio"] = round(nuclear_norm / frob_norm, 4) if frob_norm > 0 else 0
                    # 行列范数比 (检测稀疏性)
                    row_norms = np.linalg.norm(w, axis=1)
                    col_norms = np.linalg.norm(w, axis=0)
                    w_data["row_norms_cv"] = round(float(row_norms.std() / max(row_norms.mean(), 1e-10)), 4)  # 变异系数
                    w_data["col_norms_cv"] = round(float(col_norms.std() / max(col_norms.mean(), 1e-10)), 4)
                    mlp_data[w_name] = w_data
            layer_data["mlp"] = mlp_data

        weight_analysis[str(kl)] = layer_data
        print(f"  Layer {kl}: analysis complete")

    results["weight_analysis"] = weight_analysis

    # ---- 4. 全局权重统计 ----
    # 收集所有层的LN gamma统计
    all_ln_gamma_stats = {}
    for l in range(n_layers):
        layer = layers[l]
        for ln_name in ["input_layernorm", "ln_1"]:
            if hasattr(layer, ln_name):
                ln = getattr(layer, ln_name)
                if hasattr(ln, "weight"):
                    w = ln.weight.detach().cpu().float().numpy()
                    if ln_name not in all_ln_gamma_stats:
                        all_ln_gamma_stats[ln_name] = {"means": [], "stds": [], "gt1_ratios": []}
                    all_ln_gamma_stats[ln_name]["means"].append(round(float(w.mean()), 4))
                    all_ln_gamma_stats[ln_name]["stds"].append(round(float(w.std()), 4))
                    all_ln_gamma_stats[ln_name]["gt1_ratios"].append(round(float((w > 1.0).mean()), 4))

    # 收集所有层的o_proj和down_proj的SVD top-1
    all_o_proj_s1 = []
    all_down_proj_s1 = []
    for l in range(n_layers):
        layer = layers[l]
        if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
            w = layer.self_attn.o_proj.weight.detach().cpu().float().numpy()
            S = np.linalg.svd(w, compute_uv=False)
            all_o_proj_s1.append(round(float(S[0]), 4))
        if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
            w = layer.mlp.down_proj.weight.detach().cpu().float().numpy()
            S = np.linalg.svd(w, compute_uv=False)
            all_down_proj_s1.append(round(float(S[0]), 4))

    results["global_stats"] = {
        "ln_gamma": all_ln_gamma_stats,
        "o_proj_s1_by_layer": all_o_proj_s1,
        "down_proj_s1_by_layer": all_down_proj_s1,
        "o_proj_s1_trend": {
            "L0": all_o_proj_s1[0] if all_o_proj_s1 else None,
            "L_mid": all_o_proj_s1[n_layers//2] if len(all_o_proj_s1) > n_layers//2 else None,
            "L_last": all_o_proj_s1[-1] if all_o_proj_s1 else None,
        },
        "down_proj_s1_trend": {
            "L0": all_down_proj_s1[0] if all_down_proj_s1 else None,
            "L_mid": all_down_proj_s1[n_layers//2] if len(all_down_proj_s1) > n_layers//2 else None,
            "L_last": all_down_proj_s1[-1] if all_down_proj_s1 else None,
        },
    }

    # 打印关键结果
    print(f"\n  === P432 Key Results for {model_name} ===")
    for ln_name in all_ln_gamma_stats:
        stats = all_ln_gamma_stats[ln_name]
        print(f"  {ln_name} gamma>1 ratio: L0={stats['gt1_ratios'][0]:.3f}, L_mid={stats['gt1_ratios'][n_layers//2]:.3f}, L_last={stats['gt1_ratios'][-1]:.3f}")
        print(f"  {ln_name} gamma mean: L0={stats['means'][0]:.4f}, L_mid={stats['means'][n_layers//2]:.4f}, L_last={stats['means'][-1]:.4f}")
    print(f"  o_proj S1: L0={all_o_proj_s1[0] if all_o_proj_s1 else 'N/A'}, L_mid={all_o_proj_s1[n_layers//2] if len(all_o_proj_s1) > n_layers//2 else 'N/A'}, L_last={all_o_proj_s1[-1] if all_o_proj_s1 else 'N/A'}")
    print(f"  down_proj S1: L0={all_down_proj_s1[0] if all_down_proj_s1 else 'N/A'}, L_mid={all_down_proj_s1[n_layers//2] if len(all_down_proj_s1) > n_layers//2 else 'N/A'}, L_last={all_down_proj_s1[-1] if all_down_proj_s1 else 'N/A'}")

    # 保存
    ts = time.strftime("%Y%m%d_%H%M")
    outf = OUT_DIR / f"phase_lxxxviii_p432_{model_name}_{ts}.json"
    with open(outf, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {outf}")

    return results


# ========== P433: 信号传播统一公式与预测力验证 ==========

def run_p433(model, tokenizer, device, model_name, p431_results, p432_results):
    print(f"\n{'='*60}")
    print(f"P433: Unified Signal Propagation Formula - {model_name}")
    print(f"{'='*60}")

    layers = get_layers(model)
    n_layers = len(layers)
    embed = model.get_input_embeddings()
    d_model = embed.weight.shape[1]
    beta = 1.0  # 小信号

    prompts = PROMPTS[:10]

    results = {"model": model_name, "exp": "p433", "n_layers": n_layers}

    # ---- 核心验证: 累积增益预测 vs 实际增益 ----
    # 从P431获取逐层的LN_gain和J_components
    layer_summary = p431_results.get("layer_summary", {})

    # 实际测量: 在L0注入信号, 测量每层的实际信号大小
    test_dims = ["logic", "sentiment", "style"]
    actual_gains = {str(l): [] for l in range(n_layers)}

    for dim_name in test_dims:
        w1, w2 = DIM_PAIRS[dim_name]
        direction, norm = get_dimension_direction(model, tokenizer, w1, w2)
        if norm < 1e-8:
            continue
        direction_t = torch.tensor(direction * beta, dtype=torch.float32)

        for pi, prompt in enumerate(prompts):
            toks = tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = toks.input_ids

            # 收集每层input的信号大小
            baseline_inputs = {}
            intervened_inputs = {}

            def make_pre_hook_b(store, layer_idx):
                def hook_fn(module, args):
                    x = args[0] if isinstance(args, tuple) else args
                    store[layer_idx] = x[0, -1, :].detach().cpu().float()
                    return args
                return hook_fn

            # Baseline
            hooks_b = []
            for l in range(n_layers):
                hooks_b.append(layers[l].register_forward_pre_hook(make_pre_hook_b(baseline_inputs, l)))
            with torch.no_grad():
                _ = model(input_ids)
            for h in hooks_b:
                h.remove()

            # Intervened
            hooks_i = []
            for l in range(n_layers):
                hooks_i.append(layers[l].register_forward_pre_hook(make_pre_hook_b(intervened_inputs, l)))

            def inj_hook_fn(module, input, output, d=direction_t):
                modified = output.clone()
                modified[0, -1, :] += d.to(output.device)
                return modified

            h_embed = embed.register_forward_hook(inj_hook_fn)
            with torch.no_grad():
                _ = model(input_ids)
            h_embed.remove()
            for h in hooks_i:
                h.remove()

            # 计算每层实际信号大小
            dh_L0 = (intervened_inputs[0] - baseline_inputs[0]).norm().item() if 0 in intervened_inputs and 0 in baseline_inputs else 0

            for l in range(n_layers):
                if l in intervened_inputs and l in baseline_inputs:
                    dh = (intervened_inputs[l] - baseline_inputs[l]).norm().item()
                    actual_gains[str(l)].append(dh / dh_L0 if dh_L0 > 1e-10 else 0)

    # 平均
    avg_actual = {}
    for l in range(n_layers):
        vals = actual_gains[str(l)]
        if vals:
            avg_actual[str(l)] = round(float(np.mean(vals)), 4)

    # 累积预测增益
    cum_pred = 1.0
    pred_vs_actual = {}
    for l in range(n_layers):
        ls = layer_summary.get(str(l), {})
        per_layer_pred = ls.get("predicted_gain", 1.0)
        cum_pred *= per_layer_pred
        actual = avg_actual.get(str(l), 0)
        ratio = cum_pred / actual if actual > 1e-10 else float('inf')
        pred_vs_actual[str(l)] = {
            "cum_predicted": round(cum_pred, 4),
            "actual": round(actual, 4),
            "ratio": round(ratio, 4),
        }

    results["pred_vs_actual"] = pred_vs_actual

    # 计算R² (对数空间, 避免极端值)
    log_pred = []
    log_actual = []
    for l in range(n_layers):
        pva = pred_vs_actual.get(str(l), {})
        p = pva.get("cum_predicted", 0)
        a = pva.get("actual", 0)
        if p > 0 and a > 0:
            log_pred.append(np.log(max(p, 1e-10)))
            log_actual.append(np.log(max(a, 1e-10)))

    if len(log_pred) > 2:
        ss_res = sum((p - a) ** 2 for p, a in zip(log_pred, log_actual))
        ss_tot = sum((a - np.mean(log_actual)) ** 2 for a in log_actual)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    else:
        r2 = 0

    results["log_r2"] = round(r2, 4)

    # ---- 方向2: 纯LN累积预测 ----
    cum_ln_only = 1.0
    ln_pred_vs_actual = {}
    for l in range(n_layers):
        ls = layer_summary.get(str(l), {})
        ln_gain = ls.get("ln_gain", 1.0)
        cum_ln_only *= ln_gain
        actual = avg_actual.get(str(l), 0)
        ratio = cum_ln_only / actual if actual > 1e-10 else float('inf')
        ln_pred_vs_actual[str(l)] = {
            "cum_ln_only": round(cum_ln_only, 4),
            "actual": round(actual, 4),
            "ratio": round(ratio, 4),
        }

    results["ln_only_pred_vs_actual"] = ln_pred_vs_actual

    # LN-only R²
    log_ln_pred = []
    for l in range(n_layers):
        lva = ln_pred_vs_actual.get(str(l), {})
        p = lva.get("cum_ln_only", 0)
        a = lva.get("actual", 0)
        if p > 0 and a > 0:
            log_ln_pred.append(np.log(max(p, 1e-10)))

    if len(log_ln_pred) > 2:
        ss_res_ln = sum((p - a) ** 2 for p, a in zip(log_ln_pred, log_actual))
        r2_ln = 1 - ss_res_ln / ss_tot if ss_tot > 0 else 0
    else:
        r2_ln = 0

    results["ln_only_log_r2"] = round(r2_ln, 4)

    # ---- 方向3: 修正公式 (残差连接修正) ----
    # Pre-LN架构: output = x + attn(LN(x)) + mlp(LN(x + attn(LN(x))))
    # 一阶信号传播: dh_out = dh_in + attn_J * LN_gain * dh_in + mlp_J * LN_gain2 * dh_in_attn
    # 简化: dh_out/dh_in = 1 + attn_J * LN_gain + mlp_J * LN_gain2 * (1 + attn_J * LN_gain)
    # 但这个展开太复杂, 用简化版:
    # dh_out/dh_in ≈ 1 + attn_J + mlp_J  (当J<1时, 即压缩)

    # 尝试经验修正: observed per-layer gain = LN_gain^α × J^β
    # 其中α,β通过回归确定

    # 收集(ln_gain, j_comp, actual_per_layer_gain)三元组
    ln_gains = []
    j_comps = []
    actual_per_layer = []

    # 从actual gains计算per-layer gain
    prev_actual = 1.0
    for l in range(n_layers):
        actual = avg_actual.get(str(l), 0)
        per_layer_actual = actual / prev_actual if prev_actual > 1e-10 else 0
        ls = layer_summary.get(str(l), {})
        lg = ls.get("ln_gain", 0)
        jc = ls.get("j_components", 0)
        if lg > 0 and jc > 0 and per_layer_actual > 0:
            ln_gains.append(np.log(lg))
            j_comps.append(np.log(jc))
            actual_per_layer.append(np.log(per_layer_actual))
        prev_actual = actual

    if len(actual_per_layer) > 3:
        # 线性回归: log(actual) = α * log(ln_gain) + β * log(j_comp) + c
        X = np.column_stack([ln_gains, j_comps, np.ones(len(ln_gains))])
        y = np.array(actual_per_layer)
        try:
            coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta_coeff, const = coeffs
            # R² of regression
            y_pred = X @ coeffs
            ss_res_reg = ((y - y_pred) ** 2).sum()
            ss_tot_reg = ((y - y.mean()) ** 2).sum()
            r2_reg = 1 - ss_res_reg / ss_tot_reg if ss_tot_reg > 0 else 0
            results["empirical_formula"] = {
                "formula": "log(gain) = α * log(LN_gain) + β * log(J_comp) + c",
                "alpha": round(float(alpha), 4),
                "beta": round(float(beta_coeff), 4),
                "const": round(float(const), 4),
                "r2": round(float(r2_reg), 4),
            }
        except Exception as e:
            results["empirical_formula"] = {"error": str(e)}
    else:
        results["empirical_formula"] = {"error": "insufficient data"}

    # 打印关键结果
    print(f"\n  === P433 Key Results for {model_name} ===")
    print(f"  LN*J predicted R2 (log): {r2:.4f}")
    print(f"  LN-only predicted R2 (log): {r2_ln:.4f}")
    for l in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        pva = pred_vs_actual.get(str(l), {})
        lva = ln_pred_vs_actual.get(str(l), {})
        print(f"  L{l}: LN×J_pred={pva.get('cum_predicted','N/A')}, LN_only={lva.get('cum_ln_only','N/A')}, actual={pva.get('actual','N/A')}")

    if "alpha" in results.get("empirical_formula", {}):
        ef = results["empirical_formula"]
        print(f"  Empirical: alpha={ef['alpha']}, beta={ef['beta']}, c={ef['const']}, R2={ef['r2']}")

    # 保存
    ts = time.strftime("%Y%m%d_%H%M")
    outf = OUT_DIR / f"phase_lxxxviii_p433_{model_name}_{ts}.json"
    with open(outf, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {outf}")

    return results


# ========== P434: 权重→激活因果方程 ==========

def run_p434(model, tokenizer, device, model_name, p431_results, p432_results, p433_results):
    print(f"\n{'='*60}")
    print(f"P434: Weight→Activation Causal Equation - {model_name}")
    print(f"{'='*60}")

    layers = get_layers(model)
    n_layers = len(layers)
    embed = model.get_input_embeddings()
    d_model = embed.weight.shape[1]

    results = {"model": model_name, "exp": "p434", "n_layers": n_layers, "d_model": d_model}

    # ---- 核心方法: 线性化因果方程 ----
    # h_l = (I + J_l) * LN_l * h_{l-1} + b_l
    # 其中 J_l = attn_Jacobian + mlp_Jacobian (来自P431)
    # LN_l = LayerNorm算子 (来自P431的ln_gain)
    # b_l = 层偏置

    # 验证: 用h_0预测h_k, 看R²
    # h_k ≈ (∏_l (I + J_l) * LN_l) * h_0 + cumulative_bias

    # 由于计算完整雅可比矩阵太昂贵(d_model × d_model),
    # 我们用信号传播方法验证:
    # 对N个随机方向d_i, 测量 (h_k(x + ε*d_i) - h_k(x)) / ε
    # 然后检查是否等于 predicted_gain_k * d_i

    n_directions = 20  # 随机方向数
    eps = 0.1  # 微小扰动

    prompts = PROMPTS[:5]
    sample_layers = [0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-2, n_layers-1]
    sample_layers = sorted(set(sample_layers))

    layer_summary = p431_results.get("layer_summary", {})
    p433_pva = p433_results.get("pred_vs_actual", {})

    causal_results = {}

    for pi, prompt in enumerate(prompts):
        toks = tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = toks.input_ids

        # 收集baseline每层hidden state
        baseline_hs = {}

        def make_hs_hook(store, layer_idx):
            def hook_fn(module, args):
                x = args[0] if isinstance(args, tuple) else args
                store[layer_idx] = x[0, -1, :].detach().cpu().float()
                return args
            return hook_fn

        hooks = []
        for l in sample_layers:
            hooks.append(layers[l].register_forward_pre_hook(make_hs_hook(baseline_hs, l)))

        with torch.no_grad():
            _ = model(input_ids)
        for h in hooks:
            h.remove()

        # 对每个随机方向做微扰
        direction_results = {str(l): {"cos_sim": [], "norm_ratio": []} for l in sample_layers}

        for di in range(n_directions):
            # 随机方向
            rand_dir = torch.randn(d_model, dtype=torch.float32)
            rand_dir = rand_dir / rand_dir.norm() * eps

            # 在embed注入
            intervened_hs = {}
            hooks_i = []
            for l in sample_layers:
                hooks_i.append(layers[l].register_forward_pre_hook(make_hs_hook(intervened_hs, l)))

            def inj_hook_fn(module, input, output, d=rand_dir):
                modified = output.clone()
                modified[0, -1, :] += d.to(output.device)
                return modified

            h_embed = embed.register_forward_hook(inj_hook_fn)
            with torch.no_grad():
                _ = model(input_ids)
            h_embed.remove()
            for h in hooks_i:
                h.remove()

            # 比较实际变化 vs 预测变化
            for l in sample_layers:
                if l in baseline_hs and l in intervened_hs:
                    actual_delta = intervened_hs[l] - baseline_hs[l]
                    actual_norm = actual_delta.norm().item()

                    # 预测: 信号增益 × 初始信号方向
                    pva = p433_pva.get(str(l), {})
                    predicted_gain = pva.get("cum_predicted", 1.0)

                    # 方向预测: 如果线性, 则方向应与L0的delta同方向
                    if 0 in baseline_hs and 0 in intervened_hs:
                        l0_delta = intervened_hs[0] - baseline_hs[0]
                        l0_norm = l0_delta.norm().item()
                        if actual_norm > 1e-10 and l0_norm > 1e-10:
                            cos_sim = float(torch.nn.functional.cosine_similarity(
                                actual_delta.unsqueeze(0), l0_delta.unsqueeze(0)
                            ).item())
                            norm_ratio = actual_norm / (predicted_gain * l0_norm) if predicted_gain * l0_norm > 1e-10 else 0
                            direction_results[str(l)]["cos_sim"].append(cos_sim)
                            direction_results[str(l)]["norm_ratio"].append(norm_ratio)

        # 汇总此prompt的结果
        for l in sample_layers:
            dr = direction_results[str(l)]
            if dr["cos_sim"]:
                if str(l) not in causal_results:
                    causal_results[str(l)] = {"cos_sim": [], "norm_ratio": []}
                causal_results[str(l)]["cos_sim"].extend(dr["cos_sim"])
                causal_results[str(l)]["norm_ratio"].extend(dr["norm_ratio"])

    # 最终汇总
    causal_summary = {}
    for l in sample_layers:
        cr = causal_results.get(str(l), {})
        cos_vals = cr.get("cos_sim", [])
        norm_vals = cr.get("norm_ratio", [])
        if cos_vals:
            causal_summary[str(l)] = {
                "mean_cos_sim": round(float(np.mean(cos_vals)), 4),
                "std_cos_sim": round(float(np.std(cos_vals)), 4),
                "mean_norm_ratio": round(float(np.mean(norm_vals)), 4),
                "std_norm_ratio": round(float(np.std(norm_vals)), 4),
                "linearity_score": round(float(np.mean([abs(c) for c in cos_vals])), 4),  # 方向保持度
                "n_samples": len(cos_vals),
            }

    results["causal_summary"] = causal_summary

    # ---- 方向保持度的深层含义 ----
    # cos_sim ≈ 1: 线性传播 (信号方向不变, 只有幅度变化)
    # cos_sim ≈ 0: 正交散射 (信号方向完全随机化)
    # cos_sim < 0: 信号反转 (某些维度被翻转)
    # norm_ratio ≈ predicted_gain: 预测准确
    # norm_ratio ≠ predicted_gain: 预测偏差

    # 线性度评分
    linearity_by_layer = {}
    for l in sample_layers:
        cs = causal_summary.get(str(l), {})
        linearity_by_layer[str(l)] = cs.get("linearity_score", 0)

    results["linearity_by_layer"] = linearity_by_layer

    # 打印关键结果
    print(f"\n  === P434 Key Results for {model_name} ===")
    print(f"  Layer | linearity | cos_sim | norm_ratio | n_samples")
    for l in sample_layers:
        cs = causal_summary.get(str(l), {})
        print(f"  L{l:3d} | {cs.get('linearity_score',0):9.4f} | {cs.get('mean_cos_sim',0):7.4f} | {cs.get('mean_norm_ratio',0):10.4f} | {cs.get('n_samples',0)}")

    # 整体线性度
    all_lin = [v for v in linearity_by_layer.values()]
    if all_lin:
        print(f"\n  Overall linearity: {np.mean(all_lin):.4f}")

    # 保存
    ts = time.strftime("%Y%m%d_%H%M")
    outf = OUT_DIR / f"phase_lxxxviii_p434_{model_name}_{ts}.json"
    with open(outf, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Saved to {outf}")

    return results


# ========== 主程序 ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--exp", type=str, required=True, choices=["p431", "p432", "p433", "p434", "all"])
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)

    # 加载之前的实验结果(如果需要)
    def load_latest_result(exp_name, model_name):
        """加载最近的实验结果文件"""
        import glob
        pattern = str(OUT_DIR / f"phase_lxxxviii_{exp_name}_{model_name}_*.json")
        files = sorted(glob.glob(pattern))
        if files:
            latest = files[-1]
            print(f"  Loading {exp_name} results from {latest}")
            with open(latest, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    if args.exp == "p431" or args.exp == "all":
        p431_results = run_p431(model, tokenizer, device, args.model)
    else:
        p431_results = load_latest_result("p431", args.model)

    if args.exp == "p432" or args.exp == "all":
        p432_results = run_p432(model, tokenizer, device, args.model)
    else:
        p432_results = load_latest_result("p432", args.model)

    if args.exp == "p433" or args.exp == "all":
        p433_results = run_p433(model, tokenizer, device, args.model, p431_results, p432_results)
    else:
        p433_results = {}

    if args.exp == "p434" or args.exp == "all":
        p434_results = run_p434(model, tokenizer, device, args.model, p431_results, p432_results, p433_results)
    else:
        p434_results = {}

    # 释放GPU
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"\nDone! GPU memory freed.")


if __name__ == "__main__":
    main()
