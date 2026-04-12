"""
Phase LXXXIII-P411/412/413/414: 暗能量解剖与V_lang边界
================================================================================

阶段F核心任务 - 暗能量的精确解剖 + V_lang真正边界:

P411: L26层attention/MLP贡献分解
  - P409发现: DS7B L26层dh_norm暴增+1800
  - 核心问题: attention和MLP各自贡献多少?
  - 方法: 逐层分离attn_output和mlp_output, 计算各自的dh分量
  - 关键: dh_attn = hook_attn_output - base_attn_output
         dh_mlp = hook_mlp_output - base_mlp_output

P412: V_lang边界搜索——扩展到50+词对
  - P407发现: 25个词对中, GLM4全部功能, Qwen3=17, DS7B=16
  - 核心问题: V_lang的真正维度上限是多少?
  - 方法: 定义50+新的词对方向, 测量Δlogit
  - 关键: 找到功能维度的饱和点

P413: 结构维度"复活"实验
  - P407发现: position/completeness/age等在Qwen3中是结构维度
  - 核心问题: 更大β或不同prompt能否"复活"这些维度?
  - 方法: β=8→32, 多个prompt测试, 逐层注入而非L0
  - 关键: 区分"维度不存在"vs"信号太弱"

P414: Δlogit分布的精确建模
  - P410发现: eff_gaussian在DS7B上误差1.33
  - 核心问题: Δlogit分布是否偏离高斯?
  - 方法: 拟合Student-t、混合高斯、Laplace分布
  - 关键: 找到精确描述Δlogit分布的参数族

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

# P412: 扩展词对 - 新增30个维度
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

ALL_50_DIMS = {**DIM_PAIRS, **EXTRA_DIM_PAIRS, **EXTENDED_DIM_PAIRS}

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


# ========== P411: L26层attention/MLP贡献分解 ==========

def run_p411(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P411: Attention/MLP decomposition of dark energy - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()

    layers = get_layers(model)
    n_layers = len(layers)

    test_dims = ["style", "logic", "grammar", "sentiment"]
    beta = 8.0

    dim_info = {}
    for name in test_dims:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        w_pos_raw = get_w_lm(model, tokenizer, pos)[1]
        w_neg_raw = get_w_lm(model, tokenizer, neg)[1]
        w_diff = w_pos_raw - w_neg_raw
        w_diff_norm = float(np.linalg.norm(w_diff))
        pos_id = tokenizer.encode(pos, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg, add_special_tokens=False)[0]
        dim_info[name] = {
            "direction": direction, "w_diff": w_diff, "w_diff_norm": w_diff_norm,
            "pos_id": pos_id, "neg_id": neg_id,
        }

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # ★ 核心: 分离attention和MLP输出 ★
    # TransformerLayer的forward: h_out = h + attn(h) + mlp(h + attn(h))
    # 我们需要hook住self_attn和mlp的输出

    # 扫描关键层: 最后5层 + 之前每5层一次
    scan_layers = list(range(max(0, n_layers - 5), n_layers))
    for l in range(0, n_layers - 5, 5):
        if l not in scan_layers:
            scan_layers.append(l)
    scan_layers.sort()
    print(f"  Scanning {len(scan_layers)} layers: {scan_layers}")

    # ===== 基线 =====
    captured_attn_base = {}
    captured_mlp_base = {}
    captured_resid_base = {}

    def make_hook(storage, key):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[key] = output[0].detach().float()
            else:
                storage[key] = output.detach().float()
        return hook

    handles = []
    for l in scan_layers:
        layer = layers[l]
        # Hook attention output
        if hasattr(layer, "self_attn"):
            handles.append(layer.self_attn.register_forward_hook(make_hook(captured_attn_base, f"L{l}")))
        elif hasattr(layer, "attention"):
            handles.append(layer.attention.register_forward_hook(make_hook(captured_attn_base, f"L{l}")))
        # Hook MLP output
        if hasattr(layer, "mlp"):
            handles.append(layer.mlp.register_forward_hook(make_hook(captured_mlp_base, f"L{l}")))
        elif hasattr(layer, "feed_forward"):
            handles.append(layer.feed_forward.register_forward_hook(make_hook(captured_mlp_base, f"L{l}")))
        # Hook residual stream (layer output)
        handles.append(layer.register_forward_hook(make_hook(captured_resid_base, f"L{l}")))

    with torch.no_grad():
        _ = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids)
    for h in handles:
        h.remove()

    # ===== 干预 =====
    results_per_dim = {}
    for dim_name in dim_info:
        direction = dim_info[dim_name]["direction"]
        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        captured_attn_int = {}
        captured_mlp_int = {}
        captured_resid_int = {}

        handles = []
        for l in scan_layers:
            layer = layers[l]
            if hasattr(layer, "self_attn"):
                handles.append(layer.self_attn.register_forward_hook(make_hook(captured_attn_int, f"L{l}")))
            elif hasattr(layer, "attention"):
                handles.append(layer.attention.register_forward_hook(make_hook(captured_attn_int, f"L{l}")))
            if hasattr(layer, "mlp"):
                handles.append(layer.mlp.register_forward_hook(make_hook(captured_mlp_int, f"L{l}")))
            elif hasattr(layer, "feed_forward"):
                handles.append(layer.feed_forward.register_forward_hook(make_hook(captured_mlp_int, f"L{l}")))
            handles.append(layer.register_forward_hook(make_hook(captured_resid_int, f"L{l}")))

        with torch.no_grad():
            _ = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids)
        for h in handles:
            h.remove()

        # 逐层分析attn vs mlp贡献
        layer_analysis = {}
        for l in scan_layers:
            key = f"L{l}"
            # residual dh
            if key in captured_resid_int and key in captured_resid_base:
                dh_resid = (captured_resid_int[key][0, -1, :] - captured_resid_base[key][0, -1, :]).cpu().numpy()
            else:
                dh_resid = None

            # attention dh
            if key in captured_attn_int and key in captured_attn_base:
                dh_attn = (captured_attn_int[key][0, -1, :] - captured_attn_base[key][0, -1, :]).cpu().numpy()
            else:
                dh_attn = None

            # MLP dh
            if key in captured_mlp_int and key in captured_mlp_base:
                dh_mlp = (captured_mlp_int[key][0, -1, :] - captured_mlp_base[key][0, -1, :]).cpu().numpy()
            else:
                dh_mlp = None

            analysis = {}
            if dh_resid is not None:
                analysis["dh_resid_norm"] = float(np.linalg.norm(dh_resid))
            if dh_attn is not None:
                analysis["dh_attn_norm"] = float(np.linalg.norm(dh_attn))
            if dh_mlp is not None:
                analysis["dh_mlp_norm"] = float(np.linalg.norm(dh_mlp))

            # 贡献比例
            if dh_attn is not None and dh_mlp is not None and dh_resid is not None:
                # dh_resid ≈ dh_attn + dh_mlp (残差连接)
                # 但实际上dh_resid可能不完全等于两者之和(因为有层归一化等)
                # 计算两者对resid的投影
                if analysis["dh_resid_norm"] > 1e-8:
                    cos_attn_resid = float(np.dot(dh_attn, dh_resid) / (np.linalg.norm(dh_attn) * analysis["dh_resid_norm"]))
                    cos_mlp_resid = float(np.dot(dh_mlp, dh_resid) / (np.linalg.norm(dh_mlp) * analysis["dh_resid_norm"]))
                    analysis["cos_attn_resid"] = cos_attn_resid
                    analysis["cos_mlp_resid"] = cos_mlp_resid
                    # 投影能量比
                    proj_attn = float(np.dot(dh_attn, dh_resid) / analysis["dh_resid_norm"])
                    proj_mlp = float(np.dot(dh_mlp, dh_resid) / analysis["dh_resid_norm"])
                    analysis["proj_attn_ratio"] = proj_attn / analysis["dh_resid_norm"]
                    analysis["proj_mlp_ratio"] = proj_mlp / analysis["dh_resid_norm"]

            layer_analysis[str(l)] = analysis

        # 找最大增幅层
        resid_norms = {l: layer_analysis[l].get("dh_resid_norm", 0) for l in layer_analysis}
        sorted_layers = sorted(resid_norms.keys(), key=lambda x: int(x))
        if len(sorted_layers) >= 2:
            max_increase = 0
            max_inc_layer = ""
            for i in range(1, len(sorted_layers)):
                inc = resid_norms[sorted_layers[i]] - resid_norms[sorted_layers[i-1]]
                if inc > max_increase:
                    max_increase = inc
                    max_inc_layer = sorted_layers[i]
            
            l_key = max_inc_layer
            attn_ratio = layer_analysis[l_key].get("proj_attn_ratio", 0)
            mlp_ratio = layer_analysis[l_key].get("proj_mlp_ratio", 0)
            print(f"\n  {dim_name}: Max increase at L{l_key} (+{max_increase:.1f})")
            print(f"    At L{l_key}: dh_resid={resid_norms[l_key]:.1f}, attn_ratio={attn_ratio:.4f}, mlp_ratio={mlp_ratio:.4f}")
            print(f"    dh_attn_norm={layer_analysis[l_key].get('dh_attn_norm', 0):.1f}, dh_mlp_norm={layer_analysis[l_key].get('dh_mlp_norm', 0):.1f}")

        results_per_dim[dim_name] = {
            "layer_analysis": layer_analysis,
            "max_increase_layer": max_inc_layer if len(sorted_layers) >= 2 else "N/A",
            "max_increase_val": max_increase if len(sorted_layers) >= 2 else 0,
        }

    # 打印最后3层的attn/mlp比例
    print(f"\n  === Last 3 layers: attn vs mlp ===")
    last3 = [str(l) for l in scan_layers[-3:]]
    for l_key in last3:
        print(f"  L{l_key}:", end="")
        for dim_name in dim_info:
            la = results_per_dim[dim_name]["layer_analysis"].get(l_key, {})
            ar = la.get("proj_attn_ratio", 0)
            mr = la.get("proj_mlp_ratio", 0)
            an = la.get("dh_attn_norm", 0)
            mn = la.get("dh_mlp_norm", 0)
            print(f"  {dim_name}(attn={an:.0f},mlp={mn:.0f},r={ar:.2f}/{mr:.2f})", end="")
        print()

    results = {
        "n_layers_total": n_layers,
        "scan_layers": scan_layers,
        "results_per_dim": results_per_dim,
    }
    return results


# ========== P412: V_lang边界搜索——扩展到50+词对 ==========

def run_p412(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P412: V_lang boundary search - 50+ dimensions - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    beta = 8.0

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 基线
    with torch.no_grad():
        logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()

    results_per_dim = {}
    n_functional = 0
    n_structural = 0
    n_marginal = 0
    n_error = 0

    for dim_name, pairs in ALL_50_DIMS.items():
        pos, neg = pairs[0]
        try:
            direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
            if norm < 1e-8:
                print(f"  {dim_name}: SKIP (norm too small)")
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

            func_flag = "F" if is_functional else ("S" if is_structural else "M")
            print(f"  {func_flag} {dim_name} ({pos}/{neg}): |dlogit|={abs(dlogit):.3f}")

        except Exception as e:
            print(f"  {dim_name}: ERROR - {e}")
            n_error += 1

    print(f"\n  === V_lang Boundary Summary ===")
    print(f"  Total tested: {len(results_per_dim)}")
    print(f"  Functional (|dlogit|>1.0): {n_functional}")
    print(f"  Structural (|dlogit|<0.5): {n_structural}")
    print(f"  Marginal (0.5-1.0): {n_marginal}")
    print(f"  Errors: {n_error}")

    # 按dlogit排序top 20
    sorted_dims = sorted(results_per_dim.items(), key=lambda x: abs(x[1]["dlogit"]), reverse=True)
    print(f"\n  Top 20 functional dimensions:")
    for name, r in sorted_dims[:20]:
        func = "F" if r["is_functional"] else ("S" if r["is_structural_only"] else "M")
        print(f"    {func} {name}: |dlogit|={abs(r['dlogit']):.3f}")

    # 正交性检查: 功能维度之间是否正交?
    print(f"\n  === Orthogonality among top functional dims ===")
    top_func = [name for name, r in sorted_dims if r["is_functional"]][:10]
    if len(top_func) >= 2:
        dirs = {}
        for name in top_func:
            pos, neg = ALL_50_DIMS[name][0]
            d, _ = get_dimension_direction(model, tokenizer, pos, neg)
            dirs[name] = d
        
        for i in range(min(5, len(top_func))):
            for j in range(i+1, min(5, len(top_func))):
                cos = float(np.dot(dirs[top_func[i]], dirs[top_func[j]]))
                print(f"    cos({top_func[i]}, {top_func[j]}) = {cos:.4f}")

    results = {
        "n_total": len(results_per_dim),
        "n_functional": n_functional,
        "n_structural": n_structural,
        "n_marginal": n_marginal,
        "results_per_dim": results_per_dim,
    }
    return results


# ========== P413: 结构维度"复活"实验 ==========

def run_p413(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P413: Structural dimension revival - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    layers = get_layers(model)
    n_layers = len(layers)

    # 之前发现的结构维度
    structural_dims = ["position", "completeness", "age", "importance", "weight"]
    # 对照: 已知的功能维度
    functional_dims = ["style", "politeness", "value", "specificity"]

    all_test_dims = structural_dims + functional_dims
    betas = [8.0, 16.0, 32.0, 64.0]
    # 逐层注入: 在不同层注入信号
    inject_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]

    results = {}

    for dim_name in all_test_dims:
        if dim_name in DIM_PAIRS:
            pos, neg = DIM_PAIRS[dim_name][0]
        elif dim_name in EXTRA_DIM_PAIRS:
            pos, neg = EXTRA_DIM_PAIRS[dim_name][0]
        else:
            continue

        try:
            direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
            if norm < 1e-8:
                continue
            pos_id = tokenizer.encode(pos, add_special_tokens=False)[0]
            neg_id = tokenizer.encode(neg, add_special_tokens=False)[0]
        except:
            continue

        dim_results = {}

        # ===== 实验1: 增大β =====
        print(f"\n  {dim_name} ({pos}/{neg}):")
        print(f"    Beta scaling:")
        inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

        with torch.no_grad():
            logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()

        for beta in betas:
            w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
            inputs_embeds_int = inputs_embeds_base.clone()
            inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

            with torch.no_grad():
                logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()

            delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
            delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
            dlogit = delta_pos - delta_neg

            dim_results[f"beta_{int(beta)}"] = {"dlogit": dlogit, "delta_pos": delta_pos, "delta_neg": delta_neg}
            print(f"      β={int(beta)}: dlogit={dlogit:.3f} (pos={delta_pos:.3f}, neg={delta_neg:.3f})")

        # ===== 实验2: 逐层注入 =====
        print(f"    Layer injection (β=16):")
        beta_inj = 16.0
        for inject_l in inject_layers:
            # 使用hook在指定层注入
            w_tensor = torch.tensor(direction * beta_inj, dtype=torch.float32, device=device)

            captured = {}
            def make_inject_hook(w_inject, layer_idx):
                count = [0]
                def hook(module, input, output):
                    if count[0] == 0 and isinstance(output, tuple):
                        out = output[0].clone()
                        out[0, -1, :] += w_inject.to(out.dtype)
                        count[0] += 1
                        return (out,) + output[1:]
                    return output
                return hook

            handle = layers[inject_l].register_forward_hook(make_inject_hook(w_tensor, inject_l))

            with torch.no_grad():
                logits_int = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()
            handle.remove()

            delta_pos = float(logits_int[pos_id].cpu() - logits_base[pos_id].cpu())
            delta_neg = float(logits_int[neg_id].cpu() - logits_base[neg_id].cpu())
            dlogit = delta_pos - delta_neg

            dim_results[f"layer_{inject_l}"] = {"dlogit": dlogit}
            print(f"      L{inject_l}: dlogit={dlogit:.3f}")

        # ===== 实验3: 多prompt =====
        print(f"    Multi-prompt (β=16):")
        for pi, prompt_i in enumerate(PROMPTS[:4]):
            toks_i = tokenizer(prompt_i, return_tensors="pt").to(device)
            input_ids_i = toks_i.input_ids
            seq_len_i = input_ids_i.shape[1]
            inputs_embeds_i = embed(input_ids_i).detach().clone().to(model.dtype)
            position_ids_i = torch.arange(seq_len_i, device=device).unsqueeze(0)

            with torch.no_grad():
                logits_base_i = model(inputs_embeds=inputs_embeds_i, position_ids=position_ids_i).logits[0, -1, :].float()

            w_tensor = torch.tensor(direction * 16.0, dtype=torch.float32, device=device)
            inputs_embeds_int_i = inputs_embeds_i.clone()
            inputs_embeds_int_i[0, -1, :] += w_tensor.to(model.dtype)

            with torch.no_grad():
                logits_int_i = model(inputs_embeds=inputs_embeds_int_i, position_ids=position_ids_i).logits[0, -1, :].float()

            delta_pos = float(logits_int_i[pos_id].cpu() - logits_base_i[pos_id].cpu())
            delta_neg = float(logits_int_i[neg_id].cpu() - logits_base_i[neg_id].cpu())
            dlogit = delta_pos - delta_neg

            dim_results[f"prompt_{pi}"] = {"dlogit": dlogit, "prompt": prompt_i[:30]}
            print(f"      P{pi}: dlogit={dlogit:.3f}")

        # 判断是否"可复活"
        max_dlogit_beta = max(abs(dim_results.get(f"beta_{int(b)}", {}).get("dlogit", 0)) for b in betas)
        max_dlogit_layer = max(abs(dim_results.get(f"layer_{l}", {}).get("dlogit", 0)) for l in inject_layers)
        max_dlogit_prompt = max(abs(dim_results.get(f"prompt_{pi}", {}).get("dlogit", 0)) for pi in range(4))

        can_revive = max(max_dlogit_beta, max_dlogit_layer, max_dlogit_prompt) > 1.0
        results[dim_name] = {
            "dim_results": dim_results,
            "max_dlogit_beta": max_dlogit_beta,
            "max_dlogit_layer": max_dlogit_layer,
            "max_dlogit_prompt": max_dlogit_prompt,
            "can_revive": can_revive,
            "is_structural": dim_name in structural_dims,
        }

        revive_flag = " ★ CAN REVIVE ★" if can_revive and dim_name in structural_dims else ""
        print(f"    Summary: max_beta={max_dlogit_beta:.3f}, max_layer={max_dlogit_layer:.3f}, max_prompt={max_dlogit_prompt:.3f}{revive_flag}")

    # 统计
    n_revived = sum(1 for r in results.values() if r["can_revive"] and r["is_structural"])
    n_still_structural = sum(1 for r in results.values() if not r["can_revive"] and r["is_structural"])
    print(f"\n  === Revival Summary ===")
    print(f"  Structural dims revived: {n_revived}")
    print(f"  Structural dims still dead: {n_still_structural}")

    return results


# ========== P414: Δlogit分布的精确建模 ==========

def run_p414(model, tokenizer, device, model_name):
    print(f"\n{'='*60}")
    print(f"P414: Precise modeling of Δlogit distribution - {model_name}")
    print(f"{'='*60}")

    embed = model.get_input_embeddings()
    lm_head = model.lm_head
    W_lm = lm_head.weight.detach().cpu().float().numpy()
    vocab_size, hidden_dim = W_lm.shape

    test_dims = ["style", "logic", "grammar", "sentiment", "tense", "certainty"]
    beta = 8.0

    dim_info = {}
    for name in test_dims:
        pos, neg = DIM_PAIRS[name][0]
        direction, norm = get_dimension_direction(model, tokenizer, pos, neg)
        w_pos_raw = get_w_lm(model, tokenizer, pos)[1]
        w_neg_raw = get_w_lm(model, tokenizer, neg)[1]
        w_diff = w_pos_raw - w_neg_raw
        w_diff_norm = float(np.linalg.norm(w_diff))
        pos_id = tokenizer.encode(pos, add_special_tokens=False)[0]
        neg_id = tokenizer.encode(neg, add_special_tokens=False)[0]
        dim_info[name] = {
            "direction": direction, "w_diff": w_diff, "w_diff_norm": w_diff_norm,
            "pos_id": pos_id, "neg_id": neg_id,
        }

    prompt = PROMPTS[0]
    toks = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = toks.input_ids
    seq_len = input_ids.shape[1]
    inputs_embeds_base = embed(input_ids).detach().clone().to(model.dtype)
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)

    # 多个随机方向做对照
    np.random.seed(42)
    n_random = 5
    random_directions = [np.random.randn(hidden_dim) / np.sqrt(hidden_dim) for _ in range(n_random)]

    # 对每个维度, 收集完整的Δlogit分布
    results_per_dim = {}

    for dim_name in list(dim_info.keys()) + [f"random_{i}" for i in range(n_random)]:
        if dim_name.startswith("random_"):
            direction = random_directions[int(dim_name.split("_")[1])]
        else:
            direction = dim_info[dim_name]["direction"]

        w_tensor = torch.tensor(direction * beta, dtype=torch.float32, device=device)
        inputs_embeds_int = inputs_embeds_base.clone()
        inputs_embeds_int[0, -1, :] += w_tensor.to(model.dtype)

        with torch.no_grad():
            logits_int = model(inputs_embeds=inputs_embeds_int, position_ids=position_ids).logits[0, -1, :].float()

        with torch.no_grad():
            logits_base = model(inputs_embeds=inputs_embeds_base, position_ids=position_ids).logits[0, -1, :].float()

        delta_logits = (logits_int - logits_base).cpu().numpy()

        # ★★★ 分布分析 ★★★
        mean_dl = float(np.mean(delta_logits))
        std_dl = float(np.std(delta_logits))
        median_dl = float(np.median(delta_logits))
        skew_dl = float(np.mean((delta_logits - mean_dl)**3) / max(std_dl**3, 1e-10))
        kurt_dl = float(np.mean((delta_logits - mean_dl)**4) / max(std_dl**4, 1e-10) - 3)  # excess kurtosis

        # 分位数
        q01 = float(np.percentile(delta_logits, 1))
        q05 = float(np.percentile(delta_logits, 5))
        q95 = float(np.percentile(delta_logits, 95))
        q99 = float(np.percentile(delta_logits, 99))

        # 重尾指数 (tail index)
        # 用Hill estimator: α = 1/mean(log(X_i / X_threshold)) for X_i > X_threshold
        abs_sorted = np.sort(np.abs(delta_logits))[::-1]
        n_tail = max(100, int(0.01 * len(abs_sorted)))
        if abs_sorted[n_tail] > 0:
            hill_alpha = 1.0 / np.mean(np.log(abs_sorted[:n_tail] / abs_sorted[n_tail]))
        else:
            hill_alpha = float('inf')

        # ★★★ 分布拟合 ★★★
        # 1. Gaussian: N(μ, σ²) - 已知mean和std
        # 2. Student-t: 需要估计自由度ν
        #    kurtosis of t(ν) = 6/(ν-4) for ν>4
        #    所以 ν ≈ 4 + 6/kurtosis (如果kurtosis>0)
        if kurt_dl > 0:
            nu_t = 4 + 6 / kurt_dl
        else:
            nu_t = 100  # 近似高斯
        # Student-t的σ: Var[t(ν)] = σ² * ν/(ν-2) for ν>2
        if nu_t > 2:
            sigma_t = std_dl * np.sqrt((nu_t - 2) / nu_t)
        else:
            sigma_t = std_dl

        # 3. Laplace: f(x) = 1/(2b) * exp(-|x-μ|/b)
        #    E[|X-μ|] = b, Var = 2b²
        b_laplace = float(np.mean(np.abs(delta_logits - mean_dl)))
        sigma_laplace = b_laplace * np.sqrt(2)

        # 4. 混合高斯: 两分量
        # 简化: 用em算法的1步近似
        # 分成|Δlogit|>2σ和<=2σ两组
        mask_large = np.abs(delta_logits - mean_dl) > 2 * std_dl
        mask_small = ~mask_large
        n_large = int(np.sum(mask_large))
        n_small = int(np.sum(mask_small))
        if n_large > 0 and n_small > 0:
            mu_small = float(np.mean(delta_logits[mask_small]))
            std_small = float(np.std(delta_logits[mask_small]))
            mu_large = float(np.mean(delta_logits[mask_large]))
            std_large = float(np.std(delta_logits[mask_large]))
            pi_mix = n_small / len(delta_logits)  # 小分量权重
        else:
            mu_small = mean_dl
            std_small = std_dl
            mu_large = mean_dl
            std_large = std_dl * 3
            pi_mix = 0.9

        # ★★★ 拟合优度: 用KS-like统计量 ★★★
        # 对每种分布, 计算: 在目标词对处的实际|Δlogit| vs 预测
        if not dim_name.startswith("random_"):
            pos_id = dim_info[dim_name]["pos_id"]
            neg_id = dim_info[dim_name]["neg_id"]
            actual_dlogit = float(delta_logits[pos_id] - delta_logits[neg_id])
        else:
            actual_dlogit = 0

        # 计算eff_L1
        total_L1 = float(np.sum(np.abs(delta_logits)))
        eff_L1 = abs(actual_dlogit) / max(total_L1, 1e-10) if not dim_name.startswith("random_") else 0

        # Gaussian预测eff
        eff_gauss = abs(actual_dlogit) / max(vocab_size * std_dl * np.sqrt(2 / np.pi), 1e-10) if not dim_name.startswith("random_") else 0

        # Student-t预测eff
        # E[|X|] for t(ν, σ) ≈ σ * 2*sqrt(ν)*Γ((ν+1)/2) / ((ν-1)*Γ(ν/2)*sqrt(π))
        # 简化: 对于大ν ≈ σ*sqrt(2/π), 对小ν更重尾→E[|X|]更大→eff更小
        from scipy.special import gamma as gamma_func
        if nu_t > 1:
            E_abs_t = sigma_t * 2 * np.sqrt(nu_t - 2) * gamma_func((nu_t - 1) / 2) / ((nu_t - 2) * gamma_func((nu_t - 2) / 2) * np.sqrt(np.pi))
        else:
            E_abs_t = sigma_t * np.sqrt(2 / np.pi)
        eff_t = abs(actual_dlogit) / max(vocab_size * E_abs_t, 1e-10) if not dim_name.startswith("random_") else 0

        # Laplace预测eff
        eff_laplace = abs(actual_dlogit) / max(vocab_size * b_laplace, 1e-10) if not dim_name.startswith("random_") else 0

        # 混合高斯预测eff
        E_abs_mix = pi_mix * std_small * np.sqrt(2 / np.pi) + (1 - pi_mix) * std_large * np.sqrt(2 / np.pi)
        eff_mix = abs(actual_dlogit) / max(vocab_size * E_abs_mix, 1e-10) if not dim_name.startswith("random_") else 0

        results_per_dim[dim_name] = {
            "mean": mean_dl, "std": std_dl, "median": median_dl,
            "skewness": skew_dl, "excess_kurtosis": kurt_dl,
            "q01": q01, "q05": q05, "q95": q95, "q99": q99,
            "hill_alpha": hill_alpha,
            "nu_t": nu_t, "sigma_t": sigma_t,
            "b_laplace": b_laplace, "sigma_laplace": sigma_laplace,
            "mix_pi": pi_mix, "mix_std_small": std_small, "mix_std_large": std_large,
            "actual_dlogit": actual_dlogit,
            "eff_L1": eff_L1,
            "eff_gauss": eff_gauss, "eff_t": eff_t, "eff_laplace": eff_laplace, "eff_mix": eff_mix,
        }

        is_rand = dim_name.startswith("random_")
        print(f"\n  {dim_name}:")
        print(f"    mean={mean_dl:.4f}, std={std_dl:.4f}, skew={skew_dl:.4f}, kurt={kurt_dl:.4f}")
        print(f"    Hill α={hill_alpha:.2f}, Student-t ν={nu_t:.1f}")
        print(f"    Laplace b={b_laplace:.4f}, Mix: π={pi_mix:.3f}, σ_s={std_small:.4f}, σ_l={std_large:.4f}")
        if not is_rand:
            print(f"    actual_dlogit={actual_dlogit:.3f}")
            print(f"    eff: actual={eff_L1:.6e}, gauss={eff_gauss:.6e}, t={eff_t:.6e}, laplace={eff_laplace:.6e}, mix={eff_mix:.6e}")
            # 最佳拟合
            errors = {
                "gauss": abs(eff_gauss - eff_L1) / max(eff_L1, 1e-10),
                "t": abs(eff_t - eff_L1) / max(eff_L1, 1e-10),
                "laplace": abs(eff_laplace - eff_L1) / max(eff_L1, 1e-10),
                "mix": abs(eff_mix - eff_L1) / max(eff_L1, 1e-10),
            }
            best = min(errors, key=errors.get)
            print(f"    Best fit: {best} (error={errors[best]:.2f})")

    # 全局最佳拟合
    print(f"\n  === Global best distribution fit ===")
    non_random = {k: v for k, v in results_per_dim.items() if not k.startswith("random_")}
    for dist_name in ["gauss", "t", "laplace", "mix"]:
        errors = [abs(v[f"eff_{dist_name}"] - v["eff_L1"]) / max(v["eff_L1"], 1e-10) for v in non_random.values()]
        print(f"  {dist_name}: mean error = {np.mean(errors):.2f}")

    results = {
        "W_lm_shape": [vocab_size, hidden_dim],
        "results_per_dim": results_per_dim,
    }
    return results


# ========== Main ==========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen3", choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=str, default="all", choices=["p411", "p412", "p413", "p414", "all"])
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)
    timestamp = time.strftime("%Y%m%d_%H%M", time.localtime())

    results = {}
    exps = ["p411", "p412", "p413", "p414"] if args.exp == "all" else [args.exp]

    for exp in exps:
        try:
            if exp == "p411":
                results["p411"] = run_p411(model, tokenizer, device, args.model)
            elif exp == "p412":
                results["p412"] = run_p412(model, tokenizer, device, args.model)
            elif exp == "p413":
                results["p413"] = run_p413(model, tokenizer, device, args.model)
            elif exp == "p414":
                results["p414"] = run_p414(model, tokenizer, device, args.model)
        except Exception as e:
            print(f"Error in {exp}: {e}")
            traceback.print_exc()
            results[exp] = {"error": str(e)}

    out_file = OUT_DIR / f"phase_lxxxiii_p411_414_{args.model}_{timestamp}.json"

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
