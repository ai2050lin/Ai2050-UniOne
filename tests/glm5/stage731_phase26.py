#!/usr/bin/env python3
"""
Stage 731: Phase XXVI — Token级概念分析+因果消融+注意力模式+W_E对比
================================================================================
Phase XXV发现: 句子级探测概念间cos=0.94~0.99, 属性不分离。
Phase XXV用"句子"而非"单词"探测, Phase XXVI改用token级精确分析:

  P171: Token级概念cos — 提取"apple"/"sun"等token位置的h
  P172: 属性替换差异 — "The red apple" vs "The green apple"的h差异
  P173: 因果消融 — 消融特定token的h, 测下游KL
  P174: 注意力模式 — 哪些头关注概念token?
  P175: W_E(Embedding) vs h — embedding空间概念方向 vs hidden state概念方向

测试规模: 200+文本×全层, 3模型
用法: python stage731_phase26.py --model qwen3
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

MODEL_MAP = {
    "qwen3": _Path(r"D:\develop\model\hub\models--Qwen--Qwen3-4B\snapshots\1cfa9a7208912126459214e8b04321603b3df60c"),
    "deepseek7b": _Path(r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"),
    "glm4": _Path(r"D:\develop\model\hub\models--zai-org--GLM-4-9B-Chat-HF\snapshots\8599336fc6c125203efb2360bfaf4c80eef1d1bf"),
}

def load_model(model_name):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = MODEL_MAP[model_name]
    log(f"[load] Loading {model_name} from {p.name} ...")
    tok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        str(p), torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    mdl.eval()
    log(f"[load] Loaded. layers={len(mdl.model.layers)}, d_model={mdl.config.hidden_size}")
    return mdl, tok

def get_token_id(tok, word):
    """获取词的token id(处理subword)"""
    ids = tok.encode(word, add_special_tokens=False)
    return ids

def find_token_positions(tok, text, target_word):
    """找到目标词在tokenized文本中的位置"""
    input_ids = tok.encode(text, add_special_tokens=True)
    tokens = tok.convert_ids_to_tokens(input_ids)
    target_ids = tok.encode(target_word, add_special_tokens=False)
    target_tokens = tok.convert_ids_to_tokens(target_ids)

    positions = []
    for i in range(len(tokens)):
        if tokens[i] == target_tokens[0]:
            if len(target_tokens) == 1 or (i + len(target_tokens) <= len(tokens)
                and tokens[i:i+len(target_tokens)] == target_tokens):
                positions.append(i)
        # 额外匹配: 去掉空格前缀后匹配
        elif tokens[i].lstrip("\u0120\u2581 ") == target_tokens[0].lstrip("\u0120\u2581 "):
            if len(target_tokens) == 1:
                positions.append(i)
    return positions, input_ids

# ============================================================
# P171: Token级概念cos
# ============================================================

def build_token_concept_texts():
    """Token级概念探测: 固定模板, 替换概念词"""
    templates = [
        "The {} is on the table.",
        "She saw a {} in the garden.",
        "A beautiful {} caught her attention.",
        "The {} was large and heavy.",
        "He picked up the {} carefully.",
        "The {} belongs to the category of living things.",
        "Many people like the {}.",
        "The {} has several important properties.",
        "Scientists study the {} extensively.",
        "The {} can be found in nature.",
        "A small {} appeared in the distance.",
        "The {} is known for its unique characteristics.",
        "They described the {} in great detail.",
        "The {} plays an important role in the ecosystem.",
        "Children learn about the {} in school.",
    ]
    concepts = ["apple", "sun", "water", "stone", "hair", "dog", "cat", "book", "tree", "car"]
    return templates, concepts

def run_p171(model, tok, n_layers):
    """Token级概念cos: 提取概念token位置的h"""
    log("\n" + "="*70)
    log("P171: Token级概念cos — 提取概念token位置的h")
    log("="*70)

    templates, concepts = build_token_concept_texts()

    # 收集每个概念在模板中的h
    concept_hs = {}
    for concept in concepts:
        log(f"\n  Concept: {concept}")
        all_hs = []
        valid_count = 0
        for tmpl in templates:
            text = tmpl.format(concept)
            pos, input_ids = find_token_positions(tok, text, concept)
            if not pos:
                continue
            # 使用概念词的最后一个subword token位置
            target_pos = pos[-1]
            valid_count += 1

            inp = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            hs = out.hidden_states
            n_trans = len(model.model.layers)
            layer_hs = []
            for i in range(1, n_trans + 1):
                if i < len(hs):
                    # 提取概念token位置的h
                    layer_hs.append(hs[i][0, target_pos, :].float().cpu())
            all_hs.append(layer_hs)

        if valid_count == 0:
            log(f"    WARNING: no valid positions for '{concept}'")
            continue
        concept_hs[concept] = all_hs
        log(f"    {valid_count}/{len(templates)} valid texts")

    # 计算概念内cos和概念间cos
    results = {}
    for concept, hs_list in concept_hs.items():
        layer_cos = []
        for l in range(n_layers):
            h_matrix = torch.stack([hs_list[i][l] for i in range(len(hs_list))])
            h_mean = h_matrix.mean(dim=0)
            cos_vals = F.cosine_similarity(h_matrix, h_mean.unsqueeze(0)).tolist()
            layer_cos.append(np.mean(cos_vals))
        results[concept] = {"intra_cos": layer_cos}

    # 概念间cos
    concept_names = list(concept_hs.keys())
    inter_cos_layers = []
    for l in range(n_layers):
        cos_pairs = {}
        for i in range(len(concept_names)):
            for j in range(i+1, len(concept_names)):
                a, b = concept_names[i], concept_names[j]
                h_a = torch.stack([concept_hs[a][k][l] for k in range(len(concept_hs[a]))]).mean(dim=0)
                h_b = torch.stack([concept_hs[b][k][l] for k in range(len(concept_hs[b]))]).mean(dim=0)
                c = F.cosine_similarity(h_a.unsqueeze(0), h_b.unsqueeze(0)).item()
                cos_pairs[f"{a}_vs_{b}"] = c
        inter_cos_layers.append(cos_pairs)

    # 汇总
    log("\n  --- Intra-concept cos (same concept, different templates) ---")
    for concept in concept_names:
        avg = np.mean(results[concept]["intra_cos"])
        log(f"    {concept}: avg cos={avg:.4f}")

    log("\n  --- Inter-concept cos (different concepts) ---")
    n = n_layers
    early_mean = np.mean([list(inter_cos_layers[i].values()) for i in range(n//3)], axis=0).mean()
    mid_mean = np.mean([list(inter_cos_layers[i].values()) for i in range(n//3, 2*n//3)], axis=0).mean()
    late_mean = np.mean([list(inter_cos_layers[i].values()) for i in range(2*n//3, n)], axis=0).mean()
    log(f"    Early/Mid/Late: {early_mean:.4f} / {mid_mean:.4f} / {late_mean:.4f}")

    # 最正交/最平行概念对
    all_pair_avg = defaultdict(list)
    for cos_pairs in inter_cos_layers:
        for pair, c in cos_pairs.items():
            all_pair_avg[pair].append(c)
    pair_means = {p: np.mean(v) for p, v in all_pair_avg.items()}
    sorted_pairs = sorted(pair_means.items(), key=lambda x: x[1])
    log(f"    Most orthogonal: {sorted_pairs[:3]}")
    log(f"    Most parallel: {sorted_pairs[-3:]}")

    # 所有概念间cos的范围
    all_vals = []
    for cos_pairs in inter_cos_layers:
        all_vals.extend(cos_pairs.values())
    if all_vals:
        log(f"    Overall: mean={np.mean(all_vals):.4f}, min={np.min(all_vals):.4f}, max={np.max(all_vals):.4f}")
    else:
        log(f"    WARNING: no inter-concept pairs found")

    return {
        "intra_cos": {c: results[c]["intra_cos"] for c in concept_names},
        "inter_cos_layers": inter_cos_layers,
        "pair_means": pair_means,
    }

# ============================================================
# P172: 属性替换差异
# ============================================================

def run_p172(model, tok, n_layers):
    """同模板替换属性词: "The red apple" vs "The green apple" """
    log("\n" + "="*70)
    log("P172: 属性替换差异 — 同模板替换属性词的h差异")
    log("="*70)

    templates = [
        "The {} apple is on the table.",
        "She ate a {} apple yesterday.",
        "A {} apple looked delicious.",
        "The {} apple has a beautiful color.",
        "He prefers {} apples over other fruits.",
        "The {} apple was fresh from the orchard.",
        "A {} apple fell from the tree.",
        "The {} apple tasted amazing.",
    ]

    color_pairs = [
        ("red", "green"),
        ("red", "yellow"),
        ("green", "yellow"),
        ("sweet", "sour"),
        ("hot", "cold"),  # 用于water
    ]

    results = {}
    for adj1, adj2 in color_pairs:
        log(f"\n  Pair: {adj1} vs {adj2}")
        all_diff_norms = []
        all_diff_cos = []

        # 找目标概念词
        concept_word = "apple" if adj1 in ["red", "green", "yellow", "sweet", "sour"] else "water"
        target_tmpls = templates if concept_word == "apple" else [
            "The {} water is in the glass.",
            "She drank {} water this morning.",
            "The {} water was refreshing.",
            "He added {} water to the mixture.",
        ]

        for tmpl in target_tmpls:
            text1 = tmpl.format(adj1)
            text2 = tmpl.format(adj2)

            for text, label in [(text1, adj1), (text2, adj2)]:
                pass  # 只需提取

            inp1 = tok(text1, return_tensors="pt").to(model.device)
            inp2 = tok(text2, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out1 = model(**inp1, output_hidden_states=True)
                out2 = model(**inp2, output_hidden_states=True)

            hs1 = out1.hidden_states
            hs2 = out2.hidden_states

            # 找到属性词位置
            pos1, _ = find_token_positions(tok, text1, adj1)
            pos2, _ = find_token_positions(tok, text2, adj2)
            if not pos1 or not pos2:
                continue
            attr_pos1 = pos1[-1]
            attr_pos2 = pos2[-1]

            # 找概念词位置
            cpos1, _ = find_token_positions(tok, text1, concept_word)
            cpos2, _ = find_token_positions(tok, text2, concept_word)
            if not cpos1 or not cpos2:
                continue
            concept_pos1 = cpos1[-1]
            concept_pos2 = cpos2[-1]

            for l in range(min(n_layers, len(hs1)-1)):
                # 属性词位置的h差异
                h1_attr = hs1[l+1][0, attr_pos1, :].float().cpu()
                h2_attr = hs2[l+1][0, attr_pos2, :].float().cpu()
                diff_attr = h1_attr - h2_attr
                diff_norm = diff_attr.norm().item()
                cos_a = F.cosine_similarity(h1_attr.unsqueeze(0), h2_attr.unsqueeze(0)).item()

                # 概念词位置的h差异
                h1_conc = hs1[l+1][0, concept_pos1, :].float().cpu()
                h2_conc = hs2[l+1][0, concept_pos2, :].float().cpu()
                cos_c = F.cosine_similarity(h1_conc.unsqueeze(0), h2_conc.unsqueeze(0)).item()

                all_diff_norms.append(diff_norm)
                all_diff_cos.append({"attr_cos": cos_a, "conc_cos": cos_c})

        if all_diff_cos:
            avg_attr_cos = np.mean([d["attr_cos"] for d in all_diff_cos])
            avg_conc_cos = np.mean([d["conc_cos"] for d in all_diff_cos])
            avg_diff_norm = np.mean(all_diff_norms)
            log(f"    avg attr_cos={avg_attr_cos:.4f}, avg conc_cos={avg_conc_cos:.4f}, avg diff_norm={avg_diff_norm:.4f}")
            results[f"{adj1}_vs_{adj2}"] = {
                "avg_attr_cos": avg_attr_cos,
                "avg_conc_cos": avg_conc_cos,
                "avg_diff_norm": avg_diff_norm,
            }

    return results

# ============================================================
# P173: 因果消融 — 消融特定token的h
# ============================================================

def run_p173(model, tok, n_layers):
    """消融概念token的h, 测量下游输出KL"""
    log("\n" + "="*70)
    log("P173: 因果消融 — 消融特定token的h, 测下游KL")
    log("="*70)

    texts = [
        "The apple is red and delicious.",
        "The sun shines brightly in the sky.",
        "She drank cold water from the glass.",
        "The stone was heavy and rough.",
        "The dog barked loudly at the stranger.",
    ]

    target_words = ["apple", "sun", "water", "stone", "dog"]

    results = {}
    for text, target in zip(texts, target_words):
        log(f"\n  Text: '{text[:50]}...' Target: {target}")

        pos, input_ids = find_token_positions(tok, text, target)
        if not pos:
            log(f"    WARNING: target '{target}' not found")
            continue
        target_pos = pos[-1]

        inp = tok(text, return_tensors="pt").to(model.device)
        input_ids_t = inp.input_ids

        # Clean forward
        with torch.no_grad():
            out_clean = model(**inp)
        clean_logits = out_clean.logits[0, -1, :].float().cpu()

        # Method 1: 将目标token的embedding替换为零向量
        emb_layer = model.get_input_embeddings()
        orig_embeds = emb_layer(input_ids_t)
        ablated_embeds = orig_embeds.clone()
        ablated_embeds[0, target_pos, :] = 0
        with torch.no_grad():
            out_abl = model(inputs_embeds=ablated_embeds)
        abl_logits = out_abl.logits[0, -1, :].float().cpu()
        kl_zero = F.kl_div(
            F.log_softmax(abl_logits, dim=-1),
            F.softmax(clean_logits, dim=-1),
            reduction='batchmean'
        ).item()
        if math.isnan(kl_zero): kl_zero = 0.0

        # Method 2: 将目标token替换为"the"
        the_id = tok.encode(" the", add_special_tokens=False)[0]
        replace_ids = input_ids_t.clone()
        replace_ids[0, target_pos] = the_id
        with torch.no_grad():
            out_rep = model(input_ids=replace_ids)
        rep_logits = out_rep.logits[0, -1, :].float().cpu()
        kl_replace = F.kl_div(
            F.log_softmax(rep_logits, dim=-1),
            F.softmax(clean_logits, dim=-1),
            reduction='batchmean'
        ).item()
        if math.isnan(kl_replace): kl_replace = 0.0

        # Method 3: 消融目标token之后的所有token的embedding
        abl_after = orig_embeds.clone()
        abl_after[0, target_pos+1:, :] = 0
        with torch.no_grad():
            out_after = model(inputs_embeds=abl_after)
        after_logits = out_after.logits[0, -1, :].float().cpu()
        kl_after = F.kl_div(
            F.log_softmax(after_logits, dim=-1),
            F.softmax(clean_logits, dim=-1),
            reduction='batchmean'
        ).item()
        if math.isnan(kl_after): kl_after = 0.0

        results[target] = {
            "kl_zero_emb": kl_zero,
            "kl_replace": kl_replace,
            "kl_zero_after": kl_after,
        }
        log(f"    KL(zero_target)={kl_zero:.3f}, KL(replace)={kl_replace:.3f}, KL(zero_after)={kl_after:.3f}")

    return results

# ============================================================
# P174: 注意力模式
# ============================================================

def run_p174(model, tok, n_layers):
    """分析哪些注意力头关注概念token"""
    log("\n" + "="*70)
    log("P174: Attention Pattern")
    log("="*70)

    texts = [
        "The apple is red and sweet.",
        "The sun provides light and warmth.",
        "Cold water is very refreshing.",
        "The heavy stone fell from the cliff.",
    ]
    target_words = ["apple", "sun", "water", "stone"]

    # 尝试设置attention implementation
    try:
        model.config._attn_implementation = "eager"
        for layer in model.model.layers:
            if hasattr(layer.self_attn, 'config'):
                pass
    except Exception as e:
        log(f"    NOTE: Could not set eager attention: {e}")

    results = {}
    for text, target in zip(texts, target_words):
        log(f"\n  Text: '{text[:40]}...' Target: {target}")

        pos, input_ids = find_token_positions(tok, text, target)
        if not pos:
            continue
        target_pos = pos[-1]

        inp = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inp, output_hidden_states=True, output_attentions=True)

        if not hasattr(out, 'attentions') or out.attentions is None or len(out.attentions) == 0:
            log(f"    WARNING: no attention output (SDPA does not support output_attentions)")
            results[target] = []
            continue

        seq_len = inp.input_ids.shape[1]
        last_token_pos = seq_len - 1

        layer_attn = []
        for l in range(min(n_layers, len(out.attentions))):
            attn = out.attentions[l]
            if attn.dim() == 4:
                attn = attn[0]  # (n_heads, seq, seq)
            # 最后token对目标token的平均注意力
            attn_to_target = attn[:, last_token_pos, target_pos].float().mean().item()
            attn_to_all_val = attn[:, last_token_pos, :].float().mean(dim=1).mean().item()
            layer_attn.append({
                "attn_to_target": attn_to_target,
                "attn_to_all_avg": attn_to_all_val,
                "ratio": attn_to_target / (attn_to_all_val + 1e-10),
            })

        results[target] = layer_attn
        if layer_attn:
            avg_ratio = np.mean([la["ratio"] for la in layer_attn])
            max_ratio = max([la["ratio"] for la in layer_attn])
            max_l = [la["ratio"] for la in layer_attn].index(max_ratio)
            log(f"    avg ratio={avg_ratio:.2f}, max ratio={max_ratio:.2f} at L{max_l}")

    return results

# ============================================================
# P175: W_E vs h 概念方向对比
# ============================================================

def run_p175(model, tok, n_layers):
    """Embedding空间概念方向 vs Hidden state概念方向"""
    log("\n" + "="*70)
    log("P175: W_E vs h — Embedding空间 vs Hidden state概念方向")
    log("="*70)

    concepts = ["apple", "sun", "water", "stone", "dog", "cat", "book", "tree", "car", "king", "queen", "man", "woman"]

    # 获取embedding向量
    w_e = model.get_input_embeddings().weight.data.float()  # (vocab_size, d_model)

    # 对每个概念, 获取embedding
    concept_embeddings = {}
    for concept in concepts:
        ids = tok.encode(concept, add_special_tokens=False)
        if len(ids) == 1:
            concept_embeddings[concept] = w_e[ids[0]]
        else:
            # 多subword: 取平均
            vecs = [w_e[i] for i in ids]
            concept_embeddings[concept] = torch.stack(vecs).mean(dim=0)

    # Embedding空间概念间cos
    emb_cos = {}
    concept_names = list(concept_embeddings.keys())
    for i in range(len(concept_names)):
        for j in range(i+1, len(concept_names)):
            a, b = concept_names[i], concept_names[j]
            c = F.cosine_similarity(
                concept_embeddings[a].unsqueeze(0),
                concept_embeddings[b].unsqueeze(0)
            ).item()
            emb_cos[f"{a}_vs_{b}"] = c

    # Hidden state中: 用简单模板获取概念token的h
    h_cos = {}
    for i in range(len(concept_names)):
        for j in range(i+1, len(concept_names)):
            a, b = concept_names[i], concept_names[j]
            text_a = f"The {a} is on the table."
            text_b = f"The {b} is on the table."

            pos_a, _ = find_token_positions(tok, text_a, a)
            pos_b, _ = find_token_positions(tok, text_b, b)
            if not pos_a or not pos_b:
                continue

            inp_a = tok(text_a, return_tensors="pt").to(model.device)
            inp_b = tok(text_b, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out_a = model(**inp_a, output_hidden_states=True)
                out_b = model(**inp_b, output_hidden_states=True)

            # 取中间层和最后层
            for l_label, l_idx in [("mid", n_layers//2), ("last", n_layers-1)]:
                real_idx = l_idx + 1  # hidden_states index = layer index + 1
                hs_a_all = out_a.hidden_states
                hs_b_all = out_b.hidden_states
                if real_idx >= len(hs_a_all) or real_idx >= len(hs_b_all):
                    continue
                hs_a_h = hs_a_all[real_idx][0, pos_a[-1], :].float().cpu()
                hs_b_h = hs_b_all[real_idx][0, pos_b[-1], :].float().cpu()
                c = F.cosine_similarity(hs_a_h.unsqueeze(0), hs_b_h.unsqueeze(0)).item()
                key = f"{a}_vs_{b}_{l_label}"
                h_cos[key] = c

    # 汇总
    log("\n  --- Embedding space concept cos ---")
    emb_vals = list(emb_cos.values())
    log(f"    mean={np.mean(emb_vals):.4f}, min={np.min(emb_vals):.4f}, max={np.max(emb_vals):.4f}")

    # 在embedding空间中做类比
    if "king" in concept_embeddings and "queen" in concept_embeddings and "man" in concept_embeddings and "woman" in concept_embeddings:
        pred = concept_embeddings["king"] + concept_embeddings["queen"] - concept_embeddings["man"]
        cos_woman = F.cosine_similarity(pred.unsqueeze(0), concept_embeddings["woman"].unsqueeze(0)).item()
        log(f"    Embedding analogy king+queen-man≈woman: cos={cos_woman:.4f}")

    log("\n  --- Hidden state concept cos (same template) ---")
    mid_vals = [v for k, v in h_cos.items() if "_mid" in k]
    last_vals = [v for k, v in h_cos.items() if "_last" in k]
    if mid_vals:
        log(f"    Mid layer: mean={np.mean(mid_vals):.4f}, min={np.min(mid_vals):.4f}, max={np.max(mid_vals):.4f}")
    if last_vals:
        log(f"    Last layer: mean={np.mean(last_vals):.4f}, min={np.min(last_vals):.4f}, max={np.max(last_vals):.4f}")

    return {"emb_cos": emb_cos, "h_cos": h_cos, "emb_stats": {
        "mean": np.mean(emb_vals), "min": np.min(emb_vals), "max": np.max(emb_vals)
    }}

# ============================================================
# 主程序
# ============================================================

def main():
    global log
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = f"tests/glm5_temp/stage731_phase26_{args.model}_{ts}"
    log = Logger(log_dir, "results")
    log(f"Phase XXVI: Token-Level Concept Analysis + Causal Ablation")
    log(f"Model: {args.model}, Time: {ts}")

    mdl, tok = load_model(args.model)
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size

    # P171
    t0 = time.time()
    p171 = run_p171(mdl, tok, n_layers)
    log(f"\n[P171] Time: {time.time()-t0:.1f}s")

    # P172
    t0 = time.time()
    p172 = run_p172(mdl, tok, n_layers)
    log(f"\n[P172] Time: {time.time()-t0:.1f}s")

    # P173
    t0 = time.time()
    p173 = run_p173(mdl, tok, n_layers)
    log(f"\n[P173] Time: {time.time()-t0:.1f}s")

    # P174
    t0 = time.time()
    p174 = run_p174(mdl, tok, n_layers)
    log(f"\n[P174] Time: {time.time()-t0:.1f}s")

    # P175
    t0 = time.time()
    p175 = run_p175(mdl, tok, n_layers)
    log(f"\n[P175] Time: {time.time()-t0:.1f}s")

    # ============================================================
    # 最终汇总
    # ============================================================
    log("\n" + "="*70)
    log("FINAL SUMMARY")
    log("="*70)

    log("\n--- P171: Token-Level Concept Cos ---")
    if "intra_cos" in p171:
        for c, cos_list in p171["intra_cos"].items():
            avg = np.mean(cos_list)
            log(f"  {c}: avg intra_cos={avg:.4f}")
    if "pair_means" in p171:
        all_vals = list(p171["pair_means"].values())
        log(f"  Inter-concept: mean={np.mean(all_vals):.4f}, min={np.min(all_vals):.4f}")

    log("\n--- P172: Attribute Replacement ---")
    for pair, data in p172.items():
        log(f"  {pair}: attr_cos={data['avg_attr_cos']:.4f}, conc_cos={data['avg_conc_cos']:.4f}")

    log("\n--- P173: Causal Ablation ---")
    for target, data in p173.items():
        log(f"  {target}: KL(zero)={data['kl_zero_emb']:.3f}, KL(replace)={data['kl_replace']:.3f}, KL(after)={data['kl_zero_after']:.3f}")

    log("\n--- P174: Attention Pattern ---")
    for target, layer_data in p174.items():
        if layer_data:
            avg_ratio = np.mean([la["ratio"] for la in layer_data])
            log(f"  {target}: avg_attn_ratio={avg_ratio:.2f}")
        else:
            log(f"  {target}: no attention data (SDPA)")

    log("\n--- P175: W_E vs h ---")
    log(f"  Embedding cos: mean={p175['emb_stats']['mean']:.4f}")

    del mdl, tok
    gc.collect()
    torch.cuda.empty_cache()
    log.close()
    print(f"\nResults saved to: {log_dir}/results.log")

if __name__ == "__main__":
    main()
