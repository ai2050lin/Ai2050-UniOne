#!/usr/bin/env python3
"""
Stage 732: Phase XXVII — 逻辑推理追踪+全层EMB→HS变换+float32因果验证
================================================================================
Phase XXVI发现: Token级概念间cos=0.50~0.71(有明确区分), EMB cos=0.04~0.08。
但P173因果消融KL=0(bfloat16精度不足), 且只测了mid/last层的EMB→HS变换。

Phase XXVII核心任务:
  P176: float32因果消融 — 用torch.float32重新运行P173(修复KL=0)
  P177: 全层EMB→HS变换追踪 — 每层的inter-concept cos, 找"概念分离"关键层
  P178: 逻辑推理逐层追踪 — "All A are B. X is A. Therefore X is B."
  P179: 属性因果验证 — 替换"red"→"green"后测下游输出变化(float32)
  P180: 大规模概念图谱 — 30+概念的token级cos矩阵

测试规模: 300+文本×全层, 3模型
用法: python stage732_phase27.py --model qwen3
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
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    log(f"[load] Loaded. layers={n_layers}, d_model={d_model}")
    return mdl, tok

def load_model_float32(model_name):
    """加载float32模型用于因果消融"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    p = MODEL_MAP[model_name]
    log(f"[load-fp32] Loading {model_name} in float32 ...")
    tok = AutoTokenizer.from_pretrained(str(p), trust_remote_code=True)
    mdl = AutoModelForCausalLM.from_pretrained(
        str(p), torch_dtype=torch.float32, device_map="auto", trust_remote_code=True
    )
    mdl.eval()
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size
    log(f"[load-fp32] Loaded. layers={n_layers}, d_model={d_model}")
    return mdl, tok

def find_token_positions(tok, text, target_word):
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
        elif tokens[i].lstrip("\u0120\u2581 ") == target_tokens[0].lstrip("\u0120\u2581 "):
            if len(target_tokens) == 1:
                positions.append(i)
    return positions, input_ids

def compute_kl(p, q, eps=1e-10):
    """计算KL(p||q)"""
    p = p.float()
    q = q.float()
    log_p = torch.log(p + eps)
    log_q = torch.log(q + eps)
    kl = (p * (log_p - log_q)).sum(-1).mean().item()
    return kl

def get_output_probs(model, tok, text, target_pos=None):
    """获取模型输出logits并转为probs"""
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=False)
    logits = out.logits  # [1, seq_len, vocab]
    if target_pos is not None:
        logits = logits[:, target_pos, :]
    else:
        logits = logits[:, -1, :]
    probs = F.softmax(logits.float(), dim=-1)
    return probs

# ============================================================
# P176: float32因果消融 — 修复P173的KL=0
# ============================================================

def run_p176(model_name):
    """float32因果消融: 验证消融概念token的embedding对下游的影响"""
    log("\n" + "="*70)
    log("P176: float32因果消融 — 修复P173的KL=0")
    log("="*70)

    log(f"\n  Loading model in float32 for causal ablation...")
    mdl, tok = load_model_float32(model_name)
    n_layers = len(mdl.model.layers)

    # 消融目标
    ablation_texts = [
        ("The apple is on the table.", "apple"),
        ("She ate a red apple yesterday.", "apple"),
        ("The sun rises in the east.", "sun"),
        ("He gave the dog a bone.", "dog"),
        ("The tree grows tall and strong.", "tree"),
        ("Water is essential for life.", "water"),
        ("The stone was heavy and rough.", "stone"),
        ("She read an interesting book.", "book"),
        ("A beautiful cat sat on the mat.", "cat"),
        ("The car drove down the road.", "car"),
    ]

    results = {}

    for text, target_word in ablation_texts:
        log(f"\n  Text: '{text}', target: '{target_word}'")

        pos_list, input_ids = find_token_positions(tok, text, target_word)
        if not pos_list:
            log(f"    WARNING: token '{target_word}' not found, skipping")
            continue
        target_pos = pos_list[-1]

        # 原始输出
        probs_orig = get_output_probs(mdl, tok, text)
        top5_orig = torch.topk(probs_orig, 5)
        top5_words_orig = tok.convert_ids_to_tokens(top5_orig.indices[0].tolist())

        # 方法1: 零化target token的embedding
        inp = tok(text, return_tensors="pt").to(mdl.device)
        embeds = mdl.get_input_embeddings()(inp.input_ids).clone()
        embeds_orig = embeds.clone()
        embeds[0, target_pos, :] = 0.0  # 零化
        with torch.no_grad():
            out_zero = mdl(inputs_embeds=embeds)
        probs_zero = F.softmax(out_zero.logits[:, -1, :].float(), dim=-1)
        kl_zero = compute_kl(probs_orig, probs_zero)

        # 方法2: 替换为随机embedding
        embeds_rand = embeds_orig.clone()
        embeds_rand[0, target_pos, :] = torch.randn_like(embeds_rand[0, target_pos, :]) * 0.1
        with torch.no_grad():
            out_rand = mdl(inputs_embeds=embeds_rand)
        probs_rand = F.softmax(out_rand.logits[:, -1, :].float(), dim=-1)
        kl_rand = compute_kl(probs_orig, probs_rand)

        # 方法3: 替换为"the"的embedding
        the_ids = tok.encode("the", add_special_tokens=False)
        the_emb = mdl.get_input_embeddings()(torch.tensor(the_ids).to(mdl.device))[0]
        embeds_the = embeds_orig.clone()
        embeds_the[0, target_pos, :] = the_emb
        with torch.no_grad():
            out_the = mdl(inputs_embeds=embeds_the)
        probs_the = F.softmax(out_the.logits[:, -1, :].float(), dim=-1)
        kl_the = compute_kl(probs_orig, probs_the)

        # 方法4: 逐步增大零化幅度
        scale_results = {}
        for scale in [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]:
            embeds_sc = embeds_orig.clone()
            embeds_sc[0, target_pos, :] = embeds_orig[0, target_pos, :] * (1.0 - scale)
            with torch.no_grad():
                out_sc = mdl(inputs_embeds=embeds_sc)
            probs_sc = F.softmax(out_sc.logits[:, -1, :].float(), dim=-1)
            kl_sc = compute_kl(probs_orig, probs_sc)
            scale_results[f"scale_{scale}"] = round(kl_sc, 6)

        # 方法5: 替换为同类但不同概念的embedding
        concept_map = {
            "apple": "orange", "sun": "moon", "dog": "cat", "tree": "flower",
            "water": "milk", "stone": "brick", "book": "magazine", "car": "bus"
        }
        replace_word = concept_map.get(target_word, "thing")
        rep_ids = tok.encode(replace_word, add_special_tokens=False)
        rep_emb = mdl.get_input_embeddings()(torch.tensor(rep_ids).to(mdl.device))[0]
        embeds_rep = embeds_orig.clone()
        embeds_rep[0, target_pos, :] = rep_emb
        with torch.no_grad():
            out_rep = mdl(inputs_embeds=embeds_rep)
        probs_rep = F.softmax(out_rep.logits[:, -1, :].float(), dim=-1)
        kl_rep = compute_kl(probs_orig, probs_rep)

        top5_zero = torch.topk(probs_zero, 5)
        top5_words_zero = tok.convert_ids_to_tokens(top5_zero.indices[0].tolist()) if top5_zero.indices.dim() > 1 else tok.convert_ids_to_tokens(top5_zero.indices.tolist())

        log(f"    KL(zero_emb)={kl_zero:.6f}, KL(rand_emb)={kl_rand:.6f}, KL(the_emb)={kl_the:.6f}, KL(replace={replace_word})={kl_rep:.6f}")
        log(f"    Scale sweep: {json.dumps(scale_results)}")
        log(f"    Top5 orig: {top5_words_orig}")
        log(f"    Top5 zero: {top5_words_zero}")

        results[target_word] = {
            "KL_zero": round(kl_zero, 6),
            "KL_rand": round(kl_rand, 6),
            "KL_the": round(kl_the, 6),
            "KL_replace": round(kl_rep, 6),
            "replace_word": replace_word,
            "scale_results": scale_results,
            "top5_orig": top5_words_orig,
            "top5_zero": top5_words_zero,
        }

    # 汇总
    all_kl_zero = [r["KL_zero"] for r in results.values()]
    all_kl_rand = [r["KL_rand"] for r in results.values()]
    all_kl_the = [r["KL_the"] for r in results.values()]
    all_kl_rep = [r["KL_replace"] for r in results.values()]

    log(f"\n  === P176 Summary ===")
    log(f"    KL(zero_emb): mean={np.mean(all_kl_zero):.6f}, max={np.max(all_kl_zero):.6f}, nonzero={sum(1 for k in all_kl_zero if k > 0.0001)}/{len(all_kl_zero)}")
    log(f"    KL(rand_emb): mean={np.mean(all_kl_rand):.6f}, max={np.max(all_kl_rand):.6f}, nonzero={sum(1 for k in all_kl_rand if k > 0.0001)}/{len(all_kl_rand)}")
    log(f"    KL(the_emb): mean={np.mean(all_kl_the):.6f}, max={np.max(all_kl_the):.6f}, nonzero={sum(1 for k in all_kl_the if k > 0.0001)}/{len(all_kl_the)}")
    log(f"    KL(replace): mean={np.mean(all_kl_rep):.6f}, max={np.max(all_kl_rep):.6f}, nonzero={sum(1 for k in all_kl_rep if k > 0.0001)}/{len(all_kl_rep)}")

    # 判断float32是否修复了KL=0
    if np.mean(all_kl_rep) > 0.001:
        log(f"\n  ** float32修复了KL=0! 替换概念embedding有显著因果效应 **")
    elif np.mean(all_kl_rep) > 0.0001:
        log(f"\n  ** float32部分修复: KL>0但效应很小 **")
    else:
        log(f"\n  ** float32仍未修复KL=0! 问题可能不是精度, 而是inputs_embeds被忽略 **")

    del mdl
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================
# P177: 全层EMB→HS变换追踪
# ============================================================

def run_p177(model, tok, n_layers):
    """全层EMB→HS变换: 每层的inter-concept cos"""
    log("\n" + "="*70)
    log("P177: 全层EMB→HS变换追踪 — 每层的inter-concept cos")
    log("="*70)

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
        "I bought a new {} at the store.",
        "The {} is a common household item.",
        "Everyone needs a {} in daily life.",
        "The {} was discovered long ago.",
        "A {} can be used for many purposes.",
    ]

    concepts = ["apple", "sun", "water", "stone", "dog", "cat", "book", "tree",
                "car", "house", "flower", "bird", "fish", "mountain", "river",
                "fire", "cloud", "moon", "star", "rain"]

    # 提取embedding空间的概念方向
    log(f"\n  Extracting embedding directions for {len(concepts)} concepts...")
    emb_vectors = {}
    for concept in concepts:
        ids = tok.encode(concept, add_special_tokens=False)
        if len(ids) == 1:
            with torch.no_grad():
                emb = model.get_input_embeddings()(torch.tensor(ids).to(model.device))
            emb_vectors[concept] = emb[0].float().cpu()
        else:
            # 多subword: 拼接embedding
            embs = []
            for tid in ids:
                with torch.no_grad():
                    e = model.get_input_embeddings()(torch.tensor([tid]).to(model.device))
                embs.append(e[0].float().cpu())
            emb_vectors[concept] = sum(embs) / len(embs)

    # Embedding空间的cos
    log(f"\n  --- Embedding space concept pairs ---")
    emb_concept_names = list(emb_vectors.keys())
    emb_cos_matrix = {}
    for i in range(len(emb_concept_names)):
        for j in range(i+1, len(emb_concept_names)):
            a, b = emb_concept_names[i], emb_concept_names[j]
            c = F.cosine_similarity(emb_vectors[a].unsqueeze(0), emb_vectors[b].unsqueeze(0)).item()
            emb_cos_matrix[f"{a}_{b}"] = c

    emb_all_cos = list(emb_cos_matrix.values())
    log(f"    Embedding inter-concept cos: mean={np.mean(emb_all_cos):.4f}, min={np.min(emb_all_cos):.4f}, max={np.max(emb_all_cos):.4f}")

    # 全层hidden state追踪
    log(f"\n  Extracting hidden states across all {n_layers} layers...")
    concept_hs = {}  # concept -> list of (layer, h_vector)

    for concept in concepts:
        log(f"    Processing concept: {concept}")
        layer_hs_sum = [torch.zeros(model.config.hidden_size) for _ in range(n_layers)]
        valid_count = 0

        for tmpl in templates:
            text = tmpl.format(concept)
            pos_list, input_ids = find_token_positions(tok, text, concept)
            if not pos_list:
                continue
            target_pos = pos_list[-1]
            valid_count += 1

            inp = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            hs = out.hidden_states

            for l_idx in range(n_layers):
                layer_idx = l_idx + 1  # hs[0]=embedding, hs[1]=layer0 output
                if layer_idx < len(hs):
                    layer_hs_sum[l_idx] += hs[layer_idx][0, target_pos, :].float().cpu()

        if valid_count == 0:
            log(f"      WARNING: no valid positions")
            continue

        # 平均
        concept_hs[concept] = [h / valid_count for h in layer_hs_sum]
        log(f"      {valid_count}/{len(templates)} valid texts")

    # 每层inter-concept cos
    log(f"\n  --- Per-layer inter-concept cos ---")
    layer_inter_cos = []
    for l in range(n_layers):
        cos_pairs = {}
        cnames = list(concept_hs.keys())
        for i in range(len(cnames)):
            for j in range(i+1, len(cnames)):
                a, b = cnames[i], cnames[j]
                c = F.cosine_similarity(concept_hs[a][l].unsqueeze(0), concept_hs[b][l].unsqueeze(0)).item()
                cos_pairs[f"{a}_{b}"] = c
        mean_cos = np.mean(list(cos_pairs.values()))
        std_cos = np.std(list(cos_pairs.values()))
        layer_inter_cos.append({"mean": mean_cos, "std": std_cos, "min": np.min(list(cos_pairs.values())),
                                "max": np.max(list(cos_pairs.values()))})
        if l % 5 == 0 or l == n_layers - 1:
            log(f"    Layer {l}: mean={mean_cos:.4f}, std={std_cos:.4f}, range=[{np.min(list(cos_pairs.values())):.4f}, {np.max(list(cos_pairs.values())):.4f}]")

    # 找关键层: EMB→HS变化最大的层
    log(f"\n  --- EMB→HS transformation: key layers ---")
    emb_mean = np.mean(emb_all_cos)
    log(f"    Embedding mean cos: {emb_mean:.4f}")

    # 每层与embedding cos的差异
    deltas = []
    for l in range(n_layers):
        delta = layer_inter_cos[l]["mean"] - emb_mean
        deltas.append(delta)
        if l % 5 == 0 or l == n_layers - 1:
            log(f"    Layer {l}: HS cos={layer_inter_cos[l]['mean']:.4f}, delta from EMB={delta:+.4f}")

    # 最小cos的层 = 概念分离最大的层
    min_layer = np.argmin([c["mean"] for c in layer_inter_cos])
    max_layer = np.argmax([c["mean"] for c in layer_inter_cos])
    log(f"\n    Best separation layer: L{min_layer} (cos={layer_inter_cos[min_layer]['mean']:.4f})")
    log(f"    Worst separation layer: L{max_layer} (cos={layer_inter_cos[max_layer]['mean']:.4f})")
    log(f"    Separation range: {emb_mean:.4f} (EMB) → {layer_inter_cos[min_layer]['mean']:.4f} (best) → {layer_inter_cos[-1]['mean']:.4f} (last)")

    # 层级分析
    n = n_layers
    early = np.mean([layer_inter_cos[i]["mean"] for i in range(n//3)])
    mid = np.mean([layer_inter_cos[i]["mean"] for i in range(n//3, 2*n//3)])
    late = np.mean([layer_inter_cos[i]["mean"] for i in range(2*n//3, n)])
    log(f"    Early/Mid/Late: {early:.4f} / {mid:.4f} / {late:.4f}")

    # Embedding到HS的变换幅度: cos(emb_dir, hs_dir) per concept
    log(f"\n  --- Per-concept EMB→HS alignment ---")
    concept_alignments = {}
    for concept in concept_hs:
        emb_v = emb_vectors.get(concept)
        hs_v = concept_hs[concept][-1]  # last layer
        if emb_v is not None:
            align = F.cosine_similarity(emb_v.unsqueeze(0), hs_v.unsqueeze(0)).item()
            concept_alignments[concept] = align
            log(f"    {concept}: cos(EMB, HS_last)={align:.4f}")

    # 最正交/最对齐的概念
    sorted_align = sorted(concept_alignments.items(), key=lambda x: x[1])
    log(f"    Most changed by layers: {sorted_align[:3]}")
    log(f"    Least changed by layers: {sorted_align[-3:]}")

    return {
        "emb_mean_cos": emb_mean,
        "layer_inter_cos": layer_inter_cos,
        "best_sep_layer": min_layer,
        "worst_sep_layer": max_layer,
        "early_mid_late": [early, mid, late],
        "concept_alignments": concept_alignments,
        "n_concepts": len(concepts),
    }


# ============================================================
# P178: 逻辑推理逐层追踪
# ============================================================

def run_p178(model, tok, n_layers):
    """逻辑推理逐层追踪: "All A are B. X is A. Therefore X is B." """
    log("\n" + "="*70)
    log("P178: 逻辑推理逐层追踪")
    log("="*70)

    # 逻辑推理文本集
    reasoning_texts = [
        # 三段论
        ("All cats are animals. Tom is a cat. Therefore, Tom is", "animal", "三段论(cats→animals)"),
        ("All birds can fly. Robin is a bird. Therefore, Robin can", "fly", "三段论(birds→fly)"),
        ("All roses are flowers. This plant is a rose. Therefore, this plant is", "a flower", "三段论(roses→flowers)"),
        ("All metals conduct electricity. Copper is a metal. Therefore, copper", "conducts", "三段论(metals→conduct)"),
        ("All mammals are warm-blooded. Whales are mammals. Therefore, whales are", "warm-blooded", "三段论(mammals→warm)"),

        # 条件推理
        ("If it rains, the ground gets wet. It is raining. Therefore, the ground is", "wet", "条件推理(rain→wet)"),
        ("If you study hard, you will pass. You studied hard. Therefore, you will", "pass", "条件推理(study→pass)"),
        ("If the temperature drops below zero, water freezes. The temperature is minus five. Therefore, water", "freezes", "条件推理(freeze)"),

        # 传递性
        ("A is greater than B. B is greater than C. Therefore, A is greater than", "C", "传递性(greater)"),
        ("Alice is older than Bob. Bob is older than Carol. Therefore, Alice is older than", "Carol", "传递性(older)"),
        ("Red is warmer than blue. Blue is warmer than green. Therefore, red is warmer than", "green", "传递性(warmer)"),

        # 反事实推理
        ("If the coin is fair, the probability of heads is 0.5. The coin landed heads. This is", "expected", "反事实(probability)"),
        ("Normally, water boils at 100 degrees. At high altitude, water boils at a lower temperature. Therefore, on a mountain, water boils", "earlier", "反事实(boiling)"),

        # 否定推理
        ("All dogs are animals. Cats are not dogs. Cats may or may not be", "animals", "否定推理(not→uncertain)"),
        ("No fish can fly. Salmon is a fish. Therefore, salmon cannot", "fly", "否定推理(no)"),
        ("Not all birds can fly. Penguins are birds. Therefore, penguins", "cannot fly", "否定推理(not all)"),

        # 量词推理
        ("Some students passed the exam. John is a student. Therefore, John", "may or may not have passed", "量词(some)"),
        ("Most people like chocolate. Mary is a person. Therefore, Mary probably", "likes chocolate", "量词(most)"),
        ("Every square is a rectangle. This shape is a square. Therefore, this shape is", "a rectangle", "量词(every)"),

        # 组合属性
        ("The box is red and heavy. The other box is red and light. What do they have in common? They are both", "red", "组合属性(common)"),
        ("John is tall and smart. Mary is short and smart. Both John and Mary are", "smart", "组合属性(intersection)"),
    ]

    results = []

    for text, expected, desc in reasoning_texts:
        log(f"\n  [{desc}] '{text}' → expected: '{expected}'")

        inp = tok(text, return_tensors="pt").to(model.device)
        seq_len = inp.input_ids.shape[1]

        with torch.no_grad():
            out = model(**inp, output_hidden_states=True)

        hs = out.hidden_states
        logits = out.logits

        # 最后一步的top预测
        last_logits = logits[0, -1, :].float()
        last_probs = F.softmax(last_logits, dim=-1)
        top5 = torch.topk(last_probs, 5)
        top5_tokens = tok.convert_ids_to_tokens(top5.indices.tolist())
        top5_probs = top5.values.tolist()

        # 检查expected是否在top5中
        expected_ids = tok.encode(expected, add_special_tokens=False)
        expected_matched = False
        for eid in expected_ids:
            prob = last_probs[eid].item()
            if prob > 0.01:
                expected_matched = True
                log(f"    Expected '{expected}' (id={eid}): prob={prob:.4f}")
                break
        if not expected_matched:
            log(f"    Expected '{expected}': NOT found in high-prob tokens")

        log(f"    Top5: {list(zip(top5_tokens, [f'{p:.4f}' for p in top5_probs]))}")

        # 逐层hidden state的"推理信息"测量
        # 用"前提token"和"结论token"位置的h差异来追踪推理过程
        layer_norms = []
        layer_cos_with_final = []
        h_final = hs[-1][0, -1, :].float().cpu()

        for l in range(1, min(len(hs), n_layers + 1)):
            h = hs[l][0, -1, :].float().cpu()
            norm_val = h.norm().item()
            cos_val = F.cosine_similarity(h.unsqueeze(0), h_final.unsqueeze(0)).item()
            layer_norms.append(norm_val)
            layer_cos_with_final.append(cos_val)

        # 找"推理跃迁"层: cos变化最大的层
        cos_diffs = []
        for i in range(len(layer_cos_with_final) - 1):
            diff = layer_cos_with_final[i+1] - layer_cos_with_final[i]
            cos_diffs.append(diff)
        if cos_diffs:
            max_diff_idx = np.argmax(np.abs(cos_diffs))
            log(f"    Reasoning transition layer: L{max_diff_idx} (cos diff={cos_diffs[max_diff_idx]:+.4f})")
            log(f"    Layer norms: first={layer_norms[0]:.1f}, last={layer_norms[-1]:.1f}, ratio={layer_norms[-1]/layer_norms[0]:.1f}x")

        # early/mid/late cos with final
        n = len(layer_cos_with_final)
        early_cos = np.mean(layer_cos_with_final[:n//3])
        mid_cos = np.mean(layer_cos_with_final[n//3:2*n//3])
        late_cos = np.mean(layer_cos_with_final[2*n//3:])

        results.append({
            "desc": desc,
            "expected": expected,
            "top5": list(zip(top5_tokens, [round(p, 4) for p in top5_probs])),
            "expected_matched": expected_matched,
            "reasoning_transition_layer": int(max_diff_idx) if cos_diffs else -1,
            "early_mid_late_cos": [round(early_cos, 4), round(mid_cos, 4), round(late_cos, 4)],
            "norm_ratio": round(layer_norms[-1]/layer_norms[0], 1) if layer_norms[0] > 0 else 0,
            "cos_with_final": [round(c, 4) for c in layer_cos_with_final],
            "layer_norms": [round(n_val, 1) for n_val in layer_norms],
        })

    # 汇总
    log(f"\n  === P178 Summary ===")
    matched_count = sum(1 for r in results if r["expected_matched"])
    log(f"    Reasoning accuracy (expected in top-prob): {matched_count}/{len(results)} ({100*matched_count/len(results):.0f}%)")

    # 推理跃迁层统计
    transition_layers = [r["reasoning_transition_layer"] for r in results if r["reasoning_transition_layer"] >= 0]
    if transition_layers:
        log(f"    Reasoning transition layers: mean={np.mean(transition_layers):.1f}, std={np.std(transition_layers):.1f}, range=[{np.min(transition_layers)}, {np.max(transition_layers)}]")

    # 分类准确率
    categories = defaultdict(lambda: {"total": 0, "matched": 0})
    for r in results:
        desc = r["desc"]
        cat = desc.split("(")[0] if "(" in desc else desc
        categories[cat]["total"] += 1
        if r["expected_matched"]:
            categories[cat]["matched"] += 1

    log(f"\n  --- Per-category accuracy ---")
    for cat, vals in sorted(categories.items()):
        acc = 100 * vals["matched"] / vals["total"] if vals["total"] > 0 else 0
        log(f"    {cat}: {vals['matched']}/{vals['total']} ({acc:.0f}%)")

    return results


# ============================================================
# P179: 属性因果验证 — float32
# ============================================================

def run_p179(model_name):
    """属性因果验证: 替换属性词后测下游输出变化(float32)"""
    log("\n" + "="*70)
    log("P179: 属性因果验证 — float32属性替换因果效应")
    log("="*70)

    log(f"\n  Loading model in float32...")
    mdl, tok = load_model_float32(model_name)
    n_layers = len(mdl.model.layers)

    # 属性替换文本对
    attr_tests = [
        ("The red apple is on the table.", "The green apple is on the table.", "red→green", "apple"),
        ("She ate a sweet apple.", "She ate a sour apple.", "sweet→sour", "apple"),
        ("The hot water burned his hand.", "The cold water refreshed him.", "hot→cold", "water"),
        ("The tall tree provided shade.", "The short tree offered no shade.", "tall→short", "tree"),
        ("The heavy stone was hard to lift.", "The light stone was easy to carry.", "heavy→light", "stone"),
        ("The fast car raced down the highway.", "The slow car crawled along the road.", "fast→slow", "car"),
        ("The big dog barked loudly.", "The small dog barked quietly.", "big→small", "dog"),
        ("The thick book took weeks to read.", "The thin book was finished in a day.", "thick→thin", "book"),
        ("The bright sun blinded him.", "The dim sun was barely visible.", "bright→dim", "sun"),
        ("The loud noise woke everyone up.", "The quiet noise went unnoticed.", "loud→quiet", "noise"),
        ("The old house needed repairs.", "The new house was in perfect condition.", "old→new", "house"),
        ("The deep river was dangerous to cross.", "The shallow river was easy to wade through.", "deep→shallow", "river"),
        ("The sharp knife cut easily.", "The dull knife struggled to cut.", "sharp→dull", "knife"),
        ("The smooth surface reflected light.", "The rough surface scattered light.", "smooth→rough", "surface"),
        ("The warm fire heated the room.", "The cold fire went out quickly.", "warm→cold", "fire"),
    ]

    results = []

    for text_orig, text_modified, attr_pair, target_word in attr_tests:
        log(f"\n  [{attr_pair}] '{text_orig}' → '{text_modified}'")

        # 原始输出
        probs_orig = get_output_probs(mdl, tok, text_orig)
        top5_orig = torch.topk(probs_orig, 5)
        top5_words_orig = tok.convert_ids_to_tokens(top5_orig.indices[0].tolist())

        # 修改后输出
        probs_mod = get_output_probs(mdl, tok, text_modified)
        top5_mod = torch.topk(probs_mod, 5)
        top5_words_mod = tok.convert_ids_to_tokens(top5_mod.indices[0].tolist())

        # KL divergence
        kl = compute_kl(probs_orig, probs_mod)

        # top-1是否改变
        top1_changed = top5_orig.indices[0, 0].item() != top5_mod.indices[0, 0].item()

        # top-5 overlap
        orig_set = set(top5_orig.indices[0].tolist())
        mod_set = set(top5_mod.indices[0].tolist())
        top5_overlap = len(orig_set & mod_set) / 5.0

        # 因果消融: 零化属性词embedding
        attr_word_orig = attr_pair.split("→")[0]
        pos_list, _ = find_token_positions(tok, text_orig, attr_word_orig)
        if pos_list:
            target_pos = pos_list[-1]
            inp = tok(text_orig, return_tensors="pt").to(mdl.device)
            embeds = mdl.get_input_embeddings()(inp.input_ids).clone()
            embeds[0, target_pos, :] = 0.0
            with torch.no_grad():
                out_abl = mdl(inputs_embeds=embeds)
            probs_abl = F.softmax(out_abl.logits[:, -1, :].float(), dim=-1)
            kl_abl = compute_kl(probs_orig, probs_abl)
        else:
            kl_abl = 0.0

        log(f"    KL(orig,modified)={kl:.6f}, KL(orig,ablated)={kl_abl:.6f}")
        log(f"    Top1 changed: {top1_changed}, Top5 overlap: {top5_overlap:.2f}")
        log(f"    Top5 orig: {top5_words_orig[:3]}")
        log(f"    Top5 mod:  {top5_words_mod[:3]}")

        results.append({
            "attr_pair": attr_pair,
            "text_orig": text_orig,
            "text_mod": text_modified,
            "target_word": target_word,
            "KL_modified": round(kl, 6),
            "KL_ablated": round(kl_abl, 6),
            "top1_changed": top1_changed,
            "top5_overlap": round(top5_overlap, 2),
            "top5_orig": top5_words_orig[:3],
            "top5_mod": top5_words_mod[:3],
        })

    # 汇总
    all_kl_mod = [r["KL_modified"] for r in results]
    all_kl_abl = [r["KL_ablated"] for r in results]
    all_top1_chg = sum(1 for r in results if r["top1_changed"])
    all_top5_ol = [r["top5_overlap"] for r in results]

    log(f"\n  === P179 Summary ===")
    log(f"    KL(orig,modified): mean={np.mean(all_kl_mod):.6f}, max={np.max(all_kl_mod):.6f}")
    log(f"    KL(orig,ablated): mean={np.mean(all_kl_abl):.6f}, max={np.max(all_kl_abl):.6f}")
    log(f"    Top-1 changed: {all_top1_chg}/{len(results)} ({100*all_top1_chg/len(results):.0f}%)")
    log(f"    Top-5 overlap: mean={np.mean(all_top5_ol):.2f}")
    log(f"    Attribute causal effect ratio (mod/abl): {np.mean(all_kl_mod)/max(np.mean(all_kl_abl), 1e-10):.1f}x")

    del mdl
    torch.cuda.empty_cache()
    gc.collect()

    return results


# ============================================================
# P180: 大规模概念图谱
# ============================================================

def run_p180(model, tok, n_layers):
    """30+概念的全层cos矩阵"""
    log("\n" + "="*70)
    log("P180: 大规模概念图谱 — 30+概念的token级cos矩阵")
    log("="*70)

    templates = [
        "The {} is on the table.",
        "She saw a {} in the garden.",
        "A beautiful {} caught her attention.",
        "The {} was large and heavy.",
        "He picked up the {} carefully.",
        "The {} has several important properties.",
        "Many people like the {}.",
        "Scientists study the {} extensively.",
        "The {} can be found in nature.",
        "A small {} appeared in the distance.",
        "The {} is known for its unique characteristics.",
        "They described the {} in great detail.",
        "Children learn about the {} in school.",
        "I bought a new {} at the store.",
        "The {} is a common household item.",
        "Everyone needs a {} in daily life.",
        "A {} can be used for many purposes.",
        "The {} was discovered long ago.",
        "People often discuss the {}.",
        "The {} plays an important role in our lives.",
    ]

    # 30个概念，分5个语义类别
    concept_categories = {
        "自然物": ["apple", "sun", "water", "stone", "tree", "flower", "mountain", "river", "fire", "rain"],
        "动物": ["dog", "cat", "bird", "fish", "horse", "elephant", "lion", "rabbit"],
        "人造物": ["car", "house", "book", "knife", "phone", "chair", "table", "cup"],
        "抽象": ["love", "truth", "justice", "freedom", "beauty", "knowledge"],
    }

    all_concepts = []
    cat_labels = {}
    for cat, concepts in concept_categories.items():
        for c in concepts:
            all_concepts.append(c)
            cat_labels[c] = cat

    log(f"  Total concepts: {len(all_concepts)} across {len(concept_categories)} categories")

    # 提取每层每概念的h
    concept_hs = {}
    for concept in all_concepts:
        layer_hs_sum = [torch.zeros(model.config.hidden_size) for _ in range(n_layers)]
        valid_count = 0

        for tmpl in templates:
            text = tmpl.format(concept)
            pos_list, _ = find_token_positions(tok, text, concept)
            if not pos_list:
                continue
            target_pos = pos_list[-1]
            valid_count += 1

            inp = tok(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inp, output_hidden_states=True)
            hs = out.hidden_states

            for l_idx in range(n_layers):
                layer_idx = l_idx + 1
                if layer_idx < len(hs):
                    layer_hs_sum[l_idx] += hs[layer_idx][0, target_pos, :].float().cpu()

        if valid_count == 0:
            continue
        concept_hs[concept] = [h / valid_count for h in layer_hs_sum]

    valid_concepts = list(concept_hs.keys())
    log(f"  Valid concepts: {len(valid_concepts)}")

    # 计算最后层的cos矩阵
    log(f"\n  --- Last-layer cos matrix (selected) ---")
    cos_matrix_last = {}
    for i in range(len(valid_concepts)):
        for j in range(i+1, len(valid_concepts)):
            a, b = valid_concepts[i], valid_concepts[j]
            c = F.cosine_similarity(concept_hs[a][-1].unsqueeze(0), concept_hs[b][-1].unsqueeze(0)).item()
            cos_matrix_last[f"{a}_{b}"] = c

    # 按类别统计
    log(f"\n  --- Intra-category vs Inter-category cos ---")
    intra_cos = []
    inter_cos = []
    for i in range(len(valid_concepts)):
        for j in range(i+1, len(valid_concepts)):
            a, b = valid_concepts[i], valid_concepts[j]
            c = cos_matrix_last.get(f"{a}_{b}", cos_matrix_last.get(f"{b}_{a}", 0.5))
            if cat_labels.get(a) == cat_labels.get(b):
                intra_cos.append(c)
            else:
                inter_cos.append(c)

    log(f"    Intra-category cos: mean={np.mean(intra_cos):.4f}, std={np.std(intra_cos):.4f}, n={len(intra_cos)}")
    log(f"    Inter-category cos: mean={np.mean(inter_cos):.4f}, std={np.std(inter_cos):.4f}, n={len(inter_cos)}")
    log(f"    Separation: {np.mean(intra_cos):.4f} vs {np.mean(inter_cos):.4f} (lower=more separated)")

    # 每个类别的内部cos
    log(f"\n  --- Per-category internal cos ---")
    for cat, concepts in concept_categories.items():
        cat_cos = []
        for i in range(len(concepts)):
            for j in range(i+1, len(concepts)):
                a, b = concepts[i], concepts[j]
                if a in concept_hs and b in concept_hs:
                    c = F.cosine_similarity(concept_hs[a][-1].unsqueeze(0), concept_hs[b][-1].unsqueeze(0)).item()
                    cat_cos.append(c)
        if cat_cos:
            log(f"    {cat}: mean={np.mean(cat_cos):.4f}, std={np.std(cat_cos):.4f}, n={len(cat_cos)}")

    # 全层追踪: 每层的类别分离度
    log(f"\n  --- Per-layer category separation ---")
    layer_sep = []
    for l in range(n_layers):
        l_intra = []
        l_inter = []
        for i in range(len(valid_concepts)):
            for j in range(i+1, len(valid_concepts)):
                a, b = valid_concepts[i], valid_concepts[j]
                c = F.cosine_similarity(concept_hs[a][l].unsqueeze(0), concept_hs[b][l].unsqueeze(0)).item()
                if cat_labels.get(a) == cat_labels.get(b):
                    l_intra.append(c)
                else:
                    l_inter.append(c)
        if l_intra and l_inter:
            sep = np.mean(l_inter) - np.mean(l_intra)
            layer_sep.append({"intra": np.mean(l_intra), "inter": np.mean(l_inter), "separation": sep})
            if l % 5 == 0 or l == n_layers - 1:
                log(f"    Layer {l}: intra={np.mean(l_intra):.4f}, inter={np.mean(l_inter):.4f}, sep={sep:+.4f}")

    if layer_sep:
        best_sep_layer = np.argmax([s["separation"] for s in layer_sep])
        log(f"\n    Best category separation layer: L{best_sep_layer} (sep={layer_sep[best_sep_layer]['separation']:+.4f})")

    # 最正交/最平行的跨类别概念对
    cross_cat_cos = []
    for i in range(len(valid_concepts)):
        for j in range(i+1, len(valid_concepts)):
            a, b = valid_concepts[i], valid_concepts[j]
            if cat_labels.get(a) != cat_labels.get(b):
                c = cos_matrix_last.get(f"{a}_{b}", cos_matrix_last.get(f"{b}_{a}", 0.5))
                cross_cat_cos.append((a, b, cat_labels[a], cat_labels[b], c))
    cross_cat_cos.sort(key=lambda x: x[4])

    log(f"\n    Most orthogonal cross-category pairs:")
    for a, b, ca, cb, c in cross_cat_cos[:5]:
        log(f"      {a}({ca}) vs {b}({cb}): cos={c:.4f}")
    log(f"    Most parallel cross-category pairs:")
    for a, b, ca, cb, c in cross_cat_cos[-5:]:
        log(f"      {a}({ca}) vs {b}({cb}): cos={c:.4f}")

    return {
        "n_concepts": len(valid_concepts),
        "n_categories": len(concept_categories),
        "intra_cat_cos": {"mean": round(np.mean(intra_cos), 4), "std": round(np.std(intra_cos), 4)},
        "inter_cat_cos": {"mean": round(np.mean(inter_cos), 4), "std": round(np.std(inter_cos), 4)},
        "layer_separation": [{"layer": i, **s} for i, s in enumerate(layer_sep)],
    }


# ============================================================
# 主函数
# ============================================================

def main():
    global log

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()

    model_name = args.model
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = f"d:/develop/TransformerLens-main/tests/glm5_temp/stage732_phase27_{model_name}_{ts}"
    log = Logger(log_dir, f"phase27_{model_name}")

    log(f"{'='*70}")
    log(f"Phase XXVII: 逻辑推理追踪+全层EMB→HS变换+float32因果验证")
    log(f"Model: {model_name}, Time: {ts}")
    log(f"Log dir: {log_dir}")
    log(f"{'='*70}")

    all_results = {}

    # P176: float32因果消融 (需要单独加载float32模型)
    try:
        t0 = time.time()
        all_results["P176"] = run_p176(model_name)
        log(f"\n[P176] Done in {time.time()-t0:.0f}s")
    except Exception as e:
        log(f"\n[P176] ERROR: {e}")
        import traceback; traceback.print_exc()

    # 释放显存
    torch.cuda.empty_cache()
    gc.collect()

    # P177-P180: 用bfloat16模型
    try:
        log(f"\n[load] Loading bfloat16 model for P177-P180...")
        mdl, tok = load_model(model_name)
        n_layers = len(mdl.model.layers)
    except Exception as e:
        log(f"[load] ERROR: {e}")
        return

    # P177: 全层EMB→HS变换追踪
    try:
        t0 = time.time()
        all_results["P177"] = run_p177(mdl, tok, n_layers)
        log(f"\n[P177] Done in {time.time()-t0:.0f}s")
    except Exception as e:
        log(f"\n[P177] ERROR: {e}")
        import traceback; traceback.print_exc()

    # P178: 逻辑推理逐层追踪
    try:
        t0 = time.time()
        all_results["P178"] = run_p178(mdl, tok, n_layers)
        log(f"\n[P178] Done in {time.time()-t0:.0f}s")
    except Exception as e:
        log(f"\n[P178] ERROR: {e}")
        import traceback; traceback.print_exc()

    # P180: 大规模概念图谱
    try:
        t0 = time.time()
        all_results["P180"] = run_p180(mdl, tok, n_layers)
        log(f"\n[P180] Done in {time.time()-t0:.0f}s")
    except Exception as e:
        log(f"\n[P180] ERROR: {e}")
        import traceback; traceback.print_exc()

    del mdl
    torch.cuda.empty_cache()
    gc.collect()

    # P179: 属性因果验证 (需要float32)
    try:
        t0 = time.time()
        all_results["P179"] = run_p179(model_name)
        log(f"\n[P179] Done in {time.time()-t0:.0f}s")
    except Exception as e:
        log(f"\n[P179] ERROR: {e}")
        import traceback; traceback.print_exc()

    # 保存结果
    results_path = os.path.join(log_dir, f"results_phase27_{model_name}.json")
    # 将numpy和torch类型转换为可序列化类型
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(i) for i in obj]
        return obj

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(convert(all_results), f, ensure_ascii=False, indent=2)
    log(f"\n[save] Results saved to {results_path}")

    log.close()
    print(f"\nDone! Log dir: {log_dir}")


if __name__ == "__main__":
    main()
