#!/usr/bin/env python3
"""
Stage 730: Phase XXV — 概念探测与属性编码(微观编码分析)
================================================================================
Phase I~XXIV完成了宏观几何: 曲率(PR=3~19), 弯曲路径(84°), 全息编码。
Phase XXV转向微观编码: 概念/属性/抽象层级在h中的编码方式。

  P166: 概念探测 — 同一概念不同描述, 找共享方向(概念方向)
  P167: 属性编码 — 同一概念不同属性, 找属性对应的h变化方向
  P168: 抽象层级 — "苹果→水果→食物→物体"的cos/距离层级
  P169: 词嵌入类比 — king+queen-man≈woman在hidden state中是否成立
  P170: 概念方向正交性 — 不同概念的方向是否正交/平行

测试规模: 150+文本×全层, 3模型
用法: python stage730_phase25.py --model qwen3
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

# ============================================================
# 文本集设计
# ============================================================

def build_concept_texts():
    """概念探测文本: 同一概念多种描述"""
    groups = {
        "apple": [
            "A red apple is on the table.",
            "She ate a sweet green apple.",
            "Apple trees grow in temperate climates.",
            "Apple juice is a popular beverage.",
            "The apple fell from the tree.",
            "He bought organic apples at the market.",
            "Apple pie is a classic American dessert.",
            "An apple a day keeps the doctor away.",
            "The apple has a round shape and crisp texture.",
        ],
        "sun": [
            "The sun rises in the east every morning.",
            "Sunlight provides energy for photosynthesis.",
            "The sun is a massive star at the center of our solar system.",
            "Sunshine warms the earth during the day.",
            "Solar panels convert sunlight into electricity.",
            "The sunset painted the sky with golden hues.",
            "Sun protection is important for skin health.",
            "The sun emits light and heat.",
            "Ancient civilizations worshipped the sun as a god.",
        ],
        "water": [
            "Water covers about 71% of the earth surface.",
            "She drank a glass of cold water.",
            "The river flows with crystal clear water.",
            "Water boils at 100 degrees Celsius.",
            "Clean drinking water is essential for human survival.",
            "The ocean water is salty and deep.",
            "Water freezes into ice at zero degrees.",
            "Plants absorb water through their roots.",
            "Water cycle includes evaporation and precipitation.",
        ],
        "stone": [
            "The stone was heavy and rough.",
            "They built the wall with large stones.",
            "A precious stone sparkled in the ring.",
            "Stone tools were used by early humans.",
            "The child threw a stone into the lake.",
            "The philosopher Socrates sat on a stone.",
            "Mountain stones are formed over millions of years.",
            "A stepping stone crossed the stream.",
            "The castle was made of gray stone.",
        ],
        "hair": [
            "She has long flowing black hair.",
            "He needs a haircut next week.",
            "Hair color can be changed with dye.",
            "The hairbrush smoothed her tangled hair.",
            "Hair grows about half an inch per month.",
            "Gray hair is a sign of aging.",
            "Curly hair requires special conditioning.",
            "The hair salon offers various styles.",
            "Animal hair is often called fur.",
        ],
    }
    return groups

def build_attribute_texts():
    """属性编码文本: 同一概念不同属性"""
    groups = {
        "apple_color": {
            "red": [
                "The apple is bright red.",
                "A red apple looked delicious.",
                "She picked the reddest apple from the tree.",
                "Red apple skin shines in the sunlight.",
            ],
            "green": [
                "The apple is green and tart.",
                "A green apple is not yet ripe.",
                "Granny Smith apples are famously green.",
                "The green apple has a sour taste.",
            ],
            "yellow": [
                "The apple is golden yellow.",
                "Yellow apples are sweet and juicy.",
                "A yellow apple fell from the branch.",
                "Golden Delicious apples are yellow.",
            ],
        },
        "apple_taste": {
            "sweet": [
                "The apple tastes very sweet.",
                "A sweet apple is perfect for dessert.",
                "Honeycrisp apples are incredibly sweet.",
                "She loves the sweet flavor of ripe apples.",
            ],
            "sour": [
                "The apple has a sour taste.",
                "A sour apple made her pucker.",
                "Lemon apples are extremely sour.",
                "The sour green apple was not pleasant to eat.",
            ],
            "crisp": [
                "The apple is crisp and fresh.",
                "A crisp apple makes a satisfying crunch.",
                "The crisp texture of the apple was delightful.",
                "He bit into the crisp red apple.",
            ],
        },
        "water_temp": {
            "hot": [
                "The hot water steamed in the cup.",
                "She poured hot water for tea.",
                "Hot water can burn your skin.",
                "The hot water bottle kept him warm.",
            ],
            "cold": [
                "The cold water was refreshing.",
                "She added ice to make the water cold.",
                "Cold water flows from the mountain spring.",
                "He drank cold water after exercise.",
            ],
        },
    }
    return groups

def build_abstraction_texts():
    """抽象层级文本: 苹果→水果→食物→物体"""
    groups = {
        "apple": [
            "An apple is a round fruit that grows on trees.",
            "The apple is crisp and sweet.",
            "Apple cultivation originated in Central Asia.",
            "The apple belongs to the Rosaceae family.",
            "Apple seeds contain small amounts of cyanide.",
        ],
        "fruit": [
            "Fruits are the seed-bearing structures of flowering plants.",
            "A healthy diet includes plenty of fresh fruit.",
            "Tropical fruits include mangoes and pineapples.",
            "Fruit consumption is linked to reduced disease risk.",
            "The fruit basket contained various seasonal fruits.",
        ],
        "food": [
            "Food provides essential nutrients for the body.",
            "The food industry is a major economic sector.",
            "Organic food is grown without synthetic pesticides.",
            "Food security is a global challenge.",
            "Traditional food varies greatly across cultures.",
        ],
        "object": [
            "An object is any material thing that can be perceived.",
            "Physical objects have mass and occupy space.",
            "The object fell to the ground due to gravity.",
            "Everyday objects surround us in our environment.",
            "The object had a smooth metallic surface.",
        ],
        "entity": [
            "An entity is something that exists independently.",
            "Legal entities have rights and obligations.",
            "The entity was difficult to classify.",
            "Abstract entities include concepts and numbers.",
            "The philosophical entity debate continues today.",
        ],
    }
    return groups

def build_analogy_texts():
    """词嵌入类比验证文本"""
    pairs = {
        "king": [
            "The king ruled the kingdom with wisdom.",
            "King Arthur led the Knights of the Round Table.",
            "The king wore a golden crown.",
        ],
        "queen": [
            "The queen ruled the kingdom alongside the king.",
            "Queen Victoria reigned over the British Empire.",
            "The queen wore an elegant dress.",
        ],
        "man": [
            "The man walked through the park.",
            "A tall man stood at the entrance.",
            "The old man sat on the bench.",
        ],
        "woman": [
            "The woman walked through the park.",
            "A tall woman stood at the entrance.",
            "The old woman sat on the bench.",
        ],
        "boy": [
            "The boy played in the playground.",
            "A young boy ran across the field.",
            "The boy was eager to learn.",
        ],
        "girl": [
            "The girl played in the playground.",
            "A young girl ran across the field.",
            "The girl was eager to learn.",
        ],
        "prince": [
            "The prince was next in line for the throne.",
            "Prince charming rescued the princess.",
            "The young prince studied diplomacy.",
        ],
        "princess": [
            "The princess lived in a beautiful castle.",
            "The princess was known for her kindness.",
            "Princess Diana was beloved worldwide.",
        ],
        "doctor": [
            "The doctor examined the patient carefully.",
            "A doctor must study medicine for many years.",
            "The doctor prescribed medication.",
        ],
        "nurse": [
            "The nurse cared for the patient.",
            "A nurse assists the doctor in treatment.",
            "The nurse recorded the vital signs.",
        ],
    }
    return pairs

# ============================================================
# 模型加载
# ============================================================

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

def get_hidden(model, tok, text, layers=None):
    """提取所有层的hidden states (最后一个token)"""
    inp = tok(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inp, output_hidden_states=True)
    hs = out.hidden_states  # tuple of (1, seq, d)
    # 提取每个transformer layer的输出 (索引1到L)
    n_trans = len(model.model.layers)
    layer_hs = []
    for i in range(1, n_trans + 1):
        if i < len(hs):
            layer_hs.append(hs[i][0, -1, :].float().cpu())
    return layer_hs

# ============================================================
# P166: 概念探测
# ============================================================

def run_p166(model, tok, concept_groups, n_layers):
    """同一概念不同描述 → 找共享方向(概念方向)"""
    log("\n" + "="*70)
    log("P166: 概念探测 — 找共享方向(概念方向)")
    log("="*70)

    results = {}
    for concept, texts in concept_groups.items():
        log(f"\n--- Concept: {concept} ({len(texts)} texts) ---")
        # 收集每个文本的h
        all_h = []
        for t in texts:
            hs = get_hidden(model, tok, t)
            all_h.append(hs)

        # 逐层分析
        layer_results = []
        for l in range(n_layers):
            h_matrix = torch.stack([all_h[i][l] for i in range(len(texts))])  # (N, d)
            h_mean = h_matrix.mean(dim=0)

            # 去均值后的协方差
            h_centered = h_matrix - h_mean
            cov = (h_centered.T @ h_centered) / len(texts)
            eigenvalues, eigenvectors = torch.linalg.eigh(cov)
            eigenvalues = eigenvalues.flip(0)
            eigenvectors = eigenvectors.flip(1)

            # Top-1方向的解释方差比
            total_var = eigenvalues.sum()
            top1_var = eigenvalues[0]
            top1_ratio = (top1_var / total_var).item()

            # Top-3累计
            top3_ratio = (eigenvalues[:3].sum() / total_var).item()

            # 每个文本到均值的cos相似度
            cos_to_mean = F.cosine_similarity(h_matrix, h_mean.unsqueeze(0)).tolist()

            # 参与比
            ev = eigenvalues[eigenvalues > 1e-10]
            if len(ev) > 0:
                pr = (ev.sum()**2) / (ev**2).sum()
            else:
                pr = 0

            layer_results.append({
                "top1_ratio": top1_ratio,
                "top3_ratio": top3_ratio,
                "cos_mean": np.mean(cos_to_mean),
                "cos_std": np.std(cos_to_mean),
                "cos_min": np.min(cos_to_mean),
                "pr": pr.item(),
                "h_mean_norm": h_mean.norm().item(),
            })

        results[concept] = layer_results
        # 汇总
        avg_top1 = np.mean([r["top1_ratio"] for r in layer_results])
        avg_cos = np.mean([r["cos_mean"] for r in layer_results])
        avg_pr = np.mean([r["pr"] for r in layer_results])
        log(f"  avg Top-1 ratio: {avg_top1:.4f}, avg cos_to_mean: {avg_cos:.4f}, avg PR: {avg_pr:.1f}")

        # 层间模式(early/mid/late)
        n = n_layers
        early = np.mean([r["cos_mean"] for r in layer_results[:n//3]])
        mid = np.mean([r["cos_mean"] for r in layer_results[n//3:2*n//3]])
        late = np.mean([r["cos_mean"] for r in layer_results[2*n//3:]])
        log(f"  Early/Mid/Late cos_to_mean: {early:.4f} / {mid:.4f} / {late:.4f}")

    return results

# ============================================================
# P167: 属性编码
# ============================================================

def run_p167(model, tok, attr_groups, n_layers):
    """同一概念不同属性 → 找属性对应的h变化方向"""
    log("\n" + "="*70)
    log("P167: 属性编码 — 属性是独立维度还是概念向量的分量?")
    log("="*70)

    results = {}
    for group_name, attrs in attr_groups.items():
        log(f"\n--- Group: {group_name} ---")
        group_results = {}
        for attr_name, texts in attrs.items():
            log(f"  Attribute: {attr_name} ({len(texts)} texts)")
            all_h = []
            for t in texts:
                hs = get_hidden(model, tok, t)
                all_h.append(hs)

            layer_results = []
            for l in range(n_layers):
                h_matrix = torch.stack([all_h[i][l] for i in range(len(texts))])
                h_mean = h_matrix.mean(dim=0)
                cos_to_mean = F.cosine_similarity(h_matrix, h_mean.unsqueeze(0)).tolist()
                layer_results.append({
                    "cos_mean": np.mean(cos_to_mean),
                    "cos_std": np.std(cos_to_mean),
                    "h_mean": h_mean,
                    "h_mean_norm": h_mean.norm().item(),
                })
            group_results[attr_name] = layer_results
            avg_cos = np.mean([r["cos_mean"] for r in layer_results])
            log(f"    avg cos_to_mean: {avg_cos:.4f}")

        # 属性间方向比较
        attr_names = list(attrs.keys())
        layer_attr_cos = []
        for l in range(n_layers):
            cos_matrix = {}
            for i in range(len(attr_names)):
                for j in range(i+1, len(attr_names)):
                    a, b = attr_names[i], attr_names[j]
                    h_a = group_results[a][l]["h_mean"]
                    h_b = group_results[b][l]["h_mean"]
                    c = F.cosine_similarity(h_a.unsqueeze(0), h_b.unsqueeze(0)).item()
                    cos_matrix[f"{a}_vs_{b}"] = c
            layer_attr_cos.append(cos_matrix)

        results[group_name] = {
            "attributes": group_results,
            "inter_attr_cos": layer_attr_cos,
        }

        # 汇总属性间cos
        for l_idx in [n_layers//4, n_layers//2, 3*n_layers//4]:
            if l_idx < len(layer_attr_cos):
                cos_vals = list(layer_attr_cos[l_idx].values())
                log(f"  L{l_idx} inter-attr cos: mean={np.mean(cos_vals):.4f}, "
                    f"min={np.min(cos_vals):.4f}, max={np.max(cos_vals):.4f}")

    return results

# ============================================================
# P168: 抽象层级
# ============================================================

def run_p168(model, tok, abstraction_groups, n_layers):
    """抽象层级: "苹果→水果→食物→物体→实体"的cos/距离"""
    log("\n" + "="*70)
    log("P168: 抽象层级 — 概念抽象度的几何关系")
    log("="*70)

    # 收集每个抽象层级的h_mean
    level_means = {}
    level_names = list(abstraction_groups.keys())
    for level_name, texts in abstraction_groups.items():
        log(f"\n  Level: {level_name} ({len(texts)} texts)")
        all_h = []
        for t in texts:
            hs = get_hidden(model, tok, t)
            all_h.append(hs)

        layer_means = []
        layer_inters = []
        for l in range(n_layers):
            h_matrix = torch.stack([all_h[i][l] for i in range(len(texts))])
            h_mean = h_matrix.mean(dim=0)
            layer_means.append(h_mean)

            # 层内一致性
            cos_vals = F.cosine_similarity(h_matrix, h_mean.unsqueeze(0)).tolist()
            layer_inters.append(np.mean(cos_vals))

        level_means[level_name] = {
            "means": layer_means,
            "intra_cos": layer_inters,
        }
        avg_cos = np.mean(layer_inters)
        log(f"    avg intra-cos: {avg_cos:.4f}")

    # 层级间cos矩阵(每个层)
    results = {"intra_cos": {}, "inter_cos_matrix": {}}
    for ln in level_names:
        results["intra_cos"][ln] = level_means[ln]["intra_cos"]

    # 选几个代表性层
    for l_idx in [0, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]:
        cos_mat = {}
        for i in range(len(level_names)):
            for j in range(i+1, len(level_names)):
                a, b = level_names[i], level_names[j]
                h_a = level_means[a]["means"][l_idx]
                h_b = level_means[b]["means"][l_idx]
                c = F.cosine_similarity(h_a.unsqueeze(0), h_b.unsqueeze(0)).item()
                cos_mat[f"{a}_vs_{b}"] = c
        results["inter_cos_matrix"][f"L{l_idx}"] = cos_mat

    log("\n  --- Inter-level cos (selected layers) ---")
    for l_key, cos_mat in results["inter_cos_matrix"].items():
        cos_vals = list(cos_mat.values())
        log(f"  {l_key}: mean={np.mean(cos_vals):.4f}, std={np.std(cos_vals):.4f}, "
            f"min={np.min(cos_vals):.4f}, max={np.max(cos_vals):.4f}")
        # 特别关注相邻层级的cos
        adjacent_pairs = [k for k in cos_mat if "_vs_" in k and
                         any(level_names[i] in k and level_names[i+1] in k
                             for i in range(len(level_names)-1))]
        if adjacent_pairs:
            adj_cos = [cos_mat[k] for k in adjacent_pairs]
            log(f"    adjacent pairs cos: mean={np.mean(adj_cos):.4f}")

    return results

# ============================================================
# P169: 词嵌入类比验证
# ============================================================

def run_p169(model, tok, analogy_pairs, n_layers):
    """king+queen-man≈woman 在hidden state中是否成立"""
    log("\n" + "="*70)
    log("P169: 词嵌入类比验证 — hidden state中的代数结构")
    log("="*70)

    # 收集每个词的h_mean
    word_means = {}
    for word, texts in analogy_pairs.items():
        all_h = []
        for t in texts:
            hs = get_hidden(model, tok, t)
            all_h.append(hs)
        layer_means = []
        for l in range(n_layers):
            h_matrix = torch.stack([all_h[i][l] for i in range(len(texts))])
            layer_means.append(h_mean := h_matrix.mean(dim=0))
        word_means[word] = layer_means
        log(f"  {word}: {len(texts)} texts processed")

    # 类比测试
    analogies = [
        ("king", "queen", "man", "woman"),
        ("boy", "girl", "man", "woman"),
        ("prince", "princess", "king", "queen"),
        ("doctor", "nurse", "man", "woman"),
    ]

    results = {}
    for a1, a2, b1, b2 in analogies:
        log(f"\n  Analogy: {a1}+{a2}-{b1} ≈ {b2}?")
        layer_accuracies = []
        layer_cos = []
        for l in range(n_layers):
            h_a1 = word_means[a1][l]
            h_a2 = word_means[a2][l]
            h_b1 = word_means[b1][l]
            h_b2 = word_means[b2][l]

            # king + queen - man = ?
            predicted = h_a1 + h_a2 - h_b1
            # 与woman的cos
            cos_predicted = F.cosine_similarity(predicted.unsqueeze(0), h_b2.unsqueeze(0)).item()

            # 也计算与所有词的cos, 看woman是否排第一
            all_cos = {}
            for w in word_means:
                c = F.cosine_similarity(predicted.unsqueeze(0), word_means[w][l].unsqueeze(0)).item()
                all_cos[w] = c
            sorted_words = sorted(all_cos.items(), key=lambda x: -x[1])
            rank = [i for i, (w, _) in enumerate(sorted_words) if w == b2][0] + 1

            layer_cos.append(cos_predicted)
            layer_accuracies.append(rank == 1)

        results[f"{a1}+{a2}-{b1}={b2}"] = {
            "layer_cos": layer_cos,
            "accuracy": sum(layer_accuracies) / len(layer_accuracies),
            "best_layer_cos": max(layer_cos),
            "best_layer_idx": layer_cos.index(max(layer_cos)),
        }
        log(f"    max cos={max(layer_cos):.4f} at L{layer_cos.index(max(layer_cos))}, "
            f"accuracy={sum(layer_accuracies)}/{len(layer_accuracies)}")

    return results

# ============================================================
# P170: 概念方向正交性
# ============================================================

def run_p170(model, tok, concept_groups, n_layers):
    """不同概念("苹果"vs"太阳")的方向是否正交"""
    log("\n" + "="*70)
    log("P170: 概念方向正交性 — 不同概念的方向关系")
    log("="*70)

    # 收集每个概念的h_mean
    concept_means = {}
    concept_names = list(concept_groups.keys())
    for concept, texts in concept_groups.items():
        all_h = []
        for t in texts:
            hs = get_hidden(model, tok, t)
            all_h.append(hs)
        layer_means = []
        for l in range(n_layers):
            h_matrix = torch.stack([all_h[i][l] for i in range(len(texts))])
            layer_means.append(h_mean := h_matrix.mean(dim=0))
        concept_means[concept] = layer_means

    # 逐层计算概念间cos
    results = {}
    for l in range(n_layers):
        cos_vals = []
        cos_pairs = {}
        for i in range(len(concept_names)):
            for j in range(i+1, len(concept_names)):
                a, b = concept_names[i], concept_names[j]
                c = F.cosine_similarity(
                    concept_means[a][l].unsqueeze(0),
                    concept_means[b][l].unsqueeze(0)
                ).item()
                cos_pairs[f"{a}_vs_{b}"] = c
                cos_vals.append(c)
        results[f"L{l}"] = cos_pairs

    # 汇总
    all_cos = []
    for l_key, cos_pairs in results.items():
        vals = list(cos_pairs.values())
        all_cos.extend(vals)

    log(f"\n  Overall: mean cos={np.mean(all_cos):.4f}, std={np.std(all_cos):.4f}, "
        f"min={np.min(all_cos):.4f}, max={np.max(all_cos):.4f}")

    # 逐层汇总
    layer_stats = []
    for l in range(n_layers):
        vals = list(results[f"L{l}"].values())
        layer_stats.append({
            "mean": np.mean(vals),
            "std": np.std(vals),
            "min": np.min(vals),
            "max": np.max(vals),
        })

    log(f"\n  Layer pattern:")
    n = n_layers
    early = np.mean([s["mean"] for s in layer_stats[:n//3]])
    mid = np.mean([s["mean"] for s in layer_stats[n//3:2*n//3]])
    late = np.mean([s["mean"] for s in layer_stats[2*n//3:]])
    log(f"    Early/Mid/Late: {early:.4f} / {mid:.4f} / {late:.4f}")

    # 最正交/最平行的概念对
    all_pair_cos = {}
    for l_key, cos_pairs in results.items():
        for pair, c in cos_pairs.items():
            if pair not in all_pair_cos:
                all_pair_cos[pair] = []
            all_pair_cos[pair].append(c)

    log(f"\n  Pair-level averages:")
    sorted_pairs = sorted(all_pair_cos.items(), key=lambda x: np.mean(x[1]))
    for pair, cos_list in sorted_pairs[:3]:  # 最正交
        log(f"    Most orthogonal: {pair}: mean cos={np.mean(cos_list):.4f}")
    for pair, cos_list in sorted_pairs[-3:]:  # 最平行
        log(f"    Most parallel: {pair}: mean cos={np.mean(cos_list):.4f}")

    return {"layer_stats": layer_stats, "pair_avg": {p: np.mean(c) for p, c in all_pair_cos.items()}}

# ============================================================
# 主程序
# ============================================================

def main():
    global log
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, choices=["qwen3", "deepseek7b", "glm4"])
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M")
    log_dir = f"tests/glm5_temp/stage730_phase25_{args.model}_{ts}"
    log = Logger(log_dir, "results")
    log(f"Phase XXV: Concept Probing & Attribute Encoding")
    log(f"Model: {args.model}, Time: {ts}")

    mdl, tok = load_model(args.model)
    n_layers = len(mdl.model.layers)
    d_model = mdl.config.hidden_size

    # 构建文本集
    concept_groups = build_concept_texts()
    attr_groups = build_attribute_texts()
    abstraction_groups = build_abstraction_texts()
    analogy_pairs = build_analogy_texts()

    # P166
    t0 = time.time()
    p166 = run_p166(mdl, tok, concept_groups, n_layers)
    log(f"\n[P166] Time: {time.time()-t0:.1f}s")

    # P167
    t0 = time.time()
    p167 = run_p167(mdl, tok, attr_groups, n_layers)
    log(f"\n[P167] Time: {time.time()-t0:.1f}s")

    # P168
    t0 = time.time()
    p168 = run_p168(mdl, tok, abstraction_groups, n_layers)
    log(f"\n[P168] Time: {time.time()-t0:.1f}s")

    # P169
    t0 = time.time()
    p169 = run_p169(mdl, tok, analogy_pairs, n_layers)
    log(f"\n[P169] Time: {time.time()-t0:.1f}s")

    # P170
    t0 = time.time()
    p170 = run_p170(mdl, tok, concept_groups, n_layers)
    log(f"\n[P170] Time: {time.time()-t0:.1f}s")

    # ============================================================
    # 最终汇总
    # ============================================================
    log("\n" + "="*70)
    log("FINAL SUMMARY")
    log("="*70)

    log("\n--- P166: Concept Probing ---")
    for concept in concept_groups:
        data = p166[concept]
        avg_cos = np.mean([r["cos_mean"] for r in data])
        avg_top1 = np.mean([r["top1_ratio"] for r in data])
        avg_pr = np.mean([r["pr"] for r in data])
        log(f"  {concept}: avg_cos={avg_cos:.4f}, avg_top1_ratio={avg_top1:.4f}, avg_PR={avg_pr:.1f}")

    log("\n--- P167: Attribute Encoding ---")
    for group_name, group_data in p167.items():
        log(f"  {group_name}:")
        inter_cos = group_data.get("inter_attr_cos", group_data.get("inter_cos", {}))
        if isinstance(inter_cos, list):
            for l_idx in [0, n_layers//2, n_layers-1]:
                if l_idx < len(inter_cos):
                    vals = list(inter_cos[l_idx].values())
                    log(f"    L{l_idx}: inter-attr cos mean={np.mean(vals):.4f}")
        else:
            for l_key in ["L0", f"L{n_layers//2}", f"L{n_layers-1}"]:
                if l_key in inter_cos:
                    vals = list(inter_cos[l_key].values())
                    log(f"    {l_key}: inter-attr cos mean={np.mean(vals):.4f}")

    log("\n--- P168: Abstraction Hierarchy ---")
    for l_key, cos_mat in p168["inter_cos_matrix"].items():
        vals = list(cos_mat.values())
        log(f"  {l_key}: inter-level cos mean={np.mean(vals):.4f}")
    # 抽象度趋势: apple-fruit vs apple-entity
    log("  Abstraction distance trend:")
    for l_key in list(p168["inter_cos_matrix"].keys()):
        cm = p168["inter_cos_matrix"][l_key]
        apple_fruit = cm.get("apple_vs_fruit", 0)
        apple_food = cm.get("apple_vs_food", 0)
        apple_object = cm.get("apple_vs_object", 0)
        apple_entity = cm.get("apple_vs_entity", 0)
        log(f"    {l_key}: apple-fruit={apple_fruit:.4f}, apple-food={apple_food:.4f}, "
            f"apple-object={apple_object:.4f}, apple-entity={apple_entity:.4f}")

    log("\n--- P169: Analogy Verification ---")
    for analogy_key, data in p169.items():
        log(f"  {analogy_key}: best_cos={data['best_layer_cos']:.4f}(L{data['best_layer_idx']}), "
            f"accuracy={data['accuracy']:.2f}")

    log("\n--- P170: Concept Orthogonality ---")
    all_means = [s["mean"] for s in p170["layer_stats"]]
    log(f"  Overall inter-concept cos: mean={np.mean(all_means):.4f}, "
        f"std={np.std(all_means):.4f}")
    log(f"  Most orthogonal pair: {min(p170['pair_avg'].items(), key=lambda x: x[1])}")
    log(f"  Most parallel pair: {max(p170['pair_avg'].items(), key=lambda x: x[1])}")

    del mdl, tok
    gc.collect()
    torch.cuda.empty_cache()
    log.close()
    print(f"\nResults saved to: {log_dir}/results.log")

if __name__ == "__main__":
    main()
