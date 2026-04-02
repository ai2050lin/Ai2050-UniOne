# -*- coding: utf-8 -*-
"""
Stage461: 单层偏置 + 逐层SVD + 100维高维分析 — 双模型
======================================================

Stage460瓶颈：6层×hidden_dim拼接(24576维)导致SVD效率低
Stage461改进：
  1. 逐层独立分析（每层单独做SVD）
  2. 单层偏置矩阵维度=hidden_dim(3584/4096)而非24576
  3. 找到信息密度最高的"黄金层"
  4. 对黄金层做100维SVD
  5. 双模型交叉验证

核心假设：
  - 单层SVD方差解释 > 6层拼接SVD（因为维度更低，噪声更少）
  - 存在一个"黄金层"集中编码概念语义
  - 100维SVD在单层上能达到50%+方差

模型: Qwen3-4B + DeepSeek-7B
"""

from __future__ import annotations

import gc
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qwen3_language_shared import (
    PROJECT_ROOT,
    capture_qwen_mlp_payloads,
    discover_layers,
    move_batch_to_model_device,
    remove_hooks,
    QWEN3_MODEL_PATH,
)

DEEPSEEK7B_MODEL_PATH = Path(
    r"D:\develop\model\hub\models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B\snapshots\916b56a44061fd5cd7d6a8fb632557ed4f724f60"
)

TIMESTAMP = "20260401"
OUTPUT_DIR = PROJECT_ROOT / "tests" / "codex_temp" / f"stage461_single_layer_svd_{TIMESTAMP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 复用Stage460的概念集（减少概念数到300提高速度）
CONCEPTS = {}
_all_words = set()
for cat_name, cat_data in {
    "fruit": ["apple", "banana", "orange", "grape", "mango", "peach", "lemon",
              "cherry", "berry", "melon", "kiwi", "plum", "pear", "fig", "lime",
              "coconut", "pineapple", "strawberry", "watermelon", "blueberry"],
    "animal": ["dog", "cat", "bird", "fish", "horse", "lion", "tiger", "elephant",
               "whale", "shark", "snake", "eagle", "wolf", "bear", "monkey",
               "rabbit", "deer", "fox", "owl", "dolphin",
               "penguin", "parrot", "frog", "turtle", "crocodile",
               "giraffe", "zebra", "gorilla", "kangaroo", "panda"],
    "vehicle": ["car", "bus", "train", "plane", "ship", "bicycle", "motorcycle",
                "truck", "helicopter", "rocket", "boat", "submarine", "tractor",
                "van", "taxi", "ambulance", "firetruck", "scooter", "tram", "ferry"],
    "profession": ["doctor", "nurse", "teacher", "engineer", "lawyer", "chef", "artist",
                   "musician", "writer", "painter", "scientist", "programmer", "pilot",
                   "soldier", "firefighter", "police", "farmer", "baker", "butcher", "driver",
                   "architect", "carpenter", "plumber", "electrician", "mechanic",
                   "surgeon", "dentist", "pharmacist", "veterinarian", "therapist"],
    "clothing": ["shirt", "pants", "dress", "jacket", "shoes", "hat", "socks",
                 "gloves", "scarf", "tie", "belt", "coat", "sweater", "boots",
                 "sandals", "uniform", "jeans", "shorts", "skirt", "blazer"],
    "furniture": ["chair", "table", "bed", "sofa", "desk", "bookcase", "cabinet",
                  "wardrobe", "dresser", "shelf", "stool", "bench", "lamp", "rug",
                  "curtain", "mirror", "couch", "armchair", "ottoman", "crib"],
    "food": ["bread", "rice", "cake", "cookie", "pizza", "pasta", "soup",
             "salad", "sandwich", "steak", "chicken", "egg", "cheese", "butter",
             "milk", "yogurt", "ice_cream", "chocolate", "candy", "waffle"],
    "color": ["red", "blue", "green", "yellow", "orange", "purple", "pink",
              "brown", "black", "white", "gray", "navy", "beige", "ivory",
              "coral", "crimson", "maroon", "olive", "amber", "lavender"],
    "emotion": ["happy", "sad", "angry", "fear", "love", "hate", "joy", "sorrow",
                "pride", "shame", "guilt", "envy", "jealousy", "hope", "despair",
                "anxiety", "calm", "excitement", "boredom", "loneliness"],
    "natural": ["mountain", "river", "ocean", "forest", "desert", "island",
                "valley", "cave", "volcano", "waterfall", "lake", "glacier",
                "meadow", "cliff", "swamp", "prairie", "jungle", "reef",
                "canyon", "plateau"],
}.items():
    words = [w for w in cat_data if w not in _all_words]
    _all_words.update(words)
    if words:
        CONCEPTS[cat_name] = {"label": cat_name, "words": words}


def sanitize_for_json(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    return obj


def load_model(model_path: Path):
    import os
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    want_cuda = torch.cuda.is_available()
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path), local_files_only=True, trust_remote_code=True, use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "pretrained_model_name_or_path": str(model_path),
        "local_files_only": True, "trust_remote_code": True,
        "low_cpu_mem_usage": True, "torch_dtype": torch.bfloat16,
    }
    if want_cuda:
        load_kwargs["device_map"] = "auto"
    else:
        load_kwargs["device_map"] = "cpu"
        load_kwargs["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(**load_kwargs)
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation("eager")
    model.eval()

    layer_count = len(discover_layers(model))
    neuron_dim = discover_layers(model)[0].mlp.gate_proj.out_features
    print(f"  Layers: {layer_count}, NeuronDim: {neuron_dim}")
    return model, tokenizer, layer_count, neuron_dim


def extract_word_per_layer(model, tokenizer, word, layer_count):
    """提取单个词在每层的MLP neuron_in激活"""
    prompt = f"The {word}"
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=32)
    token_ids = encoded["input_ids"][0].tolist()

    word_tokens = tokenizer.encode(word, add_special_tokens=False)
    target_pos = None
    for i, tid in enumerate(token_ids):
        if tid in word_tokens:
            target_pos = i
            break
    if target_pos is None:
        target_pos = -1

    layer_payload_map = {i: "neuron_in" for i in range(layer_count)}
    buffers, handles = capture_qwen_mlp_payloads(model, layer_payload_map)

    try:
        encoded = move_batch_to_model_device(model, encoded)
        with torch.no_grad():
            model(**encoded)

        per_layer = {}
        for li in range(layer_count):
            buf = buffers[li]
            if buf is not None:
                pos = target_pos if target_pos >= 0 else buf.shape[1] + target_pos
                pos = max(0, min(pos, buf.shape[1] - 1))
                per_layer[li] = buf[0, pos].float().numpy()
        return per_layer
    finally:
        remove_hooks(handles)


def extract_all_activations(model, tokenizer, concepts, layer_count):
    """提取所有概念在每层的激活"""
    all_activations = {}
    total = sum(len(c["words"]) for c in concepts.values())
    done = 0

    for cat_name, cat_data in concepts.items():
        all_activations[cat_name] = {}
        for word in cat_data["words"]:
            try:
                acts = extract_word_per_layer(model, tokenizer, word, layer_count)
                if acts:
                    all_activations[cat_name][word] = acts
                    done += 1
            except Exception:
                pass

    print(f"  Total: {done}/{total}")
    return all_activations


def per_layer_bias_analysis(all_activations, concepts, layer_count):
    """逐层独立分析：每层单独构建偏置矩阵+计算SVD"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD

    layer_results = {}

    for li in range(layer_count):
        # 收集该层所有概念的激活
        word_vecs = []
        word_labels = []
        word_cats = []

        for cat_name, words_acts in all_activations.items():
            if cat_name not in concepts:
                continue
            # 计算类别基底
            cat_vecs = []
            for word, acts in words_acts.items():
                if li in acts:
                    cat_vecs.append(acts[li])
            if not cat_vecs:
                continue
            basis = np.mean(cat_vecs, axis=0)

            # 计算偏置
            for word, acts in words_acts.items():
                if li in acts:
                    bias = acts[li] - basis
                    word_vecs.append(bias)
                    word_labels.append(word)
                    word_cats.append(cat_name)

        if len(word_vecs) < 20:
            continue

        bias_matrix = np.array(word_vecs)
        n_concepts, dim = bias_matrix.shape

        # 标准化
        scaler = StandardScaler()
        normed = scaler.fit_transform(bias_matrix)

        # SVD（最多100维）
        n_comp = min(100, min(normed.shape) - 1)
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        comp = svd.fit_transform(normed)

        var_exp = svd.explained_variance_ratio_
        cum_var = np.cumsum(var_exp)

        # 因子-类别eta²
        cat_to_idx = defaultdict(list)
        for i, cat in enumerate(word_cats):
            cat_to_idx[cat].append(i)

        factor_eta = []
        for fi in range(min(20, n_comp)):
            scores = comp[:, fi]
            grand_mean = np.mean(scores)
            ss_between = sum(len(idxs) * (np.mean(scores[idxs]) - grand_mean)**2
                           for idxs in cat_to_idx.values())
            ss_total = np.sum((scores - grand_mean)**2)
            eta_sq = ss_between / (ss_total + 1e-10)
            factor_eta.append(float(eta_sq))

        # 里程碑
        milestones = {}
        for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
            idx = np.searchsorted(cum_var, threshold)
            if idx < len(cum_var):
                milestones[f"{int(threshold*100)}%"] = int(idx + 1)

        layer_results[f"layer_{li}"] = {
            "n_concepts": n_concepts,
            "dim": dim,
            "top10_var": float(cum_var[9]) if n_comp >= 10 else None,
            "top20_var": float(cum_var[19]) if n_comp >= 20 else None,
            "top50_var": float(cum_var[49]) if n_comp >= 50 else None,
            "top100_var": float(cum_var[min(n_comp-1, 99)]),
            "max_eta2": float(max(factor_eta)) if factor_eta else 0,
            "mean_eta2_top10": float(np.mean(factor_eta[:10])) if factor_eta else 0,
            "milestones": milestones,
        }

    return layer_results


def golden_layer_deep_analysis(all_activations, concepts, golden_layer):
    """对黄金层做深入100维分析"""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD

    # 构建该层偏置矩阵
    word_vecs = []
    word_labels = []
    word_cats = []

    for cat_name, words_acts in all_activations.items():
        if cat_name not in concepts:
            continue
        cat_vecs = []
        for word, acts in words_acts.items():
            if golden_layer in acts:
                cat_vecs.append(acts[golden_layer])
        if not cat_vecs:
            continue
        basis = np.mean(cat_vecs, axis=0)

        for word, acts in words_acts.items():
            if golden_layer in acts:
                bias = acts[golden_layer] - basis
                word_vecs.append(bias)
                word_labels.append(word)
                word_cats.append(cat_name)

    bias_matrix = np.array(word_vecs)

    # 100维SVD
    scaler = StandardScaler()
    normed = scaler.fit_transform(bias_matrix)

    n_comp = min(100, min(normed.shape) - 1)
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    comp = svd.fit_transform(normed)

    var_exp = svd.explained_variance_ratio_
    cum_var = np.cumsum(var_exp)

    # 因子语义（前30个）
    factor_semantics = []
    for i in range(min(30, n_comp)):
        scores = comp[:, i]
        top_pos = np.argsort(scores)[-5:][::-1]
        top_neg = np.argsort(scores)[:5]
        factor_semantics.append({
            "factor": i,
            "variance": float(var_exp[i]),
            "cum_var": float(cum_var[i]),
            "top_positive": [(word_labels[j], float(scores[j])) for j in top_pos],
            "top_negative": [(word_labels[j], float(scores[j])) for j in top_neg],
        })

    # 因子-类别eta²（全100维）
    cat_to_idx = defaultdict(list)
    for i, cat in enumerate(word_cats):
        cat_to_idx[cat].append(i)

    factor_cat_eta = []
    for fi in range(n_comp):
        scores = comp[:, fi]
        grand_mean = np.mean(scores)
        ss_between = sum(len(idxs) * (np.mean(scores[idxs]) - grand_mean)**2
                       for idxs in cat_to_idx.values())
        ss_total = np.sum((scores - grand_mean)**2)
        eta_sq = ss_between / (ss_total + 1e-10)
        factor_cat_eta.append(float(eta_sq))

    # 类内分散度
    intra_disp = {}
    for cat, idxs in cat_to_idx.items():
        cat_comp = comp[idxs]
        intra_disp[cat] = float(np.mean(np.std(cat_comp, axis=0)))

    return {
        "n_concepts": len(word_labels),
        "n_components": n_comp,
        "variance_explained": var_exp.tolist(),
        "cumulative_variance": cum_var.tolist(),
        "factor_semantics": factor_semantics,
        "factor_cat_eta2": factor_cat_eta,
        "intra_dispersion": intra_disp,
        "n_high_eta": sum(1 for e in factor_cat_eta if e > 0.3),
    }


def run_model_experiment(model_name: str, model_path: Path) -> Dict:
    """单个模型的完整逐层分析"""
    print(f"\n{'='*70}")
    print(f"  Stage461: {model_name}")
    print(f"{'='*70}")

    # 1. 加载模型
    print(f"\n[1/5] Loading model...")
    t0 = time.time()
    model, tokenizer, layer_count, neuron_dim = load_model(model_path)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # 2. 提取所有概念在所有层的激活
    print(f"\n[2/5] Extracting activations per layer...")
    t0 = time.time()
    all_activations = extract_all_activations(model, tokenizer, CONCEPTS, layer_count)
    print(f"  Extracted in {time.time()-t0:.1f}s")

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

    # 3. 逐层SVD分析
    print(f"\n[3/5] Per-layer SVD analysis...")
    t0 = time.time()
    layer_results = per_layer_bias_analysis(all_activations, CONCEPTS, layer_count)

    # 找黄金层（top20_var最高的层）
    best_layer = max(layer_results, key=lambda k: layer_results[k].get("top20_var", 0) or 0)
    best_data = layer_results[best_layer]
    print(f"  Best layer: {best_layer} (top20={best_data.get('top20_var', 'N/A')}, "
          f"max_eta²={best_data.get('max_eta2', 'N/A')})")

    # 打印所有层摘要
    print(f"\n  Per-layer summary:")
    print(f"  {'Layer':<10} {'N':>4} {'Top10':>8} {'Top20':>8} {'Top50':>8} {'Top100':>8} {'MaxEta':>8}")
    for li_str in sorted(layer_results.keys()):
        lr = layer_results[li_str]
        print(f"  {li_str:<10} {lr['n_concepts']:>4} "
              f"{lr.get('top10_var',0)*100:>7.1f}% "
              f"{lr.get('top20_var',0)*100:>7.1f}% "
              f"{lr.get('top50_var',0)*100:>7.1f}% "
              f"{lr.get('top100_var',0)*100:>7.1f}% "
              f"{lr.get('max_eta2',0):>7.3f}")

    print(f"  Per-layer done in {time.time()-t0:.1f}s")

    # 4. 黄金层深入分析
    golden_li = int(best_layer.split("_")[1])
    print(f"\n[4/5] Golden layer deep analysis (L{golden_li})...")
    t0 = time.time()
    deep = golden_layer_deep_analysis(all_activations, CONCEPTS, golden_li)
    print(f"  100-dim SVD: {deep['cumulative_variance'][min(99, len(deep['cumulative_variance'])-1)]*100:.1f}%")
    print(f"  High eta² factors (>0.3): {deep['n_high_eta']}")
    print(f"  Done in {time.time()-t0:.1f}s")

    # 5. 重建精度
    print(f"\n[5/5] Reconstruction analysis...")
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import TruncatedSVD

    recon = {}
    for dim in [10, 20, 30, 50, 80, 100]:
        if dim >= deep["n_concepts"]:
            continue
        # 需要重新构建该层偏置矩阵
        word_vecs = []
        for cat_name, words_acts in all_activations.items():
            if cat_name not in CONCEPTS:
                continue
            cat_vecs = [acts[golden_li] for word, acts in words_acts.items() if golden_li in acts]
            if not cat_vecs:
                continue
            basis = np.mean(cat_vecs, axis=0)
            for word, acts in words_acts.items():
                if golden_li in acts:
                    word_vecs.append(acts[golden_li] - basis)

        bm = np.array(word_vecs)
        scaler = StandardScaler()
        normed = scaler.fit_transform(bm)
        svd = TruncatedSVD(n_components=dim, random_state=42)
        approx = svd.inverse_transform(svd.fit_transform(normed))
        residuals = normed - approx
        mse = float(np.mean(residuals**2))

        cos_sims = []
        for i in range(min(len(normed), 100)):
            n1 = np.linalg.norm(normed[i])
            n2 = np.linalg.norm(approx[i])
            if n1 > 1e-8 and n2 > 1e-8:
                cos_sims.append(float(np.dot(normed[i], approx[i]) / (n1 * n2)))

        recon[f"dim_{dim}"] = {
            "var": float(np.sum(svd.explained_variance_ratio_)),
            "cos": float(np.mean(cos_sims)) if cos_sims else 0,
            "mse": mse,
        }
        print(f"    dim={dim}: var={recon[f'dim_{dim}']['var']*100:.1f}%, cos={recon[f'dim_{dim}']['cos']:.3f}")

    return {
        "model_name": model_name,
        "layer_count": layer_count,
        "neuron_dim": neuron_dim,
        "golden_layer": golden_li,
        "layer_results": layer_results,
        "deep_analysis": deep,
        "reconstruction": recon,
    }


def build_report(all_results: Dict) -> str:
    lines = []
    lines.append("# Stage461: 单层偏置 + 逐层SVD + 100维分析 — 双模型")
    lines.append("")
    lines.append(f"**时间**: 2026-04-01 09:45")
    lines.append(f"**模型**: Qwen3-4B + DeepSeek-7B")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 1. 黄金层发现
    lines.append("## 1. 黄金层发现")
    lines.append("")
    for key, r in all_results.items():
        lines.append(f"### {r['model_name']}: 黄金层 = L{r['golden_layer']}")
        lines.append("")
        lr = r["layer_results"][f"layer_{r['golden_layer']}"]
        lines.append(f"- Top-10方差: {lr.get('top10_var',0)*100:.1f}%")
        lines.append(f"- Top-20方差: {lr.get('top20_var',0)*100:.1f}%")
        lines.append(f"- Top-50方差: {lr.get('top50_var',0)*100:.1f}%")
        lines.append(f"- Top-100方差: {lr.get('top100_var',0)*100:.1f}%")
        lines.append(f"- 最高eta²: {lr.get('max_eta2',0):.3f}")
        lines.append("")

    # 2. 逐层SVD摘要表
    lines.append("## 2. 逐层SVD方差解释（Top-20）")
    lines.append("")
    for key, r in all_results.items():
        lines.append(f"### {r['model_name']}")
        lines.append("")
        lines.append("| 层 | 概念数 | Top10 | Top20 | Top50 | Top100 | Max eta² |")
        lines.append("|----|-------|-------|-------|-------|--------|----------|")
        for li_str in sorted(r["layer_results"].keys()):
            lr = r["layer_results"][li_str]
            mark = " ★" if int(li_str.split("_")[1]) == r["golden_layer"] else ""
            lines.append(f"| {li_str}{mark} | {lr['n_concepts']} | "
                        f"{lr.get('top10_var',0)*100:.1f}% | {lr.get('top20_var',0)*100:.1f}% | "
                        f"{lr.get('top50_var',0)*100:.1f}% | {lr.get('top100_var',0)*100:.1f}% | "
                        f"{lr.get('max_eta2',0):.3f} |")
        lines.append("")

    # 3. 黄金层100维详细分析
    lines.append("## 3. 黄金层100维SVD分析")
    lines.append("")
    for key, r in all_results.items():
        d = r["deep_analysis"]
        lines.append(f"### {r['model_name']} (L{r['golden_layer']})")
        lines.append(f"- 概念数: {d['n_concepts']}, 维度: {d['n_components']}")
        cum = d["cumulative_variance"]
        lines.append(f"- Top-10: {cum[9]*100:.1f}%, Top-20: {cum[19]*100:.1f}%, "
                    f"Top-50: {cum[49]*100:.1f}%, Top-100: {cum[min(99,len(cum)-1)]*100:.1f}%")
        lines.append(f"- eta²>0.3的因子数: {d['n_high_eta']}")
        lines.append("")

        # 因子语义
        lines.append("#### 前20个因子语义")
        lines.append("")
        lines.append("| Factor | Var | Top+ | Top- |")
        lines.append("|--------|-----|------|------|")
        for fs in d["factor_semantics"][:20]:
            tp = ", ".join(f"{w}({s:.1f})" for w, s in fs["top_positive"][:3])
            tn = ", ".join(f"{w}({s:.1f})" for w, s in fs["top_negative"][:3])
            lines.append(f"| F{fs['factor']} | {fs['variance']*100:.2f}% | {tp} | {tn} |")
        lines.append("")

    # 4. 重建精度
    lines.append("## 4. 黄金层重建精度")
    lines.append("")
    lines.append("| 维度 | " + " | ".join(r["model_name"] for r in all_results.values()) + " |")
    lines.append("|------|" + "|".join(["------"]*len(all_results)) + "|")
    for dim_key in ["dim_10", "dim_20", "dim_30", "dim_50", "dim_80", "dim_100"]:
        row = [dim_key.replace("dim_", "")]
        for r in all_results.values():
            if dim_key in r["reconstruction"]:
                d = r["reconstruction"][dim_key]
                row.append(f"var={d['var']*100:.1f}%, cos={d['cos']:.3f}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    # 5. 核心结论
    lines.append("")
    lines.append("## 5. 核心结论")
    lines.append("")
    for key, r in all_results.items():
        d = r["deep_analysis"]
        cum = d["cumulative_variance"]
        lines.append(f"### {r['model_name']}")
        lines.append(f"- 黄金层L{r['golden_layer']}: 100维SVD解释{cum[min(99,len(cum)-1)]*100:.1f}%方差")
        lines.append(f"- 压缩比: {r['neuron_dim']}→100 = {r['neuron_dim']/100:.0f}:1")
        lines.append(f"- eta²>0.3因子: {d['n_high_eta']}/{d['n_components']}")
        lines.append("")

    # 单层vs6层对比
    lines.append("### 单层 vs 6层拼接对比（Stage460）")
    lines.append("")
    lines.append("| 方法 | Qwen3 Top-80 | DeepSeek Top-80 |")
    lines.append("|------|-------------|----------------|")
    lines.append("| 6层拼接(Stage460) | 44.1% | 39.5% |")
    for key, r in all_results.items():
        cum = r["deep_analysis"]["cumulative_variance"]
        var80 = cum[min(79, len(cum)-1)]*100
        if key == "qwen3":
            lines.append(f"| 单层L{r['golden_layer']}(Stage461) | {var80:.1f}% | - |")
        else:
            lines.append(f"| 单层L{r['golden_layer']}(Stage461) | - | {var80:.1f}% |")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  Stage461: Single-Layer Bias + Per-Layer SVD + 100-dim Analysis")
    print("  Models: Qwen3-4B -> DeepSeek-7B")
    print("=" * 70)

    all_results = {}

    # Qwen3-4B
    print("\n" + "#" * 70)
    print("# Round 1: Qwen3-4B")
    print("#" * 70)
    t0 = time.time()
    all_results["qwen3"] = run_model_experiment("Qwen3-4B", QWEN3_MODEL_PATH)
    print(f"\n  Qwen3-4B done in {time.time()-t0:.1f}s")

    # DeepSeek-7B
    print("\n" + "#" * 70)
    print("# Round 2: DeepSeek-7B")
    print("#" * 70)
    t0 = time.time()
    all_results["deepseek"] = run_model_experiment("DeepSeek-7B", DEEPSEEK7B_MODEL_PATH)
    print(f"\n  DeepSeek-7B done in {time.time()-t0:.1f}s")

    # 保存
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_for_json(all_results), f, ensure_ascii=False, indent=2)

    report = build_report(all_results)
    report_path = OUTPUT_DIR / "REPORT.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    # 摘要
    print("\n" + "=" * 70)
    print("  Stage461 Summary")
    print("=" * 70)
    for key, r in all_results.items():
        d = r["deep_analysis"]
        cum = d["cumulative_variance"]
        print(f"\n  {r['model_name']}:")
        print(f"    Golden Layer: L{r['golden_layer']}")
        print(f"    Top-20: {cum[19]*100:.1f}%, Top-50: {cum[49]*100:.1f}%, Top-100: {cum[min(99,len(cum)-1)]*100:.1f}%")
        print(f"    High eta² factors: {d['n_high_eta']}")

    print(f"\n  Report: {report_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
