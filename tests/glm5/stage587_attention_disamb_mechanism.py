#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
stage587: 四模型attention消歧精确机制分析
目标：
  1. 对20个歧义词在四模型上提取消歧峰值层attention pattern
  2. 对比同词不同语境的attention pattern差异（哪些head区分语境）
  3. 逐层追踪attention消歧的"浮现-衰减"曲线
  4. 分析attention对上下文词的关注模式（功能词 vs 内容词）
  5. 验证跨模型不变量：80%走attention路径、消歧峰值在~25%层
模型：Qwen3 / DeepSeek7B / GLM4 / Gemma4（依次运行，避免GPU溢出）
"""

from __future__ import annotations
import sys, json, time, gc, torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    load_model_bundle, free_model, get_model_device,
    discover_layers, ZeroModule, move_batch_to_model_device
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")

# 20个歧义词及两个语境（沿用stage573）
POLYSEMY_WORDS = [
    {
        "word": "bank",
        "ctx1": "The river bank was muddy after the heavy rain.",
        "ctx2": "The bank approved the loan for the new business.",
        "sense1": "river", "sense2": "finance",
    },
    {
        "word": "plant",
        "ctx1": "The factory plant employs over five hundred workers.",
        "ctx2": "She watered the plant in the garden every morning.",
        "sense1": "factory", "sense2": "botany",
    },
    {
        "word": "spring",
        "ctx1": "The hot spring resort attracts many tourists each year.",
        "ctx2": "Spring is the most beautiful season of the year.",
        "sense1": "water", "sense2": "season",
    },
    {
        "word": "apple",
        "ctx1": "She ate a sweet red apple from the orchard.",
        "ctx2": "Apple released the new iPhone at the conference.",
        "sense1": "fruit", "sense2": "company",
    },
    {
        "word": "pool",
        "ctx1": "They swam in the swimming pool all afternoon.",
        "ctx2": "The car pool arrangement saved everyone money.",
        "sense1": "water", "sense2": "shared",
    },
    {
        "word": "seal",
        "ctx1": "The seal balanced a ball on its nose at the zoo.",
        "ctx2": "Please seal the envelope before mailing it.",
        "sense1": "animal", "sense2": "close",
    },
    {
        "word": "fair",
        "ctx1": "The county fair had rides and games for everyone.",
        "ctx2": "The judge made a fair and impartial decision.",
        "sense1": "event", "sense2": "just",
    },
    {
        "word": "match",
        "ctx1": "He struck the match to light the candle.",
        "ctx2": "The football match was exciting to watch.",
        "sense1": "fire", "sense2": "game",
    },
    {
        "word": "ruler",
        "ctx1": "She used a ruler to measure the desk accurately.",
        "ctx2": "The cruel ruler oppressed the people for decades.",
        "sense1": "tool", "sense2": "leader",
    },
    {
        "word": "light",
        "ctx1": "Please turn on the light in the dark room.",
        "ctx2": "The package was very light and easy to carry.",
        "sense1": "illumination", "sense2": "weight",
    },
    {
        "word": "nail",
        "ctx1": "He hammered the nail into the wooden board.",
        "ctx2": "She painted her fingernail a bright red color.",
        "sense1": "metal", "sense2": "body",
    },
    {
        "word": "bat",
        "ctx1": "The baseball bat was made of solid wood.",
        "ctx2": "The bat flew out of the cave at sunset.",
        "sense1": "equipment", "sense2": "animal",
    },
    {
        "word": "square",
        "ctx1": "The town square was decorated for the festival.",
        "ctx2": "A square has four equal sides and four right angles.",
        "sense1": "place", "sense2": "shape",
    },
    {
        "word": "draft",
        "ctx1": "The soldier was drafted into the army last year.",
        "ctx2": "She wrote the first draft of her novel quickly.",
        "sense1": "military", "sense2": "writing",
    },
    {
        "word": "ring",
        "ctx1": "He bought a diamond ring for the proposal.",
        "ctx2": "Please ring the bell when you arrive at the door.",
        "sense1": "jewelry", "sense2": "sound",
    },
    {
        "word": "bow",
        "ctx1": "The archer drew the bow and aimed at the target.",
        "ctx2": "She tied a bow on the birthday gift ribbon.",
        "sense1": "weapon", "sense2": "knot",
    },
    {
        "word": "palm",
        "ctx1": "The palm tree swayed gently in the tropical breeze.",
        "ctx2": "He read her future in the palm of her hand.",
        "sense1": "tree", "sense2": "hand",
    },
    {
        "word": "club",
        "ctx1": "He hit the golf ball with a heavy iron club.",
        "ctx2": "She goes to the dance club every Friday night.",
        "sense1": "equipment", "sense2": "social",
    },
    {
        "word": "jam",
        "ctx1": "She spread strawberry jam on the toast for breakfast.",
        "ctx2": "Traffic jam delayed us for over an hour this morning.",
        "sense1": "food", "sense2": "congestion",
    },
    {
        "word": "shed",
        "ctx1": "He keeps the lawnmower in the garden shed.",
        "ctx2": "The snake shed its skin and grew a new one.",
        "sense1": "building", "sense2": "discard",
    },
]


def cos(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def get_attn_all_heads(model, tokenizer, sentence, layer_idx):
    """提取指定层所有head的attention pattern → (n_heads, seq, seq)"""
    layers = discover_layers(model)
    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)

    attn_data = {}

    def hook_fn(module, inp, output):
        if isinstance(output, tuple) and len(output) > 1:
            attn_data["weights"] = output[1].detach().float().cpu()
        return output

    handles = [layers[layer_idx].self_attn.register_forward_hook(hook_fn)]
    with torch.no_grad():
        model(**enc)
    for h in handles:
        h.remove()

    if "weights" in attn_data:
        return attn_data["weights"][0]  # (n_heads, seq, seq)
    return None


def get_all_layer_attn(model, tokenizer, sentence, layer_indices=None):
    """提取多个层的所有head attention pattern"""
    layers = discover_layers(model)
    n_layers = len(layers)
    if layer_indices is None:
        # 每隔几层采样 + 关键层
        step = max(1, n_layers // 12)
        layer_indices = sorted(set([0, 1, 2, 3] + list(range(0, n_layers, step)) + [n_layers - 1]))
        layer_indices = [l for l in layer_indices if l < n_layers]

    enc = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    enc = move_batch_to_model_device(model, enc)

    all_attn = {}
    hooks = []

    for li in layer_indices:
        def make_hook(idx):
            def hook_fn(module, inp, output):
                if isinstance(output, tuple) and len(output) > 1:
                    all_attn[idx] = output[1].detach().float().cpu()[0]
                return output
            return hook_fn
        hooks.append(layers[li].self_attn.register_forward_hook(make_hook(li)))

    with torch.no_grad():
        model(**enc)
    for h in hooks:
        h.remove()

    return all_attn


def layer_wise_disamb(model, tokenizer, s1, s2):
    """逐层消歧cosine距离"""
    enc1 = tokenizer(s1, return_tensors="pt", truncation=True, max_length=64)
    enc2 = tokenizer(s2, return_tensors="pt", truncation=True, max_length=64)
    enc1 = move_batch_to_model_device(model, enc1)
    enc2 = move_batch_to_model_device(model, enc2)
    with torch.no_grad():
        out1 = model(**enc1, output_hidden_states=True)
        out2 = model(**enc2, output_hidden_states=True)
    scores = []
    for h1, h2 in zip(out1.hidden_states, out2.hidden_states):
        scores.append(cos(h1[0, -1, :].float().cpu(), h2[0, -1, :].float().cpu()))
    return scores


def find_disamb_peak_layer(scores):
    """找消歧度最大的层（1-cos最小，即cos最小）"""
    # scores有n_layers+1个元素（含embedding层），映射到0-indexed层
    min_cos = min(scores[1:])  # 跳过embedding层(L0)
    peak = scores[1:].index(min_cos) + 1  # +1因为跳过了L0
    return peak, 1 - min_cos


def attn_pattern_similarity(attn1, attn2):
    """计算两组attention pattern之间的相似度（逐head cosine）"""
    # attn: (n_heads, seq1, seq1), attn2: (n_heads, seq2, seq2)
    # 取公共长度（通常seq1≈seq2）
    min_seq = min(attn1.shape[-1], attn2.shape[-1])
    a1 = attn1[:, :min_seq, :min_seq]  # (n_heads, min_seq, min_seq)
    a2 = attn2[:, :min_seq, :min_seq]

    n_heads = a1.shape[0]
    head_sims = []
    for h in range(n_heads):
        # 把每个head的pattern展平成向量
        v1 = a1[h].flatten()
        v2 = a2[h].flatten()
        if v1.norm() > 1e-8 and v2.norm() > 1e-8:
            head_sims.append(cos(v1, v2))
        else:
            head_sims.append(1.0)
    return head_sims


def classify_path(scores, peak_layer, peak_disamb):
    """判断消歧路径类型"""
    end_disamb = 1 - scores[-1]
    if peak_disamb < 0.05:
        return "none", peak_disamb, end_disamb
    if end_disamb < peak_disamb * 0.6:
        return "attention", peak_disamb, end_disamb
    elif end_disamb >= peak_disamb * 0.8:
        return "mlp", peak_disamb, end_disamb
    else:
        return "mixed", peak_disamb, end_disamb


def identify_discriminating_heads(head_sims):
    """找出区分两个语境最多的head（similarity最低的head）"""
    sims = np.array(head_sims)
    mean_sim = sims.mean()
    # 找sim < mean - 1std的head（显著区分语境的head）
    std_sim = sims.std()
    threshold = max(mean_sim - std_sim, 0.0)
    disc_heads = np.where(sims < threshold)[0]
    return {
        "mean_sim": round(float(mean_sim), 4),
        "std_sim": round(float(std_sim), 4),
        "min_sim": round(float(sims.min()), 4),
        "max_sim": round(float(sims.max()), 4),
        "disc_head_indices": disc_heads.tolist(),
        "n_disc_heads": len(disc_heads),
        "n_total_heads": len(sims),
    }


def analyze_context_attention(attn_weights, tokens, target_token_pos=-1):
    """分析target token对所有context token的attention分布"""
    # attn_weights: (n_heads, seq, seq)
    n_heads, seq_len, _ = attn_weights.shape

    # target对各位置的attention → 各head取mean
    if target_token_pos < 0:
        target_token_pos = seq_len - 1

    target_attn = attn_weights[:, target_token_pos, :].mean(dim=0)  # (seq,)

    # 分类：功能词 vs 内容词
    func_words = {"the", "a", "an", "is", "was", "in", "on", "at", "to", "for",
                  "of", "and", "or", "it", "he", "she", "they", "with", "by",
                  "from", "that", "this", "his", "her", "its", "their", "has",
                  "had", "have", "been", "be", "are", "do", "does", "did"}

    func_attn = 0.0
    content_attn = 0.0
    func_count = 0
    content_count = 0
    details = []

    for i, tok in enumerate(tokens):
        tok_lower = tok.lower().strip()
        attn_val = target_attn[i].item()
        is_func = tok_lower in func_words
        details.append({"pos": i, "token": tok[:15], "attn": round(attn_val, 4), "is_func": is_func})
        if is_func:
            func_attn += attn_val
            func_count += 1
        else:
            content_attn += attn_val
            content_count += 1

    return {
        "func_attn_ratio": round(func_attn / max(func_attn + content_attn, 1e-8), 4),
        "content_attn_ratio": round(content_attn / max(func_attn + content_attn, 1e-8), 4),
        "func_count": func_count,
        "content_count": content_count,
        "top3_attended": sorted(details, key=lambda x: -x["attn"])[:3],
    }


def run_single_model(model_key):
    """单个模型的完整分析"""
    print(f"\n{'='*60}")
    print(f"加载模型: {model_key}")
    print(f"{'='*60}")
    t0 = time.time()

    model, tokenizer = load_model_bundle(model_key)
    layers = discover_layers(model)
    n_layers = len(layers)
    device = get_model_device(model)
    print(f"  层数={n_layers}, 设备={device}")

    # 先获取一个sample来确定peak layer近似位置
    sample_scores = layer_wise_disamb(model, tokenizer,
        POLYSEMY_WORDS[0]["ctx1"], POLYSEMY_WORDS[0]["ctx2"])
    peak_pct = sample_scores.index(min(sample_scores)) / max(n_layers - 1, 1)
    print(f"  示例bank消歧峰值层比例: {peak_pct:.2%}")

    # 采样层：覆盖L0-L3 + peak附近 + 晚层
    peak_approx = int(n_layers * 0.25)
    scan_layers = sorted(set(
        [0, 1, 2, 3, 4] +
        [max(0, peak_approx - 2), peak_approx, peak_approx + 1, peak_approx + 2] +
        [n_layers // 2] +
        [n_layers - 4, n_layers - 3, n_layers - 2, n_layers - 1]
    ))
    scan_layers = [l for l in scan_layers if l < n_layers]

    word_results = []
    path_counts = {"attention": 0, "mlp": 0, "mixed": 0, "none": 0}

    for wi, pw in enumerate(POLYSEMY_WORDS):
        word = pw["word"]
        print(f"\n  [{wi+1}/20] {word}...")
        wt0 = time.time()

        # 1. 逐层消歧分析
        disamb_scores = layer_wise_disamb(model, tokenizer, pw["ctx1"], pw["ctx2"])
        peak_layer, peak_disamb = find_disamb_peak_layer(disamb_scores)
        # 确保peak_layer在有效层范围内（hidden_states有n_layers+1项）
        peak_layer = min(peak_layer, n_layers - 1)
        path_type, peak_d, end_d = classify_path(disamb_scores, peak_layer, peak_disamb)
        path_counts[path_type] += 1

        # 2. 在峰值层提取attention pattern
        peak_attn1 = get_attn_all_heads(model, tokenizer, pw["ctx1"], peak_layer)
        peak_attn2 = get_attn_all_heads(model, tokenizer, pw["ctx2"], peak_layer)

        head_analysis = {}
        context_analysis = {}
        if peak_attn1 is not None and peak_attn2 is not None:
            head_sims = attn_pattern_similarity(peak_attn1, peak_attn2)
            head_analysis = identify_discriminating_heads(head_sims)

            # 分析target token对context的attention
            tok1 = tokenizer.tokenize(pw["ctx1"])
            tok2 = tokenizer.tokenize(pw["ctx2"])
            context_analysis["ctx1"] = analyze_context_attention(peak_attn1, tok1)
            context_analysis["ctx2"] = analyze_context_attention(peak_attn2, tok2)

        # 3. 在L0和末层也提取attention（验证入口/出口瓶颈）
        l0_attn1 = get_attn_all_heads(model, tokenizer, pw["ctx1"], 0)
        l0_attn2 = get_attn_all_heads(model, tokenizer, pw["ctx2"], 0)
        l0_head_sims = []
        if l0_attn1 is not None and l0_attn2 is not None:
            l0_head_sims = attn_pattern_similarity(l0_attn1, l0_attn2)

        last_attn1 = get_attn_all_heads(model, tokenizer, pw["ctx1"], n_layers - 1)
        last_attn2 = get_attn_all_heads(model, tokenizer, pw["ctx2"], n_layers - 1)
        last_head_sims = []
        if last_attn1 is not None and last_attn2 is not None:
            last_head_sims = attn_pattern_similarity(last_attn1, last_attn2)

        # 4. 逐层attention dissimilarity曲线（选几个关键层）
        layer_attn_dissim = {}
        for li in scan_layers:
            a1 = get_attn_all_heads(model, tokenizer, pw["ctx1"], li)
            a2 = get_attn_all_heads(model, tokenizer, pw["ctx2"], li)
            if a1 is not None and a2 is not None:
                hs = attn_pattern_similarity(a1, a2)
                layer_attn_dissim[str(li)] = {
                    "mean_sim": round(float(np.mean(hs)), 4),
                    "min_sim": round(float(np.min(hs)), 4),
                    "std_sim": round(float(np.std(hs)), 4),
                }

        wt_elapsed = time.time() - wt0
        print(f"    路径={path_type}, 峰值层=L{peak_layer}, 峰值消歧={peak_disamb:.4f}, "
              f"区分head数={head_analysis.get('n_disc_heads', 'N/A')}, "
              f"耗时={wt_elapsed:.1f}s")

        word_results.append({
            "word": word,
            "path_type": path_type,
            "peak_layer": peak_layer,
            "peak_layer_pct": round(peak_layer / max(n_layers - 1, 1), 4),
            "peak_disamb": round(peak_disamb, 4),
            "end_disamb": round(end_d, 4),
            "disamb_decay_ratio": round(end_d / max(peak_disamb, 1e-8), 4),
            "head_analysis": head_analysis,
            "context_analysis": context_analysis,
            "l0_mean_attn_sim": round(float(np.mean(l0_head_sims)), 4) if l0_head_sims else None,
            "last_mean_attn_sim": round(float(np.mean(last_head_sims)), 4) if last_head_sims else None,
            "layer_attn_dissim": layer_attn_dissim,
            "time_s": round(wt_elapsed, 1),
        })

    total_time = time.time() - t0

    # 汇总统计
    attn_count = path_counts["attention"] + path_counts["mixed"]
    total_count = sum(path_counts.values())
    attn_ratio = attn_count / max(total_count, 1)

    peak_layers_pct = [r["peak_layer_pct"] for r in word_results if r["path_type"] != "none"]
    mean_peak_pct = np.mean(peak_layers_pct) if peak_layers_pct else 0

    disc_head_counts = [r["head_analysis"].get("n_disc_heads", 0) for r in word_results
                        if "n_disc_heads" in r.get("head_analysis", {})]
    mean_disc_heads = np.mean(disc_head_counts) if disc_head_counts else 0

    # 功能词attention比例
    func_ratios = []
    for r in word_results:
        ca = r.get("context_analysis", {})
        if "ctx1" in ca:
            func_ratios.append(ca["ctx1"]["func_attn_ratio"])
            func_ratios.append(ca["ctx2"]["func_attn_ratio"])
    mean_func_ratio = np.mean(func_ratios) if func_ratios else 0

    summary = {
        "model": model_key,
        "n_layers": n_layers,
        "total_time_s": round(total_time, 1),
        "path_distribution": path_counts,
        "attention_ratio": round(attn_ratio, 4),
        "mean_peak_layer_pct": round(mean_peak_pct, 4),
        "mean_disc_heads": round(mean_disc_heads, 2),
        "mean_func_attn_ratio": round(mean_func_ratio, 4),
        "word_results": word_results,
    }

    # 释放模型
    free_model(model)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(3)

    return summary


def main():
    print("=" * 60)
    print("Stage 587: 四模型attention消歧精确机制分析")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    model_keys = ["qwen3", "deepseek7b", "glm4", "gemma4"]
    all_summaries = {}

    for mk in model_keys:
        try:
            summary = run_single_model(mk)
            all_summaries[mk] = summary
        except Exception as e:
            print(f"\n  模型 {mk} 出错: {e}")
            import traceback
            traceback.print_exc()
            all_summaries[mk] = {"error": str(e)}

    # 跨模型对比
    cross_model = {}
    for mk, s in all_summaries.items():
        if "error" in s:
            continue
        cross_model[mk] = {
            "attention_ratio": s["attention_ratio"],
            "mean_peak_layer_pct": s["mean_peak_layer_pct"],
            "mean_disc_heads": s["mean_disc_heads"],
            "mean_func_attn_ratio": s["mean_func_attn_ratio"],
        }

    final = {
        "timestamp": TIMESTAMP,
        "stage": "587",
        "title": "四模型attention消歧精确机制分析",
        "cross_model_summary": cross_model,
        "models": all_summaries,
    }

    out_path = OUTPUT_DIR / f"stage587_attention_disamb_mechanism_{TIMESTAMP}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")

    # 打印关键结果
    print("\n" + "=" * 60)
    print("跨模型对比总结")
    print("=" * 60)
    for mk, cm in cross_model.items():
        print(f"\n  {mk}:")
        print(f"    attention路径比例: {cm['attention_ratio']:.1%}")
        print(f"    消歧峰值层位置: {cm['mean_peak_layer_pct']:.1%}")
        print(f"    平均区分head数: {cm['mean_disc_heads']:.1f}")
        print(f"    功能词attention比例: {cm['mean_func_attn_ratio']:.1%}")

    # 不变量验证
    print("\n" + "=" * 60)
    print("不变量验证")
    print("=" * 60)
    attn_ratios = [cm["attention_ratio"] for cm in cross_model.values()]
    peak_pcts = [cm["mean_peak_layer_pct"] for cm in cross_model.values()]
    disc_heads = [cm["mean_disc_heads"] for cm in cross_model.values()]
    func_ratios = [cm["mean_func_attn_ratio"] for cm in cross_model.values()]

    print(f"\n  INV-40验证(80%走attention): {attn_ratios}")
    print(f"    范围: {min(attn_ratios):.1%} ~ {max(attn_ratios):.1%}")
    print(f"    跨模型一致: {'YES' if max(attn_ratios) - min(attn_ratios) < 0.2 else 'NO'}")

    print(f"\n  消歧峰值层位置: {peak_pcts}")
    print(f"    范围: {min(peak_pcts):.1%} ~ {max(peak_pcts):.1%}")

    print(f"\n  区分head数: {disc_heads}")
    print(f"    范围: {min(disc_heads):.1f} ~ {max(disc_heads):.1f}")

    print(f"\n  功能词attention比例: {func_ratios}")
    print(f"    范围: {min(func_ratios):.1%} ~ {max(func_ratios):.1%}")


if __name__ == "__main__":
    main()
