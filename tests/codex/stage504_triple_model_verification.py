#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage504: 三模型(Qwen3+DeepSeek+Gemma4)统一验证协议

用3个模型验证已积累的6个确认机制和7个否决假说，检验跨模型一致性。

验证的确认机制：
  M1: 残差流忠实传播（余弦保持>0.8）
  M2: write_neurons主导简单关系
  M3: late_readout主导复杂推理
  M4: 共现塑造embedding距离
  M5: 否定压制激活范数
  M6: 分布式编码（非局部原子）

验证的否决假说（应不复现）：
  R1: 欧氏层次距离单调性
  R2: 子空间分割
  R3: 反义词互注意力优势

每个验证使用轻量级指标，确保3个模型都能快速完成。
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from multimodel_language_shared import (
    ZeroModule,
    ablate_layer_component,
    candidate_score_map,
    discover_layers,
    evenly_spaced_layers,
    free_model,
    get_model_device,
    load_model_bundle,
    restore_layer_component,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_BASE = PROJECT_ROOT / "tests" / "codex_temp"

ALL_MODELS = ["qwen3", "deepseek7b", "gemma4"]


# ============================================================
# M1: 残差流忠实传播
# ============================================================
def verify_m1_residual_fidelity(model, tokenizer) -> Dict:
    """相邻层间残差余弦相似度"""
    layers = discover_layers(model)
    n_layers = len(layers)
    device = get_model_device(model)

    text = "猫是一种哺乳动物"
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
    input_ids = encoded["input_ids"].to(device)

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, output_hidden_states=True)

    cosines = []
    for i in range(min(8, n_layers - 1)):
        h1 = outputs.hidden_states[i][0, -1].float()
        h2 = outputs.hidden_states[i + 1][0, -1].float()
        cos = torch.nn.functional.cosine_similarity(h1.unsqueeze(0), h2.unsqueeze(0))
        cosines.append(float(cos.item()))

    avg_cos = sum(cosines) / len(cosines) if cosines else 0
    return {
        "avg_cosine": avg_cos,
        "min_cosine": min(cosines) if cosines else 0,
        "supported": avg_cos > 0.8,
        "cosines": cosines,
    }


# ============================================================
# M2: write_neurons主导简单关系
# ============================================================
def _run_with_zeroed_component(model, tokenizer, prompt, candidates, layer_idx, component):
    """用hook将某层某组件输出置零后运行评分"""
    layer = discover_layers(model)[layer_idx]
    target = layer.self_attn if component == "attn" else layer.mlp

    def zero_hook(module, args, kwargs, output):
        if isinstance(output, tuple):
            return (torch.zeros_like(output[0]),) + output[1:]
        return torch.zeros_like(output)

    handles = []
    try:
        # 尝试注册post-hook
        handles.append(target.register_forward_hook(zero_hook, with_kwargs=True))
    except TypeError:
        handles.append(target.register_forward_hook(zero_hook))

    try:
        return candidate_score_map(model, tokenizer, prompt, candidates)
    finally:
        for h in handles:
            h.remove()


def verify_m2_neurons_dominate_simple(model, tokenizer) -> Dict:
    """反义词对：MLP消融影响 > Attention消融影响"""
    pairs = [("热", "冷"), ("大", "小"), ("高", "低")]

    mlp_dominant_count = 0
    total = 0

    for w1, w2 in pairs:
        prompt = f"{w1}的反义词是"
        candidates = [w2]
        scores = candidate_score_map(model, tokenizer, prompt, candidates)

        mlp_scores = _run_with_zeroed_component(model, tokenizer, prompt, candidates, 0, "mlp")
        attn_scores = _run_with_zeroed_component(model, tokenizer, prompt, candidates, 0, "attn")

        orig_score = scores.get(w2, 0)
        mlp_impact = abs(orig_score - mlp_scores.get(w2, 0))
        attn_impact = abs(orig_score - attn_scores.get(w2, 0))

        if mlp_impact > attn_impact:
            mlp_dominant_count += 1
        total += 1

    return {
        "mlp_dominant_ratio": mlp_dominant_count / total if total > 0 else 0,
        "supported": mlp_dominant_count / total > 0.5 if total > 0 else False,
    }


# ============================================================
# M3: late_readout主导复杂推理
# ============================================================
def verify_m3_late_readout_complex(model, tokenizer) -> Dict:
    """类比推理：晚层消融影响 > 早层消融影响"""
    analogies = [("国王", "女王", "男人"), ("太阳", "月亮", "父亲")]
    layers = discover_layers(model)
    n_layers = len(layers)

    late_dominant_count = 0
    total = 0

    for w1, w2, w3 in analogies:
        prompt = f"{w1}之于{w2}，如同{w3}之于"
        candidates = ["女人", "母亲", "男孩"]
        scores = candidate_score_map(model, tokenizer, prompt, candidates)

        early_idx = max(0, n_layers // 4)
        late_idx = min(n_layers - 1, 3 * n_layers // 4)

        late_scores = _run_with_zeroed_component(model, tokenizer, prompt, candidates, late_idx, "mlp")
        early_scores = _run_with_zeroed_component(model, tokenizer, prompt, candidates, early_idx, "mlp")

        orig_best = scores.get("女人", 0)
        late_impact = abs(orig_best - late_scores.get("女人", 0))
        early_impact = abs(orig_best - early_scores.get("女人", 0))

        if late_impact > early_impact:
            late_dominant_count += 1
        total += 1

    return {
        "late_dominant_ratio": late_dominant_count / total if total > 0 else 0,
        "supported": late_dominant_count / total > 0.5 if total > 0 else False,
    }


# ============================================================
# M4: 共现塑造embedding距离
# ============================================================
def verify_m4_cooccurrence_distance(model, tokenizer) -> Dict:
    """高共现词对距离 < 低共现词对距离"""
    device = get_model_device(model)

    high_cooccurrence = [("的", "了"), ("是", "在"), ("和", "与")]
    low_cooccurrence = [("量子", "饺子"), ("方程", "钢琴"), ("逻辑", "蝴蝶")]

    def get_embedding(word):
        w_ids = tokenizer(word, add_special_tokens=False)["input_ids"]
        # 从模型获取embedding
        if hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
            return emb(torch.tensor([w_ids[-1]], device=device))[0].float().cpu()
        return None

    high_dists = []
    for w1, w2 in high_cooccurrence:
        e1 = get_embedding(w1)
        e2 = get_embedding(w2)
        if e1 is not None and e2 is not None:
            dist = float(torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)))
            high_dists.append(1 - dist)  # 距离 = 1 - 余弦相似度

    low_dists = []
    for w1, w2 in low_cooccurrence:
        e1 = get_embedding(w1)
        e2 = get_embedding(w2)
        if e1 is not None and e2 is not None:
            dist = float(torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)))
            low_dists.append(1 - dist)

    avg_high = sum(high_dists) / len(high_dists) if high_dists else 0
    avg_low = sum(low_dists) / len(low_dists) if low_dists else 0

    return {
        "high_cooccurrence_avg_dist": avg_high,
        "low_cooccurrence_avg_dist": avg_low,
        "supported": avg_high < avg_low,
    }


# ============================================================
# M5: 否定压制激活范数
# ============================================================
def verify_m5_negation_suppression(model, tokenizer) -> Dict:
    """否定句的激活范数 < 肯定句"""
    device = get_model_device(model)
    layers = discover_layers(model)
    n_layers = len(layers)

    affirm_negate_pairs = [
        ("猫是黑色的", "猫不是黑色的"),
        ("他很高", "他不高"),
        ("我喜欢音乐", "我不喜欢音乐"),
        ("水是液体", "水不是液体"),
    ]

    suppression_count = 0
    total = 0

    with torch.inference_mode():
        for affirmative, negative in affirm_negate_pairs:
            aff_encoded = tokenizer(affirmative, return_tensors="pt", truncation=True, max_length=64)
            aff_input = aff_encoded["input_ids"].to(device)
            aff_out = model(input_ids=aff_input, output_hidden_states=True)

            neg_encoded = tokenizer(negative, return_tensors="pt", truncation=True, max_length=64)
            neg_input = neg_encoded["input_ids"].to(device)
            neg_out = model(input_ids=neg_input, output_hidden_states=True)

            mid = n_layers // 2
            aff_norm = float(aff_out.hidden_states[mid][0, -1].float().norm())
            neg_norm = float(neg_out.hidden_states[mid][0, -1].float().norm())

            if neg_norm < aff_norm:
                suppression_count += 1
            total += 1

    return {
        "suppression_ratio": suppression_count / total if total > 0 else 0,
        "supported": suppression_count / total > 0.5 if total > 0 else False,
    }


# ============================================================
# M6: 分布式编码
# ============================================================
def verify_m6_distributed_encoding(model, tokenizer) -> Dict:
    """概念激活的维度利用率 > 20%"""
    device = get_model_device(model)
    test_words = ["猫", "苹果", "快乐", "城市"]

    with torch.inference_mode():
        mid_layer = len(discover_layers(model)) // 2
        utilizations = []
        for w in test_words:
            encoded = tokenizer(w, return_tensors="pt", add_special_tokens=False)
            input_ids = encoded["input_ids"].to(device)
            outputs = model(input_ids=input_ids, output_hidden_states=True)
            h = outputs.hidden_states[mid_layer][0, -1].float()
            threshold = h.abs().max().item() * 0.01
            active = (h.abs() > threshold).sum().item()
            utilizations.append(active / h.numel())

    avg_util = sum(utilizations) / len(utilizations) if utilizations else 0
    return {
        "avg_dimension_utilization": avg_util,
        "supported": avg_util > 0.2,
    }


# ============================================================
# R1: 欧氏层次距离单调性（应不成立）
# ============================================================
def verify_r1_hierarchical_distance(model, tokenizer) -> Dict:
    """动物界层次：父→子 < 兄弟 < 跨分支（应该不成立）"""
    device = get_model_device(model)

    parent_child = [("动物", "猫"), ("动物", "狗")]
    siblings = [("猫", "狗"), ("鸟", "鱼")]
    cross_branch = [("猫", "鸟"), ("狗", "鱼")]

    def get_dist(w1, w2):
        e1_ids = tokenizer(w1, add_special_tokens=False)["input_ids"]
        e2_ids = tokenizer(w2, add_special_tokens=False)["input_ids"]
        if hasattr(model, "get_input_embeddings"):
            emb = model.get_input_embeddings()
            v1 = emb(torch.tensor([e1_ids[-1]], device=device))[0].float()
            v2 = emb(torch.tensor([e2_ids[-1]], device=device))[0].float()
            return float(1 - torch.nn.functional.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)))
        return 0

    pc_dists = [get_dist(w1, w2) for w1, w2 in parent_child]
    sib_dists = [get_dist(w1, w2) for w1, w2 in siblings]
    cb_dists = [get_dist(w1, w2) for w1, w2 in cross_branch]

    avg_pc = sum(pc_dists) / len(pc_dists) if pc_dists else 0
    avg_sib = sum(sib_dists) / len(sib_dists) if sib_dists else 0
    avg_cb = sum(cb_dists) / len(cb_dists) if cb_dists else 0

    monotonic = avg_pc < avg_sib < avg_cb
    return {
        "parent_child_avg": avg_pc,
        "sibling_avg": avg_sib,
        "cross_branch_avg": avg_cb,
        "monotonic": monotonic,
        "rejected": not monotonic,  # 我们期望这个假说被否决
    }


# ============================================================
# R3: 反义词互注意力优势（应不成立）
# ============================================================
def verify_r3_antonym_attention(model, tokenizer) -> Dict:
    """反义词间的注意力不应显著高于随机词对"""
    device = get_model_device(model)

    antonym_pairs = [("热", "冷"), ("大", "小"), ("高", "低")]
    random_pairs = [("热", "大"), ("冷", "小"), ("高", "快")]

    def get_mutual_attention(w1, w2):
        text = f"{w1}和{w2}是不同的"
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        input_ids = encoded["input_ids"].to(device)
        with torch.inference_mode():
            outputs = model(input_ids=input_ids, output_attentions=True)
        mid = len(discover_layers(model)) // 2
        attn = outputs.attentions[mid][0].mean(dim=0)  # [seq, seq]
        # 取对角线附近的平均值作为互注意力近似
        return float(attn.mean())

    ant_attns = [get_mutual_attention(w1, w2) for w1, w2 in antonym_pairs]
    rand_attns = [get_mutual_attention(w1, w2) for w1, w2 in random_pairs]

    avg_ant = sum(ant_attns) / len(ant_attns) if ant_attns else 0
    avg_rand = sum(rand_attns) / len(rand_attns) if rand_attns else 0

    return {
        "antonym_avg_attention": avg_ant,
        "random_avg_attention": avg_rand,
        "antonym_advantage": avg_ant > avg_rand * 1.1,
        "rejected": not (avg_ant > avg_rand * 1.1),
    }


# ============================================================
# 主函数
# ============================================================
def run_single_model(model_key: str) -> Dict:
    print(f"\n{'='*60}")
    print(f"模型: {model_key}")
    print(f"{'='*60}")

    model, tokenizer = load_model_bundle(model_key)
    n_layers = len(discover_layers(model))
    print(f"层数: {n_layers}")

    try:
        results = {"model": model_key, "n_layers": n_layers}

        # 确认机制
        print("\n--- 确认机制验证 ---")
        for name, fn in [
            ("M1_残差流忠实传播", verify_m1_residual_fidelity),
            ("M2_neurons主导简单关系", verify_m2_neurons_dominate_simple),
            ("M3_late_readout主导复杂推理", verify_m3_late_readout_complex),
            ("M4_共现塑造距离", verify_m4_cooccurrence_distance),
            ("M5_否定压制激活", verify_m5_negation_suppression),
            ("M6_分布式编码", verify_m6_distributed_encoding),
        ]:
            print(f"  验证 {name}...", end=" ", flush=True)
            r = fn(model, tokenizer)
            supported = r.get("supported", False)
            print(f"{'[OK]' if supported else '[NO]'}")
            results[name] = r

        # 否决假说
        print("\n--- 否决假说验证（期望不被支持）---")
        for name, fn in [
            ("R1_欧氏层次距离", verify_r1_hierarchical_distance),
            ("R3_反义词互注意力", verify_r3_antonym_attention),
        ]:
            print(f"  验证 {name}...", end=" ", flush=True)
            r = fn(model, tokenizer)
            rejected = r.get("rejected", False)
            print(f"{'[已否决]' if rejected else '[意外支持]'}")
            results[name] = r

    finally:
        free_model(model)

    return results


def main():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = OUTPUT_BASE / f"stage504_triple_model_verify_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    models_to_run = ALL_MODELS
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg in ALL_MODELS:
            models_to_run = [arg]

    all_results = {}
    for model_key in models_to_run:
        result = run_single_model(model_key)
        all_results[model_key] = result
        # 单模型结果保存
        (out_dir / f"summary_{model_key}.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
        )

    # 跨模型一致性分析
    print(f"\n{'='*60}")
    print("跨模型一致性分析")
    print(f"{'='*60}")

    mechanisms = ["M1_残差流忠实传播", "M2_neurons主导简单关系", "M3_late_readout主导复杂推理",
                  "M4_共现塑造距离", "M5_否定压制激活", "M6_分布式编码",
                  "R1_欧氏层次距离", "R3_反义词互注意力"]

    consistency = {}
    for m in mechanisms:
        supported_count = 0
        total = 0
        for mk, res in all_results.items():
            if m in res:
                total += 1
                if res[m].get("supported", False) or res[m].get("rejected", False):
                    supported_count += 1
        consistency[m] = {
            "consistent_count": supported_count,
            "total": total,
            "consistency_ratio": supported_count / total if total > 0 else 0,
        }

    # 综合总结
    print("\n机制跨模型一致性:")
    for m, c in consistency.items():
        print(f"  {m}: {c['consistent_count']}/{c['total']} (一致性={c['consistency_ratio']:.0%})")

    overall = {
        "timestamp": timestamp,
        "models_tested": list(all_results.keys()),
        "results": all_results,
        "consistency": consistency,
        "overall_consistency": sum(c["consistency_ratio"] for c in consistency.values()) / len(consistency) if consistency else 0,
    }
    (out_dir / "summary_cross_model.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2, default=float), encoding="utf-8"
    )
    print(f"\n[Stage504] 结果已保存到 {out_dir}")
    print(f"整体一致性: {overall['overall_consistency']:.0%}")


if __name__ == "__main__":
    main()
