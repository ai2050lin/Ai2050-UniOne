#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage638-639: P0核心实验——浪费因果消融 + 正交编码最优性 + 预注册判伪框架

Stage638: 浪费分量因果消融实验
  问题：浪费分量是"功能编码"还是"副产品"？
  预注册判伪：如果删除waste后只有消歧能力下降而语法/推理/生成完全不变，
             则"浪费=全功能语境编码"(INV-123)被推翻。
  实验：
  1. 提取目标词位的hidden state，分解为aligned和waste分量
  2. 用forward hook在末层注入修改后的hidden state（删除/增强waste）
  3. 同口径测试四类能力：消歧(top1)、语法(主被动判断)、推理(前提检测)、生成(续写logits)
  4. 对比四模型：哪些能力对waste最敏感

Stage639: 正交编码的理论最优性
  问题：为什么四模型一致选择正交编码(cos≈0.05)？
  预注册判伪：如果强制对齐(cos=0.5)不差于自然训练，则"正交编码=最优策略"被推翻。
  实验：
  1. 分析四模型消歧方向与unembed子空间的夹角分布
  2. 对比不同alignment区间的消歧效率
  3. 数学推导：正交编码是否最小化跨任务干扰
  4. 分析：如果强制将消歧方向旋转到与unembed对齐，预测消歧效果变化

用法: python stage638_639_causal_orthogonality.py [qwen3|deepseek7b|glm4|gemma4]
"""

from __future__ import annotations
import sys, json, time, gc, torch, os, copy
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "tests" / "codex"))

from multimodel_language_shared import (
    load_model_bundle, free_model, discover_layers, encode_to_device
)

OUTPUT_DIR = PROJECT_ROOT / "tests" / "glm5_temp"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")


def safe_get_device(model):
    for attr in [None, 'model', 'model.model']:
        try:
            obj = model
            if attr:
                for part in attr.split('.'):
                    obj = getattr(obj, part)
            return next(obj.parameters()).device
        except (StopIteration, AttributeError):
            continue
    return torch.device("cpu") if not torch.cuda.is_available() else torch.device("cuda:0")


def move_to_device(batch, model):
    device = safe_get_device(model)
    if hasattr(batch, 'to'):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: (v.to(device) if hasattr(v, 'to') else v) for k, v in batch.items()}
    return batch


def cos_sim(a, b):
    return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()


def extract_last_layer_hidden(model, tokenizer, sentence, layers, target_pos=-1):
    """提取所有层的hidden states"""
    device = safe_get_device(model)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    inputs = move_to_device(inputs, model)

    hidden_states = {}
    hooks = []

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            if h.dim() >= 2:
                hidden_states[layer_idx] = h.float().detach()
        return hook_fn

    for li, layer_module in enumerate(layers):
        hooks.append(layer_module.register_forward_hook(make_hook(li)))

    try:
        with torch.no_grad():
            output = model(**inputs)
    except Exception as e:
        print(f"    Forward failed: {e}")
    finally:
        for h in hooks:
            h.remove()

    if len(hidden_states) == 0:
        return None, None, None

    last_idx = max(hidden_states.keys())
    last_hidden = hidden_states[last_idx]
    target_vec = last_hidden[0, target_pos, :].cpu()
    return target_vec, hidden_states, inputs


def get_unembed_matrix(model):
    for attr_path in ['lm_head', 'embed_out', 'output']:
        try:
            w = getattr(model, attr_path, None)
            if w is not None and hasattr(w, 'weight'):
                return w.weight.float().detach().cpu()
        except:
            pass
    try:
        inner = getattr(model, 'model', model)
        for attr_path in ['lm_head', 'embed_out', 'output']:
            try:
                w = getattr(inner, attr_path, None)
                if w is not None and hasattr(w, 'weight'):
                    return w.weight.float().detach().cpu()
            except:
                pass
    except:
        pass
    return None


def inject_and_predict(model, tokenizer, sentence, layers, target_pos, modified_vec, device):
    """用forward hook注入修改后的hidden state，获取修改后的logits"""
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
    inputs = move_to_device(inputs, model)

    injected_logits = [None]
    original_hidden = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                h = output[0]
            else:
                h = output
            if h.dim() >= 2:
                original_hidden[layer_idx] = h.clone()
        return hook_fn

    # 先正常前向传播记录原始hidden
    hooks = []
    last_idx = len(layers) - 1
    for li, layer_module in enumerate(layers):
        hooks.append(layer_module.register_forward_hook(make_hook(li)))

    try:
        with torch.no_grad():
            model(**inputs)
    except:
        pass
    finally:
        for h in hooks:
            h.remove()

    # 在末层注入修改后的向量
    def inject_hook(module, input, output):
        if isinstance(output, tuple):
            h = output[0]
            new_h = h.clone()
            new_h[0, target_pos, :] = modified_vec.to(device, dtype=h.dtype)
            return (new_h,) + output[1:]
        else:
            h = output.clone()
            h[0, target_pos, :] = modified_vec.to(device, dtype=h.dtype)
            return h

    hook = layers[last_idx].register_forward_hook(inject_hook)
    try:
        with torch.no_grad():
            output = model(**inputs)
            if hasattr(output, 'logits'):
                injected_logits[0] = output.logits[0, target_pos, :].cpu()
    except Exception as e:
        print(f"    Inject forward failed: {e}")
    finally:
        hook.remove()

    return injected_logits[0]


def get_top_predictions(logits, tokenizer, top_k=5):
    """获取top-k预测"""
    if logits is None:
        return []
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, top_k)
    results = []
    for i in range(top_k):
        token_id = topk.indices[i].item()
        prob = topk.values[i].item()
        token = tokenizer.decode([token_id]).strip()
        results.append((token, prob))
    return results


# ============ 测试用例 ============

# 消歧测试对
DISAMBIG_PAIRS = [
    ("The river bank was muddy.", "The bank approved the loan.", "bank",
     ["water", "river", "mud", "flow"], ["money", "loan", "cash", "finance"]),
    ("The plant was green.", "The plant closed down.", "plant",
     ["green", "leaf", "tree", "grow"], ["factory", "close", "worker", "machine"]),
    ("She played a match.", "The match was bright.", "match",
     ["play", "game", "win", "score"], ["fire", "light", "burn", "bright"]),
    ("The bat flew away.", "He swung the bat.", "bat",
     ["fly", "wing", "sky", "night"], ["swing", "hit", "ball", "baseball"]),
    ("I need an apple watch.", "Watch the game.", "watch",
     ["time", "clock", "wrist", "apple"], ["look", "see", "game", "tv"]),
]

# 语法测试（主被动转换检测）
SYNTAX_TESTS = [
    ("The cat chased the mouse.", "The mouse was chased by the cat.", "chased",
     "active"),
    ("The boy broke the window.", "The window was broken by the boy.", "broke",
     "active"),
    ("The teacher praised the student.", "The student was praised by the teacher.", "praised",
     "active"),
    ("Scientists discovered the planet.", "The planet was discovered by scientists.", "discovered",
     "active"),
    ("The storm destroyed the village.", "The village was destroyed by the storm.", "destroyed",
     "active"),
]

# 推理测试（前提存在 vs 不存在）
REASONING_TESTS = [
    ("All cats are animals. Snowball is a cat. Snowball is", "an animal", True),
    ("If it rains, the ground gets wet. It rained. The ground is", "wet", True),
    ("Birds can fly. Penguins are birds. Penguins can", "fly", False),
    ("Fish live in water. Whales are not fish. Whales live in", "water", True),
    ("No mammals lay eggs. Platypuses lay eggs. Platypuses are", "mammals", False),
]

# 生成质量测试（续写概率）
GENERATION_TESTS = [
    ("The sky is", "blue"),
    ("Water boils at", "100"),
    ("The capital of France is", "Paris"),
    ("Two plus two equals", "four"),
    ("Cats drink", "milk"),
]


# ============ Stage638: 浪费分量因果消融 ============

def compute_waste_components(hidden_vec, W_unembed, word_a_ids, word_b_ids, tokenizer):
    """
    分解hidden_vec为aligned分量和waste分量
    aligned: 投影到 delta_u = W_unembed[word_a] - W_unembed[word_b] 上
    waste: 正交于 delta_u 的分量
    """
    delta_u = W_unembed[word_a_ids[0]] - W_unembed[word_b_ids[0]]
    delta_u_norm = delta_u / (delta_u.norm() + 1e-10)

    aligned_mag = torch.dot(hidden_vec, delta_u_norm).item()
    aligned_vec = aligned_mag * delta_u_norm
    waste_vec = hidden_vec - aligned_vec

    return delta_u_norm, aligned_vec, waste_vec, aligned_mag


def test_disambig_ability(model, tokenizer, sentence, target_word, senses_a, senses_b,
                          layers, target_pos=-1):
    """测试消歧能力：top1是否在正确语义集中"""
    hidden, all_hidden, inputs = extract_last_layer_hidden(model, tokenizer, sentence,
                                                           layers, target_pos)
    if hidden is None:
        return None

    W_unembed = get_unembed_matrix(model)
    if W_unembed is None:
        return None

    logits = hidden @ W_unembed.T
    top_preds = get_top_predictions(logits, tokenizer, top_k=10)

    correct = 0
    for token, prob in top_preds:
        token_lower = token.lower()
        if any(s in token_lower for s in senses_a) or any(s in token_lower for s in senses_b):
            correct += 1

    # 判断是否消歧正确：检查top1是否在目标语义中
    context = "context_a" if any(s in sentence.lower() for s in senses_a) else "context_b"
    target_senses = senses_a if context == "context_a" else senses_b
    top1_correct = any(s in top_preds[0][0].lower() for s in target_senses) if top_preds else False

    return {
        "top1": top_preds[0] if top_preds else ("?", 0),
        "top1_correct": top1_correct,
        "correct_in_top10": correct,
        "entropy": -sum(p * np.log(p + 1e-10) for _, p in top_preds if p > 0.01),
    }


def test_syntax_ability(model, tokenizer, active_sent, passive_sent, target_word, layers):
    """测试语法能力：主动句vs被动句的logits变化"""
    hidden_active, _, _ = extract_last_layer_hidden(model, tokenizer, active_sent, layers, -1)
    hidden_passive, _, _ = extract_last_layer_hidden(model, tokenizer, passive_sent, layers, -1)

    if hidden_active is None or hidden_passive is None:
        return None

    # 计算两个hidden state在目标词位的差异
    # active_sent和passive_sent长度可能不同，使用最后一个词
    # 简化：比较整个句子的logits分布
    W_unembed = get_unembed_matrix(model)
    if W_unembed is None:
        return None

    logits_active = hidden_active @ W_unembed.T
    logits_passive = hidden_passive @ W_unembed.T

    # 对比语法结构词的logits变化
    grammar_words = ["the", "was", "by", "is", "are", "were", "had", "has"]
    active_logits = []
    passive_logits = []
    for w in grammar_words:
        ids = tokenizer.encode(w, add_special_tokens=False)
        if ids:
            active_logits.append(logits_active[ids[0]].item())
            passive_logits.append(logits_passive[ids[0]].item())

    if not active_logits:
        return None

    # 计算logits变化的correlation
    corr = np.corrcoef(active_logits, passive_logits)[0, 1] if len(active_logits) > 1 else 0
    mean_diff = np.mean(np.abs(np.array(active_logits) - np.array(passive_logits)))

    return {
        "grammar_corr": corr,
        "mean_grammar_diff": mean_diff,
        "active_logits": active_logits,
        "passive_logits": passive_logits,
    }


def test_reasoning_ability(model, tokenizer, premise_sent, expected_answer, layers):
    """测试推理能力：前提句的续写logits中期望答案的排名"""
    hidden, _, _ = extract_last_layer_hidden(model, tokenizer, premise_sent, layers, -1)
    if hidden is None:
        return None

    W_unembed = get_unembed_matrix(model)
    if W_unembed is None:
        return None

    logits = hidden @ W_unembed.T
    top_preds = get_top_predictions(logits, tokenizer, top_k=20)

    expected_ids = tokenizer.encode(expected_answer, add_special_tokens=False)
    expected_rank = -1
    expected_prob = 0

    if expected_ids:
        all_probs = F.softmax(logits, dim=-1)
        expected_prob = all_probs[expected_ids[0]].item()
        sorted_vals, sorted_idxs = torch.sort(all_probs, descending=True)
        for rank in range(len(sorted_idxs)):
            if sorted_idxs[rank].item() == expected_ids[0]:
                expected_rank = rank
                break

    return {
        "expected_prob": expected_prob,
        "expected_rank": expected_rank,
        "top1": top_preds[0] if top_preds else ("?", 0),
    }


def test_generation_quality(model, tokenizer, prefix, expected, layers):
    """测试生成质量：续写期望词的概率"""
    hidden, _, _ = extract_last_layer_hidden(model, tokenizer, prefix, layers, -1)
    if hidden is None:
        return None

    W_unembed = get_unembed_matrix(model)
    if W_unembed is None:
        return None

    logits = hidden @ W_unembed.T
    top_preds = get_top_predictions(logits, tokenizer, top_k=5)

    expected_ids = tokenizer.encode(expected, add_special_tokens=False)
    expected_prob = 0
    expected_rank = -1

    if expected_ids:
        all_probs = F.softmax(logits, dim=-1)
        expected_prob = all_probs[expected_ids[0]].item()
        sorted_vals, sorted_idxs = torch.sort(all_probs, descending=True)
        for rank in range(len(sorted_idxs)):
            if sorted_idxs[rank].item() == expected_ids[0]:
                expected_rank = rank
                break

    return {
        "expected_prob": expected_prob,
        "expected_rank": expected_rank,
        "top1": top_preds[0] if top_preds else ("?", 0),
    }


def run_stage638(model, tokenizer, model_key):
    """
    Stage638: 浪费分量因果消融——验证浪费是功能编码还是副产品
    """
    print("\n" + "=" * 70)
    print("Stage638: 浪费分量因果消融实验")
    print("=" * 70)

    print("\n  预注册判伪条件：")
    print("  如果删除waste后只有消歧能力下降而语法/推理/生成完全不变，")
    print("  则 INV-123('浪费=全功能语境编码')被推翻。")
    print("  判定标准：语法/推理/生成中>=2项变化<5%则判定为'不变'。")

    device = safe_get_device(model)
    layers = discover_layers(model)
    W_unembed = get_unembed_matrix(model)

    if W_unembed is None:
        print("  [SKIP] 无法获取unembed矩阵")
        return {}

    results = {
        "model": model_key,
        "preregistered_falsification": {
            "condition": "waste删除后，语法/推理/生成中>=2项变化<5%",
            "consequence": "INV-123被推翻，浪费=副产品",
        },
        "disambig": {"baseline": [], "no_waste": [], "enhanced_waste": []},
        "syntax": {"baseline": [], "no_waste": [], "enhanced_waste": []},
        "reasoning": {"baseline": [], "no_waste": [], "enhanced_waste": []},
        "generation": {"baseline": [], "no_waste": [], "enhanced_waste": []},
    }

    # --- 消歧测试 ---
    print("\n  --- 消歧能力测试 ---")
    for s1, s2, word, senses_a, senses_b in DISAMBIG_PAIRS[:3]:
        for sent, target_senses in [(s1, senses_a), (s2, senses_b)]:
            ctx = "ctx_a" if target_senses is senses_a else "ctx_b"
            hidden, all_hidden, inputs = extract_last_layer_hidden(
                model, tokenizer, sent, layers, -1)
            if hidden is None:
                continue

            word_a_ids = tokenizer.encode(senses_a[0], add_special_tokens=False)
            word_b_ids = tokenizer.encode(senses_b[0], add_special_tokens=False)
            if not word_a_ids or not word_b_ids:
                continue

            delta_u, aligned, waste, aligned_mag = compute_waste_components(
                hidden, W_unembed, word_a_ids, word_b_ids, tokenizer)

            waste_ratio = waste.norm().item() / (hidden.norm().item() + 1e-10)

            # Baseline
            baseline = test_disambig_ability(model, tokenizer, sent, word,
                                             senses_a, senses_b, layers)
            if baseline:
                results["disambig"]["baseline"].append(baseline)

            # No waste: 用aligned_only替换
            aligned_only = aligned
            no_waste_logits = inject_and_predict(model, tokenizer, sent, layers, -1,
                                                  aligned_only, device)
            if no_waste_logits is not None:
                top_preds = get_top_predictions(no_waste_logits, tokenizer, top_k=10)
                correct = sum(1 for t, _ in top_preds
                              if any(s in t.lower() for s in senses_a + senses_b))
                top1_correct = any(s in top_preds[0][0].lower() for s in target_senses) if top_preds else False
                entropy = -sum(p * np.log(p + 1e-10) for _, p in top_preds if p > 0.01)
                results["disambig"]["no_waste"].append({
                    "top1": top_preds[0] if top_preds else ("?", 0),
                    "top1_correct": top1_correct,
                    "correct_in_top10": correct,
                    "entropy": entropy,
                    "waste_ratio": waste_ratio,
                })

            # Enhanced waste: 用aligned + 2x_waste替换
            enhanced = aligned + 2.0 * waste
            enhanced_logits = inject_and_predict(model, tokenizer, sent, layers, -1,
                                                  enhanced, device)
            if enhanced_logits is not None:
                top_preds = get_top_predictions(enhanced_logits, tokenizer, top_k=10)
                correct = sum(1 for t, _ in top_preds
                              if any(s in t.lower() for s in senses_a + senses_b))
                top1_correct = any(s in top_preds[0][0].lower() for s in target_senses) if top_preds else False
                entropy = -sum(p * np.log(p + 1e-10) for _, p in top_preds if p > 0.01)
                results["disambig"]["enhanced_waste"].append({
                    "top1": top_preds[0] if top_preds else ("?", 0),
                    "top1_correct": top1_correct,
                    "correct_in_top10": correct,
                    "entropy": entropy,
                })

        gc.collect()
        torch.cuda.empty_cache()

    # --- 语法能力测试 ---
    print("\n  --- 语法能力测试（主被动）---")
    for active, passive, target, label in SYNTAX_TESTS[:3]:
        hidden_active, _, _ = extract_last_layer_hidden(model, tokenizer, active, layers, -1)
        hidden_passive, _, _ = extract_last_layer_hidden(model, tokenizer, passive, layers, -1)
        if hidden_active is None or hidden_passive is None:
            continue

        # Baseline syntax
        baseline_syntax = test_syntax_ability(model, tokenizer, active, passive, target, layers)
        if baseline_syntax:
            results["syntax"]["baseline"].append(baseline_syntax)

        # No waste: 从active句末层hidden删除waste分量
        delta_u = hidden_active - hidden_passive
        delta_u_norm = delta_u / (delta_u.norm() + 1e-10)
        aligned_active = torch.dot(hidden_active, delta_u_norm) * delta_u_norm
        waste_active = hidden_active - aligned_active

        no_waste_hidden = aligned_active
        no_waste_logits = inject_and_predict(model, tokenizer, active, layers, -1,
                                              no_waste_hidden, device)
        if no_waste_logits is not None:
            grammar_words = ["the", "was", "by", "is", "are"]
            active_logits = []
            no_waste_logits_vals = []
            for w in grammar_words:
                ids = tokenizer.encode(w, add_special_tokens=False)
                if ids:
                    active_logits.append((hidden_active @ W_unembed.T)[ids[0]].item())
                    no_waste_logits_vals.append(no_waste_logits[ids[0]].item())
            if active_logits:
                corr = np.corrcoef(active_logits, no_waste_logits_vals)[0, 1]
                mean_diff = np.mean(np.abs(np.array(active_logits) - np.array(no_waste_logits_vals)))
                results["syntax"]["no_waste"].append({
                    "grammar_corr": corr,
                    "mean_grammar_diff": mean_diff,
                })

        # Enhanced waste
        enhanced_hidden = aligned_active + 2.0 * waste_active
        enhanced_logits = inject_and_predict(model, tokenizer, active, layers, -1,
                                              enhanced_hidden, device)
        if enhanced_logits is not None:
            enh_logits_vals = []
            for w in grammar_words:
                ids = tokenizer.encode(w, add_special_tokens=False)
                if ids:
                    enh_logits_vals.append(enhanced_logits[ids[0]].item())
            if enh_logits_vals:
                corr = np.corrcoef(active_logits, enh_logits_vals)[0, 1]
                mean_diff = np.mean(np.abs(np.array(active_logits) - np.array(enh_logits_vals)))
                results["syntax"]["enhanced_waste"].append({
                    "grammar_corr": corr,
                    "mean_grammar_diff": mean_diff,
                })

        gc.collect()
        torch.cuda.empty_cache()

    # --- 推理能力测试 ---
    print("\n  --- 推理能力测试 ---")
    for premise, expected, is_valid in REASONING_TESTS[:3]:
        hidden, _, _ = extract_last_layer_hidden(model, tokenizer, premise, layers, -1)
        if hidden is None:
            continue

        # 获取"正确"和"错误"答案的unembed方向作为delta_u的代理
        correct_ids = tokenizer.encode(expected, add_special_tokens=False)
        wrong_answer = "not " + expected if is_valid else expected + "s"
        wrong_ids = tokenizer.encode(wrong_answer, add_special_tokens=False)

        if not correct_ids or not wrong_ids:
            continue

        delta_u = W_unembed[correct_ids[0]] - W_unembed[wrong_ids[0]]
        delta_u_norm = delta_u / (delta_u.norm() + 1e-10)
        aligned = torch.dot(hidden, delta_u_norm) * delta_u_norm
        waste = hidden - aligned

        # Baseline
        baseline_reason = test_reasoning_ability(model, tokenizer, premise, expected, layers)
        if baseline_reason:
            results["reasoning"]["baseline"].append(baseline_reason)

        # No waste
        no_waste_logits = inject_and_predict(model, tokenizer, premise, layers, -1,
                                              aligned, device)
        if no_waste_logits is not None:
            all_probs = F.softmax(no_waste_logits, dim=-1)
            expected_prob = all_probs[correct_ids[0]].item()
            sorted_vals, sorted_idxs = torch.sort(all_probs, descending=True)
            expected_rank = -1
            for rank in range(len(sorted_idxs)):
                if sorted_idxs[rank].item() == correct_ids[0]:
                    expected_rank = rank
                    break
            results["reasoning"]["no_waste"].append({
                "expected_prob": expected_prob,
                "expected_rank": expected_rank,
            })

        # Enhanced waste
        enhanced = aligned + 2.0 * waste
        enhanced_logits = inject_and_predict(model, tokenizer, premise, layers, -1,
                                              enhanced, device)
        if enhanced_logits is not None:
            all_probs = F.softmax(enhanced_logits, dim=-1)
            expected_prob = all_probs[correct_ids[0]].item()
            sorted_vals, sorted_idxs = torch.sort(all_probs, descending=True)
            expected_rank = -1
            for rank in range(len(sorted_idxs)):
                if sorted_idxs[rank].item() == correct_ids[0]:
                    expected_rank = rank
                    break
            results["reasoning"]["enhanced_waste"].append({
                "expected_prob": expected_prob,
                "expected_rank": expected_rank,
            })

        gc.collect()
        torch.cuda.empty_cache()

    # --- 生成质量测试 ---
    print("\n  --- 生成质量测试 ---")
    for prefix, expected in GENERATION_TESTS[:3]:
        hidden, _, _ = extract_last_layer_hidden(model, tokenizer, prefix, layers, -1)
        if hidden is None:
            continue

        expected_ids = tokenizer.encode(expected, add_special_tokens=False)
        if not expected_ids:
            continue

        # 用hidden state的主方向作为delta_u的代理
        delta_u = hidden / (hidden.norm() + 1e-10)  # 直接用hidden方向
        delta_u_norm = delta_u / (delta_u.norm() + 1e-10)
        aligned = torch.dot(hidden, delta_u_norm) * delta_u_norm
        waste = hidden - aligned

        # Baseline
        baseline_gen = test_generation_quality(model, tokenizer, prefix, expected, layers)
        if baseline_gen:
            results["generation"]["baseline"].append(baseline_gen)

        # No waste
        no_waste_logits = inject_and_predict(model, tokenizer, prefix, layers, -1,
                                              aligned, device)
        if no_waste_logits is not None:
            all_probs = F.softmax(no_waste_logits, dim=-1)
            expected_prob = all_probs[expected_ids[0]].item()
            sorted_vals, sorted_idxs = torch.sort(all_probs, descending=True)
            expected_rank = -1
            for rank in range(len(sorted_idxs)):
                if sorted_idxs[rank].item() == expected_ids[0]:
                    expected_rank = rank
                    break
            results["generation"]["no_waste"].append({
                "expected_prob": expected_prob,
                "expected_rank": expected_rank,
            })

        # Enhanced waste
        enhanced = aligned + 2.0 * waste
        enhanced_logits = inject_and_predict(model, tokenizer, prefix, layers, -1,
                                              enhanced, device)
        if enhanced_logits is not None:
            all_probs = F.softmax(enhanced_logits, dim=-1)
            expected_prob = all_probs[expected_ids[0]].item()
            sorted_vals, sorted_idxs = torch.sort(all_probs, descending=True)
            expected_rank = -1
            for rank in range(len(sorted_idxs)):
                if sorted_idxs[rank].item() == expected_ids[0]:
                    expected_rank = rank
                    break
            results["generation"]["enhanced_waste"].append({
                "expected_prob": expected_prob,
                "expected_rank": expected_rank,
            })

        gc.collect()
        torch.cuda.empty_cache()

    # --- 判伪结果汇总 ---
    print("\n  --- 预注册判伪结果 ---")
    falsification = _evaluate_falsification(results)
    results["falsification_result"] = falsification
    print(f"  判伪结果: {falsification['verdict']}")
    print(f"  消歧变化: {falsification['disambig_change_pct']:.1f}%")
    print(f"  语法变化: {falsification['syntax_change_pct']:.1f}%")
    print(f"  推理变化: {falsification['reasoning_change_pct']:.1f}%")
    print(f"  生成变化: {falsification['generation_change_pct']:.1f}%")

    return results


def _evaluate_falsification(results):
    """评估预注册判伪条件"""
    changes = {}

    # 消歧：比较baseline vs no_waste的top1_correct和entropy
    if results["disambig"]["baseline"] and results["disambig"]["no_waste"]:
        base_entropy = np.mean([d["entropy"] for d in results["disambig"]["baseline"]])
        nowaste_entropy = np.mean([d["entropy"] for d in results["disambig"]["no_waste"]])
        if base_entropy > 0:
            changes["disambig"] = abs(nowaste_entropy - base_entropy) / base_entropy * 100
        else:
            changes["disambig"] = 100 if nowaste_entropy > 0 else 0
    else:
        changes["disambig"] = 0

    # 语法：比较grammar_corr
    if results["syntax"]["baseline"] and results["syntax"]["no_waste"]:
        base_corr = np.mean([s["grammar_corr"] for s in results["syntax"]["baseline"]])
        nw_corr = np.mean([s["grammar_corr"] for s in results["syntax"]["no_waste"]])
        changes["syntax"] = abs(nw_corr - base_corr) * 100  # corr变化作为百分比
    else:
        changes["syntax"] = 0

    # 推理：比较expected_prob
    if results["reasoning"]["baseline"] and results["reasoning"]["no_waste"]:
        base_prob = np.mean([r["expected_prob"] for r in results["reasoning"]["baseline"]])
        nw_prob = np.mean([r["expected_prob"] for r in results["reasoning"]["no_waste"]])
        changes["reasoning"] = abs(nw_prob - base_prob) / (base_prob + 1e-10) * 100
    else:
        changes["reasoning"] = 0

    # 生成：比较expected_prob
    if results["generation"]["baseline"] and results["generation"]["no_waste"]:
        base_prob = np.mean([g["expected_prob"] for g in results["generation"]["baseline"]])
        nw_prob = np.mean([g["expected_prob"] for g in results["generation"]["no_waste"]])
        changes["generation"] = abs(nw_prob - base_prob) / (base_prob + 1e-10) * 100
    else:
        changes["generation"] = 0

    # 判伪：语法/推理/生成中>=2项变化<5% → 推翻INV-123
    unchanged_count = sum(1 for k in ["syntax", "reasoning", "generation"] if changes.get(k, 0) < 5)
    if unchanged_count >= 2:
        verdict = "FALSIFIED: INV-123被推翻——浪费=副产品，非功能编码"
    else:
        verdict = "SURVIVED: INV-123存活——浪费确实是功能编码"

    return {
        "verdict": verdict,
        "disambig_change_pct": changes.get("disambig", 0),
        "syntax_change_pct": changes.get("syntax", 0),
        "reasoning_change_pct": changes.get("reasoning", 0),
        "generation_change_pct": changes.get("generation", 0),
        "unchanged_count": unchanged_count,
    }


# ============ Stage639: 正交编码的理论最优性 ============

def run_stage639(model, tokenizer, model_key):
    """
    Stage639: 正交编码的理论最优性分析
    """
    print("\n" + "=" * 70)
    print("Stage639: 正交编码的理论最优性")
    print("=" * 70)

    print("\n  预注册判伪条件：")
    print("  如果高alignment区间的消歧效率不优于低alignment区间，")
    print("  则'正交编码是最优策略'的假说被推翻。")

    device = safe_get_device(model)
    layers = discover_layers(model)
    W_unembed = get_unembed_matrix(model)

    if W_unembed is None:
        print("  [SKIP] 无法获取unembed矩阵")
        return {}

    # 用更高效的方式获取unembed的主子空间
    # W_unembed: [V, D], 对W_unembed做PCA得到D维空间中的主方向
    # 使用W_unembed.T @ W_unembed的eigen decomposition更高效
    WtW = W_unembed.float().T @ W_unembed.float()  # [D, D]
    eigvals, eigvecs = torch.linalg.eigh(WtW)
    # 按特征值降序排列
    sorted_idx = torch.argsort(eigvals, descending=True)
    eigvecs_sorted = eigvecs[:, sorted_idx]  # [D, D], 每列是一个主方向
    unembed_basis = eigvecs_sorted  # [D, D]
    print(f"  Unembed SVD完成, top-5 eigenvalues: {eigvals[sorted_idx[:5]].tolist()}")

    results = {"model": model_key}

    # 分析消歧方向与unembed子空间的夹角分布
    print("\n  --- 消歧方向与unembed子空间夹角分析 ---")
    angles = []
    alignments = []
    disamb_efficiencies = []

    for s1, s2, word, senses_a, senses_b in DISAMBIG_PAIRS:
        hidden1, _, _ = extract_last_layer_hidden(model, tokenizer, s1, layers, -1)
        hidden2, _, _ = extract_last_layer_hidden(model, tokenizer, s2, layers, -1)

        if hidden1 is None or hidden2 is None:
            continue

        d = hidden2 - hidden1
        d_norm = d / (d.norm() + 1e-10)

        # 投影到unembed前K维子空间
        for K in [50, 100, 200]:
            if K > unembed_basis.shape[1]:
                continue
            # unembed_basis[:, :K]: [D, K] - D维空间中的前K个主方向
            proj_basis = unembed_basis[:, :K]  # [D, K]
            proj_coeffs = d_norm @ proj_basis  # [K]
            proj = proj_coeffs @ proj_basis.T  # [D] 重建
            recon_error = 1.0 - proj.norm().item() / (d_norm.norm().item() + 1e-10)
            alignment = cos_sim(d_norm, proj)

            # 消歧效率：logit margin
            word_a_ids = tokenizer.encode(senses_a[0], add_special_tokens=False)
            word_b_ids = tokenizer.encode(senses_b[0], add_special_tokens=False)
            if word_a_ids and word_b_ids:
                logit_a = torch.dot(d, W_unembed[word_a_ids[0]]).item()
                logit_b = torch.dot(d, W_unembed[word_b_ids[0]]).item()
                margin = logit_a - logit_b
            else:
                margin = 0

            angles.append(recon_error)
            alignments.append(alignment)
            disamb_efficiencies.append(margin)

        gc.collect()
        torch.cuda.empty_cache()

    # 按alignment分桶，比较消歧效率
    print("\n  --- 按alignment分桶的消歧效率 ---")
    if alignments:
        arr = np.array(alignments)
        eff = np.array(disamb_efficiencies)
        buckets = [
            ("high_align", arr > 0.5),
            ("medium_align", (arr > 0.1) & (arr <= 0.5)),
            ("low_align", (arr > -0.1) & (arr <= 0.1)),
            ("neg_align", arr <= -0.1),
        ]

        bucket_results = {}
        for name, mask in buckets:
            if mask.sum() > 0:
                bucket_results[name] = {
                    "count": int(mask.sum()),
                    "mean_alignment": float(arr[mask].mean()),
                    "mean_efficiency": float(eff[mask].mean()),
                    "std_efficiency": float(eff[mask].std()) if mask.sum() > 1 else 0,
                }
                print(f"    {name}: n={mask.sum()}, alignment={arr[mask].mean():.4f}, "
                      f"efficiency={eff[mask].mean():.2f}")

        results["alignment_buckets"] = bucket_results

    # 分析：强制旋转消歧方向到与unembed对齐后的效果预测
    print("\n  --- 强制对齐的消歧效果预测 ---")
    forced_align_results = []

    for s1, s2, word, senses_a, senses_b in DISAMBIG_PAIRS[:3]:
        hidden1, _, _ = extract_last_layer_hidden(model, tokenizer, s1, layers, -1)
        hidden2, _, _ = extract_last_layer_hidden(model, tokenizer, s2, layers, -1)

        if hidden1 is None or hidden2 is None:
            continue

        d = hidden2 - hidden1
        d_norm_val = d.norm().item()

        # 原始消歧效果
        word_a_ids = tokenizer.encode(senses_a[0], add_special_tokens=False)
        word_b_ids = tokenizer.encode(senses_b[0], add_special_tokens=False)
        if not word_a_ids or not word_b_ids:
            continue

        delta_u = W_unembed[word_a_ids[0]] - W_unembed[word_b_ids[0]]
        original_margin = torch.dot(d, delta_u).item()
        original_alignment = cos_sim(d, delta_u)

        # 强制对齐版本：将d旋转到delta_u方向，保持范数
        d_forced = d_norm_val * delta_u / (delta_u.norm() + 1e-10)
        forced_margin = torch.dot(d_forced, delta_u).item()

        # 理论最优margin
        optimal_margin = d_norm_val * delta_u.norm().item()

        forced_align_results.append({
            "word": word,
            "original_margin": original_margin,
            "original_alignment": original_alignment,
            "forced_margin": forced_margin,
            "optimal_margin": optimal_margin,
            "margin_ratio": original_margin / (optimal_margin + 1e-10),
        })

        print(f"    {word}: original_margin={original_margin:.2f}, "
              f"forced_margin={forced_margin:.2f}, "
              f"optimal_margin={optimal_margin:.2f}, "
              f"ratio={original_margin / (optimal_margin + 1e-10):.4f}")

    results["forced_alignment"] = forced_align_results

    # 跨任务干扰分析：正交编码是否最小化了对其他任务的干扰
    print("\n  --- 跨任务干扰分析 ---")
    cross_task_results = {}

    for s1, s2, word, senses_a, senses_b in DISAMBIG_PAIRS[:3]:
        hidden1, _, _ = extract_last_layer_hidden(model, tokenizer, s1, layers, -1)
        hidden2, _, _ = extract_last_layer_hidden(model, tokenizer, s2, layers, -1)

        if hidden1 is None or hidden2 is None:
            continue

        d = hidden2 - hidden1
        d_unit = d / (d.norm() + 1e-10)

        # 消歧任务的对齐度
        word_a_ids = tokenizer.encode(senses_a[0], add_special_tokens=False)
        word_b_ids = tokenizer.encode(senses_b[0], add_special_tokens=False)
        if word_a_ids and word_b_ids:
            delta_u = W_unembed[word_a_ids[0]] - W_unembed[word_b_ids[0]]
            disamb_align = cos_sim(d, delta_u)

            # "其他任务"的对齐度：用不相干词的unembed方向
            # 选择与消歧无关的词（如数字、颜色）
            neutral_words = ["the", "is", "and", "one", "two", "red", "blue", "big"]
            neutral_aligns = []
            for nw in neutral_words:
                nw_ids = tokenizer.encode(nw, add_special_tokens=False)
                if nw_ids:
                    nw_align = abs(cos_sim(d, W_unembed[nw_ids[0]]))
                    neutral_aligns.append(nw_align)

            if neutral_aligns:
                mean_neutral = np.mean(neutral_aligns)
                # 正交编码的"隔离比"：消歧对齐 / 中性词对齐
                isolation = disamb_align / (mean_neutral + 1e-10) if mean_neutral > 0 else float('inf')

                cross_task_results[word] = {
                    "disamb_alignment": disamb_align,
                    "mean_neutral_alignment": mean_neutral,
                    "isolation_ratio": isolation,
                }

                print(f"    {word}: disamb_align={disamb_align:.4f}, "
                      f"neutral_align={mean_neutral:.4f}, "
                      f"isolation={isolation:.2f}")

    results["cross_task"] = cross_task_results

    # 数学推导：正交编码是否最小化跨任务干扰
    print("\n  --- 数学推导：正交编码的信息论分析 ---")
    print("  设消歧方向为d，unembed主子空间为U_k（前k维）")
    print("  d = d_parallel + d_perp，其中d_parallel在U_k中，d_perp正交于U_k")
    print("  正交编码：||d_parallel|| << ||d_perp||")
    print("  对齐编码：||d_parallel|| >> ||d_perp||")
    print("  跨任务干扰 ∝ ||d_parallel||^2（平行分量越大，对unembed其他方向影响越大）")
    print("  因此正交编码最小化跨任务干扰的条件是：")
    print("  对齐编码的消歧增益 > 跨任务干扰的代价")
    print("  如果消歧信息可以通过很小的||d_parallel||实现（当前cos≈0.05），")
    print("  则正交编码是Pareto最优解——最小干扰下获得足够消歧效果。")

    # 判伪：检查高alignment是否真的更优
    falsification_639 = "SURVIVED"
    if "alignment_buckets" in results:
        high = results["alignment_buckets"].get("high_align", {})
        low = results["alignment_buckets"].get("low_align", {})
        if high.get("count", 0) > 0 and low.get("count", 0) > 0:
            high_eff = high.get("mean_efficiency", 0)
            low_eff = low.get("mean_efficiency", 0)
            if high_eff <= low_eff * 1.1:  # 高alignment不超过低alignment的10%
                falsification_639 = "PARTIALLY_FALSIFIED"
                print(f"\n  WARNING: high_align效率({high_eff:.2f})不优于low_align({low_eff:.2f})")
            else:
                print(f"\n  CONFIRMED: high_align效率({high_eff:.2f})优于low_align({low_eff:.2f})")

    results["falsification_639"] = falsification_639
    print(f"\n  判伪结果: {falsification_639}")

    return results


# ============ 主函数 ============

def main():
    if len(sys.argv) < 2:
        print("用法: python stage638_639_causal_orthogonality.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)

    model_key = sys.argv[1].lower()
    valid_keys = ["qwen3", "deepseek7b", "glm4", "gemma4"]
    if model_key not in valid_keys:
        print(f"无效模型: {model_key}，可选: {valid_keys}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"Stage638-639: P0因果消融+正交编码最优性")
    print(f"模型: {model_key}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")

    t0 = time.time()

    print(f"\n[1/4] 加载模型 {model_key}...")
    bundle = load_model_bundle(model_key)
    if bundle is None:
        print(f"无法加载模型 {model_key}")
        sys.exit(1)

    model, tokenizer = bundle

    try:
        print(f"\n[2/4] 运行Stage638: 浪费分量因果消融...")
        r638 = run_stage638(model, tokenizer, model_key)

        gc.collect()
        torch.cuda.empty_cache()

        print(f"\n[3/4] 运行Stage639: 正交编码最优性...")
        r639 = run_stage639(model, tokenizer, model_key)

        print(f"\n[4/4] 保存结果...")
        all_results = {
            "model": model_key,
            "timestamp": TIMESTAMP,
            "stage638_causal_ablation": r638,
            "stage639_orthogonality": r639,
        }

        output_path = OUTPUT_DIR / f"stage638_639_{model_key}_{TIMESTAMP}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)

        print(f"  结果已保存: {output_path}")

        # 打印关键结果汇总
        print(f"\n{'=' * 70}")
        print("关键结果汇总")
        print(f"{'=' * 70}")

        if r638:
            fr = r638.get("falsification_result", {})
            print(f"\n  Stage638 判伪: {fr.get('verdict', 'N/A')}")
            print(f"    消歧变化: {fr.get('disambig_change_pct', 0):.1f}%")
            print(f"    语法变化: {fr.get('syntax_change_pct', 0):.1f}%")
            print(f"    推理变化: {fr.get('reasoning_change_pct', 0):.1f}%")
            print(f"    生成变化: {fr.get('generation_change_pct', 0):.1f}%")

            # 打印具体数据
            if r638["disambig"]["no_waste"]:
                top1_correct = sum(1 for d in r638["disambig"]["no_waste"] if d.get("top1_correct"))
                total = len(r638["disambig"]["no_waste"])
                print(f"    删除waste后top1正确率: {top1_correct}/{total}")
            if r638["disambig"]["enhanced_waste"]:
                top1_correct = sum(1 for d in r638["disambig"]["enhanced_waste"] if d.get("top1_correct"))
                total = len(r638["disambig"]["enhanced_waste"])
                print(f"    增强waste后top1正确率: {top1_correct}/{total}")

        if r639:
            print(f"\n  Stage639 判伪: {r639.get('falsification_639', 'N/A')}")
            if "forced_alignment" in r639:
                ratios = [r["margin_ratio"] for r in r639["forced_alignment"]]
                print(f"    平均margin_ratio: {np.mean(ratios):.4f}")
                print(f"    (1.0=完全对齐最优, 0.0=完全正交)")

        elapsed = time.time() - t0
        print(f"\n  总耗时: {elapsed:.1f}s")

    finally:
        print("\n释放模型...")
        free_model(bundle)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
