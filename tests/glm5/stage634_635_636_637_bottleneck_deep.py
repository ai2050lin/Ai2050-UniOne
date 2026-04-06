#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage634-635-636-637: 四大瓶颈深度分析

Stage634: DS7B消歧方向反向分析
  问题：DS7B的logit margin为负，消歧方向与代表词方向相反
  实验：
  - 分析DS7B的unembed矩阵结构：奇异值谱、代表词方向投影
  - 找到DS7B实际使用什么方向消歧（不是代表词方向，那是什么？）
  - 比较DS7B与其他三模型的unembed结构差异
  - 假说：DS7B可能用"反义词方向"或"上下文-字面方向"消歧

Stage635: "浪费能量"功能定位
  问题：92-97%能量与unembed读出方向正交，这些能量编码了什么？
  实验：
  - 将92%的"浪费分量"投影到语义探针/语法探针/推理探针
  - 测试强制删除"浪费分量"后模型其他能力的变化
  - 量化"浪费分量"中各功能（语法/知识/风格）的比例

Stage636: 非消歧维度功能分解
  问题：消歧仅占3-19维，其余2500+维编码了什么？
  实验：
  - 对完整hidden state做SVD，分解为"消歧子空间"和"非消歧子空间"
  - 在非消歧子空间中搜索：是否承载语法/知识/频率/位置信息
  - 用探针向量（语法探针、知识探针等）扫描非消歧维度

Stage637: Gemma4 d_norm不增长归因
  问题：Gemma4的d_norm增长仅1.1x（其他模型18-124x）
  实验：
  - 逐层分析Gemma4的delta_k范数（是每层增量太小，还是有正负抵消）
  - 分析Gemma4的层归一化（LayerNorm）对消歧方向的压缩效果
  - 对比Gemma4 vs Qwen3的per-layer rotation angle和norm growth
  - 测试：去掉LN后，d_norm是否会增长

用法: python stage634_635_636_637_bottleneck_deep.py [qwen3|deepseek7b|glm4|gemma4]
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


def extract_all_layer_hidden(model, tokenizer, sentence, layers, target_pos=-1):
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
                hidden_states[layer_idx] = h[0, target_pos, :].float().detach().cpu()
        return hook_fn

    for li, layer_module in enumerate(layers):
        hooks.append(layer_module.register_forward_hook(make_hook(li)))

    try:
        with torch.no_grad():
            model(**inputs)
    except Exception as e:
        print(f"    Forward failed: {e}")
    finally:
        for h in hooks:
            h.remove()

    if len(hidden_states) == 0:
        return None
    return hidden_states


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


def get_embed_matrix(model):
    """Get embedding matrix"""
    try:
        inner = getattr(model, 'model', model)
        for attr_path in ['embed_tokens', 'wte', 'word_embeddings']:
            try:
                w = getattr(inner, attr_path, None)
                if w is not None and hasattr(w, 'weight'):
                    return w.weight.float().detach().cpu()
            except:
                pass
    except:
        pass
    return None


# ============ 歧义词对（5+20）============
CORE_PAIRS = [
    ("The river bank was muddy.", "The bank approved the loan.", "bank",
     ["water", "river", "mud", "flow"], ["money", "loan", "cash", "finance"]),
    ("The plant was green.", "The plant closed down.", "plant",
     ["green", "leaf", "tree", "grow"], ["factory", "close", "worker", "machine"]),
    ("The bat flew away.", "He swung the bat.", "bat",
     ["fly", "wing", "sky", "night"], ["swing", "hit", "ball", "baseball"]),
    ("I need an apple watch.", "Watch the game.", "watch",
     ["time", "clock", "wrist", "apple"], ["look", "see", "game", "tv"]),
    ("She played a match.", "The match was bright.", "match",
     ["play", "game", "win", "score"], ["fire", "light", "burn", "bright"]),
]

EXTENDED_PAIRS = [
    ("The river bank was muddy.", "The bank approved the loan.", "bank",
     ["water", "river", "mud", "flow"], ["money", "loan", "cash", "finance"]),
    ("The plant was green.", "The plant closed down.", "plant",
     ["green", "leaf", "tree", "grow"], ["factory", "close", "worker", "machine"]),
    ("The bat flew away.", "He swung the bat.", "bat",
     ["fly", "wing", "sky", "night"], ["swing", "hit", "ball", "baseball"]),
    ("I need an apple watch.", "Watch the game.", "watch",
     ["time", "clock", "wrist", "apple"], ["look", "see", "game", "tv"]),
    ("She played a match.", "The match was bright.", "match",
     ["play", "game", "win", "score"], ["fire", "light", "burn", "bright"]),
    ("The spring was warm.", "The spring bounced back.", "spring",
     ["warm", "season", "flower", "April"], ["bounce", "coil", "jump", "elastic"]),
    ("The bow was tied.", "She took a bow.", "bow",
     ["tie", "ribbon", "knot", "gift"], ["respect", "applaud", "perform", "stage"]),
    ("The light was on.", "The bag was light.", "light",
     ["on", "bright", "lamp", "glow"], ["weight", "heavy", "bag", "carry"]),
    ("The right turn.", "You are right.", "right",
     ["turn", "left", "direction", "road"], ["correct", "yes", "true", "wrong"]),
    ("The rock was hard.", "She loves rock music.", "rock",
     ["hard", "stone", "mountain", "heavy"], ["music", "band", "guitar", "song"]),
    ("The star was bright.", "She is a star.", "star",
     ["bright", "sky", "night", "moon"], ["famous", "celebrity", "movie", "actress"]),
    ("The bar served drinks.", "He passed the bar.", "bar",
     ["drink", "alcohol", "beer", "serve"], ["law", "exam", "lawyer", "legal"]),
    ("The ring was golden.", "Give me a ring.", "ring",
     ["golden", "jewelry", "finger", "diamond"], ["call", "phone", "telephone", "dial"]),
    ("The head was injured.", "She is the head.", "head",
     ["injured", "brain", "face", "hair"], ["leader", "boss", "chief", "manager"]),
    ("The square was large.", "Three is not a square.", "square",
     ["large", "area", "city", "park"], ["math", "number", "root", "shape"]),
    ("The table was set.", "Look at the table.", "table",
     ["set", "food", "dinner", "chair"], ["data", "chart", "list", "number"]),
    ("The pool was deep.", "The pool game ended.", "pool",
     ["deep", "water", "swim", "blue"], ["game", "billiard", "cue", "eight"]),
    ("The saw cut wood.", "I saw him yesterday.", "saw",
     ["cut", "wood", "tool", "sharp"], ["see", "yesterday", "look", "watch"]),
    ("The fair was fun.", "That is not fair.", "fair",
     ["fun", "carnival", "ride", "game"], ["just", "equal", "right", "honest"]),
    ("The suit was expensive.", "That suit you.", "suit",
     ["expensive", "wear", "formal", "tie"], ["fit", "match", "appropriate", "work"]),
    ("The band played music.", "The rubber band broke.", "band",
     ["music", "play", "guitar", "sing"], ["rubber", "stretch", "elastic", "break"]),
]


# ============ 探针词组 ============
SEMANTIC_PROBES = {
    "animal": ["cat", "dog", "bird", "fish", "horse"],
    "food": ["apple", "bread", "rice", "meat", "cake"],
    "tool": ["hammer", "knife", "saw", "drill", "wrench"],
    "color": ["red", "blue", "green", "yellow", "white"],
    "emotion": ["happy", "sad", "angry", "fear", "love"],
    "action": ["run", "eat", "sleep", "walk", "think"],
    "location": ["house", "city", "road", "river", "mountain"],
    "time": ["today", "yesterday", "morning", "night", "hour"],
    "quantity": ["one", "two", "many", "few", "all"],
    "abstract": ["love", "truth", "beauty", "justice", "freedom"],
}

SYNTAX_PROBES = {
    "determiner": ["the", "a", "an", "this", "that"],
    "preposition": ["in", "on", "at", "to", "from"],
    "conjunction": ["and", "but", "or", "because", "although"],
    "pronoun": ["he", "she", "it", "they", "we"],
    "auxiliary": ["is", "was", "has", "will", "can"],
    "negation": ["not", "never", "no", "none", "without"],
}


# ============ Stage634: DS7B消歧方向反向分析 ============

def run_stage634(model, tokenizer, model_key):
    """DS7B消歧方向反向分析——为什么消歧方向与代表词方向相反？"""
    print("\n" + "=" * 70)
    print("Stage634: 消歧方向反向分析 (所有模型)")
    print("=" * 70)

    device = safe_get_device(model)
    layers = discover_layers(model)
    W_unembed = get_unembed_matrix(model)

    if W_unembed is None:
        print("  [SKIP] 无法获取unembed矩阵")
        return {}

    V, D = W_unembed.shape
    print(f"  Unembed shape: [{V}, {D}]")

    # 1. Unembed矩阵全局结构分析
    U_u, S_u, Vt_u = torch.linalg.svd(W_unembed, full_matrices=False)
    print(f"  Unembed SVD: top10 singular values = {S_u[:10].tolist()[:5]}...")
    print(f"  Unembed top5 singular ratio: {(S_u[:5].sum() / S_u.sum()).item():.4f}")

    # 2. 分析每个歧义词的消歧方向
    results = {}
    all_word_results = []

    for idx, (sA, sB, word, meaningA_tokens, meaningB_tokens) in enumerate(CORE_PAIRS):
        print(f"\n  [{idx+1}/5] word: {word}")

        meaningA_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningA_tokens]
        meaningB_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningB_tokens]

        hiddens_A = extract_all_layer_hidden(model, tokenizer, sA, layers)
        hiddens_B = extract_all_layer_hidden(model, tokenizer, sB, layers)

        if hiddens_A is None or hiddens_B is None:
            print(f"    [SKIP]")
            continue

        n_layers = len(hiddens_A)
        d_final = hiddens_A[n_layers - 1] - hiddens_B[n_layers - 1]

        # delta_u: 代表词方向
        delta_u_meanA = torch.stack([W_unembed[maid] for maid in meaningA_ids]).mean(dim=0)
        delta_u_meanB = torch.stack([W_unembed[mbid] for mbid in meaningB_ids]).mean(dim=0)
        delta_u = delta_u_meanA - delta_u_meanB

        d_norm = torch.norm(d_final).item()
        alignment = cos_sim(d_final, delta_u) if d_norm > 1e-8 else 0.0

        print(f"    alignment = {alignment:.4f} (cos(d, delta_u))")
        print(f"    d_norm = {d_norm:.4f}")

        # 3. 如果alignment为负（DS7B情况），寻找实际消歧方向
        if alignment < 0:
            print(f"    [REVERSED] alignment < 0, searching for actual disambiguation direction...")

            # 3a. 用反方向测试
            d_reversed = -d_final
            logit_margin_normal = (d_final @ delta_u).item()
            logit_margin_reversed = (d_reversed @ delta_u).item()
            print(f"    Normal logit margin: {logit_margin_normal:.4f}")
            print(f"    Reversed logit margin: {logit_margin_reversed:.4f}")

            # 3b. 寻找unembed中最与d_final对齐的top-20 token
            all_logits = d_final @ W_unembed.T
            topk_vals, topk_ids = torch.topk(all_logits, 20)
            topk_tokens = [tokenizer.decode([t]).strip() for t in topk_ids.tolist()]
            print(f"    Top-20 tokens aligned with d_final: {topk_tokens[:10]}")

            # 3c. 寻找最与-d_final对齐的top-20 token
            all_logits_neg = -d_final @ W_unembed.T
            topk_vals_neg, topk_ids_neg = torch.topk(all_logits_neg, 20)
            topk_tokens_neg = [tokenizer.decode([t]).strip() for t in topk_ids_neg.tolist()]
            print(f"    Top-20 tokens aligned with -d_final: {topk_tokens_neg[:10]}")

            # 3d. 分析：A句子的top预测词 vs B句子的top预测词
            logits_A = hiddens_A[n_layers - 1] @ W_unembed.T
            logits_B = hiddens_B[n_layers - 1] @ W_unembed.T
            probs_A = F.softmax(logits_A, dim=-1)
            probs_B = F.softmax(logits_B, dim=-1)
            top5_A = torch.topk(probs_A, 5)
            top5_B = torch.topk(probs_B, 5)
            top5_tokens_A = [tokenizer.decode([t]).strip() for t in top5_A[1].tolist()]
            top5_tokens_B = [tokenizer.decode([t]).strip() for t in top5_B[1].tolist()]
            print(f"    A-sentence top-5: {top5_tokens_A[:5]}")
            print(f"    B-sentence top-5: {top5_tokens_B[:5]}")

            # 3e. 用A-B的实际top差异定义delta_u_actual
            diff_logits = logits_A - logits_B
            top_diff_vals, top_diff_ids = torch.topk(diff_logits, 10)
            top_diff_neg_vals, top_diff_neg_ids = torch.topk(-diff_logits, 10)

            # delta_u_actual: A比B更喜欢的前10个token vs B比A更喜欢的前10个token
            actualA_vecs = torch.stack([W_unembed[tid] for tid in top_diff_ids[:5]])
            actualB_vecs = torch.stack([W_unembed[tid] for tid in top_diff_neg_ids[:5]])
            delta_u_actual = actualA_vecs.mean(dim=0) - actualB_vecs.mean(dim=0)

            alignment_actual = cos_sim(d_final, delta_u_actual)
            print(f"    alignment with actual delta_u: {alignment_actual:.4f}")

        # 4. 逐层alignment变化
        layer_alignments = []
        for l in range(n_layers):
            d_l = hiddens_A[l] - hiddens_B[l]
            d_norm_l = torch.norm(d_l).item()
            if d_norm_l > 1e-8:
                al = cos_sim(d_l, delta_u)
                layer_alignments.append(al)

        # 找alignment变号的层
        sign_changes = 0
        for i in range(1, len(layer_alignments)):
            if layer_alignments[i-1] * layer_alignments[i] < 0:
                sign_changes += 1
                print(f"    Sign change at L{i-1}->L{i}: {layer_alignments[i-1]:.4f} -> {layer_alignments[i]:.4f}")

        # 5. 消歧方向在unembed子空间中的投影
        d_proj_unembed = (W_unembed.T @ (W_unembed @ d_final)) / (W_unembed @ d_final).norm()**2
        # 用SVD近似
        k_approx = min(200, D)
        U_k = U_u[:, :k_approx]
        S_k = S_u[:k_approx]
        Vt_k = Vt_u[:k_approx, :]
        d_in_subspace = (Vt_k.T @ (Vt_k @ d_final))
        reconstruction_error = torch.norm(d_final - d_in_subspace).item() / (torch.norm(d_final).item() + 1e-10)
        print(f"    Unembed {k_approx}-dim reconstruction of d: error={reconstruction_error:.4f}")

        word_result = {
            "word": word,
            "alignment": alignment,
            "d_norm": d_norm,
            "sign_changes": sign_changes,
            "mean_layer_alignment": float(np.mean(layer_alignments)) if layer_alignments else 0,
            "min_layer_alignment": min(layer_alignments) if layer_alignments else 0,
            "max_layer_alignment": max(layer_alignments) if layer_alignments else 0,
            "unembed_reconstruction_error": reconstruction_error,
        }
        all_word_results.append(word_result)

    if not all_word_results:
        return {}

    summary = {
        "mean_alignment": float(np.mean([w["alignment"] for w in all_word_results])),
        "reversed_count": sum(1 for w in all_word_results if w["alignment"] < 0),
        "mean_sign_changes": float(np.mean([w["sign_changes"] for w in all_word_results])),
        "mean_unembed_recon_error": float(np.mean([w["unembed_reconstruction_error"] for w in all_word_results])),
    }

    results["stage634"] = {"summary": summary, "words": all_word_results}

    print(f"\n  === Stage634 Summary ===")
    print(f"  Mean alignment: {summary['mean_alignment']:.4f}")
    print(f"  Reversed count: {summary['reversed_count']}/5")
    print(f"  Mean sign changes: {summary['mean_sign_changes']:.1f}")
    print(f"  Mean unembed recon error: {summary['mean_unembed_recon_error']:.4f}")

    return results


# ============ Stage635: "浪费能量"功能定位 ============

def build_probe_vector(W_unembed, tokenizer, probe_words):
    """Build a probe vector from a set of words"""
    ids = [tokenizer.encode(w, add_special_tokens=False)[-1] for w in probe_words]
    vecs = torch.stack([W_unembed[i] for i in ids if i < W_unembed.shape[0]])
    if len(vecs) == 0:
        return None
    return vecs.mean(dim=0)


def run_stage635(model, tokenizer, model_key):
    """"浪费能量"功能定位——92-97%非对齐能量的语义分析"""
    print("\n" + "=" * 70)
    print("Stage635: Waste Energy Functional Analysis")
    print("=" * 70)

    device = safe_get_device(model)
    layers = discover_layers(model)
    W_unembed = get_unembed_matrix(model)

    if W_unembed is None:
        print("  [SKIP]")
        return {}

    V, D = W_unembed.shape

    # 1. Build all probe vectors
    all_probes = {}
    for category, words in {**SEMANTIC_PROBES, **SYNTAX_PROBES}.items():
        pv = build_probe_vector(W_unembed, tokenizer, words)
        if pv is not None:
            all_probes[category] = pv

    print(f"  Built {len(all_probes)} probe vectors")

    # 2. For each ambiguous word pair, decompose d into aligned vs waste
    results = {}
    all_word_results = []

    for idx, (sA, sB, word, meaningA_tokens, meaningB_tokens) in enumerate(CORE_PAIRS):
        print(f"\n  [{idx+1}/5] word: {word}")

        meaningA_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningA_tokens]
        meaningB_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningB_tokens]

        hiddens_A = extract_all_layer_hidden(model, tokenizer, sA, layers)
        hiddens_B = extract_all_layer_hidden(model, tokenizer, sB, layers)

        if hiddens_A is None or hiddens_B is None:
            print(f"    [SKIP]")
            continue

        n_layers = len(hiddens_A)
        d_final = hiddens_A[n_layers - 1] - hiddens_B[n_layers - 1]
        d_norm = torch.norm(d_final).item()

        if d_norm < 1e-8:
            print(f"    [SKIP] d_norm too small")
            continue

        # delta_u
        delta_u_meanA = torch.stack([W_unembed[maid] for maid in meaningA_ids]).mean(dim=0)
        delta_u_meanB = torch.stack([W_unembed[mbid] for mbid in meaningB_ids]).mean(dim=0)
        delta_u = delta_u_meanA - delta_u_meanB
        delta_u_unit = delta_u / (torch.norm(delta_u).item() + 1e-10)

        # Decompose d_final into aligned + waste
        d_aligned = torch.dot(d_final, delta_u_unit) * delta_u_unit
        d_waste = d_final - d_aligned
        d_aligned_norm = torch.norm(d_aligned).item()
        d_waste_norm = torch.norm(d_waste).item()
        aligned_ratio = d_aligned_norm / d_norm
        waste_ratio = d_waste_norm / d_norm

        print(f"    d_norm={d_norm:.4f}, aligned={d_aligned_norm:.4f}({aligned_ratio:.4f}), waste={d_waste_norm:.4f}({waste_ratio:.4f})")

        # 3. Project waste onto each probe
        waste_probe_alignments = {}
        for cat, pv in all_probes.items():
            pv_norm = torch.norm(pv).item()
            if pv_norm < 1e-8:
                continue
            pv_unit = pv / pv_norm
            proj_aligned = abs(torch.dot(d_aligned, pv_unit).item())
            proj_waste = abs(torch.dot(d_waste, pv_unit).item())
            proj_total = abs(torch.dot(d_final, pv_unit).item())
            waste_probe_alignments[cat] = {
                "aligned_proj": proj_aligned,
                "waste_proj": proj_waste,
                "total_proj": proj_total,
                "waste_fraction": proj_waste / (proj_total + 1e-10),
            }

        # 4. Find top probes for waste
        sorted_probes = sorted(waste_probe_alignments.items(),
                               key=lambda x: x[1]["waste_proj"], reverse=True)

        print(f"    Top waste-aligned probes:")
        for cat, vals in sorted_probes[:5]:
            print(f"      {cat}: waste_proj={vals['waste_proj']:.4f}, "
                  f"total={vals['total_proj']:.4f}, waste_frac={vals['waste_fraction']:.4f}")

        # 5. Test: removing waste component and check top-1 accuracy
        logits_A_orig = hiddens_A[n_layers - 1] @ W_unembed.T
        probs_A_orig = F.softmax(logits_A_orig, dim=-1)
        top1_orig = torch.argmax(probs_A_orig).item()
        top1_correct_orig = int(any(maid == top1_orig for maid in meaningA_ids))

        # Remove waste: use only aligned component
        h_A_no_waste = hiddens_A[n_layers - 1] - d_waste / 2  # Keep only aligned part
        logits_A_no_waste = h_A_no_waste @ W_unembed.T
        probs_A_no_waste = F.softmax(logits_A_no_waste, dim=-1)
        top1_no_waste = torch.argmax(probs_A_no_waste).item()
        top1_correct_no_waste = int(any(maid == top1_no_waste for maid in meaningA_ids))

        # Keep only waste: remove aligned component
        h_A_only_waste = hiddens_A[n_layers - 1] - d_aligned / 2
        logits_A_only_waste = h_A_only_waste @ W_unembed.T
        probs_A_only_waste = F.softmax(logits_A_only_waste, dim=-1)
        top1_only_waste = torch.argmax(probs_A_only_waste).item()
        top1_correct_only_waste = int(any(maid == top1_only_waste for maid in meaningA_ids))

        entropy_orig = -(probs_A_orig * torch.log(probs_A_orig + 1e-10)).sum().item()
        entropy_no_waste = -(probs_A_no_waste * torch.log(probs_A_no_waste + 1e-10)).sum().item()

        print(f"    Top-1 correct: orig={top1_correct_orig}, no_waste={top1_correct_no_waste}, only_waste={top1_correct_only_waste}")
        print(f"    Entropy: orig={entropy_orig:.2f}, no_waste={entropy_no_waste:.2f}")

        # 6. SVD of waste component
        if d_waste_norm > 1e-8:
            # Use multiple layers to build waste matrix
            waste_layers = []
            for l in range(1, n_layers):
                d_l = hiddens_A[l] - hiddens_B[l]
                du_l_norm = torch.norm(delta_u).item()
                if du_l_norm > 1e-8:
                    aligned_l = torch.dot(d_l, delta_u_unit) * delta_u_unit
                    waste_l = d_l - aligned_l
                    if torch.norm(waste_l).item() > 1e-8:
                        waste_layers.append(waste_l)

            if len(waste_layers) >= 2:
                waste_matrix = torch.stack(waste_layers)
                U_w, S_w, Vt_w = torch.linalg.svd(waste_matrix, full_matrices=False)
                cumul_w = torch.cumsum(S_w ** 2, dim=0) / (torch.sum(S_w ** 2) + 1e-10)
                waste_eff_rank_90 = (cumul_w < 0.90).sum().item() + 1
                print(f"    Waste SVD: eff_rank_90 = {waste_eff_rank_90}")
            else:
                waste_eff_rank_90 = -1
        else:
            waste_eff_rank_90 = -1

        word_result = {
            "word": word,
            "aligned_ratio": aligned_ratio,
            "waste_ratio": waste_ratio,
            "top1_correct_orig": top1_correct_orig,
            "top1_correct_no_waste": top1_correct_no_waste,
            "top1_correct_only_waste": top1_correct_only_waste,
            "entropy_orig": entropy_orig,
            "entropy_no_waste": entropy_no_waste,
            "waste_eff_rank_90": waste_eff_rank_90,
            "top_waste_probes": {cat: vals["waste_proj"] for cat, vals in sorted_probes[:5]},
        }
        all_word_results.append(word_result)

    if not all_word_results:
        return {}

    summary = {
        "mean_aligned_ratio": float(np.mean([w["aligned_ratio"] for w in all_word_results])),
        "mean_waste_ratio": float(np.mean([w["waste_ratio"] for w in all_word_results])),
        "mean_waste_eff_rank_90": float(np.mean([w["waste_eff_rank_90"] for w in all_word_results if w["waste_eff_rank_90"] > 0])),
        "top1_preserved_rate": float(np.mean([w["top1_correct_no_waste"] for w in all_word_results])),
    }

    results["stage635"] = {"summary": summary, "words": all_word_results}

    print(f"\n  === Stage635 Summary ===")
    print(f"  Mean aligned ratio: {summary['mean_aligned_ratio']:.4f}")
    print(f"  Mean waste ratio: {summary['mean_waste_ratio']:.4f}")
    print(f"  Mean waste eff_rank_90: {summary['mean_waste_eff_rank_90']:.1f}")
    print(f"  Top-1 preserved (no waste): {summary['top1_preserved_rate']:.2f}")

    return results


# ============ Stage636: 非消歧维度功能分解 ============

def run_stage636(model, tokenizer, model_key):
    """非消歧维度功能分解——消歧子空间vs非消歧子空间"""
    print("\n" + "=" * 70)
    print("Stage636: Non-Disambiguation Dimension Functional Decomposition")
    print("=" * 70)

    device = safe_get_device(model)
    layers = discover_layers(model)
    W_unembed = get_unembed_matrix(model)
    W_embed = get_embed_matrix(model)

    if W_unembed is None:
        print("  [SKIP]")
        return {}

    V, D = W_unembed.shape
    print(f"  Hidden dim: {D}, Vocab: {V}")

    # 1. Build all probe vectors from semantic and syntax categories
    all_probes = {}
    for category, words in {**SEMANTIC_PROBES, **SYNTAX_PROBES}.items():
        pv = build_probe_vector(W_unembed, tokenizer, words)
        if pv is not None:
            all_probes[category] = pv
    print(f"  Probes: {len(all_probes)}")

    results = {}
    all_word_results = []

    for idx, (sA, sB, word, meaningA_tokens, meaningB_tokens) in enumerate(CORE_PAIRS):
        print(f"\n  [{idx+1}/5] word: {word}")

        meaningA_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningA_tokens]
        meaningB_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningB_tokens]

        hiddens_A = extract_all_layer_hidden(model, tokenizer, sA, layers)
        hiddens_B = extract_all_layer_hidden(model, tokenizer, sB, layers)

        if hiddens_A is None or hiddens_B is None:
            print(f"    [SKIP]")
            continue

        n_layers = len(hiddens_A)

        # 2. Build disambiguation direction matrix across all layers
        d_matrix = torch.stack([hiddens_A[l] - hiddens_B[l] for l in range(n_layers)])
        U_d, S_d, Vt_d = torch.linalg.svd(d_matrix, full_matrices=False)

        # Determine disambiguation subspace dimension (90% energy)
        cumul_d = torch.cumsum(S_d ** 2, dim=0) / (torch.sum(S_d ** 2) + 1e-10)
        disamb_dim_90 = (cumul_d < 0.90).sum().item() + 1
        disamb_dim_99 = (cumul_d < 0.99).sum().item() + 1

        print(f"    Disamb subspace: dim_90={disamb_dim_90}, dim_99={disamb_dim_99}")
        print(f"    Non-disamb dims: {D - disamb_dim_90}")

        # 3. Disambiguation subspace basis (top-K right singular vectors)
        disamb_basis = Vt_d[:disamb_dim_90, :]  # K x D
        # Non-disambiguation subspace basis (remaining)
        non_disamb_basis = Vt_d[disamb_dim_90:, :]  # (D-K) x D

        # 4. Project probes onto disamb vs non-disamb subspaces
        probe_projections = {}
        for cat, pv in all_probes.items():
            pv_norm = torch.norm(pv).item()
            if pv_norm < 1e-8:
                continue

            # Project onto disambiguation subspace
            proj_disamb = disamb_basis @ pv  # K-dimensional
            proj_disamb_norm = torch.norm(proj_disamb).item()

            # Project onto non-disambiguation subspace
            proj_non_disamb = non_disamb_basis @ pv  # (D-K)-dimensional
            proj_non_disamb_norm = torch.norm(proj_non_disamb).item()

            total_proj = proj_disamb_norm + proj_non_disamb_norm
            disamb_fraction = proj_disamb_norm / (total_proj + 1e-10)
            non_disamb_fraction = proj_non_disamb_norm / (total_proj + 1e-10)

            probe_projections[cat] = {
                "disamb_norm": proj_disamb_norm,
                "non_disamb_norm": proj_non_disamb_norm,
                "disamb_fraction": disamb_fraction,
                "non_disamb_fraction": non_disamb_fraction,
            }

        # 5. Project the full hidden state of sentence A (final layer) into both subspaces
        h_A = hiddens_A[n_layers - 1]
        h_A_disamb = disamb_basis @ h_A
        h_A_non_disamb = non_disamb_basis @ h_A
        h_A_disamb_norm = torch.norm(h_A_disamb).item()
        h_A_non_disamb_norm = torch.norm(h_A_non_disamb).item()
        h_A_total_norm = torch.norm(h_A).item()

        h_B = hiddens_B[n_layers - 1]
        h_B_disamb = disamb_basis @ h_B
        h_B_non_disamb = non_disamb_basis @ h_B

        # Cosine in disamb subspace vs non-disamb subspace
        cos_disamb = cos_sim(h_A_disamb, h_B_disamb) if torch.norm(h_A_disamb) > 1e-8 else 0
        cos_non_disamb = cos_sim(h_A_non_disamb, h_B_non_disamb) if torch.norm(h_A_non_disamb) > 1e-8 else 0

        print(f"    Hidden A: disamb={h_A_disamb_norm:.4f}({h_A_disamb_norm/h_A_total_norm:.4f}), "
              f"non_disamb={h_A_non_disamb_norm:.4f}({h_A_non_disamb_norm/h_A_total_norm:.4f})")
        print(f"    Cos in disamb subspace: {cos_disamb:.4f}")
        print(f"    Cos in non-disamb subspace: {cos_non_disamb:.4f}")

        # 6. Top probes that live in non-disamb subspace
        sorted_probes = sorted(probe_projections.items(),
                               key=lambda x: x[1]["non_disamb_norm"], reverse=True)
        print(f"    Top non-disamb probes:")
        for cat, vals in sorted_probes[:5]:
            print(f"      {cat}: disamb_frac={vals['disamb_fraction']:.4f}, "
                  f"non_disamb_frac={vals['non_disamb_fraction']:.4f}")

        # 7. Test: reconstruct hidden from non-disamb subspace only
        h_A_recon_non_disamb = non_disamb_basis.T @ h_A_non_disamb
        logits_recon = h_A_recon_non_disamb @ W_unembed.T
        probs_recon = F.softmax(logits_recon, dim=-1)
        top1_recon = torch.argmax(probs_recon).item()
        top1_correct_recon = int(any(maid == top1_recon for maid in meaningA_ids))
        entropy_recon = -(probs_recon * torch.log(probs_recon + 1e-10)).sum().item()

        # Original
        logits_orig = h_A @ W_unembed.T
        probs_orig = F.softmax(logits_orig, dim=-1)
        entropy_orig = -(probs_orig * torch.log(probs_orig + 1e-10)).sum().item()

        print(f"    Non-disamb only: top1_correct={top1_correct_recon}, entropy={entropy_recon:.2f} vs orig={entropy_orig:.2f}")

        word_result = {
            "word": word,
            "disamb_dim_90": disamb_dim_90,
            "disamb_dim_99": disamb_dim_99,
            "h_A_disamb_fraction": h_A_disamb_norm / h_A_total_norm,
            "h_A_non_disamb_fraction": h_A_non_disamb_norm / h_A_total_norm,
            "cos_disamb_subspace": cos_disamb,
            "cos_non_disamb_subspace": cos_non_disamb,
            "non_disamb_top1_correct": top1_correct_recon,
            "entropy_orig": entropy_orig,
            "entropy_recon": entropy_recon,
            "probe_projections": {cat: {
                "disamb_frac": vals["disamb_fraction"],
                "non_disamb_frac": vals["non_disamb_fraction"],
            } for cat, vals in sorted_probes[:5]},
        }
        all_word_results.append(word_result)

    if not all_word_results:
        return {}

    summary = {
        "mean_disamb_dim_90": float(np.mean([w["disamb_dim_90"] for w in all_word_results])),
        "mean_disamb_dim_99": float(np.mean([w["disamb_dim_99"] for w in all_word_results])),
        "mean_non_disamb_fraction": float(np.mean([w["h_A_non_disamb_fraction"] for w in all_word_results])),
        "mean_cos_non_disamb": float(np.mean([w["cos_non_disamb_subspace"] for w in all_word_results])),
        "non_disamb_top1_rate": float(np.mean([w["non_disamb_top1_correct"] for w in all_word_results])),
    }

    results["stage636"] = {"summary": summary, "words": all_word_results}

    print(f"\n  === Stage636 Summary ===")
    print(f"  Mean disamb dim_90: {summary['mean_disamb_dim_90']:.1f}")
    print(f"  Mean non-disamb fraction: {summary['mean_non_disamb_fraction']:.4f}")
    print(f"  Mean cos in non-disamb: {summary['mean_cos_non_disamb']:.4f}")
    print(f"  Non-disamb only top-1 correct: {summary['non_disamb_top1_rate']:.2f}")

    return results


# ============ Stage637: Gemma4 d_norm不增长归因 ============

def run_stage637(model, tokenizer, model_key):
    """Gemma4 d_norm不增长归因分析（所有模型做对比）"""
    print("\n" + "=" * 70)
    print("Stage637: d_norm Growth Attribution (All Models)")
    print("=" * 70)

    device = safe_get_device(model)
    layers = discover_layers(model)
    W_unembed = get_unembed_matrix(model)

    if W_unembed is None:
        print("  [SKIP]")
        return {}

    results = {}
    all_word_results = []

    for idx, (sA, sB, word, meaningA_tokens, meaningB_tokens) in enumerate(CORE_PAIRS):
        print(f"\n  [{idx+1}/5] word: {word}")

        meaningA_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningA_tokens]
        meaningB_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningB_tokens]

        hiddens_A = extract_all_layer_hidden(model, tokenizer, sA, layers)
        hiddens_B = extract_all_layer_hidden(model, tokenizer, sB, layers)

        if hiddens_A is None or hiddens_B is None:
            print(f"    [SKIP]")
            continue

        n_layers = len(hiddens_A)

        # 1. Per-layer d_norm
        d_norms = []
        delta_norms = []
        delta_angles = []
        cancel_ratios = []

        for l in range(n_layers):
            d_l = hiddens_A[l] - hiddens_B[l]
            d_norms.append(torch.norm(d_l).item())

        # 2. Per-layer delta analysis
        for l in range(1, n_layers):
            delta_A = hiddens_A[l] - hiddens_A[l-1]
            delta_B = hiddens_B[l] - hiddens_B[l-1]
            delta_diff = delta_A - delta_B

            delta_norms.append(torch.norm(delta_diff).item())

            # 3. Angle between delta_diff and d_{l-1}
            d_prev = hiddens_A[l-1] - hiddens_B[l-1]
            if torch.norm(d_prev) > 1e-8 and torch.norm(delta_diff) > 1e-8:
                c = cos_sim(delta_diff, d_prev)
                angle = np.degrees(np.arccos(np.clip(c, -1, 1)))
                delta_angles.append(angle)
            else:
                delta_angles.append(0.0)

            # 4. Cancellation ratio: how much delta_diff opposes d direction
            if d_norms[l-1] > 1e-8:
                d_prev_unit = (hiddens_A[l-1] - hiddens_B[l-1])
                d_prev_unit = d_prev_unit / torch.norm(d_prev_unit)
                projection = torch.dot(delta_diff, d_prev_unit).item()
                # Negative projection means delta_diff opposes d growth
                cancel_ratio = -projection / (torch.norm(delta_diff).item() + 1e-10)
                cancel_ratios.append(max(0, cancel_ratio))  # Only count opposing
            else:
                cancel_ratios.append(0.0)

        # 5. Compute norm growth decomposition
        # d_l = d_{l-1} + delta_diff_l
        # ||d_l||^2 = ||d_{l-1}||^2 + ||delta_diff||^2 + 2*dot(d_{l-1}, delta_diff)
        # The cross term determines growth vs shrinkage
        growth_terms = []
        for l in range(1, n_layers):
            delta_diff = (hiddens_A[l] - hiddens_A[l-1]) - (hiddens_B[l] - hiddens_B[l-1])
            d_prev = hiddens_A[l-1] - hiddens_B[l-1]
            cross_term = 2 * torch.dot(d_prev, delta_diff).item()
            delta_sq = torch.norm(delta_diff).item() ** 2
            growth_terms.append({
                "layer": l,
                "cross_term": cross_term,
                "delta_sq": delta_sq,
                "net_growth": cross_term + delta_sq,
            })

        # 6. Summary statistics
        total_growth = d_norms[-1] / (d_norms[0] + 1e-10)
        mean_delta_norm = float(np.mean(delta_norms))
        mean_delta_angle = float(np.mean(delta_angles))
        mean_cancel_ratio = float(np.mean(cancel_ratios))

        # Layers where d_norm decreases (net_growth < 0)
        shrink_layers = sum(1 for gt in growth_terms if gt["net_growth"] < 0)

        # Peak d_norm layer
        peak_layer = np.argmax(d_norms)

        print(f"    Total growth: {total_growth:.2f}x")
        print(f"    Peak at L{peak_layer} (d_norm={d_norms[peak_layer]:.4f})")
        print(f"    Mean delta_norm: {mean_delta_norm:.4f}")
        print(f"    Mean delta_angle: {mean_delta_angle:.1f} deg")
        print(f"    Mean cancel_ratio: {mean_cancel_ratio:.4f}")
        print(f"    Shrink layers: {shrink_layers}/{n_layers-1}")

        # 7. Hidden state norm analysis (is norm itself shrinking?)
        hA_norms = [torch.norm(hiddens_A[l]).item() for l in range(n_layers)]
        hB_norms = [torch.norm(hiddens_B[l]).item() for l in range(n_layers)]
        hA_growth = hA_norms[-1] / (hA_norms[0] + 1e-10)
        hB_growth = hB_norms[-1] / (hB_norms[0] + 1e-10)
        print(f"    h_A norm growth: {hA_growth:.2f}x")
        print(f"    h_B norm growth: {hB_growth:.2f}x")

        # 8. First 5 and last 5 layers detail
        print(f"    Per-layer d_norm (first/last 5):")
        print(f"      Early: {d_norms[:5]}")
        print(f"      Late:  {d_norms[-5:]}")

        word_result = {
            "word": word,
            "total_growth": total_growth,
            "peak_layer": int(peak_layer),
            "mean_delta_norm": mean_delta_norm,
            "mean_delta_angle": mean_delta_angle,
            "mean_cancel_ratio": mean_cancel_ratio,
            "shrink_layers": shrink_layers,
            "total_layers": n_layers - 1,
            "hA_growth": hA_growth,
            "hB_growth": hB_growth,
        }
        all_word_results.append(word_result)

    if not all_word_results:
        return {}

    summary = {
        "mean_total_growth": float(np.mean([w["total_growth"] for w in all_word_results])),
        "mean_peak_layer_frac": float(np.mean([w["peak_layer"] / w["total_layers"] for w in all_word_results])),
        "mean_delta_angle": float(np.mean([w["mean_delta_angle"] for w in all_word_results])),
        "mean_cancel_ratio": float(np.mean([w["mean_cancel_ratio"] for w in all_word_results])),
        "mean_shrink_fraction": float(np.mean([w["shrink_layers"] / w["total_layers"] for w in all_word_results])),
        "mean_hA_growth": float(np.mean([w["hA_growth"] for w in all_word_results])),
        "mean_hB_growth": float(np.mean([w["hB_growth"] for w in all_word_results])),
    }

    results["stage637"] = {"summary": summary, "words": all_word_results}

    print(f"\n  === Stage637 Summary ===")
    print(f"  Mean total d_norm growth: {summary['mean_total_growth']:.2f}x")
    print(f"  Mean peak layer: {summary['mean_peak_layer_frac']:.2f}")
    print(f"  Mean delta angle: {summary['mean_delta_angle']:.1f} deg")
    print(f"  Mean cancel ratio: {summary['mean_cancel_ratio']:.4f}")
    print(f"  Mean shrink fraction: {summary['mean_shrink_fraction']:.4f}")
    print(f"  Mean h_A norm growth: {summary['mean_hA_growth']:.2f}x")
    print(f"  Mean h_B norm growth: {summary['mean_hB_growth']:.2f}x")

    return results


# ============ Main ============

def main():
    if len(sys.argv) < 2:
        print("Usage: python stage634_635_636_637_bottleneck_deep.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)

    model_key = sys.argv[1].lower()
    valid_keys = {"qwen3", "deepseek7b", "glm4", "gemma4"}
    if model_key not in valid_keys:
        print(f"Invalid model: {model_key}. Choose from: {valid_keys}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"Stage634-637: Bottleneck Deep Analysis — {model_key.upper()}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")

    # Load model
    print(f"\nLoading {model_key}...")
    t0 = time.time()
    model, tokenizer = load_model_bundle(model_key)
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s")

    all_results = {}

    try:
        # Run all stages
        all_results.update(run_stage634(model, tokenizer, model_key))
        all_results.update(run_stage635(model, tokenizer, model_key))
        all_results.update(run_stage636(model, tokenizer, model_key))
        all_results.update(run_stage637(model, tokenizer, model_key))
    except Exception as e:
        print(f"\n  ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Free model
        print(f"\nFreeing {model_key}...")
        free_model(model)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save results
    output_path = OUTPUT_DIR / f"stage634_637_{model_key}_{TIMESTAMP}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to: {output_path}")

    # Print final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY — {model_key.upper()}")
    print(f"{'='*70}")
    for stage_name, stage_data in all_results.items():
        if "summary" in stage_data:
            print(f"\n  {stage_name}:")
            for k, v in stage_data["summary"].items():
                print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
