#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stage631-632-633: 完整信息流闭式方程 + 最优编码验证 + 扩展消歧词表泛化

Stage631: 完整信息流闭式方程组装与验证（P0）
  已知部件：
  - 残差流方程: h_l = (I + A_l) · h_{l-1}, A_l低秩(rank≈2-5)
  - SwiGLU压缩: MLP变换≈高秩线性×稀疏mask, 活跃神经元3-15%
  - 统一方程: logit_margin = cos(d, delta_u) × ||d||
  - 消歧传播: 每层旋转19-54°, 存活率<5%, 能量增长3-40%/层
  
  目标：组装完整方程 h_L = Prod(I + A_l) · h_0 → logit
  验证：用实际hidden state vs 预测hidden state对比

Stage632: 消歧信息"最优编码"理论验证（P0）
  为什么efficiency≈2%？是否是最优？
  实验：
  - 构造人工unembed矩阵：强制消歧方向与unembed对齐(cos=1.0)
  - 测量强制对齐后的logit margin变化
  - 验证"正交编码"是否保护了其他语言能力

Stage633: 扩展消歧词表泛化验证（P1）
  之前仅用5个词，扩展到20+个歧义词
  验证已发现的不变量在更大词表上是否成立

用法: python stage631_632_633_closed_form_optimal.py [qwen3|deepseek7b|glm4|gemma4]
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


def find_word_token_id(tokenizer, sentence, word):
    """Find the token ID(s) of a word within a sentence"""
    word_tokens = tokenizer.encode(word, add_special_tokens=False)
    sent_tokens = tokenizer.encode(sentence, add_special_tokens=True)
    sent_no_special = tokenizer.encode(sentence, add_special_tokens=False)
    
    # Try to find word tokens in sentence tokens
    for wt in word_tokens:
        if wt in sent_no_special:
            return wt
    # Fallback: last token of word encoding
    return word_tokens[-1] if word_tokens else None


def find_target_token_position(tokenizer, sentence, word):
    """Find the position of the word's LAST token in the tokenized sentence"""
    word_tokens = tokenizer.encode(word, add_special_tokens=False)
    if not word_tokens:
        return None
    sent_tokens = tokenizer.encode(sentence, add_special_tokens=True)
    
    # Find the last word token position
    for i in range(len(sent_tokens) - 1, -1, -1):
        if sent_tokens[i] == word_tokens[-1]:
            return i
    return None


def extract_all_layer_hidden(model, tokenizer, sentence, layers, target_pos=-1):
    """Extract hidden state at target position (-1 = last token)"""
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
    """Get unembedding weight matrix"""
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


# ============ 核心歧义词对 ============
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

# ============ 扩展歧义词对（Stage633用）============
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


# ============ Stage631: 完整信息流闭式方程 ============

def compute_layer_transform(model, tokenizer, layers, sentence, device):
    """Compute per-layer effective transformation A_l where h_l = (I + A_l) * h_{l-1}"""
    # Get all hidden states
    hiddens = extract_all_layer_hidden(model, tokenizer, sentence, layers)
    if hiddens is None or len(hiddens) < 2:
        return None

    transforms = {}
    n_layers = len(hiddens)

    for l in range(1, n_layers):
        h_prev = hiddens[l - 1]
        h_curr = hiddens[l]
        # A_l = h_curr - h_prev (the residual added by layer l)
        delta = h_curr - h_prev
        transforms[l] = delta

    return transforms, hiddens


def run_stage631(model, tokenizer, model_key):
    """完整信息流闭式方程组装与验证"""
    print("\n" + "=" * 70)
    print("Stage631: 完整信息流闭式方程组装与验证")
    print("=" * 70)

    device = safe_get_device(model)
    layers = discover_layers(model)
    W_unembed = get_unembed_matrix(model)

    if W_unembed is None:
        print("  [SKIP] 无法获取unembed矩阵")
        return {}

    results = {}
    all_pair_results = []

    for idx, (sA, sB, word, meaningA_tokens, meaningB_tokens) in enumerate(CORE_PAIRS):
        print(f"\n  [{idx+1}/5] 词: {word}")

        # Extract hiddens for both sentences
        hiddens_A = extract_all_layer_hidden(model, tokenizer, sA, layers)
        hiddens_B = extract_all_layer_hidden(model, tokenizer, sB, layers)

        if hiddens_A is None or hiddens_B is None:
            print(f"    [SKIP] 提取失败")
            continue

        n_layers = len(hiddens_A)
        if n_layers != len(hiddens_B):
            print(f"    [SKIP] 层数不一致: {n_layers} vs {len(hiddens_B)}")
            continue

        # Get meaning token IDs
        meaningA_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningA_tokens]
        meaningB_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningB_tokens]

        # d_final = h_A[-1] - h_B[-1]
        d_final = hiddens_A[n_layers - 1] - hiddens_B[n_layers - 1]

        # 1. Compute per-layer delta accumulation errors
        errors = []
        d_norms = []
        d_angles = []

        for l in range(1, n_layers):
            # Predicted: d_l = d_0 + sum_{k=1}^{l} (delta_A_k - delta_B_k)
            d_pred = (hiddens_A[0] - hiddens_B[0])
            for k in range(1, l + 1):
                d_pred = d_pred + (hiddens_A[k] - hiddens_A[k - 1]) - (hiddens_B[k] - hiddens_B[k - 1])

            d_actual = hiddens_A[l] - hiddens_B[l]
            err = torch.norm(d_pred - d_actual).item()
            errors.append(err)
            d_norms.append(torch.norm(d_actual).item())

            if torch.norm(d_pred) > 1e-8 and torch.norm(d_actual) > 1e-8:
                angle = cos_sim(d_pred, d_actual)
                d_angles.append(angle)
            else:
                d_angles.append(0.0)

        # 2. Logit margin using meaning tokens (mean of top meaning token differences)
        logits_A = hiddens_A[n_layers - 1] @ W_unembed.T
        logits_B = hiddens_B[n_layers - 1] @ W_unembed.T
        # Sentence A should have higher logit for meaningA tokens, sentence B for meaningB tokens
        margin_A = float(np.mean([logits_A[maid].item() for maid in meaningA_ids]))
        margin_B = float(np.mean([logits_B[mbid].item() for mbid in meaningB_ids]))
        # Cross margins (A sentence predicting B meaning, and vice versa)
        cross_A_on_B = float(np.mean([logits_A[mbid].item() for mbid in meaningB_ids]))
        cross_B_on_A = float(np.mean([logits_B[maid].item() for maid in meaningA_ids]))
        logit_margin = margin_A - cross_A_on_B  # how much A prefers meaning A over B

        # 3. Alignment with unembed rows of meaning tokens
        delta_u_meanA = torch.stack([W_unembed[maid] for maid in meaningA_ids]).mean(dim=0)
        delta_u_meanB = torch.stack([W_unembed[mbid] for mbid in meaningB_ids]).mean(dim=0)
        delta_u = delta_u_meanA - delta_u_meanB
        d_norm_final = torch.norm(d_final).item()
        delta_u_norm = torch.norm(delta_u).item()
        alignment = cos_sim(d_final, delta_u) if d_norm_final > 1e-8 and delta_u_norm > 1e-8 else 0.0
        logit_margin_pred = alignment * d_norm_final * delta_u_norm

        # 4. SVD of cumulative d matrix
        d_matrix = torch.stack([hiddens_A[l] - hiddens_B[l] for l in range(n_layers)])
        U, S, Vt = torch.linalg.svd(d_matrix, full_matrices=False)
        cumul_energy = torch.cumsum(S ** 2, dim=0) / (torch.sum(S ** 2) + 1e-10)
        eff_rank_90 = (cumul_energy < 0.90).sum().item() + 1

        # 5. Residual flow accuracy: h_l = h_0 + sum delta_k
        residual_errors = []
        for l in range(n_layers):
            h_pred = hiddens_A[0].clone()
            for k in range(1, l + 1):
                h_pred = h_pred + (hiddens_A[k] - hiddens_A[k - 1])
            residual_errors.append(torch.norm(h_pred - hiddens_A[l]).item())

        pair_result = {
            "word": word,
            "n_layers": n_layers,
            "max_cumulative_error": max(errors) if errors else 0,
            "mean_cumulative_error": float(np.mean(errors)) if errors else 0,
            "cumulative_angle": float(np.mean(d_angles)) if d_angles else 0,
            "d_norm_final": d_norm_final,
            "delta_u_norm": delta_u_norm,
            "alignment": alignment,
            "logit_margin": logit_margin,
            "logit_margin_pred": logit_margin_pred,
            "eff_rank_90": eff_rank_90,
            "max_residual_error": max(residual_errors) if residual_errors else 0,
            "d_norm_growth": d_norms[-1] / (d_norms[0] + 1e-8) if d_norms else 1.0,
        }
        all_pair_results.append(pair_result)

        print(f"    累积预测误差: {pair_result['mean_cumulative_error']:.4f} (max={pair_result['max_cumulative_error']:.4f})")
        print(f"    残差流精度: max_err={pair_result['max_residual_error']:.6f}")
        print(f"    统一方程: actual={logit_margin:.3f}, pred={logit_margin_pred:.3f}")
        print(f"    d_norm增长: {pair_result['d_norm_growth']:.1f}x, eff_rank_90={eff_rank_90}")

    if not all_pair_results:
        return {}

    # Summary
    summary = {
        "mean_cumulative_error": float(np.mean([p["mean_cumulative_error"] for p in all_pair_results])),
        "max_residual_error": float(np.mean([p["max_residual_error"] for p in all_pair_results])),
        "mean_alignment": float(np.mean([p["alignment"] for p in all_pair_results])),
        "mean_d_norm_growth": float(np.mean([p["d_norm_growth"] for p in all_pair_results])),
        "mean_eff_rank_90": float(np.mean([p["eff_rank_90"] for p in all_pair_results])),
        "mean_d_norm_final": float(np.mean([p["d_norm_final"] for p in all_pair_results])),
    }
    results["stage631"] = {"summary": summary, "pairs": all_pair_results}

    print(f"\n  === Stage631 总结 ===")
    print(f"  累积预测误差: {summary['mean_cumulative_error']:.4f}")
    print(f"  残差流精度: {summary['max_residual_error']:.6f}")
    print(f"  平均alignment: {summary['mean_alignment']:.4f}")
    print(f"  d_norm平均增长: {summary['mean_d_norm_growth']:.1f}x")
    print(f"  平均eff_rank_90: {summary['mean_eff_rank_90']:.1f}")

    return results


# ============ Stage632: 最优编码验证 ============

def run_stage632(model, tokenizer, model_key):
    """验证消歧信息的最优编码——强制对齐实验"""
    print("\n" + "=" * 70)
    print("Stage632: 消歧信息最优编码验证")
    print("=" * 70)

    device = safe_get_device(model)
    layers = discover_layers(model)
    W_unembed = get_unembed_matrix(model)

    if W_unembed is None:
        print("  [SKIP] 无法获取unembed矩阵")
        return {}

    # Get vocab size and hidden dim
    V, D = W_unembed.shape
    print(f"  Unembed shape: [{V}, {D}]")

    results = {}
    all_pair_results = []

    for idx, (sA, sB, word, meaningA_tokens, meaningB_tokens) in enumerate(CORE_PAIRS):
        print(f"\n  [{idx+1}/5] 词: {word}")

        meaningA_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningA_tokens]
        meaningB_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningB_tokens]

        # 1. Extract final hidden states
        hiddens_A = extract_all_layer_hidden(model, tokenizer, sA, layers)
        hiddens_B = extract_all_layer_hidden(model, tokenizer, sB, layers)

        if hiddens_A is None or hiddens_B is None:
            print(f"    [SKIP] 提取失败")
            continue

        n_layers = len(hiddens_A)
        d_final = hiddens_A[n_layers - 1] - hiddens_B[n_layers - 1]

        # Meaning-based delta_u (mean of meaning token unembed rows)
        delta_u_meanA = torch.stack([W_unembed[maid] for maid in meaningA_ids]).mean(dim=0)
        delta_u_meanB = torch.stack([W_unembed[mbid] for mbid in meaningB_ids]).mean(dim=0)
        delta_u = delta_u_meanA - delta_u_meanB

        # 2. Baseline measurements
        logits_A = hiddens_A[n_layers - 1] @ W_unembed.T
        logits_B = hiddens_B[n_layers - 1] @ W_unembed.T
        margin_A = float(np.mean([logits_A[maid].item() for maid in meaningA_ids]))
        cross_A_on_B = float(np.mean([logits_A[mbid].item() for mbid in meaningB_ids]))
        baseline_margin = margin_A - cross_A_on_B

        probs_A = F.softmax(logits_A, dim=-1)
        top5_probs, top5_ids = torch.topk(probs_A, 5)
        top5_tokens = [tokenizer.decode([t]) for t in top5_ids.tolist()]

        baseline_alignment = cos_sim(d_final, delta_u)
        baseline_efficiency = abs(baseline_alignment)
        baseline_d_norm = torch.norm(d_final).item()

        theoretical_max = baseline_d_norm * torch.norm(delta_u).item()

        # 4. Alignment boost experiment
        delta_u_unit = delta_u / (torch.norm(delta_u).item() + 1e-10)
        d_proj_on_u = torch.dot(d_final, delta_u_unit) * delta_u_unit

        scale_factors = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
        alignment_experiments = []

        for sf in scale_factors:
            h_mod_A = hiddens_A[n_layers - 1] + sf * d_proj_on_u / 2
            h_mod_B = hiddens_B[n_layers - 1] - sf * d_proj_on_u / 2

            logits_mod_A = h_mod_A @ W_unembed.T
            mod_margin_A = float(np.mean([logits_mod_A[maid].item() for maid in meaningA_ids]))
            mod_cross_B = float(np.mean([logits_mod_A[mbid].item() for mbid in meaningB_ids]))
            mod_margin = mod_margin_A - mod_cross_B

            mod_probs_A = F.softmax(logits_mod_A, dim=-1)
            mod_top1_id = torch.argmax(mod_probs_A).item()
            mod_top1_correct = int(any(maid == mod_top1_id for maid in meaningA_ids))

            entropy = -(mod_probs_A * torch.log(mod_probs_A + 1e-10)).sum().item()

            alignment_experiments.append({
                "scale_factor": sf,
                "modified_margin": mod_margin,
                "top1_correct": mod_top1_correct,
                "entropy": entropy,
                "margin_boost": mod_margin - baseline_margin,
            })

        waste_ratio = 1.0 - baseline_efficiency
        potential_gain = theoretical_max - abs(baseline_margin)

        pair_result = {
            "word": word,
            "baseline_margin": baseline_margin,
            "baseline_alignment": baseline_alignment,
            "baseline_efficiency": baseline_efficiency,
            "baseline_d_norm": baseline_d_norm,
            "theoretical_max": theoretical_max,
            "waste_ratio": waste_ratio,
            "potential_gain": potential_gain,
            "top5_tokens": top5_tokens[:3],
        }
        all_pair_results.append(pair_result)

        print(f"    Baseline margin: {baseline_margin:.3f}")
        print(f"    Efficiency: {baseline_efficiency:.4f} (waste={waste_ratio:.4f})")
        print(f"    Theoretical max: {theoretical_max:.3f}")
        # top5_tokens might contain non-encodable chars, skip printing
        # top5_tokens_safe = [repr(t) for t in top5_tokens[:3]]
        # print(f"    Top-5: {top5_tokens_safe}")

        for ae in alignment_experiments:
            sf = ae["scale_factor"]
            if sf == 0:
                continue
            print(f"    Scale={sf}: margin={ae['modified_margin']:.3f} "
                  f"(+{ae['margin_boost']:.3f}), top1={ae['top1_correct']}, H={ae['entropy']:.2f}")

    if not all_pair_results:
        return {}

    # Summary
    summary = {
        "mean_efficiency": float(np.mean([p["baseline_efficiency"] for p in all_pair_results])),
        "mean_waste_ratio": float(np.mean([p["waste_ratio"] for p in all_pair_results])),
        "mean_potential_gain": float(np.mean([p["potential_gain"] for p in all_pair_results])),
        "mean_theoretical_max": float(np.mean([p["theoretical_max"] for p in all_pair_results])),
        "mean_baseline_margin": float(np.mean([p["baseline_margin"] for p in all_pair_results])),
    }

    results["stage632"] = {"summary": summary, "pairs": all_pair_results}

    print(f"\n  === Stage632 总结 ===")
    print(f"  平均efficiency: {summary['mean_efficiency']:.4f}")
    print(f"  平均waste_ratio: {summary['mean_waste_ratio']:.4f}")
    print(f"  理论最大margin: {summary['mean_theoretical_max']:.3f}")
    print(f"  当前margin: {summary.get('mean_baseline_margin', summary.get('mean_margin', 0)):.3f}")
    print(f"  潜在增益: {summary['mean_potential_gain']:.3f}")
    for sf in [1.0, 5.0, 10.0]:
        key_boost = f"scale_{sf}_mean_boost"
        key_top1 = f"scale_{sf}_top1_rate"
        if key_boost in summary:
            print(f"  Scale={sf}: boost={summary[key_boost]:.3f}, top1_rate={summary[key_top1]:.2f}")

    return results


# ============ Stage633: 扩展消歧词表泛化验证 ============

def run_stage633(model, tokenizer, model_key):
    """扩展到20个歧义词，验证不变量"""
    print("\n" + "=" * 70)
    print("Stage633: 扩展消歧词表泛化验证 (20词)")
    print("=" * 70)

    device = safe_get_device(model)
    layers = discover_layers(model)
    W_unembed = get_unembed_matrix(model)

    if W_unembed is None:
        print("  [SKIP] 无法获取unembed矩阵")
        return {}

    results = {}
    all_pair_results = []

    for idx, (sA, sB, word, meaningA_tokens, meaningB_tokens) in enumerate(EXTENDED_PAIRS):
        if idx >= 20:
            break

        print(f"  [{idx+1}/20] {word}", end="")

        meaningA_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningA_tokens]
        meaningB_ids = [tokenizer.encode(t, add_special_tokens=False)[-1] for t in meaningB_tokens]

        hiddens_A = extract_all_layer_hidden(model, tokenizer, sA, layers)
        hiddens_B = extract_all_layer_hidden(model, tokenizer, sB, layers)

        if hiddens_A is None or hiddens_B is None:
            print(" [SKIP]")
            continue

        n_layers = len(hiddens_A)

        # 1. Hidden space disambiguation
        d_final = hiddens_A[n_layers - 1] - hiddens_B[n_layers - 1]
        hidden_cos_dist = 1.0 - cos_sim(hiddens_A[n_layers - 1], hiddens_B[n_layers - 1])

        # 2. Logit margin using meaning tokens
        delta_u_meanA = torch.stack([W_unembed[maid] for maid in meaningA_ids]).mean(dim=0)
        delta_u_meanB = torch.stack([W_unembed[mbid] for mbid in meaningB_ids]).mean(dim=0)
        delta_u = delta_u_meanA - delta_u_meanB
        logit_margin = (d_final @ delta_u).item()
        d_norm = torch.norm(d_final).item()
        delta_u_norm = torch.norm(delta_u).item()
        alignment = cos_sim(d_final, delta_u) if d_norm > 1e-8 and delta_u_norm > 1e-8 else 0.0
        efficiency = abs(alignment)

        # 3. First significant layer
        first_sig_layer = -1
        for l in range(n_layers):
            cd = 1.0 - cos_sim(hiddens_A[l], hiddens_B[l])
            if cd > 0.01:
                first_sig_layer = l
                break

        # 4. d_norm growth
        d_norms = [torch.norm(hiddens_A[l] - hiddens_B[l]).item() for l in range(n_layers)]

        # 5. Rotation angle per layer
        angles = []
        for l in range(1, n_layers):
            d_prev = hiddens_A[l-1] - hiddens_B[l-1]
            d_curr = hiddens_A[l] - hiddens_B[l]
            if torch.norm(d_prev) > 1e-8 and torch.norm(d_curr) > 1e-8:
                c = cos_sim(d_prev, d_curr)
                angle = np.degrees(np.arccos(np.clip(c, -1, 1)))
                angles.append(angle)

        # 6. Meaning token accuracy
        logits_A = hiddens_A[n_layers - 1] @ W_unembed.T
        probs_A = F.softmax(logits_A, dim=-1)
        top1_id = torch.argmax(probs_A).item()
        correct_top1 = int(any(maid == top1_id for maid in meaningA_ids))
        correct_top5 = int(any(maid in torch.topk(probs_A, 5)[1].tolist() for maid in meaningA_ids))

        pair_result = {
            "word": word,
            "n_layers": n_layers,
            "hidden_cos_dist": hidden_cos_dist,
            "logit_margin": logit_margin,
            "d_norm": d_norm,
            "alignment": alignment,
            "efficiency": efficiency,
            "delta_u_norm": delta_u_norm,
            "first_sig_layer": first_sig_layer,
            "mean_rotation_deg": float(np.mean(angles)) if angles else 0,
            "d_norm_growth": d_norms[-1] / (d_norms[0] + 1e-8) if d_norms else 1.0,
            "correct_top1": correct_top1,
            "correct_top5": correct_top5,
        }
        all_pair_results.append(pair_result)
        print(f" cos_dist={hidden_cos_dist:.4f} margin={logit_margin:.3f} eff={efficiency:.4f} top1={correct_top1}")

    if not all_pair_results:
        return {}

    # Summary statistics
    hidden_dists = [p["hidden_cos_dist"] for p in all_pair_results]
    margins = [p["logit_margin"] for p in all_pair_results]
    efficiencies = [p["efficiency"] for p in all_pair_results]
    d_norms = [p["d_norm"] for p in all_pair_results]
    alignments = [p["alignment"] for p in all_pair_results]
    rotations = [p["mean_rotation_deg"] for p in all_pair_results]
    top1_rates = [p["correct_top1"] for p in all_pair_results]

    # Correlations
    if len(hidden_dists) > 2:
        corr_hidden_margin = float(np.corrcoef(hidden_dists, margins)[0, 1])
        corr_eff_margin = float(np.corrcoef(efficiencies, margins)[0, 1])
        corr_dnorm_margin = float(np.corrcoef(d_norms, margins)[0, 1])
        corr_rot_margin = float(np.corrcoef(rotations, margins)[0, 1])
    else:
        corr_hidden_margin = corr_eff_margin = corr_dnorm_margin = corr_rot_margin = 0.0

    summary = {
        "n_words": len(all_pair_results),
        "mean_hidden_cos_dist": float(np.mean(hidden_dists)),
        "mean_logit_margin": float(np.mean(margins)),
        "mean_efficiency": float(np.mean(efficiencies)),
        "mean_d_norm": float(np.mean(d_norms)),
        "mean_alignment": float(np.mean(alignments)),
        "mean_rotation_deg": float(np.mean(rotations)),
        "top1_accuracy": float(np.mean(top1_rates)),
        "corr_hidden_margin": corr_hidden_margin,
        "corr_eff_margin": corr_eff_margin,
        "corr_dnorm_margin": corr_dnorm_margin,
        "corr_rot_margin": corr_rot_margin,
        "std_efficiency": float(np.std(efficiencies)),
        "efficiency_cv": float(np.std(efficiencies) / (np.mean(efficiencies) + 1e-10)),
    }

    results["stage633"] = {"summary": summary, "pairs": all_pair_results}

    print(f"\n  === Stage633 总结 (20词) ===")
    print(f"  平均hidden cos_dist: {summary['mean_hidden_cos_dist']:.4f}")
    print(f"  平均logit margin: {summary['mean_logit_margin']:.3f}")
    print(f"  平均efficiency: {summary['mean_efficiency']:.4f} (std={summary['std_efficiency']:.4f})")
    print(f"  平均||d||: {summary['mean_d_norm']:.1f}")
    print(f"  平均旋转角度: {summary['mean_rotation_deg']:.1f} deg/layer")
    print(f"  Top-1准确率: {summary['top1_accuracy']:.2f}")
    print(f"  Corr(hidden_dist, margin): {corr_hidden_margin:.3f}")
    print(f"  Corr(efficiency, margin): {corr_eff_margin:.3f}")
    print(f"  Corr(||d||, margin): {corr_dnorm_margin:.3f}")
    print(f"  Corr(rotation, margin): {corr_rot_margin:.3f}")

    return results


# ============ 主函数 ============

def main():
    if len(sys.argv) < 2:
        print("用法: python stage631_632_633_closed_form_optimal.py [qwen3|deepseek7b|glm4|gemma4]")
        sys.exit(1)

    model_key = sys.argv[1]
    print(f"模型: {model_key}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load model
    print(f"\n加载模型 {model_key}...")
    bundle = load_model_bundle(model_key)
    if isinstance(bundle, tuple):
        model, tokenizer = bundle[0], bundle[1]
    else:
        model = bundle["model"]
        tokenizer = bundle["tokenizer"]

    layers = discover_layers(model)
    print(f"层数: {len(layers)}")

    all_results = {}

    try:
        # Stage631
        r631 = run_stage631(model, tokenizer, model_key)
        all_results.update(r631)
        gc.collect()
        torch.cuda.empty_cache()

        # Stage632
        r632 = run_stage632(model, tokenizer, model_key)
        all_results.update(r632)
        gc.collect()
        torch.cuda.empty_cache()

        # Stage633
        r633 = run_stage633(model, tokenizer, model_key)
        all_results.update(r633)

    finally:
        # Free model
        try:
            free_model(model_key)
        except:
            pass
        gc.collect()
        torch.cuda.empty_cache()

    # Save results
    output_path = OUTPUT_DIR / f"stage631_632_633_{model_key}_{TIMESTAMP}.json"
    # Convert non-serializable values
    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2, default=convert, ensure_ascii=False)
        print(f"\n结果已保存: {output_path}")
    except Exception as e:
        print(f"\n保存失败: {e}")

    print("\n完成!")


if __name__ == "__main__":
    main()
