"""
Phase 1: 表征涌现与参数级语言编码综合分析
================================================
从参数编码级别系统性分析语言机制:
  1. Embedding 结构分析 (SVD/PCA/聚类)
  2. 残差流几何演化 (逐层范数/角度/有效秩)
  3. 注意力头信息路由 (功能分类/信息流图谱)
  4. FFN 变换类型分类 (存储型/变换型/门控型)

使用模型: GPT-2 (快速迭代) + Qwen2.5-0.5B (跨架构验证)
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# ============================================================
# 配置
# ============================================================
OUTPUT_DIR = Path("d:/ai2050/TransformerLens-Project/tests/glm5/phase1_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 测试句子集 - 覆盖语法/逻辑/风格/语义四个维度
TEST_SENTENCES = {
    "syntax": {
        "svo_active": [
            "The cat catches the mouse.",
            "The dog chases the ball.",
            "The student reads the book.",
            "The farmer grows the wheat.",
            "The artist paints the canvas.",
        ],
        "svo_passive": [
            "The mouse is caught by the cat.",
            "The ball is chased by the dog.",
            "The book is read by the student.",
            "The wheat is grown by the farmer.",
            "The canvas is painted by the artist.",
        ],
        "subject_verb_agree_singular": [
            "The cat is sleeping.",
            "A bird flies south.",
            "My friend works hard.",
            "This problem seems difficult.",
            "Every child needs love.",
        ],
        "subject_verb_agree_plural": [
            "The cats are sleeping.",
            "The birds fly south.",
            "My friends work hard.",
            "These problems seem difficult.",
            "All children need love.",
        ],
        "embedding_clause": [
            "I know that he left early.",
            "She said that the rain will stop.",
            "They believe that the answer is correct.",
            "We hope that the project succeeds.",
            "He denied that he broke the window.",
        ],
    },
    "logic": {
        "syllogism_premise": [
            "All birds can fly. Penguins are birds.",
            "All mammals are warm-blooded. Whales are mammals.",
            "All roses are flowers. This is a rose.",
            "All metals conduct electricity. Copper is a metal.",
            "All students must study. She is a student.",
        ],
        "syllogism_conclusion_valid": [
            "Therefore penguins can fly.",
            "Therefore whales are warm-blooded.",
            "Therefore this is a flower.",
            "Therefore copper conducts electricity.",
            "Therefore she must study.",
        ],
        "syllogism_conclusion_invalid": [
            "Therefore penguins cannot fly.",
            "Therefore whales are cold-blooded.",
            "Therefore this is not a flower.",
            "Therefore copper does not conduct electricity.",
            "Therefore she does not need to study.",
        ],
        "negation_affirm": [
            "The sky is blue.",
            "Water boils at 100 degrees.",
            "The Earth orbits the Sun.",
            "Gravity pulls objects down.",
            "Light travels faster than sound.",
        ],
        "negation_negated": [
            "The sky is not blue.",
            "Water does not boil at 100 degrees.",
            "The Earth does not orbit the Sun.",
            "Gravity does not pull objects down.",
            "Light does not travel faster than sound.",
        ],
        "conditional_if": [
            "If it rains, then the ground gets wet.",
            "If you study hard, then you pass the exam.",
            "If the temperature drops, then water freezes.",
            "If the bridge collapses, then traffic stops.",
            "If the signal is red, then cars must stop.",
        ],
    },
    "style": {
        "formal": [
            "The aforementioned research demonstrates significant findings.",
            "It is imperative that appropriate measures be implemented.",
            "The committee has reached a consensus regarding this matter.",
            "Further investigation is warranted to substantiate these claims.",
            "The implications of this study extend beyond its immediate scope.",
        ],
        "casual": [
            "That research shows some really cool stuff.",
            "We totally need to do something about this.",
            "Everyone agreed on what to do about it.",
            "We should look into this more to be sure.",
            "This study matters for a bunch of reasons.",
        ],
        "academic": [
            "The empirical evidence suggests a statistically significant correlation.",
            "This finding is consistent with prior theoretical frameworks.",
            "The methodology employed follows established protocols in the field.",
            "These results warrant further replication across diverse populations.",
            "The theoretical implications challenge existing paradigms.",
        ],
        "narrative": [
            "She walked into the room and everything changed.",
            "The old man smiled, remembering days long past.",
            "Rain hammered against the window like tiny fists.",
            "He couldn't believe what he saw on the other side.",
            "The forest grew darker with every step she took.",
        ],
    },
    "semantics": {
        "animals": [
            "The cat purrs softly on the mat.",
            "Dogs are known for their loyalty.",
            "Eagles soar high above the mountains.",
            "Fish swim in the deep blue ocean.",
            "Elephants are the largest land animals.",
        ],
        "food": [
            "Apples are rich in vitamins and fiber.",
            "Fresh bread smells wonderful in the morning.",
            "Sushi is a traditional Japanese dish.",
            "Chocolate contains compounds that boost mood.",
            "Rice is a staple food in many countries.",
        ],
        "tools": [
            "The hammer drives nails into wood.",
            "Scissors are used for cutting paper.",
            "The wrench tightens bolts and nuts.",
            "A compass helps find direction.",
            "The saw cuts through thick timber.",
        ],
        "abstract": [
            "Justice requires fairness and equality.",
            "Freedom carries both rights and responsibilities.",
            "Truth is often more complex than it appears.",
            "Knowledge grows through continuous inquiry.",
            "Beauty exists in many unexpected forms.",
        ],
        "numbers_quantity": [
            "There are seven days in a week.",
            "The team scored three goals in the match.",
            "Approximately fifty people attended the event.",
            "The building has twenty floors.",
            "She counted exactly one hundred coins.",
        ],
    },
}


def load_model(model_name="gpt2"):
    """加载模型和tokenizer，返回model和config"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading model: {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # 精确分析用float32
        device_map="auto",
    )
    model.eval()

    # 获取模型配置信息
    config = {
        "model_name": model_name,
        "hidden_size": model.config.hidden_size,
        "num_layers": model.config.n_layer if hasattr(model.config, 'n_layer') else model.config.num_hidden_layers,
        "num_heads": model.config.n_head if hasattr(model.config, 'n_head') else model.config.num_attention_heads,
        "head_dim": model.config.hidden_size // (model.config.n_head if hasattr(model.config, 'n_head') else model.config.num_attention_heads),
        "vocab_size": model.config.vocab_size,
        "num_parameters": sum(p.numel() for p in model.parameters()),
    }
    print(f"  hidden_size={config['hidden_size']}, layers={config['num_layers']}, "
          f"heads={config['num_heads']}, params={config['num_parameters']:,}")
    return model, tokenizer, config


# ============================================================
# 1. Embedding 结构分析
# ============================================================
def analyze_embedding_structure(model, tokenizer, config):
    """分析 Token Embedding 和 Position Embedding 的数学结构"""
    print("\n" + "=" * 60)
    print("1. Embedding Structure Analysis")
    print("=" * 60)

    results = {}

    # --- 1a. Token Embedding SVD 分析 ---
    W_embed = model.get_input_embeddings().weight.data.cpu().numpy()  # [vocab_size, hidden_size]
    results["token_embed_shape"] = list(W_embed.shape)

    # SVD 分解
    U, S, Vt = np.linalg.svd(W_embed, full_matrices=False)
    results["singular_values_top20"] = S[:20].tolist()
    results["singular_values_cumvar_top20"] = np.cumsum(S[:20] ** 2) / np.sum(S ** 2).tolist()

    # 分析前几个主成分对应什么
    # 对每个主成分方向，找投影最大和最小的token
    top_components = []
    for i in range(min(5, len(S))):
        direction = Vt[i]  # 第i主成分方向
        projections = W_embed @ direction  # 每个token在该方向上的投影
        top_indices = np.argsort(projections)[-10:][::-1]
        bottom_indices = np.argsort(projections)[:10]
        top_tokens = [tokenizer.decode([idx]).strip() for idx in top_indices]
        bottom_tokens = [tokenizer.decode([idx]).strip() for idx in bottom_indices]
        explained_var = (S[i] ** 2) / np.sum(S ** 2)
        top_components.append({
            "component": i,
            "singular_value": float(S[i]),
            "explained_variance": float(explained_var),
            "top_tokens": top_tokens,
            "bottom_tokens": bottom_tokens,
        })
    results["top_components"] = top_components

    # --- 1b. Embedding 聚类分析 ---
    # 选取有意义的token子集进行聚类
    target_words = [
        "cat", "dog", "bird", "fish", "horse",  # 动物
        "apple", "bread", "rice", "fish", "egg",  # 食物
        "hammer", "knife", "saw", "key", "pen",   # 工具
        "run", "walk", "eat", "think", "sleep",   # 动词
        "big", "small", "fast", "slow", "hot",    # 形容词
        "the", "a", "is", "are", "not",           # 功能词
    ]
    target_tokens = []
    target_embeddings = []
    target_categories = []
    cat_sizes = {"animal": 5, "food": 5, "tool": 5, "verb": 5, "adj": 5, "func": 5}

    for word in target_words:
        ids = tokenizer.encode(word, add_special_tokens=False)
        if len(ids) == 1:
            target_tokens.append(word)
            target_embeddings.append(W_embed[ids[0]])
            for cat, size in cat_sizes.items():
                if len(target_categories) < sum(list(cat_sizes.values())[:list(cat_sizes.keys()).index(cat) + 1]):
                    target_categories.append(cat)
                    break

    if len(target_embeddings) > 1:
        embeddings = np.array(target_embeddings)
        # 计算词类内的平均距离 vs 词类间的平均距离
        intra_dists = []
        inter_dists = []
        unique_cats = list(set(target_categories))
        for cat in unique_cats:
            cat_embs = embeddings[np.array(target_categories) == cat]
            if len(cat_embs) > 1:
                from itertools import combinations
                for i, j in combinations(range(len(cat_embs)), 2):
                    intra_dists.append(np.linalg.norm(cat_embs[i] - cat_embs[j]))

        for i, cat1 in enumerate(unique_cats):
            for cat2 in unique_cats[i + 1:]:
                embs1 = embeddings[np.array(target_categories) == cat1]
                embs2 = embeddings[np.array(target_categories) == cat2]
                for e1 in embs1:
                    for e2 in embs2:
                        inter_dists.append(np.linalg.norm(e1 - e2))

        results["embedding_clustering"] = {
            "num_tokens_analyzed": len(target_tokens),
            "categories": unique_cats,
            "mean_intra_category_dist": float(np.mean(intra_dists)) if intra_dists else None,
            "mean_inter_category_dist": float(np.mean(inter_dists)) if inter_dists else None,
            "intra_over_inter_ratio": float(np.mean(intra_dists) / np.mean(inter_dists)) if intra_dists and inter_dists else None,
        }

    # --- 1c. Embedding-Unembedding 对称性分析 ---
    W_unembed = model.get_output_embeddings().weight.data.cpu().numpy()
    # 注意: LM head 可能与 embedding 共享权重
    embed_unembed_cosine = np.dot(W_embed.flatten(), W_unembed.flatten()) / (
        np.linalg.norm(W_embed.flatten()) * np.linalg.norm(W_unembed.flatten())
    )
    results["embed_unembed_cosine_similarity"] = float(embed_unembed_cosine)

    # 各层的偏差量（如果共享权重则为0）
    if W_embed.shape == W_unembed.shape:
        diff = W_embed - W_unembed
        results["embed_unembed_diff_norm"] = float(np.linalg.norm(diff))
        results["embed_unembed_diff_relative"] = float(np.linalg.norm(diff) / np.linalg.norm(W_embed))
    else:
        results["embed_unembed_shape_mismatch"] = f"{W_embed.shape} vs {W_unembed.shape}"

    print(f"  Top singular values: {[f'{s:.1f}' for s in S[:5]]}")
    print(f"  CumVar(top5): {results['singular_values_cumvar_top20'][4]:.4f}")
    print(f"  Embed-Unembed cosine: {embed_unembed_cosine:.4f}")
    if "intra_over_inter_ratio" in results.get("embedding_clustering", {}):
        print(f"  Clustering intra/inter ratio: {results['embedding_clustering']['intra_over_inter_ratio']:.4f}")

    return results


# ============================================================
# 2. 残差流几何演化分析
# ============================================================
def analyze_residual_stream_geometry(model, tokenizer, config):
    """追踪残差流在每一层的几何特征变化"""
    print("\n" + "=" * 60)
    print("2. Residual Stream Geometry Analysis")
    print("=" * 60)

    # 选取代表性句子
    test_cases = {
        "syntax_svo": "The cat catches the mouse.",
        "syntax_passive": "The mouse is caught by the cat.",
        "logic_syllogism": "All birds can fly. Penguins are birds.",
        "logic_negation": "The sky is not blue.",
        "style_formal": "The aforementioned research demonstrates significant findings.",
        "style_casual": "That research shows some really cool stuff.",
        "semantic_animal": "The cat purrs softly on the mat.",
        "semantic_abstract": "Justice requires fairness and equality.",
    }

    results = {}
    all_residuals = defaultdict(list)  # layer -> list of residual vectors

    hooks = []
    residual_cache = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            # 对于 GPT-2 的 transformer block，output 是 tuple (hidden_state, ...)
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            residual_cache[layer_idx] = hidden.detach().cpu()
        return hook_fn

    # 注册 hooks
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2
        for i, block in enumerate(model.transformer.h):
            hooks.append(block.register_forward_hook(make_hook(i)))
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Qwen
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))

    # 运行前向传播
    model.eval()
    with torch.no_grad():
        for case_name, sentence in test_cases.items():
            inputs = tokenizer(sentence, return_tensors="pt", truncation=True, max_length=64)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            _ = model(**inputs)

            layer_data = []
            for layer_idx in sorted(residual_cache.keys()):
                residual = residual_cache[layer_idx]  # [1, seq_len, hidden_size]
                residual = residual[0]  # [seq_len, hidden_size]

                # 计算几何特征
                norm = float(torch.norm(residual, dim=-1).mean())  # 平均范数
                mean_vec = residual.mean(dim=0)  # [hidden_size] 平均向量
                centered = residual - mean_vec
                cov = (centered.T @ centered) / (residual.shape[0] - 1)

                # 有效秩
                eigenvalues = torch.linalg.eigvalsh(cov.to(torch.float32))
                eigenvalues = eigenvalues[eigenvalues > 1e-6]
                effective_rank = float(torch.sum(eigenvalues) / torch.max(eigenvalues)) if len(eigenvalues) > 0 else 0.0

                # 信号维度 (近似)
                effective_dim = float(torch.sum(eigenvalues / eigenvalues.sum() > 0.01))

                layer_data.append({
                    "layer": layer_idx,
                    "mean_norm": norm,
                    "effective_rank": effective_rank,
                    "effective_dim": effective_dim,
                    "max_eigenvalue": float(eigenvalues.max()) if len(eigenvalues) > 0 else 0.0,
                })

            results[case_name] = {
                "sentence": sentence,
                "num_tokens": residual_cache[0].shape[1] if 0 in residual_cache else 0,
                "layer_data": layer_data,
            }

    # 移除 hooks
    for h in hooks:
        h.remove()

    # 计算跨句子对的方向变化
    pair_results = {}
    pairs = [
        ("syntax_svo", "syntax_passive", "syntax_active_vs_passive"),
        ("style_formal", "style_casual", "style_formal_vs_casual"),
        ("logic_syllogism", "logic_negation", "logic_vs_negation"),
    ]
    for k1, k2, pair_name in pairs:
        if k1 in results and k2 in results:
            angles = []
            for ld1, ld2 in zip(results[k1]["layer_data"], results[k2]["layer_data"]):
                # 用层间方向差异度量
                angles.append(abs(ld1["mean_norm"] - ld2["mean_norm"]))
            pair_results[pair_name] = {
                "case1": k1,
                "case2": k2,
                "layer_norm_diffs": angles,
                "max_diff_layer": int(np.argmax(angles)) if angles else -1,
            }
    results["pairwise_analysis"] = pair_results

    # 打印摘要
    for case_name, data in results.items():
        if "layer_data" in data:
            ld = data["layer_data"]
            norms = [d["mean_norm"] for d in ld]
            ranks = [d["effective_rank"] for d in ld]
            print(f"  [{case_name}] norm: {norms[0]:.2f} → {norms[-1]:.2f}, "
                  f"rank: {ranks[0]:.1f} → {ranks[-1]:.1f}")

    return results


# ============================================================
# 3. 注意力头信息路由分析
# ============================================================
def analyze_attention_routing(model, tokenizer, config):
    """分析注意力头的路由模式和信息流"""
    print("\n" + "=" * 60)
    print("3. Attention Head Routing Analysis")
    print("=" * 60)

    num_layers = config["num_layers"]
    num_heads = config["num_heads"]
    head_dim = config["head_dim"]

    # 测试用句对
    routing_tests = {
        "prev_token": {
            "description": "Previous Token Head: high attention to position-1",
            "sentences": [
                "The cat sat on the mat and looked around.",
                "A beautiful sunset painted the sky with golden light.",
                "She carefully opened the old wooden door slowly.",
            ],
        },
        "induction": {
            "description": "Induction Head: high attention to previous occurrence",
            "sentences": [
                "The cat sat on the mat. The dog sat on the mat.",
                "Harry went to school. Hermione went to school.",
                "One plus one equals two. Two plus two equals four.",
            ],
        },
        "syntax_dependency": {
            "description": "Syntax dependency: subject-verb, modifier-head",
            "sentence_pairs": [
                ("The tall building stands in the city center.",
                 "Building tall stands in the city center."),  # 语序打乱
            ],
        },
        "logic_flow": {
            "description": "Logic flow: premise to conclusion",
            "sentence_pairs": [
                ("All birds can fly. Penguins are birds. So penguins can fly.",
                 "All birds can fly. Penguins are birds. So penguins cannot fly."),
            ],
        },
        "style_shift": {
            "description": "Style sensitivity: formal vs casual",
            "sentence_pairs": [
                ("The committee has reached a consensus regarding this matter.",
                 "Everyone agreed on what to do about it."),
            ],
        },
    }

    results = {}
    attn_cache = {}

    # 注册 attention hooks
    hooks = []

    def make_attn_hook(layer_idx):
        def hook_fn(module, input, output):
            # 尝试提取注意力权重
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[-1]  # 通常是 (batch, heads, seq, seq)
                if attn_weights is not None and isinstance(attn_weights, torch.Tensor):
                    attn_cache[layer_idx] = attn_weights.detach().cpu()
            else:
                # 有些模型不输出attention weights，需要从内部获取
                pass
        return hook_fn

    # 尝试获取注意力权重
    # GPT-2: model.transformer.h[i].attn
    # Qwen: model.model.layers[i].self_attn
    attn_modules = []
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        for i, block in enumerate(model.transformer.h):
            if hasattr(block, 'attn'):
                attn_modules.append((i, block.attn))
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'self_attn'):
                attn_modules.append((i, layer.self_attn))

    has_attention_weights = False

    # 方法1: 使用 model 的 output_attentions 参数
    try:
        model.config.output_attentions = True
        model.eval()

        with torch.no_grad():
            # 先测一个句子确认能获取attention
            test_input = tokenizer("The cat sat on the mat.", return_tensors="pt", truncation=True, max_length=64)
            test_input = {k: v.to(model.device) for k, v in test_input.items()}
            test_output = model(**test_input, output_attentions=True)

            if test_output.attentions is not None and len(test_output.attentions) > 0:
                has_attention_weights = True
                print(f"  Attention weights available: {len(test_output.attentions)} layers")
                print(f"  Per-layer shape: {test_output.attentions[0].shape}")
            else:
                print("  WARNING: output_attentions=True did not return attention weights")
    except Exception as e:
        print(f"  WARNING: Could not get attention weights: {e}")

    if not has_attention_weights:
        print("  Falling back to attention pattern extraction via hooks...")
        # 使用自定义方法提取注意力模式
        for layer_idx, attn_module in attn_modules:
            hooks.append(attn_module.register_forward_hook(make_attn_hook(layer_idx)))

    # 对每个测试运行分析
    for test_name, test_info in routing_tests.items():
        print(f"\n  --- {test_name}: {test_info['description']} ---")

        if "sentences" in test_info:
            # 单句分析
            for sent in test_info["sentences"]:
                layer_attention_profiles = []

                with torch.no_grad():
                    inputs = tokenizer(sent, return_tensors="pt", truncation=True, max_length=64)
                    inputs = {k: v.to(model.device) for k, v in inputs.items()}

                    if has_attention_weights:
                        output = model(**inputs, output_attentions=True)
                        attentions = output.attentions  # tuple of [batch, heads, seq, seq]
                    else:
                        attn_cache.clear()
                        _ = model(**inputs)
                        attentions = [attn_cache[i] for i in sorted(attn_cache.keys())]

                if attentions:
                    for layer_idx, attn in enumerate(attentions):
                        if attn is None:
                            continue
                        attn_np = attn[0].numpy()  # [heads, seq, seq]
                        seq_len = attn_np.shape[-1]

                        # 对每个头计算路由模式特征
                        head_profiles = []
                        for head_idx in range(attn_np.shape[0]):
                            a = attn_np[head_idx]  # [seq, seq]

                            # 路由模式指标
                            # 1. 对角线强度 (prev token)
                            diag_strength = np.mean(np.diag(a, k=-1)[:-1]) if seq_len > 1 else 0.0

                            # 2. 归纳强度 (当前token对上一个同token的注意力)
                            induction_strength = 0.0
                            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
                            token_ids = inputs["input_ids"][0].cpu().numpy()
                            for pos in range(seq_len):
                                current_token = token_ids[pos]
                                for prev_pos in range(pos):
                                    if token_ids[prev_pos] == current_token:
                                        induction_strength += a[pos, prev_pos]
                            induction_strength /= max(seq_len, 1)

                            # 3. 远程依赖强度 (距离>3的位置)
                            local_mask = np.zeros_like(a)
                            for i in range(seq_len):
                                for j in range(max(0, i - 3), min(seq_len, i + 4)):
                                    local_mask[i, j] = 1.0
                            remote_strength = np.sum(a * (1 - local_mask)) / max(np.sum(a), 1e-8)

                            # 4. 集中度 (entropy的逆)
                            row_entropy = -np.sum(a * np.log(a + 1e-10), axis=-1).mean()
                            max_entropy = np.log(seq_len)
                            concentration = 1.0 - row_entropy / max_entropy if max_entropy > 0 else 0.0

                            head_profiles.append({
                                "head": head_idx,
                                "diag_strength": float(diag_strength),
                                "induction_strength": float(induction_strength),
                                "remote_strength": float(remote_strength),
                                "concentration": float(concentration),
                            })

                        layer_attention_profiles.append({
                            "layer": layer_idx,
                            "heads": head_profiles,
                        })

                    results[f"{test_name}_{sent[:20]}"] = {
                        "test": test_name,
                        "sentence": sent,
                        "layers": layer_attention_profiles,
                    }

        elif "sentence_pairs" in test_info:
            # 句对对比分析
            for pair in test_info["sentence_pairs"]:
                sent_a, sent_b = pair
                pair_diffs = []

                with torch.no_grad():
                    inputs_a = tokenizer(sent_a, return_tensors="pt", truncation=True, max_length=64)
                    inputs_b = tokenizer(sent_b, return_tensors="pt", truncation=True, max_length=64)
                    inputs_a = {k: v.to(model.device) for k, v in inputs_a.items()}
                    inputs_b = {k: v.to(model.device) for k, v in inputs_b.items()}

                    if has_attention_weights:
                        output_a = model(**inputs_a, output_attentions=True)
                        output_b = model(**inputs_b, output_attentions=True)
                        att_a = output_a.attentions
                        att_b = output_b.attentions
                    else:
                        attn_cache.clear()
                        _ = model(**inputs_a)
                        att_a = [attn_cache[i] for i in sorted(attn_cache.keys())]
                        attn_cache.clear()
                        _ = model(**inputs_b)
                        att_b = [attn_cache[i] for i in sorted(attn_cache.keys())]

                if att_a and att_b:
                    for layer_idx, (a, b) in enumerate(zip(att_a, att_b)):
                        if a is None or b is None:
                            continue
                        # 对齐到相同长度
                        min_seq = min(a.shape[-1], b.shape[-1])
                        a_np = a[0, :, :min_seq, :min_seq].numpy()
                        b_np = b[0, :, :min_seq, :min_seq].numpy()

                        diff = np.abs(a_np - b_np).mean()
                        pair_diffs.append(float(diff))

                    results[f"{test_name}_{sent_a[:20]}"] = {
                        "test": test_name,
                        "sentence_a": sent_a,
                        "sentence_b": sent_b,
                        "layer_diffs": pair_diffs,
                        "max_diff_layer": int(np.argmax(pair_diffs)) if pair_diffs else -1,
                        "mean_diff": float(np.mean(pair_diffs)) if pair_diffs else 0.0,
                    }

                    print(f"    A: {sent_a[:40]}...")
                    print(f"    B: {sent_b[:40]}...")
                    print(f"    Mean attention diff: {results[f'{test_name}_{sent_a[:20]}']['mean_diff']:.6f}")
                    print(f"    Max diff at layer: {results[f'{test_name}_{sent_a[:20]}']['max_diff_layer']}")

    # 清理 hooks
    for h in hooks:
        h.remove()
    model.config.output_attentions = False

    # 汇总: 找到每个功能的最强头
    head_function_summary = defaultdict(list)
    for key, val in results.items():
        if "layers" in val:
            for layer_data in val["layers"]:
                for head_data in layer_data["heads"]:
                    head_id = f"L{layer_data['layer']}_H{head_data['head']}"
                    head_function_summary[head_id].append({
                        "test": val["test"],
                        **head_data,
                    })

    # 对每个头计算功能偏好
    head_rankings = {}
    for head_id, profiles in head_function_summary.items():
        avg_diag = np.mean([p["diag_strength"] for p in profiles])
        avg_induction = np.mean([p["induction_strength"] for p in profiles])
        avg_remote = np.mean([p["remote_strength"] for p in profiles])
        avg_conc = np.mean([p["concentration"] for p in profiles])

        # 功能标签
        scores = {
            "prev_token": avg_diag,
            "induction": avg_induction,
            "remote_dep": avg_remote,
            "focused": avg_conc,
        }
        dominant = max(scores, key=scores.get)

        head_rankings[head_id] = {
            **scores,
            "dominant_function": dominant,
            "num_tests": len(profiles),
        }

    results["head_function_summary"] = dict(head_rankings)

    # 打印Top头
    print(f"\n  Top heads by function:")
    for func in ["prev_token", "induction", "remote_dep"]:
        top_heads = sorted(
            [(h, v) for h, v in head_rankings.items() if v["dominant_function"] == func],
            key=lambda x: x[1][func], reverse=True
        )[:3]
        if top_heads:
            print(f"    {func}: {[(h, f'{v[func]:.4f}') for h, v in top_heads]}")

    return results


# ============================================================
# 4. FFN 变换类型分析
# ============================================================
def analyze_ffn_transformations(model, tokenizer, config):
    """分析FFN神经元的变换类型：存储型/变换型/门控型"""
    print("\n" + "=" * 60)
    print("4. FFN Transformation Type Analysis")
    print("=" * 60)

    results = {}

    # 获取FFN权重
    ffn_weights = []
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-2: each block has mlp (MLP)
        for i, block in enumerate(model.transformer.h):
            if hasattr(block, 'mlp') and hasattr(block.mlp, 'c_fc') and hasattr(block.mlp, 'c_proj'):
                W_in = block.mlp.c_fc.weight.data.cpu().numpy()   # [4d, d]
                W_out = block.mlp.c_proj.weight.data.cpu().numpy()  # [d, 4d]
                ffn_weights.append({
                    "layer": i,
                    "W_in": W_in,
                    "W_out": W_out.T,  # [4d, d] for consistency
                })
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Qwen: each layer has mlp
        for i, layer in enumerate(model.model.layers):
            if hasattr(layer, 'mlp'):
                # Qwen2 uses gate_proj, up_proj, down_proj
                if hasattr(layer.mlp, 'gate_proj') and hasattr(layer.mlp, 'down_proj'):
                    W_gate = layer.mlp.gate_proj.weight.data.cpu().numpy()  # [intermediate, hidden]
                    W_up = layer.mlp.up_proj.weight.data.cpu().numpy()
                    W_down = layer.mlp.down_proj.weight.data.cpu().numpy()  # [hidden, intermediate]
                    ffn_weights.append({
                        "layer": i,
                        "W_gate": W_gate,
                        "W_up": W_up,
                        "W_down": W_down,
                        "type": "gated",  # SwiGLU style
                    })
                elif hasattr(layer.mlp, 'fc1') and hasattr(layer.mlp, 'fc2'):
                    W_in = layer.mlp.fc1.weight.data.cpu().numpy()
                    W_out = layer.mlp.fc2.weight.data.cpu().numpy()
                    ffn_weights.append({
                        "layer": i,
                        "W_in": W_in,
                        "W_out": W_out.T,
                    })

    if not ffn_weights:
        print("  WARNING: Could not extract FFN weights")
        results["error"] = "Could not extract FFN weights"
        return results

    print(f"  Found {len(ffn_weights)} FFN layers")

    # 对每个FFN层分析神经元类型
    layer_results = []
    global_transform_counts = {"storage": 0, "rotator": 0, "projector": 0, "amplifier": 0, "other": 0}
    total_neurons = 0

    for ffn_info in ffn_weights:
        layer_idx = ffn_info["layer"]
        layer_data = {
            "layer": layer_idx,
            "transform_counts": {},
        }

        if ffn_info.get("type") == "gated":
            # SwiGLU: output = (silu(x @ W_gate)) * (x @ W_up) @ W_down
            W_gate = ffn_info["W_gate"]  # [intermediate, hidden]
            W_up = ffn_info["W_up"]
            W_down = ffn_info["W_down"]
            n_neurons = W_gate.shape[0]

            for neuron_idx in range(min(n_neurons, 500)):  # 采样分析
                w_gate = W_gate[neuron_idx]  # [hidden]
                w_up = W_up[neuron_idx]
                w_down = W_down[:, neuron_idx]  # [hidden]

                # Gate 向量分析: 它"看"输入的什么方向
                gate_norm = np.linalg.norm(w_gate)
                up_norm = np.linalg.norm(w_up)

                # Gate 和 Up 的余弦相似度
                cos_gate_up = np.dot(w_gate, w_up) / (gate_norm * up_norm + 1e-10)

                # W_down 的范数 (输出强度)
                down_norm = np.linalg.norm(w_down)

                # 分类
                if cos_gate_up > 0.8:
                    transform_type = "amplifier"  # gate和up同向 → 选择性放大
                elif abs(cos_gate_up) < 0.3:
                    transform_type = "rotator"  # gate和up正交 → 方向旋转
                elif cos_gate_up < -0.5:
                    transform_type = "inverter"  # gate和up反向 → 选择性抑制
                else:
                    transform_type = "projector"  # 中间情况 → 投影型

                global_transform_counts[transform_type] = global_transform_counts.get(transform_type, 0) + 1
        else:
            # 标准FFN: output = activation(x @ W_in) @ W_out
            W_in = ffn_info["W_in"]  # [4d, d]
            W_out = ffn_info["W_out"]  # [4d, d]
            n_neurons = W_in.shape[0]

            for neuron_idx in range(min(n_neurons, 500)):
                w_in = W_in[neuron_idx]   # [d]
                w_out = W_out[neuron_idx]  # [d]

                in_norm = np.linalg.norm(w_in)
                out_norm = np.linalg.norm(w_out)
                cos_in_out = np.dot(w_in, w_out) / (in_norm * out_norm + 1e-10)

                # 变换类型分类
                if abs(cos_in_out) > 0.9:
                    transform_type = "storage"  # 输入输出同向 → 存储型(近似自编码)
                elif abs(cos_in_out) < 0.2:
                    transform_type = "rotator"  # 输入输出正交 → 方向旋转型
                elif cos_in_out > 0.5:
                    transform_type = "amplifier"  # 正向偏转 → 放大型
                elif cos_in_out < -0.5:
                    transform_type = "inverter"  # 反向 → 反转型
                else:
                    transform_type = "projector"  # 投影型

                global_transform_counts[transform_type] = global_transform_counts.get(transform_type, 0) + 1

        total_neurons += min(n_neurons, 500)

        layer_data["num_neurons_sampled"] = min(n_neurons, 500)
        layer_data["num_neurons_total"] = n_neurons
        layer_results.append(layer_data)

    # 计算比例
    transform_ratios = {k: v / total_neurons for k, v in global_transform_counts.items()}
    results["transform_counts"] = global_transform_counts
    results["transform_ratios"] = transform_ratios
    results["total_neurons_analyzed"] = total_neurons
    results["layer_details"] = layer_results

    print(f"  Total neurons sampled: {total_neurons}")
    for t, ratio in sorted(transform_ratios.items(), key=lambda x: -x[1]):
        print(f"    {t}: {ratio:.4f} ({global_transform_counts[t]} neurons)")

    # 分析变换类型的层级分布
    layer_type_dist = []
    for lr in layer_results:
        layer_type_dist.append({
            "layer": lr["layer"],
            "num_neurons": lr["num_neurons_sampled"],
        })

    results["layer_type_distribution"] = layer_type_dist

    return results


# ============================================================
# 5. 表征对计算的依赖性分析
# ============================================================
def analyze_representation_computation_dependency(model, tokenizer, config):
    """分析表征质量对计算层的依赖程度"""
    print("\n" + "=" * 60)
    print("5. Representation-Computation Dependency Analysis")
    print("=" * 60)

    results = {}

    # 测试词对：语义相似但不相同的词
    word_pairs = [
        ("cat", "dog"),       # 同类动物
        ("cat", "car"),       # 不同类
        ("happy", "joyful"),  # 同义词
        ("happy", "sad"),     # 反义词
        ("king", "queen"),    # 相关词
        ("king", "crown"),    # 关联词
        ("run", "walk"),      # 同类动词
        ("run", "think"),     # 不同类动词
        ("big", "small"),     # 反义形容词
        ("big", "red"),       # 无关形容词
    ]

    # 计算在Embedding空间中的距离
    W_embed = model.get_input_embeddings().weight.data.cpu().numpy()

    embedding_distances = []
    for w1, w2 in word_pairs:
        ids1 = tokenizer.encode(w1, add_special_tokens=False)
        ids2 = tokenizer.encode(w2, add_special_tokens=False)
        if ids1 and ids2:
            v1 = W_embed[ids1[0]]
            v2 = W_embed[ids2[0]]
            dist = float(np.linalg.norm(v1 - v2))
            cos = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10))
            embedding_distances.append({
                "word1": w1, "word2": w2,
                "embed_dist": dist,
                "embed_cosine": cos,
            })

    results["embedding_word_distances"] = embedding_distances

    # 计算残差流中各层对同一词对的距离变化
    hooks = []
    residual_cache = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            residual_cache[layer_idx] = hidden.detach().cpu()
        return hook_fn

    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        for i, block in enumerate(model.transformer.h):
            hooks.append(block.register_forward_hook(make_hook(i)))
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for i, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_hook(make_hook(i)))

    # 用简单句子获取每层的残差流
    layer_word_distances = defaultdict(list)

    with torch.no_grad():
        for w1, w2 in word_pairs:
            # 构造句子，让模型处理这个词
            sent1 = f"The {w1} is here."
            sent2 = f"The {w2} is here."

            ids1 = tokenizer.encode(sent1, return_tensors="pt", truncation=True, max_length=32)
            ids2 = tokenizer.encode(sent2, return_tensors="pt", truncation=True, max_length=32)

            # 找到目标词的位置
            target_pos1 = None
            target_pos2 = None
            tokens1 = tokenizer.convert_ids_to_tokens(ids1[0])
            tokens2 = tokenizer.convert_ids_to_tokens(ids2[0])
            for i, t in enumerate(tokens1):
                if w1.lower() in t.lower():
                    target_pos1 = i
                    break
            for i, t in enumerate(tokens2):
                if w2.lower() in t.lower():
                    target_pos2 = i
                    break

            if target_pos1 is None or target_pos2 is None:
                continue

            # 运行前向传播
            residual_cache.clear()
            ids1 = ids1.to(model.device)
            _ = model(input_ids=ids1)
            residuals1 = {k: v[0, target_pos1].numpy() for k, v in residual_cache.items()}

            residual_cache.clear()
            ids2 = ids2.to(model.device)
            _ = model(input_ids=ids2)
            residuals2 = {k: v[0, target_pos2].numpy() for k, v in residual_cache.items()}

            # 计算每层的距离
            for layer_idx in sorted(residuals1.keys()):
                if layer_idx in residuals2:
                    dist = float(np.linalg.norm(residuals1[layer_idx] - residuals2[layer_idx]))
                    cos = float(np.dot(residuals1[layer_idx], residuals2[layer_idx]) /
                                (np.linalg.norm(residuals1[layer_idx]) *
                                 np.linalg.norm(residuals2[layer_idx]) + 1e-10))
                    layer_word_distances[layer_idx].append({
                        "word1": w1, "word2": w2,
                        "target_pos": target_pos1,
                        "layer": layer_idx,
                        "residual_dist": dist,
                        "residual_cosine": cos,
                    })

    for h in hooks:
        h.remove()

    # 汇总每层的平均距离
    layer_summary = []
    for layer_idx in sorted(layer_word_distances.keys()):
        dists = [d["residual_dist"] for d in layer_word_distances[layer_idx]]
        coss = [d["residual_cosine"] for d in layer_word_distances[layer_idx]]
        layer_summary.append({
            "layer": layer_idx,
            "mean_dist": float(np.mean(dists)),
            "std_dist": float(np.std(dists)),
            "mean_cosine": float(np.mean(coss)),
        })

    results["layer_word_distances"] = dict(layer_word_distances)
    results["layer_distance_summary"] = layer_summary

    # 打印摘要
    print(f"  Word pairs analyzed: {len(word_pairs)}")
    if layer_summary:
        print(f"  Embedding mean dist: {np.mean([d['embed_dist'] for d in embedding_distances]):.4f}")
        print(f"  Layer 0 mean dist: {layer_summary[0]['mean_dist']:.4f}")
        if len(layer_summary) > 1:
            print(f"  Layer {config['num_layers']-1} mean dist: {layer_summary[-1]['mean_dist']:.4f}")
            # 距离变化趋势
            trend = layer_summary[-1]["mean_dist"] - layer_summary[0]["mean_dist"]
            print(f"  Distance change (L0→L{config['num_layers']-1}): {trend:+.4f}")
            if trend > 0:
                print("  → 表征在深层变得更加区分 (语义分离)")
            else:
                print("  → 表征在深层变得更加相似 (语义合并)")

    return results


# ============================================================
# Main
# ============================================================
def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 运行 GPT-2
    print("=" * 60)
    print("PHASE 1 ANALYSIS: GPT-2")
    print("=" * 60)
    model_gpt2, tok_gpt2, cfg_gpt2 = load_model("gpt2")

    results = {
        "timestamp": timestamp,
        "models": {},
    }

    # 1. Embedding
    results["models"]["gpt2"] = {
        "config": cfg_gpt2,
        "embedding": analyze_embedding_structure(model_gpt2, tok_gpt2, cfg_gpt2),
        "residual_geometry": analyze_residual_stream_geometry(model_gpt2, tok_gpt2, cfg_gpt2),
        "attention_routing": analyze_attention_routing(model_gpt2, tok_gpt2, cfg_gpt2),
        "ffn_transformations": analyze_ffn_transformations(model_gpt2, tok_gpt2, cfg_gpt2),
        "computation_dependency": analyze_representation_computation_dependency(model_gpt2, tok_gpt2, cfg_gpt2),
    }

    # 释放 GPT-2 显存
    del model_gpt2
    torch.cuda.empty_cache()

    # 运行 Qwen2.5-0.5B
    print("\n" + "=" * 60)
    print("PHASE 1 ANALYSIS: Qwen2.5-0.5B")
    print("=" * 60)

    try:
        model_qwen, tok_qwen, cfg_qwen = load_model("Qwen/Qwen2.5-0.5B")
        results["models"]["qwen2.5-0.5b"] = {
            "config": cfg_qwen,
            "embedding": analyze_embedding_structure(model_qwen, tok_qwen, cfg_qwen),
            "residual_geometry": analyze_residual_stream_geometry(model_qwen, tok_qwen, cfg_qwen),
            "attention_routing": analyze_attention_routing(model_qwen, tok_qwen, cfg_qwen),
            "ffn_transformations": analyze_ffn_transformations(model_qwen, tok_qwen, cfg_qwen),
            "computation_dependency": analyze_representation_computation_dependency(model_qwen, tok_qwen, cfg_qwen),
        }
        del model_qwen
        torch.cuda.empty_cache()
    except RuntimeError as e:
        print(f"  Qwen2.5-0.5B failed (likely OOM): {e}")
        print("  Skipping Qwen analysis. Use a smaller model or quantization.")
        results["models"]["qwen2.5-0.5b"] = {"error": str(e)}

    # 保存结果
    output_path = OUTPUT_DIR / f"phase1_comprehensive_{timestamp}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 60}")

    return results


if __name__ == "__main__":
    main()
