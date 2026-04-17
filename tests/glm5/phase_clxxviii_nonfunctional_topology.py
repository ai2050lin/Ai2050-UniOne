"""
Phase CLXXVIII: 非功能空间的拓扑结构分析
=========================================
核心目标: 分析残差流中94-97%非功能空间的组织结构

关键问题:
1. 非功能空间中是否存在可识别的子结构(语义聚类/知识区域)?
2. 功能方向是否控制非功能空间的信息选择("控制-内容"假说)?
3. 不同类型文本在非功能空间中的分布模式?
4. 非功能空间在中间层的中英文对比(避免Embedding干扰)?

实验设计:
  P781: 非功能空间的语义聚类
    - 收集大量句子的残差流, 投影到非功能空间
    - 对非功能空间表示做聚类, 识别子结构
    - 分析聚类是否对应语义类别(动物/颜色/职业...)

  P782: "控制-内容"假说验证
    - 在功能方向上做干预, 测量非功能空间的变化
    - 如果功能方向控制内容选择, 则改变功能应改变非功能空间
    - 量化: 功能干预 vs 非功能变化的耦合度

  P783: 文本类型在非功能空间的分布
    - 收集不同类型文本(诗歌/科学/对话/新闻/代码)
    - 分析其在非功能空间中的分布差异
    - 文本类型是否可从非功能空间线性分离?

  P784: 中间层中英文功能对比
    - 在Layer 5-10做中英文功能方向对比(避免Embedding干扰)
    - 验证: 中间层的EN-ZH余弦是否高于Layer 0
    - 如果是, 则功能基底在中间层可能更通用
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import argparse
import json
import gc
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from collections import defaultdict

from model_utils import load_model, get_model_info, get_layers, release_model


def to_numpy(tensor_or_array):
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array.astype(np.float32)
    return tensor_or_array.detach().cpu().float().numpy().astype(np.float32)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ============================================================
# 功能句子对 (英文)
# ============================================================

ENGLISH_FUNCTIONAL_PAIRS = {
    'syntax': [
        ("The cat sits on the mat", "The cats sit on the mat"),
        ("She walks to school", "She walked to school"),
        ("The dog chased the cat", "The cat was chased by the dog"),
        ("He is running fast", "Is he running fast?"),
        ("A bird flies in the sky", "Birds fly in the sky"),
        ("The man reads a book", "The men read books"),
        ("She has eaten dinner", "She had eaten dinner"),
        ("They will go home", "They went home"),
        ("I can see the mountain", "Can I see the mountain?"),
        ("The children play outside", "The child plays outside"),
    ],
    'semantic': [
        ("The cat sat on the mat", "The dog sat on the mat"),
        ("She walked to school", "She drove to school"),
        ("The king ruled the kingdom", "The queen ruled the kingdom"),
        ("He ate an apple", "He ate an orange"),
        ("The sun is bright", "The moon is bright"),
        ("The man is tall", "The woman is tall"),
        ("I love music", "I hate music"),
        ("The car is fast", "The bicycle is fast"),
        ("She bought a house", "She rented a house"),
        ("The doctor cured patients", "The teacher taught students"),
    ],
    'style': [
        ("The cat sat on the mat", "The feline rested upon the rug"),
        ("She walked to school", "She proceeded to the educational institution"),
        ("He is very happy", "He is exceedingly joyful"),
        ("The food was good", "The cuisine was delectable"),
        ("I think this is right", "In my humble opinion, this appears correct"),
        ("The movie was bad", "The cinematic experience was abysmal"),
        ("She said hello", "She articulated a greeting"),
        ("The house is big", "The dwelling is commodious"),
        ("He ran fast", "He sprinted with celerity"),
        ("It was a good day", "It proved to be a splendid day"),
    ],
    'tense': [
        ("I walk to school", "I walked to school"),
        ("She reads a book", "She read a book"),
        ("They play outside", "They played outside"),
        ("He runs every day", "He ran every day"),
        ("We eat dinner together", "We ate dinner together"),
        ("The sun rises at dawn", "The sun rose at dawn"),
        ("She writes in her journal", "She wrote in her journal"),
        ("I understand the problem", "I understood the problem"),
        ("The children sing songs", "The children sang songs"),
        ("He drives to work", "He drove to work"),
    ],
    'polarity': [
        ("She is happy", "She is not happy"),
        ("The movie was good", "The movie was not good"),
        ("He can swim", "He cannot swim"),
        ("I like this song", "I do not like this song"),
        ("The test was easy", "The test was not easy"),
        ("She will come", "She will not come"),
        ("They are rich", "They are not rich"),
        ("We have time", "We do not have time"),
        ("The food is fresh", "The food is not fresh"),
        ("He always arrives early", "He never arrives early"),
    ],
}

# 中文功能句子对
CHINESE_FUNCTIONAL_PAIRS = {
    'syntax': [
        ("猫坐在垫子上", "猫们坐在垫子上"),
        ("她走路去学校", "她走路去了学校"),
        ("狗追了猫", "猫被狗追了"),
        ("他跑得很快", "他跑得快吗"),
        ("一只鸟在天上飞", "鸟在天上飞"),
    ],
    'semantic': [
        ("猫坐在垫子上", "狗坐在垫子上"),
        ("她走路去学校", "她开车去学校"),
        ("国王统治王国", "女王统治王国"),
        ("他吃了一个苹果", "他吃了一个橘子"),
        ("太阳很亮", "月亮很亮"),
    ],
    'style': [
        ("猫坐在垫子上", "猫咪安歇于地毯之上"),
        ("她走路去学校", "她步行前往学府"),
        ("他非常高兴", "他喜不自胜"),
        ("食物很好吃", "佳肴美味可口"),
        ("我觉得这是对的", "依我之见此乃正确"),
    ],
    'tense': [
        ("我走路去学校", "我走路去了学校"),
        ("她读一本书", "她读了一本书"),
        ("他们在外面玩", "他们在外面玩了"),
        ("他每天跑步", "他每天跑了步"),
        ("我们一起吃晚饭", "我们一起吃了晚饭"),
    ],
    'polarity': [
        ("她很高兴", "她不高兴"),
        ("电影很好看", "电影不好看"),
        ("他会游泳", "他不会游泳"),
        ("我喜欢这首歌", "我不喜欢这首歌"),
        ("考试很简单", "考试不简单"),
    ],
}

# 语义类别句子 (用于非功能空间聚类分析)
SEMANTIC_CATEGORY_SENTENCES = {
    'animals': [
        "The cat sleeps peacefully on the sofa",
        "A dog barks at the mailman every morning",
        "The horse galloped across the open field",
        "Birds build nests in tall trees",
        "The fish swims in the clear pond",
    ],
    'colors': [
        "The red rose blooms in the garden",
        "She wore a blue dress to the party",
        "The green grass covers the hillside",
        "A yellow sunflower reaches for the sky",
        "The purple sunset painted the horizon",
    ],
    'occupations': [
        "The doctor examined the patient carefully",
        "A teacher explains the lesson to students",
        "The engineer designed a new bridge",
        "The chef prepared a delicious meal",
        "The scientist published important research",
    ],
    'emotions': [
        "She felt overwhelming joy at the news",
        "His anger flared when he heard the lie",
        "The child showed fear of the dark room",
        "Love filled her heart completely",
        "Sadness clouded his usual cheerful mood",
    ],
    'locations': [
        "The mountain towered above the valley",
        "Waves crashed against the rocky shore",
        "The forest stretched endlessly before them",
        "A desert stretched under the hot sun",
        "The city skyline glittered at night",
    ],
    'science': [
        "Electrons orbit the atomic nucleus",
        "Gravity pulls objects toward the earth",
        "DNA carries genetic information in cells",
        "Light travels at incredible speed",
        "Chemical reactions transform substances",
    ],
    'food': [
        "The bread baked golden in the oven",
        "Fresh fruit filled the market stalls",
        "She stirred the soup with a wooden spoon",
        "The cheese aged perfectly in the cellar",
        "Rice is a staple food in many cultures",
    ],
    'technology': [
        "The computer processed data rapidly",
        "Software updates improve system security",
        "The internet connects people worldwide",
        "Artificial intelligence advances quickly",
        "Robots assemble cars in the factory",
    ],
}

# 不同类型文本
TEXT_TYPE_SENTENCES = {
    'poetry': [
        "The moonlight dances on the water",
        "Stars whisper secrets to the night",
        "Autumn leaves fall like golden rain",
        "Dawn breaks with rosy fingers of light",
        "The wind sings through ancient trees",
    ],
    'science': [
        "The experiment confirmed the hypothesis",
        "Data analysis revealed significant correlations",
        "The method controls for confounding variables",
        "Results are consistent with theoretical predictions",
        "The study employs a randomized design",
    ],
    'dialogue': [
        "What do you think about this idea",
        "I agree that we should proceed carefully",
        "Could you explain that point again",
        "Let me think about it for a moment",
        "That sounds like a reasonable approach",
    ],
    'narrative': [
        "She walked slowly through the old town",
        "The door creaked open revealing darkness",
        "He found the letter hidden in the drawer",
        "Morning light streamed through the window",
        "They met at the corner of the street",
    ],
    'code': [
        "The function returns a boolean value",
        "Initialize the array with default values",
        "The loop iterates over each element",
        "Import the required module at the top",
        "The variable stores the current state",
    ],
}


def _collect_residual_stream(model, tokenizer, device, sentence, max_length=64):
    """对单个句子收集每层输出的最后一个token残差流"""
    ids = tokenizer.encode(sentence, return_tensors='pt', max_length=max_length, truncation=True).to(device)
    layers_out = {}

    def make_hook(storage, layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                storage[layer_idx] = to_numpy(output[0])
            else:
                storage[layer_idx] = to_numpy(output)
        return hook

    hooks = []
    for i, layer in enumerate(get_layers(model)):
        h = layer.register_forward_hook(make_hook(layers_out, i))
        hooks.append(h)

    with torch.no_grad():
        _ = model(ids)

    for h in hooks:
        h.remove()

    residual = {}
    for layer_idx in sorted(layers_out.keys()):
        arr = layers_out[layer_idx]
        if arr.ndim == 3:
            residual[layer_idx] = arr[0, -1, :]
        elif arr.ndim == 2:
            residual[layer_idx] = arr[-1, :]
        else:
            residual[layer_idx] = arr
    return residual


def extract_functional_directions(model, tokenizer, device, pairs_dict, layer_idx=0):
    """提取指定层的功能方向"""
    dim_names = list(pairs_dict.keys())
    directions = {}

    for dim_name in dim_names:
        pairs = pairs_dict[dim_name]
        diff_vectors = []

        for s1, s2 in pairs:
            r1 = _collect_residual_stream(model, tokenizer, device, s1)
            r2 = _collect_residual_stream(model, tokenizer, device, s2)
            if layer_idx in r1 and layer_idx in r2:
                diff = r1[layer_idx] - r2[layer_idx]
                diff_vectors.append(diff)

        if not diff_vectors:
            continue

        mean_diff = np.mean(diff_vectors, axis=0)
        mean_diff = mean_diff / (np.linalg.norm(mean_diff) + 1e-30)
        directions[dim_name] = mean_diff

    return directions


def get_functional_subspace_basis(model, tokenizer, device, layer_idx=0):
    """获取功能子空间的正交基"""
    dim_names = list(ENGLISH_FUNCTIONAL_PAIRS.keys())
    directions = extract_functional_directions(
        model, tokenizer, device, ENGLISH_FUNCTIONAL_PAIRS, layer_idx
    )

    if not directions:
        return None, []

    available_dims = [d for d in dim_names if d in directions]
    D = np.stack([directions[d] for d in available_dims], axis=1)  # [d_model, n_funcs]

    # Gram-Schmidt 正交化
    Q, R = np.linalg.qr(D)
    return Q, available_dims  # Q: [d_model, n_funcs], 正交基


# ============================================================
# P781: 非功能空间的语义聚类
# ============================================================

def P781_semantic_clustering(model, tokenizer, device, model_name, results):
    """
    在非功能空间中对语义类别句子做聚类分析
    """
    print("\n--- P781: 非功能空间的语义聚类 ---")

    n_vocab, d_model = model.lm_head.weight.shape
    n_layers = len(get_layers(model))

    # 使用中间层 (Layer n_layers//4) 避免Embedding和输出层的干扰
    target_layer = n_layers // 4
    print(f"  使用 Layer {target_layer} 进行分析")

    # 获取功能子空间正交基
    Q_func, func_dims = get_functional_subspace_basis(model, tokenizer, device, target_layer)

    if Q_func is None:
        print("  未能获取功能子空间基, 跳过")
        return

    n_func = Q_func.shape[1]
    print(f"  功能子空间维度: {n_func}")

    # 收集所有语义类别句子的残差流
    category_names = list(SEMANTIC_CATEGORY_SENTENCES.keys())
    all_representations = []  # [(category, sentence, full_residual)]
    all_func_repr = []  # 功能空间投影
    all_nonfunc_repr = []  # 非功能空间投影

    for cat_name in category_names:
        sentences = SEMANTIC_CATEGORY_SENTENCES[cat_name]
        print(f"  收集 {cat_name} ({len(sentences)} 句)...")

        for sent in sentences:
            r = _collect_residual_stream(model, tokenizer, device, sent)
            if target_layer in r:
                h = r[target_layer]  # [d_model]
                # 功能空间投影
                h_func = Q_func @ (Q_func.T @ h)
                # 非功能空间投影
                h_nonfunc = h - h_func

                all_representations.append((cat_name, sent, h))
                all_func_repr.append(h_func)
                all_nonfunc_repr.append(h_nonfunc)

    if len(all_representations) < 10:
        print("  样本不足, 跳过")
        return

    func_matrix = np.stack(all_func_repr)  # [n_samples, d_model]
    nonfunc_matrix = np.stack(all_nonfunc_repr)  # [n_samples, d_model]
    labels = [r[0] for r in all_representations]

    # 分析1: 功能空间 vs 非功能空间的类别区分度
    # 用类内距离/类间距离比来衡量
    def compute_separability(matrix, labels):
        """计算聚类可分性: 类间距离 / 类内距离"""
        unique_labels = list(set(labels))
        global_center = np.mean(matrix, axis=0)

        # 类间散度
        between_var = 0
        for label in unique_labels:
            mask = np.array(labels) == label
            center = np.mean(matrix[mask], axis=0)
            between_var += np.sum(mask) * np.sum((center - global_center) ** 2)

        # 类内散度
        within_var = 0
        for label in unique_labels:
            mask = np.array(labels) == label
            center = np.mean(matrix[mask], axis=0)
            within_var += np.sum(np.sum((matrix[mask] - center) ** 2, axis=1))

        return float(between_var / (within_var + 1e-30))

    func_sep = compute_separability(func_matrix, labels)
    nonfunc_sep = compute_separability(nonfunc_matrix, labels)

    print(f"\n  语义类别可分性:")
    print(f"    功能空间: between/within = {func_sep:.4f}")
    print(f"    非功能空间: between/within = {nonfunc_sep:.4f}")
    print(f"    非功能/功能比: {nonfunc_sep / (func_sep + 1e-30):.2f}x")

    # 分析2: 非功能空间的PCA降维后聚类
    # 用PCA降到低维, 计算类间余弦矩阵
    from sklearn.decomposition import PCA
    n_pca = min(20, nonfunc_matrix.shape[0] - 1, nonfunc_matrix.shape[1])
    pca = PCA(n_components=n_pca)
    nonfunc_pca = pca.fit_transform(nonfunc_matrix)

    # 类间余弦矩阵
    unique_cats = list(set(labels))
    cat_centers = {}
    for cat in unique_cats:
        mask = np.array(labels) == cat
        cat_centers[cat] = np.mean(nonfunc_pca[mask], axis=0)

    cat_cos_matrix = {}
    for i, c1 in enumerate(unique_cats):
        for j, c2 in enumerate(unique_cats):
            if i < j:
                cos = float(np.dot(cat_centers[c1], cat_centers[c2]) /
                           (np.linalg.norm(cat_centers[c1]) * np.linalg.norm(cat_centers[c2]) + 1e-30))
                cat_cos_matrix[f"{c1}_vs_{c2}"] = cos

    # 最相似和最不相似的类别对
    sorted_pairs = sorted(cat_cos_matrix.items(), key=lambda x: x[1], reverse=True)
    print(f"\n  非功能空间中最相似的类别对:")
    for pair, cos in sorted_pairs[:3]:
        print(f"    {pair}: cos={cos:.4f}")
    print(f"  最不相似的类别对:")
    for pair, cos in sorted_pairs[-3:]:
        print(f"    {pair}: cos={cos:.4f}")

    # 分析3: 功能空间PCA
    func_pca = PCA(n_components=min(n_pca, func_matrix.shape[0] - 1))
    func_pca_data = pca.fit_transform(func_matrix)

    # 解释方差比
    print(f"\n  非功能空间PCA方差解释比 (前5维): {pca.explained_variance_ratio_[:5]}")

    results['p781_semantic_clustering'] = {
        'target_layer': target_layer,
        'n_func_dims': n_func,
        'n_samples': len(all_representations),
        'n_categories': len(unique_cats),
        'func_separability': func_sep,
        'nonfunc_separability': nonfunc_sep,
        'nonfunc_func_ratio': nonfunc_sep / (func_sep + 1e-30),
        'category_cosine_matrix': cat_cos_matrix,
        'pca_explained_variance_ratio': pca.explained_variance_ratio_[:10].tolist(),
        'most_similar_pairs': [(p, c) for p, c in sorted_pairs[:5]],
        'least_similar_pairs': [(p, c) for p, c in sorted_pairs[-5:]],
    }


# ============================================================
# P782: "控制-内容"假说验证
# ============================================================

def P782_control_content_hypothesis(model, tokenizer, device, model_name, results):
    """
    验证: 功能方向是否控制非功能空间的信息选择
    方法: 在功能方向上做干预, 测量非功能空间的变化
    """
    print("\n--- P782: 控制-内容假说验证 ---")

    n_vocab, d_model = model.lm_head.weight.shape
    n_layers = len(get_layers(model))
    target_layer = n_layers // 4

    # 获取功能子空间正交基
    Q_func, func_dims = get_functional_subspace_basis(model, tokenizer, device, target_layer)

    if Q_func is None:
        print("  未能获取功能子空间基, 跳过")
        return

    n_func = Q_func.shape[1]

    # 测试句子
    test_sentences = [
        "The cat sat on the mat",
        "She walked to school",
        "The doctor helped patients",
        "The sun shines brightly",
        "He reads books every day",
    ]

    dim_names = list(ENGLISH_FUNCTIONAL_PAIRS.keys())
    coupling_results = {}

    for intervene_dim in func_dims:
        dim_idx = func_dims.index(intervene_dim)
        intervene_dir = Q_func[:, dim_idx]  # 功能方向 [d_model]

        coupling_scores = []

        for sent in test_sentences:
            # 获取原始残差流
            r_orig = _collect_residual_stream(model, tokenizer, device, sent)
            if target_layer not in r_orig:
                continue

            h_orig = r_orig[target_layer]
            h_func_orig = Q_func @ (Q_func.T @ h_orig)
            h_nonfunc_orig = h_orig - h_func_orig

            # 在功能方向上做干预 (添加一个扰动)
            scale = 0.5 * float(np.linalg.norm(h_func_orig))
            h_modified = h_orig + scale * intervene_dir

            # 计算干预后非功能空间的变化
            # 注意: 干预是在残差流上做的, 不经过后续层
            # 所以这里直接测量残差流的变化
            h_func_mod = Q_func @ (Q_func.T @ h_modified)
            h_nonfunc_mod = h_modified - h_func_mod

            # 非功能空间的变化量
            nonfunc_change = float(np.linalg.norm(h_nonfunc_mod - h_nonfunc_orig))
            func_change = float(np.linalg.norm(h_func_mod - h_func_orig))

            # 耦合度 = 非功能变化 / 功能变化
            coupling = nonfunc_change / (func_change + 1e-30)
            coupling_scores.append(coupling)

        avg_coupling = float(np.mean(coupling_scores)) if coupling_scores else 0
        coupling_results[intervene_dim] = {
            'avg_coupling': avg_coupling,
            'coupling_scores': coupling_scores,
        }
        print(f"  {intervene_dim}: 平均耦合度 = {avg_coupling:.4f}")

    # 理论分析:
    # 如果功能和非功能完全独立, 耦合度应该=0 (功能干预不影响非功能)
    # 如果功能控制非功能, 耦合度应该>0
    # 但在残差流中, 干预是线性的, 所以耦合度应该=0 (理论上)
    # 关键: 需要看经过后续层后的非线性效应

    # 补充: 实际经过模型的效果 (在功能方向干预后看最终logit)
    print("\n  补充: 功能干预对最终logit的影响 (经过后续层)")

    # 选一个句子和一个干预维度
    test_sent = "The cat sat on the mat"
    intervene_dim = 'syntax'
    dim_idx = func_dims.index(intervene_dim) if intervene_dim in func_dims else 0
    intervene_dir = Q_func[:, dim_idx]

    # 获取原始logit
    ids = tokenizer.encode(test_sent, return_tensors='pt').to(device)
    with torch.no_grad():
        orig_logits = to_numpy(model(ids).logits[0, -1, :])

    # 在目标层做干预 (需要用hook)
    logit_changes = {}
    for scale in [0.1, 0.5, 1.0, 2.0]:
        modified_logits_list = []

        def intervention_hook(module, input, output, s=scale):
            if isinstance(output, tuple):
                out = output[0]
            else:
                out = output
            # 在功能方向上添加扰动
            perturbation = torch.tensor(s * intervene_dir, dtype=out.dtype, device=out.device)
            modified = out + perturbation.unsqueeze(0).unsqueeze(0)
            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        layer = get_layers(model)[target_layer]
        hook = layer.register_forward_hook(intervention_hook)

        with torch.no_grad():
            mod_logits = to_numpy(model(ids).logits[0, -1, :])

        hook.remove()

        # logit变化
        logit_diff = mod_logits - orig_logits
        logit_l2 = float(np.linalg.norm(logit_diff))
        logit_cos = float(np.dot(orig_logits, mod_logits) /
                         (np.linalg.norm(orig_logits) * np.linalg.norm(mod_logits) + 1e-30))

        # top-5 token 变化
        top5_orig = set(np.argsort(orig_logits)[-5:][::-1])
        top5_mod = set(np.argsort(mod_logits)[-5:][::-1])
        top5_overlap = len(top5_orig & top5_mod) / 5.0

        logit_changes[f"scale_{scale}"] = {
            'logit_l2': logit_l2,
            'logit_cos': logit_cos,
            'top5_overlap': top5_overlap,
        }
        print(f"    scale={scale}: logit_L2={logit_l2:.2f}, cos={logit_cos:.4f}, top5_overlap={top5_overlap:.2f}")

    results['p782_control_content'] = {
        'target_layer': target_layer,
        'coupling_results': coupling_results,
        'logit_intervention': logit_changes,
        'intervene_dim': intervene_dim,
    }


# ============================================================
# P783: 文本类型在非功能空间的分布
# ============================================================

def P783_text_type_distribution(model, tokenizer, device, model_name, results):
    """
    分析不同类型文本在非功能空间中的分布差异
    """
    print("\n--- P783: 文本类型在非功能空间的分布 ---")

    n_vocab, d_model = model.lm_head.weight.shape
    n_layers = len(get_layers(model))
    target_layer = n_layers // 4

    # 获取功能子空间正交基
    Q_func, func_dims = get_functional_subspace_basis(model, tokenizer, device, target_layer)

    if Q_func is None:
        print("  未能获取功能子空间基, 跳过")
        return

    n_func = Q_func.shape[1]

    # 收集不同类型文本的残差流
    type_names = list(TEXT_TYPE_SENTENCES.keys())
    type_representations = {}

    for type_name in type_names:
        sentences = TEXT_TYPE_SENTENCES[type_name]
        print(f"  收集 {type_name} ({len(sentences)} 句)...")

        func_reprs = []
        nonfunc_reprs = []

        for sent in sentences:
            r = _collect_residual_stream(model, tokenizer, device, sent)
            if target_layer in r:
                h = r[target_layer]
                h_func = Q_func @ (Q_func.T @ h)
                h_nonfunc = h - h_func

                func_reprs.append(h_func)
                nonfunc_reprs.append(h_nonfunc)

        if func_reprs:
            type_representations[type_name] = {
                'func': np.stack(func_reprs),
                'nonfunc': np.stack(nonfunc_reprs),
            }

    # 分析1: 每种文本类型在非功能空间的范数
    print(f"\n  文本类型的非功能空间范数:")
    type_norms = {}
    for type_name in type_names:
        if type_name not in type_representations:
            continue
        nonfunc_norms = np.linalg.norm(type_representations[type_name]['nonfunc'], axis=1)
        func_norms = np.linalg.norm(type_representations[type_name]['func'], axis=1)
        avg_nonfunc = float(np.mean(nonfunc_norms))
        avg_func = float(np.mean(func_norms))
        type_norms[type_name] = {
            'avg_nonfunc_norm': avg_nonfunc,
            'avg_func_norm': avg_func,
            'ratio': avg_nonfunc / (avg_func + 1e-30),
        }
        print(f"    {type_name}: nonfunc={avg_nonfunc:.2f}, func={avg_func:.2f}, ratio={avg_nonfunc/(avg_func+1e-30):.2f}x")

    # 分析2: 跨类型余弦矩阵 (非功能空间)
    type_centers_nonfunc = {}
    type_centers_func = {}
    for type_name in type_names:
        if type_name not in type_representations:
            continue
        type_centers_nonfunc[type_name] = np.mean(type_representations[type_name]['nonfunc'], axis=0)
        type_centers_func[type_name] = np.mean(type_representations[type_name]['func'], axis=0)

    # 非功能空间跨类型余弦
    nonfunc_cos = {}
    for i, t1 in enumerate(type_names):
        for j, t2 in enumerate(type_names):
            if i < j and t1 in type_centers_nonfunc and t2 in type_centers_nonfunc:
                cos = float(np.dot(type_centers_nonfunc[t1], type_centers_nonfunc[t2]) /
                           (np.linalg.norm(type_centers_nonfunc[t1]) * np.linalg.norm(type_centers_nonfunc[t2]) + 1e-30))
                nonfunc_cos[f"{t1}_vs_{t2}"] = cos

    # 功能空间跨类型余弦
    func_cos = {}
    for i, t1 in enumerate(type_names):
        for j, t2 in enumerate(type_names):
            if i < j and t1 in type_centers_func and t2 in type_centers_func:
                cos = float(np.dot(type_centers_func[t1], type_centers_func[t2]) /
                           (np.linalg.norm(type_centers_func[t1]) * np.linalg.norm(type_centers_func[t2]) + 1e-30))
                func_cos[f"{t1}_vs_{t2}"] = cos

    # 对比: 非功能空间和功能空间的类型区分能力
    avg_nonfunc_cos = float(np.mean(list(nonfunc_cos.values()))) if nonfunc_cos else 0
    avg_func_cos = float(np.mean(list(func_cos.values()))) if func_cos else 0

    print(f"\n  跨类型中心余弦 (越高=越相似, 区分度越低):")
    print(f"    非功能空间: avg_cos = {avg_nonfunc_cos:.4f}")
    print(f"    功能空间: avg_cos = {avg_func_cos:.4f}")
    print(f"    非功能空间区分度(1-cos): {1-avg_nonfunc_cos:.4f}")
    print(f"    功能空间区分度(1-cos): {1-avg_func_cos:.4f}")

    # 分析3: 线性可分性 (用PCA + 类间/类内距离比)
    all_nonfunc = []
    all_labels = []
    for type_name in type_names:
        if type_name not in type_representations:
            continue
        all_nonfunc.append(type_representations[type_name]['nonfunc'])
        all_labels.extend([type_name] * len(type_representations[type_name]['nonfunc']))

    all_func = []
    for type_name in type_names:
        if type_name not in type_representations:
            continue
        all_func.append(type_representations[type_name]['func'])

    nonfunc_matrix = np.concatenate(all_nonfunc, axis=0)
    func_matrix = np.concatenate(all_func, axis=0)

    # 类间/类内距离比
    def separability(mat, labels):
        unique = list(set(labels))
        global_center = np.mean(mat, axis=0)
        between = 0
        within = 0
        for label in unique:
            mask = np.array(labels) == label
            center = np.mean(mat[mask], axis=0)
            between += np.sum(mask) * np.sum((center - global_center) ** 2)
            within += np.sum(np.sum((mat[mask] - center) ** 2, axis=1))
        return float(between / (within + 1e-30))

    nonfunc_sep = separability(nonfunc_matrix, all_labels)
    func_sep = separability(func_matrix, all_labels)

    print(f"\n  文本类型可分性:")
    print(f"    非功能空间: {nonfunc_sep:.4f}")
    print(f"    功能空间: {func_sep:.4f}")

    results['p783_text_type'] = {
        'target_layer': target_layer,
        'type_norms': type_norms,
        'nonfunc_avg_cos': avg_nonfunc_cos,
        'func_avg_cos': avg_func_cos,
        'nonfunc_separability': nonfunc_sep,
        'func_separability': func_sep,
        'nonfunc_cos_matrix': nonfunc_cos,
        'func_cos_matrix': func_cos,
    }


# ============================================================
# P784: 中间层中英文功能对比
# ============================================================

def P784_midlayer_cross_language(model, tokenizer, device, model_name, results):
    """
    在中间层做中英文功能方向对比, 避免Embedding层干扰
    """
    print("\n--- P784: 中间层中英文功能对比 ---")

    n_layers = len(get_layers(model))
    dim_names = list(ENGLISH_FUNCTIONAL_PAIRS.keys())

    # 测试多个中间层
    test_layers = [0, n_layers // 8, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2]
    test_layers = [l for l in test_layers if l < n_layers]

    en_zh_cos_by_layer = {}

    for layer_idx in test_layers:
        print(f"\n  Layer {layer_idx}:")

        # 英文功能方向
        en_dirs = extract_functional_directions(
            model, tokenizer, device, ENGLISH_FUNCTIONAL_PAIRS, layer_idx
        )

        # 中文功能方向
        zh_dirs = extract_functional_directions(
            model, tokenizer, device, CHINESE_FUNCTIONAL_PAIRS, layer_idx
        )

        common_dims = [d for d in dim_names if d in en_dirs and d in zh_dirs]

        layer_cos = {}
        for dim_name in common_dims:
            cos = float(np.dot(en_dirs[dim_name], zh_dirs[dim_name]))
            layer_cos[dim_name] = cos
            print(f"    {dim_name}: EN-ZH cos = {cos:.4f}")

        # 正交性 (英文方向之间)
        en_orth = {}
        for i in range(len(common_dims)):
            for j in range(i + 1, len(common_dims)):
                d1, d2 = common_dims[i], common_dims[j]
                cos = float(np.dot(en_dirs[d1], en_dirs[d2]))
                en_orth[f"{d1}_vs_{d2}"] = cos

        en_zh_cos_by_layer[layer_idx] = {
            'en_zh_cos': layer_cos,
            'en_orthogonality': en_orth,
            'avg_en_zh_cos': float(np.mean(list(layer_cos.values()))) if layer_cos else 0,
            'avg_en_orth': float(np.mean([abs(c) for c in en_orth.values()])) if en_orth else 0,
        }

    # 找到EN-ZH余弦最高的层
    layer_avg_cos = {l: info['avg_en_zh_cos'] for l, info in en_zh_cos_by_layer.items()}
    best_layer = max(layer_avg_cos, key=layer_avg_cos.get) if layer_avg_cos else 0

    print(f"\n  各层EN-ZH平均余弦:")
    for layer_idx, avg_cos in sorted(layer_avg_cos.items()):
        print(f"    Layer {layer_idx}: avg_cos = {avg_cos:.4f}")
    print(f"  最佳层: Layer {best_layer} (avg_cos = {layer_avg_cos.get(best_layer, 0):.4f})")

    # 各层正交性变化
    print(f"\n  各层英文功能方向正交性:")
    for layer_idx, info in sorted(en_zh_cos_by_layer.items()):
        print(f"    Layer {layer_idx}: avg_|cos| = {info['avg_en_orth']:.4f}")

    results['p784_midlayer_cross_language'] = {
        'test_layers': test_layers,
        'en_zh_cos_by_layer': en_zh_cos_by_layer,
        'best_layer': best_layer,
        'best_layer_avg_cos': layer_avg_cos.get(best_layer, 0),
    }


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase CLXXVIII: 非功能空间拓扑结构分析")
    parser.add_argument('--model', type=str, required=True,
                       choices=['glm4', 'qwen3', 'deepseek7b'],
                       help='model name')
    args = parser.parse_args()

    model_name = args.model
    print(f"=== Phase CLXXVIII: Non-functional Space Topology - {model_name} ===")

    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    print(f"Model: {info.n_layers} layers, d_model={info.d_model}, vocab={info.vocab_size}")

    results = {
        'model': model_name,
        'model_info': {
            'n_layers': info.n_layers,
            'd_model': info.d_model,
            'vocab_size': info.vocab_size,
        },
        'timestamp': datetime.now().isoformat(),
    }

    # 运行实验
    P781_semantic_clustering(model, tokenizer, device, model_name, results)
    P782_control_content_hypothesis(model, tokenizer, device, model_name, results)
    P783_text_type_distribution(model, tokenizer, device, model_name, results)
    P784_midlayer_cross_language(model, tokenizer, device, model_name, results)

    # 保存结果
    output_dir = Path(__file__).parent / 'results' / 'phase_clxxviii'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_results.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("Model released")


if __name__ == '__main__':
    main()
