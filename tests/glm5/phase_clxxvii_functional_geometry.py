"""
Phase CLXXVII: 功能几何学精确刻画
=====================================
核心目标: 基于CLXXVI的发现(功能方向天然正交, ~35维功能子空间),
          精确刻画语言功能空间的几何结构

关键问题:
1. 功能方向在多层间的旋转是否构成连续群?
2. 99%非功能维度是什么? 知识? 上下文? 推理链?
3. 中英文的功能基底是否相同?
4. 功能子空间的曲率/拓扑结构

实验设计:
  P771: 功能方向的精确提取(每维度50+句子对)
    - 用对比学习思路: 对每个功能维度构造大量正负样本对
    - 计算平均差异方向, 做PCA提取主成分
    - 与CLXXVI的10对比较, 验证稳定性

  P772: 功能空间的旋转动力学
    - 提取每层的功能方向矩阵 D_l = [d_syntax_l, d_semantic_l, ...]
    - 计算相邻层的旋转矩阵 R_l = D_{l+1} @ D_l^T
    - 验证: R_l是否构成SO(n)群的路径? 是否可参数化?

  P773: 非功能维度分析
    - 将残差流分解为功能子空间 + 正交补空间
    - 分析正交补空间的SVD结构
    - 识别正交补空间中的子结构(知识? 上下文?)

  P774: 中英文功能基底对比
    - 对中文句子做同样的功能方向提取
    - 比较中英文的功能方向: 是否旋转等价?
    - 如果是, 则功能基底是语言无关的!
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
from scipy.linalg import orthogonal_procrustes, logm, expm
from scipy.spatial.transform import Rotation

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
# 大规模功能句子对 (每维度15+对, 英文)
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
        ("We have finished work", "We had finished work"),
        ("John gives Mary a gift", "Mary is given a gift by John"),
        ("The sun rises early", "Does the sun rise early?"),
        ("A tree grows tall", "Trees grow tall"),
        ("She writes letters", "She wrote letters"),
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
        ("The river flows south", "The river flows north"),
        ("He reads fiction", "He reads nonfiction"),
        ("The bird sang loudly", "The bird sang quietly"),
        ("Winter is cold", "Summer is cold"),
        ("The doctor cured patients", "The teacher taught students"),
        ("Gold is expensive", "Silver is expensive"),
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
        ("The book is interesting", "The literary work is captivating"),
        ("We had fun", "We experienced considerable amusement"),
        ("The weather is nice", "The meteorological conditions are favorable"),
        ("She made a mistake", "She committed an error"),
        ("The test was hard", "The examination was formidable"),
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
        ("We build sandcastles", "We built sandcastles"),
        ("The bird flies away", "The bird flew away"),
        ("She paints beautiful pictures", "She painted beautiful pictures"),
        ("I know the answer", "I knew the answer"),
        ("They swim in the lake", "They swam in the lake"),
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
        ("This is possible", "This is impossible"),
        ("The door is open", "The door is closed"),
        ("She agrees with us", "She disagrees with us"),
        ("The project succeeded", "The project failed"),
        ("He is present", "He is absent"),
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
        ("这个男人读一本书", "这些男人读书"),
        ("她已经吃了晚饭", "她以前吃过晚饭"),
        ("他们会回家", "他们回家了"),
        ("我能看到山", "我能看到山吗"),
        ("孩子们在外面玩", "孩子在外面玩"),
    ],
    'semantic': [
        ("猫坐在垫子上", "狗坐在垫子上"),
        ("她走路去学校", "她开车去学校"),
        ("国王统治王国", "女王统治王国"),
        ("他吃了一个苹果", "他吃了一个橘子"),
        ("太阳很亮", "月亮很亮"),
        ("这个男人很高", "这个女人很高"),
        ("我喜欢音乐", "我讨厌音乐"),
        ("车很快", "自行车很快"),
        ("她买了一栋房子", "她租了一栋房子"),
        ("医生治好了病人", "老师教了学生"),
    ],
    'style': [
        ("猫坐在垫子上", "猫咪安歇于地毯之上"),
        ("她走路去学校", "她步行前往学府"),
        ("他非常高兴", "他喜不自胜"),
        ("食物很好吃", "佳肴美味可口"),
        ("我觉得这是对的", "依我之见此乃正确"),
        ("这个电影不好看", "此影片实在不堪入目"),
        ("她说了你好", "她致以问候"),
        ("房子很大", "居所甚为宽敞"),
        ("他跑得快", "他疾步如飞"),
        ("这是个好日子", "今日实乃良辰吉日"),
    ],
    'tense': [
        ("我走路去学校", "我走路去了学校"),
        ("她读一本书", "她读了一本书"),
        ("他们在外面玩", "他们在外面玩了"),
        ("他每天跑步", "他每天跑了步"),
        ("我们一起吃晚饭", "我们一起吃了晚饭"),
        ("太阳在黎明升起", "太阳在黎明升起了"),
        ("她在日记里写字", "她在日记里写了字"),
        ("我明白这个问题", "我明白了这个问题"),
        ("孩子们唱歌", "孩子们唱了歌"),
        ("他开车去上班", "他开车去上了班"),
    ],
    'polarity': [
        ("她很高兴", "她不高兴"),
        ("电影很好看", "电影不好看"),
        ("他会游泳", "他不会游泳"),
        ("我喜欢这首歌", "我不喜欢这首歌"),
        ("考试很简单", "考试不简单"),
        ("她会来", "她不会来"),
        ("他们很有钱", "他们没有钱"),
        ("我们有时间", "我们没有时间"),
        ("食物很新鲜", "食物不新鲜"),
        ("他总是早到", "他从没早到"),
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

    # 提取每个层最后一个token的表示
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
    """提取指定层的功能方向(PCA主成分)"""
    dim_names = list(pairs_dict.keys())
    directions = {}
    direction_pcas = {}

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

        # 堆叠差异向量
        diff_matrix = np.stack(diff_vectors)  # [n_pairs, d_model]

        # 主方向: 平均差异向量
        mean_diff = np.mean(diff_matrix, axis=0)
        mean_diff = mean_diff / (np.linalg.norm(mean_diff) + 1e-30)
        directions[dim_name] = mean_diff

        # PCA获取子空间
        diff_centered = diff_matrix - np.mean(diff_matrix, axis=0, keepdims=True)
        cov = np.cov(diff_centered.T)
        if cov.ndim == 0:
            cov = np.array([[cov]])
        try:
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            # 降序排列
            idx = np.argsort(eigenvalues)[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]

            # 有效秩
            total_var = np.sum(eigenvalues)
            if total_var > 0:
                effective_rank = np.sum(eigenvalues / np.max(eigenvalues)) ** 2 / len(eigenvalues)
            else:
                effective_rank = 0

            direction_pcas[dim_name] = {
                'eigenvalues': eigenvalues[:20],
                'eigenvectors_top10': eigenvectors[:, :10],
                'effective_rank': effective_rank,
                'n_pairs': len(diff_vectors),
            }
        except:
            direction_pcas[dim_name] = {
                'eigenvalues': [],
                'effective_rank': 0,
                'n_pairs': len(diff_vectors),
            }

    return directions, direction_pcas


# ============================================================
# P771: 精确功能方向提取
# ============================================================

def P771_precise_functional_directions(model, tokenizer, device, model_name, results):
    """
    用15对句子提取更精确的功能方向, 与CLXXVI的10对对比
    """
    print("\n--- P771: 精确功能方向提取 ---")

    n_vocab, d_model = model.lm_head.weight.shape
    n_layers = len(get_layers(model))

    # 提取关键层的功能方向
    key_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 2, n_layers - 1]
    key_layers = [l for l in key_layers if l < n_layers]

    all_layer_directions = {}
    all_layer_pcas = {}
    stability_metrics = {}

    for layer_idx in key_layers:
        print(f"  Layer {layer_idx}...")
        directions, pcas = extract_functional_directions(
            model, tokenizer, device, ENGLISH_FUNCTIONAL_PAIRS, layer_idx
        )
        all_layer_directions[layer_idx] = directions
        all_layer_pcas[layer_idx] = pcas

    # 跨层方向稳定性: 相邻层方向的余弦相似度
    dim_names = list(ENGLISH_FUNCTIONAL_PAIRS.keys())
    for dim_name in dim_names:
        if dim_name not in all_layer_directions.get(0, {}):
            continue
        stability_metrics[dim_name] = {}
        sorted_layers = sorted(all_layer_directions.keys())
        for i in range(len(sorted_layers) - 1):
            l1, l2 = sorted_layers[i], sorted_layers[i + 1]
            if dim_name in all_layer_directions[l1] and dim_name in all_layer_directions[l2]:
                d1 = all_layer_directions[l1][dim_name]
                d2 = all_layer_directions[l2][dim_name]
                cos = float(np.dot(d1, d2) / (np.linalg.norm(d1) * np.linalg.norm(d2) + 1e-30))
                stability_metrics[dim_name][f"L{l1}_L{l2}"] = cos

    # 正交性验证 (关键层)
    orthogonality = {}
    for layer_idx in key_layers:
        if layer_idx not in all_layer_directions:
            continue
        dirs = all_layer_directions[layer_idx]
        layer_orth = {}
        dim_list = [d for d in dim_names if d in dirs]
        for i in range(len(dim_list)):
            for j in range(i + 1, len(dim_list)):
                d1, d2 = dim_list[i], dim_list[j]
                cos = float(np.dot(dirs[d1], dirs[d2]))
                layer_orth[f"{d1}_vs_{d2}"] = cos
        orthogonality[layer_idx] = layer_orth

    # 有效秩
    effective_ranks = {}
    for layer_idx in key_layers:
        if layer_idx not in all_layer_pcas:
            continue
        effective_ranks[layer_idx] = {}
        for dim_name in dim_names:
            if dim_name in all_layer_pcas[layer_idx]:
                effective_ranks[layer_idx][dim_name] = float(all_layer_pcas[layer_idx][dim_name]['effective_rank'])

    results['p771_precise_directions'] = {
        'stability_metrics': stability_metrics,
        'orthogonality': orthogonality,
        'effective_ranks': effective_ranks,
        'key_layers': key_layers,
    }

    # 打印总结
    print(f"\n  跨层方向稳定性 (Layer 0 vs Last Layer):")
    for dim_name in dim_names:
        if dim_name in stability_metrics:
            pairs = list(stability_metrics[dim_name].values())
            print(f"    {dim_name}: avg_cos = {np.mean(pairs):.4f}")

    print(f"\n  Layer 0 正交性:")
    if 0 in orthogonality:
        for pair_name, cos_val in sorted(orthogonality[0].items(), key=lambda x: abs(x[1]), reverse=True)[:5]:
            print(f"    {pair_name}: cos = {cos_val:.4f}")

    print(f"\n  Layer 0 有效秩:")
    if 0 in effective_ranks:
        for dim_name, rank in effective_ranks[0].items():
            print(f"    {dim_name}: {rank:.2f}")


# ============================================================
# P772: 功能空间旋转动力学
# ============================================================

def P772_rotation_dynamics(model, tokenizer, device, model_name, results):
    """
    提取每层的功能方向矩阵, 分析相邻层的旋转
    """
    print("\n--- P772: 功能空间旋转动力学 ---")

    n_vocab, d_model = model.lm_head.weight.shape
    n_layers = len(get_layers(model))
    dim_names = list(ENGLISH_FUNCTIONAL_PAIRS.keys())

    # 对每一层提取功能方向
    print("  提取全层功能方向...")
    all_directions = {}  # layer_idx -> {dim_name: direction}

    # 为了效率, 只采样部分层
    sample_layers = list(range(0, n_layers, max(1, n_layers // 20)))
    if (n_layers - 1) not in sample_layers:
        sample_layers.append(n_layers - 1)

    for layer_idx in sample_layers:
        directions, _ = extract_functional_directions(
            model, tokenizer, device, ENGLISH_FUNCTIONAL_PAIRS, layer_idx
        )
        all_directions[layer_idx] = directions

    # 对每层构建方向矩阵 D_l = [d1, d2, d3, d4, d5]  (5个功能维度)
    direction_matrices = {}
    for layer_idx in sorted(all_directions.keys()):
        dirs = all_directions[layer_idx]
        available_dims = [d for d in dim_names if d in dirs]
        if len(available_dims) >= 2:
            D = np.stack([dirs[d] for d in available_dims], axis=1)  # [d_model, n_dims]
            direction_matrices[layer_idx] = (D, available_dims)

    # 计算相邻层的旋转矩阵 R_l = D_{l+1} @ D_l^T
    rotations = {}
    rotation_angles = {}
    sorted_layers = sorted(direction_matrices.keys())

    for i in range(len(sorted_layers) - 1):
        l1, l2 = sorted_layers[i], sorted_layers[i + 1]
        D1, dims1 = direction_matrices[l1]
        D2, dims2 = direction_matrices[l2]

        # 只用共同维度
        common_dims = [d for d in dims1 if d in dims2]
        if len(common_dims) < 2:
            continue

        idx1 = [dims1.index(d) for d in common_dims]
        idx2 = [dims2.index(d) for d in common_dims]

        D1_sub = D1[:, idx1]  # [d_model, n_common]
        D2_sub = D2[:, idx2]

        # 功能子空间内的旋转: R = (D2_sub^T @ D1_sub)
        # 这是 n_common x n_common 的矩阵
        R_inner = D2_sub.T @ D1_sub  # [n_common, n_common]

        # 计算旋转角
        try:
            # SVD分解 R_inner = U @ S @ V^T
            U, S, Vt = np.linalg.svd(R_inner)
            # 旋转角: arccos of singular values
            angles = np.arccos(np.clip(S, -1, 1))
            rotation_angles[f"L{l1}_L{l2}"] = {
                'angles_deg': [float(a * 180 / np.pi) for a in angles],
                'singular_values': S.tolist(),
                'det': float(np.linalg.det(R_inner)),
                'frobenius_dist_from_identity': float(np.linalg.norm(R_inner - np.eye(len(common_dims)), 'fro')),
            }
        except:
            rotation_angles[f"L{l1}_L{l2}"] = {'angles_deg': [], 'singular_values': [], 'det': 0}

        rotations[f"L{l1}_L{l2}"] = R_inner.tolist()

    # 分析旋转角的变化模式
    angle_summary = {}
    for key, info in rotation_angles.items():
        if info['angles_deg']:
            angle_summary[key] = {
                'max_angle': max(info['angles_deg']),
                'mean_angle': float(np.mean(info['angles_deg'])),
                'det': info['det'],
                'frob_dist': info.get('frobenius_dist_from_identity', 0),
            }

    # 检查是否构成群: 旋转矩阵的行列式应为±1
    dets = [info['det'] for info in rotation_angles.values() if 'det' in info and info['det'] != 0]

    results['p772_rotation_dynamics'] = {
        'rotation_angles': angle_summary,
        'sample_layers': sample_layers,
        'det_analysis': {
            'mean_det': float(np.mean(dets)) if dets else 0,
            'std_det': float(np.std(dets)) if dets else 0,
            'det_close_to_pm1': all(abs(d) - 1 < 0.3 for d in dets) if dets else False,
        },
    }

    print(f"\n  旋转角分析 (采样 {len(sample_layers)} 层):")
    for key, info in sorted(angle_summary.items()):
        print(f"    {key}: max_angle={info['max_angle']:.1f}°, mean={info['mean_angle']:.1f}°, det={info['det']:.4f}")

    if dets:
        print(f"\n  行列式分析: mean_det={np.mean(dets):.4f}, std={np.std(dets):.4f}")
        print(f"  det接近±1? {all(abs(d) - 1 < 0.3 for d in dets)}")


# ============================================================
# P773: 非功能维度分析
# ============================================================

def P773_nonfunctional_analysis(model, tokenizer, device, model_name, results):
    """
    分析残差流中99%非功能维度的结构
    """
    print("\n--- P773: 非功能维度分析 ---")

    n_vocab, d_model = model.lm_head.weight.shape
    n_layers = len(get_layers(model))
    dim_names = list(ENGLISH_FUNCTIONAL_PAIRS.keys())

    # 收集Layer 0的功能方向
    directions_l0, _ = extract_functional_directions(
        model, tokenizer, device, ENGLISH_FUNCTIONAL_PAIRS, layer_idx=0
    )

    if not directions_l0:
        print("  未能提取功能方向, 跳过P773")
        return

    # 构建功能子空间投影矩阵
    available_dims = [d for d in dim_names if d in directions_l0]
    D = np.stack([directions_l0[d] for d in available_dims], axis=1)  # [d_model, n_funcs]

    # 用PCA获取完整的功能子空间(每维度多主成分)
    # 收集所有句子对的差异向量
    all_diff_vectors = []
    for dim_name, pairs in ENGLISH_FUNCTIONAL_PAIRS.items():
        for s1, s2 in pairs:
            r1 = _collect_residual_stream(model, tokenizer, device, s1)
            r2 = _collect_residual_stream(model, tokenizer, device, s2)
            if 0 in r1 and 0 in r2:
                all_diff_vectors.append(r1[0] - r2[0])

    if not all_diff_vectors:
        print("  未能收集差异向量, 跳过")
        return

    diff_matrix = np.stack(all_diff_vectors)  # [n_total_pairs, d_model]

    # SVD分解差异矩阵
    print(f"  差异矩阵形状: {diff_matrix.shape}")
    U, S, Vt = np.linalg.svd(diff_matrix, full_matrices=False)
    # Vt: [min(n, d), d_model] — 行是右奇异向量

    # 累积方差解释比
    total_var = np.sum(S ** 2)
    cum_var_ratio = np.cumsum(S ** 2) / total_var

    # 找到90%和99%方差需要的维度
    n_90 = int(np.searchsorted(cum_var_ratio, 0.90)) + 1
    n_99 = int(np.searchsorted(cum_var_ratio, 0.99)) + 1
    n_999 = int(np.searchsorted(cum_var_ratio, 0.999)) + 1

    print(f"  功能子空间SVD:")
    print(f"    90%方差需要 {n_90} 维")
    print(f"    99%方差需要 {n_99} 维")
    print(f"    99.9%方差需要 {n_999} 维")

    # 功能子空间 vs 全空间的投影
    # 功能子空间: Vt[:n_99, :] 张成的空间
    V_func = Vt[:n_99, :]  # [n_99, d_model]
    P_func = V_func.T @ V_func  # [d_model, d_model] 投影矩阵

    # 非功能子空间: I - P_func
    P_nonfunc = np.eye(d_model) - P_func

    # 分析非功能空间: 收集一些实际残差流
    # 使用一些句子, 分析其在非功能空间中的结构
    test_sentences = [
        "The weather is beautiful today",
        "Scientists discovered a new species",
        "She opened the door quietly",
        "The quantum computer solved the problem",
        "Mountains rise above the clouds",
    ]

    residuals_func = []
    residuals_nonfunc = []

    for sent in test_sentences:
        r = _collect_residual_stream(model, tokenizer, device, sent)
        if 0 in r:
            h = r[0]
            h_func = P_func @ h
            h_nonfunc = P_nonfunc @ h
            residuals_func.append(float(np.linalg.norm(h_func)))
            residuals_nonfunc.append(float(np.linalg.norm(h_nonfunc)))

    # 非功能子空间分析 (避免大矩阵分配)
    # 方法: 用W_U^T @ W_U计算右奇异向量(更省内存)
    W_U = to_numpy(model.lm_head.weight.data)  # [n_vocab, d_model]

    # W_U^T @ W_U 的特征分解 = V @ S^2 @ V^T
    print("  计算W_U的右奇异向量(通过W_U^T@W_U)...")
    # 分批计算W_U^T @ W_U
    WuTWu = np.zeros((d_model, d_model), dtype=np.float32)
    batch_size = 5000
    for start in range(0, W_U.shape[0], batch_size):
        end = min(start + batch_size, W_U.shape[0])
        WuTWu += W_U[start:end].T @ W_U[start:end]

    eigenvalues, eigenvectors = np.linalg.eigh(WuTWu)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # 奇异值 = sqrt(特征值)
    S_wu = np.sqrt(np.maximum(eigenvalues, 0))

    # 分析W_U的奇异向量与功能空间的对齐
    print(f"\n  W_U奇异向量与功能空间的对齐:")
    func_alignment = {}
    for i in range(min(30, len(eigenvectors))):
        v = eigenvectors[:, i]  # [d_model]
        proj_norm = float(np.linalg.norm(V_func @ v))
        total_norm = float(np.linalg.norm(v))
        alignment = proj_norm / total_norm if total_norm > 0 else 0
        func_alignment[i] = alignment
        if i < 10:
            print(f"    SV{i}: 功能对齐度 = {alignment:.4f}")

    # 找到功能空间主要对应的W_U奇异向量
    high_align_sv = [(i, a) for i, a in func_alignment.items() if a > 0.1]
    print(f"\n  功能对齐度>0.1的W_U奇异向量: {len(high_align_sv)} 个")
    for i, a in high_align_sv[:10]:
        # 对应的top token
        loadings = W_U @ eigenvectors[:, i]  # [n_vocab]
        top_idx = np.argmax(np.abs(loadings))
        try:
            top_token = tokenizer.decode([top_idx])
            # 安全编码: 只保留ASCII可打印字符
            top_token = top_token.encode('ascii', 'replace').decode('ascii')
        except:
            top_token = str(top_idx)
        print(f"    SV{i}: alignment={a:.4f}, top_token={top_token}")

    # 功能空间和W_U top奇异向量的能量分布
    total_wu_var = float(np.sum(S_wu[:50] ** 2))
    wu_func_energy = 0
    for i in range(min(50, len(eigenvectors))):
        v = eigenvectors[:, i]
        alignment = float(np.linalg.norm(V_func @ v) / (np.linalg.norm(v) + 1e-30))
        wu_func_energy += S_wu[i] ** 2 * alignment ** 2
    wu_func_ratio = wu_func_energy / total_wu_var if total_wu_var > 0 else 0

    print(f"\n  W_U能量分布:")
    print(f"    功能空间占比: {wu_func_ratio:.4f}")
    print(f"    非功能空间占比: {1-wu_func_ratio:.4f}")

    # 功能/非功能能量比
    func_energy = float(np.mean(residuals_func)) if residuals_func else 0
    nonfunc_energy = float(np.mean(residuals_nonfunc)) if residuals_nonfunc else 0

    results['p773_nonfunctional'] = {
        'functional_svd': {
            'n_90': n_90,
            'n_99': n_99,
            'n_999': n_999,
            'top_20_singular_values': S[:20].tolist(),
            'cum_var_ratio_top20': cum_var_ratio[:20].tolist(),
        },
        'wu_analysis': {
            'wu_func_energy_ratio': wu_func_ratio,
            'wu_nonfunc_energy_ratio': 1 - wu_func_ratio,
            'func_alignment': func_alignment,
            'n_high_align_sv': len(high_align_sv),
        },
        'energy_split': {
            'functional_norm_mean': func_energy,
            'nonfunctional_norm_mean': nonfunc_energy,
            'ratio': nonfunc_energy / (func_energy + 1e-30),
        },
    }


# ============================================================
# P774: 中英文功能基底对比
# ============================================================

def P774_cross_language_comparison(model, tokenizer, device, model_name, results):
    """
    对比中英文的功能方向, 检查是否旋转等价
    """
    print("\n--- P774: 中英文功能基底对比 ---")

    dim_names = list(ENGLISH_FUNCTIONAL_PAIRS.keys())

    # 英文功能方向 (Layer 0)
    en_dirs, en_pcas = extract_functional_directions(
        model, tokenizer, device, ENGLISH_FUNCTIONAL_PAIRS, layer_idx=0
    )

    # 中文功能方向 (Layer 0)
    zh_dirs, zh_pcas = extract_functional_directions(
        model, tokenizer, device, CHINESE_FUNCTIONAL_PAIRS, layer_idx=0
    )

    # 比较同维度的中英文方向
    cross_lang_cos = {}
    common_dims = [d for d in dim_names if d in en_dirs and d in zh_dirs]

    for dim_name in common_dims:
        cos = float(np.dot(en_dirs[dim_name], zh_dirs[dim_name]))
        cross_lang_cos[dim_name] = cos
        print(f"  {dim_name}: EN-ZH cos = {cos:.4f}")

    # 构建方向矩阵
    if len(common_dims) >= 2:
        D_en = np.stack([en_dirs[d] for d in common_dims], axis=1)  # [d_model, n_dims]
        D_zh = np.stack([zh_dirs[d] for d in common_dims], axis=1)

        # 正交Procrustes: 找旋转R使 D_zh ≈ D_en @ R
        R, scale = orthogonal_procrustes(D_zh, D_en)

        # 重建误差
        D_zh_reconstructed = D_en @ R
        recon_error = float(np.linalg.norm(D_zh - D_zh_reconstructed, 'fro'))
        recon_cos = float(np.trace(D_zh.T @ D_zh_reconstructed) / (
            np.linalg.norm(D_zh, 'fro') * np.linalg.norm(D_zh_reconstructed, 'fro') + 1e-30
        ))

        # R的性质
        R_det = float(np.linalg.det(R))
        R_orth_err = float(np.linalg.norm(R @ R.T - np.eye(len(common_dims)), 'fro'))

        print(f"\n  Procrustes分析:")
        print(f"    重建Frobenius误差: {recon_error:.4f}")
        print(f"    重建余弦: {recon_cos:.4f}")
        print(f"    R行列式: {R_det:.4f}")
        print(f"    R正交性误差: {R_orth_err:.6f}")

        # 逐维度重建质量
        per_dim_recon = {}
        for i, dim_name in enumerate(common_dims):
            orig = D_zh[:, i]
            recon = D_zh_reconstructed[:, i]
            cos = float(np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-30))
            per_dim_recon[dim_name] = cos
            print(f"    {dim_name} 重建cos: {cos:.4f}")

        results['p774_cross_language'] = {
            'direct_cosine': cross_lang_cos,
            'procrustes': {
                'reconstruction_error': recon_error,
                'reconstruction_cosine': recon_cos,
                'R_determinant': R_det,
                'R_orthogonality_error': R_orth_err,
                'R_matrix': R.tolist(),
                'per_dim_reconstruction_cosine': per_dim_recon,
            },
            'n_common_dims': len(common_dims),
        }
    else:
        results['p774_cross_language'] = {
            'direct_cosine': cross_lang_cos,
            'procrustes': None,
            'n_common_dims': len(common_dims),
        }

    # 有效秩对比
    rank_comparison = {}
    for dim_name in common_dims:
        en_rank = float(en_pcas[dim_name]['effective_rank']) if dim_name in en_pcas else 0
        zh_rank = float(zh_pcas[dim_name]['effective_rank']) if dim_name in zh_pcas else 0
        rank_comparison[dim_name] = {'en_rank': en_rank, 'zh_rank': zh_rank}

    results['p774_cross_language']['rank_comparison'] = rank_comparison

    # 也在最后层做对比
    n_layers = len(get_layers(model))
    last_layer = n_layers - 1

    en_dirs_last, _ = extract_functional_directions(
        model, tokenizer, device, ENGLISH_FUNCTIONAL_PAIRS, layer_idx=last_layer
    )
    zh_dirs_last, _ = extract_functional_directions(
        model, tokenizer, device, CHINESE_FUNCTIONAL_PAIRS, layer_idx=last_layer
    )

    last_layer_cos = {}
    for dim_name in common_dims:
        if dim_name in en_dirs_last and dim_name in zh_dirs_last:
            cos = float(np.dot(en_dirs_last[dim_name], zh_dirs_last[dim_name]))
            last_layer_cos[dim_name] = cos

    results['p774_cross_language']['last_layer_cosine'] = last_layer_cos
    print(f"\n  最后层 EN-ZH cos:")
    for dim_name, cos_val in last_layer_cos.items():
        print(f"    {dim_name}: {cos_val:.4f}")


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Phase CLXXVII: 功能几何学精确刻画")
    parser.add_argument('--model', type=str, required=True,
                       choices=['glm4', 'qwen3', 'deepseek7b'],
                       help='模型名称')
    args = parser.parse_args()

    model_name = args.model
    print(f"=== Phase CLXXVII: 功能几何学精确刻画 - {model_name} ===")

    # 加载模型
    model, tokenizer, device = load_model(model_name)
    info = get_model_info(model, model_name)
    print(f"模型信息: {info.n_layers}层, d_model={info.d_model}, vocab={info.vocab_size}")

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
    P771_precise_functional_directions(model, tokenizer, device, model_name, results)
    P772_rotation_dynamics(model, tokenizer, device, model_name, results)
    P773_nonfunctional_analysis(model, tokenizer, device, model_name, results)
    P774_cross_language_comparison(model, tokenizer, device, model_name, results)

    # 保存结果
    output_dir = Path(__file__).parent / 'results' / 'phase_clxxvii'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}_results.json"

    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, cls=NumpyEncoder, indent=2, ensure_ascii=False)

    print(f"\n结果已保存到: {output_file}")

    # 释放模型
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("模型已释放")


if __name__ == '__main__':
    main()
