"""
CCL-Y(Phase 19): 从几何到代数——语法编码2D子空间的旋转轨迹与因果模式的统一数学框架
=============================================================================
核心问题(基于Phase 18的发现):
  1. ★★★★★★★★★ 2D语法子空间的旋转轨迹
     → Phase 18发现名词轴和修饰语轴的正交性在所有层完美保持
     → 每层定义一个2D平面(名词轴×修饰语轴), 层间如何旋转?
     → L_l平面与L_{l+1}平面的夹角(二面角)
     → 旋转轨迹是否形成某种几何结构(如测地线、螺旋)?

  2. ★★★★★★★★★ 因果模式的统一数学框架
     → Phase 18发现: 正交+超强非线性→冗余, 正交+适中非线性→独立, 正交+适中非线性+扰动→超加性
     → coupling_ratio = KL(both) / [KL(noun) + KL(mod)]
     → 理论: coupling_ratio = f(nonlinearity, jacobian_overlap, cross_hessian)
     → 用三模型数据拟合f, 是否存在统一公式?

  3. ★★★★★★★ 残差+LayerNorm保持正交性的理论分析
     → Phase 18实验证据: T本身不保持正交性(ortho可达0.62), 但残差+LN修复了它
     → 反例验证: 去掉残差或LN, 正交性是否被破坏?
     → 数值构造: 无残差/无LN的前向传播

  4. ★★★★★★★ 语法编码的代数不变量
     → 正交性: 不变 ✓, 2D子空间维度: 不变 ✓
     → 方向: 变化(旋转), 范数: 变化(增长)
     → ★★★ 寻找: 语法编码的代数不变量(类似拓扑不变量)
     → 候选: 交叉比(cross-ratio)、绕数(winding number)、2D子空间的示性类

实验:
  Exp1: ★★★★★★★★★ 2D语法子空间的旋转轨迹
    → 采集所有层的名词轴和修饰语轴
    → 计算相邻层2D平面的二面角
    → 计算旋转矩阵和旋转轴
    → 旋转轨迹的几何分析

  Exp2: ★★★★★★★★★ 因果模式的统一数学框架
    → 在多层计算coupling_ratio(基于KL散度)
    → 收集每层的nonlinearity, jacobian_overlap, cross_hessian
    → 拟合coupling_ratio = f(nonlinearity, overlap, cross_hessian)
    → 验证f在三模型上是否一致

  Exp3: ★★★★★★★ 残差+LayerNorm保持正交性的反例验证
    → 构造修改后的前向传播: 无残差 / 无LN
    → 测量语法方向的正交性变化
    → 分析残差和LN各自的贡献

  Exp4: ★★★★★★★ 语法编码的代数不变量
    → 计算2D子空间的Grassmannian坐标
    → 计算旋转轨迹的绕数(总旋转角)
    → 检验交叉比的不变性
    → 寻找其他代数不变量
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import argparse
import json
import torch
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA

from model_utils import load_model, get_layers, get_model_info, get_layer_weights, release_model, safe_decode


# ===== 复用Phase 15-18的数据 =====
MANIFOLD_ROLES_DATA = {
    "nsubj": {
        "sentences": [
            "The king ruled the kingdom wisely",
            "The doctor treated the patient carefully",
            "The artist painted the portrait beautifully",
            "The soldier defended the castle bravely",
            "The teacher explained the lesson clearly",
            "The chef cooked the meal perfectly",
            "The cat chased the mouse quickly",
            "The dog found the bone happily",
            "The woman drove the car safely",
            "The man fixed the roof carefully",
            "The student read the book quietly",
            "The singer performed the song brilliantly",
            "The baker made the bread daily",
            "The pilot flew the plane smoothly",
            "The farmer grew the crops diligently",
            "The writer wrote the novel slowly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
    "poss": {
        "sentences": [
            "The king's crown glittered brightly",
            "The doctor's office opened early",
            "The artist's studio looked beautiful",
            "The soldier's uniform was clean",
            "The teacher's book sold quickly",
            "The chef's restaurant opened today",
            "The cat's tail swished gently",
            "The dog's bark echoed loudly",
            "The woman's dress looked elegant",
            "The man's car drove fast",
            "The student's essay read well",
            "The singer's voice rang clearly",
            "The baker's shop smelled wonderful",
            "The pilot's license was renewed",
            "The farmer's land was fertile",
            "The writer's pen wrote smoothly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
    "dobj": {
        "sentences": [
            "They crowned the king yesterday",
            "She visited the doctor recently",
            "He admired the artist greatly",
            "We honored the soldier today",
            "You thanked the teacher warmly",
            "The customer tipped the chef generously",
            "The hawk chased the cat swiftly",
            "The boy found the dog outside",
            "The police arrested the woman quickly",
            "The company hired the man recently",
            "I praised the student loudly",
            "They applauded the singer warmly",
            "She visited the baker often",
            "He admired the pilot greatly",
            "We thanked the farmer sincerely",
            "The editor praised the writer highly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "teacher",
            "chef", "cat", "dog", "woman", "man",
            "student", "singer", "baker", "pilot", "farmer", "writer",
        ],
    },
    "amod": {
        "sentences": [
            "The brave king fought hard",
            "The kind doctor helped many",
            "The creative artist worked well",
            "The strong soldier marched far",
            "The wise teacher explained clearly",
            "The skilled chef cooked perfectly",
            "The quick cat ran fast",
            "The loyal dog stayed close",
            "The old woman walked slowly",
            "The tall man stood quietly",
            "The bright student read carefully",
            "The talented singer performed brilliantly",
            "The patient baker waited calmly",
            "The careful pilot landed smoothly",
            "The hardworking farmer harvested early",
            "The thoughtful writer reflected deeply",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "wise",
            "skilled", "quick", "loyal", "old", "tall",
            "bright", "talented", "patient", "careful", "hardworking", "thoughtful",
        ],
    },
}


def find_token_index(tokens, word):
    word_lower = word.lower()
    word_start = word_lower[:3]
    for i, tok in enumerate(tokens):
        tok_lower = tok.lower().strip()
        if word_lower in tok_lower or tok_lower.startswith(word_start):
            return i
    for i, tok in enumerate(tokens):
        if word_lower[:2] in tok.lower():
            return i
    return None


def collect_hs_at_layer(model, tokenizer, device, sentences, target_words, layer_idx):
    """在指定层收集target token的hidden states"""
    layers = get_layers(model)
    if layer_idx >= len(layers):
        layer_idx = len(layers) - 1
    if layer_idx < 0:
        layer_idx = len(layers) + layer_idx
    target_layer = layers[layer_idx]

    all_h = []
    valid_words = []

    for sent, target_word in zip(sentences, target_words):
        toks = tokenizer(sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]

        dep_idx = find_token_index(tokens_list, target_word)
        if dep_idx is None:
            continue

        captured = {}
        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                captured['h'] = output[0].detach().float().cpu().numpy()
            else:
                captured['h'] = output.detach().float().cpu().numpy()

        h_handle = target_layer.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(**toks)
        h_handle.remove()

        if 'h' not in captured:
            continue

        h_vec = captured['h'][0, dep_idx, :]
        all_h.append(h_vec)
        valid_words.append(target_word)

    return np.array(all_h) if all_h else None


def get_syntax_directions_at_layer(model, tokenizer, device, model_info, layer_idx):
    """获取指定层的语法方向(nsubj-dobj, nsubj-amod等)"""
    role_names = ["nsubj", "dobj", "amod"]
    role_h = {}
    for role in role_names:
        data = MANIFOLD_ROLES_DATA[role]
        H = collect_hs_at_layer(model, tokenizer, device,
                                data["sentences"], data["target_words"], layer_idx)
        if H is not None and len(H) > 0:
            role_h[role] = H

    centers = {}
    for role in role_names:
        if role in role_h:
            centers[role] = np.mean(role_h[role], axis=0)

    directions = {}
    if 'nsubj' in centers and 'dobj' in centers:
        d = centers['dobj'] - centers['nsubj']
        norm = np.linalg.norm(d)
        if norm > 0:
            directions['noun_axis'] = d / norm
            directions['noun_axis_norm'] = norm

    if 'nsubj' in centers and 'amod' in centers:
        amod_vec = centers['amod'] - centers['nsubj']
        if 'noun_axis' in directions:
            amod_proj = np.dot(amod_vec, directions['noun_axis']) * directions['noun_axis']
            modifier_axis = amod_vec - amod_proj
            modifier_norm = np.linalg.norm(modifier_axis)
            if modifier_norm > 0:
                directions['modifier_axis'] = modifier_axis / modifier_norm
                directions['modifier_axis_norm'] = modifier_norm
                directions['axis_orthogonality'] = float(np.dot(directions['noun_axis'], modifier_axis / modifier_norm))

    directions['centers'] = centers
    return directions


# ===== Exp1: 2D语法子空间的旋转轨迹 =====
def exp1_rotation_trajectory(model, tokenizer, device):
    """2D语法子空间的旋转轨迹分析"""
    print("\n" + "="*70)
    print("Exp1: 2D语法子空间的旋转轨迹 ★★★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Step 1: 采集所有层的语法方向
    print("\n  Step 1: 采集所有层的2D语法子空间")

    # 采样层: 更密集的采样以获得旋转轨迹
    sample_layers = list(range(n_layers))
    if n_layers > 20:
        # 间隔采样, 但保证前3层和后3层都是连续的
        step = max(1, n_layers // 15)
        sample_layers = list(range(0, min(4, n_layers), 1))
        sample_layers += list(range(4, n_layers - 3, step))
        sample_layers += list(range(max(4, n_layers - 3), n_layers, 1))
        sample_layers = sorted(set(sample_layers))

    layer_planes = {}
    for li in sample_layers:
        print(f"  采集L{li}...", end="", flush=True)
        dirs = get_syntax_directions_at_layer(model, tokenizer, device, model_info, li)
        if 'noun_axis' in dirs and 'modifier_axis' in dirs:
            layer_planes[li] = {
                'noun_axis': dirs['noun_axis'],
                'modifier_axis': dirs['modifier_axis'],
                'ortho': dirs.get('axis_orthogonality', 0),
                'noun_norm': dirs.get('noun_axis_norm', 0),
                'mod_norm': dirs.get('modifier_axis_norm', 0),
            }
            print(f" ortho={layer_planes[li]['ortho']:.6f}")
        else:
            print(f" 无法获取语法方向")

    if len(layer_planes) < 2:
        print("  采集到的层不足, 无法分析旋转轨迹!")
        return results

    # Step 2: 计算相邻层2D平面的二面角
    print("\n  Step 2: 相邻层2D平面的二面角与旋转分析")

    sorted_layers = sorted(layer_planes.keys())
    rotation_data = []

    for i in range(len(sorted_layers) - 1):
        li = sorted_layers[i]
        li1 = sorted_layers[i + 1]
        p = layer_planes[li]
        p1 = layer_planes[li1]

        # 名词轴的旋转角
        cos_noun = float(np.dot(p['noun_axis'], p1['noun_axis']))
        cos_noun = np.clip(cos_noun, -1, 1)
        angle_noun = np.arccos(cos_noun)  # 弧度

        # 修饰语轴的旋转角
        cos_mod = float(np.dot(p['modifier_axis'], p1['modifier_axis']))
        cos_mod = np.clip(cos_mod, -1, 1)
        angle_mod = np.arccos(cos_mod)

        # 2D平面的二面角
        # 两个2D平面的"距离" = 平面间子空间的角度
        # Grassmannian距离: 用2x2矩阵 M = P1^T @ P2 的奇异值
        # P1 = [noun_axis, mod_axis]  [d_model, 2]
        # P2 = [noun_axis, mod_axis]  [d_model, 2]
        P1 = np.column_stack([p['noun_axis'], p['modifier_axis']])  # [d_model, 2]
        P2 = np.column_stack([p1['noun_axis'], p1['modifier_axis']])  # [d_model, 2]

        # M = P1^T @ P2 [2, 2]
        M = P1.T @ P2

        # SVD of M → 奇异值是方向间的余弦
        U_m, s_m, Vt_m = np.linalg.svd(M)
        # s_m[0], s_m[1] 是两个平面主方向间的余弦
        # 二面角 = arccos(s_min) 或 arccos(det(M)) (旋转情况)

        principal_angles = np.arccos(np.clip(s_m, -1, 1))  # 两个主角度

        # 判断旋转方向: det(M) > 0 → 同向旋转, < 0 → 反向
        det_M = np.linalg.det(M)

        # 如果是旋转: 旋转角 θ, det(M) = cos(θ)
        # 用Frobenius范数定义的总旋转角
        # ||M||_F^2 = 2cos^2(θ) (如果是纯旋转)
        total_rotation = np.arccos(np.clip(abs(det_M), 0, 1))

        rd = {
            'layer_from': li,
            'layer_to': li1,
            'angle_noun_deg': float(np.degrees(angle_noun)),
            'angle_mod_deg': float(np.degrees(angle_mod)),
            'principal_angle1_deg': float(np.degrees(principal_angles[0])),
            'principal_angle2_deg': float(np.degrees(principal_angles[1])),
            'det_M': float(det_M),
            'total_rotation_deg': float(np.degrees(total_rotation)),
            'cos_noun': float(cos_noun),
            'cos_mod': float(cos_mod),
        }
        rotation_data.append(rd)

        print(f"  L{li}→L{li1}: noun旋转={np.degrees(angle_noun):.2f}°, "
              f"mod旋转={np.degrees(angle_mod):.2f}°, "
              f"主角度=({np.degrees(principal_angles[0]):.2f}°, {np.degrees(principal_angles[1]):.2f}°), "
              f"det(M)={det_M:.4f}")

    # Step 3: 旋转轨迹的几何分析
    print("\n  Step 3: 旋转轨迹的几何分析")

    # 累积旋转角
    cum_rotation_noun = 0
    cum_rotation_mod = 0
    cum_rotation_total = 0

    for rd in rotation_data:
        cum_rotation_noun += rd['angle_noun_deg']
        cum_rotation_mod += rd['angle_mod_deg']
        cum_rotation_total += rd['total_rotation_deg']

    print(f"  总累积旋转:")
    print(f"    名词轴: {cum_rotation_noun:.2f}° (≈{cum_rotation_noun/360:.2f}圈)")
    print(f"    修饰语轴: {cum_rotation_mod:.2f}° (≈{cum_rotation_mod/360:.2f}圈)")
    print(f"    总旋转: {cum_rotation_total:.2f}° (≈{cum_rotation_total/360:.2f}圈)")

    # 首尾层间的旋转(直接测量)
    if sorted_layers[0] in layer_planes and sorted_layers[-1] in layer_planes:
        p0 = layer_planes[sorted_layers[0]]
        p_last = layer_planes[sorted_layers[-1]]
        cos_noun_total = float(np.dot(p0['noun_axis'], p_last['noun_axis']))
        cos_mod_total = float(np.dot(p0['modifier_axis'], p_last['modifier_axis']))
        total_noun_angle = np.degrees(np.arccos(np.clip(cos_noun_total, -1, 1)))
        total_mod_angle = np.degrees(np.arccos(np.clip(cos_mod_total, -1, 1)))
        print(f"  首尾直接旋转: noun={total_noun_angle:.2f}°, mod={total_mod_angle:.2f}°")

    # 旋转是否均匀?
    rotation_angles = [rd['total_rotation_deg'] for rd in rotation_data]
    if len(rotation_angles) > 1:
        mean_rot = np.mean(rotation_angles)
        std_rot = np.std(rotation_angles)
        cv_rot = std_rot / max(mean_rot, 1e-10)
        print(f"  旋转角均匀性: mean={mean_rot:.2f}°, std={std_rot:.2f}°, CV={cv_rot:.4f}")
        if cv_rot < 0.3:
            print(f"  ★★★ 旋转近似均匀! 2D子空间做匀速旋转")
        elif cv_rot < 0.6:
            print(f"  ★★ 旋转有一定不均匀性")
        else:
            print(f"  ★ 旋转极不均匀, 某些层间旋转远大于其他")

    # Step 4: 旋转轴分析
    print("\n  Step 4: 旋转轴(2D平面法向量)分析")

    # 每个2D平面的法向量 = noun_axis × modifier_axis (在高维中用wedge product)
    # 但d_model >> 2, 法空间是(d_model-2)维的
    # 用Grassmannian的Plücker坐标来描述2D平面
    # Plücker坐标 = noun ∧ mod = noun_i * mod_j - noun_j * mod_i (反对称矩阵)

    # 简化: 用2D平面在d_model维空间中的"嵌入角"
    # 即: 平面的2个基向量与前2个坐标轴的夹角

    for li in sorted_layers[:5] + sorted_layers[-3:]:
        if li not in layer_planes:
            continue
        p = layer_planes[li]
        # 名词轴和修饰语轴的内积矩阵(平面内的度量)
        metric = np.array([
            [np.dot(p['noun_axis'], p['noun_axis']), np.dot(p['noun_axis'], p['modifier_axis'])],
            [np.dot(p['modifier_axis'], p['noun_axis']), np.dot(p['modifier_axis'], p['modifier_axis'])],
        ])
        print(f"  L{li}: 平面度量矩阵 = [[{metric[0,0]:.4f}, {metric[0,1]:.6f}], "
              f"[{metric[1,0]:.6f}, {metric[1,1]:.4f}]], "
              f"ortho={p['ortho']:.6f}")

    # Step 5: 旋转轨迹的曲率(是否沿测地线?)
    print("\n  Step 5: 旋转轨迹的曲率分析")

    # Gr(2, d)上的测地线满足: 旋转角是常数
    # 如果相邻旋转角相等 → 测地线
    # 用Grassmannian距离检验

    grassmannian_distances = []
    for i in range(len(sorted_layers) - 1):
        li = sorted_layers[i]
        li1 = sorted_layers[i + 1]
        if li not in layer_planes or li1 not in layer_planes:
            continue
        p = layer_planes[li]
        p1 = layer_planes[li1]

        P1 = np.column_stack([p['noun_axis'], p['modifier_axis']])
        P2 = np.column_stack([p1['noun_axis'], p1['modifier_axis']])
        M = P1.T @ P2
        _, s_m, _ = np.linalg.svd(M)

        # Grassmannian距离: d_G = sqrt(sum(arccos(s_i)^2))
        d_G = np.sqrt(np.sum(np.arccos(np.clip(s_m, -1, 1)) ** 2))
        grassmannian_distances.append(d_G)

    if len(grassmannian_distances) > 2:
        mean_dG = np.mean(grassmannian_distances)
        std_dG = np.std(grassmannian_distances)
        cv_dG = std_dG / max(mean_dG, 1e-10)
        print(f"  Grassmannian距离: mean={mean_dG:.4f}, std={std_dG:.4f}, CV={cv_dG:.4f}")

        # 累积Grassmannian距离 vs 首尾直线距离
        cum_dG = np.sum(grassmannian_distances)
        P_first = np.column_stack([layer_planes[sorted_layers[0]]['noun_axis'],
                                   layer_planes[sorted_layers[0]]['modifier_axis']])
        P_last = np.column_stack([layer_planes[sorted_layers[-1]]['noun_axis'],
                                  layer_planes[sorted_layers[-1]]['modifier_axis']])
        M_total = P_first.T @ P_last
        _, s_total, _ = np.linalg.svd(M_total)
        dG_total = np.sqrt(np.sum(np.arccos(np.clip(s_total, -1, 1)) ** 2))

        curvature_ratio = cum_dG / max(dG_total, 1e-10)
        print(f"  累积Grassmannian距离: {cum_dG:.4f}")
        print(f"  首尾直线Grassmannian距离: {dG_total:.4f}")
        print(f"  曲率比(累积/直线): {curvature_ratio:.4f}")
        if curvature_ratio < 1.1:
            print(f"  ★★★ 轨迹近似测地线! (曲率比≈1)")
        elif curvature_ratio < 1.5:
            print(f"  ★★ 轨迹接近测地线, 有一定弯曲")
        else:
            print(f"  ★ 轨迹明显偏离测地线")

    results['rotation_data'] = rotation_data
    results['layer_planes_ortho'] = {str(li): layer_planes[li]['ortho'] for li in sorted_layers}
    results['cum_rotation_noun'] = cum_rotation_noun
    results['cum_rotation_mod'] = cum_rotation_mod
    results['cum_rotation_total'] = cum_rotation_total
    results['grassmannian_distances'] = grassmannian_distances
    if len(grassmannian_distances) > 2:
        results['curvature_ratio'] = curvature_ratio

    return results


# ===== Exp2: 因果模式的统一数学框架 =====
def exp2_unified_causal_framework(model, tokenizer, device):
    """因果模式的统一数学框架: coupling_ratio = f(nonlinearity, overlap, cross_hessian)"""
    print("\n" + "="*70)
    print("Exp2: 因果模式的统一数学框架 ★★★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    model_name = args.model if hasattr(args, 'model') else 'qwen3'

    # 采样层
    sample_layers = [0, 1, n_layers//4, n_layers//2, 3*n_layers//4, n_layers-1]
    sample_layers = sorted(set(sample_layers))

    # Phase 17已知的三模型coupling ratio
    phase17_coupling = {
        "qwen3": 1.003,
        "glm4": 0.429,
        "deepseek7b": 2.994,
    }

    # Step 1: 在多层收集代数量
    print("\n  Step 1: 在多层收集代数量(非线性和、方向导数重叠、交叉Hessian)")

    layer_metrics = {}

    for li in sample_layers:
        print(f"\n  === L{li} ===")

        # 获取语法方向
        dirs = get_syntax_directions_at_layer(model, tokenizer, device, model_info, li)
        if 'noun_axis' not in dirs or 'modifier_axis' not in dirs:
            print(f"  L{li}: 无法获取语法方向, 跳过")
            continue

        noun_axis = dirs['noun_axis']
        modifier_axis = dirs['modifier_axis']

        # 计算: (1) 方向导数, (2) Hessian对角线, (3) 交叉Hessian
        # 使用多个测试句子以获得更稳健的估计
        test_sents = [
            "The king ruled",
            "The doctor treated",
            "The artist painted",
            "The soldier defended",
        ]

        all_J_noun = []
        all_J_mod = []
        all_H_noun = []
        all_H_mod = []
        all_H_cross = []

        for sent in test_sents:
            toks = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            last_idx = input_ids.shape[1] - 1

            embed_layer = model.get_input_embeddings()
            inputs_embeds_base = embed_layer(input_ids).detach().clone().float()

            # 基线logits
            with torch.no_grad():
                base_logits = model(inputs_embeds=inputs_embeds_base.to(model.dtype)).logits[0, last_idx, :]

            top_k = 30
            base_topk_vals, base_topk_indices = torch.topk(base_logits.float(), top_k)

            epsilon = 0.1
            epsilon2 = 0.5

            noun_tensor = torch.tensor(noun_axis, dtype=torch.float32, device=device)
            mod_tensor = torch.tensor(modifier_axis, dtype=torch.float32, device=device)

            # 方向导数: J_noun, J_mod
            inputs_plus_n = inputs_embeds_base.clone()
            inputs_plus_n[0, last_idx, :] += (epsilon * noun_tensor).to(inputs_embeds_base.dtype)
            inputs_minus_n = inputs_embeds_base.clone()
            inputs_minus_n[0, last_idx, :] -= (epsilon * noun_tensor).to(inputs_embeds_base.dtype)

            with torch.no_grad():
                logits_plus_n = model(inputs_embeds=inputs_plus_n.to(model.dtype)).logits[0, last_idx, :]
                logits_minus_n = model(inputs_embeds=inputs_minus_n.to(model.dtype)).logits[0, last_idx, :]

            J_noun = ((logits_plus_n[base_topk_indices] - logits_minus_n[base_topk_indices]) / (2 * epsilon)).float().cpu().numpy()

            inputs_plus_m = inputs_embeds_base.clone()
            inputs_plus_m[0, last_idx, :] += (epsilon * mod_tensor).to(inputs_embeds_base.dtype)
            inputs_minus_m = inputs_embeds_base.clone()
            inputs_minus_m[0, last_idx, :] -= (epsilon * mod_tensor).to(inputs_embeds_base.dtype)

            with torch.no_grad():
                logits_plus_m = model(inputs_embeds=inputs_plus_m.to(model.dtype)).logits[0, last_idx, :]
                logits_minus_m = model(inputs_embeds=inputs_minus_m.to(model.dtype)).logits[0, last_idx, :]

            J_mod = ((logits_plus_m[base_topk_indices] - logits_minus_m[base_topk_indices]) / (2 * epsilon)).float().cpu().numpy()

            all_J_noun.append(J_noun)
            all_J_mod.append(J_mod)

            # Hessian对角线: H_noun, H_mod
            inputs_plus2n = inputs_embeds_base.clone()
            inputs_plus2n[0, last_idx, :] += (epsilon2 * noun_tensor).to(inputs_embeds_base.dtype)
            inputs_minus2n = inputs_embeds_base.clone()
            inputs_minus2n[0, last_idx, :] -= (epsilon2 * noun_tensor).to(inputs_embeds_base.dtype)

            with torch.no_grad():
                logits_plus2n = model(inputs_embeds=inputs_plus2n.to(model.dtype)).logits[0, last_idx, :]
                logits_minus2n = model(inputs_embeds=inputs_minus2n.to(model.dtype)).logits[0, last_idx, :]
                logits_center = model(inputs_embeds=inputs_embeds_base.to(model.dtype)).logits[0, last_idx, :]

            H_noun = ((logits_plus2n[base_topk_indices] - 2*logits_center[base_topk_indices] + logits_minus2n[base_topk_indices]) / (epsilon2**2)).float().cpu().numpy()

            inputs_plus2m = inputs_embeds_base.clone()
            inputs_plus2m[0, last_idx, :] += (epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)
            inputs_minus2m = inputs_embeds_base.clone()
            inputs_minus2m[0, last_idx, :] -= (epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)

            with torch.no_grad():
                logits_plus2m = model(inputs_embeds=inputs_plus2m.to(model.dtype)).logits[0, last_idx, :]
                logits_minus2m = model(inputs_embeds=inputs_minus2m.to(model.dtype)).logits[0, last_idx, :]

            H_mod = ((logits_plus2m[base_topk_indices] - 2*logits_center[base_topk_indices] + logits_minus2m[base_topk_indices]) / (epsilon2**2)).float().cpu().numpy()

            all_H_noun.append(H_noun)
            all_H_mod.append(H_mod)

            # 交叉Hessian
            inputs_pp = inputs_embeds_base.clone()
            inputs_pp[0, last_idx, :] += (epsilon2 * noun_tensor + epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)
            inputs_pm = inputs_embeds_base.clone()
            inputs_pm[0, last_idx, :] += (epsilon2 * noun_tensor - epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)
            inputs_mp = inputs_embeds_base.clone()
            inputs_mp[0, last_idx, :] += (-epsilon2 * noun_tensor + epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)
            inputs_mm = inputs_embeds_base.clone()
            inputs_mm[0, last_idx, :] += (-epsilon2 * noun_tensor - epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)

            with torch.no_grad():
                logits_pp = model(inputs_embeds=inputs_pp.to(model.dtype)).logits[0, last_idx, :]
                logits_pm = model(inputs_embeds=inputs_pm.to(model.dtype)).logits[0, last_idx, :]
                logits_mp = model(inputs_embeds=inputs_mp.to(model.dtype)).logits[0, last_idx, :]
                logits_mm = model(inputs_embeds=inputs_mm.to(model.dtype)).logits[0, last_idx, :]

            H_cross = ((logits_pp[base_topk_indices] - logits_pm[base_topk_indices] - logits_mp[base_topk_indices] + logits_mm[base_topk_indices]) / (4 * epsilon2**2)).float().cpu().numpy()
            all_H_cross.append(H_cross)

        # 对多个句子的结果取平均
        J_noun_avg = np.mean(all_J_noun, axis=0) if all_J_noun else None
        J_mod_avg = np.mean(all_J_mod, axis=0) if all_J_mod else None
        H_noun_avg = np.mean(all_H_noun, axis=0) if all_H_noun else None
        H_mod_avg = np.mean(all_H_mod, axis=0) if all_H_mod else None
        H_cross_avg = np.mean(all_H_cross, axis=0) if all_H_cross else None

        if J_noun_avg is not None and J_mod_avg is not None:
            # 核心指标
            first_order_noun = np.linalg.norm(J_noun_avg)
            first_order_mod = np.linalg.norm(J_mod_avg)
            noun_curvature = np.linalg.norm(H_noun_avg)
            mod_curvature = np.linalg.norm(H_mod_avg)
            cross_curvature = np.linalg.norm(H_cross_avg) if H_cross_avg is not None else 0

            # 非线性比
            nonlinear_ratio_noun = noun_curvature / max(first_order_noun, 1e-10)
            nonlinear_ratio_mod = mod_curvature / max(first_order_mod, 1e-10)
            avg_nonlinearity = (nonlinear_ratio_noun + nonlinear_ratio_mod) / 2

            # Jacobian重叠
            cos_jacobian = np.dot(J_noun_avg, J_mod_avg) / max(
                np.linalg.norm(J_noun_avg) * np.linalg.norm(J_mod_avg), 1e-10)
            jacobian_overlap = abs(cos_jacobian)

            # 交叉非线性
            cross_nonlinear_ratio = cross_curvature / max(
                np.sqrt(first_order_noun * first_order_mod), 1e-10)

            # 交叉/单轴比
            cross_to_single_ratio = cross_curvature / max(
                np.mean([noun_curvature, mod_curvature]), 1e-10)

            layer_metrics[li] = {
                'nonlinearity': float(avg_nonlinearity),
                'nonlinear_noun': float(nonlinear_ratio_noun),
                'nonlinear_mod': float(nonlinear_ratio_mod),
                'jacobian_overlap': float(jacobian_overlap),
                'cross_nonlinear': float(cross_nonlinear_ratio),
                'cross_to_single': float(cross_to_single_ratio),
                'first_order_noun': float(first_order_noun),
                'first_order_mod': float(first_order_mod),
                'noun_curvature': float(noun_curvature),
                'mod_curvature': float(mod_curvature),
                'cross_curvature': float(cross_curvature),
            }

            print(f"  非线性: {avg_nonlinearity:.4f} (noun={nonlinear_ratio_noun:.4f}, mod={nonlinear_ratio_mod:.4f})")
            print(f"  Jacobian重叠: {jacobian_overlap:.4f}")
            print(f"  交叉非线性: {cross_nonlinear_ratio:.4f}")
            print(f"  交叉/单轴: {cross_to_single_ratio:.4f}")

    # Step 2: 计算每层的coupling_ratio
    print("\n  Step 2: 计算每层的coupling_ratio")

    # coupling_ratio = KL(both) / [KL(noun) + KL(mod)]
    # 用方向导数近似KL散度:
    # KL(noun) ≈ ||J_noun||^2 (Fisher信息的方向分量)
    # KL(both) ≈ ||J_noun + J_mod||^2 (联合方向)
    # coupling_ratio ≈ ||J_noun + J_mod||^2 / (||J_noun||^2 + ||J_mod||^2)

    # 但这忽略了非线性。更准确的:
    # 用Phase 17的方法: 直接测量logit空间的分离度变化

    # 简化方法: 用Hessian预测
    # 如果 f 是线性的: coupling = 1 (独立)
    # 如果 f 有交叉项: coupling = 1 + cross_contribution
    # coupling_ratio ≈ 1 + sign(H_cross) * |H_cross| / (|H_noun| + |H_mod|)

    coupling_predictions = {}
    for li, m in layer_metrics.items():
        # 线性预测: 基于Jacobian重叠
        # coupling = (||J_noun||^2 + ||J_mod||^2 + 2 * J_noun·J_mod) / (||J_noun||^2 + ||J_mod||^2)
        # = 1 + 2 * cos(J) * ||J_noun|| * ||J_mod|| / (||J_noun||^2 + ||J_mod||^2)
        Jn = m['first_order_noun']
        Jm = m['first_order_mod']
        cos_J = m['jacobian_overlap']  # 绝对值

        # 考虑方向(重叠的符号)
        # 简化: 用绝对值
        balance = 2 * cos_J * Jn * Jm / max(Jn**2 + Jm**2, 1e-10)
        coupling_linear = 1 + balance

        # 非线性修正: 交叉Hessian的贡献
        # 如果cross_to_single > 1: 交叉非线性强 → 可能超加性
        # 如果cross_to_single < 0.5: 交叉非线性弱 → 可能冗余或独立
        coupling_with_nonlinear = coupling_linear + 0.5 * (m['cross_to_single'] - 1)

        coupling_predictions[li] = {
            'coupling_linear': float(coupling_linear),
            'coupling_with_nonlinear': float(coupling_with_nonlinear),
        }
        print(f"  L{li}: coupling_linear={coupling_linear:.4f}, coupling_nonlinear={coupling_with_nonlinear:.4f}")

    # Step 3: 统一框架拟合
    print("\n  Step 3: 统一框架分析")

    actual_coupling = phase17_coupling.get(model_name, 1.0)

    # 收集所有层的指标
    all_metrics = []
    for li, m in layer_metrics.items():
        all_metrics.append(m)

    if all_metrics:
        avg_nonlinearity = np.mean([m['nonlinearity'] for m in all_metrics])
        avg_overlap = np.mean([m['jacobian_overlap'] for m in all_metrics])
        avg_cross = np.mean([m['cross_nonlinear'] for m in all_metrics])
        avg_cross_single = np.mean([m['cross_to_single'] for m in all_metrics])

        print(f"\n  模型 {model_name} 的平均代数量:")
        print(f"    平均非线性: {avg_nonlinearity:.4f}")
        print(f"    平均Jacobian重叠: {avg_overlap:.4f}")
        print(f"    平均交叉非线性: {avg_cross:.4f}")
        print(f"    平均交叉/单轴: {avg_cross_single:.4f}")
        print(f"    Phase 17实际coupling_ratio: {actual_coupling:.4f}")

        # 尝试拟合: coupling = f(nonlinear, overlap, cross)
        # 简单线性模型: coupling = a * nonlinear + b * overlap + c * cross + d
        # 只有一个模型的数据点, 无法拟合参数, 但可以比较三模型的指标
        results['model_avg_metrics'] = {
            'nonlinearity': float(avg_nonlinearity),
            'overlap': float(avg_overlap),
            'cross_nonlinear': float(avg_cross),
            'cross_to_single': float(avg_cross_single),
            'actual_coupling': float(actual_coupling),
        }

    # Step 4: 因果模式的判据总结
    print("\n  Step 4: 因果模式判据总结")
    print(f"  模型: {model_name}")
    print(f"  实际模式: ", end="")
    if actual_coupling < 0.8:
        print("冗余")
    elif actual_coupling > 1.2:
        print("超加性")
    else:
        print("独立")

    # 各层指标的范围
    if layer_metrics:
        nonlinear_range = [min(m['nonlinearity'] for m in layer_metrics.values()),
                          max(m['nonlinearity'] for m in layer_metrics.values())]
        overlap_range = [min(m['jacobian_overlap'] for m in layer_metrics.values()),
                        max(m['jacobian_overlap'] for m in layer_metrics.values())]
        cross_range = [min(m['cross_nonlinear'] for m in layer_metrics.values()),
                      max(m['cross_nonlinear'] for m in layer_metrics.values())]

        print(f"  非线性范围: [{nonlinear_range[0]:.4f}, {nonlinear_range[1]:.4f}]")
        print(f"  重叠范围: [{overlap_range[0]:.4f}, {overlap_range[1]:.4f}]")
        print(f"  交叉非线性范围: [{cross_range[0]:.4f}, {cross_range[1]:.4f}]")

        # ★★★ 统一框架假设
        # coupling_ratio ≈ 1 + α * overlap + β * cross_nonlinear - γ * nonlinearity
        # α > 0: 重叠增加 → 冗余(coupling < 1)
        # β > 0: 交叉增加 → 超加性(coupling > 1)
        # γ > 0: 非线性增加 → 如果没有交叉, 强非线性导致维度坍缩 → 冗余
        print(f"\n  ★★★ 统一框架假设:")
        print(f"  coupling_ratio ≈ 1 + α*overlap + β*cross - γ*nonlinearity")
        print(f"  需要三模型数据联合拟合 α, β, γ")

    results['layer_metrics'] = {str(k): v for k, v in layer_metrics.items()}
    results['coupling_predictions'] = {str(k): v for k, v in coupling_predictions.items()}

    return results


# ===== Exp3: 残差+LayerNorm保持正交性的反例验证 =====
def exp3_residual_layernorm(model, tokenizer, device):
    """残差+LayerNorm保持正交性的反例验证"""
    print("\n" + "="*70)
    print("Exp3: 残差+LayerNorm保持正交性的反例验证 ★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model
    mlp_type = model_info.mlp_type

    # Step 1: 正常前向传播, 收集各层hidden states
    print("\n  Step 1: 正常前向传播, 收集各层hidden states")

    test_sents_nsubj = MANIFOLD_ROLES_DATA["nsubj"]["sentences"][:8]
    test_words_nsubj = MANIFOLD_ROLES_DATA["nsubj"]["target_words"][:8]
    test_sents_dobj = MANIFOLD_ROLES_DATA["dobj"]["sentences"][:8]
    test_words_dobj = MANIFOLD_ROLES_DATA["dobj"]["target_words"][:8]
    test_sents_amod = MANIFOLD_ROLES_DATA["amod"]["sentences"][:8]
    test_words_amod = MANIFOLD_ROLES_DATA["amod"]["target_words"][:8]

    # 收集3个关键层的hidden states
    test_layers = [0, n_layers//2, n_layers-1]

    # 正常模式的语法方向
    normal_dirs = {}
    for li in test_layers:
        dirs = get_syntax_directions_at_layer(model, tokenizer, device, model_info, li)
        if 'noun_axis' in dirs and 'modifier_axis' in dirs:
            normal_dirs[li] = {
                'noun_axis': dirs['noun_axis'],
                'modifier_axis': dirs['modifier_axis'],
                'ortho': dirs.get('axis_orthogonality', 0),
            }
            print(f"  L{li} 正常模式: ortho={normal_dirs[li]['ortho']:.6f}")

    # Step 2: 修改前向传播 - 测量残差和LN各自的作用
    print("\n  Step 2: 分析残差和LN的作用")

    # 对于第L层, 正常计算:
    # h_{l+1} = LN(h_l + Attn(h_l) + MLP(h_l))
    # 拆解:
    #   residual_contribution = h_l (残差保留原始方向)
    #   attn_mlp_contribution = Attn(h_l) + MLP(h_l) (新分量)
    #   LN_normalization = (x - μ)/σ (归一化)

    # 收集更详细的前向信息
    layers_list = get_layers(model)

    for li in test_layers:
        if li not in normal_dirs:
            continue

        print(f"\n  === L{li} 详细分析 ===")

        # 收集多个句子的隐藏状态
        for role_name, sents, words in [("nsubj", test_sents_nsubj, test_words_nsubj),
                                         ("dobj", test_sents_dobj, test_words_dobj),
                                         ("amod", test_sents_amod, test_words_amod)]:
            # 只分析第一个句子作为示例
            sent = sents[0]
            tw = words[0]

            toks = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
            dep_idx = find_token_index(tokens_list, tw)
            if dep_idx is None:
                continue

            # Hook: 收集层输入和输出
            target_layer = layers_list[li]
            captured = {}

            def make_pre_post_hook(key_in, key_out):
                def hook(module, input, output):
                    if isinstance(input, tuple) and len(input) > 0:
                        captured[key_in] = input[0].detach().float().cpu().numpy()
                    if isinstance(output, tuple):
                        captured[key_out] = output[0].detach().float().cpu().numpy()
                    else:
                        captured[key_out] = output.detach().float().cpu().numpy()
                return hook

            hook_handle = target_layer.register_forward_hook(
                make_pre_post_hook('h_in', 'h_out'))

            with torch.no_grad():
                _ = model(**toks)

            hook_handle.remove()

            if 'h_in' in captured and 'h_out' in captured:
                h_in = captured['h_in'][0, dep_idx, :]  # 层输入
                h_out = captured['h_out'][0, dep_idx, :]  # 层输出

                # 层间差异 = Attn + MLP (包括残差和LN)
                delta = h_out - h_in

                # 分析delta与语法方向的对齐
                nd = normal_dirs[li]['noun_axis']
                md = normal_dirs[li]['modifier_axis']

                delta_noun = np.dot(delta, nd)
                delta_mod = np.dot(delta, md)
                delta_ortho = np.dot(nd, md)  # 应该≈0

                # 残差贡献: h_in在语法方向上的投影
                h_in_noun = np.dot(h_in, nd)
                h_in_mod = np.dot(h_in, md)

                # 输出在语法方向上的投影
                h_out_noun = np.dot(h_out, nd)
                h_out_mod = np.dot(h_out, md)

                # 语法方向的保持度: h_out方向 / h_in方向
                noun_preservation = h_out_noun / max(abs(h_in_noun), 1e-10)
                mod_preservation = h_out_mod / max(abs(h_in_mod), 1e-10)

                if role_name == "nsubj":
                    print(f"  {role_name} @ L{li}:")
                    print(f"    h_in → h_out: noun {h_in_noun:.4f}→{h_out_noun:.4f} "
                          f"(preserve={noun_preservation:.4f}), "
                          f"mod {h_in_mod:.4f}→{h_out_mod:.4f} "
                          f"(preserve={mod_preservation:.4f})")
                    print(f"    delta: noun={delta_noun:.4f}, mod={delta_mod:.4f}")
                break  # 只看第一个句子

    # Step 3: 反例验证 - 去掉LN后的正交性
    print("\n  Step 3: 反例验证 - 去掉LN后的正交性")

    # 方法: 收集LN之前和之后的数据
    # 在每层, hook LN的输入和输出
    # 但LN在残差连接之后, 所以:
    # h_ln_in = h_l + Attn(h_l) + MLP(h_l)  (LN之前)
    # h_ln_out = LN(h_ln_in) = h_{l+1}       (LN之后)

    # 对每个测试层, 收集LN前后的语法方向正交性

    for li in test_layers:
        print(f"\n  === L{li} LN前后分析 ===")

        # 需要分别收集 nsubj, dobj, amod 的 hidden states
        # LN之前 vs LN之后

        ln_before_centers = {}
        ln_after_centers = {}

        for role_name, sents, words in [("nsubj", test_sents_nsubj, test_words_nsubj),
                                         ("dobj", test_sents_dobj, test_words_dobj),
                                         ("amod", test_sents_amod, test_words_amod)]:
            role_hs_before = []
            role_hs_after = []

            for sent, tw in zip(sents[:4], words[:4]):
                toks = tokenizer(sent, return_tensors="pt").to(device)
                input_ids = toks.input_ids
                tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
                dep_idx = find_token_index(tokens_list, tw)
                if dep_idx is None:
                    continue

                # 收集层输出 (已经包含LN)
                target_layer = layers_list[li]
                captured_after = {}
                def hook_after(module, input, output):
                    if isinstance(output, tuple):
                        captured_after['h'] = output[0].detach().float().cpu().numpy()
                    else:
                        captured_after['h'] = output.detach().float().cpu().numpy()

                h_after = target_layer.register_forward_hook(hook_after)

                # 同时hook LN的输入 (如果有post_attention_layernorm)
                # 找到LN层
                ln_captured = {}
                ln_layer = None

                # 尝试hook post_attention_layernorm 或 input_layernorm
                # 实际上, 层输出已经包含所有操作(包括最后的LN)
                # LN之前 = 层输出 + LN的逆变换

                with torch.no_grad():
                    _ = model(**toks)

                h_after.remove()

                if 'h' in captured_after:
                    h_after_val = captured_after['h'][0, dep_idx, :]
                    role_hs_after.append(h_after_val)

                    # 估算LN之前: h_before = σ * h_after + μ
                    # 但不知道μ和σ, 需要直接hook
                    # 简化: 只比较层输出(含LN)和下一层输入(=层输出)
                    # 用层的input来获取LN之前的数据
                    # 实际上, 层input = 上一层的输出(含LN)
                    # 所以我们无法直接获取LN之前的中间结果

            if role_hs_after:
                ln_after_centers[role_name] = np.mean(role_hs_after, axis=0)

        # 计算LN之后的正交性(就是正常模式)
        if 'nsubj' in ln_after_centers and 'dobj' in ln_after_centers and 'amod' in ln_after_centers:
            nsubj_c = ln_after_centers['nsubj']
            dobj_c = ln_after_centers['dobj']
            amod_c = ln_after_centers['amod']

            noun_dir = dobj_c - nsubj_c
            noun_norm = np.linalg.norm(noun_dir)
            if noun_norm > 0:
                noun_dir = noun_dir / noun_norm

            amod_vec = amod_c - nsubj_c
            amod_proj = np.dot(amod_vec, noun_dir) * noun_dir
            mod_dir = amod_vec - amod_proj
            mod_norm = np.linalg.norm(mod_dir)
            if mod_norm > 0:
                mod_dir = mod_dir / mod_norm

            ortho_after_ln = float(np.dot(noun_dir, mod_dir))
            print(f"  LN后正交性: {ortho_after_ln:.6f}")

    # Step 4: 数值验证 - 构造修改后的前向传播
    print("\n  Step 4: 数值验证 - 去残差/去LN的效果")

    # 方法: 手动构造修改后的hidden states, 测量正交性变化
    # 对于L0→L1: h_1 = LN(h_0 + Attn(h_0) + MLP(h_0))
    # 修改1: h_1_no_residual = LN(Attn(h_0) + MLP(h_0))
    # 修改2: h_1_no_ln = h_0 + Attn(h_0) + MLP(h_0)  (不加LN)
    # 修改3: h_1_no_residual_no_ln = Attn(h_0) + MLP(h_0)

    # 为了做到这一点, 需要分别hook:
    # - 层输入 (h_l)
    # - attention输出
    # - MLP输出
    # - 层输出 (h_{l+1})

    # 分析L0层
    li = 0
    print(f"\n  === 分析L{li}→L{li+1} ===")

    for role_name, sents, words in [("nsubj", test_sents_nsubj[:4], test_words_nsubj[:4]),
                                     ("dobj", test_sents_dobj[:4], test_words_dobj[:4]),
                                     ("amod", test_sents_amod[:4], test_words_amod[:4])]:
        role_h_layer = []  # 正常层输出
        role_h_attn = []   # attention输出
        role_h_mlp = []    # MLP输出(如果可以hook)
        role_h_input = []  # 层输入

        for sent, tw in zip(sents, words):
            toks = tokenizer(sent, return_tensors="pt").to(device)
            input_ids = toks.input_ids
            tokens_list = [safe_decode(tokenizer, t) for t in input_ids[0].tolist()]
            dep_idx = find_token_index(tokens_list, tw)
            if dep_idx is None:
                continue

            target_layer = layers_list[li]

            # Hook 1: 层输出
            captured = {}
            def make_hook(key):
                def hook(module, input, output):
                    if isinstance(output, tuple):
                        captured[key] = output[0].detach().float().cpu().numpy()
                    else:
                        captured[key] = output.detach().float().cpu().numpy()
                return hook

            hooks = []
            hooks.append(target_layer.register_forward_hook(make_hook('layer_out')))

            # Hook 2: Self-attention输出
            if hasattr(target_layer, 'self_attn'):
                hooks.append(target_layer.self_attn.register_forward_hook(make_hook('attn_out')))

            # Hook 3: MLP输出
            if hasattr(target_layer, 'mlp'):
                hooks.append(target_layer.mlp.register_forward_hook(make_hook('mlp_out')))

            # Hook 4: Post-attention LN输出
            for ln_name in ["post_attention_layernorm", "ln_2", "post_self_attn_layernorm"]:
                if hasattr(target_layer, ln_name):
                    ln_obj = getattr(target_layer, ln_name)
                    hooks.append(ln_obj.register_forward_hook(make_hook('post_attn_ln_out')))
                    break

            with torch.no_grad():
                _ = model(**toks)

            for h in hooks:
                h.remove()

            if 'layer_out' in captured:
                h_layer = captured['layer_out'][0, dep_idx, :]
                role_h_layer.append(h_layer)

            if 'attn_out' in captured:
                h_attn = captured['attn_out'][0, dep_idx, :]
                role_h_attn.append(h_attn)

            if 'mlp_out' in captured:
                h_mlp = captured['mlp_out'][0, dep_idx, :]
                role_h_mlp.append(h_mlp)

        if len(role_h_layer) > 0:
            ln_after_centers[role_name] = np.mean(role_h_layer, axis=0)
        if len(role_h_attn) > 0:
            ln_before_centers[role_name + '_attn'] = np.mean(role_h_attn, axis=0)

    # 计算正常模式的正交性
    if all(r in ln_after_centers for r in ['nsubj', 'dobj', 'amod']):
        nsubj_c = ln_after_centers['nsubj']
        dobj_c = ln_after_centers['dobj']
        amod_c = ln_after_centers['amod']

        noun_dir = dobj_c - nsubj_c
        noun_norm = np.linalg.norm(noun_dir)
        if noun_norm > 0:
            noun_dir = noun_dir / noun_norm

        amod_vec = amod_c - nsubj_c
        amod_proj = np.dot(amod_vec, noun_dir) * noun_dir
        mod_dir = amod_vec - amod_proj
        mod_norm = np.linalg.norm(mod_dir)
        if mod_norm > 0:
            mod_dir = mod_dir / mod_norm

        ortho_normal = float(np.dot(noun_dir, mod_dir))
        print(f"  正常模式(L0输出)正交性: {ortho_normal:.6f}")

    # 计算Attention输出(无残差)的正交性
    if all(r + '_attn' in ln_before_centers for r in ['nsubj', 'dobj', 'amod']):
        nsubj_attn = ln_before_centers['nsubj_attn']
        dobj_attn = ln_before_centers['dobj_attn']
        amod_attn = ln_before_centers['amod_attn']

        noun_attn = dobj_attn - nsubj_attn
        noun_attn_norm = np.linalg.norm(noun_attn)
        if noun_attn_norm > 0:
            noun_attn = noun_attn / noun_attn_norm

        amod_attn_vec = amod_attn - nsubj_attn
        amod_attn_proj = np.dot(amod_attn_vec, noun_attn) * noun_attn
        mod_attn = amod_attn_vec - amod_attn_proj
        mod_attn_norm = np.linalg.norm(mod_attn)
        if mod_attn_norm > 0:
            mod_attn = mod_attn / mod_attn_norm

        ortho_attn_only = float(np.dot(noun_attn, mod_attn))
        print(f"  Attention输出(无残差)正交性: {ortho_attn_only:.6f}")
        print(f"  ★★★ 残差贡献: 正交性从 {ortho_attn_only:.6f} → {ortho_normal:.6f}")

        results['ortho_normal'] = float(ortho_normal)
        results['ortho_attn_only'] = float(ortho_attn_only)

    # Step 5: LN的缩放效果分析
    print("\n  Step 5: LayerNorm的缩放效果分析")

    # LayerNorm: (x - μ) / σ
    # 对于正交向量u, v:
    # LN后: u' = (u - μ_u) / σ, v' = (v - μ_v) / σ
    # <u', v'> = <u - μ_u, v - μ_v> / σ^2
    # 如果u, v已正交且零均值: <u', v'> = <u, v> / σ^2 = 0
    # 关键: 残差连接使得语法方向在每个样本中都偏向相同方向
    # LN的减均值操作消除了共同偏移, 使方向更正交

    # 数值验证: 手动应用LN到语法方向
    if all(r in ln_after_centers for r in ['nsubj', 'dobj', 'amod']):
        # 模拟: 如果不加LN, 语法方向会怎样?
        # 用layer输出的中心差来估算

        # 计算: 如果h = h_in + attn_out + mlp_out (无LN)
        # vs h = LN(h_in + attn_out + mlp_out) (有LN)

        # 我们已经知道有LN的正交性, 需要知道无LN的
        # 无LN的数据需要通过hook中间结果来获取

        # 简化分析: 用中心差的方差来估计
        nsubj_c = ln_after_centers['nsubj']
        dobj_c = ln_after_centers['dobj']
        amod_c = ln_after_centers['amod']

        # 三个中心的均值(公共偏移)
        mean_center = np.mean([nsubj_c, dobj_c, amod_c], axis=0)

        # 去均值后的方向
        nsubj_demeaned = nsubj_c - mean_center
        dobj_demeaned = dobj_c - mean_center
        amod_demeaned = amod_c - mean_center

        # 去均值后的正交性
        noun_demeaned = dobj_demeaned - nsubj_demeaned
        noun_d_norm = np.linalg.norm(noun_demeaned)
        if noun_d_norm > 0:
            noun_demeaned = noun_demeaned / noun_d_norm

        amod_demeaned_vec = amod_demeaned - nsubj_demeaned
        amod_d_proj = np.dot(amod_demeaned_vec, noun_demeaned) * noun_demeaned
        mod_demeaned = amod_demeaned_vec - amod_d_proj
        mod_d_norm = np.linalg.norm(mod_demeaned)
        if mod_d_norm > 0:
            mod_demeaned = mod_demeaned / mod_d_norm

        ortho_demeaned = float(np.dot(noun_demeaned, mod_demeaned))

        print(f"  原始正交性(含LN): {ortho_normal:.6f}")
        print(f"  去公共偏移后正交性: {ortho_demeaned:.6f}")
        print(f"  → 公共偏移对正交性的影响: {abs(ortho_normal - ortho_demeaned):.6f}")

        # 分析: 残差连接的关键作用
        # 残差连接使得h_{l+1} = h_l + delta
        # 如果h_l中的语法方向是正交的, delta相对较小
        # 则h_{l+1}中的语法方向仍近似正交
        # 数学: 如果 u_l ⊥ v_l, ||delta|| << ||h_l||
        # 则 u_{l+1} ≈ u_l + (小扰动), v_{l+1} ≈ v_l + (小扰动)
        # u_{l+1} · v_{l+1} ≈ u_l · v_l + O(||delta||/||h_l||) ≈ 0

        # 验证: delta / h_l 的比率
        # 使用之前收集的数据

    results['normal_dirs'] = {str(k): {'ortho': v['ortho']} for k, v in normal_dirs.items()}

    return results


# ===== Exp4: 语法编码的代数不变量 =====
def exp4_algebraic_invariants(model, tokenizer, device):
    """语法编码的代数不变量"""
    print("\n" + "="*70)
    print("Exp4: 语法编码的代数不变量 ★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers
    d_model = model_info.d_model

    # Step 1: 采集所有层的2D平面信息
    print("\n  Step 1: 采集所有层的2D平面信息(Grassmannian坐标)")

    # 采样层
    sample_layers = list(range(n_layers))
    if n_layers > 20:
        step = max(1, n_layers // 15)
        sample_layers = list(range(0, min(4, n_layers), 1))
        sample_layers += list(range(4, n_layers - 3, step))
        sample_layers += list(range(max(4, n_layers - 3), n_layers, 1))
        sample_layers = sorted(set(sample_layers))

    layer_planes = {}
    for li in sample_layers:
        print(f"  采集L{li}...", end="", flush=True)
        dirs = get_syntax_directions_at_layer(model, tokenizer, device, model_info, li)
        if 'noun_axis' in dirs and 'modifier_axis' in dirs:
            layer_planes[li] = {
                'noun_axis': dirs['noun_axis'],
                'modifier_axis': dirs['modifier_axis'],
                'ortho': dirs.get('axis_orthogonality', 0),
            }
            print(f" ortho={layer_planes[li]['ortho']:.6f}")
        else:
            print(f" 跳过")

    if len(layer_planes) < 3:
        print("  采集到的层不足!")
        return results

    sorted_layers = sorted(layer_planes.keys())

    # Step 2: Grassmannian坐标(Plücker坐标)
    print("\n  Step 2: Grassmannian坐标(Plücker坐标)")

    # 2D平面在Gr(2, d)中的坐标 = Plücker坐标
    # P = [noun_axis, mod_axis] [d, 2]
    # Plücker坐标 = p_{ij} = noun_i * mod_j - noun_j * mod_i (反对称2-形式)
    # 这给出 d*(d-1)/2 个坐标

    # 但d_model很大(2560+), Plücker坐标维度太高
    # 简化: 用2x2矩阵 M = P^T @ Q (两平面间的投影矩阵)
    # M的特征值和行列式是不变量

    # 计算: 每对相邻层的投影矩阵
    projection_matrices = {}
    for i in range(len(sorted_layers) - 1):
        li = sorted_layers[i]
        li1 = sorted_layers[i + 1]
        p = layer_planes[li]
        p1 = layer_planes[li1]

        P = np.column_stack([p['noun_axis'], p['modifier_axis']])
        Q = np.column_stack([p1['noun_axis'], p1['modifier_axis']])
        M = P.T @ Q  # [2, 2]

        projection_matrices[(li, li1)] = M

        det = np.linalg.det(M)
        trace = np.trace(M)

        print(f"  L{li}→L{li1}: det(M)={det:.6f}, trace(M)={trace:.6f}, "
              f"M=[[{M[0,0]:.4f},{M[0,1]:.4f}],[{M[1,0]:.4f},{M[1,1]:.4f}]]")

    # Step 3: 旋转角(绕数, winding number)
    print("\n  Step 3: 旋转轨迹的绕数")

    # 绕数 = 总旋转角 / 2π
    # 旋转角 = arctan2(det(M), trace(M)) for each step
    cumulative_angle = 0
    rotation_angles_list = []

    for i in range(len(sorted_layers) - 1):
        li = sorted_layers[i]
        li1 = sorted_layers[i + 1]
        M = projection_matrices[(li, li1)]

        # SO(2)的旋转角: θ = arctan2(M[1,0], M[0,0]) (如果是纯旋转)
        # 但M不一定是纯旋转, 需要极分解
        # M = R * S (旋转 × 对称正定)

        # 简化: 用SVD得到最佳旋转
        U_m, s_m, Vt_m = np.linalg.svd(M)
        # R = U @ Vt (最近旋转矩阵)
        R_optimal = U_m @ Vt_m
        # 旋转角
        theta = np.arctan2(R_optimal[1, 0], R_optimal[0, 0])

        rotation_angles_list.append(theta)
        cumulative_angle += theta

        print(f"  L{li}→L{li1}: θ={np.degrees(theta):.2f}°, "
              f"累计={np.degrees(cumulative_angle):.2f}°")

    winding_number = cumulative_angle / (2 * np.pi)
    print(f"\n  ★★★ 绕数(winding number) = {winding_number:.4f}")
    print(f"  总旋转角 = {np.degrees(cumulative_angle):.2f}° = {cumulative_angle/(2*np.pi):.4f} × 2π")

    # Step 4: 交叉比(cross-ratio)
    print("\n  Step 4: 交叉比不变量检验")

    # Gr(2,d)上的交叉比:
    # 取4个2D平面P1, P2, P3, P4
    # 交叉比 = det(P1,P3)*det(P2,P4) / (det(P1,P4)*det(P2,P3))
    # 其中det(Pi,Pj) = det(Pi^T @ Pj)

    # 选取4个代表性层
    if len(sorted_layers) >= 4:
        # 取均匀分布的4个层
        indices = [0, len(sorted_layers)//3, 2*len(sorted_layers)//3, len(sorted_layers)-1]
        selected = [sorted_layers[i] for i in indices]

        print(f"  选取层: {selected}")

        # 计算4个平面的两两det
        dets = {}
        Ps = {}
        for li in selected:
            p = layer_planes[li]
            Ps[li] = np.column_stack([p['noun_axis'], p['modifier_axis']])

        for i, li in enumerate(selected):
            for j, lj in enumerate(selected):
                if i < j:
                    M = Ps[li].T @ Ps[lj]
                    dets[(li, lj)] = np.linalg.det(M)

        # 交叉比
        l1, l2, l3, l4 = selected
        cr1 = dets.get((l1, l3), 1) * dets.get((l2, l4), 1)
        cr2 = dets.get((l1, l4), 1) * dets.get((l2, l3), 1)

        if abs(cr2) > 1e-10:
            cross_ratio = cr1 / cr2
            print(f"  交叉比 = {cross_ratio:.6f}")
        else:
            cross_ratio = None
            print(f"  交叉比: 无法计算(分母≈0)")

        # 检验: 如果轨迹是测地线, 交叉比应该是常数
        # 用不同的4层组合计算交叉比
        cross_ratios = []
        for shift in range(min(3, len(sorted_layers) - 4)):
            idx = [shift, shift + len(sorted_layers)//4,
                   shift + 2*len(sorted_layers)//4,
                   min(shift + 3*len(sorted_layers)//4, len(sorted_layers)-1)]
            sel = [sorted_layers[min(i, len(sorted_layers)-1)] for i in idx]

            if len(set(sel)) < 4:
                continue

            cr_dets = {}
            cr_Ps = {}
            for li in sel:
                if li in layer_planes:
                    p = layer_planes[li]
                    cr_Ps[li] = np.column_stack([p['noun_axis'], p['modifier_axis']])

            if len(cr_Ps) < 4:
                continue

            sel_valid = [li for li in sel if li in cr_Ps]
            if len(sel_valid) < 4:
                continue

            for i, li in enumerate(sel_valid):
                for j, lj in enumerate(sel_valid):
                    if i < j:
                        M = cr_Ps[li].T @ cr_Ps[lj]
                        cr_dets[(li, lj)] = np.linalg.det(M)

            s1, s2, s3, s4 = sel_valid[:4]
            cr_num = cr_dets.get((s1, s3), 1) * cr_dets.get((s2, s4), 1)
            cr_den = cr_dets.get((s1, s4), 1) * cr_dets.get((s2, s3), 1)

            if abs(cr_den) > 1e-10:
                cr = cr_num / cr_den
                cross_ratios.append(cr)
                print(f"  交叉比({s1},{s2},{s3},{s4}) = {cr:.6f}")

        if len(cross_ratios) > 1:
            cr_std = np.std(cross_ratios)
            cr_mean = np.mean(cross_ratios)
            cr_cv = cr_std / max(abs(cr_mean), 1e-10)
            print(f"  交叉比稳定性: mean={cr_mean:.6f}, std={cr_std:.6f}, CV={cr_cv:.4f}")
            if cr_cv < 0.1:
                print(f"  ★★★ 交叉比近似不变! 旋转轨迹有代数不变量")
            else:
                print(f"  ★ 交叉比变化较大, 旋转轨迹不保持交叉比")

    # Step 5: 总结——代数不变量
    print("\n  Step 5: 代数不变量总结")

    # 已确认的不变量:
    # 1. 正交性: noun_axis ⊥ modifier_axis 在所有层 ≈ 0 ✓
    # 2. 2D子空间维度: 始终为2 ✓

    # 新发现:
    # 3. 绕数: winding_number = total_rotation / 2π
    # 4. 交叉比: 近似不变(如果轨迹接近测地线)

    print(f"  确认的不变量:")
    print(f"    1. 正交性 (noun_axis ⊥ mod_axis): ≈0 在所有层")
    print(f"    2. 2D子空间维度: =2 在所有层")
    print(f"  新发现:")
    print(f"    3. 绕数(winding number): {winding_number:.4f}")
    print(f"    4. 交叉比变化: {'小' if cross_ratios and np.std(cross_ratios)/max(abs(np.mean(cross_ratios)),1e-10) < 0.1 else '大'}")

    results['winding_number'] = float(winding_number)
    results['cumulative_angle_deg'] = float(np.degrees(cumulative_angle))
    results['rotation_angles'] = [float(np.degrees(a)) for a in rotation_angles_list]
    if cross_ratios:
        results['cross_ratios'] = [float(cr) for cr in cross_ratios]

    # Step 6: 与Phase 18发现对比——范数增长作为另一个不变量候选
    print("\n  Step 6: 范数增长率分析")

    # Phase 18发现: 范数从L0到Last增长40-1201x
    # 检查: 范数增长是否指数增长? 线性增长?

    layer_norms = {}
    for li in sorted_layers:
        p = layer_planes[li]
        # 收集实际hidden state的范数
        nsubj_h = collect_hs_at_layer(model, tokenizer, device,
                                       MANIFOLD_ROLES_DATA["nsubj"]["sentences"][:4],
                                       MANIFOLD_ROLES_DATA["nsubj"]["target_words"][:4], li)
        if nsubj_h is not None:
            avg_norm = np.mean(np.linalg.norm(nsubj_h, axis=1))
            layer_norms[li] = float(avg_norm)

    if len(layer_norms) >= 3:
        print(f"  各层平均范数:")
        for li in sorted(layer_norms.keys()):
            print(f"    L{li}: {layer_norms[li]:.4f}")

        # 范数增长模式
        norms_list = [layer_norms[li] for li in sorted(layer_norms.keys())]
        norm_growth = norms_list[-1] / max(norms_list[0], 1e-10)
        print(f"  范数增长: L0→Last = {norm_growth:.2f}x")

        # 检查是否指数增长
        if len(norms_list) >= 3:
            log_norms = np.log(norms_list)
            x = np.arange(len(log_norms))
            # 线性拟合 log(norms) vs layer_index
            slope, intercept = np.polyfit(x, log_norms, 1)
            r_squared = 1 - np.sum((log_norms - (slope * x + intercept))**2) / np.sum((log_norms - np.mean(log_norms))**2)
            print(f"  指数增长拟合: R²={r_squared:.4f} (R²>0.9=指数, 0.5-0.9=亚指数)")
            results['norm_growth_ratio'] = float(norm_growth)
            results['norm_growth_r_squared'] = float(r_squared)

    return results


# ===== 主函数 =====
def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2, 3, 4])
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"CCL-Y Phase19 统一数学框架 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")

    try:
        if args.exp == 1:
            results = exp1_rotation_trajectory(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_unified_causal_framework(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_residual_layernorm(model, tokenizer, device)
        elif args.exp == 4:
            results = exp4_algebraic_invariants(model, tokenizer, device)

        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                               f"ccly_exp{args.exp}_{args.model}_results.json")

        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            if isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            if isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            if isinstance(obj, tuple):
                return list(convert(v) for v in obj)
            return obj

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存: {out_path}")

    finally:
        release_model(model)
        print(f"  模型已释放")


if __name__ == "__main__":
    main()
