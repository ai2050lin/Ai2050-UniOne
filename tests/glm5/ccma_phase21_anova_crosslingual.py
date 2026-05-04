"""
CCMA(Phase 21): 方差分解与跨语言验证——从"依赖"到"谁主导"
=============================================================================
核心问题(基于Phase 20用户批评的修正):

  硬伤1修正: 曲率比∝sqrt(N)只是统计标度律, ≠微观随机
    → 旋转轨迹表现为扩散型统计结构, 而非简单测地线
    → 确定性混沌/约束流形系统也可产生扩散标度

  硬伤2修正: LN没有"失效", 变化的是有效归一化增益
    → LN始终执行公式, 是有效增益在变化

  硬伤3修正: 只对nsubj/dobj/amod验证了正交性
    → 降级为"核心语法因子呈近正交组织"
    → 不是普遍语言规律

  硬伤4修正: dG_total≈2.1只是候选普适量
    → 3个decoder-only模型太同质

  因果判据: 需要方差分解(ANOVA)来区分Model vs Role vs 交互效应

实验:
  Exp1: ★★★★★★★★★ 方差分解(ANOVA)
    → 用Phase 20 Exp4的数据: 3模型 × 3角色组合 × 3指标
    → Y_ijk = μ + α_i(Model) + β_j(Role) + (αβ)_ij + ε_ijk
    → 确定各指标的方差来源

  Exp2: ★★★★★★★★ 跨语言正交性验证
    → 中文: 主语/宾语/修饰语(nsubj/dobj/amod)的语法角色夹角
    → 日语: が/を/形容動詞 的语法角色夹角
    → 核心语法因子是否跨语言呈近正交组织?

  Exp3: ★★★★★★★ 扩散型标度律的微观结构分析
    → 曲率比∝sqrt(N)不等于随机, 分析微观动力学
    → 逐层旋转角的自相关: 是否有长程关联?
    → 旋转角的方向记忆: 相邻层旋转方向是否相关?
    → 如果有记忆 → 约束流形上的确定性运动(不是纯随机)
    → 如果无记忆 → 接近纯随机游走

  Exp4: ★★★★★ 相对范数与probing准确率
    → 用线性probe测量每层的语法角色分类准确率
    → 与相对范数曲线的相关性
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
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode


# ===== 英文语法角色数据 =====
EN_ROLES_DATA = {
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
}

# ===== 中文语法角色数据 =====
ZH_ROLES_DATA = {
    "nsubj": {
        "sentences": [
            "国王明智地统治了王国",
            "医生仔细地治疗了病人",
            "艺术家精美地画了肖像",
            "士兵勇敢地保卫了城堡",
            "老师清楚地讲解了课程",
            "厨师完美地烹饪了餐点",
            "猫快速地追逐了老鼠",
            "狗高兴地找到了骨头",
            "女士安全地驾驶了汽车",
            "男士仔细地修理了屋顶",
            "学生安静地读了书",
            "歌手出色地演唱了歌曲",
            "面包师每天烤了面包",
            "飞行员平稳地驾驶了飞机",
            "农民勤奋地种植了庄稼",
            "作家慢慢地写了小说",
        ],
        "target_words": [
            "国王", "医生", "艺术家", "士兵", "老师",
            "厨师", "猫", "狗", "女士", "男士",
            "学生", "歌手", "面包师", "飞行员", "农民", "作家",
        ],
    },
    "dobj": {
        "sentences": [
            "他们昨天加冕了国王",
            "她最近拜访了医生",
            "他非常钦佩艺术家",
            "我们今天表彰了士兵",
            "你热情地感谢了老师",
            "顾客慷慨地给了厨师小费",
            "鹰迅速地追逐了猫",
            "男孩在外面找到了狗",
            "警察快速地逮捕了女士",
            "公司最近雇佣了男士",
            "我大声地表扬了学生",
            "他们热情地鼓掌了歌手",
            "她经常拜访面包师",
            "他非常钦佩飞行员",
            "我们真诚地感谢了农民",
            "编辑高度赞扬了作家",
        ],
        "target_words": [
            "国王", "医生", "艺术家", "士兵", "老师",
            "厨师", "猫", "狗", "女士", "男士",
            "学生", "歌手", "面包师", "飞行员", "农民", "作家",
        ],
    },
    "amod": {
        "sentences": [
            "勇敢的国王奋力战斗",
            "善良的医生帮助了很多人",
            "有创意的艺术家工作得很好",
            "强壮的士兵行军很远",
            "明智的老师讲解得很清楚",
            "熟练的厨师烹饪得很完美",
            "敏捷的猫跑得很快",
            "忠诚的狗待得很近",
            "年老的女士走得很慢",
            "高大的男士站得很安静",
            "聪明的学生读得很仔细",
            "有才华的歌手演唱得出色",
            "耐心的面包师等待得很平静",
            "小心的飞行员降落得很平稳",
            "勤劳的农民收获得很早",
            "深思的作家反思得很深",
        ],
        "target_words": [
            "勇敢", "善良", "创意", "强壮", "明智",
            "熟练", "敏捷", "忠诚", "年老", "高大",
            "聪明", "才华", "耐心", "小心", "勤劳", "深思",
        ],
    },
}

# ===== 日语语法角色数据 =====
JA_ROLES_DATA = {
    "nsubj": {
        "sentences": [
            "王は賢く王国を統治した",
            "医者は丁寧に患者を治療した",
            "芸術家は美しく肖像画を描いた",
            "兵士は勇敢に城を守った",
            "先生は明確に授業を説明した",
            "料理人は完璧に食事を調理した",
            "猫は素早くネズミを追いかけた",
            "犬は嬉しそうに骨を見つけた",
            "女性は安全に車を運転した",
            "男性は慎重に屋根を修理した",
            "学生は静かに本を読んだ",
            "歌手は見事に歌を歌った",
            "パン屋は毎日パンを焼いた",
            "パイロットは順調に飛行機を操縦した",
            "農家は勤勉に作物を育てた",
            "作家はゆっくりと小説を書いた",
        ],
        "target_words": [
            "王", "医者", "芸術家", "兵士", "先生",
            "料理人", "猫", "犬", "女性", "男性",
            "学生", "歌手", "パン屋", "パイロット", "農家", "作家",
        ],
    },
    "dobj": {
        "sentences": [
            "彼らは昨日王を即位させた",
            "彼女は最近医者を訪問した",
            "彼は芸術家を大いに称賛した",
            "私たちは今日兵士を称えた",
            "あなたは先生に感謝した",
            "客は料理人に惜しみなくチップを渡した",
            "鷹は素早く猫を追いかけた",
            "少年は外で犬を見つけた",
            "警察は素早く女性を逮捕した",
            "会社は最近男性を雇った",
            "私は大声で学生を褒めた",
            "彼らは歌手に温かく拍手した",
            "彼女はよくパン屋を訪問した",
            "彼はパイロットを大いに称賛した",
            "私たちは農家に心から感謝した",
            "編集者は作家を高く評価した",
        ],
        "target_words": [
            "王", "医者", "芸術家", "兵士", "先生",
            "料理人", "猫", "犬", "女性", "男性",
            "学生", "歌手", "パン屋", "パイロット", "農家", "作家",
        ],
    },
    "amod": {
        "sentences": [
            "勇敢な王は激しく戦った",
            "親切な医者は多くの人を助けた",
            "創造的な芸術家は上手に働いた",
            "強い兵士は遠くまで行軍した",
            "賢明な先生は明確に説明した",
            "熟練した料理人は完璧に調理した",
            "速い猫は素早く走った",
            "忠実な犬は近くに留まった",
            "年老いた女性はゆっくり歩いた",
            "背の高い男性は静かに立った",
            "聡明な学生は丁寧に読んだ",
            "才能ある歌手は見事に歌った",
            "忍耐強いパン屋は穏やかに待った",
            "慎重なパイロットはスムーズに着陸した",
            "勤勉な農家は早く収穫した",
            "思慮深い作家は深く考察した",
        ],
        "target_words": [
            "勇敢", "親切", "創造", "強い", "賢明",
            "熟練", "速い", "忠実", "年老い", "背高い",
            "聡明", "才能", "忍耐", "慎重", "勤勉", "思慮",
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
    # For CJK characters, try exact match on first character
    if len(word) > 0:
        first_char = word[0]
        for i, tok in enumerate(tokens):
            if first_char in tok:
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


def compute_syntax_angle(role_h_dict):
    """计算语法方向之间的夹角"""
    centers = {role: np.mean(hs, axis=0) for role, hs in role_h_dict.items()}
    roles = sorted(centers.keys())

    if len(roles) < 3:
        return None

    # 基准角色
    base_role = roles[0]
    base_c = centers[base_role]

    # 其他角色的方向
    dir1 = centers[roles[1]] - base_c
    dir2 = centers[roles[2]] - base_c

    norm1 = np.linalg.norm(dir1)
    norm2 = np.linalg.norm(dir2)

    if norm1 < 1e-10 or norm2 < 1e-10:
        return None

    dir1_unit = dir1 / norm1
    dir2_unit = dir2 / norm2

    # 正交化前的夹角
    cos_raw = np.dot(dir1_unit, dir2_unit)
    angle_raw_deg = np.degrees(np.arccos(np.clip(abs(cos_raw), 0, 1)))

    # Gram-Schmidt正交化后的正交性
    ortho_component = dir2_unit - np.dot(dir2_unit, dir1_unit) * dir1_unit
    ortho_norm = np.linalg.norm(ortho_component)
    cos_after_gs = np.dot(dir1_unit, ortho_component / max(ortho_norm, 1e-10))

    return {
        'cos_raw': float(cos_raw),
        'angle_raw_deg': float(angle_raw_deg),
        'cos_after_gs': float(cos_after_gs),
        'roles': roles,
    }


def compute_grassmannian_distance(p1_noun, p1_mod, p2_noun, p2_mod):
    """计算两个2D平面的Grassmannian距离"""
    P1 = np.column_stack([p1_noun, p1_mod])
    P2 = np.column_stack([p2_noun, p2_mod])
    M = P1.T @ P2
    _, s_m, _ = np.linalg.svd(M)
    d_G = np.sqrt(np.sum(np.arccos(np.clip(s_m, -1, 1)) ** 2))
    return d_G


# ===== Exp1: 方差分解(ANOVA) =====
def exp1_anova(model, tokenizer, device):
    """方差分解: Model vs Role vs 交互效应"""
    print("\n" + "="*70)
    print("Exp1: 方差分解(ANOVA) ★★★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers

    # 在中间层测试所有角色组合
    test_li = n_layers // 2
    print(f"  测试层: L{test_li}")

    role_combos = [
        ("nsubj", "dobj", "amod"),
        ("nsubj", "dobj", "poss"),
        ("nsubj", "amod", "poss"),
    ]

    combo_metrics = {}

    for base, noun_role, mod_role in role_combos:
        print(f"\n  组合: 基准={base}, 名词方向={noun_role}, 修饰语方向={mod_role}")

        role_hs = {}
        for role in [base, noun_role, mod_role]:
            data = EN_ROLES_DATA[role]
            H = collect_hs_at_layer(model, tokenizer, device,
                                    data["sentences"][:8], data["target_words"][:8], test_li)
            if H is not None:
                role_hs[role] = H

        if len(role_hs) < 3:
            print(f"  数据不足, 跳过")
            continue

        # 计算方向
        base_c = np.mean(role_hs[base], axis=0)
        noun_c = np.mean(role_hs[noun_role], axis=0)
        mod_c = np.mean(role_hs[mod_role], axis=0)

        noun_dir = noun_c - base_c
        noun_norm = np.linalg.norm(noun_dir)
        if noun_norm > 0:
            noun_dir = noun_dir / noun_norm

        mod_vec = mod_c - base_c
        mod_proj = np.dot(mod_vec, noun_dir) * noun_dir
        modifier_dir = mod_vec - mod_proj
        modifier_norm = np.linalg.norm(modifier_dir)
        if modifier_norm > 0:
            modifier_dir = modifier_dir / modifier_norm

        ortho = float(np.dot(noun_dir, modifier_dir))

        # 计算方向导数
        test_sent = "The king ruled"
        toks = tokenizer(test_sent, return_tensors="pt").to(device)
        input_ids = toks.input_ids
        last_idx = input_ids.shape[1] - 1

        embed_layer = model.get_input_embeddings()
        inputs_embeds_base = embed_layer(input_ids).detach().clone().float()

        with torch.no_grad():
            base_logits = model(inputs_embeds=inputs_embeds_base.to(model.dtype)).logits[0, last_idx, :]

        top_k = 30
        base_topk_vals, base_topk_indices = torch.topk(base_logits.float(), top_k)

        epsilon = 0.1
        epsilon2 = 0.5

        noun_tensor = torch.tensor(noun_dir, dtype=torch.float32, device=device)
        mod_tensor = torch.tensor(modifier_dir, dtype=torch.float32, device=device)

        # Jacobian
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

        # Hessian
        with torch.no_grad():
            logits_center = model(inputs_embeds=inputs_embeds_base.to(model.dtype)).logits[0, last_idx, :]

        inputs_plus2n = inputs_embeds_base.clone()
        inputs_plus2n[0, last_idx, :] += (epsilon2 * noun_tensor).to(inputs_embeds_base.dtype)
        inputs_minus2n = inputs_embeds_base.clone()
        inputs_minus2n[0, last_idx, :] -= (epsilon2 * noun_tensor).to(inputs_embeds_base.dtype)

        with torch.no_grad():
            logits_plus2n = model(inputs_embeds=inputs_plus2n.to(model.dtype)).logits[0, last_idx, :]
            logits_minus2n = model(inputs_embeds=inputs_minus2n.to(model.dtype)).logits[0, last_idx, :]

        H_noun = ((logits_plus2n[base_topk_indices] - 2*logits_center[base_topk_indices] + logits_minus2n[base_topk_indices]) / (epsilon2**2)).float().cpu().numpy()

        inputs_plus2m = inputs_embeds_base.clone()
        inputs_plus2m[0, last_idx, :] += (epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)
        inputs_minus2m = inputs_embeds_base.clone()
        inputs_minus2m[0, last_idx, :] -= (epsilon2 * mod_tensor).to(inputs_embeds_base.dtype)

        with torch.no_grad():
            logits_plus2m = model(inputs_embeds=inputs_plus2m.to(model.dtype)).logits[0, last_idx, :]
            logits_minus2m = model(inputs_embeds=inputs_minus2m.to(model.dtype)).logits[0, last_idx, :]

        H_mod = ((logits_plus2m[base_topk_indices] - 2*logits_center[base_topk_indices] + logits_minus2m[base_topk_indices]) / (epsilon2**2)).float().cpu().numpy()

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

        # 指标
        first_order_noun = np.linalg.norm(J_noun)
        first_order_mod = np.linalg.norm(J_mod)
        noun_curvature = np.linalg.norm(H_noun)
        mod_curvature = np.linalg.norm(H_mod)
        cross_curvature = np.linalg.norm(H_cross)

        nonlinear_ratio_noun = noun_curvature / max(first_order_noun, 1e-10)
        nonlinear_ratio_mod = mod_curvature / max(first_order_mod, 1e-10)
        avg_nonlinearity = (nonlinear_ratio_noun + nonlinear_ratio_mod) / 2

        cos_jacobian = np.dot(J_noun, J_mod) / max(
            np.linalg.norm(J_noun) * np.linalg.norm(J_mod), 1e-10)
        jacobian_overlap = abs(cos_jacobian)

        cross_to_single = cross_curvature / max(
            np.mean([noun_curvature, mod_curvature]), 1e-10)

        combo_key = f"{base}-{noun_role}-{mod_role}"
        combo_metrics[combo_key] = {
            'nonlinearity': float(avg_nonlinearity),
            'jacobian_overlap': float(jacobian_overlap),
            'cross_to_single': float(cross_to_single),
            'orthogonality': float(ortho),
        }

        print(f"  非线性: {avg_nonlinearity:.4f}")
        print(f"  Jacobian重叠: {jacobian_overlap:.4f}")
        print(f"  交叉/单轴: {cross_to_single:.4f}")
        print(f"  正交性: {ortho:.6f}")

    results['combo_metrics'] = combo_metrics

    # ===== 方差分解 =====
    print(f"\n  ===== 方差分解 =====")

    # 加载已有模型数据
    temp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'glm5_temp')

    all_data = []
    for model_name in ['qwen3', 'glm4', 'deepseek7b']:
        result_path = os.path.join(temp_dir, f"cclz_exp4_{model_name}_results.json")
        if os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                model_data = json.load(f)
            if 'role_combination_metrics' in model_data:
                for combo_key, metrics in model_data['role_combination_metrics'].items():
                    all_data.append({
                        'model': model_name,
                        'role_combo': combo_key,
                        'nonlinearity': metrics.get('nonlinearity', 0),
                        'jacobian_overlap': metrics.get('jacobian_overlap', 0),
                        'cross_to_single': metrics.get('cross_to_single', 0),
                    })

    # 加入当前模型数据
    for combo_key, metrics in combo_metrics.items():
        all_data.append({
            'model': args.model,
            'role_combo': combo_key,
            'nonlinearity': metrics['nonlinearity'],
            'jacobian_overlap': metrics['jacobian_overlap'],
            'cross_to_single': metrics['cross_to_single'],
        })

    if len(all_data) >= 6:
        models = list(set(d['model'] for d in all_data))
        combos = list(set(d['role_combo'] for d in all_data))

        print(f"  数据: {len(all_data)} 条, {len(models)} 模型, {len(combos)} 角色组合")

        for metric_name in ['nonlinearity', 'jacobian_overlap', 'cross_to_single']:
            values = [d[metric_name] for d in all_data]
            grand_mean = np.mean(values)
            total_ss = np.sum((np.array(values) - grand_mean) ** 2)

            # Model效应
            model_means = {}
            for m in models:
                m_vals = [d[metric_name] for d in all_data if d['model'] == m]
                model_means[m] = np.mean(m_vals)
            model_ss = sum(len([d for d in all_data if d['model'] == m]) * (model_means[m] - grand_mean)**2
                          for m in models)

            # Role效应
            combo_means = {}
            for c in combos:
                c_vals = [d[metric_name] for d in all_data if d['role_combo'] == c]
                combo_means[c] = np.mean(c_vals)
            role_ss = sum(len([d for d in all_data if d['role_combo'] == c]) * (combo_means[c] - grand_mean)**2
                         for c in combos)

            # 残差(包含交互效应)
            residual_ss = total_ss - model_ss - role_ss

            # 比例
            if total_ss > 1e-10:
                model_pct = model_ss / total_ss * 100
                role_pct = role_ss / total_ss * 100
                residual_pct = residual_ss / total_ss * 100
            else:
                model_pct = role_pct = residual_pct = 0

            print(f"\n  {metric_name}:")
            print(f"    总方差: {total_ss:.4f}")
            print(f"    Model效应: {model_pct:.1f}% (SS={model_ss:.4f})")
            print(f"    Role效应: {role_pct:.1f}% (SS={role_ss:.4f})")
            print(f"    残差(含交互): {residual_pct:.1f}% (SS={residual_ss:.4f})")

            if model_pct > role_pct and model_pct > residual_pct:
                print(f"    → ★★★ Model主效应主导!")
            elif role_pct > model_pct and role_pct > residual_pct:
                print(f"    → ★★★ Role主效应主导!")
            elif residual_pct > model_pct and residual_pct > role_pct:
                print(f"    → ★★★ 交互效应主导!")
            else:
                print(f"    → 效应混合")

            results[f'{metric_name}_model_pct'] = float(model_pct)
            results[f'{metric_name}_role_pct'] = float(role_pct)
            results[f'{metric_name}_residual_pct'] = float(residual_pct)

    return results


# ===== Exp2: 跨语言正交性验证 =====
def exp2_crosslingual_orthogonality(model, tokenizer, device):
    """跨语言正交性验证"""
    print("\n" + "="*70)
    print("Exp2: 跨语言正交性验证 ★★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers

    test_layers = [0, n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers - 1]

    for lang_name, roles_data in [("English", EN_ROLES_DATA), ("Chinese", ZH_ROLES_DATA), ("Japanese", JA_ROLES_DATA)]:
        print(f"\n  ===== {lang_name} =====")

        lang_results = {}

        for li in test_layers:
            print(f"\n  L{li}:")

            role_hs = {}
            for role in ["nsubj", "dobj", "amod"]:
                if role not in roles_data:
                    continue
                data = roles_data[role]
                H = collect_hs_at_layer(model, tokenizer, device,
                                        data["sentences"][:8], data["target_words"][:8], li)
                if H is not None and len(H) >= 3:
                    role_hs[role] = H
                    print(f"    {role}: {len(H)} samples, norm={np.mean(np.linalg.norm(H, axis=1)):.2f}")

            if len(role_hs) < 3:
                print(f"    数据不足, 跳过")
                continue

            # 计算夹角
            angle_info = compute_syntax_angle(role_hs)
            if angle_info is not None:
                print(f"    正交化前夹角: {angle_info['angle_raw_deg']:.1f}°")
                print(f"    正交化前cos: {angle_info['cos_raw']:.6f}")
                print(f"    Gram-Schmidt后cos: {angle_info['cos_after_gs']:.6f}")

                lang_results[li] = angle_info

        # 总结
        if lang_results:
            angles = [v['angle_raw_deg'] for v in lang_results.values()]
            cos_vals = [abs(v['cos_raw']) for v in lang_results.values()]

            mean_angle = np.mean(angles)
            mean_cos = np.mean(cos_vals)

            print(f"\n  {lang_name}总结:")
            print(f"    平均正交化前夹角: {mean_angle:.1f}°")
            print(f"    平均|cos|: {mean_cos:.4f}")

            if mean_angle > 80:
                print(f"    → ★★★ {lang_name}核心语法因子呈近正交组织!")
            elif mean_angle > 60:
                print(f"    → {lang_name}核心语法因子夹角中等, 部分正交")
            else:
                print(f"    → {lang_name}核心语法因子不正交")

            results[f'{lang_name}_mean_angle'] = float(mean_angle)
            results[f'{lang_name}_mean_cos'] = float(mean_cos)
            results[f'{lang_name}_layer_details'] = {str(k): v for k, v in lang_results.items()}

    # ===== 跨语言对比 =====
    print(f"\n  ===== 跨语言对比 =====")

    for metric_name in ['mean_angle', 'mean_cos']:
        lang_vals = {}
        for lang_name in ["English", "Chinese", "Japanese"]:
            key = f'{lang_name}_{metric_name}'
            if key in results:
                lang_vals[lang_name] = results[key]

        if len(lang_vals) >= 2:
            vals = list(lang_vals.values())
            print(f"  {metric_name}: {lang_vals}")
            print(f"    跨语言范围: [{min(vals):.2f}, {max(vals):.2f}]")
            print(f"    跨语言变异系数: {np.std(vals)/max(np.mean(vals),1e-10):.4f}")

    # ===== 随机方向对照(跨语言) =====
    print(f"\n  ===== 跨语言随机方向对照 =====")

    random_groups_en = [
        ["The weather is nice today", "The sun shines brightly", "The rain falls gently",
         "The wind blows softly", "The snow melts slowly", "The clouds drift apart",
         "The storm passed quickly", "The fog cleared away"],
        ["She runs every morning", "He swims in the pool", "They dance all night",
         "We sing together now", "I read before sleeping", "You write very well",
         "It plays the music", "She paints the wall"],
        ["The table is wooden", "The chair looks old", "The lamp shines dim",
         "The door closes shut", "The window opens wide", "The floor feels cold",
         "The ceiling hangs low", "The wall stands firm"],
    ]

    random_groups_zh = [
        ["今天天气很好", "太阳明晃晃地照耀", "雨轻轻地落下",
         "风轻柔地吹着", "雪慢慢地融化", "云慢慢地飘散",
         "暴风雨很快过去了", "雾气渐渐散开"],
        ["她每天早上跑步", "他在游泳池里游泳", "他们整晚跳舞",
         "我们一起唱歌", "我睡觉前读书", "你写得很好",
         "它播放着音乐", "她粉刷了墙壁"],
        ["桌子是木头的", "椅子看起来很旧", "灯发出昏暗的光",
         "门紧紧地关上了", "窗户开得很大", "地板感觉很冷",
         "天花板挂得很低", "墙立得很稳"],
    ]

    random_groups_ja = [
        ["今日は天気がいい", "太陽が明るく輝く", "雨が静かに降る",
         "風が優しく吹く", "雪がゆっくり溶ける", "雲が漂っていく",
         "嵐はすぐに過ぎた", "霧が晴れてきた"],
        ["彼女は毎朝走る", "彼はプールで泳ぐ", "彼らは一晩中踊る",
         "私たちは一緒に歌う", "私は寝る前に読む", "あなたは上手に書く",
         "それは音楽を再生する", "彼女は壁を塗る"],
        ["テーブルは木製だ", "椅子は古く見える", "ランプは薄暗い",
         "ドアは閉まっている", "窓は大きく開く", "床は冷たく感じる",
         "天井は低く垂れている", "壁はしっかり立っている"],
    ]

    for lang_name, rand_groups in [("English", random_groups_en),
                                    ("Chinese", random_groups_zh),
                                    ("Japanese", random_groups_ja)]:
        print(f"\n  {lang_name} 随机方向对照:")

        test_li = n_layers // 2
        random_centers = []

        for gi, group in enumerate(rand_groups):
            hs = []
            for sent in group:
                try:
                    toks = tokenizer(sent, return_tensors="pt").to(device)
                    captured = {}
                    layers_list = get_layers(model)
                    target_layer = layers_list[test_li]
                    def hook_fn(module, input, output):
                        if isinstance(output, tuple):
                            captured['h'] = output[0].detach().float().cpu().numpy()
                        else:
                            captured['h'] = output.detach().float().cpu().numpy()
                    h_handle = target_layer.register_forward_hook(hook_fn)
                    with torch.no_grad():
                        _ = model(**toks)
                    h_handle.remove()
                    if 'h' in captured:
                        last_idx = captured['h'].shape[1] - 1
                        hs.append(captured['h'][0, last_idx, :])
                except Exception as e:
                    pass

            if hs:
                center = np.mean(hs, axis=0)
                random_centers.append(center)

        if len(random_centers) >= 3:
            c0, c1, c2 = random_centers[0], random_centers[1], random_centers[2]
            dir_01 = c1 - c0
            dir_02 = c2 - c0
            norm_01 = np.linalg.norm(dir_01)
            norm_02 = np.linalg.norm(dir_02)

            if norm_01 > 0 and norm_02 > 0:
                cos_random = np.dot(dir_01 / norm_01, dir_02 / norm_02)
                angle_random = np.degrees(np.arccos(np.clip(abs(cos_random), 0, 1)))
                print(f"    随机方向夹角: {angle_random:.1f}° (cos={cos_random:.4f})")

                results[f'{lang_name}_random_angle'] = float(angle_random)
                results[f'{lang_name}_random_cos'] = float(cos_random)

    return results


# ===== Exp3: 扩散型标度律的微观结构分析 =====
def exp3_micro_structure(model, tokenizer, device):
    """扩散型标度律的微观结构: 旋转是否有记忆?"""
    print("\n" + "="*70)
    print("Exp3: 扩散型标度律的微观结构 ★★★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers

    # 采集所有层的2D平面
    print("\n  Step 1: 采集所有层的2D语法子空间")

    all_layer_planes = {}
    for li in range(n_layers):
        print(f"  L{li}...", end="", flush=True)

        role_hs = {}
        for role in ["nsubj", "dobj", "amod"]:
            data = EN_ROLES_DATA[role]
            H = collect_hs_at_layer(model, tokenizer, device,
                                    data["sentences"][:8], data["target_words"][:8], li)
            if H is not None:
                role_hs[role] = H

        if len(role_hs) < 3:
            print(f" 跳过")
            continue

        centers = {role: np.mean(hs, axis=0) for role, hs in role_hs.items()}

        noun_dir = centers['dobj'] - centers['nsubj']
        noun_norm = np.linalg.norm(noun_dir)
        if noun_norm > 0:
            noun_dir = noun_dir / noun_norm

        amod_vec = centers['amod'] - centers['nsubj']
        amod_proj = np.dot(amod_vec, noun_dir) * noun_dir
        mod_dir = amod_vec - amod_proj
        mod_norm = np.linalg.norm(mod_dir)
        if mod_norm > 0:
            mod_dir = mod_dir / mod_norm

        all_layer_planes[li] = {
            'noun_axis': noun_dir,
            'modifier_axis': mod_dir,
        }
        print(f" ok")

    if len(all_layer_planes) < 4:
        print("  采集不足!")
        return results

    # Step 2: 逐层旋转角
    print("\n  Step 2: 逐层旋转角")

    sorted_layers = sorted(all_layer_planes.keys())
    step_dGs = []
    for i in range(len(sorted_layers) - 1):
        li = sorted_layers[i]
        li1 = sorted_layers[i + 1]
        p = all_layer_planes[li]
        p1 = all_layer_planes[li1]
        dG = compute_grassmannian_distance(
            p['noun_axis'], p['modifier_axis'],
            p1['noun_axis'], p1['modifier_axis']
        )
        step_dGs.append((li, li1, dG))
        print(f"  L{li}→L{li1}: dG={np.degrees(dG):.2f}°")

    # Step 3: 旋转角的自相关分析
    print("\n  Step 3: 旋转角的自相关分析")

    dG_values = [dG for _, _, dG in step_dGs]

    if len(dG_values) >= 4:
        # lag-1自相关
        dG_arr = np.array(dG_values)
        mean_dG = np.mean(dG_arr)
        dG_centered = dG_arr - mean_dG

        if np.sum(dG_centered**2) > 0:
            autocorr_lag1 = np.sum(dG_centered[:-1] * dG_centered[1:]) / np.sum(dG_centered**2)
            autocorr_lag2 = np.sum(dG_centered[:-2] * dG_centered[2:]) / np.sum(dG_centered**2) if len(dG_centered) > 2 else 0

            print(f"  lag-1自相关: {autocorr_lag1:.4f}")
            print(f"  lag-2自相关: {autocorr_lag2:.4f}")

            if autocorr_lag1 > 0.3:
                print(f"  → ★★★ 旋转角有正自相关! 相邻层旋转方向有记忆!")
                print(f"  → 不是纯随机游走, 更像约束流形上的确定性运动")
            elif autocorr_lag1 < -0.3:
                print(f"  → 旋转角有负自相关! 交替大/小旋转")
                print(f"  → 类似于振荡行为")
            else:
                print(f"  → 旋转角近似独立, 接近随机游走")

            results['autocorr_lag1'] = float(autocorr_lag1)
            results['autocorr_lag2'] = float(autocorr_lag2)

    # Step 4: 旋转方向的分析
    # 定义: 旋转方向 = 相邻层的Grassmannian旋转的"方向"
    # 用旋转轴向量来表示
    print("\n  Step 4: 旋转方向记忆分析")

    # 更细致的分析: 相邻3层的旋转是否沿相似方向?
    # 用连续3层dG的符号一致性来衡量
    if len(step_dGs) >= 3:
        # 将dG分为"前半大"和"前半小时"
        # 随机游走: 前后半旋转大小没有关联
        # 确定性运动: 可能有关联

        # 旋转角的符号变化
        dG_changes = np.diff(dG_values)
        sign_changes = np.sum(np.diff(np.sign(dG_changes)) != 0) if len(dG_changes) > 1 else 0

        # 预期: 随机游走, sign_changes ≈ (n-2)/2
        expected_changes = max(len(dG_changes) - 1, 1) / 2

        print(f"  旋转角变化的符号翻转次数: {sign_changes}")
        print(f"  随机游走期望: {expected_changes:.1f}")

        if sign_changes < expected_changes * 0.5:
            print(f"  → 旋转角变化有趋势性(符号很少翻转) → 确定性趋势!")
        elif sign_changes > expected_changes * 1.5:
            print(f"  → 旋转角变化振荡频繁 → 振荡型确定性运动")
        else:
            print(f"  → 旋转角变化接近随机 → 统计上与随机游走一致")

        results['sign_changes'] = int(sign_changes)
        results['expected_changes'] = float(expected_changes)

    # Step 5: 长程关联的Hurst指数
    print("\n  Step 5: Hurst指数(长程关联)")

    if len(dG_values) >= 8:
        # R/S分析
        dG_arr = np.array(dG_values)

        def hurst_rs(ts):
            """简化的R/S分析"""
            N = len(ts)
            if N < 4:
                return 0.5

            max_k = min(N // 2, 20)
            rs_values = []

            for k in [2, 4, 8, 16]:
                if k > N // 2:
                    continue
                n_sub = N // k
                rs_list = []
                for i in range(n_sub):
                    sub = ts[i*k:(i+1)*k]
                    if len(sub) < 2:
                        continue
                    mean_sub = np.mean(sub)
                    cum_dev = np.cumsum(sub - mean_sub)
                    R = np.max(cum_dev) - np.min(cum_dev)
                    S = np.std(sub)
                    if S > 0:
                        rs_list.append(R / S)
                if rs_list:
                    rs_values.append((k, np.mean(rs_list)))

            if len(rs_values) >= 2:
                log_k = np.log([k for k, _ in rs_values])
                log_rs = np.log([rs for _, rs in rs_values])
                slope, _ = np.polyfit(log_k, log_rs, 1)
                return slope
            return 0.5

        hurst = hurst_rs(dG_arr)
        print(f"  Hurst指数: {hurst:.4f}")
        print(f"  H=0.5 → 纯随机游走")
        print(f"  H>0.5 → 长程正相关(趋势持续) → 确定性成分")
        print(f"  H<0.5 → 长程负相关(均值回归) → 反持续")

        if hurst > 0.6:
            print(f"  → ★★★ H>0.6, 有长程正相关! 旋转轨迹有确定性趋势!")
            print(f"  → 曲率比∝sqrt(N)只是粗粒度标度律, 微观有结构!")
        elif hurst < 0.4:
            print(f"  → H<0.4, 均值回归行为")
        else:
            print(f"  → H≈0.5, 统计上与随机游走一致")
            print(f"  → 注意: 不排除确定性系统(某些确定性混沌H≈0.5)")

        results['hurst_exponent'] = float(hurst)

    results['step_dGs'] = [(li, li1, float(dG)) for li, li1, dG in step_dGs]

    return results


# ===== Exp4: 相对范数与probing准确率 =====
def exp4_relative_norm_probing(model, tokenizer, device):
    """相对范数曲线与probing准确率的相关性"""
    print("\n" + "="*70)
    print("Exp4: 相对范数与probing准确率 ★★★★★")
    print("="*70)

    results = {}
    model_info = get_model_info(model, args.model if hasattr(args, 'model') else 'qwen3')
    n_layers = model_info.n_layers

    # 收集所有层的hidden states和标签
    print("\n  Step 1: 收集所有层的语法角色hidden states")

    roles = ["nsubj", "dobj", "amod", "poss"]
    sample_per_role = 8

    all_layer_data = {}

    for li in range(n_layers):
        print(f"  L{li}...", end="", flush=True)

        X_all = []  # hidden states
        y_all = []  # role labels

        for role_idx, role in enumerate(roles):
            data = EN_ROLES_DATA[role]
            H = collect_hs_at_layer(model, tokenizer, device,
                                    data["sentences"][:sample_per_role],
                                    data["target_words"][:sample_per_role], li)
            if H is not None:
                X_all.append(H)
                y_all.extend([role_idx] * len(H))

        if len(X_all) >= 3 and len(y_all) >= 12:
            X = np.vstack(X_all)
            y = np.array(y_all)

            # Probing准确率
            try:
                clf = LogisticRegression(max_iter=200, multi_class='ovr')
                scores = cross_val_score(clf, X, y, cv=min(3, len(set(y))), scoring='accuracy')
                probe_acc = float(np.mean(scores))
            except Exception as e:
                probe_acc = 0.0

            # 相对范数
            centers = {}
            for role_idx, role in enumerate(roles):
                mask = y == role_idx
                if np.sum(mask) > 0:
                    centers[role] = np.mean(X[mask], axis=0)

            if 'nsubj' in centers and 'dobj' in centers:
                noun_dir = centers['dobj'] - centers['nsubj']
                noun_norm = np.linalg.norm(noun_dir)
                avg_hs_norm = np.mean(np.linalg.norm(X, axis=1))
                relative_norm = noun_norm / max(avg_hs_norm, 1e-10)
            else:
                relative_norm = 0

            all_layer_data[li] = {
                'probe_acc': probe_acc,
                'relative_norm': relative_norm,
                'n_samples': len(y),
            }

            print(f" probe={probe_acc:.3f}, rel_norm={relative_norm:.4f}")
        else:
            print(f" 数据不足")

    # Step 2: 相关性分析
    print("\n  Step 2: 相对范数 vs Probing准确率的相关性")

    if len(all_layer_data) >= 3:
        layers = sorted(all_layer_data.keys())
        rel_norms = [all_layer_data[li]['relative_norm'] for li in layers]
        probe_accs = [all_layer_data[li]['probe_acc'] for li in layers]

        # Pearson相关
        if np.std(rel_norms) > 0 and np.std(probe_accs) > 0:
            corr = np.corrcoef(rel_norms, probe_accs)[0, 1]
            print(f"  Pearson相关: {corr:.4f}")

            if corr > 0.7:
                print(f"  → 相对范数与probing准确率强相关!")
                print(f"  → 语法方向的可见度确实反映了语法理解能力")
            elif corr > 0.4:
                print(f"  → 中等相关, 相对范数部分反映语法能力")
            else:
                print(f"  → 弱相关, 相对范数不直接反映语法能力")

            results['probe_norm_corr'] = float(corr)

        # 找到相对范数达到峰值的层
        peak_layer_rel = layers[np.argmax(rel_norms)]
        peak_layer_probe = layers[np.argmax(probe_accs)]
        print(f"\n  相对范数峰值层: L{peak_layer_rel}")
        print(f"  Probing准确率峰值层: L{peak_layer_probe}")

        # 增长速率分析
        if len(rel_norms) >= 4:
            rel_growth = np.diff(rel_norms)
            probe_growth = np.diff(probe_accs)

            # 前1/3 vs 后1/3
            n = len(rel_growth)
            first_third = rel_growth[:n//3]
            last_third = rel_growth[2*n//3:]

            if len(first_third) > 0 and len(last_third) > 0:
                print(f"\n  相对范数增长速率:")
                print(f"    前1/3层: mean={np.mean(first_third):.6f}")
                print(f"    后1/3层: mean={np.mean(last_third):.6f}")

                if np.mean(first_third) > 0 and np.mean(last_third) <= 0:
                    print(f"    → 前层增长, 后层持平/下降 → '学习→使用'转折!")
                elif np.mean(first_third) > np.mean(last_third):
                    print(f"    → 前层增长快, 后层增长慢 → 饱和趋势")

    results['layer_data'] = {str(k): v for k, v in all_layer_data.items()}

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
    print(f"CCMA Phase21 方差分解与跨语言验证 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")

    try:
        if args.exp == 1:
            results = exp1_anova(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_crosslingual_orthogonality(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_micro_structure(model, tokenizer, device)
        elif args.exp == 4:
            results = exp4_relative_norm_probing(model, tokenizer, device)

        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                               f"ccma_exp{args.exp}_{args.model}_results.json")

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
