"""
CCL-S(Phase 13): 语法编码的真正数学结构
=============================================================================
核心问题:
  Phase 12修正了"全息编码"假说: 冗余度≈n_bands是高维空间的一般性质, 非语法特有
  真正需要解释的是:
  1. 为什么4类语法角色只需要2-5维? 这是语法的特殊性质还是一般性质?
  2. nsubj-poss不可分→模型是否将"名词性成分"视为同一类?
  3. 语法角色是否形成子空间? 名词/动词/修饰/功能各有独立子空间?
  4. 语法(2-5维)→语义(5-10维)的维度增加来自哪里?

实验:
  Exp1: ★★★★★ 名词性角色超类假说
    → 系统测试: nsubj/poss/det/nummod/compound是否共享表示?
    → 如果所有"名词性"角色的CV≈随机: 名词性角色形成"超类"
    → 如果某些对可区分: 更细粒度的角色区分

  Exp2: ★★★★★ 语法角色子空间正交性
    → 测量名词/动词/形容词/副词子空间的正交性
    → 如果子空间正交: 语法角色有独立的编码子空间
    → 如果子空间重叠: 语法角色共享表示空间

  Exp3: ★★★★★ 语法低维性的对照实验
    → 创建随机4类数据, 测量其内在维度
    → 对比: 语法4类的2-5维 vs 随机4类的?维
    → 如果随机4类也是2-5维: 低维性不是语法特有
    → 如果随机4类需要更多维: 语法的低维性是特殊性质

  Exp4: ★★★★★ 语法→语义的维度桥梁
    → 设计4级分类任务, 从纯语法到纯语义
    → 级别1: 纯语法(4语法角色)
    → 级别2: 语法-语义(有生命主语/无生命主语/有生命宾语/无生命宾语)
    → 级别3: 词汇语义(4类语义关系)
    → 级别4: 概念语义(4类抽象概念)
    → 测量每级的内在维度, 找到维度跃迁点
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from model_utils import load_model, get_layers, get_model_info, release_model, safe_decode, get_W_U


# ===== Exp1: 名词性角色超类假说 =====
# 核心设计: 系统测试所有"名词性"依赖角色
# 如果nsubj/poss/det/nummod/compound都不可分→名词性超类成立
# 如果某些对可分→需要更细粒度的理论

NOMINAL_ROLES_DATA = {
    "nsubj": {
        "desc": "主语名词",
        "sentences": [
            "The king ruled the kingdom", "The doctor treated the patient",
            "The artist painted the portrait", "The soldier defended the castle",
            "The cat sat on the mat", "The dog ran through the park",
            "The woman drove the car", "The man fixed the roof",
            "The student read the textbook", "The teacher explained the lesson",
            "The president signed the bill", "The chef cooked the meal",
            "The baker baked the bread", "The pilot flew the airplane",
            "The singer sang a song", "The farmer grew crops",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "woman", "man", "student", "teacher", "president", "chef",
            "baker", "pilot", "singer", "farmer",
        ],
    },
    "poss": {
        "desc": "所有格修饰语",
        "sentences": [
            "The king's crown glittered brightly", "The doctor's office opened early",
            "The artist's studio looked beautiful", "The soldier's uniform was clean",
            "The cat's tail swished gently", "The dog's bark echoed loudly",
            "The woman's dress looked elegant", "The man's car drove fast",
            "The student's essay read well", "The teacher's book sold quickly",
            "The president's speech inspired many", "The chef's restaurant opened today",
            "The baker's shop smelled wonderful", "The pilot's license was renewed",
            "The singer's voice rang clearly", "The farmer's land was fertile",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "woman", "man", "student", "teacher", "president", "chef",
            "baker", "pilot", "singer", "farmer",
        ],
    },
    "dobj": {
        "desc": "直接宾语",
        "sentences": [
            "They crowned the king yesterday", "She visited the doctor recently",
            "He admired the artist greatly", "We honored the soldier today",
            "She chased the cat away", "He found the dog outside",
            "The police arrested the man quickly", "The company hired the woman recently",
            "I praised the student loudly", "You thanked the teacher warmly",
            "The nation elected the president fairly", "The customer tipped the chef generously",
            "They hired the baker recently", "She admired the pilot greatly",
            "We praised the singer loudly", "He visited the farmer often",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "man", "woman", "student", "teacher", "president", "chef",
            "baker", "pilot", "singer", "farmer",
        ],
    },
    "iobj": {
        "desc": "间接宾语(给某人)",
        "sentences": [
            "She gave the king the letter", "He sent the doctor the report",
            "They offered the artist a prize", "We handed the soldier the flag",
            "I gave the cat some food", "She threw the dog a bone",
            "He told the woman the truth", "They brought the man the tools",
            "I gave the student a book", "She sent the teacher a message",
            "They gave the president the file", "He offered the chef the recipe",
            "She gave the baker the order", "They sent the pilot the schedule",
            "We offered the singer the role", "He gave the farmer the seeds",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "woman", "man", "student", "teacher", "president", "chef",
            "baker", "pilot", "singer", "farmer",
        ],
    },
    "nmod": {
        "desc": "名词修饰语(介词宾语中的名词)",
        "sentences": [
            "He stood by the king proudly", "She worked with the doctor carefully",
            "They performed for the artist beautifully", "We marched behind the soldier bravely",
            "The dog played with the cat gently", "The child ran after the dog quickly",
            "He talked to the woman politely", "She sat beside the man quietly",
            "I studied with the student earnestly", "They learned from the teacher eagerly",
            "The advisor briefed the president thoroughly", "The apprentice watched the chef closely",
            "They shopped at the baker daily", "He flew with the pilot safely",
            "She rehearsed with the singer diligently", "They walked with the farmer slowly",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "woman", "man", "student", "teacher", "president", "chef",
            "baker", "pilot", "singer", "farmer",
        ],
    },
}

# 对照组: 非名词性角色
NON_NOMINAL_ROLES_DATA = {
    "amod": {
        "desc": "形容词修饰语",
        "sentences": [
            "The brave king fought hard", "The kind doctor helped many",
            "The creative artist worked well", "The strong soldier marched far",
            "The beautiful cat sat quietly", "The large dog ran swiftly",
            "The old woman walked slowly", "The tall man stood quietly",
            "The bright student read carefully", "The wise teacher explained clearly",
            "The powerful president decided firmly", "The skilled chef cooked perfectly",
            "The patient baker waited calmly", "The careful pilot landed smoothly",
            "The talented singer performed brilliantly", "The hardworking farmer harvested early",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "beautiful", "large",
            "old", "tall", "bright", "wise", "powerful", "skilled",
            "patient", "careful", "talented", "hardworking",
        ],
    },
    "advmod": {
        "desc": "副词修饰语",
        "sentences": [
            "The king ruled wisely forever", "The doctor worked carefully always",
            "The artist painted beautifully daily", "The soldier fought bravely there",
            "The cat ran quickly home", "The dog barked loudly today",
            "The woman drove slowly home", "The man spoke quietly now",
            "The student read carefully alone", "The teacher spoke clearly again",
            "The president spoke firmly today", "The chef worked quickly then",
            "The baker baked freshly daily", "The pilot flew steadily onward",
            "The singer sang softly tonight", "The farmer worked diligently always",
        ],
        "target_words": [
            "wisely", "carefully", "beautifully", "bravely", "quickly", "loudly",
            "slowly", "quietly", "carefully", "clearly", "firmly", "quickly",
            "freshly", "steadily", "softly", "diligently",
        ],
    },
}

# 所有角色合并
ALL_ROLES_DATA = {**NOMINAL_ROLES_DATA, **NON_NOMINAL_ROLES_DATA}


# ===== Exp2: 子空间正交性 =====
# 名词/动词/形容词/副词的子空间分析
SUBSPACE_DATA = {
    "nouns": {
        "desc": "名词(句首主语位)",
        "sentences": [
            "Kings ruled the ancient kingdom", "Doctors treated the sick patient",
            "Artists painted the beautiful portrait", "Soldiers defended the mighty castle",
            "Cats sat on the soft mat", "Dogs ran through the green park",
            "Women drove the new car", "Men fixed the broken roof",
            "Students read the thick textbook", "Teachers explained the complex lesson",
            "Presidents signed the important bill", "Chefs cooked the delicious meal",
            "Bakers baked the fresh bread", "Pilots flew the large airplane",
            "Singers sang the happy song", "Farmers grew the golden crops",
        ],
        "target_words": [
            "Kings", "Doctors", "Artists", "Soldiers", "Cats", "Dogs",
            "Women", "Men", "Students", "Teachers", "Presidents", "Chefs",
            "Bakers", "Pilots", "Singers", "Farmers",
        ],
    },
    "verbs": {
        "desc": "动词(谓语位)",
        "sentences": [
            "The king ruled wisely", "The doctor treated carefully",
            "The artist painted beautifully", "The soldier fought bravely",
            "The cat jumped quickly", "The dog ran swiftly",
            "The woman drove carefully", "The man spoke loudly",
            "The student studied hard", "The teacher taught well",
            "The president decided firmly", "The chef cooked expertly",
            "The baker baked daily", "The pilot flew safely",
            "The singer performed brilliantly", "The farmer worked diligently",
        ],
        "target_words": [
            "ruled", "treated", "painted", "fought", "jumped", "ran",
            "drove", "spoke", "studied", "taught", "decided", "cooked",
            "baked", "flew", "performed", "worked",
        ],
    },
    "adjectives": {
        "desc": "形容词(定语位)",
        "sentences": [
            "The brave king fought", "The kind doctor helped",
            "The creative artist worked", "The strong soldier marched",
            "The beautiful cat sat", "The large dog ran",
            "The old woman walked", "The tall man stood",
            "The bright student read", "The wise teacher explained",
            "The powerful president decided", "The skilled chef cooked",
            "The patient baker waited", "The careful pilot landed",
            "The talented singer performed", "The hardworking farmer harvested",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "beautiful", "large",
            "old", "tall", "bright", "wise", "powerful", "skilled",
            "patient", "careful", "talented", "hardworking",
        ],
    },
    "adverbs": {
        "desc": "副词(状语位)",
        "sentences": [
            "He ruled wisely always", "She worked carefully forever",
            "They painted beautifully daily", "He fought bravely there",
            "It ran quickly home", "It barked loudly today",
            "She drove slowly home", "He spoke quietly now",
            "She studied carefully alone", "He explained clearly again",
            "She decided firmly today", "He cooked quickly then",
            "She baked freshly daily", "He flew steadily onward",
            "She sang softly tonight", "He worked diligently always",
        ],
        "target_words": [
            "wisely", "carefully", "beautifully", "bravely", "quickly", "loudly",
            "slowly", "quietly", "carefully", "clearly", "firmly", "quickly",
            "freshly", "steadily", "softly", "diligently",
        ],
    },
}


# ===== Exp3: 语法低维性的对照 =====
# 核心: 语法4类 vs 随机4类 vs 语义4类 的内在维度对比

# 语法-语义过渡数据
SYNTAX_SEMANTIC_BRIDGE_DATA = {
    # 级别1: 纯语法(4语法角色)
    "pure_syntax": {
        "desc": "纯语法: nsubj/dobj/amod/advmod",
        "categories": {
            "nsubj": {
                "sentences": [
                    "The king ruled the kingdom", "The doctor treated the patient",
                    "The artist painted the portrait", "The soldier defended the castle",
                    "The cat sat on the mat", "The dog ran through the park",
                    "The woman drove the car", "The man fixed the roof",
                    "The student read the textbook", "The teacher explained the lesson",
                    "The president signed the bill", "The chef cooked the meal",
                ],
            },
            "dobj": {
                "sentences": [
                    "They crowned the king yesterday", "She visited the doctor recently",
                    "He admired the artist greatly", "We honored the soldier today",
                    "She chased the cat away", "He found the dog outside",
                    "The police arrested the man quickly", "The company hired the woman recently",
                    "I praised the student loudly", "You thanked the teacher warmly",
                    "The nation elected the president fairly", "The customer tipped the chef generously",
                ],
            },
            "amod": {
                "sentences": [
                    "The brave king fought hard", "The kind doctor helped many",
                    "The creative artist worked well", "The strong soldier marched far",
                    "The beautiful cat sat quietly", "The large dog ran swiftly",
                    "The old woman walked slowly", "The tall man stood quietly",
                    "The bright student read carefully", "The wise teacher explained clearly",
                    "The powerful president decided firmly", "The skilled chef cooked perfectly",
                ],
            },
            "advmod": {
                "sentences": [
                    "The king ruled wisely forever", "The doctor worked carefully always",
                    "The artist painted beautifully daily", "The soldier fought bravely there",
                    "The cat ran quickly home", "The dog barked loudly today",
                    "The woman drove slowly home", "The man spoke quietly now",
                    "The student read carefully alone", "The teacher spoke clearly again",
                    "The president spoke firmly today", "The chef worked quickly then",
                ],
            },
        },
    },
    # 级别2: 语法-语义(有生命主语/无生命主语/有生命宾语/无生命宾语)
    "syntax_semantic": {
        "desc": "语法-语义: 有生命主语/无生命主语/有生命宾语/无生命宾语",
        "categories": {
            "animate_subj": {
                "sentences": [
                    "The boy opened the door", "The girl read the book",
                    "The man drove the car", "The woman cooked the meal",
                    "The child played the game", "The teacher wrote the letter",
                    "The doctor examined the patient", "The farmer planted the seed",
                    "The artist drew the picture", "The singer performed the song",
                    "The builder made the wall", "The nurse gave the medicine",
                ],
            },
            "inanimate_subj": {
                "sentences": [
                    "The wind opened the door", "The fire burned the book",
                    "The machine drove the process", "The oven cooked the meal",
                    "The computer played the game", "The pen wrote the letter",
                    "The microscope examined the cell", "The rain watered the seed",
                    "The camera drew the image", "The radio performed the broadcast",
                    "The crane made the building", "The pump gave the water",
                ],
            },
            "animate_obj": {
                "sentences": [
                    "The door hit the boy suddenly", "The book inspired the girl greatly",
                    "The car injured the man badly", "The meal fed the woman well",
                    "The game entertained the child fully", "The letter informed the teacher clearly",
                    "The test challenged the doctor much", "The land supported the farmer well",
                    "The paint covered the artist lightly", "The stage frightened the singer greatly",
                    "The tool helped the builder greatly", "The medicine cured the nurse quickly",
                ],
            },
            "inanimate_obj": {
                "sentences": [
                    "He opened the door quickly", "She read the book carefully",
                    "He drove the car safely", "She cooked the meal perfectly",
                    "He played the game well", "She wrote the letter neatly",
                    "He examined the document thoroughly", "She planted the flower carefully",
                    "He painted the wall beautifully", "She sang the song softly",
                    "He built the structure solidly", "She prepared the medicine carefully",
                ],
            },
        },
    },
    # 级别3: 词汇语义(4类语义关系)
    "lexical_semantic": {
        "desc": "词汇语义: 情感/反义/上位/下位",
        "categories": {
            "emotion": {
                "sentences": [
                    "She felt joyful about the news", "He was angry at the delay",
                    "They were sad about the loss", "She felt fearful of the dark",
                    "He was disgusted by the smell", "They were surprised by the result",
                    "She felt proud of her work", "He was ashamed of his mistake",
                    "They were jealous of the success", "She felt guilty about the lie",
                    "He was hopeful about the future", "They were anxious about the test",
                ],
            },
            "antonym": {
                "sentences": [
                    "Hot and cold are opposites", "Big and small are contrasts",
                    "Fast and slow differ greatly", "Light and dark contrast sharply",
                    "Rich and poor are extremes", "Strong and weak vary widely",
                    "Young and old differ clearly", "Happy and sad are opposites",
                    "High and low are contrasts", "Hard and soft differ greatly",
                    "Loud and quiet vary widely", "Clean and dirty are opposites",
                ],
            },
            "hypernym": {
                "sentences": [
                    "A dog is a kind of animal", "A rose is a type of flower",
                    "An oak is a species of tree", "A car is a type of vehicle",
                    "A knife is a kind of tool", "A shirt is a type of clothing",
                    "Rice is a kind of grain", "Gold is a type of metal",
                    "A diamond is a kind of gem", "A piano is a type of instrument",
                    "A chair is a kind of furniture", "A book is a type of publication",
                ],
            },
            "hyponym": {
                "sentences": [
                    "An animal could be a dog", "A flower might be a rose",
                    "A tree could be an oak", "A vehicle might be a car",
                    "A tool could be a knife", "Clothing might include shirts",
                    "A grain could be rice", "A metal might be gold",
                    "A gem could be a diamond", "An instrument might be a piano",
                    "Furniture could include chairs", "A publication might be a book",
                ],
            },
        },
    },
    # 级别4: 概念语义(4类抽象概念)
    "conceptual_semantic": {
        "desc": "概念语义: 因果/条件/递进/转折",
        "categories": {
            "causal": {
                "sentences": [
                    "The rain caused the flood", "The heat melted the ice",
                    "The wind broke the window", "The fire burned the forest",
                    "The cold froze the lake", "The storm damaged the roof",
                    "The drought killed the crops", "The earthquake destroyed the building",
                    "The explosion shattered the glass", "The virus infected the patient",
                    "The medicine cured the disease", "The exercise strengthened the muscle",
                ],
            },
            "conditional": {
                "sentences": [
                    "If it rains the ground gets wet", "If you study you will pass",
                    "If she runs she will win", "If they work they get paid",
                    "If we try we can succeed", "If he rests he recovers",
                    "If the sun shines the ice melts", "If you eat you feel full",
                    "If she practices she improves", "If they build they create",
                    "If we plan we achieve", "If he reads he learns",
                ],
            },
            "progressive": {
                "sentences": [
                    "He worked hard and earned more", "She studied more and improved greatly",
                    "They practiced daily and mastered the skill", "He saved money and bought a house",
                    "She trained hard and won the race", "They invested wisely and grew wealthy",
                    "He exercised regularly and got stronger", "She wrote daily and published books",
                    "They researched deeply and discovered truth", "He practiced medicine and saved lives",
                    "She painted often and created beauty", "They explored widely and found treasure",
                ],
            },
            "adversative": {
                "sentences": [
                    "He tried hard but still failed", "She studied hard yet barely passed",
                    "They planned carefully but things went wrong", "He was tired yet kept working",
                    "She was afraid but faced the challenge", "They were outnumbered yet won the battle",
                    "He was ill but attended the meeting", "She was young yet very wise",
                    "They were poor but remained happy", "He was wrong but refused to admit",
                    "She was hurt but kept smiling", "They were late but finished first",
                ],
            },
        },
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


def collect_hidden_states_multirole(model, tokenizer, device, role_names, data_dict, layer_idx=-1):
    """收集多个语法角色的hidden states"""
    all_h = []
    all_labels = []

    layers = get_layers(model)
    target_layer = layers[layer_idx] if layer_idx >= 0 else layers[-1]

    for role_idx, role in enumerate(role_names):
        data = data_dict[role]
        for sent, target_word in zip(data["sentences"], data["target_words"]):
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
            all_labels.append(role_idx)

    return np.array(all_h), np.array(all_labels)


def collect_hidden_states_categories(model, tokenizer, device, data_dict, layer_idx=-1):
    """收集多类别hidden states(取句末token)"""
    all_h = []
    all_labels = []

    layers = get_layers(model)
    target_layer = layers[layer_idx] if layer_idx >= 0 else layers[-1]

    for cat_idx, (cat_name, cat_data) in enumerate(data_dict.items()):
        sents = cat_data["sentences"] if "sentences" in cat_data else cat_data
        for sent in sents:
            toks = tokenizer(sent, return_tensors="pt").to(device)

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

            h_vec = captured['h'][0, -1, :]
            all_h.append(h_vec)
            all_labels.append(cat_idx)

    return np.array(all_h), np.array(all_labels)


def compute_intrinsic_dim(H, labels, threshold=0.95):
    """计算内在维度"""
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)

    probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    cv_full = cross_val_score(probe, H_scaled, labels, cv=5, scoring='accuracy')
    full_acc = cv_full.mean()

    max_dim = min(H.shape[0], H.shape[1]) - 1
    dims_to_test = [d for d in [1, 2, 3, 5, 7, 10, 15, 20, 30, 50] if d < max_dim]

    dim_curve = {}
    intrinsic_dim = None

    for dim in dims_to_test:
        pca = PCA(n_components=dim)
        H_pca = pca.fit_transform(H)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv = cross_val_score(probe, H_pca, labels, cv=5, scoring='accuracy')
        dim_curve[dim] = float(cv.mean())

        if intrinsic_dim is None and cv.mean() >= full_acc * threshold:
            intrinsic_dim = dim

    return {
        'full_acc': float(full_acc),
        'dim_curve': dim_curve,
        'intrinsic_dim': intrinsic_dim,
        'threshold': threshold,
    }


def compute_pairwise_cv(H, labels, role_names):
    """计算所有角色对之间的分类CV"""
    results = {}
    n_roles = len(role_names)

    for i in range(n_roles):
        for j in range(i + 1, n_roles):
            mask = (labels == i) | (labels == j)
            H_pair = H[mask]
            labels_pair = labels[mask]

            if len(H_pair) < 8:
                continue

            scaler = StandardScaler()
            H_scaled = scaler.fit_transform(H_pair)
            probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            cv = cross_val_score(probe, H_scaled, labels_pair, cv=5, scoring='accuracy')

            pair_key = f"{role_names[i]}_vs_{role_names[j]}"
            results[pair_key] = float(cv.mean())

    return results


def compute_subspace_overlap(H1, H2, n_components=10):
    """计算两个子空间的重叠度(Grassmann距离)"""
    n1 = min(n_components, min(H1.shape[0], H1.shape[1]) - 1)
    n2 = min(n_components, min(H2.shape[0], H2.shape[1]) - 1)
    n_comp = min(n1, n2)

    if n_comp < 1:
        return None

    pca1 = PCA(n_components=n_comp)
    pca2 = PCA(n_components=n_comp)

    H1_pca = pca1.fit_transform(StandardScaler().fit_transform(H1))
    H2_pca = pca2.fit_transform(StandardScaler().fit_transform(H2))

    # 子空间基
    U1 = pca1.components_.T  # [d_model, n_comp]
    U2 = pca2.components_.T  # [d_model, n_comp]

    # 投影矩阵: U1 @ U1^T 将向量投影到U1子空间
    # 子空间重叠 = ||U1^T @ U2||_F^2 / n_comp
    # 这测量U2的基向量在U1子空间中的能量
    proj_matrix = U1.T @ U2  # [n_comp, n_comp]
    singular_values = np.linalg.svd(proj_matrix, compute_uv=False)
    # Principal angles
    cos_angles = np.clip(singular_values, 0, 1)
    # Grassmann距离
    angles = np.arccos(cos_angles)
    mean_angle = np.mean(angles)
    # 重叠度 = cos(mean_angle) ∈ [0, 1]
    overlap = float(np.cos(mean_angle))

    return {
        'overlap': overlap,
        'mean_angle': float(np.degrees(mean_angle)),
        'principal_angles_deg': [float(np.degrees(a)) for a in angles],
        'singular_values': singular_values.tolist(),
    }


# ===== Exp1: 名词性角色超类假说 =====
def exp1_nominal_superclass(model, tokenizer, device):
    """系统测试名词性角色的共享表示结构"""
    print("\n" + "="*70)
    print("Exp1: 名词性角色超类假说 ★★★★★")
    print("="*70)

    results = {}

    # Part A: 名词性角色之间的两两CV
    print(f"\n  Part A: 名词性角色之间的两两分类CV")
    nominal_roles = list(NOMINAL_ROLES_DATA.keys())
    print(f"  名词性角色: {nominal_roles}")

    H_nom, labels_nom = collect_hidden_states_multirole(
        model, tokenizer, device, nominal_roles, NOMINAL_ROLES_DATA)
    print(f"  样本数: {len(H_nom)}")

    if len(H_nom) >= 20:
        # 所有名词性角色的全局分类
        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H_nom)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv_all = cross_val_score(probe, H_scaled, labels_nom, cv=5, scoring='accuracy')
        print(f"  所有名词性角色分类CV: {cv_all.mean():.4f}")
        results['nominal_all_cv'] = float(cv_all.mean())

        # 两两CV
        pairwise_cv = compute_pairwise_cv(H_nom, labels_nom, nominal_roles)
        print(f"\n  名词性角色两两CV:")
        for pair, cv in sorted(pairwise_cv.items()):
            marker = "→ 不可分!" if cv < 0.55 else "→ 可区分" if cv > 0.7 else "→ 弱区分"
            print(f"    {pair}: CV={cv:.4f} {marker}")
        results['nominal_pairwise_cv'] = pairwise_cv

        # 内在维度
        dim_result = compute_intrinsic_dim(H_nom, labels_nom, threshold=0.95)
        print(f"  名词性角色内在维度(95%): {dim_result['intrinsic_dim']}")
        print(f"  名词性角色全空间CV: {dim_result['full_acc']:.4f}")
        results['nominal_intrinsic_dim'] = dim_result

    # Part B: 名词性 vs 非名词性
    print(f"\n  Part B: 名词性 vs 非名词性角色")
    all_roles = list(ALL_ROLES_DATA.keys())
    H_all, labels_all = collect_hidden_states_multirole(
        model, tokenizer, device, all_roles, ALL_ROLES_DATA)
    print(f"  样本数: {len(H_all)}")

    if len(H_all) >= 20:
        # 名词性(0-4) vs 非名词性(4-6)
        is_nominal = labels_all < len(nominal_roles)
        labels_binary = is_nominal.astype(int)

        scaler = StandardScaler()
        H_scaled = scaler.fit_transform(H_all)
        probe = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv_binary = cross_val_score(probe, H_scaled, labels_binary, cv=5, scoring='accuracy')
        print(f"  名词性 vs 非名词性 分类CV: {cv_binary.mean():.4f}")
        results['nominal_vs_nonnominal_cv'] = float(cv_binary.mean())

        # 所有角色分类
        probe_all = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv_all = cross_val_score(probe_all, H_scaled, labels_all, cv=5, scoring='accuracy')
        print(f"  所有角色分类CV: {cv_all.mean():.4f}")
        results['all_roles_cv'] = float(cv_all.mean())

        # 两两CV
        pairwise_all = compute_pairwise_cv(H_all, labels_all, all_roles)
        print(f"\n  名词性 vs 非名词性 两两CV:")
        for pair, cv in sorted(pairwise_all.items()):
            # 判断是否为名词性-非名词性对
            parts = pair.split("_vs_")
            is_cross = (parts[0] in nominal_roles) != (parts[1] in nominal_roles)
            marker = "★跨类" if is_cross else "  同类"
            print(f"    {marker} {pair}: CV={cv:.4f}")
        results['all_pairwise_cv'] = pairwise_all

    # Part C: 名词性角色超类检测
    print(f"\n  Part C: 名词性角色超类检测")
    if len(H_nom) >= 20:
        # 如果名词性角色的pairwise CV普遍低(< 0.6), 则超类成立
        nom_cvs = [v for k, v in pairwise_cv.items() if not k.startswith('dobj')]
        avg_nom_cv = np.mean(nom_cvs) if nom_cvs else 0

        # dobj也是名词性的, 但它作为宾语可能有不同表示
        dobj_cvs = {k: v for k, v in pairwise_cv.items() if 'dobj' in k}
        other_cvs = {k: v for k, v in pairwise_cv.items() if 'dobj' not in k}

        print(f"  名词性角色间(不含dobj)平均CV: {avg_nom_cv:.4f}")
        print(f"  含dobj的CV: {dobj_cvs}")
        print(f"  不含dobj的CV: {other_cvs}")

        if avg_nom_cv < 0.55:
            print(f"  → 名词性角色超类成立: nsubj/poss/iobj/nmod共享表示!")
        else:
            print(f"  → 名词性角色有区分, 超类假说不完全成立")

        results['superclass_test'] = {
            'avg_nominal_cv_excl_dobj': float(avg_nom_cv),
            'dobj_cvs': dobj_cvs,
            'other_cvs': other_cvs,
            'superclass_holds': avg_nom_cv < 0.55,
        }

    return results


# ===== Exp2: 语法角色子空间正交性 =====
def exp2_subspace_orthogonality(model, tokenizer, device):
    """测量名词/动词/形容词/副词子空间的正交性"""
    print("\n" + "="*70)
    print("Exp2: 语法角色子空间正交性 ★★★★★")
    print("="*70)

    results = {}

    # 收集4类词性的hidden states
    pos_categories = list(SUBSPACE_DATA.keys())
    pos_H = {}
    for pos in pos_categories:
        data = SUBSPACE_DATA[pos]
        H, _ = collect_hidden_states_multirole(
            model, tokenizer, device, [pos], {pos: data}, layer_idx=-1)
        pos_H[pos] = H
        print(f"  {pos}: {len(H)} 样本, dim={H.shape[1] if len(H) > 0 else 'N/A'}")

    # Part A: 4类词性的内在维度
    print(f"\n  Part A: 4类词性各自的内在维度")
    H_all, labels_all = collect_hidden_states_multirole(
        model, tokenizer, device, pos_categories, SUBSPACE_DATA)
    print(f"  总样本数: {len(H_all)}")

    if len(H_all) >= 20:
        dim_result = compute_intrinsic_dim(H_all, labels_all, threshold=0.95)
        print(f"  4类词性内在维度(95%): {dim_result['intrinsic_dim']}")
        print(f"  全空间CV: {dim_result['full_acc']:.4f}")
        results['pos_dim'] = dim_result

        # 两两CV
        pairwise = compute_pairwise_cv(H_all, labels_all, pos_categories)
        print(f"\n  词性两两CV:")
        for pair, cv in sorted(pairwise.items()):
            print(f"    {pair}: CV={cv:.4f}")
        results['pos_pairwise_cv'] = pairwise

    # Part B: 子空间重叠度
    print(f"\n  Part B: 子空间重叠度(Grassmann距离)")
    overlap_results = {}
    for i, pos1 in enumerate(pos_categories):
        for j, pos2 in enumerate(pos_categories):
            if j <= i:
                continue
            if len(pos_H[pos1]) < 8 or len(pos_H[pos2]) < 8:
                continue
            overlap = compute_subspace_overlap(pos_H[pos1], pos_H[pos2], n_components=5)
            if overlap is not None:
                pair_key = f"{pos1}_vs_{pos2}"
                overlap_results[pair_key] = overlap
                print(f"    {pair_key}: overlap={overlap['overlap']:.3f}, "
                      f"mean_angle={overlap['mean_angle']:.1f}°")

    results['subspace_overlap'] = overlap_results

    # Part C: 子空间维度分析
    print(f"\n  Part C: 每个词性的PCA方差解释比")
    for pos in pos_categories:
        H = pos_H[pos]
        if len(H) < 5:
            continue
        max_pca = min(10, min(H.shape[0], H.shape[1]) - 1)
        pca = PCA(n_components=max_pca)
        H_pca = pca.fit_transform(StandardScaler().fit_transform(H))
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        # 90%方差需要多少维?
        dim_90 = np.searchsorted(cumvar, 0.90) + 1
        dim_95 = np.searchsorted(cumvar, 0.95) + 1 if cumvar[-1] >= 0.95 else None
        print(f"    {pos}: 90%方差需{dim_90}维, 95%方差需{dim_95}维, "
              f"前5维累计={cumvar[min(4,len(cumvar)-1)]:.3f}")
        results[f'{pos}_pca_dim'] = {'dim_90': int(dim_90), 'dim_95': dim_95,
                                     'cumvar_top5': float(cumvar[min(4,len(cumvar)-1)])}

    return results


# ===== Exp3: 语法低维性的对照实验 =====
def exp3_syntax_lowdim_control(model, tokenizer, device):
    """语法4类 vs 随机4类 vs 语义4类的内在维度对比"""
    print("\n" + "="*70)
    print("Exp3: 语法低维性的对照实验 ★★★★★")
    print("="*70)

    results = {}

    # Part A: 语法4类的内在维度(已知2-5维)
    print(f"\n  Part A: 语法4类(nsubj/dobj/amod/advmod)")
    syntax_data = SYNTAX_SEMANTIC_BRIDGE_DATA["pure_syntax"]["categories"]
    H_syn, labels_syn = collect_hidden_states_categories(
        model, tokenizer, device, syntax_data)
    print(f"  样本数: {len(H_syn)}")

    if len(H_syn) >= 20:
        dim_syn = compute_intrinsic_dim(H_syn, labels_syn, threshold=0.95)
        print(f"  语法内在维度(95%): {dim_syn['intrinsic_dim']}")
        print(f"  全空间CV: {dim_syn['full_acc']:.4f}")
        results['syntax_dim'] = dim_syn

    # Part B: 随机4类(随机分配标签到语法数据)
    print(f"\n  Part B: 随机4类(打乱语法数据的标签)")
    if len(H_syn) >= 20:
        np.random.seed(42)
        random_labels = np.random.permutation(labels_syn)
        dim_random = compute_intrinsic_dim(H_syn, random_labels, threshold=0.95)
        print(f"  随机标签内在维度(95%): {dim_random['intrinsic_dim']}")
        print(f"  随机标签全空间CV: {dim_random['full_acc']:.4f}")
        print(f"  → 如果随机标签的内在维度也很低: 低维性不是语法特有")
        print(f"  → 如果随机标签的内在维度高: 语法的低维性是特殊性质")
        results['random_dim'] = dim_random

    # Part C: 随机4类(从全部hidden states中随机采样4簇)
    print(f"\n  Part C: 随机4簇(在hidden space中生成4个随机簇)")
    if len(H_syn) >= 20:
        d_model = H_syn.shape[1]
        n_per_cluster = len(H_syn) // 4
        np.random.seed(123)
        # 4个随机中心
        centers = np.random.randn(4, d_model) * 0.1  # 小方差 → 接近
        # 从每个中心生成样本
        H_random_clusters = []
        labels_random_clusters = []
        for ci in range(4):
            cluster = centers[ci] + np.random.randn(n_per_cluster, d_model) * 0.5
            H_random_clusters.append(cluster)
            labels_random_clusters.extend([ci] * n_per_cluster)
        H_random = np.vstack(H_random_clusters)
        labels_random = np.array(labels_random_clusters)

        dim_random_clusters = compute_intrinsic_dim(H_random, labels_random, threshold=0.95)
        print(f"  随机簇内在维度(95%): {dim_random_clusters['intrinsic_dim']}")
        print(f"  随机簇全空间CV: {dim_random_clusters['full_acc']:.4f}")
        results['random_clusters_dim'] = dim_random_clusters

    # Part D: 语义4类的内在维度
    print(f"\n  Part D: 语义4类(情感/反义/上位/下位)")
    sem_data = SYNTAX_SEMANTIC_BRIDGE_DATA["lexical_semantic"]["categories"]
    H_sem, labels_sem = collect_hidden_states_categories(
        model, tokenizer, device, sem_data)
    print(f"  样本数: {len(H_sem)}")

    if len(H_sem) >= 20:
        dim_sem = compute_intrinsic_dim(H_sem, labels_sem, threshold=0.95)
        print(f"  语义内在维度(95%): {dim_sem['intrinsic_dim']}")
        print(f"  全空间CV: {dim_sem['full_acc']:.4f}")
        results['semantic_dim'] = dim_sem

    # Part E: 推理4类的内在维度
    print(f"\n  Part E: 推理4类(因果/条件/递进/转折)")
    reas_data = SYNTAX_SEMANTIC_BRIDGE_DATA["conceptual_semantic"]["categories"]
    H_reas, labels_reas = collect_hidden_states_categories(
        model, tokenizer, device, reas_data)
    print(f"  样本数: {len(H_reas)}")

    if len(H_reas) >= 20:
        dim_reas = compute_intrinsic_dim(H_reas, labels_reas, threshold=0.95)
        print(f"  推理内在维度(95%): {dim_reas['intrinsic_dim']}")
        print(f"  全空间CV: {dim_reas['full_acc']:.4f}")
        results['reasoning_dim'] = dim_reas

    # Part F: 对比总结
    print(f"\n  Part F: 对比总结")
    dims = {}
    for name, key in [('语法4类', 'syntax_dim'), ('随机标签', 'random_dim'),
                       ('随机簇', 'random_clusters_dim'), ('语义4类', 'semantic_dim'),
                       ('推理4类', 'reasoning_dim')]:
        if key in results:
            d = results[key].get('intrinsic_dim', 'N/A')
            cv = results[key].get('full_acc', 'N/A')
            print(f"    {name}: dim={d}, CV={cv if isinstance(cv, str) else f'{cv:.4f}'}")
            dims[name] = d

    results['comparison'] = dims

    # 判断语法的低维性是否特殊
    syn_dim = dims.get('语法4类', None)
    rand_dim = dims.get('随机标签', None)
    if syn_dim is not None and rand_dim is not None:
        if syn_dim <= rand_dim:
            print(f"\n  ★ 语法4类维度({syn_dim}) ≤ 随机标签维度({rand_dim})")
            print(f"    → 语法的低维性不是特殊的! 打乱标签也能达到同样低维!")
            print(f"    → 说明: 4类样本的内在维度低是样本量少的必然结果")
        else:
            print(f"\n  ★ 语法4类维度({syn_dim}) > 随机标签维度({rand_dim})")
            print(f"    → 语法的维度比随机还高, 这不符合预期...")
            print(f"    → 可能原因: 语法角色确实有不同的表示")

    return results


# ===== Exp4: 语法→语义的维度桥梁 =====
def exp4_syntax_semantic_bridge(model, tokenizer, device):
    """从纯语法到纯语义的维度谱, 找到维度跃迁点"""
    print("\n" + "="*70)
    print("Exp4: 语法→语义的维度桥梁 ★★★★★")
    print("="*70)

    results = {}

    levels = [
        ("纯语法", "pure_syntax"),
        ("语法-语义", "syntax_semantic"),
        ("词汇语义", "lexical_semantic"),
        ("概念语义", "conceptual_semantic"),
    ]

    for level_name, level_key in levels:
        print(f"\n  Level: {level_name}")
        data = SYNTAX_SEMANTIC_BRIDGE_DATA[level_key]["categories"]
        H, labels = collect_hidden_states_categories(
            model, tokenizer, device, data)
        print(f"  样本数: {len(H)}")

        if len(H) >= 20:
            dim_result = compute_intrinsic_dim(H, labels, threshold=0.95)
            dim_result_90 = compute_intrinsic_dim(H, labels, threshold=0.90)
            print(f"  内在维度(95%): {dim_result['intrinsic_dim']}")
            print(f"  内在维度(90%): {dim_result_90['intrinsic_dim']}")
            print(f"  全空间CV: {dim_result['full_acc']:.4f}")

            results[level_key] = {
                'dim_95': dim_result['intrinsic_dim'],
                'dim_90': dim_result_90['intrinsic_dim'],
                'full_cv': float(dim_result['full_acc']),
                'n_samples': len(H),
                'n_categories': len(data),
                'dim_curve': dim_result['dim_curve'],
            }
        else:
            print(f"  样本不足, 跳过")

    # 维度桥梁分析
    print(f"\n  ★ 维度桥梁分析:")
    dim_spectrum = []
    for level_name, level_key in levels:
        if level_key in results:
            d95 = results[level_key]['dim_95']
            d90 = results[level_key]['dim_90']
            cv = results[level_key]['full_cv']
            dim_spectrum.append((level_name, d95, d90, cv))
            print(f"    {level_name}: dim_95={d95}, dim_90={d90}, CV={cv:.4f}")

    # 跃迁点检测
    print(f"\n  ★ 维度跃迁点:")
    for i in range(len(dim_spectrum) - 1):
        name1, d95_1, d90_1, cv1 = dim_spectrum[i]
        name2, d95_2, d90_2, cv2 = dim_spectrum[i + 1]
        if d95_1 is not None and d95_2 is not None:
            ratio = d95_2 / max(d95_1, 1)
            print(f"    {name1}→{name2}: {d95_1}→{d95_2} (ratio={ratio:.2f})"
                  f" {'★ 跃迁!' if ratio > 1.5 else ''}")

    results['dim_spectrum'] = [(n, d95, d90, float(cv)) for n, d95, d90, cv in dim_spectrum]

    return results


# ===== 主函数 =====
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                       choices=["qwen3", "glm4", "deepseek7b"])
    parser.add_argument("--exp", type=int, required=True,
                       choices=[1, 2, 3, 4])
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"CCL-S Phase13 语法编码的真正数学结构 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")

    try:
        if args.exp == 1:
            results = exp1_nominal_superclass(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_subspace_orthogonality(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_syntax_lowdim_control(model, tokenizer, device)
        elif args.exp == 4:
            results = exp4_syntax_semantic_bridge(model, tokenizer, device)

        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                               f"ccls_exp{args.exp}_{args.model}_results.json")

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
                return tuple(convert(v) for v in obj)
            return obj

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)
        print(f"\n  结果已保存: {out_path}")

    finally:
        release_model(model)
        print(f"  模型已释放")


if __name__ == "__main__":
    main()
