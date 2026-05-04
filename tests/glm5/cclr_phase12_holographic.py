"""
CCL-R(Phase 12): 冗余编码的信息论结构与维度谱
=============================================================================
核心问题:
  Phase 11发现: 冗余度≈4.9(5个频段), 语法→推理维度跃迁比3-5
  但:
  1. 冗余度≈5是否因为恰好分了5个频段? 如果分10/20个频段, 冗余度是否≈10/20?
     → 验证"全息编码"假说: 每个频段都包含完整信息
  2. nsubj-poss重合是数据artifact(相同target_words), 控制词汇后是否仍然重合?
  3. 语法→语义→推理→创造性, 内在维度是否形成某种级联/维度谱?
  4. 全息编码是W_U特有的还是随机投影也有?

实验:
  Exp1: ★★★★★ 冗余度 vs 频段数 (2/3/5/7/10/15/20)
    → 关键假说: 冗余度 ≈ 频段数? (全息编码)
    → 如果是: 每个频段都包含100%信息, 冗余度=n_bands
    → 如果不是: 存在最优频段数, 语法信息在某些频段更集中

  Exp2: ★★★★★ 控制词汇的语法几何
    → 对同一角色用完全不同的词汇
    → 测试: nsubj-poss是否仍然重合?
    → 如果不重合: 之前的重合是词汇artifact
    → 如果仍然重合: 语法角色有固定的几何位置!

  Exp3: ★★★★★ 维度谱: 语法→语义→推理→创造性
    → 语法(4角色): 已知2-5维
    → 语义(情感/反义词/上位词/下位词): ?维
    → 推理(因果/条件/递进/转折): 已知10-15维
    → 创造性(比喻/反讽/双关/隐喻): ?维
    → 维度谱是否形成级联? 维度跃迁是否有固定比例?

  Exp4: ★★★★★ 全息编码 vs 随机投影
    → 用随机正交基代替W_U特征向量
    → 测试: 随机投影的冗余度是否也≈n_bands?
    → 如果是: 全息编码不是W_U特有, 而是高维空间的一般性质
    → 如果不是: W_U特征空间有特殊的全息结构
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


# ===== 数据集: 复用Phase 11的4角色数据 =====
SYNTAX_ROLES_DATA = {
    "nsubj": {
        "desc": "主语名词",
        "sentences": [
            "The king ruled the kingdom", "The doctor treated the patient",
            "The artist painted the portrait", "The soldier defended the castle",
            "The cat sat on the mat", "The dog ran through the park",
            "The woman drove the car", "The man fixed the roof",
            "The student read the textbook", "The teacher explained the lesson",
            "The president signed the bill", "The chef cooked the meal",
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "woman", "man", "student", "teacher", "president", "chef",
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
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "man", "woman", "student", "teacher", "president", "chef",
        ],
    },
    "amod": {
        "desc": "形容词修饰语",
        "sentences": [
            "The brave king fought hard", "The kind doctor helped many",
            "The creative artist worked well", "The strong soldier marched far",
            "The beautiful cat sat quietly", "The large dog ran swiftly",
            "The old woman walked slowly", "The tall man stood quietly",
            "The bright student read carefully", "The wise teacher explained clearly",
            "The powerful president decided firmly", "The skilled chef cooked perfectly",
        ],
        "target_words": [
            "brave", "kind", "creative", "strong", "beautiful", "large",
            "old", "tall", "bright", "wise", "powerful", "skilled",
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
        ],
        "target_words": [
            "wisely", "carefully", "beautifully", "bravely", "quickly", "loudly",
            "slowly", "quietly", "carefully", "clearly", "firmly", "quickly",
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
        ],
        "target_words": [
            "king", "doctor", "artist", "soldier", "cat", "dog",
            "woman", "man", "student", "teacher", "president", "chef",
        ],
    },
}

# ===== Exp2: 控制词汇的语法几何 =====
# 核心设计: 同一角色用完全不同的名词集合
# nsubj角色: 用不同的名词(人/动物/物体/抽象概念)
# poss角色: 用相同的名词, 但在所有格位置
# 如果nsubj-poss仍然重合: 说明重合是词汇artifact
# 如果不重合: 说明语法角色确实有不同的几何位置

CONTROLLED_VOCAB_DATA = {
    "nsubj_concrete": {
        "desc": "主语-具体名词(人/动物)",
        "sentences": [
            "The baker baked fresh bread", "The pilot flew the airplane",
            "The singer sang a song", "The farmer grew crops",
            "The hunter tracked the deer", "The fisher caught fish",
            "The builder constructed walls", "The driver steered the truck",
            "The nurse cared for patients", "The clerk filed documents",
            "The guard watched the gate", "The monk prayed silently",
        ],
        "target_words": [
            "baker", "pilot", "singer", "farmer", "hunter", "fisher",
            "builder", "driver", "nurse", "clerk", "guard", "monk",
        ],
    },
    "nsubj_abstract": {
        "desc": "主语-抽象名词(概念)",
        "sentences": [
            "Justice prevailed in the end", "Truth emerged from the investigation",
            "Love conquered all obstacles", "Fear gripped the village",
            "Hope inspired the people", "Time heals all wounds",
            "Knowledge expanded through study", "Freedom required sacrifice",
            "Courage overcame the challenge", "Wisdom guided the decision",
            "Chaos followed the collapse", "Peace settled over the land",
        ],
        "target_words": [
            "Justice", "Truth", "Love", "Fear", "Hope", "Time",
            "Knowledge", "Freedom", "Courage", "Wisdom", "Chaos", "Peace",
        ],
    },
    "nsubj_objects": {
        "desc": "主语-无生命物体",
        "sentences": [
            "The rock rolled down the hill", "The river flowed through the valley",
            "The tree grew tall and strong", "The wind blew across the plain",
            "The fire burned through the night", "The ice melted in the sun",
            "The cloud drifted over the mountain", "The star shone in the sky",
            "The building towered over the city", "The road led to the village",
            "The bridge crossed the canyon", "The lamp illuminated the room",
        ],
        "target_words": [
            "rock", "river", "tree", "wind", "fire", "ice",
            "cloud", "star", "building", "road", "bridge", "lamp",
        ],
    },
    "poss_concrete": {
        "desc": "所有格-与nsubj_concrete相同的名词",
        "sentences": [
            "The baker's recipe was famous", "The pilot's license was valid",
            "The singer's voice was beautiful", "The farmer's land was fertile",
            "The hunter's dog was loyal", "The fisher's boat was sturdy",
            "The builder's tools were sharp", "The driver's route was long",
            "The nurse's uniform was clean", "The clerk's desk was organized",
            "The guard's post was secure", "The monk's robe was simple",
        ],
        "target_words": [
            "baker", "pilot", "singer", "farmer", "hunter", "fisher",
            "builder", "driver", "nurse", "clerk", "guard", "monk",
        ],
    },
    "poss_abstract": {
        "desc": "所有格-与nsubj_abstract相同的名词",
        "sentences": [
            "Justice's scale was balanced", "Truth's power was undeniable",
            "Love's strength was infinite", "Fear's grip was tight",
            "Hope's light was bright", "Time's passage was inevitable",
            "Knowledge's value was clear", "Freedom's price was high",
            "Courage's reward was great", "Wisdom's source was deep",
            "Chaos's end was certain", "Peace's foundation was strong",
        ],
        "target_words": [
            "Justice", "Truth", "Love", "Fear", "Hope", "Time",
            "Knowledge", "Freedom", "Courage", "Wisdom", "Chaos", "Peace",
        ],
    },
    "dobj_concrete": {
        "desc": "直接宾语-与nsubj_concrete相同的名词",
        "sentences": [
            "They hired the baker recently", "She admired the pilot greatly",
            "We praised the singer loudly", "He visited the farmer often",
            "They followed the hunter closely", "She watched the fisher carefully",
            "We trusted the builder completely", "He rewarded the driver generously",
            "They thanked the nurse warmly", "She noticed the clerk immediately",
            "We respected the guard deeply", "He remembered the monk fondly",
        ],
        "target_words": [
            "baker", "pilot", "singer", "farmer", "hunter", "fisher",
            "builder", "driver", "nurse", "clerk", "guard", "monk",
        ],
    },
}

# ===== Exp3: 维度谱数据 =====
SEMANTIC_DATA = {
    "emotion": {
        "desc": "情感语义",
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
        "desc": "反义关系",
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
        "desc": "上位词关系(具体→抽象)",
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
        "desc": "下位词关系(抽象→具体)",
        "sentences": [
            "An animal could be a dog", "A flower might be a rose",
            "A tree could be an oak", "A vehicle might be a car",
            "A tool could be a knife", "Clothing might include shirts",
            "A grain could be rice", "A metal might be gold",
            "A gem could be a diamond", "An instrument might be a piano",
            "Furniture could include chairs", "A publication might be a book",
        ],
    },
}

REASONING_DATA = {
    "causal": {
        "desc": "因果推理",
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
        "desc": "条件推理",
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
        "desc": "递进推理",
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
        "desc": "转折推理",
        "sentences": [
            "He tried hard but still failed", "She studied hard yet barely passed",
            "They planned carefully but things went wrong", "He was tired yet kept working",
            "She was afraid but faced the challenge", "They were outnumbered yet won the battle",
            "He was ill but attended the meeting", "She was young yet very wise",
            "They were poor but remained happy", "He was wrong but refused to admit",
            "She was hurt but kept smiling", "They were late but finished first",
        ],
    },
}

CREATIVITY_DATA = {
    "metaphor": {
        "desc": "比喻(概念映射)",
        "sentences": [
            "Time is a river flowing onward", "Love is a fire burning bright",
            "The mind is a garden growing ideas", "Life is a journey full of paths",
            "Knowledge is light illuminating darkness", "Hope is an anchor holding firm",
            "Fear is a shadow following closely", "Memory is a mirror reflecting the past",
            "Words are weapons cutting deep", "Silence is golden beyond measure",
            "The world is a stage for all", "History is a teacher guiding us",
        ],
    },
    "irony": {
        "desc": "反讽(字面与意图相反)",
        "sentences": [
            "What wonderful weather we are having in the storm",
            "Oh great another meeting that could have been an email",
            "How lovely to be stuck in traffic for hours",
            "Fantastic the project deadline moved up again",
            "Wonderful the system crashed right before saving",
            "Perfect timing the bus just left without us",
            "Great the phone died right when it rang",
            "How splendid to work on the holiday weekend",
            "Lovely the rain started just as we left",
            "Excellent the presentation deleted itself somehow",
            "Amazing the alarm failed on the important day",
            "Brilliant the recipe forgot the main ingredient",
        ],
    },
    "pun": {
        "desc": "双关(一词多义)",
        "sentences": [
            "The bicycle fell over because it was two tired",
            "Time flies like an arrow but fruit flies like a banana",
            "The man who invented the door knocker got a No-bell prize",
            "She had a photographic memory but never developed it",
            "The cross-eyed teacher could not control his pupils",
            "I used to be a baker but I lost interest in the dough",
            "The short fortune teller escaped from prison as a small medium at large",
            "A boiled egg in the morning is hard to beat",
            "The guy who fell onto the upholstery machine recovered fully",
            "I am reading a book about anti-gravity and cannot put it down",
            "The police station got robbed and the detectives are investigating",
            "The calendar thief got twelve months for his crime",
        ],
    },
    "metonymy": {
        "desc": "借代(以部分代整体)",
        "sentences": [
            "The White House announced new policies today", "Wall Street reacted to the news",
            "Hollywood released several blockbusters this year", "The crown ordered new taxes",
            "The pen is mightier than the sword", "The bench delivered a landmark ruling",
            "The press covered the story extensively", "The sword conquered the territory",
            "The stage prepared for opening night", "The badge arrested the suspect",
            "The throne commanded absolute obedience", "The stethoscope examined the patient",
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
        for sent in cat_data["sentences"]:
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


def compute_W_U_eigenspectrum(W_U):
    """计算W_U的特征值谱和特征向量"""
    d_model = W_U.shape[1]
    W_U_f64 = W_U.astype(np.float64)
    WtW = W_U_f64.T @ W_U_f64
    eigenvalues, eigenvectors = np.linalg.eigh(WtW)
    eigenvalues = eigenvalues[::-1]
    eigenvectors = eigenvectors[:, ::-1]
    return eigenvalues, eigenvectors


def compute_intrinsic_dim(H, labels, threshold=0.95):
    """计算内在维度(达到全空间准确率threshold%所需的最小PCA维度)"""
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)

    # 全空间准确率
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


# ===== Exp1: 冗余度 vs 频段数 =====
def exp1_redundancy_vs_bands(model, tokenizer, device):
    """测量不同频段数下的冗余度, 验证全息编码假说"""
    print("\n" + "="*70)
    print("Exp1: 冗余度 vs 频段数 ★★★★★")
    print("="*70)

    # 收集4角色hidden states
    syntax_roles = ["nsubj", "dobj", "amod", "advmod"]
    H, labels = collect_hidden_states_multirole(
        model, tokenizer, device, syntax_roles, SYNTAX_ROLES_DATA)
    print(f"  收集到 {len(H)} 个样本, {len(syntax_roles)} 角色")

    n_samples = len(H)
    n_roles = len(syntax_roles)
    d_model = H.shape[1]

    # W_U特征谱
    W_U = get_W_U(model)
    eigenvalues, eigenvectors = compute_W_U_eigenspectrum(W_U)

    # Part A: 全空间互信息
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)
    probe_full = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    probe_full.fit(H_scaled, labels)

    probs_full = probe_full.predict_proba(H_scaled)
    probs_full = np.clip(probs_full, 1e-10, 1.0)
    probs_full /= probs_full.sum(axis=1, keepdims=True)
    H_role_given_full = -np.mean(np.sum(probs_full * np.log2(probs_full), axis=1))
    I_full = np.log2(n_roles) - H_role_given_full

    print(f"\n  I(H_full; role) = {I_full:.3f} bits")
    print(f"  H(role) = {np.log2(n_roles):.3f} bits")

    # Part B: 不同频段数的冗余度
    n_bands_list = [2, 3, 5, 7, 10, 15, 20]
    results_by_bands = {}

    print(f"\n  Part B: 不同频段数的冗余度")
    print(f"  {'n_bands':>8s} {'Redundancy':>12s} {'I/full':>8s} {'Avg Band MI':>12s} {'Avg Band %':>12s}")

    for n_bands in n_bands_list:
        band_size = d_model // n_bands

        band_mi_list = []
        for bi in range(n_bands):
            start = bi * band_size
            end = min((bi + 1) * band_size, d_model)
            if end <= start:
                continue
            U_band = eigenvectors[:, start:end]
            H_band = H @ U_band

            scaler_b = StandardScaler()
            H_band_scaled = scaler_b.fit_transform(H_band)
            probe_b = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            probe_b.fit(H_band_scaled, labels)

            probs_b = probe_b.predict_proba(H_band_scaled)
            probs_b = np.clip(probs_b, 1e-10, 1.0)
            probs_b /= probs_b.sum(axis=1, keepdims=True)
            H_cond_b = -np.mean(np.sum(probs_b * np.log2(probs_b), axis=1))
            I_band = np.log2(n_roles) - H_cond_b
            band_mi_list.append(float(I_band))

        sum_band_mi = sum(band_mi_list)
        redundancy = sum_band_mi / max(I_full, 0.001)
        avg_band_mi = np.mean(band_mi_list)
        avg_band_pct = avg_band_mi / max(I_full, 0.001) * 100

        results_by_bands[n_bands] = {
            'band_mi_list': band_mi_list,
            'sum_band_mi': float(sum_band_mi),
            'redundancy': float(redundancy),
            'avg_band_mi': float(avg_band_mi),
            'avg_band_pct': float(avg_band_pct),
        }

        print(f"  {n_bands:8d} {redundancy:12.2f} {sum_band_mi:8.3f} {avg_band_mi:12.3f} {avg_band_pct:11.1f}%")

    # Part C: 分析趋势
    redundancies = [results_by_bands[nb]['redundancy'] for nb in n_bands_list]
    avg_pcts = [results_by_bands[nb]['avg_band_pct'] for nb in n_bands_list]

    print(f"\n  Part C: 趋势分析")
    print(f"  冗余度范围: {min(redundancies):.2f} - {max(redundancies):.2f}")
    print(f"  平均频段MI%: {avg_pcts[-1]:.1f}% (20 bands) - {avg_pcts[0]:.1f}% (2 bands)")

    # 全息编码判定
    # 如果冗余度≈频段数: 每个频段都包含完整信息 → 全息编码
    # 如果冗余度<频段数: 信息在频段间分散 → 非全息
    print(f"\n  ★ 全息编码判定:")
    for nb in n_bands_list:
        r = results_by_bands[nb]['redundancy']
        ratio = r / nb
        print(f"    n_bands={nb:2d}: redundancy={r:.2f}, redundancy/n_bands={ratio:.3f}"
              f" {'→ 全息!' if ratio > 0.8 else '→ 非全息'}")

    # Part D: 使用PCA代替W_U特征空间的频段分析
    print(f"\n  Part D: PCA空间的频段分析(对照)")
    max_pca = min(20, min(H.shape[0], H.shape[1]) - 1)
    pca = PCA(n_components=max_pca)
    H_pca = pca.fit_transform(H)

    for n_bands in [5, 10]:
        band_size = max_pca // n_bands
        band_mi_list = []
        for bi in range(n_bands):
            start = bi * band_size
            end = min((bi + 1) * band_size, max_pca)
            if end <= start:
                continue
            H_band = H_pca[:, start:end]

            scaler_b = StandardScaler()
            H_band_scaled = scaler_b.fit_transform(H_band)
            probe_b = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            probe_b.fit(H_band_scaled, labels)

            probs_b = probe_b.predict_proba(H_band_scaled)
            probs_b = np.clip(probs_b, 1e-10, 1.0)
            probs_b /= probs_b.sum(axis=1, keepdims=True)
            H_cond_b = -np.mean(np.sum(probs_b * np.log2(probs_b), axis=1))
            I_band = np.log2(n_roles) - H_cond_b
            band_mi_list.append(float(I_band))

        sum_mi = sum(band_mi_list)
        red = sum_mi / max(I_full, 0.001)
        print(f"    PCA bands={n_bands}: redundancy={red:.2f}, avg_pct={np.mean(band_mi_list)/max(I_full,0.001)*100:.1f}%")

    results = {
        'I_full': float(I_full),
        'H_role': float(np.log2(n_roles)),
        'n_samples': n_samples,
        'results_by_bands': results_by_bands,
    }

    return results


# ===== Exp2: 控制词汇的语法几何 =====
def exp2_controlled_vocab_geometry(model, tokenizer, device):
    """控制词汇变量, 测试语法角色的几何结构"""
    print("\n" + "="*70)
    print("Exp2: 控制词汇的语法几何 ★★★★★")
    print("="*70)

    results = {}

    # Test 1: nsubj_concrete vs poss_concrete (相同名词, 不同角色)
    print(f"\n  Test 1: nsubj(具体) vs poss(具体) — 相同名词, 不同角色")
    role_names_1 = ["nsubj_concrete", "poss_concrete"]
    H1, labels1 = collect_hidden_states_multirole(
        model, tokenizer, device, role_names_1, CONTROLLED_VOCAB_DATA)
    print(f"  样本数: {len(H1)}")

    if len(H1) >= 8:
        # PCA降维
        max_dim = min(5, min(H1.shape[0], H1.shape[1]) - 1)
        pca1 = PCA(n_components=max_dim)
        H1_pca = pca1.fit_transform(H1)

        # 质心
        centers1 = {}
        for ri, role in enumerate(role_names_1):
            mask = labels1 == ri
            centers1[role] = H1_pca[mask].mean(axis=0)

        d_nsubj_poss = np.linalg.norm(centers1["nsubj_concrete"] - centers1["poss_concrete"])
        print(f"  d(nsubj_concrete, poss_concrete) = {d_nsubj_poss:.3f}")
        print(f"  PCA解释方差: {pca1.explained_variance_ratio_[:3]}")

        # 分类准确率
        scaler1 = StandardScaler()
        H1_scaled = scaler1.fit_transform(H1)
        probe1 = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv1 = cross_val_score(probe1, H1_scaled, labels1, cv=5, scoring='accuracy')
        print(f"  nsubj vs poss 分类CV: {cv1.mean():.4f}")
        print(f"  {'角色可区分!' if cv1.mean() > 0.6 else '角色不可区分(artifact!)'}")

        results['test1'] = {
            'd_nsubj_poss': float(d_nsubj_poss),
            'pca_explained_var': pca1.explained_variance_ratio_.tolist(),
            'classification_cv': float(cv1.mean()),
            'centers': {r: centers1[r].tolist() for r in role_names_1},
        }

    # Test 2: nsubj_concrete vs dobj_concrete (相同名词, 不同角色)
    print(f"\n  Test 2: nsubj(具体) vs dobj(具体) — 相同名词, 不同角色")
    role_names_2 = ["nsubj_concrete", "dobj_concrete"]
    H2, labels2 = collect_hidden_states_multirole(
        model, tokenizer, device, role_names_2, CONTROLLED_VOCAB_DATA)
    print(f"  样本数: {len(H2)}")

    if len(H2) >= 8:
        max_dim = min(5, min(H2.shape[0], H2.shape[1]) - 1)
        pca2 = PCA(n_components=max_dim)
        H2_pca = pca2.fit_transform(H2)

        centers2 = {}
        for ri, role in enumerate(role_names_2):
            mask = labels2 == ri
            centers2[role] = H2_pca[mask].mean(axis=0)

        d_nsubj_dobj = np.linalg.norm(centers2["nsubj_concrete"] - centers2["dobj_concrete"])
        print(f"  d(nsubj_concrete, dobj_concrete) = {d_nsubj_dobj:.3f}")

        scaler2 = StandardScaler()
        H2_scaled = scaler2.fit_transform(H2)
        probe2 = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv2 = cross_val_score(probe2, H2_scaled, labels2, cv=5, scoring='accuracy')
        print(f"  nsubj vs dobj 分类CV: {cv2.mean():.4f}")

        results['test2'] = {
            'd_nsubj_dobj': float(d_nsubj_dobj),
            'classification_cv': float(cv2.mean()),
        }

    # Test 3: nsubj_concrete vs nsubj_abstract vs nsubj_objects (不同名词, 相同角色)
    print(f"\n  Test 3: nsubj(具体) vs nsubj(抽象) vs nsubj(物体) — 不同名词, 相同角色")
    role_names_3 = ["nsubj_concrete", "nsubj_abstract", "nsubj_objects"]
    H3, labels3 = collect_hidden_states_multirole(
        model, tokenizer, device, role_names_3, CONTROLLED_VOCAB_DATA)
    print(f"  样本数: {len(H3)}")

    if len(H3) >= 10:
        max_dim = min(5, min(H3.shape[0], H3.shape[1]) - 1)
        pca3 = PCA(n_components=max_dim)
        H3_pca = pca3.fit_transform(H3)

        centers3 = {}
        for ri, role in enumerate(role_names_3):
            mask = labels3 == ri
            centers3[role] = H3_pca[mask].mean(axis=0)

        for i, r1 in enumerate(role_names_3):
            for j, r2 in enumerate(role_names_3):
                if j > i:
                    d = np.linalg.norm(centers3[r1] - centers3[r2])
                    print(f"  d({r1}, {r2}) = {d:.3f}")

        # 分类准确率(相同角色不同词汇能否区分?)
        scaler3 = StandardScaler()
        H3_scaled = scaler3.fit_transform(H3)
        probe3 = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv3 = cross_val_score(probe3, H3_scaled, labels3, cv=5, scoring='accuracy')
        print(f"  词汇类型分类CV: {cv3.mean():.4f}")
        print(f"  {'词汇语义有区分度' if cv3.mean() > 0.5 else '词汇语义区分度低'}")

        results['test3'] = {
            'cv_vocab_type': float(cv3.mean()),
            'centers': {r: centers3[r].tolist() for r in role_names_3},
        }

    # Test 4: nsubj_abstract vs poss_abstract (相同抽象名词, 不同角色)
    print(f"\n  Test 4: nsubj(抽象) vs poss(抽象) — 相同名词, 不同角色")
    role_names_4 = ["nsubj_abstract", "poss_abstract"]
    H4, labels4 = collect_hidden_states_multirole(
        model, tokenizer, device, role_names_4, CONTROLLED_VOCAB_DATA)
    print(f"  样本数: {len(H4)}")

    if len(H4) >= 8:
        max_dim = min(5, min(H4.shape[0], H4.shape[1]) - 1)
        pca4 = PCA(n_components=max_dim)
        H4_pca = pca4.fit_transform(H4)

        centers4 = {}
        for ri, role in enumerate(role_names_4):
            mask = labels4 == ri
            centers4[role] = H4_pca[mask].mean(axis=0)

        d_nsubj_poss_abs = np.linalg.norm(centers4["nsubj_abstract"] - centers4["poss_abstract"])
        print(f"  d(nsubj_abstract, poss_abstract) = {d_nsubj_poss_abs:.3f}")

        scaler4 = StandardScaler()
        H4_scaled = scaler4.fit_transform(H4)
        probe4 = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
        cv4 = cross_val_score(probe4, H4_scaled, labels4, cv=5, scoring='accuracy')
        print(f"  nsubj vs poss (抽象词) 分类CV: {cv4.mean():.4f}")
        print(f"  {'角色可区分!' if cv4.mean() > 0.6 else '角色不可区分(artifact!)'}")

        results['test4'] = {
            'd_nsubj_poss_abstract': float(d_nsubj_poss_abs),
            'classification_cv': float(cv4.mean()),
        }

    # 总结
    print(f"\n  ★ 控制词汇实验总结:")
    if 'test1' in results and 'test4' in results:
        cv_concrete = results['test1']['classification_cv']
        cv_abstract = results['test4']['classification_cv']
        d_concrete = results['test1']['d_nsubj_poss']
        d_abstract = results['test4']['d_nsubj_poss_abstract']
        print(f"    nsubj vs poss (具体词): CV={cv_concrete:.4f}, d={d_concrete:.3f}")
        print(f"    nsubj vs poss (抽象词): CV={cv_abstract:.4f}, d={d_abstract:.3f}")
        if cv_concrete > 0.6 or cv_abstract > 0.6:
            print(f"    → nsubj和poss在控制词汇后可以区分, 之前重合是artifact!")
        else:
            print(f"    → nsubj和poss即使控制词汇也无法区分, 可能真有共享结构!")

    return results


# ===== Exp3: 维度谱 =====
def exp3_dimension_spectrum(model, tokenizer, device):
    """测量语法→语义→推理→创造性的内在维度谱"""
    print("\n" + "="*70)
    print("Exp3: 维度谱 — 语法→语义→推理→创造性 ★★★★★")
    print("="*70)

    results = {}

    # Level 1: 语法(4角色)
    print(f"\n  Level 1: 语法(4角色)")
    syntax_roles = ["nsubj", "dobj", "amod", "advmod"]
    H_syn, labels_syn = collect_hidden_states_multirole(
        model, tokenizer, device, syntax_roles, SYNTAX_ROLES_DATA)
    print(f"  样本数: {len(H_syn)}")

    if len(H_syn) >= 10:
        syn_result = compute_intrinsic_dim(H_syn, labels_syn, threshold=0.95)
        syn_result_90 = compute_intrinsic_dim(H_syn, labels_syn, threshold=0.90)
        print(f"  语法内在维度(95%): {syn_result['intrinsic_dim']}")
        print(f"  语法内在维度(90%): {syn_result_90['intrinsic_dim']}")
        print(f"  全空间CV: {syn_result['full_acc']:.4f}")
        results['syntax'] = {
            'dim_95': syn_result['intrinsic_dim'],
            'dim_90': syn_result_90['intrinsic_dim'],
            'full_cv': float(syn_result['full_acc']),
            'n_samples': len(H_syn),
            'n_categories': len(syntax_roles),
            'dim_curve': syn_result['dim_curve'],
        }

    # Level 2: 语义(4类)
    print(f"\n  Level 2: 语义(4类: 情感/反义/上位/下位)")
    H_sem, labels_sem = collect_hidden_states_categories(
        model, tokenizer, device, SEMANTIC_DATA)
    print(f"  样本数: {len(H_sem)}")

    if len(H_sem) >= 10:
        sem_result = compute_intrinsic_dim(H_sem, labels_sem, threshold=0.95)
        sem_result_90 = compute_intrinsic_dim(H_sem, labels_sem, threshold=0.90)
        print(f"  语义内在维度(95%): {sem_result['intrinsic_dim']}")
        print(f"  语义内在维度(90%): {sem_result_90['intrinsic_dim']}")
        print(f"  全空间CV: {sem_result['full_acc']:.4f}")
        results['semantic'] = {
            'dim_95': sem_result['intrinsic_dim'],
            'dim_90': sem_result_90['intrinsic_dim'],
            'full_cv': float(sem_result['full_acc']),
            'n_samples': len(H_sem),
            'n_categories': len(SEMANTIC_DATA),
            'dim_curve': sem_result['dim_curve'],
        }

    # Level 3: 推理(4类)
    print(f"\n  Level 3: 推理(4类: 因果/条件/递进/转折)")
    H_reas, labels_reas = collect_hidden_states_categories(
        model, tokenizer, device, REASONING_DATA)
    print(f"  样本数: {len(H_reas)}")

    if len(H_reas) >= 10:
        reas_result = compute_intrinsic_dim(H_reas, labels_reas, threshold=0.95)
        reas_result_90 = compute_intrinsic_dim(H_reas, labels_reas, threshold=0.90)
        print(f"  推理内在维度(95%): {reas_result['intrinsic_dim']}")
        print(f"  推理内在维度(90%): {reas_result_90['intrinsic_dim']}")
        print(f"  全空间CV: {reas_result['full_acc']:.4f}")
        results['reasoning'] = {
            'dim_95': reas_result['intrinsic_dim'],
            'dim_90': reas_result_90['intrinsic_dim'],
            'full_cv': float(reas_result['full_acc']),
            'n_samples': len(H_reas),
            'n_categories': len(REASONING_DATA),
            'dim_curve': reas_result['dim_curve'],
        }

    # Level 4: 创造性(4类)
    print(f"\n  Level 4: 创造性(4类: 比喻/反讽/双关/借代)")
    H_crea, labels_crea = collect_hidden_states_categories(
        model, tokenizer, device, CREATIVITY_DATA)
    print(f"  样本数: {len(H_crea)}")

    if len(H_crea) >= 10:
        crea_result = compute_intrinsic_dim(H_crea, labels_crea, threshold=0.95)
        crea_result_90 = compute_intrinsic_dim(H_crea, labels_crea, threshold=0.90)
        print(f"  创造性内在维度(95%): {crea_result['intrinsic_dim']}")
        print(f"  创造性内在维度(90%): {crea_result_90['intrinsic_dim']}")
        print(f"  全空间CV: {crea_result['full_acc']:.4f}")
        results['creativity'] = {
            'dim_95': crea_result['intrinsic_dim'],
            'dim_90': crea_result_90['intrinsic_dim'],
            'full_cv': float(crea_result['full_acc']),
            'n_samples': len(H_crea),
            'n_categories': len(CREATIVITY_DATA),
            'dim_curve': crea_result['dim_curve'],
        }

    # 维度谱总结
    print(f"\n  ★ 维度谱总结:")
    dim_spectrum = []
    for level_name in ['syntax', 'semantic', 'reasoning', 'creativity']:
        if level_name in results:
            dim_95 = results[level_name].get('dim_95', 'N/A')
            dim_90 = results[level_name].get('dim_90', 'N/A')
            print(f"    {level_name:>12s}: dim_95={dim_95}, dim_90={dim_90}")
            if dim_95 is not None:
                dim_spectrum.append(dim_95)

    # 维度跃迁比
    print(f"\n  ★ 维度跃迁比:")
    for i in range(len(dim_spectrum) - 1):
        ratio = dim_spectrum[i+1] / max(dim_spectrum[i], 1)
        print(f"    {['语法','语义','推理','创造性'][i]}→{['语法','语义','推理','创造性'][i+1]}: {ratio:.2f}")

    return results


# ===== Exp4: 全息编码 vs 随机投影 =====
def exp4_holographic_vs_random(model, tokenizer, device):
    """比较W_U特征投影和随机投影的冗余度"""
    print("\n" + "="*70)
    print("Exp4: 全息编码 vs 随机投影 ★★★★★")
    print("="*70)

    # 收集4角色hidden states
    syntax_roles = ["nsubj", "dobj", "amod", "advmod"]
    H, labels = collect_hidden_states_multirole(
        model, tokenizer, device, syntax_roles, SYNTAX_ROLES_DATA)
    print(f"  收集到 {len(H)} 个样本")

    n_roles = len(syntax_roles)
    d_model = H.shape[1]

    # 全空间互信息
    scaler = StandardScaler()
    H_scaled = scaler.fit_transform(H)
    probe_full = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
    probe_full.fit(H_scaled, labels)

    probs_full = probe_full.predict_proba(H_scaled)
    probs_full = np.clip(probs_full, 1e-10, 1.0)
    probs_full /= probs_full.sum(axis=1, keepdims=True)
    H_role_given_full = -np.mean(np.sum(probs_full * np.log2(probs_full), axis=1))
    I_full = np.log2(n_roles) - H_role_given_full

    print(f"  I(H_full; role) = {I_full:.3f} bits")

    # Part A: W_U特征投影的冗余度(5和10频段)
    W_U = get_W_U(model)
    eigenvalues, eigenvectors = compute_W_U_eigenspectrum(W_U)

    print(f"\n  Part A: W_U特征投影")
    wu_results = {}
    for n_bands in [5, 10]:
        band_size = d_model // n_bands
        band_mi_list = []
        for bi in range(n_bands):
            start = bi * band_size
            end = min((bi + 1) * band_size, d_model)
            if end <= start:
                continue
            U_band = eigenvectors[:, start:end]
            H_band = H @ U_band

            scaler_b = StandardScaler()
            H_band_scaled = scaler_b.fit_transform(H_band)
            probe_b = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            probe_b.fit(H_band_scaled, labels)

            probs_b = probe_b.predict_proba(H_band_scaled)
            probs_b = np.clip(probs_b, 1e-10, 1.0)
            probs_b /= probs_b.sum(axis=1, keepdims=True)
            H_cond_b = -np.mean(np.sum(probs_b * np.log2(probs_b), axis=1))
            I_band = np.log2(n_roles) - H_cond_b
            band_mi_list.append(float(I_band))

        sum_mi = sum(band_mi_list)
        redundancy = sum_mi / max(I_full, 0.001)
        avg_pct = np.mean(band_mi_list) / max(I_full, 0.001) * 100
        wu_results[n_bands] = {
            'redundancy': float(redundancy),
            'avg_band_pct': float(avg_pct),
        }
        print(f"    W_U bands={n_bands}: redundancy={redundancy:.2f}, avg_band_pct={avg_pct:.1f}%")

    # Part B: 随机正交投影的冗余度(5和10频段)
    print(f"\n  Part B: 随机正交投影")
    # 生成随机正交矩阵
    np.random.seed(42)
    Q, _ = np.linalg.qr(np.random.randn(d_model, d_model))

    random_results = {}
    for n_bands in [5, 10]:
        band_size = d_model // n_bands
        band_mi_list = []
        for bi in range(n_bands):
            start = bi * band_size
            end = min((bi + 1) * band_size, d_model)
            if end <= start:
                continue
            Q_band = Q[:, start:end]
            H_band = H @ Q_band

            scaler_b = StandardScaler()
            H_band_scaled = scaler_b.fit_transform(H_band)
            probe_b = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            probe_b.fit(H_band_scaled, labels)

            probs_b = probe_b.predict_proba(H_band_scaled)
            probs_b = np.clip(probs_b, 1e-10, 1.0)
            probs_b /= probs_b.sum(axis=1, keepdims=True)
            H_cond_b = -np.mean(np.sum(probs_b * np.log2(probs_b), axis=1))
            I_band = np.log2(n_roles) - H_cond_b
            band_mi_list.append(float(I_band))

        sum_mi = sum(band_mi_list)
        redundancy = sum_mi / max(I_full, 0.001)
        avg_pct = np.mean(band_mi_list) / max(I_full, 0.001) * 100
        random_results[n_bands] = {
            'redundancy': float(redundancy),
            'avg_band_pct': float(avg_pct),
        }
        print(f"    Random bands={n_bands}: redundancy={redundancy:.2f}, avg_band_pct={avg_pct:.1f}%")

    # Part C: PCA投影的冗余度(前50维)
    print(f"\n  Part C: PCA投影(前50维)")
    max_pca = min(50, min(H.shape[0], H.shape[1]) - 1)
    pca = PCA(n_components=max_pca)
    H_pca = pca.fit_transform(H)

    pca_results = {}
    for n_bands in [5, 10]:
        band_size = max_pca // n_bands
        band_mi_list = []
        for bi in range(n_bands):
            start = bi * band_size
            end = min((bi + 1) * band_size, max_pca)
            if end <= start:
                continue
            H_band = H_pca[:, start:end]

            scaler_b = StandardScaler()
            H_band_scaled = scaler_b.fit_transform(H_band)
            probe_b = LogisticRegression(solver='lbfgs', max_iter=2000, C=1.0)
            probe_b.fit(H_band_scaled, labels)

            probs_b = probe_b.predict_proba(H_band_scaled)
            probs_b = np.clip(probs_b, 1e-10, 1.0)
            probs_b /= probs_b.sum(axis=1, keepdims=True)
            H_cond_b = -np.mean(np.sum(probs_b * np.log2(probs_b), axis=1))
            I_band = np.log2(n_roles) - H_cond_b
            band_mi_list.append(float(I_band))

        sum_mi = sum(band_mi_list)
        redundancy = sum_mi / max(I_full, 0.001)
        avg_pct = np.mean(band_mi_list) / max(I_full, 0.001) * 100
        pca_results[n_bands] = {
            'redundancy': float(redundancy),
            'avg_band_pct': float(avg_pct),
        }
        print(f"    PCA bands={n_bands}: redundancy={redundancy:.2f}, avg_band_pct={avg_pct:.1f}%")

    # 总结
    print(f"\n  ★ 全息编码 vs 随机投影:")
    for n_bands in [5, 10]:
        wu_red = wu_results[n_bands]['redundancy']
        rand_red = random_results[n_bands]['redundancy']
        pca_red = pca_results[n_bands]['redundancy']
        print(f"    n_bands={n_bands}: W_U={wu_red:.2f}, Random={rand_red:.2f}, PCA={pca_red:.2f}")
        if wu_red > rand_red * 1.1:
            print(f"      → W_U投影冗余度显著高于随机投影, W_U有特殊结构!")
        elif abs(wu_red - rand_red) < 0.3:
            print(f"      → W_U和随机投影冗余度接近, 全息编码是高维空间的一般性质!")

    results = {
        'I_full': float(I_full),
        'wu_results': wu_results,
        'random_results': random_results,
        'pca_results': pca_results,
    }

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
    print(f"CCL-R Phase12 冗余编码信息论结构与维度谱 | Model={args.model} | Exp={args.exp}")
    print(f"{'='*70}")

    model, tokenizer, device = load_model(args.model)
    model_info = get_model_info(model, args.model)
    print(f"  Model: {model_info.model_class}, Layers={model_info.n_layers}, "
          f"d_model={model_info.d_model}")

    try:
        if args.exp == 1:
            results = exp1_redundancy_vs_bands(model, tokenizer, device)
        elif args.exp == 2:
            results = exp2_controlled_vocab_geometry(model, tokenizer, device)
        elif args.exp == 3:
            results = exp3_dimension_spectrum(model, tokenizer, device)
        elif args.exp == 4:
            results = exp4_holographic_vs_random(model, tokenizer, device)

        # 保存结果
        out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               '..', 'glm5_temp')
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir,
                               f"cclr_exp{args.exp}_{args.model}_results.json")

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
