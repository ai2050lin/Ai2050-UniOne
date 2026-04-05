"""
stage560: 大规模验证——100+词多种词性
目标：用大规模词汇验证之前发现的不变量是否成立
- 100+词（名词/动词/形容词/功能词/副词/介词）
- 验证家族内聚性是否在不同词性中都成立
- 验证语境方差>词义方差的普适性
- 验证消歧神经元比例(56%)是否在不同模型中一致
- 编码距离矩阵的维度分析

使用Qwen3快速验证。
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_language_shared import load_qwen3_model, discover_layers
from multimodel_language_shared import encode_to_device, evenly_spaced_layers, free_model

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'glm5', 'temp')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 120个词，覆盖6个词性
LARGE_VOCAB = {
    "noun_animal": ["cat", "dog", "horse", "bird", "fish", "lion", "tiger", "bear",
                    "elephant", "monkey", "rabbit", "fox", "wolf", "deer", "eagle",
                    "whale", "snake", "frog", "ant", "bee"],
    "noun_object": ["table", "chair", "book", "phone", "car", "house", "door", "window",
                    "key", "cup", "plate", "knife", "clock", "lamp", "shoe",
                    "bag", "ring", "coin", "flag", "rope"],
    "verb_motion": ["run", "walk", "jump", "fly", "swim", "climb", "crawl", "dance",
                    "throw", "catch", "push", "pull", "carry", "lift", "drop",
                    "slide", "roll", "spin", "bounce", "crash"],
    "verb_cognition": ["think", "know", "believe", "remember", "forget", "learn", "understand",
                       "imagine", "wonder", "doubt", "realize", "notice", "recognize",
                       "discover", "solve", "decide", "choose", "plan", "predict", "guess"],
    "adj_physical": ["red", "blue", "green", "yellow", "black", "white", "big", "small",
                     "tall", "short", "long", "wide", "thin", "thick", "heavy",
                     "soft", "hard", "rough", "smooth", "sharp"],
    "adj_abstract": ["good", "bad", "happy", "sad", "fast", "slow", "easy", "hard",
                     "new", "old", "young", "rich", "poor", "strong", "weak",
                     "safe", "dangerous", "clean", "dirty", "free"],
}


def compute_effective_dim(vectors):
    if len(vectors) < 2:
        return 1.0
    X = np.array(vectors)
    X = X - X.mean(axis=0)
    cov = (X.T @ X) / (len(X) - 1)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 0)
    eigenvalues = eigenvalues[::-1]
    total = eigenvalues.sum()
    if total < 1e-10:
        return 1.0
    cumsum = np.cumsum(eigenvalues) / total
    for k in range(1, len(cumsum)):
        if cumsum[k] >= 0.9:
            return float(k + 1)
    return float(len(eigenvalues))


def experiment1_family_cohesion_large(model, tokenizer, n_layers):
    """120词家族内聚性验证"""
    print(f"\n{'='*60}")
    print(f"  实验1：大规模家族内聚性（120词×6类）")
    print(f"{'='*60}")

    sample_layers = evenly_spaced_layers(model, count=7)

    all_encodings = {}
    for family, words in LARGE_VOCAB.items():
        family_encs = {}
        for li in sample_layers:
            encs = []
            for w in words:
                encoded = encode_to_device(model, tokenizer, w)
                with torch.no_grad():
                    out = model(**encoded, output_hidden_states=True)
                encs.append(out.hidden_states[li][0, -1].cpu().float().numpy())
            family_encs[li] = np.array(encs)
        all_encodings[family] = family_encs

    # 末层 intra/inter
    last_layer = sample_layers[-1]
    print(f"\n  末层 L{last_layer} 家族内聚性:")
    print(f"  家族            | intra   | inter   | ratio  | eff_dim")
    print(f"  ----------------+---------+---------+--------+--------")

    family_names = list(LARGE_VOCAB.keys())
    for fam in family_names:
        encs = all_encodings[fam][last_layer]

        # intra
        intra_dists = []
        for i in range(len(encs)):
            for j in range(i + 1, len(encs)):
                d = 1 - F.cosine_similarity(
                    torch.tensor(encs[i]), torch.tensor(encs[j]), dim=0
                ).item()
                intra_dists.append(d)

        # inter
        inter_dists = []
        for other_fam in family_names:
            if other_fam == fam:
                continue
            other_encs = all_encodings[other_fam][last_layer]
            for i in range(len(encs)):
                for j in range(len(other_encs)):
                    d = 1 - F.cosine_similarity(
                        torch.tensor(encs[i]), torch.tensor(other_encs[j]), dim=0
                    ).item()
                    inter_dists.append(d)

        ratio = np.mean(intra_dists) / np.mean(inter_dists) if np.mean(inter_dists) > 1e-10 else 0
        eff_dim = compute_effective_dim(encs)

        print(f"  {fam:16s} | {np.mean(intra_dists):.4f}  | {np.mean(inter_dists):.4f}  | {ratio:.4f}  | {eff_dim:.0f}")

    return {}


def experiment2_context_variance_large(model, tokenizer, n_layers):
    """大规模语境方差验证"""
    print(f"\n{'='*60}")
    print(f"  实验2：大规模语境方差验证（20词×5语境）")
    print(f"{'='*60}")

    # 20个词，每个5个不同语境
    test_words = ["apple", "bank", "light", "cat", "run", "big", "fast", "book",
                  "house", "think", "good", "new", "red", "car", "water",
                  "fire", "stone", "dream", "music", "gold"]

    context_templates = [
        "the {}",           # 冠词+名词
        "a {}",             # 不定冠词
        "{} is good",       # 判断句
        "I like {}",        # 偏好句
        "the {} was there", # 事件句
    ]

    sample_layers = evenly_spaced_layers(model, count=7)
    last_layer = sample_layers[-1]

    # 词义方差：同语境不同词
    meaning_variances = []
    for tmpl in context_templates:
        word_encs = []
        for w in test_words:
            sent = tmpl.format(w)
            encoded = encode_to_device(model, tokenizer, sent)
            token_ids = encoded["input_ids"][0].tolist()
            with torch.no_grad():
                out = model(**encoded, output_hidden_states=True)
            # 取目标词位置
            tokens = [tokenizer.convert_ids_to_tokens(t) for t in token_ids]
            target_pos = len(tokens) - 1
            word_encs.append(out.hidden_states[last_layer][0, target_pos].cpu().float().numpy())

        # 计算词义方差
        word_encs = np.array(word_encs)
        # 所有词对的cos_d
        dists = []
        for i in range(len(word_encs)):
            for j in range(i + 1, len(word_encs)):
                d = 1 - F.cosine_similarity(
                    torch.tensor(word_encs[i]), torch.tensor(word_encs[j]), dim=0
                ).item()
                dists.append(d)
        meaning_variances.append(np.mean(dists))
        print(f"  模板 '{tmpl:25s}': 词义方差={np.mean(dists):.4f}")

    # 语境方差：同词不同语境
    context_variances = []
    for w in test_words[:10]:  # 只取10个词避免太慢
        word_ctx_encs = []
        for tmpl in context_templates:
            sent = tmpl.format(w)
            encoded = encode_to_device(model, tokenizer, sent)
            with torch.no_grad():
                out = model(**encoded, output_hidden_states=True)
            tokens = [tokenizer.convert_ids_to_tokens(t) for t in encoded["input_ids"][0].tolist()]
            target_pos = len(tokens) - 1
            word_ctx_encs.append(out.hidden_states[last_layer][0, target_pos].cpu().float().numpy())

        word_ctx_encs = np.array(word_ctx_encs)
        dists = []
        for i in range(len(word_ctx_encs)):
            for j in range(i + 1, len(word_ctx_encs)):
                d = 1 - F.cosine_similarity(
                    torch.tensor(word_ctx_encs[i]), torch.tensor(word_ctx_encs[j]), dim=0
                ).item()
                dists.append(d)
        context_variances.append(np.mean(dists))

    print(f"\n  汇总:")
    print(f"    词义方差(5模板平均): {np.mean(meaning_variances):.4f}")
    print(f"    语境方差(10词平均):  {np.mean(context_variances):.4f}")
    print(f"    语境/词义 比值:      {np.mean(context_variances)/np.mean(meaning_variances):.2f}x")

    return {}


def experiment3_distance_matrix_large(model, tokenizer, n_layers):
    """120词距离矩阵+有效维度"""
    print(f"\n{'='*60}")
    print(f"  实验3：120词距离矩阵维度分析")
    print(f"{'='*60}")

    sample_layers = evenly_spaced_layers(model, count=7)

    all_words = []
    for fam_words in LARGE_VOCAB.values():
        all_words.extend(fam_words)

    # 采样40个词（每类取一部分，减少计算量）
    np.random.seed(42)
    sample_words = []
    for fam, words in LARGE_VOCAB.items():
        sample_words.extend(np.random.choice(words, size=min(7, len(words)), replace=False).tolist())
    sample_words = sample_words[:40]  # 限制40个

    print(f"  采样词数: {len(sample_words)}")

    print(f"  层  | 有效维度 | mean_cos_d | std_cos_d")
    for li in sample_layers:
        encs = []
        for w in sample_words:
            encoded = encode_to_device(model, tokenizer, w)
            with torch.no_grad():
                out = model(**encoded, output_hidden_states=True)
            encs.append(out.hidden_states[li][0, -1].cpu().float().numpy())

        encs = np.array(encs)
        eff_dim = compute_effective_dim(encs)

        dists = []
        for i in range(len(encs)):
            for j in range(i + 1, len(encs)):
                d = 1 - F.cosine_similarity(
                    torch.tensor(encs[i]), torch.tensor(encs[j]), dim=0
                ).item()
                dists.append(d)

        print(f"  L{li:2d} | {eff_dim:8.0f} | {np.mean(dists):10.4f} | {np.std(dists):.4f}")

    return {}


def experiment4_disambig_neuron_ratio(model, tokenizer, n_layers):
    """消歧神经元比例：120词中找歧义词验证"""
    print(f"\n{'='*60}")
    print(f"  实验4：消歧神经元比例（多歧义词）")
    print(f"{'='*60}")

    # 更多歧义词
    polysemous_words = {
        "bank": [("river bank", "the bank gave a loan")],
        "apple": [("red apple fruit", "Apple company iPhone")],
        "light": [("the light is bright", "the box is light")],
        "bat": [("the bat flew away", "baseball bat hit")],
        "match": [("the match was exciting", "strike a match")],
        "plant": [("the plant grew tall", "power plant energy")],
        "ring": [("gold ring finger", "ring the bell")],
        "watch": [("I watch TV", "wrist watch time")],
        "spring": [("spring season warm", "spring water flow")],
        "right": [("turn right direction", "human rights law")],
    }

    patch_layer = min(8, n_layers - 1)
    dim = 2560  # Qwen3 hidden dim

    print(f"  L{patch_layer} 消歧神经元比例:")
    ratios = []
    for word, (ctx1, ctx2) in polysemous_words.items():
        encs = []
        for ctx in [ctx1, ctx2]:
            encoded = encode_to_device(model, tokenizer, ctx)
            token_ids = encoded["input_ids"][0].tolist()
            tokens = [tokenizer.convert_ids_to_tokens(t) for t in token_ids]
            target_pos = len(tokens) - 1
            with torch.no_grad():
                out = model(**encoded, output_hidden_states=True)
            encs.append(out.hidden_states[patch_layer][0, target_pos].cpu().float().numpy())

        stacked = np.array(encs)
        n_std = stacked.std(axis=0)
        n_mean = stacked.mean(axis=0)
        cv = n_std / (np.abs(n_mean) + 1e-10)
        n_sig = int(np.sum(cv > 1.0))
        pct = n_sig / dim * 100
        ratios.append(pct)

        # cos_d between contexts
        cos_d = 1 - F.cosine_similarity(
            torch.tensor(encs[0]), torch.tensor(encs[1]), dim=0
        ).item()
        print(f"    {word:8s}: cos_d={cos_d:.4f}, sig_neurons={n_sig:4d}/{dim} ({pct:.1f}%)")

    print(f"\n  平均消歧神经元比例: {np.mean(ratios):.1f}%")
    print(f"  范围: [{min(ratios):.1f}%, {max(ratios):.1f}%]")

    return {}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage560: 大规模验证——120+词多种词性")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    print(f"  Qwen3 n_layers={n_layers}, dim=2560")

    try:
        r1 = experiment1_family_cohesion_large(model, tokenizer, n_layers)
        r2 = experiment2_context_variance_large(model, tokenizer, n_layers)
        r3 = experiment3_distance_matrix_large(model, tokenizer, n_layers)
        r4 = experiment4_disambig_neuron_ratio(model, tokenizer, n_layers)
    finally:
        free_model(model)

    out_path = os.path.join(OUTPUT_DIR, "stage560_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({}, f, ensure_ascii=False, indent=2)
    print(f"\n  results saved: {out_path}")
    print(f"\n  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
