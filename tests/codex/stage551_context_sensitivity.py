"""
stage551: 上下文敏感性量化
目标：同一个词在不同语境中的编码差异有多大？
- "apple" 在 "red apple", "apple pie", "apple is a fruit", "I eat an apple" 中
- 量化：词义方差 vs 语境方差 → 编码中多少是词义，多少是语境
- 同时验证不同长度的输入对维度坍缩的影响

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

# 核心：同一个词放在不同语境中
CONTEXT_EXPERIMENTS = {
    "apple": [
        "apple",                     # 无语境（单token）
        "red apple",                 # 形容词修饰
        "green apple",               # 不同形容词
        "apple pie",                 # 复合名词
        "apple juice",               # 复合名词2
        "An apple is a fruit",       # 定义句
        "I eat an apple every day",  # 动作句
        "The apple fell from the tree",  # 事件句
        "Apple is a technology company",  # 歧义句（Apple=公司）
    ],
    "bank": [
        "bank",
        "river bank",
        "the bank of the river",
        "I went to the bank",
        "the bank gave me a loan",
        "blood bank",
        "bank robbery",
        "the river bank was muddy",
        "he sat on the bank fishing",
    ],
    "light": [
        "light",
        "the light is on",
        "turn off the light",
        "she is light",
        "a light rain",
        "the light of the sun",
        "travel at the speed of light",
        "light blue",
        "he painted with light strokes",
    ],
}

# 用于维度坍缩实验：不同长度的输入
LENGTH_EXPERIMENTS = {
    "1-token": ["apple", "cat", "hammer", "sun", "freedom", "red"],
    "3-token": ["red apple", "big cat", "old hammer", "the sun", "true freedom", "dark red"],
    "6-token": ["a red apple is sweet", "the big cat sleeps", "an old hammer works", "the bright sun shines", "true freedom matters", "paint it dark red"],
    "10-token": [
        "a red apple is a very sweet fruit",
        "the big cat sleeps on the soft bed",
        "an old hammer can fix many things",
        "the bright sun shines in the sky",
        "true freedom matters for everyone",
        "paint the wall with dark red paint",
    ],
    "15-token": [
        "a red apple is a very sweet fruit that grows on trees",
        "the big cat sleeps on the soft bed all day long",
        "an old hammer can fix many things around the old house",
        "the bright sun shines in the clear blue sky every morning",
        "true freedom matters for everyone in this modern society",
        "they decided to paint the wall with dark red paint color",
    ],
}


def get_word_encoding(model, tokenizer, sentence, target_word, layer_idx):
    """获取句中目标词的编码"""
    encoded = encode_to_device(model, tokenizer, sentence)
    input_ids = encoded["input_ids"]

    # 找target_word的token位置
    target_tokens = tokenizer.encode(target_word, add_special_tokens=False)
    target_pos = None
    for tid in target_tokens:
        matches = (input_ids[0] == tid).nonzero(as_tuple=True)[0]
        if len(matches) > 0:
            target_pos = matches[0].item()
            break
    if target_pos is None:
        target_pos = input_ids.shape[1] - 1

    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx + 1]
    return hs[0, target_pos, :], target_pos


def get_last_token_encoding(model, tokenizer, sentence, layer_idx):
    """获取最后一个token的编码"""
    encoded = encode_to_device(model, tokenizer, sentence)
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    hs = outputs.hidden_states[layer_idx + 1]
    return hs[0, -1, :]


def compute_effective_dim(matrix):
    """计算有效维度"""
    n = matrix.shape[0]
    if n < 2:
        return n
    centered = matrix - matrix.mean(dim=0, keepdim=True)
    cov = (centered.T @ centered) / max(n - 1, 1)
    eigenvalues, _ = torch.linalg.eigh(cov)
    top_eigs = eigenvalues[eigenvalues > 1e-10].flip(0).cpu().numpy()
    if len(top_eigs) == 0:
        return 0
    cumsum = np.cumsum(top_eigs) / max(top_eigs.sum(), 1e-10)
    return int(np.searchsorted(cumsum, 0.90) + 1)


def main():
    print(f"{'='*60}")
    print(f"  stage551: 上下文敏感性量化 (Qwen3)")
    print(f"{'='*60}")

    model, tokenizer = load_qwen3_model()
    layers = discover_layers(model)
    n_layers = len(layers)
    sample_layers = evenly_spaced_layers(model, count=10)
    print(f"  层数: {n_layers}, 采样: {sample_layers}")

    # ========== 实验1: 上下文敏感性 ==========
    print(f"\n{'='*60}")
    print(f"  [实验1] 同词不同语境 - 编码差异量化")
    print(f"{'='*60}")

    context_results = {}
    for word, sentences in CONTEXT_EXPERIMENTS.items():
        print(f"\n  --- 词: {word} ({len(sentences)}种语境) ---")
        word_data = {}
        
        for li in sample_layers:
            encodings = []
            for sent in sentences:
                enc, pos = get_word_encoding(model, tokenizer, sent, word, li)
                encodings.append(enc.float().cpu())
            
            matrix = torch.stack(encodings)  # [n_contexts, hidden_dim]
            
            # 计算所有配对的cosine距离
            n_ctx = len(sentences)
            all_dists = []
            for i in range(n_ctx):
                for j in range(i + 1, n_ctx):
                    d = 1 - F.cosine_similarity(encodings[i].unsqueeze(0), encodings[j].unsqueeze(0)).item()
                    all_dists.append(d)
            
            # 方差分解
            mean_enc = matrix.mean(dim=0)  # "词义中心"
            l2_to_center = [float(torch.norm(e - mean_enc)) for e in encodings]
            
            word_data[str(li)] = {
                "mean_cosine_dist": round(float(np.mean(all_dists)), 6),
                "std_cosine_dist": round(float(np.std(all_dists)), 6),
                "max_cosine_dist": round(float(np.max(all_dists)), 6),
                "min_cosine_dist": round(float(np.min(all_dists)), 6),
                "mean_l2_to_center": round(float(np.mean(l2_to_center)), 4),
                "std_l2_to_center": round(float(np.std(l2_to_center)), 4),
            }
        
        # 打印首末层
        for li in [sample_layers[0], sample_layers[-1]]:
            d = word_data[str(li)]
            print(f"    L{li}: cos_mean={d['mean_cosine_dist']:.4f}+/-{d['std_cosine_dist']:.4f}, "
                  f"L2_to_center={d['mean_l2_to_center']:.2f}+/-{d['std_l2_to_center']:.2f}")
        
        # 打印具体语境间的距离（末层）
        last_li = sample_layers[-1]
        print(f"    末层(L{last_li})语境对距离:")
        pairs_to_show = [(0, 1), (0, 2), (0, 5), (1, 5)]  # 无语境vs有语境
        for a, b in pairs_to_show:
            if a < len(sentences) and b < len(sentences):
                enc_a, _ = get_word_encoding(model, tokenizer, sentences[a], word, last_li)
                enc_b, _ = get_word_encoding(model, tokenizer, sentences[b], word, last_li)
                d = 1 - F.cosine_similarity(enc_a.float().unsqueeze(0), enc_b.float().unsqueeze(0)).item()
                l2 = float(torch.norm(enc_a.float() - enc_b.float()))
                print(f"      [{a}]'{sentences[a][:20]}' vs [{b}]'{sentences[b][:20]}': cos_d={d:.4f}, L2={l2:.2f}")
        
        context_results[word] = word_data

    # ========== 实验2: 词义方差 vs 语境方差 ==========
    print(f"\n{'='*60}")
    print(f"  [实验2] 词义方差 vs 语境方差")
    print(f"{'='*60}")

    # 用"apple", "banana", "cherry"作为不同词义
    # 同一词（apple）在不同语境中的方差 = "语境方差"
    # 不同词（apple, banana, cherry）在同一语境模板中的方差 = "词义方差"

    meaning_words = ["apple", "banana", "cherry"]
    context_templates = [
        "red {}", "a {} is sweet", "I eat a {} every day",
        "the {} fell from the tree", "a {} pie is delicious",
    ]

    last_li = sample_layers[-1]
    print(f"\n  末层(L{last_li})方差分解:")

    # 语境方差：同一词在不同模板中的编码方差
    for word in meaning_words:
        context_encs = []
        for tmpl in context_templates:
            sent = tmpl.format(word)
            enc, _ = get_word_encoding(model, tokenizer, sent, word, last_li)
            context_encs.append(enc.float().cpu())
        ctx_matrix = torch.stack(context_encs)
        # 配对距离的均值
        ctx_dists = []
        for i in range(len(ctx_matrix)):
            for j in range(i + 1, len(ctx_matrix)):
                d = 1 - F.cosine_similarity(ctx_matrix[i].unsqueeze(0), ctx_matrix[j].unsqueeze(0)).item()
                ctx_dists.append(d)
        print(f"    '{word}' 语境方差(cos_d): mean={np.mean(ctx_dists):.4f}, std={np.std(ctx_dists):.4f}")

    # 词义方差：不同词在同一模板中的编码方差
    for tmpl in context_templates:
        meaning_encs = []
        for word in meaning_words:
            sent = tmpl.format(word)
            enc, _ = get_word_encoding(model, tokenizer, sent, word, last_li)
            meaning_encs.append(enc.float().cpu())
        mean_matrix = torch.stack(meaning_encs)
        mean_dists = []
        for i in range(len(mean_matrix)):
            for j in range(i + 1, len(mean_matrix)):
                d = 1 - F.cosine_similarity(mean_matrix[i].unsqueeze(0), mean_matrix[j].unsqueeze(0)).item()
                mean_dists.append(d)
        print(f"    模板'{tmpl}' 词义方差(cos_d): mean={np.mean(mean_dists):.4f}, std={np.std(mean_dists):.4f}")

    # ========== 实验3: 输入长度 vs 维度坍缩 ==========
    print(f"\n{'='*60}")
    print(f"  [实验3] 输入长度 vs 维度坍缩")
    print(f"{'='*60}")

    length_results = {}
    for length_label, sentences in LENGTH_EXPERIMENTS.items():
        print(f"\n  --- {length_label} ({len(sentences[0].split())} 词) ---")
        encodings = {}
        for li in sample_layers:
            layer_encs = []
            for sent in sentences:
                enc = get_last_token_encoding(model, tokenizer, sent, li)
                layer_encs.append(enc.float().cpu())
            encodings[li] = torch.stack(layer_encs)

        # 有效维度
        print(f"    有效维度:")
        for li in sample_layers:
            dim = compute_effective_dim(encodings[li])
            print(f"      L{li}: {dim}")

        length_results[length_label] = {
            "n_words": len(sentences[0].split()),
            "n_sentences": len(sentences),
            "effective_dims": {str(li): compute_effective_dim(encodings[li]) for li in sample_layers},
        }

        # 检测坍缩
        dims_list = [length_results[length_label]["effective_dims"][str(li)] for li in sample_layers]
        collapsed = False
        for i in range(1, len(dims_list)):
            if dims_list[i] <= 1 and dims_list[i-1] > 1:
                print(f"    >> 坍缩在 L{sample_layers[i]}")
                collapsed = True
                break
        if not collapsed and dims_list[0] > 1:
            print(f"    >> 无坍缩")

    # ========== 总结 ==========
    print(f"\n{'='*60}")
    print(f"  拼图总结")
    print(f"{'='*60}")
    print(f"\n  [维度坍缩 vs 输入长度]:")
    for label, data in length_results.items():
        first_dim = data["effective_dims"][str(sample_layers[0])]
        min_dim = min(data["effective_dims"].values())
        print(f"    {label:>8} ({data['n_words']:>2}词): L0_dim={first_dim:>2}, min_dim={min_dim:>2}")

    # 保存
    result = {
        "context_sensitivity": context_results,
        "length_vs_collapse": length_results,
    }
    out_path = os.path.join(OUTPUT_DIR, "stage551_context_sensitivity.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"\n  保存到: {out_path}")

    free_model(model)


if __name__ == "__main__":
    main()
