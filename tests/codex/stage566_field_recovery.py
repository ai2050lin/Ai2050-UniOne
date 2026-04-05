"""
stage566: 场恢复原型——从attention pattern直接恢复语义场
动机：拓扑（距离矩阵）跨架构不一致，但场统计（attention pattern）
可能是跨模型更一致的"语言"。

本实验验证：
1. 同一概念在不同句子中是否产生相似的attention pattern
2. 不同概念在同层是否产生可区分的attention pattern
3. attention pattern是否比hidden states更跨层一致

使用Qwen3。
"""
import sys, os, json, time
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.dirname(__file__))

from qwen3_language_shared import load_qwen3_model, discover_layers
from multimodel_language_shared import encode_to_device, free_model

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'glm5', 'temp')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def find_target_position(tokenizer, token_ids, target_word):
    tokens = [tokenizer.convert_ids_to_tokens(t) for t in token_ids]
    for pos in range(len(tokens) - 1, -1, -1):
        if target_word in tokens[pos].lower():
            return pos
    return len(tokens) - 1


def get_attention_for_target(model, tokenizer, sentence, target_word, layer_idx):
    """获取target word在某层的attention pattern（作为row vector）"""
    enc = encode_to_device(model, tokenizer, sentence)
    pos = find_target_position(tokenizer, enc["input_ids"][0].tolist(), target_word)
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True, output_attentions=True)
    if out.attentions is not None:
        attn = out.attentions[layer_idx][0, :, pos, :].float().cpu().numpy()  # (num_heads, seq_len)
        hidden = out.hidden_states[layer_idx][0, pos].cpu().float().numpy()
        return attn, hidden, pos
    return None, None, pos


def attn_pattern_similarity(attn1, attn2):
    """两个attention pattern的相似度（展平后cosine，截断到相同长度）"""
    v1 = attn1.flatten().astype(np.float32)
    v2 = attn2.flatten().astype(np.float32)
    min_len = min(len(v1), len(v2))
    v1, v2 = v1[:min_len], v2[:min_len]
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm < 1e-10:
        return 0.0
    return float(np.dot(v1, v2) / norm)


def experiment1_concept_field_consistency(model, tokenizer, n_layers):
    """实验1：同一概念在不同句中的attention pattern一致性"""
    print(f"\n{'='*60}")
    print(f"  实验1：概念场一致性（同一词不同语境的attention pattern）")
    print(f"{'='*60}")

    concept_sentences = {
        "bank": [
            "The river bank was muddy",
            "I walked along the bank",
            "The bank gave me a loan",
            "She deposited money in the bank",
        ],
        "apple": [
            "The red apple is sweet",
            "I ate an apple for lunch",
            "Apple released a new phone",
            "Apple is a technology company",
        ],
    }

    results = {}
    for concept, sentences in concept_sentences.items():
        print(f"\n  === {concept} ({len(sentences)} sentences) ===")
        # 按意义分组（前半=意义1，后半=意义2）
        group1 = sentences[:2]
        group2 = sentences[2:]

        layer_samples = list(range(0, min(n_layers, 36), 4))
        if layer_samples[-1] != min(n_layers, 35):
            layer_samples.append(min(n_layers, 35))

        print(f"  {'Layer':>5s} | {'intra1':>7s} | {'intra2':>7s} | {'inter':>7s} | {'ratio':>6s}")
        print(f"  {'-'*5} | {'-'*7} | {'-'*7} | {'-'*7} | {'-'*6}")

        layer_results = {}
        for li in layer_samples:
            attns_g1 = []
            attns_g2 = []
            for s in group1:
                a, _, _ = get_attention_for_target(model, tokenizer, s, concept, li)
                if a is not None:
                    attns_g1.append(a)
            for s in group2:
                a, _, _ = get_attention_for_target(model, tokenizer, s, concept, li)
                if a is not None:
                    attns_g2.append(a)

            if len(attns_g1) < 2 or len(attns_g2) < 1:
                continue

            # 组内相似度
            intra1 = np.mean([attn_pattern_similarity(attns_g1[i], attns_g1[j])
                              for i in range(len(attns_g1)) for j in range(i+1, len(attns_g1))])
            intra2_sim = 1.0  # 单个没有组内

            # 组间相似度
            inter = np.mean([attn_pattern_similarity(a1, a2) for a1 in attns_g1 for a2 in attns_g2])

            ratio = inter / (intra1 + 1e-10)

            layer_results[f"L{li}"] = {"intra1": round(intra1, 4), "inter": round(inter, 4), "ratio": round(ratio, 4)}
            print(f"  L{li:3d}   | {intra1:7.4f} | {intra2_sim:7.4f} | {inter:7.4f} | {ratio:6.3f}")

        results[concept] = layer_results

    return results


def experiment2_cross_concept_field(model, tokenizer, n_layers):
    """实验2：不同概念的同层attention pattern可区分性"""
    print(f"\n{'='*60}")
    print(f"  实验2：跨概念场区分度")
    print(f"{'='*60}")

    concepts = {
        "bank": "The river bank was muddy and steep",
        "apple": "The red apple is sweet and delicious",
        "cat": "The cat sat on the mat quietly",
        "run": "The boy decided to run fast",
        "happy": "She felt very happy today",
        "the": "The weather was nice",
    }

    layer_samples = list(range(0, min(n_layers, 36), 6))
    if layer_samples[-1] != min(n_layers, 35):
        layer_samples.append(min(n_layers, 35))

    # 收集每个概念在每层的attention pattern
    concept_fields = {}
    for concept, sentence in concepts.items():
        concept_fields[concept] = {}
        for li in layer_samples:
            a, _, _ = get_attention_for_target(model, tokenizer, sentence, concept, li)
            if a is not None:
                concept_fields[concept][li] = a

    # 计算跨概念距离矩阵
    for li in layer_samples:
        available = {c: concept_fields[c].get(li) for c in concepts if li in concept_fields[c]}
        if len(available) < 2:
            continue

        names = list(available.keys())
        print(f"\n  L{li} attention pattern距离矩阵:")
        header = f"  {'':>8s}" + "".join([f" {n[:6]:>7s}" for n in names])
        print(header)

        for i, n1 in enumerate(names):
            row = f"  {n1[:6]:>8s}"
            for j, n2 in enumerate(names):
                if i == j:
                    row += f"   1.000"
                else:
                    a1, a2 = available[n1], available[n2]
                    min_len = min(a1.shape[1], a2.shape[1])
                    a1_trunc = a1[:, :min_len].flatten()
                    a2_trunc = a2[:, :min_len].flatten()
                    norm = np.linalg.norm(a1_trunc) * np.linalg.norm(a2_trunc)
                    if norm < 1e-10:
                        sim = 0.0
                    else:
                        sim = float(np.dot(a1_trunc, a2_trunc) / norm)
                    row += f" {sim:7.4f}"
            print(row)

    return {}


def experiment3_field_vs_hidden_consistency(model, tokenizer, n_layers):
    """实验3：attention field vs hidden state的跨层一致性对比"""
    print(f"\n{'='*60}")
    print(f"  实验3：场(attention) vs 隐藏状态(hidden)的跨层一致性")
    print(f"{'='*60}")

    sentences = [
        "The river bank was muddy and steep",
        "The bank gave me a loan today",
    ]

    layer_pairs = list(range(0, min(n_layers, 36), 4))

    for si, sent in enumerate(sentences):
        print(f"\n  Sentence {si+1}: '{sent[:35]}...'")
        print(f"  {'L_pair':>8s} | {'attn_cos':>8s} | {'hidden_cos':>10s}")
        print(f"  {'-'*8} | {'-'*8} | {'-'*10}")

        attns = {}
        hiddens = {}
        for li in layer_pairs:
            a, h, _ = get_attention_for_target(model, tokenizer, sent, "bank", li)
            if a is not None:
                attns[li] = a
                hiddens[li] = h

        for i in range(len(layer_pairs) - 1):
            l1, l2 = layer_pairs[i], layer_pairs[i+1]
            if l1 in attns and l2 in attns:
                a_cos = attn_pattern_similarity(attns[l1], attns[l2])
                h_cos = F.cosine_similarity(
                    torch.tensor(hiddens[l1]).unsqueeze(0),
                    torch.tensor(hiddens[l2]).unsqueeze(0), dim=1
                ).item()
                print(f"  L{l1}-L{l2}  | {a_cos:8.4f} | {h_cos:10.4f}")

    return {}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage566: 场恢复原型")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    print(f"  Qwen3 n_layers={n_layers}")

    try:
        r1 = experiment1_concept_field_consistency(model, tokenizer, n_layers)
        r2 = experiment2_cross_concept_field(model, tokenizer, n_layers)
        r3 = experiment3_field_vs_hidden_consistency(model, tokenizer, n_layers)
    finally:
        free_model(model)

    out_path = os.path.join(OUTPUT_DIR, "stage566_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump({"exp1": r1, "exp2": "见输出", "exp3": "见输出"}, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n  results saved: {out_path}")
    print(f"\n  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
