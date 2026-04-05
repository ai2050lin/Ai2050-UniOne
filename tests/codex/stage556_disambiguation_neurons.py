"""
stage556: 消歧机制的神经元级分析
目标：bank/apple/light在不同语境中，哪些神经元活动不同？
- L8（消歧峰值层）的神经元差异分析
- 歧义神经元 vs 非歧义神经元
- 消歧子回路识别：差异最大的Top-K神经元
- 跨消歧层(L0→L35)的神经元轨迹

使用Qwen3。
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

# 歧义词实验
DISAMBIG_WORDS = {
    "bank": {
        "contexts": [
            ("river bank", "河岸"),
            ("the bank of the river", "河岸-长"),
            ("the bank gave me a loan", "银行"),
            ("I went to the bank", "银行-短"),
            ("blood bank", "血库"),
        ],
        "target_word": "bank",
    },
    "apple": {
        "contexts": [
            ("red apple is sweet", "水果"),
            ("I eat an apple", "水果-动作"),
            ("Apple is a technology company", "公司"),
            ("Apple released a new phone", "公司-产品"),
        ],
        "target_word": "apple",
    },
    "light": {
        "contexts": [
            ("the light is on", "光照"),
            ("sunlight is bright", "光照-同义"),
            ("the box is very light", "轻"),
            ("a light rain fell", "轻-雨"),
        ],
        "target_word": "light",
    },
}


def find_target_position(tokenizer, token_ids, target_word):
    """找到目标词在token序列中的位置"""
    tokens = [tokenizer.convert_ids_to_tokens(t) for t in token_ids]
    # 从后往前找（目标词通常在后面）
    for pos in range(len(tokens) - 1, -1, -1):
        if target_word in tokens[pos].lower():
            return pos
    return len(tokens) - 1  # fallback: last token


def get_target_encoding(model, encoded, target_pos, layer_idx):
    """获取特定层目标位置的编码"""
    with torch.no_grad():
        out = model(**encoded, output_hidden_states=True)
    return out.hidden_states[layer_idx][0, target_pos].cpu().float().numpy()


def experiment1_neuron_disambiguation(model, tokenizer, n_layers):
    """L8层的神经元差异分析"""
    print(f"\n{'='*60}")
    print(f"  实验1：L8消歧峰值的神经元差异")
    print(f"{'='*60}")

    # L8作为消歧峰值层
    peak_layer = min(8, n_layers - 1)
    results = {}

    for word, config in DISAMBIG_WORDS.items():
        print(f"\n  === {word} (L{peak_layer}) ===")

        # 收集各语境的编码
        context_encodings = {}
        for ctx_sent, ctx_desc in config["contexts"]:
            encoded = encode_to_device(model, tokenizer, ctx_sent)
            token_ids = encoded["input_ids"][0].tolist()
            target_pos = find_target_position(tokenizer, token_ids, config["target_word"])
            enc = get_target_encoding(model, encoded, target_pos, peak_layer)
            context_encodings[ctx_desc] = enc
            tokens_safe = [tokenizer.convert_ids_to_tokens(t).encode('ascii', 'replace').decode()
                          for t in token_ids]
            print(f"    '{ctx_sent}' → pos={target_pos}, tokens={tokens_safe}")

        # 计算所有语境对之间的神经元差异
        ctx_names = list(context_encodings.keys())
        all_encodings = np.array(list(context_encodings.values()))

        # 每个神经元的标准差（跨语境）
        neuron_std = all_encodings.std(axis=0)
        neuron_mean = all_encodings.mean(axis=0)

        # 按变异系数排序
        cv = neuron_std / (np.abs(neuron_mean) + 1e-10)
        top_k = min(20, len(cv))
        top_neuron_idx = np.argsort(-cv)[:top_k]

        print(f"\n    Top-{top_k} 消歧神经元 (按变异系数排序):")
        print(f"    神经元ID | CV     | 均值    | 标准差  | 各语境值(前3语境)")
        for rank, idx in enumerate(top_neuron_idx):
            vals = all_encodings[:, idx]
            vals_str = ", ".join(f"{v:+.3f}" for v in vals[:3])
            print(f"    N{idx:6d} | {cv[idx]:.4f} | {neuron_mean[idx]:+.4f} | {neuron_std[idx]:.4f} | [{vals_str}]")

        # 消歧语境间的cos距离
        print(f"\n    语境间cos距离矩阵:")
        print(f"    {'':>16s}", end="")
        for name in ctx_names[:4]:
            short = name[:5]
            print(f" | {short:>6s}", end="")
        print()
        for i, name_i in enumerate(ctx_names[:4]):
            short_i = name_i[:5]
            print(f"    {short_i:>16s}", end="")
            for j, name_j in enumerate(ctx_names[:4]):
                if i == j:
                    print(f" | {'--':>6s}", end="")
                else:
                    cos_d = 1 - F.cosine_similarity(
                        torch.tensor(context_encodings[name_i]),
                        torch.tensor(context_encodings[name_j]),
                        dim=0
                    ).item()
                    print(f" | {cos_d:6.4f}", end="")
            print()

        # 消歧神经元数量
        significant_neurons = np.sum(cv > 1.0)  # CV>1的神经元
        total_neurons = len(cv)
        print(f"\n    显著消歧神经元(CV>1): {significant_neurons}/{total_neurons} "
              f"({significant_neurons/total_neurons*100:.1f}%)")

        # 用Top-K神经元重建的距离 vs 全维度距离
        top_k_list = [5, 10, 20, 50, 100]
        print(f"\n    Top-K神经元重建 vs 全维度距离 (cosine):")
        for k in top_k_list:
            if k > total_neurons:
                continue
            selected = top_neuron_idx[:k]
            # 用选中的神经元计算cos距离
            pair_corrs = []
            full_corrs = []
            for i in range(len(ctx_names)):
                for j in range(i + 1, len(ctx_names)):
                    sub_i = context_encodings[ctx_names[i]][selected]
                    sub_j = context_encodings[ctx_names[j]][selected]
                    cos_sub = 1 - F.cosine_similarity(
                        torch.tensor(sub_i), torch.tensor(sub_j), dim=0
                    ).item()
                    cos_full = 1 - F.cosine_similarity(
                        torch.tensor(context_encodings[ctx_names[i]]),
                        torch.tensor(context_encodings[ctx_names[j]]),
                        dim=0
                    ).item()
                    pair_corrs.append(cos_sub)
                    full_corrs.append(cos_full)

            r = np.corrcoef(pair_corrs, full_corrs)[0, 1] if np.std(pair_corrs) > 1e-10 else 0
            print(f"      K={k:3d}: r={r:.4f} (与全维度的Pearson相关)")

        results[word] = {
            "significant_neurons": int(significant_neurons),
            "total_neurons": total_neurons,
            "pct": round(significant_neurons / total_neurons * 100, 1),
            "top_cv": [round(float(cv[i]), 4) for i in top_neuron_idx[:10]],
        }

    return results


def experiment2_layer_trajectory(model, tokenizer, n_layers):
    """消歧能力逐层变化轨迹"""
    print(f"\n{'='*60}")
    print(f"  实验2：消歧能力逐层轨迹")
    print(f"{'='*60}")

    for word, config in DISAMBIG_WORDS.items():
        # 取两个最不同的语境
        ctx_pairs = []
        for i, (s1, d1) in enumerate(config["contexts"]):
            for j, (s2, d2) in enumerate(config["contexts"]):
                if j > i:
                    ctx_pairs.append((s1, d1, s2, d2))

        # 只取前3对（避免太多）
        ctx_pairs = ctx_pairs[:3]

        print(f"\n  === {word} ===")

        # 细粒度采样
        all_layers_list = list(range(0, min(n_layers, 20))) + \
                          list(range(20, n_layers, max(1, (n_layers - 20) // 15)))
        if (n_layers - 1) not in all_layers_list:
            all_layers_list.append(n_layers - 1)
        all_layers_list = sorted(set(all_layers_list))

        for s1, d1, s2, d2 in ctx_pairs:
            enc1 = encode_to_device(model, tokenizer, s1)
            enc2 = encode_to_device(model, tokenizer, s2)
            pos1 = find_target_position(tokenizer, enc1["input_ids"][0].tolist(), config["target_word"])
            pos2 = find_target_position(tokenizer, enc2["input_ids"][0].tolist(), config["target_word"])

            with torch.no_grad():
                out1 = model(**enc1, output_hidden_states=True)
                out2 = model(**enc2, output_hidden_states=True)

            print(f"  '{d1}' vs '{d2}':")
            print(f"    层  | cos_d  | L2_dist | 消歧神经元数(CV>1)")
            for li in all_layers_list:
                e1 = out1.hidden_states[li][0, pos1].cpu().float().numpy()
                e2 = out2.hidden_states[li][0, pos2].cpu().float().numpy()
                cos_d = 1 - F.cosine_similarity(
                    torch.tensor(e1), torch.tensor(e2), dim=0
                ).item()
                l2_d = float(np.linalg.norm(e1 - e2))
                # CV>1的神经元数
                stacked = np.stack([e1, e2])
                n_std = stacked.std(axis=0)
                n_mean = stacked.mean(axis=0)
                cv = n_std / (np.abs(n_mean) + 1e-10)
                n_sig = int(np.sum(cv > 1.0))

                norm_pos = li / max(n_layers - 1, 1)
                marker = ""
                if li == 8:
                    marker = " ← L8!"
                print(f"    L{li:2d}{marker:5s} | {cos_d:.4f} | {l2_d:7.1f} | {n_sig:5d}")

    return {}


def experiment3_neuron_overlap(model, tokenizer, n_layers):
    """不同歧义词的消歧神经元重叠度"""
    print(f"\n{'='*60}")
    print(f"  实验3：消歧神经元重叠（bank vs apple vs light）")
    print(f"{'='*60}")

    peak_layer = min(8, n_layers - 1)

    # 收集每个词的消歧神经元
    word_top_neurons = {}
    for word, config in DISAMBIG_WORDS.items():
        encodings = []
        for ctx_sent, ctx_desc in config["contexts"]:
            encoded = encode_to_device(model, tokenizer, ctx_sent)
            token_ids = encoded["input_ids"][0].tolist()
            target_pos = find_target_position(tokenizer, token_ids, config["target_word"])
            enc = get_target_encoding(model, encoded, target_pos, peak_layer)
            encodings.append(enc)

        stacked = np.array(encodings)
        n_std = stacked.std(axis=0)
        n_mean = stacked.mean(axis=0)
        cv = n_std / (np.abs(n_mean) + 1e-10)
        top_idx = set(np.argsort(-cv)[:50].tolist())  # Top 50
        word_top_neurons[word] = top_idx
        n_sig = int(np.sum(cv > 1.0))
        print(f"  {word}: 显著消歧神经元={n_sig}, Top50索引范围={min(top_idx)}-{max(top_idx)}")

    # 计算重叠
    words = list(word_top_neurons.keys())
    print(f"\n  Top50神经元重叠:")
    for i in range(len(words)):
        for j in range(i + 1, len(words)):
            w1, w2 = words[i], words[j]
            overlap = word_top_neurons[w1] & word_top_neurons[w2]
            union = word_top_neurons[w1] | word_top_neurons[w2]
            jaccard = len(overlap) / len(union) if len(union) > 0 else 0
            print(f"    {w1} ∩ {w2}: {len(overlap)}/{len(union)} (Jaccard={jaccard:.4f})")

    return {}


def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage556: 消歧机制神经元级分析")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    print(f"  Qwen3 n_layers={n_layers}")

    try:
        r1 = experiment1_neuron_disambiguation(model, tokenizer, n_layers)
        r2 = experiment2_layer_trajectory(model, tokenizer, n_layers)
        r3 = experiment3_neuron_overlap(model, tokenizer, n_layers)
    finally:
        free_model(model)

    # 保存
    out_path = os.path.join(OUTPUT_DIR, "stage556_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(r1, f, ensure_ascii=False, indent=2)
    print(f"\n  结果已保存: {out_path}")

    print(f"\n  总耗时: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
