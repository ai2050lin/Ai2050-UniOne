"""
stage563: 输入标准化研究——不同输入格式对编码场的影响
目标：确定应该用什么格式作为标准输入来分析编码结构。

四种输入格式的对比：
1. 单token "apple"
2. 修饰语+名词 "red apple"
3. 完整句式 "The red apple is sweet"
4. 多句语境 "The red apple is sweet. I love fruits."

对每种格式测量：
- 编码有效维度
- 家族内聚性(intra vs inter)
- 消歧神经元的比例和集中度
- 逐层cos_d轨迹

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


def effective_dimension(vectors, threshold=0.9):
    """PCA有效维度"""
    if len(vectors) < 2:
        return 1
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    cov = np.cov(centered, rowvar=False)
    if cov.ndim < 2:
        return 1
    eigenvalues = np.sort(np.abs(np.linalg.eigvalsh(cov)))[::-1]
    total = eigenvalues.sum() + 1e-10
    cumsum = np.cumsum(eigenvalues) / total
    return max(1, int(np.searchsorted(cumsum, threshold) + 1))


def experiment1_format_dimension_comparison(model, tokenizer, n_layers):
    """实验1：不同输入格式的有效维度对比"""
    print(f"\n{'='*60}")
    print(f"  实验1：输入格式 vs 有效维度")
    print(f"{'='*60}")

    # 同一组水果词，不同格式
    fruit_words = ["apple", "banana", "orange", "grape", "mango", "peach", "pear", "cherry", "lemon", "kiwi"]

    formats = {
        "single_token": [w for w in fruit_words],
        "adj_noun": [f"red {w}" for w in fruit_words],
        "full_sentence": [f"The {w} is sweet and delicious" for w in fruit_words],
        "multi_sentence": [f"The {w} is sweet and delicious. I love fruits." for w in fruit_words],
    }

    layer_samples = list(range(0, n_layers, max(1, n_layers // 10)))
    if layer_samples[-1] != n_layers - 1:
        layer_samples.append(n_layers - 1)
    results = {}

    for fmt_name, sentences in formats.items():
        print(f"\n  Format: {fmt_name} (sentences example: '{sentences[0][:40]}...')")

        fmt_result = {}
        for li in layer_samples:
            encodings = []
            for sent in sentences:
                enc = encode_to_device(model, tokenizer, sent)
                token_ids = enc["input_ids"][0].tolist()
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True)
                # 取最后一个token的编码
                h = out.hidden_states[li][0, -1].cpu().float().numpy()
                encodings.append(h)

            enc_matrix = np.array(encodings)
            ed = effective_dimension(enc_matrix)
            fmt_result[f"L{li}"] = ed

        print(f"    Layers: {', '.join([f'{k}={v}' for k, v in fmt_result.items()])}")
        results[fmt_name] = fmt_result

    # 打印对比表
    print(f"\n  对比表（有效维度）:")
    header = f"  {'Layer':>5s} | " + " | ".join([f"{f[:8]:>8s}" for f in formats.keys()])
    print(header)
    for li in layer_samples:
        row = f"  L{li:3d}   |"
        for fmt_name in formats.keys():
            row += f" {results[fmt_name][f'L{li}']:>8d} |"
        print(row)

    return results


def experiment2_format_cohesion(model, tokenizer, n_layers):
    """实验2：不同输入格式的家族内聚性"""
    print(f"\n{'='*60}")
    print(f"  实验2：输入格式 vs 家族内聚性(intra/inter ratio)")
    print(f"{'='*60}")

    families = {
        "fruit": ["apple", "banana", "orange", "grape", "mango"],
        "animal": ["cat", "dog", "bird", "fish", "horse"],
        "color": ["red", "blue", "green", "yellow", "purple"],
    }

    formats = {
        "single": lambda w: w,
        "adj_noun": lambda w: f"the {w}",
        "sentence": lambda w: f"The {w} is very common in nature",
    }

    results = {}
    for fmt_name, fmt_fn in formats.items():
        print(f"\n  Format: {fmt_name}")
        fmt_results = {}

        for fam_name, words in families.items():
            # 编码
            encodings = {}
            for w in words:
                sent = fmt_fn(w)
                enc = encode_to_device(model, tokenizer, sent)
                with torch.no_grad():
                    out = model(**enc, output_hidden_states=True)
                h = out.hidden_states[-1][0, -1].cpu().float()
                encodings[w] = h

            # intra-family距离
            intra_dists = []
            for i in range(len(words)):
                for j in range(i+1, len(words)):
                    cos = F.cosine_similarity(
                        encodings[words[i]].unsqueeze(0),
                        encodings[words[j]].unsqueeze(0), dim=1
                    ).item()
                    intra_dists.append(1 - cos)

            # inter-family距离
            inter_dists = []
            other_families = [f for f in families.keys() if f != fam_name]
            for other_fam in other_families:
                other_words = families[other_fam]
                for ow in other_words:
                    if ow in encodings:
                        for w in words:
                            cos = F.cosine_similarity(
                                encodings[w].unsqueeze(0),
                                encodings[ow].unsqueeze(0), dim=1
                            ).item()
                            inter_dists.append(1 - cos)

            intra_mean = np.mean(intra_dists)
            inter_mean = np.mean(inter_dists)
            ratio = intra_mean / (inter_mean + 1e-10)

            fmt_results[fam_name] = {
                "intra": round(intra_mean, 4),
                "inter": round(inter_mean, 4),
                "ratio": round(ratio, 4),
            }
            print(f"    {fam_name:8s}: intra={intra_mean:.4f}, inter={inter_mean:.4f}, ratio={ratio:.4f}")

        results[fmt_name] = fmt_results

    return results


def experiment3_format_disambig_neuron_density(model, tokenizer, n_layers):
    """实验3：不同输入格式的消歧神经元比例"""
    print(f"\n{'='*60}")
    print(f"  实验3：输入格式 vs 消歧神经元集中度")
    print(f"{'='*60}")

    formats = {
        "single": "bank",
        "adj_noun": "river bank",
        "sentence_river": "The river bank was muddy",
        "sentence_finance": "The bank gave a loan",
    }

    patch_layer = min(8, n_layers - 1)
    results = {}

    for fmt_name, sentence in formats.items():
        enc = encode_to_device(model, tokenizer, sentence)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
        h = out.hidden_states[patch_layer][0, -1].cpu().float().numpy()

        # CV分析
        mean_val = np.abs(h).mean()
        std_val = h.std()
        cv = std_val / (mean_val + 1e-10)

        # 用另一个语境的编码来计算消歧神经元
        if "river" in sentence or "finance" in sentence:
            other_ctx = "The bank gave a loan today" if "river" in sentence else "The river bank was muddy"
            other_enc = encode_to_device(model, tokenizer, other_ctx)
            with torch.no_grad():
                other_out = model(**other_enc, output_hidden_states=True)
            other_h = other_out.hidden_states[patch_layer][0, -1].cpu().float().numpy()

            diff = np.abs(h - other_h)
            n_diff_std = diff.std(axis=0)
            n_diff_mean = diff.mean(axis=0)
            cv_disambig = n_diff_std / (np.abs(n_diff_mean) + 1e-10)
            threshold = np.percentile(cv_disambig, 75)
            active_ratio = (cv_disambig > threshold).mean()
        else:
            cv_disambig = cv
            active_ratio = (cv > np.percentile(cv, 75)).mean()

        results[fmt_name] = {
            "cv_mean": round(float(cv.mean()), 4),
            "cv_max": round(float(cv.max()), 4),
            "active_ratio": round(float(active_ratio), 4),
        }
        print(f"  {fmt_name:20s}: cv_mean={cv.mean():.4f}, cv_max={cv.max():.4f}, active={active_ratio:.4f}")

    return results


def main():
    t0 = time.time()
    print("=" * 60)
    print("  stage563: 输入标准化研究")
    print("=" * 60)

    model, tokenizer = load_qwen3_model()
    n_layers = len(discover_layers(model))
    print(f"  Qwen3 n_layers={n_layers}")

    try:
        r1 = experiment1_format_dimension_comparison(model, tokenizer, n_layers)
        r2 = experiment2_format_cohesion(model, tokenizer, n_layers)
        r3 = experiment3_format_disambig_neuron_density(model, tokenizer, n_layers)
    finally:
        free_model(model)

    all_results = {"exp1_dimension": r1, "exp2_cohesion": r2, "exp3_neurons": r3}
    out_path = os.path.join(OUTPUT_DIR, "stage563_results.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  results saved: {out_path}")
    print(f"\n  total time: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
